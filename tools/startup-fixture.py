#!/usr/bin/env python3
"""Operator/test-harness-launched CPU-only neutral journal producer, protocol v1.

This is not a model or KV implementation: it stores an exactly computed synthetic
answer in a separately launched bounded cache process. The Grill never invokes
this program. It proves lifecycle/request recording, not deployment qualification.
"""
import argparse
import hashlib
import http.client
import http.server
import json
import os
from pathlib import Path
import time

CAP = 2 * 1024 * 1024


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()


def process(pid):
    text = Path(f"/proc/{pid}/stat").read_text()
    fields = text[text.rfind(")") + 1:].split()
    return {"pid": pid, "start_ticks": int(fields[19]),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


class BoundedServer(http.server.HTTPServer):
    allow_reuse_address = True

    def get_request(self):
        conn, address = super().get_request()
        conn.settimeout(3)
        return conn, address


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def body(self):
        count = int(self.headers.get("Content-Length", "0"))
        if not 0 < count <= CAP:
            raise ValueError("body bound")
        raw = self.rfile.read(count)
        if len(raw) != count:
            raise ValueError("truncated request")
        return raw

    def send(self, code, body, media="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", media)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()
        self.close_connection = True


def cache_server(args):
    values = {}
    requests = 0

    class CacheHandler(Handler):
        def do_POST(self):
            nonlocal requests
            requests += 1
            if self.path != "/cache" or requests > 256:
                self.send(429, b"{}")
                return
            data = json.loads(self.body())
            key = data["identity"]
            if not isinstance(key, str) or len(key) != 64:
                self.send(400, b"{}")
                return
            hit = key in values and not data["store"] and not args.recompute
            # Independent exact synthetic computation; not an echoed expected_answer.
            value = values[key] if hit else str(sum(range(7)))
            if len(values) < 16 or key in values:
                values[key] = value
            else:
                self.send(507, b"{}")
                return
            self.send(200, encoded({"answer": value, "result": "stored" if data["store"]
                                   else "persisted_hit" if hit else "recomputed",
                                   "hash_seed_zero": os.environ.get("PYTHONHASHSEED") == "0",
                                   "cache_server": process(os.getpid())}))

    server = BoundedServer(("127.0.0.1", args.port), CacheHandler)
    server.timeout = 0.05
    end = time.monotonic() + args.lifetime
    try:
        while time.monotonic() < end and requests <= 256:
            server.handle_request()
    finally:
        server.server_close()


def engine(args):
    plan_bytes = Path(args.plan).read_bytes()
    if len(plan_bytes) > 4 * 1024 * 1024:
        raise ValueError("plan size bound")
    plan = json.loads(plan_bytes)
    restart = plan["study"]["kind"] == "restart_cache"
    identity = plan["study"]["identity"] if restart else None
    if restart and (not args.cache_port or not args.cache_pid):
        raise ValueError("restart fixture needs explicitly launched cache process")
    ids = ([r["id"] for r in plan["study"]["post_restart"]] if args.stage == "reload"
           else ["store"] if args.stage == "store" else ["first_inference"])
    if not 1 <= len(ids) <= 16:
        raise ValueError("request list bound")
    origin = time.monotonic_ns()
    engine_id = process(os.getpid())
    clock = {"id": f"engine-{engine_id['pid']}-{engine_id['start_ticks']}", "kind": "monotonic",
             "units": "microseconds", "resolution_us": 1,
             "synchronization": "single-process-origin"}
    fd = os.open(args.events, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    sequence = 0
    retained = 0

    def event(kind, **fields):
        nonlocal sequence, retained
        row = {"version": 1, "sequence": sequence, "clock": dict(clock),
               "offset_us": (time.monotonic_ns() - origin) // 1000,
               "engine": engine_id, "event": {"kind": kind, **fields}}
        if args.fault == "clock" and kind == "ready":
            row["clock"]["id"] = "different-origin"
        if args.fault == "stale" and kind == "ready":
            row["engine"] = {**engine_id, "start_ticks": engine_id["start_ticks"] + 1}
        raw = encoded(row) + b"\n"
        retained += len(raw)
        if retained > plan["event_bytes"] or sequence >= plan["max_events"]:
            raise ValueError("journal bound")
        os.write(fd, raw)
        os.fsync(fd)
        sequence += 1

    def state(name, value):
        event("state", state={"class": name, "identity": value, "declared": "unknown", "observed": "unknown"})

    seen = 0
    closed = False
    hits = 0
    queries = 0
    compute = 0
    epoch = int(time.time())
    event("launched")
    event("controls", smoke_suppressed=args.fault != "hidden", shape_warmup_suppressed=True,
          hash_seed_contract=("PYTHONHASHSEED=0:engine+cache_server"
                              if restart and os.environ.get("PYTHONHASHSEED") == "0" else "unavailable"))
    state("engine", {"kind": "process", "identity": engine_id})
    if restart and args.fault != "missing_class":
        state("cache_server", {"kind": "process", "identity": process(args.cache_pid)})
    # This process does not observe model weights, compiled caches or filesystem
    # cache state. Preserve their unavailability rather than inventing identities.
    for name in ["filesystem", "compiled_artifact_cache", "weights", "prefix_offload"]:
        state(name, None)
    time.sleep(args.listen_ms / 1000)
    if args.fault == "crash":
        os.close(fd)
        return

    class EngineHandler(Handler):
        def do_GET(self):
            if self.path != "/metrics":
                self.send(404, b"{}")
                return
            if args.fault == "missing_metrics":
                self.send(200, b"# missing selected counters\n", "text/plain")
                return
            values = {"vllm:prefix_cache_queries_total": queries,
                      "vllm:prefix_cache_hits_total": hits,
                      "vllm:external_prefix_cache_queries_total": queries,
                      "vllm:external_prefix_cache_hits_total": hits}
            lines = []
            for name, value in values.items():
                lines.append(f'{name}{{model_name="neutral"}} {value}')
                lines.append(f'{name.removesuffix("_total")}_created{{model_name="neutral"}} {epoch}')
            self.send(200, ("\n".join(lines) + "\n").encode(), "text/plain")

        def do_POST(self):
            nonlocal seen, closed, queries, hits, compute
            if self.path != "/v1/chat/completions" or closed:
                self.send(409, b"{}")
                return
            raw = self.body()
            body = json.loads(raw)
            index = seen
            seen += 1
            request_id = ids[index] if index < len(ids) else "extra"
            if args.fault == "reordered":
                request_id = "unexpected-order"
            event("request_start", id=request_id, body_sha256=digest(raw))
            if args.fault == "inference":
                event("failure", detail="synthetic inference failure")
                self.send(500, b'{"error":"synthetic"}')
                closed = True
                return
            if args.fault == "timeout":
                time.sleep(min(args.lifetime, 10))
            time.sleep(args.inference_ms / 1000)
            queries += 7
            outcome = "unavailable"
            answer = str(sum(range(7)))
            actual_identity = None
            cached = 0
            if restart:
                actual_identity = dict(identity)
                actual_identity["prompt_sha256"] = digest(body["messages"][0]["content"].encode())
                actual_identity["cache_salt"] = body.get("cache_salt", "")
                if args.fault == "identity":
                    actual_identity["cache_salt"] += "-drift"
                conn = http.client.HTTPConnection("127.0.0.1", args.cache_port, timeout=3)
                try:
                    conn.request("POST", "/cache", encoded({"identity": digest(encoded(actual_identity)),
                                                             "store": args.stage == "store"}))
                    response = conn.getresponse()
                    data = json.loads(response.read(CAP + 1))
                    if (response.status != 200 or data["cache_server"] != process(args.cache_pid)
                            or not data["hash_seed_zero"]):
                        raise ValueError("cache-server identity, hash seed or lookup failed")
                    answer, outcome = data["answer"], data["result"]
                finally:
                    conn.close()
                cached = 7 if outcome == "persisted_hit" else 0
                hits += cached
                compute += 7 - cached
            response_body = encoded({"choices": [{"index": 0, "message": {"role": "assistant", "content": answer},
                                                    "finish_reason": "stop"}],
                                     "usage": {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8,
                                               "prompt_tokens_details": {"cached_tokens": cached}}})
            if args.fault == "missing_hit":
                outcome = "unavailable"
            if args.fault != "missing_reload":
                event("inference_complete", id=request_id, response_sha256=digest(response_body),
                      cache_identity=actual_identity, cache_result=outcome)
            if args.fault == "extra":
                event("request_start", id="extra", body_sha256=digest(raw))
            if seen == len(ids):
                # Close inference admission before publishing final coverage; metrics remain readable.
                closed = True
                if args.fault != "unsealed":
                    event("sealed", requests=seen)
            self.send(200, response_body)

    server = BoundedServer(("127.0.0.1", args.port), EngineHandler)
    server.timeout = 0.05
    event("listening")
    event("communication_start")
    time.sleep(args.communication_ms / 1000)
    if args.fault != "missing_substage":
        event("communication_end")
    time.sleep(args.ready_ms / 1000)
    if args.fault == "malformed":
        os.write(fd, b'{"malformed":\n')
        os.fsync(fd)
    else:
        event("ready", contract="bad-ready" if args.fault == "bad_ready" else "neutral-ready-v1")
    if args.fault == "hidden":
        event("request_start", id="launcher-smoke", body_sha256="0" * 64)
    end = time.monotonic() + args.lifetime
    try:
        while time.monotonic() < end:
            server.handle_request()
    finally:
        server.server_close()
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["engine", "cache"])
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--lifetime", type=float, default=120)
    parser.add_argument("--plan")
    parser.add_argument("--events")
    parser.add_argument("--stage", choices=["startup", "store", "reload"], default="startup")
    parser.add_argument("--cache-port", type=int)
    parser.add_argument("--cache-pid", type=int)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--listen-ms", type=int, default=80)
    parser.add_argument("--communication-ms", type=int, default=100)
    parser.add_argument("--ready-ms", type=int, default=120)
    parser.add_argument("--inference-ms", type=int, default=140)
    parser.add_argument("--fault", choices=["none", "clock", "stale", "crash", "inference", "timeout",
                        "identity", "hidden", "extra", "reordered", "missing_reload", "missing_hit",
                        "missing_class", "missing_metrics", "missing_substage", "unsealed", "malformed", "bad_ready"], default="none")
    args = parser.parse_args()
    if not 1 <= args.lifetime <= 180 or not 1024 <= args.port <= 65535:
        parser.error("finite lifetime 1..180 seconds and ordinary port required")
    if any(not 0 <= getattr(args, key) <= 2000 for key in ["listen_ms", "communication_ms", "ready_ms", "inference_ms"]):
        parser.error("fixture delays must be within 0..2000 ms")
    if args.mode == "cache":
        cache_server(args)
    else:
        if not args.plan or not args.events:
            parser.error("engine requires explicit plan and fresh events path")
        engine(args)


if __name__ == "__main__":
    main()
