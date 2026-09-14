#!/usr/bin/env python3
"""Opt-in, in-process vLLM journal bridge. The operator owns this invocation.

No service manager, subprocess, remote control, source edits or cache emulation.
Only main() imports serving libraries; importing this file is CPU/stdlib-only.
The fixed source pins below are reviewed public bytes, not installed-version guesses.
"""
import argparse
import asyncio
import contextvars
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import stat
import sys
import time

SOURCE_CONTRACT = "vllm-0.27.0-uvicorn-0.34.0-sha256-v1"
SOURCE_PINS = {
    "vllm/entrypoints/openai/api_server.py": "cd4b83e85dc9d5aae808348d336e59b3064bee697012750d23c786de082a1c53",
    "vllm/entrypoints/launcher.py": "caf4c4517a62f05abe5998409de73af70e91d99dfab3819a4b359ba44af0da91",
    "vllm/v1/engine/async_llm.py": "81a0cae6d5da22140f509a59d6c6bb8fc6ee1572da2a2cd793b5830222d18bcc",
    "uvicorn/server.py": "8dd3d150523fd140a9981c41f0fae963b869b20c064dd94ab7422bc453748e6f",
}
CAP = 4 * 1024 * 1024
BODY_CAP = 2 * 1024 * 1024
REQUEST = contextvars.ContextVar("startup_runtime_request", default=None)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def bounded_read(path, cap=CAP):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as source:
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise ValueError("source must be a regular file")
        raw = source.read(cap + 1)
    if len(raw) > cap:
        raise ValueError("source byte bound exceeded")
    return raw


def process():
    pid = os.getpid()
    text = Path(f"/proc/{pid}/stat").read_text()
    fields = text[text.rfind(")") + 1:].split()
    return {"pid": pid, "start_ticks": int(fields[19]),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def verify_sources(paths):
    if set(paths) != set(SOURCE_PINS):
        raise ValueError("runtime source membership mismatch")
    for name, expected in SOURCE_PINS.items():
        if digest(bounded_read(paths[name])) != expected:
            raise ValueError("runtime source drift: " + name)


class Journal:
    def __init__(self, plan_path, events, stage, source_contract, source_check):
        raw = bounded_read(plan_path)
        self.plan = json.loads(raw)
        self.plan_hash = digest(raw)
        self.producer_hash = digest(bounded_read(__file__))
        if self.plan["adapter_sha256"] != self.producer_hash:
            raise ValueError("bridge source pin mismatch")
        self.window = self.plan["runtime_window_us"]
        if (not isinstance(self.window, int) or not 1000 <= self.window < self.plan["deadline_us"]
                or not 8 <= self.plan["max_events"] <= 1024
                or not 1024 <= self.plan["event_bytes"] <= CAP
                or not 1000 <= self.plan["deadline_us"] <= 3600000000):
            raise ValueError("runtime journal bounds")
        self.expected = ([r["id"] for r in self.plan["study"]["post_restart"]] if stage == "reload"
                         else ["store"] if stage == "store" else ["first_inference"])
        if not 1 <= len(self.expected) <= 16:
            raise ValueError("runtime request bound")
        self.identity = self.plan["study"].get("identity")
        self.engine = process()
        self.origin = time.monotonic_ns()
        self.clock = {"id": f"runtime-{self.engine['pid']}-{self.engine['start_ticks']}",
                      "kind": "monotonic", "units": "microseconds", "resolution_us": 1,
                      "synchronization": "single-process-origin"}
        self.fd = os.open(events, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        self.sequence = self.bytes = self.requests = self.active = 0
        self.closed = self.sealed = self.ready_seen = self.listening_seen = False
        self.engine_hook = False
        self.source_check = source_check
        self.timer = None
        self.emit("runtime_start", producer_sha256=self.producer_hash, source_contract=source_contract,
                  pid_namespace=os.readlink("/proc/self/ns/pid"),
                  time_namespace=os.readlink("/proc/self/ns/time"))
        # Frontend process identity is real; it does not identify every engine worker.
        for condition in self.plan["states"]:
            name = condition["class"]
            identity = {"kind": "process", "identity": self.engine} if name == "engine" else None
            self.emit("state", state={"class": name, "identity": identity,
                      "declared": condition["declared"], "observed": "unknown"})
        # Do not promote frontend env or a launcher assertion to cache-server facts.
        self.emit("controls", smoke_suppressed=False, shape_warmup_suppressed=False,
                  hash_seed_contract="PYTHONHASHSEED=0:frontend-only" if
                  os.environ.get("PYTHONHASHSEED") == "0" else "unavailable")

    def emit(self, kind, **fields):
        if self.sealed or os.getpid() != self.engine["pid"]:
            raise RuntimeError("journal sealed or inherited into another process")
        row = {"version": 1, "sequence": self.sequence, "clock": self.clock,
               "offset_us": (time.monotonic_ns() - self.origin) // 1000,
               "engine": self.engine, "event": {"kind": kind, **fields}}
        raw = encoded(row) + b"\n"
        if (self.sequence >= self.plan["max_events"] or self.bytes + len(raw) > self.plan["event_bytes"]
                or row["offset_us"] > self.plan["deadline_us"]):
            self.closed = True
            raise RuntimeError("runtime journal budget exhausted; no seal")
        # A short write must not become a silently complete journal.
        view = memoryview(raw)
        while view:
            count = os.write(self.fd, view)
            if count <= 0:
                raise OSError("short journal write")
            view = view[count:]
        os.fsync(self.fd)
        self.bytes += len(raw)
        self.sequence += 1

    def engine_request(self):
        context = REQUEST.get()
        if context is None or not context["active"]:
            self.emit("failure", detail="unobserved/internal AsyncLLM.add_request outside ASGI request interval")
            return
        context["engine_requests"] += 1
        self.emit("runtime_engine_request", id=context["id"])

    def ready(self):
        if self.ready_seen:
            raise RuntimeError("duplicate ASGI startup completion")
        self.ready_seen = True
        self.emit("ready", contract="asgi-lifespan-startup-complete-v1")
        self.timer = asyncio.create_task(self.coverage())

    def listening(self, server):
        # The return of Server.startup alone is insufficient on failed lifespan.
        sockets = [sock for listener in server.servers for sock in (listener.sockets or [])]
        import socket
        if not server.started or not sockets or not all(
                sock.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) == 1 for sock in sockets):
            raise RuntimeError("Uvicorn startup returned without an actual listening socket")
        if self.listening_seen:
            raise RuntimeError("multiple frontend listeners are unsupported")
        self.listening_seen = True
        self.emit("listening")

    async def coverage(self):
        await asyncio.sleep(self.window / 1000000)
        # Atomic on the serving event loop: no newly admitted mutating request
        # can race the active count or final seal. Existing requests must drain.
        self.closed = True
        while self.active:
            if (time.monotonic_ns() - self.origin) // 1000 >= self.plan["deadline_us"]:
                return  # Missing seal, never a successful partial interval.
            await asyncio.sleep(0.01)
        try:
            self.source_check()
            if not self.listening_seen:
                self.emit("failure", detail="coverage ended without a listening socket")
                return
            self.emit("runtime_coverage", window_us=self.window, admission_closed=True,
                      active_requests=0, engine_hook=self.engine_hook)
            self.emit("sealed", requests=self.requests)
            self.sealed = True
        except Exception:
            if not self.sealed:
                self.emit("failure", detail="runtime source drift or coverage failure")

    def close(self):
        if self.timer is not None:
            self.timer.cancel()
        os.close(self.fd)


class Middleware:
    """Outermost ASGI observer: all non-readonly HTTP and every websocket.

    Only exact GET/HEAD /health, /metrics and /v1/models are exempt. This
    deliberately counts unknown routes/methods rather than guessing inference.
    The journal retains digests, never auth headers or external prompt bodies.
    """
    def __init__(self, app, journal):
        self.app, self.journal = app, journal

    async def __call__(self, scope, receive, send):
        j = self.journal
        if scope["type"] == "lifespan":
            async def lifecycle(message):
                if message["type"] == "lifespan.startup.complete":
                    j.ready()
                elif message["type"] == "lifespan.startup.failed":
                    j.emit("failure", detail="ASGI lifespan startup failed")
                await send(message)
            await self.app(scope, receive, lifecycle)
            return
        if scope["type"] == "http" and scope["method"] in ("GET", "HEAD") and scope["path"] in (
                "/health", "/metrics", "/v1/models"):
            await self.app(scope, receive, send)
            return
        if scope["type"] not in ("http", "websocket"):
            raise RuntimeError("unsupported ASGI scope")
        if j.closed:
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1013})
            else:
                await send({"type": "http.response.start", "status": 503,
                            "headers": [(b"content-type", b"application/json")]})
                await send({"type": "http.response.body", "body": b'{"error":"observation admission closed"}'})
            return
        j.requests += 1
        j.active += 1
        number = j.requests
        context = None
        token = None
        try:
            if j.active != 1 or number > j.plan["max_events"]:
                j.closed = True
                raise ValueError("overlapping or excessive runtime requests")
            if scope["type"] == "websocket":
                j.emit("request_start", id=f"external-{number}", body_sha256=digest(b"websocket"))
                j.emit("failure", detail="websocket inference outside supported finite request contract")
                await send({"type": "websocket.close", "code": 1008})
                return
            body = bytearray()
            async with asyncio.timeout(j.plan["limits"]["total_ms"] / 1000):
                while True:
                    message = await receive()
                    if message["type"] != "http.request":
                        raise ValueError("request disconnected before complete body")
                    part = message.get("body", b"")
                    if len(body) + len(part) > BODY_CAP:
                        raise ValueError("request body exceeds finite bridge bound")
                    body.extend(part)
                    if not message.get("more_body", False):
                        break
            headers = scope.get("headers", [])
            ids = [value for key, value in headers if key.lower() == b"x-grill-startup-request"]
            plans = [value for key, value in headers if key.lower() == b"x-grill-startup-plan"]
            request_id = f"external-{number}"
            if len(ids) == len(plans) == 1 and plans[0] == j.plan_hash.encode():
                candidate = ids[0].decode("ascii")
                if candidate in j.expected:
                    request_id = candidate
            j.emit("request_start", id=request_id, body_sha256=digest(body))
            if not j.ready_seen:
                j.emit("failure", detail="request before ASGI readiness")
            context = {"id": request_id, "active": True, "engine_requests": 0}
            token = REQUEST.set(context)
            delivered = False
            response_hash = hashlib.sha256()
            response_bytes = 0
            status_code = None
            complete = False

            async def replay_receive():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()

            async def response(message):
                nonlocal response_bytes, status_code, complete
                if message["type"] == "http.response.start":
                    status_code = message["status"]
                elif message["type"] == "http.response.body":
                    part = message.get("body", b"")
                    response_bytes += len(part)
                    if response_bytes > j.plan["limits"]["response_bytes"]:
                        raise ValueError("response exceeds finite bridge bound")
                    response_hash.update(part)
                await send(message)
                if message["type"] == "http.response.body" and not message.get("more_body", False):
                    complete = True
                    if status_code != 200:
                        j.emit("failure", detail="runtime returned non-200 inference response")
                    # Cache identity is request exposure, not proof of KV identity.
                    identity = None
                    if j.identity is not None:
                        parsed = json.loads(body)
                        identity = dict(j.identity)
                        identity["prompt_sha256"] = digest(parsed["messages"][0]["content"].encode())
                        identity["cache_salt"] = parsed.get("cache_salt", "")
                    j.emit("inference_complete", id=request_id, response_sha256=response_hash.hexdigest(),
                           cache_identity=identity, cache_result="unavailable")
            async with asyncio.timeout(j.plan["limits"]["total_ms"] / 1000):
                await self.app(scope, replay_receive, response)
            if not complete:
                j.emit("failure", detail="runtime response did not complete")
            if j.engine_hook and context["engine_requests"] != 1:
                j.emit("failure", detail="inference response requires exactly one observed AsyncLLM.add_request")
        except BaseException:
            j.emit("failure", detail="runtime request failed, disconnected or exceeded bound")
            raise
        finally:
            if context is not None:
                context["active"] = False
            if token is not None:
                REQUEST.reset(token)
            j.active -= 1


def install(journal, server_class, engine_class):
    """Fixed reviewed hook points; also usable with the ordinary ASGI fixture."""
    startup = server_class.startup
    add_request = engine_class.add_request

    async def observed_startup(server, sockets=None):
        if server.config.workers != 1 or server.config.lifespan != "on":
            raise ValueError("bridge requires one frontend and explicit lifespan=on")
        server.config.loaded_app = Middleware(server.config.loaded_app, journal)
        await startup(server, sockets=sockets)
        journal.listening(server)

    async def observed_add_request(engine, *args, **kwargs):
        journal.engine_request()
        return await add_request(engine, *args, **kwargs)

    server_class.startup = observed_startup
    engine_class.add_request = observed_add_request
    journal.engine_hook = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--stage", choices=["startup", "store", "reload"], required=True)
    parser.add_argument("serving_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    serving_args = args.serving_args
    if serving_args[:1] != ["--"]:
        parser.error("use -- followed by the existing vLLM api_server arguments")
    # Locate public source bytes without importing vLLM or touching devices.
    paths = {name: importlib.metadata.distribution(name.split("/")[0]).locate_file(name)
             for name in SOURCE_PINS}
    verify_sources(paths)
    journal = Journal(args.plan, args.events, args.stage, SOURCE_CONTRACT, lambda: verify_sources(paths))
    try:
        if journal.plan["adapter"] != "runtime_vllm_v1":
            raise ValueError("real runtime requires runtime_vllm_v1 plan")
        # Imports execute only on explicit operator launch, never during fixture import.
        import uvicorn.server
        import vllm.v1.engine.async_llm
        for name, module in [("uvicorn/server.py", uvicorn.server),
                             ("vllm/v1/engine/async_llm.py", vllm.v1.engine.async_llm)]:
            if Path(module.__file__).resolve() != Path(paths[name]).resolve():
                raise ValueError("imported runtime source differs from verified distribution")
        install(journal, uvicorn.server.Server, vllm.v1.engine.async_llm.AsyncLLM)
        # This is the existing single-worker entrypoint in this same OS process,
        # not a child launcher or configurable module/plugin execution framework.
        sys.argv = ["vllm.entrypoints.openai.api_server", *serving_args[1:]]
        import vllm.entrypoints.openai.api_server as api
        if Path(api.__file__).resolve() != Path(paths["vllm/entrypoints/openai/api_server.py"]).resolve():
            raise ValueError("imported api_server source mismatch")
        launcher = sys.modules[api.serve_http.__module__]
        if Path(launcher.__file__).resolve() != Path(paths["vllm/entrypoints/launcher.py"]).resolve():
            raise ValueError("imported launcher source mismatch")
        verify_sources(paths)
        api.cli_env_setup()
        parsed = api.make_arg_parser(api.FlexibleArgumentParser()).parse_args()
        api.validate_parsed_serve_args(parsed)
        api.uvloop.run(api.run_server(parsed, lifespan="on"))
    except BaseException:
        if not journal.sealed:
            journal.emit("failure", detail="runtime entrypoint failed")
        raise
    finally:
        journal.close()


if __name__ == "__main__":
    main()
