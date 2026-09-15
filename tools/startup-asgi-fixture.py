#!/usr/bin/env python3
"""Finite CPU-only ASGI server using the real bridge, without serving libraries.

The fixture's HTTP/1 transport is deliberately narrow. It exercises lifecycle,
outer-ASGI request coverage and engine-admission hooks, not Uvicorn/vLLM itself.
"""
import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location("startup_runtime", Path(__file__).with_name("startup-runtime.py"))
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


async def serve(args):
    source = Path(__file__).with_name("startup-runtime.py")
    initial_hash = bridge.digest(bridge.bounded_read(source))

    def source_check():
        if args.fault == "source_drift" or bridge.digest(bridge.bounded_read(source)) != initial_hash:
            raise ValueError("synthetic source drift")

    adapter = json.loads(bridge.bounded_read(args.plan))["adapter"]
    bootstrap = adapter == "runtime_asgi_fixture_bootstrap_v1"
    if adapter not in ("runtime_asgi_fixture_v1", "runtime_asgi_fixture_bootstrap_v1"):
        raise ValueError("fixture cannot claim real runtime source execution")
    contract = "ordinary-asgi-fixture-bootstrap-v1" if bootstrap else "ordinary-asgi-fixture-v1"
    journal = bridge.Journal(args.plan, args.events, args.stage, contract, source_check)
    if args.fault == "namespace":
        # Adversarial retained source identity, not a fabricated host PID mapping.
        # Alter the initial event before the collector attaches; replay must refuse.
        rows = Path(args.events).read_bytes().splitlines()
        row = json.loads(rows[0])
        row["event"]["pid_namespace"] = "pid:[0]"
        rows[0] = bridge.encoded(row)
        Path(args.events).write_bytes(b"\n".join(rows) + b"\n")
        # No more writes: malformed/incompatible source is a retained failure.
        await asyncio.sleep(args.lifetime)
        journal.close()
        return

    class Engine:
        queries = 0
        async def add_request(self):
            await asyncio.sleep(args.inference_ms / 1000)
            self.queries += 7
            return str(sum(range(7)))

    engine = Engine()

    class CoreClient:
        @staticmethod
        def make_async_mp_client():
            time.sleep(args.bootstrap_ms / 1000)
            if args.fault == "bootstrap_failed":
                raise RuntimeError("synthetic engine construction failure")
            return engine

    if bootstrap:
        bridge.install_bootstrap(journal, CoreClient)

    def construct():
        if args.fault == "missing_bootstrap_start":
            journal.emit("bootstrap_end")
        elif args.fault == "missing_bootstrap_end":
            journal.emit("bootstrap_start")
        elif args.fault != "missing_bootstrap":
            if args.fault == "duplicate_bootstrap_start":
                journal.emit("bootstrap_start")
            if args.fault == "reordered_bootstrap":
                journal.emit("bootstrap_end")
            assert CoreClient.make_async_mp_client() is engine
            if args.fault == "duplicate_bootstrap_end":
                journal.emit("bootstrap_end")

    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            message = await receive()
            assert message["type"] == "lifespan.startup"
            await asyncio.sleep(args.ready_ms / 1000)
            if args.fault == "invalid_ready":
                await send({"type": "lifespan.startup.failed", "message": "synthetic failure"})
                return
            await send({"type": "lifespan.startup.complete"})
            message = await receive()
            assert message["type"] == "lifespan.shutdown"
            await send({"type": "lifespan.shutdown.complete"})
            return
        media = b"application/json"
        if scope["method"] == "GET" and scope["path"] == "/metrics":
            media = b"text/plain"
            lines = []
            # Synthetic computation exposure only; no cached or persisted KV.
            for name, value in [("prefix_cache_queries", engine.queries), ("prefix_cache_hits", 0),
                                ("external_prefix_cache_queries", 0), ("external_prefix_cache_hits", 0)]:
                lines.extend([f'vllm:{name}_total{{model_name="neutral"}} {value}',
                              f'vllm:{name}_created{{model_name="neutral"}} 1'])
            raw = ("\n".join(lines) + "\n").encode()
        elif scope["method"] == "GET":
            # Simple synchronization only, not a readiness inference.
            raw = b"{}"
        else:
            await receive()
            answer = await engine.add_request()
            if args.fault == "hidden_inline":
                await engine.add_request()
            raw = bridge.encoded({"choices": [{"index": 0, "message": {"role": "assistant", "content": answer},
                                              "finish_reason": "stop"}],
                                  "usage": {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8,
                                            "prompt_tokens_details": {"cached_tokens": 0}}})
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", media), (b"content-length", str(len(raw)).encode())]})
        await send({"type": "http.response.body", "body": raw})

    class Server:
        def __init__(self):
            self.config = SimpleNamespace(workers=1, lifespan="on", loaded_app=app)
            self.started = False
            self.servers = []
            self.life_in = asyncio.Queue()
            self.life_out = asyncio.Queue()
            self.life_task = None
            self.handlers = set()

        async def startup(self, sockets=None):
            self.life_task = asyncio.create_task(self.config.loaded_app(
                {"type": "lifespan"}, self.life_in.get, self.life_out.put))
            await self.life_in.put({"type": "lifespan.startup"})
            message = await self.life_out.get()
            if message["type"] != "lifespan.startup.complete":
                return
            await asyncio.sleep(args.listen_ms / 1000)
            self.servers = [await asyncio.start_server(self.http, "127.0.0.1", args.port, limit=65536)]
            self.started = True

        async def http(self, reader, writer):
            task = asyncio.current_task()
            self.handlers.add(task)
            try:
                async with asyncio.timeout(5):
                    header = await reader.readuntil(b"\r\n\r\n")
                    lines = header[:-4].split(b"\r\n")
                    method, path, version = lines[0].decode("ascii").split(" ")
                    if version != "HTTP/1.1":
                        raise ValueError("fixture HTTP version")
                    headers = [tuple(part.strip() for part in line.split(b":", 1)) for line in lines[1:]]
                    lengths = [int(value) for key, value in headers if key.lower() == b"content-length"]
                    length = lengths[0] if lengths else 0
                    if len(lengths) > 1 or not 0 <= length <= bridge.BODY_CAP:
                        raise ValueError("fixture HTTP body bound")
                    body = await reader.readexactly(length)
                    received = False

                    async def receive():
                        nonlocal received
                        if not received:
                            received = True
                            return {"type": "http.request", "body": body}
                        await reader.read(1)
                        return {"type": "http.disconnect"}

                    async def send(message):
                        if message["type"] == "http.response.start":
                            writer.write(f"HTTP/1.1 {message['status']} Fixture\r\n".encode())
                            for key, value in message.get("headers", []):
                                writer.write(key + b": " + value + b"\r\n")
                            writer.write(b"Connection: close\r\n\r\n")
                        else:
                            writer.write(message.get("body", b""))
                        await writer.drain()
                    await self.config.loaded_app({"type": "http", "method": method, "path": path,
                                                  "headers": headers}, receive, send)
            except (Exception, asyncio.CancelledError):
                pass  # Bridge already retains ASGI failures; malformed transport never qualifies.
            finally:
                writer.close()
                await writer.wait_closed()
                self.handlers.discard(task)

    bridge.install(journal, Server, Engine)
    server = Server()
    try:
        if bootstrap and args.fault != "late_bootstrap":
            construct()
        await server.startup()
        if bootstrap and args.fault == "late_bootstrap":
            construct()
        if args.fault == "missing_seal" and journal.timer is not None:
            journal.timer.cancel()
        if args.fault == "hidden_engine":
            await engine.add_request()  # No HTTP context: real admission hook must retain this.
        await asyncio.sleep(args.lifetime)
    except Exception:
        if not journal.sealed:
            journal.emit("failure", detail="ordinary ASGI fixture startup failed")
        await asyncio.sleep(args.lifetime)
    finally:
        for listener in server.servers:
            listener.close()
            await listener.wait_closed()
        for task in tuple(server.handlers):
            task.cancel()
        if server.handlers:
            await asyncio.gather(*tuple(server.handlers), return_exceptions=True)
        if server.life_task is not None and not server.life_task.done():
            await server.life_in.put({"type": "lifespan.shutdown"})
            await server.life_task
        journal.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--stage", choices=["startup", "store", "reload"], default="startup")
    parser.add_argument("--fault", choices=[
        "none", "invalid_ready", "namespace", "source_drift", "missing_seal", "hidden_engine", "hidden_inline",
        "missing_bootstrap", "missing_bootstrap_start", "missing_bootstrap_end", "bootstrap_failed",
        "duplicate_bootstrap_start", "duplicate_bootstrap_end", "reordered_bootstrap", "late_bootstrap",
    ], default="none")
    parser.add_argument("--bootstrap-ms", type=int, default=60)
    parser.add_argument("--ready-ms", type=int, default=80)
    parser.add_argument("--listen-ms", type=int, default=100)
    parser.add_argument("--inference-ms", type=int, default=120)
    parser.add_argument("--lifetime", type=float, default=20)
    args = parser.parse_args()
    if (not 0 < args.port < 65536 or not 0 < args.lifetime <= 120
            or any(not 0 <= value <= 5000 for value in
                   (args.bootstrap_ms, args.ready_ms, args.listen_ms, args.inference_ms))):
        parser.error("finite fixture bounds exceeded")
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
