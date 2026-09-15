#!/usr/bin/env python3
"""Ordinary finite HTTP/LRU fixture; never a model/KV qualification source.

Runs real insertion/lookup/allocation-driven eviction under the same concrete
journal hook API. Replies derive a fact from actual acquired parent messages.
"""
import argparse
from collections import OrderedDict
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location("retention_journal", Path(__file__).with_name("retention-journal.py"))
journal_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(journal_module)


class Block:
    def __init__(self, index):
        self.block_id = index
        self.block_hash = None
        self.is_null = False


class BlockMap:
    def __init__(self):
        self.entries = {}

    def contain(self, key, block_id):
        return key in self.entries and self.entries[key].block_id == block_id

    def get_one_block(self, key):
        return self.entries.get(key)


class Pool:
    def __init__(self, capacity):
        self.num_gpu_blocks = capacity
        self.cached_block_hash_to_block = BlockMap()
        self.blocks = [Block(i) for i in range(capacity)]
        self.lru = OrderedDict((b.block_id, b) for b in self.blocks)

    def get_num_free_blocks(self):
        return len(self.lru)

    def _remove_cached_block_hashes(self, block):
        key = block.block_hash
        if key is None:
            return []
        del self.cached_block_hash_to_block.entries[key]
        block.block_hash = None
        return [key]

    def _maybe_evict_cached_block(self, block):
        return bool(self._remove_cached_block_hashes(block))

    def get_new_blocks(self, count):
        chosen = []
        for _ in range(count):
            _, block = self.lru.popitem(last=False)
            self._maybe_evict_cached_block(block)
            chosen.append(block)
        return chosen

    def _insert_block_hash(self, key, block, num_tokens):
        block.block_hash = key
        self.cached_block_hash_to_block.entries[key] = block
        self.lru[block.block_id] = block

    def cache_full_blocks(self, request, blocks, *args, **kwargs):
        self._insert_block_hash(request.key, blocks[0], 8)

    def reset_prefix_cache(self):
        raise AssertionError("fixture never resets cache")


class Manager:
    def __init__(self, pool):
        self.block_pool = pool

    def get_computed_blocks(self, request):
        block = None if request.skip_read else self.block_pool.cached_block_hash_to_block.get_one_block(request.key)
        if block:
            self.block_pool.lru.move_to_end(block.block_id)
        return SimpleNamespace(blocks=([block] if block else [],)), 8 if block else 0, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=1)
    parser.add_argument("--max-seconds", type=int, default=120)
    parser.add_argument("--delay-ms", type=int, default=30)
    parser.add_argument("--missing-counters", action="store_true")
    parser.add_argument("--miss-on-probe", action="store_true")
    parser.add_argument("--fail-after", type=int)
    args = parser.parse_args()
    if not 1 <= args.capacity <= 128 or not 1 <= args.max_seconds <= 300 or not 0 <= args.delay_ms <= 3000:
        parser.error("fixture bounds exceeded")
    journal = journal_module.Journal(args.journal, source=journal_module.FIXTURE,
                                     max_window_us=300000000)
    journal_module.install(journal, Pool, Manager)
    manager = Manager(Pool(args.capacity))
    totals = {"queries": 0, "hits": 0, "tokens": 0, "cached": 0, "compute": 0, "requests": 0}
    epoch = time.time()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, status, content, kind):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path != "/metrics":
                self.reply(404, b"", "text/plain")
                return
            names = {
                "prefix_cache_queries": totals["queries"],
                "prefix_cache_hits": totals["hits"],
                "prompt_tokens": totals["tokens"],
                "prompt_tokens_cached": totals["cached"],
            }
            lines = []
            labels = 'model_name="fixture-model",engine="0"'
            if not args.missing_counters:
                for name, value in names.items():
                    lines.extend([f"vllm:{name}_total{{{labels}}} {value}",
                                  f"vllm:{name}_created{{{labels}}} {epoch}"])
                for source, value in [("local_compute", totals["compute"]), ("local_cache_hit", totals["cached"]), ("external_kv_transfer", 0)]:
                    selected = labels + ',source="' + source + '"'
                    lines.extend([f"vllm:prompt_tokens_by_source_total{{{selected}}} {value}",
                                  f"vllm:prompt_tokens_by_source_created{{{selected}}} {epoch}"])
            self.reply(200, ("\n".join(lines) + "\n").encode(), "text/plain; version=0.0.4")

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            if self.path != "/v1/chat/completions" or not 0 < length <= 2097152:
                self.reply(400, b"{}", "application/json")
                return
            body = json.loads(self.rfile.read(length))
            totals["requests"] += 1
            if args.fail_after is not None and totals["requests"] > args.fail_after:
                # A finite ordinary allocation fails its own fixed allowance;
                # this is provider-reported fixture OOM, not a real device OOM.
                try:
                    FixedAllocator(0).allocate(1)
                except MemoryError:
                    self.reply(503, b'{"error":{"code":"out_of_memory","type":"fixture_allocation_limit"}}', "application/json")
                    return
            messages = body["messages"]
            salt = body.get("cache_salt", "flat-" + str(totals["requests"]))
            root = messages[0]["content"]
            key = hashlib.sha256((salt + "\0" + root).encode()).digest()
            skip = args.miss_on_probe and "Probe" in messages[-1]["content"]
            request = SimpleNamespace(cache_salt=salt, key=key, skip_read=skip)
            _, cached, _ = manager.get_computed_blocks(request)
            if not cached and not manager.block_pool.cached_block_hash_to_block.get_one_block(key):
                blocks = manager.block_pool.get_new_blocks(1)
                manager.block_pool.cache_full_blocks(request, blocks)
            prompt = 16 + len(messages) * 8
            totals["queries"] += prompt
            totals["hits"] += cached
            totals["tokens"] += prompt
            totals["cached"] += cached
            totals["compute"] += prompt - cached
            # Actual parent output is parsed, not replaced with expected answers.
            parent = next((m["content"] for m in reversed(messages) if m["role"] == "assistant"), None)
            if parent is not None:
                fact = json.loads(parent)["fact"]
            elif root.startswith("Remember "):
                fact = root[len("Remember "):].rstrip(".")
            else:
                fact = "sapphire"
            content = json.dumps({"fact": fact}, separators=(",", ":"))
            time.sleep(args.delay_ms / 1000)
            usage = {"prompt_tokens": prompt, "completion_tokens": 8, "total_tokens": prompt + 8,
                     "prompt_tokens_details": {"cached_tokens": cached}}
            if body.get("stream"):
                chunks = [{"choices": [{"delta": {"content": content}}]},
                          {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": usage}]
                encoded = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
                self.reply(200, encoded.encode(), "text/event-stream")
            else:
                self.reply(200, json.dumps({"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}], "usage": usage}).encode(), "application/json")

    server = HTTPServer(("127.0.0.1", 0), Handler)
    server.timeout = 0.2
    args.ready.write_text(json.dumps({"port": server.server_port}))
    deadline = time.monotonic() + args.max_seconds
    try:
        while time.monotonic() < deadline:
            server.handle_request()
    finally:
        server.server_close()
        journal.close()


class FixedAllocator:
    def __init__(self, limit):
        self.limit = limit
        self.used = 0

    def allocate(self, count):
        if count > self.limit - self.used:
            raise MemoryError("fixed fixture allocation ceiling")
        block = bytearray(count)
        self.used += len(block)
        return block


if __name__ == "__main__":
    main()
