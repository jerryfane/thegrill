#!/usr/bin/env python3
"""Finite explicit backend hook; no launcher, service control, flags or devices.

An operator calls install_vllm(journal) in the owning engine process before any
cache activity. This is deliberately NOT a frontend ASGI journal: it observes
actual BlockPool insertion, allocation-driven removal and computed-block lookup.
Importing this module uses only the standard library; it never imports vLLM.
"""
import contextvars
import functools
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import stat
import threading
import time

SOURCE = "vllm-0.27.0-block-pool-v1"
FIXTURE = "ordinary-lru-fixture-v1"
PINS = {
    "vllm/v1/core/block_pool.py": "51cad2fd425128a0ff433ca4685acfc022e40659153ea5d7180fc586e1eebb6c",
    "vllm/v1/core/kv_cache_manager.py": "70f7f608c0963af155540630a5633483e5c19c0c0280dfd621760d5db2a419a6",
    "vllm/v1/request.py": "6085b0668f41d56cd81ef483c06456f4ebe88bc1d46c8131d7134182d7893012",
}


def bounded_read(path, cap):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("source must be a regular nonsymlink file")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read(cap + 1)
        if len(data) > cap:
            raise ValueError("source exceeds byte bound")
        return data
    finally:
        os.close(fd)


def verify_sources(paths):
    if set(paths) != set(PINS):
        raise ValueError("exact backend source membership required")
    for name, path in paths.items():
        if hashlib.sha256(bounded_read(path, 2 * 1024 * 1024)).hexdigest() != PINS[name]:
            raise ValueError("unsupported backend source bytes: " + name)


class Journal:
    """One process, one cache pool, finite append-only source observations.

    Exhaustion stops RECORDING, not serving. An explicit failure event consumes
    the reserved last event/bytes; replay withholds all continuity claims.
    """
    def __init__(self, path, *, source, max_events=10000, max_bytes=8388608,
                 max_window_us=60000000):
        if source not in (SOURCE, FIXTURE):
            raise ValueError("unsupported journal source")
        if not 8 <= max_events <= 10000 or not 4096 <= max_bytes <= 16777216:
            raise ValueError("invalid finite journal bounds")
        if not 1000 <= max_window_us <= 3600000000:
            raise ValueError("invalid journal duration")
        self.fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        self.origin = time.monotonic_ns()
        self.pid = os.getpid()
        self.incarnation = os.urandom(16).hex()
        self.source = source
        self.sequence = 0
        self.bytes = 0
        self.overhead_ns = 0
        self.max_events = max_events
        self.max_bytes = max_bytes
        self.max_window_us = max_window_us
        self.failed = False
        self.lock = threading.RLock()
        self.pool = None
        self.check_hooks = lambda: True
        producer = hashlib.sha256(bounded_read(__file__, 1024 * 1024)).hexdigest()
        self.emit("start", producer_sha256=producer, source_pins=PINS if source == SOURCE else {},
                  max_events=max_events, max_bytes=max_bytes, max_window_us=max_window_us)

    def emit(self, kind, **data):
        with self.lock:
            if self.failed:
                return
            started = time.monotonic_ns()
            elapsed = (started - self.origin) // 1000
            if os.getpid() != self.pid or not self.check_hooks():
                kind, data = "failure", {"reason": "source_or_process_changed"}
            elif self.sequence >= self.max_events - 1 or elapsed >= self.max_window_us:
                kind, data = "failure", {"reason": "finite_budget_exhausted"}
            event = {"version": 1, "source": self.source, "incarnation": self.incarnation,
                     "sequence": self.sequence, "offset_us": elapsed,
                     "clock": "producer_monotonic_microseconds", "overhead_us": self.overhead_ns // 1000,
                     "event": {"kind": kind, **data}}
            encoded = (json.dumps(event, separators=(",", ":")) + "\n").encode()
            if len(encoded) > 65536 or self.bytes + len(encoded) > self.max_bytes - 1024:
                event["event"] = {"kind": "failure", "reason": "byte_or_event_budget_exhausted"}
                encoded = (json.dumps(event, separators=(",", ":")) + "\n").encode()
                kind = "failure"
            try:
                view = memoryview(encoded)
                while view:
                    n = os.write(self.fd, view)
                    if n <= 0:
                        raise OSError("journal short write")
                    view = view[n:]
                os.fsync(self.fd)
            except OSError:
                self.failed = True  # Partial last line is rejected by replay.
                return  # Observation failure must not change the serving result.
            self.sequence += 1
            self.bytes += len(encoded)
            self.overhead_ns += time.monotonic_ns() - started
            if kind == "failure":
                self.failed = True

    def bind_pool(self, pool):
        if self.pool is None:
            self.pool = pool
        if self.pool is not pool or pool.num_gpu_blocks > 65536:
            self.emit("failure", reason="multiple_or_oversized_pool")
            return False
        return True

    def close(self):
        os.close(self.fd)


def key_hash(value, journal):
    if not isinstance(value, bytes) or len(value) > 1024:
        journal.emit("failure", reason="unsupported_block_hash_representation")
        return None
    return hashlib.sha256(value).hexdigest()


def install(journal, pool_class, manager_class):
    """Fixed concrete API, also exercised by the ordinary LRU fixture.

    Production calls ONLY install_vllm, which verifies exact source bytes.
    """
    if getattr(pool_class, "_grill_retention_hook", False):
        raise ValueError("retention hook already installed")
    allocating = contextvars.ContextVar("grill_allocating", default=False)
    caching = contextvars.ContextVar("grill_caching", default=None)
    hooked = []

    def wrap(cls, name, factory):
        original = getattr(cls, name)
        replacement = functools.wraps(original)(factory(original))
        setattr(cls, name, replacement)
        hooked.append((cls, name, replacement))

    def allocation(original):
        def call(pool, num_blocks):
            journal.bind_pool(pool)
            journal.emit("allocate", requested_blocks=num_blocks,
                         free_blocks=pool.get_num_free_blocks(), total_blocks=pool.num_gpu_blocks)
            token = allocating.set(True)
            try:
                return original(pool, num_blocks)
            finally:
                allocating.reset(token)
        return call

    def evict(original):
        def call(pool, block):
            # _remove_cached_block_hashes gives the actual removed aliases, not
            # preemption, miss, configured pool size or a guessed eviction count.
            token = caching.set(("eviction", allocating.get()))
            try:
                return original(pool, block)
            finally:
                caching.reset(token)
        return call

    def remove(original):
        def call(pool, block):
            removed = original(pool, block)
            if removed:
                context = caching.get()
                if context == ("eviction", True):
                    absent = [key_hash(k, journal) for k in removed
                              if pool.cached_block_hash_to_block.get_one_block(k) is None]
                    if absent:
                        journal.emit("evict", keys=absent)
                else:
                    # Promotion, movement, explicit eviction and reset are not
                    # allocation-pressure eviction; conservatively break coverage.
                    journal.emit("failure", reason="non_allocation_cache_invalidation")
            return removed
        return call

    def cache(original):
        def call(pool, request, *args, **kwargs):
            journal.bind_pool(pool)
            salt = getattr(request, "cache_salt", None)
            token = caching.set(salt if isinstance(salt, str) and len(salt) <= 256 else None)
            try:
                return original(pool, request, *args, **kwargs)
            finally:
                caching.reset(token)
        return call

    def insert(original):
        def call(pool, block_hash, block, num_tokens):
            was_present = pool.cached_block_hash_to_block.contain(block_hash, block.block_id)
            result = original(pool, block_hash, block, num_tokens)
            if not was_present and pool.cached_block_hash_to_block.contain(block_hash, block.block_id):
                salt = caching.get()
                if not isinstance(salt, str):
                    journal.emit("failure", reason="unattributed_cache_insertion")
                else:
                    journal.emit("store", cache_salt=salt, keys=[key_hash(block_hash, journal)])
            return result
        return call

    def lookup(original):
        def call(manager, request):
            journal.bind_pool(manager.block_pool)
            result = original(manager, request)
            blocks, cached_tokens, _ = result
            salt = getattr(request, "cache_salt", None)
            keys = [key_hash(block.block_hash, journal) for group in blocks.blocks for block in group
                    if block.block_hash is not None and not block.is_null]
            if len(keys) > 1024 or not isinstance(salt, str) or len(salt) > 256:
                journal.emit("failure", reason="unbounded_or_unattributed_lookup")
            else:
                journal.emit("lookup", cache_salt=salt, keys=keys, cached_tokens=cached_tokens)
            return result
        return call

    def reset(original):
        def call(pool, *args, **kwargs):
            journal.emit("failure", reason="explicit_cache_reset")
            return original(pool, *args, **kwargs)
        return call

    wrap(pool_class, "get_new_blocks", allocation)
    wrap(pool_class, "_maybe_evict_cached_block", evict)
    wrap(pool_class, "_remove_cached_block_hashes", remove)
    wrap(pool_class, "cache_full_blocks", cache)
    wrap(pool_class, "_insert_block_hash", insert)
    wrap(pool_class, "reset_prefix_cache", reset)
    wrap(manager_class, "get_computed_blocks", lookup)
    pool_class._grill_retention_hook = True
    journal.check_hooks = lambda: all(getattr(cls, name, None) is fn for cls, name, fn in hooked)
    journal.emit("installed", hook="allocation-remove-store-lookup-v1")


def install_vllm(journal):
    """Operator-invoked in the single owning backend process before cache use.

    No environment flag is set, no cache is reset, and no server is launched.
    vLLM imports happen only here, never during source inspection or CPU fixtures.
    """
    if journal.source != SOURCE:
        raise ValueError("production journal source mismatch")
    pool = importlib.import_module("vllm.v1.core.block_pool")
    manager = importlib.import_module("vllm.v1.core.kv_cache_manager")
    request = importlib.import_module("vllm.v1.request")
    verify_sources({name: inspect.getsourcefile(module) for name, module in zip(PINS, (pool, manager, request))})
    install(journal, pool.BlockPool, manager.KVCacheManager)
