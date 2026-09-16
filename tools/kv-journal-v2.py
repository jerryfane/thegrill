"""Explicit registered-state/L1 observation protocol; stdlib-only on import.

Callbacks are not device completion. Only an existing source event query, followed
by weak allocation validation, can close a copied operation. No device work is
created by this observer and no MemoryObj or tensor is retained by the journal.

Observation callbacks run synchronously while journal.condition is held. They
must be nonblocking: never wait for the dispatcher, MQ loop, or another thread
that may need this condition. Existing event queries are nonblocking; lifetime
checks may take only reentrant/leaf locks. No callback arguments escape observe.
"""

import contextvars
import functools
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import stat
import sys
import threading
import time
import weakref

SOURCE = "vllm-752a3a504-lmcache-ddc5fa34-kv-copy-v3"
SCOPE = "registered-state-l1-engine-restart-v3"
POLICY = {"dcp": "1", "groups": "attention-only", "request_configs": "absent",
          "event_backend": "default-torch-cuda", "registration": "single-generation",
          "admission": "candidate-not-independent-or-native"}
ASYNC = "vllm/v1/engine/async_llm.py"
OP = contextvars.ContextVar("grill_kv_operation_v2", default=None)
CALLBACK = contextvars.ContextVar("grill_kv_callback_v2", default=None)
PAGE = 64


def bounded_read(path, cap=4 * 1024 * 1024):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("source is not a regular file")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(cap + 1)
        if len(raw) > cap:
            raise ValueError("source exceeds bound")
        return raw
    finally:
        os.close(fd)


def load_adjacent(name):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def process():
    pid = os.getpid()
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "start_ticks": int(fields[19]),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def encoded(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode()


def key_value(key):
    return {"hash": key.chunk_hash.hex(), "model": key.model_name,
            "kv_rank": key.kv_rank, "group": key.object_group_id, "salt": key.cache_salt}


def integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("integer outside observation bounds")
    return value


class Journal:
    operation_context = OP
    scope = SCOPE

    def __init__(self, path, *, role, max_events=10000, max_bytes=8 * 1024 * 1024,
                 max_window_us=60000000):
        if role not in ("worker", "cache", "frontend"):
            raise ValueError("unknown journal role")
        integer(max_events, 8, 10000)
        integer(max_bytes, 4096, 16 * 1024 * 1024)
        integer(max_window_us, 1000, 3600000000)
        self.fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        self.role, self.process = role, process()
        self.incarnation = os.urandom(16).hex()
        self.max_events, self.max_bytes, self.max_window_us = max_events, max_bytes, max_window_us
        self.origin = time.monotonic_ns()
        self.sequence = self.bytes = self.active = 0
        self.failed = self.closed = self.sealed = False
        self.condition = threading.Condition(threading.RLock())
        self.hooks, self.operations, self.registrations = [], {}, {}
        self.lifetime = None
        self.event_guard = None
        self.cache_seal_guard = None
        self.event_owners = {}
        self.worker_pending = {}
        self.worker_generation = 0
        self.worker_futures = {}
        self.emit("start", source=SOURCE, scope=SCOPE, role=role, process=self.process,
                  observation_policy=POLICY,
                  producer_sha256=hashlib.sha256(bounded_read(__file__)).hexdigest(),
                  components={name: hashlib.sha256(bounded_read(Path(__file__).with_name(name))).hexdigest()
                              for name in ("kv-registration.py", "kv-registration-hooks.py", "kv-lifetime-v2.py",
                                           "kv-transfer-v2.py", "kv-v2-pins.json")},
                  max_events=max_events, max_bytes=max_bytes, max_window_us=max_window_us,
                  hash_randomization=sys.flags.hash_randomization,
                  pythonhashseed=os.environ.get("PYTHONHASHSEED"))

    def emit(self, kind, **data):
        with self.condition:
            if self.failed or self.sealed:
                return
            if kind in ("worker_registration", "cache_registration"):
                engine = data["engine_instance"]
                if engine in self.registrations or self.closed:
                    kind, data = "failure", {"reason": "duplicate_or_closed_registration"}
                else:
                    self.registrations[engine] = {"kind": kind, **data}
            elapsed = (time.monotonic_ns() - self.origin) // 1000
            if (os.getpid() != self.process["pid"] or self.sequence >= self.max_events - 1
                    or elapsed > self.max_window_us):
                kind, data = "failure", {"reason": "finite_process_budget"}
            row = {"version": 3, "sequence": self.sequence, "incarnation": self.incarnation,
                   "offset_us": elapsed, "event": {"kind": kind, **data}}
            try:
                raw = encoded(row) + b"\n"
                if len(raw) > 65536 or self.bytes + len(raw) > self.max_bytes - 1024:
                    row["event"] = {"kind": "failure", "reason": "byte_budget"}
                    raw, kind = encoded(row) + b"\n", "failure"
                view = memoryview(raw)
                while view:
                    written = os.write(self.fd, view)
                    if written <= 0:
                        raise OSError("short journal write")
                    view = view[written:]
                os.fsync(self.fd)
                self.sequence += 1
                self.bytes += len(raw)
                self.failed = kind == "failure"
            except (OSError, ValueError, TypeError):
                self.failed = True

    def observe(self, callback, *args):
        """Invoke inline, in order, without retaining arguments after return."""
        with self.condition:
            try:
                return callback(*args)
            except Exception as error:
                reason = "observation_exception"
                if type(error) is ValueError and len(error.args) == 1 and type(error.args[0]) is str:
                    reason += ":" + error.args[0][:256]
                self.emit("failure", reason=reason)
                return None

    def replace(self, owner, name, replacement):
        setattr(owner, name, replacement)
        self.hooks.append((owner, name, replacement))

    def pages(self, operation, field, group, items):
        integer(len(items), 0, 65536)
        for offset in range(0, max(1, len(items)), PAGE):
            self.emit("page", operation=operation, field=field, group=group,
                      offset=offset, total=len(items), items=items[offset:offset + PAGE])

    def harvest(self):
        with self.condition:
            if self.failed or self.sealed:
                return
            for op in self.operations.values():
                pending = op.get("event")
                if (pending is None or op.get("device_complete")
                        or not op["returned"] or not op["success"]):
                    continue
                backend, event = pending
                if self.event_guard is None:
                    raise ValueError("event backend has not been admitted")
                self.event_guard(backend, event)
                ready = backend.query_event(event)
                if type(ready) is not bool:
                    raise ValueError("event query did not return a boolean")
                if not ready:
                    continue
                # Query happens before any weak operand is dereferenced.
                for token in op["allocations"]:
                    self.lifetime.validate(token)
                op["device_complete"] = True
                op.pop("event")
                self.emit("device_complete", operation=op["id"])
            self.condition.notify_all()

    def begin(self, direction, key, engine, owner):
        with self.condition:
            self.harvest()
            if self.closed or self.failed:
                raise ValueError("operation outside open journal")
            if engine not in self.registrations or len(self.operations) >= 1024:
                raise ValueError("unregistered operation or operation budget")
            integer(key.world_size, 1, 64)
            integer(key.worker_id, 0, key.world_size - 1)
            integer(key.num_kv_readers, 1, 64)
            if key.request_configs is not None:
                raise ValueError("request-scoped configuration is outside the observer contract")
            for value in (key.request_id, key.model_name):
                if type(value) is not str or not 0 < len(value.encode()) <= 256:
                    raise ValueError("unbounded request identity")
            op = {"id": len(self.operations) + 1, "direction": direction, "engine": engine,
                  "start": key.start, "end": key.end, "allocations": set(),
                  "callback": False, "delivered": False, "device_complete": False, "finalized": False,
                  "returned": False, "success": False, "recorded": False, "keys": None, "raw": None}
            self.operations[op["id"]] = op
            self.active += 1
            self.emit("operation", operation=op["id"], direction=direction,
                      session=key.request_id, engine=engine, rank=key.worker_id,
                      world_size=key.world_size, model=key.model_name, salt=key.cache_salt,
                      start=key.start, end=key.end, num_kv_readers=key.num_kv_readers)
            return op

    def selected(self, values):
        op = OP.get()
        if op is None:
            raise ValueError("unbound key selection")
        if op["keys"] is not None:
            raise ValueError("duplicate key selection")
        groups = values["obj_keys_per_obj_group"]
        integer(len(groups), 1, 64)
        op["keys"] = []
        for group, keys in enumerate(groups):
            integer(len(keys), 1, 1024)
            row = [key_value(key) for key in keys]
            op["keys"].append(row)
            self.pages(op["id"], "keys", group, row)

    def raw_blocks(self, context, blocks):
        op = OP.get()
        if op is None or op["raw"] is not None:
            raise ValueError("unbound or duplicate raw blocks")
        registration = self.registrations[op["engine"]]
        if len(blocks) != len(registration["kernel_groups"]):
            raise ValueError("raw kernel inventory mismatch")
        op["raw"] = []
        for kid, values in enumerate(blocks):
            integer(len(values), 1, 65536)
            nb = registration["kernel_groups"][kid]["shape"]["nb"]
            raw = [integer(value, 0, nb - 1) for value in values]
            op["raw"].append(raw)
            self.pages(op["id"], "blocks", kid, raw)
        chunk = registration["lmcache_tokens_per_chunk"]
        chunks = len(op["keys"][0])
        for group, obj in enumerate(registration["object_groups"]):
            excluded = []
            window = obj["num_chunks_in_sw"]
            skip = max(0, chunks - window) if op["direction"] == "retrieve" and window > 0 else 0
            for index in range(chunks):
                all_null = True
                for kid in obj["kernel_group_ids"]:
                    bpc = chunk // registration["kernel_groups"][kid]["tokens_per_block"]
                    if len(op["raw"][kid]) != chunks * bpc:
                        raise ValueError("raw block coverage differs from full selected range")
                    all_null &= not any(op["raw"][kid][index * bpc:(index + 1) * bpc])
                if index < skip or (op["direction"] == "store" and all_null):
                    excluded.append({"index": index, "reason": "window_prefix" if index < skip else "all_null"})
            if excluded:
                self.pages(op["id"], "exclusions", group, excluded)

    def recorded_event(self, values):
        op = OP.get()
        if op is None or op["recorded"] or op["device_complete"]:
            raise ValueError("unbound or duplicate source event")
        if not op.get("transfer_key") or values.get("transfer_key") != op["transfer_key"]:
            raise ValueError("completion event is not bound to the executed transfer generation")
        backend, event = values["event_backend"], values["event"]
        if self.event_guard is None:
            raise ValueError("event backend has not been admitted")
        self.event_guard(backend, event)
        previous = self.event_owners.get(id(event))
        if previous is not None and previous() is event:
            raise ValueError("source event reused across transfer generations")
        # Keep history weakly: stale event reuse must fail without extending
        # device object lifetimes after the operation's completion.
        self.event_owners[id(event)] = weakref.ref(event)
        op["recorded"] = True
        op["event"] = (backend, event)
        self.emit("event_recorded", operation=op["id"])

    def submit(self, original, stream, kind, keys):
        op = OP.get()
        def observe_submission():
            if op is None or op["callback"] or not op["recorded"]:
                raise ValueError("unbound callback or missing recorded event")
            expected = "finish_write" if op["direction"] == "store" else "finish_read_prefetched"
            if kind != expected:
                raise ValueError("callback direction mismatch")
            integer(len(keys), 1, 65536)
            op["callback_keys"] = [key_value(key) for key in keys]
            op["callback_kind"] = kind
            op["callback"] = True
            self.pages(op["id"], "callback_keys", 0, op["callback_keys"])
            self.emit("callback_submitted", operation=op["id"], callback=kind)
            return True
        if self.observe(observe_submission):
            return original(stream, f"grill_kv_{kind}_v2", (op["id"], keys))
        return original(stream, kind, keys)

    def handler(self, kind, original):
        @functools.wraps(original)
        def call(payload):
            operation, keys = payload
            op = self.operations.get(operation)
            def lookup():
                if (op is None or not op["callback"] or op["delivered"]
                        or op["callback_kind"] != kind
                        or op["callback_keys"] != [key_value(key) for key in keys]):
                    raise ValueError("unmatched callback delivery")
                return op
            observed = self.observe(lookup)
            token = CALLBACK.set(observed)
            try:
                result = original(keys)
                if observed is not None:
                    observed["delivered"] = True
                    self.emit("callback_delivered", operation=operation, callback=kind)
                return result
            except BaseException:
                self.emit("failure", reason="callback_source_exception")
                raise
            finally:
                CALLBACK.reset(token)
                with self.condition:
                    self.condition.notify_all()
        return call

    def finalized(self, values):
        op = CALLBACK.get()
        if (op is None or op["direction"] != "store" or not op["callback"]
                or op.get("callback_kind") != "finish_write" or op.get("finalized")):
            raise ValueError("unbound, wrong-direction or duplicate storage finalization")
        successful = [key_value(key) for key in values["successful_keys"]]
        failed = [key_value(key) for key in values["failed_keys"]]
        results = [{"key": key, "success": True} for key in successful]
        results.extend({"key": key, "success": False} for key in failed)
        self.pages(op["id"], "finalized", 0, results)
        fields = ("hash", "model", "kv_rank", "group", "salt")
        expected = op["callback_keys"]
        expected_keys = {tuple(key[field] for field in fields) for key in expected}
        successful_keys = {tuple(key[field] for field in fields) for key in successful}
        if (failed or not expected or len(expected_keys) != len(expected)
                or len(successful_keys) != len(successful)
                or successful_keys != expected_keys):
            self.emit("failure", reason="storage_finalization_incomplete")
            return
        self.emit("finalized", operation=op["id"])
        op["finalized"] = not self.failed

    def seal(self, timeout=5.0):
        if not 0 <= timeout <= 60:
            raise ValueError("invalid drain timeout")
        with self.condition:
            self.closed = True
            deadline = time.monotonic() + timeout
            def pending():
                return len(self.worker_pending) + sum(
                    not (op["returned"] and op["success"] and op["callback"]
                         and op["delivered"] and op["device_complete"]
                         and (op.get("direction") == "retrieve"
                              or (op.get("direction") == "store" and op.get("finalized") is True)))
                    for op in self.operations.values())
            while (self.active or pending()) and not self.failed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.condition.wait(remaining)
            if self.role == "cache":
                if self.cache_seal_guard is None:
                    self.emit("failure", reason="cache_lifecycle_not_installed")
                else:
                    self.observe(self.cache_seal_guard)
            intact = bool(self.hooks) and all(getattr(owner, name) is fn for owner, name, fn in self.hooks)
            complete = not self.failed and intact and not self.active and not pending()
            self.emit("fence", admission_closed=True, active=self.active, pending=pending(),
                      operations=len(self.operations), complete=complete, hooks_intact=intact)
            self.sealed = True
            os.close(self.fd)
            return complete and not self.failed


def install_cache(journal, modules, sources):
    return load_adjacent("kv-transfer-v2.py").install_cache(journal, modules, sources)


def install_worker(journal, modules, sources):
    return load_adjacent("kv-transfer-v2.py").install_worker(journal, modules, sources)


def install_lmcache(journal):
    hooks = load_adjacent("kv-transfer-v2.py")
    modules, sources = hooks.loaded_sources("cache")
    return hooks.install_cache(journal, modules, sources)


def install_vllm_worker(journal):
    hooks = load_adjacent("kv-transfer-v2.py")
    modules, sources = hooks.loaded_sources("worker")
    return hooks.install_worker(journal, modules, sources)


def install_frontend(journal, engine_class, runtime_request, source):
    return load_adjacent("kv-transfer-v2.py").install_frontend(journal, engine_class, runtime_request, source)


class WorkerExtension:
    def __new__(cls, *args, **kwargs):
        directory = Path(os.environ["THEGRILL_KV_V2_WORKER_DIR"])
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError("explicit private worker journal directory required")
        identity = process()
        path = directory / f"worker-{identity['boot_id']}-{identity['pid']}-{identity['start_ticks']}.jsonl"
        journal = Journal(path, role="worker", max_window_us=int(os.environ["THEGRILL_KV_V2_WINDOW_US"]))
        try:
            install_vllm_worker(journal)
            instance = super().__new__(cls)
            instance._grill_kv_v2_journal = journal
            return instance
        except BaseException:
            journal.emit("failure", reason="worker_installation_failed")
            journal.seal(0)
            raise

    def grill_kv_v2_seal(self, timeout=5.0):
        journal = self._grill_kv_v2_journal
        complete = journal.seal(timeout)
        return {"complete": complete, "process": journal.process, "incarnation": journal.incarnation}
