#!/usr/bin/env python3
"""Opt-in pinned LMCache source observations; importing this file is stdlib-only.

Installation is an operator action in each owning process, before serving work.
No launcher, receiver, service control, device probe, or GPU synchronization.
The callback dispatcher already ordered behind the copy is the completion fence.
"""
import ast
import contextvars
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import stat
import threading
import math
import time

SOURCE = "vllm-487ecf187-lmcache-3e11b8ed-kv-copy-v1"
SCOPE = "lmcache-driven-full-attention-l1-engine-restart"
TRANSFER = "lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py"
STORAGE = "lmcache/v1/distributed/storage_manager.py"
COMPLETION = "lmcache/v1/multiprocess/native_completion.py"
PINS = {
    TRANSFER: "7e42b2c1caeb88fd6a2416e0e193451bf83a8413daa3c8d00f31ff9b2a580da7",
    STORAGE: "a2cdf1277fbc4bf49aaf687d057d44e617ef78457e9a4e83e8ea866ab7825384",
    COMPLETION: "58c69916f74b72afb5a755cadf4a87164f34597858a6489509fcbe11e9b866d0",
    "lmcache/v1/gpu_connector/gpu_ops.py": "de0bdd322b8eb8a4078c9a83d8ef8d6bf5e62e67092ab3a07a07f04c69d9fb90",
    "lmcache/v1/distributed/api.py": "cfa411afbebb06c29d21a45c40a7f964748d749d3d1c928215dfaefcf0ef4609",
    "lmcache/v1/multiprocess/custom_types.py": "6e9e19fea6682cdf0c8837c1ac00d77c06413e2a4b740b34c93698d958e05dd7",
}
ASYNC = "vllm/v1/engine/async_llm.py"
ASYNC_SHA256 = "bceed0b3f5f0c834fef79525f2462a092f082390f0070526280abc95945837dd"
WORKER = "lmcache/integration/vllm/vllm_multi_process_adapter.py"
WORKER_SHA256 = "6fe6598e77e7872829fd0f4b0dc0e23ed2c79a3294c61a69045f7e89bae31ecb"
WORKER_BASE = "vllm/v1/worker/worker_base.py"
WORKER_BASE_SHA256 = "7da44338c2645ebf03d23394e452b31a8e3da1011fd1b42fcfcccfe99551b3fe"
WORKER_ACTIVE = contextvars.ContextVar("grill_kv_worker_active", default=False)
OP = contextvars.ContextVar("grill_kv_operation", default=None)
CALLBACK = contextvars.ContextVar("grill_kv_callback", default=None)


def bounded_read(path, cap=4 * 1024 * 1024):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("source/journal must be a regular file")
        raw = os.read(fd, cap + 1)
        if len(raw) > cap:
            raise ValueError("source/journal exceeds bound")
        return raw
    finally:
        os.close(fd)


def encoded(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode()


def process():
    pid = os.getpid()
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "start_ticks": int(fields[19]),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def key_value(key):
    return {"hash": key.chunk_hash.hex(), "model": key.model_name, "kv_rank": key.kv_rank,
            "group": key.object_group_id, "salt": key.cache_salt}


class Journal:
    """Finite synchronous records, with an explicit terminal delivery fence.

    Each operation is registered before the selected source body runs. Missing
    native callbacks, serialization/drop/handler failures cannot produce a clean
    drain: pending operation accounting is independent of EventBus sequences.
    Observation errors never change backend return values or swallow its errors.
    """
    def __init__(self, path, *, role, max_events=10000, max_bytes=8 * 1024 * 1024,
                 max_window_us=60000000):
        if role not in ("cache", "frontend", "worker"):
            raise ValueError("unsupported journal role")
        if not 8 <= max_events <= 10000 or not 4096 <= max_bytes <= 16 * 1024 * 1024:
            raise ValueError("invalid journal bounds")
        if not 1000 <= max_window_us <= 3600000000:
            raise ValueError("invalid journal window")
        self.fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        self.role, self.process = role, process()
        self.incarnation = os.urandom(16).hex()
        self.max_events, self.max_bytes = max_events, max_bytes
        self.max_window_us = max_window_us
        self.origin = time.monotonic_ns()
        self.sequence = self.bytes = self.operation_count = self.active = 0
        self.failed = self.closed = self.sealed = False
        self.pending = {}
        self.operations = {}
        self.condition = threading.Condition(threading.RLock())
        self.hooks = []
        self.emit("start", source=SOURCE, scope=SCOPE, role=role, process=self.process,
                  producer_sha256=hashlib.sha256(bounded_read(__file__)).hexdigest(),
                  max_events=max_events, max_bytes=max_bytes, max_window_us=max_window_us)

    def emit(self, kind, **data):
        with self.condition:
            if self.failed or self.sealed:
                return
            elapsed = (time.monotonic_ns() - self.origin) // 1000
            if (os.getpid() != self.process["pid"] or self.sequence >= self.max_events - 1
                    or elapsed > self.max_window_us):
                kind, data = "failure", {"reason": "process_or_finite_budget"}
            row = {"version": 1, "sequence": self.sequence, "incarnation": self.incarnation,
                   "offset_us": elapsed, "event": {"kind": kind, **data}}
            try:
                raw = encoded(row) + b"\n"
                if len(raw) > 65536 or self.bytes + len(raw) > self.max_bytes - 1024:
                    row["event"] = {"kind": "failure", "reason": "byte_budget"}
                    raw, kind = encoded(row) + b"\n", "failure"
                view = memoryview(raw)
                while view:
                    n = os.write(self.fd, view)
                    if n <= 0:
                        raise OSError("short write")
                    view = view[n:]
                os.fsync(self.fd)
                self.sequence += 1
                self.bytes += len(raw)
                self.failed = kind == "failure"
            except (OSError, ValueError, TypeError):
                self.failed = True

    def observe(self, callback, *args):
        try:
            return callback(*args)
        except Exception:
            self.emit("failure", reason="observation_exception")
            return None

    def replace(self, owner, name, replacement):
        setattr(owner, name, replacement)
        self.hooks.append((owner, name, replacement))

    def seal(self, timeout=5.0):
        if not 0 <= timeout <= 60:
            raise ValueError("invalid finite drain timeout")
        with self.condition:
            self.closed = True
            deadline = time.monotonic() + timeout
            while (self.active or self.pending) and not self.failed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.condition.wait(remaining)
            intact = bool(self.hooks) and all(getattr(o, n) is f for o, n, f in self.hooks)
            complete = (not self.failed and intact and not self.active and not self.pending
                        and all(o.get("terminal") for o in self.operations.values()))
            self.emit("fence", admission_closed=True, active=self.active,
                      pending=len(self.pending), operations=self.operation_count,
                      complete=complete, hooks_intact=intact)
            self.sealed = True
            os.close(self.fd)
            return complete and not self.failed

    def begin(self, kind, key, instance):
        with self.condition:
            if self.closed and not self.sealed:
                self.emit("failure", reason="source_operation_arrived_during_drain")
            if self.closed or self.failed:
                return None
            if len(self.operations) >= min(self.max_events // 2, 1024):
                self.emit("failure", reason="operation_budget")
                return None
            if (not isinstance(key.request_id, str) or not 0 < len(key.request_id) <= 256
                    or not isinstance(key.model_name, str) or not 0 < len(key.model_name) <= 256
                    or not isinstance(key.cache_salt, str) or len(key.cache_salt) > 128):
                self.emit("failure", reason="request_identity_bound")
                return None
            self.operation_count += 1
            op = {"id": self.operation_count, "kind": kind, "session": key.request_id,
                  "engine": instance, "rank": key.worker_id, "world_size": key.world_size,
                  "model": key.model_name, "salt": key.cache_salt,
                  "start": key.start, "end": key.end, "expected": [], "copied": [],
                  "supported": True, "terminal": False, "submitted": False}
            self.operations[op["id"]] = op
            self.active += 1
            self.emit("operation", operation=op["id"], direction=kind, session=op["session"],
                      engine=instance, rank=key.worker_id, world_size=key.world_size,
                      model=key.model_name, salt=key.cache_salt, start=key.start, end=key.end)
            return op

    def selected(self, values):
        op = OP.get()
        if op is None:
            return
        ctx = values["cache_context"]
        groups = values["obj_keys_per_obj_group"]
        manager = ctx.kv_layer_groups_manager
        op["keys"] = groups
        op["chunk_size"] = values["self"]._ctx.chunk_size
        op["supported"] = (0 < len(groups) <= 64 and 0 < len(groups[0]) <= 1024
                           and values.get("skip_first_n_tokens", 0) == 0
                           and op["chunk_size"] > 0
                           and op["end"] - op["start"] == len(groups[0]) * op["chunk_size"]
                           and all(len(keys) == len(groups[0]) for keys in groups)
                           and all(manager.get_attn_desc().is_full_attention(i)
                                   for i in range(len(groups))))
        if not op["supported"]:
            self.emit("unsupported", operation=op["id"], reason="window_skip_or_group_bounds")
            return
        packed_rank = ((op["world_size"] << 24) | (op["rank"] << 16)
                       | (op["world_size"] << 8) | op["rank"])
        for group, keys in enumerate(groups):
            kernel_groups = manager.object_groups[group].kernel_group_indices
            layouts = [ctx.get_kernel_group_shape_dtype(op["chunk_size"], kg) for kg in kernel_groups]
            size = sum(math.prod(shape) * dtype.itemsize for shape, dtype in layouts)
            if not kernel_groups or size <= 0:
                op["supported"] = False
            for index, key in enumerate(keys):
                value = key_value(key)
                if (value["kv_rank"] != packed_rank or value["group"] != group
                        or value["model"] != op["model"] or value["salt"] != op["salt"]):
                    op["supported"] = False
                op["expected"].append({"key": value, "start": op["start"] + index * op["chunk_size"],
                                       "end": op["start"] + (index + 1) * op["chunk_size"],
                                       "bytes": size})
        self.emit("selected", operation=op["id"], extents=op["expected"], supported=op["supported"])

    def transfer(self, original, *args, **kwargs):
        op = OP.get()
        values = self.observe(lambda: inspect.signature(original).bind(*args, **kwargs).arguments)
        before = len(op.get("native", [])) if op else 0
        result = original(*args, **kwargs)
        if op is not None and values is not None:
            self.observe(self.copied, op, values, before)
        return result

    def native(self, values):
        op = OP.get()
        if op is not None:
            if len(op.setdefault("native", [])) >= 64:
                self.emit("failure", reason="native_group_budget")
            else:
                op["native"].append(values["object_group_id"])

    def copied(self, op, values, before):
        group = values["object_group_id"]
        objs = values["memory_objs"]
        # A successful helper return without an actual nonempty native execution
        # (empty reservations, skipped batches, tensorless cleanup) is not a copy.
        native = op.get("native", [])[before:]
        if (not op["supported"] or not objs
                or native != [group] or values["skip_first_n_tokens"] != 0):
            op["supported"] = False
            return
        keys = op["keys"][group]
        if len(keys) != len(objs):
            op["supported"] = False
            return
        extents = []
        for index, (key, mo) in enumerate(zip(keys, objs, strict=True)):
            if mo is None:
                op["supported"] = False
                continue
            size = mo.get_size()
            if type(size) is not int or size <= 0:
                op["supported"] = False
                return
            extents.append({"key": key_value(key), "start": op["start"] + index * op["chunk_size"],
                            "end": op["start"] + (index + 1) * op["chunk_size"], "bytes": size})
        op["copied"].extend(extents)
        self.emit("copy", operation=op["id"], extents=extents, outcome="enqueued",
                  mode="native-object-group")

    def worker_topology(self, worker):
        strategy = worker.parallel_strategy
        if (strategy.mla_only is not False or strategy.n_servers != 1
                or strategy.vllm_world_size != worker.world_size
                or strategy.vllm_worker_id != worker.worker_id):
            self.emit("failure", reason="unsupported_worker_parallel_strategy")

    def worker_submission(self, values, direction):
        if not WORKER_ACTIVE.get():
            return
        worker, key = values["self"], values["key"]
        strategy = worker.parallel_strategy
        self.emit("worker_submission", direction=direction, session=key.request_id,
                  engine=worker.instance_id, rank=key.worker_id, world_size=key.world_size,
                  model=key.model_name, salt=key.cache_salt, start=key.start, end=key.end,
                  mla_only=strategy.mla_only, n_servers=strategy.n_servers,
                  vllm_world_size=strategy.vllm_world_size, vllm_worker_id=strategy.vllm_worker_id)

    def worker_store(self, values):
        self.worker_submission(values, "store")

    def worker_retrieve(self, values):
        self.worker_submission(values, "retrieve")

    def submit(self, original, stream, kind, keys):
        op = OP.get()
        if op is not None:
            def record():
                with self.condition:
                    if kind not in ("finish_write", "finish_read_prefetched") or not 0 < len(keys) <= 65536:
                        raise ValueError("unsupported callback kind or key bound")
                    if op["id"] in self.pending:
                        raise ValueError("duplicate operation callback")
                    op["callback"] = kind
                    op["callback_keys"] = [key_value(k) for k in keys]
                    self.pending[op["id"]] = op
                    op["submitted"] = True
                    self.emit("callback_submitted", operation=op["id"], callback=kind,
                              keys=op["callback_keys"])
                    return True
            if self.observe(record):
                # Preserve the original handler and exact keys; add a lossless
                # operation ID through the existing native opaque payload.
                return original(stream, f"grill_kv_{kind}_v1", (op["id"], keys))
        if not self.sealed:
            self.emit("failure", reason="unbound_callback_submission")
        return original(stream, kind, keys)

    def finalized(self, values):
        op = CALLBACK.get()
        if op is not None:
            results = [{"key": key_value(k), "success": True} for k in values["successful_keys"]]
            results.extend({"key": key_value(k), "success": False} for k in values["failed_keys"])
            op["finalized"] = results
            self.emit("finalized", operation=op["id"], results=results)

    def handler(self, kind, original):
        @functools.wraps(original)
        def call(payload):
            operation_id, keys = payload
            def lookup():
                with self.condition:
                    op = self.pending.get(operation_id)
                    if (op is None or op["callback"] != kind
                            or op["callback_keys"] != [key_value(k) for k in keys]):
                        raise ValueError("unmatched native callback delivery")
                    return op
            op = self.observe(lookup)
            token = CALLBACK.set(op)
            try:
                result = original(keys)
                if op is not None:
                    def completed():
                        self.emit("device_complete", operation=op["id"], callback=kind,
                                  keys=[key_value(k) for k in keys])
                        op["terminal"] = True
                    self.observe(completed)
                return result
            finally:
                CALLBACK.reset(token)
                with self.condition:
                    if op is not None:
                        self.pending.pop(operation_id, None)
                    self.condition.notify_all()
        return call


def instrument(raw, name, namespace, after_assignment=None, after_call=None):
    """Compile one hash-verified source body with a concrete post-call observer.

    No source file is rewritten. Backend imports/decorators are not executed by
    the CPU harness. Production uses the existing module globals and decorators.
    """
    tree = ast.parse(raw)
    candidates = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                  and n.name == name]
    if len(candidates) != 1:
        raise ValueError("ambiguous pinned source function")
    node = candidates[0]
    count = 0

    class Observe(ast.NodeTransformer):
        def visit_Assign(self, statement):
            nonlocal count
            if after_assignment and any(isinstance(t, ast.Name) and t.id == after_assignment[0]
                                        for t in statement.targets):
                count += 1
                return [statement, ast.parse(f"_grill_kv.observe(_grill_kv.{after_assignment[1]}, locals())").body[0]]
            return statement

        def visit_Expr(self, statement):
            nonlocal count
            call = statement.value
            if (after_call and isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                    and call.func.id == after_call[0]):
                count += 1
                return [statement, ast.parse(f"_grill_kv.observe(_grill_kv.{after_call[1]}, locals())").body[0]]
            return statement

    node = Observe().visit(node)
    if count != 1:
        raise ValueError("pinned instrumentation boundary changed")
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    local = {}
    exec(compile(module, "<grill-pinned-kv-hook>", "exec"), namespace, local)
    return local[name]


def install_cache(journal, transfer, storage, completion, sources):
    """Before construction, atomically install the pinned cache-process hooks.

    This explicit low-level boundary also serves the CPU source harness.
    Native operators use install_lmcache(), which verifies loaded module paths.
    """
    import gc
    if journal.role != "cache" or journal.hooks or "_grill_kv" in transfer.__dict__:
        raise ValueError("wrong role or duplicate installation")
    cls = transfer.LMCacheDrivenTransferModule
    if any(isinstance(obj, (cls, completion.DeviceHostFuncDispatcher)) for obj in gc.get_objects()):
        raise ValueError("late installation: transfer module or dispatcher already exists")
    for path, expected in PINS.items():
        if hashlib.sha256(sources[path]).hexdigest() != expected:
            raise ValueError(f"source pin mismatch: {path}")
    # Compile everything first. No replaced method or handler is visible on a
    # failed compilation; installation is single-threaded before construction.
    replacements = []
    transfer.__dict__["_grill_kv"] = journal
    storage.__dict__["_grill_kv"] = journal
    try:
        for name in ("store", "retrieve"):
            fn = instrument(sources[TRANSFER], name, transfer.__dict__,
                            after_assignment=("obj_keys_per_obj_group", "selected"))
            def wrapper(original, direction):
                @functools.wraps(original)
                def call(self, key, instance_id, *args, **kwargs):
                    op = journal.observe(journal.begin, direction, key, instance_id)
                    token = OP.set(op)
                    outcome = False
                    try:
                        result = original(self, key, instance_id, *args, **kwargs)
                        outcome = result[1] is True
                        return result
                    finally:
                        OP.reset(token)
                        if op is not None:
                            with journal.condition:
                                journal.emit("result", operation=op["id"], success=outcome,
                                             supported=op["supported"])
                                if not op["submitted"]:
                                    op["terminal"] = True
                                journal.active -= 1
                                journal.condition.notify_all()
                return call
            replacements.append((cls, name, wrapper(fn, name)))
        native = instrument(sources[TRANSFER], "_run_object_group_transfer_plan", transfer.__dict__,
                            after_call=("execute_object_group_transfer", "native"))
        replacements.append((transfer, "_run_object_group_transfer_plan", native))
        helper = transfer.transfer_kv_per_object_group
        replacements.append((transfer, "transfer_kv_per_object_group",
                             functools.wraps(helper)(lambda *a, **k: journal.transfer(helper, *a, **k))))
        submit = transfer.submit_callback_to_stream
        replacements.append((transfer, "submit_callback_to_stream",
                             functools.wraps(submit)(lambda *a, **k: journal.submit(submit, *a, **k))))
        finish = instrument(sources[STORAGE], "finish_write", storage.__dict__,
                            after_assignment=("failed_keys", "finalized"))
        replacements.append((storage.StorageManager, "finish_write", finish))
        register = completion.DeviceHostFuncDispatcher.register
        @functools.wraps(register)
        def observed_register(dispatcher, kind, handler, payload_type):
            @functools.wraps(handler)
            def unobserved(keys):
                if not journal.sealed:
                    journal.emit("failure", reason="unbound_native_callback_delivery")
                return handler(keys)
            result = register(dispatcher, kind, unobserved, payload_type)
            if kind in ("finish_write", "finish_read_prefetched"):
                journal.observe(lambda: register(dispatcher, f"grill_kv_{kind}_v1",
                                                 journal.handler(kind, handler), tuple[int, payload_type]))
            return result
        replacements.append((completion.DeviceHostFuncDispatcher, "register", observed_register))
    except BaseException:
        transfer.__dict__.pop("_grill_kv", None)
        storage.__dict__.pop("_grill_kv", None)
        raise
    for owner, name, replacement in replacements:
        journal.replace(owner, name, replacement)
    journal.emit("installed", pins=PINS, scope=SCOPE)


def install_lmcache(journal):
    """Operator-invoked only, in the cache server before transfer construction."""
    import importlib
    modules = {path: importlib.import_module(path[:-3].replace("/", ".")) for path in PINS}
    sources = {path: bounded_read(inspect.getsourcefile(module)) for path, module in modules.items()}
    install_cache(journal, modules[TRANSFER], modules[STORAGE], modules[COMPLETION], sources)


def install_frontend(journal, engine_class, runtime_request, source):
    """Observe actual _add_request EngineCoreRequest IDs, including child IDs.

    runtime_request is startup-runtime.REQUEST, not a caller-supplied ID map.
    Call before frontend serving. No OpenAI response-ID inference is involved.
    """
    import gc
    if (journal.role != "frontend" or journal.hooks
            or getattr(engine_class._add_request, "_grill_kv_frontend", False)
            or any(isinstance(obj, engine_class) for obj in gc.get_objects())
            or hashlib.sha256(source).hexdigest() != ASYNC_SHA256):
        raise ValueError("frontend installation role/source/duplicate/late mismatch")
    original = engine_class._add_request
    @functools.wraps(original)
    async def observed(engine, request, prompt, parent_req, index, queue):
        context = runtime_request.get()
        with journal.condition:
            if journal.closed and not journal.sealed:
                journal.emit("failure", reason="frontend_admission_arrived_during_drain")
            tracking = not journal.closed and not journal.failed
            if tracking:
                journal.active += 1
        try:
            result = await original(engine, request, prompt, parent_req, index, queue)
            if tracking:
                def binding():
                    if context is None or not context["active"]:
                        journal.emit("failure", reason="engine_admission_outside_runtime_request")
                    else:
                        journal.emit("binding", request=context["id"], engine_request=request.request_id,
                                      parent=None if parent_req is None else parent_req.request_id,
                                      index=index)
                journal.observe(binding)
            return result
        finally:
            if tracking:
                with journal.condition:
                    journal.active -= 1
                    journal.condition.notify_all()
    observed._grill_kv_frontend = True
    journal.replace(engine_class, "_add_request", observed)
    journal.emit("installed", pins={ASYNC: ASYNC_SHA256}, scope=SCOPE)


def install_worker(journal, module, source):
    """Install before any LMCacheMPWorkerAdapter exists in an engine worker."""
    import gc
    cls = module.LMCacheMPWorkerAdapter
    if (journal.role != "worker" or journal.hooks or "_grill_kv" in module.__dict__
            or hashlib.sha256(source).hexdigest() != WORKER_SHA256
            or any(isinstance(obj, cls) for obj in gc.get_objects())):
        raise ValueError("worker source/role mismatch or duplicate/late installation")
    module.__dict__["_grill_kv"] = journal
    replacements = []
    try:
        for direction in ("store", "retrieve"):
            name = f"submit_{direction}_request"
            fn = instrument(source, name, module.__dict__,
                            after_assignment=("future", f"worker_{direction}"))
            def wrapper(original):
                @functools.wraps(original)
                def call(*args, **kwargs):
                    with journal.condition:
                        if journal.closed and not journal.sealed:
                            journal.emit("failure", reason="worker_submission_arrived_during_drain")
                        tracking = not journal.closed and not journal.failed
                        if tracking:
                            journal.active += 1
                    if tracking:
                        journal.observe(journal.worker_topology, args[0] if args else kwargs.get("self"))
                    token = WORKER_ACTIVE.set(tracking)
                    try:
                        return original(*args, **kwargs)
                    finally:
                        WORKER_ACTIVE.reset(token)
                        if tracking:
                            with journal.condition:
                                journal.active -= 1
                                journal.condition.notify_all()
                return call
            replacements.append((cls, name, wrapper(fn)))
    except BaseException:
        module.__dict__.pop("_grill_kv", None)
        raise
    for owner, name, replacement in replacements:
        journal.replace(owner, name, replacement)
    journal.emit("installed", pins={WORKER: WORKER_SHA256}, scope=SCOPE)


def install_vllm_worker(journal):
    """Operator-invoked only; no backend imports at module-import time."""
    import importlib
    module = importlib.import_module(WORKER[:-3].replace("/", "."))
    install_worker(journal, module, bounded_read(inspect.getsourcefile(module)))


class WorkerExtension:
    """Explicit vLLM worker_extension_cls; works in independently spawned workers.

    Resolve this class through the existing worker extension configuration.
    __new__ runs before the original worker __init__, not in the frontend.
    Sealing is one method on vLLM's existing collective_rpc, not a new endpoint.
    """
    def __new__(cls, *args, **kwargs):
        import sys
        base = sys.modules.get("vllm.v1.worker.worker_base")
        if base is None or hashlib.sha256(bounded_read(base.__file__)).hexdigest() != WORKER_BASE_SHA256:
            raise ValueError("unsupported worker construction source")
        directory = Path(os.environ["THEGRILL_KV_WORKER_DIR"])
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError("explicit private worker journal directory required")
        identity = process()
        name = f"worker-{identity['boot_id']}-{identity['pid']}-{identity['start_ticks']}.jsonl"
        window = int(os.environ["THEGRILL_KV_WINDOW_US"])
        journal = Journal(directory / name, role="worker", max_window_us=window)
        try:
            install_vllm_worker(journal)
            instance = super().__new__(cls)
            instance._grill_kv_journal = journal
            return instance
        except BaseException:
            journal.emit("failure", reason="worker_extension_installation_failed")
            journal.seal(timeout=0)
            raise

    def grill_kv_seal(self, timeout=5.0):
        """Close only observation admission; do not stop worker/cache work."""
        journal = self._grill_kv_journal
        complete = journal.seal(timeout=timeout)
        return {"complete": complete, "process": journal.process,
                "incarnation": journal.incarnation}
