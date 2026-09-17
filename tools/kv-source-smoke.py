#!/usr/bin/env python3
"""Offline CPU source-boundary smoke. Never imports a serving/device backend.

Runs the installed hook around pinned store/retrieve, native-plan construction,
StorageManager.finish_write and dispatcher bodies. Only dependency boundaries
(device buffers/kernels, allocator, native callback queue, IPC) are controlled.
No generated receipt is native evidence. Source files stay outside the repo.
"""
import argparse
import ast
import asyncio
import contextlib
import contextvars
import dataclasses
import enum
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import types


def load_hook():
    spec = importlib.util.spec_from_file_location("kv_journal", Path(__file__).with_name("kv-journal.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_function(raw, name, namespace, owner=None):
    tree = ast.parse(raw)
    if owner is not None:
        tree = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == owner)
    nodes = [node for node in ast.walk(tree)
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    if len(nodes) != 1:
        raise ValueError(f"ambiguous source body: {name}")
    node = nodes[0]
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), "<pinned-cpu-source>", "exec"), namespace)
    return namespace[name]


def read_sources(hook, verified, additional):
    result = {}
    for path, digest in {**hook.PINS, hook.ASYNC: hook.ASYNC_SHA256, hook.WORKER: hook.WORKER_SHA256,
                         hook.WORKER_BASE: hook.WORKER_BASE_SHA256}.items():
        root = verified / "lmcache" if path.startswith("lmcache/") else additional
        candidate = root / path
        if not candidate.exists():
            candidate = additional / path
        if not candidate.exists() and path == hook.WORKER_BASE:
            candidate = additional / "vllm" / path
        raw = hook.bounded_read(candidate)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"CPU source pin mismatch: {path}")
        result[path] = raw
    return result


def source_class(raw, name, namespace):
    nodes = [n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef) and n.name == name]
    if len(nodes) != 1:
        raise AssertionError(f"source class count: {name}")
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                            nodes[0]], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), f"<pinned-{name}>", "exec"), namespace)
    return namespace[name]


def key_schema(sources):
    """Execute the pinned key schemas and converter, never a fixture rank encoding."""
    module = types.ModuleType("cpu_key_schema")
    module.__dict__.update(dataclass=dataclasses.dataclass, field=dataclasses.field)
    sys.modules[module.__name__] = module
    for name, path in (
        ("ObjectKey", "lmcache/v1/distributed/api.py"),
        ("IPCCacheServerKey", "lmcache/v1/multiprocess/custom_types.py"),
    ):
        source_class(sources[path], name, module.__dict__)
    module.convert = source_function(sources["lmcache/v1/distributed/api.py"], "ipc_key_to_object_keys",
                                     module.__dict__)
    return module


class Buffer:
    def __init__(self, pointers, data=b"\x00" * 8):
        self.data = bytearray(data)
        self.nbytes = len(data)
        self.pointer = len(pointers) + 1
        pointers[self.pointer] = self

    def data_ptr(self):
        return self.pointer

    def numel(self):
        return len(self.data)


class Memory:
    def __init__(self, buffer, tensorless=False):
        self.buffer = buffer
        self.raw_tensor = None if tensorless else buffer
        self.data_ptr = buffer.pointer
        self.meta = types.SimpleNamespace(address=0)

    def get_size(self):
        return self.buffer.nbytes


def environment(hook, sources, case, groups=1, chunks=1):
    ns = types.SimpleNamespace
    pointers, queue, stored = {}, [], {}
    schema = key_schema(sources)
    logger = ns(debug=lambda *a: None, info=lambda *a: None, warning=lambda *a: None,
                error=lambda *a: None, exception=lambda *a: None)
    class Direction(enum.Enum):
        D2H = 1
        H2D = 2
    class L1Error(enum.Enum):
        SUCCESS = 1
        FAILED = 2
    class GDSMemoryObject:
        pass
    executed = []
    gpu = Buffer(pointers, b"model-kv")
    staging = Buffer(pointers)
    def execute(direction, device, pin_chunk, specs, steps):
        if case == "copy-failure":
            raise RuntimeError("controlled native copy failure")
        for copies, launches in steps:
            if not launches:
                raise AssertionError("source omitted kernel launch")
            if direction == Direction.D2H:
                staging.data[:] = gpu.data
            for dst, src, count, offset in copies:
                pointers[dst].data[:count] = pointers[src].data[:count]
            if direction == Direction.H2D:
                gpu.data[:] = staging.data
        executed.append(direction.name)
    device_ops = ns(KernelGroupSpec=lambda *a: a, LaunchVar=lambda *a: a,
                    BatchStep=lambda *a: a, StagingCopy=lambda *a: a,
                    execute_object_group_transfer=execute)
    event_types = ns(**{name: name for name in ("MP_STORE_SUBMITTED", "MP_TOKENS", "MP_STORE_START", "MP_STORE_END",
                                               "MP_RETRIEVE_SUBMITTED", "MP_RETRIEVE_START", "MP_RETRIEVE_END", "SM_WRITE_FINISHED")})
    bus = ns(publish=lambda *a: None, publish_on_stream=lambda *a: None,
             has_subscribers=lambda *a: False)
    common = {"__name__": "cpu_pinned", "time": time, "threading": threading,
              "_lmcache_nvtx_annotate": lambda f: f, "enable_tracing": lambda: lambda f: f,
              "logger": logger, "Event": lambda **k: ns(**k), "EventType": event_types,
              "ObjectKey": schema.ObjectKey, "MemoryObj": Memory, "GDSMemoryObject": GDSMemoryObject,
              "BaseCacheContext": object, "IPCCacheServerKey": schema.IPCCacheServerKey,
              "Sequence": list, "Any": object, "Generator": object,
              "islice": itertools.islice, "torch": ns(Tensor=object),
              "torch_dev": ns(device=lambda *a: contextlib.nullcontext(), stream=lambda *a: contextlib.nullcontext()),
              "lmcache_native": ns(TransferDirection=Direction), "device_ops": device_ops,
              "LazyMemoryAllocator": ns(PIN_CHUNK_SIZE=4096),
              "MemoryLayoutDesc": lambda **k: ns(**k), "L1Error": L1Error,
              "_HAS_NATIVE_OBJECT_GROUP_TRANSFER": True}
    transfer = types.ModuleType("cpu_transfer")
    transfer.__dict__.update(common)
    gpu_ops = dict(common)
    source_function(sources["lmcache/v1/gpu_connector/gpu_ops.py"], "build_staging_copies", gpu_ops)
    transfer.build_staging_copies = gpu_ops["build_staging_copies"]
    for name in ("get_layout_desc", "batched_iteration_with_skip", "all_null_chunk_masks",
                 "downsample_and_stage_block_ids", "_recalculate_blocks_to_skip",
                 "_run_object_group_transfer_plan", "transfer_kv_per_object_group"):
        source_function(sources[hook.TRANSFER], name, transfer.__dict__)
    class Transfer:
        pass
    for name in ("store", "retrieve"):
        setattr(Transfer, name, source_function(sources[hook.TRANSFER], name, transfer.__dict__))
    transfer.LMCacheDrivenTransferModule = Transfer
    storage = types.ModuleType("cpu_storage")
    storage.__dict__.update(common)
    class Storage:
        def reserve_write(self, keys, layout, mode):
            if case == "empty-success":
                return {}
            values = {key: Memory(Buffer(pointers), case in ("tensorless", "cleanup-finalization")) for key in keys}
            if case == "partial" and values:
                values.pop(next(iter(values)))
            stored.update(values)
            if case == "cleanup-finalization":
                self.finish_write(list(values))
                return {}
            return values
        @contextlib.contextmanager
        def read_prefetched_results(self, keys):
            yield [stored[key] for key in keys if key in stored]
        def finish_read_prefetched(self, keys):
            if case == "handler-failure":
                raise RuntimeError("controlled completion handler failure")
    Storage.finish_write = source_function(sources[hook.STORAGE], "finish_write", storage.__dict__)
    storage.StorageManager = Storage
    completion = types.ModuleType("cpu_completion")
    completion.__dict__.update(common)
    completion._Registration = lambda **k: ns(**k)
    completion.msgspec = ns(msgpack=ns(Decoder=lambda **k: ns(decode=lambda value: value),
                                      encode=lambda value: value))
    def drain():
        items = list(queue)
        queue.clear()
        return [] if case == "lost-native-callback" else items
    completion._device_ops = ns(drain_recorded_completions=drain,
                                record_completion_on_stream=lambda ptr, kind, keys: queue.append((kind, keys)))
    transfer.submit_callback_to_stream = source_function(sources[hook.COMPLETION], "submit_callback_to_stream", completion.__dict__)
    class Dispatcher:
        def __init__(self):
            self._registry = {}
            self._registry_lock = threading.Lock()
            self._exception_counts = {}
            self._dispatched_count = 0
    for name in ("register", "_drain_once"):
        setattr(Dispatcher, name, source_function(sources[hook.COMPLETION], name, completion.__dict__))
    completion.DeviceHostFuncDispatcher = Dispatcher
    attention = ns(is_full_attention=lambda group: case != "window",
                   num_chunks_in_sw=[1 if case == "window" else -1] * groups)
    manager = ns(num_object_groups=groups, num_kernel_groups=groups,
                 object_groups=[ns(kernel_group_indices=[group]) for group in range(groups)],
                 get_attn_desc=lambda: attention, get_subchunk_sw_size_tokens=lambda group: 2)
    ctx = ns(kv_layer_groups_manager=manager, lmcache_tokens_per_chunk=2,
             calculate_num_blocks=lambda tokens, group: tokens // 2,
             get_kernel_group_shape_dtype=lambda tokens, group: ((2,), ns(itemsize=4)),
             get_kernel_group_kv_pointers=lambda group: gpu,
             get_temp_kernel_group_buffer=lambda slot, group: staging,
             get_temp_object_group_buffer=lambda slot, group: staging,
             get_shape_desc=lambda group: (), get_slots_per_chunk_in_sw=lambda group: 2,
             get_engine_kv_format=lambda group: "cpu", max_batch_size=1,
             stage_block_ids=lambda blocks: [Buffer(pointers, bytes(group)) for group in blocks],
             device="controlled-cpu-boundary", stream=object(), cupy_stream=ns(ptr=1))
    backend = ns(create_event=lambda *a: object(), record_event=lambda *a: None,
                 import_event=lambda *a: object(), wait_event=lambda *a: None,
                 export_event=lambda *a: b"cpu-completion-handle")
    def make_instance():
        instance = Transfer()
        instance.get_and_touch_context_entry = lambda identity: ns(cache_context=ctx, model_name="cpu-model", event_backend=backend)
        sm = Storage()
        sm._event_bus = bus
        sm._l1_manager = ns(finish_write=lambda keys: {key: L1Error.FAILED if case == "finalization-failure" else L1Error.SUCCESS for key in keys})
        instance._ctx = ns(chunk_size=2, event_bus=bus, storage_manager=sm,
                           resolve_obj_keys=lambda key, group_ids: schema.convert(
                               key, [f"chunk-{i}".encode() for i in range(chunks)],
                               [group + (1 if case == "wrong-group" else 0) for group in group_ids]))
        dispatcher = Dispatcher()
        dispatcher.register("finish_write", sm.finish_write, list)
        dispatcher.register("finish_read_prefetched", sm.finish_read_prefetched, list)
        return instance, dispatcher
    return transfer, storage, completion, make_instance, gpu, stored, executed


def positive(rows):
    events = [row["event"] for row in rows]
    if not events or events[-1]["kind"] != "fence" or not events[-1]["complete"]:
        return False
    if any(e["kind"] in ("failure", "unsupported") for e in events):
        return False
    copied = []
    for operation in (1, 2):
        own = [e for e in events if e.get("operation") == operation]
        selected = [e["extents"] for e in own if e["kind"] == "selected" and e["supported"]]
        extents = [x for e in own if e["kind"] == "copy" for x in e["extents"]]
        if len(selected) != 1 or not extents or extents != selected[0]:
            return False
        results = [e for e in own if e["kind"] == "result"]
        if len(results) != 1 or not results[0]["success"] or not results[0]["supported"]:
            return False
        keys = [json.dumps(x["key"], sort_keys=True) for x in extents]
        if len(set(keys)) != len(keys):
            return False
        callback = "finish_write" if operation == 1 else "finish_read_prefetched"
        for kind in ("callback_submitted", "device_complete"):
            joined = [e for e in own if e["kind"] == kind and e["callback"] == callback]
            if len(joined) != 1 or sorted(json.dumps(k, sort_keys=True) for k in joined[0]["keys"]) != sorted(keys):
                return False
        if operation == 1:
            finalized = [e["results"] for e in own if e["kind"] == "finalized"]
            if (len(finalized) != 1 or not all(x["success"] for x in finalized[0])
                    or sorted(json.dumps(x["key"], sort_keys=True) for x in finalized[0]) != sorted(keys)):
                return False
        copied.append(extents)
    return copied[0] == copied[1]


def run_case(hook, sources, root, case, rank=0, world_size=1, groups=1):
    chunks = 2 if case == "partial" else 1
    transfer, storage, completion, make_instance, gpu, stored, executed = environment(hook, sources, case, groups, chunks)
    path = root / f"{case}.jsonl"
    journal = hook.Journal(path, role="cache", max_events=8 if case == "journal-budget" else 10000)
    hook.install_cache(journal, transfer, storage, completion, sources)
    if case == "observation-exception":
        def fail_observation(values):
            raise RuntimeError("controlled source observation failure")
        journal.selected = fail_observation
    instance, dispatcher = make_instance()
    key = transfer.IPCCacheServerKey(request_id="engine-store", worker_id=rank, world_size=world_size,
                                    model_name="cpu-model", cache_salt="cpu-salt", start=0, end=2 * chunks,
                                    token_ids=tuple(range(2 * chunks)))
    block_ids = [[0 if case == "null-chunk" else 1] * chunks for _ in range(groups)]
    if case == "store-underflow":
        block_ids = [[] for _ in range(groups)]
    store_result = instance.store(key, 101 + rank, block_ids, b"cpu-event")
    dispatcher._drain_once()
    gpu.data[:] = b"\x00" * 8
    key = dataclasses.replace(key, request_id="engine-reload",
                              cache_salt="different" if case == "wrong-salt" else key.cache_salt)
    reload_blocks = [[] if case == "retrieve-underflow" else [1] * chunks for _ in range(groups)]
    retrieve_result = instance.retrieve(key, (101 if case == "same-engine" else 202) + rank,
                                        reload_blocks, b"cpu-event",
                                        skip_first_n_tokens=2 if case == "skip" else 0)
    dispatcher._drain_once()
    journal.seal(timeout=0)
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    supported = positive(rows)
    if case == "same-engine":
        # Lifecycle is independent of successful copies.
        operations = [r["event"] for r in rows if r["event"]["kind"] == "operation"]
        supported = supported and operations[0]["engine"] != operations[1]["engine"]
    if case == "positive":
        assert supported and bytes(gpu.data) == b"model-kv"
        assert executed == ["D2H"] * groups + ["H2D"] * groups
        assert store_result[1] is True and retrieve_result[1] is True
    elif case == "observation-exception":
        assert not supported and store_result[1] is True and retrieve_result[1] is True
        assert bytes(gpu.data) == b"model-kv"  # Observation did not change serving/copy results.
    else:
        assert not supported, f"nonqualifying case accepted: {case}"
    if case == "partial":
        # A real nonempty partial copy/finalization, not another empty store.
        own = [r["event"] for r in rows if r["event"].get("operation") == 1]
        assert sum(len(e["extents"]) for e in own if e["kind"] == "copy") == 1
        assert any(e["kind"] == "finalized" and len(e["results"]) == 1 for e in own)
    if case in ("store-underflow", "retrieve-underflow"):
        operation = 1 if case == "store-underflow" else 2
        own = [r["event"] for r in rows if r["event"].get("operation") == operation]
        assert (store_result if operation == 1 else retrieve_result)[1] is False
        assert not any(e["kind"] in ("copy", "callback_submitted", "device_complete") for e in own)
    return {"case": case, "copy_contract": supported, "source_store_result": store_result[1],
            "source_retrieve_result": retrieve_result[1], "native_boundary_calls": executed}


def drain_race(hook, sources, out):
    """A later same-key operation must not drain an older missing callback."""
    transfer, storage, completion, make_instance, gpu, stored, executed = environment(hook, sources, "positive")
    path = out / "drain-race.jsonl"
    journal = hook.Journal(path, role="cache")
    hook.install_cache(journal, transfer, storage, completion, sources)
    instance, dispatcher = make_instance()
    key = transfer.IPCCacheServerKey(request_id="engine-store", worker_id=0, world_size=1,
                                    model_name="cpu-model", cache_salt="cpu-salt", start=0, end=2,
                                    token_ids=(0, 1))
    assert instance.store(key, 101, [[1]], b"event")[1]
    dispatcher._drain_once()
    key = dataclasses.replace(key, request_id="engine-reload")
    assert instance.retrieve(key, 202, [[1]], b"event")[1]
    held = completion._device_ops.drain_recorded_completions()
    assert len(held) == 1
    result = []
    sealing = threading.Thread(target=lambda: result.append(journal.seal(timeout=2)))
    sealing.start()
    with journal.condition:
        assert journal.condition.wait_for(lambda: journal.closed, timeout=1)
    # Runs the real pinned retrieve after observation admission closed.
    assert instance.retrieve(key, 202, [[1]], b"event")[1]
    dispatcher._drain_once()  # Only the later, unobserved same-key callback.
    sealing.join(timeout=3)
    assert not sealing.is_alive() and result == [False]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert not positive(rows)
    assert not any(row["event"]["kind"] == "device_complete"
                   and row["event"].get("operation") == 2 for row in rows)
    assert bytes(gpu.data) == b"model-kv"  # Serving behavior was preserved.
    return {"case": "same-key-arrival-during-drain", "copy_contract": False}


def frontend_child(hook, sources, out, stage, case="positive"):
    namespace = {"__name__": "cpu_async", "logger": types.SimpleNamespace(info=lambda *a: None)}
    class Engine:
        pass
    Engine._add_request = source_function(sources[hook.ASYNC], "_add_request", namespace)
    request_context = contextvars.ContextVar("cpu_runtime_request", default=None)
    journal = hook.Journal(out / "frontend.jsonl", role="frontend")
    hook.install_frontend(journal, Engine, request_context, sources[hook.ASYNC])
    engine = Engine()
    observed = []
    async def core_add(request):
        observed.append(request.request_id)
    engine.engine_core = types.SimpleNamespace(add_request_async=core_add)
    engine.output_processor = types.SimpleNamespace(add_request=lambda *args: None)
    engine.log_requests = False
    token = request_context.set(None if case == "outside-context" else {"id": stage, "active": True})
    requests = [f"engine-{stage}"]
    if case == "multiple-admissions":
        requests.append(f"engine-{stage}-extra")
    try:
        for request_id in requests:
            asyncio.run(engine._add_request(types.SimpleNamespace(request_id=request_id),
                                           None, None, 0, object()))
    finally:
        request_context.reset(token)
    assert observed == requests  # Both rejected observation cases preserve actual source admission.
    assert journal.seal(timeout=0) == (case != "outside-context")


def worker_child(hook, sources, out, stage, rank, world_size, case="positive"):
    """Pinned WorkerWrapperBase construction calls the real extension in this child."""
    import os
    ns = types.SimpleNamespace
    schema = key_schema(sources)
    strategy_type = source_class(sources[hook.WORKER], "ParallelStrategy", schema.__dict__)
    namespace = {"__name__": "cpu_worker", "_lmcache_nvtx_annotate": lambda f: f,
                 "LoadStoreOp": object, "_IpcEvent": object,
                 "IPCCacheServerKey": schema.IPCCacheServerKey}
    module = types.ModuleType("cpu_worker")
    module.__dict__.update(namespace)
    class Worker:
        pass
    for name in ("_create_key", "submit_store_request", "submit_retrieve_request",
                 "world_size", "worker_id", "is_kv_writer"):
        setattr(Worker, name, source_function(sources[hook.WORKER], name, module.__dict__,
                                            owner="LMCacheMPWorkerAdapter"))
    module.LMCacheMPWorkerAdapter = Worker
    module.__file__ = str(out / "adapter-source.py")
    Path(module.__file__).write_bytes(sources[hook.WORKER])
    submitted = []
    class BaseWorker:
        pass
    class DeviceWorker(BaseWorker):
        def __init__(self, **kwargs):
            # This initializer is reached only after WorkerExtension.__new__.
            worker = Worker()
            worker.instance_id = (101 if stage == "store" else 202) + rank
            worker.model_name = "cpu-model"
            worker.parallel_strategy = strategy_type(
                mla_only=case == "mla", vllm_world_size=world_size, vllm_worker_id=rank,
                tp_size=world_size, pp_size=1, n_servers=2 if case == "multi-server" else 1)
            worker._ensure_heartbeat_started = lambda: None  # Controlled dependency; no health call.
            worker.is_healthy = True
            worker.kv_caches, worker.blocks_in_chunk = [], 1
            worker._block_ids_per_group = lambda op: [[1]]
            def submit(*args, **kwargs):
                submitted.append((args, kwargs))
                return object()
            worker.transfer_ctx = ns(submit_store=submit, submit_retrieve=submit)
            worker.store_futures, worker.store_events = {}, {}
            worker.retrieve_futures, worker.retrieve_events = {}, {}
            op = ns(token_ids=[1, 2], start=0, end=2, flat_block_ids=[1], skip_first_n_tokens=0)
            name = "submit_store_request" if stage == "store" else "submit_retrieve_request"
            getattr(worker, name)(f"engine-{stage}", op, object(), cache_salt="cpu-salt")
    base_module = types.ModuleType("vllm.v1.worker.worker_base")
    base_module.__file__ = str(out / "worker-base-source.py")
    Path(base_module.__file__).write_bytes(sources[hook.WORKER_BASE])
    plugins = types.ModuleType("vllm.plugins")
    plugins.load_general_plugins = lambda: None
    # Controlled import boundaries: none of these modules imports backend code.
    fake_modules = {"vllm": types.ModuleType("vllm"), "vllm.plugins": plugins,
                    "vllm.v1.worker.worker_base": base_module,
                    hook.WORKER[:-3].replace("/", "."): module}
    saved = {name: sys.modules.get(name) for name in fake_modules}
    config = ns(parallel_config=ns(worker_cls="cpu.DeviceWorker", worker_extension_cls="kv-journal.WorkerExtension"),
                model_config=ns(multimodal_config=None), enable_trace_function_call_for_thread=lambda: None)
    wrapper_namespace = {"__name__": "cpu_wrapper", "instrument": lambda **k: lambda f: f,
                         "logger": ns(info=lambda *a: None, warning_once=lambda *a: None),
                         "resolve_obj_by_qualname": lambda name: DeviceWorker if name == "cpu.DeviceWorker" else hook.WorkerExtension,
                         "set_current_vllm_config": lambda config: contextlib.nullcontext()}
    class Wrapper:
        rpc_rank = 0
    for name in ("init_worker", "__getattr__"):
        setattr(Wrapper, name, source_function(sources[hook.WORKER_BASE], name, wrapper_namespace,
                                             owner="WorkerWrapperBase"))
    old_env = {name: os.environ.get(name) for name in ("THEGRILL_KV_WORKER_DIR", "THEGRILL_KV_WINDOW_US")}
    try:
        sys.modules.update(fake_modules)
        os.environ["THEGRILL_KV_WORKER_DIR"] = str(out)
        os.environ["THEGRILL_KV_WINDOW_US"] = "60000000"
        wrapper = Wrapper()
        wrapper.init_worker([{"vllm_config": config}])
        class Engine:
            pass
        Engine.collective_rpc = source_function(sources[hook.ASYNC], "collective_rpc", {},
                                                owner="AsyncLLM")
        async def dispatch(method, timeout, args, kwargs):
            return [getattr(wrapper, method)(*args, **(kwargs or {}))]
        engine = Engine()
        engine.engine_core = ns(collective_rpc_async=dispatch)
        result = asyncio.run(engine.collective_rpc("grill_kv_seal", timeout=1, kwargs={"timeout":0}))[0]
        assert result["complete"] == (case == "positive")
        identity = result["process"]
        original = out / f"worker-{identity['boot_id']}-{identity['pid']}-{identity['start_ticks']}.jsonl"
        original.rename(out / "worker.jsonl")
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
        for name, old in old_env.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old
    assert len(submitted) == 1 and submitted[0][0][2] == (101 if stage == "store" else 202) + rank
    assert submitted[0][0][1].request_id == f"engine-{stage}"
    events = [json.loads(line)["event"] for line in (out / "worker.jsonl").read_text().splitlines()]
    if case == "positive":
        assert len([event for event in events if event["kind"] == "worker_submission"]) == 1
    else:
        assert any(event["kind"] == "failure" and event["reason"] == "unsupported_worker_parallel_strategy"
                   for event in events)


def consumer_smoke(hook, args):
    """Execute the actual Rust consumer on source-generated process journals."""
    root = args.out / "consumer"
    root.mkdir()
    def child(role, rank=0, label=None, extra=()):
        out = root / (label or f"{role}-{rank}")
        command = [sys.executable, "-I", "-S", str(Path(__file__).resolve()),
                   "--verified-sources", str(args.verified_sources),
                   "--additional-sources", str(args.additional_sources), "--out", str(out),
                   "--child", role, "--rank", str(rank), "--world-size", "2", "--groups", "2",
                   *extra]
        subprocess.run(command, check=True, capture_output=True, timeout=30)
        return out
    frontends = [child(f"frontend-{stage}") / "frontend.jsonl" for stage in ("store", "reload")]
    workers = [[child(f"worker-{stage}", rank) / "worker.jsonl" for rank in range(2)]
               for stage in ("store", "reload")]
    caches = [child("cache", rank) / "positive.jsonl" for rank in range(2)]
    expectation = {"version": 1, "source": hook.SOURCE, "scope": hook.SCOPE,
                   "producer_sha256": hashlib.sha256(Path(hook.__file__).read_bytes()).hexdigest(),
                   "model": "cpu-model", "salt": "cpu-salt", "world_size": 2, "groups": 2,
                   "chunk_size": 2, "start": 0, "end": 2, "store_request": "store", "reload_request": "reload"}
    selection = root / "expectation.json"
    selection.write_text(json.dumps(expectation))
    def invoke(cache_paths=caches, frontend_paths=frontends, worker_paths=workers, expect_path=selection):
        command = [str(args.binary), "startup", "verify-kv", str(expect_path),
                   "--store-frontend", str(frontend_paths[0]), "--reload-frontend", str(frontend_paths[1]),
                   "--producer-source", str(Path(hook.__file__))]
        for path in cache_paths:
            command += ["--cache-events", str(path)]
        for stage, paths in zip(("store", "reload"), worker_paths, strict=True):
            for path in paths:
                command += [f"--{stage}-worker", str(path)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        report = json.loads(result.stdout) if result.stdout.strip().startswith("{") else None
        return result, report
    result, report = invoke()
    assert report and report["source_copy_supported"], (result.returncode, result.stdout, result.stderr)
    assert report["outcome"] == "INCONCLUSIVE" and not report["admission_warmup_eligible"]
    (root / "positive-report.json").write_text(result.stdout)
    outcomes = [{"case": "multi-producer-multi-rank-multi-group", "source_copy_supported": True,
                 "admission_warmup_eligible": False}]
    def reject(name, reason=None, **kwargs):
        result, report = invoke(**kwargs)
        assert not report or not report["source_copy_supported"], (name, result.stdout, result.stderr)
        if reason is not None:
            assert report and reason in report["reasons"], (name, result.stdout, result.stderr)
        outcomes.append({"case": name, "source_copy_supported": False})
    reject("missing-rank", cache_paths=caches[:1])
    reject("missing-worker", worker_paths=[workers[0][:1], workers[1]])
    reload_rows = [json.loads(line) for line in workers[1][0].read_text().splitlines()]
    old_worker = json.loads(workers[0][0].read_text().splitlines()[0])
    reload_rows[0]["event"]["process"] = old_worker["event"]["process"]
    same_birth = root / "same-worker-birth.jsonl"
    same_birth.write_text("".join(json.dumps(row) + "\n" for row in reload_rows))
    reject("unchanged-worker-process",
           reason="KV worker process did not restart; adapter generation alone is insufficient",
           worker_paths=[workers[0], [same_birth, workers[1][1]]])
    reused_cohort = []
    for rank in range(2):
        rows = [json.loads(line) for line in workers[1][rank].read_text().splitlines()]
        old_other_rank = json.loads(workers[0][1 - rank].read_text().splitlines()[0])
        old_same_rank = json.loads(workers[0][rank].read_text().splitlines()[0])
        rows[0]["event"]["process"] = old_other_rank["event"]["process"]
        assert rows[0]["event"]["process"] != old_same_rank["event"]["process"]
        path = root / f"reused-worker-cohort-{rank}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        reused_cohort.append(path)
    # Every rank changed process, but no process in the old cohort was replaced.
    # Keep valid reload sessions/directions/new adapter IDs and journal incarnations.
    reject("permuted-old-worker-cohort",
           reason="KV worker process did not restart; adapter generation alone is insufficient",
           worker_paths=[workers[0], reused_cohort])
    reject("unchanged-frontend-process", frontend_paths=[frontends[0], frontends[0]])
    reject("duplicate-cache-process", cache_paths=[caches[0], caches[0]])
    duplicate_rank = child("worker-store", label="worker-duplicate-rank") / "worker.jsonl"
    reject("duplicate-rank-worker-set", worker_paths=[[workers[0][0], duplicate_rank], workers[1]])
    for case in ("outside-context", "multiple-admissions"):
        frontend = child("frontend-store", label=f"frontend-{case}", extra=("--frontend-case", case)) / "frontend.jsonl"
        reject(f"frontend-{case}", frontend_paths=[frontend, frontends[1]])
    cache_rows = [[json.loads(line) for line in path.read_text().splitlines()] for path in caches]
    reload_events = [[row["event"] for row in rows if row["event"].get("operation") == 2]
                     for rows in cache_rows]
    swapped = []
    for producer, rows in enumerate(cache_rows):
        replacement_events = iter(reload_events[1 - producer])
        for row in rows:
            if row["event"].get("operation") == 2:
                row["event"] = next(replacement_events)
        assert next(replacement_events, None) is None
        path = root / f"wrong-cache-process-{producer}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        swapped.append(path)
    reject("cache-replacement-between-store-and-reload",
           reason="KV reload precedes matching finalized store in the same cache process",
           cache_paths=swapped)
    for case in ("mla", "multi-server"):
        worker = child("worker-store", label=f"worker-{case}", extra=("--worker-case", case)) / "worker.jsonl"
        reject(f"unsupported-worker-{case}", worker_paths=[[worker, workers[0][1]], workers[1]])
    def mutate(name, change):
        rows = [json.loads(line) for line in caches[0].read_text().splitlines()]
        change(rows)
        # Model loss before receiver sequence assignment, not merely a visible gap.
        for index, row in enumerate(rows):
            row["sequence"] = index
        path = root / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        reject(name, cache_paths=[path, caches[1]])
    mutate("missing-terminal-fence", lambda rows: rows.pop())
    mutate("pre-sequence-observation-loss", lambda rows: rows.pop(3))
    mutate("wrong-worker-rank", lambda rows: next(r["event"] for r in rows if r["event"]["kind"] == "operation").update(rank=1))
    mutate("wrong-instance", lambda rows: next(r["event"] for r in rows if r["event"]["kind"] == "operation").update(engine=999))
    mutate("wrong-request", lambda rows: next(r["event"] for r in rows if r["event"]["kind"] == "operation").update(session="not-the-engine-request"))
    mutate("wrong-copied-key", lambda rows: next(r["event"] for r in rows if r["event"]["kind"] == "copy")["extents"][0]["key"].update(hash="aa"))
    mutate("wrong-copied-salt", lambda rows: next(r["event"] for r in rows if r["event"]["kind"] == "copy")["extents"][0]["key"].update(salt="other"))
    mutate("wrong-copied-group", lambda rows: next(r["event"] for r in rows if r["event"]["kind"] == "copy")["extents"][0]["key"].update(group=9))
    mutate("ordinal-used-as-kv-rank", lambda rows: next(r["event"] for r in rows if r["event"]["kind"] == "copy")["extents"][0]["key"].update(kv_rank=0))
    mutate("wrong-copied-model", lambda rows: next(r["event"] for r in rows if r["event"]["kind"] == "copy")["extents"][0]["key"].update(model="other-model"))
    mutate("wrong-finalized-key", lambda rows: next(r["event"] for r in rows if r["event"]["kind"] == "finalized")["results"][0]["key"].update(hash="bb"))
    mutate("failed-finalization", lambda rows: next(r["event"] for r in rows if r["event"]["kind"] == "finalized")["results"][0].update(success=False))
    mutate("incomplete-drain", lambda rows: rows[-1]["event"].update(pending=1, complete=False))
    mutate("cache-incarnation-changed", lambda rows: rows[-1].update(incarnation="f" * 32))
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verified-sources", required=True, type=Path)
    parser.add_argument("--additional-sources", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--binary", type=Path, help="Already-built grill-perf; exercises its real verify-kv consumer")
    parser.add_argument("--candidate-source-root", type=Path,
                        help="Additionally exercise exact candidate adapter failure/completion paths")
    parser.add_argument("--child", choices=["cache", "frontend-store", "frontend-reload", "worker-store", "worker-reload"])
    parser.add_argument("--frontend-case", choices=["positive", "outside-context", "multiple-admissions"], default="positive")
    parser.add_argument("--worker-case", choices=["positive", "mla", "multi-server"], default="positive")
    parser.add_argument("--rank", type=int, default=0, choices=range(64))
    parser.add_argument("--world-size", type=int, default=1, choices=range(1, 65))
    parser.add_argument("--groups", type=int, default=1, choices=range(1, 65))
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False, parents=True)
    hook = load_hook()
    sources = read_sources(hook, args.verified_sources, args.additional_sources)
    if args.child:
        if args.child == "cache":
            run_case(hook, sources, args.out, "positive", args.rank, args.world_size, args.groups)
        elif args.child.startswith("frontend-"):
            frontend_child(hook, sources, args.out, args.child.removeprefix("frontend-"), args.frontend_case)
        else:
            worker_child(hook, sources, args.out, args.child.removeprefix("worker-"), args.rank, args.world_size, args.worker_case)
        return
    cases = ("positive", "empty-success", "tensorless", "cleanup-finalization", "copy-failure",
             "partial", "null-chunk", "finalization-failure", "lost-native-callback",
             "handler-failure", "journal-budget", "observation-exception",
             "wrong-group", "wrong-salt", "same-engine", "skip", "window",
             "store-underflow", "retrieve-underflow")
    results = [run_case(hook, sources, args.out, case) for case in cases]
    results.append(drain_race(hook, sources, args.out))
    consumer = consumer_smoke(hook, args) if args.binary else []
    summary = {"execution": "cpu-source-boundary-not-native-qualification", "cases": results,
               "consumer_cases": consumer, "consumer_executed": args.binary is not None,
               "source_pins": {**hook.PINS, hook.ASYNC: hook.ASYNC_SHA256, hook.WORKER: hook.WORKER_SHA256,
                               hook.WORKER_BASE: hook.WORKER_BASE_SHA256}}
    if args.candidate_source_root:
        candidate = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("kv-observer-regression.py")),
             "--source-root", str(args.candidate_source_root)],
            check=True, capture_output=True, text=True, timeout=60)
        summary["candidate_source_paths"] = json.loads(candidate.stdout)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
