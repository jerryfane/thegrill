"""Pinned v2 source hook composition. Import performs no backend operation."""

import ast
import contextvars
import functools
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
from pathlib import Path
import types
import sys
import weakref
import pkgutil
import threading

TRANSFER = "lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py"
STORAGE = "lmcache/v1/distributed/storage_manager.py"
COMPLETION = "lmcache/v1/multiprocess/native_completion.py"
PERIODIC = "lmcache/v1/periodic_thread.py"
TRACE = "lmcache/v1/mp_observability/trace/decorator.py"
NVTX = "nvtx/nvtx.py"
UTILS = "lmcache/utils.py"
WORKER = "lmcache/integration/vllm/vllm_multi_process_adapter.py"
CONNECTOR = "lmcache/integration/vllm/lmcache_mp_connector.py"
LAZY = "lmcache/v1/memory_allocators/lazy_memory_allocator.py"
MIXED = "lmcache/v1/memory_allocators/mixed_memory_allocator.py"
WORKER_BASE = "vllm/v1/worker/worker_base.py"
ASYNC = "vllm/v1/engine/async_llm.py"
MEMORY = "lmcache/v1/memory_management.py"
ALLOCATOR = "lmcache/v1/memory_allocators/tensor_memory_allocator.py"
L1 = "lmcache/v1/distributed/l1_manager.py"
GLOBAL = "_thegrill_transfer_v2"



def adjacent(name):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


registration = adjacent("kv-registration-hooks.py")


def loaded_sources(role):
    # Only explicit operator entrypoints call this loader. Read distribution
    # bytes before importing any backend module.
    journal = adjacent("kv-journal-v2.py")
    expected = registration.pins(role)
    paths = {path: importlib.metadata.distribution(path.split("/")[0]).locate_file(path)
             for path in expected}
    sources = {path: journal.bounded_read(actual) for path, actual in paths.items()}
    registration._validate_sources(sources, expected)
    needed = ([TRANSFER, STORAGE, COMPLETION, PERIODIC, TRACE, NVTX, UTILS,
               MEMORY, ALLOCATOR, L1, LAZY, MIXED] if role == "cache"
              else [CONNECTOR, WORKER, WORKER_BASE, NVTX, UTILS])
    modules = {}
    for path in needed:
        module = importlib.import_module(path[:-3].replace("/", "."))
        if Path(module.__file__).resolve() != Path(paths[path]).resolve():
            raise ValueError("loaded module differs from selected distribution")
        modules[path] = module
    for path, actual in paths.items():
        if not path.endswith(".py"):
            continue
        name = path[:-3].replace("/", ".").removesuffix(".__init__")
        loaded = sys.modules.get(name)
        if loaded is not None and Path(loaded.__file__).resolve() != Path(actual).resolve():
            raise ValueError("loaded transitive source differs from selected distribution")
    return modules, sources

def geometry_roster(sources):
    """Check both discovered module membership and effective registry classes."""
    root = "lmcache/v1/gpu_connector/kv_format/"
    for folder, base_name, table_name, key_name in (
            ("specs", "KVFormatSpec", "SPECS", "engine_kv_format"),
            ("detectors", "EngineDetector", "DETECTORS", "engine_type")):
        prefix = root + folder + "/"
        expected_modules = {Path(path).stem for path in sources
                            if path.startswith(prefix) and path.endswith(".py")
                            and Path(path).stem != "__init__"}
        registry_name = (prefix + "registry") .replace("/", ".")
        registry = sys.modules.get(registry_name)
        if registry is None:
            raise ValueError("geometry registry is not loaded")
        actual_modules = {item.name for item in pkgutil.iter_modules([str(Path(registry.__file__).parent)])}
        if actual_modules != expected_modules:
            raise ValueError("discovered geometry module roster differs from pinned source")
        base = getattr(sys.modules[prefix[:-1].replace("/", ".") + ".base"], base_name)
        expected_classes = {}
        for module_name in sorted(expected_modules - {"base", "registry"}):
            path = prefix + module_name + ".py"
            module = sys.modules.get(path[:-3].replace("/", "."))
            if module is None:
                raise ValueError("discovered geometry module missing")
            for node in ast.parse(sources[path]).body:
                if not isinstance(node, ast.ClassDef):
                    continue
                cls = getattr(module, node.name, None)
                if not isinstance(cls, type) or not issubclass(cls, base):
                    raise ValueError("discovered geometry class differs from pinned source")
                for method in node.body:
                    if isinstance(method, ast.FunctionDef):
                        function = vars(cls).get(method.name)
                        if isinstance(function, (staticmethod, classmethod)):
                            function = function.__func__
                        if isinstance(function, property):
                            function = function.fget
                        require_source_function(function, sources[path], f"{node.name}.{method.name}")
                key_nodes = [n.value for n in node.body if isinstance(n, ast.Assign)
                             and any(isinstance(t, ast.Name) and t.id == key_name for t in n.targets)]
                if len(key_nodes) != 1 or not isinstance(key_nodes[0], ast.Attribute):
                    raise ValueError("unknown geometry registry key declaration")
                key = getattr(cls, key_name)
                if key.name != key_nodes[0].attr or key in expected_classes:
                    raise ValueError("geometry registry key collision or changed identity")
                expected_classes[key] = cls
        if getattr(registry, table_name) != expected_classes:
            raise ValueError("effective geometry class roster differs from pinned source")
        if folder == "specs" and registry._SPECS_BY_VALUE != {int(k): v for k, v in expected_classes.items()}:
            raise ValueError("effective integer format registry differs from pinned source")



def require_source_function(function, source, qualified, *, shell=None):
    """Verify the executing body, not an arbitrary __wrapped__ assertion.

    Only the pinned L1 lock, tracing and NVTX shells are supported. Their
    actual delegate cells must match; decorator metadata is not authority.
    """
    original = getattr(function, "__func__", function)
    if shell is not None:
        shell_source, shell_qualified, module = shell
        expected_shell = registration.source_code(shell_source, module.__file__, shell_qualified)
        if (not isinstance(original, types.FunctionType)
                or original.__globals__ is not module.__dict__
                or registration._code_signature(original.__code__) != registration._code_signature(expected_shell)):
            raise ValueError("unsupported or substituted source decorator shell")
        cells = dict(zip(original.__code__.co_freevars, original.__closure__ or (), strict=True))
        inner = cells["func"].cell_contents
        if shell_qualified == "l1_mgr_synchronized.<locals>.wrapper":
            if set(cells) != {"func"} or hasattr(original, "__wrapped__"):
                raise ValueError("L1 synchronization shell binding changed")
        elif shell_qualified == "enable_tracing.<locals>.deco.<locals>.wrapper":
            if (getattr(original, "__wrapped__", None) is not inner
                    or set(cells) != {"func", "param_set", "resolved_qualname", "sig"}):
                raise ValueError("tracing shell binding changed")
            signature = inspect.signature(inner, follow_wrapped=False)
            if (cells["sig"].cell_contents != signature
                    or cells["param_set"].cell_contents != frozenset(
                        name for name in signature.parameters if name not in ("self", "cls"))
                    or cells["resolved_qualname"].cell_contents != f"{inner.__module__}.{inner.__qualname__}"):
                raise ValueError("unsupported tracing capture configuration")
        elif shell_qualified == "annotate.__call__.<locals>.inner":
            if (set(cells) != {"func", "self"}
                    or getattr(original, "__wrapped__", None) is not inner
                    or type(cells["self"].cell_contents) is not module.annotate):
                raise ValueError("NVTX annotation shell binding changed")
        else:
            raise ValueError("unqualified source decorator")
        original = inner
    if not isinstance(original, types.FunctionType) or hasattr(original, "__wrapped__"):
        raise ValueError("expected an unwrapped pinned Python implementation")
    expected = registration.source_code(source, "<pinned-operation>", qualified)
    if registration._code_signature(expected) != registration._code_signature(original.__code__):
        raise ValueError("active implementation differs from pinned source")
    return original

def passive_event_guard(transfer, sources):
    """Admit the source-verified default CUDA backend before instrumentation.

    The checked default selector constructs only a Python wrapper, not an
    event. Reject isolated IPC before accessing the lazy backend property.
    """
    platform = sys.modules.get("lmcache.v1.platform")
    event_module = sys.modules.get("lmcache.v1.platform.base.event_ipc")
    torch = sys.modules.get("torch")
    cuda = sys.modules.get("lmcache.v1.platform.cuda")
    isolated = sys.modules.get("lmcache.v1.platform.isolated_ipc")
    spec = getattr(platform, "current_device_spec", None)
    if (cuda is None or isolated is None or isolated._enabled is not False
            or type(spec) is not cuda.CudaDeviceSpec):
        raise ValueError("isolated or unknown event backend configuration is unsupported")
    cuda_source = sources["lmcache/v1/platform/cuda/__init__.py"]
    require_source_function(cuda._select_event_ipc_backend, cuda_source, "_select_event_ipc_backend")
    require_source_function(cuda.CudaDeviceSpec.event_ipc_backend.fget, cuda_source,
                            "CudaDeviceSpec.event_ipc_backend")
    require_source_function(cuda.is_isolated_ipc,
        sources["lmcache/v1/platform/isolated_ipc.py"], "is_isolated_ipc")
    if cuda.is_isolated_ipc is not isolated.is_isolated_ipc:
        raise ValueError("CUDA selector configuration binding changed")
    default = getattr(event_module, "DefaultEventIPCBackend", None)
    if default is None:
        raise ValueError("missing default event backend type")
    require_source_function(default.__init__,
        sources["lmcache/v1/platform/base/event_ipc.py"], "DefaultEventIPCBackend.__init__")
    backend = spec.event_ipc_backend
    if (backend is None or default is None or type(backend) is not default
            or torch is None or transfer.torch_dev is not torch.cuda
            or backend._event_module is not torch.cuda or backend.device_type != "cuda"):
        raise ValueError("passive observer requires a preselected default CUDA event backend")
    query = require_source_function(backend.query_event,
        sources["lmcache/v1/platform/base/event_ipc.py"], "DefaultEventIPCBackend.query_event")
    event_type = torch.cuda.Event
    event_query = event_type.query
    def guard(actual, event):
        if (actual is not backend or type(actual) is not default
                or isolated._enabled is not False or spec._event_backend_cache is not backend
                or actual._event_module is not torch.cuda or actual.device_type != "cuda"
                or getattr(actual.query_event, "__func__", None) is not query
                or event_type.query is not event_query
                or (event is not None and (type(event) is not event_type
                    or getattr(event.query, "__func__", None) is not event_query))):
            raise ValueError("event backend or query implementation changed")
    return guard



def instrument(module, source, qualified, boundaries, *, shell=None):
    """Compile only the exact source function with post-statement observations.

    boundaries is a list of (AST dump predicate, observer method, expected count).
    Decorators are not replayed; the selected function retains its existing
    decorator shell through an explicit source-body replacement at installation.
    """
    tree = ast.parse(source)
    parts = qualified.split(".")
    body = tree.body
    for part in parts:
        candidates = [node for node in body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == part]
        if len(candidates) != 1:
            raise ValueError("ambiguous qualified source boundary")
        node = candidates[0]
        body = node.body
    if not isinstance(node, ast.FunctionDef):
        raise ValueError("source boundary is not synchronous")
    owner = module
    for part in parts[:-1]:
        owner = getattr(owner, part)
    original = getattr(owner, parts[-1])
    if not callable(original) or getattr(original, "_grill_transfer_v2", False):
        raise ValueError("duplicate/incompatible source hook")
    unwrapped = require_source_function(original, source, qualified, shell=shell)
    if not isinstance(unwrapped, types.FunctionType) or unwrapped.__globals__ is not module.__dict__:
        raise ValueError("source hook is not original module code")
    if unwrapped.__code__.co_freevars:
        raise ValueError("standalone source body requires no closure cells")
    # Exact source code comparison refuses unrelated instrumentation.
    expected = registration.source_code(source, module.__file__, qualified)
    if registration._code_signature(expected) != registration._code_signature(unwrapped.__code__):
        raise ValueError("loaded method does not match pinned body")
    node.decorator_list = []
    counts = [0] * len(boundaries)
    class Rewrite(ast.NodeTransformer):
        def visit_FunctionDef(self, function):
            if function is not node:
                return function
            return self.generic_visit(function)
        def visit(self, statement):
            result = super().visit(statement)
            if not isinstance(result, ast.stmt):
                return result
            observers = []
            for index, (predicate, method, _) in enumerate(boundaries):
                if predicate(result):
                    counts[index] += 1
                    observers.extend(ast.parse(f"{GLOBAL}.journal.observe({GLOBAL}.{method}, locals())").body)
            return [result, *observers] if observers else result
    rewritten = Rewrite().visit(node)
    if isinstance(rewritten, list):
        raise ValueError("observer selected function itself")
    if counts != [count for _, _, count in boundaries]:
        raise ValueError("pinned observation boundary count mismatch")
    replacement = registration.compile_body(tree, rewritten, module, module.__file__, unwrapped)
    def preserve_shell(shell, replacement):
        if shell is unwrapped:
            return replacement
        inner = getattr(shell, "__wrapped__", None)
        if inner is None or shell.__closure__ is None:
            raise ValueError("unsupported existing decorator shell")
        nested = preserve_shell(inner, replacement)
        cells = tuple(types.CellType(nested) if cell.cell_contents is inner else cell
                      for cell in shell.__closure__)
        if all(a is b for a, b in zip(cells, shell.__closure__)):
            raise ValueError("decorator does not delegate through its wrapped function")
        clone = types.FunctionType(shell.__code__, shell.__globals__, shell.__name__,
                                   shell.__defaults__, cells)
        functools.update_wrapper(clone, shell)
        clone.__kwdefaults__ = shell.__kwdefaults__
        clone.__wrapped__ = nested
        return clone
    replacement = preserve_shell(original, replacement)
    replacement._grill_transfer_v2 = True
    return owner, parts[-1], original, replacement


def assigned(name):
    return lambda node: isinstance(node, (ast.Assign, ast.AnnAssign)) and any(
        isinstance(target, ast.Name) and target.id == name
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target]))


def called(name):
    def match(node):
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            return False
        fn = node.value.func
        return (isinstance(fn, ast.Name) and fn.id == name) or (
            isinstance(fn, ast.Attribute) and fn.attr == name)
    return match


class Observer:
    def __init__(self, journal):
        self.journal = journal
        self.op_context = journal.operation_context
        self.helper_context = contextvars.ContextVar("grill_transfer_helper_v2", default=None)
        self.l1 = None
        self.cache_owner = self.dispatcher = None
        self.cache_stop_observed = False
        self.dispatcher_methods = {}

    @property
    def current_helper(self):
        helper = self.helper_context.get()
        if helper is None:
            raise ValueError("unbound transfer helper observation")
        return helper

    def native_batch(self, values):
        helper = self.current_helper
        if not values["staging"] or len(values["launches"]) != len(values["kernel_group_ids"]):
            raise ValueError("empty native staging or incomplete kernel plan")
        helper["native_batches"].append((values["start_object_idx"], values["batch_len"], list(values["kernel_group_ids"])))

    def native_done(self, values):
        if (self.native_function is None
                or self.device_ops.execute_object_group_transfer is not self.native_function):
            raise ValueError("native object-group implementation changed")
        if (self.transfer_module.device_ops is not self.device_ops
                or self.transfer_module._HAS_TRANSFER_PHASE_TIMING is not self.phase_timing):
            raise ValueError("native object-group implementation changed")
        helper = self.current_helper
        op = self.op_context.get()
        if op is None or values["transfer_key"] != op.get("transfer_key"):
            raise ValueError("native plan belongs to a different transfer generation")
        if not helper["native_batches"] or len(values["batch_steps"]) != len(helper["native_batches"]):
            raise ValueError("native plan did not execute complete batches")
        for start, count, kernels in helper["native_batches"]:
            self.execution(helper, start, count, kernels, "native-object-group", "complete")

    def fallback_kernel(self, values):
        if values["end_block_pos"] <= values["start_block_pos"] or values["recalculated_skip_blocks"] != 0:
            raise ValueError("empty or partial fallback kernel")
        operation = self.device_ops.multi_layer_block_kv_transfer
        if getattr(operation, "__func__", operation) is not self.paged_function:
            raise ValueError("active paged implementation changed")
        if self.paged_mode == "torch-per-kernel" and (
                self.paged_function.__globals__["torch_ops"].multi_layer_block_kv_transfer is not self.torch_function):
            raise ValueError("active torch delegate changed")
        mode = self.paged_mode
        self.execution(self.current_helper, values["start_object_idx"], values["batch_len"],
                       [values["kernel_group_id"]], mode, "kernel")

    def fallback_h2d(self, values):
        self.execution(self.current_helper, values["start_object_idx"] + values["chunk_idx"],
                       1, [], "staging", "host-to-staging")

    def fallback_d2h(self, values):
        self.execution(self.current_helper, values["start_object_idx"] + values["chunk_idx"],
                       1, [], "staging", "staging-to-host")

    def execution(self, helper, start, count, kernels, mode, leg):
        if type(count) is not int or not 1 <= count <= 1024:
            raise ValueError("execution batch bound")
        for index in range(start, start + count):
            if index not in helper["allocations"]:
                raise ValueError("execution has no bound live operand")
        self.journal.emit("execution", operation=helper["operation"], group=helper["group"],
                          start=start, count=count, kernels=kernels, mode=mode, leg=leg)

    def helper(self, original, *args, **kwargs):
        values = inspect.signature(original).bind(*args, **kwargs).arguments
        op = self.op_context.get()
        helper = {"operation": op["id"] if op else 0, "group": values["object_group_id"],
                  "allocations": {}, "native_batches": []}
        def bind():
            if op is None or values["skip_first_n_tokens"] != 0 or self.l1 is None:
                raise ValueError("unbound or partial helper")
            transfer_key = values["transfer_key"]
            if type(transfer_key) is not str or not transfer_key or len(transfer_key) > 256:
                raise ValueError("missing source transfer generation")
            if op.setdefault("transfer_key", transfer_key) != transfer_key:
                raise ValueError("object groups belong to different transfer generations")
            registration = self.journal.registrations[op["engine"]]
            group = helper["group"]
            objects = values["memory_objs"]
            keys = op["keys"][group]
            if len(objects) != len(keys):
                raise ValueError("operand/key length mismatch")
            l1 = self.l1()
            if l1 is None:
                raise ValueError("selected L1 manager disappeared")
            for index, obj in enumerate(objects):
                if obj is not None:
                    token = self.journal.lifetime.bind(keys[index], obj, registration, group, l1)
                    helper["allocations"][index] = token
                    op["allocations"].add(token)
                    self.journal.emit("operand", operation=op["id"], group=group,
                                      index=index, allocation=token)
        self.journal.observe(bind)
        token = self.helper_context.set(helper)
        try:
            return original(*args, **kwargs)
        finally:
            self.helper_context.reset(token)

    def worker_submission(self, values, direction):
        if direction == "retrieve" and values["op"].skip_first_n_tokens != 0:
            raise ValueError("partial-prefix retrieve is outside the observer contract")
        worker, key = values["self"], values["key"]
        if self.journal.closed or self.journal.failed or worker.instance_id not in self.journal.registrations:
            raise ValueError("unregistered or closed worker submission")
        if (type(key.world_size) is not int or not 1 <= key.world_size <= 64
                or type(key.worker_id) is not int or not 0 <= key.worker_id < key.world_size):
            raise ValueError("worker submission rank outside observation bounds")
        if key.request_configs is not None:
            raise ValueError("request-scoped configuration is outside the observer contract")
        expected_readers = worker.parallel_strategy.num_kv_readers
        if type(key.num_kv_readers) is not int or not 1 <= key.num_kv_readers <= 64 or key.num_kv_readers != expected_readers:
            raise ValueError("submission reader count differs from registered strategy")
        identity = (worker.instance_id, direction, key.request_id)
        if identity in self.journal.worker_pending or self.journal.worker_generation >= 1024:
            raise ValueError("duplicate outstanding worker transfer or operation budget")
        future = values["future"]
        previous = self.journal.worker_futures.get(id(future))
        if previous is not None and previous() is future:
            raise ValueError("future reused across transfer generations")
        future_ref = weakref.ref(future)
        self.journal.worker_futures[id(future)] = future_ref
        self.journal.worker_generation += 1
        generation = self.journal.worker_generation
        self.journal.worker_pending[identity] = (future_ref, generation)
        self.journal.emit("worker_submission", direction=direction, engine=worker.instance_id,
                          session=key.request_id, rank=key.worker_id, world_size=key.world_size,
                          model=key.model_name, salt=key.cache_salt, start=key.start, end=key.end,
                          num_kv_readers=key.num_kv_readers, generation=generation)

    def worker_store(self, values):
        self.worker_submission(values, "store")

    def worker_retrieve(self, values):
        self.worker_submission(values, "retrieve")

    def worker_result(self, values, direction):
        prefix = "s" if direction == "store" else "r"
        identity = (values["self"].instance_id, direction, values["request_id"])
        pending = self.journal.worker_pending.get(identity)
        if pending is None or pending[0]() is not values[f"{prefix}_future"]:
            raise ValueError("worker result is not the exact submitted future")
        self.journal.emit("worker_result", engine=identity[0], direction=direction,
                          session=identity[2], generation=pending[1],
                          success=values[f"{prefix}_result"] is True)
        if values[f"{prefix}_result"] is not True:
            raise ValueError("worker transfer did not return success")
        del self.journal.worker_pending[identity]

    def worker_store_result(self, values):
        self.worker_result(values, "store")

    def worker_retrieve_result(self, values):
        self.worker_result(values, "retrieve")

    def worker_call(self, original):
        @functools.wraps(original)
        def call(worker, *args, **kwargs):
            def healthy():
                if not worker.is_healthy:
                    raise ValueError("unhealthy worker cannot qualify")
                if worker.error_block_ids or getattr(worker, "_dropped_retrieves", ()):
                    raise ValueError("worker reports failed or dropped retrieves")
            self.journal.observe(healthy)
            try:
                result = original(worker, *args, **kwargs)
            except BaseException:
                self.journal.emit("failure", reason="worker_source_exception")
                raise
            self.journal.observe(healthy)
            if original.__name__ == "get_block_ids_with_load_errors" and result:
                self.journal.emit("failure", reason="worker_load_errors")
            return result
        return call

    def bind_cache(self, values):
        owner = values["self"]
        dispatcher = owner._device_host_func_dispatcher
        if (type(owner) is not self.transfer_module.LMCacheDrivenTransferModule
                or type(dispatcher) is not self.dispatcher_type
                or self.cache_owner is not None):
            raise ValueError("cache lifecycle owner is not a fresh selected instance")
        self.cache_owner = weakref.ref(owner)
        self.dispatcher = weakref.ref(dispatcher)
        self.dispatcher_health(dispatcher)

    def dispatcher_health(self, instance, *, stopped=False):
        if (self.dispatcher is None or self.dispatcher() is not instance
                or type(instance) is not self.dispatcher_type):
            raise ValueError("unbound or substituted completion dispatcher")
        for name, function in self.dispatcher_methods.items():
            method = getattr(instance, name)
            if (getattr(method, "__self__", None) is not instance
                    or getattr(method, "__func__", None) is not function):
                raise ValueError("completion dispatcher lifecycle binding changed")
        if (type(instance._failed_runs) is not int or instance._failed_runs != 0
                or type(instance._exception_counts) is not dict or instance._exception_counts):
            raise ValueError("completion dispatcher reports failed runs or callbacks")
        if stopped:
            thread = instance._thread
            if (instance._running is not False
                    or (thread is not None and (
                        type(thread) is not threading.Thread
                        or getattr(thread.is_alive, "__func__", None) is not threading.Thread.is_alive
                        or thread.is_alive()))):
                raise ValueError("completion dispatcher stop did not terminate its thread")

    def drain_failed(self, values):
        self.journal.emit("failure", reason="completion_dispatcher_drain_or_delivery_failed")

    def callback_binding(self, kind, handler):
        owner = getattr(handler, "__self__", None)
        expected = self.callback_methods[kind]
        cache = self.cache_owner() if self.cache_owner is not None else None
        if (cache is None or owner is not cache._ctx.storage_manager
                or type(owner) is not self.storage_type
                or getattr(handler, "__func__", None) is not expected
                or getattr(getattr(owner, kind), "__func__", None) is not expected
                or kind in vars(owner)):
            raise ValueError("storage finalization callback binding changed")
        l1 = owner._l1_manager
        name = "finish_write" if kind == "finish_write" else "finish_read"
        if (type(l1) is not self.journal.lifetime.l1_manager_type
                or name in vars(l1)
                or getattr(getattr(l1, name), "__func__", None) is not self.l1_callback_methods[name]):
            raise ValueError("L1 finalization callback binding changed")

    def cache_fence(self):
        owner = self.cache_owner() if self.cache_owner is not None else None
        dispatcher = self.dispatcher() if self.dispatcher is not None else None
        if (not self.cache_stop_observed or owner is None or dispatcher is None
                or owner._device_host_func_dispatcher is not dispatcher):
            raise ValueError("cache fence precedes verified owning dispatcher stop")
        self.dispatcher_health(dispatcher, stopped=True)

    def cache_stopped(self, values):
        # The source has already joined/drained. Observe only; never join again.
        owner = values["self"]
        if self.cache_owner is None or self.cache_owner() is not owner:
            raise ValueError("cache close belongs to an unbound lifecycle")
        self.cache_fence()
        if not self.journal.sealed:
            self.journal.seal(timeout=0)


def atomic_install(install):
    @functools.wraps(install)
    def call(journal, modules, sources):
        missing = object()
        changes = []
        original_replace = journal.replace
        hook_start = len(journal.hooks)
        original_globals = [(module, module.__dict__.get(GLOBAL, missing)) for module in modules.values()]
        def tracked(owner, name, replacement):
            original = vars(owner).get(name, missing)
            changes.append((owner, name, original, replacement))
            original_replace(owner, name, replacement)
        journal.replace = tracked
        try:
            return install(journal, modules, sources)
        except BaseException:
            for owner, name, original, replacement in reversed(changes):
                if vars(owner).get(name, missing) is replacement:
                    if original is missing:
                        delattr(owner, name)
                    else:
                        setattr(owner, name, original)
            for module, original in original_globals:
                if original is missing:
                    module.__dict__.pop(GLOBAL, None)
                else:
                    module.__dict__[GLOBAL] = original
            del journal.hooks[hook_start:]
            raise
        finally:
            journal.replace = original_replace
    return call


@atomic_install
def install_cache(journal, modules, sources):
    import gc
    expected = registration.pins("cache")
    registration._validate_sources(sources, expected)
    geometry_roster(sources)
    if journal.role != "cache" or journal.hooks or journal.closed:
        raise ValueError("invalid cache installation state")
    transfer, storage, completion = (modules[path] for path in (TRANSFER, STORAGE, COMPLETION))
    periodic, trace = modules[PERIODIC], modules[TRACE]
    dispatcher = completion.DeviceHostFuncDispatcher
    if (dispatcher.__bases__ != (periodic.PeriodicThread,)
            or completion.PeriodicThread is not periodic.PeriodicThread
            or completion.PeriodicThreadRegistry is not periodic.PeriodicThreadRegistry
            or transfer.DeviceHostFuncDispatcher is not dispatcher):
        raise ValueError("completion dispatcher periodic lifecycle ownership changed")
    lifecycle_bindings = []
    for owner, path, names in (
            (periodic.PeriodicThread, PERIODIC, ("__init__", "start", "stop", "_run_loop")),
            (dispatcher, COMPLETION, ("__init__", "start", "stop", "register", "_execute", "_drain_once"))):
        for name in names:
            function = getattr(owner, name)
            require_source_function(function, sources[path], f"{owner.__name__}.{name}")
            if function.__globals__ is not modules[path].__dict__:
                raise ValueError("lifecycle method defining globals changed")
            cells = dict(zip(function.__code__.co_freevars, function.__closure__ or (), strict=True))
            if "__class__" in cells and cells["__class__"].cell_contents is not owner:
                raise ValueError("lifecycle super class binding changed")
            lifecycle_bindings.append((owner, name, function))
    require_source_function(trace.enable_tracing, sources[TRACE], "enable_tracing")
    require_source_function(trace.publish_call_event, sources[TRACE], "publish_call_event")
    trace_shell = (sources[TRACE], "enable_tracing.<locals>.deco.<locals>.wrapper", trace)
    lock_shell = (sources[L1], "l1_mgr_synchronized.<locals>.wrapper", modules[L1])
    require_source_function(modules[L1].l1_mgr_synchronized, sources[L1], "l1_mgr_synchronized")
    if storage.enable_tracing is not trace.enable_tracing:
        raise ValueError("storage tracing decorator defining binding changed")
    nvtx, utils = modules[NVTX], modules[UTILS]
    require_source_function(utils._lmcache_nvtx_annotate, sources[UTILS], "_lmcache_nvtx_annotate")
    require_source_function(utils._get_color_for_nvtx, sources[UTILS], "_get_color_for_nvtx")
    if utils.annotate is not nvtx.annotate:
        # The pinned no-NVTX fallback is a direct Python no-op decorator.
        require_source_function(utils.annotate, sources[UTILS], "annotate")
    nvtx_shell = (sources[NVTX], "annotate.__call__.<locals>.inner", nvtx)
    def annotation_shell(function):
        return nvtx_shell if hasattr(function, "__wrapped__") else None
    for module in (transfer, modules[ALLOCATOR], modules[MIXED]):
        if module._lmcache_nvtx_annotate is not utils._lmcache_nvtx_annotate:
            raise ValueError("NVTX decorator factory defining binding changed")
    event_guard = passive_event_guard(transfer, sources)
    journal.registration_guard = lambda worker: geometry_roster(sources)
    cls = transfer.LMCacheDrivenTransferModule
    if any(isinstance(obj, (cls, completion.DeviceHostFuncDispatcher)) for obj in gc.get_objects()):
        raise ValueError("cache hooks must precede construction")
    observer = Observer(journal)
    observer.dispatcher_type = dispatcher
    observer.storage_type = storage.StorageManager
    journal.cache_seal_guard = observer.cache_fence
    for name, path in (
            ("is_observability_enabled", "lmcache/v1/mp_observability/event_bus.py"),
            ("get_event_bus", "lmcache/v1/mp_observability/event_bus.py"),
            ("next_transfer_key", "lmcache/v1/mp_observability/event.py")):
        require_source_function(getattr(transfer, name), sources[path], name)
    observer.device_ops = transfer.device_ops
    observer.transfer_module = transfer
    observer.phase_timing = transfer._HAS_TRANSFER_PHASE_TIMING
    if type(observer.phase_timing) is not bool:
        raise ValueError("invalid native phase timing selection")
    native_enabled = transfer._HAS_NATIVE_OBJECT_GROUP_TRANSFER
    if type(native_enabled) is not bool:
        raise ValueError("invalid native object-group selection flag")
    observer.native_function = None
    if native_enabled:
        native = transfer.device_ops.execute_object_group_transfer
        if not isinstance(native, types.BuiltinFunctionType) or native.__module__ != "lmcache.cuda_ops":
            raise ValueError("native object-group implementation origin is unverified")
        observer.native_function = native
    paged = transfer.device_ops.multi_layer_block_kv_transfer
    if isinstance(paged, types.BuiltinFunctionType) and paged.__module__ == "lmcache.cuda_ops":
        observer.paged_function, observer.paged_mode = paged, "native-per-kernel"
    else:
        observer.paged_function = require_source_function(paged,
            sources["lmcache/v1/platform/base/device_ops.py"], "DeviceOps.multi_layer_block_kv_transfer")
        observer.torch_function = require_source_function(
            observer.paged_function.__globals__["torch_ops"].multi_layer_block_kv_transfer,
            sources["lmcache/v1/platform/torch_ops.py"], "multi_layer_block_kv_transfer")
        observer.paged_mode = "torch-per-kernel"
    snapshot = adjacent("kv-registration.py")
    lifetime_module = adjacent("kv-lifetime-v2.py")
    for path, owner in ((ALLOCATOR, modules[ALLOCATOR].TensorMemoryAllocator),
                        (LAZY, modules[LAZY].LazyMemoryAllocator),
                        (MIXED, modules[MIXED].MixedMemoryAllocator)):
        for name in ("free", "batched_free"):
            function = getattr(owner, name)
            original = require_source_function(function, sources[path], f"{owner.__name__}.{name}",
                                               shell=annotation_shell(function))
            if original.__globals__ is not modules[path].__dict__:
                raise ValueError("allocator defining globals changed")
    for name in ("invalidate", "set_used_size", "is_valid", "get_shapes", "get_dtypes", "get_size"):
        function = getattr(modules[MEMORY].TensorMemoryObj, name)
        require_source_function(function, sources[MEMORY], f"TensorMemoryObj.{name}")
        if function.__globals__ is not modules[MEMORY].__dict__:
            raise ValueError("allocation reader defining globals changed")
    for name in ("reserve_write", "finish_write", "finish_read"):
        original = require_source_function(getattr(modules[L1].L1Manager, name), sources[L1],
                                           f"L1Manager.{name}", shell=lock_shell)
        if original.__globals__ is not modules[L1].__dict__:
            raise ValueError("L1 evidence method defining globals changed")
    original = require_source_function(storage.StorageManager.finish_read_prefetched, sources[STORAGE],
                                       "StorageManager.finish_read_prefetched", shell=trace_shell)
    if original.__globals__ is not storage.__dict__:
        raise ValueError("storage finalization defining globals changed")
    journal.lifetime = lifetime_module.Lifetime(journal, memory_type=modules[MEMORY].TensorMemoryObj,
        tensor_allocator_type=modules[ALLOCATOR].TensorMemoryAllocator, l1_manager_type=modules[L1].L1Manager,
        allocator_roots=((modules[LAZY].LazyMemoryAllocator, "_allocator"),
                         (modules[MIXED].MixedMemoryAllocator, "pin_allocator")))
    journal.event_guard = event_guard
    staged = []
    staged.append(instrument(transfer, sources[TRANSFER], "LMCacheDrivenTransferModule.__init__", [
        (lambda node: isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Attribute) and target.attr == "_device_host_func_dispatcher"
            for target in node.targets), "bind_cache", 1)]))
    staged.append(instrument(transfer, sources[TRANSFER], "LMCacheDrivenTransferModule.close", [
        (called("stop"), "cache_stopped", 1)]))
    for direction in ("store", "retrieve"):
        owner, name, original, fn = instrument(transfer, sources[TRANSFER], f"LMCacheDrivenTransferModule.{direction}", [
            (assigned("obj_keys_per_obj_group"), "journal.selected", 1),
            (called("record_event"), "journal.recorded_event", 2)],
            shell=annotation_shell(getattr(cls, direction)))
        def wrapper(fn, direction):
            @functools.wraps(fn)
            def call(owner, key, instance_id, *args, **kwargs):
                def manager():
                    if observer.cache_owner is None or observer.cache_owner() is not owner:
                        raise ValueError("transfer does not belong to selected cache lifecycle")
                    observer.dispatcher_health(owner._device_host_func_dispatcher)
                    actual = owner._ctx.storage_manager._l1_manager
                    if observer.l1 is not None and observer.l1() is not actual:
                        raise ValueError("multiple selected L1 managers")
                    observer.l1 = weakref.ref(actual)
                journal.observe(manager)
                op = journal.observe(journal.begin, direction, key, instance_id, owner)
                token = observer.op_context.set(op)
                success = False
                try:
                    result = fn(owner, key, instance_id, *args, **kwargs)
                    success = (type(result) is tuple and len(result) == 2
                               and type(result[0]) is bytes and bool(result[0])
                               and result[1] is True)
                    return result
                finally:
                    observer.op_context.reset(token)
                    if op is not None:
                        with journal.condition:
                            op["returned"], op["success"] = True, success
                            journal.emit("result", operation=op["id"], success=success)
                            if not success:
                                journal.emit("failure", reason="transfer_failed_or_missing_completion_handle")
                            journal.active -= 1
                            journal.condition.notify_all()
            return call
        staged.append((owner, name, original, wrapper(fn, direction)))
    staged.append(instrument(transfer, sources[TRANSFER], "_run_object_group_transfer_plan", [
        (lambda node: called("append")(node) and isinstance(node.value.func.value, ast.Name)
         and node.value.func.value.id == "batch_steps", "native_batch", 1),
        (called("execute_object_group_transfer"), "native_done", 1)]))
    owner, name, original, fn = instrument(transfer, sources[TRANSFER], "transfer_kv_per_object_group", [
        (called("multi_layer_block_kv_transfer"), "fallback_kernel", 1),
        (called("lmcache_memcpy_async_h2d"), "fallback_h2d", 1),
        (called("lmcache_memcpy_async_d2h"), "fallback_d2h", 1)])
    staged.append((owner, name, original, functools.wraps(original)(lambda *a, **k: observer.helper(fn, *a, **k))))
    staged.append(instrument(storage, sources[STORAGE], "StorageManager.finish_write", [
        (assigned("failed_keys"), "journal.finalized", 1)], shell=trace_shell))
    downsample = transfer.downsample_and_stage_block_ids
    require_source_function(downsample, sources[TRANSFER], "downsample_and_stage_block_ids")
    if downsample.__globals__ is not transfer.__dict__:
        raise ValueError("block staging defining globals changed")
    @functools.wraps(downsample)
    def raw(context, blocks):
        journal.observe(journal.raw_blocks, context, blocks)
        return downsample(context, blocks)
    staged.append((transfer, "downsample_and_stage_block_ids", downsample, raw))
    submit = transfer.submit_callback_to_stream
    require_source_function(submit, sources[COMPLETION], "submit_callback_to_stream")
    if (submit is not completion.submit_callback_to_stream
            or submit.__globals__ is not completion.__dict__):
        raise ValueError("stream callback submission binding changed")
    staged.append((transfer, "submit_callback_to_stream", submit,
                   functools.wraps(submit)(lambda *a, **k: journal.submit(submit, *a, **k))))
    dispatcher = completion.DeviceHostFuncDispatcher
    register = dispatcher.register
    @functools.wraps(register)
    def registered(instance, kind, handler, payload_type):
        journal.observe(observer.dispatcher_health, instance)
        if kind in ("finish_write", "finish_read_prefetched"):
            journal.observe(observer.callback_binding, kind, handler)
        @functools.wraps(handler)
        def unbound(keys):
            journal.emit("failure", reason="unbound_callback_delivery")
            return handler(keys)
        result = register(instance, kind, unbound, payload_type)
        if kind in ("finish_write", "finish_read_prefetched"):
            @functools.wraps(handler)
            def checked(keys):
                journal.observe(observer.callback_binding, kind, handler)
                return handler(keys)
            journal.observe(register, instance, f"grill_kv_{kind}_v2", journal.handler(kind, checked), tuple[int, payload_type])
        return result
    staged.append((dispatcher, "register", register, registered))
    owner, name, drain, observed_drain = instrument(completion, sources[COMPLETION],
        "DeviceHostFuncDispatcher._drain_once", [
            (lambda node: (called("exception")(node) or called("warning")(node))
             and isinstance(node.value.func, ast.Attribute)
             and isinstance(node.value.func.value, ast.Name)
             and node.value.func.value.id == "logger", "drain_failed", 3)])
    @functools.wraps(drain)
    def drained(instance, *args, **kwargs):
        journal.observe(observer.dispatcher_health, instance)
        try:
            result = observed_drain(instance, *args, **kwargs)
        except BaseException:
            journal.emit("failure", reason="completion_dispatcher_escaped_exception")
            raise
        journal.observe(observer.dispatcher_health, instance)
        journal.observe(journal.harvest)
        return result
    staged.append((dispatcher, "_drain_once", drain, drained))
    stop = dispatcher.stop
    @functools.wraps(stop)
    def stopped(instance, *args, **kwargs):
        journal.observe(observer.dispatcher_health, instance)
        try:
            result = stop(instance, *args, **kwargs)
        except BaseException:
            journal.emit("failure", reason="completion_dispatcher_stop_exception")
            raise
        def finished():
            observer.dispatcher_health(instance, stopped=True)
            observer.cache_stop_observed = True
        journal.observe(finished)
        return result
    staged.append((dispatcher, "stop", stop, stopped))
    release = cls._release_entries
    require_source_function(release, sources[TRANSFER], "LMCacheDrivenTransferModule._release_entries")
    @functools.wraps(release)
    def releasing(instance, entries):
        if entries and not journal.sealed:
            journal.emit("failure", reason="registration_generation_invalidated")
        return release(instance, entries)
    staged.append((cls, "_release_entries", release, releasing))
    globals_added, changed = [], []
    hook_start = len(journal.hooks)
    try:
        for module in (transfer, storage, completion):
            if GLOBAL in module.__dict__:
                raise ValueError("observer global already installed")
            module.__dict__[GLOBAL] = observer
            globals_added.append(module)
        registration.install_cache(journal, transfer, snapshot_module=snapshot,
                                   sources={path: sources[path] for path in registration.CACHE_PINS})
        changed.extend(journal.lifetime.install())
        for owner, name, original, replacement in staged:
            changed.append((owner, name, original, replacement))
            journal.replace(owner, name, replacement)
    except BaseException:
        for owner, name, original, replacement in reversed(changed):
            if getattr(owner, name) is replacement:
                setattr(owner, name, original)
        for module in globals_added:
            module.__dict__.pop(GLOBAL, None)
        del journal.hooks[hook_start:]
        journal.emit("failure", reason="cache_installation_failed")
        raise
    observer.dispatcher_methods = {name: getattr(dispatcher, name) for name in
        ("__init__", "start", "stop", "register", "_execute", "_drain_once", "_run_loop")}
    observer.callback_methods = {name: getattr(storage.StorageManager, name)
                                 for name in ("finish_write", "finish_read_prefetched")}
    observer.l1_callback_methods = {name: getattr(modules[L1].L1Manager, name)
                                   for name in ("finish_write", "finish_read")}
    # Keep defining functions and inherited lifecycle bindings intact for the
    # entire window, accounting for the replacements staged above.
    replaced = {(owner, name) for owner, name, _, _ in staged}
    journal.hooks.extend(binding for binding in lifecycle_bindings if binding[:2] not in replaced)
    for owner, names in (
            (modules[MEMORY].TensorMemoryObj, ("is_valid", "get_shapes", "get_dtypes", "get_size")),
            (modules[L1].L1Manager, ("finish_write", "finish_read")),
            (storage.StorageManager, ("finish_read_prefetched",)),
            (trace, ("enable_tracing", "publish_call_event")),
            (nvtx, ("annotate", "libnvtx_push_range", "libnvtx_pop_range")),
            (utils, ("_lmcache_nvtx_annotate", "_get_color_for_nvtx", "annotate")),
            (completion, ("submit_callback_to_stream", "PeriodicThread", "PeriodicThreadRegistry")),
            (transfer, ("DeviceHostFuncDispatcher",))):
        journal.hooks.extend((owner, name, getattr(owner, name)) for name in names)
    for name in ("is_observability_enabled", "get_event_bus", "next_transfer_key"):
        journal.hooks.append((transfer, name, getattr(transfer, name)))
    journal.hooks.append((transfer, "_HAS_NATIVE_OBJECT_GROUP_TRANSFER", native_enabled))
    journal.hooks.append((transfer, "_HAS_TRANSFER_PHASE_TIMING", observer.phase_timing))
    journal.hooks.append((transfer, "device_ops", observer.device_ops))
    if observer.native_function is not None:
        journal.hooks.append((observer.device_ops, "execute_object_group_transfer", observer.native_function))
    journal.emit("installed", pins=expected, scope=journal.scope)


@atomic_install
def install_worker(journal, modules, sources):
    expected = registration.pins("worker")
    registration._validate_sources(sources, expected)
    if journal.role != "worker" or journal.hooks or journal.closed:
        raise ValueError("invalid worker installation state")
    adapter = modules[WORKER]
    nvtx, utils = modules[NVTX], modules[UTILS]
    require_source_function(utils._lmcache_nvtx_annotate, sources[UTILS], "_lmcache_nvtx_annotate")
    require_source_function(utils._get_color_for_nvtx, sources[UTILS], "_get_color_for_nvtx")
    if adapter._lmcache_nvtx_annotate is not utils._lmcache_nvtx_annotate:
        raise ValueError("worker NVTX decorator defining binding changed")
    if utils.annotate is not nvtx.annotate:
        require_source_function(utils.annotate, sources[UTILS], "annotate")
    def annotation_shell(function):
        return ((sources[NVTX], "annotate.__call__.<locals>.inner", nvtx)
                if hasattr(function, "__wrapped__") else None)
    journal.event_guard = passive_event_guard(adapter, sources)
    geometry_roster(sources)
    def registration_guard(worker):
        geometry_roster(sources)
        context_module = sys.modules["lmcache.v1.multiprocess.transfer_context.worker_transfer"]
        transport = sys.modules["lmcache.v1.multiprocess.transport.zmq_impl.client"]
        context = worker.transfer_ctx
        if (type(context) is not context_module.LMCacheDrivenTransferContext
                or type(context._req_client) is not transport.ZmqMultiprocessClient
                or context._closed or not context._registration_request_sent
                or context._device is None or worker.lazy_offload is not False
                or worker.experimental or worker.dispatcher is not None):
            raise ValueError("unsupported worker transport, lifecycle or optional feature configuration")
        journal.event_guard(context._event_backend, None)
    journal.registration_guard = registration_guard
    observer = Observer(journal)
    snapshot = adjacent("kv-registration.py")
    staged = [instrument(adapter, sources[WORKER], f"LMCacheMPWorkerAdapter.submit_{direction}_request",
                         [(assigned("future"), f"worker_{direction}", 1)],
                         shell=annotation_shell(getattr(adapter.LMCacheMPWorkerAdapter,
                                                       f"submit_{direction}_request")))
              for direction in ("store", "retrieve")]
    staged = [(owner, name, original, observer.worker_call(replacement))
              for owner, name, original, replacement in staged]
    for name in ("get_finished", "get_finished_with_lazy_offload"):
        owner, method, original, replacement = instrument(
            adapter, sources[WORKER], f"LMCacheMPWorkerAdapter.{name}", [
                (assigned("s_result"), "worker_store_result", 1),
                (assigned("r_result"), "worker_retrieve_result", 1)],
            shell=annotation_shell(getattr(adapter.LMCacheMPWorkerAdapter, name)))
        staged.append((owner, method, original, observer.worker_call(replacement)))
    owner = adapter.LMCacheMPWorkerAdapter
    name = "get_block_ids_with_load_errors"
    original = getattr(owner, name)
    require_source_function(original, sources[WORKER], f"LMCacheMPWorkerAdapter.{name}")
    staged.append((owner, name, original, observer.worker_call(original)))
    base = modules[WORKER_BASE].WorkerWrapperBase
    execute = base.execute_model
    require_source_function(execute, sources[WORKER_BASE], "WorkerWrapperBase.execute_model")
    if execute.__globals__ is not modules[WORKER_BASE].__dict__:
        raise ValueError("scheduling delegate defining globals changed")
    @functools.wraps(execute)
    def scheduled(instance, scheduler_output, *args, **kwargs):
        def observe():
            if journal.closed:
                raise ValueError("scheduled work outside journal")
            ids = sorted(scheduler_output.num_scheduled_tokens)
            if len(ids) > 64 or any(type(value) is not str or not 0 < len(value.encode()) <= 256 for value in ids):
                raise ValueError("scheduled request bound")
            counts = scheduler_output.num_scheduled_tokens.values()
            if any(type(n) is not int or n < 0 for n in counts):
                raise ValueError("scheduled token count")
            journal.emit("scheduled_step", req_ids=ids,
                         total_tokens=sum(scheduler_output.num_scheduled_tokens.values()),
                         connector_meta=scheduler_output.kv_connector_metadata is not None)
        journal.observe(observe)
        return execute(instance, scheduler_output, *args, **kwargs)
    staged.append((base, "execute_model", execute, scheduled))
    if GLOBAL in adapter.__dict__:
        raise ValueError("duplicate worker observer")
    adapter.__dict__[GLOBAL] = observer
    try:
        registration.install_worker(journal, modules[CONNECTOR], adapter, snapshot_module=snapshot,
                                    sources={path: sources[path] for path in registration.WORKER_PINS})
        for owner, name, original, replacement in staged:
            journal.replace(owner, name, replacement)
    except BaseException:
        adapter.__dict__.pop(GLOBAL, None)
        journal.emit("failure", reason="worker_installation_failed")
        raise
    for owner, names in (
            (nvtx, ("annotate", "libnvtx_push_range", "libnvtx_pop_range")),
            (utils, ("_lmcache_nvtx_annotate", "_get_color_for_nvtx", "annotate"))):
        journal.hooks.extend((owner, name, getattr(owner, name)) for name in names)
    journal.emit("installed", pins=expected, scope=journal.scope)


def install_frontend(journal, engine_class, runtime_request, source):
    # The runtime752 bootstrap already verifies the complete prospective source
    # set; this hook additionally refuses methods from a different source body.
    if journal.role != "frontend" or journal.hooks or journal.closed:
        raise ValueError("invalid frontend installation state")
    original = engine_class._add_request
    if getattr(original, "_grill_kv_v2", False):
        raise ValueError("frontend already instrumented")
    source_hash = hashlib.sha256(source).hexdigest()
    if source_hash != registration.pins("frontend")[ASYNC]:
        raise ValueError("frontend source is not the selected public752 revision")
    original_source = inspect.getsourcefile(original)
    if original_source is None or hashlib.sha256(Path(original_source).read_bytes()).hexdigest() != source_hash:
        raise ValueError("loaded frontend differs from supplied source")
    @functools.wraps(original)
    async def observed(engine, request, prompt, parent_req, index, queue):
        context = runtime_request.get()
        with journal.condition:
            admitted = not journal.closed and not journal.failed
            if admitted:
                journal.active += 1
            else:
                journal.emit("failure", reason="frontend_admission_after_close")
        try:
            result = await original(engine, request, prompt, parent_req, index, queue)
            if admitted:
                if context is None or not context["active"]:
                    journal.emit("failure", reason="unbound_engine_request")
                else:
                    journal.emit("binding", request=context["id"], engine_request=request.request_id,
                                 parent=None if parent_req is None else parent_req.request_id, index=index)
            return result
        finally:
            if admitted:
                with journal.condition:
                    journal.active -= 1
                    journal.condition.notify_all()
    observed._grill_kv_v2 = True
    journal.replace(engine_class, "_add_request", observed)
    journal.emit("installed", pins={ASYNC: source_hash}, scope=journal.scope)
