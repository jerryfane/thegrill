"""Source-pinned observers for registered KV topology and layout metadata.

This module imports only the Python standard library.  It installs observers at
three pinned LMCache source boundaries without importing a backend or retaining
configuration, adapter, context, or tensor objects strongly.  Registration
metadata establishes registered state only; it is not persistence, transfer, or
device-completion evidence.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import json
from pathlib import Path
import types
import weakref
from collections.abc import Mapping
from typing import Any

def pins(role):
    inventory = json.loads(Path(__file__).with_name("kv-v2-pins.json").read_text())
    if role not in inventory:
        raise ValueError("unsupported source role")
    return inventory[role]


# Registration uses a strict subset of the complete installation inventory.
WORKER_PINS = {path: digest for path, digest in pins("worker").items()
               if path.startswith("lmcache/")}
_CACHE_REGISTRATION_PATHS = (
    "lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py",
    "lmcache/v1/multiprocess/engine_context.py",
    "lmcache/v1/multiprocess/protocols/engine.py",
    "lmcache/v1/multiprocess/custom_types.py",
    "lmcache/v1/multiprocess/group_view.py",
    "lmcache/v1/kv_layer_groups.py",
    "lmcache/v1/platform/base/cache_context.py",
    "lmcache/v1/platform/cuda/cache_context.py",
    "lmcache/v1/distributed/api.py",
    "lmcache/utils.py",
    "lmcache/v1/gpu_connector/utils.py",
    "lmcache/lmcache_native.pyi",
)
CACHE_PINS = {path: digest for path, digest in pins("cache").items()
              if path in _CACHE_REGISTRATION_PATHS}

__all__ = ["WORKER_PINS", "CACHE_PINS", "install_worker", "install_cache"]

_CONNECTOR_PATH = "lmcache/integration/vllm/lmcache_mp_connector.py"
_ADAPTER_PATH = "lmcache/integration/vllm/vllm_multi_process_adapter.py"
_TRANSFER_PATH = "lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py"
_CONNECTOR_MODULE = "lmcache.integration.vllm.lmcache_mp_connector"
_ADAPTER_MODULE = "lmcache.integration.vllm.vllm_multi_process_adapter"
_TRANSFER_MODULE = "lmcache.v1.multiprocess.modules.lmcache_driven_transfer"
_OBSERVER_GLOBAL = "_thegrill_kv_registration_observer_v2"
_MARKER = "_thegrill_kv_registration_hook_v2"
_MAX_WORKER_IDENTITIES = 64
_MAX_CACHE_IDENTITIES = 128
_MAX_ENGINE_INSTANCE = (1 << 63) - 1
_MISSING = object()


def _validate_sources(sources: Mapping[str, bytes], pins: Mapping[str, str]) -> None:
    if not isinstance(sources, Mapping):
        raise TypeError("sources must be a mapping")
    if set(sources) != set(pins):
        missing = sorted(set(pins) - set(sources))
        extra = sorted(set(sources) - set(pins))
        raise ValueError(f"pinned source set mismatch: missing={missing}, extra={extra}")
    for path, expected in pins.items():
        source = sources[path]
        if type(source) is not bytes:
            raise TypeError(f"pinned source is not bytes: {path}")
        if hashlib.sha256(source).hexdigest() != expected:
            raise ValueError(f"pinned source hash mismatch: {path}")


def _require_module(module: Any, expected_name: str) -> None:
    if not isinstance(module, types.ModuleType) or module.__name__ != expected_name:
        raise ValueError(f"incompatible loaded module: expected {expected_name}")
    if _OBSERVER_GLOBAL in module.__dict__:
        raise ValueError(f"registration observer already installed in {expected_name}")


def _parse_method(
    source: bytes,
    path: str,
    class_name: str,
    method_name: str,
) -> tuple[ast.Module, ast.FunctionDef | ast.AsyncFunctionDef]:
    try:
        text = source.decode("utf-8")
        tree = ast.parse(text, filename=path)
    except (UnicodeDecodeError, SyntaxError) as error:
        raise ValueError(f"pinned source cannot be parsed: {path}") from error
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if len(classes) != 1:
        raise ValueError(f"ambiguous pinned class boundary: {class_name}")
    methods = [
        node
        for node in classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    ]
    if len(methods) != 1:
        raise ValueError(f"ambiguous pinned method boundary: {class_name}.{method_name}")
    return tree, methods[0]


def _walk_code_objects(code: types.CodeType):
    yield code
    for value in code.co_consts:
        if isinstance(value, types.CodeType):
            yield from _walk_code_objects(value)


def source_code(source, path, qualified_name, lineno=None):
    module_code = compile(source, path, "exec", dont_inherit=True)
    candidates = [
        code for code in _walk_code_objects(module_code)
        if (getattr(code, "co_qualname", None) == qualified_name
            or (not hasattr(code, "co_qualname")
                and code.co_name == qualified_name.rsplit(".", 1)[-1]
                and code.co_firstlineno == lineno))
    ]
    if len(candidates) != 1:
        raise ValueError(f"ambiguous compiled method boundary: {qualified_name}")
    return candidates[0]


def _constant_signature(value: Any) -> Any:
    if isinstance(value, types.CodeType):
        return ("code", _code_signature(value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_constant_signature(item) for item in value))
    if isinstance(value, frozenset):
        return ("frozenset", frozenset(_constant_signature(item) for item in value))
    return (type(value), value)


def _code_signature(code: types.CodeType) -> tuple[Any, ...]:
    return (
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_flags,
        code.co_code,
        tuple(_constant_signature(value) for value in code.co_consts),
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
    )


def _literal_defaults(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[tuple[Any, ...] | None, dict[str, Any] | None]:
    try:
        positional = tuple(ast.literal_eval(value) for value in method.args.defaults)
        keyword = {
            argument.arg: ast.literal_eval(value)
            for argument, value in zip(
                method.args.kwonlyargs, method.args.kw_defaults, strict=True
            )
            if value is not None
        }
    except (ValueError, TypeError) as error:
        raise ValueError(f"non-literal defaults at pinned method {method.name}") from error
    return positional or None, keyword or None


def _require_original_method(
    module: types.ModuleType,
    source: bytes,
    path: str,
    class_name: str,
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[type, types.FunctionType]:
    owner = module.__dict__.get(class_name)
    if not isinstance(owner, type):
        raise ValueError(f"loaded class is missing: {class_name}")
    current = vars(owner).get(method.name)
    if not isinstance(current, types.FunctionType):
        raise ValueError(f"loaded method is not a direct function: {class_name}.{method.name}")
    if getattr(current, _MARKER, False):
        raise ValueError(f"method is already observed: {class_name}.{method.name}")
    if current.__module__ != module.__name__ or current.__qualname__ != (
        f"{class_name}.{method.name}"
    ):
        raise ValueError(f"loaded method ownership changed: {class_name}.{method.name}")
    if current.__globals__ is not module.__dict__:
        raise ValueError(f"loaded method globals changed: {class_name}.{method.name}")
    if "__class__" in current.__code__.co_freevars:
        class_index = current.__code__.co_freevars.index("__class__")
        closure = current.__closure__
        if closure is None or closure[class_index].cell_contents is not owner:
            raise ValueError(
                f"loaded method class binding changed: {class_name}.{method.name}"
            )
    expected = source_code(source, path, f"{class_name}.{method.name}", method.lineno)
    if _code_signature(current.__code__) != _code_signature(expected):
        raise ValueError(f"loaded method body changed: {class_name}.{method.name}")
    defaults, kwdefaults = _literal_defaults(method)
    if current.__defaults__ != defaults or current.__kwdefaults__ != kwdefaults:
        raise ValueError(f"loaded method defaults changed: {class_name}.{method.name}")
    return owner, current


def compile_body(tree, method, module, path, original):
    futures = [node for node in tree.body
               if isinstance(node, ast.ImportFrom) and node.module == "__future__"]
    unit = ast.fix_missing_locations(ast.Module(body=[*futures, method], type_ignores=[]))
    local = {}
    exec(compile(unit, path, "exec", dont_inherit=True), module.__dict__, local)
    replacement = local[method.name]
    replacement.__defaults__ = original.__defaults__
    replacement.__kwdefaults__ = original.__kwdefaults__
    return replacement


def _compile_method(
    tree: ast.Module,
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    module: types.ModuleType,
    path: str,
    original: types.FunctionType,
) -> types.FunctionType:
    if method.decorator_list:
        raise ValueError(f"decorated pinned method cannot be replayed: {method.name}")
    if "__class__" in original.__code__.co_freevars:
        raise ValueError(f"class-closure method cannot be compiled standalone: {method.name}")
    replacement = compile_body(tree, method, module, path, original)
    functools.update_wrapper(replacement, original)
    setattr(replacement, _MARKER, True)
    return replacement


def _is_connector_adapter_assignment(node: ast.AST) -> bool:
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        return False
    target = node.targets[0]
    value = node.value
    return (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
        and target.attr == "worker_adapter"
        and isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "LMCacheMPWorkerAdapter"
    )


def _validate_connector_boundary(method: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
    matches = [node for node in ast.walk(method) if _is_connector_adapter_assignment(node)]
    if len(matches) != 1:
        raise ValueError("pinned connector worker-adapter boundary changed")


def _is_register_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    call = node.value
    function = call.func
    if not (
        isinstance(function, ast.Attribute)
        and function.attr == "register"
        and isinstance(function.value, ast.Name)
        and function.value.id == "transfer_ctx"
    ):
        return False
    layout_keywords = [
        keyword
        for keyword in call.keywords
        if keyword.arg == "layout_hints"
        and isinstance(keyword.value, ast.Name)
        and keyword.value.id == "layout_hints"
    ]
    return len(layout_keywords) == 1


def _instrument_worker_method(
    tree: ast.Module,
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    module: types.ModuleType,
    path: str,
    original: types.FunctionType,
) -> types.FunctionType:
    matches = [node for node in ast.walk(method) if _is_register_call(node)]
    if len(matches) != 1:
        raise ValueError("pinned worker registration ACK boundary changed")
    target = matches[0]
    _insert_observer(method, target, "observe_worker(self, layout_hints)")
    return _compile_method(tree, method, module, path, original)


def _is_cache_context_insertion(node: ast.AST) -> bool:
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        return False
    target = node.targets[0]
    if not (
        isinstance(target, ast.Subscript)
        and isinstance(target.value, ast.Attribute)
        and isinstance(target.value.value, ast.Name)
        and target.value.value.id == "self"
        and target.value.attr == "_cache_contexts"
        and isinstance(target.slice, ast.Name)
        and target.slice.id == "instance_id"
    ):
        return False
    value = node.value
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "ContextEntry"
    )


def _instrument_cache_method(
    tree: ast.Module,
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    module: types.ModuleType,
    path: str,
    original: types.FunctionType,
) -> types.FunctionType:
    matches = [
        statement
        for statement in method.body
        if isinstance(statement, ast.With)
        and sum(
            1 for node in ast.walk(statement) if _is_cache_context_insertion(node)
        )
        == 1
    ]
    if len(matches) != 1:
        raise ValueError("pinned cache ContextEntry boundary changed")
    target = matches[0]
    _insert_observer(method, target, "observe_cache(locals())")
    attempt = ast.parse(f"{_OBSERVER_GLOBAL}.observe_attempt(instance_id)").body[0]
    method.body.insert(1 if ast.get_docstring(method) is not None else 0, attempt)
    return _compile_method(tree, method, module, path, original)


def _insert_observer(method, target, call):
    observer = ast.parse(f"{_OBSERVER_GLOBAL}.{call}").body[0]
    ast.copy_location(observer, target)

    class InsertAfter(ast.NodeTransformer):
        def visit(self, node):
            if node is target:
                return [node, observer]
            return super().visit(node)

    InsertAfter().visit(method)


def _require_open(journal: Any) -> None:
    if journal.closed or journal.sealed or journal.failed:
        raise ValueError("registration observation is outside the open journal")


def _require_scalar(value: Any, expected: type, name: str) -> Any:
    if type(value) is not expected:
        raise ValueError(f"registered configuration scalar has wrong type: {name}")
    return value


def _require_engine_instance(value: Any) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_ENGINE_INSTANCE:
        raise ValueError("registered engine instance is outside the contract")
    return value


def _require_explicit_layout(layout_hints: Any) -> None:
    if type(layout_hints) is not dict or set(layout_hints) != {"kv_layout"}:
        raise ValueError("layout_hints must contain the required kv_layout field")


def _emit_snapshot(journal: Any, snapshot: Any, expected_kind: str) -> None:
    if type(snapshot) is not dict or snapshot.get("kind") != expected_kind:
        raise ValueError(f"snapshot did not produce {expected_kind}")
    _require_explicit_layout(snapshot.get("layout_hints", _MISSING))
    journal.emit(**snapshot)


class _WorkerObserver:
    __slots__ = (
        "_journal",
        "_snapshot",
        "_adapter_type",
        "_configs",
        "_observed",
        "_instances",
    )

    def __init__(self, journal: Any, snapshot: Any, adapter_type: type) -> None:
        self._journal = journal
        self._snapshot = snapshot
        self._adapter_type = adapter_type
        self._configs: weakref.WeakKeyDictionary[Any, types.SimpleNamespace] = (
            weakref.WeakKeyDictionary()
        )
        self._observed: weakref.WeakSet[Any] = weakref.WeakSet()
        self._instances: set[int] = set()

    def observe_connector(self, connector: Any, vllm_config: Any) -> None:
        self._journal.observe(self._capture_connector, connector, vllm_config)

    def _capture_connector(self, connector: Any, vllm_config: Any) -> None:
        attributes = vars(connector)
        if "worker_adapter" not in attributes:
            return
        _require_open(self._journal)
        adapter = attributes["worker_adapter"]
        if type(adapter) is not self._adapter_type:
            raise ValueError("connector has an unknown worker adapter")
        if adapter in self._configs or adapter in self._observed:
            raise ValueError("duplicate worker adapter association")
        if len(self._configs) + len(self._observed) >= _MAX_WORKER_IDENTITIES:
            raise ValueError("worker adapter association budget exceeded")

        parallel = vllm_config.parallel_config
        model = vllm_config.model_config
        scalar_config = types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(
                decode_context_parallel_size=_require_scalar(
                    parallel.decode_context_parallel_size, int, "decode_context_parallel_size"
                ),
                data_parallel_size=_require_scalar(
                    parallel.data_parallel_size, int, "data_parallel_size"
                ),
                world_size=_require_scalar(parallel.world_size, int, "world_size"),
                rank=_require_scalar(parallel.rank, int, "rank"),
                tensor_parallel_size=_require_scalar(
                    parallel.tensor_parallel_size, int, "tensor_parallel_size"
                ),
                pipeline_parallel_size=_require_scalar(
                    parallel.pipeline_parallel_size, int, "pipeline_parallel_size"
                ),
            ),
            model_config=types.SimpleNamespace(
                use_mla=_require_scalar(model.use_mla, bool, "use_mla"),
                is_hybrid=_require_scalar(model.is_hybrid, bool, "is_hybrid"),
            ),
        )
        self._configs[adapter] = scalar_config

    def observe_worker(self, worker: Any, layout_hints: Any) -> None:
        self._journal.observe(self._observe_worker, worker, layout_hints)

    def _observe_worker(self, worker: Any, layout_hints: Any) -> None:
        _require_open(self._journal)
        self._journal.registration_guard(worker)
        if type(worker) is not self._adapter_type:
            raise ValueError("worker registration came from an unknown adapter")
        if worker in self._observed:
            raise ValueError("duplicate worker registration observation")
        if worker not in self._configs:
            raise ValueError("worker registration has no connector association")
        instance = _require_engine_instance(worker.instance_id)
        if instance in self._instances:
            raise ValueError("duplicate worker engine instance")
        if len(self._instances) >= _MAX_WORKER_IDENTITIES:
            raise ValueError("worker registration identity budget exceeded")

        scalar_config = self._configs.pop(worker)
        self._observed.add(worker)
        self._instances.add(instance)
        snapshot = self._snapshot(worker, scalar_config, layout_hints)
        _emit_snapshot(self._journal, snapshot, "worker_registration")


class _CacheObserver:
    __slots__ = ("_journal", "_snapshot", "_instances")

    def __init__(self, journal: Any, snapshot: Any) -> None:
        self._journal = journal
        self._snapshot = snapshot
        self._instances: set[int] = set()

    def observe_attempt(self, instance_id: int) -> None:
        def check():
            _require_open(self._journal)
            if instance_id in self._instances:
                raise ValueError("registration generation cannot be reused")
        self._journal.observe(check)

    def observe_cache(self, values: dict[str, Any]) -> None:
        self._journal.observe(self._observe_cache, values)

    def _observe_cache(self, values: dict[str, Any]) -> None:
        _require_open(self._journal)
        self._journal.registration_guard(None)
        if type(values) is not dict:
            raise ValueError("cache observer did not receive method locals")
        instance = _require_engine_instance(values.get("instance_id", _MISSING))
        if instance in self._instances:
            raise ValueError("duplicate cache registration observation")
        if len(self._instances) >= _MAX_CACHE_IDENTITIES:
            raise ValueError("cache registration identity budget exceeded")

        self._instances.add(instance)
        snapshot = self._snapshot(values)
        _emit_snapshot(self._journal, snapshot, "cache_registration")


def _wrap_connector_init(
    original: types.FunctionType,
    observer: _WorkerObserver,
) -> types.FunctionType:
    @functools.wraps(original)
    def observed(connector: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(connector, *args, **kwargs)
        vllm_config = args[0] if args else kwargs.get("vllm_config", _MISSING)
        observer.observe_connector(connector, vllm_config)
        return result

    setattr(observed, _MARKER, True)
    return observed


def _apply_replacements(
    journal: Any,
    changes: list[tuple[Any, str, Any, Any]],
) -> None:
    attempted: list[tuple[Any, str, Any, Any]] = []
    try:
        for owner, name, replacement, original in changes:
            namespace = vars(owner)
            if original is _MISSING:
                if name in namespace:
                    raise ValueError(f"hook global changed during installation: {name}")
            elif namespace.get(name, _MISSING) is not original:
                raise ValueError(f"hook target changed during installation: {name}")
            attempted.append((owner, name, replacement, original))
            journal.replace(owner, name, replacement)
            if vars(owner).get(name, _MISSING) is not replacement:
                raise RuntimeError(f"journal did not install replacement: {name}")
    except BaseException:
        for owner, name, replacement, original in reversed(attempted):
            if vars(owner).get(name, _MISSING) is not replacement:
                continue
            if original is _MISSING:
                delattr(owner, name)
            else:
                setattr(owner, name, original)
        raise


def install_worker(
    journal: Any,
    connector_module: types.ModuleType,
    adapter_module: types.ModuleType,
    *,
    snapshot_module: Any,
    sources: Mapping[str, bytes],
) -> None:
    """Install the source-pinned worker registered-state observers."""
    _require_open(journal)
    _validate_sources(sources, WORKER_PINS)
    snapshot = snapshot_module.worker_registration
    _require_module(connector_module, _CONNECTOR_MODULE)
    _require_module(adapter_module, _ADAPTER_MODULE)

    connector_tree, connector_method = _parse_method(
        sources[_CONNECTOR_PATH],
        _CONNECTOR_PATH,
        "LMCacheMPConnector",
        "__init__",
    )
    adapter_tree, adapter_method = _parse_method(
        sources[_ADAPTER_PATH],
        _ADAPTER_PATH,
        "LMCacheMPWorkerAdapter",
        "_send_register_kv_caches_request",
    )
    connector_class, connector_init = _require_original_method(
        connector_module,
        sources[_CONNECTOR_PATH],
        _CONNECTOR_PATH,
        "LMCacheMPConnector",
        connector_method,
    )
    adapter_class, register_method = _require_original_method(
        adapter_module,
        sources[_ADAPTER_PATH],
        _ADAPTER_PATH,
        "LMCacheMPWorkerAdapter",
        adapter_method,
    )
    if connector_module.__dict__.get("LMCacheMPWorkerAdapter") is not adapter_class:
        raise ValueError("connector worker-adapter binding is incompatible")
    _validate_connector_boundary(connector_method)

    observed_register = _instrument_worker_method(
        adapter_tree,
        adapter_method,
        adapter_module,
        _ADAPTER_PATH,
        register_method,
    )
    observer = _WorkerObserver(journal, snapshot, adapter_class)
    observed_init = _wrap_connector_init(connector_init, observer)

    _apply_replacements(
        journal,
        [
            (connector_module, _OBSERVER_GLOBAL, observer, _MISSING),
            (adapter_module, _OBSERVER_GLOBAL, observer, _MISSING),
            (connector_class, "__init__", observed_init, connector_init),
            (
                adapter_class,
                "_send_register_kv_caches_request",
                observed_register,
                register_method,
            ),
        ],
    )


def install_cache(
    journal: Any,
    transfer_module: types.ModuleType,
    *,
    snapshot_module: Any,
    sources: Mapping[str, bytes],
) -> None:
    """Install the source-pinned cache registered-state observer."""
    _require_open(journal)
    _validate_sources(sources, CACHE_PINS)
    snapshot = snapshot_module.cache_registration
    _require_module(transfer_module, _TRANSFER_MODULE)

    transfer_tree, transfer_method = _parse_method(
        sources[_TRANSFER_PATH],
        _TRANSFER_PATH,
        "LMCacheDrivenTransferModule",
        "register_kv_cache",
    )
    transfer_class, register_method = _require_original_method(
        transfer_module,
        sources[_TRANSFER_PATH],
        _TRANSFER_PATH,
        "LMCacheDrivenTransferModule",
        transfer_method,
    )
    observed_register = _instrument_cache_method(
        transfer_tree,
        transfer_method,
        transfer_module,
        _TRANSFER_PATH,
        register_method,
    )
    observer = _CacheObserver(journal, snapshot)

    _apply_replacements(
        journal,
        [
            (transfer_module, _OBSERVER_GLOBAL, observer, _MISSING),
            (transfer_class, "register_kv_cache", observed_register, register_method),
        ],
    )
