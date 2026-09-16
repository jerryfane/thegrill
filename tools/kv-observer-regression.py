#!/usr/bin/env python3
"""CPU-only candidate observer regressions; no backend imports or admission.

Execute exact pinned adapter method bodies with inert transport futures. Device
and transport behavior are inputs, not simulated proof of a copy. The assertions
cover source failure bookkeeping, observer identity and incomplete fences.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import tempfile
import sys
from unittest.mock import patch
import types


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), Path(__file__).with_name(name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Future:
    def __init__(self, result=True, ready=True, error=None):
        self.value, self.ready, self.error = result, ready, error
        self.queries = self.results = 0

    def query(self):
        self.queries += 1
        if self.error is not None:
            raise self.error
        return self.ready

    def result(self, timeout):
        self.results += 1
        return self.value


class Event:
    def __init__(self, ready=True):
        self.ready, self.queries = ready, 0

    def query(self):
        self.queries += 1
        return self.ready


def adapter_module(hooks, raw, journal):
    module = types.ModuleType('cpu_candidate_adapter')
    module.__file__ = '<pinned-candidate-adapter>'
    module.logger = logging.getLogger('candidate-regression')
    module.IPCCacheServerKey = types.SimpleNamespace
    # Annotation carriers only; runtime method bodies remain source-derived.
    module.LoadStoreOp = types.SimpleNamespace
    module._IpcEvent = Event
    module.Any = object
    cls = type('LMCacheMPWorkerAdapter', (), {})
    module.LMCacheMPWorkerAdapter = cls
    names = ('submit_retrieve_request', 'submit_store_request', 'get_finished', 'get_finished_with_lazy_offload',
             'get_block_ids_with_load_errors', '_process_finished_stores',
             '_update_and_get_finished_store', '_create_key')
    tree = ast.parse(raw)
    class_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls.__name__)
    for name in names:
        node = next(n for n in class_node.body if isinstance(n, ast.FunctionDef) and n.name == name)
        code = hooks.registration.source_code(raw, module.__file__, f'{cls.__name__}.{name}')
        fn = types.FunctionType(code, module.__dict__, name)
        fn.__defaults__, fn.__kwdefaults__ = hooks.registration._literal_defaults(node)
        setattr(cls, name, fn)
    observer = hooks.Observer(journal)
    module.__dict__[hooks.GLOBAL] = observer
    for name in ('get_finished', 'get_finished_with_lazy_offload'):
        owner, method, _, fn = hooks.instrument(module, raw, f'{cls.__name__}.{name}', [
            (hooks.assigned('s_result'), 'worker_store_result', 1),
            (hooks.assigned('r_result'), 'worker_retrieve_result', 1)])
        journal.replace(owner, method, observer.worker_call(fn))
    for direction in ('store', 'retrieve'):
        owner, method, _, fn = hooks.instrument(module, raw, f'{cls.__name__}.submit_{direction}_request', [
            (hooks.assigned('future'), f'worker_{direction}', 1)])
        journal.replace(owner, method, observer.worker_call(fn))
    journal.replace(cls, 'get_block_ids_with_load_errors', observer.worker_call(cls.get_block_ids_with_load_errors))
    return cls, observer


def worker_case(hooks, protocol, raw, root, case, lazy, direction):
    path = root / f'{direction}-{case}-{lazy}.jsonl'
    journal = protocol.Journal(path, role='worker')
    cls, observer = adapter_module(hooks, raw, journal)
    worker = cls()
    worker.instance_id, worker.model_name, worker.world_size, worker.worker_id = 17, 'synthetic', 1, 0
    worker.parallel_strategy = types.SimpleNamespace(num_kv_readers=1)
    worker.is_healthy = case != 'drop'
    worker.lazy_offload = lazy
    worker.is_kv_writer = True
    worker.request_telemetry = types.SimpleNamespace(on_request_store_finished=lambda **kwargs: None)
    worker.dispatcher = None
    worker.kv_caches, worker.blocks_in_chunk = {}, 1
    worker.error_block_ids, worker._dropped_retrieves = set(), set()
    worker.store_futures, worker.retrieve_futures = {}, {}
    worker.store_events, worker.retrieve_events = {}, {}
    worker.finished_stores, worker.previously_finished, worker._returned_finished = set(), set(), set()
    worker._completed_store_requests = {}
    worker._ensure_heartbeat_started = lambda: None
    worker._block_ids_per_group = lambda op: [op.flat_block_ids]
    error = RuntimeError('synthetic transport error') if case == 'exception' else None
    future = Future(result=case != 'false', ready=case != 'incomplete', error=error)
    worker.transfer_ctx = types.SimpleNamespace(submit_retrieve=lambda *a, **k: future,
                                               submit_store=lambda *a, **k: future)
    journal.registrations[17] = {}
    op = types.SimpleNamespace(token_ids=[1], start=0, end=1, flat_block_ids=[7],
                               skip_first_n_tokens=1 if case == 'partial-prefix' else 0)
    request_configs = {'synthetic': True} if case == 'request-config' else None
    getattr(worker, f'submit_{direction}_request')('request', op, Event(), request_configs=request_configs)
    if case == 'unhealthy-drain':
        worker.is_healthy = False
    if case == 'stale':
        if direction == 'retrieve':
            worker.retrieve_futures['request'] = (Future(), [7])
        else:
            worker.store_futures['request'] = Future()
    try:
        result = worker.get_finished_with_lazy_offload() if lazy else worker.get_finished(set())
    except RuntimeError as caught:
        assert caught is error and case == 'exception'
    else:
        assert case != 'exception'
        if direction == 'retrieve' and case in ('false', 'drop', 'unhealthy-drain'):
            assert 'request' in result[1]
            assert worker.get_block_ids_with_load_errors() == {7}
        if direction == 'retrieve' and case in ('success', 'request-config', 'partial-prefix', 'stale'):
            assert 'request' in result[1]
        if direction == 'store' and case in ('success', 'false', 'request-config', 'stale'):
            assert 'request' not in worker.store_futures
            assert ('request' in worker._completed_store_requests if lazy else 'request' in worker.finished_stores)
        if case == 'incomplete':
            assert not result[1] and 'request' in getattr(worker, direction + '_futures')
    # Only the runtime's existing query/result calls are allowed.
    assert future.queries == (0 if case in ('drop', 'unhealthy-drain', 'stale') else 1)
    assert future.results == (1 if case in ('success', 'false', 'request-config', 'partial-prefix') else 0)
    complete = journal.seal(0)
    assert complete == (case == 'success'), (case, lazy, complete)
    rows = [json.loads(line)['event'] for line in path.read_text().splitlines()]
    assert not any(row['kind'] == 'device_complete' for row in rows)
    return {'case': case, 'lazy': lazy, 'direction': direction, 'complete': complete}


def finalization_cases(protocol, root):
    outcomes = []
    for case in ('success', 'reordered', 'missing', 'partial', 'wrong', 'duplicate',
                 'failed', 'repeated', 'wrong-direction'):
        path = root / f'finalization-{case}.jsonl'
        # This isolates the shared operation-finalization state machine, not a
        # cache installer/lifecycle proof. Whole cache fences additionally need
        # the real source-owned dispatcher stop (covered by installer smokes).
        journal = protocol.Journal(path, role='frontend')
        journal.replace(types.SimpleNamespace(), 'fixture_hook', len)
        journal.registrations[1] = {}
        key = types.SimpleNamespace(world_size=1, worker_id=0, num_kv_readers=1,
                                    request_configs=None, request_id='request',
                                    model_name='fixture', cache_salt='fixture', start=0, end=256)
        direction = 'retrieve' if case == 'wrong-direction' else 'store'
        op = journal.begin(direction, key, 1, None)
        # Isolate storage finalization after the other transfer witnesses.
        op.update(returned=True, success=True, recorded=True, device_complete=True)
        journal.active = 0
        keys = [types.SimpleNamespace(chunk_hash=bytes([n]) * 32, model_name='fixture',
                                      kv_rank=0, object_group_id=0, cache_salt='fixture')
                for n in (1, 2, 3)]
        callback_kind = 'finish_read_prefetched' if direction == 'retrieve' else 'finish_write'
        queued = []
        token = protocol.OP.set(op)
        try:
            journal.submit(lambda stream, kind, payload: queued.append(payload),
                           None, callback_kind, keys[:2])
        finally:
            protocol.OP.reset(token)
        def finish(callback_keys):
            if case == 'missing':
                return
            successful, failed = callback_keys, []
            if case == 'reordered':
                successful = list(reversed(callback_keys))
            elif case == 'partial':
                successful = callback_keys[:1]
            elif case == 'wrong':
                successful = [keys[0], keys[2]]
            elif case == 'duplicate':
                successful = [keys[0], keys[0]]
            elif case == 'failed':
                successful, failed = callback_keys[:1], callback_keys[1:]
            values = {'successful_keys': successful, 'failed_keys': failed}
            journal.observe(journal.finalized, values)
            if case == 'repeated':
                journal.observe(journal.finalized, values)
        journal.handler(callback_kind, finish)(queued[0])
        complete = journal.seal(0)
        expected = case in ('success', 'reordered')
        assert complete is expected, (case, complete)
        rows = [json.loads(line)['event'] for line in path.read_text().splitlines()]
        assert any(row['kind'] == 'fence' and row['complete'] for row in rows) is expected
        outcomes.append({'case': case, 'complete': complete})
    return outcomes


def event_cases(hooks, protocol, event_source, root):
    code = hooks.registration.source_code(event_source, '<pinned-event-backend>',
                                          'DefaultEventIPCBackend.query_event')
    backend_type = type('DefaultEventIPCBackend', (), {
        'query_event': types.FunctionType(code, {'__name__': 'cpu_event_backend'}, 'query_event')})
    backend = backend_type()
    outcomes = []
    for case in ('success', 'false', 'not-returned', 'not-ready', 'missing'):
        path = root / f'event-{case}.jsonl'
        journal = protocol.Journal(path, role='cache')
        journal.event_guard = lambda actual, event: None
        journal.lifetime = types.SimpleNamespace(validate=lambda token: None)
        event = Event(ready=case != 'not-ready')
        op = {'id': 1, 'allocations': set(), 'recorded': False, 'device_complete': False,
              'returned': case != 'not-returned', 'success': case != 'false',
              'callback': True, 'delivered': True, 'transfer_key': 'source-generation-1',
              'direction': 'retrieve'}
        journal.operations[1] = op
        token = protocol.OP.set(op)
        try:
            if case != 'missing':
                journal.recorded_event({'event_backend': backend, 'event': event,
                                        'transfer_key': 'source-generation-1'})
        finally:
            protocol.OP.reset(token)
        journal.harvest()
        assert op['device_complete'] == (case == 'success')
        assert event.queries == (1 if case in ('success', 'not-ready') else 0)
        if case == 'success':
            other = dict(op, id=2, recorded=False, device_complete=False)
            token = protocol.OP.set(other)
            try:
                try:
                    journal.recorded_event({'event_backend': backend, 'event': event,
                                            'transfer_key': 'source-generation-1'})
                except ValueError:
                    pass
                else:
                    raise AssertionError('reused event accepted')
            finally:
                protocol.OP.reset(token)
        journal.seal(0)
        outcomes.append({'case': case, 'device_complete': op['device_complete']})
    return outcomes


def backend_cases(hooks, source_root):
    paths = ('lmcache/v1/platform/base/event_ipc.py', 'lmcache/v1/platform/cuda/__init__.py',
             'lmcache/v1/platform/isolated_ipc.py')
    sources = {p: (source_root / p).read_bytes() for p in paths}
    hooks.registration._validate_sources(sources, {p: hooks.registration.pins('cache')[p] for p in paths})
    def function(path, qualified, module):
        code = hooks.registration.source_code(sources[path], path, qualified)
        return types.FunctionType(code, module.__dict__, qualified.rsplit('.', 1)[-1])
    event_module = types.ModuleType('lmcache.v1.platform.base.event_ipc')
    backend_type = type('DefaultEventIPCBackend', (), {})
    event_module.DefaultEventIPCBackend = backend_type
    for name in ('__init__', 'query_event'):
        setattr(backend_type, name, function(paths[0], 'DefaultEventIPCBackend.' + name, event_module))
    isolated = types.ModuleType('lmcache.v1.platform.isolated_ipc')
    isolated._enabled = False
    isolated.is_isolated_ipc = function(paths[2], 'is_isolated_ipc', isolated)
    cuda = types.ModuleType('lmcache.v1.platform.cuda')
    cuda.is_isolated_ipc = isolated.is_isolated_ipc
    cuda._select_event_ipc_backend = function(paths[1], '_select_event_ipc_backend', cuda)
    spec_type = type('CudaDeviceSpec', (), {'_event_backend_cache': None})
    cuda.CudaDeviceSpec = spec_type
    for name in ('device_type', 'event_ipc_backend'):
        setattr(spec_type, name, property(function(paths[1], 'CudaDeviceSpec.' + name, cuda)))
    spec = spec_type()
    torch = types.ModuleType('torch')
    torch.cuda = types.SimpleNamespace(Event=Event)
    platform = types.ModuleType('lmcache.v1.platform')
    platform.current_device_spec = spec
    bindings = {module.__name__: module for module in (event_module, isolated, cuda, torch, platform)}
    with patch.dict(sys.modules, bindings):
        transfer = types.SimpleNamespace(torch_dev=torch.cuda)
        guard = hooks.passive_event_guard(transfer, sources)
        event = Event()
        guard(spec._event_backend_cache, event)
        assert event.queries == 0
        for case in ('isolated', 'cached-unknown'):
            spec._event_backend_cache = None
            isolated._enabled = case == 'isolated'
            if case == 'cached-unknown':
                spec._event_backend_cache = types.SimpleNamespace(
                    query_event=lambda event: (_ for _ in ()).throw(AssertionError('must not query')))
            try:
                hooks.passive_event_guard(transfer, sources)
            except ValueError:
                pass
            else:
                raise AssertionError('unsupported backend admitted')
            assert event.queries == 0
    return ['default-source-selector-no-query', 'isolated-rejected', 'cached-unknown-rejected']


def metadata_cases():
    snapshot = load('kv-registration.py')
    ns = types.SimpleNamespace
    info = ns(engine_group_id=0, layer_indices=(0,), tokens_per_block=16,
              sw_size_tokens=-1, recurrent_state=False, extra_object_group_tag=0)
    strategy = ns(dcp_size=1, vllm_world_size=2, vllm_worker_id=0, tp_size=2,
                  pp_size=1, n_servers=1, mla_only=True, kv_tp_size=2, num_kv_readers=2)
    worker = ns(parallel_strategy=strategy, world_size=1, worker_id=0, is_kv_writer=True,
                kv_caches={'layer': None}, engine_group_infos=[info], instance_id=17, model_name='synthetic')
    pc = ns(decode_context_parallel_size=1, data_parallel_size=1, world_size=2, rank=0,
            tensor_parallel_size=2, pipeline_parallel_size=1)
    config = ns(parallel_config=pc, model_config=ns(use_mla=True, is_hybrid=False))
    positive = snapshot.worker_registration(worker, config, {'kv_layout': 'NHD'})
    assert positive['cache_partition'] == {'world_size': 1, 'rank': 0, 'kv_tp_size': 2, 'is_writer': True}
    for owner, name, value in ((strategy, 'dcp_size', 2), (pc, 'decode_context_parallel_size', 2),
                               (strategy, 'num_kv_readers', 1), (info, 'recurrent_state', True),
                               (info, 'extra_object_group_tag', 1)):
        old = getattr(owner, name)
        setattr(owner, name, value)
        try:
            snapshot.worker_registration(worker, config, {'kv_layout': 'NHD'})
        except ValueError:
            pass
        else:
            raise AssertionError(f'unsupported topology accepted: {name}')
        finally:
            setattr(owner, name, old)
    return ['mla-reader-projection', 'worker-dcp-rejected', 'connector-dcp-rejected',
            'reader-drift-rejected', 'recurrent-rejected', 'aux-rejected']


def native_binding_case(hooks, protocol, root):
    path = root / 'native-binding.jsonl'
    journal = protocol.Journal(path, role='cache')
    observer = hooks.Observer(journal)
    observer.native_function = len
    observer.device_ops = types.SimpleNamespace(execute_object_group_transfer=len)
    observer.phase_timing = False
    observer.transfer_module = types.SimpleNamespace(device_ops=observer.device_ops,
                                                      _HAS_TRANSFER_PHASE_TIMING=False)
    helper = {'operation': 1, 'group': 0, 'native_batches': [(0, 1, [0])], 'allocations': {0: 1}}
    operation = observer.op_context.set({'transfer_key': 'generation-1'})
    token = observer.helper_context.set(helper)
    try:
        # Deliberately no removed local callable in this source-bound callback.
        observer.native_done({'batch_steps': [object()], 'transfer_key': 'generation-1'})
        observer.device_ops.execute_object_group_transfer = sum
        try:
            observer.native_done({'batch_steps': [object()], 'transfer_key': 'generation-1'})
        except ValueError:
            pass
        else:
            raise AssertionError('changed native callable accepted')
    finally:
        observer.helper_context.reset(token)
        observer.op_context.reset(operation)
        journal.seal(0)
    executions = [json.loads(row)['event'] for row in path.read_text().splitlines()
                  if json.loads(row)['event']['kind'] == 'execution']
    assert len(executions) == 1 and executions[0]['operation'] == 1
    return 'removed-local-and-active-callable-binding'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, type=Path)
    args = parser.parse_args()
    hooks, protocol = load('kv-transfer-v2.py'), load('kv-journal-v2.py')
    raw = (args.source_root / hooks.WORKER).read_bytes()
    if hashlib.sha256(raw).hexdigest() != '67bff9d5cfcd94cd7b540d27f5908f1c43cb7f215246a4a5cd76572de31406ff':
        raise ValueError('requires the exact ddc5fa34 candidate adapter source')
    event_source = (args.source_root / 'lmcache/v1/platform/base/event_ipc.py').read_bytes()
    if hashlib.sha256(event_source).hexdigest() != '7635c30a5be4534bc592554514787906b0d819ee3112458e3dd1ce1ed6e0c0a0':
        raise ValueError('requires the exact candidate event source')
    with tempfile.TemporaryDirectory(prefix='kv-candidate-cpu-') as directory:
        root = Path(directory)
        workers = [worker_case(hooks, protocol, raw, root, case, lazy, direction)
                   for direction in ('store', 'retrieve') for lazy in (False, True)
                   for case in ('success', 'false', 'incomplete', 'exception', 'drop', 'unhealthy-drain',
                                'stale', 'request-config', 'partial-prefix')
                   if direction == 'retrieve' or case != 'partial-prefix']
        events = event_cases(hooks, protocol, event_source, root)
        finalizations = finalization_cases(protocol, root)
        backends = backend_cases(hooks, args.source_root)
        topology = metadata_cases()
        native_binding = native_binding_case(hooks, protocol, root)
    print(json.dumps({'scope': 'cpu-source-path-not-native-admission', 'workers': workers,
                      'events': events, 'finalizations': finalizations,
                      'backends': backends, 'topology': topology,
                      'native_binding': native_binding}))


if __name__ == '__main__':
    main()
