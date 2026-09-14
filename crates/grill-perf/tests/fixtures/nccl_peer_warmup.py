"""CPU-only transport/tensor fixture exercising the actual producer's two ranks.

No Torch, CUDA or NCCL library is imported. Only the peer's first required
warmup is corrupted; all measured reductions remain correct on both ranks.
"""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import types

producer, example = map(Path, sys.argv[1:3])
namespace = runpy.run_path(str(producer))
plan = json.loads(example.read_text())
program_hash = hashlib.sha256(producer.read_bytes()).hexdigest()
plan['program_sha256'] = program_hash
plan['sources'][0]['sha256'] = program_hash
plan['revision'] = hashlib.sha256(b'torch:cpu-fixture;nccl:0.0.0;devices:cpu-fixture-no-device').hexdigest()

class Tensor:
    def __init__(self, value=0):
        self.value = value
    def remainder(self, divisor):
        return Tensor(self.value % divisor)
    def __mul__(self, other):
        return Tensor(self.value * other)
    def __add__(self, other):
        return Tensor(self.value + other)
    def __ne__(self, other):
        return Tensor(int(self.value != other.value))
    def copy_(self, other):
        self.value = other.value
    def sum(self):
        return self
    def item(self):
        return self.value

with tempfile.TemporaryDirectory() as temp:
    plan_path = Path(temp) / 'plan.json'
    plan_path.write_text(json.dumps(plan))
    for corrupt_peer in (False, True):
        gathered = {}
        for rank in (1, 0):
            state = {'iteration': 0, 'initialized': False}
            torch = types.ModuleType('torch')
            distributed = types.ModuleType('torch.distributed')
            torch.__version__ = 'cpu-fixture'
            torch.int64 = 'int64'
            torch.device = lambda *args: 'cpu-fixture-no-device'
            torch.arange = lambda *args, **kwargs: Tensor()
            torch.empty = lambda *args, **kwargs: Tensor()
            torch.equal = lambda left, right: left.value == right.value
            torch.cuda = types.SimpleNamespace(
                current_device=lambda: 0,
                synchronize=lambda: None,
                get_device_name=lambda *args: 'cpu-fixture-no-device',
                nccl=types.SimpleNamespace(version=lambda: (0, 0, 0)))
            def init(*args, **kwargs):
                state['initialized'] = True
            def reduce(buffer, **kwargs):
                buffer.value = 1 + int(corrupt_peer and rank == 1 and state['iteration'] == 0)
                state['iteration'] += 1
            def gather(payload, object_list):
                gathered[rank] = payload
                if rank == 0:
                    object_list[:] = [gathered[0], gathered[1]]
            distributed.init_process_group = init
            distributed.get_rank = lambda: rank
            distributed.get_world_size = lambda: 2
            distributed.barrier = lambda: None
            distributed.all_reduce = reduce
            distributed.ReduceOp = types.SimpleNamespace(SUM='sum')
            distributed.gather_object = gather
            distributed.is_initialized = lambda: state['initialized']
            distributed.destroy_process_group = lambda: state.update(initialized=False)
            torch.distributed = distributed
            sys.modules['torch'] = torch
            sys.modules['torch.distributed'] = distributed
            os.environ.update(RANK=str(rank), WORLD_SIZE='2', MASTER_ADDR='cpu-fixture', MASTER_PORT='1', LOCAL_RANK='0')
            sys.argv = [str(producer), '--plan', str(plan_path), '--stdout']
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                assert namespace['main']() == 0
            if rank == 0:
                body = json.loads(output.getvalue())
                assert body['correctness']['mismatches'] == 0
                warmup_failures = [failure for failure in body['failures']
                                   if failure['kind'] == 'correctness-failed' and 'warmup' in failure['detail']]
                assert bool(warmup_failures) == corrupt_peer, body['failures']
                if corrupt_peer:
                    assert any('rank 1' in failure['detail'] for failure in warmup_failures)
                print(json.dumps({'corrupt_peer_warmup': corrupt_peer, 'plan': plan, 'artifact': body}))
