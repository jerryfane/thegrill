"""CPU-only admission fixture for the NCCL producer's pre-launch allowance.

No Torch, CUDA or NCCL library is loaded: a meta-path finder refuses `torch`
and `torch.distributed` outright, so the producer can only reach its device
section by importing a module this fixture vetoes. An under-budget plan must
therefore be refused by the producer's own admission with its exit-2 abort
before that import is attempted, while the shipped example plan passes
admission and proceeds to the refused import (exit 3). Both cases run in one
environment; only the declared allowance differs.
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


class NoTorch:
    """Refuse every device runtime import; nothing here loads torch or a GPU."""

    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError(f"{name} is refused by the CPU admission fixture")
        return None


sys.meta_path.insert(0, NoTorch())

producer, example = map(Path, sys.argv[1:3])
namespace = runpy.run_path(str(producer))
plan = json.loads(example.read_text())
program_hash = hashlib.sha256(producer.read_bytes()).hexdigest()
plan["program_sha256"] = program_hash
plan["sources"][0]["sha256"] = program_hash

os.environ.update(
    RANK="0",
    WORLD_SIZE="2",
    MASTER_ADDR="cpu-fixture",
    MASTER_PORT="1",
    LOCAL_RANK="0",
)


def invoke(case: dict, plan_path: Path) -> tuple[object, str]:
    """Run the producer's main against one plan; return its exit and stderr."""
    plan_path.write_text(json.dumps(case))
    sys.argv = [str(producer), "--plan", str(plan_path), "--stdout"]
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = namespace["main"]()
        except SystemExit as exit:
            code = exit.code
    return code, err.getvalue()


with tempfile.TemporaryDirectory() as temp:
    plan_path = Path(temp) / "plan.json"
    mutators = {
        "under_budget_work": lambda case: case["allowance"].update(work_units=1),
        "under_budget_memory": lambda case: case["allowance"].update(memory_bytes=1),
    }
    results = {}
    for name, mutate in mutators.items():
        case = json.loads(json.dumps(plan))
        mutate(case)
        code, err = invoke(case, plan_path)
        assert code == 2, f"{name}: exit {code}, stderr={err}"
        assert "below the frozen collective" in err, f"{name}: stderr={err}"
        results[name] = code
    code, err = invoke(plan, plan_path)
    assert code == 3, f"the admitted plan never reached the device import: exit {code}, stderr={err}"
    results["adequate_allowance"] = code
    print(json.dumps(results))
