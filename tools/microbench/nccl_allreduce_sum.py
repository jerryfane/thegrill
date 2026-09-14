#!/usr/bin/env python3
"""Fixed all-reduce collective adapter for grill-perf microbenchmark evidence.

Operator-owned launch. The collector never discovers ranks, never places
processes and never wraps this adapter in a launcher: whichever `--program` the
collector is given is hashed and must equal `plan.program_sha256`, and this file
verifies the same hash on itself, so the program must be this adapter file
directly. A launcher such as `torchrun` has a different hash and is therefore
not a valid `--program`.

Participants are started explicitly, one process per rank, each executing this
pinned file directly with the process-group environment already set:

  RANK=<rank> WORLD_SIZE=<world> MASTER_ADDR=<host> MASTER_PORT=<port>
  [LOCAL_RANK=<local device index>] python3 tools/microbench/nccl_allreduce_sum.py \
      --plan plan.json [--stdout | --out <file>] [--checkout <bytes-source root>]

Rank 0 emits the merged artifact (stdout for the collector, or `--out` for the
operator-owned flow); the other ranks print nothing. MASTER_ADDR and MASTER_PORT
are required for a rendezvous, and the process-group timeout is the plan's
declared deadline, so a missing or dead participant fails inside a bounded wait
instead of hanging. Any participant that fails init or passes the deadline
exits non-zero with its traceback, the surviving ranks time out, and the
missing ranks stay visible as retained `rank-missing` failures in rank 0's
artifact. No launcher, cluster manager or remote control is invented here.

Frozen collective: `torch.distributed` NCCL `all_reduce` SUM over `int64`
tensors with `numel` elements per rank, world and rank list declared in the plan.
Per-rank input is `input[i] = (i % 251) + rank`; the exact expected result is
`world * (i % 251) + world * (world - 1) / 2`. Element dtypes are integers, so
the reference is exact equality, not a tolerance.

Sample unit: one complete world repetition is one sample. Each rank records its
own duration on its own monotonic clock; durations from the same repetition may
only be maximized, and rank absolute timestamps are never subtracted across
ranks. Synchronization is a device synchronize plus a completed collective
barrier immediately before and after the timed collective, so the retained
duration is a synchronized host-monotonic completion latency, explicitly not
device-only kernel time. Each sample carries the integer nanosecond reading
verbatim plus the exact rational nanosecond duration `{value, 1}`.

Byte definitions (NVIDIA/nccl-tests `doc/PERFORMANCE.md`):

  * algorithmic payload bytes = numel * sizeof(dtype)   (the collector's
    `algorithm_payload_bytes`)
  * `2 * (world - 1) / world` is the nccl-tests all-reduce *bus bandwidth*
    normalization, retained separately and never described as measured
    physical link traffic.

The adapter never translates either bandwidth into a serving-speed claim.

Runtime pins are observations of what this process actually saw (installed
torch/NCCL versions, participating device names, the participant count and
element shape it reduced, and how many declared sources it could verify from
bytes versus echo as declarations). None of them is authenticated execution
evidence and none is echoed from a declared implementation pin.

Identity is observed, never declared. `kernel_revision` is the runtime
descriptor this process actually produced (torch and NCCL versions plus the
participating device names) and `revision` is the SHA-256 of that descriptor's
UTF-8 bytes, computed here. The plan's `revision` is the operator's *expected*
hash: it is validated structurally only, and a mismatch is left in the artifact
for the collector to reject rather than being suppressed. A different loaded
torch/NCCL build or device set is a legitimate observed axis; the measurement
program itself must not change.

Warmups are required repetitions whose timings never enter the measured
samples, and whose correctness failures remain failures: a warmup mismatch is
retained as a failure naming the warmup population, while only measured
repetitions contribute to `correctness.mismatches`.

Usage, collector-launched participant (the collector's own environment supplies
RANK/WORLD_SIZE/MASTER_*; peers are started by the operator):

    RANK=0 WORLD_SIZE=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
      grill-perf microbench capture \
        --adapter nccl-allreduce-sum \
        --plan plan.json \
        --program tools/microbench/nccl_allreduce_sum.py \
        --authorize-device-window \
        -- --stdout --checkout /path/to/nccl-tests

Operator-owned placement, where the collector only imports the result:

    RANK=0 WORLD_SIZE=2 MASTER_ADDR=host-a MASTER_PORT=29500 python3 tools/microbench/nccl_allreduce_sum.py \
      --plan plan.json --out rank0-artifact.json &
    RANK=1 WORLD_SIZE=2 MASTER_ADDR=host-a MASTER_PORT=29500 python3 tools/microbench/nccl_allreduce_sum.py \
      --plan plan.json &
    wait
    grill-perf microbench import --artifact rank0-artifact.json --plan plan.json --out evidence/
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
import traceback

ADAPTER = "nccl-allreduce-sum"
KIND = "microbench-artifact-v1"
VERSION = 1
REFERENCE_ID = "world*(i%251)+world*(world-1)/2"
INPUT_ID = "input[i]=(i%251)+rank"
CLOCK = {
    "id": "rank-monotonic-ns",
    "kind": "host_monotonic",
    "units": "nanoseconds",
    "resolution_ns": 1,
    "synchronization": "collective-barrier-before-after",
}
FROZEN_OPERATION = {
    "kind": "all-reduce",
    "op": "sum",
    "dtype": "int64",
    "numel": 262144,
    "world": 2,
    "ranks": [0, 1],
    "input": INPUT_ID,
    "reference": REFERENCE_ID,
}

ARTIFACT_KEYS = {
    "kind",
    "version",
    "adapter",
    "revision",
    "operation",
    "provenance",
    "submitted_provenance",
    "program_sha256",
    "kernel_revision",
    "acquisition",
    "sources",
    "observations",
    "topology",
    "clock",
    "execution",
    "samples",
    "correctness",
    "failures",
}
FAILURE_KINDS = {
    "deadline-exceeded",
    "allowance-exceeded",
    "tier-fallback",
    "rank-missing",
    "incomplete-samples",
    "correctness-failed",
    "non-finite",
    "adapter-error",
}
OBSERVATION_NAMES = {
    "torch-version",
    "nccl-version",
    "exl3-module-sha256",
    "device",
    "experts",
    "topk",
    "tokens",
    "hidden",
    "intermediate",
    "cap",
    "skew-milli",
    "routing-seed",
    "layer-seed",
    "parity-routing-seed",
    "activation-seed",
    "fallback-tier",
    "world",
    "numel",
    "dtype",
    "op",
    "sources-verified",
    "sources-declared",
}
REQUIRED_OBSERVATIONS = {
    "torch-version",
    "nccl-version",
    "device",
    "world",
    "numel",
    "dtype",
    "op",
    "sources-verified",
    "sources-declared",
}
PINNED_OBSERVATIONS = {"numel": "262144", "dtype": "int64", "op": "sum"}


def abort(message: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"{ADAPTER}: {message}", file=sys.stderr, flush=True)
    sys.exit(2)


def canonical_revision(kernel_revision: str) -> str:
    """SHA-256 of the UTF-8 bytes of the observed descriptor, no newline."""
    return hashlib.sha256(kernel_revision.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_duration(nanoseconds: int) -> dict:
    """Integer nanosecond readings are already exact: `value / 1`."""
    return {"numerator": int(nanoseconds), "denominator": 1}


def artifact_body(
    plan: dict,
    sources: list[dict],
    observations: list[dict],
    kernel_revision: str,
    world: int,
    samples: list[dict],
    execution: dict,
    correctness: dict,
    failures: list[dict],
) -> dict:
    """Assemble the wire document. Pure: no device or environment access."""
    return {
        "kind": KIND,
        "version": VERSION,
        "adapter": ADAPTER,
        # The observed descriptor's digest, never the plan's expected hash.
        "revision": canonical_revision(kernel_revision),
        "operation": plan["operation"],
        "provenance": "native_observed",
        "submitted_provenance": None,
        "program_sha256": plan["program_sha256"],
        "kernel_revision": kernel_revision,
        # The complete prospective acquisition object, copied verbatim: it
        # binds identical duration bodies to distinct acquisitions without
        # synthesizing any timestamp.
        "acquisition": plan["acquisition"],
        "sources": sources,
        "observations": observations,
        "topology": {
            "scope": "ranks",
            "world": world,
            "ranks": [{"index": rank, "device": rank} for rank in range(world)],
        },
        "clock": CLOCK,
        "execution": execution,
        "samples": samples,
        "correctness": correctness,
        "failures": failures,
    }


def validate_body(body: dict, plan: dict) -> None:
    """Fail closed if the document would not match the collector's contract."""
    if set(body) != ARTIFACT_KEYS:
        abort(
            "artifact keys differ from the collector contract: "
            f"missing={sorted(ARTIFACT_KEYS - set(body))} extra={sorted(set(body) - ARTIFACT_KEYS)}"
        )
    if body["kind"] != KIND or body["version"] != VERSION or body["adapter"] != ADAPTER:
        abort("artifact kind/version/adapter mismatch")
    # The artifact revision is the digest of the observed descriptor. The
    # plan's expected hash is only structurally validated: a mismatch is
    # retained for the collector to reject, never suppressed here.
    if not isinstance(body["revision"], str) or len(body["revision"]) != 64:
        abort("artifact revision must be the SHA-256 hex of the observed descriptor")
    if body["revision"] != canonical_revision(body["kernel_revision"]):
        abort("artifact revision is not the SHA-256 of artifact.kernel_revision")
    if body["acquisition"] != plan.get("acquisition"):
        abort("artifact acquisition must be the plan's prospective acquisition object")
    if body["operation"] != plan.get("operation"):
        abort("artifact operation does not match the plan")
    if body["clock"] != CLOCK:
        abort("artifact clock is not the declared contract")
    if body["program_sha256"] != plan.get("program_sha256"):
        abort("artifact program_sha256 does not match the plan")
    if not isinstance(body["kernel_revision"], str) or not body["kernel_revision"]:
        abort("kernel_revision must be a nonempty observed runtime identity")
    for source in body["sources"]:
        if set(source) != {"path", "sha256"} or len(source["sha256"]) != 64:
            abort("artifact source entries must be {path, sha256}")
    names = [observation["name"] for observation in body["observations"]]
    if set(names) != REQUIRED_OBSERVATIONS or len(names) != len(set(names)):
        abort(f"observation set mismatch: {sorted(set(names) ^ REQUIRED_OBSERVATIONS)}")
    for observation in body["observations"]:
        if observation["name"] not in OBSERVATION_NAMES:
            abort(f"observation name {observation['name']!r} is not in the closed set")
        expected = PINNED_OBSERVATIONS.get(observation["name"])
        if expected is not None and observation["value"] != expected:
            abort(f"observation {observation['name']} is {observation['value']!r}, expected {expected!r}")
    topology = body["topology"]
    world = int(body["operation"]["world"])
    if topology.get("scope") != "ranks" or topology.get("world") != world:
        abort("rank topology must declare the operation's world")
    if len(topology.get("ranks", [])) != world:
        abort("rank topology must list every participating rank")
    execution = body["execution"]
    if execution["warmups"] != plan["warmups"] or execution["iterations"] != plan["iterations"]:
        abort("execution warmups/iterations do not match the plan")
    seen = set()
    for position, sample in enumerate(body["samples"]):
        if set(sample) != {"index", "rank", "repetition", "raw", "duration"}:
            abort("sample keys differ from the collector contract")
        if sample["index"] != position:
            abort("sample index must be the retained position")
        key = (sample["repetition"], sample["rank"])
        if key in seen:
            abort(f"duplicate rank sample {key}")
        seen.add(key)
        if sample["rank"] is None or not 0 <= sample["rank"] < world:
            abort("rank sample outside the declared world")
        if sample["repetition"] is None or not 0 <= sample["repetition"] < plan["iterations"]:
            abort("repetition outside the declared iteration count")
        if not isinstance(sample["raw"], str) or not sample["raw"].isdigit():
            abort("nanosecond raw must be a plain nonnegative integer")
        duration = sample["duration"]
        if set(duration) != {"numerator", "denominator"}:
            abort("sample duration must be a rational object")
        if duration["denominator"] != 1 or duration["numerator"] != int(sample["raw"]):
            abort("nanosecond samples must retain the exact integer duration")
    correctness = body["correctness"]
    if correctness.get("kind") != "exact-reduction" or correctness.get("reference") != REFERENCE_ID:
        abort("correctness record is not the exact reduction reference")
    if correctness.get("tolerance_elems") != 0:
        abort("the exact integer reference has no tolerance")
    for failure in body["failures"]:
        if set(failure) != {"kind", "detail"} or failure["kind"] not in FAILURE_KINDS:
            abort(f"retained failure {failure.get('kind')!r} is not in the closed set")


def check_plan(plan: dict) -> None:
    if plan.get("kind") != "microbench-plan-v1" or plan.get("version") != VERSION:
        abort("plan kind/version is not microbench-plan-v1 version 1")
    if plan.get("adapter") != ADAPTER:
        abort(f"plan adapter {plan.get('adapter')!r} is not {ADAPTER!r}")
    operation = plan.get("operation") or {}
    if operation.get("kind") != "all-reduce":
        abort("plan operation is not the frozen all-reduce collective")
    for key in ("op", "dtype", "numel"):
        if operation.get(key) != FROZEN_OPERATION[key]:
            abort(f"plan operation {key} is {operation.get(key)!r}, expected {FROZEN_OPERATION[key]!r}")
    if operation.get("input") != INPUT_ID or operation.get("reference") != REFERENCE_ID:
        abort("plan must use the pinned input generation and exact reduction reference")
    world = operation.get("world")
    ranks = operation.get("ranks")
    if not isinstance(world, int) or world < 2 or ranks != list(range(world)):
        abort("plan must list a fixed world >= 2 and exactly ranks 0..world-1")
    if plan.get("clock") != CLOCK:
        abort("plan clock is not the pinned rank-monotonic-ns contract")
    revision = plan.get("revision")
    if not isinstance(revision, str) or len(revision) != 64 or not all(
        character in "0123456789abcdef" for character in revision
    ):
        abort("plan revision must be the expected SHA-256 hex of the observed descriptor")
    if not plan.get("program_sha256"):
        abort("plan must pin program_sha256 for an external adapter")
    if not plan.get("sources"):
        abort("plan must declare its pinned sources")
    allowance = plan.get("allowance") or {}
    for key in ("work_units", "memory_bytes", "deadline_ms"):
        if not isinstance(allowance.get(key), int) or allowance[key] <= 0:
            abort(f"plan allowance {key} must be a positive integer")
    correctness = plan.get("correctness") or {}
    if correctness.get("reference") != REFERENCE_ID:
        abort("plan correctness reference is not the exact all-reduce sum")
    if (correctness.get("tolerance") or {}).get("kind") != "exact-int64":
        abort("plan correctness tolerance is not exact int64 equality")


def verify_sources(plan: dict, checkout: str | None) -> tuple[list[dict], int, int]:
    """Verify every pin this host can resolve; echo the rest as declarations.

    `program_sha256` is the one pin the collector verifies directly, because the
    collector hashes the program it launched. A declared source path that does
    not exist on the producing host stays a declaration, is counted as one, and
    is retained verbatim rather than silently dropped or rewritten.
    """
    self_path = os.path.abspath(__file__)
    if sha256_file(self_path) != plan.get("program_sha256"):
        abort("this adapter does not match plan.program_sha256; re-pin instead of running it")
    observed = []
    verified = 0
    declared_count = 0
    for source in plan["sources"]:
        declared = source.get("path")
        declared_sha = source.get("sha256")
        if not isinstance(declared, str) or not isinstance(declared_sha, str):
            abort("plan sources must be {path, sha256} objects")
        resolved = None
        if os.path.basename(declared) == os.path.basename(self_path):
            resolved = self_path
        elif os.path.isabs(declared) and os.path.isfile(declared):
            resolved = declared
        elif checkout and os.path.isfile(os.path.join(checkout, declared)):
            resolved = os.path.join(checkout, declared)
        if resolved is not None:
            observed_sha = sha256_file(resolved)
            if observed_sha != declared_sha:
                abort(f"pinned source hash mismatch for {declared}")
            observed.append({"path": os.path.realpath(resolved), "sha256": observed_sha})
            verified += 1
        else:
            observed.append({"path": declared, "sha256": declared_sha})
            declared_count += 1
    if verified < 1:
        abort("no declared source could be verified from bytes on this host")
    return observed, verified, declared_count


def main() -> int:
    parser = argparse.ArgumentParser(description="pinned NCCL all-reduce sum microbenchmark adapter")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", default=None, help="rank 0 writes the merged artifact here")
    parser.add_argument("--stdout", action="store_true", help="rank 0 prints the merged artifact")
    parser.add_argument(
        "--checkout",
        default=None,
        help="optional root used to resolve declared source pins on this host",
    )
    args = parser.parse_args()

    with open(args.plan, "rb") as handle:
        plan = json.loads(handle.read())
    check_plan(plan)
    sources, verified, declared_count = verify_sources(plan, args.checkout)

    # Explicit process-group environment. No launcher is required or wrapped:
    # whoever starts this pinned file sets these variables, for the participant
    # the collector launches and for every peer participant alike.
    required = ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        abort(
            "the process-group environment must set "
            + ", ".join(missing)
            + " explicitly (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT; LOCAL_RANK selects the "
            "device when present); this adapter is executed directly, never through a launcher"
        )
    operation = plan["operation"]
    world = int(operation["world"])
    iterations = int(plan["iterations"])
    warmups = int(plan["warmups"])
    deadline_ms = plan["allowance"]["deadline_ms"]
    if int(os.environ["WORLD_SIZE"]) != world:
        abort(
            f"launched world {os.environ['WORLD_SIZE']} does not match the declared world {world}; "
            "re-pin the plan instead of running a different topology"
        )
    rank_index = int(os.environ["RANK"]) if os.environ["RANK"].isdigit() else -1
    if not 0 <= rank_index < world:
        abort(f"RANK {os.environ['RANK']!r} is outside the declared world {world}")
    if not os.environ["MASTER_PORT"].isdigit():
        abort(f"MASTER_PORT {os.environ['MASTER_PORT']!r} is not a port number")
    started = time.monotonic()
    rank = None
    failures = []
    timed_out = False
    total_mismatches = 0
    devices = []

    try:
        import torch
        import torch.distributed as distributed

        local_index = (
            int(os.environ["LOCAL_RANK"]) if os.environ.get("LOCAL_RANK") else torch.cuda.current_device()
        )
        local = torch.device("cuda", local_index)
        distributed.init_process_group(
            "nccl",
            timeout=datetime.timedelta(milliseconds=deadline_ms),
            device_id=local,
        )
        rank = distributed.get_rank()
        observed_world = distributed.get_world_size()
        if observed_world != world:
            abort(f"process group world {observed_world} does not match the declared {world}")

        numel = int(operation["numel"])
        index = torch.arange(numel, dtype=torch.int64, device=local)
        partial = index.remainder(251)
        expected = partial * world + (world * (world - 1)) // 2
        input_template = partial + rank

        def reset(buffer: "torch.Tensor") -> None:  # type: ignore[name-defined]
            # Reset outside every timed region so each repetition reduces the
            # declared input rather than the previous round's result.
            buffer.copy_(input_template)

        buffer = torch.empty(numel, dtype=torch.int64, device=local)
        mismatches = 0

        def run_once() -> tuple[int, int]:
            """Run one complete repetition; return (elapsed_ns, mismatches).

            Nothing is recorded here: the caller decides whether the repetition
            is a warmup (required, timed but excluded from the samples) or a
            measured repetition (sample plus measured correctness population).
            """
            reset(buffer)
            torch.cuda.synchronize()
            distributed.barrier()
            origin = time.perf_counter_ns()
            distributed.all_reduce(buffer, op=distributed.ReduceOp.SUM)
            torch.cuda.synchronize()
            elapsed = time.perf_counter_ns() - origin
            distributed.barrier()
            local_mismatch = 0
            if not bool(torch.equal(buffer, expected)):
                local_mismatch = int((buffer != expected).sum().item())
            return elapsed, local_mismatch

        warmup_mismatches = 0
        for _ in range(warmups):
            _, local_mismatch = run_once()
            warmup_mismatches += local_mismatch
        if warmup_mismatches:
            # Required warmups are part of the cell: a warmup correctness
            # failure stays a retained failure and never becomes a measured
            # observation, but it is not discarded either.
            failures.append(
                {
                    "kind": "correctness-failed",
                    "detail": (
                        f"rank {rank} warmup repetitions differed in {warmup_mismatches} elements "
                        "(warmup population, excluded from the measured samples)"
                    ),
                }
            )
        times = []
        for _ in range(iterations):
            if time.monotonic() - started > deadline_ms / 1000.0:
                timed_out = True
                failures.append(
                    {
                        "kind": "deadline-exceeded",
                        "detail": f"rank {rank} exceeded the declared {deadline_ms} ms deadline",
                    }
                )
                break
            elapsed, local_mismatch = run_once()
            mismatches += local_mismatch
            times.append(elapsed)
        if mismatches:
            failures.append(
                {
                    "kind": "correctness-failed",
                    "detail": f"rank {rank} measured repetitions differed in {mismatches} elements",
                }
            )

        gathered = [None for _ in range(world)]
        distributed.gather_object(
            {
                "rank": rank,
                "times": times,
                "mismatches": mismatches,
                "device": torch.cuda.get_device_name(local),
            },
            object_list=gathered if rank == 0 else None,
        )
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001  (retain the traceback; missing ranks stay failures)
        traceback.print_exc()
        return 3
    finally:
        try:
            import torch.distributed as distributed_cleanup

            if distributed_cleanup.is_initialized():
                distributed_cleanup.destroy_process_group()
        except BaseException:  # noqa: BLE001
            pass

    if rank != 0:
        return 0

    # The observed runtime descriptor: the versions actually loaded in this
    # process plus the participating device names reported by the gather. Its
    # SHA-256 is the artifact revision, so a different loaded build or device
    # set is a different observed identity while the program stays pinned.
    devices = sorted({str(entry["device"]) for entry in gathered if entry is not None})
    kernel_revision = (
        f"torch:{torch.__version__};"
        f"nccl:{'.'.join(str(part) for part in torch.cuda.nccl.version())};"
        f"devices:{','.join(devices) if devices else 'unavailable'}"
    )

    # Canonical grid: repetition-major, one entry per rank per repetition.
    indexed = {int(entry["rank"]): entry for entry in gathered if entry is not None}
    samples = []
    for repetition in range(iterations):
        for index_rank in range(world):
            entry = indexed.get(index_rank)
            times = entry["times"] if entry else []
            if repetition < len(times):
                nanoseconds = int(times[repetition])
                samples.append(
                    {
                        "index": len(samples),
                        "rank": index_rank,
                        "repetition": repetition,
                        "raw": str(nanoseconds),
                        "duration": sample_duration(nanoseconds),
                    }
                )
    missing = [
        missing_rank
        for missing_rank in range(world)
        if indexed.get(missing_rank) is None or len(indexed[missing_rank]["times"]) != iterations
    ]
    for missing_rank in missing:
        failures.append(
            {
                "kind": "rank-missing",
                "detail": f"rank {missing_rank} did not contribute {iterations} complete repetitions",
            }
        )
    total_mismatches = sum(int(entry["mismatches"]) for entry in indexed.values())
    complete = not missing and not timed_out and len(samples) == world * iterations
    if len(samples) != world * iterations:
        failures.append(
            {
                "kind": "incomplete-samples",
                "detail": f"retained {len(samples)} of {world * iterations} per-rank samples",
            }
        )
    work_units = numel * (warmups + iterations)
    memory_bytes = numel * 8 * 3
    if work_units > plan["allowance"]["work_units"]:
        failures.append(
            {
                "kind": "allowance-exceeded",
                "detail": f"work {work_units} exceeds declared {plan['allowance']['work_units']}",
            }
        )
    if memory_bytes > plan["allowance"]["memory_bytes"]:
        failures.append(
            {
                "kind": "allowance-exceeded",
                "detail": f"memory {memory_bytes} exceeds declared {plan['allowance']['memory_bytes']}",
            }
        )
    observations = [
        {"name": "torch-version", "value": str(torch.__version__)},
        {"name": "nccl-version", "value": ".".join(str(part) for part in torch.cuda.nccl.version())},
        {"name": "device", "value": ",".join(devices) if devices else "unavailable"},
        {"name": "world", "value": str(world)},
        {"name": "numel", "value": str(numel)},
        {"name": "dtype", "value": str(operation["dtype"])},
        {"name": "op", "value": str(operation["op"])},
        {"name": "sources-verified", "value": str(verified)},
        {"name": "sources-declared", "value": str(declared_count)},
    ]
    body = artifact_body(
        plan,
        sources,
        observations,
        kernel_revision,
        world,
        samples,
        {
            "warmups": warmups,
            "iterations": iterations,
            "deadline_ms": deadline_ms,
            "work_units": work_units,
            "memory_bytes": memory_bytes,
            "completed": complete,
            "timed_out": timed_out,
            "observed_fallback": None,
        },
        {
            "kind": "exact-reduction",
            "reference": REFERENCE_ID,
            "mismatches": int(total_mismatches),
            "tolerance_elems": 0,
            "finite": True,
            "passed": total_mismatches == 0,
        },
        failures,
    )
    validate_body(body, plan)
    encoded = json.dumps(body, indent=2, sort_keys=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
    if args.stdout or not args.out:
        sys.stdout.write(encoded + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
