#!/usr/bin/env python3
"""Fixed-cell EXL3 E3 grouped expert adapter for grill-perf microbenchmark evidence.

This is the *only* device kernel adapter in the #54 slice, and it is bound to a
single prospectively fixed cell. It deliberately does not reproduce
`main()` from the pinned public anchor: no `--skews` ladder, no `--e3-caps`
ladder, no best-median summary, no `torch.cuda.empty_cache()`, and no serving
process is touched. The operation is drawn from the pinned anchor's own
`make_layer`, `routing`, `time_fn` and `set_cap` functions; its parity check
uses `_err_stats` and `_assert_e3_within` from the pinned tolerance file.

Pinned public sources (verified by hash from bytes read at run time; a mismatch
aborts before any device work):

  MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks
  revision f906ee990596486e10ddbe381efa6f0e496f77e3
    tests/bench_e3_microbench.py  f57e4fa6726b0110c44ce56b0b0968ba425e566ace6d89f2db3872aae13889ec
    tests/test_exl3_overlay.py    8777254d47cdbb89186ba3bd1ac2437cbebc7987e0791d36c68e7bccb3df2b35

Every retained sample preserves the timer's own representation
(`repr(float(value))`, including scientific notation) and the exact rational
nanosecond duration converted from that decimal. Nothing is rounded to a whole
nanosecond and no digit is rejected for being finer than the declared clock
resolution: the declared resolution describes the advertised timer class, not a
quantization of the reported value. Non-finite or negative samples cannot be
represented as a duration, so they are retained as a failure and the grid stays
incomplete rather than being silently repaired. The same rule applies to the
parity statistics: they are emitted as `repr(float(value))` of the values the
pinned parity helpers returned, never re-rounded to a fixed number of decimals.
The reference maximum `ref_max` is the E2 value, because the pinned
`_assert_e3_within` derives both the per-key floor and the coarse bound from
`e2["ref_max"]`.

Identity is observed, never declared. `kernel_revision` is the runtime
descriptor this process actually produced (installed torch and NCCL versions
plus the SHA-256 of the loaded exl3 module file), and `revision` is the SHA-256
of that descriptor's UTF-8 bytes, computed here. The plan's `revision` is the
operator's *expected* hash: it is validated structurally only, and a mismatch is
left in the artifact for the collector to reject rather than being suppressed.
The fallback observation carries the actual raw overlay token (`grouped`,
`kernel` or `unavailable`), while `execution.observed_fallback` is the
normalized wire value (`e3-grouped`, `e2-kernel` or absent). A lower tier or an
unavailable tier is a retained failure, not a reason to withhold the document.

Every runtime pin is reported as an observation of what the process actually
saw (installed torch/NCCL versions, the loaded exl3 module file hash, the device
name, the parameters actually used, how many pinned sources were verified from
bytes rather than echoed). No declared implementation pin is echoed back as
observed kernel provenance, and none of these observations is authenticated
execution evidence.

Allowance admission. Before the torch import and before any device work, the
declared allowance must already cover the frozen cell's known minimum: the
projection work `tokens * topk * 3 * 2 * hidden * intermediate` for the required
warmups, the measured iterations and the three parity passes, plus the fp16
activation payload `tokens * hidden * 2` the timed kernel cannot run without.
The retained `execution.work_units` is that same derivation, so the accounting
cannot be understated relative to what was admitted. The retained
`execution.memory_bytes` is the process-local PyTorch allocator peak
(`memory_allocated` / `max_memory_allocated`): it is never total device, driver
or NVIDIA memory, it says nothing about non-PyTorch workspaces such as the CUDA
context or allocator caches, and it is not a physical OOM bound.

Usage (inside a separately authorized device window only):

    grill-perf microbench capture \
      --adapter exl3-e3-grouped \
      --plan plan.json \
      --program tools/microbench/exl3_e3_grouped.py \
      --authorize-device-window \
      -- --checkout /path/to/GLM-5.3-Flash-EXL3-2x-DGX-Sparks
"""

from __future__ import annotations

import argparse
import decimal
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback

ADAPTER = "exl3-e3-grouped"
KIND = "microbench-artifact-v1"
VERSION = 1
CLOCK_UNIT_NS = 1_000_000  # milliseconds -> nanoseconds

ANCHOR = "tests/bench_e3_microbench.py"
PARITY = "tests/test_exl3_overlay.py"
ANCHOR_SHA256 = "f57e4fa6726b0110c44ce56b0b0968ba425e566ace6d89f2db3872aae13889ec"
PARITY_SHA256 = "8777254d47cdbb89186ba3bd1ac2437cbebc7987e0791d36c68e7bccb3df2b35"

ROUTING_ID = "synthetic-zipf-permuted"
ACTIVATION_ID = "randn-fp16-activation-seed-3"
WEIGHTS_ID = "random-k4-trellis-shared-gate-up-suh"
REFERENCE_ID = "apply-exl3-python-loop"

TOLERANCE_MILLI = {
    "factor_milli": 1500,
    "abs_rel_micro": 1000,
    "nrmse_abs_micro": 100,
    "coarse_abs_milli": 150,
    "coarse_factor_milli": 80,
}

CLOCK = {
    "id": "cuda-event-ms",
    "kind": "device_event",
    "units": "milliseconds",
    "resolution_ns": 1000,
    "synchronization": "device-event-timing-after-synchronize",
}

FROZEN_OPERATION = {
    "kind": "exl3-experts",
    "hidden": 4096,
    "intermediate": 1024,
    "experts": 288,
    "topk": 8,
    "tokens": 1024,
    "layer_seed": 0,
    "routing_seed": 1,
    "parity_routing_seed": 3,
    "activation_seed": 3,
    "skew_milli": 1000,
    "tier": "e3-grouped",
    "cap": 32,
    "dtype": "float16",
    "routing": ROUTING_ID,
    "activation": ACTIVATION_ID,
    "weights": WEIGHTS_ID,
}

WARMUPS = 2
ITERATIONS = 5

# Passes of the pinned `apply_exl3_experts` entry point the cell always runs
# before timing starts: once as the E2 kernel reference and twice as the E3
# candidate and its repeat. The pinned Python reference loop is a different
# implementation and is not counted as grouped-expert work.
PARITY_PASSES = 3
# Projections per grouped expert (gate, up, down) and the two elementary
# operations one multiply-accumulate is counted as, matching this cell's
# existing work definition.
EXPERT_PROJECTIONS = 3
FACTOR_PER_MULTIPLY_ACCUMULATE = 2
FP16_BYTES = 2

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
RAW_FALLBACK_TOKENS = {"grouped", "kernel", "unavailable"}
NORMALIZED_FALLBACK = {"grouped": "e3-grouped", "kernel": "e2-kernel"}
U64_MAX = (1 << 64) - 1
COLLECTOR_DECIMAL_DIGITS = 32
COLLECTOR_DECIMAL_EXPONENT = 30
REQUIRED_OBSERVATIONS = {
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
    "sources-verified",
    "sources-declared",
}
# Values the collector pins against the frozen cell. `fallback-tier` is not
# pinned here: it is an observed raw token whose admissible values are checked
# separately, because a lower or unavailable tier must still produce a
# parseable, retained-failure document.
PINNED_OBSERVATIONS = {
    "experts": "288",
    "topk": "8",
    "tokens": "1024",
    "hidden": "4096",
    "intermediate": "1024",
    "cap": "32",
    "skew-milli": "1000",
    "routing-seed": "1",
    "layer-seed": "0",
    "parity-routing-seed": "3",
    "activation-seed": "3",
}


def abort(message: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"{ADAPTER}: {message}", file=sys.stderr, flush=True)
    sys.exit(2)


def canonical_revision(kernel_revision: str) -> str:
    """SHA-256 of the UTF-8 bytes of the observed descriptor, no newline."""
    return hashlib.sha256(kernel_revision.encode("utf-8")).hexdigest()


def normalize_fallback(raw: str) -> str | None:
    """Normalize the ACTUAL overlay token for the wire; unknown stays absent."""
    return NORMALIZED_FALLBACK.get(raw)


def collector_decimal(text: str) -> bool:
    """Mirror the collector's bounded decimal grammar for retained strings.

    Mirrors `microbench::exact_decimal`: no sign, no empty integer part, at most
    32 mantissa digits, an optional `e`/`E` exponent of magnitude at most 30, and
    at most 64 characters.
    """
    if not text or len(text) > 64:
        return False
    mantissa = text
    exponent = ""
    for marker in ("e", "E"):
        if marker in mantissa:
            mantissa, _, exponent = mantissa.partition(marker)
            break
    if exponent:
        digits = exponent[1:] if exponent[:1] in "+-" else exponent
        if not digits.isdigit() or int(digits) > COLLECTOR_DECIMAL_EXPONENT:
            return False
    whole, _, fraction = mantissa.partition(".")
    if not whole.isdigit():
        return False
    if fraction and not fraction.isdigit():
        return False
    return len(whole) + len(fraction) <= COLLECTOR_DECIMAL_DIGITS


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def minimum_allowance(operation: dict) -> tuple[int, int]:
    """Sound minimum `(work_units, memory_bytes)` for the frozen E3 cell.

    Mirrors `microbench::minimum_allowance` in
    crates/grill-perf/src/microbench.rs; both sides derive the same numbers, or
    one could admit a plan the other refuses.

    `work_units` counts one multiply-accumulate as two, as this cell's existing
    definition does, over the projection shape
    `tokens * topk * 3 * 2 * hidden * intermediate`, for every pass the cell
    executes: the required warmups, the measured iterations and the three
    parity passes. `memory_bytes` is the fp16 activation payload
    `tokens * hidden * 2` the timed kernel cannot run without — a minimum, not
    the retained observation, which stays the larger process-local PyTorch
    allocator peak and is never total device, driver or NVIDIA memory.
    """
    per_pass = (
        int(operation["tokens"])
        * int(operation["topk"])
        * EXPERT_PROJECTIONS
        * FACTOR_PER_MULTIPLY_ACCUMULATE
        * int(operation["hidden"])
        * int(operation["intermediate"])
    )
    work_units = per_pass * (WARMUPS + ITERATIONS + PARITY_PASSES)
    memory_bytes = int(operation["tokens"]) * int(operation["hidden"]) * FP16_BYTES
    return work_units, memory_bytes


def sample_duration(value: object) -> tuple[str, dict] | str:
    """Return `(raw, duration)` or a failure token.

    `raw` is the timer's own representation, never a re-formatted value; the
    duration is the exact rational nanosecond conversion of that decimal.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "adapter-error"
    milliseconds = float(value)
    if not math.isfinite(milliseconds):
        return "non-finite"
    if milliseconds < 0:
        return "negative"
    text = repr(milliseconds)
    nanoseconds = decimal.Decimal(text) * CLOCK_UNIT_NS
    numerator, denominator = nanoseconds.as_integer_ratio()
    if numerator < 0 or numerator > U64_MAX or denominator < 1 or denominator > U64_MAX:
        # The collector retains exact rationals as two u64 members; a value
        # outside that range is retained as a failure, never truncated.
        return "unrepresentable"
    return text, {"numerator": numerator, "denominator": denominator}


def artifact_body(
    plan: dict,
    sources: list[dict],
    observations: list[dict],
    kernel_revision: str,
    device: int,
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
        "operation": FROZEN_OPERATION,
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
            "scope": "device",
            "world": None,
            "ranks": [{"index": 0, "device": device}],
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
    if body["operation"] != FROZEN_OPERATION:
        abort("artifact operation is not the frozen cell")
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
    raw_fallback = dict((o["name"], o["value"]) for o in body["observations"])["fallback-tier"]
    if not isinstance(raw_fallback, str) or not raw_fallback or len(raw_fallback) > 256:
        abort("fallback-tier must preserve the actual raw overlay token")
    normalized = normalize_fallback(raw_fallback)
    execution_fallback = body["execution"]["observed_fallback"]
    if normalized != execution_fallback:
        abort(
            f"execution.observed_fallback {execution_fallback!r} is not the normalization "
            f"of the raw token {raw_fallback!r}"
        )
    if normalized is None and not any(
        failure["kind"] == "tier-fallback" for failure in body["failures"]
    ):
        abort("an unavailable or unknown fallback token requires a retained tier-fallback failure")
    topology = body["topology"]
    if topology.get("scope") != "device" or len(topology.get("ranks", [])) != 1:
        abort("device topology must declare exactly one rank")
    execution = body["execution"]
    if execution["warmups"] != WARMUPS or execution["iterations"] != ITERATIONS:
        abort("execution warmups/iterations are not the frozen cell")
    for position, sample in enumerate(body["samples"]):
        if set(sample) != {"index", "rank", "repetition", "raw", "duration"}:
            abort("sample keys differ from the collector contract")
        if sample["index"] != position or sample["rank"] is not None or sample["repetition"] is not None:
            abort("device samples must be a flat 0..iterations-1 sequence")
        if not isinstance(sample["raw"], str) or not sample["raw"]:
            abort("sample raw must be the timer's nonempty decimal representation")
        duration = sample["duration"]
        if set(duration) != {"numerator", "denominator"}:
            abort("sample duration must be a rational object")
        if not all(isinstance(duration[key], int) for key in ("numerator", "denominator")):
            abort("sample duration members must be integers")
        if duration["denominator"] <= 0 or duration["numerator"] < 0:
            abort("sample duration must be a nonnegative rational")
    correctness = body["correctness"]
    if correctness.get("kind") != "e3-parity" or correctness.get("reference") != REFERENCE_ID:
        abort("correctness record is not the pinned E3 parity check")
    for field in ("ref_max",):
        if not collector_decimal(str(correctness.get(field))):
            abort(f"correctness {field} is not a bounded decimal the collector accepts")
    for block in ("e2", "e3"):
        for field in ("maxabs", "per_token_max", "per_token_p99", "nrmse"):
            if not collector_decimal(str(correctness.get(block, {}).get(field))):
                abort(f"correctness {block}.{field} is not a bounded decimal the collector accepts")
    if correctness.get("tolerance") != dict({"kind": "e3-rel"}, **TOLERANCE_MILLI):
        abort("correctness tolerance is not the frozen E3 tolerance")
    for failure in body["failures"]:
        if set(failure) != {"kind", "detail"} or failure["kind"] not in FAILURE_KINDS:
            abort(f"retained failure {failure.get('kind')!r} is not in the closed set")


def check_plan(plan: dict) -> None:
    if plan.get("kind") != "microbench-plan-v1" or plan.get("version") != VERSION:
        abort("plan kind/version is not microbench-plan-v1 version 1")
    if plan.get("adapter") != ADAPTER:
        abort(f"plan adapter {plan.get('adapter')!r} is not {ADAPTER!r}")
    if plan.get("operation") != FROZEN_OPERATION:
        abort("plan operation is not the frozen E3 grouped cell; no shape, cap or skew search exists")
    if plan.get("clock") != CLOCK:
        abort("plan clock is not the pinned cuda-event-ms contract")
    if plan.get("warmups") != WARMUPS or plan.get("iterations") != ITERATIONS:
        abort(f"plan must declare warmups {WARMUPS} and iterations {ITERATIONS}")
    if (plan.get("correctness") or {}).get("reference") != REFERENCE_ID:
        abort("plan correctness reference is not the pinned LinearEXL3 loop")
    tolerance = (plan.get("correctness") or {}).get("tolerance")
    if not isinstance(tolerance, dict) or tolerance.get("kind") != "e3-rel":
        abort("plan correctness tolerance is not the frozen E3 relative tolerance")
    for key, value in TOLERANCE_MILLI.items():
        if tolerance.get(key) != value:
            abort(f"plan tolerance {key} is {tolerance.get(key)!r}, expected {value}")
    allowance = plan.get("allowance") or {}
    for key in ("work_units", "memory_bytes", "deadline_ms"):
        if not isinstance(allowance.get(key), int) or allowance[key] <= 0:
            abort(f"plan allowance {key} must be a positive integer")
    # Admission happens before the torch import and before any device or window
    # is opened: a declared allowance below the frozen cell's known minimum is
    # not an allowance, and it must not be discovered after the kernel ran.
    minimum_work, minimum_memory = minimum_allowance(FROZEN_OPERATION)
    if allowance["work_units"] < minimum_work:
        abort(
            f"plan allowance work_units {allowance['work_units']} is below the frozen E3 cell "
            f"minimum {minimum_work}"
        )
    if allowance["memory_bytes"] < minimum_memory:
        abort(
            f"plan allowance memory_bytes {allowance['memory_bytes']} is below the frozen E3 "
            f"activation payload floor {minimum_memory}"
        )
    revision = plan.get("revision")
    if not isinstance(revision, str) or len(revision) != 64 or not all(
        character in "0123456789abcdef" for character in revision
    ):
        abort("plan revision must be the expected SHA-256 hex of the observed descriptor")
    if not plan.get("program_sha256"):
        abort("plan must pin program_sha256 for an external adapter")
    if not plan.get("sources"):
        abort("plan must declare its pinned sources")


def verify_sources(plan: dict, checkout: str) -> tuple[list[dict], int, int]:
    """Verify every declared pin from bytes on this host and report it.

    Returns the observed sources plus the verified/declared counts. This
    adapter resolves every pin inside the pinned checkout, so nothing is echoed
    as a declaration; Grill still checks the observed hash set against the
    declaration.
    """
    self_path = os.path.abspath(__file__)
    if sha256_file(self_path) != plan.get("program_sha256"):
        abort("this adapter does not match plan.program_sha256; re-pin instead of running it")
    declared_hashes = set()
    observed = []
    verified = 0
    for source in plan["sources"]:
        declared = source.get("path")
        declared_sha = source.get("sha256")
        if not isinstance(declared, str) or not isinstance(declared_sha, str):
            abort("plan sources must be {path, sha256} objects")
        if os.path.basename(declared) == os.path.basename(self_path):
            resolved = self_path
        elif os.path.isabs(declared):
            resolved = declared
        else:
            resolved = os.path.join(checkout, declared)
        if not os.path.isfile(resolved):
            abort(f"pinned source is missing: {declared}")
        observed_sha = sha256_file(resolved)
        if observed_sha != declared_sha:
            abort(f"pinned source hash mismatch for {declared}: {observed_sha} != {declared_sha}")
        observed.append({"path": os.path.realpath(resolved), "sha256": observed_sha})
        declared_hashes.add(observed_sha)
        verified += 1
    if declared_hashes != {source["sha256"] for source in plan["sources"]}:
        abort("observed source hashes differ from the declared pin set")
    if not os.path.isfile(os.path.join(checkout, ANCHOR)) or not os.path.isfile(
        os.path.join(checkout, PARITY)
    ):
        abort("the pinned checkout does not contain both pinned source files")
    return observed, verified, 0


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        abort(f"cannot load pinned module at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parity(anchor, parity_module, torch, apply_exl3_experts, apply_exl3_python_loop, layer):
    """Production-geometry parity before any timing, using the pinned tolerances."""
    from vllm.model_executor.layers.quantization.exl3 import _record_exl3_fat_resolution

    hidden = FROZEN_OPERATION["hidden"]
    tokens = FROZEN_OPERATION["tokens"]
    cap = FROZEN_OPERATION["cap"]

    os.environ["EXL3_FAT_GROUPED"] = "1"
    _record_exl3_fat_resolution(layer)
    if layer._exl3_fat_effective_tier != "grouped":
        abort(
            "E3 grouped tier is not effective: "
            f"{layer._exl3_fat_effective_tier} ({layer._exl3_fat_tier_reason})"
        )
    generator = torch.Generator().manual_seed(FROZEN_OPERATION["activation_seed"])
    activation = torch.randn(tokens, hidden, generator=generator).half().cuda()
    ids, weights = anchor.routing(tokens, skew=1.0, seed=FROZEN_OPERATION["parity_routing_seed"])
    loop = apply_exl3_python_loop(activation, ids.long(), weights, layer._exl3_inners, None, 10.0)

    anchor.set_cap(layer, cap)
    os.environ["EXL3_FAT_GROUPED"] = "0"
    _record_exl3_fat_resolution(layer)
    reference = apply_exl3_experts(activation, ids, weights, layer)
    fallback_e2 = layer._exl3_last_fat_fallback
    e2 = parity_module._err_stats(loop, reference)

    os.environ["EXL3_FAT_GROUPED"] = "1"
    _record_exl3_fat_resolution(layer)
    candidate = apply_exl3_experts(activation, ids, weights, layer)
    fallback_e3 = layer._exl3_last_fat_fallback
    candidate_repeat = apply_exl3_experts(activation, ids, weights, layer)
    e3 = parity_module._err_stats(loop, candidate)

    detail = None
    passed = False
    if fallback_e2 != "kernel":
        detail = f"E2 reference fallback was {fallback_e2!r}, expected 'kernel'"
    elif fallback_e3 != "grouped":
        detail = f"E3 fallback was {fallback_e3!r}; a lower tier was silently used"
    else:
        try:
            parity_module._assert_e3_within("prod_geometry", e2, e3)
            passed = True
        except AssertionError as error:
            detail = str(error)
    finite = bool(e3["finite"]) and bool(e2["finite"])
    if not finite:
        passed = False
        detail = detail or "parity statistics were not finite"
    del activation, loop, reference, candidate, candidate_repeat
    return {"e2": e2, "e3": e3, "passed": passed, "finite": finite, "detail": detail}


def observed_decimal(value: float) -> str:
    """The float's own representation, never re-rounded to fixed decimals.

    `repr` of a Python float is the shortest round-trip decimal for that value,
    so the retained text is exactly what the pinned parity helper produced; it
    may use scientific notation, which the collector accepts.
    """
    return repr(float(value))


def main() -> int:
    parser = argparse.ArgumentParser(description="pinned EXL3 E3 grouped microbenchmark adapter")
    parser.add_argument("--plan", required=True)
    parser.add_argument(
        "--checkout",
        required=True,
        help="root of the pinned MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks checkout",
    )
    parser.add_argument("--out", default=None, help="write the artifact body to this file")
    parser.add_argument("--stdout", action="store_true", help="print the artifact body to stdout")
    args = parser.parse_args()

    with open(args.plan, "rb") as handle:
        plan = json.loads(handle.read())

    check_plan(plan)
    sources, verified, declared = verify_sources(plan, args.checkout)

    anchor_path = os.path.join(args.checkout, ANCHOR)
    parity_path = os.path.join(args.checkout, PARITY)
    deadline_ms = plan["allowance"]["deadline_ms"]
    started = time.monotonic()

    try:
        import torch  # noqa: PLC0415  (imported only inside an authorized run)

        anchor = load_module(anchor_path, "grill_pinned_bench_e3_microbench")
        parity_module = load_module(parity_path, "grill_pinned_test_exl3_overlay")
        if (
            round(parity_module.E3_TOL_FACTOR * 1000) != TOLERANCE_MILLI["factor_milli"]
            or round(parity_module.E3_TOL_ABS_REL * 1_000_000) != TOLERANCE_MILLI["abs_rel_micro"]
            or round(parity_module.E3_TOL_NRMSE_ABS * 1_000_000)
            != TOLERANCE_MILLI["nrmse_abs_micro"]
        ):
            abort("pinned tolerance constants do not match the frozen E3 tolerances")

        from vllm.model_executor.layers.quantization.exl3 import (
            _record_exl3_fat_resolution,
            apply_exl3_experts,
            apply_exl3_python_loop,
            exl3_fat_moe_symbols,
        )

        if not exl3_fat_moe_symbols():
            abort("E3 kernels are not loaded in this vllm build")

        os.environ.update(
            {
                "EXL3_FAT_KERNEL": "1",
                "EXL3_FAT_SORTED": "0",
                "EXL3_FAT_BATCHED": "0",
                "EXL3_MOE_ROW_TILE": "0",
            }
        )
        module_path = getattr(sys.modules[apply_exl3_experts.__module__], "__file__", None)
        if not module_path or not os.path.isfile(module_path):
            abort("cannot observe the loaded exl3 module file")
        device = torch.cuda.get_device_name(torch.cuda.current_device())
        kernel_revision = (
            f"torch:{torch.__version__};"
            f"nccl:{'.'.join(str(part) for part in torch.cuda.nccl.version())};"
            f"exl3-module:{sha256_file(module_path)}"
        )

        layer = anchor.make_layer(
            n_exp=FROZEN_OPERATION["experts"],
            hidden=FROZEN_OPERATION["hidden"],
            inter=FROZEN_OPERATION["intermediate"],
            seed=FROZEN_OPERATION["layer_seed"],
        )
        grade = parity(
            anchor,
            parity_module,
            torch,
            apply_exl3_experts,
            apply_exl3_python_loop,
            layer,
        )

        # Timing cell: one prospectively fixed shape, routing and tier.
        generator = torch.Generator().manual_seed(FROZEN_OPERATION["activation_seed"])
        activation = (
            torch.randn(FROZEN_OPERATION["tokens"], FROZEN_OPERATION["hidden"], generator=generator)
            .half()
            .cuda()
        )
        ids, weights = anchor.routing(
            FROZEN_OPERATION["tokens"],
            skew=FROZEN_OPERATION["skew_milli"] / 1000.0,
            seed=FROZEN_OPERATION["routing_seed"],
        )
        os.environ["EXL3_FAT_GROUPED"] = "1"
        _record_exl3_fat_resolution(layer)
        if layer._exl3_fat_effective_tier != "grouped":
            abort(
                "E3 grouped tier is not effective for the timing cell: "
                f"{layer._exl3_fat_effective_tier} ({layer._exl3_fat_tier_reason})"
            )
        anchor.set_cap(layer, FROZEN_OPERATION["cap"])
        torch.cuda.synchronize()
        before_bytes = torch.cuda.memory_allocated()

        failures = []
        timed_out = time.monotonic() - started > deadline_ms / 1000.0
        samples = []
        raw_fallback = None
        if not timed_out:
            times_ms = anchor.time_fn(
                lambda: apply_exl3_experts(activation, ids, weights, layer),
                iters=ITERATIONS,
                warm=WARMUPS,
            )
            raw_fallback = layer._exl3_last_fat_fallback
            for index, value in enumerate(times_ms):
                converted = sample_duration(value)
                if isinstance(converted, str):
                    # A value that cannot be represented as a duration is
                    # retained as a failure; the grid stays incomplete.
                    failures.append(
                        {
                            "kind": "non-finite" if converted == "non-finite" else "adapter-error",
                            "detail": f"sample {index} could not be represented: {converted}",
                        }
                    )
                    continue
                raw, duration = converted
                samples.append(
                    {
                        "index": len(samples),
                        "rank": None,
                        "repetition": None,
                        "raw": raw,
                        "duration": duration,
                    }
                )
        torch.cuda.synchronize()
        peak_bytes = torch.cuda.max_memory_allocated()

        work_units, minimum_memory = minimum_allowance(FROZEN_OPERATION)
        memory_bytes = int(max(peak_bytes, before_bytes))
        if memory_bytes < minimum_memory:
            # The observed allocator peak cannot be smaller than the activation
            # the timed kernel read; an understated observation is retained as a
            # failure instead of being emitted as if it were sufficient.
            failures.append(
                {
                    "kind": "adapter-error",
                    "detail": (
                        f"observed process-local allocator peak {memory_bytes} is below the frozen "
                        f"activation payload floor {minimum_memory}"
                    ),
                }
            )
        # The pinned source token for a grouped E3 fallback is the literal
        # "grouped"; "e3-grouped" is only this collector's normalized spelling.
        normalized_fallback = normalize_fallback(raw_fallback) if raw_fallback else None
        if raw_fallback != "grouped":
            failures.append(
                {
                    "kind": "tier-fallback",
                    "detail": (
                        "last fallback after timing was "
                        f"{raw_fallback if raw_fallback is not None else 'unavailable'!r}, "
                        "not the pinned 'grouped' token"
                    ),
                }
            )
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
        if time.monotonic() - started > deadline_ms / 1000.0:
            timed_out = True
            failures.append(
                {
                    "kind": "deadline-exceeded",
                    "detail": f"adapter exceeded the declared {deadline_ms} ms deadline",
                }
            )
        if len(samples) != ITERATIONS and not timed_out:
            failures.append(
                {
                    "kind": "incomplete-samples",
                    "detail": f"retained {len(samples)} of {ITERATIONS} measured samples",
                }
            )
        if not grade["passed"]:
            failures.append(
                {
                    "kind": "correctness-failed",
                    "detail": grade["detail"] or "E3 parity failed",
                }
            )
        if not grade["finite"]:
            failures.append({"kind": "non-finite", "detail": "parity statistics were not finite"})

        observations = [
            {"name": "torch-version", "value": str(torch.__version__)},
            {"name": "nccl-version", "value": ".".join(str(part) for part in torch.cuda.nccl.version())},
            {"name": "exl3-module-sha256", "value": sha256_file(module_path)},
            {"name": "device", "value": device},
            {"name": "experts", "value": str(FROZEN_OPERATION["experts"])},
            {"name": "topk", "value": str(FROZEN_OPERATION["topk"])},
            {"name": "tokens", "value": str(FROZEN_OPERATION["tokens"])},
            {"name": "hidden", "value": str(FROZEN_OPERATION["hidden"])},
            {"name": "intermediate", "value": str(FROZEN_OPERATION["intermediate"])},
            {"name": "cap", "value": str(FROZEN_OPERATION["cap"])},
            {"name": "skew-milli", "value": str(FROZEN_OPERATION["skew_milli"])},
            {"name": "routing-seed", "value": str(FROZEN_OPERATION["routing_seed"])},
            {"name": "layer-seed", "value": str(FROZEN_OPERATION["layer_seed"])},
            {"name": "parity-routing-seed", "value": str(FROZEN_OPERATION["parity_routing_seed"])},
            {"name": "activation-seed", "value": str(FROZEN_OPERATION["activation_seed"])},
            {
                # Never echo the declared tier as an observation: when timing
                # did not run there is nothing observed to report, and the
                # collector rejects an unavailable value rather than passing it.
                "name": "fallback-tier",
                # The actual raw overlay token, preserved verbatim.
                "value": raw_fallback if raw_fallback is not None else "unavailable",
            },
            {"name": "sources-verified", "value": str(verified)},
            {"name": "sources-declared", "value": str(declared)},
        ]

        body = artifact_body(
            plan,
            sources,
            observations,
            kernel_revision,
            int(torch.cuda.current_device()),
            samples,
            {
                "warmups": WARMUPS,
                "iterations": ITERATIONS,
                "deadline_ms": deadline_ms,
                "work_units": work_units,
                "memory_bytes": memory_bytes,
                "completed": not timed_out and len(samples) == ITERATIONS,
                "timed_out": timed_out,
                "observed_fallback": normalized_fallback,
            },
            {
                "kind": "e3-parity",
                "reference": REFERENCE_ID,
                "finite": grade["finite"],
                # The pinned `_assert_e3_within` derives the per-key floor and
                # the coarse bound from e2["ref_max"], so that is the reference
                # maximum retained here.
                "ref_max": observed_decimal(grade["e2"]["ref_max"]),
                "e2": {
                    "maxabs": observed_decimal(grade["e2"]["maxabs"]),
                    "per_token_max": observed_decimal(grade["e2"]["per_token_max"]),
                    "per_token_p99": observed_decimal(grade["e2"]["per_token_p99"]),
                    "nrmse": observed_decimal(grade["e2"]["nrmse"]),
                },
                "e3": {
                    "maxabs": observed_decimal(grade["e3"]["maxabs"]),
                    "per_token_max": observed_decimal(grade["e3"]["per_token_max"]),
                    "per_token_p99": observed_decimal(grade["e3"]["per_token_p99"]),
                    "nrmse": observed_decimal(grade["e3"]["nrmse"]),
                },
                "tolerance": dict({"kind": "e3-rel"}, **TOLERANCE_MILLI),
                "passed": bool(grade["passed"]),
                "outcome": "pass" if grade["passed"] else "fail",
                "detail": grade["detail"],
            },
            failures,
        )
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001  (retain the traceback for the operator)
        traceback.print_exc()
        return 3

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
