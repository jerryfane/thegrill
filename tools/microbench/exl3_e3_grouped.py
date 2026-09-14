#!/usr/bin/env python3
"""Fixed-cell EXL3 E3 grouped expert adapter for grill-perf microbenchmark evidence.

This is the *only* device kernel adapter in the #54 slice, and it is bound to a
single prospectively fixed cell. It deliberately does not reproduce
`main()` from the pinned public anchor: no `--skews` ladder, no `--e3-caps`
ladder, no best-median summary, no `torch.cuda.empty_cache()`, and no serving
process is touched. The operation is drawn from the pinned anchor's own
`make_layer`, `routing`, `time_fn` and `set_cap` functions; its parity check
uses `_err_stats` and `_assert_e3_within` from the pinned tolerance file.

Pinned public sources (verified by hash at run time; a mismatch aborts before
any device work):

  MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks
  revision f906ee990596486e10ddbe381efa6f0e496f77e3
    tests/bench_e3_microbench.py  f57e4fa6726b0110c44ce56b0b0968ba425e566ace6d89f2db3872aae13889ec
    tests/test_exl3_overlay.py    8777254d47cdbb89186ba3bd1ac2437cbebc7987e0791d36c68e7bccb3df2b35

Frozen cell: apply_exl3_experts, hidden 4096, intermediate 1024, experts 288,
topk 8, tokens 1024, layer_seed 0, routing_seed 1, parity routing seed 3,
activation seed 3, skew 1.0, tier E3 grouped, cap 32, warmups 2, measured
iterations 5; synthetic Zipf routing, never served routing; parity at
production geometry asserted before any timing; every measured sample requires
the effective tier and the last fallback to be grouped, so a silent lower-tier
fallback is withheld rather than timed as E3.

Declared deviations from the anchor, both additive:
  * the timed activation tensor is drawn from a seeded generator using the
    declared activation seed (the anchor draws it unseeded); shape, dtype and
    routing are unchanged;
  * the pinned candidate change axis is the adapter revision string in the
    plan, so comparisons here are revision-to-revision, never cell searches.

Raw device-event millisecond samples are retained with three fractional
digits (microsecond precision) and converted to exact nanoseconds; no
sub-microsecond digit is invented.

Usage (inside a separately authorized device window only):

    grill-perf microbench capture \
      --adapter exl3-e3-grouped \
      --plan plan.json \
      --program tools/microbench/exl3_e3_grouped.py \
      --authorize-device-window \
      -- --checkout /path/to/GLM-5.3-Flash-EXL3-2x-DGX-Sparks

Without `--out`, the artifact body is printed to stdout for the collector.
"""

from __future__ import annotations

import argparse
import decimal
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback

ADAPTER = "exl3-e3-grouped"
KIND = "microbench-artifact-v1"
VERSION = 1
REVISION_PIN = "f906ee990596486e10ddbe381efa6f0e496f77e3"

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
SAMPLE_DECIMALS = 3


def abort(message: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"{ADAPTER}: {message}", file=sys.stderr, flush=True)
    sys.exit(2)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ns_from_ms_text(text: str) -> int:
    """Exact millisecond-text to nanosecond conversion, mirroring the collector."""
    value = decimal.Decimal(text)
    scaled = value * 1_000_000
    if scaled != scaled.to_integral_value():
        raise ValueError(f"duration {text} ms is finer than one nanosecond")
    return int(scaled)


def ms_text(milliseconds: float) -> str:
    """Three fractional digits: microsecond precision, never sub-microsecond."""
    return f"{milliseconds:.{SAMPLE_DECIMALS}f}"


def decimal9(value: float) -> str:
    """Plain fixed-point text; fixed exponent form is rejected by the collector."""
    return f"{value:.9f}"


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        abort(f"cannot load pinned module at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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
    tolerance = (plan.get("correctness") or {}).get("tolerance")
    if (plan.get("correctness") or {}).get("reference") != REFERENCE_ID:
        abort("plan correctness reference is not the pinned LinearEXL3 loop")
    if not isinstance(tolerance, dict) or tolerance.get("kind") != "e3-rel":
        abort("plan correctness tolerance is not the frozen E3 relative tolerance")
    for key, value in TOLERANCE_MILLI.items():
        if tolerance.get(key) != value:
            abort(f"plan tolerance {key} is {tolerance.get(key)!r}, expected {value}")
    allowance = plan.get("allowance") or {}
    for key in ("work_units", "memory_bytes", "deadline_ms"):
        if not isinstance(allowance.get(key), int) or allowance[key] <= 0:
            abort(f"plan allowance {key} must be a positive integer")
    if not plan.get("program_sha256"):
        abort("plan must pin program_sha256 for an external adapter")


def verify_sources(plan: dict, checkout: str) -> list[dict]:
    """Verify every declared source pin and return the exact declared list."""
    self_path = os.path.abspath(__file__)
    observed_self = sha256_file(self_path)
    sources = plan.get("sources") or []
    if not sources:
        abort("plan must declare its pinned sources")
    reported = []
    for source in sources:
        declared = source.get("path")
        declared_sha = source.get("sha256")
        if not isinstance(declared, str) or not isinstance(declared_sha, str):
            abort("plan sources must be {path, sha256} objects")
        resolved = declared
        if declared == os.path.relpath(self_path) or os.path.basename(declared) == os.path.basename(
            self_path
        ):
            resolved = self_path
        elif not os.path.isabs(declared):
            resolved = os.path.join(checkout, declared)
        if not os.path.isfile(resolved):
            abort(f"pinned source is missing: {declared}")
        observed = sha256_file(resolved)
        if observed != declared_sha:
            abort(f"pinned source hash mismatch for {declared}: {observed} != {declared_sha}")
        reported.append({"path": declared, "sha256": observed})
    if observed_self != plan.get("program_sha256"):
        abort("this adapter does not match plan.program_sha256; re-pin instead of running it")
    return reported


def parity(anchor, parity_module, torch, apply_exl3_experts, apply_exl3_python_loop, layer, seed_routing, seed_activation):
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
    generator = torch.Generator().manual_seed(seed_activation)
    activation = torch.randn(tokens, hidden, generator=generator).half().cuda()
    ids, weights = anchor.routing(tokens, skew=1.0, seed=seed_routing)
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
    repeat = parity_module._err_stats(candidate, candidate_repeat)

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
    finite = bool(e3["finite"]) and bool(e2["finite"]) and bool(repeat["finite"])
    if not finite:
        passed = False
        detail = detail or "parity statistics were not finite"
    del activation, loop, reference, candidate, candidate_repeat
    return {
        "e2": e2,
        "e3": e3,
        "repeat": repeat,
        "passed": passed,
        "finite": finite,
        "detail": detail,
        "fallback": fallback_e3,
    }


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
    sources = verify_sources(plan, args.checkout)

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
        kernel_revision = (
            f"torch:{torch.__version__};"
            f"nccl:{'.'.join(str(part) for part in torch.cuda.nccl.version())};"
            f"anchor:{REVISION_PIN}"
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
            FROZEN_OPERATION["parity_routing_seed"],
            FROZEN_OPERATION["activation_seed"],
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
        observed_fallback = None
        if not timed_out:
            times_ms = anchor.time_fn(
                lambda: apply_exl3_experts(activation, ids, weights, layer),
                iters=ITERATIONS,
                warm=WARMUPS,
            )
            observed_fallback = layer._exl3_last_fat_fallback
            if observed_fallback != "grouped":
                failures.append(
                    {
                        "kind": "tier-fallback",
                        "detail": f"last fallback after timing was {observed_fallback!r}, not 'grouped'",
                    }
                )
            for index, value in enumerate(times_ms):
                text = ms_text(float(value))
                try:
                    nanos = ns_from_ms_text(text)
                except ValueError as error:
                    failures.append({"kind": "adapter-error", "detail": str(error)})
                    nanos = 0
                samples.append(
                    {
                        "index": index,
                        "rank": None,
                        "repetition": None,
                        "raw": text,
                        "duration_ns": nanos,
                    }
                )
        torch.cuda.synchronize()
        peak_bytes = torch.cuda.max_memory_allocated()

        work_units = (
            FROZEN_OPERATION["tokens"]
            * FROZEN_OPERATION["topk"]
            * 3
            * 2
            * FROZEN_OPERATION["hidden"]
            * FROZEN_OPERATION["intermediate"]
        )
        memory_bytes = int(max(peak_bytes, before_bytes))
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

        artifact = {
            "kind": KIND,
            "version": VERSION,
            "adapter": ADAPTER,
            "revision": plan["revision"],
            "operation": FROZEN_OPERATION,
            "provenance": "native_observed",
            "submitted_provenance": None,
            "program_sha256": plan["program_sha256"],
            "kernel_revision": kernel_revision,
            "sources": sources,
            "topology": {
                "scope": "device",
                "world": None,
                "ranks": [{"index": 0, "device": int(torch.cuda.current_device())}],
            },
            "clock": CLOCK,
            "execution": {
                "warmups": WARMUPS,
                "iterations": ITERATIONS,
                "deadline_ms": deadline_ms,
                "work_units": work_units,
                "memory_bytes": memory_bytes,
                "completed": not timed_out and len(samples) == ITERATIONS,
                "timed_out": timed_out,
                "observed_fallback": observed_fallback,
            },
            "samples": samples,
            "correctness": {
                "kind": "e3-parity",
                "reference": REFERENCE_ID,
                "finite": grade["finite"],
                "ref_max": decimal9(grade["e3"]["ref_max"]),
                "e2": {
                    "maxabs": decimal9(grade["e2"]["maxabs"]),
                    "per_token_max": decimal9(grade["e2"]["per_token_max"]),
                    "per_token_p99": decimal9(grade["e2"]["per_token_p99"]),
                    "nrmse": decimal9(grade["e2"]["nrmse"]),
                },
                "e3": {
                    "maxabs": decimal9(grade["e3"]["maxabs"]),
                    "per_token_max": decimal9(grade["e3"]["per_token_max"]),
                    "per_token_p99": decimal9(grade["e3"]["per_token_p99"]),
                    "nrmse": decimal9(grade["e3"]["nrmse"]),
                },
                "tolerance": dict({"kind": "e3-rel"}, **TOLERANCE_MILLI),
                "passed": bool(grade["passed"]),
                "outcome": "pass" if grade["passed"] else "fail",
                "detail": grade["detail"],
            },
            "failures": failures,
        }
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001  (retain the traceback for the operator)
        traceback.print_exc()
        return 3

    body = json.dumps(artifact, indent=2, sort_keys=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(body + "\n")
    if args.stdout or not args.out:
        sys.stdout.write(body + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
