#!/usr/bin/env python3
"""Offline observed-envelope policy oracle; maintainer-only experiment.

This driver independently recomputes the private policy.rs `evaluate` gate
decision (pooled reference extrema versus candidate extrema under integer
basis-point tolerances) using exact `fractions.Fraction` arithmetic plus an
explicit u128 checked-multiplication model. It never collects evidence, never
contacts a serving endpoint and never changes any verdict. It is a bounded
offline cross-check of gate arithmetic only; it does not calibrate statistical
confidence, does not claim guaranteed sensitivity and does not counterfeit the
full native `decide` workflow (evidence loading, coverage, role ordering and
output-amount matching are exercised by the existing real CLI policy tests).

Usage:
    python3 tools/calibrate-policy.py --out DIR [--seed N] [--replicates N]
    python3 tools/calibrate-policy.py --out DIR --production-results FILE \
        --production-binary FILE

The production cross-check consumes the JSONL written by the ignored Rust hook
`policy::policy_calibration::crosscheck_corpus` (GRILL_POLICY_CALIBRATION_*
environment variables); see docs/performance/CALIBRATION.md for the harness
pattern this follows.
"""

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import sys


# Metric names are the exact serde snake_case names admitted by policy.rs.
# Direction is intrinsic to the metric: wave_latency_us is adverse-high;
# the three rate metrics are adverse-low.
METRICS = {
    "wave_latency_us": "latency",
    "achieved_completion_tokens_per_second": "rate",
    "decode_tokens_per_second": "rate",
    "prefill_tokens_per_second": "rate",
}
U128_MAX = (1 << 128) - 1
# Schema bounds enforced by policy admission (policy.rs parse).
MAX_REGRESSION_BPS = 9999
MAX_REFERENCE_SPREAD_BPS = 1_000_000
U64_MAX = (1 << 64) - 1
# Corpus rows are bounded: extrema are u64 numerator/denominator pairs and the
# hook refuses inputs past this byte cap.
CORPUS_CAP_BYTES = 128 * 1024 * 1024
OUTCOMES = ("PASS", "INCONCLUSIVE", "REGRESSION", "ERROR")
REASONS = (
    "nonpositive_reference",
    "reference_spread_exceeded",
    "envelope_straddles_tolerance",
    "arithmetic_overflow",
)

# Prospective scenario families. Each is generated for both metric directions.
# They characterize gate behavior under named synthetic conditions; they are
# not qualified serving models and their frequencies are not confidence levels.
SCENARIOS = {
    # Candidate extrema identical to reference extrema: must always PASS.
    "unchanged": {"kind": "unchanged"},
    # Candidate extrema inside half the tolerance envelope: must always PASS.
    "noise_within_tolerance": {"kind": "noise", "max_fraction_of_tolerance": 0.5},
    # Candidate extrema up to twice the tolerance: PASS/INCONCLUSIVE/REGRESSION
    # mix; the unresolved-gate frequency is reported, not tuned.
    "noise_straddling_tolerance": {"kind": "noise", "max_fraction_of_tolerance": 2.0},
    # Reference extrema spread beyond the declared spread budget: must always
    # resolve reference_spread_exceeded (INCONCLUSIVE).
    "reference_drift": {"kind": "reference_drift"},
    # Reference and candidate shift together by the same factor: the gate sees
    # no relative change and must PASS; absolute drift alone is not regression.
    "correlated_shift": {"kind": "correlated"},
    # Adverse candidate shift strictly beyond tolerance: must always REGRESSION.
    "slowdown_beyond_tolerance": {"kind": "shift", "adverse": True,
                                  "min_fraction_of_tolerance": 1.5,
                                  "max_fraction_of_tolerance": 4.0},
    # Adverse candidate shift strictly inside tolerance: must always PASS.
    # A zero tolerance admits no within-tolerance slowdown, so this family
    # only draws nonzero tolerances.
    "slowdown_within_tolerance": {"kind": "shift", "adverse": True,
                                  "nonzero_tolerance": True,
                                  "min_fraction_of_tolerance": 0.25,
                                  "max_fraction_of_tolerance": 0.75},
    # Favorable candidate shift beyond tolerance: must always PASS; improvement
    # never rescues or masks a gate in this single-gate corpus.
    "improvement_beyond_tolerance": {"kind": "shift", "adverse": False,
                                     "min_fraction_of_tolerance": 1.5,
                                     "max_fraction_of_tolerance": 4.0},
    # One candidate extremum improves while the other regresses beyond
    # tolerance: the envelope straddles and must resolve INCONCLUSIVE.
    "mixed_extrema_straddle": {"kind": "straddle"},
}


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def scaled(numerator, scale, denominator):
    """policy.rs `scaled`: checked u128 a.numerator * scale * b.denominator."""
    first = numerator * scale
    if first > U128_MAX:
        return None
    product = first * denominator
    return product if product <= U128_MAX else None


def ratio(a, b):
    """policy.rs `ratio`: (a.n as f64 / a.d as f64) / (b.n as f64 / b.d as f64)."""
    return (float(a.numerator) / float(a.denominator)) / (float(b.numerator) / float(b.denominator))


def oracle(metric, tolerance, spread, ref_low, ref_high, cand_low, cand_high):
    """Independent recomputation of the private `evaluate` gate.

    `ref_*`/`cand_*` are Fractions; `metric` is a policy.rs Metric name.
    Returns {"decision", "evaluate_error", "adverse_bounds"} matching the
    gate fields the hook observes: `evaluate_error` is the reason `evaluate`
    returns as Err (null on Ok), and `decision` applies decide()'s mapping
    (arithmetic_overflow -> ERROR, other Err -> INCONCLUSIVE).
    """
    direction = METRICS[metric]
    report = {"decision": None, "evaluate_error": None, "adverse_bounds": None}
    if ref_low.numerator == 0:
        report.update(decision="INCONCLUSIVE", evaluate_error="nonpositive_reference")
        return report
    lhs = scaled(ref_high.numerator, 10000, ref_low.denominator)
    rhs = scaled(ref_low.numerator, 10000 + spread, ref_high.denominator)
    if lhs is None or rhs is None:
        report.update(decision="ERROR", evaluate_error="arithmetic_overflow")
        return report
    if ref_high / ref_low - 1 > Fraction(spread, 10000):
        report.update(decision="INCONCLUSIVE", evaluate_error="reference_spread_exceeded")
        return report
    if direction == "latency":
        report["adverse_bounds"] = [ratio(cand_low, ref_high) - 1.0,
                                    ratio(cand_high, ref_low) - 1.0]
        passed = scaled(cand_high.numerator, 10000, ref_low.denominator)
        passed_rhs = scaled(ref_low.numerator, 10000 + tolerance, cand_high.denominator)
        regressed = scaled(cand_low.numerator, 10000, ref_high.denominator)
        regressed_rhs = scaled(ref_high.numerator, 10000 + tolerance, cand_low.denominator)
        products = (passed, passed_rhs, regressed, regressed_rhs)
        if any(p is None for p in products):
            report.update(decision="ERROR", evaluate_error="arithmetic_overflow")
            return report
        passed = cand_high / ref_low - 1 <= Fraction(tolerance, 10000)
        regressed = cand_low / ref_high - 1 > Fraction(tolerance, 10000)
    else:
        report["adverse_bounds"] = [1.0 - ratio(cand_high, ref_low),
                                    1.0 - ratio(cand_low, ref_high)]
        passed = scaled(cand_low.numerator, 10000, ref_high.denominator)
        passed_rhs = scaled(ref_high.numerator, 10000 - tolerance, cand_low.denominator)
        regressed = scaled(cand_high.numerator, 10000, ref_low.denominator)
        regressed_rhs = scaled(ref_low.numerator, 10000 - tolerance, cand_high.denominator)
        products = (passed, passed_rhs, regressed, regressed_rhs)
        if any(p is None for p in products):
            report.update(decision="ERROR", evaluate_error="arithmetic_overflow")
            return report
        passed = 1 - cand_low / ref_high <= Fraction(tolerance, 10000)
        regressed = 1 - cand_high / ref_low > Fraction(tolerance, 10000)
    if passed:
        report["decision"] = "PASS"
    elif regressed:
        report["decision"] = "REGRESSION"
    else:
        report.update(decision="INCONCLUSIVE", evaluate_error="envelope_straddles_tolerance")
    return report


def pair(value):
    return [value.numerator, value.denominator]


def row(case_id, metric, tolerance, spread, ref_low, ref_high, cand_low, cand_high):
    for value in (ref_low, ref_high, cand_low, cand_high):
        if not (0 <= value.numerator <= U64_MAX and 1 <= value.denominator <= U64_MAX):
            raise ValueError(f"{case_id}: extremum outside u64 pair bounds")
    if ref_low > ref_high or cand_low > cand_high:
        raise ValueError(f"{case_id}: extrema not ordered")
    expected = oracle(metric, tolerance, spread, ref_low, ref_high, cand_low, cand_high)
    return {
        "id": case_id,
        "metric": metric,
        "direction": METRICS[metric],
        "max_regression_bps": tolerance,
        "max_reference_spread_bps": spread,
        "reference": {"low": pair(ref_low), "high": pair(ref_high)},
        "candidate": {"low": pair(cand_low), "high": pair(cand_high)},
        "expected": expected,
    }


def boundary_checks():
    """Deterministic exact-boundary, direction, zero and overflow cases."""
    f = Fraction
    cases = [
        # Latency: worst bound exactly at tolerance passes; strictly beyond
        # regresses; best bound exactly at tolerance is not a regression.
        ("latency_pass_exact", "wave_latency_us", 500, 0,
         f(10000), f(10000), f(10000), f(10500)),
        ("latency_regression_strict", "wave_latency_us", 500, 0,
         f(10000), f(10000), f(10501), f(11000)),
        ("latency_regression_boundary_is_not_regression", "wave_latency_us", 500, 0,
         f(10000), f(10000), f(10500), f(10501)),
        ("latency_improvement", "wave_latency_us", 0, 0,
         f(10000), f(10000), f(5000), f(9000)),
        # Rate: worst adverse bound exactly at tolerance passes; strictly beyond
        # regresses; best adverse bound exactly at tolerance is not a regression.
        ("rate_pass_exact", "achieved_completion_tokens_per_second", 500, 0,
         f(10000), f(10000), f(9500), f(10000)),
        ("rate_regression_strict", "achieved_completion_tokens_per_second", 500, 0,
         f(10000), f(10000), f(9000), f(9499)),
        ("rate_regression_boundary_is_not_regression", "achieved_completion_tokens_per_second", 500, 0,
         f(10000), f(10000), f(9400), f(9500)),
        ("rate_improvement", "decode_tokens_per_second", 0, 0,
         f(10000), f(10000), f(11000), f(20000)),
        # Direction inversion: the same extrema swap outcomes across directions.
        ("direction_latency_adverse_high", "wave_latency_us", 0, 0,
         f(10000), f(10000), f(10001), f(10001)),
        ("direction_rate_adverse_low", "prefill_tokens_per_second", 0, 0,
         f(10000), f(10000), f(9999), f(9999)),
        # Zero values: a zero reference minimum is nonpositive_reference; a
        # defined zero candidate is a rate regression but a latency pass.
        ("zero_reference_minimum", "wave_latency_us", 0, 0,
         f(0), f(10000), f(10000), f(10000)),
        ("zero_candidate_rate", "achieved_completion_tokens_per_second", 9999, 0,
         f(10000), f(10000), f(0), f(0)),
        ("zero_candidate_latency", "wave_latency_us", 0, 0,
         f(10000), f(10000), f(0), f(0)),
        # Reference spread: exact budget boundary passes the spread gate;
        # one ulp beyond resolves reference_spread_exceeded.
        ("spread_exact_boundary", "wave_latency_us", 0, 10000,
         f(10000), f(20000), f(10000), f(10000)),
        ("spread_beyond_budget", "wave_latency_us", 0, 10000,
         f(10000), f(20001), f(10000), f(10000)),
        # Envelope straddling: candidate extrema bracket the tolerance bound.
        ("latency_straddle", "wave_latency_us", 500, 0,
         f(10000), f(10000), f(9000), f(11000)),
        ("rate_straddle", "achieved_completion_tokens_per_second", 500, 0,
         f(10000), f(10000), f(9000), f(11000)),
    ]
    # u128 checked products are num * scale * den with u64-bounded inputs, so
    # overflow needs large numerators AND denominators. The spread check runs
    # first: an overflowing spread product resolves arithmetic_overflow (ERROR),
    # never reference_spread_exceeded.
    cases.append(("overflow_spread_products_fit", "wave_latency_us", 0,
                  MAX_REFERENCE_SPREAD_BPS,
                  f(1, U64_MAX), f(U64_MAX // 10000, 1), f(1, U64_MAX), f(1, U64_MAX)))
    cases.append(("overflow_spread_products_exceed", "wave_latency_us", 0,
                  MAX_REFERENCE_SPREAD_BPS,
                  f(1, U64_MAX), f(U64_MAX, 1), f(1, U64_MAX), f(1, U64_MAX)))
    # Spread check passes (equal extrema) but a decision product overflows.
    cases.append(("overflow_decision_products_fit", "wave_latency_us",
                  MAX_REGRESSION_BPS, 0,
                  f(1, U64_MAX), f(1, U64_MAX), f(U64_MAX // 10000, 1), f(U64_MAX // 10000, 1)))
    cases.append(("overflow_decision_products_exceed", "wave_latency_us",
                  MAX_REGRESSION_BPS, 0,
                  f(1, U64_MAX), f(1, U64_MAX), f(U64_MAX, 1), f(U64_MAX, 1)))
    expected = {
        "latency_pass_exact": ("PASS", None),
        "latency_regression_strict": ("REGRESSION", None),
        "latency_regression_boundary_is_not_regression":
            ("INCONCLUSIVE", "envelope_straddles_tolerance"),
        "latency_improvement": ("PASS", None),
        "rate_pass_exact": ("PASS", None),
        "rate_regression_strict": ("REGRESSION", None),
        "rate_regression_boundary_is_not_regression":
            ("INCONCLUSIVE", "envelope_straddles_tolerance"),
        "rate_improvement": ("PASS", None),
        "direction_latency_adverse_high": ("REGRESSION", None),
        "direction_rate_adverse_low": ("REGRESSION", None),
        "zero_reference_minimum": ("INCONCLUSIVE", "nonpositive_reference"),
        "zero_candidate_rate": ("REGRESSION", None),
        "zero_candidate_latency": ("PASS", None),
        "spread_exact_boundary": ("PASS", None),
        "spread_beyond_budget": ("INCONCLUSIVE", "reference_spread_exceeded"),
        "latency_straddle": ("INCONCLUSIVE", "envelope_straddles_tolerance"),
        "rate_straddle": ("INCONCLUSIVE", "envelope_straddles_tolerance"),
        "overflow_spread_products_fit": ("INCONCLUSIVE", "reference_spread_exceeded"),
        "overflow_spread_products_exceed": ("ERROR", "arithmetic_overflow"),
        "overflow_decision_products_fit": ("REGRESSION", None),
        "overflow_decision_products_exceed": ("ERROR", "arithmetic_overflow"),
    }
    rows = []
    for case_id, metric, tolerance, spread, rl, rh, cl, ch in cases:
        entry = row("boundary/" + case_id, metric, tolerance, spread, rl, rh, cl, ch)
        decision, error = expected[case_id]
        actual = entry["expected"]
        if (actual["decision"], actual["evaluate_error"]) != (decision, error):
            raise ValueError(
                f"boundary check {case_id}: {actual['decision']}/{actual['evaluate_error']} "
                f"!= declared {decision}/{error}")
        rows.append(entry)
    return rows


def draw_rational(rng, minimum=1, maximum=1_000_000):
    return Fraction(rng.randint(minimum, maximum), rng.randint(1, 65_536))


def bound(value):
    """Keep generated rational extrema exact; reject out-of-schema inputs."""
    if not (0 <= value.numerator <= U64_MAX and 1 <= value.denominator <= U64_MAX):
        raise ValueError("generated extremum outside u64 pair bounds")
    return value


def draw_extrema(rng, center, half_width):
    """Ordered extrema around `center` with relative half-width <= half_width."""
    half = Fraction(rng.randint(0, 10**6), 10**6) * half_width
    low, high = bound(center * (1 - half)), bound(center * (1 + half))
    return min(low, high), max(low, high)


def sample(rng, name, spec, direction):
    """One prospectively generated gate input; returns (tolerance, spread,
    ref_low, ref_high, cand_low, cand_high).

    Families with a guaranteed outcome constrain the declared spread budget to
    at most tolerance/4 so the reference envelope cannot itself decide the
    gate; mixed-outcome families draw the budget freely. The actual reference
    spread always stays within the declared budget except in reference_drift.
    """
    tolerance = rng.choice([100, 500, 2000] if spec.get("nonzero_tolerance")
                           else [0, 100, 500, 2000])
    kind = spec["kind"]
    guaranteed = kind in ("unchanged", "shift", "correlated") or (
        kind == "noise" and spec["max_fraction_of_tolerance"] <= 0.5)
    if guaranteed:
        spread = rng.randint(0, tolerance // 4)
    else:
        spread = rng.choice([0, 5000, 20000])
    base = draw_rational(rng)
    ref_low = base
    ref_high = bound(base * (1 + Fraction(rng.randint(0, spread), 10000)))
    # Shift magnitudes are fractions of the declared tolerance; a zero
    # tolerance still gets a nonzero shift so the scenario stays meaningful.
    effective = Fraction(max(tolerance, 100), 10000)
    if kind == "unchanged":
        cand_low, cand_high = ref_low, ref_high
    elif kind == "noise":
        # Candidate extrema jitter around the same center as the reference.
        width = Fraction(tolerance, 10000) * Fraction(str(spec["max_fraction_of_tolerance"]))
        cand_low, cand_high = draw_extrema(rng, base, width)
    elif kind == "reference_drift":
        # Reference extrema spread beyond the declared budget.
        ref_high = bound(base * (1 + Fraction(spread + rng.randint(1, 5000), 10000)))
        cand_low, cand_high = ref_low, ref_low
    elif kind == "correlated":
        # A common factor shifts both reference and candidate periods. This
        # models an unidentifiable shared change, not independent trial noise.
        factor = 1 + Fraction(rng.randint(-200, 200), 100) * effective
        ref_low, ref_high = bound(ref_low * factor), bound(ref_high * factor)
        cand_low, cand_high = ref_low, ref_high
    elif kind == "shift":
        fraction = Fraction(rng.randint(
            int(spec["min_fraction_of_tolerance"] * 100),
            int(spec["max_fraction_of_tolerance"] * 100)), 100)
        magnitude = effective * fraction
        if spec["adverse"]:
            factor = (1 + magnitude) if direction == "latency" else 1 / (1 + magnitude)
        else:
            factor = 1 / (1 + magnitude) if direction == "latency" else (1 + magnitude)
        cand_low = cand_high = bound(base * factor)
    elif kind == "straddle":
        # Both extrema land outside the tolerance band on opposite sides, so
        # the envelope always straddles: never PASS, never REGRESSION.
        below = effective * Fraction(rng.randint(150, 400), 100)
        above = effective * Fraction(rng.randint(150, 400), 100)
        cand_low, cand_high = bound(base * (1 - below)), bound(base * (1 + above))
        cand_low, cand_high = min(cand_low, cand_high), max(cand_low, cand_high)
    else:
        raise ValueError(name)
    return tolerance, spread, ref_low, ref_high, cand_low, cand_high


def close(actual, expected):
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            close(a, e) for a, e in zip(actual, expected))
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isclose(
            actual, expected, rel_tol=1e-12, abs_tol=1e-12)
    return actual == expected


EXPECTED_FIELDS = ("decision", "evaluate_error", "adverse_bounds")


def cross_check(path, binary, corpus, pins):
    """Compare hook output row-by-row; disagreements are reported, not hidden."""
    expected_header = {
        "kind": "policy-evaluate-crosscheck-v1",
        "corpus_sha256": digest(corpus),
        "policy_sha256": pins["crates/grill-perf/src/policy.rs"],
        "evaluator_sha256": digest(binary),
    }
    count = 0
    disagreements = []
    with path.open() as results, corpus.open() as inputs:
        header_line = next(results, None)
        if header_line is None:
            raise ValueError("production results missing identity header")
        header = json.loads(header_line)
        if header != expected_header:
            raise ValueError("production cross-check header does not match runtime identities")
        for line in inputs:
            row = json.loads(line)
            actual_line = next(results, None)
            if actual_line is None:
                raise ValueError(f"production results truncated before {row['id']}")
            actual = json.loads(actual_line)
            count += 1
            if actual["id"] != row["id"]:
                raise ValueError("production row identity/order mismatch")
            for field in EXPECTED_FIELDS:
                if field not in actual["report"] or not close(
                    actual["report"][field], row["expected"][field]
                ):
                    disagreements.append({"id": row["id"], "field": field,
                                          "expected": row["expected"][field],
                                          "actual": actual["report"].get(field)})
            pooled = [
                {"numerator": n, "denominator": d}
                for n, d in (row["reference"]["low"], row["reference"]["high"])
            ]
            if actual["report"].get("pooled_reference_range") != pooled:
                disagreements.append({"id": row["id"], "field": "pooled_reference_range",
                                      "expected": pooled,
                                      "actual": actual["report"].get("pooled_reference_range")})
        if next(results, None) is not None:
            raise ValueError("unexpected extra production rows")
    return {
        "status": "matched" if not disagreements else "mismatched",
        "rows": count,
        "arithmetic_disagreements": {"count": len(disagreements),
                                     "first": disagreements[:20]},
        "results_sha256": digest(path),
        "evaluator_sha256": expected_header["evaluator_sha256"],
        "build_provenance": "operator must independently verify binary/source correspondence",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new output directory; never overwritten")
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--production-results", type=Path)
    parser.add_argument("--production-binary", type=Path)
    args = parser.parse_args()
    if not 1 <= args.replicates <= 10000:
        parser.error("--replicates must be in 1..10000; choose before inspecting results")
    if bool(args.production_results) != bool(args.production_binary):
        parser.error("production results and binary must be supplied together")
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve(), root / "docs/performance/CALIBRATION.md",
             root / "docs/performance/CONTRACT.md",
             root / "Cargo.toml", root / "Cargo.lock", root / "crates/grill-perf/Cargo.toml",
             root / "crates/grill-perf/tests/support/policy_calibration.rs"]
    paths.extend(sorted((root / "crates/grill-perf/src").glob("*.rs")))
    pins = {str(path.relative_to(root)): digest(path) for path in paths}
    checks = boundary_checks()
    args.out.mkdir(parents=True, exist_ok=False)
    corpus = args.out / "corpus.jsonl"
    scenarios = []
    with corpus.open("wb") as output:
        for entry in checks:
            output.write(encoded(entry))
        for name, spec in SCENARIOS.items():
            for direction in ("latency", "rate"):
                metric = "wave_latency_us" if direction == "latency" else "achieved_completion_tokens_per_second"
                seed = int.from_bytes(
                    hashlib.sha256(f"{args.seed}/{name}/{direction}".encode()).digest(), "big")
                rng = random.Random(seed)
                counts = Counter()
                unresolved = 0
                for replicate in range(args.replicates):
                    tolerance, spread, rl, rh, cl, ch = sample(rng, name, spec, direction)
                    entry = row(f"{name}/{direction}/{replicate}", metric, tolerance,
                                spread, rl, rh, cl, ch)
                    output.write(encoded(entry))
                    expected = entry["expected"]
                    counts[expected["decision"]] += 1
                    if expected["evaluate_error"] is not None:
                        counts["reason:" + expected["evaluate_error"]] += 1
                    if expected["decision"] == "INCONCLUSIVE":
                        unresolved += 1
                scenarios.append({
                    "name": name, "direction": direction, "parameters": spec,
                    "rng_seed": seed,
                    "outcomes": {key: counts[key] for key in OUTCOMES},
                    "reasons": {key[7:]: counts[key] for key in counts
                                if key.startswith("reason:")},
                    "unresolved_gate_frequency": {
                        "numerator": unresolved, "denominator": args.replicates,
                        "rate": unresolved / args.replicates},
                })
    if corpus.stat().st_size > CORPUS_CAP_BYTES:
        raise ValueError("generated corpus exceeds the hook input cap; reduce --replicates")
    production = {"status": "not_run",
                  "reason": "requires the ignored policy::policy_calibration::crosscheck_corpus hook"}
    if args.production_results:
        production = cross_check(args.production_results, args.production_binary, corpus, pins)
    if any(digest(root / name) != sha for name, sha in pins.items()):
        raise ValueError("identity files changed during calibration; report withheld")
    report = {
        "kind": "policy-evaluate-offline-calibration-v1",
        "seed": args.seed, "replicates_per_scenario_direction": args.replicates,
        "identities_sha256": pins, "corpus_sha256": digest(corpus),
        "runtime": {"python": sys.version, "platform": platform.platform()},
        "boundary_checks": {"status": "passed",
                            "cases": [entry["id"] for entry in checks]},
        "production_cross_check": production,
        "method": {
            "unit": "one policy gate: pooled reference extrema versus candidate extrema",
            "oracle": "stdlib fractions.Fraction plus an explicit u128 checked-multiplication model",
            "directions": {"latency": "adverse-high (wave_latency_us)",
                           "rate": "adverse-low (achieved/decode/prefill tokens per second)"},
            "decision_rule": "PASS requires the worst adverse bound within tolerance; "
                             "REGRESSION requires the best adverse bound strictly beyond "
                             "tolerance; otherwise INCONCLUSIVE; overflow is ERROR",
            "assumptions": "extrema are supplied directly; wave/lane observation, coverage, "
                           "role ordering, output-amount matching and evidence verification "
                           "are outside this corpus and covered by existing real CLI policy tests",
        },
        "budget": {"boundary_cases": len(checks),
                   "scenario_rows": args.replicates * len(SCENARIOS) * 2,
                   "live_requests": 0,
                   "stopping": "fixed replicates; no extension, replacement, retries or "
                               "outcome-conditioned exclusions"},
        "scenarios": scenarios,
        "qualification": "offline synthetic oracle only; not live qualification, not a "
                         "statistical-confidence calibration and not a product gate",
        "limitations": [
            "gate inputs are supplied as extrema; no claim about how real captures produce them",
            "scenario frequencies describe this fixed synthetic experiment, not serving behavior",
            "no statistical-confidence, guaranteed-sensitivity, equivalence or causal claim",
            "coverage, eligibility and evidence-verification gates outside pure evaluate are "
            "exercised by the existing real CLI policy tests, not by this corpus",
            "arithmetic disagreements are reported under production_cross_check, never edited",
        ],
    }
    (args.out / "report.json").write_bytes(encoded(report))
    print(args.out / "report.json")
    if production.get("status") == "mismatched":
        sys.exit(1)


if __name__ == "__main__":
    main()
