# Kernel and collective evidence protocol (#54)

HTTP output speed cannot prove that an isolated kernel or collective changed.
`grill-perf microbench` keeps those observations in their own closed artifact
family so the distinction is enforced by the tool rather than by a report
author's memory.

Three states are separate and never silently upgraded:

| State | Meaning |
|---|---|
| `declared` | An operator statement with no measurement bytes. A declared artifact is rejected as measurement evidence. |
| `imported` | Bytes this collector parsed but did not produce. Importing always records `imported`, even when the payload claims native execution; the payload's own claim is retained in `submitted_provenance` for audit. |
| `native_observed` | Bytes produced by a program this collector launched, or by the in-process CPU reference. This records *observation*, never authenticated execution. |

A kernel or collective result never sets a serving-speed claim. That needs a
separately linked serving acquisition.

## Commands

```
grill-perf microbench capture --adapter <id> --plan PLAN --out DIR [--program PATH] [--authorize-device-window] [-- ARGS...]
grill-perf microbench import --artifact FILE --plan PLAN --out DIR
grill-perf microbench inspect PATH
grill-perf microbench compare BASELINE CANDIDATE [--reference A2]
```

`capture` writes `plan.json`, `artifact.json` and `receipt.json` into a fresh
`DIR`; an existing directory is never overwritten. `import` writes the same
three files with `provenance: imported`, so retained producer bytes can be
compared offline without ever being labelled as exercised.

Exit codes follow the existing observed-envelope order: `0` PASS, `1` ERROR
(invalid, corrupt or identity-drifted evidence), `2` INCONCLUSIVE (retained but
unavailable), `3` REGRESSION. `compare` uses the shared `envelope.rs` exact
rational arithmetic — no second statistical engine.

## Closed adapters and frozen cells

| Adapter | Scope | Where it executes | Frozen cell |
|---|---|---|---|
| `cpu-sum-u64-reference` | CPU | This binary, in process | `sum_u64` over `0..4096`, exact reference `n*(n-1)/2`, warmups 2, measured iterations 5 |
| `exl3-e3-grouped` | Device | Operator program, explicitly launched | `apply_exl3_experts`, hidden 4096, intermediate 1024, experts 288, topk 8, tokens 1024, layer_seed 0, routing_seed 1, parity routing seed 3, activation seed 3, skew 1.0, tier E3 grouped, cap 32, warmups 2, measured iterations 5 |
| `nccl-allreduce-sum` | Ranks | Operator program under `torchrun` | `all_reduce` SUM, `int64`, numel 262144, declared world and ranks 0..world-1, warmups 1, measured iterations 5 |

The frozen cells are checked in code, not only documented: a plan that changes
the shape, cap, skew, tier, dtype, routing, clock or bit-width is rejected with
`IdentityDrift` before anything runs. There is no shape, cap, skew, tier or
world search, no best-median summary, no `torch.cuda.empty_cache()`, no remote
discovery and no script-hook framework. A plan names one adapter; the device
adapters additionally name one explicit `--program` whose SHA-256 the collector
hashes and checks against `plan.program_sha256`.

Device and collective capture is behind `--authorize-device-window`. Without
it, `capture` refuses before creating any output directory. Under the current
CPU-only authorization only the in-process CPU reference is exercised natively;
the device adapters are delivered callable code and remain unexercised.

## Pinned public sources

The two device adapters bind to reviewed public sources. The hashes below were
recorded before implementation and are re-verified at run time; a mismatch
aborts before any device work. Plans ship with the same pins and the collector
requires `artifact.sources == plan.sources` exactly.

```
MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks @ f906ee990596486e10ddbe381efa6f0e496f77e3
  tests/bench_e3_microbench.py   f57e4fa6726b0110c44ce56b0b0968ba425e566ace6d89f2db3872aae13889ec
  tests/test_exl3_overlay.py     8777254d47cdbb89186ba3bd1ac2437cbebc7987e0791d36c68e7bccb3df2b35
NVIDIA/nccl-tests doc/PERFORMANCE.md
    2242493053a6f9d5db1c6542a8ce7740be919778645125b94eeae440efd31edf
```

`exl3_e3_grouped.py` imports `make_layer`, `routing`, `time_fn` and `set_cap`
from the pinned anchor and `_err_stats` / `_assert_e3_within` from the pinned
tolerance file; it never calls the anchor's `main()`. `nccl_allreduce_sum.py`
uses the pinned byte definitions for its reported bandwidth labels.

A declared source path that does not exist on the producing host stays a
declaration retained verbatim, never silently dropped or rewritten. The one pin
the collector verifies directly is `program_sha256`, because the collector
hashes the program it launched.

## Correctness is not optional

`exl3-e3-grouped` runs the production-geometry parity check *before* any timing
with the frozen tolerances (`E3_TOL_FACTOR = 1.5`, `E3_TOL_ABS_REL = 1e-3`,
`E3_TOL_NRMSE_ABS = 1e-4`, coarse bound `max(0.15, 0.08 * max(1.0, ref_max))`).
The collector recomputes every bound exactly from the retained `ref_max` and E2
statistics and compares the E3 statistics against them, so a widened tolerance
or an out-of-contract tolerance value is an identity error rather than a
tolerated difference. Non-finite statistics fail closed.

The effective tier and the last observed fallback must both be `grouped`; a
silent lower-tier fallback is withheld (`TierFallback`), never timed as E3.
Numerical work uses synthetic Zipf routing, labelled synthetic in the operation
identity — it is not the served routing distribution.

`nccl-allreduce-sum` verifies exact integer equality of the complete tensor
against `world*(i%251) + world*(world-1)/2`; the reduction reference has no
tolerance. Faster timing can never outweigh a correctness failure: a failed
parity or a nonzero mismatch count makes the acquisition ineligible and the
comparison ERROR or INCONCLUSIVE, never PASS.

## Clocks, units and samples

Every artifact carries exactly one clock identity, kind, units, resolution and
synchronization contract, and the plan must declare the identical contract:

| Adapter | Clock | Units | Resolution | Synchronization |
|---|---|---|---|---|
| `cpu-sum-u64-reference` | `process-monotonic-ns` | nanoseconds | 1 ns | read before and after the operation |
| `exl3-e3-grouped` | `cuda-event-ms` | milliseconds | 1000 ns | CUDA events with a device synchronize before each read |
| `nccl-allreduce-sum` | `rank-monotonic-ns` | nanoseconds | 1 ns | completed collective barrier before and after the timed collective |

Each sample retains the exact decimal text produced by the adapter timer and
the exact nanosecond integer converted from that text. The collector re-derives
the integer and rejects any mismatch, any decimal finer than one nanosecond,
and any value with digits finer than the declared resolution. Sub-microsecond
device-event digits are therefore not representable and are never invented.

Durations from different clock identities are never subtracted. Rank scope
records only per-rank durations; one completed world repetition is one sample,
and the acquisition statistic is the exact mean of the complete per-repetition
maxima. Rank absolute timestamps are never compared across ranks. The
collective duration is a synchronized host-monotonic completion latency, not
device-only kernel time, and is labelled as such.

## Bytes, bandwidth and the corrected definitions

Algorithmic payload bytes are `numel * sizeof(dtype)`; for the frozen
collective that is `262144 * 8 = 2097152` bytes. Algorithm bandwidth is
`payload_bytes / duration`, an exact rational.

`2*(world-1)/world` is the nccl-tests **bus bandwidth** normalization for
all-reduce. It is retained and labelled separately
(`bus_scope: "nccl-tests all-reduce bus-bandwidth normalization; not measured
physical link traffic"`) and is never described as measured link traffic, nor
substituted for the algorithm payload. `microbench inspect` prints both.

## Work, memory and time allowances

Every plan declares finite `work_units`, `memory_bytes` and `deadline_ms`
before execution. The collector rejects a plan whose declared allowance cannot
cover the frozen cell, and withholds evidence whose observed work or memory
exceeds the declaration. An operator program is killed at the declared Grill-side
deadline; the partial run is retained as a failed receipt with no artifact, so a
timeout can never masquerade as an empty successful measurement.

## Comparison contract

`compare` requires three distinct, strictly ordered acquisition identities
`A < B < A2`. Every pin — adapter, program hash, sources, operation, clock,
warmups, iterations, allowance, correctness contract and thresholds — must
match across roles; the only admitted difference axis is `revision`, which is
reported in the output. Distinct ids, strictly increasing declared starts and
non-duplicated evidence are enforced.

The statistic is the exact arithmetic mean per acquisition. The reference range
spans A and A2; the candidate range is the candidate acquisition. The shared
envelope assessment then applies the plan's `adverse_bps` and `spread_bps`.
These are finite empirical statistics of one declared cell, not precision,
confidence, IID or production-percentile guarantees, and not universal capacity
or adoption verdicts.

## CPU reference smoke plan

Authoritative CPU exercise, to be run by the integrator on a stable head that
includes this slice (never during concurrent writers):

```sh
cd <checkout>
cargo test -p grill-perf --locked --bin grill-perf microbench_model -- --test-threads=1
cargo test -p grill-perf --locked --test cli microbench -- --test-threads=1

# Real native CPU reference capture, inspection and comparison:
TMP=$(mktemp -d)
cp crates/grill-perf/examples/microbench-cpu-sum-u64.json "$TMP/plan.json"
grill-perf microbench capture --adapter cpu-sum-u64-reference \
  --plan "$TMP/plan.json" --out "$TMP/capture" --json
grill-perf microbench inspect "$TMP/capture" --json
```

The capture runs a real `sum_u64` reduction over `0..4096` with warmups 2 and
five measured iterations, records exact nanosecond samples on the process
monotonic clock, and validates the exact `n*(n-1)/2` reference. To compare, run
`capture` three times with distinct `acquisition.id`/`started_unix_ms` values
and pass the directories to `microbench compare` as A, B and `--reference A2`.
Copy the example plan and change only those two fields and `revision`.

Device adapters are *not* exercised by this plan. A device window additionally
requires the pinned checkout paths, `--authorize-device-window`, and the
operator's own launch placement:

```sh
grill-perf microbench capture --adapter exl3-e3-grouped \
  --plan crates/grill-perf/examples/microbench-exl3-e3-grouped.json \
  --program tools/microbench/exl3_e3_grouped.py \
  --authorize-device-window \
  -- --checkout /path/to/GLM-5.3-Flash-EXL3-2x-DGX-Sparks

grill-perf microbench capture --adapter nccl-allreduce-sum \
  --plan crates/grill-perf/examples/microbench-nccl-allreduce-sum.json \
  --program torchrun --authorize-device-window \
  -- --nproc-per-node=2 --standalone tools/microbench/nccl_allreduce_sum.py --stdout
```

The pinned example plans carry the shipped adapter hashes; editing an adapter
without re-pinning its plan is rejected by design.

## What this protocol deliberately does not claim

* Device, fabric or collective exercise. The adapters are delivered callable
  and unexercised; every device artifact is `imported` or a synthetic CPU
  fixture until a separately authorized window produces real samples.
* Serving-speed impact. A kernel or collective PASS is a measurement-layer
  observation only and needs a separately linked serving acquisition.
* Precision, confidence, IID, production percentiles, capacity or adoption.
* Authenticated execution. `native_observed` means this collector launched a
  program and retained its bytes within a bounded deadline; it cannot attest
  what that program actually did.
