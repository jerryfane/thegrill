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

Every runtime pin inside an artifact (installed versions, loaded module hash,
device names, the parameters actually used, how many pinned sources were read
from bytes) is an unauthenticated observation of what the producing process saw.
No declared implementation pin is echoed back as observed kernel provenance. A
kernel or collective result never sets a serving-speed claim; that needs a
separately linked serving acquisition.

## Commands

```
grill-perf microbench capture --adapter <id> --plan PLAN --out DIR [--program PATH] [--authorize-device-window] [-- ARGS...]
grill-perf microbench import --artifact FILE --plan PLAN --out DIR
grill-perf microbench inspect PATH
grill-perf microbench compare --study STUDY --baseline DIR --candidate DIR --reference DIR
```

`capture` writes `plan.json`, `artifact.json` and `receipt.json` into a fresh
`DIR`; an existing directory is never overwritten. `import` writes the same
three files with `provenance: imported`, so retained producer bytes can be
compared offline without ever being labelled as exercised.

Exit codes follow the existing observed-envelope order: `0` PASS, `1` ERROR
(invalid, corrupt, drifted or incomplete-declaration evidence), `2`
INCONCLUSIVE (retained but unavailable), `3` REGRESSION. `compare` uses the
shared `envelope.rs` exact rational arithmetic — no second statistical engine
and no floating-point decision path.

## Closed adapters and frozen cells

| Adapter | Scope | Where it executes | Frozen cell |
|---|---|---|---|
| `cpu-sum-u64-reference` | CPU | This binary, in process | `sum_u64` over `0..4096`, exact reference `n*(n-1)/2`, warmups 2, measured iterations 5 |
| `exl3-e3-grouped` | Device | Operator program, explicitly launched | `apply_exl3_experts`, hidden 4096, intermediate 1024, experts 288, topk 8, tokens 1024, layer_seed 0, routing_seed 1, parity routing seed 3, activation seed 3, skew 1.0, tier E3 grouped, cap 32, warmups 2, measured iterations 5 |
| `nccl-allreduce-sum` | Ranks | Operator program under `torchrun` | `all_reduce` SUM, `int64`, numel 262144, declared world and ranks 0..world-1, warmups 1, measured iterations 5 |

The frozen cells are checked in code, not only documented: a plan that changes
the shape, cap, skew, tier, dtype, routing, clock, warmup count or bit-width is
rejected with `IdentityDrift` / `WarmupCountMismatch` / `IterationCountMismatch`
before anything runs. There is no shape, cap, skew, tier or world search, no
best-median summary, no `torch.cuda.empty_cache()`, no remote discovery and no
script-hook framework. A plan names one adapter; the device adapters
additionally name one explicit `--program` whose SHA-256 the collector hashes
and checks against `plan.program_sha256`.

Device and collective capture is behind `--authorize-device-window`. Without
it, `capture` refuses before creating any output directory. Under the current
CPU-only authorization only the in-process CPU reference is exercised natively;
the device adapters are delivered callable code and remain unexercised.

## Pinned public sources

The two device adapters bind to reviewed public sources. The hashes below were
recorded before implementation and are re-verified at run time from the bytes
the adapter reads; a mismatch aborts before any device work.

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

**Sources are observed, not echoed.** A plan pins `sources[].sha256` (paths are
declared hints). The artifact retains the sources the producer actually read
with the hashes it computed from those bytes; the collector requires the
observed hash set to equal the pinned hash set exactly, so a producer cannot
drop, add or substitute a pin, while a different filesystem path for the same
bytes is fine. Each adapter also reports how many pins it verified from bytes
(`sources-verified`) and how many it could only echo as declarations
(`sources-declared`). The E3 adapter requires every pin to be verified; the
collective adapter always verifies at least its own program and may declare a
reference document it cannot resolve on the producing host. The one pin the
collector verifies directly is `program_sha256`, because it hashes the program
it launched.

## Correctness is not optional

`exl3-e3-grouped` runs the production-geometry parity check *before* any timing
with the frozen tolerances (`E3_TOL_FACTOR = 1.5`, `E3_TOL_ABS_REL = 1e-3`,
`E3_TOL_NRMSE_ABS = 1e-4`, coarse bound `max(0.15, 0.08 * max(1.0, ref_max))`).
The collector recomputes every bound exactly from the retained `ref_max` and E2
statistics and compares the E3 statistics against them, so a widened tolerance
or an out-of-contract tolerance value is an identity error rather than a
tolerated difference. Non-finite statistics fail closed; the coarse bound is
strict, the others inclusive.

The effective tier and the last observed fallback must both be `grouped`, and
the reported `fallback-tier` observation must say so; a silent lower-tier
fallback is withheld (`TierFallback`) or rejected as drift, never timed as E3.
Numerical work uses synthetic Zipf routing, labelled synthetic in the operation
identity — it is not the served routing distribution.

`nccl-allreduce-sum` verifies exact integer equality of the complete tensor
against `world*(i%251) + world*(world-1)/2`; the reduction reference has no
tolerance. Faster timing can never outweigh a correctness failure: a failed
parity or a nonzero mismatch count makes the acquisition ineligible and the
comparison ERROR or INCONCLUSIVE, never PASS.

## Clocks, units and samples

Every artifact carries exactly one clock identity, kind, units, declared
resolution and synchronization contract, and the plan must declare the identical
contract:

| Adapter | Clock | Units | Declared resolution | Synchronization |
|---|---|---|---|---|
| `cpu-sum-u64-reference` | `process-monotonic-ns` | nanoseconds | 1 ns | read before and after the operation |
| `exl3-e3-grouped` | `cuda-event-ms` | milliseconds | 1000 ns | CUDA events with a device synchronize before each read |
| `nccl-allreduce-sum` | `rank-monotonic-ns` | nanoseconds | 1 ns | completed collective barrier before and after the timed collective |

Each sample retains two things:

* `raw` — the timer's own decimal representation, preserved verbatim. This
  includes scientific notation (`1.234e-05`) and any number of fractional
  digits the timer returned;
* `duration` — the exact rational nanosecond conversion of `raw`, reduced, with
  `{numerator, denominator}`. A fractional nanosecond is retained as a
  fraction, never rounded.

Concrete examples, all accepted:

| `raw` (ms) | `duration` (ns) |
|---|---|
| `31.234` | `31234000/1` |
| `31.2345678` | `156172839/5` |
| `0.123456789012345` | `24691357802469/200000000` |
| `1.234e-05` | `617/50` |

The declared clock resolution describes the advertised timer class. It **never**
rounds, quantizes or gates a retained value: a digit finer than the declared
resolution is preserved rather than rejected, and the exact decimal expansion
of a value a timer returned as a float is not a precision or accuracy claim.
Zero, negative, non-finite, over-long and non-representable values fail closed
(`DurationOutOfRange`, `SamplePrecision`, `SampleOverflow`).

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
exceeds the declaration. An operator program is killed at the declared
Grill-side deadline; the partial run is retained as a failed receipt with no
artifact, so a timeout can never masquerade as an empty successful measurement.

## Study shape: at least three complete acquisitions per role

A native domain comparison needs a **prospectively declared study**, written
before any acquisition runs:

```json
{
  "kind": "microbench-study-v1",
  "version": 1,
  "adapter": "cpu-sum-u64-reference",
  "revision_axis": { "baseline": "cpu-sum-u64-reference-v1", "candidate": "cpu-sum-u64-reference-v2" },
  "thresholds": { "adverse_bps": 500, "spread_bps": 2000 },
  "minimum_acquisitions": 3,
  "roles": {
    "baseline": ["baseline00", "baseline01", "baseline02"],
    "candidate": ["candidate00", "candidate01", "candidate02"],
    "reference": ["reference00", "reference01", "reference02"]
  },
  "started_unix_ms": 1757000000000
}
```

* `minimum_acquisitions` must be at least `3`, and every role must declare at
  least that many slots; each slot names one acquisition directory inside the
  role directory, and the slot id must equal that acquisition's plan
  `acquisition.id`.
* The declared `revision_axis` is the only permitted difference between the
  arms: baseline and reference members must carry `baseline`, candidate members
  `candidate`, and every other pin (adapter, program hash, source pin set,
  operation, clock, warmups, iterations, allowance, correctness contract,
  thresholds) must match across the entire study. A study whose two revisions
  are identical is an A/A control and is rejected.
* Declared starts must be strictly increasing inside each role, and every
  baseline start must precede every candidate start, which must precede every
  reference start. No acquisition may start before `started_unix_ms`.
* Every acquisition in a study must have been recorded by one collector
  identity; a mixed set is `CollectorMismatch`.

Each acquisition directory is exactly what `capture`/`import` produce
(`plan.json`, `artifact.json`, `receipt.json`), so a role directory looks like:

```
baseline/
  baseline00/{plan.json,artifact.json,receipt.json}
  baseline01/...
  baseline02/...
```

`compare` loads exactly the declared membership. A directory that is present but
not declared is an error (`UnexpectedAcquisition`); a declared slot that is
missing, failed, timed out, incomplete or unusable withholds the comparison
(`MissingAcquisition` plus the specific reason). **Missing or failed
acquisitions are never replaced, filtered or dropped, and there is no
one-acquisition-per-role shortcut.**

## Comparison contract

Within one acquisition the statistic is the exact arithmetic mean of the
measured samples, or for rank scope the exact mean of the complete
per-repetition maxima. Roles are then compared by statistic:

* the reference envelope pools the acquisition statistics of the **baseline and
  reference (A/A2) roles**;
* the candidate envelope is the candidate role's acquisition statistics;
* the shared envelope assessment applies the study's `adverse_bps` and
  `spread_bps` in exact rational arithmetic.

The result retains every acquisition statistic, both role revisions, the
per-role and pooled ranges, the adverse bounds, the uniform collector identity,
the evaluator binary digest, the study digest and every reason code. These are
finite empirical statistics of one declared cell, not precision, confidence,
IID or production-percentile guarantees, and not universal capacity or adoption
verdicts.

## CPU reference smoke plan

Authoritative CPU exercise, to be run by the integrator on a stable head that
includes this slice (never during concurrent writers):

```sh
cd <checkout>
cargo test -p grill-perf --locked --bin grill-perf microbench_model -- --test-threads=1
cargo test -p grill-perf --locked --test cli microbench -- --test-threads=1

TMP=$(mktemp -d)
# Nine real native CPU captures: three baseline, three candidate, three
# reference. Copy the example plan and change only acquisition.id,
# acquisition.started_unix_ms and (for the candidate arm) revision.
plan() { # $1=slot $2=started $3=revision
  sed -e "s/\"a01\"/\"$1\"/" -e "s/1757000000000/$2/" \
      -e "s/cpu-sum-u64-reference-v1/$3/" \
      crates/grill-perf/examples/microbench-cpu-sum-u64.json > "$TMP/$1.json"
}
mkdir -p "$TMP/baseline" "$TMP/candidate" "$TMP/reference"
for index in 00 01 02; do
  plan "baseline$index" $((1757000000000 + 10#$index * 1000)) cpu-sum-u64-reference-v1
  plan "candidate$index" $((1757000060000 + 10#$index * 1000)) cpu-sum-u64-reference-v2
  plan "reference$index" $((1757000120000 + 10#$index * 1000)) cpu-sum-u64-reference-v1
  for role in baseline candidate reference; do
    grill-perf microbench capture --adapter cpu-sum-u64-reference \
      --plan "$TMP/$role$index.json" --out "$TMP/$role/$role$index" --json
    grill-perf microbench inspect "$TMP/$role/$role$index" --json
  done
done
grill-perf microbench compare \
  --study crates/grill-perf/examples/microbench-study-cpu.json \
  --baseline "$TMP/baseline" --candidate "$TMP/candidate" --reference "$TMP/reference" --json
```

The study example must be edited to match the slot ids and start times used
above (its `started_unix_ms` must not be later than any capture). Each capture
runs a real `sum_u64` reduction over `0..4096` with warmups 2 and five measured
iterations, records exact nanosecond rational samples on the process monotonic
clock, and validates the exact `n*(n-1)/2` reference. The candidate arm should
differ only in `revision` for an A/A control, or in the declared code change
under test; the tool does not care which, it enforces that everything else
matches.

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
without re-pinning its plan is rejected by design, and each adapter refuses to
run when its own bytes do not match `plan.program_sha256`.

## What this protocol deliberately does not claim

* Device, fabric or collective exercise. The adapters are delivered callable
  and unexercised; every device artifact is `imported` or a synthetic CPU
  fixture until a separately authorized window produces real samples.
* Serving-speed impact. A kernel or collective PASS is a measurement-layer
  observation only and needs a separately linked serving acquisition.
* Precision, confidence, IID, production percentiles, capacity or adoption.
* Authenticated execution. `native_observed` means this collector launched a
  program and retained its bytes within a bounded deadline, and that the
  program's own observations were structurally consistent with the frozen cell;
  it cannot attest what that program actually did.
