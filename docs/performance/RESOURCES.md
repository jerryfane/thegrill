# Bounded host resource observations

`grill-perf resource` is a separate version-1 domain evidence path. It does not
change serving plan/workload meanings or collect serving requests. This source
slice implements an ordinary Linux host producer, retained raw replay, finite
import/inspect, and prospective A/B/A2 resource comparisons. It does **not**
implement a capacity runner, a retention-pressure runner, GPU telemetry execution,
or a claim of real-adapter/live qualification.

## Commands

All resource commands print JSON. Capture observes only explicitly declared
sources; it never launches a process, searches for a PID, discovers a cgroup,
reads process command lines/environment, or calls a serving endpoint.

```text
grill-perf resource capture --plan STUDY.json --role a --phase warmup --index 0 --out NEW_CAPTURE
grill-perf resource capture --plan STUDY.json --role a --phase measured --index 0 --out NEW_CAPTURE
grill-perf resource inspect CAPTURE --plan STUDY.json
grill-perf resource import CAPTURE_JSON --out NEW_IMPORTED_CAPTURE
grill-perf resource compare --plan STUDY.json --a A_WARMUP A_0 A_1 A_2 --b B_WARMUP B_0 B_1 B_2 --a2 A2_WARMUP A2_0 A2_1 A2_2
```

The plan declares exactly three ordered arms (`a`, `b`, `a2`), a fixed duration,
a source configuration for each arm, unique ordered warmup and measured IDs,
collector/exposure/deployment SHA-256 pins, a named candidate change, and every
gate. Each arm requires 1–20 warmups and 3–1000 measured acquisitions; counts and
source contracts must match. A/A2 deployment pins must match. All declared
acquisitions remain required, even when cancelled or failed. There is no retry,
replacement, upward search, selected best sample, or automatic plan relaxation.

`collector_sha256` is the actual executable SHA-256; capture checks it before
sampling. `exposure_pin` binds the operator's work declaration, not authenticated
execution of that work. Process PIDs and cgroup paths are arm-specific selections;
logical source IDs, ownership, other source/adapter fields, and budgets must
match. The current comparison requires one identical collector binary throughout;
collector/adapter implementation changes are not an admitted candidate axis.

Inspection without `--plan` verifies raw evidence but does not infer requested
gates. With `--plan`, it recomputes each declared summary. Capture exit 0 means
its requested summaries are available, not comparison PASS. Exit 2 retains
nonqualifying observations. Comparison exits reuse the policy contract:
PASS=0, ERROR=1, INCONCLUSIVE=2, REGRESSION=3, with
ERROR > REGRESSION > INCONCLUSIVE > PASS aggregate precedence. Import/inspect
exit 0 means valid storage/replay, not source completeness or native exercise.

## Explicit sources and finite budgets

`ResourcesConfig` has `version: 1`, `sources`, `cadence_us`, `max_gap_us`,
`max_read_us`, `max_samples`, `deadline_us`, `per_sample_bytes`, and
`raw_total_bytes`. All fields are required. There are at most 16 selected sources,
10,000 total source snapshots, 1 MiB raw bytes per sampling cycle, 8 MiB raw bytes
over the observer, and a one-hour observer deadline. Cadence is 1 ms–10 s; the
maximum allowed observed gap is explicit, at most 60 s. Byte ceilings bound
retained bytes, not total RSS. JSON artifacts have a separate 64 MiB ceiling.
A missing, permission-denied, truncated, malformed, changed, or late source stays
visible; exhausted budgets never become complete exposure.

A neutral selected-process source looks like this (the PID is supplied explicitly
by the owner of the ordinary fixture, not discovered by Grill):

```json
{"id":"fixture-process","target":{"kind":"process","pid":4242},"ownership":{"kind":"process_address_space"}}
```

Supported native sources:

| Target | Exact files | Semantics |
|---|---|---|
| `process {pid}` | `/proc/PID/stat`, `status`, then `stat` again | process CPU ticks, RSS, lifetime RSS high-water mark |
| `cgroup_v2 {path}` | explicit cgroup-v2 `memory.current`, `memory.peak`, `memory.max`, `cpu.stat` | used/peak/limit bytes, cumulative CPU microseconds |
| `host_cpu` | `/proc/stat` | aggregate **busy** CPU ticks, host-wide shared scope |
| `host_memory` | `/proc/meminfo` | MemFree bytes and MemTotal capacity, descriptive, not model-owned memory |

Native source directories are held open, filesystem type is checked, and fixed
source files use bounded nonblocking/no-follow reads rather than immutable-file
size assumptions. Cgroup paths stay under `/sys/fs/cgroup`; directory inode/device
identity detects replacement. Proc stat is split after the **final** closing
parenthesis, and PID/starttime is checked both within a sampling cycle and across
cycles. Child CPU is excluded. Process `utime` already includes guest time, so
it is counted once; host guest/guest_nice fields are not added again. Host idle
and iowait are not busy CPU. CLK_TCK is observed via sysconf, never hardcoded.
Cgroup `max`, absent files, finite limits, lifetime peaks, and sampled maxima are
separate values. Native process names present in raw stat/status remain private.

Every snapshot identifies its source, clock, read start/end offsets, incarnation,
typed metric/unit/value, raw bytes, and failure. The retained read interval is
collector overhead, not atomic-source-read precision. Native observations are
unauthenticated observations; ownership labels do not prove model attribution.
Host and cgroup measurements can include unrelated activity. No metric is summed
across ranks, devices, host/device unified memory, or shared ownership groups.

Byte/count/deadline admission is finite, but userspace cannot preempt a stalled
kernel filesystem read. Deadline and cancellation checks occur between reads;
an overlong read is retained as a coverage failure, not a hard-real-time claim.

## Clocks, exposure, and exact summaries

Offsets are integer microseconds on an identified monotonic origin. Native
resolution is observed with `clock_getres` and floored to the retained 1 us
quantization. Acquisitions may have different origins, but comparisons require
the same clock kind, unit, resolution, and synchronization contract. Unix
provenance timestamps establish declared A/B/A2 ordering only; they never enter
a duration or integral.

The ordinary-process command samples once before its requested interval and
continues for the prospectively fixed duration. It retains actual boundaries;
clock wake/read overshoot is not relabelled as exact requested time. Comparison
requires actual duration in `[duration_us, duration_us + max_gap_us]`. This is a
prospectively bounded observation-window comparison, not exactly equal CPU work.

Supported lower-is-better gates:

* `sampled_maximum`: RSS, used/allocated/reserved memory, KV used bytes, or
  instantaneous power. Only in-interval samples enter the maximum, with complete
  bracketing/cadence coverage required. It is not a true peak. Lifetime peaks,
  free memory, limits, and capacity remain descriptive, not these maxima gates.
* `cpu_time`: difference of cumulative CPU counters, converted exactly to CPU
  microseconds. Both measured boundaries must equal actual source observation
  offsets. Unaligned boundaries are `unsupported_boundary`, not interpolated CPU.
* `cpu_utilization_one_cpu`: CPU time / measured wall duration as a rational
  logical-CPU ratio, allowed to exceed one. No machine-core denominator or clamp.
* `sampled_energy_estimate`: explicitly labelled piecewise-linear trapezoidal
  integral of compatible complete microwatt observations, in microjoules.
  Segments are clipped to the measured interval with exact checked rational
  arithmetic. A single power sample cannot establish energy, and this is not
  directly measured physical energy. This authorization supports imported power
  evidence only, not a native GPU/provider producer.

Missing endpoints, cadence gaps, permission failures, cancellation, budget
exhaustion, resets, source/clock changes, and unknown ownership prevent favorable
qualification. Rational arithmetic overflow is ERROR, not rounding or saturation.
Full per-acquisition samples feed the existing exact envelope implementation;
all required warmups must qualify, and none enters measured counts. There are no
confidence, causality, or universal-capacity claims.

## Shared serving observer API (Main integration)

The final API in `resources.rs` is:

```text
host_clock(id: String) -> Result<Clock>
Observer::start(config: ResourcesConfig, origin: std::time::Instant, clock: Clock) -> Result<Observer>
Observer::sample(&mut self) -> Result<bool>
Observer::last_observed_us(&self) -> Option<u64>
Observer::finish(self, measured: MeasuredInterval, cancelled: bool) -> Result<Observation>
validate_observation(&Observation) -> Result<()>
summarize(&Observation, &Gate) -> Summary
```

Main supplies the **existing capture-scoped monotonic Instant**, its matching
clock identity, and cadence calls in the existing acquisition lifecycle. The
observer has no thread, timer, endpoint client, request sender, or independent
serving collector. `sample` returns false after budget exhaustion and preserves
the reason. `finish` takes an explicit `{clock, started_us, settled_us}` measured
interval: first required step start through last required step settlement,
including prime/control/inter-step work, not disjoint measured-step windows.
Observer setup may precede it and final observation may follow it. Caller-supplied
`cancelled` retains cancellation. Main retains the returned observation alongside
its final workload6 acquisition types and preserves per-step coverage separately.

The conservative CPU endpoint gate can be unavailable for serving acquisitions
whose required-step boundaries do not coincide with actual CPU sample offsets;
the API does not fabricate those missing counter boundaries. Source comparisons
must retain that limitation. Serving acquisition attachment/wiring is owned by
Main and is not supplied by a second runner here.

## Import, privacy, and replay

`resource import` reads one bounded `Capture` JSON and **always** changes outer
provenance to `imported`, even if input claims `native_observed`. Original bytes
are retained as `imported.json`; the projection is independently checked on load.
Native captures retain exact `study.json`, `observation.json` (including selected
raw bytes), and the digest-bearing `resource.json` receipt. Files are exclusively
created mode 0600 inside fresh 0700 directories using the existing evidence
helpers. Native destination admission precedes source reads; unfinished output
has no authoritative receipt. Raw sampling state is bounded in memory until
publication; abrupt process/host loss before publication can leave only the
prospective study, not recoverable sampled data. Cooperative cancellation does
publish the acquired evidence.

Imported target identity is explicit:
`imported {adapter, device, rank}` plus an ownership declaration and source
incarnation. Each snapshot retains `readings.json` bytes containing the closed
`Reading` list. Replay reparses those bytes and rejects unit/metric disagreement.
Device/provider values are **imported declarations**, not authenticated adapter
execution. Imported arithmetic can yield an explicitly imported comparison but
cannot establish real-adapter exercise or live qualification. Full evidence and
plans are private replay inputs, not public-safe exports.

## Later CPU verification, not executed during concurrent authoring

After Main approves a stable integration head, run:

```sh
cargo test -p grill-perf --locked resources::tests
cargo build -p grill-perf --locked
python3 tools/smoke-resources.py target/debug/grill-perf --out /tmp/grill-resource-smoke-new
```

The smoke authors a prospectively pinned plan for its own finite 8 MiB ordinary
process, collects one warmup plus three measured acquisitions in each A/B/A2
role, exercises the actual native producer, replays every capture, compares,
and verifies import cannot upgrade provenance. It retains all failures and never
retries. It starts/terminates only its own test process; no serving process,
GPU runtime, metrics, health, model, or device access is involved. Unchanged
control variation or timing gaps may prevent PASS; inspect retained evidence
rather than replacing acquisitions. For outside-checkout verification, pass the
staged binary's absolute path and a fresh private output directory.

Fixtures additionally cover final-parenthesis parsing, guest exclusion, PID reuse,
resets, permission/byte/sample/deadline/cancellation gaps, clock and unit mismatch,
overflow, unknown/shared ownership, missing rank visibility, unlimited versus
missing limits, CPU endpoint alignment, clipped rational energy, exact tolerance
boundaries, incomplete A/B/A2 populations, raw corruption, and imported provenance.
They are authored source, not an executed verification claim.

## Remaining capacity and retention dependencies

No capacity or retention command is implemented by this independent host slice.
Capacity requires a finalized workload6 acquisition integration with finite
prospective cells, success/quality/resource conditions, explicit stop condition
and OOM response, retained failed cells, and operator-owned recovery. Retention
requires the same sequence runner, prime/pressure/probe/recovery cases, actual
acquired histories, and finalized #51 accounting/continuity evidence. A miss
alone is not eviction. Missing continuity cannot qualify eviction. Neither
feature can be replaced by host resource JSON or a small fixture claim; issue52
and broad coverage remain open until those dependencies, independent review,
CPU/native staging, real adapters, and separately authorized qualification finish.
