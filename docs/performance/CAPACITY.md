# Finite capacity and history retention

These are **opt-in native workload6 observations**, not safe-capacity discovery.
`capacity-study-v1` prospectively lists every cell, required response/quality and
resource condition, traffic/retention/wall allowance, stopping rule and OOM
response. No request is added after seeing an outcome. There is no upward search,
retry, replacement, cache reset/flush, restart, flag change or model-name inference.
The operator owns serving setup, risk acceptance and recovery.

## Commands

```sh
grill-perf capacity preflight crates/grill-perf/examples/capacity-v1.json

grill-perf capacity run crates/grill-perf/examples/capacity-v1.json \
  --endpoint http://127.0.0.1:8000/v1/chat/completions --local-http \
  --model operator-selected-model --out results/finite-capacity

grill-perf capacity inspect results/finite-capacity
```

The first command is offline admission, not permission to send traffic. Replace
endpoint/model declarations and prepare a **new** output directory under an
existing parent. `run` uses the ordinary native collector, per-request deadlines,
sequence state, output checks and evidence replay. It stops at the first
unsuccessful cell or its global cooperative wall deadline. Metrics2 scraping has
its own existing finite budgets; a scrape/publication/kernel read can overshoot a
userspace deadline, but cannot authorize the next serving request. No hard-real-
time wall limit is claimed.

`inspect` makes no network or device calls and replays the native cells. Exit 0
from inspection means evidence was read, not a capacity verdict. Collection exits
0 only if all listed cells succeeded, otherwise 2; admission/storage errors use 1.

The report includes every planned cell and request count, dispatched failures,
undispatched remainders, correctness requirements, resource summaries and native
run references. Explicit JSON `error.code: "out_of_memory"` is retained as
**provider-reported OOM**, not a verified device allocation diagnosis. Other
service/transport/deadline failures retain their actual status and raw response;
unknown causes stay unknown. A started cell without settlement is unsuccessful/
unknown, never automatically repeated. A forced crash may leave only its admission
marker and partial native evidence.
Dispatched/undispatched counts are null when corrupted or reserved-unsettled
native evidence cannot establish the count; unknown is never reported as zero.

`largest_successful_tested_cell` selects the last successful member of the
prospectively ordered, componentwise nondecreasing context/load list. It never
invents an untested cell or extrapolates beyond the list. Equal context/load cells
remain distinct declared observations. The context token ceiling is an operator
bound, checked against available provider prompt usage; it is **not** measured
context occupancy or evidence that the prompt filled that context. Missing required
usage withholds success. Provider counts remain unauthenticated.
`observed_prompt_tokens` separately retains the minimum/maximum actually reported
counts, with `prompt_usage_observed_requests` against the fixed request population.

`quality: all_sequence_checks` requires every actual workload6 parent/control/
response check, including warmups. `protocol_only` explicitly checks response and
usage eligibility, not semantic correctness, and makes no semantic capacity claim.
`resources.kind: not_required_no_resource_claim` explicitly has no resource/headroom
requirement. `resources.kind: limits` requires attached, complete resource coverage
and every listed `{summary,maximum}` absolute bound. The index selects a prospective
resource summary; units are exactly that summary's units. Unsupported/missing
statistics fail the condition, not zero or a favorable throughput fallback.

The finite ceilings are 16 cells, 10,000 requests total, 1 billion requested output
tokens, one hour declared wall allowance and 16 GiB retained file payload allowance.
Admission charges all native wave reservation/receipt/response/trace ceilings,
metrics and attached resources, not just successful responses. These are serialized
payload protection bounds, **not RSS guarantees, reserved disk space, model KV
capacity or universal production-safe limits**.

## Complete-acquisition resources

Workload6 may carry `resources: {version:1,mode:"review_only",observer,summaries,
retained_bytes}`. Earlier workload versions reject this field. All fields of the
existing closed `ResourcesConfig` and resource source contracts are unchanged;
[resource semantics and limitations](RESOURCES.md) still apply. The finite
`resource-acquisitions-v6.json` example uses an intentionally explicit placeholder
PID, which must be replaced with the caller-selected ordinary process. No process
or device is discovered automatically.

One caller-owned observer uses the **same capture `Instant` and plan clock ID** as
the existing workload6 step clocks. It samples before acquisition start, at admitted
cadence during collection, and after settlement, then drains before the run receipt.
The measured interval is the first required step's start through the last required
step's settlement, including required prime/control steps and inter-step work.
It is not a sum of disjoint wave windows. Warmups retain their own complete group.
Inter-step synchronous work can delay sampling; max-gap checks conservatively make
incomplete exposure unavailable instead of moving boundaries. Setup and publication
are outside the measured span where the original step-clock contract puts them.

Raw bytes, source/incarnation identities, sample/read windows, cancellation,
failures, cadence/budgets and read overhead live once in `resources-NNNNNN.json`
per acquisition. Step receipts do not copy the full raw observations or histories.
Replay verifies ordered required membership, each published step clock, exact full
span, raw parsing, provenance and source settings; absent sidecars are unavailable.
`retained_bytes` must cover the 64 MiB artifact ceiling for **every** acquisition,
up to 1 GiB total. Each observer retains only its existing bounded raw samples;
reports retain summaries, not copies of those samples.

These serving attachments are deliberately **review-only**. Native throughput
policies reject them rather than silently PASS a resource claim; descriptive
`compare` exposes resources and exits 2. Capacity can separately require absolute
resource limits in its prospective plan. CPU cumulative-counter integration still
requires actual retained samples exactly at both measurement boundaries. Typical
serving boundaries are not sample times, so `unsupported_boundary` remains honest:
there is no interpolation, invented boundary sample or renamed CPU measurement.
Sampled maxima are not true peaks; sampled energy is a labelled trapezoidal estimate;
shared source activity is not attributed solely to the model. No host/device/rank
memory sums or safe-headroom inference are made.

## Retention phases and actual eviction

A capacity cell can declare `retention` with `prime`, ordered nonempty `pressure`,
`probe`, ordered nonempty `recovery`, `intended_pressure`, condition, exact source /
producer pin, exact `metrics_labels`, and finite journal byte/event/window bounds.
These IDs must cover the native conversation steps **exactly in order**. Pressure
uses separate root histories; probes and recovery use actual acquired parents in
the prime history. The existing collector remains the only sequence runner; retained
assistant/tool outputs, parent IDs and acquisition-salted history namespace are not
substituted with fixture answers.

`observe_only` reports observations but never satisfies a retention qualification.
`eviction_and_recovery` requires complete correctness, actual target-key removal and
a later source-reported recovery hit. `retained_hit` requires a probe hit without an
observed target-key eviction. A miss does not imply eviction, and a hit despite
intended pressure is preserved rather than called a failed pressure generator.
Actual allocation requests/free/total blocks are reported separately from the
operator's intended pressure. A block removal does not prove whole-history loss
or that it caused a subsequent miss.

Metrics2 retains and replays its actual exporter epochs, full-label identities and
accounting. The retention gate requires the exact selected labels' prefix
queries/hits, prompt total/cached and all three prompt sources to remain continuous,
plus both existing prompt-source conservation checks. Missing or reset counters,
missing epochs and inconsistent accounting withhold qualification. Unrelated draft
accounting is not relabelled as cache evidence. Server-wide accounting remains
server-wide, not a per-request attribution guarantee. Preemption is never eviction.

### Concrete backend journal hook

`tools/retention-journal.py` supplies a minimal **operator-invoked backend hook**,
not a plugin registry, server launcher or frontend log importer. Its production
entrypoint is `install_vllm(journal)`, called in the one owning engine process
before any pool activity; importing the script itself uses only Python's standard
library. The operator loads this exact file explicitly and constructs:

```python
journal = retention_journal.Journal(
    "/operator/chosen/fresh-events.jsonl",
    source=retention_journal.SOURCE,
    max_events=10000,
    max_bytes=8388608,
    max_window_us=300000000,
)
retention_journal.install_vllm(journal)
```

This snippet is an integration call in an **already operator-owned engine startup
path**, not an instruction to restart or patch a deployed service. The CLI never
installs the hook, imports vLLM, launches a producer, attaches automatically, sets
environment/serving flags, or resets a cache. Multiple pools/processes, source changes,
partial-cache promotion/movement, explicit invalidation/reset, unbounded identities
and recording exhaustion deliberately withhold continuity. It neither supports nor
claims distributed-pool aggregation or external/persisted KV eviction.

Supported public source contract: vLLM **v0.27.0**, exact file SHA-256:

| Public source | SHA-256 |
|---|---|
| [block_pool.py](https://github.com/vllm-project/vllm/blob/v0.27.0/vllm/v1/core/block_pool.py) | `51cad2fd425128a0ff433ca4685acfc022e40659153ea5d7180fc586e1eebb6c` |
| [kv_cache_manager.py](https://github.com/vllm-project/vllm/blob/v0.27.0/vllm/v1/core/kv_cache_manager.py) | `70f7f608c0963af155540630a5633483e5c19c0c0280dfd621760d5db2a419a6` |
| [request.py](https://github.com/vllm-project/vllm/blob/v0.27.0/vllm/v1/request.py) | `6085b0668f41d56cd81ef483c06456f4ebe88bc1d46c8131d7134182d7893012` |

The hook observes actual `KVCacheManager.get_computed_blocks`, actual
`BlockPool._insert_block_hash` under `cache_full_blocks(request,...)`, and actual
removed hashes returned by `_remove_cached_block_hashes` under allocation-driven
`get_new_blocks` / `_maybe_evict_cached_block`. It records key loss only when no
alternative block remains under that key. Promotion/movement or explicit eviction
is not falsely called allocation-pressure eviction. It observes real `cache_salt`
and opaque hashed backend keys; no private token IDs or model weights are exported.
Existing source files are not copied into the project. Other versions/recipe patches
are unsupported until reviewed with a new contract, never accepted by overwriting
a pin with an installed hash.

The caller supplies the existing regular nonsymlink journal with
`capacity run ... --eviction-journal PATH --metrics-url URL --metrics-isolation ...`.
The collector holds its file descriptor and retains before/after prefix continuity,
source sequence/incarnation/clock, source/producer pins, failures and all bounded raw
JSONL. Journal time has its own origin: it is not subtracted from serving timestamps.
Exact ordered lookup salts link to actual workload6 histories. Duplicate/missing
requests, missing insertions, unsupported key transitions, incomplete event lines,
changed source or budget exhaustion withhold eviction claims. Maxima/eviction are
never inferred merely from a configured pool or a declaration. Journals are local
unauthenticated observations, not hostile-writer-proof execution attestations.

The existing startup ASGI/AsyncLLM bridge is a different frontend source. It does
not provide backend eviction or persisted-KV evidence and is not used as a substitute.

## Controlled CPU fixtures and qualification boundary

The ordinary fixture implements a real finite LRU dictionary of blocks under the
same insertion/allocation/removal/lookup hook interface, plus actual HTTP response
collection and metrics2 conservation counters. Its replies use the actual retained
parent answer. A one-block cache evicts under a separately declared pressure root;
an eight-block cache retains it; a deliberate read-bypass produces a miss without
removing the retained key. Its finite allocator failure is a controlled fixture
failure, not a GPU OOM. Source identity is `ordinary-lru-fixture-v1`, never vLLM.

Main's later CPU commands, not live endpoint qualification:

```sh
cargo test -p grill-perf --locked --test cli serving_resource_tests -- --test-threads=1
cargo build -p grill-perf --locked
python3 tools/smoke-capacity.py --binary target/debug/grill-perf --out /tmp/grill-capacity-fresh
```

The tests cover complete warmup/measured span alignment, raw replay, unsupported
CPU boundaries, cooperative cancellation and old-version rejection. The smoke
owns only its finite ordinary fixture processes and verifies a successful then
failed then undispatched cell, actual eviction/recovery, retained hit despite
intended pressure, missing counters and a non-eviction miss. It preserves every
command/stdout/stderr/native artifact and never replaces a failed attempt.

Authored fixtures and source-pin review alone do not qualify a real adapter,
installed runtime, GPU, serving setup, OOM recovery or production capacity. Real
runtime source compatibility, separate approved live execution, independent review
and safe operational recovery remain separate requirements. Evidence directories
contain exact requests/responses and source selections; they are **not automatically
public-safe exports**. Review local artifacts before sharing.
