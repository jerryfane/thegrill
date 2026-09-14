# Startup and operator-restart cache evidence

`grill-perf startup` is a separate finite domain workflow. It never starts,
stops, restarts, flushes or configures a service. Default `baseline`, `check`,
`run`, serving workload versions 1–5, their namespaces, and `lifecycle.rs`
capture ownership keep their existing meanings.

## Commands and execution boundary

```text
grill-perf startup observe PLAN --slot N --pid PID --events JOURNAL \
  --producer-source SOURCE --out NEW_DIRECTORY [--previous PREVIOUS_CAPTURE]
grill-perf startup store PLAN --slot N --pid PID --cache-pid CACHE_PID \
  --events JOURNAL --producer-source SOURCE --out STORE_DIRECTORY \
  [--previous PREVIOUS_CAPTURE]
# Operator/test harness restarts the engine, not the cache process.
grill-perf startup reload PLAN --slot N --pid NEW_PID --cache-pid CACHE_PID \
  --events NEW_JOURNAL --producer-source SOURCE --store STORE_DIRECTORY \
  --out RELOAD_DIRECTORY [--previous PREVIOUS_CAPTURE]
grill-perf startup import SOURCE_DIRECTORY --out NEW_DIRECTORY
grill-perf startup inspect CAPTURE_DIRECTORY
grill-perf startup compare PLAN ORDERED_PATH_MANIFEST
```

All output is JSON. `observe` only accepts a startup study; `store` and `reload`
only accept a restart-cache study. The path manifest is a JSON array of capture
directory paths in the plan's A/B/A2 slot order. `--previous` binds the preceding
whole acquisition's receipt, including failures; store and reload use the same
predecessor. The first slot has no predecessor. A store alone is not a completed
restart-cache acquisition and cannot enter comparison.

Use new output directories. Captures retain the exact `plan.json`, pinned source
bytes in `producer.bin`, raw `events.bin`, authoritative `capture.json`, and each
`request-NN/body.bin`, `response.bin`, and actual metrics2 companions when
selected. Reload embeds the verified store under `store/`; it is independently
replayable without the original store path. Existing evidence read/write,
non-symlink regular-file checks, exclusive private creation and publication
primitives are reused. Filesystem failure can leave an inspectable partial
output without an authoritative receipt; OS/filesystem stalls are not hard
real-time deadlines.

`import` always writes an imported wrapper and preserves the original receipt
as `imported-capture.json`, including on repeated import. Original acquisition
links remain checkable. A payload claiming native execution is still imported.
Native records are unauthenticated observations, **not cryptographic execution
attestations**. Possessing producer source with the correct hash does not prove
that an untrusted producer ran it.

## Neutral executable fixture

`tools/startup-fixture.py` supplies `neutral_journal_v1`, an explicit local-process
adapter. The **operator or test harness** invokes its `engine` and, for restart
studies, separate `cache` modes. The Rust CLI does not execute the script or
interpret input paths as commands. No source discovery or plugin registry exists.

The engine independently delays binding/listening, communication initialization,
readiness and inference, emits an append-only NDJSON journal, and exposes a
bounded ordinary loopback Chat Completions response and synthetic metrics.
The cache process really retains a computed synthetic answer across engine
restart; this is neither model KV nor an approximation of a model's performance.
The exact synthetic operation is the sum of integers zero through six, answer
`21`. Request response correctness is checked against the plan's exact answer,
not inferred from a model name, health string or listening socket.

Native attachment reads only the explicitly selected engine/cache PIDs and
journal. PID identity is `{pid,start_ticks,boot_id}`; parsing `/proc/PID/stat`
starts after its final closing parenthesis. Zombies, process exit, PID reuse,
journal inode replacement, rewritten prefixes, malformed lines, exhausted
budgets, failed responses and missing completion are retained as nonqualifying
observations. Polling is every 10 ms, with finite event count, raw bytes and a
whole observation deadline. There are no inference retries or replacement
acquisitions. Only the prospective request list is dispatched, after supported
readiness; all declared metrics snapshots are budgeted separately.

The script's `sealed` event means **inference admission has closed**, while
read-only metrics remain available. It is emitted after all accepted inference
requests have been recorded. Complete coverage is therefore a property of this
specific instrumented producer, not of arbitrary logs with similarly named
fields. Smoke suppression, shape-warmup suppression and stable hash control
must be source-observed; an operator attestation cannot replace them.

## Opt-in serving-runtime bridge

`tools/startup-runtime.py` implements `runtime_vllm_v1`. It is an explicit
**operator-invoked, in-process entrypoint**, not a command that Grill launches.
It calls the existing single-worker `api_server.run_server` with ASGI lifespan
enabled; there is no subprocess, restart, service manager or arbitrary-module
hook. The neutral producer above remains a separate CPU fixture.

Supported public source is byte-pinned, not selected by model name or image tag:

| Source | Revision | Required SHA-256 |
|---|---|---|
| [vLLM api_server.py](https://github.com/vllm-project/vllm/blob/v0.27.0/vllm/entrypoints/openai/api_server.py) | `v0.27.0` | `cd4b83e85dc9d5aae808348d336e59b3064bee697012750d23c786de082a1c53` |
| [vLLM launcher.py](https://github.com/vllm-project/vllm/blob/v0.27.0/vllm/entrypoints/launcher.py) | `v0.27.0` | `caf4c4517a62f05abe5998409de73af70e91d99dfab3819a4b359ba44af0da91` |
| [vLLM async_llm.py](https://github.com/vllm-project/vllm/blob/v0.27.0/vllm/v1/engine/async_llm.py) | `v0.27.0` | `81a0cae6d5da22140f509a59d6c6bb8fc6ee1572da2a2cd793b5830222d18bcc` |
| [Uvicorn server.py](https://github.com/encode/uvicorn/blob/0.34.0/uvicorn/server.py) | `0.34.0` | `8dd3d150523fd140a9981c41f0fae963b869b20c064dd94ab7422bc453748e6f` |

Pins are checked against bounded installed source files before serving imports
and again before sealing. Imported module locations must match those files.
The producer source retained as `producer.bin` binds these exact pins and the
`vllm-0.27.0-uvicorn-0.34.0-sha256-v1` event contract. This is native
**unauthenticated** source observation, not execution authentication or proof
that every dependency is unmodified. Different upstream/recipe-patched bytes
are unsupported; do not replace a pin with the local hash to make it pass.
Review a source-contract revision instead. No deployment-local source is copied.

### Launch, readiness and inference

The bridge records `runtime_start` at its journal entrypoint before importing
serving libraries. It is explicitly **not OS process launch**: interpreter,
bridge imports and initial pin checking precede this anchor. Inspection reports
`launch_anchor:"bridge-entrypoint-not-os-launch"`; milestone durations begin
there. Full OS exec-to-ready remains unavailable.

The outermost ASGI wrapper records `ready` only on
`lifespan.startup.complete` (`asgi-lifespan-startup-complete-v1`). In the pinned
vLLM entrypoint this follows engine-client construction and app-state
initialization. This is ASGI/application readiness, not independent proof of
model correctness. `listening` is emitted separately after the pinned
`uvicorn.Server.startup` returns with `server.started` and actual sockets whose
`SO_ACCEPTCONN` is one. Uvicorn completes lifespan **before** starting listeners;
ready can therefore precede listening. The collector waits for both. No
health string, log line, model name or merely bound socket substitutes.

`inference_complete` hashes the actual ASGI response entity after its final
body send. Favorable first-valid inference still requires the existing Rust
wire collector's complete valid response, exact expected answer, usage and
matching digest. Bootstrap/communication substage timing is not observed by
this bridge: a `communication` gate remains unavailable, not total startup
time relabelled as communication.

### Complete finite request exposure

Declare `runtime_window_us` in the plan (1 ms or more, strictly below
`deadline_us`). From entrypoint initialization through at least this duration
after ASGI ready, instrumentation records all mutating/unknown HTTP routes and
all websocket admissions, including requests without Grill headers. Only
GET/HEAD `/health`, `/metrics`, `/v1/models` bypass body observation. Unknown
routes are counted conservatively, not guessed to be harmless. A websocket is
recorded and rejected as unsupported; overlap closes admission and withholds
serial-order qualification instead of allocating unbounded concurrent bodies.

Grill adds `x-grill-startup-request` and `x-grill-startup-plan` for correlation;
it does not change the exact body or stable `cache_salt`. Unmarked requests
receive `external-N` IDs and cannot masquerade as first reload just by arriving
first. Headers remain unauthenticated correlations, not bearer credentials.
`AsyncLLM.add_request` is wrapped before engine initialization:
`runtime_engine_request` records actual admissions in the active ASGI context.
Internal/background calls without a live context produce a retained failure.
Missing engine admissions cannot be repaired by a 200 response alone.

At the prospective timer boundary the bridge closes **its inference admission**
and drains already admitted requests before emitting `runtime_coverage`, then
`sealed`. It does not stop the server; read-only metrics stay accessible.
Later inference attempts get 503 and cannot reach the wrapped app. This
behavior is opt-in and must be accepted when the operator selects the bridge.
It never seals just because the expected count has arrived: external requests
after the final Grill response but before the timer still invalidate ordering.
Timeout, changed source, partial drain, absent instrumentation or missing seal
withhold the claim. Byte/event caps and per-request deadlines fail closed.

The complete supported scope is **one frontend's ASGI and AsyncLLM admissions**.
Additional frontends, direct engine-core clients and backend/device graph/shape
warmup are not proven absent by these hooks. The bridge therefore does not emit
favorable smoke/shape suppression controls, and it cannot currently qualify
cross-restart KV persistence. `no_warming_attestation` cannot change that.

### Shared namespace identity and operator invocation

Run collector and producer in the **same PID and time namespaces**, with the
same `/proc` process view: for example, both inside the operator-owned serving
container, or both in a deliberately shared host namespace. `--pid` is the
actual producer PID in that namespace, not a container PID copied into a host
command. The collector checks `/proc/self/ns/{pid,time}` against
`/proc/PID/ns/{pid,time}`, saves both namespace identities, and requires
the runtime event's identity plus PID/start ticks/boot ID to agree.
Different namespaces fail closed. Automatic host/container PID translation,
namespace entry and container control are not implemented.

The observed `engine` process for this adapter is the API/frontend process,
not every backend worker. `--cache-pid` retains the kernel-observed identity of
the explicitly selected cache-server process, but this bridge does not infer
which cache process served a request.
Filesystem residency, compiled artifact/weight/prefix-offload identity and
temperature stay unknown. A matching frontend `PYTHONHASHSEED=0` is recorded
only as `frontend-only`; cache-server seed and KV transfer are unobserved.
Real response cached-token usage and selected metrics2 evidence are retained,
but they cannot identify external persisted KV by themselves.
`cache_result` is `unavailable`, never a synthetic answer-cache hit or an
inferred transfer. Store/reload remain callable and honestly nonqualifying for
these missing predicates.

For a separately authorized launch, prepare the normal complete startup plan,
set `adapter:"runtime_vllm_v1"`, pin `adapter_sha256` to the exact bridge file
and `collector_sha256` to the exact binary, and prospectively set the complete
runtime window/deadline/request allowances. Source-only example:

```sh
# Operator-owned serving invocation, not a Grill action:
PYTHONHASHSEED=0 python3 tools/startup-runtime.py \
  --plan PLAN --events NEW_JOURNAL --stage startup -- \
  --model MODEL --host 127.0.0.1 --port 8000
# A separate shell in the SAME PID/time namespaces; actual running frontend PID:
grill-perf startup observe PLAN --slot 0 --pid PID \
  --events NEW_JOURNAL --producer-source tools/startup-runtime.py --out NEW_CAPTURE
```

Use the existing `store`/`reload` commands and matching bridge `--stage` for
restart evidence; operator alone replaces the serving invocation between them.
Do not invoke recipe smoke/shape warmups hoping the bridge will ignore them.
The existing recipe launcher remains intact and owns deployment lifecycle.
This source support has not itself exercised or qualified a recipe/image.


## Plan v1 and finite exposure

`startup::Plan` is a closed `version:1`, `kind:"startup-study-v1"` object. Its
required fields are:

- `study_id`, collector binary SHA-256, adapter enum and exact adapter source
  SHA-256; every capture rechecks these pins.
- `runtime_window_us` is required only for `runtime_vllm_v1` and
  `runtime_asgi_fixture_v1`; it is absent for existing adapters. These explicit
  adapter contracts extend the closed plan without reinterpreting old bytes.
- `setup_axis` and `setup_sha256:[A,B,A2]`: prospectively declared setup
  fingerprints. A and A2 must match. These remain declarations, not observed
  proof that the only effective difference was that axis.
- `slots:[{id,arm,warmup,index}]`: arms `a`, `b`, `a2`, in that order, each with
  1–20 warmups followed by 3–100 measured acquisitions. Equal counts in all arms,
  contiguous indices within warmup/measured phases, unique IDs, at most 360 slots.
- `study:{kind:"startup"}` or the explicit restart-cache contract below.
- `states:[{class,declared,required_observed}]`: 1–6 unique prospective class
  conditions. Temperatures are `cold`, `warm`, `unknown`. An unknown observation
  never proves a requested cold or warm state. Both declared and observed state
  remain visible; a compiled-artifact cold condition is not a KV miss.
- `request:{model,prompt,max_tokens,expected_answer}`: bounded exact prompt and
  expected answer. The narrow producer uses nonstreaming, one-choice text,
  temperature zero, output cap 1–32768, without hidden sampling or namespace
  changes. All repeated list entries use this same exact request body.
- `endpoint`, `local_http`, optional credential environment **name**, optional
  finalized `metrics::v2::Config`; no credential value is retained and model
  credentials are never forwarded to metrics. Metrics2 is mandatory for restart.
- `max_events` 8–1024; `event_bytes` 1 KiB–4 MiB; `deadline_us` 1 ms–1 hour;
  `limits` with total HTTP 1–600000 ms, idle bounded by total, response at most
  1 MiB, aggregate retained-response allowance at most 16 MiB. Admission checks
  the full request/scrape deadline allowance and request/response byte ceilings.
  Encoded requests stay within the existing 2 MiB ceiling. These are serialized
  buffer ceilings, not an RSS guarantee.
- Nonempty `gates`, unique targets among `listening`, `ready`,
  `first_valid_inference`, `communication`, `reload`, each with prospective
  `max_regression_bps` and `max_reference_spread_bps`. Reload is only legal in a
  restart study. There is no token-rate or C1 conversion for these gates.

`tools/smoke-startup.py` contains a complete generated prospective plan example
with every field and the exact finalized metrics2 configuration. It pins the
binary and adapter file actually supplied to the fixture, rather than embedding
an obsolete sample fingerprint in documentation.

## Events, clocks and milestones

Each NDJSON event has `version:1`, a contiguous `sequence`, `clock`, `offset_us`,
engine incarnation, and a tagged `event`. The clock has an ID/origin,
`kind:"monotonic"`, `units:"microseconds"`, positive `resolution_us`, and
`synchronization:"single-process-origin"`. Resolution describes timestamp
quantization, not a claim of hardware timer precision.

| Event | Supported meaning |
|---|---|
| `launched` | First event from the attached process incarnation |
| `listening` | Producer has bound/listened; never model-ready or valid inference |
| `communication_start/end` | Explicit paired communication-initialization substage |
| `ready` | Exact `neutral-ready-v1` producer readiness contract; no model-name heuristic |
| `controls` | Source-observed smoke/shape suppression and hash-seed contract, before requests |
| `state` | Per-class declared/observed temperature plus appropriately typed identity |
| `request_start` | Request ID and SHA-256 of exact received body, in observed order |
| `inference_complete` | Request ID, retained response digest and supported per-request cache result |
| `failure` | Retained producer failure; cannot disappear behind faster samples |
| `sealed` | Final complete request count after inference admission closes |
| `runtime_start` | Bridge-entrypoint anchor, producer pin, source contract and observed namespaces; not OS launch |
| `runtime_engine_request` | Actual frontend engine admission correlated with the current ASGI request |
| `runtime_coverage` | Prospective minimum window elapsed, admission closed, active requests drained, hook installed |


For the neutral producer, `launched` is its first explicit source event, not the OS
`exec` instant. Interpreter/import work preceding the journal origin is outside
these source-clock durations; full OS-process-launch latency is unavailable.
First-valid-inference requires an actual native-collected, complete valid
response with exact expected answer, supported usage, stop termination and
matching source response digest. Failed/missing client evidence cannot be
replaced by an adapter event. Header/body arrival and health checks do not count.

Anchor-to-listen, anchor-to-ready and anchor-to-first-valid-inference subtract
only timestamps with identical clock contract/origin **and** process incarnation.
Communication duration uses its own supported start/end pair. Missing or
cross-clock boundaries stay unavailable. HTTP reload duration is separately
client-clock `settle_us`; it is never subtracted from a source-clock timestamp.
All request/metrics offsets within a capture share a capture-scoped client
origin. Across acquisitions, duration comparison requires matching clock kind,
units, resolution and synchronization, but not the same origin ID.

## Explicit restart-cache study

The `restart_cache` study adds:

```text
identity: CacheIdentity {
  version: 1, prompt_sha256, source_sha256, cache_salt,
  hash_seed_contract: "PYTHONHASHSEED=0:engine+cache_server"
}
post_restart: [{id:"first_reload"}, ...]  # 1–16 unique ordered requests
classes: [{class:"engine",transition:"changed"},
          {class:"cache_server",transition:"persisted"}, ...]
no_warming_attestation: boolean         # declaration only, never sufficient
```

The exact cache salt and prompt/source pins are shared by store and reload.
There is no random acquisition namespace, salt suffix, seed increment or
per-acquisition derivation. The prompt hash is checked against retained prompt
bytes; the source hash is a reviewed source declaration pinned in the plan.
Existing conversation namespaces are unchanged.

Engine/cache-server identity is a process incarnation. Filesystem,
compiled-artifact cache, weights and prefix-offload identities are instead
`{content_sha256,generation}` artifact identities. Required class transitions
are independently assessed. Same engine means invalid restart evidence;
missing relevant class evidence means unavailable; restarting the cache server
is not persistent KV across an engine-only restart.


The neutral producer observes engine/cache-server process identities only.
Its nonprocess identity and temperature fields remain explicitly unavailable
rather than hashing class names or pretending synthetic weights were observed.
Artifact identity predicates can be replayed from imported evidence, but native
qualification for those classes needs an actual supported source.
The first observed post-restart request must be `first_reload`. Source request
starts and finishes must exactly match the finite collected list, received
body hashes and retained response hashes, with a complete admission-closure
record and no preceding/extra request. Missing suppression or ordering evidence
withholds persistence even when the operator asserts there was no warming.

Actual `persisted_hit`, `recomputed`, `stored`, and `unavailable` results stay
distinct. A favorable persisted result requires the supported neutral producer's
actual cache lookup plus positive request cached-token evidence, matched stable
identity, changed engine, persisted cache server, and actual metrics2 snapshots.
The narrow neutral source additionally requires one complete full-label external
query/hit diagnostic pair with positive deltas; multiple-label or absent/zero
diagnostics withhold the claim, without summing or attributing server counters.
Metrics2 parsing, cache diagnostics, continuity views and accounting consistency
are reused without substituting a parallel schema. **Server-wide metrics are
still diagnostic, not per-request attribution**; no local/external counter alias
or inferred hit from speed is introduced. A missing promised hit or recomputation
is nonqualifying, not a zero-valued successful reload sample.

## Public recipe source mapping versus qualification

The pinned public DeepSeek recipe at
[`f3d76450e075d5f89698aed2e865655f0b7eeb1c`](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/tree/f3d76450e075d5f89698aed2e865655f0b7eeb1c)
provides these **source-reviewed constraints only**:

- [`lmcache/README.md`](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/blob/f3d76450e075d5f89698aed2e865655f0b7eeb1c/lmcache/README.md):
  stable `PYTHONHASHSEED=0` on engines and cache servers; independently surviving
  cache servers; a ZMQ port answer is listening, not ready or valid inference.
- [`docs/ENVS.md`](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/blob/f3d76450e075d5f89698aed2e865655f0b7eeb1c/docs/ENVS.md):
  launcher smoke precedes the default shape warmup; qualifying first reload
  would need no smoke and `DSPARK_BOOT_SHAPE_WARMUP=0`, or an operator launch path
  without those requests, **with actual complete observation of that fact**.

These uninstrumented recipe logs cannot establish complete request coverage,
substage boundaries, hidden-warming absence, actual restart identity or a
persisted first reload. `uninstrumented_recipe_logs` is an explicitly
nonqualifying imported/declaration source category, not a parser or a native
producer for those logs. No real DeepSeek/GLM startup adapter or backend has been
exercised or qualified by this source delivery. Production source support for
those unavailable facts requires separately reviewed instrumentation and an
explicit disruptive observation window; do not translate the neutral fixture's
result into model-KV persistence or communication speedup.

## Offline decision and limitations

Comparisons reuse `envelope::range` and `envelope::assess` with exact rational
microsecond samples and lower-is-better direction. A and A2 form the reference
range; every declared warmup and measurement must qualify. Native captures must
have distinct engine incarnations; the acquisition chain and plan fingerprints
must agree. Actual prompt/completion-token exposure must match. No dropping failed
trials, best-case selection, threshold adjustment or survivor-only gate is allowed.

Outcomes and exits retain shared policy precedence:
`ERROR` (1) > `REGRESSION` (3) > `INCONCLUSIVE` (2) > `PASS` (0).
An imported startup arithmetic PASS is explicitly `imported-comparison-only`,
not adapter exercise. Restart-cache persistence cannot PASS on imported evidence.
No outcome establishes causality, universal no-regression, production percentile,
serving speed, real-adapter exercise or live qualification.

## Later CPU verification

These exact commands are intended for the integration owner on a stable head:

```sh
cargo build -p grill-perf --locked
python3 tools/smoke-startup.py --binary target/debug/grill-perf --case startup
python3 tools/smoke-startup.py --binary target/debug/grill-perf --case restart
python3 tools/smoke-startup.py --binary target/debug/grill-perf --case comparison
# Or all three in one finite invocation:
python3 tools/smoke-startup.py --binary target/debug/grill-perf --case all
# Real bridge code, ordinary ASGI fixture; no vLLM/Uvicorn/device imports:
python3 tools/smoke-startup-runtime.py --binary target/debug/grill-perf
```

The smoke authors actual delayed ordinary processes, source failures and
malformed data, wrong clocks and stale/same/missing incarnations, identity drift,
hidden/extra/reordered requests, missing reload/hit/metrics evidence, full
recomputation, imported-provenance rejection, exact 5% boundary arithmetic,
insufficient acquisitions and broken order. It uses only synthetic loopback
fixtures. Its arithmetic inputs are explicitly imported synthetic mutations,
not relabelled live observations. Execution is a separate verification gate;
source authoring alone does not claim these commands passed.

The runtime smoke preserves the existing neutral cases and adds real bridge
lifecycle/admission execution through `tools/startup-asgi-fixture.py`
(`runtime_asgi_fixture_v1`, explicitly not vLLM execution): independent
ASGI-ready/listen/inference delays, external requests before and after the
planned request, direct internal engine admissions, failed ASGI startup,
container-namespace mismatch, source drift and missing seal. A separate real
store/operator-owned process replacement/first-reload fixture preserves stable
request identity and verifies unavailable KV accounting never becomes PASS.
It compares actual CLI capture with offline replay. Both CPU execution and
pinned-library/live qualification remain separate checks; authoring these
fixtures is not a pass.
