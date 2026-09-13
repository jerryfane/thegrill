# Shared recipe workflow

For recipe-facing baseline/change/check integration, use the
[canonical model-agnostic recipe](RECIPES.md). Its explicit selection manifest
is separate from the historical bundle described below; no existing bundle
entry, workload byte or advanced-policy result is reinterpreted.

The small selected concurrency examples preserve the recipe control distinction:
`concurrency-selection-v1.json` requests `chat_template_kwargs.thinking: false`;
`concurrency-enable-thinking-selection-v1.json` requests
`chat_template_kwargs.enable_thinking: false`. Select according to an explicitly
qualified template, never from the model name. A neutral third recipe uses the
same path without adding a mapping. These are descriptive selected captures,
not the original sparkDash workload or its observed-envelope decision policy.

The checked-in [recipe manifest](../../crates/grill-perf/examples/recipes-v1.json)
selects the frozen sparkDash decode/prefill workloads for DeepSeek and materialized
GLM variants. The GLM variants change only the workload name and the explicit
thinking-key declaration. `bundle verify` checks that relation offline; it does
not resolve, rewrite or collect workloads. Different thinking declarations have
different normalized workload identities. Do not compare across recipes as if
the workloads were equivalent.

These command blocks are the recipe entrypoints. There is no second collector,
shell orchestration tool, model discovery, forward creation, health check,
deployment change, cache flush, weight download, retry or upload step. Operators
manage serving state separately and obtain approval before model collection.
Unsupported request controls must fail qualification; never strip a field to
make a server accept the workload.

## Install and pin the actual artifact

Follow [download, checksum verification and installation](INSTALL.md) before
executing an archive. Use an explicitly approved release, or a separately reviewed
staged archive while publication is pending; staged artifacts are not promised
public release downloads. The supported native Linux targets and runtime
requirements are stated there. No Rust checkout is required for this workflow.

Set `GRILL_ROOT` to the absolute extracted package directory:

```sh
GRILL_PERF="$GRILL_ROOT/bin/grill-perf"
WORKLOADS="$GRILL_ROOT/workloads"
"$GRILL_PERF" --version
"$GRILL_PERF" bundle verify "$WORKLOADS/recipes-v1.json" --json
sha256sum "$GRILL_PERF"
```

Retain the reviewed archive checksum/build receipt and actual binary/workload
identities. Use this absolute executable path throughout, not an unrelated binary
on `PATH`. Keep packaged workloads immutable during inspection and collection.
The Git revision is not the workload `source_sha256`; the latter hashes exact
file bytes. `workload_sha256` is the typed normalized workload identity.

The source-build fallback remains documented in [INSTALL.md](INSTALL.md).
Historical source `a6729f9ab5410587bb9da1adb7b34944a9cfc436` was CPU-qualified
on native ARM64 for the original four-entry workflow. That historical record is
not a current binary pin, live qualification or qualification of another target.

## Verify and select the exact workload

```sh
MANIFEST="$WORKLOADS/recipes-v1.json"
"$GRILL_PERF" bundle verify "$MANIFEST" --json
```

The verifier admits a closed, bounded manifest with exactly four entries and the
explicit `deepseek`/`glm` decode/prefill mappings. Each entry declares its fixed
leaf filename, raw and normalized digests, and the GLM entries declare their
corresponding `base`. The manifest has no self-hash or containing Git revision.
Verification rejects unsupported versions, unknown fields, incorrect mappings,
unsafe paths, symlink files or roots, hash mismatches and semantic variant drift.
Its report contains the manifest digest, entry identities, declared controls and
request/token ceilings. Keep the source directory immutable during verification
and collection; verification is not a filesystem snapshot or signature.

| Setting | Decode, either recipe | Prefill, either recipe |
|---|---|---|
| Request profile | `vllm-fixed-v1`, streaming | `vllm-fixed-v1`, streaming |
| Output | exact 400 tokens | exact 8 tokens |
| Sampling | temperature zero, top_p one, no seed | temperature zero, top_p one, no seed |
| Cache | `observe` | `observe`; salted filled prompts |
| Requests including warmup | 72: 18 warmup, 54 measured | 16: 4 warmup, 12 measured |
| Output token ceiling | 28,800 | 128 |
| Total / idle deadline, milliseconds | 360000 / 60000 | 600000 / 600000 |
| Response / wave allowance, bytes | 1048576 / 67108864 | 65536 / 4194304 |

The workloads retain their exact case/cell schedules and declared warmups and
trials. Prompt size names are not tokenizer measurements. Exact output requests
use `min_tokens`, `max_tokens` and `ignore_eos`; successful length stops are not
automatically censored failures. Eligibility and provider usage remain decisive.
`observe` does not establish a cold cache. Review reported generated channels and
reasoning-token evidence: a declared thinking key does not prove the template
honored it.

DeepSeek sends `chat_template_kwargs: {"thinking": false}`; GLM sends
`chat_template_kwargs: {"enable_thinking": false}`. Neither mapping automatically
qualifies the other. Preserve the frozen sparkDash files and attribution to
[MiaAI-Lab's sparkDash](https://github.com/MiaAI-Lab/sparkDash).

### Phase counts are not protocol equivalence

The separate [v3 phase-budget option](README.md#phase-specific-output-budgets)
can request capped warmup32/measured400 without changing these frozen exact400
recipes. It is a new workload identity, not a repair or relabeling of old evidence.
The historical installation pin above predates v3; use an approved source revision
containing the feature, not a historical qualification receipt as a feature claim.

At pinned upstream sparkDash revision
`d0c7f71296a1071d0d75f95b14c21413d4d06321`,
[`DecodeBench.js`](https://github.com/MiaAI-Lab/sparkDash/blob/d0c7f71296a1071d0d75f95b14c21413d4d06321/server/collectors/DecodeBench.js)
uses capped 32-token best-effort warmup. Measured output defaults to 400 with
`min_tokens`, `ignore_eos` and `stop: []`; an HTTP 400 path can strip fill-force
fields and resend. A capped400 TheGrill request therefore differs from that
initial measured request, even when its maximum token count matches.
Upstream [`LlmStreaming.js`](https://github.com/MiaAI-Lab/sparkDash/blob/d0c7f71296a1071d0d75f95b14c21413d4d06321/server/collectors/LlmStreaming.js)
also sends `enable_thinking`, `thinking` and `thinking_mode` together and has
thinking-control fallback behavior. TheGrill sends one explicitly declared
thinking-key mapping, never infers it from a model name and never strips controls
or retries. It retains warmup evidence and stops on ineligible responses rather
than treating warmup as best-effort.

Neither matching 32/400 counts nor selecting exact measured output establishes
full protocol equivalence: control sets, failure handling, schedules and timing
semantics remain separate. Qualify the declared workload and template rather
than claiming source fidelity from token counts alone.

### Measurement compatibility matrix

This records how the existing selected workloads relate to the pinned upstream
protocol. "Match" means equal under the stated prerequisites; "different" is an
intentional, declared difference; "unsupported" is deliberately absent. It adds
no workload variant, declared field or runtime behavior, and matching prompt text
alone never establishes protocol equivalence.

| Aspect | Existing TheGrill behavior | Pinned upstream `d0c7f712` behavior | Classification |
|---|---|---|---|
| Prompt bytes and concurrent lane variation | Frozen decode cases contain no `{salt}`, so concurrent lanes send the same case text. Where explicitly present, `{salt}` renders as `namespace[..16]-wave.index-lane`; the frozen prefill workload places it before its repeated fill. | `pickDecodeBenchPrompts` returns the base prompt for one lane and appends ` (stream i/n)` per lane at higher concurrency; `buildPrefillPrompt` leads every size with `[prefill-bench <uuid>]`. | Decode base prompts match; concurrent prompt variation differs. Prefill salt placement is analogous, not identical prompt bytes or proof of cache isolation. |
| Warmup schedule, output and failure policy | Per-cell `warmup_trials`; frozen decode declares seven warmup waves (18 warmup requests) and prefill four, each at the declared exact output (400 and 8). Prefill warmups use each cell's full prompt size. Warmup is retained evidence: policy requires every declared warmup eligible, and a new v3 run stops later admission on failure without retry. | DecodeBench runs one capped 32-token warmup per job before the concurrency loop and ignores its failure; PrefillBench runs one best-effort warmup with an estimated 512-token prompt and an eight-token output cap. | Different: count, prompt size, output allowance and failure handling. The generic v3 `warmup_output` declaration can explicitly separate output allowance without rewriting these frozen workloads. |
| Measured `min_tokens`, `ignore_eos`, `stop` | Exact output sets `min_tokens` and `ignore_eos: true`; the request schema has no `stop` field, so no `stop` is serialized, and a failed response is never re-sent with stripped controls. | DecodeBench adds `min_tokens`, `ignore_eos: true` and `stop: []`; an HTTP 400 carrying fill-force fields re-sends once through `stripFillForceFields`, which deletes all three. PrefillBench instead caps output at eight tokens without those fill-force fields. | Decode's initial `min_tokens`/`ignore_eos` match; prefill's output contract differs. `stop` is unsupported and its omission is not proven equivalent to `[]`; control-stripping retries are intentionally unsupported. |
| Explicit thinking mapping | Exactly one declared mapping per workload: legacy `thinking` sends `chat_template_kwargs.thinking`; `thinking_control` `vllm-enable-thinking-v1` sends `chat_template_kwargs.enable_thinking`. Both cannot be non-null and neither is inferred from a model name. | `applyThinkingFlags` sends `enable_thinking`, `thinking` and `thinking_mode` together (off for these benches), and an HTTP 400 retry re-adds all three plus top-level `thinking`/`enable_thinking`. | Different: one declared key, no fallback. |
| Reported versus estimated usage | Required metric counts come from provider-reported usage. Missing counts remain unavailable; the policy's matched-output gate requires completion counts for every metric. Counts are never estimated from text or SSE events. | Prefers `usage.completion_tokens` but falls back to `estimateTokenCount` (about four characters per token) for decode and prompt counts when usage is absent. | Different: estimates are unsupported. |
| Text-event versus settle timing window | Policy `decode_tokens_per_second` uses `(n - 1) * 1e6 / (settle_us - first_generated_text_us)`, including terminal delay and parsing. The text-event rate uses `(n - 1) * 1e6 / (last_generated_text_us - first_generated_text_us)` and is an observation, not a selectable policy metric. | `decodeTps` uses the first-visible-token to last-visible-token window, excluding stream teardown. | Different for the policy gate; the text-event formula is not silently aliased to settle-window decode. |
| Prefill cache exclusion | The derived prefill rate, and the policy gate, require provider-reported cached prompt tokens absent or zero; otherwise the sample is null. Absent cache telemetry is unknown, not evidence of a cold cache. A salt appended after a shared prefix does not isolate that prefix. | `buildPrefillPrompt` puts a fresh random salt at the start to reduce cross-run prefix reuse, runs one request per size and derives `prompt_tokens / TTFT` with a text-estimate fallback. | Analogous leading-salt placement; cache-field exclusion and usage fallback differ. |
| Aggregation and failed-cell behavior | A fixed wave needs every lane eligible; an incomplete response nulls the wave completion total, and a failure or ineligibility stops later admission without retries. Policy requires complete eligible waves and observations per cell and metric; a missing or failed cell cannot pass and every gate reason is retained. | Aggregates only successful streams (`streamsOk`) into mean/median/min/max, reports `streamsFailed` and an error string, and continues to the next concurrency; the aggregate decode rate uses the earliest first-token to latest last-token window. | Different: fail-stop completeness versus a success-only partial aggregate. |

This matrix introduces no new workload variant. Any necessary change identified
by a pilot requires an explicit new declaration and identity, not an edit to a
frozen workload or a silently weakened protocol.

## Create and approve the policy before collection

Choose the recipe and one workload first. A decode study and a prefill study are
separate acquisitions with separate source pins and policy scopes. Use the
actual binary and exact selected workload to obtain the pins:

```sh
sha256sum "$GRILL_PERF" "$WORKLOAD"
```

Create `POLICY` as an absolute filename outside the checkout, using a JSON editor.
Copy the first digest into `collector_sha256` and the second into
`workload_source_sha256`. Do not use the Git SHA, Cargo.lock digest, manifest digest
or normalized workload digest for those fields. Choose an identifier, required
metrics and practical regression/reference-spread tolerances with the study
reviewer, independently of the collected outcomes. Approve the complete policy
before the first run; retain exactly the same policy bytes for A, B and A2.
`run --policy FILE` validates and persists those bytes before dispatch. It does
not prove independent preregistration or physical server restoration.

The following policies are **deliberately invalid illustrations, not runnable
policies**. Every `REPLACE_...` string must be replaced. Hashes must become actual
lowercase SHA256 strings; each tolerance must become an operator-approved JSON
integer, not a string. No tolerance below is a recommendation or implicit
default. These examples choose one metric per cell; the operator may explicitly
add other required metrics to every relevant cell before approval. Do not remove
cells. Decode requires the complete decode scope shown here for either recipe:

```json
{
  "version": 1,
  "method": "observed-envelope-v1",
  "id": "REPLACE_APPROVED_POLICY_ID",
  "collector_sha256": "REPLACE_ACTUAL_BINARY_SHA256",
  "workload_source_sha256": "REPLACE_SELECTED_DECODE_FILE_SHA256",
  "min_trials": 3,
  "cells": [
    {"cell":"structured-1","metrics":[{"metric":"decode_tokens_per_second","max_regression_bps":"REPLACE_APPROVED_INTEGER","max_reference_spread_bps":"REPLACE_APPROVED_INTEGER"}]},
    {"cell":"structured-2","metrics":[{"metric":"decode_tokens_per_second","max_regression_bps":"REPLACE_APPROVED_INTEGER","max_reference_spread_bps":"REPLACE_APPROVED_INTEGER"}]},
    {"cell":"structured-4","metrics":[{"metric":"decode_tokens_per_second","max_regression_bps":"REPLACE_APPROVED_INTEGER","max_reference_spread_bps":"REPLACE_APPROVED_INTEGER"}]},
    {"cell":"structured-8","metrics":[{"metric":"decode_tokens_per_second","max_regression_bps":"REPLACE_APPROVED_INTEGER","max_reference_spread_bps":"REPLACE_APPROVED_INTEGER"}]},
    {"cell":"prose-1","metrics":[{"metric":"decode_tokens_per_second","max_regression_bps":"REPLACE_APPROVED_INTEGER","max_reference_spread_bps":"REPLACE_APPROVED_INTEGER"}]},
    {"cell":"code-1","metrics":[{"metric":"decode_tokens_per_second","max_regression_bps":"REPLACE_APPROVED_INTEGER","max_reference_spread_bps":"REPLACE_APPROVED_INTEGER"}]},
    {"cell":"json-1","metrics":[{"metric":"decode_tokens_per_second","max_regression_bps":"REPLACE_APPROVED_INTEGER","max_reference_spread_bps":"REPLACE_APPROVED_INTEGER"}]}
  ]
}
```

For a prefill study, use this separate complete policy instead:

```json
{
  "version": 1,
  "method": "observed-envelope-v1",
  "id": "REPLACE_APPROVED_POLICY_ID",
  "collector_sha256": "REPLACE_ACTUAL_BINARY_SHA256",
  "workload_source_sha256": "REPLACE_SELECTED_PREFILL_FILE_SHA256",
  "min_trials": 3,
  "cells": [
    {"cell":"prefill-4k-1","metrics":[{"metric":"prefill_tokens_per_second","max_regression_bps":"REPLACE_APPROVED_INTEGER","max_reference_spread_bps":"REPLACE_APPROVED_INTEGER"}]},
    {"cell":"prefill-8k-1","metrics":[{"metric":"prefill_tokens_per_second","max_regression_bps":"REPLACE_APPROVED_INTEGER","max_reference_spread_bps":"REPLACE_APPROVED_INTEGER"}]},
    {"cell":"prefill-16k-1","metrics":[{"metric":"prefill_tokens_per_second","max_regression_bps":"REPLACE_APPROVED_INTEGER","max_reference_spread_bps":"REPLACE_APPROVED_INTEGER"}]},
    {"cell":"prefill-32k-1","metrics":[{"metric":"prefill_tokens_per_second","max_regression_bps":"REPLACE_APPROVED_INTEGER","max_reference_spread_bps":"REPLACE_APPROVED_INTEGER"}]}
  ]
}
```

The metric enum also permits `wave_latency_us` and
`achieved_completion_tokens_per_second`; direction is intrinsic to the metric.
Every workload cell must appear exactly once with nonempty unique metrics.
Admission enforces the declared trial minimum against the workload. The decision
additionally requires warmup in every cell; admission alone does not enforce that
condition. Basis-point schema bounds are engineering limits, not scientific advice.
The reference-spread bound is an observed variability gate, not a confidence bound.
See the [performance contract](CONTRACT.md) for policy admission and decision
arithmetic. No policy generator subcommand is provided.

Prepare complete private deployment JSON files for A and B, containing nonempty
`model_revision`, `runtime`, `hardware` and `settings` declarations. Describe the
actual context limit, quantization, tensor parallelism, speculative configuration
and other relevant serving settings there; do not paste invented example values.
Reuse A's exact deployment declaration for the restored repeat. Keep the same
model selector and stable endpoint for A/A2; changing a forward port prevents the
declared reference identity from matching. Record B's actual declaration rather
than copying A's declaration when the configuration changed.
Native `run` admission permits absent deployment fields; it does not certify
reference completeness. Missing reference declarations prevent a qualified
`decide` result even when collection was admitted.

## One baseline/change/check decision path

Choose exactly one workload file before creating its policy. These frozen bundle
entries are explicit data choices, not model detection or interchangeable
qualifications:

| Declared mapping | Decode file | Prefill file |
|---|---|---|
| `enable_thinking: false` (historical GLM entry) | `glm-decode-v1.json` | `glm-prefill-v1.json` |
| `thinking: false` (historical DeepSeek entry) | `sparkdash-decode-v1.json` | `sparkdash-prefill-v1.json` |

Set `WORKLOAD` to the chosen absolute file under `WORKLOADS`, or another explicitly
reviewed workload. A new backend uses declared data through these same commands,
not a new bundle mapping. Qualify its controls separately.

Set `POLICY`, `DEPLOYMENT_A`, `DEPLOYMENT_B`, `A`, `B` and `A2` to absolute paths.
Each output directory must be new with an existing parent. Set `ENDPOINT_A`,
`ENDPOINT_B` and `MODEL` explicitly. These commands assume HTTPS; literal-loopback
HTTP additionally needs `--local-http` on each preflight/run. Add
`--auth-env MODEL_API_KEY` only when the independently supplied credential is
required; its value never belongs in the policy, workload or command arguments.

First inspect and admit both declared configurations offline:

```sh
"$GRILL_PERF" bundle inspect "$WORKLOAD"
"$GRILL_PERF" preflight "$WORKLOAD" --policy "$POLICY" --endpoint "$ENDPOINT_A" \
  --model "$MODEL" --deployment "$DEPLOYMENT_A"
"$GRILL_PERF" preflight "$WORKLOAD" --policy "$POLICY" --endpoint "$ENDPOINT_B" \
  --model "$MODEL" --deployment "$DEPLOYMENT_B"
```

Preflight prints each run's complete warmup/measured request and output ceilings.
Approve the whole three-role window, not just one run's allowance: the unchanged
workload is executed once in each of A, B and A2. A separate unchanged-control
exercise adds its own explicitly planned traffic. Per-request deadlines remain
finite; no automatic probes, extra repetitions or budget expansion occur.

Capture the baseline with the policy already bound:

```sh
"$GRILL_PERF" run "$WORKLOAD" --policy "$POLICY" --endpoint "$ENDPOINT_A" \
  --model "$MODEL" --deployment "$DEPLOYMENT_A" --out "$A" --json
```

After A completes, make the separately authorized serving change, then capture B:

```sh
"$GRILL_PERF" run "$WORKLOAD" --policy "$POLICY" --endpoint "$ENDPOINT_B" \
  --model "$MODEL" --deployment "$DEPLOYMENT_B" --out "$B" --json
```

After B completes, restore and independently verify A through the operator's
approved procedure. Capture a fresh A2, then check the retained evidence offline:

```sh
"$GRILL_PERF" run "$WORKLOAD" --policy "$POLICY" --endpoint "$ENDPOINT_A" \
  --model "$MODEL" --deployment "$DEPLOYMENT_A" --out "$A2" --json
"$GRILL_PERF" decide "$A" "$B" --reference "$A2" --json
```

This is the native policy-bound `run`/`decide` path, not a new mode of the
eight-acquisition `baseline`/`check` capture commands. Explicit selections there
remain descriptive; the default C1 assessment retains its own semantics.
Do not pass a capture root as a
native policy run or choose one of its acquisitions after seeing the results.
A final A/B/A2 verdict cannot precede the post-candidate reference.

For an unchanged-control exercise, declare the same deployment in all three roles
and make no serving change. Its result tests the stated policy against those
observed periods; PASS does not establish causality or future repeatability.

## Read decisions without broadening the claim

`compare` remains descriptive: successful comparison is eligibility, never PASS.
Only a successfully printed versioned `decide` envelope constitutes a decision.
Do not infer a verdict from an exit code or missing output; parsing/output errors
can share exit values with decision outcomes. Inspect `decision`, `eligibility`,
all scoped gates, coverage and bounded reason codes. The policy evaluates the
observed envelope, not statistical significance, equivalence, intelligence,
causality or universal no-regression.

A, B and A2 must all capture the same exact policy before dispatch. Missing
bindings are INCONCLUSIVE; conflicting present bindings or corrupt evidence are
ERROR. Insufficient observations, unqualified references, copied acquisitions,
out-of-order declared starts or missing metrics cannot silently become PASS.
Declared start order and matching reference declarations do not authenticate
nonoverlap or physical restoration. Paused/resumed sessions do not qualify as
uninterrupted performance acquisitions. Preserve every cell and metric; do not
select only favorable gates or treat a withheld comparison percentage as zero.

### Supported policy eligibility conditions

These are the existing conditions under which an explicitly chosen workload can
support a `decide` verdict. Failing a required condition prevents PASS; unresolved
and invalid gates remain visible alongside any separately resolved regression.

- The policy must bind prospective exact pins before dispatch: `collector_sha256`
  is the actual executable and `workload_source_sha256` is the exact selected
  workload file bytes. Baseline, candidate and reference must all carry the same
  captured policy bytes.
- A separately collected A2 reference is required and must declare a start after
  the candidate. A missing repeat, or a baseline/reference identity that is not a
  declared match, is INCONCLUSIVE; `compare` alone is eligibility-only.
- Every declared warmup must exist and be eligible. A cell that declares no
  warmup cannot pass.
- Every declared cell and metric must show complete eligible coverage: the
  expected measured waves and observations for the declared trials and
  concurrency. Missing observations are reported, never treated as zero.
- The ordered measured lanes must report equal completion counts across baseline,
  candidate and reference, and this applies to every policy metric including
  latency-only gates. A completed capture without matched reported counts is not
  eligible.
- The pooled baseline/reference range must sit around a positive minimum and fit
  the declared reference-spread budget.
- Decode and prefill ranges pool lane observations from every measured wave, so
  concurrent lanes contribute dispersion rather than independent repetitions; the
  declared minimum trial count is a floor, not a precision or power guarantee.
- Every gate keeps its own coverage and bounded reason codes. A resolved
  REGRESSION outranks an unresolved INCONCLUSIVE gate; the aggregate outcome is
  the maximum over gates, and PASS requires every gate rather than a favorable
  filtered subset.

Explicit-selection baseline/check captures and raw `compare` output remain
descriptive; neither produces this policy PASS/REGRESSION vocabulary.
See the [policy guide](README.md#captured-observed-envelope-policy) for the closed schema
and decision arithmetic.

Maintain separate source-reviewed, loopback-tested and live-qualified statuses
for each recipe and workload. Both explicit control mappings and all four
workload entries were exercised through the CLI against synthetic responses.
Live qualification remains pending in the adopted pilot order: GLM EXL3 first,
then DeepSeek DSpark2, then Qwen27 SGLang. Each window is separately authorized;
the operator owns every serving change, restoration, credential and traffic
budget, and TheGrill adds no launcher. Earlier model smoke receipts do not
qualify this new bundle/policy workflow.

The Qwen pilot is source-reviewed from the authoritative pinned
[`ndec.py`](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark/blob/9fb18edf8cfb3364e8aa89258e6d5ab1fe1fd11a/bench/ndec.py):
one 16-token warmup, then two nonstreaming calls per prompt at caps 60 and 600
across two prompts, reporting `(c600 - c60) / (t600 - t60)` from provider
completion counts and wall times. That differential estimate is a separate
diagnostic, not streaming decode equivalence, and TheGrill does not implement it.

Use the [manual reviewed report template](SHARED-REPORT-TEMPLATE.md) only after
privacy review. It is not an exporter or replayable evidence package. No upstream
recipe files are changed by these commands; maintainers may separately adopt
these public command blocks without copying private deployment code or evidence.
