# Bounded conversation checks

Use the normal capture workflow with the explicit selection:

```sh
grill-perf baseline --selection /absolute/path/to/conversation-selection-v2.json \
  --endpoint "$ENDPOINT" --model "$MODEL" --deployment "$DECLARATION" \
  --auth-env MODEL_API_TOKEN --out "$PRIVATE/baseline"
grill-perf check "$PRIVATE/baseline" --change none \
  --deployment "$DECLARATION" --out "$PRIVATE/control"
grill-perf compare "$PRIVATE/baseline" "$PRIVATE/control" --json
```

Keep `conversation-v2.json` alongside its selection manifest. Use `--local-http`
only for an explicitly allowed local HTTP fixture. A serving change is made by
the operator, never TheGrill; select its declared changed field for a candidate.
`check` inherits the pinned workload, controls and endpoint from the baseline.
No agent, cache reset, tool process or experiment coordinator is involved.

## Scope and budget

The pinned scenario has thirteen ordered C1 steps: alpha/beta prime, reuse,
alternation, edit/restore, fixed tool continuity and short prime/reuse. After the
short history is introduced, `return-beta` observes cache reuse without requiring
a hit; `recover-beta` is parented to that actual return and requires a reported
hit. These requirements are prospective, not relaxed after an eviction.

Each capture contains eight whole-sequence acquisitions: **104 requests, at most
13,312 requested output tokens**, including all primes and follow-ups. There are
no hidden warmups or replacement requests. The default whole-capture deadline is
300 seconds, including preparation and native evidence publication; each request
is capped at 128 output tokens, 64 KiB response and 10 seconds. A candidate/control
has a separate equal allowance.

Workload v2 bounds a sequence to sixteen steps. Each encoded accumulated request
is limited to 128 KiB and 64 messages; retained histories have a 128 KiB bound per
step. A limit, interruption, missing required evidence or failed declared check
stops later admission and retains the partial sequence. Sequences cannot resume.
A new capture requires a new explicit allowance; failure never triggers one.

The explicit `vllm-conversation-v2` profile declares streaming at workload level.
Effective settings derive from each prospective expected response kind: factual
steps stream with usage requested; tool steps require a full nonstreaming
response. There is no independent per-step streaming knob and no streamed tool
assembly. Requests include the declared `enable_thinking=false` mapping without
retry or field stripping. The provider must support these controls, `cache_salt`,
full tool messages and the declared usage fields; TheGrill cannot attest that
the backend honored them.

Factual steps report client-observed first-generated-text, first-answer-text,
last-generated-text, terminal and settlement timing through the existing
collector. A delayed terminal is not the last answer arrival. Tool
first-output/first-answer latency and decode-only rates remain **unavailable**:
body or header arrival is not a generated-token observation.

The selected `v2` conversation path retains its exact workload bytes and pins.
The separately versioned streamed-tool path below does not upgrade historical
receipts. The earlier all-nonstreaming prototype was development-only,
not a published contract or a separately supported capability. Replay of its
private development receipts uses the preserved pre-consolidation source and
binary, not a compatibility branch in the current collector. Text-only published
profiles still reject tools; the separate quality evaluator is unchanged.

## Prefix and lineage contract

Each step names a history and optionally an earlier parent in that same history.
Its messages append only declared user inputs to the parent's **actual retained
assistant output**, not an expected or repaired answer. Editing and restoring
parent the same pre-edit state; the restore does not inherit the edited sibling.
Independent histories use different namespace/history cache salts. A new native
acquisition gets a new namespace; no real server cache is flushed.

Root repeated-fill content is deterministic and has no per-attempt text salt.
Continuation steps cannot fill or request cold-zero semantics. Existing workload
v1 filled prompts retain their original salting behavior. Prefix-cache hits are
provider reports: priming does not prove reuse, and token block alignment,
eviction and backend replay can change observed cached tokens. A declared hit
requires a positive reported count; missing telemetry or a zero count cannot
pass that requirement. `observe` permits unknown telemetry without inventing a
hit. The original alternation still prospectively requires retention. A zero
count there fails cache eligibility and stops admission. Only the newly declared
`return-beta` uses `observe`; its warm recovery requires a positive count.

This is a **small bounded retention scenario**, not guaranteed real-backend
eviction or capacity pressure. The CPU fixture explicitly simulates **two
retained history slots**, least-recently-used replacement and one latest request
per retained history. It computes actual common byte prefixes within the matching
salt, maps four bytes to a synthetic token, and rounds down to sixteen-token
blocks. Introducing the short history evicts beta in this declared simulation;
beta's return misses and its parent-linked recovery hits. Those fixture rules are
not a product cache implementation, tokenizer model or claim about a serving
backend's retention policy. A real backend may retain beta and report a hit on
the observation step. No real cache is reset or filled until eviction occurs.

Tool support is deliberately one fixed `lookup_fact` function. A step declares
one expected key and local fixture result; no arbitrary function runs. The
response must contain one full call, the correct name and JSON arguments, a
bounded unique ID and `tool_calls` finish. The following request uses the actual
call/arguments/ID and the declared fixture result. Invalid or duplicate arguments
are rejected, never repaired. Every tool step requires a parent-linked factual
follow-up, so a tool call alone cannot claim continuity.

A tool step that also requires a reported prefix hit needs the declaration to
be part of the shared leading prompt. On chat templates that render `tools`
before the first message (the GLM template does), a step-scoped declaration
changes the leading prompt, so the tool step shares no cached blocks with the
tool-free requests that precede it and a required hit cannot be observed.
`expect: {"kind":"tool", ..., "shared": true}` declares the same fixed
`lookup_fact` array on every step of that history instead: factual steps send
`tool_choice: "none"` and must still answer factually, while the tool step
keeps its forced call. All tool steps in one history must agree on `shared`,
and the declaration requires workload version 5 or 6 where the bounded tool
trace allowance exists. `tool_choice` is a decoding control, but a backend may
omit tools from template inputs when its value is `"none"`. Shared-prefix
qualification therefore requires confirming that the serving configuration
retains the declarations; acceptance of the request alone is insufficient.
Historical workloads without `shared` keep their
exact request bytes, hashes and outcomes; the flag is opt-in and never
retroactively applied.

## Reading the evidence

`Attempt.sequence.correct` records factual/structural correctness.
`canonical_match` records whether factual JSON used its canonical spelling;
`strict_match` applies only where a strict output string was declared beforehand.
Equivalent JSON whitespace is semantically valid and the original bytes remain
in later history. A declared strict failure stops admission without relabeling
it as a factual error. Genuine stale facts and cross-history leakage fail.

Correctness, performance eligibility and comparison outcome are separate.
A semantically wrong response may have valid timing and usage; those observations
remain visible, but it cannot authorize subsequent steps or a successful capture.
Per-step native waves retain request bytes, raw output, parent/history identity,
completion/cache observations, errors and available provider metrics. Offline
replay reconstructs requests from actual parent responses and checks the same
lineage and semantics. Provider-wide snapshots, if collected through advanced
`run`, do not prove a per-step cause without label/reset/traffic accounting;
unsupported preemption telemetry remains unknown.

Streamed factual lineage concatenates only answer-content deltas accepted by the
existing semantic decoder and SSE parser through the exact accepted terminal
boundary. It preserves decoded content formatting, excludes reasoning and
post-terminal bytes, and bounds the reconstructed answer by the response ceiling.

Successful selected comparisons report `DESCRIPTIVE`, displayed as
`COMPLETE - DESCRIPTIVE ONLY`; this means required evidence and declared checks
completed, not a measured speedup or equivalence. Incomplete captures remain
inconclusive and invalid evidence remains invalid.

Selected captures remain descriptive per step/acquisition. They can show improved
long-history reuse alongside slower short follow-ups; no pooled improvement,
model-quality score, C1 confidence interval or production qualification follows.
The controlled CPU fixtures exercise protocol, linkage, simulated eviction,
recovery and failure handling—not real-model correctness or cache capacity.

The old `v2` profile still has no tool first-output observation. Guaranteed
real-backend eviction/capacity pressure, live backend qualification and causal
per-step attribution of provider-wide counters or preemptions remain unqualified.
Unsupported telemetry stays unknown. Live qualification needs its own finite
approved window. Raw requests, responses, declarations and endpoints remain
private; publish only a reviewed sanitized summary.

## Streamed fixed tools: workload 5 / conversation-v3

The explicit `vllm-conversation-v3` profile uses workload version **5** and native
plan version **4**. It streams the existing fixed `lookup_fact` step as well as
factual responses. The old workload 2 / conversation-v2 path, selected capture
bytes and nonstreaming tool semantics are unchanged. This is not a new
conversation runner, repetition contract or automatic performance decision.

The compact `crates/grill-perf/examples/conversation-tools-v3.json` example is
selected explicitly through the advanced native workflow:

```sh
grill-perf run /absolute/path/to/conversation-tools-v3.json \
  --endpoint "$ENDPOINT" --model "$MODEL" --auth-env MODEL_API_TOKEN \
  --out "$PRIVATE/tools"
grill-perf compare "$PRIVATE/tools" "$PRIVATE/tools" --json
```

Only run against a provider in a separately approved finite window. This example
declares two requests, each capped at 128 output tokens, 64 KiB response and ten
seconds, including the required follow-up. It does not require or prove a cache
hit. No policy, resume, schedule, hidden warmup or repeated trial is admitted by
workload 5. The existing sixteen-step, 64-message and 128 KiB accumulated
request/history bounds remain in force.

Two tool observations have deliberately different meanings:

- `first_tool_delta_us`: receipt of the first accepted **nonempty name or
  argument fragment**. An ID-only, role-only or empty fragment does not qualify.
- `first_validated_tool_call_us`: receipt of the earliest assembled candidate
  with the complete valid ID, fixed name and exact JSON arguments, **only if**
  the final call and terminal state validate. A syntactically valid JSON prefix
  followed by contradictory fragments never becomes a validated call.

Both are lane-dispatch-relative client monotonic microseconds, not provider
compute timestamps. They are absent, not zero, when unavailable. Their order is
headers ≤ first body ≤ first tool delta ≤ validated call ≤ terminal ≤ settlement,
where each optional observation is checked explicitly. Tool observations never
populate the generated-text/answer fields or a fabricated text channel. A delayed
`[DONE]` is a later terminal, not later generated tool output. A canceled or
otherwise incomplete attempt retains any first delta but exposes no validated
call.

The streamed schema accepts one call at index zero, a whole bounded ASCII ID
exactly once, function type exactly once, fragmented `lookup_fact` name and at
most 4096 decoded argument bytes. The arguments must be exactly one object with
one `key` string equal to the declared fixture key; duplicate/unknown fields and
trailing values fail. IDs cannot duplicate an earlier call in the retained
parent history. Unknown tools, multiple calls and answer text in the tool step
fail; no arbitrary tool runs.

Successful streamed fixed calls accept `tool_calls` or `stop`, followed by
`[DONE]`. The pinned [vLLM named-tool implementation](https://github.com/vllm-project/vllm/blob/487ecf187/vllm/entrypoints/openai/chat_completion/serving.py)
uses `stop` for the explicitly named function selected by this profile.
This applies to the streamed workload-5 and workload-6 paths only; historical
nonstreaming tool semantics remain unchanged. `length`, missing finish/terminal
events, malformed arguments and incomplete calls still fail. Neither successful
finish spelling bypasses tool identity, history or usage validation.

The follow-up is built by the existing history machinery using the actual
assembled ID/name/argument bytes and declared local fixture result. Formatting
inside the encoded arguments is retained. A parent-linked factual follow-up is
required prospectively and must itself complete its existing correctness checks
before continuity is successful. A valid call alone is not a conversation pass.

The added fixtures are synthetic protocol coverage, not real-provider
compatibility or live qualification. Automatic tool-latency comparison gates,
larger/repeated histories and provider qualification are separate work.

For tool steps only, `Timing.tool_stream_arrivals` retains at most 4096
`{end_offset, observed_us}` records: the exclusive retained-body offset and
lane-dispatch-relative arrival time for each nonempty received chunk. The
prospective per-tool wave-buffer allowance includes an additional 1 MiB for
this bounded trace, its receipt encoding and fixed-call context. Reaching the
chunk cap before settlement retains a `response_limit` failure, not an
unbounded metadata stream. Offsets must strictly advance through exactly the
retained bytes; clocks must be monotonic and inside the observed response.
Offline replay feeds those exact byte ranges with their recorded arrival times
to the same SSE and fixed-call validators and checks the three distinct
boundaries. Bytes alone do not establish wall timing, and client receipts
cannot authenticate the clock observations against a dishonest collector.
