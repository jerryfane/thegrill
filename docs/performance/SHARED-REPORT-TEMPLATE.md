# Manually reviewed shared recipe report

Copy this template into a separately reviewed public report. Replace bracketed
fields only with explicitly approved public values. Omit a field or mark it
`withheld` when publication is not approved; never paste private text to explain
why it was withheld. This summary is not raw replay evidence, an authenticated
execution record or an automatic safe-to-publish export. The CLI has no exporter
or uploader.

## Publication gate

- Public pins, change description and scope approved: [yes / no]
- Privacy review completed: [yes / no]
- Entire report restricted to the approved fields below: [yes / no]

Do not publish while any gate is `no`. Do not attach private deployment JSON,
local paths, endpoints, model/provider selectors, credentials, raw prompts,
responses, raw timing/usage metrics, metrics labels, stderr or loader diagnostic
text. Do not paste complete run/compare output. Review policy identifiers and
all other operator-supplied identifiers before publishing them. Hashes also need
publication approval; a digest is not an authorization to disclose its source.

## Public provenance

| Field | Approved public value |
|---|---|
| Externally pinned full source Git SHA | [SHA or withheld] |
| Actual collector binary SHA256 | [digest or withheld] |
| Evaluator binary SHA256 | [digest or withheld] |
| Package version | [version or withheld] |
| Cargo.lock SHA256 | [digest or withheld] |
| Build profile, command and deliberate flags | [reviewed build context or withheld] |
| Rust/Cargo versions, target, OS/architecture | [reviewed context or withheld] |
| Outside-checkout native install smoke, by supported target | [verified / not run / failed] |
| Manifest SHA256 | [digest or withheld] |
| Recipe | [deepseek / glm / another explicitly approved recipe] |
| Claim profile and version | [approved public profile or withheld] |
| Default versus stress scope | [documented default / explicitly bounded stress case] |
| Workload or domain operation | [approved public identity or withheld] |
| Exact workload source SHA256 | [digest or withheld] |
| Normalized workload SHA256 | [digest or withheld] |
| Captured policy SHA256 and approved identifier | [pins or withheld] |
| Role-labelled A, B, A2 evidence fingerprints | [digests or withheld] |
| Implementation and exact source review | [implemented and reviewed / incomplete / pending] |
| CPU protocol verification | [passed / not run / failed] |
| Real source adapter exercise | [exercised / not run / failed / not applicable with reason] |
| Live backend qualification | [qualified within stated scope / not run / failed / not applicable with reason] |

Source Git revision, workload source bytes and normalized workload identity are
separate pins. A binary rebuilt from the same source need not have the same
binary digest. DeepSeek qualification does not qualify GLM, or vice versa.

Imported observations, operator declarations, CPU fixtures and exercised adapters
are different evidence. A completed parser or a successful import does not attest
that the declared operation ran. Record those distinctions for each claim, not
only once for the entire report.

## Claim-to-evidence map

Select the rows relevant to the proposed change. These are evidence requirements,
not assertions that every capability is installed or live-qualified. Include
every performance claim made by the PR or its README; explicitly mark unsupported
or missing evidence. Do not replace a missing claim with a favorable unrelated
serving result.

| Claim family | Required evidence and interpretation |
|---|---|
| Generation, graph, quantization or draft-setting serving impact | Matched pinned serving workloads and all prospective gates; link separate numerical/semantic quality checks. |
| Prefill and long context | Actual tokenizer and encoded input/output budgets, thinking/cache controls and matching latency or throughput observations; token counts are not byte limits. |
| Concurrency and throughput | Declared load, complete admitted waves and every lane; do not call correlated peers independent trials. |
| Mixed prefill/decode or staggered arrivals | Named per-lane controls and latency gates, matched solo cases, actual trigger/dispatch and required overlap evidence; retain never-triggered and never-overlapped attempts. |
| Cache reuse, history edit/branch/restore | Linked ordered history and actual supported cache evidence; zero hits, missing fields or a faster response do not establish reuse. |
| First output, fairness, tails or whole conversations | Precisely named observation, clock, population and sample unit; complete per-lane/per-step gates and prospective exposure. Withhold unsupported percentiles or sparse-tail claims. |
| Tool first output | First accepted name/argument delta separately from the fully validated fixed call and linked follow-up; no body/header timestamp substitute. |
| Prefix accounting, preemptions or draft acceptance | Full source/label identity, explicit denominators and continuity over the claimed acquisition; no attribution from unisolated server-wide counters. |
| CPU/RAM/device memory, power/KV headroom or capacity | Typed source scope, units, cadence, gaps and overhead; finite successful and failed cells. Sampled maximum is not true peak; largest tested success is not universal capacity. |
| Startup/bootstrap or cross-restart cache reload | Separate launch/listen/ready/first-valid events and clock domains; declared versus observed cold/warm state, stable cache identity and first reload before any warmup. |
| Kernel or collective performance | Pinned operation/input/layout/dtype/topology, actual synchronization/timing and correctness evidence; explicit collective byte denominator. Serving impact requires separate serving evidence. |
| Numerical, semantic or tool quality | Link the existing applicable quality checks and all failures; throughput and an envelope PASS do not establish correctness. |

For each selected row record:

- Approved public claim identifier and the exact changed axis: [value or withheld]
- Required workload/cell/lane/step or operation identities: [approved values or withheld]
- Required automatic gates versus reviewed descriptive evidence: [explicit handling]
- Four evidence statuses from the provenance table: [statuses, separately]
- Missing, failed, incompatible or inconclusive evidence: [approved bounded codes]
- Applicable quality-check evidence and disposition: [approved reference or withheld]
- Reviewer and maintainer disposition: [approved reference / pending / rejected]

## Prospective controls and finite allowances

- Workload, policy, collector, evaluator, serving and adapter identities pinned
  before collection: [yes / no / unverified; approved public pins only]
- Input/history byte limits and tokenizer/output controls checked against the
  actual encoded requests: [yes / no / unverified]
- Measured acquisition unit and warmup/prime exclusions: [approved definition]
- Total attempts, concurrency, input/retained bytes, output tokens and time
  allowances, including warmup and telemetry: [approved ceilings or withheld]
- Default configuration matches the README reference measurement:
  [yes / no / unverified; describe any approved exception]
- Traffic/device/restart window and restoration plan separately authorized:
  [yes / no / not applicable]
- Thresholds and eligibility rules declared prospectively, with no
  outcome-conditioned exclusions or replacement requests: [yes / no]

Required-hit and cache-disabled workloads cannot qualify one another. Differently
pinned cold and hit workloads are separate scopes, not a matched native A/B pair.
Inspect retained evidence before assuming a missing historical field describes
the current backend. Never record missing counters as zero.

## Approved change and evaluated scope

- Public change identifier: [approved public issue/commit identifier or withheld]
- Public change description: [independently reviewed public description or withheld]
- Declared policy approval before collection: [yes / no / unverified]
- Same captured policy in all roles: [yes / no / unknown]
- Distinct ordered acquisitions: [qualified / unqualified / unknown]
- Declared A/A2 reference identity: [declared_match / unqualified / unknown]
- Operator-reviewed nonoverlap and restoration: [reviewed / unverified]

Repeat a row for **every** required cell/lane/step or domain gate, including missing or unfavorable
gates. Use only public bundled cell IDs, metric enum values, numeric policy
thresholds and coverage counts from reviewed decision output. Do not include raw
sample values, private labels or copied diagnostic prose.

| Cell / lane / step or operation | Metric and sample unit | Regression tolerance | Reference spread limit | Expected / observed coverage by A, B, A2 | Gate decision | Bounded reason codes |
|---|---|---|---|---|---|---|
| [public identity] | [metric enum and unit] | [approved value and unit] | [approved value and unit] | [reviewed coverage counts] | [PASS / REGRESSION / INCONCLUSIVE / ERROR / review-only] | [codes only] |

## Decision and limitations

- Successfully parsed versioned decision envelope: [yes / no]
- Overall decision: [PASS / REGRESSION / INCONCLUSIVE / ERROR / no decision]
- Eligibility: [qualified / unqualified / unknown]
- Aggregate bounded reason codes: [codes only]
- Complete scope retained, including missing observations: [yes / no]
- Every failed, cancelled, missing-trigger, non-overlap and incomplete acquisition
  retained in private replay evidence: [yes / no]
- Offline validation command and exact evaluator identity: [approved command/pin]
- Corrupt, incompatible and unsupported required evidence rejected:
  [verified / unverified / failed]
- Public summary versus private replay inputs explicitly distinguished:
  [yes / no]

Only a complete, eligible PASS satisfies an automatic gate. INCONCLUSIVE is not
success merely because it is not REGRESSION; review-only evidence is not an
automatic PASS. A failure remains visible even if another acquisition succeeds.

## Maintainer scope and exceptions

- Required claim scopes agreed by the maintainer: [approved reference / pending]
- Every tradeoff or exception explicitly accepted or rejected:
  [approved disposition / pending]
- Broad adoption coverage reconciled, including remaining linked acceptance:
  [complete with evidence / incomplete]

Until coverage and maintainer agreement are complete, describe any accepted
subset as a subset. Do not propose mandatory all-claim adoption as finished.
Existing recipe launchers, numerical checks and specialized diagnostics remain
in place unless a separately reviewed equivalent replacement is demonstrated.

The decision concerns the declared observed-envelope policy and this workload,
binary, scope and acquisition set only. It is not a significance test, confidence
interval, causal attribution, equivalence test, universal no-regression result,
intelligence score or maximum-capacity claim. `compare` success does not mean
PASS. A missing decision envelope is not a decision. Matching declarations do
not independently prove model identity, nonoverlap or physical restoration.
Thinking controls are declarations, and `observe` does not prove a cold cache.
This reviewed summary omits private replay evidence; readers cannot reconstruct
the underlying observations from it alone.
