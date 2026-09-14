# Bounded provider accounting protocol (metrics version 2)

This document describes `metrics::v2`, the explicit second version of The Grill's
bounded provider-snapshot protocol (issue #51). Version 1 is frozen and
unchanged: same four-name allowlist, same receipt identity, same derived
`counters`/`acceptance` views, same bytes. Version 2 is opt-in per run with
`--metrics-version 2`.

## What these numbers are, and are not

| Evidence | Meaning | Usable as |
|---|---|---|
| `usage.prompt_tokens_details.cached_tokens` (request) | what the server reported for **one response** | request-level cache observation under its declared mechanism |
| version 1 snapshots | four server-wide series, matched by full label set | server-wide diagnostics only |
| version 2 snapshots | closed allowlist of server-wide counters and gauges | server-wide accounting with explicit continuity limits |
| attributed measurement | none of the above | requires a supported isolation/accounting contract and actual source evidence |

Version 2 never sums across ranks, replicas, engines or models, and never
reconciles server-wide counters against request usage. Every derived value keys
on one exact label identity (labels minus the selector label it is derived
from). The wave view repeats
`attribution = "server_wide_counter_delta_not_request_attributed"` and the fixed
caveat list, including
`accepted_tokens_include_terminal_accepted_work_that_was_not_emitted`: accepted
speculative tokens are not emitted tokens and must not be substituted for them.

## Selected names

The allowlist is closed and validated by exact comparison; names are never
discovered or fuzzy-matched at runtime. Sources are the pinned vLLM Prometheus
exporter (`vllm/v1/metrics/loggers.py`, `vllm/v1/spec_decode/metrics.py`,
`vllm/v1/metrics/reader.py`) and retained captures of that exporter. Exposition
spelling includes the exporter's `_total` suffix on counters.

| Series | Unit | Kind | Meaning |
|---|---|---|---|
| `vllm:num_requests_running` | requests | gauge | requests currently running in the engine |
| `vllm:num_requests_waiting` | requests | gauge | requests currently waiting to be scheduled |
| `vllm:num_preemptions_total` | preemptions | counter | cumulative engine preemptions |
| `vllm:prefix_cache_queries_total` | tokens | counter | prefix cache queried tokens |
| `vllm:prefix_cache_hits_total` | tokens | counter | prefix cache hit (cached) tokens |
| `vllm:external_prefix_cache_queries_total` | tokens | counter | KV-connector cross-instance cache queried tokens |
| `vllm:external_prefix_cache_hits_total` | tokens | counter | KV-connector cross-instance cache hit tokens |
| `vllm:prompt_tokens_total` | tokens | counter | prefill prompt tokens processed by the engine |
| `vllm:prompt_tokens_by_source_total{source}` | tokens | counter | prefill tokens split by the engine's reported source |
| `vllm:prompt_tokens_cached_total` | tokens | counter | cached prompt tokens (local plus external) |
| `vllm:spec_decode_num_drafts_total` | drafts | counter | speculative draft rounds |
| `vllm:spec_decode_num_draft_tokens_total` | tokens | counter | speculative draft tokens |
| `vllm:spec_decode_num_accepted_tokens_total` | tokens | counter | speculatively accepted tokens (terminal accepted work included) |
| `vllm:spec_decode_num_accepted_tokens_per_pos_total{position}` | tokens | counter | draft rounds whose accepted prefix reached this position |

The closed `source` set is `local_compute`, `local_cache_hit`,
`external_kv_transfer`. A selected series with an unknown `source` value or a
non-numeric `position` label is retained in the raw bytes and reported in the
wave's `excluded` list with a fixed reason; it never silently becomes a cache,
local-compute or per-position claim.

## Continuity, resets and epochs

Continuity is evaluated over the **entire captured acquisition**: every wave's
before and after snapshot, in order, including between-wave boundaries. For each
exact identity the acquisition view reports `observations`, `first`, `last`,
`delta`, the exporter `epoch` and one status:

| Status | Meaning |
|---|---|
| `continuous` | present in every required snapshot, non-decreasing, with an unchanged exporter epoch |
| `reset_observed` | a later observation is below an earlier one |
| `disappeared` / `appeared_later` | the identity is missing from a snapshot after / before its first appearance |
| `epoch_changed` / `epoch_missing` / `epoch_undeclared` | the exporter `_created` epoch for that counter is not attested as constant |
| `snapshot_incomplete` | at least one required snapshot is not `complete` |
| `no_snapshots` / `nonfinite_value` | no observation, or a nonfinite sample |

Two nondecreasing samples are explicitly **not** continuity, and never evidence
of zero preemptions. A reset can be followed by regrowth between two samples, so
counters additionally need the exporter's own creation-epoch gauge
(`<name>_created`) constant across every observation. The epoch gauges are
exporter attestations of a counter incarnation, not a substitute for a declared
isolation contract.

## Derived accounting

- Cache view (per identity): `queries`/`hits` deltas and `ratio = hits/queries`.
  Zero queries is `zero_queries` with no ratio (never clamped). `hits > queries`
  is `inconsistent`. `local_compute`, `local_cache_hits`,
  `external_kv_transfer` and `cached_tokens` are reported beside it so a
  "cache hit" claim is never aliased onto local compute.
- Draft view (per identity): `rounds`, `draft_tokens`, `accepted_tokens` and
  `token_acceptance = accepted/draft_tokens`. Accepted tokens include terminal
  accepted work that was not emitted.
- Per-position acceptance (per identity + position): `accepted` divided by the
  provider's own position-exposure denominator, `vllm:spec_decode_num_drafts_total`
  (the pinned provider documents the per-position vector as
  `spec_decode_num_accepted_tokens_per_pos / spec_decode_num_drafts`). Total
  draft tokens are never assumed to be the per-position denominator. Zero
  exposure is `zero_exposure` with no ratio; `accepted > exposure` is
  `inconsistent`.
- Accounting reconciliation (per identity, explicit rules, exact integer
  deltas): `prompt_tokens_total = local_compute + local_cache_hit +
  external_kv_transfer`; `prompt_tokens_cached_total = local_cache_hit +
  external_kv_transfer`; `sum(accepted_tokens_per_pos) =
  accepted_tokens_total`; the accepted-per-position vector is non-increasing in
  position. Each rule reports `consistent`, `inconsistent` or `unavailable`.
  These sums are within one identity only.

## Budgets, cancellation and offsets

Every scrape charges request, byte, time, series and overhead budgets; all are
finite, retained and re-validated offline:

- per scrape: 2 s deadline, 1 MiB body, 64 KiB line, 256 selected series,
  16 labels and 4096 label-set bytes per series;
- whole acquisition: 30 s charged time, 16 MiB retained raw bytes,
  twice the planned wave count in requests, `waves × 2 × 256` selected series
  and 1 s of scrape/parse overhead;
- exhaustion retains `skipped_budget` without another call, as in version 1;
- an in-flight scrape cancelled by the acquisition latch retains bounded partial
  raw bytes as `cancelled` and never counts as complete evidence.

Both snapshot offsets and the measured-lane origin are expressed against the
same local monotonic origin for that wave; wall timestamps are provenance only
and are never subtracted from monotonic offsets. Replay re-derives and checks
that the before scrape finished before the measured origin and that the after
scrape started after lane settlement, with checked `u64` arithmetic.

## Metrics credentials

`--metrics-auth-env NAME` names an environment variable holding a metrics-only
credential. `Plan.metrics` retains the **name only**. The value is resolved at
capture time, sent only on the metrics request, and never recorded. Version 1
never sends metrics authorization. Declaring the model's `--auth-env` name for
metrics is rejected before any dispatch, and declaring a metrics credential for
version 1 is rejected. Metrics snapshots remain excluded from the model
transport policy except for the declared credential.

## Required telemetry (policy integration contract)

Optional version 2 diagnostics never change performance eligibility. Required
evidence is a separate, explicit declaration owned by policy2:

```json
{"name":"vllm:num_preemptions_total",
 "labels":{"engine":"0","model_name":"..."},
 "predicate":"zero_counter_delta"}
```

- `name` must be a member of the closed allowlist (never an epoch gauge).
- `labels` is the exact identity: `engine` and `model_name` are free-form, and
  the metric's own selector (`source` or `position`) is required with a value
  from the closed set. Wildcards, unknown label names and empty requirements are
  invalid policy.
- `present` requires the series in every required snapshot (no attribution);
  `continuous_counter` additionally requires whole-capture continuity with an
  unchanged epoch; `zero_counter_delta` additionally requires an explicit
  `exclusive-single-acquisition` isolation declaration and an observed total of
  exactly zero.
- `metrics::v2::assess` returns one `RequirementResult` per requirement and an
  aggregate `AssessmentOutcome`:

| Outcome | Meaning | Required handling |
|---|---|---|
| `satisfied` | every promised observation holds | continue to the other gates |
| `refuted` | a promised zero total advanced | never PASS |
| `unavailable` (with a fixed reason such as `scope_not_isolated`, `reset_observed`, `epoch_changed`, `missing_required_snapshot`, `series_absent`, `contradictory_accounting`) | the promise is not observed | INCONCLUSIVE, never PASS |
| `error` (`no_metrics2`, `invalid_requirement`) | the promise cannot be evaluated | ERROR |

Required declarations are invalid policy without version 2 metrics, and a
version 1 summary never satisfies a requirement.

## Limits of this protocol

- Only the pinned vLLM exporter family is supported. Another provider's
  differently named counters are unsupported, not auto-discovered.
- Isolation is an operator declaration retained in the plan
  (`shared-server-unattributed` by default); the protocol does not claim to
  verify it, and with the default declaration zero-counter claims are withheld.
- Independent collector traffic in an acquisition is not sufficient evidence of
  a worker-counter reset, and an exporter identity that does not attest epochs
  cannot support continuity claims.
- Version 2 is implemented and CPU-fixture verified only; it has not been
  exercised against a live endpoint, and no serving window is implied.
