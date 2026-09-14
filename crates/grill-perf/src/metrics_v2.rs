//! Explicit version-2 provider accounting protocol (#51).
//!
//! Version 1 (`super`) is frozen: same allowlist, receipt identity, snapshot
//! fields and derived views. This module is an independently version-validated
//! successor: a closed selected-name allowlist with typed unit/meaning/source,
//! full label identities, per-acquisition continuity with exporter epochs,
//! finite scrape/request/byte/series/time/overhead budgets with bounded partial
//! cancellation, and cache/preemption/draft/per-position accounting.
//!
//! Grounding: every selected name and label is taken from the pinned vLLM
//! Prometheus exporter (`vllm/v1/metrics/loggers.py`,
//! `vllm/v1/spec_decode/metrics.py`, `vllm/v1/metrics/reader.py`) and from
//! retained live captures of that exporter. Names are never discovered,
//! fuzzy-matched or summed across ranks, replicas or models at runtime.
//!
//! Claims this protocol deliberately does NOT make: server-wide counters are
//! never attributed to this workload or reconciled against request usage;
//! a nondecreasing pair of samples is not continuity; accepted tokens are not
//! emitted tokens.

use super::{Budget, Reference, Result};
use crate::evidence;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::time::Duration;

// Selected series names, in the provider's exposition spelling. Counters carry
// the exporter's `_total` suffix; gauges do not.
const RUNNING: &str = "vllm:num_requests_running";
const WAITING: &str = "vllm:num_requests_waiting";
const PREEMPTIONS: &str = "vllm:num_preemptions_total";
const PREFIX_QUERIES: &str = "vllm:prefix_cache_queries_total";
const PREFIX_HITS: &str = "vllm:prefix_cache_hits_total";
const EXTERNAL_QUERIES: &str = "vllm:external_prefix_cache_queries_total";
const EXTERNAL_HITS: &str = "vllm:external_prefix_cache_hits_total";
const TOKENS_TOTAL: &str = "vllm:prompt_tokens_total";
const TOKENS_BY_SOURCE: &str = "vllm:prompt_tokens_by_source_total";
const TOKENS_CACHED: &str = "vllm:prompt_tokens_cached_total";
const DRAFTS: &str = "vllm:spec_decode_num_drafts_total";
const DRAFT_TOKENS: &str = "vllm:spec_decode_num_draft_tokens_total";
const ACCEPTED_TOKENS: &str = "vllm:spec_decode_num_accepted_tokens_total";
const ACCEPTED_POS: &str = "vllm:spec_decode_num_accepted_tokens_per_pos_total";

pub const ALLOWLIST: [&str; 14] = [
    RUNNING,
    WAITING,
    PREEMPTIONS,
    PREFIX_QUERIES,
    PREFIX_HITS,
    EXTERNAL_QUERIES,
    EXTERNAL_HITS,
    TOKENS_TOTAL,
    TOKENS_BY_SOURCE,
    TOKENS_CACHED,
    DRAFTS,
    DRAFT_TOKENS,
    ACCEPTED_TOKENS,
    ACCEPTED_POS,
];

// Exporter creation-epoch gauges (`_created`): the provider's own incarnation
// timestamp per counter series. A change means a new counter incarnation, so
// continuity claims become unavailable. These are exporter attestations, not a
// substitute for a declared isolation contract.
pub const EPOCHS: [&str; 12] = [
    "vllm:num_preemptions_created",
    "vllm:prefix_cache_queries_created",
    "vllm:prefix_cache_hits_created",
    "vllm:external_prefix_cache_queries_created",
    "vllm:external_prefix_cache_hits_created",
    "vllm:prompt_tokens_created",
    "vllm:prompt_tokens_by_source_created",
    "vllm:prompt_tokens_cached_created",
    "vllm:spec_decode_num_drafts_created",
    "vllm:spec_decode_num_draft_tokens_created",
    "vllm:spec_decode_num_accepted_tokens_created",
    "vllm:spec_decode_num_accepted_tokens_per_pos_created",
];
const EPOCH_PAIRS: [(&str, &str); 12] = [
    (PREEMPTIONS, "vllm:num_preemptions_created"),
    (PREFIX_QUERIES, "vllm:prefix_cache_queries_created"),
    (PREFIX_HITS, "vllm:prefix_cache_hits_created"),
    (EXTERNAL_QUERIES, "vllm:external_prefix_cache_queries_created"),
    (EXTERNAL_HITS, "vllm:external_prefix_cache_hits_created"),
    (TOKENS_TOTAL, "vllm:prompt_tokens_created"),
    (TOKENS_BY_SOURCE, "vllm:prompt_tokens_by_source_created"),
    (TOKENS_CACHED, "vllm:prompt_tokens_cached_created"),
    (DRAFTS, "vllm:spec_decode_num_drafts_created"),
    (DRAFT_TOKENS, "vllm:spec_decode_num_draft_tokens_created"),
    (ACCEPTED_TOKENS, "vllm:spec_decode_num_accepted_tokens_created"),
    (
        ACCEPTED_POS,
        "vllm:spec_decode_num_accepted_tokens_per_pos_created",
    ),
];

// Closed selector value set of `prompt_tokens_by_source_total`.
pub const SOURCES: [&str; 3] = [
    "external_kv_transfer",
    "local_cache_hit",
    "local_compute",
];
pub const ISOLATION_EXCLUSIVE: &str = "exclusive-single-acquisition";
pub const ISOLATION_SHARED: &str = "shared-server-unattributed";

pub const CAVEATS: [&str; 4] = [
    "server_wide_counters_are_not_attributed_to_this_workload",
    "request_usage_and_provider_counters_are_not_reconciled",
    "accepted_tokens_include_terminal_accepted_work_that_was_not_emitted",
    "no_ranking_model_or_replica_summing",
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Counter,
    Gauge,
}

struct Spec {
    name: &'static str,
    unit: &'static str,
    kind: Kind,
    /// Selector label that identifies a member of the metric's own closed set.
    selector: Option<&'static str>,
    meaning: &'static str,
}

const SPECS: [Spec; 14] = [
    Spec {
        name: RUNNING,
        unit: "requests",
        kind: Kind::Gauge,
        selector: None,
        meaning: "requests currently running in the engine",
    },
    Spec {
        name: WAITING,
        unit: "requests",
        kind: Kind::Gauge,
        selector: None,
        meaning: "requests currently waiting to be scheduled",
    },
    Spec {
        name: PREEMPTIONS,
        unit: "preemptions",
        kind: Kind::Counter,
        selector: None,
        meaning: "cumulative engine preemptions",
    },
    Spec {
        name: PREFIX_QUERIES,
        unit: "tokens",
        kind: Kind::Counter,
        selector: None,
        meaning: "prefix cache queried tokens",
    },
    Spec {
        name: PREFIX_HITS,
        unit: "tokens",
        kind: Kind::Counter,
        selector: None,
        meaning: "prefix cache hit (cached) tokens",
    },
    Spec {
        name: EXTERNAL_QUERIES,
        unit: "tokens",
        kind: Kind::Counter,
        selector: None,
        meaning: "KV-connector cross-instance cache queried tokens",
    },
    Spec {
        name: EXTERNAL_HITS,
        unit: "tokens",
        kind: Kind::Counter,
        selector: None,
        meaning: "KV-connector cross-instance cache hit tokens",
    },
    Spec {
        name: TOKENS_TOTAL,
        unit: "tokens",
        kind: Kind::Counter,
        selector: None,
        meaning: "prefill prompt tokens processed by the engine",
    },
    Spec {
        name: TOKENS_BY_SOURCE,
        unit: "tokens",
        kind: Kind::Counter,
        selector: Some("source"),
        meaning: "prefill prompt tokens split by the engine's reported source",
    },
    Spec {
        name: TOKENS_CACHED,
        unit: "tokens",
        kind: Kind::Counter,
        selector: None,
        meaning: "cached prompt tokens (local plus external)",
    },
    Spec {
        name: DRAFTS,
        unit: "drafts",
        kind: Kind::Counter,
        selector: None,
        meaning: "speculative draft rounds",
    },
    Spec {
        name: DRAFT_TOKENS,
        unit: "tokens",
        kind: Kind::Counter,
        selector: None,
        meaning: "speculative draft tokens",
    },
    Spec {
        name: ACCEPTED_TOKENS,
        unit: "tokens",
        kind: Kind::Counter,
        selector: None,
        meaning: "speculatively accepted tokens, including terminal accepted work not emitted",
    },
    Spec {
        name: ACCEPTED_POS,
        unit: "tokens",
        kind: Kind::Counter,
        selector: Some("position"),
        meaning: "draft rounds whose accepted prefix reached this position",
    },
];

fn spec(name: &str) -> Option<&'static Spec> {
    SPECS.iter().find(|spec| spec.name == name)
}
fn is_counter(name: &str) -> bool {
    spec(name).is_some_and(|spec| spec.kind == Kind::Counter)
}
fn is_epoch(name: &str) -> bool {
    EPOCHS.contains(&name)
}
fn kind_of(name: &str) -> &'static str {
    if is_epoch(name) {
        "epoch"
    } else if is_counter(name) {
        "counter"
    } else {
        "gauge"
    }
}
fn unit_of(name: &str) -> &'static str {
    spec(name).map_or("unix_seconds", |spec| spec.unit)
}
fn epoch_of(name: &str) -> Option<&'static str> {
    EPOCH_PAIRS
        .iter()
        .find(|(counter, _)| *counter == name)
        .map(|(_, epoch)| *epoch)
}
fn selected(name: &str) -> bool {
    spec(name).is_some() || is_epoch(name)
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub version: u32,
    pub endpoint: String,
    pub allowlist: [String; 14],
    pub epochs: [String; 12],
    pub scope: String,
    pub source: String,
    pub isolation: String,
    pub auth_env: Option<String>,
    pub cadence: String,
    pub deadline_us: u64,
    pub body_bytes: usize,
    pub line_bytes: usize,
    pub series: usize,
    pub labels_per_series: usize,
    pub label_bytes_per_series: usize,
    pub run_budget_us: u64,
    pub retained_raw_bytes: usize,
    pub max_requests: usize,
    pub max_series: usize,
    pub max_overhead_us: u64,
}
impl Config {
    pub fn new(endpoint: String, waves: usize, auth_env: Option<String>) -> Self {
        Self {
            version: 2,
            endpoint,
            allowlist: ALLOWLIST.map(str::to_owned),
            epochs: EPOCHS.map(str::to_owned),
            scope: "server-wide-full-labels-not-workload-attributed".into(),
            source: "vllm:prometheus-exporter".into(),
            isolation: ISOLATION_SHARED.into(),
            auth_env,
            cadence: "before-finishes-before-measured-origin-after-starts-after-settlement".into(),
            deadline_us: 2_000_000,
            body_bytes: 1024 * 1024,
            line_bytes: 64 * 1024,
            series: 256,
            labels_per_series: 16,
            label_bytes_per_series: 4096,
            run_budget_us: 30_000_000,
            retained_raw_bytes: 16 * 1024 * 1024,
            max_requests: waves * 2,
            max_series: waves * 2 * 256,
            max_overhead_us: 1_000_000,
        }
    }
    pub fn validate(
        &self,
        waves: usize,
        local_http: bool,
        model_auth_env: Option<&str>,
    ) -> Result<()> {
        if let Some(name) = &self.auth_env {
            if name.is_empty()
                || name.len() > 64
                || !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
            {
                return Err("invalid metrics credential environment-variable name".into());
            }
            if model_auth_env == Some(name.as_str()) {
                return Err(
                    "metrics diagnostics must not reuse the model credential variable".into(),
                );
            }
        }
        if self.isolation != ISOLATION_EXCLUSIVE && self.isolation != ISOLATION_SHARED {
            return Err("unsupported metrics isolation declaration".into());
        }
        let url = crate::wire::endpoint(&self.endpoint, local_http)?;
        if *self != Self::new(url.to_string(), waves, self.auth_env.clone()) {
            return Err("unsupported metrics protocol or bounds".into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Consumption {
    pub snapshots: usize,
    pub series: usize,
    pub overhead_us: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Complete,
    ParseError,
    TransportError,
    HttpError,
    Unsupported,
    Deadline,
    BodyLimit,
    SkippedBudget,
    /// The acquisition was cancelled while this scrape was in flight. Bounded
    /// partial raw bytes are retained and are never continuity evidence.
    Cancelled,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Snapshot {
    /// Wall provenance only; never subtracted from monotonic offsets.
    pub started_unix_ms: u64,
    /// Scrape start relative to this wave's local monotonic origin.
    pub scrape_offset_us: u64,
    pub scrape_us: u64,
    pub duration_us: u64,
    pub charged_us: u64,
    /// Scrape plus parse overhead inside `duration_us` (duration - scrape).
    pub overhead_us: u64,
    pub allowance_us: u64,
    pub status: Status,
    pub error: Option<String>,
    pub http_status: Option<u16>,
    pub raw_bytes: usize,
    pub raw_sha256: String,
    pub series: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Receipt {
    pub version: u32,
    pub plan_sha256: String,
    pub wave: u32,
    pub before: Snapshot,
    pub after: Snapshot,
    /// Measured-lane origin relative to the same local monotonic origin used by
    /// both snapshot offsets.
    pub measured_origin_offset_us: u64,
    pub measured_origin_unix_ms: u64,
    pub measured_duration_us: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct Identity {
    pub name: String,
    pub labels: BTreeMap<String, String>,
}
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Series {
    #[serde(flatten)]
    pub identity: Identity,
    pub unit: &'static str,
    pub kind: &'static str,
    pub value: f64,
}
#[derive(Clone, Debug, Serialize)]
pub struct Sample {
    pub delta: Option<f64>,
    pub status: &'static str,
}
#[derive(Clone, Debug, Serialize)]
pub struct CounterDelta {
    #[serde(flatten)]
    pub identity: Identity,
    pub unit: &'static str,
    pub before: Option<f64>,
    pub after: Option<f64>,
    pub delta: Option<f64>,
    pub status: &'static str,
}
#[derive(Clone, Debug, Serialize)]
pub struct CacheView {
    pub labels: BTreeMap<String, String>,
    pub queries: Sample,
    pub hits: Sample,
    pub ratio: Option<f64>,
    pub status: &'static str,
    pub reason: Option<&'static str>,
    pub external_queries: Sample,
    pub external_hits: Sample,
    pub local_compute: Sample,
    pub local_cache_hits: Sample,
    pub external_kv_transfer: Sample,
    pub cached_tokens: Sample,
    pub attribution: &'static str,
}
#[derive(Clone, Debug, Serialize)]
pub struct DraftView {
    pub labels: BTreeMap<String, String>,
    pub rounds: Sample,
    pub draft_tokens: Sample,
    pub accepted_tokens: Sample,
    pub token_acceptance: Option<f64>,
    pub status: &'static str,
    pub reason: Option<&'static str>,
    pub accounting: &'static str,
}
#[derive(Clone, Debug, Serialize)]
pub struct PositionView {
    pub labels: BTreeMap<String, String>,
    pub position: u32,
    pub accepted: Sample,
    pub exposure: Sample,
    pub exposure_source: &'static str,
    pub acceptance: Option<f64>,
    pub status: &'static str,
    pub reason: Option<&'static str>,
}
#[derive(Clone, Debug, Serialize)]
pub struct AccountingCheck {
    pub labels: BTreeMap<String, String>,
    pub rule: &'static str,
    pub observed: Option<f64>,
    pub expected: Option<f64>,
    pub status: &'static str,
}
#[derive(Clone, Debug, Serialize)]
pub struct Excluded {
    pub name: String,
    pub labels: BTreeMap<String, String>,
    pub reason: &'static str,
}
#[derive(Clone, Debug, Serialize)]
pub struct WaveSummary {
    pub receipt: Receipt,
    pub overhead_us: u64,
    pub before_series: Vec<Series>,
    pub after_series: Vec<Series>,
    pub counters: Vec<CounterDelta>,
    pub preemptions: Vec<CounterDelta>,
    pub cache: Vec<CacheView>,
    pub draft: Vec<DraftView>,
    pub positions: Vec<PositionView>,
    pub accounting: Vec<AccountingCheck>,
    pub excluded: Vec<Excluded>,
    pub caveats: Vec<&'static str>,
}
#[derive(Clone, Debug, Serialize)]
pub struct Epoch {
    pub series: Option<&'static str>,
    pub observations: usize,
    pub value: Option<f64>,
    pub status: &'static str,
}
#[derive(Clone, Debug, Serialize)]
pub struct SeriesContinuity {
    #[serde(flatten)]
    pub identity: Identity,
    pub kind: &'static str,
    pub observations: usize,
    pub first: Option<f64>,
    pub last: Option<f64>,
    pub delta: Option<f64>,
    pub epoch: Option<Epoch>,
    pub status: &'static str,
}
#[derive(Clone, Debug, Serialize)]
pub struct Acquisition {
    pub snapshots: usize,
    pub complete: usize,
    pub series: Vec<SeriesContinuity>,
    pub accounting: Vec<AccountingCheck>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Requirement {
    pub name: String,
    pub labels: BTreeMap<String, String>,
    pub predicate: Predicate,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Predicate {
    Present,
    ContinuousCounter,
    ZeroCounterDelta,
}
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AssessmentOutcome {
    Satisfied,
    Refuted,
    Unavailable,
    Error,
}
#[derive(Clone, Debug, Serialize)]
pub struct RequirementResult {
    pub name: String,
    pub labels: BTreeMap<String, String>,
    pub predicate: Predicate,
    pub outcome: &'static str,
    pub reason: Option<&'static str>,
    pub detail: String,
}
#[derive(Clone, Debug, Serialize)]
pub struct Assessment {
    pub outcome: AssessmentOutcome,
    pub results: Vec<RequirementResult>,
}

fn position(text: &str) -> Option<u32> {
    if text.is_empty()
        || text.len() > 10
        || !text.bytes().all(|b| b.is_ascii_digit())
        || (text.len() > 1 && text.starts_with('0'))
    {
        return None;
    }
    text.parse::<u32>().ok()
}

pub fn parse(raw: &[u8], config: &Config) -> Result<Vec<Series>> {
    if raw.len() > config.body_bytes {
        return Err("metrics body exceeds bound".into());
    }
    let text = std::str::from_utf8(raw).map_err(|_| "metrics body is not UTF-8")?;
    let mut seen = std::collections::BTreeSet::new();
    let mut series = Vec::new();
    for line in text.lines() {
        if line.len() > config.line_bytes {
            return Err("metrics line exceeds bound".into());
        }
        let line = line.trim_matches([' ', '\t', '\r']);
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let end = line.find(['{', ' ', '\t']).unwrap_or(line.len());
        let name = &line[..end];
        if !selected(name) {
            continue;
        }
        if seen.len() >= config.series {
            return Err("metrics selected series exceed bound".into());
        }
        let (labels, value) = if let Some(rest) = line[end..].strip_prefix('{') {
            super::labels(rest, config.labels_per_series, config.label_bytes_per_series)?
        } else {
            (BTreeMap::new(), &line[end..])
        };
        if !value.starts_with([' ', '\t']) {
            return Err("missing metrics sample separator".into());
        }
        let mut fields = value.split_whitespace();
        let value = fields
            .next()
            .ok_or("missing metrics sample value")?
            .parse::<f64>()
            .map_err(|_| "invalid metrics number")?;
        if !value.is_finite() {
            return Err("nonfinite metrics number".into());
        }
        if value < 0.0 {
            return Err("negative metrics value".into());
        }
        if let Some(timestamp) = fields.next() {
            timestamp
                .parse::<i64>()
                .map_err(|_| "invalid metrics timestamp")?;
        }
        if fields.next().is_some() {
            return Err("unsupported metrics sample suffix".into());
        }
        let identity = Identity {
            name: name.into(),
            labels,
        };
        if !seen.insert(identity.clone()) {
            return Err("duplicate metrics series".into());
        }
        series.push(Series {
            identity,
            unit: unit_of(name),
            kind: kind_of(name),
            value,
        });
    }
    series.sort_by(|a, b| a.identity.cmp(&b.identity));
    Ok(series)
}

pub async fn scrape(
    client: &reqwest::Client,
    config: &Config,
    budget: &mut Budget,
    dir: &Path,
    side: &str,
    ctx: &super::CaptureCtx<'_>,
) -> Result<Snapshot> {
    let start = std::time::Instant::now();
    let mut snapshot = Snapshot {
        started_unix_ms: super::unix_ms(),
        scrape_offset_us: super::offset_us(ctx.origin, start)?,
        scrape_us: 0,
        duration_us: 0,
        charged_us: 0,
        overhead_us: 0,
        allowance_us: 0,
        status: Status::SkippedBudget,
        error: Some("whole-run telemetry budget exhausted".into()),
        http_status: None,
        raw_bytes: 0,
        raw_sha256: String::new(),
        series: 0,
    };
    let mut raw = Vec::new();
    if !budget.exhausted_v2(config) {
        let allowance = config
            .deadline_us
            .min(config.run_budget_us - budget.charged_us);
        snapshot.allowance_us = allowance;
        let operation = async {
            let mut request = client
                .get(&config.endpoint)
                .header("accept", "text/plain; version=0.0.4")
                .header("accept-encoding", "identity");
            if let Some(auth) = ctx.auth {
                request = request.header(reqwest::header::AUTHORIZATION, auth.clone());
            }
            let mut response = request
                .send()
                .await
                .map_err(|_| (Status::TransportError, "metrics request failed".to_owned()))?;
            snapshot.http_status = Some(response.status().as_u16());
            let encoded = response
                .headers()
                .get_all("content-encoding")
                .iter()
                .any(|value| match value.to_str() {
                    Ok(text) => text
                        .split(',')
                        .any(|text| !text.trim().eq_ignore_ascii_case("identity")),
                    Err(_) => true,
                });
            while let Some(chunk) = response.chunk().await.map_err(|_| {
                (
                    Status::TransportError,
                    "metrics body transfer failed".to_owned(),
                )
            })? {
                if ctx
                    .cancelled
                    .is_some_and(|flag| flag.load(Ordering::SeqCst))
                {
                    return Err((
                        Status::Cancelled,
                        "acquisition cancelled during metrics scrape".into(),
                    ));
                }
                if crate::wire::us(start) >= allowance {
                    return Err((Status::Deadline, "metrics scrape deadline expired".into()));
                }
                let remaining = config.body_bytes - raw.len();
                raw.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
                if chunk.len() > remaining {
                    return Err((Status::BodyLimit, "metrics body exceeds bound".into()));
                }
            }
            if snapshot.http_status != Some(200) {
                return Err((
                    Status::HttpError,
                    "metrics HTTP status is not successful".into(),
                ));
            }
            if encoded {
                return Err((
                    Status::Unsupported,
                    "metrics content encoding is unsupported".into(),
                ));
            }
            snapshot.series = parse(&raw, config)
                .map_err(|error| (Status::ParseError, error))?
                .len();
            if crate::wire::us(start) >= allowance {
                return Err((Status::Deadline, "metrics scrape deadline expired".into()));
            }
            Ok::<(), (Status, String)>(())
        };
        match tokio::time::timeout(Duration::from_micros(allowance), operation).await {
            Ok(Ok(())) => {
                snapshot.status = Status::Complete;
                snapshot.error = None;
            }
            Ok(Err((status, error))) => {
                snapshot.status = status;
                snapshot.error = Some(error);
            }
            Err(_) => {
                snapshot.status = Status::Deadline;
                snapshot.error = Some("metrics scrape deadline expired".into());
            }
        }
        snapshot.scrape_us = crate::wire::us(start).max(1);
    }
    snapshot.raw_bytes = raw.len();
    snapshot.raw_sha256 = evidence::digest(&raw);
    evidence::write(&dir.join(format!("metrics-{side}.bin")), &raw)?;
    if snapshot.status != Status::SkippedBudget {
        snapshot.duration_us = crate::wire::us(start).max(snapshot.scrape_us);
        snapshot.charged_us = snapshot.duration_us;
        snapshot.overhead_us = snapshot.duration_us - snapshot.scrape_us;
    }
    if snapshot.status != Status::Complete {
        snapshot.series = 0;
    }
    budget.retain_v2(config, &snapshot)?;
    Ok(snapshot)
}

pub fn publish(dir: &Path, receipt: &Receipt) -> Result<Reference> {
    let bytes = serde_json::to_vec_pretty(receipt).map_err(|error| error.to_string())?;
    if bytes.len() > super::RECEIPT_CAP {
        return Err("metrics receipt exceeds bound".into());
    }
    evidence::write(&dir.join("metrics.json"), &bytes)?;
    evidence::sync(dir)?;
    Ok(Reference {
        file: "metrics.json".into(),
        sha256: evidence::digest(&bytes),
        overhead_us: 0,
    })
}

/// Identical to [`super::load`] with the version-2 receipt, budgets and views.
pub fn load(
    dir: &Path,
    reference: &Reference,
    config: &Config,
    plan_hash: &str,
    wave: u32,
    budget: &mut Budget,
) -> Result<WaveSummary> {
    if reference.file != "metrics.json" {
        return Err("invalid metrics companion path".into());
    }
    let bytes = evidence::read(&dir.join("metrics.json"), super::RECEIPT_CAP)?;
    if evidence::digest(&bytes) != reference.sha256 {
        return Err("metrics companion hash mismatch".into());
    }
    let receipt: Receipt =
        serde_json::from_slice(&bytes).map_err(|error| format!("invalid metrics receipt: {error}"))?;
    if receipt.version != 2 || receipt.plan_sha256 != plan_hash || receipt.wave != wave {
        return Err("metrics companion lineage mismatch".into());
    }
    let before_end = receipt
        .before
        .scrape_offset_us
        .checked_add(receipt.before.duration_us)
        .ok_or("metrics offset overflow")?;
    let measured_end = receipt
        .measured_origin_offset_us
        .checked_add(receipt.measured_duration_us)
        .ok_or("metrics offset overflow")?;
    if before_end > receipt.measured_origin_offset_us
        || receipt.before.scrape_offset_us >= receipt.after.scrape_offset_us
        || measured_end > receipt.after.scrape_offset_us
        || receipt.measured_duration_us == 0
    {
        return Err("metrics measurement boundary contradicts wave settlement".into());
    }
    let snapshots_us = receipt
        .before
        .duration_us
        .checked_add(receipt.after.duration_us)
        .ok_or("metrics duration overflow")?;
    let mut parsed = Vec::with_capacity(2);
    for (side, snapshot) in [("before", &receipt.before), ("after", &receipt.after)] {
        budget.retain_v2(config, snapshot)?;
        let raw = evidence::read(&dir.join(format!("metrics-{side}.bin")), config.body_bytes)?;
        if raw.len() != snapshot.raw_bytes || evidence::digest(&raw) != snapshot.raw_sha256 {
            return Err("metrics raw evidence hash mismatch".into());
        }
        let samples = parse(&raw, config);
        match snapshot.status {
            Status::Complete if snapshot.http_status == Some(200) => {
                let samples = samples?;
                if samples.len() != snapshot.series {
                    return Err("metrics series count contradicts raw evidence".into());
                }
                parsed.push(samples);
            }
            Status::ParseError if snapshot.http_status == Some(200) && samples.is_err() => {
                parsed.push(Vec::new());
            }
            Status::Complete | Status::ParseError => {
                return Err("metrics parser status contradicts raw evidence".into());
            }
            Status::HttpError if snapshot.http_status.is_none_or(|status| status == 200) => {
                return Err("invalid metrics HTTP failure".into());
            }
            Status::BodyLimit if raw.len() != config.body_bytes => {
                return Err("invalid metrics body limit".into());
            }
            _ => parsed.push(Vec::new()),
        }
    }
    budget.finish_wave(reference.overhead_us, snapshots_us)?;
    let after_series = parsed.pop().ok_or("missing after metrics")?;
    let before_series = parsed.pop().ok_or("missing before metrics")?;
    let complete =
        receipt.before.status == Status::Complete && receipt.after.status == Status::Complete;
    let views = derive(&before_series, &after_series, complete);
    Ok(WaveSummary {
        receipt,
        overhead_us: reference.overhead_us,
        before_series,
        after_series,
        counters: views.counters,
        preemptions: views.preemptions,
        cache: views.cache,
        draft: views.draft,
        positions: views.positions,
        accounting: views.accounting,
        excluded: views.excluded,
        caveats: CAVEATS.to_vec(),
    })
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    name: String,
    labels: BTreeMap<String, String>,
}

fn base(labels: &BTreeMap<String, String>, drop: &[&str]) -> BTreeMap<String, String> {
    labels
        .iter()
        .filter(|(name, _)| !drop.contains(&name.as_str()))
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect()
}

type Slices = BTreeMap<Key, (Option<f64>, Option<f64>)>;

fn slices(before: &[Series], after: &[Series], drop: &[&str]) -> Slices {
    let mut map = Slices::new();
    for (index, list) in [before, after].into_iter().enumerate() {
        for series in list {
            let key = Key {
                name: series.identity.name.clone(),
                labels: base(&series.identity.labels, drop),
            };
            let entry = map.entry(key).or_insert((None, None));
            if index == 0 {
                entry.0 = Some(series.value);
            } else {
                entry.1 = Some(series.value);
            }
        }
    }
    map
}

fn lookup(map: &Slices, name: &str, labels: &BTreeMap<String, String>) -> (Option<f64>, Option<f64>) {
    map.iter()
        .find(|(key, _)| key.name == name && key.labels == *labels)
        .map(|(_, value)| *value)
        .unwrap_or((None, None))
}

fn sample(map: &Slices, name: &str, labels: &BTreeMap<String, String>, complete: bool) -> Sample {
    let (before, after) = lookup(map, name, labels);
    let (delta, status) = if !complete {
        (None, "not_complete")
    } else {
        match (before, after) {
            (None, Some(_)) => (None, "missing_before"),
            (Some(_), None) => (None, "missing_after"),
            (None, None) => (None, "missing"),
            (Some(before), Some(after)) if after < before => (None, "reset_observed"),
            (Some(before), Some(after)) => (Some(after - before), "available"),
        }
    };
    Sample { delta, status }
}

fn by_source(
    full: &Slices,
    source: &str,
    labels: &BTreeMap<String, String>,
    complete: bool,
) -> Sample {
    let mut with = labels.clone();
    with.insert("source".into(), source.into());
    sample(full, TOKENS_BY_SOURCE, &with, complete)
}

fn compare(
    labels: &BTreeMap<String, String>,
    rule: &'static str,
    observed: Option<f64>,
    expected: Option<f64>,
    complete: bool,
) -> AccountingCheck {
    let status = if !complete {
        "unavailable"
    } else {
        match (observed, expected) {
            (Some(observed), Some(expected)) if observed == expected => "consistent",
            (Some(_), Some(_)) => "inconsistent",
            _ => "unavailable",
        }
    };
    AccountingCheck {
        labels: labels.clone(),
        rule,
        observed,
        expected,
        status,
    }
}

fn accounting(base_map: &Slices, full: &Slices, complete: bool) -> Vec<AccountingCheck> {
    let mut identities: Vec<BTreeMap<String, String>> = Vec::new();
    for (key, _) in base_map.iter() {
        if [
            TOKENS_TOTAL,
            TOKENS_CACHED,
            TOKENS_BY_SOURCE,
            ACCEPTED_TOKENS,
            ACCEPTED_POS,
        ]
        .contains(&key.name.as_str())
            && !identities.contains(&key.labels)
        {
            identities.push(key.labels.clone());
        }
    }
    identities.sort();
    let mut checks = Vec::new();
    for labels in identities {
        let sources = by_source(full, "local_compute", &labels, complete)
            .delta
            .zip(by_source(full, "local_cache_hit", &labels, complete).delta)
            .zip(by_source(full, "external_kv_transfer", &labels, complete).delta)
            .map(|((compute, cache), external)| compute + cache + external);
        checks.push(compare(
            &labels,
            "prompt_tokens_total_equals_sum_of_reported_sources",
            sample(base_map, TOKENS_TOTAL, &labels, complete).delta,
            sources,
            complete,
        ));
        let cached = by_source(full, "local_cache_hit", &labels, complete)
            .delta
            .zip(by_source(full, "external_kv_transfer", &labels, complete).delta)
            .map(|(cache, external)| cache + external);
        checks.push(compare(
            &labels,
            "prompt_tokens_cached_total_equals_local_cache_hit_plus_external_kv_transfer",
            sample(base_map, TOKENS_CACHED, &labels, complete).delta,
            cached,
            complete,
        ));
        let mut positions: Vec<(u32, Option<f64>)> = full
            .iter()
            .filter(|(key, _)| key.name == ACCEPTED_POS && base(&key.labels, &["position"]) == labels)
            .filter_map(|(key, _)| {
                let draft_position = position(key.labels.get("position")?)?;
                Some((
                    draft_position,
                    sample(full, ACCEPTED_POS, &key.labels, complete).delta,
                ))
            })
            .collect();
        positions.sort_by_key(|(position, _)| *position);
        let position_total = positions
            .iter()
            .map(|(_, delta)| *delta)
            .collect::<Option<Vec<f64>>>()
            .map(|values| values.iter().sum());
        checks.push(compare(
            &labels,
            "sum_accepted_tokens_per_pos_equals_spec_decode_num_accepted_tokens_total",
            sample(base_map, ACCEPTED_TOKENS, &labels, complete).delta,
            position_total,
            complete,
        ));
        let violations = positions
            .windows(2)
            .filter(|pair| match (pair[0].1, pair[1].1) {
                (Some(lower), Some(higher)) => higher > lower,
                _ => false,
            })
            .count();
        checks.push(compare(
            &labels,
            "accepted_tokens_per_pos_is_non_increasing_in_position",
            complete.then_some(violations as f64),
            complete.then_some(0.0),
            complete,
        ));
    }
    checks
}

pub struct Views {
    pub counters: Vec<CounterDelta>,
    pub preemptions: Vec<CounterDelta>,
    pub cache: Vec<CacheView>,
    pub draft: Vec<DraftView>,
    pub positions: Vec<PositionView>,
    pub accounting: Vec<AccountingCheck>,
    pub excluded: Vec<Excluded>,
}

pub fn derive(before: &[Series], after: &[Series], complete: bool) -> Views {
    let full = slices(before, after, &[]);
    let base_map = slices(before, after, &["source", "position"]);
    let mut excluded: BTreeMap<(String, BTreeMap<String, String>), &'static str> = BTreeMap::new();
    for series in before.iter().chain(after.iter()) {
        let reason = if series.identity.name == TOKENS_BY_SOURCE {
            match series.identity.labels.get("source") {
                Some(source) if SOURCES.contains(&source.as_str()) => continue,
                Some(_) => "unsupported_source_value",
                None => "missing_source_selector",
            }
        } else if series.identity.name == ACCEPTED_POS {
            match series
                .identity
                .labels
                .get("position")
                .and_then(|text| position(text))
            {
                Some(_) => continue,
                None => "invalid_position_label",
            }
        } else {
            continue;
        };
        excluded.insert(
            (series.identity.name.clone(), series.identity.labels.clone()),
            reason,
        );
    }
    let counters: Vec<CounterDelta> = full
        .iter()
        .filter(|(key, _)| is_counter(&key.name))
        .map(|(key, (before, after))| {
            let (delta, status) = if !complete {
                (None, "not_complete")
            } else {
                match (before, after) {
                    (None, Some(_)) => (None, "missing_before"),
                    (Some(_), None) => (None, "missing_after"),
                    (None, None) => (None, "missing"),
                    (Some(before), Some(after)) if after < before => (None, "reset_observed"),
                    (Some(before), Some(after)) => (Some(after - before), "available"),
                }
            };
            CounterDelta {
                identity: Identity {
                    name: key.name.clone(),
                    labels: key.labels.clone(),
                },
                unit: unit_of(&key.name),
                before: *before,
                after: *after,
                delta,
                status,
            }
        })
        .collect();
    let preemptions = counters
        .iter()
        .filter(|counter| counter.identity.name == PREEMPTIONS)
        .cloned()
        .collect();
    let mut cache_labels: Vec<BTreeMap<String, String>> = Vec::new();
    for (key, _) in base_map.iter() {
        if [
            PREFIX_QUERIES,
            PREFIX_HITS,
            EXTERNAL_QUERIES,
            EXTERNAL_HITS,
            TOKENS_BY_SOURCE,
            TOKENS_CACHED,
        ]
        .contains(&key.name.as_str())
            && !cache_labels.contains(&key.labels)
        {
            cache_labels.push(key.labels.clone());
        }
    }
    cache_labels.sort();
    let cache: Vec<CacheView> = cache_labels
        .into_iter()
        .map(|labels| {
            let queries = sample(&base_map, PREFIX_QUERIES, &labels, complete);
            let hits = sample(&base_map, PREFIX_HITS, &labels, complete);
            let (ratio, status, reason) = if !complete {
                (None, "not_complete", Some("snapshot_not_complete"))
            } else if queries.status != "available" {
                (None, "unavailable", Some(queries.status))
            } else if hits.status != "available" {
                (None, "unavailable", Some(hits.status))
            } else if queries.delta == Some(0.0) {
                (None, "zero_queries", Some("zero_exposure"))
            } else if hits.delta > queries.delta {
                (None, "inconsistent", Some("hits_exceed_queries"))
            } else {
                (hits.delta.zip(queries.delta).map(|(hits, queries)| hits / queries), "available", None)
            };
            CacheView {
                labels: labels.clone(),
                queries,
                hits,
                ratio,
                status,
                reason,
                external_queries: sample(&base_map, EXTERNAL_QUERIES, &labels, complete),
                external_hits: sample(&base_map, EXTERNAL_HITS, &labels, complete),
                local_compute: by_source(&full, "local_compute", &labels, complete),
                local_cache_hits: by_source(&full, "local_cache_hit", &labels, complete),
                external_kv_transfer: by_source(&full, "external_kv_transfer", &labels, complete),
                cached_tokens: sample(&base_map, TOKENS_CACHED, &labels, complete),
                attribution: "server_wide_counter_delta_not_request_attributed",
            }
        })
        .collect();
    let mut draft_labels: Vec<BTreeMap<String, String>> = Vec::new();
    for (key, _) in base_map.iter() {
        if [DRAFTS, DRAFT_TOKENS, ACCEPTED_TOKENS].contains(&key.name.as_str())
            && !draft_labels.contains(&key.labels)
        {
            draft_labels.push(key.labels.clone());
        }
    }
    draft_labels.sort();
    let draft: Vec<DraftView> = draft_labels
        .into_iter()
        .map(|labels| {
            let rounds = sample(&base_map, DRAFTS, &labels, complete);
            let draft_tokens = sample(&base_map, DRAFT_TOKENS, &labels, complete);
            let accepted_tokens = sample(&base_map, ACCEPTED_TOKENS, &labels, complete);
            let (token_acceptance, status, reason) = if !complete {
                (None, "not_complete", Some("snapshot_not_complete"))
            } else if rounds.status != "available" {
                (None, "unavailable", Some(rounds.status))
            } else if draft_tokens.status != "available" {
                (None, "unavailable", Some(draft_tokens.status))
            } else if accepted_tokens.status != "available" {
                (None, "unavailable", Some(accepted_tokens.status))
            } else if draft_tokens.delta == Some(0.0) {
                (None, "zero_draft_tokens", Some("zero_exposure"))
            } else if accepted_tokens.delta > draft_tokens.delta {
                (None, "inconsistent", Some("accepted_exceed_draft_tokens"))
            } else {
                (
                    accepted_tokens
                        .delta
                        .zip(draft_tokens.delta)
                        .map(|(accepted, tokens)| accepted / tokens),
                    "available",
                    None,
                )
            };
            DraftView {
                labels,
                rounds,
                draft_tokens,
                accepted_tokens,
                token_acceptance,
                status,
                reason,
                accounting: "accepted_tokens_include_terminal_accepted_work_that_was_not_emitted",
            }
        })
        .collect();
    let mut positions: Vec<PositionView> = full
        .iter()
        .filter(|(key, _)| key.name == ACCEPTED_POS)
        .filter_map(|(key, _)| {
            let draft_position = position(key.labels.get("position")?)?;
            let labels = base(&key.labels, &["position"]);
            let accepted = sample(&full, ACCEPTED_POS, &key.labels, complete);
            let exposure = sample(&base_map, DRAFTS, &labels, complete);
            let (acceptance, status, reason) = if !complete {
                (None, "not_complete", Some("snapshot_not_complete"))
            } else if accepted.status != "available" {
                (None, "unavailable", Some(accepted.status))
            } else if exposure.status != "available" {
                (None, "unavailable", Some(exposure.status))
            } else if exposure.delta == Some(0.0) {
                (None, "zero_exposure", Some("zero_exposure"))
            } else if accepted.delta > exposure.delta {
                (None, "inconsistent", Some("accepted_exceeds_exposure"))
            } else {
                (
                    accepted
                        .delta
                        .zip(exposure.delta)
                        .map(|(accepted, exposure)| accepted / exposure),
                    "available",
                    None,
                )
            };
            Some(PositionView {
                labels,
                position: draft_position,
                accepted,
                exposure,
                exposure_source: DRAFTS,
                acceptance,
                status,
                reason,
            })
        })
        .collect();
    positions.sort_by(|a, b| (&a.labels, a.position).cmp(&(&b.labels, b.position)));
    Views {
        counters,
        preemptions,
        cache,
        draft,
        positions,
        accounting: accounting(&base_map, &full, complete),
        excluded: excluded
            .into_iter()
            .map(|((name, labels), reason)| Excluded {
                name,
                labels,
                reason,
            })
            .collect(),
    }
}

struct Row {
    identity: Identity,
    kind: &'static str,
    unit: &'static str,
    observations: usize,
    changed: bool,
    decreasing: bool,
    first: Option<f64>,
    last: Option<f64>,
    values: Vec<Option<f64>>,
}

/// Acquisition-wide continuity, derived from every retained snapshot of the
/// whole capture (including between-wave boundaries and every required
/// snapshot). A nondecreasing pair is not continuity: counters additionally
/// need their exporter epoch unchanged across every observation.
pub fn acquisition(waves: &[super::WaveReport]) -> Option<Acquisition> {
    let mut snapshots: Vec<&Snapshot> = Vec::new();
    let mut lists: Vec<&Vec<Series>> = Vec::new();
    for wave in waves {
        if let super::WaveReport::V2(summary) = wave {
            snapshots.push(&summary.receipt.before);
            lists.push(&summary.before_series);
            snapshots.push(&summary.receipt.after);
            lists.push(&summary.after_series);
        }
    }
    if snapshots.is_empty() {
        return None;
    }
    let total = snapshots.len();
    let complete = snapshots
        .iter()
        .filter(|snapshot| snapshot.status == Status::Complete)
        .count();
    let mut identities: Vec<Identity> = Vec::new();
    for list in &lists {
        for series in list.iter() {
            if !identities.contains(&series.identity) {
                identities.push(series.identity.clone());
            }
        }
    }
    let rows: Vec<Row> = identities
        .into_iter()
        .map(|identity| {
            let values: Vec<Option<f64>> = lists
                .iter()
                .map(|list| {
                    list.iter()
                        .find(|series| series.identity == identity)
                        .map(|series| series.value)
                })
                .collect();
            let observations = values.iter().filter(|value| value.is_some()).count();
            let mut changed = false;
            let mut decreasing = false;
            let mut previous: Option<f64> = None;
            for value in values.iter().flatten() {
                if let Some(previous) = previous {
                    if previous != *value {
                        changed = true;
                    }
                    if *value < previous {
                        decreasing = true;
                    }
                }
                previous = Some(*value);
            }
            Row {
                kind: kind_of(&identity.name),
                unit: unit_of(&identity.name),
                observations,
                changed,
                decreasing,
                first: values.iter().flatten().next().copied(),
                last: values.iter().flatten().next_back().copied(),
                values,
                identity,
            }
        })
        .collect();
    let index: BTreeMap<Identity, usize> = rows
        .iter()
        .enumerate()
        .map(|(index, row)| (row.identity.clone(), index))
        .collect();
    let mut series = Vec::with_capacity(rows.len());
    for row in &rows {
        let epoch = if row.kind == "counter" {
            match epoch_of(&row.identity.name) {
                Some(name) => {
                    let identity = Identity {
                        name: name.to_owned(),
                        labels: row.identity.labels.clone(),
                    };
                    match index.get(&identity).map(|index| &rows[*index]) {
                        Some(epoch) => Epoch {
                            series: Some(name),
                            observations: epoch.observations,
                            value: epoch.first,
                            status: if epoch.observations < total {
                                "missing"
                            } else if epoch.changed {
                                "changed"
                            } else {
                                "constant"
                            },
                        },
                        None => Epoch {
                            series: Some(name),
                            observations: 0,
                            value: None,
                            status: "missing",
                        },
                    }
                }
                None => Epoch {
                    series: None,
                    observations: 0,
                    value: None,
                    status: "undeclared",
                },
            }
        } else {
            None
        };
        let status = if row.kind == "epoch" {
            if row.observations == 0 {
                "no_snapshots"
            } else if row.observations < total {
                "disappeared"
            } else if row.changed {
                "changed"
            } else {
                "constant"
            }
        } else if complete < total {
            "snapshot_incomplete"
        } else if row.observations == 0 {
            "no_snapshots"
        } else if row.observations < total {
            if row.values.first().is_none_or(|value| value.is_none()) {
                "appeared_later"
            } else {
                "disappeared"
            }
        } else if !row.values.iter().flatten().all(|value| value.is_finite()) {
            "nonfinite_value"
        } else if row.kind == "counter" && row.decreasing {
            "reset_observed"
        } else {
            match (&epoch, row.kind) {
                (Some(epoch), "counter") if epoch.status != "constant" => match epoch.status {
                    "changed" => "epoch_changed",
                    "undeclared" => "epoch_undeclared",
                    _ => "epoch_missing",
                },
                _ => "continuous",
            }
        };
        series.push(SeriesContinuity {
            identity: row.identity.clone(),
            kind: row.kind,
            observations: row.observations,
            first: row.first,
            last: row.last,
            delta: row.first.zip(row.last).map(|(first, last)| last - first),
            epoch,
            status,
        });
    }
    let mut first_series = Vec::with_capacity(rows.len());
    let mut last_series = Vec::with_capacity(rows.len());
    for row in &rows {
        if let Some(value) = row.first {
            first_series.push(Series {
                identity: row.identity.clone(),
                unit: row.unit,
                kind: row.kind,
                value,
            });
        }
        if let Some(value) = row.last {
            last_series.push(Series {
                identity: row.identity.clone(),
                unit: row.unit,
                kind: row.kind,
                value,
            });
        }
    }
    let full = slices(&first_series, &last_series, &[]);
    let base_map = slices(&first_series, &last_series, &["source", "position"]);
    let checks = accounting(&base_map, &full, complete == total);
    Some(Acquisition {
        snapshots: total,
        complete,
        series,
        accounting: checks,
    })
}

pub fn validate_requirement(requirement: &Requirement) -> Result<()> {
    let spec = spec(&requirement.name)
        .ok_or("required telemetry names a metric outside the closed v2 allowlist")?;
    if requirement.predicate != Predicate::Present && spec.kind != Kind::Counter {
        return Err("required telemetry predicate is incompatible with the metric kind".into());
    }
    if requirement.labels.len() > 8 {
        return Err("required telemetry declares too many labels".into());
    }
    for (name, value) in &requirement.labels {
        let selector = spec
            .selector
            .is_some_and(|selector| selector == name.as_str());
        if name != "engine" && name != "model_name" && !selector {
            return Err("required telemetry declares an unsupported label".into());
        }
        if value.is_empty() || value.len() > 256 || value.chars().any(char::is_control) {
            return Err("required telemetry label value is empty or out of bounds".into());
        }
    }
    match spec.selector {
        Some("source") => {
            let value = requirement
                .labels
                .get("source")
                .ok_or("required telemetry omits the source selector")?;
            if !SOURCES.contains(&value.as_str()) {
                return Err("required telemetry declares an unsupported source".into());
            }
        }
        Some("position") => {
            let value = requirement
                .labels
                .get("position")
                .ok_or("required telemetry omits the position selector")?;
            if position(value).is_none() {
                return Err("required telemetry declares an invalid position".into());
            }
        }
        _ => (),
    }
    Ok(())
}

pub fn assess(requirements: &[Requirement], summary: &super::Summary) -> Assessment {
    if requirements.is_empty() {
        return Assessment {
            outcome: AssessmentOutcome::Satisfied,
            results: Vec::new(),
        };
    }
    let Some(acquisition) = &summary.acquisition else {
        return Assessment {
            outcome: AssessmentOutcome::Error,
            results: requirements
                .iter()
                .map(|requirement| {
                    result(
                        requirement,
                        "error",
                        Some("no_metrics2"),
                        "required telemetry needs the explicit version-2 metrics protocol",
                    )
                })
                .collect(),
        };
    };
    let isolation = match &summary.config {
        super::Protocol::V2(config) => config.isolation.as_str(),
        super::Protocol::V1(_) => "",
    };
    let results: Vec<RequirementResult> = requirements
        .iter()
        .map(|requirement| evaluate(requirement, acquisition, isolation))
        .collect();
    let outcome = if results.iter().any(|result| result.outcome == "error") {
        AssessmentOutcome::Error
    } else if results.iter().any(|result| result.outcome == "refuted") {
        AssessmentOutcome::Refuted
    } else if results.iter().any(|result| result.outcome == "unavailable") {
        AssessmentOutcome::Unavailable
    } else {
        AssessmentOutcome::Satisfied
    };
    Assessment { outcome, results }
}

fn result(
    requirement: &Requirement,
    outcome: &'static str,
    reason: Option<&'static str>,
    detail: impl Into<String>,
) -> RequirementResult {
    RequirementResult {
        name: requirement.name.clone(),
        labels: requirement.labels.clone(),
        predicate: requirement.predicate,
        outcome,
        reason,
        detail: detail.into(),
    }
}

fn evaluate(
    requirement: &Requirement,
    acquisition: &Acquisition,
    isolation: &str,
) -> RequirementResult {
    if let Err(detail) = validate_requirement(requirement) {
        return result(requirement, "error", Some("invalid_requirement"), detail);
    }
    let identity = Identity {
        name: requirement.name.clone(),
        labels: requirement.labels.clone(),
    };
    let absent = if acquisition.complete < acquisition.snapshots {
        "missing_required_snapshot"
    } else {
        "series_absent"
    };
    let Some(entry) = acquisition
        .series
        .iter()
        .find(|series| series.identity == identity)
    else {
        return result(
            requirement,
            "unavailable",
            Some(absent),
            "required series identity is not present in the captured acquisition",
        );
    };
    if let Some(check) = acquisition
        .accounting
        .iter()
        .find(|check| check.labels == requirement.labels && check.status == "inconsistent")
    {
        return result(
            requirement,
            "unavailable",
            Some("contradictory_accounting"),
            check.rule,
        );
    }
    match requirement.predicate {
        Predicate::Present => {
            if acquisition.complete == acquisition.snapshots
                && entry.observations == acquisition.snapshots
                && entry.first.is_some()
            {
                result(
                    requirement,
                    "satisfied",
                    None,
                    "required series is present in every required snapshot",
                )
            } else {
                result(
                    requirement,
                    "unavailable",
                    Some(entry.status),
                    "required series is not present in every required snapshot",
                )
            }
        }
        Predicate::ContinuousCounter => {
            if entry.status == "continuous" {
                result(
                    requirement,
                    "satisfied",
                    None,
                    "counter is non-decreasing across the whole capture with an unchanged exporter epoch",
                )
            } else {
                result(
                    requirement,
                    "unavailable",
                    Some(entry.status),
                    "counter continuity is not attested for the whole capture",
                )
            }
        }
        Predicate::ZeroCounterDelta => {
            if isolation != ISOLATION_EXCLUSIVE {
                return result(
                    requirement,
                    "unavailable",
                    Some("scope_not_isolated"),
                    "zero counter totals need an exclusive isolated acquisition declaration",
                );
            }
            if entry.status != "continuous" {
                return result(
                    requirement,
                    "unavailable",
                    Some(entry.status),
                    "zero counter totals need attested continuity, not a nondecreasing pair",
                );
            }
            match entry.delta {
                Some(delta) if delta == 0.0 => result(
                    requirement,
                    "satisfied",
                    None,
                    "zero counter delta observed over the whole capture",
                ),
                Some(_) => result(
                    requirement,
                    "refuted",
                    Some("nonzero_delta"),
                    "counter advanced during the capture",
                ),
                None => result(
                    requirement,
                    "unavailable",
                    Some("unavailable_delta"),
                    "counter delta is not observable",
                ),
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{summarize, Budget, Protocol, WaveReport};

    const IDENTITY: &str = r#"engine="0",model_name="m""#;
    // Must match `Config::new`'s deadline so a self-consistent snapshot passes.
    const ALLOWANCE_US: u64 = 2_000_000;

    fn config() -> Config {
        let mut config = Config::new("http://127.0.0.1:9/metrics".into(), 1, None);
        config.isolation = ISOLATION_EXCLUSIVE.into();
        config
    }

    fn labels(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect()
    }

    fn identity() -> Vec<(&'static str, &'static str)> {
        vec![("engine", "0"), ("model_name", "m")]
    }

    fn snapshot(offset: u64, duration: u64) -> Snapshot {
        Snapshot {
            started_unix_ms: 1_700_000_000_000,
            scrape_offset_us: offset,
            scrape_us: duration,
            duration_us: duration,
            charged_us: duration,
            overhead_us: 0,
            allowance_us: ALLOWANCE_US,
            status: Status::Complete,
            error: None,
            http_status: Some(200),
            raw_bytes: 0,
            raw_sha256: String::new(),
            series: 0,
        }
    }

    fn receipt() -> Receipt {
        Receipt {
            version: 2,
            plan_sha256: "0".repeat(64),
            wave: 0,
            before: snapshot(0, 10),
            after: snapshot(20, 10),
            measured_origin_offset_us: 11,
            measured_origin_unix_ms: 1_700_000_000_000,
            measured_duration_us: 5,
        }
    }

    fn wave_summary(before: &str, after: &str) -> WaveSummary {
        let config = config();
        let before_series = parse(before.as_bytes(), &config).unwrap();
        let after_series = parse(after.as_bytes(), &config).unwrap();
        let views = derive(&before_series, &after_series, true);
        WaveSummary {
            receipt: receipt(),
            overhead_us: 0,
            before_series,
            after_series,
            counters: views.counters,
            preemptions: views.preemptions,
            cache: views.cache,
            draft: views.draft,
            positions: views.positions,
            accounting: views.accounting,
            excluded: views.excluded,
            caveats: CAVEATS.to_vec(),
        }
    }

    fn summarize_with(config: Config, before: &str, after: &str) -> crate::metrics::Summary {
        let protocol = Protocol::V2(config);
        summarize(
            Some(&protocol),
            Budget::default(),
            vec![WaveReport::V2(wave_summary(before, after))],
        )
        .unwrap()
    }

    fn summary(before: &str, after: &str) -> crate::metrics::Summary {
        summarize_with(config(), before, after)
    }

    fn sample(text: &mut String, name: &str, labels: &str, value: f64) {
        text.push_str(&format!("{name}{{{labels}}} {value}\n"));
    }

    fn epoch(text: &mut String, name: &str, labels: &str) {
        sample(text, name, labels, 1_700_000_000.0);
    }

    fn entry<'a>(
        acquisition: &'a Acquisition,
        name: &str,
        pairs: &[(&str, &str)],
    ) -> &'a SeriesContinuity {
        let wanted = labels(pairs);
        acquisition
            .series
            .iter()
            .find(|series| series.identity.name == name && series.identity.labels == wanted)
            .unwrap()
    }

    fn requirement(name: &str, pairs: &[(&str, &str)], predicate: Predicate) -> Requirement {
        Requirement {
            name: name.into(),
            labels: labels(pairs),
            predicate,
        }
    }

    fn outcome(assessment: &Assessment) -> (AssessmentOutcome, Option<&'static str>) {
        (
            assessment.outcome,
            assessment.results.first().and_then(|result| result.reason),
        )
    }

    /// Pinned-shape vLLM accounting sample whose internal relations hold unless
    /// a field is deliberately perturbed.
    #[derive(Clone)]
    struct Fixture {
        preemptions: f64,
        preemption_epoch: f64,
        queries: f64,
        hits: f64,
        total: f64,
        local_compute: f64,
        local_cache_hit: f64,
        external_kv: f64,
        cached: f64,
        drafts: f64,
        draft_tokens: f64,
        accepted: f64,
        positions: Vec<f64>,
        omit_hits: bool,
        omit_cached: bool,
    }

    fn fixture() -> Fixture {
        Fixture {
            preemptions: 0.0,
            preemption_epoch: 1_700_000_000.0,
            queries: 100.0,
            hits: 40.0,
            total: 100.0,
            local_compute: 60.0,
            local_cache_hit: 30.0,
            external_kv: 10.0,
            cached: 40.0,
            drafts: 10.0,
            draft_tokens: 50.0,
            accepted: 12.0,
            positions: vec![10.0, 2.0],
            omit_hits: false,
            omit_cached: false,
        }
    }

    fn advanced() -> Fixture {
        Fixture {
            preemptions: 0.0,
            preemption_epoch: 1_700_000_000.0,
            queries: 140.0,
            hits: 70.0,
            total: 160.0,
            local_compute: 100.0,
            local_cache_hit: 50.0,
            external_kv: 10.0,
            cached: 60.0,
            drafts: 25.0,
            draft_tokens: 125.0,
            accepted: 26.0,
            positions: vec![20.0, 6.0],
            omit_hits: false,
            omit_cached: false,
        }
    }

    fn body(fixture: &Fixture) -> String {
        let mut text = String::new();
        sample(&mut text, PREEMPTIONS, IDENTITY, fixture.preemptions);
        sample(
            &mut text,
            "vllm:num_preemptions_created",
            IDENTITY,
            fixture.preemption_epoch,
        );
        sample(&mut text, PREFIX_QUERIES, IDENTITY, fixture.queries);
        epoch(&mut text, "vllm:prefix_cache_queries_created", IDENTITY);
        if !fixture.omit_hits {
            sample(&mut text, PREFIX_HITS, IDENTITY, fixture.hits);
            epoch(&mut text, "vllm:prefix_cache_hits_created", IDENTITY);
        }
        sample(&mut text, TOKENS_TOTAL, IDENTITY, fixture.total);
        epoch(&mut text, "vllm:prompt_tokens_created", IDENTITY);
        for (source, value) in [
            ("local_compute", fixture.local_compute),
            ("local_cache_hit", fixture.local_cache_hit),
            ("external_kv_transfer", fixture.external_kv),
        ] {
            let labels = format!("{IDENTITY},source=\"{source}\"");
            sample(&mut text, TOKENS_BY_SOURCE, &labels, value);
            epoch(&mut text, "vllm:prompt_tokens_by_source_created", &labels);
        }
        if !fixture.omit_cached {
            sample(&mut text, TOKENS_CACHED, IDENTITY, fixture.cached);
            epoch(&mut text, "vllm:prompt_tokens_cached_created", IDENTITY);
        }
        sample(&mut text, DRAFTS, IDENTITY, fixture.drafts);
        epoch(&mut text, "vllm:spec_decode_num_drafts_created", IDENTITY);
        sample(&mut text, DRAFT_TOKENS, IDENTITY, fixture.draft_tokens);
        epoch(
            &mut text,
            "vllm:spec_decode_num_draft_tokens_created",
            IDENTITY,
        );
        sample(&mut text, ACCEPTED_TOKENS, IDENTITY, fixture.accepted);
        epoch(
            &mut text,
            "vllm:spec_decode_num_accepted_tokens_created",
            IDENTITY,
        );
        for (position, value) in fixture.positions.iter().enumerate() {
            let labels = format!("{IDENTITY},position=\"{position}\"");
            sample(&mut text, ACCEPTED_POS, &labels, *value);
            epoch(
                &mut text,
                "vllm:spec_decode_num_accepted_tokens_per_pos_created",
                &labels,
            );
        }
        sample(&mut text, RUNNING, IDENTITY, 2.0);
        sample(&mut text, WAITING, IDENTITY, 1.0);
        text
    }

    #[test]
    fn cache_draft_and_per_position_accounting_use_provider_denominators() {
        let summary = wave_summary(&body(&fixture()), &body(&advanced()));
        let cache = &summary.cache[0];
        assert_eq!(cache.queries.delta, Some(40.0));
        assert_eq!(cache.hits.delta, Some(30.0));
        assert_eq!(cache.ratio, Some(0.75));
        assert_eq!(cache.status, "available");
        assert_eq!(cache.local_compute.delta, Some(40.0));
        assert_eq!(cache.local_cache_hits.delta, Some(20.0));
        assert_eq!(cache.external_kv_transfer.delta, Some(0.0));
        assert_eq!(cache.cached_tokens.delta, Some(20.0));
        assert_eq!(
            cache.attribution,
            "server_wide_counter_delta_not_request_attributed"
        );
        let draft = &summary.draft[0];
        assert_eq!(draft.rounds.delta, Some(15.0));
        assert_eq!(draft.accepted_tokens.delta, Some(14.0));
        assert_eq!(draft.token_acceptance, Some(14.0 / 75.0));
        assert_eq!(summary.positions.len(), 2);
        assert_eq!(summary.positions[0].position, 0);
        assert_eq!(summary.positions[0].exposure.delta, Some(15.0));
        assert_eq!(summary.positions[0].exposure_source, DRAFTS);
        assert_eq!(summary.positions[0].acceptance, Some(10.0 / 15.0));
        assert_eq!(summary.positions[1].acceptance, Some(4.0 / 15.0));
        assert!(summary
            .accounting
            .iter()
            .all(|check| check.status == "consistent"));
        assert!(summary
            .preemptions
            .iter()
            .all(|delta| delta.delta == Some(0.0)));
        assert_eq!(summary.caveats, CAVEATS.to_vec());
    }

    #[test]
    fn zero_exposure_and_missing_series_never_clamp_into_ratios() {
        let mut before = fixture();
        before.positions = vec![5.0];
        let mut after = fixture();
        after.positions = vec![5.0];
        after.omit_hits = true;
        let summary = wave_summary(&body(&before), &body(&after));
        let position = &summary.positions[0];
        assert_eq!(position.exposure.delta, Some(0.0));
        assert_eq!(position.acceptance, None);
        assert_eq!(position.status, "zero_exposure");
        let cache = &summary.cache[0];
        assert_eq!(cache.hits.delta, None);
        assert_eq!(cache.ratio, None);
        assert_eq!(cache.status, "unavailable");
        assert_eq!(cache.reason, Some("missing_after"));
    }

    #[test]
    fn reset_disappearance_and_epoch_change_withhold_continuity() {
        let mut before = fixture();
        before.preemptions = 5.0;
        before.omit_cached = true;
        let mut after = fixture();
        after.preemptions = 2.0;
        after.preemption_epoch = 1_700_000_500.0;
        after.omit_hits = true;
        let summary = wave_summary(&body(&before), &body(&after));
        assert_eq!(summary.preemptions[0].delta, None);
        assert_eq!(summary.preemptions[0].status, "reset_observed");
        let acquisition = summarize_with(config(), &body(&before), &body(&after))
            .acquisition
            .unwrap();
        assert_eq!(acquisition.snapshots, 2);
        assert_eq!(acquisition.complete, 2);
        let preemptions = entry(&acquisition, PREEMPTIONS, &identity());
        assert_eq!(preemptions.status, "reset_observed");
        assert_eq!(preemptions.epoch.as_ref().unwrap().status, "changed");
        assert_eq!(entry(&acquisition, PREFIX_HITS, &identity()).status, "disappeared");
        assert_eq!(
            entry(&acquisition, TOKENS_CACHED, &identity()).status,
            "appeared_later"
        );
        assert!(acquisition
            .accounting
            .iter()
            .any(|check| check.status == "unavailable"));
    }

    #[test]
    fn zero_preemptions_needs_continuity_isolation_and_epoch() {
        let requirements = [requirement(
            PREEMPTIONS,
            &identity(),
            Predicate::ZeroCounterDelta,
        )];
        let isolated = summary(&body(&fixture()), &body(&advanced()));
        assert_eq!(
            outcome(&assess(&requirements, &isolated)),
            (AssessmentOutcome::Satisfied, None)
        );

        let mut shared = config();
        shared.isolation = ISOLATION_SHARED.into();
        let shared = summarize_with(shared, &body(&fixture()), &body(&advanced()));
        assert_eq!(
            outcome(&assess(&requirements, &shared)),
            (AssessmentOutcome::Unavailable, Some("scope_not_isolated"))
        );

        // Two nondecreasing samples are not zero: the counter advanced.
        let mut nonzero = advanced();
        nonzero.preemptions = 3.0;
        let refuted = summary(&body(&fixture()), &body(&nonzero));
        assert_eq!(
            outcome(&assess(&requirements, &refuted)),
            (AssessmentOutcome::Refuted, Some("nonzero_delta"))
        );

        // A new exporter epoch with an unchanged value is still unavailable.
        let mut replaced = advanced();
        replaced.preemption_epoch = 1_700_000_500.0;
        let changed = summary(&body(&fixture()), &body(&replaced));
        assert_eq!(
            outcome(&assess(&requirements, &changed)),
            (AssessmentOutcome::Unavailable, Some("epoch_changed"))
        );

        // Nondecreasing samples with an unchanged epoch are continuity, not the
        // legacy "nondecreasing restart unverified" status.
        let acquisition = summary(&body(&fixture()), &body(&advanced()))
            .acquisition
            .unwrap();
        assert_eq!(entry(&acquisition, DRAFTS, &identity()).status, "continuous");
        assert_eq!(
            entry(&acquisition, DRAFTS, &identity())
                .epoch
                .as_ref()
                .unwrap()
                .status,
            "constant"
        );
    }

    #[test]
    fn required_telemetry_is_typed_and_fails_closed() {
        let continuous = [requirement(
            PREFIX_HITS,
            &identity(),
            Predicate::ContinuousCounter,
        )];
        let present = [requirement(RUNNING, &identity(), Predicate::Present)];
        let cached = [requirement(
            TOKENS_CACHED,
            &identity(),
            Predicate::ContinuousCounter,
        )];
        let sound = summary(&body(&fixture()), &body(&advanced()));
        assert_eq!(
            outcome(&assess(&continuous, &sound)),
            (AssessmentOutcome::Satisfied, None)
        );
        assert_eq!(
            outcome(&assess(&present, &sound)),
            (AssessmentOutcome::Satisfied, None)
        );
        assert_eq!(
            outcome(&assess(&[], &sound)),
            (AssessmentOutcome::Satisfied, None)
        );

        // A missing series is unavailable, never satisfied.
        let mut dropped = advanced();
        dropped.omit_hits = true;
        let dropped = summary(&body(&fixture()), &body(&dropped));
        assert_eq!(
            outcome(&assess(&continuous, &dropped)),
            (AssessmentOutcome::Unavailable, Some("disappeared"))
        );

        // Contradictory accounting withholds a related claim.
        let mut corrupted = advanced();
        corrupted.local_compute = 999.0;
        let corrupted = summary(&body(&fixture()), &body(&corrupted));
        assert_eq!(
            outcome(&assess(&cached, &corrupted)),
            (
                AssessmentOutcome::Unavailable,
                Some("contradictory_accounting")
            )
        );

        // Requirements promised without the version-2 protocol are errors.
        let legacy = summarize(
            Some(&Protocol::V1(crate::metrics::Config::new(
                "http://127.0.0.1:9/metrics".into(),
                1,
            ))),
            Budget::default(),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(
            outcome(&assess(&present, &legacy)),
            (AssessmentOutcome::Error, Some("no_metrics2"))
        );
    }

    #[test]
    fn requirement_validation_rejects_unknown_names_and_selectors() {
        assert!(validate_requirement(&requirement(
            "vllm:not_a_selected_metric_total",
            &identity(),
            Predicate::Present
        ))
        .is_err());
        assert!(validate_requirement(&requirement(
            RUNNING,
            &identity(),
            Predicate::ZeroCounterDelta
        ))
        .is_err());
        assert!(validate_requirement(&requirement(
            TOKENS_BY_SOURCE,
            &identity(),
            Predicate::Present
        ))
        .is_err());
        assert!(validate_requirement(&requirement(
            TOKENS_BY_SOURCE,
            &[("engine", "0"), ("source", "invented")],
            Predicate::Present
        ))
        .is_err());
        assert!(validate_requirement(&requirement(
            ACCEPTED_POS,
            &[("engine", "0"), ("position", "first")],
            Predicate::Present
        ))
        .is_err());
        assert!(validate_requirement(&requirement(
            PREEMPTIONS,
            &[("worker", "3")],
            Predicate::Present
        ))
        .is_err());
        assert!(validate_requirement(&requirement(
            TOKENS_BY_SOURCE,
            &[("engine", "0"), ("source", "local_compute")],
            Predicate::ContinuousCounter
        ))
        .is_ok());
        assert!(validate_requirement(&requirement(
            ACCEPTED_POS,
            &[("engine", "0"), ("position", "0")],
            Predicate::ContinuousCounter
        ))
        .is_ok());
    }

    #[test]
    fn budget_and_snapshot_impossible_metadata_fail_closed() {
        let config = config();
        let mut budget = Budget::default();
        let mut incomplete = snapshot(0, 10);
        incomplete.status = Status::Cancelled;
        incomplete.error = Some("acquisition cancelled during metrics scrape".into());
        incomplete.series = 3;
        assert!(budget.retain_v2(&config, &incomplete).is_err());

        let mut oversized = snapshot(0, 10);
        oversized.series = config.series + 1;
        assert!(budget.retain_v2(&config, &oversized).is_err());

        let mut miscounted = snapshot(0, 10);
        miscounted.overhead_us = 5;
        assert!(budget.retain_v2(&config, &miscounted).is_err());

        let mut valid = snapshot(0, 10);
        valid.series = 7;
        assert!(budget.retain_v2(&config, &valid).is_ok());
        let consumption = budget.v2.unwrap();
        assert_eq!(consumption.snapshots, 1);
        assert_eq!(consumption.series, 7);
        assert_eq!(consumption.overhead_us, 0);
        assert_eq!(budget.requests, 1);
        assert_eq!(budget.charged_us, 10);

        // A cancelled scrape retains bounded partial bytes but never counts as
        // complete series evidence.
        assert!(budget.retain_v2(&config, &incomplete).is_err());
    }

    #[test]
    fn accounting_contradictions_withhold_claims() {
        let mut corrupted = advanced();
        corrupted.cached = 100.0;
        let summary = summary(&body(&fixture()), &body(&corrupted));
        let acquisition = summary.acquisition.as_ref().unwrap();
        let rule = acquisition
            .accounting
            .iter()
            .find(|check| check.status == "inconsistent")
            .unwrap();
        assert_eq!(
            rule.rule,
            "prompt_tokens_cached_total_equals_local_cache_hit_plus_external_kv_transfer"
        );
        let requirements = [requirement(
            TOKENS_CACHED,
            &identity(),
            Predicate::ContinuousCounter,
        )];
        assert_eq!(
            outcome(&assess(&requirements, &summary)),
            (
                AssessmentOutcome::Unavailable,
                Some("contradictory_accounting")
            )
        );
    }

    #[test]
    fn parse_is_closed_and_rejects_negative_duplicate_and_nonfinite() {
        let config = config();
        let selected = |text: &str| parse(text.as_bytes(), &config).map(|series| series.len());
        assert_eq!(selected(&format!("{PREEMPTIONS}{{{IDENTITY}}} 1\n")), Ok(1));
        assert_eq!(
            selected("vllm:gpu_cache_usage_perc{engine=\"0\"} 0.5\n"),
            Ok(0)
        );
        assert!(selected(&format!("{PREEMPTIONS}{{{IDENTITY}}} -1\n")).is_err());
        assert!(selected(&format!("{PREEMPTIONS}{{{IDENTITY}}} NaN\n")).is_err());
        assert!(selected(&format!("{PREEMPTIONS}{{{IDENTITY}}} +Inf\n")).is_err());
        assert!(selected(&format!(
            "{PREEMPTIONS}{{{IDENTITY}}} 1\n{PREEMPTIONS}{{{IDENTITY}}} 2\n"
        ))
        .is_err());

        // Selector values outside the closed sets are excluded, never guessed.
        let mut odd = fixture();
        odd.positions = vec![1.0];
        let broken = body(&odd).replace("position=\"0\"", "position=\"zero\"");
        let mut odd = fixture();
        odd.positions = vec![2.0];
        let broken_after = body(&odd).replace("position=\"0\"", "position=\"zero\"");
        let views = derive(
            &parse(broken.as_bytes(), &config).unwrap(),
            &parse(broken_after.as_bytes(), &config).unwrap(),
            true,
        );
        assert!(views.positions.is_empty());
        assert!(views
            .excluded
            .iter()
            .any(|excluded| excluded.reason == "invalid_position_label"));

        let unknown = body(&fixture()).replace("source=\"local_compute\"", "source=\"invented\"");
        let views = derive(
            &parse(unknown.as_bytes(), &config).unwrap(),
            &parse(unknown.as_bytes(), &config).unwrap(),
            true,
        );
        assert!(views
            .excluded
            .iter()
            .any(|excluded| excluded.reason == "unsupported_source_value"));
    }
}
