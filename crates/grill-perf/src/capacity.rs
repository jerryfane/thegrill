//! Finite prospective capacity cells, using run's single native collector.
use crate::{acquisition, evidence, model::*, run};
use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

const PLAN_CAP: usize = 4 * 1024 * 1024;
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub version: u32,
    pub kind: String,
    pub cells: Vec<Cell>,
    pub max_requests: u64,
    pub max_output_tokens: u64,
    pub max_wall_us: u64,
    pub retained_bytes: u64,
    pub stop_condition: StopCondition,
    pub oom_response: OomResponse,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StopCondition {
    FirstUnsuccessfulCell,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OomResponse {
    StopOperatorRecovery,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Cell {
    pub id: String,
    pub context_token_ceiling: u64,
    pub workload: Workload,
    pub success: Success,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "present"
    )]
    pub retention: Option<crate::retention::Plan>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Success {
    pub quality: Quality,
    pub require_prompt_usage: bool,
    pub resources: ResourceCondition,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Quality {
    AllSequenceChecks,
    ProtocolOnly,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ResourceCondition {
    NotRequiredNoResourceClaim,
    Limits { limits: Vec<ResourceLimit> },
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ResourceLimit {
    pub summary: usize,
    pub maximum: u64,
}
#[derive(Serialize)]
pub struct Budget {
    pub requests: u64,
    pub output_tokens: u64,
    pub retained_bytes: u64,
    pub wall_us: u64,
}
impl Plan {
    pub fn validate(&self) -> Result<Budget> {
        if self.version != 1
            || self.kind != "capacity-study-v1"
            || self.cells.is_empty()
            || self.cells.len() > 16
            || self.max_requests == 0
            || self.max_requests > MAX_ATTEMPTS
            || self.max_output_tokens == 0
            || self.max_output_tokens > 1_000_000_000
            || self.max_wall_us == 0
            || self.max_wall_us > 3_600_000_000
            || self.retained_bytes == 0
            || self.retained_bytes > 16 * 1024 * 1024 * 1024
        {
            return Err("capacity study exceeds finite plan ceilings".into());
        }
        let mut budget = Budget {
            requests: 0,
            output_tokens: 0,
            retained_bytes: PLAN_CAP as u64,
            wall_us: self.max_wall_us,
        };
        let mut ids = std::collections::BTreeSet::new();
        let mut previous = None;
        for cell in &self.cells {
            cell.workload.validate()?;
            if !identifier(&cell.id)
                || !ids.insert(&cell.id)
                || cell.workload.version != 6
                || cell.context_token_ceiling == 0
                || cell.context_token_ceiling > 16_777_216
            {
                return Err(
                    "capacity cells require unique IDs, explicit bounded context and workload6"
                        .into(),
                );
            }
            let concurrency = cell
                .workload
                .cells
                .iter()
                .map(|c| c.concurrency)
                .max()
                .ok_or("empty capacity population")?;
            if previous.is_some_and(|(context, load)| {
                cell.context_token_ceiling < context || concurrency < load
            }) {
                return Err(
                    "ordered capacity cells must not decrease either declared context or load"
                        .into(),
                );
            }
            previous = Some((cell.context_token_ceiling, concurrency));
            if cell.success.quality == Quality::AllSequenceChecks
                && !acquisition::conversation(&cell.workload)
            {
                return Err("sequence correctness requires a conversation workload".into());
            }
            match (&cell.workload.resources, &cell.success.resources) {
                (None, ResourceCondition::NotRequiredNoResourceClaim) => (),
                (Some(config), ResourceCondition::Limits { limits }) if !limits.is_empty() && limits.len() <= 32 => {
                    let mut seen = std::collections::BTreeSet::new();
                    for limit in limits {
                        if limit.summary >= config.summaries.len() || !seen.insert(limit.summary) {
                            return Err("resource limit must select a unique prospective summary".into());
                        }
                    }
                }
                _ => return Err("attached capacity resources require explicit limits; absent resources make no resource claim".into()),
            }
            if let Some(retention) = &cell.retention {
                retention.validate(&cell.workload)?;
            }
            let b = acquisition::budget(&cell.workload)?;
            let requests = b.warmup_requests + b.control_requests + b.measured_requests;
            budget.requests = budget
                .requests
                .checked_add(requests)
                .ok_or("request budget overflow")?;
            budget.output_tokens = budget
                .output_tokens
                .checked_add(b.output_token_ceiling)
                .ok_or("token budget overflow")?;
            // Native per-wave reservation (40MiB), receipt (4MiB), exact responses,
            // sequence traces, metrics2 allowance, full native plan and resources.
            let bytes = (cell.workload.waves().len() as u64)
                .checked_mul(44 * 1024 * 1024)
                .and_then(|n| n.checked_add(b.response_byte_ceiling))
                .and_then(|n| n.checked_add(b.tool_trace_byte_ceiling))
                .and_then(|n| n.checked_add(64 * 1024 * 1024))
                .and_then(|n| {
                    n.checked_add(
                        cell.workload
                            .resources
                            .as_ref()
                            .map_or(0, |r| r.retained_bytes),
                    )
                })
                .and_then(|n| {
                    n.checked_add(
                        cell.retention
                            .as_ref()
                            .map_or(0, |r| u64::from(r.journal_bytes) * 2),
                    )
                })
                .ok_or("retention budget overflow")?;
            budget.retained_bytes = budget
                .retained_bytes
                .checked_add(bytes)
                .ok_or("retention budget overflow")?;
        }
        if budget.requests > self.max_requests
            || budget.output_tokens > self.max_output_tokens
            || budget.retained_bytes > self.retained_bytes
        {
            return Err(
                "prospective capacity population exceeds declared traffic or retention allowance"
                    .into(),
            );
        }
        Ok(budget)
    }
}
fn load_plan(path: &Path) -> Result<(Plan, Vec<u8>)> {
    let bytes = evidence::read(path, PLAN_CAP)?;
    let plan: Plan = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    plan.validate()?;
    Ok((plan, bytes))
}
#[derive(clap::Args)]
pub struct Options {
    pub plan: PathBuf,
    #[arg(long)]
    pub endpoint: String,
    #[arg(long)]
    pub model: String,
    #[arg(long)]
    pub auth_env: Option<String>,
    #[arg(long)]
    pub deployment: Option<PathBuf>,
    #[arg(long)]
    pub local_http: bool,
    #[arg(long)]
    pub metrics_url: Option<String>,
    #[arg(long)]
    pub metrics_auth_env: Option<String>,
    #[arg(long)]
    pub metrics_isolation: Option<String>,
    /// Existing source-pinned finite eviction journal; no producer launch or reset.
    #[arg(long)]
    pub eviction_journal: Option<PathBuf>,
    #[arg(long)]
    pub out: PathBuf,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Execution {
    version: u32,
    plan_sha256: String,
    endpoint: String,
    model: String,
    auth_env: Option<String>,
    local_http: bool,
    metrics_url: Option<String>,
    metrics_auth_env: Option<String>,
    metrics_isolation: Option<String>,
    deployment: Option<Deployment>,
    collector_sha256: String,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Outcome {
    version: u32,
    plan_sha256: String,
    error: Option<String>,
}
#[derive(Serialize)]
pub struct CellReport {
    pub id: String,
    pub state: &'static str,
    pub context_token_ceiling: u64,
    pub maximum_concurrency: u32,
    pub success: Success,
    pub native_run: String,
    pub planned_requests: u64,
    pub dispatched_requests: Option<u64>,
    pub undispatched_requests: Option<u64>,
    pub observed_prompt_tokens: Option<[u64; 2]>,
    pub prompt_usage_observed_requests: u64,
    pub failures: Vec<serde_json::Value>,
    pub reasons: Vec<String>,
    pub resources: Vec<crate::serving_resources::Report>,
    pub retention: Option<serde_json::Value>,
    pub evidence_sha256: Option<String>,
}
#[derive(Serialize)]
pub struct Report {
    pub version: u32,
    pub kind: &'static str,
    pub plan_sha256: String,
    pub cells: Vec<CellReport>,
    pub largest_successful_tested_cell: Option<String>,
    pub scope: &'static str,
    pub safe_universal_capacity: &'static str,
    pub recovery: &'static str,
}
fn cell_path(index: usize) -> String {
    format!("cell-{index:03}")
}
fn common(options: &Options, workload: PathBuf) -> run::CommonArgs {
    run::CommonArgs {
        workload,
        endpoint: options.endpoint.clone(),
        model: options.model.clone(),
        deployment: options.deployment.clone(),
        policy: None,
        metrics_url: options.metrics_url.clone(),
        metrics_version: if options.metrics_url.is_some() { 2 } else { 1 },
        metrics_auth_env: options.metrics_auth_env.clone(),
        metrics_isolation: options.metrics_isolation.clone(),
        auth_env: options.auth_env.clone(),
        local_http: options.local_http,
    }
}
fn collection_error(native: Option<String>, journal: Option<String>) -> Option<String> {
    match (native, journal) {
        (None, None) => None,
        (Some(error), None) => Some(format!("native collector: {error}")),
        (None, Some(error)) => Some(format!("eviction journal capture: {error}")),
        (Some(native), Some(journal)) => Some(format!(
            "native collector: {native}; eviction journal capture: {journal}"
        )),
    }
}
pub fn collect(options: &Options) -> Result<Report> {
    let (plan, bytes) = load_plan(&options.plan)?;
    if plan.cells.iter().any(|c| c.retention.is_some()) && options.metrics_url.is_none() {
        return Err("retention collection requires explicit metrics2 accounting".into());
    }
    if options.eviction_journal.is_some() && !plan.cells.iter().any(|c| c.retention.is_some()) {
        return Err("eviction journal requires prospective retention phases".into());
    }
    // Admit every native cell before first traffic using the same admission path.
    evidence::fresh(&options.out)?;
    evidence::write(&options.out.join("capacity-plan.json"), &bytes)?;
    for (i, cell) in plan.cells.iter().enumerate() {
        let path = options.out.join(format!("workload-{i:03}.json"));
        evidence::json(&path, &cell.workload)?;
        run::preflight(&common(options, path))?;
    }
    let deployment = options
        .deployment
        .as_ref()
        .map(|path| {
            serde_json::from_slice::<Deployment>(&evidence::read(path, 32 * 1024)?)
                .map_err(|e| e.to_string())
        })
        .transpose()?;
    let execution = Execution {
        version: 1,
        plan_sha256: evidence::digest(&bytes),
        endpoint: crate::wire::endpoint(&options.endpoint, options.local_http)?.to_string(),
        model: options.model.clone(),
        auth_env: options.auth_env.clone(),
        local_http: options.local_http,
        metrics_url: options.metrics_url.clone(),
        metrics_auth_env: options.metrics_auth_env.clone(),
        metrics_isolation: options.metrics_isolation.clone(),
        deployment,
        collector_sha256: evidence::binary_digest()?,
    };
    evidence::json(&options.out.join("execution.json"), &execution)?;
    evidence::sync(&options.out)?;
    let deadline = Instant::now()
        .checked_add(Duration::from_micros(plan.max_wall_us))
        .ok_or("capacity deadline overflow")?;
    for (i, cell) in plan.cells.iter().enumerate() {
        if Instant::now() >= deadline {
            break;
        }
        let journal = cell
            .retention
            .as_ref()
            .map(|r| crate::retention::JournalWindow::begin(options.eviction_journal.as_deref(), r))
            .transpose()?;
        let native = options.out.join(cell_path(i));
        evidence::json(
            &options.out.join(format!("started-{i:03}.json")),
            &Outcome {
                version: 1,
                plan_sha256: execution.plan_sha256.clone(),
                error: None,
            },
        )?;
        let result = run::execute_capacity_bounded(
            &run::Options {
                common: common(options, options.out.join(format!("workload-{i:03}.json"))),
                out: native.clone(),
                json: true,
            },
            deadline,
        );
        let journal_error = journal.and_then(|journal| journal.finish(&options.out, i).err());
        let error = collection_error(result.err(), journal_error);
        evidence::json(
            &options.out.join(format!("outcome-{i:03}.json")),
            &Outcome {
                version: 1,
                plan_sha256: execution.plan_sha256.clone(),
                error,
            },
        )?;
        evidence::sync(&options.out)?;
        let report = inspect(&options.out)?;
        if report.cells[i].state != "successful_tested" {
            break;
        }
    }
    let report = inspect(&options.out)?;
    evidence::publish(&options.out, "capacity-report.json", &report)?;
    Ok(report)
}
pub fn inspect(root: &Path) -> Result<Report> {
    evidence::directory(root)?;
    let (plan, bytes) = load_plan(&root.join("capacity-plan.json"))?;
    let execution: Execution =
        serde_json::from_slice(&evidence::read(&root.join("execution.json"), 64 * 1024)?)
            .map_err(|e| e.to_string())?;
    let plan_hash = evidence::digest(&bytes);
    if execution.version != 1 || execution.plan_sha256 != plan_hash {
        return Err("capacity execution lineage mismatch".into());
    }
    let mut report = Report {
        version: 1,
        kind: "capacity-observation-v1",
        plan_sha256: plan_hash.clone(),
        cells: Vec::new(),
        largest_successful_tested_cell: None,
        scope: "finite ordered cells only; declared context ceilings are not measured context occupancy; protocol-only quality is not semantic correctness; native observations are unauthenticated",
        safe_universal_capacity: "unknown; successful tested cells are not safe capacity, OOM headroom or permission to increase load",
        recovery: "operator-owned; no retry, replacement, search, reset, flush or restart",
    };
    let mut stopped = false;
    for (i, cell) in plan.cells.iter().enumerate() {
        let native_name = cell_path(i);
        let native = root.join(&native_name);
        let planned = cell
            .workload
            .waves()
            .iter()
            .map(|s| u64::from(s.concurrency))
            .sum::<u64>();
        let mut row = CellReport {
            id: cell.id.clone(),
            state: "undispatched",
            context_token_ceiling: cell.context_token_ceiling,
            maximum_concurrency: cell
                .workload
                .cells
                .iter()
                .map(|c| c.concurrency)
                .max()
                .unwrap_or(0),
            success: cell.success.clone(),
            native_run: native_name,
            planned_requests: planned,
            dispatched_requests: Some(0),
            undispatched_requests: Some(planned),
            observed_prompt_tokens: None,
            prompt_usage_observed_requests: 0,
            failures: Vec::new(),
            reasons: Vec::new(),
            resources: Vec::new(),
            retention: None,
            evidence_sha256: None,
        };
        let started = root.join(format!("started-{i:03}.json"));
        if !started.try_exists().map_err(|e| e.to_string())?
            && (native.try_exists().map_err(|e| e.to_string())?
                || root
                    .join(format!("outcome-{i:03}.json"))
                    .try_exists()
                    .map_err(|e| e.to_string())?)
        {
            return Err(
                "capacity cell evidence exists without prospective admission marker".into(),
            );
        }
        if started.try_exists().map_err(|e| e.to_string())? {
            if stopped {
                return Err(
                    "capacity dispatched after an unsuccessful or undispatched cell".into(),
                );
            }
            let marker: Outcome = serde_json::from_slice(&evidence::read(&started, 64 * 1024)?)
                .map_err(|e| e.to_string())?;
            if marker.version != 1 || marker.plan_sha256 != plan_hash || marker.error.is_some() {
                return Err("capacity admission marker mismatch".into());
            }
            row.state = "unsuccessful_tested";
            row.dispatched_requests = None;
            row.undispatched_requests = None;
            let outcome_path = root.join(format!("outcome-{i:03}.json"));
            if outcome_path.try_exists().map_err(|e| e.to_string())? {
                let outcome: Outcome =
                    serde_json::from_slice(&evidence::read(&outcome_path, 64 * 1024)?)
                        .map_err(|e| e.to_string())?;
                if outcome.version != 1 || outcome.plan_sha256 != plan_hash {
                    return Err("capacity outcome lineage mismatch".into());
                }
                if let Some(error) = outcome.error {
                    row.reasons.push(error);
                }
            } else {
                row.reasons
                    .push("cell did not settle; outcome unknown".into());
            }
            match evidence::load(&native) {
                Ok(mut loaded) => {
                    if loaded.plan.workload != cell.workload
                        || loaded.plan.endpoint != execution.endpoint
                        || loaded.plan.model != execution.model
                        || loaded.plan.auth_env != execution.auth_env
                        || loaded.plan.local_http != execution.local_http
                        || loaded.plan.collector_sha256 != execution.collector_sha256
                    {
                        return Err(
                            "capacity native cell differs from prospective execution".into()
                        );
                    }
                    if loaded.plan.deployment != execution.deployment {
                        return Err("capacity deployment declaration mismatch".into());
                    }
                    let expected_metrics = execution
                        .metrics_url
                        .as_ref()
                        .map(|url| {
                            crate::metrics::Protocol::new(
                                crate::wire::endpoint(url, execution.local_http)?.to_string(),
                                loaded.plan.waves.len(),
                                2,
                                execution.metrics_auth_env.clone(),
                                execution.metrics_isolation.clone(),
                                execution.auth_env.as_deref(),
                            )
                        })
                        .transpose()?;
                    if loaded.plan.metrics != expected_metrics {
                        return Err("capacity metrics controls mismatch".into());
                    }
                    if loaded.history.count != 1
                        || loaded.history.open
                        || loaded.history.last_status.as_deref() != Some("completed")
                    {
                        row.reasons
                            .push("native acquisition did not complete normally".into());
                    }
                    let mut dispatched = 0u64;
                    for (spec, wave) in loaded.plan.waves.iter().zip(&loaded.waves) {
                        let Some(wave) = wave else {
                            row.reasons
                                .push(format!("missing required wave {}", spec.index));
                            continue;
                        };
                        for attempt in &wave.attempts {
                            dispatched += u64::from(attempt.dispatched);
                            let correct = cell.success.quality != Quality::AllSequenceChecks
                                || attempt
                                    .sequence
                                    .as_ref()
                                    .is_some_and(crate::sequence::passed);
                            if !wave.eligible || !correct || attempt.status != Status::Complete {
                                let raw = evidence::read(
                                    &evidence::wave_dir(&native, spec.index)
                                        .join(format!("response-{:04}.bin", attempt.lane)),
                                    cell.workload.limits.response_bytes,
                                )?;
                                let reported_oom =
                                    serde_json::from_slice::<serde_json::Value>(&raw)
                                        .ok()
                                        .is_some_and(|body| {
                                            body.get("error")
                                                .and_then(|e| e.get("code"))
                                                .and_then(serde_json::Value::as_str)
                                                == Some("out_of_memory")
                                        });
                                row.failures.push(serde_json::json!({"wave":spec.index,"lane":attempt.lane,"status":attempt.status,
                                    "http_status":attempt.http_status,"dispatched":attempt.dispatched,"eligibility_errors":attempt.eligibility_errors,
                                    "sequence":attempt.sequence,"response_sha256":attempt.response_sha256,
                                    "provider_reported_oom":reported_oom,"oom_scope":"provider assertion only; raw body retained; operator recovery"}));
                            }
                            if cell.success.require_prompt_usage
                                && attempt.usage.prompt_tokens.is_none()
                            {
                                row.reasons
                                    .push("required provider prompt-token count missing".into());
                            }
                            if let Some(tokens) = attempt.usage.prompt_tokens {
                                row.prompt_usage_observed_requests += 1;
                                row.observed_prompt_tokens = Some(
                                    row.observed_prompt_tokens
                                        .map_or([tokens, tokens], |[low, high]| {
                                            [low.min(tokens), high.max(tokens)]
                                        }),
                                );
                            }
                            if attempt
                                .usage
                                .prompt_tokens
                                .is_some_and(|n| n > cell.context_token_ceiling)
                            {
                                row.reasons.push(
                                    "reported prompt tokens exceed declared context ceiling".into(),
                                );
                            }
                        }
                    }
                    if !loaded.states.contains(&"reserved_unsettled") {
                        row.dispatched_requests = Some(dispatched);
                        row.undispatched_requests = Some(
                            planned
                                .checked_sub(dispatched)
                                .ok_or("capacity dispatched count exceeds fixed population")?,
                        );
                    }
                    if !row.failures.is_empty() {
                        row.reasons.push("required response or correctness condition failed; OOM cause is unknown unless explicitly source-observed".into());
                    }
                    if let ResourceCondition::Limits { limits } = &cell.success.resources {
                        if loaded.resources.is_empty() {
                            row.reasons.push("resource observations missing".into());
                        }
                        for resource in &loaded.resources {
                            if !resource.complete_membership
                                || resource.cancelled
                                || resource.failure.is_some()
                            {
                                row.reasons.push("resource acquisition incomplete".into());
                            }
                            for limit in limits {
                                match resource.summaries.get(limit.summary).and_then(|s| s.value) {
                                    Some(value)
                                        if value.denominator > 0
                                            && u128::from(value.numerator)
                                                <= u128::from(limit.maximum)
                                                    * u128::from(value.denominator) => {}
                                    Some(_) => row.reasons.push("resource maximum exceeded".into()),
                                    None => row
                                        .reasons
                                        .push("required resource statistic unavailable".into()),
                                }
                            }
                        }
                    }
                    if let Some(retention) = &cell.retention {
                        let retention_report =
                            crate::retention::report(root, i, retention, &loaded)?;
                        if retention.condition != crate::retention::Condition::ObserveOnly
                            && retention_report["condition_satisfied"] != true
                        {
                            row.reasons
                                .push("required retention condition unavailable or refuted".into());
                        }
                        row.retention = Some(retention_report);
                    }
                    row.resources = std::mem::take(&mut loaded.resources);
                    row.evidence_sha256 = Some(loaded.evidence_sha256);
                    if row.reasons.is_empty() {
                        row.state = "successful_tested";
                    }
                }
                Err(error) => row
                    .reasons
                    .push(format!("native evidence unavailable or invalid: {error}")),
            }
        }
        if row.state == "unsuccessful_tested" && row.dispatched_requests.is_none_or(|n| n == 0) {
            row.state = "unsettled_or_undispatched";
        }
        if row.state == "successful_tested" {
            report.largest_successful_tested_cell = Some(row.id.clone());
        } else {
            stopped = true;
        }
        report.cells.push(row);
    }
    let saved = root.join("capacity-report.json");
    if saved.try_exists().map_err(|e| e.to_string())? {
        let saved: serde_json::Value =
            serde_json::from_slice(&evidence::read(&saved, 16 * 1024 * 1024)?)
                .map_err(|e| e.to_string())?;
        if saved != serde_json::to_value(&report).map_err(|e| e.to_string())? {
            return Err("capacity report does not replay".into());
        }
    }
    Ok(report)
}
#[derive(clap::Subcommand)]
pub enum Command {
    /// Admit the finite cell population without traffic or source reads.
    Preflight { plan: PathBuf },
    /// Collect exactly the declared cells, stopping at the first unsuccessful cell.
    Run(Box<Options>),
    /// Replay every cell and undispatched remainder without network access.
    Inspect { capture: PathBuf },
}
pub fn execute(command: Command) -> Result<u8> {
    match command {
        Command::Preflight { plan } => {
            let (plan, _) = load_plan(&plan)?;
            crate::print_json(&plan.validate()?)?;
            Ok(0)
        }
        Command::Run(options) => {
            let report = collect(&options)?;
            let complete = report.cells.iter().all(|c| c.state == "successful_tested");
            crate::print_json(&report)?;
            Ok(if complete { 0 } else { 2 })
        }
        Command::Inspect { capture } => {
            crate::print_json(&inspect(&capture)?)?;
            Ok(0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settlement_preserves_independent_collector_and_journal_failures() {
        let native = "failed to publish native receipt";
        let journal = "failed to persist journal window";
        let error = collection_error(Some(native.into()), Some(journal.into())).unwrap();
        assert!(error.contains(native));
        assert!(error.contains(journal));
        assert!(
            collection_error(Some(native.into()), None)
                .unwrap()
                .contains(native)
        );
        assert!(
            collection_error(None, Some(journal.into()))
                .unwrap()
                .contains(journal)
        );
        assert_eq!(collection_error(None, None), None);
    }
}
