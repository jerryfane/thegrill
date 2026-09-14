use crate::{
    envelope::{self, Direction, EnvelopeDecision, EnvelopeReason, Rational, range},
    evidence,
    model::*,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::Path;

pub const CAP: usize = 64 * 1024;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    version: u32,
    method: String,
    id: String,
    collector_sha256: String,
    workload_source_sha256: String,
    min_trials: u32,
    #[serde(default, deserialize_with = "Policy::deserialize_telemetry")]
    required_telemetry: Option<Vec<crate::metrics::v2::Requirement>>,
    cells: Vec<CellPolicy>,
}
impl Policy {
    fn deserialize_telemetry<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<Vec<crate::metrics::v2::Requirement>>, D::Error> {
        Vec::deserialize(deserializer).map(Some)
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CellPolicy {
    cell: String,
    metrics: Vec<MetricPolicy>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MetricPolicy {
    metric: Metric,
    #[serde(default, deserialize_with = "MetricPolicy::deserialize_lane")]
    lane: Option<String>,
    max_regression_bps: u32,
    max_reference_spread_bps: u32,
}
impl MetricPolicy {
    fn deserialize_lane<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<String>, D::Error> {
        String::deserialize(deserializer).map(Some)
    }
}
#[derive(Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
enum Metric {
    WaveLatencyUs,
    AchievedCompletionTokensPerSecond,
    DecodeTokensPerSecond,
    PrefillTokensPerSecond,
    FirstGeneratedTextUs,
    FirstAnswerTextUs,
    CompletionLatencyUs,
    WorstLaneFirstAnswerUs,
    FirstAnswerMaxMinRatio,
}
impl Metric {
    fn legacy(self) -> bool {
        matches!(
            self,
            Self::WaveLatencyUs
                | Self::AchievedCompletionTokensPerSecond
                | Self::DecodeTokensPerSecond
                | Self::PrefillTokensPerSecond
        )
    }

    fn aggregate(self) -> bool {
        matches!(self, Self::WorstLaneFirstAnswerUs | Self::FirstAnswerMaxMinRatio)
    }

    fn per_wave(self) -> bool {
        self.aggregate()
            || matches!(self, Self::WaveLatencyUs | Self::AchievedCompletionTokensPerSecond)
    }

    fn direction(self) -> Direction {
        match self {
            Self::WaveLatencyUs
            | Self::FirstGeneratedTextUs
            | Self::FirstAnswerTextUs
            | Self::CompletionLatencyUs
            | Self::WorstLaneFirstAnswerUs
            | Self::FirstAnswerMaxMinRatio => Direction::LowerBetter,
            Self::AchievedCompletionTokensPerSecond
            | Self::DecodeTokensPerSecond
            | Self::PrefillTokensPerSecond => Direction::HigherBetter,
        }
    }
}
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Reason {
    InvalidEvidence,
    InvalidPolicy,
    PolicyHashMismatch,
    PolicySourceMismatch,
    PolicyCollectorMismatch,
    PolicyScopeMismatch,
    InsufficientDeclaredTrials,
    MissingPolicy,
    ConflictingPolicy,
    MissingReference,
    IncompatibleEvidence,
    ReferenceUnqualified,
    RoleReuse,
    DeclaredStartsOutOfOrder,
    SessionIncomplete,
    WarmupIncomplete,
    MetricUnavailable,
    OutputAmountsMismatch,
    NonpositiveReference,
    ReferenceSpreadExceeded,
    EnvelopeStraddlesTolerance,
    ArithmeticOverflow,
    EvaluatorUnavailable,
    RequiredTelemetryUnavailable,
    RequiredTelemetryRefuted,
    InvalidRequiredTelemetry,
}
impl Reason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InvalidEvidence => "invalid_evidence",
            Self::InvalidPolicy => "invalid_policy",
            Self::PolicyHashMismatch => "policy_hash_mismatch",
            Self::PolicySourceMismatch => "policy_source_mismatch",
            Self::PolicyCollectorMismatch => "policy_collector_mismatch",
            Self::PolicyScopeMismatch => "policy_scope_mismatch",
            Self::InsufficientDeclaredTrials => "insufficient_declared_trials",
            Self::MissingPolicy => "missing_policy",
            Self::ConflictingPolicy => "conflicting_policy",
            Self::MissingReference => "missing_reference",
            Self::IncompatibleEvidence => "incompatible_evidence",
            Self::ReferenceUnqualified => "reference_unqualified",
            Self::RoleReuse => "role_reuse",
            Self::DeclaredStartsOutOfOrder => "declared_starts_out_of_order",
            Self::SessionIncomplete => "session_incomplete",
            Self::WarmupIncomplete => "warmup_incomplete",
            Self::MetricUnavailable => "metric_unavailable",
            Self::OutputAmountsMismatch => "output_amounts_mismatch",
            Self::NonpositiveReference => "nonpositive_reference",
            Self::ReferenceSpreadExceeded => "reference_spread_exceeded",
            Self::EnvelopeStraddlesTolerance => "envelope_straddles_tolerance",
            Self::ArithmeticOverflow => "arithmetic_overflow",
            Self::EvaluatorUnavailable => "evaluator_unavailable",
            Self::RequiredTelemetryUnavailable => "required_telemetry_unavailable",
            Self::RequiredTelemetryRefuted => "required_telemetry_refuted",
            Self::InvalidRequiredTelemetry => "invalid_required_telemetry",
        }
    }
}
fn sha(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
pub fn parse(
    bytes: &[u8],
    collector_sha256: &str,
    source_sha256: &str,
    workload: &Workload,
    telemetry: Option<&crate::metrics::Protocol>,
) -> Result<Policy, Reason> {
    if bytes.len() > CAP {
        return Err(Reason::InvalidPolicy);
    }
    let policy: Policy = serde_json::from_slice(bytes).map_err(|_| Reason::InvalidPolicy)?;
    if !matches!(
        (policy.version, policy.method.as_str()),
        (1, "observed-envelope-v1") | (2, "observed-envelope-v2")
    )
        || !identifier(&policy.id)
        || !sha(&policy.collector_sha256)
        || !sha(&policy.workload_source_sha256)
        || !(3..=100).contains(&policy.min_trials)
    {
        return Err(Reason::InvalidPolicy);
    }
    if policy.collector_sha256 != collector_sha256 {
        return Err(Reason::PolicyCollectorMismatch);
    }
    if policy.workload_source_sha256 != source_sha256 {
        return Err(Reason::PolicySourceMismatch);
    }
    // This first policy2 slice admits homogeneous flat workloads only.
    // Schedule selectors and repeated conversations require their own integration.
    if (policy.version == 1 && workload.version >= 4)
        || (policy.version == 2 && !matches!(workload.version, 1 | 3))
    {
        return Err(Reason::PolicyScopeMismatch);
    }
    if let Some(requirements) = &policy.required_telemetry {
        if policy.version != 2 || requirements.is_empty() || requirements.len() > 256 {
            return Err(Reason::InvalidPolicy);
        }
        let Some(crate::metrics::Protocol::V2(config)) = telemetry else {
            return Err(Reason::PolicyScopeMismatch);
        };
        let mut identities = HashSet::new();
        for requirement in requirements {
            if !identities.insert((&requirement.name, &requirement.labels))
                || crate::metrics::v2::validate_requirement(requirement).is_err()
            {
                return Err(Reason::InvalidRequiredTelemetry);
            }
            if requirement.predicate == crate::metrics::v2::Predicate::ZeroCounterDelta
                && config.isolation != crate::metrics::v2::ISOLATION_EXCLUSIVE
            {
                return Err(Reason::PolicyScopeMismatch);
            }
        }
    }
    if policy.cells.len() != workload.cells.len() {
        return Err(Reason::PolicyScopeMismatch);
    }
    let mut cells = HashSet::new();
    for declared in &policy.cells {
        let cell = workload
            .cells
            .iter()
            .find(|c| c.id == declared.cell)
            .ok_or(Reason::PolicyScopeMismatch)?;
        if !cells.insert(&declared.cell)
            || declared.metrics.is_empty()
            || declared.metrics.len() > if policy.version == 1 { 4 } else { 9 }
        {
            return Err(Reason::PolicyScopeMismatch);
        }
        if cell.trials < policy.min_trials {
            return Err(Reason::InsufficientDeclaredTrials);
        }
        let mut metrics = HashSet::new();
        for metric in &declared.metrics {
            if !metrics.insert(metric.metric)
                || metric.max_regression_bps > 9999
                || metric.max_reference_spread_bps > 1_000_000
                || (policy.version == 1 && !metric.metric.legacy())
                || metric.lane.is_some()
                || (metric.metric.aggregate() && cell.concurrency < 2)
            {
                return Err(Reason::InvalidPolicy);
            }
        }
    }
    Ok(policy)
}

// Declaration order is also the aggregate precedence, not a vote across gates.
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "UPPERCASE")]
pub enum Outcome {
    Pass,
    Inconclusive,
    Regression,
    Error,
}
impl Outcome {
    pub fn exit(self) -> u8 {
        match self {
            Self::Pass => 0,
            Self::Error => 1,
            Self::Inconclusive => 2,
            Self::Regression => 3,
        }
    }
}
#[derive(Serialize)]
struct Coverage {
    expected_waves: u32,
    observed_waves: usize,
    eligible_waves: usize,
    expected_observations: u32,
    observed_observations: usize,
    expected_warmups: u32,
    eligible_warmups: usize,
}
#[derive(Serialize)]
struct Roles<T> {
    baseline: T,
    candidate: T,
    reference: T,
}
#[derive(Serialize)]
struct RoleIdentity {
    plan_sha256: String,
    evidence_sha256: String,
    collector_sha256: String,
    workload_source_sha256: String,
    policy_sha256: Option<String>,
}
#[derive(Serialize)]
struct Gate {
    cell: String,
    metric: Metric,
    #[serde(skip_serializing_if = "Option::is_none")]
    sample_unit: Option<&'static str>,
    max_regression_bps: u32,
    max_reference_spread_bps: u32,
    coverage: Roles<Coverage>,
    ranges: Roles<Option<[Rational; 2]>>,
    pooled_reference_range: Option<[Rational; 2]>,
    adverse_bounds: Option<[f64; 2]>,
    decision: Outcome,
    reason_codes: Vec<Reason>,
}
#[derive(Serialize)]
pub struct Decision {
    version: u32,
    claim: &'static str,
    pub decision: Outcome,
    pub eligibility: bool,
    policy_sha256: Option<String>,
    policy_id: Option<String>,
    min_trials: Option<u32>,
    evaluator_sha256: Option<String>,
    roles: Roles<Option<RoleIdentity>>,
    gates: Vec<Gate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    required_telemetry: Option<Roles<Option<crate::metrics::v2::Assessment>>>,
    reason_codes: Vec<Reason>,
}
fn observations(
    run: Option<&evidence::Loaded>,
    cell: &Cell,
    metric: Metric,
) -> (Coverage, Vec<Rational>) {
    let mut coverage = Coverage {
        expected_waves: cell.trials,
        observed_waves: 0,
        eligible_waves: 0,
        expected_observations: cell.trials
            * if metric.per_wave() { 1 } else { cell.concurrency },
        observed_observations: 0,
        expected_warmups: cell.warmup_trials,
        eligible_warmups: 0,
    };
    let mut values = Vec::new();
    if let Some(run) = run {
        for wave in run
            .waves
            .iter()
            .flatten()
            .filter(|w| w.spec.cell == cell.id)
        {
            if wave.spec.phase == Phase::Warmup {
                coverage.eligible_warmups += usize::from(wave.eligible);
                continue;
            }
            coverage.observed_waves += 1;
            coverage.eligible_waves += usize::from(wave.eligible);
            match metric {
                Metric::WaveLatencyUs => {
                    if wave.elapsed_us > 0 {
                        values.push((wave.elapsed_us, 1).into());
                    }
                }
                Metric::AchievedCompletionTokensPerSecond => {
                    if let Some(tokens) = wave
                        .completion_tokens
                        .filter(|_| wave.eligible && wave.elapsed_us > 0)
                    {
                        values.push((tokens, wave.elapsed_us).into());
                    }
                }
                Metric::DecodeTokensPerSecond | Metric::PrefillTokensPerSecond => {
                    values.extend(
                        wave.attempts
                            .iter()
                            .filter_map(|a| {
                                if metric == Metric::DecodeTokensPerSecond {
                                    evidence::decode_sample(a)
                                } else {
                                    evidence::prefill_sample(a)
                                }
                            })
                            .map(Rational::from),
                    );
                }
                Metric::FirstGeneratedTextUs
                | Metric::FirstAnswerTextUs
                | Metric::CompletionLatencyUs => {
                    values.extend(wave.attempts.iter().filter_map(|a| latency_sample(a, metric)));
                }
                Metric::WorstLaneFirstAnswerUs | Metric::FirstAnswerMaxMinRatio => {
                    if let Some(sample) = fairness_sample(wave, metric) {
                        values.push(sample);
                    }
                }
            }
        }
    }
    coverage.observed_observations = values.len();
    (coverage, values)
}

fn latency_sample(attempt: &Attempt, metric: Metric) -> Option<Rational> {
    if !attempt.dispatched
        || attempt.status != Status::Complete
        || !attempt.eligibility_errors.is_empty()
    {
        return None;
    }
    let value = match metric {
        Metric::FirstGeneratedTextUs => attempt.timing.first_generated_text_us?,
        Metric::FirstAnswerTextUs => attempt.timing.first_answer_text_us?,
        Metric::CompletionLatencyUs => attempt.timing.settle_us,
        _ => return None,
    };
    (value > 0).then(|| (value, 1).into())
}

fn fairness_sample(wave: &Wave, metric: Metric) -> Option<Rational> {
    if !wave.eligible
        || wave.spec.concurrency < 2
        || wave.attempts.len() != wave.spec.concurrency as usize
    {
        return None;
    }
    let mut low = u64::MAX;
    let mut high = 0;
    for (lane, attempt) in wave.attempts.iter().enumerate() {
        // Full admitted-lane coverage, never a ratio over surviving peers.
        if attempt.lane as usize != lane {
            return None;
        }
        let value = latency_sample(attempt, Metric::FirstAnswerTextUs)?.numerator;
        low = low.min(value);
        high = high.max(value);
    }
    match metric {
        Metric::WorstLaneFirstAnswerUs => Some((high, 1).into()),
        Metric::FirstAnswerMaxMinRatio => Some((high, low).into()),
        _ => None,
    }
}
fn amounts_match(a: &evidence::Loaded, b: &evidence::Loaded, cell: &str) -> bool {
    a.plan
        .waves
        .iter()
        .enumerate()
        .filter(|(_, s)| s.phase == Phase::Measured && s.cell == cell)
        .all(|(i, _)| match (&a.waves[i], &b.waves[i]) {
            (Some(a), Some(b)) => a.attempts.iter().zip(&b.attempts).all(|(a, b)| {
                a.usage.completion_tokens.is_some()
                    && a.usage.completion_tokens == b.usage.completion_tokens
            }),
            _ => false,
        })
}
fn envelope_reason(reason: EnvelopeReason) -> Reason {
    match reason {
        EnvelopeReason::InvalidRational | EnvelopeReason::InvalidBounds => Reason::InvalidEvidence,
        EnvelopeReason::ArithmeticOverflow { .. } => Reason::ArithmeticOverflow,
        EnvelopeReason::NonpositiveReference => Reason::NonpositiveReference,
        EnvelopeReason::ReferenceSpreadExceeded => Reason::ReferenceSpreadExceeded,
        EnvelopeReason::EnvelopeStraddlesTolerance { .. } => Reason::EnvelopeStraddlesTolerance,
    }
}

fn evaluate(
    gate: &mut Gate,
    pooled: [Rational; 2],
    candidate: [Rational; 2],
) -> Result<(), Reason> {
    gate.pooled_reference_range = Some(pooled);
    match envelope::assess(
        pooled,
        candidate,
        gate.max_regression_bps,
        gate.max_reference_spread_bps,
        gate.metric.direction(),
    ) {
        Ok(assessment) => {
            gate.adverse_bounds = Some(assessment.adverse_bounds);
            gate.decision = match assessment.decision {
                EnvelopeDecision::Pass => Outcome::Pass,
                EnvelopeDecision::Regression => Outcome::Regression,
            };
            Ok(())
        }
        Err(reason) => {
            gate.adverse_bounds = match reason {
                EnvelopeReason::ArithmeticOverflow { adverse_bounds } => adverse_bounds,
                EnvelopeReason::EnvelopeStraddlesTolerance { adverse_bounds } => Some(adverse_bounds),
                _ => None,
            };
            Err(envelope_reason(reason))
        }
    }
}
fn add(reasons: &mut Vec<Reason>, reason: Reason) {
    if !reasons.contains(&reason) {
        reasons.push(reason);
    }
}
pub fn decide(a: &Path, b: &Path, reference: Option<&Path>) -> Decision {
    let mut result = Decision {
        version: 1,
        claim: "observed-policy-decision-not-statistical-or-causal",
        decision: Outcome::Pass,
        eligibility: false,
        policy_sha256: None,
        policy_id: None,
        min_trials: None,
        evaluator_sha256: None,
        roles: Roles {
            baseline: None,
            candidate: None,
            reference: None,
        },
        gates: Vec::new(),
        required_telemetry: None,
        reason_codes: Vec::new(),
    };
    match evidence::binary_digest() {
        Ok(hash) => result.evaluator_sha256 = Some(hash),
        Err(_) => {
            result.decision = Outcome::Error;
            add(&mut result.reason_codes, Reason::EvaluatorUnavailable);
        }
    }
    let mut load = |path: Option<&Path>| match path.map(evidence::load_verified) {
        Some(Ok(run)) if sha(&run.plan.collector_sha256) => Some(run),
        Some(Ok(_)) => {
            result.decision = Outcome::Error;
            add(&mut result.reason_codes, Reason::InvalidEvidence);
            None
        }
        Some(Err(error)) => {
            result.decision = Outcome::Error;
            add(&mut result.reason_codes, error.reason);
            None
        }
        None => {
            add(&mut result.reason_codes, Reason::MissingReference);
            None
        }
    };
    let left = load(Some(a));
    let right = load(Some(b));
    let repeat = load(reference);
    let runs = [left.as_ref(), right.as_ref(), repeat.as_ref()];
    let identity = |run: Option<&evidence::Loaded>| {
        run.map(|r| RoleIdentity {
            plan_sha256: r.plan_sha256.clone(),
            evidence_sha256: r.evidence_sha256.clone(),
            collector_sha256: r.plan.collector_sha256.clone(),
            workload_source_sha256: r.plan.source_sha256.clone(),
            policy_sha256: r.plan.policy_sha256.clone(),
        })
    };
    result.roles = Roles {
        baseline: identity(runs[0]),
        candidate: identity(runs[1]),
        reference: identity(runs[2]),
    };
    let mut bound = None;
    for run in runs.into_iter().flatten() {
        match run.plan.policy_sha256.as_ref() {
            None => add(&mut result.reason_codes, Reason::MissingPolicy),
            Some(hash) => {
                if bound.is_some_and(|prior| prior != hash) {
                    result.decision = Outcome::Error;
                    add(&mut result.reason_codes, Reason::ConflictingPolicy);
                }
                bound = Some(hash);
            }
        }
        if run.history.count != 1
            || run.history.open
            || run.history.last_status.as_deref() != Some("completed")
        {
            add(&mut result.reason_codes, Reason::SessionIncomplete);
        }
        if let Some(left) = &left
            && !evidence::compatible(&left.plan, &run.plan)
        {
            result.decision = Outcome::Error;
            add(&mut result.reason_codes, Reason::IncompatibleEvidence);
        }
    }
    for i in 0..runs.len() {
        for j in i + 1..runs.len() {
            if let (Some(a), Some(b)) = (runs[i], runs[j]) {
                if a.plan_sha256 == b.plan_sha256 || a.lineage_sha256 == b.lineage_sha256 {
                    add(&mut result.reason_codes, Reason::RoleReuse);
                }
                if a.plan.started_unix_ms >= b.plan.started_unix_ms {
                    add(&mut result.reason_codes, Reason::DeclaredStartsOutOfOrder);
                }
            }
        }
    }
    if let (Some(a), Some(a2)) = (runs[0], runs[2])
        && evidence::reference_identity(&a.plan, &a2.plan).status
            != evidence::ReferenceIdentityStatus::DeclaredMatch
    {
        add(&mut result.reason_codes, Reason::ReferenceUnqualified);
    }
    let qualified = result.reason_codes.is_empty();
    let invalid = result.decision == Outcome::Error;
    if !qualified {
        result.decision = result.decision.max(Outcome::Inconclusive);
    }
    if let Some(a) = &left {
        result.policy_sha256 = a.plan.policy_sha256.clone();
        if let Some(policy) = &a.policy {
            result.version = policy.version;
            result.policy_id = Some(policy.id.clone());
            result.min_trials = Some(policy.min_trials);
            for declared in &policy.cells {
                // Policy admission already proved exact cell coverage.
                let cell = a
                    .plan
                    .workload
                    .cells
                    .iter()
                    .find(|c| c.id == declared.cell)
                    .unwrap();
                for metric in &declared.metrics {
                    let [(ac, av), (bc, bv), (rc, rv)] =
                        runs.map(|r| observations(r, cell, metric.metric));
                    let sampled_ranges = [&av, &bv, &rv].map(|values| range(values));
                    let mut gate = Gate {
                        cell: cell.id.clone(),
                        metric: metric.metric,
                        sample_unit: (policy.version == 2).then_some(if metric.metric.per_wave() {
                            "wave-repetition"
                        } else {
                            "lane-observations-within-wave-repetitions"
                        }),
                        max_regression_bps: metric.max_regression_bps,
                        max_reference_spread_bps: metric.max_reference_spread_bps,
                        coverage: Roles {
                            baseline: ac,
                            candidate: bc,
                            reference: rc,
                        },
                        ranges: Roles {
                            baseline: sampled_ranges[0].unwrap_or(None),
                            candidate: sampled_ranges[1].unwrap_or(None),
                            reference: sampled_ranges[2].unwrap_or(None),
                        },
                        pooled_reference_range: None,
                        adverse_bounds: None,
                        decision: Outcome::Inconclusive,
                        reason_codes: result.reason_codes.clone(),
                    };
                    for reason in sampled_ranges.into_iter().filter_map(|result| result.err()) {
                        gate.decision = Outcome::Error;
                        add(&mut gate.reason_codes, envelope_reason(reason));
                    }
                    for coverage in [
                        &gate.coverage.baseline,
                        &gate.coverage.candidate,
                        &gate.coverage.reference,
                    ] {
                        if coverage.expected_warmups == 0
                            || coverage.eligible_warmups != coverage.expected_warmups as usize
                        {
                            add(&mut gate.reason_codes, Reason::WarmupIncomplete);
                        }
                        if coverage.eligible_waves != coverage.expected_waves as usize
                            || coverage.observed_observations
                                != coverage.expected_observations as usize
                        {
                            add(&mut gate.reason_codes, Reason::MetricUnavailable);
                        }
                    }
                    if let (Some(b), Some(r)) = (runs[1], runs[2])
                        && evidence::compatible(&a.plan, &b.plan)
                        && evidence::compatible(&a.plan, &r.plan)
                        && (!amounts_match(a, b, &cell.id) || !amounts_match(a, r, &cell.id))
                    {
                        add(&mut gate.reason_codes, Reason::OutputAmountsMismatch);
                    }
                    if gate.reason_codes.is_empty()
                        && let (Some(ar), Some(rr), Some(br)) = (
                            gate.ranges.baseline,
                            gate.ranges.reference,
                            gate.ranges.candidate,
                        )
                    {
                        let assessment = range(&[ar[0], ar[1], rr[0], rr[1]])
                            .map_err(envelope_reason)
                            .and_then(|pooled| pooled.ok_or(Reason::InvalidEvidence))
                            .and_then(|pooled| evaluate(&mut gate, pooled, br));
                        if let Err(reason) = assessment {
                            gate.decision = if matches!(reason, Reason::ArithmeticOverflow | Reason::InvalidEvidence) {
                                Outcome::Error
                            } else {
                                Outcome::Inconclusive
                            };
                            add(&mut gate.reason_codes, reason);
                        }
                    } else if invalid {
                        gate.decision = Outcome::Error;
                    }
                    result.decision = result.decision.max(gate.decision);
                    result.gates.push(gate);
                }
            }
        }
    }
    let mut telemetry_eligible = true;
    if let Some(requirements) = left
        .as_ref()
        .and_then(|run| run.policy.as_ref())
        .and_then(|policy| policy.required_telemetry.as_ref())
    {
        use crate::metrics::v2::AssessmentOutcome;
        let assessments = runs.map(|run| {
            run.and_then(|run| run.metrics.as_ref())
                .map(|summary| crate::metrics::v2::assess(requirements, summary))
        });
        for (role, assessment) in assessments.iter().enumerate() {
            match assessment.as_ref().map(|assessment| assessment.outcome) {
                Some(AssessmentOutcome::Satisfied) => (),
                Some(AssessmentOutcome::Refuted) if role == 1 => {
                    result.decision = result.decision.max(Outcome::Regression);
                    add(&mut result.reason_codes, Reason::RequiredTelemetryRefuted);
                }
                Some(AssessmentOutcome::Refuted) => {
                    result.decision = result.decision.max(Outcome::Inconclusive);
                    add(&mut result.reason_codes, Reason::ReferenceUnqualified);
                    telemetry_eligible = false;
                }
                Some(AssessmentOutcome::Error) => {
                    result.decision = Outcome::Error;
                    add(&mut result.reason_codes, Reason::InvalidRequiredTelemetry);
                    telemetry_eligible = false;
                }
                Some(AssessmentOutcome::Unavailable) | None => {
                    result.decision = result.decision.max(Outcome::Inconclusive);
                    add(&mut result.reason_codes, Reason::RequiredTelemetryUnavailable);
                    telemetry_eligible = false;
                }
            }
        }
        let [baseline, candidate, reference] = assessments;
        result.required_telemetry = Some(Roles { baseline, candidate, reference });
    }
    result.eligibility = qualified
        && telemetry_eligible
        && result.gates.iter().all(|g| {
            g.reason_codes.iter().all(|r| {
                matches!(
                    r,
                    Reason::ReferenceSpreadExceeded
                        | Reason::EnvelopeStraddlesTolerance
                        | Reason::NonpositiveReference
                )
            })
        });
    result
}

#[cfg(test)]
#[path = "../tests/support/policy_calibration.rs"]
mod policy_calibration;

#[cfg(test)]
#[path = "../tests/support/policy_observations.rs"]
mod policy_observations;
