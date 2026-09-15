//! Explicit workload6 populations. All offsets use one capture-local monotonic clock.
use crate::model::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub const HISTORY_CAP: usize = 256 * 1024 * 1024;
pub const WAVE_CAP: usize = 10_000;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Protocol {
    Flat {
        input_bytes: usize,
    },
    Conversation {
        repetitions: u32,
        warmup_repetitions: u32,
        measured_steps: Vec<String>,
        input_bytes: usize,
        retained_history_bytes: usize,
    },
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AcquisitionIdentity {
    pub phase: Phase,
    pub index: u32,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct StepClock {
    pub clock_id: String,
    pub kind: ClockKind,
    pub units: ClockUnits,
    pub started_offset_us: u64,
    pub settled_offset_us: u64,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ClockKind {
    StdInstantMonotonic,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ClockUnits {
    Microseconds,
}
impl Protocol {
    pub fn input_bytes(&self) -> usize {
        match self {
            Self::Flat { input_bytes } | Self::Conversation { input_bytes, .. } => *input_bytes,
        }
    }
    pub fn history_bytes(&self) -> usize {
        match self {
            Self::Flat { .. } => 0,
            Self::Conversation {
                retained_history_bytes,
                ..
            } => *retained_history_bytes,
        }
    }
    pub fn conversation(&self) -> bool {
        matches!(self, Self::Conversation { .. })
    }
    pub fn measured(&self, case: &str) -> bool {
        match self {
            Self::Flat { .. } => true,
            Self::Conversation { measured_steps, .. } => measured_steps.iter().any(|s| s == case),
        }
    }
    pub fn counts(&self, cell: &Cell) -> (u32, u32) {
        match self {
            Self::Flat { .. } => (cell.trials, cell.warmup_trials),
            Self::Conversation {
                repetitions,
                warmup_repetitions,
                ..
            } => (*repetitions, *warmup_repetitions),
        }
    }
}
pub fn conversation(workload: &Workload) -> bool {
    matches!(workload.version, 2 | 5)
        || workload
            .acquisition
            .as_ref()
            .is_some_and(Protocol::conversation)
}
pub fn input_cap(workload: &Workload) -> usize {
    workload
        .acquisition
        .as_ref()
        .map_or(REQUEST_CAP, Protocol::input_bytes)
}
pub fn namespace(original: &str, identity: AcquisitionIdentity) -> String {
    let phase = match identity.phase {
        Phase::Warmup => "warmup",
        Phase::Measured => "measured",
    };
    crate::evidence::digest(
        format!("grill-acquisition-v1:{original}:{phase}:{}", identity.index).as_bytes(),
    )
}
pub fn validate(workload: &Workload) -> Result<()> {
    if workload.version != 6 {
        return if workload.acquisition.is_none() {
            Ok(())
        } else {
            Err("acquisition requires workload6".into())
        };
    }
    let protocol = workload
        .acquisition
        .as_ref()
        .ok_or("workload6 requires explicit acquisition")?;
    if !(1..=REQUEST_CAP).contains(&protocol.input_bytes()) || workload.schedule.is_some() {
        return Err("workload6 requires bounded input_bytes and no schedule".into());
    }
    match protocol {
        Protocol::Flat { .. } => {
            if !matches!(
                workload.request.profile,
                Profile::PortableChatV1 | Profile::VllmFixedV1
            ) {
                return Err("flat acquisitions require a flat profile".into());
            }
        }
        Protocol::Conversation {
            repetitions,
            warmup_repetitions,
            measured_steps,
            retained_history_bytes,
            ..
        } => {
            if workload.request.profile != Profile::VllmConversationV3
                || !(1..=1000).contains(repetitions)
                || *warmup_repetitions > 20
                || !(1..=HISTORY_CAP).contains(retained_history_bytes)
                || measured_steps.is_empty()
                || measured_steps.len() > 128
            {
                return Err("invalid conversation acquisition population or budgets".into());
            }
            let mut seen = HashSet::new();
            for step in measured_steps {
                if !seen.insert(step) || !workload.cases.iter().any(|c| &c.id == step) {
                    return Err("measured_steps must be unique existing case IDs".into());
                }
            }
        }
    }
    Ok(())
}
pub fn waves(workload: &Workload) -> Vec<WaveSpec> {
    let protocol = workload
        .acquisition
        .as_ref()
        .expect("validated acquisition");
    let mut result = Vec::new();
    let mut append = |cell: &Cell, phase, index| {
        result.push(WaveSpec {
            index: result.len() as u32,
            phase,
            cell: cell.id.clone(),
            case: Some(cell.case.clone()),
            trial: index,
            concurrency: cell.concurrency,
            lanes: None,
            acquisition: Some(AcquisitionIdentity { phase, index }),
        })
    };
    for phase in [Phase::Warmup, Phase::Measured] {
        match protocol {
            Protocol::Flat { .. } => {
                for cell in &workload.cells {
                    let (measured, warmup) = protocol.counts(cell);
                    for index in 0..if phase == Phase::Warmup {
                        warmup
                    } else {
                        measured
                    } {
                        append(cell, phase, index);
                    }
                }
            }
            Protocol::Conversation {
                repetitions,
                warmup_repetitions,
                ..
            } => {
                for index in 0..if phase == Phase::Warmup {
                    *warmup_repetitions
                } else {
                    *repetitions
                } {
                    for cell in &workload.cells {
                        append(cell, phase, index);
                    }
                }
            }
        }
    }
    result
}

#[derive(Clone, Debug, Serialize)]
pub struct Budget {
    pub warmup_requests: u64,
    pub control_requests: u64,
    pub measured_requests: u64,
    pub output_token_ceiling: u64,
    pub encoded_input_byte_ceiling: u64,
    pub response_byte_ceiling: u64,
    pub tool_trace_byte_ceiling: u64,
    pub wall_time_ceiling_us: u64,
    pub retained_history_bytes: usize,
    pub serialized_bounds_not_rss_guarantees: bool,
}
pub fn budget(workload: &Workload) -> Result<Budget> {
    let protocol = workload.acquisition.as_ref().ok_or("missing acquisition")?;
    let mut b = Budget {
        warmup_requests: 0,
        control_requests: 0,
        measured_requests: 0,
        output_token_ceiling: 0,
        encoded_input_byte_ceiling: 0,
        response_byte_ceiling: 0,
        tool_trace_byte_ceiling: 0,
        wall_time_ceiling_us: 0,
        retained_history_bytes: protocol.history_bytes(),
        serialized_bounds_not_rss_guarantees: true,
    };
    for wave in workload.waves() {
        let n = u64::from(wave.concurrency);
        let count = if wave.phase == Phase::Warmup {
            &mut b.warmup_requests
        } else if protocol.measured(wave.case.as_deref().ok_or("missing case")?) {
            &mut b.measured_requests
        } else {
            &mut b.control_requests
        };
        *count = count.checked_add(n).ok_or("acquisition count overflow")?;
        b.output_token_ceiling = b
            .output_token_ceiling
            .checked_add(n * u64::from(workload.request.effective_output(wave.phase).tokens))
            .ok_or("output budget overflow")?;
        b.encoded_input_byte_ceiling = b
            .encoded_input_byte_ceiling
            .checked_add(n * protocol.input_bytes() as u64)
            .ok_or("input budget overflow")?;
        b.response_byte_ceiling = b
            .response_byte_ceiling
            .checked_add(n * workload.limits.response_bytes as u64)
            .ok_or("response budget overflow")?;
        if workload
            .cases
            .iter()
            .find(|c| Some(c.id.as_str()) == wave.case.as_deref())
            .and_then(|c| c.step.as_ref())
            .is_some_and(|s| matches!(s.expect, crate::sequence::Expected::Tool { .. }))
        {
            b.tool_trace_byte_ceiling = b
                .tool_trace_byte_ceiling
                .checked_add(n * TOOL_TRACE_ALLOWANCE as u64)
                .ok_or("trace budget overflow")?;
        }
        b.wall_time_ceiling_us = b
            .wall_time_ceiling_us
            .checked_add(u64::from(workload.limits.total_ms) * 1000)
            .ok_or("time budget overflow")?;
    }
    Ok(b)
}

// Performance eligibility deliberately retains timings for incorrect answers.
// Complete acquisitions additionally require every declared semantic check.
fn step_eligible(wave: &Wave) -> bool {
    wave.eligible
        && wave
            .attempts
            .iter()
            .all(|a| a.sequence.as_ref().is_none_or(crate::sequence::passed))
}

/// Complete, ordered required membership; controls and warmups cannot be dropped.
pub fn whole_samples(plan: &Plan, waves: &[Option<Wave>], phase: Phase) -> Result<Vec<u64>> {
    let Some(Protocol::Conversation {
        repetitions,
        warmup_repetitions,
        ..
    }) = &plan.workload.acquisition
    else {
        return Err("whole samples require conversation acquisitions".into());
    };
    let count = if phase == Phase::Warmup {
        *warmup_repetitions
    } else {
        *repetitions
    };
    let mut samples = Vec::with_capacity(count as usize);
    for index in 0..count {
        samples.push(whole_sample(
            plan,
            waves,
            AcquisitionIdentity { phase, index },
        )?);
    }
    Ok(samples)
}
pub fn whole_sample(
    plan: &Plan,
    waves: &[Option<Wave>],
    identity: AcquisitionIdentity,
) -> Result<u64> {
    let mut start = None;
    let mut end = 0;
    let mut members = 0;
    for spec in plan
        .waves
        .iter()
        .filter(|s| s.acquisition == Some(identity))
    {
        let wave = waves
            .get(spec.index as usize)
            .and_then(Option::as_ref)
            .ok_or("missing acquisition step")?;
        if !step_eligible(wave) || wave.spec != *spec {
            return Err("unqualified acquisition step".into());
        }
        let clock = wave
            .acquisition_clock
            .as_ref()
            .ok_or("missing acquisition clock")?;
        if clock.clock_id != wave.plan_sha256
            || clock.started_offset_us < end
            || clock.settled_offset_us < clock.started_offset_us
        {
            return Err("unordered acquisition clock".into());
        }
        start.get_or_insert(clock.started_offset_us);
        end = clock.settled_offset_us;
        members += 1;
    }
    if members != plan.workload.cells.len() {
        return Err("incomplete acquisition membership".into());
    }
    end.checked_sub(start.ok_or("empty acquisition")?)
        .filter(|n| *n > 0)
        .ok_or_else(|| "nonpositive acquisition span".into())
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Percentile {
    P95,
    P99,
}
impl Percentile {
    pub fn floor(self) -> usize {
        match self {
            Self::P95 => 200,
            Self::P99 => 1000,
        }
    }
    pub fn nearest_rank(self, values: &mut [u64]) -> Option<u64> {
        if values.len() < self.floor() {
            return None;
        }
        values.sort_unstable();
        let p = match self {
            Self::P95 => 95usize,
            Self::P99 => 99,
        };
        let rank = p.checked_mul(values.len())?.div_ceil(100);
        values.get(rank.checked_sub(1)?).copied()
    }
}

/// Preflight uses encoded current inputs and a conservative escaped-response
/// bound for unknown actual parents, never substitutes expected answers.
pub fn preflight(context: &crate::wire::BodyContext<'_>) -> Result<()> {
    let workload = context.workload;
    let protocol = workload.acquisition.as_ref().ok_or("missing acquisition")?;
    let mut history_bounds: Vec<(String, usize)> = Vec::new();
    let mut identity = None;
    for spec in workload.waves() {
        if protocol.conversation() && identity != spec.acquisition {
            history_bounds.clear();
            identity = spec.acquisition;
        }
        let mut escaped_total = 0usize;
        for lane in 0..spec.concurrency {
            let body = crate::wire::request_body(context, &spec, lane)?;
            let mut bound = body.len();
            if protocol.conversation() {
                let case = workload
                    .cases
                    .iter()
                    .find(|c| Some(c.id.as_str()) == spec.case.as_deref())
                    .ok_or("unknown case")?;
                let step = case.step.as_ref().ok_or("missing step")?;
                if let Some(parent) = &step.parent {
                    bound = bound
                        .checked_add(
                            history_bounds
                                .iter()
                                .find(|(id, _)| id == parent)
                                .ok_or("missing parent bound")?
                                .1,
                        )
                        .ok_or("history input bound overflow")?;
                }
                // Only already-declared input bytes can be known offline.
                // Unknown actual parent outputs consume the prospective cap at
                // runtime; reaching it is a retained admission failure, not a
                // shortened prompt or a substituted expected response.
                if matches!(step.expect, crate::sequence::Expected::Tool { .. }) {
                    bound = bound.checked_add(2048).ok_or("input bound overflow")?;
                }
                history_bounds.push((case.id.clone(), bound));
            }
            if bound > protocol.input_bytes() {
                return Err("expanded acquisition input allowance exceeds input_bytes".into());
            }
            let escaped = if protocol.conversation() {
                protocol
                    .input_bytes()
                    .checked_mul(6)
                    .ok_or("reservation bound overflow")?
            } else {
                serde_json::to_string(&body)
                    .map_err(|e| e.to_string())?
                    .len()
            };
            escaped_total = escaped_total
                .checked_add(escaped)
                .ok_or("reservation bound overflow")?;
        }
        if escaped_total > 32 * 1024 * 1024 {
            return Err("acquisition exceeds reservation receipt bound".into());
        }
    }
    Ok(())
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AdmissionFailure {
    pub version: u32,
    pub wave: WaveSpec,
    pub detail: String,
}
pub fn settlement<'a>(mut attempts: impl Iterator<Item = &'a Attempt>) -> Result<u64> {
    attempts.try_fold(0u64, |end, a| {
        a.timing
            .dispatch_offset_us
            .checked_add(a.timing.settle_us)
            .map(|n| end.max(n))
            .ok_or_else(|| "acquisition settlement overflow".into())
    })
}
pub fn eligibility_phase(workload: &Workload, phase: Phase) -> Phase {
    // Every required conversation cache probe qualifies even in warmup.
    if workload.version == 6 && conversation(workload) {
        Phase::Measured
    } else {
        phase
    }
}

#[derive(Serialize)]
pub struct Record {
    pub identity: AcquisitionIdentity,
    pub cell: Option<String>,
    pub required_steps: Vec<String>,
    pub measured_steps: Vec<String>,
    pub missing_steps: Vec<String>,
    pub ineligible_steps: Vec<String>,
    pub started_offset_us: Option<u64>,
    pub settled_offset_us: Option<u64>,
    pub whole_conversation_wall_us: Option<u64>,
    pub complete_eligible: bool,
}
#[derive(Serialize)]
pub struct Report {
    pub scope: &'static str,
    pub protocol: Protocol,
    pub records: Vec<Record>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resources: Option<serde_json::Value>,
}
#[derive(Serialize)]
pub struct Comparison {
    pub baseline: Report,
    pub candidate: Report,
    pub reference: Option<Report>,
}
pub fn report(run: &crate::evidence::Loaded) -> Option<Report> {
    let protocol = run.plan.workload.acquisition.as_ref()?;
    let mut records: Vec<Record> = Vec::new();
    for spec in &run.plan.waves {
        let identity = spec.acquisition?;
        let cell = (!protocol.conversation()).then(|| spec.cell.clone());
        let position = records
            .iter()
            .position(|r| r.identity == identity && r.cell == cell);
        let index = position.unwrap_or_else(|| {
            records.push(Record {
                identity,
                cell,
                required_steps: Vec::new(),
                measured_steps: Vec::new(),
                missing_steps: Vec::new(),
                ineligible_steps: Vec::new(),
                started_offset_us: None,
                settled_offset_us: None,
                whole_conversation_wall_us: None,
                complete_eligible: run.history.count == 1
                    && !run.history.open
                    && run.history.last_status.as_deref() == Some("completed"),
            });
            records.len() - 1
        });
        let record = &mut records[index];
        let step = spec.case.as_ref()?;
        record.required_steps.push(step.clone());
        if protocol.measured(step) {
            record.measured_steps.push(step.clone());
        }
        match run.waves.get(spec.index as usize).and_then(Option::as_ref) {
            Some(wave) => {
                if !step_eligible(wave) {
                    record.ineligible_steps.push(step.clone());
                    record.complete_eligible = false;
                }
                if let Some(clock) = &wave.acquisition_clock {
                    record
                        .started_offset_us
                        .get_or_insert(clock.started_offset_us);
                    record.settled_offset_us = Some(clock.settled_offset_us);
                } else {
                    record.complete_eligible = false;
                }
            }
            None => {
                record.missing_steps.push(step.clone());
                record.complete_eligible = false;
            }
        }
    }
    for record in &mut records {
        if record.complete_eligible && protocol.conversation() {
            record.whole_conversation_wall_us = record
                .settled_offset_us
                .zip(record.started_offset_us)
                .and_then(|(end, start)| end.checked_sub(start));
        }
    }
    Some(Report {
        scope: "declared-acquisitions-with-all-required-controls; capture-monotonic-microseconds; descriptive-not-an-automatic-policy-verdict",
        protocol: protocol.clone(),
        records,
        resources: run.plan.workload.resources.as_ref().map(|_| {
            serde_json::json!({
                "mode": "review_only", "acquisitions": run.resources,
                "capacity_qualification": "unavailable-through-throughput-policy"
            })
        }),
    })
}
