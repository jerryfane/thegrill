//! Resource observation attached to the existing workload6 acquisition clock.
//! One raw sidecar per acquisition; step reports never duplicate raw samples.
use crate::{acquisition::StepClock, evidence, model::*, resources};
use serde::{Deserialize, Serialize};
use std::{
    path::Path,
    time::{Duration, Instant},
};
use tokio::{sync::watch, task::JoinHandle};

pub const CAP: usize = 64 * 1024 * 1024;
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub version: u32,
    pub mode: Mode,
    pub observer: resources::ResourcesConfig,
    pub summaries: Vec<resources::Gate>,
    pub retained_bytes: u64,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    ReviewOnly,
}
impl Config {
    pub fn validate(&self, workload: &Workload) -> Result<()> {
        self.observer.validate()?;
        if self.version != 1
            || workload.version != 6
            || self.summaries.is_empty()
            || self.summaries.len() > 32
            || self.retained_bytes == 0
            || self.retained_bytes > 1024 * 1024 * 1024
        {
            return Err(
                "serving resources require workload6 and bounded review-only configuration".into(),
            );
        }
        let mut keys = std::collections::BTreeSet::new();
        for gate in &self.summaries {
            if !self.observer.sources.iter().any(|s| s.id == gate.source)
                || gate.max_regression_bps != 0
                || gate.max_reference_spread_bps != 0
                || !keys.insert(serde_json::to_string(gate).map_err(|e| e.to_string())?)
            {
                return Err("resource summary requires a unique declared source and zero comparison thresholds (review-only)".into());
            }
        }
        let groups = groups(
            &workload.waves(),
            workload
                .acquisition
                .as_ref()
                .is_some_and(crate::acquisition::Protocol::conversation),
        );
        // JSON encodes raw octets as decimal arrays; reserve the full per-acquisition
        // artifact ceiling rather than treating raw bytes as serialized bytes.
        if (groups.len() as u64)
            .checked_mul(CAP as u64)
            .is_none_or(|n| n > self.retained_bytes)
        {
            return Err(
                "declared resource retention does not cover every acquisition artifact ceiling"
                    .into(),
            );
        }
        Ok(())
    }
}

pub fn groups(waves: &[WaveSpec], conversation: bool) -> Vec<Vec<u32>> {
    let mut groups: Vec<Vec<u32>> = Vec::new();
    for (i, wave) in waves.iter().enumerate() {
        if i == 0 || !conversation || wave.acquisition != waves[i - 1].acquisition {
            groups.push(Vec::new());
        }
        groups
            .last_mut()
            .expect("group initialized")
            .push(wave.index);
    }
    groups
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Member {
    pub wave: u32,
    pub clock: StepClock,
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Capture {
    pub version: u32,
    pub plan_sha256: String,
    pub required_waves: Vec<u32>,
    pub members: Vec<Member>,
    pub complete_membership: bool,
    pub cancelled: bool,
    pub failure: Option<String>,
    pub observation: Option<resources::Observation>,
}
#[derive(Serialize)]
pub struct Report {
    pub required_waves: Vec<u32>,
    pub observed_waves: Vec<u32>,
    pub complete_membership: bool,
    pub cancelled: bool,
    pub failure: Option<String>,
    pub measured: Option<resources::MeasuredInterval>,
    pub summaries: Vec<resources::Summary>,
    pub raw_sidecar: String,
    pub sha256: Option<String>,
    pub scope: &'static str,
}
struct Active {
    required: Vec<u32>,
    members: Vec<Member>,
    stop: watch::Sender<bool>,
    task: JoinHandle<(resources::Observer, Option<String>)>,
}
pub struct Session<'a> {
    root: &'a Path,
    config: Option<&'a Config>,
    plan_hash: &'a str,
    origin: Instant,
    groups: Vec<Vec<u32>>,
    active: Option<Active>,
}
impl<'a> Session<'a> {
    pub fn new(root: &'a Path, plan: &'a Plan, plan_hash: &'a str, origin: Instant) -> Self {
        Self {
            root,
            config: plan.workload.resources.as_ref(),
            plan_hash,
            origin,
            groups: groups(
                &plan.waves,
                crate::acquisition::conversation(&plan.workload),
            ),
            active: None,
        }
    }
    pub async fn begin(&mut self, wave: u32) -> Result<()> {
        let Some(config) = self.config else {
            return Ok(());
        };
        if self
            .active
            .as_ref()
            .is_some_and(|a| a.required.contains(&wave))
        {
            return Ok(());
        }
        self.finish(false).await?;
        let required = self
            .groups
            .iter()
            .find(|g| g.first() == Some(&wave))
            .ok_or("resource acquisition boundary mismatch")?
            .clone();
        let clock = resources::host_clock(self.plan_hash.into())?;
        let mut observer = resources::Observer::start(config.observer.clone(), self.origin, clock)?;
        let initial = observer.sample();
        let cadence = Duration::from_micros(config.observer.cadence_us);
        let (stop, mut cancellation) = watch::channel(false);
        let task = tokio::spawn(async move {
            let mut failure = initial.as_ref().err().cloned();
            let mut sampling = initial.unwrap_or(false);
            let mut timer =
                tokio::time::interval_at(tokio::time::Instant::now() + cadence, cadence);
            timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                tokio::select! {
                    biased;
                    _ = cancellation.changed() => break,
                    _ = timer.tick(), if sampling => {
                        match observer.sample() {
                            Ok(keep) => sampling = keep,
                            Err(error) => { failure = Some(error); sampling = false; }
                        }
                    }
                }
            }
            if sampling && let Err(error) = observer.sample() {
                failure = Some(error);
            }
            (observer, failure)
        });
        self.active = Some(Active {
            required,
            members: Vec::new(),
            stop,
            task,
        });
        Ok(())
    }
    pub fn member(&mut self, wave: &Wave) {
        if let (Some(active), Some(clock)) = (&mut self.active, &wave.acquisition_clock) {
            active.members.push(Member {
                wave: wave.spec.index,
                clock: clock.clone(),
            });
        }
    }
    pub async fn finish(&mut self, cancelled: bool) -> Result<()> {
        let Some(active) = self.active.take() else {
            return Ok(());
        };
        let _ = active.stop.send(true);
        let (observer, failure) = active
            .task
            .await
            .map_err(|_| "resource observer task failed; no resource qualification")?;
        let complete = active
            .required
            .iter()
            .copied()
            .eq(active.members.iter().map(|m| m.wave));
        let interval = active
            .members
            .first()
            .zip(active.members.last())
            .map(|(first, last)| resources::MeasuredInterval {
                clock: self.plan_hash.into(),
                started_us: first.clock.started_offset_us,
                settled_us: last.clock.settled_offset_us,
            });
        // No dispatched step means no measured interval. Retain setup samples with
        // an explicit empty interval and incomplete membership, never call it work.
        let empty = observer.last_observed_us().unwrap_or(0);
        let observation = observer.finish(
            interval.unwrap_or(resources::MeasuredInterval {
                clock: self.plan_hash.into(),
                started_us: empty,
                settled_us: empty,
            }),
            cancelled || !complete || failure.is_some(),
        )?;
        let capture = Capture {
            version: 1,
            plan_sha256: self.plan_hash.into(),
            required_waves: active.required,
            members: active.members,
            complete_membership: complete,
            cancelled,
            failure,
            observation: Some(observation),
        };
        let bytes = serde_json::to_vec(&capture).map_err(|e| e.to_string())?;
        if bytes.len() > CAP {
            return Err("resource acquisition artifact exceeded admitted ceiling".into());
        }
        evidence::write(&self.root.join(filename(capture.required_waves[0])), &bytes)?;
        evidence::sync(self.root)
    }
}
fn filename(first: u32) -> String {
    format!("resources-{first:06}.json")
}
pub fn replay(
    root: &Path,
    plan: &Plan,
    plan_hash: &str,
    waves: &[Option<Wave>],
) -> Result<Vec<Report>> {
    let Some(config) = &plan.workload.resources else {
        return Ok(Vec::new());
    };
    let mut reports = Vec::new();
    for required in groups(
        &plan.waves,
        crate::acquisition::conversation(&plan.workload),
    ) {
        let name = filename(required[0]);
        let path = root.join(&name);
        let exists = match std::fs::symlink_metadata(&path) {
            Ok(_) => true,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => false,
            Err(e) => return Err(e.to_string()),
        };
        let mut report = Report {
            required_waves: required.clone(),
            observed_waves: Vec::new(),
            complete_membership: false,
            cancelled: false,
            failure: Some("resource acquisition not retained".into()),
            measured: None,
            summaries: Vec::new(),
            raw_sidecar: name,
            sha256: None,
            scope: "review-only complete-acquisition resources; sampled maxima are not peaks; unsupported CPU boundaries remain unavailable; source ownership is not model attribution",
        };
        if exists {
            let bytes = evidence::read(&path, CAP)?;
            let capture: Capture = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
            if capture.version != 1
                || capture.plan_sha256 != plan_hash
                || capture.required_waves != required
                || capture.members.len() > required.len()
                || !capture
                    .members
                    .iter()
                    .zip(&required)
                    .all(|(m, r)| m.wave == *r)
                || capture.complete_membership != (capture.members.len() == required.len())
            {
                return Err("resource acquisition membership mismatch".into());
            }
            for member in &capture.members {
                let wave = waves
                    .get(member.wave as usize)
                    .and_then(Option::as_ref)
                    .ok_or("resource member has no published wave")?;
                if wave.acquisition_clock.as_ref() != Some(&member.clock) {
                    return Err("resource step clock mismatch".into());
                }
            }
            let observation = capture
                .observation
                .as_ref()
                .ok_or("resource sidecar has no drained observation")?;
            resources::validate_observation(observation)?;
            if observation.config != config.observer
                || observation.clock.id != plan_hash
                || observation.provenance != resources::Provenance::NativeObserved
                || observation.binary_sha256 != plan.collector_sha256
            {
                return Err("resource source, provenance or capture clock mismatch".into());
            }
            if let Some((first, last)) = capture.members.first().zip(capture.members.last()) {
                if observation.measured
                    != (resources::MeasuredInterval {
                        clock: plan_hash.into(),
                        started_us: first.clock.started_offset_us,
                        settled_us: last.clock.settled_offset_us,
                    })
                {
                    return Err("resource interval is not complete acquisition span".into());
                }
                report.measured = Some(observation.measured.clone());
            } else if observation.measured.started_us != observation.measured.settled_us {
                return Err("undispatched acquisition has resource exposure".into());
            }
            if (!capture.complete_membership || capture.cancelled || capture.failure.is_some())
                && !observation
                    .failures
                    .contains(&resources::Failure::Cancelled)
            {
                return Err("partial resource capture concealed cancellation".into());
            }
            report.observed_waves = capture.members.iter().map(|m| m.wave).collect();
            report.complete_membership = capture.complete_membership;
            report.cancelled = capture.cancelled;
            report.failure = capture.failure;
            report.summaries = config
                .summaries
                .iter()
                .map(|g| resources::summarize(observation, g))
                .collect();
            report.sha256 = Some(evidence::digest(&bytes));
        }
        reports.push(report);
    }
    Ok(reports)
}
