//! Actual workload6 history/parent identities, metrics2 accounting and explicit
//! allocation-driven cache eviction journal. A miss or preemption is not eviction.
use crate::{evidence, model::*};
use serde::{Deserialize, Serialize};
use std::{collections::{BTreeMap, BTreeSet}, fs::File, os::unix::fs::MetadataExt, path::Path, time::Instant};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub version: u32,
    pub prime: String,
    pub pressure: Vec<String>,
    pub probe: String,
    pub recovery: Vec<String>,
    pub intended_pressure: String,
    pub condition: Condition,
    pub source: Source,
    pub producer_sha256: String,
    pub metrics_labels: BTreeMap<String, String>,
    pub journal_bytes: u32,
    pub max_events: u32,
    pub max_window_us: u64,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Condition { ObserveOnly, EvictionAndRecovery, RetainedHit }
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub enum Source {
    #[serde(rename = "vllm-0.27.0-block-pool-v1")] VllmBlockPool,
    #[serde(rename = "ordinary-lru-fixture-v1")] OrdinaryLruFixture,
}
impl Plan {
    pub fn validate(&self, workload: &Workload) -> Result<()> {
        if self.version != 1 || workload.version != 6 || !crate::acquisition::conversation(workload)
            || self.pressure.is_empty() || self.pressure.len() > 64 || self.recovery.is_empty() || self.recovery.len() > 16
            || self.intended_pressure.trim().is_empty() || self.intended_pressure.len() > 1024
            || !pin(&self.producer_sha256) || !(4096..=16*1024*1024).contains(&self.journal_bytes)
            || !(8..=10000).contains(&self.max_events) || !(1000..=3_600_000_000).contains(&self.max_window_us) {
            return Err("invalid finite history retention declaration".into());
        }
        if self.metrics_labels.is_empty() || self.metrics_labels.len() > 16
            || self.metrics_labels.iter().any(|(k,v)| k.is_empty() || k.len() + v.len() > 4096) {
            return Err("retention requires explicit bounded metrics2 label identity".into());
        }
        let order: Vec<&str> = std::iter::once(self.prime.as_str()).chain(self.pressure.iter().map(String::as_str))
            .chain(std::iter::once(self.probe.as_str())).chain(self.recovery.iter().map(String::as_str)).collect();
        if order.len() != workload.cells.len() || !order.iter().zip(&workload.cells).all(|(id,c)| *id == c.case)
            || order.iter().collect::<BTreeSet<_>>().len() != order.len() {
            return Err("retention phases must exactly cover ordered workload6 steps".into());
        }
        let step = |id: &str| workload.cases.iter().find(|c| c.id == id).and_then(|c| c.step.as_ref()).ok_or("retention phase has no actual sequence step");
        let prime = step(&self.prime)?;
        if prime.parent.is_some() { return Err("retention prime must begin an actual history".into()); }
        for id in &self.pressure {
            let pressure = step(id)?;
            if pressure.history == prime.history || pressure.parent.is_some() { return Err("pressure requires separate declared root histories".into()); }
        }
        for id in std::iter::once(&self.probe).chain(&self.recovery) {
            let follow = step(id)?;
            if follow.history != prime.history || follow.parent.is_none() { return Err("probe/recovery must use an acquired parent in the prime history".into()); }
        }
        Ok(())
    }
}

fn accounting_complete(a: &crate::metrics::v2::Acquisition, plan: &Plan, expected: usize) -> bool {
    if a.snapshots != expected || a.complete != expected { return false; }
    let selected = |name: &str, labels: &BTreeMap<String,String>| a.series.iter().find(|s| s.identity.name == name && s.identity.labels == *labels);
    for name in ["vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
        "vllm:prompt_tokens_total", "vllm:prompt_tokens_cached_total"] {
        if selected(name, &plan.metrics_labels).is_none_or(|s| s.kind != "counter" || s.status != "continuous") { return false; }
    }
    for source in ["local_compute", "local_cache_hit", "external_kv_transfer"] {
        let mut labels = plan.metrics_labels.clone();
        labels.insert("source".into(), source.into());
        if selected("vllm:prompt_tokens_by_source_total", &labels).is_none_or(|s| s.status != "continuous") { return false; }
    }
    let queries = selected("vllm:prefix_cache_queries_total", &plan.metrics_labels).and_then(|s| s.delta);
    let hits = selected("vllm:prefix_cache_hits_total", &plan.metrics_labels).and_then(|s| s.delta);
    if !matches!((queries,hits), (Some(q),Some(h)) if q >= h && h >= 0.0) { return false; }
    ["prompt_tokens_total_equals_sum_of_reported_sources",
        "prompt_tokens_cached_total_equals_local_cache_hit_plus_external_kv_transfer"].into_iter()
        .all(|rule| a.accounting.iter().any(|c| c.labels == plan.metrics_labels && c.rule == rule && c.status == "consistent"))
}
fn pin(s: &str) -> bool { s.len() == 64 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) }
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    version: u32, source: Source, producer_sha256: String,
    provenance: String, before_bytes: usize, before_sha256: String,
    after_bytes: usize, after_sha256: String, file_identity: Option<String>,
    elapsed_us: u64, failure: Option<String>,
}
pub struct JournalWindow {
    file: Option<File>, before: Vec<u8>, identity: Option<String>,
    origin: Instant, plan: Plan, failure: Option<String>,
}
fn identity(file: &File) -> Result<String> {
    let m = file.metadata().map_err(|e| e.to_string())?;
    if !m.is_file() { return Err("journal is not an ordinary regular file".into()); }
    Ok(format!("{}:{}", m.dev(), m.ino()))
}
fn read_file(file: &File, cap: usize) -> Result<Vec<u8>> {
    use std::os::unix::fs::FileExt;
    let mut bytes = Vec::new();
    let mut buffer = [0u8; 8192];
    loop {
        let n = file.read_at(&mut buffer, bytes.len() as u64).map_err(|e| e.to_string())?;
        if n == 0 { break; }
        if bytes.len() + n > cap { return Err("journal byte ceiling exceeded".into()); }
        bytes.extend_from_slice(&buffer[..n]);
    }
    Ok(bytes)
}
impl JournalWindow {
    pub fn begin(path: Option<&Path>, plan: &Plan) -> Result<Self> {
        use std::os::unix::fs::OpenOptionsExt;
        let origin = Instant::now();
        let mut window = Self { file: None, before: Vec::new(), identity: None, origin, plan: plan.clone(), failure: None };
        let Some(path) = path else { window.failure = Some("no native eviction source selected".into()); return Ok(window); };
        let opened = std::fs::OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK).open(path).map_err(|e| e.to_string());
        match opened.and_then(|file| { let id = identity(&file)?; let bytes = read_file(&file, plan.journal_bytes as usize)?; Ok((file,id,bytes)) }) {
            Ok((file,id,bytes)) => { window.file = Some(file); window.identity = Some(id); window.before = bytes; }
            Err(_) => window.failure = Some("selected journal unavailable before acquisition".into()),
        }
        Ok(window)
    }
    pub fn finish(mut self, root: &Path, index: usize) -> Result<()> {
        let mut after = Vec::new();
        if let Some(file) = &self.file {
            match read_file(file, self.plan.journal_bytes as usize) {
                Ok(bytes) if bytes.starts_with(&self.before) && identity(file).ok() == self.identity => after = bytes,
                _ => self.failure = Some("journal changed, truncated or exceeded its byte ceiling".into()),
            }
        }
        let elapsed_us = u64::try_from(self.origin.elapsed().as_micros()).map_err(|_| "journal clock overflow")?;
        if elapsed_us > self.plan.max_window_us { self.failure = Some("journal acquisition window exceeded".into()); }
        evidence::write(&root.join(format!("journal-{index:03}.jsonl")), &after)?;
        evidence::json(&root.join(format!("journal-{index:03}.json")), &Receipt { version: 1, source: self.plan.source,
            producer_sha256: self.plan.producer_sha256, provenance: "native-observed-unauthenticated-source".into(),
            before_bytes: self.before.len(), before_sha256: evidence::digest(&self.before),
            after_bytes: after.len(), after_sha256: evidence::digest(&after), file_identity: self.identity, elapsed_us, failure: self.failure })?;
        evidence::sync(root)
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Event {
    version: u32, source: Source, incarnation: String, sequence: u32,
    offset_us: u64, clock: String, overhead_us: u64, event: Kind,
}
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Kind {
    Start { producer_sha256: String, source_pins: BTreeMap<String,String>, max_events: u32, max_bytes: u32, max_window_us: u64 },
    Installed { hook: String },
    Allocate { requested_blocks: u32, free_blocks: u32, total_blocks: u32 },
    Store { cache_salt: String, keys: Vec<String> },
    Lookup { cache_salt: String, keys: Vec<String>, cached_tokens: u64 },
    Evict { keys: Vec<String> },
    Failure { reason: String },
}
fn events(raw: &[u8], plan: &Plan) -> Result<Vec<Event>> {
    if raw.is_empty() || raw.last() != Some(&b'\n') { return Err("journal missing or final event incomplete".into()); }
    let mut events = Vec::new();
    let mut live = BTreeSet::<String>::new();
    for line in raw[..raw.len()-1].split(|b| *b == b'\n') {
        if line.len() > 65536 || events.len() >= plan.max_events as usize { return Err("journal event ceiling exceeded".into()); }
        let event: Event = serde_json::from_slice(line).map_err(|e| e.to_string())?;
        if event.version != 1 || event.source != plan.source || event.sequence as usize != events.len()
            || event.clock != "producer_monotonic_microseconds" || event.incarnation.len() != 32 {
            return Err("journal source/order/clock mismatch".into());
        }
        if let Some(old) = events.last() {
            let old: &Event = old;
            if old.incarnation != event.incarnation || old.offset_us > event.offset_us || old.overhead_us > event.overhead_us {
                return Err("journal incarnation or monotonic clock changed".into());
            }
        }
        match &event.event {
            Kind::Store { cache_salt, keys } | Kind::Lookup { cache_salt, keys, .. } => {
                if cache_salt.is_empty() || cache_salt.len() > 256 || keys.len() > 1024 || keys.iter().any(|k| !pin(k)) { return Err("invalid cache source identity".into()); }
            }
            Kind::Evict { keys } if keys.is_empty() || keys.len() > 1024 || keys.iter().any(|k| !pin(k)) => return Err("invalid eviction keys".into()),
            Kind::Allocate { requested_blocks, free_blocks, total_blocks } if *total_blocks == 0 || *total_blocks > 65536 || free_blocks > total_blocks || requested_blocks > total_blocks => return Err("invalid observed block allocation pressure".into()),
            Kind::Failure {reason} if reason.is_empty() || reason.len() > 256 => return Err("invalid journal failure evidence".into()),
            _ => (),
        }
        events.push(event);
        match &events.last().expect("event retained").event {
            Kind::Store {keys,..} => live.extend(keys.iter().cloned()),
            Kind::Evict {keys} => {
                for key in keys {
                    if !live.remove(key) { return Err("eviction has no retained live insertion".into()); }
                }
            }
            Kind::Lookup {keys,cached_tokens,..} => {
                if (*cached_tokens == 0) != keys.is_empty() || keys.iter().any(|k| !live.contains(k)) {
                    return Err("reported cache lookup contradicts retained cache contents".into());
                }
            }
            Kind::Start {..} if events.len() != 1 => return Err("repeated journal start".into()),
            Kind::Installed {..} if events.len() != 2 => return Err("repeated hook installation".into()),
            _ => (),
        }
    }
    match events.first().map(|e| &e.event) {
        Some(Kind::Start { producer_sha256, source_pins, max_events, max_bytes, max_window_us })
            if producer_sha256 == &plan.producer_sha256 && *max_events == plan.max_events && *max_bytes == plan.journal_bytes
                && *max_window_us == plan.max_window_us => {
                let expected: BTreeMap<String,String> = if plan.source == Source::VllmBlockPool {
                    [("vllm/v1/core/block_pool.py","51cad2fd425128a0ff433ca4685acfc022e40659153ea5d7180fc586e1eebb6c"),
                     ("vllm/v1/core/kv_cache_manager.py","70f7f608c0963af155540630a5633483e5c19c0c0280dfd621760d5db2a419a6"),
                     ("vllm/v1/request.py","6085b0668f41d56cd81ef483c06456f4ebe88bc1d46c8131d7134182d7893012")]
                        .into_iter().map(|(a,b)|(a.into(),b.into())).collect()
                } else { BTreeMap::new() };
                if *source_pins != expected { return Err("backend source pins mismatch".into()); }
            }
        _ => return Err("journal producer or finite contract mismatch".into()),
    }
    if !matches!(events.get(1).map(|e| &e.event), Some(Kind::Installed {hook}) if hook == "allocation-remove-store-lookup-v1") {
        return Err("cache hooks were not installed before acquisition".into());
    }
    Ok(events)
}

pub fn report(root: &Path, index: usize, plan: &Plan, run: &evidence::Loaded) -> Result<serde_json::Value> {
    let mut unavailable = Vec::<String>::new();
    let mut rows = Vec::new();
    let mut journal_events = Vec::new();
    let mut boundary = 0;
    let receipt_path = root.join(format!("journal-{index:03}.json"));
    if receipt_path.try_exists().map_err(|e| e.to_string())? {
        let receipt: Receipt = serde_json::from_slice(&evidence::read(&receipt_path, 64*1024)?).map_err(|e| e.to_string())?;
        let raw = evidence::read(&root.join(format!("journal-{index:03}.jsonl")), plan.journal_bytes as usize)?;
        if receipt.version != 1 || receipt.source != plan.source || receipt.producer_sha256 != plan.producer_sha256
            || receipt.provenance != "native-observed-unauthenticated-source" || receipt.after_bytes != raw.len()
            || receipt.after_sha256 != evidence::digest(&raw) { return Err("retention journal receipt mismatch".into()); }
        if let Some(failure) = receipt.failure { unavailable.push(failure); }
        else if receipt.before_bytes > raw.len() || receipt.before_sha256 != evidence::digest(&raw[..receipt.before_bytes])
            || receipt.elapsed_us > plan.max_window_us || receipt.file_identity.is_none() { return Err("retention journal continuity mismatch".into()); }
        else {
            if receipt.before_bytes == 0 || raw[receipt.before_bytes-1] != b'\n' {
                unavailable.push("journal acquisition start is not a complete event boundary".into());
            }
            boundary = raw[..receipt.before_bytes].iter().filter(|b| **b == b'\n').count();
            match events(&raw, plan) {
                Ok(events) => journal_events = events,
                Err(error) => unavailable.push(error),
            }
        }
    } else { unavailable.push("eviction journal absent".into()); }
    let telemetry = run.metrics.as_ref().and_then(|m| m.acquisition.as_ref());
    let counters = telemetry.is_some_and(|a| accounting_complete(a, plan, run.plan.waves.len()*2));
    if !counters { unavailable.push("complete metrics2 source continuity/accounting unavailable; preemption and miss are not eviction".into()); }
    let lookup_indices: Vec<usize> = journal_events.iter().enumerate().skip(boundary).filter_map(|(i,e)| matches!(e.event, Kind::Lookup {..}).then_some(i)).collect();
    let mut allocations = Vec::new();
    let mut source_failed = false;
    for (i,event) in journal_events.iter().enumerate() {
        if let Kind::Failure {reason} = &event.event { source_failed = true; unavailable.push(reason.clone()); }
        if i >= boundary && let Kind::Allocate { requested_blocks,free_blocks,total_blocks } = &event.event {
            allocations.push(serde_json::json!({"sequence":event.sequence,"requested_blocks":requested_blocks,"free_blocks":free_blocks,"total_blocks":total_blocks}));
        }
    }
    let published = run.waves.iter().flatten().count();
    if lookup_indices.len() != published || boundary < 2 || published != run.plan.waves.len() {
        unavailable.push("journal request membership or complete acquisition coverage missing".into());
    }
    let mut eviction_recovery = true;
    let mut retained_hit = true;
    for group in crate::serving_resources::groups(&run.plan.waves, true) {
        let mut prime_keys = BTreeSet::<String>::new();
        let mut evicted = BTreeSet::<String>::new();
        let mut probe_hit = None;
        let mut recovery_hit = None;
        let mut members = Vec::new();
        let mut group_qualified = true;
        for wave_index in &group {
            let spec = &run.plan.waves[*wave_index as usize];
            let case = spec.case.as_deref().ok_or("retention wave case missing")?;
            let wave = run.waves[*wave_index as usize].as_ref();
            let attempt = wave.and_then(|w| w.attempts.first());
            let check = attempt.and_then(|a| a.sequence.as_ref());
            if !wave.is_some_and(|w| w.eligible) || !check.is_some_and(crate::sequence::passed) { group_qualified = false; }
            let expected_salt = check.map(|c| format!("{}-{}", crate::acquisition::namespace(run.plan.cache_namespace.as_deref().unwrap_or(""), spec.acquisition.expect("workload6")), c.history));
            let event_index = lookup_indices.get(*wave_index as usize).copied();
            let mut journal_hit = None;
            if let Some(ei) = event_index {
                if let Kind::Lookup {cache_salt, cached_tokens, keys} = &journal_events[ei].event {
                    if expected_salt.as_deref() != Some(cache_salt) { unavailable.push("journal lookup salt differs from actual acquired history".into()); }
                    journal_hit = Some(*cached_tokens > 0);
                    if case == plan.prime { prime_keys.extend(keys.iter().cloned()); }
                    // The previous phase ends immediately before this lookup;
                    // only allocation-driven removals before the probe count.
                    if case == plan.probe { probe_hit = journal_hit; }
                    if plan.recovery.last().map(String::as_str) == Some(case) { recovery_hit = journal_hit; }
                }
                let end = lookup_indices.get(*wave_index as usize + 1).copied().unwrap_or(journal_events.len());
                for event in &journal_events[ei+1..end] {
                    match &event.event {
                        Kind::Store {cache_salt,keys} if case == plan.prime && expected_salt.as_deref() == Some(cache_salt) => prime_keys.extend(keys.iter().cloned()),
                        Kind::Evict {keys} if case == plan.prime || plan.pressure.iter().any(|p| p == case) => evicted.extend(keys.iter().filter(|k| prime_keys.contains(*k)).cloned()),
                        _ => (),
                    }
                }
            }
            let cached = attempt.and_then(|a| a.usage.cached_prompt_tokens);
            members.push(serde_json::json!({"wave":wave_index,"case":case,"history":check.map(|c| &c.history),"parent":check.and_then(|c| c.parent.as_ref()),
                "correct":check.is_some_and(crate::sequence::passed),"reported_cached_tokens":cached,
                "reported_hit":cached.map(|n| n > 0),"journal_hit":journal_hit}));
        }
        let qualified = group_qualified && !source_failed && unavailable.is_empty();
        eviction_recovery &= qualified && !evicted.is_empty() && recovery_hit == Some(true);
        retained_hit &= qualified && probe_hit == Some(true) && evicted.is_empty();
        rows.push(serde_json::json!({"required_waves":group,"steps":members,"prime_source_keys":prime_keys,
            "observed_evicted_target_keys": if qualified {Some(&evicted)} else {None},
            "source_eviction_candidates":evicted,"probe_hit":probe_hit,"recovery_hit":recovery_hit,
            "complete_qualified":qualified}));
    }
    let satisfied = match plan.condition { Condition::ObserveOnly => false, Condition::EvictionAndRecovery => eviction_recovery, Condition::RetainedHit => retained_hit };
    Ok(serde_json::json!({"version":1,"kind":"retention-observation-v1","source":plan.source,
        "provenance":"native-observed-unauthenticated-source","fixture_not_real_backend":plan.source == Source::OrdinaryLruFixture,
        "intended_pressure":plan.intended_pressure,"observed_allocation_pressure":allocations,
        "metrics2_continuity_accounting":telemetry,"counter_coverage_complete":counters,
        "acquisitions":rows,"condition":plan.condition,"condition_satisfied":satisfied,"unavailable":unavailable,
        "scope":"source-pinned local pool only; target block removal is observed eviction, not proof of whole-history loss or miss causation; reported hit/miss and correctness remain separate; no reset/flush/restart",
        "live_qualified":false}))
}
