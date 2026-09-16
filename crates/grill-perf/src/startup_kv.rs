//! New source-copy observation identity. Never upgrades historical captures.
use super::{ProcessIdentity, Stage, Study, decode, hash, load};
use crate::{evidence, model::Result, policy::Outcome};
use clap::Args;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

const SOURCE: &str = "vllm-487ecf187-lmcache-3e11b8ed-kv-copy-v1";
const SCOPE: &str = "lmcache-driven-full-attention-l1-engine-restart";
const CAP: usize = 16 * 1024 * 1024;

#[derive(Args)]
pub struct VerifyArgs {
    /// Prospective source, model, rank/group and full token-range selection.
    pub expectation: PathBuf,
    #[arg(long, required = true)]
    pub cache_events: Vec<PathBuf>,
    #[arg(long)]
    pub store_frontend: PathBuf,
    #[arg(long)]
    pub reload_frontend: PathBuf,
    /// One closed source journal per selected pre-restart worker process.
    #[arg(long)]
    pub store_worker: Vec<PathBuf>,
    /// Corresponding post-restart worker journals; rank binding is observed.
    #[arg(long)]
    pub reload_worker: Vec<PathBuf>,
    #[arg(long)]
    pub producer_source: PathBuf,
    /// Existing retained reload capture, including its linked store capture.
    #[arg(long)]
    pub reload_capture: Option<PathBuf>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Expectation {
    version: u32,
    source: String,
    scope: String,
    producer_sha256: String,
    model: String,
    salt: String,
    world_size: usize,
    groups: usize,
    chunk_size: u64,
    start: u64,
    end: u64,
    store_request: String,
    reload_request: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
struct Key {
    hash: String,
    model: String,
    kv_rank: usize,
    group: usize,
    salt: String,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Extent {
    key: Key,
    start: u64,
    end: u64,
    bytes: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Finalized {
    key: Key,
    success: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Row {
    version: u32,
    sequence: usize,
    incarnation: String,
    offset_us: u64,
    event: Event,
}
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Event {
    Start {
        source: String,
        scope: String,
        role: String,
        process: ProcessIdentity,
        producer_sha256: String,
        max_events: usize,
        max_bytes: usize,
        max_window_us: u64,
    },
    Installed {
        pins: BTreeMap<String, String>,
        scope: String,
    },
    Binding {
        request: String,
        engine_request: String,
        parent: Option<String>,
        index: usize,
    },
    Operation {
        operation: usize,
        direction: String,
        session: String,
        engine: u64,
        rank: usize,
        world_size: usize,
        model: String,
        salt: String,
        start: u64,
        end: u64,
    },
    WorkerSubmission {
        direction: String,
        session: String,
        engine: u64,
        rank: usize,
        world_size: usize,
        model: String,
        salt: String,
        start: u64,
        end: u64,
        mla_only: bool,
        n_servers: usize,
        vllm_world_size: usize,
        vllm_worker_id: usize,
    },
    Selected {
        operation: usize,
        extents: Vec<Extent>,
        supported: bool,
    },
    Copy {
        operation: usize,
        extents: Vec<Extent>,
        outcome: String,
        mode: String,
    },
    CallbackSubmitted {
        operation: usize,
        callback: String,
        keys: Vec<Key>,
    },
    Finalized {
        operation: usize,
        results: Vec<Finalized>,
    },
    DeviceComplete {
        operation: usize,
        callback: String,
        keys: Vec<Key>,
    },
    Result {
        operation: usize,
        success: bool,
        supported: bool,
    },
    Unsupported {
        operation: usize,
        reason: String,
    },
    Failure {
        reason: String,
    },
    Fence {
        admission_closed: bool,
        active: usize,
        pending: usize,
        operations: usize,
        complete: bool,
        hooks_intact: bool,
    },
}

struct Journal {
    rows: Vec<Row>,
    process: ProcessIdentity,
    digest: String,
    bytes: usize,
}
fn source_pins(role: &str) -> BTreeMap<String, String> {
    let pairs: &[(&str, &str)] = if role == "frontend" {
        &[(
            "vllm/v1/engine/async_llm.py",
            "bceed0b3f5f0c834fef79525f2462a092f082390f0070526280abc95945837dd",
        )]
    } else if role == "worker" {
        &[(
            "lmcache/integration/vllm/vllm_multi_process_adapter.py",
            "6fe6598e77e7872829fd0f4b0dc0e23ed2c79a3294c61a69045f7e89bae31ecb",
        )]
    } else {
        &[
            (
                "lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py",
                "7e42b2c1caeb88fd6a2416e0e193451bf83a8413daa3c8d00f31ff9b2a580da7",
            ),
            (
                "lmcache/v1/distributed/storage_manager.py",
                "a2cdf1277fbc4bf49aaf687d057d44e617ef78457e9a4e83e8ea866ab7825384",
            ),
            (
                "lmcache/v1/multiprocess/native_completion.py",
                "58c69916f74b72afb5a755cadf4a87164f34597858a6489509fcbe11e9b866d0",
            ),
            (
                "lmcache/v1/gpu_connector/gpu_ops.py",
                "de0bdd322b8eb8a4078c9a83d8ef8d6bf5e62e67092ab3a07a07f04c69d9fb90",
            ),
            (
                "lmcache/v1/distributed/api.py",
                "cfa411afbebb06c29d21a45c40a7f964748d749d3d1c928215dfaefcf0ef4609",
            ),
            (
                "lmcache/v1/multiprocess/custom_types.py",
                "6e9e19fea6682cdf0c8837c1ac00d77c06413e2a4b740b34c93698d958e05dd7",
            ),
        ]
    };
    pairs
        .iter()
        .map(|(k, v)| (String::from(*k), String::from(*v)))
        .collect()
}
fn journal(path: &Path, expectation: &Expectation, role: &str, cap: usize) -> Result<Journal> {
    let raw = evidence::read(path, cap)?;
    if raw.last() != Some(&b'\n') {
        return Err("KV journal lacks terminal delivery newline".into());
    }
    let mut rows: Vec<Row> = Vec::new();
    for line in raw.split_inclusive(|b| *b == b'\n') {
        if rows.len() >= 10000 || line.len() > 65536 {
            return Err("KV journal bound exceeded".into());
        }
        let row: Row = decode(line)?;
        if row.version != 1
            || row.sequence != rows.len()
            || row.incarnation.len() != 32
            || !row.incarnation.bytes().all(|b| b.is_ascii_hexdigit())
            || rows.last().is_some_and(|previous| {
                previous.incarnation != row.incarnation
                    || previous.offset_us > row.offset_us
                    || matches!(previous.event, Event::Fence { .. })
            })
        {
            return Err("KV journal sequence, clock, incarnation or final-suffix mismatch".into());
        }
        rows.push(row);
    }
    let Some(Row {
        event:
            Event::Start {
                source,
                scope,
                role: observed_role,
                process,
                producer_sha256,
                max_events,
                max_bytes,
                max_window_us,
            },
        ..
    }) = rows.first()
    else {
        return Err("KV journal missing source start".into());
    };
    if source != SOURCE
        || scope != SCOPE
        || observed_role != role
        || producer_sha256 != &expectation.producer_sha256
        || process.pid == 0
        || process.start_ticks == 0
        || process.boot_id.is_empty()
        || !(8..=10000).contains(max_events)
        || !(4096..=CAP).contains(max_bytes)
        || !(1000..=3600000000).contains(max_window_us)
        || rows.len() > *max_events
        || raw.len() > *max_bytes
        || rows.last().is_none_or(|r| r.offset_us > *max_window_us)
    {
        return Err("KV journal source, process or prospective bounds mismatch".into());
    }
    if !matches!(rows.get(1).map(|r| &r.event), Some(Event::Installed {pins, scope})
        if *pins == source_pins(role) && scope == SCOPE)
    {
        return Err("KV installed whole source set unavailable".into());
    }
    let operations = rows
        .iter()
        .filter(|r| matches!(r.event, Event::Operation { .. }))
        .count();
    if !matches!(rows.last().map(|r| &r.event), Some(Event::Fence {
        admission_closed: true, active: 0, pending: 0, operations: n, complete: true, hooks_intact: true
    }) if *n == operations)
    {
        return Err("KV source pending/loss/terminal drain fence incomplete".into());
    }
    for (index, row) in rows.iter().enumerate() {
        match &row.event {
            Event::Failure { reason } => return Err(format!("KV source failure: {reason}")),
            Event::Unsupported { operation, reason } => {
                return Err(format!("KV operation {operation} unsupported: {reason}"));
            }
            Event::Start { .. } if index != 0 => return Err("duplicate KV source start".into()),
            Event::Installed { .. } if index != 1 => return Err("duplicate KV installation".into()),
            Event::Binding { .. } if role != "frontend" => {
                return Err("cache journal contains frontend binding".into());
            }
            Event::WorkerSubmission { .. } if role != "worker" => {
                return Err("non-worker journal contains worker submission".into());
            }
            Event::Start { .. }
            | Event::Installed { .. }
            | Event::Fence { .. }
            | Event::Binding { .. }
            | Event::WorkerSubmission { .. } => (),
            _ if role != "cache" => return Err("non-cache journal contains cache operation".into()),
            _ => (),
        }
    }
    let process = process.clone();
    Ok(Journal {
        rows,
        process,
        digest: evidence::digest(&raw),
        bytes: raw.len(),
    })
}
fn binding(journal: &Journal, request: &str) -> Result<String> {
    let bindings: Vec<_> = journal
        .rows
        .iter()
        .filter_map(|r| match &r.event {
            Event::Binding {
                request: id,
                engine_request,
                parent,
                index,
            } => Some((id, engine_request, parent, index)),
            _ => None,
        })
        .collect();
    let [(id, engine, None, 0)] = bindings.as_slice() else {
        return Err("KV scope requires exactly one observed non-child engine admission".into());
    };
    if id.as_str() != request || engine.is_empty() || engine.len() > 256 {
        return Err("KV Grill-to-engine request binding mismatch".into());
    }
    Ok((*engine).clone())
}
fn worker_bindings(
    journals: &[Journal],
    session: &str,
    direction: &str,
    expected: &Expectation,
) -> Result<BTreeMap<usize, (ProcessIdentity, String, u64)>> {
    let mut result = BTreeMap::new();
    for journal in journals {
        let mut submissions = journal.rows.iter().filter_map(|row| match &row.event {
            event @ Event::WorkerSubmission { .. } => Some(event),
            _ => None,
        });
        let Some(Event::WorkerSubmission {
            direction: dir,
            session: id,
            engine,
            rank,
            world_size: size,
            model,
            salt,
            start,
            end,
            mla_only,
            n_servers,
            vllm_world_size,
            vllm_worker_id,
        }) = submissions.next()
        else {
            return Err("KV worker submission missing or repeated".into());
        };
        if submissions.next().is_some() {
            return Err("KV worker submission missing or repeated".into());
        }
        if *mla_only || *n_servers != 1 || vllm_world_size != size || vllm_worker_id != rank {
            return Err(
                "KV worker topology unsupported; requires non-MLA single-server ordinal domain"
                    .into(),
            );
        }
        if dir != direction
            || id != session
            || *engine == 0
            || *rank >= expected.world_size
            || *size != expected.world_size
            || model != &expected.model
            || salt != &expected.salt
            || *start != expected.start
            || *end != expected.end
            || result
                .insert(
                    *rank,
                    (
                        journal.process.clone(),
                        journal.rows[0].incarnation.clone(),
                        *engine,
                    ),
                )
                .is_some()
        {
            return Err("KV actual worker rank/session/instance binding mismatch".into());
        }
    }
    if result.len() != expected.world_size {
        return Err("KV worker lifecycle required rank missing".into());
    }
    Ok(result)
}

#[derive(Default)]
struct Operation {
    direction: String,
    session: String,
    engine: u64,
    rank: usize,
    producer: usize,
    selected: Option<Vec<Extent>>,
    copied: Vec<Extent>,
    submitted: Option<Vec<Key>>,
    finalized: Option<Vec<Key>>,
    completed: bool,
    result: Option<bool>,
}
fn keys(extents: &[Extent]) -> BTreeSet<Key> {
    extents.iter().map(|x| x.key.clone()).collect()
}
fn extents_valid(extents: &[Extent], rank: usize, expected: &Expectation) -> bool {
    let n = (expected.end - expected.start) / expected.chunk_size;
    if extents.len() != n as usize * expected.groups || keys(extents).len() != extents.len() {
        return false;
    }
    let mut positions = BTreeSet::new();
    // Pinned ipc_key_to_object_keys uses ComputeKVRank(world, rank, world, rank).
    // Worker ordinal is independent of this packed topology field.
    let kv_rank = (expected.world_size << 24) | (rank << 16) | (expected.world_size << 8) | rank;
    extents.iter().all(|x| {
        x.bytes > 0
            && x.key.kv_rank == kv_rank
            && x.key.group < expected.groups
            && x.key.model == expected.model
            && x.key.salt == expected.salt
            && !x.key.hash.is_empty()
            && x.key.hash.len() <= 2048
            && x.key.hash.len().is_multiple_of(2)
            && x.key
                .hash
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            && x.start >= expected.start
            && x.end <= expected.end
            && x.end.checked_sub(x.start) == Some(expected.chunk_size)
            && (x.start - expected.start).is_multiple_of(expected.chunk_size)
            && positions.insert((x.key.group, x.start))
    })
}
fn transfers(
    caches: &[Journal],
    before: &str,
    after: &str,
    expected: &Expectation,
    workers: &BTreeMap<usize, [u64; 2]>,
) -> Result<()> {
    let mut operations: BTreeMap<(usize, usize), Operation> = BTreeMap::new();
    for (producer, cache) in caches.iter().enumerate() {
        let mut local_operations = 0;
        for row in &cache.rows {
            if let Event::Operation {
                operation,
                direction,
                session,
                engine,
                rank,
                world_size,
                model,
                salt,
                start,
                end,
            } = &row.event
            {
                if *operation != local_operations + 1
                    || operations.len() >= expected.world_size * 2
                    || *rank >= expected.world_size
                    || *world_size != expected.world_size
                    || model != &expected.model
                    || salt != &expected.salt
                    || *start != expected.start
                    || *end != expected.end
                    || !matches!(direction.as_str(), "store" | "retrieve")
                    || *engine == 0
                    || (direction == "store" && session != before)
                    || (direction == "retrieve" && session != after)
                {
                    return Err(
                        "KV operation request, rank, range, mode or identity mismatch".into(),
                    );
                }
                if workers
                    .get(rank)
                    .map(|ids| ids[usize::from(direction == "retrieve")])
                    != Some(*engine)
                {
                    return Err(
                        "KV cache-side engine instance does not match actual worker submission"
                            .into(),
                    );
                }
                if direction == "retrieve"
                    && !operations.values().any(|op| {
                        op.rank == *rank
                            && op.producer == producer
                            && op.direction == "store"
                            && op.completed
                            && op.result == Some(true)
                    })
                {
                    return Err(
                        "KV reload precedes matching finalized store in the same cache process"
                            .into(),
                    );
                }
                local_operations += 1;
                operations.insert(
                    (producer, *operation),
                    Operation {
                        direction: direction.clone(),
                        session: session.clone(),
                        engine: *engine,
                        rank: *rank,
                        producer,
                        ..Operation::default()
                    },
                );
                continue;
            }
            let id = match &row.event {
                Event::Selected { operation, .. }
                | Event::Copy { operation, .. }
                | Event::CallbackSubmitted { operation, .. }
                | Event::Finalized { operation, .. }
                | Event::DeviceComplete { operation, .. }
                | Event::Result { operation, .. } => *operation,
                _ => continue,
            };
            let op = operations
                .get_mut(&(producer, id))
                .ok_or("KV observation without an operation start")?;
            match &row.event {
                Event::Selected {
                    extents, supported, ..
                } => {
                    if !supported
                        || op.selected.is_some()
                        || !extents_valid(extents, op.rank, expected)
                    {
                        return Err("KV selected expected nonempty extents unavailable".into());
                    }
                    op.selected = Some(extents.clone());
                }
                Event::Copy {
                    extents,
                    outcome,
                    mode,
                    ..
                } => {
                    if op.selected.is_none()
                        || op.submitted.is_some()
                        || op.result.is_some()
                        || outcome != "enqueued"
                        || mode != "native-object-group"
                        || extents.is_empty()
                    {
                        return Err("KV copy outcome or source order mismatch".into());
                    }
                    if op.copied.len() + extents.len() > 65536 {
                        return Err("KV copied extent bound".into());
                    }
                    op.copied.extend_from_slice(extents);
                }
                Event::CallbackSubmitted {
                    callback,
                    keys: supplied,
                    ..
                } => {
                    if op.submitted.is_some()
                        || op.result.is_some()
                        || supplied.is_empty()
                        || callback
                            != if op.direction == "store" {
                                "finish_write"
                            } else {
                                "finish_read_prefetched"
                            }
                        || op.selected.as_ref() != Some(&op.copied)
                        || supplied.iter().cloned().collect::<BTreeSet<_>>() != keys(&op.copied)
                        || supplied.len() != op.copied.len()
                    {
                        return Err(
                            "KV callback does not bind the exact nonempty successful copy".into(),
                        );
                    }
                    op.submitted = Some(supplied.clone());
                }
                Event::Finalized { results, .. } => {
                    let good: Vec<_> = results
                        .iter()
                        .filter(|r| r.success)
                        .map(|r| r.key.clone())
                        .collect();
                    if op.direction != "store"
                        || op.completed
                        || op.finalized.is_some()
                        || results.len() != good.len()
                        || op.submitted.as_ref().is_none_or(|k| {
                            k.len() != good.len()
                                || k.iter().collect::<BTreeSet<_>>() != good.iter().collect()
                        })
                    {
                        return Err("KV successful per-key finalization unavailable".into());
                    }
                    op.finalized = Some(good);
                }
                Event::DeviceComplete {
                    callback,
                    keys: supplied,
                    ..
                } => {
                    if op.completed
                        || op.submitted.as_ref() != Some(supplied)
                        || callback
                            != if op.direction == "store" {
                                "finish_write"
                            } else {
                                "finish_read_prefetched"
                            }
                        || (op.direction == "store" && op.finalized.is_none())
                    {
                        return Err("KV device completion/finalization join mismatch".into());
                    }
                    op.completed = true;
                }
                // Record the successful transition even when this guard falls through.
                Event::Result {
                    success, supported, ..
                } if op.result.replace(*success && *supported).is_some()
                    || !success
                    || !supported =>
                {
                    return Err("KV failed, partial, skipped or duplicate operation result".into());
                }
                _ => (),
            }
        }
    }
    if operations.len() != expected.world_size * 2 {
        return Err("KV required rank operation missing".into());
    }
    for rank in 0..expected.world_size {
        let store: Vec<_> = operations
            .values()
            .filter(|op| op.rank == rank && op.direction == "store")
            .collect();
        let reload: Vec<_> = operations
            .values()
            .filter(|op| op.rank == rank && op.direction == "retrieve")
            .collect();
        let ([store], [reload]) = (store.as_slice(), reload.as_slice()) else {
            return Err("KV duplicate/missing rank store/reload".into());
        };
        if !store.completed
            || !reload.completed
            || store.result != Some(true)
            || reload.result != Some(true)
            || store.producer != reload.producer
            || store.engine == reload.engine
            || store.session == reload.session
            || store.copied != reload.copied
        {
            return Err(
                "KV stored keys/extents, changed engine or completed reload mismatch".into(),
            );
        }
    }
    Ok(())
}

#[derive(Serialize)]
pub struct Report {
    version: u32,
    observation_identity: &'static str,
    scope: &'static str,
    claim: &'static str,
    expectation_sha256: String,
    source_journal_sha256: Vec<String>,
    cache_processes: Vec<ProcessIdentity>,
    pub outcome: Outcome,
    source_copy_supported: bool,
    admission_warmup_eligible: bool,
    historical_store: Option<super::Inspection>,
    historical_reload: Option<super::Inspection>,
    reasons: Vec<String>,
}

pub fn verify(args: &VerifyArgs) -> Result<Report> {
    let bytes = evidence::read(&args.expectation, 65536)?;
    let expected: Expectation = decode(&bytes)?;
    if expected.version != 1
        || expected.source != SOURCE
        || expected.scope != SCOPE
        || !hash(&expected.producer_sha256)
        || expected.model.is_empty()
        || expected.model.len() > 256
        || expected.salt.len() > 128
        || !(1..=64).contains(&expected.world_size)
        || !(1..=64).contains(&expected.groups)
        || expected.chunk_size == 0
        || expected.end <= expected.start
        || !(expected.end - expected.start).is_multiple_of(expected.chunk_size)
        || (expected.end - expected.start) / expected.chunk_size > 1024
        || expected.store_request.is_empty()
        || expected.reload_request.is_empty()
        || expected.store_request == expected.reload_request
    {
        return Err("unsupported prospective KV source/rank/group/range selection".into());
    }
    if evidence::digest(&evidence::read(&args.producer_source, 4 * 1024 * 1024)?)
        != expected.producer_sha256
    {
        return Err("KV producer source digest mismatch".into());
    }
    if args.cache_events.is_empty() || args.cache_events.len() > expected.world_size {
        return Err("KV requires a bounded explicit cache producer file set".into());
    }
    let before = journal(&args.store_frontend, &expected, "frontend", CAP)?;
    let after = journal(
        &args.reload_frontend,
        &expected,
        "frontend",
        CAP - before.bytes,
    )?;
    let mut remaining = CAP - before.bytes - after.bytes;
    if args.store_worker.len() > expected.world_size
        || args.reload_worker.len() > expected.world_size
    {
        return Err("KV worker file set exceeds prospective ranks".into());
    }
    let mut store_workers = Vec::new();
    let mut reload_workers = Vec::new();
    for (paths, journals) in [
        (&args.store_worker, &mut store_workers),
        (&args.reload_worker, &mut reload_workers),
    ] {
        for path in paths {
            let worker = journal(path, &expected, "worker", remaining)?;
            remaining -= worker.bytes;
            journals.push(worker);
        }
    }
    let mut caches: Vec<Journal> = Vec::new();
    for path in &args.cache_events {
        let cache = journal(path, &expected, "cache", remaining)?;
        if caches.iter().any(|other| {
            other.process == cache.process || other.rows[0].incarnation == cache.rows[0].incarnation
        }) {
            return Err("duplicate cache process/incarnation in KV producer set".into());
        }
        remaining -= cache.bytes;
        caches.push(cache);
    }
    let mut digests: Vec<_> = caches.iter().map(|j| j.digest.clone()).collect();
    digests.extend([before.digest.clone(), after.digest.clone()]);
    digests.extend(
        store_workers
            .iter()
            .chain(&reload_workers)
            .map(|j| j.digest.clone()),
    );
    let mut report = Report {
        version: 1,
        observation_identity: SOURCE,
        scope: SCOPE,
        claim: "source-copy-observation-not-native-qualification",
        expectation_sha256: evidence::digest(&bytes),
        source_journal_sha256: digests,
        cache_processes: caches.iter().map(|j| j.process.clone()).collect(),
        outcome: Outcome::Inconclusive,
        source_copy_supported: false,
        admission_warmup_eligible: false,
        historical_store: None,
        historical_reload: None,
        reasons: Vec::new(),
    };
    let source_result = (|| {
        if before.process == after.process
            || before.rows[0].incarnation == after.rows[0].incarnation
        {
            return Err("KV frontend engine incarnation did not change".into());
        }
        let store_id = binding(&before, &expected.store_request)?;
        let reload_id = binding(&after, &expected.reload_request)?;
        let old_workers = worker_bindings(&store_workers, &store_id, "store", &expected)?;
        let new_workers = worker_bindings(&reload_workers, &reload_id, "retrieve", &expected)?;
        let mut workers = BTreeMap::new();
        for (rank, (_, old_incarnation, old_id)) in &old_workers {
            let (new_process, new_incarnation, new_id) = &new_workers[rank];
            // The entire worker cohorts must be disjoint, including rank permutations.
            if old_workers
                .values()
                .any(|(process, _, _)| process == new_process)
                || old_incarnation == new_incarnation
                || old_id == new_id
            {
                return Err(
                    "KV worker process did not restart; adapter generation alone is insufficient"
                        .into(),
                );
            }
            workers.insert(*rank, [*old_id, *new_id]);
        }
        transfers(&caches, &store_id, &reload_id, &expected, &workers)
    })();
    match source_result {
        Ok(()) => report.source_copy_supported = true,
        Err(reason) => report.reasons.push(reason),
    }
    if let Some(root) = &args.reload_capture {
        let reload = load(root)?;
        let store = load(&root.join("store"))?;
        if reload.capture.stage != Stage::Reload
            || store.capture.stage != Stage::Store
            || reload.capture.store_sha256.as_ref() != Some(&store.digest)
            || store.capture.engine.as_ref() != Some(&before.process)
            || reload.capture.engine.as_ref() != Some(&after.process)
            || store.capture.cache_server != reload.capture.cache_server
            || !caches
                .iter()
                .any(|j| store.capture.cache_server.as_ref() == Some(&j.process))
            || store.capture.requests.len() != 1
            || reload.capture.requests.len() != 1
            || store.capture.requests[0].id != expected.store_request
            || reload.capture.requests[0].id != expected.reload_request
            || !matches!(reload.plan.study, Study::RestartCache { .. })
        {
            report.reasons.push(
                "KV source journals do not bind retained request/process/lifecycle evidence".into(),
            );
            report.source_copy_supported = false;
        }
        // Existing outcomes and controls are retained independently; copy evidence
        // cannot erase an unobserved warmup or reinterpret an old unavailable row.
        report.admission_warmup_eligible = caches.len() == 1
            && store.inspection.outcome == Outcome::Pass
            && reload.inspection.outcome == Outcome::Pass;
        if caches.len() > 1 {
            report.reasons.push("legacy single-cache-PID admission evidence cannot qualify a distributed producer set".into());
        }
        report.historical_store = Some(store.inspection);
        report.historical_reload = Some(reload.inspection);
    } else {
        report
            .reasons
            .push("independently retained startup admission/warmup evidence absent".into());
    }
    if report.source_copy_supported && report.admission_warmup_eligible {
        report.outcome = Outcome::Pass;
    } else if !report.admission_warmup_eligible {
        report.reasons.push("admission/warmup eligibility remains unavailable or failed; KV events cannot establish it".into());
    }
    Ok(report)
}
