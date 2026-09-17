//! Version 3 candidate source-copy evidence using the frozen v2 geometry projection.
use super::{ProcessIdentity, kv_registration as registration};
use crate::{evidence, model::Result, policy::Outcome};
use clap::Args;
use registration::{
    CacheJournal, CacheRegistration, RegisteredStateTemplate, WorkerRecord, WorkerRegistration,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    path::{Path, PathBuf},
};

const SOURCE: &str = "vllm-752a3a504-lmcache-ddc5fa34-kv-copy-v3";
const SCOPE: &str = "registered-state-l1-engine-restart-v3";
const CAP: usize = 16 * 1024 * 1024;

#[derive(Args)]
pub struct VerifyArgs {
    #[arg(long)]
    pub expectation: PathBuf,
    #[arg(long, required = true)]
    pub old_worker: Vec<PathBuf>,
    #[arg(long, required = true)]
    pub reload_worker: Vec<PathBuf>,
    #[arg(long, required = true)]
    pub cache: Vec<PathBuf>,
    #[arg(long)]
    pub store_frontend: PathBuf,
    #[arg(long)]
    pub reload_frontend: PathBuf,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Expectation {
    version: u32,
    source: String,
    scope: String,
    producer_sha256: String,
    #[serde(deserialize_with = "crate::wire::object")]
    registered_state_v2: RegisteredStateTemplate,
    salt: String,
    store_request: String,
    reload_request: String,
    start: u64,
    end: u64,
}

// A streaming prepass preserves duplicate-key rejection even through internally
// tagged serde buffering and arbitrary nested source-pin maps. No Value map is
// used to normalize or compare evidence.
struct Unique;
impl<'de> Deserialize<'de> for Unique {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = Unique;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("unique-key JSON")
            }
            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                mut map: M,
            ) -> std::result::Result<Unique, M::Error> {
                let mut keys = BTreeSet::new();
                while let Some(key) = map.next_key::<String>()? {
                    if !keys.insert(key) {
                        return Err(serde::de::Error::custom("duplicate JSON key"));
                    }
                    map.next_value::<Unique>()?;
                }
                Ok(Unique)
            }
            fn visit_seq<S: serde::de::SeqAccess<'de>>(
                self,
                mut seq: S,
            ) -> std::result::Result<Unique, S::Error> {
                while seq.next_element::<Unique>()?.is_some() {}
                Ok(Unique)
            }
            fn visit_bool<E: serde::de::Error>(self, _: bool) -> std::result::Result<Unique, E> {
                Ok(Unique)
            }
            fn visit_i64<E: serde::de::Error>(self, _: i64) -> std::result::Result<Unique, E> {
                Ok(Unique)
            }
            fn visit_u64<E: serde::de::Error>(self, _: u64) -> std::result::Result<Unique, E> {
                Ok(Unique)
            }
            fn visit_f64<E: serde::de::Error>(self, _: f64) -> std::result::Result<Unique, E> {
                Err(E::custom(
                    "floating point is outside the candidate protocol",
                ))
            }
            fn visit_str<E: serde::de::Error>(self, _: &str) -> std::result::Result<Unique, E> {
                Ok(Unique)
            }
            fn visit_unit<E: serde::de::Error>(self) -> std::result::Result<Unique, E> {
                Ok(Unique)
            }
        }
        d.deserialize_any(Visitor)
    }
}
fn decode<T: serde::de::DeserializeOwned>(raw: &[u8]) -> Result<T> {
    serde_json::from_slice::<Unique>(raw).map_err(|e| e.to_string())?;
    crate::wire::object(&mut serde_json::Deserializer::from_slice(raw)).map_err(|e| e.to_string())
}
fn check(ok: bool, reason: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(format!("KV v3 candidate: {reason}"))
    }
}

fn required_option<'de, D: serde::Deserializer<'de>, T: Deserialize<'de>>(
    d: D,
) -> std::result::Result<Option<T>, D::Error> {
    Option::<T>::deserialize(d)
}
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
struct Key {
    hash: String,
    model: String,
    kv_rank: u32,
    group: usize,
    salt: String,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Finalized {
    #[serde(deserialize_with = "crate::wire::object")]
    key: Key,
    success: bool,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Component {
    kernel_group_id: usize,
    shape: [u64; 4],
    dtype: registration::Dtype,
}
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Direction {
    Store,
    Retrieve,
}
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum Field {
    Keys,
    Blocks,
    CallbackKeys,
    Finalized,
    Exclusions,
}
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Excluded {
    index: usize,
    reason: String,
}
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Items {
    Keys(#[serde(deserialize_with = "crate::wire::objects")] Vec<Key>),
    Blocks(Vec<u64>),
    Finalized(#[serde(deserialize_with = "crate::wire::objects")] Vec<Finalized>),
    Exclusions(#[serde(deserialize_with = "crate::wire::objects")] Vec<Excluded>),
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Page {
    operation: u64,
    field: Field,
    group: usize,
    offset: usize,
    total: usize,
    items: Items,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Row {
    version: u32,
    sequence: usize,
    incarnation: String,
    offset_us: u64,
    #[serde(deserialize_with = "crate::wire::object")]
    event: Event,
}
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Event {
    Start {
        source: String,
        scope: String,
        role: String,
        observation_policy: BTreeMap<String, String>,
        #[serde(deserialize_with = "crate::wire::object")]
        process: ProcessIdentity,
        producer_sha256: String,
        components: BTreeMap<String, String>,
        max_events: usize,
        max_bytes: usize,
        max_window_us: u64,
        hash_randomization: u8,
        #[serde(deserialize_with = "required_option")]
        pythonhashseed: Option<String>,
    },
    Installed {
        pins: BTreeMap<String, String>,
        scope: String,
    },
    WorkerRegistration(#[serde(deserialize_with = "crate::wire::object")] WorkerRegistration),
    CacheRegistration(#[serde(deserialize_with = "crate::wire::object")] CacheRegistration),
    Binding {
        request: String,
        engine_request: String,
        #[serde(deserialize_with = "required_option")]
        parent: Option<String>,
        index: usize,
    },
    ScheduledStep {
        req_ids: Vec<String>,
        total_tokens: u64,
        connector_meta: bool,
    },
    WorkerSubmission {
        direction: Direction,
        engine: u64,
        session: String,
        rank: usize,
        world_size: usize,
        model: String,
        salt: String,
        start: u64,
        end: u64,
        num_kv_readers: usize,
        generation: u64,
    },
    WorkerResult {
        direction: Direction,
        engine: u64,
        session: String,
        generation: u64,
        success: bool,
    },
    Operation {
        operation: u64,
        direction: Direction,
        session: String,
        engine: u64,
        rank: usize,
        world_size: usize,
        model: String,
        salt: String,
        start: u64,
        end: u64,
        num_kv_readers: usize,
    },
    Page(#[serde(deserialize_with = "crate::wire::object")] Page),
    Allocation {
        allocation: u64,
        #[serde(deserialize_with = "crate::wire::object")]
        key: Key,
        #[serde(deserialize_with = "crate::wire::objects")]
        components: Vec<Component>,
        bytes: u64,
    },
    AllocationRetired {
        allocation: u64,
        reason: String,
    },
    Operand {
        operation: u64,
        group: usize,
        index: usize,
        allocation: u64,
    },
    Execution {
        operation: u64,
        group: usize,
        start: usize,
        count: usize,
        kernels: Vec<usize>,
        mode: String,
        leg: String,
    },
    #[serde(rename = "event_recorded")]
    Recorded {
        operation: u64,
    },
    CallbackSubmitted {
        operation: u64,
        callback: String,
    },
    CallbackDelivered {
        operation: u64,
        callback: String,
    },
    Finalized {
        operation: u64,
    },
    DeviceComplete {
        operation: u64,
    },
    Result {
        operation: u64,
        success: bool,
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
    incarnation: String,
    digest: String,
}
fn component_pins() -> BTreeMap<String, String> {
    [
        (
            "kv-registration.py",
            include_bytes!("../../../tools/kv-registration.py").as_slice(),
        ),
        (
            "kv-registration-hooks.py",
            include_bytes!("../../../tools/kv-registration-hooks.py").as_slice(),
        ),
        (
            "kv-lifetime-v2.py",
            include_bytes!("../../../tools/kv-lifetime-v2.py").as_slice(),
        ),
        (
            "kv-transfer-v2.py",
            include_bytes!("../../../tools/kv-transfer-v2.py").as_slice(),
        ),
        (
            "kv-v2-pins.json",
            include_bytes!("../../../tools/kv-v2-pins.json").as_slice(),
        ),
    ]
    .into_iter()
    .map(|(name, bytes)| (name.to_owned(), evidence::digest(bytes)))
    .collect()
}
fn journal(
    path: &Path,
    expected: &Expectation,
    role: &str,
    remaining: &mut usize,
) -> Result<Journal> {
    let raw = evidence::read(path, *remaining)?;
    *remaining = remaining
        .checked_sub(raw.len())
        .ok_or("journal byte budget")?;
    check(raw.last() == Some(&b'\n'), "journal lacks terminal newline")?;
    let mut rows: Vec<Row> = Vec::new();
    for line in raw.split_inclusive(|&b| b == b'\n') {
        check(rows.len() < 10000 && line.len() <= 65536, "row bound")?;
        let row: Row = decode(line)?;
        check(
            row.version == 3
                && row.sequence == rows.len()
                && row.incarnation.len() == 32
                && row.incarnation.bytes().all(|b| b.is_ascii_hexdigit())
                && rows.last().is_none_or(|last| {
                    last.incarnation == row.incarnation
                        && last.offset_us <= row.offset_us
                        && !matches!(last.event, Event::Fence { .. })
                }),
            "journal version, order or incarnation",
        )?;
        rows.push(row);
    }
    let Some(Row {
        event:
            Event::Start {
                source,
                scope,
                role: found_role,
                process,
                observation_policy,
                producer_sha256,
                components,
                max_events,
                max_bytes,
                max_window_us,
                hash_randomization,
                pythonhashseed,
            },
        ..
    }) = rows.first()
    else {
        return Err("missing v3 candidate start".into());
    };
    check(
        source == SOURCE
            && scope == SCOPE
            && *observation_policy
                == BTreeMap::from([
                    ("dcp".into(), "1".into()),
                    ("groups".into(), "attention-only".into()),
                    ("request_configs".into(), "absent".into()),
                    ("event_backend".into(), "default-torch-cuda".into()),
                    ("registration".into(), "single-generation".into()),
                    (
                        "admission".into(),
                        "candidate-not-independent-or-native".into(),
                    ),
                ])
            && found_role == role
            && producer_sha256 == &expected.producer_sha256
            && *components == component_pins()
            && process.pid > 0
            && process.start_ticks > 0
            && !process.boot_id.is_empty()
            && process.boot_id.len() <= 256
            && (8..=10000).contains(max_events)
            && (4096..=CAP).contains(max_bytes)
            && (1000..=3600000000).contains(max_window_us)
            && rows.len() <= *max_events
            && raw.len() <= *max_bytes
            && rows
                .last()
                .is_some_and(|row| row.offset_us <= *max_window_us)
            && *hash_randomization == 0
            && pythonhashseed.as_deref() == Some("0"),
        "start identity, source, hash seed or bounds",
    )?;
    let source_sets: BTreeMap<String, BTreeMap<String, String>> =
        decode(include_bytes!("../../../tools/kv-v2-pins.json"))?;
    check(
        matches!(rows.get(1).map(|row| &row.event), Some(Event::Installed { pins, scope })
        if source_sets.get(role) == Some(pins) && scope == SCOPE),
        "whole installed source pin set",
    )?;
    let count = rows
        .iter()
        .filter(|row| matches!(row.event, Event::Operation { .. }))
        .count();
    check(
        matches!(rows.last().map(|row| &row.event), Some(Event::Fence {
        admission_closed: true, active: 0, pending: 0, operations, complete: true, hooks_intact: true
    }) if *operations == count),
        "incomplete terminal fence",
    )?;
    for (index, row) in rows.iter().enumerate() {
        let allowed = match &row.event {
            Event::Start { .. } => index == 0,
            Event::Installed { .. } => index == 1,
            Event::Fence { .. } => index + 1 == rows.len(),
            Event::Failure { reason } => return Err(format!("source failure: {reason}")),
            Event::Binding { .. } => role == "frontend",
            Event::WorkerRegistration(_)
            | Event::ScheduledStep { .. }
            | Event::WorkerResult { .. }
            | Event::WorkerSubmission { .. } => role == "worker",
            _ => role == "cache",
        };
        check(allowed, "wrong-role or duplicate lifecycle event")?;
    }
    let process = process.clone();
    let incarnation = rows[0].incarnation.clone();
    Ok(Journal {
        rows,
        process,
        incarnation,
        digest: evidence::digest(&raw),
    })
}

fn binding(j: &Journal, request: &str) -> Result<String> {
    let mut found = None;
    for row in &j.rows {
        if let Event::Binding {
            request: observed,
            engine_request,
            parent,
            index,
        } = &row.event
        {
            check(
                observed == request
                    && parent.is_none()
                    && *index == 0
                    && !engine_request.is_empty()
                    && engine_request.len() <= 256
                    && found.is_none(),
                "unrelated or repeated frontend binding",
            )?;
            found = Some(engine_request.clone());
        }
    }
    found.ok_or_else(|| "frontend has no selected request binding".into())
}
fn worker(j: &Journal) -> Result<&WorkerRegistration> {
    let records: Vec<_> = j
        .rows
        .iter()
        .filter_map(|row| match &row.event {
            Event::WorkerRegistration(w) => Some(w),
            _ => None,
        })
        .collect();
    let [registration] = records.as_slice() else {
        return Err("worker requires exactly one registration".into());
    };
    Ok(registration)
}
fn worker_submissions(
    j: &Journal,
    session: &str,
    direction: Direction,
    expected: &Expectation,
) -> Result<()> {
    let w = worker(j)?;
    let mut registered = false;
    let mut submissions = 0;
    let mut scheduled = BTreeSet::new();
    let mut pending = None;
    for row in &j.rows {
        match &row.event {
            Event::WorkerRegistration(_) => registered = true,
            Event::ScheduledStep {
                req_ids,
                total_tokens,
                connector_meta,
            } => {
                check(
                    req_ids.len() <= 64
                        && *total_tokens <= (1 << 20)
                        && req_ids.windows(2).all(|pair| pair[0] < pair[1])
                        && req_ids.iter().all(|id| id == session)
                        && (!req_ids.is_empty() || *total_tokens == 0),
                    "unbound or malformed scheduled input",
                )?;
                if *connector_meta && !registered {
                    return Err("connector scheduling precedes registration".into());
                }
                scheduled.extend(req_ids.iter().map(String::as_str));
            }
            Event::WorkerSubmission {
                direction: found,
                engine,
                session: found_session,
                rank,
                world_size,
                model,
                salt,
                start,
                end,
                num_kv_readers,
                generation,
            } => {
                check(
                    registered
                        && pending.is_none()
                        && *generation == submissions as u64 + 1
                        && *found == direction
                        && *engine == w.engine_instance
                        && found_session == session
                        && *rank == w.cache_partition.rank
                        && *world_size == w.cache_partition.world_size
                        && *num_kv_readers
                            == if w.model_flags.mla_only {
                                w.cache_partition.kv_tp_size
                            } else {
                                1
                            }
                        && model == &w.model
                        && salt == &expected.salt
                        && *start == expected.start
                        && *end == expected.end,
                    "worker submission registration/request mismatch",
                )?;
                submissions += 1;
                pending = Some(*generation);
            }
            Event::WorkerResult {
                direction: found,
                engine,
                session: found_session,
                generation,
                success,
            } => {
                check(
                    *success
                        && *found == direction
                        && *engine == w.engine_instance
                        && found_session == session
                        && pending == Some(*generation),
                    "worker result lacks exact successful submission generation",
                )?;
                pending = None;
            }
            _ => (),
        }
    }
    check(
        scheduled == BTreeSet::from([session])
            && pending.is_none()
            && submissions
                == usize::from(direction == Direction::Retrieve || w.cache_partition.is_writer),
        "missing reader/writer submission or scheduled coverage",
    )
}

#[derive(Default)]
struct Pages<'a> {
    total: usize,
    count: usize,
    keys: Vec<&'a Key>,
    blocks: Vec<u64>,
    finalized: Vec<&'a Finalized>,
    exclusions: Vec<&'a Excluded>,
}
impl<'a> Pages<'a> {
    fn append(&mut self, page: &'a Page) -> Result<()> {
        let count = match &page.items {
            Items::Keys(x) => x.len(),
            Items::Blocks(x) => x.len(),
            Items::Finalized(x) => x.len(),
            Items::Exclusions(x) => x.len(),
        };
        check(
            page.total > 0
                && page.total <= 65536
                && page.offset == self.count
                && page.offset < page.total
                && (self.count == 0 || self.total == page.total)
                && count > 0
                && count == (page.total - page.offset).min(64),
            "page offset, count or total",
        )?;
        match (&page.field, &page.items) {
            (Field::Keys | Field::CallbackKeys, Items::Keys(items)) => self.keys.extend(items),
            (Field::Blocks, Items::Blocks(items)) => self.blocks.extend(items),
            (Field::Finalized, Items::Finalized(items)) => self.finalized.extend(items),
            (Field::Exclusions, Items::Exclusions(items)) => self.exclusions.extend(items),
            _ => return Err("page item type differs from field".into()),
        }
        self.count += count;
        self.total = page.total;
        Ok(())
    }
}
struct Allocation<'a> {
    key: &'a Key,
    retired: bool,
}
struct Operation<'a> {
    engine: u64,
    rank: usize,
    direction: Direction,
    admission: usize,
    published: bool,
    pages: BTreeMap<(Field, usize), Pages<'a>>,
    operands: BTreeMap<(usize, usize), u64>,
    kernels: BTreeSet<(usize, usize, usize)>,
    staged: BTreeSet<(usize, usize)>,
    recorded: bool,
    submitted: bool,
    delivered: bool,
    finalized: bool,
    completed: Option<usize>,
    result: bool,
}
impl<'a> Operation<'a> {
    fn page(&self, field: Field, group: usize) -> Result<&Pages<'a>> {
        let page = self
            .pages
            .get(&(field, group))
            .ok_or("missing required page inventory")?;
        check(page.count == page.total, "incomplete page inventory")?;
        Ok(page)
    }
}

fn transfers(
    caches: &[Journal],
    roster: &registration::RegistryRoster,
    expected: &Expectation,
    store_session: &str,
    reload_session: &str,
) -> Result<usize> {
    let layout = &expected.registered_state_v2.cache_template;
    let chunk = layout.lmcache_tokens_per_chunk;
    let chunks = ((expected.end - expected.start) / chunk) as usize;
    let mut engine_roster = BTreeMap::new();
    for (direction, ranks) in [
        (Direction::Store, &roster.old),
        (Direction::Retrieve, &roster.reload),
    ] {
        for rank in ranks.values() {
            engine_roster.insert(rank.engine_instance, (direction, rank));
        }
    }
    let mut seen = BTreeSet::new();
    let mut copied = 0;
    for (journal_index, journal) in caches.iter().enumerate() {
        let mut registrations = BTreeSet::new();
        let mut operations: BTreeMap<u64, Operation<'_>> = BTreeMap::new();
        let mut allocations: BTreeMap<u64, Allocation<'_>> = BTreeMap::new();
        // Stored records are published only after every copy/finalization/event
        // predicate is satisfied; their sequence must precede reload admission.
        let mut resident: BTreeMap<&Key, (u64, usize, Vec<Vec<bool>>)> = BTreeMap::new();
        for row in &journal.rows {
            match &row.event {
                Event::CacheRegistration(c) => {
                    registrations.insert(c.engine_instance);
                }
                Event::Operation {
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
                    num_kv_readers,
                } => {
                    let &(phase, binding) = engine_roster
                        .get(engine)
                        .ok_or("operation engine outside roster")?;
                    check(
                        registrations.contains(engine)
                            && phase == *direction
                            && binding.cache_journal == journal_index
                            && *rank == binding.cache_rank
                            && *world_size == layout.cache_world_size
                            && *num_kv_readers
                                == if expected.registered_state_v2.model_flags.mla_only {
                                    expected.registered_state_v2.physical_template.tp_size
                                        / expected.registered_state_v2.physical_template.n_servers
                                } else {
                                    1
                                }
                            && model == &layout.model
                            && salt == &expected.salt
                            && *start == expected.start
                            && *end == expected.end
                            && session
                                == if phase == Direction::Store {
                                    store_session
                                } else {
                                    reload_session
                                }
                            && (phase == Direction::Retrieve || binding.is_writer)
                            && seen.insert(*engine)
                            && *operation == operations.len() as u64 + 1,
                        "unrelated/duplicate operation or roster mismatch",
                    )?;
                    operations.insert(
                        *operation,
                        Operation {
                            engine: *engine,
                            rank: *rank,
                            direction: *direction,
                            admission: row.sequence,
                            published: false,
                            pages: BTreeMap::new(),
                            operands: BTreeMap::new(),
                            kernels: BTreeSet::new(),
                            staged: BTreeSet::new(),
                            recorded: false,
                            submitted: false,
                            delivered: false,
                            finalized: false,
                            completed: None,
                            result: false,
                        },
                    );
                }
                Event::Page(page) => {
                    let op = operations.get_mut(&page.operation).ok_or("unbound page")?;
                    check(
                        match page.field {
                            Field::Keys | Field::Exclusions => {
                                !op.recorded
                                    && op.operands.is_empty()
                                    && page.group < layout.object_groups.len()
                            }
                            Field::Blocks => {
                                !op.recorded
                                    && op.operands.is_empty()
                                    && page.group < layout.kernel_groups.len()
                            }
                            Field::CallbackKeys => op.recorded && !op.submitted && page.group == 0,
                            Field::Finalized => op.submitted && !op.finalized && page.group == 0,
                        },
                        "page phase or group mismatch",
                    )?;
                    op.pages
                        .entry((page.field, page.group))
                        .or_default()
                        .append(page)?;
                }
                Event::Allocation {
                    allocation,
                    key,
                    components,
                    bytes,
                } => {
                    check(
                        *allocation == allocations.len() as u64 + 1
                            && key.group < layout.object_groups.len(),
                        "allocation generation or group",
                    )?;
                    let members = &layout.object_groups[key.group].kernel_group_ids;
                    check(
                        components.len() == members.len()
                            && components.iter().zip(members).all(|(component, &kid)| {
                                let kernel = &layout.kernel_groups[kid];
                                component.kernel_group_id == kid
                                    && component.shape == kernel.object_shape
                                    && component.dtype == kernel.dtype
                            }),
                        "allocation component order, shape or dtype",
                    )?;
                    let total = members.iter().try_fold(0_u64, |n, &kid| {
                        n.checked_add(layout.kernel_groups[kid].allocation_bytes)
                            .ok_or("allocation sum overflow")
                    })?;
                    check(*bytes == total, "allocation bytes")?;
                    allocations.insert(
                        *allocation,
                        Allocation {
                            key,
                            retired: false,
                        },
                    );
                }
                Event::AllocationRetired { allocation, reason } => {
                    check(
                        matches!(
                            reason.as_str(),
                            "free_started"
                                | "invalidated"
                                | "size_changed"
                                | "write_reserved"
                                | "collected"
                        ),
                        "unknown allocation retirement",
                    )?;
                    let allocation = allocations
                        .get_mut(allocation)
                        .ok_or("unknown retired generation")?;
                    check(!allocation.retired, "duplicate allocation retirement")?;
                    allocation.retired = true;
                }
                Event::Operand {
                    operation,
                    group,
                    index,
                    allocation,
                } => {
                    let op = operations.get_mut(operation).ok_or("unbound operand")?;
                    for kid in 0..layout.kernel_groups.len() {
                        op.page(Field::Blocks, kid)?;
                    }
                    let object = allocations
                        .get(allocation)
                        .ok_or("unknown allocation operand")?;
                    let keys = &op.page(Field::Keys, *group)?.keys;
                    check(
                        !op.recorded
                            && *index < keys.len()
                            && !object.retired
                            && object.key == keys[*index]
                            && op.operands.insert((*group, *index), *allocation).is_none(),
                        "operand identity, order or lifetime",
                    )?;
                }
                Event::Execution {
                    operation,
                    group,
                    start,
                    count,
                    kernels,
                    mode,
                    leg,
                } => {
                    let op = operations.get_mut(operation).ok_or("unbound execution")?;
                    check(
                        !op.recorded
                            && *group < layout.object_groups.len()
                            && *count > 0
                            && start.checked_add(*count).is_some_and(|end| end <= chunks),
                        "execution range or phase",
                    )?;
                    let members = &layout.object_groups[*group].kernel_group_ids;
                    let native = mode == "native-object-group" && leg == "complete";
                    let paged = matches!(mode.as_str(), "torch-per-kernel" | "native-per-kernel")
                        && leg == "kernel";
                    let staging = mode == "staging"
                        && matches!(leg.as_str(), "host-to-staging" | "staging-to-host");
                    check(
                        (native && kernels == members)
                            || (paged && kernels.len() == 1 && members.contains(&kernels[0]))
                            || (staging && kernels.is_empty()),
                        "unverified execution mode or kernel inventory",
                    )?;
                    for index in *start..start + count {
                        check(
                            op.operands.contains_key(&(*group, index)),
                            "execution lacks allocation operand",
                        )?;
                        if paged && op.direction == Direction::Retrieve {
                            check(
                                op.staged.contains(&(*group, index)),
                                "paged retrieve precedes host staging",
                            )?;
                        }
                        if leg == "staging-to-host" {
                            check(
                                members
                                    .iter()
                                    .all(|&kid| op.kernels.contains(&(*group, index, kid))),
                                "host copy precedes paged store kernels",
                            )?;
                        }
                        for &kid in kernels {
                            check(
                                op.kernels.insert((*group, index, kid)),
                                "duplicate kernel execution",
                            )?;
                        }
                        if native || staging {
                            check(
                                native
                                    || (op.direction == Direction::Retrieve
                                        && leg == "host-to-staging")
                                    || (op.direction == Direction::Store
                                        && leg == "staging-to-host"),
                                "wrong staging direction",
                            )?;
                            check(op.staged.insert((*group, index)), "duplicate host staging")?;
                        }
                    }
                }
                Event::Recorded { operation } => {
                    let op = operations.get_mut(operation).ok_or("unbound event")?;
                    check(!op.recorded, "duplicate source event")?;
                    op.recorded = true;
                }
                Event::CallbackSubmitted {
                    operation,
                    callback,
                } => {
                    let op = operations.get_mut(operation).ok_or("unbound callback")?;
                    check(
                        op.recorded
                            && !op.submitted
                            && callback
                                == if op.direction == Direction::Store {
                                    "finish_write"
                                } else {
                                    "finish_read_prefetched"
                                },
                        "callback submission order",
                    )?;
                    op.page(Field::CallbackKeys, 0)?;
                    op.submitted = true;
                }
                Event::CallbackDelivered {
                    operation,
                    callback,
                } => {
                    let op = operations
                        .get_mut(operation)
                        .ok_or("unbound callback delivery")?;
                    check(
                        op.submitted
                            && !op.delivered
                            && callback
                                == if op.direction == Direction::Store {
                                    "finish_write"
                                } else {
                                    "finish_read_prefetched"
                                },
                        "callback delivery order",
                    )?;
                    op.delivered = true;
                }
                Event::Finalized { operation } => {
                    let op = operations
                        .get_mut(operation)
                        .ok_or("unbound finalization")?;
                    check(
                        op.direction == Direction::Store
                            && op.submitted
                            && !op.finalized
                            && !op.delivered,
                        "write finalization order",
                    )?;
                    op.page(Field::Finalized, 0)?;
                    op.finalized = true;
                }
                Event::DeviceComplete { operation } => {
                    let op = operations.get_mut(operation).ok_or("unbound completion")?;
                    check(
                        op.recorded && op.completed.is_none(),
                        "completion before event or duplicate",
                    )?;
                    for allocation in op.operands.values() {
                        check(
                            !allocations[allocation].retired,
                            "allocation retired before completion",
                        )?;
                    }
                    op.completed = Some(row.sequence);
                }
                Event::Result { operation, success } => {
                    let op = operations.get_mut(operation).ok_or("unbound result")?;
                    check(
                        *success && !op.result && op.recorded,
                        "failed, premature or duplicate result",
                    )?;
                    op.result = true;
                }
                _ => (),
            }
            // Completion/finalization may arrive in either order. Publish store
            // residency at the last required predicate, never at callback alone.
            let changed = match &row.event {
                Event::Result { operation, .. }
                | Event::CallbackDelivered { operation, .. }
                | Event::Finalized { operation }
                | Event::DeviceComplete { operation } => operations.get_mut(operation),
                _ => None,
            };
            if let Some(op) = changed {
                if op.published
                    || op.direction != Direction::Store
                    || !op.result
                    || !op.delivered
                    || !op.finalized
                    || op.completed.is_none()
                {
                    continue;
                }
                let extents = validate_operation(
                    op,
                    layout,
                    &engine_roster[&op.engine].1.kernel_capacities,
                    expected,
                    chunks,
                )?;
                for (key, token, valid) in extents {
                    resident.entry(key).or_insert((token, row.sequence, valid));
                }
                op.published = true;
            }
        }
        for op in operations.values() {
            check(
                op.result
                    && op.delivered
                    && op.completed.is_some()
                    && (op.direction == Direction::Retrieve || op.finalized),
                "operation not fully completed",
            )?;
            if op.direction == Direction::Store {
                check(op.published, "store residency was not published")?;
                continue;
            }
            let extents = validate_operation(
                op,
                layout,
                &engine_roster[&op.engine].1.kernel_capacities,
                expected,
                chunks,
            )?;
            check(
                extents
                    .iter()
                    .any(|(_, _, valid)| valid.iter().flatten().any(|&bit| bit)),
                "reader has no nonnull retained state",
            )?;
            for (key, token, valid) in extents {
                let (stored_token, sequence, stored_valid) = resident
                    .get(key)
                    .ok_or("reload has no same-cache completed store")?;
                check(
                    *stored_token == token
                        && *sequence < op.admission
                        && stored_valid.len() == valid.len()
                        && stored_valid.iter().zip(&valid).all(|(before, after)| {
                            before.len() == after.len()
                                && before
                                    .iter()
                                    .zip(after)
                                    .all(|(&stored, &required)| !required || stored)
                        }),
                    "reload precedes store completion, changed L1 generation or missing valid kernel state",
                )?;
                copied += 1;
            }
        }
        let used_allocations: BTreeSet<_> = operations
            .values()
            .flat_map(|op| op.operands.values().copied())
            .collect();
        check(
            used_allocations == allocations.keys().copied().collect(),
            "unbound allocation generation",
        )?;
    }
    let required: BTreeSet<_> = engine_roster
        .iter()
        .filter(|(_, (direction, rank))| *direction == Direction::Retrieve || rank.is_writer)
        .map(|(&engine, _)| engine)
        .collect();
    check(
        seen == required && copied > 0,
        "incomplete writer/physical-reader operation roster or empty copy",
    )?;
    Ok(copied)
}

type ValidExtent<'a> = (&'a Key, u64, Vec<Vec<bool>>);
fn validate_operation<'a>(
    op: &Operation<'a>,
    layout: &registration::CacheTemplate,
    capacities: &[u64],
    expected: &Expectation,
    chunks: usize,
) -> Result<Vec<ValidExtent<'a>>> {
    let chunk = layout.lmcache_tokens_per_chunk;
    let mut raw = Vec::new();
    for (kid, kernel) in layout.kernel_groups.iter().enumerate() {
        let blocks = &op.page(Field::Blocks, kid)?.blocks;
        let per_chunk =
            usize::try_from(chunk / kernel.tokens_per_block).map_err(|_| "block count")?;
        check(
            blocks.len()
                == chunks
                    .checked_mul(per_chunk)
                    .ok_or("block count overflow")?
                && blocks.iter().all(|&block| block < capacities[kid]),
            "raw block coverage or registered capacity",
        )?;
        raw.push(blocks);
    }
    let mut required_kernels = 0;
    let mut extents = Vec::new();
    let shared_keys = &op.page(Field::Keys, 0)?.keys;
    for (group, object) in layout.object_groups.iter().enumerate() {
        let mut exclusions = Vec::new();
        let keys = &op.page(Field::Keys, group)?.keys;
        check(keys.len() == chunks, "candidate key coverage")?;
        let skip = if op.direction == Direction::Retrieve && object.num_chunks_in_sw > 0 {
            chunks.saturating_sub(object.num_chunks_in_sw as usize)
        } else {
            0
        };
        for (index, &key) in keys.iter().enumerate() {
            let world = layout.cache_world_size as u32;
            let rank = op.rank as u32;
            let packed = (world << 24) | (rank << 16) | (world << 8) | rank;
            check(
                key.group == group
                    && key.model == layout.model
                    && key.salt == expected.salt
                    && key.kv_rank == packed
                    && key.hash.len() == 64
                    && key.hash.bytes().all(|b| b.is_ascii_hexdigit()),
                "key identity",
            )?;
            check(
                shared_keys[index].hash == key.hash,
                "hash differs across object groups",
            )?;
            let mut valid = Vec::new();
            let mut all_null = true;
            for &kid in &object.kernel_group_ids {
                let kernel = &layout.kernel_groups[kid];
                let bpc = (chunk / kernel.tokens_per_block) as usize;
                let blocks = &raw[kid][index * bpc..(index + 1) * bpc];
                all_null &= blocks.iter().all(|&block| block == 0);
                let retained = if kernel.sw_size_tokens > 0 {
                    chunk.min(kernel.sw_size_tokens as u64)
                } else {
                    chunk
                };
                let keep = (retained / kernel.tokens_per_block) as usize;
                valid.push(
                    blocks[bpc - keep..]
                        .iter()
                        .map(|&block| block != 0)
                        .collect(),
                );
            }
            let needed = index >= skip && (op.direction == Direction::Retrieve || !all_null);
            if !needed {
                exclusions.push(Excluded {
                    index,
                    reason: if index < skip {
                        "window_prefix"
                    } else {
                        "all_null"
                    }
                    .into(),
                });
                continue;
            }
            required_kernels += object.kernel_group_ids.len();
            let &token = op
                .operands
                .get(&(group, index))
                .ok_or("required key missing live operand")?;
            check(
                object
                    .kernel_group_ids
                    .iter()
                    .all(|&kid| op.kernels.contains(&(group, index, kid)))
                    && op.staged.contains(&(group, index)),
                "required kernel or staging leg missing",
            )?;
            extents.push((key, token, valid));
        }
        let observed = op.pages.get(&(Field::Exclusions, group));
        check(
            match observed {
                Some(page) => {
                    page.count == page.total && page.exclusions.iter().copied().eq(&exclusions)
                }
                None => exclusions.is_empty(),
            },
            "missing, incomplete or unjustified exclusion witness",
        )?;
    }
    check(
        !extents.is_empty()
            && op.operands.len() == extents.len()
            && op.kernels.len() == required_kernels,
        "unexpected/missing operand or executed kernel",
    )?;
    let copied_keys: BTreeSet<_> = extents.iter().map(|(key, _, _)| *key).collect();
    check(
        copied_keys.len() == extents.len(),
        "duplicate copied key in logical chunks",
    )?;
    let callback = &op.page(Field::CallbackKeys, 0)?.keys;
    check(
        callback.len() == copied_keys.len()
            && callback.iter().copied().collect::<BTreeSet<_>>() == copied_keys,
        "callback keys differ from exact copies",
    )?;
    if op.direction == Direction::Store {
        let finalized = &op.page(Field::Finalized, 0)?.finalized;
        check(
            finalized.len() == copied_keys.len()
                && finalized.iter().all(|entry| entry.success)
                && finalized
                    .iter()
                    .map(|entry| &entry.key)
                    .collect::<BTreeSet<_>>()
                    == copied_keys,
            "write finalization coverage",
        )?;
    }
    Ok(extents)
}

#[derive(Serialize)]
pub struct Report {
    version: u32,
    observation_identity: &'static str,
    scope: &'static str,
    claim: &'static str,
    expectation_sha256: String,
    source_journal_sha256: Vec<String>,
    source_copy_supported: bool,
    admission_warmup_eligible: bool,
    physical_workers: usize,
    cache_servers: usize,
    reloaded_objects: usize,
    pub outcome: Outcome,
    reasons: Vec<String>,
}

pub fn verify(args: &VerifyArgs) -> Result<Report> {
    let raw = evidence::read(&args.expectation, 65536)?;
    let expected: Expectation = decode(&raw)?;
    registration::validate_template(&expected.registered_state_v2)?;
    let layout = &expected.registered_state_v2.cache_template;
    check(
        expected.version == 3
            && expected.source == SOURCE
            && expected.scope == SCOPE
            && expected.producer_sha256
                == evidence::digest(include_bytes!("../../../tools/kv-journal-v2.py"))
            && expected.end > expected.start
            && (expected.end - expected.start).is_multiple_of(layout.lmcache_tokens_per_chunk)
            && (expected.end - expected.start) / layout.lmcache_tokens_per_chunk <= 1024
            && expected.salt.len() <= 128
            && !expected.store_request.is_empty()
            && expected.store_request.len() <= 256
            && !expected.reload_request.is_empty()
            && expected.reload_request.len() <= 256
            && expected.store_request != expected.reload_request,
        "invalid prospective v3 candidate expectation",
    )?;
    let world = expected
        .registered_state_v2
        .physical_template
        .vllm_world_size;
    let servers = expected.registered_state_v2.physical_template.n_servers;
    check(
        args.old_worker.len() == world
            && args.reload_worker.len() == world
            && args.cache.len() == servers,
        "prospective journal count",
    )?;
    let mut remaining = CAP;
    let old_frontend = journal(&args.store_frontend, &expected, "frontend", &mut remaining)?;
    let new_frontend = journal(&args.reload_frontend, &expected, "frontend", &mut remaining)?;
    check(
        old_frontend.process != new_frontend.process
            && old_frontend.incarnation != new_frontend.incarnation,
        "frontend did not restart",
    )?;
    let store_session = binding(&old_frontend, &expected.store_request)?;
    let reload_session = binding(&new_frontend, &expected.reload_request)?;
    check(
        store_session != reload_session,
        "engine request reused across phases",
    )?;
    let mut read_many = |paths: &[PathBuf], role| -> Result<Vec<Journal>> {
        paths
            .iter()
            .map(|path| journal(path, &expected, role, &mut remaining))
            .collect()
    };
    let old = read_many(&args.old_worker, "worker")?;
    let reload = read_many(&args.reload_worker, "worker")?;
    let caches = read_many(&args.cache, "cache")?;
    for j in &old {
        worker_submissions(j, &store_session, Direction::Store, &expected)?;
    }
    for j in &reload {
        worker_submissions(j, &reload_session, Direction::Retrieve, &expected)?;
    }
    let old_records: Vec<_> = old
        .iter()
        .map(|j| {
            Ok(WorkerRecord {
                process: &j.process,
                incarnation: &j.incarnation,
                registration: worker(j)?,
            })
        })
        .collect::<Result<_>>()?;
    let reload_records: Vec<_> = reload
        .iter()
        .map(|j| {
            Ok(WorkerRecord {
                process: &j.process,
                incarnation: &j.incarnation,
                registration: worker(j)?,
            })
        })
        .collect::<Result<_>>()?;
    let cache_records: Vec<Vec<_>> = caches
        .iter()
        .map(|j| {
            j.rows
                .iter()
                .filter_map(|row| match &row.event {
                    Event::CacheRegistration(c) => Some(c),
                    _ => None,
                })
                .collect()
        })
        .collect();
    let cache_journals: Vec<_> = caches
        .iter()
        .zip(&cache_records)
        .map(|(j, registrations)| CacheJournal {
            process: &j.process,
            incarnation: &j.incarnation,
            registrations,
        })
        .collect();
    let roster = registration::verify_rosters(
        &expected.registered_state_v2,
        &old_records,
        &reload_records,
        &cache_journals,
    )?;
    check(
        roster
            .cache_slots
            .iter()
            .all(|(slot, journal)| slot == journal),
        "cache CLI order differs from prospective server slots",
    )?;
    let reloaded_objects = transfers(&caches, &roster, &expected, &store_session, &reload_session)?;
    let source_journal_sha256 = [&old_frontend, &new_frontend]
        .into_iter()
        .chain(&old)
        .chain(&reload)
        .chain(&caches)
        .map(|j| j.digest.clone())
        .collect();
    Ok(Report { version: 3, observation_identity: SOURCE, scope: SCOPE,
        claim: "registered-state-l1-source-copy-observation-not-native-qualification",
        expectation_sha256: evidence::digest(&raw), source_journal_sha256,
        source_copy_supported: true, admission_warmup_eligible: false, physical_workers: world,
        cache_servers: servers, reloaded_objects, outcome: Outcome::Inconclusive,
        reasons: vec!["Registered L1 source-copy evidence does not establish native qualification or distributed admission/warmup eligibility".into()] })
}
