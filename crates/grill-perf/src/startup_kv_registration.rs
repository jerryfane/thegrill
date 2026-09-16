//! Registered topology and layout identity only; no transfer or persistence claim.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Deserializer, Serialize};

use crate::model::Result;

const MAX_SIGNED: u64 = i64::MAX as u64;
const MAX_DIM: u64 = i32::MAX as u64;
const MAX_TOKENS: u64 = 1 << 20;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(super) enum Schema {
    #[serde(rename = "thegrill.kv.registered-state.v2")]
    V2,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(super) enum EngineType {
    #[serde(rename = "VLLM")]
    Vllm,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(super) enum RegistrationKind {
    #[serde(rename = "new")]
    New,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(super) enum KvLayout {
    #[serde(rename = "NHD")]
    Nhd,
    #[serde(rename = "HND")]
    Hnd,
}

// deserialize_with prevents serde's implicit missing-Option default. Explicit
// null is valid, but the field must occur in the object.
fn required_layout<'de, D: Deserializer<'de>>(
    d: D,
) -> std::result::Result<Option<KvLayout>, D::Error> {
    Option::<KvLayout>::deserialize(d)
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct LayoutHints {
    #[serde(deserialize_with = "required_layout")]
    pub(super) kv_layout: Option<KvLayout>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(super) enum Dtype {
    #[serde(rename = "torch.float16")]
    Float16,
    #[serde(rename = "torch.float32")]
    Float32,
    #[serde(rename = "torch.bfloat16")]
    Bfloat16,
    #[serde(rename = "torch.uint8")]
    Uint8,
}

impl Dtype {
    fn itemsize(self) -> u64 {
        match self {
            Self::Float16 | Self::Bfloat16 => 2,
            Self::Float32 => 4,
            Self::Uint8 => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(super) enum EngineKvFormat {
    #[serde(rename = "NB_NL_TWO_BS_NH_HS")]
    NbNlTwoBsNhHs,
    #[serde(rename = "NL_X_TWO_NB_BS_NH_HS")]
    NlXTwoNbBsNhHs,
    #[serde(rename = "NL_X_NB_TWO_BS_NH_HS")]
    NlXNbTwoBsNhHs,
    #[serde(rename = "NL_X_NB_BS_HS")]
    NlXNbBsHs,
    #[serde(rename = "TWO_X_NL_X_NBBS_NH_HS")]
    TwoXNlXNbbsNhHs,
    #[serde(rename = "NL_X_NBBS_ONE_HS")]
    NlXNbbsOneHs,
    #[serde(rename = "NL_X_TWO_NB_NH_BS_HS")]
    NlXTwoNbNhBsHs,
    #[serde(rename = "NL_X_NB_TWO_NH_BS_HS")]
    NlXNbTwoNhBsHs,
    #[serde(rename = "NB_NL_TWO_NH_BS_HS")]
    NbNlTwoNhBsHs,
    #[serde(rename = "TWO_X_NL_X_NB_BS_NH_HS")]
    TwoXNlXNbBsNhHs,
    #[serde(rename = "NL_X_NB_NH_BS_TWO_HS")]
    NlXNbNhBsTwoHs,
    #[serde(rename = "NL_X_NB_BS_NH_TWO_HS")]
    NlXNbBsNhTwoHs,
    #[serde(rename = "NL_X_NB_NH_BS_CS")]
    NlXNbNhBsCs,
    #[serde(rename = "NL_X_NB_BS_NH_CS")]
    NlXNbBsNhCs,
    #[serde(rename = "NL_X_NB_BSV_BSS")]
    NlXNbBsvBss,
    #[serde(rename = "NL_X_TWO_NB_NH_ONE_BS_HS")]
    NlXTwoNbNhOneBsHs,
}

impl EngineKvFormat {
    fn code(self) -> u8 {
        match self {
            Self::NbNlTwoBsNhHs => 0,
            Self::NlXTwoNbBsNhHs => 1,
            Self::NlXNbTwoBsNhHs => 2,
            Self::NlXNbBsHs => 3,
            Self::TwoXNlXNbbsNhHs => 4,
            Self::NlXNbbsOneHs => 5,
            Self::NlXTwoNbNhBsHs => 6,
            Self::NlXNbTwoNhBsHs => 7,
            Self::NbNlTwoNhBsHs => 8,
            Self::TwoXNlXNbBsNhHs => 9,
            Self::NlXNbNhBsTwoHs => 10,
            Self::NlXNbBsNhTwoHs => 11,
            Self::NlXNbNhBsCs => 12,
            Self::NlXNbBsNhCs => 13,
            Self::NlXNbBsvBss => 14,
            Self::NlXTwoNbNhOneBsHs => 15,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct PhysicalTemplate {
    pub(super) vllm_world_size: usize,
    pub(super) tp_size: usize,
    pub(super) pp_size: usize,
    pub(super) dp_size: usize,
    pub(super) n_servers: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Physical {
    pub(super) vllm_world_size: usize,
    pub(super) vllm_worker_id: usize,
    pub(super) tp_size: usize,
    pub(super) pp_size: usize,
    pub(super) dp_size: usize,
    pub(super) n_servers: usize,
}

impl Physical {
    fn topology(&self) -> PhysicalTemplate {
        PhysicalTemplate {
            vllm_world_size: self.vllm_world_size,
            tp_size: self.tp_size,
            pp_size: self.pp_size,
            dp_size: self.dp_size,
            n_servers: self.n_servers,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct ModelFlags {
    pub(super) mla_enabled: bool,
    pub(super) is_hybrid: bool,
    pub(super) mla_only: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct CachePartition {
    pub(super) world_size: usize,
    pub(super) rank: usize,
    pub(super) kv_tp_size: usize,
    pub(super) is_writer: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct RegisteredLayer {
    pub(super) tensor_index: usize,
    pub(super) name: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct WireEngineGroup {
    pub(super) wire_group_index: usize,
    pub(super) engine_group_id: usize,
    pub(super) layer_indices: Vec<usize>,
    pub(super) tokens_per_block: u64,
    pub(super) sw_size_tokens: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct WorkerRegistration {
    pub(super) schema: Schema,
    pub(super) engine_instance: u64,
    pub(super) model: String,
    pub(super) engine_type: EngineType,
    #[serde(deserialize_with = "crate::wire::object")]
    pub(super) physical: Physical,
    #[serde(deserialize_with = "crate::wire::object")]
    pub(super) model_flags: ModelFlags,
    #[serde(deserialize_with = "crate::wire::object")]
    pub(super) cache_partition: CachePartition,
    pub(super) server_slot: usize,
    #[serde(deserialize_with = "crate::wire::object")]
    pub(super) layout_hints: LayoutHints,
    #[serde(deserialize_with = "crate::wire::objects")]
    pub(super) registered_layers: Vec<RegisteredLayer>,
    #[serde(deserialize_with = "crate::wire::objects")]
    pub(super) wire_engine_groups: Vec<WireEngineGroup>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct WorkerTemplate {
    pub(super) model: String,
    pub(super) engine_type: EngineType,
    #[serde(deserialize_with = "crate::wire::object")]
    pub(super) layout_hints: LayoutHints,
    #[serde(deserialize_with = "crate::wire::objects")]
    pub(super) registered_layers: Vec<RegisteredLayer>,
    #[serde(deserialize_with = "crate::wire::objects")]
    pub(super) wire_engine_groups: Vec<WireEngineGroup>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct ShapeTemplate {
    pub(super) kv_size: u64,
    pub(super) nl: u64,
    pub(super) bs: u64,
    pub(super) nh: u64,
    pub(super) hs: u64,
    pub(super) element_size: u64,
    pub(super) block_stride_elems: u64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Shape {
    pub(super) kv_size: u64,
    pub(super) nl: u64,
    pub(super) nb: u64,
    pub(super) bs: u64,
    pub(super) nh: u64,
    pub(super) hs: u64,
    pub(super) element_size: u64,
    pub(super) block_stride_elems: u64,
}

impl ShapeTemplate {
    fn validate(&self) -> Result<()> {
        for n in [
            self.kv_size,
            self.nl,
            self.bs,
            self.nh,
            self.hs,
            self.element_size,
        ] {
            bound(n, 1, MAX_DIM, "shape dimension")?;
        }
        bound(self.block_stride_elems, 0, MAX_SIGNED, "block stride")
    }
}

impl Shape {
    fn identity(&self) -> ShapeTemplate {
        ShapeTemplate {
            kv_size: self.kv_size,
            nl: self.nl,
            bs: self.bs,
            nh: self.nh,
            hs: self.hs,
            element_size: self.element_size,
            block_stride_elems: self.block_stride_elems,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields, bound(deserialize = "S: Deserialize<'de>"))]
pub(super) struct KernelGroup<S> {
    pub(super) kernel_group_id: usize,
    pub(super) engine_group_id: usize,
    pub(super) object_group_id: usize,
    pub(super) layer_indices: Vec<usize>,
    #[serde(deserialize_with = "crate::wire::object")]
    pub(super) shape: S,
    pub(super) dtype: Dtype,
    pub(super) engine_kv_format_code: u8,
    pub(super) engine_kv_format: EngineKvFormat,
    pub(super) tokens_per_block: u64,
    pub(super) sw_size_tokens: i64,
    pub(super) object_shape: [u64; 4],
    pub(super) allocation_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct ObjectGroup {
    pub(super) object_group_id: usize,
    pub(super) kernel_group_ids: Vec<usize>,
    pub(super) num_chunks_in_sw: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct ServerPolicy {
    pub(super) separate_object_groups: bool,
    pub(super) full_sw_kv: bool,
}

// Declare the shared wire fields once without flattening either closed object.
macro_rules! cache_record {
    ($name:ident, $shape:ty, $( $extra:ident : $ty:ty ),* $(,)?) => {
        #[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
        #[serde(deny_unknown_fields)]
        pub(super) struct $name {
            $(pub(super) $extra: $ty,)*
            pub(super) model: String,
            pub(super) engine_type: EngineType,
            pub(super) cache_world_size: usize,
            #[serde(deserialize_with = "crate::wire::object")]
            pub(super) layout_hints: LayoutHints,
            #[serde(deserialize_with = "crate::wire::object")]
            pub(super) server_policy: ServerPolicy,
            pub(super) registered_tensor_count: usize,
            pub(super) excluded_tensor_indices: Vec<usize>,
            #[serde(deserialize_with = "crate::wire::objects")]
            pub(super) wire_engine_groups: Vec<WireEngineGroup>,
            pub(super) lmcache_tokens_per_chunk: u64,
            #[serde(deserialize_with = "crate::wire::objects")]
            pub(super) kernel_groups: Vec<KernelGroup<$shape>>,
            #[serde(deserialize_with = "crate::wire::objects")]
            pub(super) object_groups: Vec<ObjectGroup>,
        }
    };
}

cache_record!(CacheRegistration, Shape, schema: Schema, registration_kind: RegistrationKind, engine_instance: u64);
cache_record!(CacheTemplate, ShapeTemplate,);

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct RegisteredStateTemplate {
    pub(super) schema: Schema,
    #[serde(deserialize_with = "crate::wire::object")]
    pub(super) physical_template: PhysicalTemplate,
    #[serde(deserialize_with = "crate::wire::object")]
    pub(super) model_flags: ModelFlags,
    #[serde(deserialize_with = "crate::wire::object")]
    pub(super) worker_template: WorkerTemplate,
    #[serde(deserialize_with = "crate::wire::object")]
    pub(super) cache_template: CacheTemplate,
}

pub(super) struct WorkerRecord<'a> {
    pub(super) process: &'a super::ProcessIdentity,
    pub(super) incarnation: &'a str,
    pub(super) registration: &'a WorkerRegistration,
}

pub(super) struct CacheJournal<'a> {
    pub(super) process: &'a super::ProcessIdentity,
    pub(super) incarnation: &'a str,
    pub(super) registrations: &'a [&'a CacheRegistration],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct RankRegistration {
    pub(super) engine_instance: u64,
    pub(super) cache_journal: usize,
    pub(super) cache_rank: usize,
    pub(super) is_writer: bool,
    pub(super) kernel_capacities: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct RegistryRoster {
    pub(super) old: BTreeMap<usize, RankRegistration>,
    pub(super) reload: BTreeMap<usize, RankRegistration>,
    pub(super) cache_slots: BTreeMap<usize, usize>,
}

fn require(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(format!("registered state: {message}"))
    }
}

fn bound(value: u64, lower: u64, upper: u64, field: &str) -> Result<()> {
    if (lower..=upper).contains(&value) {
        Ok(())
    } else {
        Err(format!(
            "registered state: {field} outside {lower}..={upper}"
        ))
    }
}

fn text(value: &str) -> Result<()> {
    require(
        !value.is_empty() && value.len() <= 256,
        "name must contain 1..=256 UTF-8 bytes",
    )
}

fn product(values: &[u64]) -> Result<u64> {
    values.iter().try_fold(1_u64, |n, value| {
        n.checked_mul(*value)
            .ok_or_else(|| "registered state: product overflow".into())
    })
}

fn window(value: i64, upper: u64) -> Result<()> {
    require(
        value == -1 || (value > 0 && value as u64 <= upper),
        "invalid window",
    )
}

fn topology(p: &PhysicalTemplate, flags: &ModelFlags) -> Result<()> {
    for n in [
        p.vllm_world_size,
        p.tp_size,
        p.pp_size,
        p.dp_size,
        p.n_servers,
    ] {
        bound(n as u64, 1, 64, "physical size")?;
    }
    require(
        p.tp_size.checked_mul(p.pp_size) == Some(p.vllm_world_size)
            && p.vllm_world_size.is_multiple_of(p.n_servers)
            && p.tp_size.is_multiple_of(p.n_servers)
            && p.n_servers <= p.vllm_world_size
            && (p.n_servers == 1 || (p.dp_size == 1 && p.pp_size == 1)),
        "unsupported physical topology",
    )?;
    require(
        flags.mla_only == (flags.mla_enabled && !flags.is_hybrid),
        "MLA flags disagree",
    )
}

fn partition(p: &PhysicalTemplate, flags: &ModelFlags, rank: usize) -> CachePartition {
    let local_world = p.vllm_world_size / p.n_servers;
    let kv_tp_size = p.tp_size / p.n_servers;
    CachePartition {
        world_size: if flags.mla_only {
            p.vllm_world_size / p.tp_size
        } else {
            local_world
        },
        rank: if flags.mla_only {
            rank / p.tp_size
        } else {
            rank % local_world
        },
        kv_tp_size,
        is_writer: !flags.mla_only || rank.is_multiple_of(kv_tp_size),
    }
}

fn layers_valid(layers: &[RegisteredLayer]) -> Result<()> {
    bound(layers.len() as u64, 1, 4096, "registered tensor count")?;
    let mut names = BTreeSet::new();
    for (index, layer) in layers.iter().enumerate() {
        text(&layer.name)?;
        require(
            layer.tensor_index == index && names.insert(layer.name.as_str()),
            "layer indices or names are not unique and ordered",
        )?;
    }
    Ok(())
}

fn indices_valid(indices: &[usize], count: usize, nonempty: bool) -> Result<()> {
    require(
        (!nonempty || !indices.is_empty())
            && indices.len() <= count
            && indices.iter().all(|&i| i < count)
            && indices.windows(2).all(|pair| pair[0] < pair[1]),
        "invalid ordered group indices",
    )
}

fn wires_valid(wires: &[WireEngineGroup], count: usize) -> Result<BTreeSet<usize>> {
    bound(wires.len() as u64, 1, 64, "wire group count")?;
    let mut used = BTreeSet::new();
    let mut spaces = BTreeSet::new();
    for (index, wire) in wires.iter().enumerate() {
        require(
            wire.wire_group_index == index && wire.engine_group_id < 64,
            "wire or engine group ID outside domain",
        )?;
        indices_valid(&wire.layer_indices, count, true)?;
        for &layer in &wire.layer_indices {
            require(used.insert(layer), "layer belongs to multiple wire groups")?;
        }
        spaces.insert(wire.engine_group_id);
        bound(wire.tokens_per_block, 1, MAX_TOKENS, "tokens per block")?;
        window(wire.sw_size_tokens, MAX_TOKENS)?;
    }
    require(
        spaces.iter().copied().eq(0..spaces.len()),
        "engine group IDs are not dense",
    )?;
    Ok(used)
}

pub(super) fn validate_worker(w: &WorkerRegistration) -> Result<()> {
    bound(w.engine_instance, 0, MAX_SIGNED, "engine instance")?;
    text(&w.model)?;
    let p = w.physical.topology();
    topology(&p, &w.model_flags)?;
    let rank = w.physical.vllm_worker_id;
    require(rank < p.vllm_world_size, "physical rank outside world")?;
    require(
        w.cache_partition == partition(&p, &w.model_flags, rank),
        "cache partition or writer disagrees with physical strategy",
    )?;
    require(
        w.server_slot == rank / (p.vllm_world_size / p.n_servers),
        "server slot disagrees with source routing",
    )?;
    layers_valid(&w.registered_layers)?;
    wires_valid(&w.wire_engine_groups, w.registered_layers.len())?;
    Ok(())
}

fn validate_layout(c: &CacheTemplate) -> Result<()> {
    text(&c.model)?;
    bound(c.cache_world_size as u64, 1, 64, "cache world")?;
    bound(
        c.registered_tensor_count as u64,
        1,
        4096,
        "registered tensor count",
    )?;
    bound(
        c.lmcache_tokens_per_chunk,
        1,
        MAX_TOKENS,
        "tokens per chunk",
    )?;
    require(
        !c.server_policy.full_sw_kv,
        "full_sw_kv is outside VLLM scope",
    )?;
    let mut used = wires_valid(&c.wire_engine_groups, c.registered_tensor_count)?;
    indices_valid(&c.excluded_tensor_indices, c.registered_tensor_count, false)?;
    for &index in &c.excluded_tensor_indices {
        require(
            used.insert(index),
            "excluded tensor is also in a wire group",
        )?;
    }
    require(
        used.len() == c.registered_tensor_count,
        "tensor coverage is incomplete",
    )?;
    require(
        c.kernel_groups.len() == c.wire_engine_groups.len(),
        "wire/kernel count mismatch",
    )?;
    bound(c.object_groups.len() as u64, 1, 64, "object group count")?;
    let chunk = c.lmcache_tokens_per_chunk;
    let mut expected_objects: Vec<ObjectGroup> = Vec::new();
    for (kid, kernel) in c.kernel_groups.iter().enumerate() {
        let wire = &c.wire_engine_groups[kid];
        require(
            kernel.kernel_group_id == kid
                && kernel.engine_group_id == wire.engine_group_id
                && kernel.layer_indices == wire.layer_indices
                && kernel.tokens_per_block == wire.tokens_per_block
                && kernel.sw_size_tokens == wire.sw_size_tokens,
            "wire/kernel identity mismatch",
        )?;
        kernel.shape.validate()?;
        let s = &kernel.shape;
        require(
            s.nl == kernel.layer_indices.len() as u64 && s.element_size == kernel.dtype.itemsize(),
            "layer count or exact dtype width mismatch",
        )?;
        require(
            kernel.engine_kv_format_code == kernel.engine_kv_format.code(),
            "format code/name mismatch",
        )?;
        let tokens = kernel.tokens_per_block;
        let sw = kernel.sw_size_tokens;
        require(
            tokens % s.bs == 0
                && chunk.is_multiple_of(tokens)
                && !(sw > 0 && (sw as u64) < chunk && !(sw as u64).is_multiple_of(tokens)),
            "logical/physical block or window alignment mismatch",
        )?;
        let retained = if sw == -1 || sw as u64 >= chunk {
            chunk
        } else {
            sw as u64
        };
        let expected_shape = [
            s.kv_size,
            s.nl,
            product(&[retained, s.bs])? / tokens,
            product(&[s.nh, s.hs])?,
        ];
        for &dim in &kernel.object_shape {
            bound(dim, 1, MAX_DIM, "object dimension")?;
        }
        require(
            kernel.object_shape == expected_shape,
            "object shape differs from registered allocation formula",
        )?;
        let bytes = product(&[
            kernel.object_shape[0],
            kernel.object_shape[1],
            kernel.object_shape[2],
            kernel.object_shape[3],
            kernel.dtype.itemsize(),
        ])?;
        bound(bytes, 1, MAX_SIGNED, "allocation bytes")?;
        require(
            kernel.allocation_bytes == bytes,
            "allocation bytes mismatch",
        )?;
        let class = if !c.server_policy.separate_object_groups || sw == -1 {
            -1
        } else {
            let numerator = (sw as u64)
                .checked_add(chunk - 1)
                .ok_or("registered state: window sum overflow")?;
            (numerator / chunk) as i64
        };
        window(class, 1024)?;
        let oid = match expected_objects
            .iter()
            .position(|obj| obj.num_chunks_in_sw == class)
        {
            Some(oid) => oid,
            None => {
                let oid = expected_objects.len();
                expected_objects.push(ObjectGroup {
                    object_group_id: oid,
                    kernel_group_ids: Vec::new(),
                    num_chunks_in_sw: class,
                });
                oid
            }
        };
        require(
            kernel.object_group_id == oid,
            "kernel object owner differs from window partition",
        )?;
        expected_objects[oid].kernel_group_ids.push(kid);
    }
    require(
        c.object_groups == expected_objects,
        "object membership, order or window policy mismatch",
    )?;
    for object in &c.object_groups {
        let total = object.kernel_group_ids.iter().try_fold(0_u64, |n, &kid| {
            n.checked_add(c.kernel_groups[kid].allocation_bytes)
                .ok_or("registered state: object allocation sum overflow")
        })?;
        bound(total, 1, MAX_SIGNED, "object allocation total")?;
    }
    Ok(())
}

pub(super) fn validate_template(t: &RegisteredStateTemplate) -> Result<()> {
    topology(&t.physical_template, &t.model_flags)?;
    let w = &t.worker_template;
    text(&w.model)?;
    layers_valid(&w.registered_layers)?;
    wires_valid(&w.wire_engine_groups, w.registered_layers.len())?;
    let c = &t.cache_template;
    validate_layout(c)?;
    require(
        w.model == c.model
            && w.engine_type == c.engine_type
            && w.layout_hints == c.layout_hints
            && w.wire_engine_groups == c.wire_engine_groups
            && w.registered_layers.len() == c.registered_tensor_count
            && c.cache_world_size == partition(&t.physical_template, &t.model_flags, 0).world_size,
        "worker/cache prospective templates disagree",
    )
}

fn layout_matches(a: &CacheRegistration, b: &CacheTemplate) -> bool {
    a.model == b.model
        && a.engine_type == b.engine_type
        && a.cache_world_size == b.cache_world_size
        && a.layout_hints == b.layout_hints
        && a.server_policy == b.server_policy
        && a.registered_tensor_count == b.registered_tensor_count
        && a.excluded_tensor_indices == b.excluded_tensor_indices
        && a.wire_engine_groups == b.wire_engine_groups
        && a.lmcache_tokens_per_chunk == b.lmcache_tokens_per_chunk
        && a.object_groups == b.object_groups
        && a.kernel_groups.len() == b.kernel_groups.len()
        && a.kernel_groups.iter().zip(&b.kernel_groups).all(|(x, y)| {
            x.kernel_group_id == y.kernel_group_id
                && x.engine_group_id == y.engine_group_id
                && x.object_group_id == y.object_group_id
                && x.layer_indices == y.layer_indices
                && x.shape.identity() == y.shape
                && x.dtype == y.dtype
                && x.engine_kv_format_code == y.engine_kv_format_code
                && x.engine_kv_format == y.engine_kv_format
                && x.tokens_per_block == y.tokens_per_block
                && x.sw_size_tokens == y.sw_size_tokens
                && x.object_shape == y.object_shape
                && x.allocation_bytes == y.allocation_bytes
        })
}

pub(super) fn verify_rosters(
    template: &RegisteredStateTemplate,
    old: &[WorkerRecord<'_>],
    reload: &[WorkerRecord<'_>],
    caches: &[CacheJournal<'_>],
) -> Result<RegistryRoster> {
    validate_template(template)?;
    let p = &template.physical_template;
    require(
        old.len() == p.vllm_world_size && reload.len() == p.vllm_world_size,
        "both phases require every physical worker, including nonwriters",
    )?;
    require(
        caches.len() == p.n_servers,
        "cache journal count differs from server count",
    )?;
    let mut processes = BTreeSet::new();
    let mut incarnations = BTreeSet::new();
    let mut registrations = BTreeMap::new();
    for (journal, cache) in caches.iter().enumerate() {
        require(
            processes.insert((
                cache.process.pid,
                cache.process.start_ticks,
                cache.process.boot_id.as_str(),
            )) && incarnations.insert(cache.incarnation),
            "duplicate cache process or incarnation",
        )?;
        require(!cache.registrations.is_empty(), "empty cache journal")?;
        for &registration in cache.registrations {
            // Every layout field must equal the already validated template;
            // only runtime engine identity and block capacities are additional.
            bound(
                registration.engine_instance,
                0,
                MAX_SIGNED,
                "engine instance",
            )?;
            for kernel in &registration.kernel_groups {
                bound(kernel.shape.nb, 1, MAX_DIM, "registered block capacity")?;
            }
            require(
                layout_matches(registration, &template.cache_template),
                "cache registration differs from prospective layout",
            )?;
            require(
                registrations
                    .insert(registration.engine_instance, (journal, registration))
                    .is_none(),
                "duplicate cache engine registration",
            )?;
        }
    }
    let mut instances = BTreeSet::new();
    let mut cache_slots = BTreeMap::new();
    let mut phases = [BTreeMap::new(), BTreeMap::new()];
    for (phase, workers) in [old, reload].into_iter().enumerate() {
        for record in workers {
            let w = record.registration;
            validate_worker(w)?;
            require(
                processes.insert((
                    record.process.pid,
                    record.process.start_ticks,
                    record.process.boot_id.as_str(),
                )) && incarnations.insert(record.incarnation),
                "worker process or incarnation overlaps another worker phase or cache",
            )?;
            require(
                instances.insert(w.engine_instance),
                "worker engine instance overlaps within or across phases",
            )?;
            let expected = &template.worker_template;
            require(
                w.physical.topology() == *p
                    && w.model_flags == template.model_flags
                    && w.model == expected.model
                    && w.engine_type == expected.engine_type
                    && w.layout_hints == expected.layout_hints
                    && w.registered_layers == expected.registered_layers
                    && w.wire_engine_groups == expected.wire_engine_groups,
                "worker differs from prospective registered state",
            )?;
            let &(journal, c) = registrations
                .get(&w.engine_instance)
                .ok_or("registered state: worker has no cache registration")?;
            require(
                w.model == c.model
                    && w.engine_type == c.engine_type
                    && w.cache_partition.world_size == c.cache_world_size
                    && w.layout_hints == c.layout_hints
                    && w.wire_engine_groups == c.wire_engine_groups
                    && w.registered_layers.len() == c.registered_tensor_count,
                "worker/cache engine registration join mismatch",
            )?;
            if let Some(previous) = cache_slots.insert(w.server_slot, journal) {
                require(
                    previous == journal,
                    "server slot changed cache journal across workers or restart",
                )?;
            }
            let rank = RankRegistration {
                engine_instance: w.engine_instance,
                cache_journal: journal,
                cache_rank: w.cache_partition.rank,
                is_writer: w.cache_partition.is_writer,
                kernel_capacities: c
                    .kernel_groups
                    .iter()
                    .map(|kernel| kernel.shape.nb)
                    .collect(),
            };
            require(
                phases[phase]
                    .insert(w.physical.vllm_worker_id, rank)
                    .is_none(),
                "duplicate physical rank",
            )?;
        }
        require(
            phases[phase].keys().copied().eq(0..p.vllm_world_size),
            "physical rank coverage is not exact",
        )?;
    }
    require(
        registrations.len() == instances.len(),
        "unrelated cache registration",
    )?;
    require(
        cache_slots.keys().copied().eq(0..p.n_servers)
            && cache_slots
                .values()
                .copied()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .eq(0..caches.len()),
        "server slots and cache journals are not a bijection",
    )?;
    let [old, reload] = phases;
    Ok(RegistryRoster {
        old,
        reload,
        cache_slots,
    })
}
