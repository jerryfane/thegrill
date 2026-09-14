//! Kernel and collective microbenchmark evidence (#54).
//!
//! HTTP output speed cannot prove that an isolated kernel or collective
//! changed, so this module keeps kernel/fabric observations in their own
//! closed artifact family. Three provenance states are distinct and never
//! silently upgraded:
//!
//! * `declared` — an operator statement with no measurement bytes. A declared
//!   artifact carries no samples and is rejected as measurement evidence.
//! * `imported` — bytes this collector parsed but did not produce. Importing
//!   always records `imported`, even when the payload claims native execution,
//!   and the payload's own claim is retained separately for audit.
//! * `native_observed` — bytes produced by a program this collector launched,
//!   or by the in-process CPU reference below. This records *observation*,
//!   never authenticated execution: the collector cannot attest what a child
//!   process actually did.
//!
//! Only one adapter executes inside this binary: the deterministic CPU
//! `sum_u64` reference over the vector `0..bound`. It proves operation
//! recording, inspection and exact comparison, and it is the only adapter that
//! may be reported as native-observed under the current CPU-only
//! authorization. The EXL3 E3 grouped expert adapter and the NCCL all-reduce
//! adapter are explicitly launched programs; their device execution is a
//! separately authorized window and their artifacts stay unexercised until
//! then.
//!
//! Grounding for the two shipped device adapters, pinned before
//! implementation (hashes recorded in this file, in the plans shipped under
//! `crates/grill-perf/examples/` and in `docs/performance/KERNEL-FABRIC.md`):
//!
//! * `MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks` at
//!   `f906ee990596486e10ddbe381efa6f0e496f77e3`:
//!   `tests/bench_e3_microbench.py` (operation and synthetic-routing source
//!   anchor) and `tests/test_exl3_overlay.py` (frozen parity tolerances).
//! * `NVIDIA/nccl-tests` `doc/PERFORMANCE.md` for the all-reduce byte
//!   definitions used by the reported bandwidth labels.
//!
//! Algorithmic payload bytes are `numel * sizeof(dtype)`. The factor
//! `2*(world-1)/world` is the nccl-tests *bus bandwidth* normalization for
//! all-reduce; it is labelled separately and is not measured physical link
//! traffic. A kernel or collective result never sets a serving-speed claim.
//!
//! Clock discipline: every artifact carries exactly one clock identity, kind,
//! unit and synchronization contract, and every duration is retained as its
//! bounded exact decimal text plus the exact nanosecond integer converted from
//! it. Digits finer than the declared timer resolution are rejected rather
//! than silently rounded. Durations from different clock identities are never
//! subtracted; for rank scope only complete per-repetition maxima are used.

use crate::envelope::{self, Direction, EnvelopeDecision, EnvelopeReason, Rational};
use crate::evidence;
use crate::model::{FILE_CAP, Result};
use crate::policy::Outcome;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub const KIND: &str = "microbench-artifact-v1";
pub const PLAN_KIND: &str = "microbench-plan-v1";
pub const RECEIPT_KIND: &str = "microbench-receipt-v1";
pub const VERSION: u32 = 1;

/// Bounded operator program output; a larger artifact is rejected.
pub const OUTPUT_CAP: usize = 1024 * 1024;
/// Bounded retained stderr for a failed operator program.
pub const STDERR_CAP: usize = 64 * 1024;
/// Fractional digits accepted in any retained decimal value.
pub const DECIMALS: usize = 9;
/// Largest accepted single duration (~13 days) keeps exact arithmetic bounded.
pub const MAX_DURATION_NS: u64 = 1 << 50;
/// Largest accepted rank count for a declared collective.
pub const MAX_RANKS: u32 = 64;

/// Frozen revision of the pinned public E3 sources.
pub const E3_SOURCE_REVISION: &str = "f906ee990596486e10ddbe381efa6f0e496f77e3";
/// `tests/bench_e3_microbench.py` at that revision.
pub const E3_OPERATION_SOURCE_SHA256: &str =
    "f57e4fa6726b0110c44ce56b0b0968ba425e566ace6d89f2db3872aae13889ec";
/// `tests/test_exl3_overlay.py` at that revision: `_err_stats`,
/// `_assert_e3_within` and the frozen `E3_TOL_*` constants.
pub const E3_PARITY_SOURCE_SHA256: &str =
    "8777254d47cdbb89186ba3bd1ac2437cbebc7987e0791d36c68e7bccb3df2b35";
/// `NVIDIA/nccl-tests` `doc/PERFORMANCE.md`, the byte-definition source.
pub const NCCL_BYTE_SOURCE_SHA256: &str =
    "2242493053a6f9d5db1c6542a8ce7740be919778645125b94eeae440efd31edf";

pub const E3_ROUTING_ID: &str = "synthetic-zipf-permuted";
pub const E3_ACTIVATION_ID: &str = "randn-fp16-activation-seed-3";
pub const E3_WEIGHTS_ID: &str = "random-k4-trellis-shared-gate-up-suh";
pub const E3_REFERENCE_ID: &str = "apply-exl3-python-loop";
pub const SUM_REFERENCE_ID: &str = "n*(n-1)/2";
pub const REDUCE_INPUT_ID: &str = "input[i]=(i%251)+rank";
pub const REDUCE_REFERENCE_ID: &str = "world*(i%251)+world*(world-1)/2";

/// Frozen E3 tolerances from the pinned parity source: factor 1.5,
/// absolute-relative floor 1e-3, nRMSE floor 1e-4, coarse bound
/// `max(0.15, 0.08 * max(1.0, ref_max))`.
pub const E3_TOLERANCE: Tolerance = Tolerance::E3Rel {
    factor_milli: 1500,
    abs_rel_micro: 1000,
    nrmse_abs_micro: 100,
    coarse_abs_milli: 150,
    coarse_factor_milli: 80,
};

fn push(list: &mut Vec<Reason>, reason: Reason) {
    if !list.contains(&reason) {
        list.push(reason);
    }
}

/// Closed adapter set. No registry, discovery or script hook: a plan names one
/// adapter and, for the two external adapters, one explicitly declared program.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum AdapterId {
    #[serde(rename = "cpu-sum-u64-reference")]
    #[value(name = "cpu-sum-u64-reference")]
    CpuSumU64Reference,
    #[serde(rename = "exl3-e3-grouped")]
    #[value(name = "exl3-e3-grouped")]
    Exl3E3Grouped,
    #[serde(rename = "nccl-allreduce-sum")]
    #[value(name = "nccl-allreduce-sum")]
    NcclAllreduceSum,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    Cpu,
    Device,
    Ranks,
}

impl AdapterId {
    pub fn name(self) -> &'static str {
        match self {
            Self::CpuSumU64Reference => "cpu-sum-u64-reference",
            Self::Exl3E3Grouped => "exl3-e3-grouped",
            Self::NcclAllreduceSum => "nccl-allreduce-sum",
        }
    }

    pub fn scope(self) -> Scope {
        match self {
            Self::CpuSumU64Reference => Scope::Cpu,
            Self::Exl3E3Grouped => Scope::Device,
            Self::NcclAllreduceSum => Scope::Ranks,
        }
    }

    /// Device and collective adapters are always operator programs; the CPU
    /// reference is executed in-process.
    fn external(self) -> bool {
        !matches!(self, Self::CpuSumU64Reference)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Provenance {
    Declared,
    Imported,
    NativeObserved,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ClockKind {
    ProcessMonotonic,
    DeviceEvent,
    HostMonotonic,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Units {
    Nanoseconds,
    Milliseconds,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Synchronization {
    /// Reading a monotonic clock immediately before and after the operation.
    ProcessClockReadBeforeAfter,
    /// CUDA events recorded around the operation with a device synchronize
    /// before reading each elapsed time.
    DeviceEventTimingAfterSynchronize,
    /// A completed collective barrier immediately before and after the timed
    /// collective, so the recorded duration spans the synchronized operation.
    CollectiveBarrierBeforeAfter,
}

impl Units {
    fn nanoseconds(self) -> u64 {
        match self {
            Self::Nanoseconds => 1,
            Self::Milliseconds => 1_000_000,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Nanoseconds => "nanoseconds",
            Self::Milliseconds => "milliseconds",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Dtype {
    U64,
    Int64,
    Float16,
}

impl Dtype {
    fn bytes(self) -> u64 {
        match self {
            Self::U64 | Self::Int64 => 8,
            Self::Float16 => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Tier {
    E2Kernel,
    E3Grouped,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReduceOp {
    Sum,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Clock {
    pub id: String,
    pub kind: ClockKind,
    pub units: Units,
    pub resolution_ns: u64,
    pub synchronization: Synchronization,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Source {
    pub path: String,
    pub sha256: String,
}

/// The declared, fixable operation identity. Only the frozen cells below are
/// admitted: no shape, cap, skew or tier search exists.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum Operation {
    SumU64 {
        bound: u32,
        dtype: Dtype,
    },
    Exl3Experts {
        hidden: u32,
        intermediate: u32,
        experts: u32,
        topk: u32,
        tokens: u32,
        layer_seed: u64,
        routing_seed: u64,
        parity_routing_seed: u64,
        activation_seed: u64,
        skew_milli: u32,
        tier: Tier,
        cap: u32,
        dtype: Dtype,
        routing: String,
        activation: String,
        weights: String,
    },
    AllReduce {
        op: ReduceOp,
        dtype: Dtype,
        numel: u32,
        world: u32,
        ranks: Vec<u32>,
        input: String,
        reference: String,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum Tolerance {
    ExactU64,
    ExactInt64,
    E3Rel {
        factor_milli: u32,
        abs_rel_micro: u32,
        nrmse_abs_micro: u32,
        coarse_abs_milli: u32,
        coarse_factor_milli: u32,
    },
}

impl Tolerance {
    fn factor_milli(self) -> u64 {
        match self {
            Self::E3Rel { factor_milli, .. } => u64::from(factor_milli),
            _ => 0,
        }
    }

    fn abs_rel_micro(self) -> u64 {
        match self {
            Self::E3Rel { abs_rel_micro, .. } => u64::from(abs_rel_micro),
            _ => 0,
        }
    }

    fn nrmse_abs_micro(self) -> u64 {
        match self {
            Self::E3Rel { nrmse_abs_micro, .. } => u64::from(nrmse_abs_micro),
            _ => 0,
        }
    }

    fn coarse_abs_milli(self) -> u64 {
        match self {
            Self::E3Rel { coarse_abs_milli, .. } => u64::from(coarse_abs_milli),
            _ => 0,
        }
    }

    fn coarse_factor_milli(self) -> u64 {
        match self {
            Self::E3Rel {
                coarse_factor_milli, ..
            } => u64::from(coarse_factor_milli),
            _ => 0,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CorrectnessContract {
    pub reference: String,
    pub tolerance: Tolerance,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Allowance {
    pub work_units: u64,
    pub memory_bytes: u64,
    pub deadline_ms: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Thresholds {
    pub adverse_bps: u32,
    pub spread_bps: u32,
}

/// Distinct ordered acquisition identity. Roles are compared only when their
/// declared starts are strictly ordered A < B < A2 and their ids differ.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Acquisition {
    pub id: String,
    pub started_unix_ms: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub kind: String,
    pub version: u32,
    pub adapter: AdapterId,
    /// The candidate change axis. Every other pin must match across roles.
    pub revision: String,
    pub program_sha256: Option<String>,
    pub sources: Vec<Source>,
    pub operation: Operation,
    pub clock: Clock,
    pub warmups: u32,
    pub iterations: u32,
    pub allowance: Allowance,
    pub correctness: CorrectnessContract,
    pub thresholds: Option<Thresholds>,
    pub acquisition: Acquisition,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RankRef {
    pub index: u32,
    pub device: Option<u32>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Topology {
    pub scope: Scope,
    pub world: Option<u32>,
    pub ranks: Vec<RankRef>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Execution {
    pub warmups: u32,
    pub iterations: u32,
    pub deadline_ms: u64,
    pub work_units: u64,
    pub memory_bytes: u64,
    pub completed: bool,
    pub timed_out: bool,
    /// The last observed EXL3 fallback tier, when the adapter reports one. It
    /// must equal the declared tier; a silent lower-tier fallback withholds
    /// the measurement.
    pub observed_fallback: Option<Tier>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Sample {
    pub index: u32,
    pub rank: Option<u32>,
    pub repetition: Option<u32>,
    /// Exact retained decimal text as produced by the adapter timer.
    pub raw: String,
    /// Exact nanosecond conversion of `raw`. Never a re-rounded float.
    pub duration_ns: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ParityStats {
    pub maxabs: String,
    pub per_token_max: String,
    pub per_token_p99: String,
    pub nrmse: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CheckOutcome {
    Pass,
    Fail,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum Correctness {
    ExactSum {
        reference: String,
        computed: String,
        bound: String,
        passed: bool,
    },
    ExactReduction {
        reference: String,
        mismatches: u64,
        tolerance_elems: u64,
        finite: bool,
        passed: bool,
    },
    E3Parity {
        reference: String,
        finite: bool,
        ref_max: String,
        e2: ParityStats,
        e3: ParityStats,
        tolerance: Tolerance,
        passed: bool,
        outcome: CheckOutcome,
        detail: Option<String>,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum FailureKind {
    DeadlineExceeded,
    AllowanceExceeded,
    TierFallback,
    RankMissing,
    IncompleteSamples,
    CorrectnessFailed,
    NonFinite,
    AdapterError,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Failure {
    pub kind: FailureKind,
    pub detail: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    pub kind: String,
    pub version: u32,
    pub adapter: AdapterId,
    pub revision: String,
    pub operation: Operation,
    /// Assigned by this collector; never taken from the payload.
    pub provenance: Provenance,
    /// Whatever the payload claimed about itself, retained for audit.
    pub submitted_provenance: Option<Provenance>,
    pub program_sha256: Option<String>,
    pub kernel_revision: Option<String>,
    pub sources: Vec<Source>,
    pub topology: Topology,
    pub clock: Clock,
    pub execution: Execution,
    pub samples: Vec<Sample>,
    pub correctness: Correctness,
    pub failures: Vec<Failure>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReceiptStatus {
    Captured,
    Imported,
    Failed,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Receipt {
    pub kind: String,
    pub version: u32,
    pub adapter: AdapterId,
    pub provenance: Provenance,
    pub status: ReceiptStatus,
    pub plan_sha256: String,
    pub artifact_sha256: Option<String>,
    pub collector_sha256: String,
    pub program: Option<String>,
    pub program_sha256: Option<String>,
    pub source_sha256: Option<String>,
    pub started_unix_ms: u64,
    pub observation_unix_ms: u64,
    pub duration_ms: u64,
    pub command: Vec<String>,
    pub failure: Option<String>,
    pub claim: String,
}

/// Domain-owned rejection codes. Precedence and exits are the existing
/// `policy::Outcome` order: ERROR(1) invalid or corrupt, INCONCLUSIVE(2)
/// retained but unavailable, REGRESSION(3), PASS(0).
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Reason {
    InvalidPlan,
    InvalidArtifact,
    MissingArtifact,
    CaptureFailed,
    AdapterMismatch,
    IdentityDrift,
    ClockMismatch,
    UnsynchronizedClock,
    IterationCountMismatch,
    SampleGridIncomplete,
    SamplePrecision,
    DurationOutOfRange,
    NonFinite,
    CorrectnessFailed,
    RetainedFailure,
    DeadlineExceeded,
    AllowanceExceeded,
    TierFallback,
    MissingThresholds,
    MissingReference,
    RoleReuse,
    DeclaredStartsOutOfOrder,
    StatisticUnavailable,
    InvalidRational,
    InvalidBounds,
    ArithmeticOverflow,
    NonpositiveReference,
    ReferenceSpreadExceeded,
    EnvelopeStraddlesTolerance,
}

fn envelope_reason(reason: EnvelopeReason) -> Reason {
    match reason {
        EnvelopeReason::InvalidRational => Reason::InvalidRational,
        EnvelopeReason::InvalidBounds => Reason::InvalidBounds,
        EnvelopeReason::ArithmeticOverflow { .. } => Reason::ArithmeticOverflow,
        EnvelopeReason::NonpositiveReference => Reason::NonpositiveReference,
        EnvelopeReason::ReferenceSpreadExceeded => Reason::ReferenceSpreadExceeded,
        EnvelopeReason::EnvelopeStraddlesTolerance { .. } => Reason::EnvelopeStraddlesTolerance,
    }
}

/// Bounded exact decimal-to-rational conversion. Rejects signs, exponents,
/// empty integer parts, more than [`DECIMALS`] fractional digits and values
/// beyond the bounded magnitude.
pub fn exact_decimal(text: &str) -> Option<Rational> {
    if text.is_empty() || text.len() > 32 {
        return None;
    }
    let (whole, fraction) = match text.split_once('.') {
        Some((whole, fraction)) => (whole, fraction),
        None => (text, ""),
    };
    if whole.is_empty() || !whole.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    if fraction.len() > DECIMALS || !fraction.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let whole: u64 = whole.parse().ok()?;
    let scale = 10u64.checked_pow(fraction.len() as u32)?;
    let fraction: u64 = if fraction.is_empty() {
        0
    } else {
        fraction.parse().ok()?
    };
    Some(Rational {
        numerator: whole.checked_mul(scale)?.checked_add(fraction)?,
        denominator: scale,
    })
}

/// Exact nanosecond conversion of a retained decimal in the declared units.
/// A value finer than one nanosecond is rejected instead of rounded.
pub fn duration_ns(text: &str, units: Units) -> Option<u64> {
    let value = exact_decimal(text)?;
    let scaled = value.numerator.checked_mul(units.nanoseconds())?;
    if value.denominator == 0 || scaled % value.denominator != 0 {
        return None;
    }
    let nanoseconds = scaled / value.denominator;
    (nanoseconds <= MAX_DURATION_NS).then_some(nanoseconds)
}

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let next = a % b;
        a = b;
        b = next;
    }
    a
}

fn reduce(numerator: u128, denominator: u128) -> Option<Rational> {
    if denominator == 0 {
        return None;
    }
    let divisor = gcd(numerator, denominator).max(1);
    Some(Rational {
        numerator: u64::try_from(numerator / divisor).ok()?,
        denominator: u64::try_from(denominator / divisor).ok()?,
    })
}

fn multiply(a: Rational, b: Rational) -> Option<Rational> {
    reduce(
        u128::from(a.numerator) * u128::from(b.numerator),
        u128::from(a.denominator) * u128::from(b.denominator),
    )
}

fn scale(a: Rational, numerator: u64, denominator: u64) -> Option<Rational> {
    multiply(a, Rational { numerator, denominator })
}

fn add(a: Rational, b: Rational) -> Option<Rational> {
    reduce(
        u128::from(a.numerator) * u128::from(b.denominator)
            + u128::from(b.numerator) * u128::from(a.denominator),
        u128::from(a.denominator) * u128::from(b.denominator),
    )
}

fn compare(a: Rational, b: Rational) -> std::cmp::Ordering {
    (u128::from(a.numerator) * u128::from(b.denominator))
        .cmp(&(u128::from(b.numerator) * u128::from(a.denominator)))
}

fn larger(a: Rational, b: Rational) -> Rational {
    if compare(a, b).is_lt() { b } else { a }
}

fn display(value: Rational) -> f64 {
    value.numerator as f64 / value.denominator as f64
}

fn sha256(text: &str) -> bool {
    text.len() == 64
        && text
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn identifier(text: &str) -> bool {
    !text.is_empty()
        && text.len() <= 256
        && text
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.')
}

fn unix_ms() -> Result<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .map_err(|e| format!("system clock before the Unix epoch: {e}"))
}

/// The single admitted cell per adapter. Pinning this in code, not only in a
/// document, is what makes "no cap/skew/tier search" a checked property.
pub fn frozen_operation(adapter: AdapterId) -> Operation {
    match adapter {
        AdapterId::CpuSumU64Reference => Operation::SumU64 {
            bound: 4096,
            dtype: Dtype::U64,
        },
        AdapterId::Exl3E3Grouped => Operation::Exl3Experts {
            hidden: 4096,
            intermediate: 1024,
            experts: 288,
            topk: 8,
            tokens: 1024,
            layer_seed: 0,
            routing_seed: 1,
            parity_routing_seed: 3,
            activation_seed: 3,
            skew_milli: 1000,
            tier: Tier::E3Grouped,
            cap: 32,
            dtype: Dtype::Float16,
            routing: E3_ROUTING_ID.into(),
            activation: E3_ACTIVATION_ID.into(),
            weights: E3_WEIGHTS_ID.into(),
        },
        AdapterId::NcclAllreduceSum => Operation::AllReduce {
            op: ReduceOp::Sum,
            dtype: Dtype::Int64,
            numel: 262_144,
            world: 2,
            ranks: vec![0, 1],
            input: REDUCE_INPUT_ID.into(),
            reference: REDUCE_REFERENCE_ID.into(),
        },
    }
}

/// Admit only the frozen operation identity. The collective's world and rank
/// list are operator-declared and shape-checked rather than hard-pinned: a
/// fabric run may legitimately use a different participant count, and the
/// observed world is cross-checked against the declaration everywhere.
pub fn operation_admitted(adapter: AdapterId, operation: &Operation) -> bool {
    match (adapter, operation) {
        (
            AdapterId::NcclAllreduceSum,
            Operation::AllReduce {
                op,
                dtype,
                numel,
                world,
                ranks,
                input,
                reference,
            },
        ) => {
            *op == ReduceOp::Sum
                && *dtype == Dtype::Int64
                && *numel == 262_144
                && *world >= 2
                && *world <= MAX_RANKS
                && ranks.len() == *world as usize
                && ranks.iter().enumerate().all(|(index, rank)| *rank == index as u32)
                && input == REDUCE_INPUT_ID
                && reference == REDUCE_REFERENCE_ID
        }
        _ => *operation == frozen_operation(adapter),
    }
}

pub fn required_clock(adapter: AdapterId) -> Clock {
    match adapter {
        AdapterId::CpuSumU64Reference => Clock {
            id: "process-monotonic-ns".into(),
            kind: ClockKind::ProcessMonotonic,
            units: Units::Nanoseconds,
            resolution_ns: 1,
            synchronization: Synchronization::ProcessClockReadBeforeAfter,
        },
        AdapterId::Exl3E3Grouped => Clock {
            id: "cuda-event-ms".into(),
            kind: ClockKind::DeviceEvent,
            units: Units::Milliseconds,
            resolution_ns: 1000,
            synchronization: Synchronization::DeviceEventTimingAfterSynchronize,
        },
        AdapterId::NcclAllreduceSum => Clock {
            id: "rank-monotonic-ns".into(),
            kind: ClockKind::HostMonotonic,
            units: Units::Nanoseconds,
            resolution_ns: 1,
            synchronization: Synchronization::CollectiveBarrierBeforeAfter,
        },
    }
}

pub fn frozen_correctness(adapter: AdapterId) -> CorrectnessContract {
    match adapter {
        AdapterId::CpuSumU64Reference => CorrectnessContract {
            reference: SUM_REFERENCE_ID.into(),
            tolerance: Tolerance::ExactU64,
        },
        AdapterId::Exl3E3Grouped => CorrectnessContract {
            reference: E3_REFERENCE_ID.into(),
            tolerance: E3_TOLERANCE,
        },
        AdapterId::NcclAllreduceSum => CorrectnessContract {
            reference: REDUCE_REFERENCE_ID.into(),
            tolerance: Tolerance::ExactInt64,
        },
    }
}

pub fn frozen_warmups(adapter: AdapterId) -> u32 {
    match adapter {
        AdapterId::CpuSumU64Reference | AdapterId::Exl3E3Grouped => 2,
        AdapterId::NcclAllreduceSum => 1,
    }
}

pub fn frozen_iterations(adapter: AdapterId) -> u32 {
    match adapter {
        AdapterId::CpuSumU64Reference | AdapterId::Exl3E3Grouped | AdapterId::NcclAllreduceSum => 5,
    }
}

fn reference_sum(bound: u32) -> u64 {
    let n = u64::from(bound);
    n * (n - 1) / 2
}

/// Exact payload bytes for the operation: `numel * sizeof(dtype)`. This is the
/// nccl-tests *algorithm* denominator, not the bus normalization.
pub fn algorithm_payload_bytes(operation: &Operation) -> Option<u64> {
    match operation {
        Operation::AllReduce { dtype, numel, .. } => u64::from(*numel).checked_mul(dtype.bytes()),
        _ => None,
    }
}

/// The nccl-tests all-reduce bus normalization `2*(world-1)/world`, retained
/// as an exact rational and labelled as a normalization, never as measured
/// physical link traffic.
pub fn bus_normalization(world: u32) -> Option<Rational> {
    if world < 2 {
        return None;
    }
    reduce(u128::from(2 * (world - 1)), u128::from(world))
}

/// Descriptive derivation for a rank-scoped operation. Both bandwidth values
/// are exact rationals derived from the acquisition statistic; the floating
/// display exists only for readability.
#[derive(Serialize)]
pub struct Derived {
    pub definition: &'static str,
    pub definition_source_sha256: &'static str,
    pub payload_bytes: u64,
    pub clock_id: String,
    pub units: &'static str,
    pub duration: Rational,
    pub duration_display_ns: f64,
    pub algorithm_bandwidth_bps: Rational,
    pub algorithm_bandwidth_display: f64,
    pub bus_normalization: Rational,
    pub bus_normalization_display: f64,
    pub bus_bandwidth_bps: Rational,
    pub bus_bandwidth_display: f64,
    pub bus_scope: &'static str,
}

fn derived(operation: &Operation, clock: &Clock, duration: Rational) -> Option<Derived> {
    let payload = algorithm_payload_bytes(operation)?;
    let world = match operation {
        Operation::AllReduce { world, .. } => *world,
        _ => return None,
    };
    if duration.numerator == 0 {
        return None;
    }
    let normalization = bus_normalization(world)?;
    // Exact bytes per second: payload * 1e9 / duration.
    let algorithm = reduce(
        u128::from(payload) * 1_000_000_000 * u128::from(duration.denominator),
        u128::from(duration.numerator),
    )?;
    let bus = multiply(algorithm, normalization)?;
    Some(Derived {
        definition: "algorithm bandwidth = payload_bytes / duration; payload_bytes = numel * sizeof(dtype)",
        definition_source_sha256: NCCL_BYTE_SOURCE_SHA256,
        payload_bytes: payload,
        clock_id: clock.id.clone(),
        units: clock.units.name(),
        duration,
        duration_display_ns: display(duration),
        algorithm_bandwidth_bps: algorithm,
        algorithm_bandwidth_display: display(algorithm),
        bus_normalization: normalization,
        bus_normalization_display: display(normalization),
        bus_bandwidth_bps: bus,
        bus_bandwidth_display: display(bus),
        bus_scope: "nccl-tests all-reduce bus-bandwidth normalization; not measured physical link traffic",
    })
}

struct Findings {
    invalid: Vec<Reason>,
    unavailable: Vec<Reason>,
}

impl Findings {
    fn new() -> Self {
        Self {
            invalid: Vec::new(),
            unavailable: Vec::new(),
        }
    }

    fn invalidate(&mut self, reason: Reason) {
        push(&mut self.invalid, reason);
    }

    fn withhold(&mut self, reason: Reason) {
        push(&mut self.unavailable, reason);
    }

    fn clean(&self) -> bool {
        self.invalid.is_empty() && self.unavailable.is_empty()
    }
}

/// Validate the prospective declaration on its own. The frozen cell is checked
/// here so a plan cannot drift into an unapproved shape, cap, tier or clock,
/// and the declared allowance must already cover the reference requirement.
pub fn check_plan(plan: &Plan) -> Vec<Reason> {
    let mut reasons = Vec::new();
    if plan.kind != PLAN_KIND || plan.version != VERSION {
        push(&mut reasons, Reason::InvalidPlan);
        return reasons;
    }
    if !identifier(&plan.revision) {
        push(&mut reasons, Reason::InvalidPlan);
    }
    if !identifier(&plan.acquisition.id) || plan.acquisition.started_unix_ms == 0 {
        push(&mut reasons, Reason::InvalidPlan);
    }
    if !operation_admitted(plan.adapter, &plan.operation) {
        push(&mut reasons, Reason::IdentityDrift);
    }
    if plan.clock != required_clock(plan.adapter) {
        push(&mut reasons, Reason::ClockMismatch);
    }
    if plan.correctness != frozen_correctness(plan.adapter) {
        push(&mut reasons, Reason::IdentityDrift);
    }
    if plan.warmups != frozen_warmups(plan.adapter)
        || plan.iterations != frozen_iterations(plan.adapter)
    {
        push(&mut reasons, Reason::IterationCountMismatch);
    }
    if plan.allowance.work_units == 0
        || plan.allowance.memory_bytes == 0
        || plan.allowance.deadline_ms == 0
        || plan.allowance.deadline_ms > 3_600_000
    {
        push(&mut reasons, Reason::InvalidPlan);
    }
    if let Some(thresholds) = plan.thresholds
        && (thresholds.adverse_bps > 9999 || thresholds.spread_bps > 1_000_000)
    {
        push(&mut reasons, Reason::InvalidBounds);
    }
    for source in &plan.sources {
        if source.path.is_empty() || source.path.len() > 4096 || !sha256(&source.sha256) {
            push(&mut reasons, Reason::InvalidPlan);
        }
    }
    match plan.adapter {
        AdapterId::CpuSumU64Reference => {
            if plan.program_sha256.is_some() || !plan.sources.is_empty() {
                push(&mut reasons, Reason::InvalidPlan);
            }
            if let Operation::SumU64 { bound, .. } = &plan.operation {
                let work = u64::from(*bound)
                    .checked_mul(u64::from(plan.warmups) + u64::from(plan.iterations));
                let memory = u64::from(*bound).checked_mul(8);
                match (work, memory) {
                    (Some(work), Some(memory))
                        if work <= plan.allowance.work_units
                            && memory <= plan.allowance.memory_bytes => {}
                    _ => push(&mut reasons, Reason::AllowanceExceeded),
                }
            }
        }
        AdapterId::Exl3E3Grouped => {
            if !plan.program_sha256.as_deref().is_some_and(sha256)
                || !pinned(&plan.sources, E3_OPERATION_SOURCE_SHA256)
                || !pinned(&plan.sources, E3_PARITY_SOURCE_SHA256)
            {
                push(&mut reasons, Reason::IdentityDrift);
            }
        }
        AdapterId::NcclAllreduceSum => {
            if !plan.program_sha256.as_deref().is_some_and(sha256)
                || !pinned(&plan.sources, NCCL_BYTE_SOURCE_SHA256)
            {
                push(&mut reasons, Reason::IdentityDrift);
            }
            if let Operation::AllReduce { world, ranks, .. } = &plan.operation
                && (*world < 2
                    || *world > MAX_RANKS
                    || ranks.len() != *world as usize
                    || ranks.iter().enumerate().any(|(i, r)| *r != i as u32))
            {
                push(&mut reasons, Reason::IdentityDrift);
            }
        }
    }
    reasons
}

fn pinned(sources: &[Source], sha: &str) -> bool {
    sources.iter().any(|source| source.sha256 == sha)
}

/// Structural then plan-relative validation. `plan` is absent only for
/// standalone inspection of a retained artifact, where plan-dependent checks
/// cannot run and are reported as such.
pub fn check_artifact(plan: Option<&Plan>, artifact: &Artifact) -> Findings {
    let mut findings = Findings::new();
    if artifact.kind != KIND || artifact.version != VERSION {
        findings.invalidate(Reason::InvalidArtifact);
        return findings;
    }
    if artifact.provenance == Provenance::Declared {
        findings.invalidate(Reason::InvalidArtifact);
    }
    if !identifier(&artifact.revision)
        || artifact
            .program_sha256
            .as_deref()
            .is_some_and(|hash| !sha256(hash))
    {
        findings.invalidate(Reason::InvalidArtifact);
    }
    if artifact.adapter.scope() != artifact.topology.scope {
        findings.invalidate(Reason::IdentityDrift);
    }
    if !operation_admitted(artifact.adapter, &artifact.operation) {
        findings.invalidate(Reason::IdentityDrift);
    }
    if artifact.clock != required_clock(artifact.adapter) {
        findings.invalidate(Reason::ClockMismatch);
    }
    if !artifact.adapter.external()
        && (artifact.program_sha256.is_some() || !artifact.sources.is_empty())
    {
        findings.invalidate(Reason::InvalidArtifact);
    }
    if artifact.adapter.external()
        && (artifact.program_sha256.is_none()
            || artifact.kernel_revision.as_deref().is_none_or(str::is_empty))
    {
        findings.invalidate(Reason::IdentityDrift);
    }
    if artifact.execution.warmups != frozen_warmups(artifact.adapter)
        || artifact.execution.iterations != frozen_iterations(artifact.adapter)
    {
        findings.invalidate(Reason::IterationCountMismatch);
    }
    if artifact.execution.timed_out {
        findings.withhold(Reason::DeadlineExceeded);
    }
    if !artifact.execution.completed {
        findings.withhold(Reason::SampleGridIncomplete);
    }
    if artifact.adapter == AdapterId::Exl3E3Grouped {
        let tier = match &artifact.operation {
            Operation::Exl3Experts { tier, .. } => Some(*tier),
            _ => None,
        };
        if tier.is_none() || artifact.execution.observed_fallback != tier {
            findings.withhold(Reason::TierFallback);
        }
    }
    check_grid(artifact, &mut findings);
    check_correctness(artifact, &mut findings);
    if artifact
        .failures
        .iter()
        .any(|failure| matches!(failure.kind, FailureKind::CorrectnessFailed | FailureKind::NonFinite))
    {
        findings.invalidate(Reason::CorrectnessFailed);
    } else if !artifact.failures.is_empty() {
        findings.withhold(Reason::RetainedFailure);
    }
    if let Some(plan) = plan {
        if plan.adapter != artifact.adapter {
            findings.invalidate(Reason::AdapterMismatch);
        }
        if plan.revision != artifact.revision
            || plan.operation != artifact.operation
            || plan.clock != artifact.clock
            || plan.warmups != artifact.execution.warmups
            || plan.iterations != artifact.execution.iterations
            || plan.sources != artifact.sources
            || plan.program_sha256 != artifact.program_sha256
        {
            findings.invalidate(Reason::IdentityDrift);
        }
        if artifact.execution.work_units > plan.allowance.work_units
            || artifact.execution.memory_bytes > plan.allowance.memory_bytes
        {
            findings.withhold(Reason::AllowanceExceeded);
        }
        if artifact.execution.deadline_ms > plan.allowance.deadline_ms {
            findings.invalidate(Reason::InvalidArtifact);
        }
        match &plan.operation {
            Operation::AllReduce { world, ranks, .. } => {
                if artifact.topology.world != Some(*world)
                    || artifact.topology.ranks.len() != ranks.len()
                {
                    findings.invalidate(Reason::IdentityDrift);
                }
            }
            // The CPU reference bound is verified from the operation itself in
            // `check_correctness`, so it holds with or without a retained plan.
            Operation::SumU64 { .. } | Operation::Exl3Experts { .. } => (),
        }
    }
    findings
}

fn check_grid(artifact: &Artifact, findings: &mut Findings) {
    let units = artifact.clock.units;
    let resolution = artifact.clock.resolution_ns;
    let world = artifact.topology.world.unwrap_or(0);
    let mut seen = std::collections::BTreeSet::new();
    for (position, sample) in artifact.samples.iter().enumerate() {
        let Some(nanoseconds) = duration_ns(&sample.raw, units) else {
            findings.invalidate(Reason::SamplePrecision);
            continue;
        };
        if nanoseconds != sample.duration_ns {
            findings.invalidate(Reason::SamplePrecision);
        }
        if resolution == 0 || nanoseconds % resolution != 0 {
            findings.invalidate(Reason::SamplePrecision);
        }
        if nanoseconds == 0 || nanoseconds > MAX_DURATION_NS {
            findings.invalidate(Reason::DurationOutOfRange);
        }
        let key = match artifact.topology.scope {
            Scope::Ranks => match (sample.repetition, sample.rank) {
                (Some(repetition), Some(rank)) => {
                    if rank >= world || repetition >= artifact.execution.iterations {
                        findings.invalidate(Reason::SampleGridIncomplete);
                        continue;
                    }
                    (repetition, rank)
                }
                _ => {
                    findings.invalidate(Reason::SampleGridIncomplete);
                    continue;
                }
            },
            Scope::Cpu | Scope::Device => {
                if sample.rank.is_some() || sample.repetition.is_some() {
                    findings.invalidate(Reason::SampleGridIncomplete);
                    continue;
                }
                (0, 0)
            }
        };
        if !seen.insert(key) || sample.index != position as u32 {
            findings.invalidate(Reason::SampleGridIncomplete);
        }
    }
    let expected = match artifact.topology.scope {
        Scope::Ranks => {
            if world < 2 || world > MAX_RANKS || artifact.topology.ranks.len() != world as usize {
                findings.invalidate(Reason::SampleGridIncomplete);
                return;
            }
            u64::from(world) * u64::from(artifact.execution.iterations)
        }
        Scope::Cpu | Scope::Device => u64::from(artifact.execution.iterations),
    };
    if artifact.samples.len() as u64 != expected {
        findings.invalidate(Reason::SampleGridIncomplete);
    }
}

fn check_correctness(artifact: &Artifact, findings: &mut Findings) {
    match &artifact.correctness {
        Correctness::ExactSum {
            reference,
            computed,
            bound,
            passed,
        } => {
            if reference != SUM_REFERENCE_ID {
                findings.invalidate(Reason::IdentityDrift);
            }
            let (Some(computed), Some(bound)) = (exact_decimal(computed), exact_decimal(bound)) else {
                findings.invalidate(Reason::NonFinite);
                return;
            };
            if compare(computed, bound).is_ne() || !*passed {
                findings.invalidate(Reason::CorrectnessFailed);
            }
            // The declared reference must be the independent closed form for
            // the declared bound, so neither value can drift independently.
            if let Operation::SumU64 { bound: declared, .. } = &artifact.operation
                && compare(
                    bound,
                    Rational {
                        numerator: reference_sum(*declared),
                        denominator: 1,
                    },
                )
                .is_ne()
            {
                findings.invalidate(Reason::CorrectnessFailed);
            }
        }
        Correctness::ExactReduction {
            reference,
            mismatches,
            tolerance_elems,
            finite,
            passed,
        } => {
            if reference != REDUCE_REFERENCE_ID || *tolerance_elems != 0 {
                findings.invalidate(Reason::IdentityDrift);
            }
            if !*finite {
                findings.invalidate(Reason::NonFinite);
            }
            if *mismatches != 0 || !*passed {
                findings.invalidate(Reason::CorrectnessFailed);
            }
        }
        Correctness::E3Parity {
            reference,
            finite,
            ref_max,
            e2,
            e3,
            tolerance,
            passed,
            outcome,
            ..
        } => {
            if reference != E3_REFERENCE_ID {
                findings.invalidate(Reason::IdentityDrift);
            }
            if *tolerance != E3_TOLERANCE {
                findings.invalidate(Reason::IdentityDrift);
            }
            if !*finite {
                findings.invalidate(Reason::NonFinite);
            }
            let (Some(ref_max), Some(e2), Some(e3)) =
                (exact_decimal(ref_max), stats(e2), stats(e3))
            else {
                findings.invalidate(Reason::NonFinite);
                return;
            };
            match e3_bounds(*tolerance, ref_max, &e2) {
                Some(bounds) => {
                    let violated = compare(e3[0], bounds[0]).is_gt()
                        || compare(e3[1], bounds[1]).is_gt()
                        || compare(e3[2], bounds[2]).is_gt()
                        || compare(e3[3], bounds[3]).is_gt()
                        || !compare(e3[0], bounds[4]).is_lt();
                    if violated || *outcome != CheckOutcome::Pass || !*passed {
                        findings.invalidate(Reason::CorrectnessFailed);
                    }
                }
                None => findings.invalidate(Reason::ArithmeticOverflow),
            }
        }
    }
}

fn stats(value: &ParityStats) -> Option<[Rational; 4]> {
    Some([
        exact_decimal(&value.maxabs)?,
        exact_decimal(&value.per_token_max)?,
        exact_decimal(&value.per_token_p99)?,
        exact_decimal(&value.nrmse)?,
    ])
}

/// Recompute the pinned `_assert_e3_within` bounds exactly. `e2` is the frozen
/// E2-vs-loop reference statistics; the E3 values must fit these bounds, so a
/// widened tolerance cannot appear in retained evidence.
fn e3_bounds(tolerance: Tolerance, ref_max: Rational, e2: &[Rational; 4]) -> Option<[Rational; 5]> {
    let floor = scale(ref_max, tolerance.abs_rel_micro(), 1_000_000)?;
    let factor = Rational {
        numerator: tolerance.factor_milli(),
        denominator: 1000,
    };
    let mut bounds = [Rational {
        numerator: 0,
        denominator: 1,
    }; 5];
    for index in 0..3 {
        bounds[index] = add(multiply(e2[index], factor)?, floor)?;
    }
    let nrmse_floor = Rational {
        numerator: tolerance.nrmse_abs_micro(),
        denominator: 1_000_000,
    };
    bounds[3] = add(multiply(e2[3], factor)?, nrmse_floor)?;
    let one = Rational {
        numerator: 1,
        denominator: 1,
    };
    let coarse_abs = Rational {
        numerator: tolerance.coarse_abs_milli(),
        denominator: 1000,
    };
    let coarse_factor = scale(
        larger(one, ref_max),
        tolerance.coarse_factor_milli(),
        1000,
    )?;
    bounds[4] = larger(coarse_abs, coarse_factor);
    Some(bounds)
}

/// One measured acquisition yields one exact statistic: the arithmetic mean of
/// its measured samples, or for rank scope the mean of the complete
/// per-repetition maxima. Never a best case, percentile or survivor subset.
fn statistic(artifact: &Artifact) -> Option<Rational> {
    let values = match artifact.topology.scope {
        Scope::Ranks => {
            let world = artifact.topology.world?;
            if world < 2
                || artifact.samples.len() as u64
                    != u64::from(world) * u64::from(artifact.execution.iterations)
            {
                return None;
            }
            let mut per_repetition = Vec::new();
            for repetition in 0..artifact.execution.iterations {
                let worst = artifact
                    .samples
                    .iter()
                    .filter(|sample| sample.repetition == Some(repetition))
                    .map(|sample| sample.duration_ns)
                    .max()?;
                per_repetition.push(worst);
            }
            per_repetition
        }
        Scope::Cpu | Scope::Device => {
            if artifact.samples.len() != artifact.execution.iterations as usize {
                return None;
            }
            artifact
                .samples
                .iter()
                .map(|sample| sample.duration_ns)
                .collect()
        }
    };
    if values.is_empty() {
        return None;
    }
    let mut total = 0u128;
    for value in &values {
        total = total.checked_add(u128::from(*value))?;
    }
    reduce(total, values.len() as u128)
}

fn exerciser(provenance: Provenance, adapter: AdapterId) -> &'static str {
    match (provenance, adapter.scope()) {
        (Provenance::Declared, _) => "operator-declared",
        (Provenance::Imported, _) => "imported-bytes",
        (Provenance::NativeObserved, Scope::Cpu) => "native-cpu-reference",
        (Provenance::NativeObserved, Scope::Device) => "native-device-program-observation",
        (Provenance::NativeObserved, Scope::Ranks) => "native-collective-program-observation",
    }
}

fn scope_sentence(provenance: Provenance, adapter: AdapterId) -> String {
    match (provenance, adapter.scope()) {
        (Provenance::Imported, _) => format!(
            "Imported comparison only for {}. Bytes this collector did not produce cannot establish that {} was exercised, and any provenance claim inside the payload is retained but never upgraded. Device or fabric qualification requires a separately authorized native capture.",
            adapter.name(),
            adapter.name()
        ),
        (Provenance::Declared, _) => format!(
            "Operator declaration only for {}; no measurement bytes were produced or parsed.",
            adapter.name()
        ),
        (Provenance::NativeObserved, Scope::Cpu) => format!(
            "Native CPU protocol exercise for {}. This proves native operation execution, sample recording, inspection and exact comparison; it does not demonstrate GPU, device or collective support, and it never sets a serving-speed claim.",
            adapter.name()
        ),
        (Provenance::NativeObserved, Scope::Device) => format!(
            "This collector launched the {} program and retained its synchronized device-event samples on the declared clock. Native observation is not authenticated execution, and a kernel result never sets a serving-speed claim; serving impact needs a separately linked serving acquisition.",
            adapter.name()
        ),
        (Provenance::NativeObserved, Scope::Ranks) => format!(
            "This collector launched the {} program and retained complete per-rank samples on the declared clock. One completed world repetition is one sample; ranks are never pooled, and rank durations from the same repetition may only be maximized. Native observation is not authenticated execution, and a collective result never sets a serving-speed claim.",
            adapter.name()
        ),
    }
}

// ---------------------------------------------------------------------------
// Native CPU reference adapter
// ---------------------------------------------------------------------------

fn reduce_sum(data: &[u64]) -> u64 {
    data.iter()
        .fold(0u64, |total, value| total.wrapping_add(*value))
}

/// Deterministic native CPU reference: `sum_u64` over `0..bound`, with the
/// exact closed-form reference `n*(n-1)/2` computed independently. The reduced
/// value passes through `black_box` on both sides so the timed reduction
/// cannot be optimized away.
fn cpu_reference(plan: &Plan, bound: u32) -> Result<Artifact> {
    let data: Vec<u64> = (0..u64::from(bound)).collect();
    let memory_bytes = (data.len() as u64)
        .checked_mul(8)
        .ok_or("CPU reference memory accounting overflow")?;
    let work_units = u64::from(bound)
        .checked_mul(u64::from(plan.warmups) + u64::from(plan.iterations))
        .ok_or("CPU reference work accounting overflow")?;
    if memory_bytes > plan.allowance.memory_bytes || work_units > plan.allowance.work_units {
        return Err(format!(
            "declared allowance is below the reference requirement: work {work_units}/{}, memory {memory_bytes}/{}",
            plan.allowance.work_units, plan.allowance.memory_bytes
        ));
    }
    let deadline = Duration::from_millis(plan.allowance.deadline_ms);
    let started = Instant::now();
    let mut warm = 0u64;
    for _ in 0..plan.warmups {
        warm = warm.wrapping_add(std::hint::black_box(reduce_sum(std::hint::black_box(&data))));
    }
    std::hint::black_box(warm);
    let mut samples = Vec::new();
    let mut failures = Vec::new();
    let mut timed_out = false;
    for index in 0..plan.iterations {
        if started.elapsed() >= deadline {
            timed_out = true;
            failures.push(Failure {
                kind: FailureKind::DeadlineExceeded,
                detail: format!(
                    "reference deadline of {} ms elapsed before iteration {index}",
                    plan.allowance.deadline_ms
                ),
            });
            break;
        }
        let origin = Instant::now();
        let observed = std::hint::black_box(reduce_sum(std::hint::black_box(&data)));
        let elapsed = origin.elapsed();
        if observed != reference_sum(bound) {
            failures.push(Failure {
                kind: FailureKind::CorrectnessFailed,
                detail: format!(
                    "iteration {index} reduced {observed} instead of {}",
                    reference_sum(bound)
                ),
            });
        }
        samples.push(Sample {
            index,
            rank: None,
            repetition: None,
            raw: elapsed.as_nanos().to_string(),
            duration_ns: u64::try_from(elapsed.as_nanos())
                .map_err(|_| "reference duration does not fit nanoseconds".to_string())?,
        });
    }
    Ok(Artifact {
        kind: KIND.into(),
        version: VERSION,
        adapter: plan.adapter,
        revision: plan.revision.clone(),
        operation: plan.operation.clone(),
        provenance: Provenance::NativeObserved,
        submitted_provenance: None,
        program_sha256: None,
        kernel_revision: None,
        sources: Vec::new(),
        topology: Topology {
            scope: Scope::Cpu,
            world: None,
            ranks: Vec::new(),
        },
        clock: plan.clock.clone(),
        execution: Execution {
            warmups: plan.warmups,
            iterations: plan.iterations,
            deadline_ms: plan.allowance.deadline_ms,
            work_units,
            memory_bytes,
            completed: !timed_out && samples.len() == plan.iterations as usize,
            timed_out,
            observed_fallback: None,
        },
        samples,
        correctness: Correctness::ExactSum {
            reference: SUM_REFERENCE_ID.into(),
            computed: reference_sum(bound).to_string(),
            bound: reference_sum(bound).to_string(),
            passed: true,
        },
        failures,
    })
}

// ---------------------------------------------------------------------------
// Explicit operator programs
// ---------------------------------------------------------------------------

struct ProgramRun {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    stdout_truncated: bool,
    status: Option<i32>,
    timed_out: bool,
    spawn_error: Option<String>,
    elapsed_ms: u64,
}

fn read_bounded(reader: impl std::io::Read, cap: usize) -> (Vec<u8>, bool) {
    use std::io::Read;
    let mut bytes = Vec::new();
    let mut limited = reader.take(cap as u64 + 1);
    let read_error = limited.read_to_end(&mut bytes).is_err();
    let truncated = read_error || bytes.len() > cap;
    bytes.truncate(cap);
    (bytes, truncated)
}

/// Launch one explicitly declared program with a bounded Grill-side deadline.
/// The program is invoked as `PROGRAM [extra..] --plan PLAN`, so a wrapper such
/// as `torchrun --nproc-per-node N SCRIPT` receives the plan as a trailing
/// argument for the script. Only stdout is parsed as the artifact body, and
/// stdout is never interpreted as shell.
fn run_program(program: &Path, extra: &[String], plan_path: &Path, deadline: Duration) -> ProgramRun {
    let mut command = Command::new(program);
    command
        .args(extra)
        .arg("--plan")
        .arg(plan_path)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let started = Instant::now();
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            return ProgramRun {
                stdout: Vec::new(),
                stderr: Vec::new(),
                stdout_truncated: false,
                status: None,
                timed_out: false,
                spawn_error: Some(format!("spawn {}: {error}", program.display())),
                elapsed_ms: started.elapsed().as_millis() as u64,
            };
        }
    };
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let out_reader = std::thread::spawn(move || match stdout {
        Some(stream) => read_bounded(stream, OUTPUT_CAP),
        None => (Vec::new(), false),
    });
    let err_reader = std::thread::spawn(move || match stderr {
        Some(stream) => read_bounded(stream, STDERR_CAP),
        None => (Vec::new(), false),
    });
    let mut status = None;
    let mut timed_out = false;
    let mut wait_error = None;
    loop {
        match child.try_wait() {
            Ok(Some(state)) => {
                status = state.code();
                break;
            }
            Ok(None) => (),
            Err(error) => {
                let _ = child.kill();
                wait_error = Some(format!("wait for {}: {error}", program.display()));
                break;
            }
        }
        if started.elapsed() >= deadline {
            let _ = child.kill();
            timed_out = true;
            let _ = child.wait();
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    let (stdout, stdout_truncated) = out_reader.join().unwrap_or((Vec::new(), false));
    let (stderr, _) = err_reader.join().unwrap_or((Vec::new(), false));
    ProgramRun {
        stdout,
        stderr,
        stdout_truncated,
        status,
        timed_out,
        spawn_error: wait_error,
        elapsed_ms: started.elapsed().as_millis() as u64,
    }
}

/// Hash an explicitly declared program file. Symlinks are resolved first
/// because interpreters and launchers are commonly symlinked; the digest is of
/// the resolved bytes.
fn hash_program(path: &Path) -> Result<String> {
    let resolved =
        std::fs::canonicalize(path).map_err(|e| format!("resolve program {}: {e}", path.display()))?;
    let metadata = std::fs::metadata(&resolved).map_err(|e| e.to_string())?;
    if !metadata.is_file() || metadata.len() > FILE_CAP as u64 {
        return Err(format!(
            "program is not a bounded regular file: {}",
            path.display()
        ));
    }
    let bytes = std::fs::read(&resolved).map_err(|e| format!("read program: {e}"))?;
    Ok(evidence::digest(&bytes))
}

// ---------------------------------------------------------------------------
// Reports
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct CaptureReport {
    pub claim: String,
    pub status: &'static str,
    pub adapter: AdapterId,
    pub exerciser: &'static str,
    pub provenance: Provenance,
    pub plan_sha256: String,
    pub artifact_sha256: Option<String>,
    pub out: String,
    pub program: Option<String>,
    pub program_sha256: Option<String>,
    pub samples: usize,
    pub statistic: Option<Rational>,
    pub invalid: Vec<Reason>,
    pub unavailable: Vec<Reason>,
    pub failure: Option<String>,
    pub scope: String,
}

impl CaptureReport {
    pub fn exit(&self) -> u8 {
        if !self.invalid.is_empty() || (self.artifact_sha256.is_none() && self.failure.is_some()) {
            Outcome::Error.exit()
        } else if !self.unavailable.is_empty() {
            Outcome::Inconclusive.exit()
        } else {
            Outcome::Pass.exit()
        }
    }

    fn provenance_name(&self) -> &'static str {
        match self.provenance {
            Provenance::Declared => "declared",
            Provenance::Imported => "imported",
            Provenance::NativeObserved => "native-observed",
        }
    }
}

#[derive(Serialize)]
pub struct Inspection {
    pub claim: &'static str,
    pub path: String,
    pub kind: String,
    pub version: u32,
    pub adapter: AdapterId,
    pub revision: String,
    pub exerciser: &'static str,
    pub provenance: Provenance,
    pub submitted_provenance: Option<Provenance>,
    pub operation: Operation,
    pub plan_sha256: Option<String>,
    pub plan_verified: bool,
    pub artifact_sha256: String,
    /// Collector that produced this receipt, when one was retained.
    pub collector_sha256: Option<String>,
    pub program_sha256: Option<String>,
    pub kernel_revision: Option<String>,
    pub sources: Vec<Source>,
    pub topology: Topology,
    pub clock: Clock,
    pub execution: Execution,
    pub samples: usize,
    pub statistic: Option<Rational>,
    pub statistic_definition: &'static str,
    pub correctness: Correctness,
    pub failures: Vec<Failure>,
    pub derived: Option<Derived>,
    pub receipt: Option<Receipt>,
    pub invalid: Vec<Reason>,
    pub unavailable: Vec<Reason>,
    pub scope: String,
}

impl Inspection {
    pub fn exit(&self) -> u8 {
        if !self.invalid.is_empty() {
            Outcome::Error.exit()
        } else if !self.unavailable.is_empty() {
            Outcome::Inconclusive.exit()
        } else {
            Outcome::Pass.exit()
        }
    }

    fn provenance_name(&self) -> &'static str {
        match self.provenance {
            Provenance::Declared => "declared",
            Provenance::Imported => "imported",
            Provenance::NativeObserved => "native-observed",
        }
    }
}

#[derive(Clone, Serialize)]
pub struct Identity {
    pub plan_sha256: String,
    pub artifact_sha256: String,
    pub collector_sha256: Option<String>,
    pub program_sha256: Option<String>,
    pub revision: String,
    pub acquisition: Acquisition,
}

#[derive(Clone, Serialize)]
pub struct Role {
    pub identity: Option<Identity>,
    pub provenance: Option<Provenance>,
    pub exerciser: &'static str,
    pub samples: usize,
    pub statistic: Option<Rational>,
    pub statistic_display_ns: Option<f64>,
    pub eligible: bool,
    pub invalid: Vec<Reason>,
    pub unavailable: Vec<Reason>,
    /// Never serialized: the validated plan feeds pin comparison only.
    #[serde(skip)]
    plan: Option<Plan>,
}

#[derive(Serialize)]
pub struct Roles {
    pub baseline: Role,
    pub candidate: Role,
    pub reference: Role,
}

#[derive(Serialize)]
pub struct Decision {
    pub version: u32,
    pub claim: &'static str,
    pub scope: String,
    pub statistic: &'static str,
    pub statistic_definition: &'static str,
    pub direction: &'static str,
    pub axis: &'static str,
    pub decision: Outcome,
    pub adverse_bounds: Option<[f64; 2]>,
    pub thresholds: Option<Thresholds>,
    pub roles: Roles,
    pub reference_range: Option<[Rational; 2]>,
    pub candidate_range: Option<[Rational; 2]>,
    pub reason_codes: Vec<Reason>,
}

impl Decision {
    pub fn exit(&self) -> u8 {
        self.decision.exit()
    }
}

fn unavailable_role(reason: Reason, exerciser: &'static str) -> Role {
    Role {
        identity: None,
        provenance: None,
        exerciser,
        samples: 0,
        statistic: None,
        statistic_display_ns: None,
        eligible: false,
        invalid: Vec::new(),
        unavailable: vec![reason],
        references: Vec::new(),
        plan: None,
    }
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

#[derive(clap::Args)]
pub struct CaptureOptions {
    /// Closed adapter identifier.
    #[arg(long, value_enum)]
    pub adapter: AdapterId,
    /// Prospective declaration validated against the produced artifact.
    #[arg(long)]
    pub plan: PathBuf,
    /// Fresh output directory; existing evidence is never overwritten.
    #[arg(long)]
    pub out: PathBuf,
    /// Explicit adapter program for the two external adapters.
    #[arg(long)]
    pub program: Option<PathBuf>,
    /// Required for device and collective adapters: names the separately
    /// authorized device window. The CPU reference never needs it.
    #[arg(long)]
    pub authorize_device_window: bool,
    /// Arguments passed through to the program after `--`.
    #[arg(last = true)]
    pub args: Vec<String>,
    #[arg(long)]
    pub json: bool,
}

#[derive(clap::Args)]
pub struct ImportOptions {
    /// Retained artifact bytes from a producer this collector did not launch.
    #[arg(long)]
    pub artifact: PathBuf,
    #[arg(long)]
    pub plan: PathBuf,
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long)]
    pub json: bool,
}

#[derive(clap::Args)]
pub struct InspectOptions {
    /// Artifact file, or a directory holding `artifact.json`/`plan.json`.
    pub path: PathBuf,
    #[arg(long)]
    pub json: bool,
}

#[derive(clap::Args)]
pub struct CompareOptions {
    pub baseline: PathBuf,
    pub candidate: PathBuf,
    #[arg(long)]
    pub reference: Option<PathBuf>,
    #[arg(long)]
    pub json: bool,
}

fn parse_plan(bytes: &[u8]) -> Result<Plan> {
    let plan: Plan =
        serde_json::from_slice(bytes).map_err(|e| format!("invalid microbench plan: {e}"))?;
    let reasons = check_plan(&plan);
    if !reasons.is_empty() {
        return Err(format!(
            "microbench plan rejected: {reasons:?}; the frozen cell, clock, correctness contract and pins are not negotiable in this slice"
        ));
    }
    Ok(plan)
}

fn parse_artifact(bytes: &[u8]) -> Result<Artifact> {
    serde_json::from_slice(bytes).map_err(|e| format!("invalid microbench artifact: {e}"))
}

/// Publish `plan.json`, optional `artifact.json` and `receipt.json` into a
/// directory created by the caller. A failed producer still retains its plan
/// and receipt.
fn publish(
    out: &Path,
    plan_bytes: &[u8],
    artifact: Option<&Artifact>,
    receipt: &Receipt,
) -> Result<(String, Option<String>)> {
    let plan_path = out.join("plan.json");
    if std::fs::symlink_metadata(&plan_path).is_err() {
        evidence::write(&plan_path, plan_bytes)?;
    }
    let artifact_sha = match artifact {
        Some(artifact) => {
            let bytes = evidence::json(&out.join("artifact.json"), artifact)?;
            Some(evidence::digest(&bytes))
        }
        None => None,
    };
    let mut receipt = receipt.clone();
    receipt.artifact_sha256 = artifact_sha.clone();
    evidence::publish(out, "receipt.json", &receipt)?;
    evidence::sync(out)?;
    Ok((evidence::digest(plan_bytes), artifact_sha))
}

/// Fields every receipt shares; the producer-specific fields are filled in at
/// the call site so no constructor grows an argument list.
struct ReceiptSeed {
    adapter: AdapterId,
    provenance: Provenance,
    status: ReceiptStatus,
    claim: &'static str,
    plan_sha256: String,
    collector_sha256: String,
    started_unix_ms: u64,
    observation_unix_ms: u64,
    duration_ms: u64,
}

impl ReceiptSeed {
    fn build(self) -> Receipt {
        Receipt {
            kind: RECEIPT_KIND.into(),
            version: VERSION,
            adapter: self.adapter,
            provenance: self.provenance,
            status: self.status,
            plan_sha256: self.plan_sha256,
            artifact_sha256: None,
            collector_sha256: self.collector_sha256,
            program: None,
            program_sha256: None,
            source_sha256: None,
            started_unix_ms: self.started_unix_ms,
            observation_unix_ms: self.observation_unix_ms,
            duration_ms: self.duration_ms,
            command: Vec::new(),
            failure: None,
            claim: self.claim.into(),
        }
    }
}

pub fn capture(options: &CaptureOptions) -> Result<CaptureReport> {
    let plan_bytes = evidence::read(&options.plan, FILE_CAP)?;
    let plan = parse_plan(&plan_bytes)?;
    if plan.adapter != options.adapter {
        return Err(format!(
            "plan declares adapter {} but --adapter is {}",
            plan.adapter.name(),
            options.adapter.name()
        ));
    }
    let collector_sha256 = evidence::binary_digest()?;
    let plan_sha256 = evidence::digest(&plan_bytes);
    let started_unix_ms = unix_ms()?;
    if plan.acquisition.started_unix_ms > started_unix_ms {
        return Err(
            "plan declares an acquisition start in the future; declare a plan before running it"
                .into(),
        );
    }
    if plan.adapter.external() && !options.authorize_device_window {
        return Err(
            "device and collective capture is outside the CPU-only authorization; pass --authorize-device-window only inside a separately approved window".into(),
        );
    }
    let mut command = vec![format!(
        "grill-perf microbench capture --adapter {}",
        plan.adapter.name()
    )];
    evidence::fresh(&options.out)?;
    let (artifact, mut receipt, failure) = if plan.adapter.external() {
        let program = options.program.as_deref().ok_or(
            "this adapter requires an explicitly declared --program; there is no discovery",
        )?;
        let program_sha256 = hash_program(program)?;
        if plan.program_sha256.as_deref() != Some(program_sha256.as_str()) {
            return Err(
                "the declared program does not match plan.program_sha256; re-pin the plan instead of running a different producer"
                    .into(),
            );
        }
        command.push(format!("program={}", program.display()));
        command.extend(options.args.iter().cloned());
        let plan_path = options.out.join("plan.json");
        evidence::write(&plan_path, &plan_bytes)?;
        let run = run_program(
            program,
            &options.args,
            &plan_path,
            Duration::from_millis(plan.allowance.deadline_ms),
        );
        let source_sha256 = Some(evidence::digest(&run.stdout));
        let failure = if let Some(error) = &run.spawn_error {
            Some(error.clone())
        } else if run.timed_out {
            Some(format!(
                "program exceeded the declared {} ms deadline and was terminated",
                plan.allowance.deadline_ms
            ))
        } else if run.status != Some(0) {
            Some(format!("program exited with status {:?}", run.status))
        } else if run.stdout_truncated {
            Some("program output exceeded the bounded artifact allowance".into())
        } else {
            None
        };
        let parsed = if failure.is_none() {
            match parse_artifact(&run.stdout) {
                Ok(artifact) => Some(artifact),
                Err(error) => {
                    let mut receipt = ReceiptSeed {
                        adapter: plan.adapter,
                        provenance: Provenance::NativeObserved,
                        status: ReceiptStatus::Failed,
                        claim: "observed-producer-not-authenticated-execution",
                        plan_sha256: plan_sha256.clone(),
                        collector_sha256: collector_sha256.clone(),
                        started_unix_ms,
                        observation_unix_ms: unix_ms()?,
                        duration_ms: run.elapsed_ms,
                    }
                    .build();
                    receipt.program = Some(program.display().to_string());
                    receipt.program_sha256 = Some(program_sha256.clone());
                    receipt.source_sha256 = source_sha256.clone();
                    receipt.command = command.clone();
                    receipt.failure = Some(error.clone());
                    publish(&options.out, &plan_bytes, None, &receipt)?;
                    return Ok(CaptureReport {
                        claim: "observed-producer-not-authenticated-execution".into(),
                        status: "failed",
                        adapter: plan.adapter,
                        exerciser: exerciser(Provenance::NativeObserved, plan.adapter),
                        provenance: Provenance::NativeObserved,
                        plan_sha256,
                        artifact_sha256: None,
                        out: options.out.display().to_string(),
                        program: Some(program.display().to_string()),
                        program_sha256: Some(program_sha256),
                        samples: 0,
                        statistic: None,
                        invalid: Vec::new(),
                        unavailable: vec![Reason::InvalidArtifact],
                        failure: Some(error),
                        scope: scope_sentence(Provenance::NativeObserved, plan.adapter),
                    });
                }
            }
        } else {
            None
        };
        let mut artifact = parsed;
        if let Some(payload) = artifact.as_mut()
            && payload.provenance != Provenance::NativeObserved
        {
            payload.submitted_provenance = Some(payload.provenance);
            payload.provenance = Provenance::NativeObserved;
        }
        let mut receipt = ReceiptSeed {
            adapter: plan.adapter,
            provenance: Provenance::NativeObserved,
            status: if failure.is_some() {
                ReceiptStatus::Failed
            } else {
                ReceiptStatus::Captured
            },
            claim: "observed-producer-not-authenticated-execution",
            plan_sha256: plan_sha256.clone(),
            collector_sha256: collector_sha256.clone(),
            started_unix_ms,
            observation_unix_ms: unix_ms()?,
            duration_ms: run.elapsed_ms,
        }
        .build();
        receipt.program = Some(program.display().to_string());
        receipt.program_sha256 = Some(program_sha256);
        receipt.source_sha256 = source_sha256;
        receipt.command = command;
        receipt.failure = failure.clone();
        (artifact, receipt, failure)
    } else {
        if options.program.is_some() {
            return Err(
                "the in-process CPU reference adapter rejects --program; an operator program needs an external adapter id"
                    .into(),
            );
        }
        let bound = match plan.operation {
            Operation::SumU64 { bound, .. } => bound,
            _ => return Err("the CPU reference requires the sum-u64 operation".into()),
        };
        let artifact = cpu_reference(&plan, bound)?;
        let failed = artifact.failures.iter().any(|failure| {
            matches!(
                failure.kind,
                FailureKind::CorrectnessFailed | FailureKind::NonFinite | FailureKind::DeadlineExceeded
            )
        });
        let mut receipt = ReceiptSeed {
            adapter: plan.adapter,
            provenance: Provenance::NativeObserved,
            status: if failed {
                ReceiptStatus::Failed
            } else {
                ReceiptStatus::Captured
            },
            claim: "native-in-process-cpu-reference",
            plan_sha256: plan_sha256.clone(),
            collector_sha256: collector_sha256.clone(),
            started_unix_ms,
            observation_unix_ms: unix_ms()?,
            duration_ms: 0,
        }
        .build();
        receipt.command = command;
        receipt.failure = failed.then(|| "the native reference retained a failure".to_string());
        let failure = receipt.failure.clone();
        (Some(artifact), receipt, failure)
    };
    let findings = match &artifact {
        Some(artifact) => check_artifact(Some(&plan), artifact),
        None => {
            let mut findings = Findings::new();
            findings.withhold(Reason::CaptureFailed);
            findings
        }
    };
    receipt.failure = receipt.failure.clone().or_else(|| failure.clone());
    let (plan_sha256, artifact_sha256) =
        publish(&options.out, &plan_bytes, artifact.as_ref(), &receipt)?;
    Ok(CaptureReport {
        claim: receipt.claim.clone(),
        status: if artifact.is_some() && failure.is_none() {
            "captured"
        } else {
            "failed"
        },
        adapter: plan.adapter,
        exerciser: exerciser(receipt.provenance, plan.adapter),
        provenance: receipt.provenance,
        plan_sha256,
        artifact_sha256,
        out: options.out.display().to_string(),
        program: receipt.program.clone(),
        program_sha256: receipt.program_sha256.clone(),
        samples: artifact.as_ref().map_or(0, |artifact| artifact.samples.len()),
        statistic: artifact.as_ref().and_then(statistic),
        invalid: findings.invalid,
        unavailable: findings.unavailable,
        failure,
        scope: scope_sentence(receipt.provenance, plan.adapter),
    })
}

pub fn import(options: &ImportOptions) -> Result<CaptureReport> {
    let plan_bytes = evidence::read(&options.plan, FILE_CAP)?;
    let plan = parse_plan(&plan_bytes)?;
    let artifact_bytes = evidence::read(&options.artifact, FILE_CAP)?;
    let mut artifact = parse_artifact(&artifact_bytes)?;
    let collector_sha256 = evidence::binary_digest()?;
    let observation_unix_ms = unix_ms()?;
    if plan.acquisition.started_unix_ms > observation_unix_ms {
        return Err("plan declares an acquisition start in the future".into());
    }
    // Importing never upgrades provenance, whatever the payload claims.
    if artifact.provenance == Provenance::NativeObserved {
        artifact.submitted_provenance = Some(Provenance::NativeObserved);
    }
    artifact.provenance = Provenance::Imported;
    let findings = check_artifact(Some(&plan), &artifact);
    let plan_sha256 = evidence::digest(&plan_bytes);
    let mut receipt = ReceiptSeed {
        adapter: plan.adapter,
        provenance: Provenance::Imported,
        status: ReceiptStatus::Imported,
        claim: "imported-bytes-not-authenticated-execution",
        plan_sha256,
        collector_sha256,
        started_unix_ms: plan.acquisition.started_unix_ms,
        observation_unix_ms,
        duration_ms: 0,
    }
    .build();
    receipt.program_sha256 = artifact.program_sha256.clone();
    receipt.source_sha256 = Some(evidence::digest(&artifact_bytes));
    receipt.command = vec!["grill-perf microbench import".into()];
    evidence::fresh(&options.out)?;
    let (plan_sha256, artifact_sha256) =
        publish(&options.out, &plan_bytes, Some(&artifact), &receipt)?;
    Ok(CaptureReport {
        claim: "imported-bytes-not-authenticated-execution".into(),
        status: "imported",
        adapter: plan.adapter,
        exerciser: exerciser(Provenance::Imported, plan.adapter),
        provenance: Provenance::Imported,
        plan_sha256,
        artifact_sha256,
        out: options.out.display().to_string(),
        program: None,
        program_sha256: artifact.program_sha256.clone(),
        samples: artifact.samples.len(),
        statistic: statistic(&artifact),
        invalid: findings.invalid,
        unavailable: findings.unavailable,
        failure: None,
        scope: scope_sentence(Provenance::Imported, plan.adapter),
    })
}

struct Loaded {
    plan: Plan,
    plan_sha256: String,
    artifact: Artifact,
    artifact_sha256: String,
    receipt: Option<Receipt>,
    findings: Findings,
}

enum LoadError {
    /// The evidence is absent or unreadable: unavailable, not corrupt.
    Unavailable(Reason),
    /// The evidence is present but invalid: never silently accepted.
    Invalid(Reason),
}

fn load_dir(path: &Path) -> std::result::Result<Loaded, LoadError> {
    evidence::directory(path).map_err(|_| LoadError::Unavailable(Reason::MissingArtifact))?;
    let plan_bytes = evidence::read(&path.join("plan.json"), FILE_CAP)
        .map_err(|_| LoadError::Unavailable(Reason::MissingArtifact))?;
    let plan = parse_plan(&plan_bytes).map_err(|_| LoadError::Invalid(Reason::InvalidPlan))?;
    let artifact_bytes = evidence::read(&path.join("artifact.json"), FILE_CAP)
        .map_err(|_| LoadError::Unavailable(Reason::MissingArtifact))?;
    let artifact =
        parse_artifact(&artifact_bytes).map_err(|_| LoadError::Invalid(Reason::InvalidArtifact))?;
    let receipt = evidence::read(&path.join("receipt.json"), FILE_CAP)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Receipt>(&bytes).ok());
    let findings = check_artifact(Some(&plan), &artifact);
    Ok(Loaded {
        plan,
        plan_sha256: evidence::digest(&plan_bytes),
        artifact,
        artifact_sha256: evidence::digest(&artifact_bytes),
        receipt,
        findings,
    })
}

pub fn inspect(options: &InspectOptions) -> Result<Inspection> {
    if options.path.is_dir() {
        return match load_dir(&options.path) {
            Ok(loaded) => {
                let plan_sha256 = loaded.plan_sha256.clone();
                Ok(inspection(options, plan_sha256, loaded))
            }
            Err(LoadError::Unavailable(reason)) => Err(format!(
                "{}: {reason:?}; the directory must hold plan.json and artifact.json",
                options.path.display()
            )),
            Err(LoadError::Invalid(reason)) => {
                Err(format!("{}: {reason:?}", options.path.display()))
            }
        };
    }
    let bytes = evidence::read(&options.path, FILE_CAP)?;
    let artifact_sha256 = evidence::digest(&bytes);
    let artifact = parse_artifact(&bytes)?;
    let sibling = options.path.parent().map(|parent| parent.join("plan.json"));
    let (plan, plan_sha256) = match sibling
        .as_deref()
        .and_then(|path| evidence::read(path, FILE_CAP).ok())
        .and_then(|bytes| {
            parse_plan(&bytes)
                .ok()
                .map(|plan| (plan, evidence::digest(&bytes)))
        }) {
        Some((plan, sha)) => (Some(plan), Some(sha)),
        None => (None, None),
    };
    let findings = check_artifact(plan.as_ref(), &artifact);
    let receipt = options
        .path
        .parent()
        .and_then(|parent| evidence::read(&parent.join("receipt.json"), FILE_CAP).ok())
        .and_then(|bytes| serde_json::from_slice::<Receipt>(&bytes).ok());
    Ok(inspection_of(
        options.path.display().to_string(),
        plan_sha256,
        artifact_sha256,
        artifact,
        receipt,
        findings,
    ))
}

fn inspection(options: &InspectOptions, plan_sha256: String, loaded: Loaded) -> Inspection {
    inspection_of(
        options.path.display().to_string(),
        Some(plan_sha256),
        loaded.artifact_sha256,
        loaded.artifact,
        loaded.receipt,
        loaded.findings,
    )
}

fn inspection_of(
    path: String,
    plan_sha256: Option<String>,
    artifact_sha256: String,
    artifact: Artifact,
    receipt: Option<Receipt>,
    findings: Findings,
) -> Inspection {
    let statistic = statistic(&artifact);
    let derived = statistic.and_then(|value| derived(&artifact.operation, &artifact.clock, value));
    let collector_sha256 = receipt
        .as_ref()
        .map(|receipt| receipt.collector_sha256.clone());
    Inspection {
        claim: "microbench-artifact-inspection-not-qualification",
        path,
        kind: artifact.kind.clone(),
        version: artifact.version,
        adapter: artifact.adapter,
        revision: artifact.revision.clone(),
        exerciser: exerciser(artifact.provenance, artifact.adapter),
        provenance: artifact.provenance,
        submitted_provenance: artifact.submitted_provenance,
        operation: artifact.operation.clone(),
        plan_verified: plan_sha256.is_some(),
        plan_sha256,
        artifact_sha256,
        collector_sha256,
        program_sha256: artifact.program_sha256.clone(),
        kernel_revision: artifact.kernel_revision.clone(),
        sources: artifact.sources.clone(),
        topology: artifact.topology.clone(),
        clock: artifact.clock.clone(),
        execution: artifact.execution.clone(),
        samples: artifact.samples.len(),
        statistic,
        statistic_definition: "exact arithmetic mean of the measured samples in this acquisition (rank scope: mean of complete per-repetition maxima)",
        correctness: artifact.correctness.clone(),
        failures: artifact.failures.clone(),
        derived,
        receipt,
        invalid: findings.invalid,
        unavailable: findings.unavailable,
        scope: scope_sentence(artifact.provenance, artifact.adapter),
    }
}

fn load_role(path: &Path, missing: Reason) -> Role {
    match load_dir(path) {
        Ok(loaded) => {
            let eligible = loaded.findings.clean();
            let statistic = if eligible {
                statistic(&loaded.artifact)
            } else {
                None
            };
            Role {
                identity: Some(Identity {
                    plan_sha256: loaded.plan_sha256,
                    artifact_sha256: loaded.artifact_sha256,
                    collector_sha256: loaded
                        .receipt
                        .as_ref()
                        .map(|receipt| receipt.collector_sha256.clone()),
                    program_sha256: loaded.artifact.program_sha256.clone(),
                    revision: loaded.artifact.revision.clone(),
                    acquisition: loaded.plan.acquisition.clone(),
                }),
                provenance: Some(loaded.artifact.provenance),
                exerciser: exerciser(loaded.artifact.provenance, loaded.artifact.adapter),
                samples: loaded.artifact.samples.len(),
                statistic,
                statistic_display_ns: statistic.map(display),
                eligible: eligible && statistic.is_some(),
                invalid: loaded.findings.invalid,
                unavailable: loaded.findings.unavailable,
                plan: Some(loaded.plan),
            }
        }
        Err(LoadError::Invalid(reason)) => {
            let mut role = unavailable_role(missing, "none");
            role.invalid = vec![reason];
            role
        }
        Err(LoadError::Unavailable(_)) => unavailable_role(missing, "none"),
    }
}

/// Every pin except the declared candidate axis must match across roles. The
/// acquisition identity itself is required to differ and be strictly ordered.
fn pins_match(a: &Plan, b: &Plan) -> bool {
    a.kind == b.kind
        && a.version == b.version
        && a.adapter == b.adapter
        && a.program_sha256 == b.program_sha256
        && a.sources == b.sources
        && a.operation == b.operation
        && a.clock == b.clock
        && a.warmups == b.warmups
        && a.iterations == b.iterations
        && a.allowance == b.allowance
        && a.correctness == b.correctness
        && a.thresholds == b.thresholds
}

pub fn compare(baseline: &Path, candidate: &Path, reference: Option<&Path>) -> Decision {
    let mut reason_codes = Vec::new();
    let mut decision = Outcome::Pass;
    let roles = [
        load_role(baseline, Reason::MissingArtifact),
        load_role(candidate, Reason::MissingArtifact),
        match reference {
            Some(path) => load_role(path, Reason::MissingReference),
            None => unavailable_role(Reason::MissingReference, "none"),
        },
    ];
    if reference.is_none() {
        push(&mut reason_codes, Reason::MissingReference);
    }
    for role in &roles {
        for reason in &role.invalid {
            decision = Outcome::Error;
            push(&mut reason_codes, *reason);
        }
        for reason in &role.unavailable {
            decision = decision.max(Outcome::Inconclusive);
            push(&mut reason_codes, *reason);
        }
    }
    let plans: [Option<&Plan>; 3] = [
        roles[0].plan.as_ref(),
        roles[1].plan.as_ref(),
        roles[2].plan.as_ref(),
    ];
    if let [Some(a), Some(b), Some(a2)] = plans {
        for (left, right) in [(a, b), (a, a2), (b, a2)] {
            if !pins_match(left, right) {
                decision = Outcome::Error;
                push(&mut reason_codes, Reason::IdentityDrift);
            }
        }
        if a.acquisition.id == b.acquisition.id
            || a.acquisition.id == a2.acquisition.id
            || b.acquisition.id == a2.acquisition.id
        {
            decision = Outcome::Error;
            push(&mut reason_codes, Reason::RoleReuse);
        }
        if !(a.acquisition.started_unix_ms < b.acquisition.started_unix_ms
            && b.acquisition.started_unix_ms < a2.acquisition.started_unix_ms)
        {
            decision = Outcome::Error;
            push(&mut reason_codes, Reason::DeclaredStartsOutOfOrder);
        }
    }
    let thresholds = plans[0].and_then(|plan| plan.thresholds);
    if thresholds.is_none() {
        decision = decision.max(Outcome::Inconclusive);
        push(&mut reason_codes, Reason::MissingThresholds);
    }
    let statistics = [roles[0].statistic, roles[1].statistic, roles[2].statistic];
    let mut adverse_bounds = None;
    let mut reference_range = None;
    let mut candidate_range = None;
    if decision == Outcome::Pass
        && thresholds.is_some()
        && let (Some(a), Some(b), Some(a2)) = (statistics[0], statistics[1], statistics[2])
    {
        let pooled = envelope::range(&[a, a2])
            .and_then(|range| range.ok_or(EnvelopeReason::InvalidRational));
        let candidate_span =
            envelope::range(&[b]).and_then(|range| range.ok_or(EnvelopeReason::InvalidRational));
        match (pooled, candidate_span) {
            (Ok(pooled), Ok(candidate_span)) => {
                reference_range = Some(pooled);
                candidate_range = Some(candidate_span);
                let thresholds = thresholds.unwrap();
                match envelope::assess(
                    pooled,
                    candidate_span,
                    thresholds.adverse_bps,
                    thresholds.spread_bps,
                    Direction::LowerBetter,
                ) {
                    Ok(assessment) => {
                        adverse_bounds = Some(assessment.adverse_bounds);
                        decision = decision.max(match assessment.decision {
                            EnvelopeDecision::Pass => Outcome::Pass,
                            EnvelopeDecision::Regression => Outcome::Regression,
                        });
                    }
                    Err(reason) => {
                        adverse_bounds = match reason {
                            EnvelopeReason::ArithmeticOverflow { adverse_bounds } => adverse_bounds,
                            EnvelopeReason::EnvelopeStraddlesTolerance { adverse_bounds } => {
                                Some(adverse_bounds)
                            }
                            _ => None,
                        };
                        let mapped = envelope_reason(reason);
                        decision = decision.max(match mapped {
                            Reason::ArithmeticOverflow | Reason::InvalidRational => Outcome::Error,
                            _ => Outcome::Inconclusive,
                        });
                        push(&mut reason_codes, mapped);
                    }
                }
            }
            (Err(reason), _) | (_, Err(reason)) => {
                decision = Outcome::Error;
                push(&mut reason_codes, envelope_reason(reason));
            }
        }
    } else if decision == Outcome::Pass {
        decision = Outcome::Inconclusive;
        push(&mut reason_codes, Reason::StatisticUnavailable);
    }
    let adapter = plans[0].map(|plan| plan.adapter);
    let provenance = roles[0].provenance;
    let scope = match (provenance, adapter) {
        (Some(Provenance::NativeObserved), Some(adapter)) => {
            scope_sentence(Provenance::NativeObserved, adapter)
        }
        (Some(Provenance::Imported), Some(adapter)) => scope_sentence(Provenance::Imported, adapter),
        (Some(Provenance::Declared), Some(adapter)) => scope_sentence(Provenance::Declared, adapter),
        _ => "Unavailable evidence: the baseline acquisition could not be loaded, so no comparison scope exists.".into(),
    };
    let exerciser = roles[0].exerciser;
    Decision {
        version: VERSION,
        claim: "observed-microbench-comparison-not-serving-speed",
        scope: format!(
            "{} Exerciser: {}. This is a measurement-layer observation only; kernel or collective results never establish serving-speed impact, capacity, adoption or live qualification.",
            scope, exerciser
        ),
        statistic: "mean-duration-ns",
        statistic_definition: "exact arithmetic mean per acquisition; rank scope uses complete per-repetition maxima",
        direction: "lower_better",
        axis: "adapter-revision",
        decision,
        adverse_bounds,
        thresholds,
        roles: Roles {
            baseline: roles[0].clone(),
            candidate: roles[1].clone(),
            reference: roles[2].clone(),
        },
        reference_range,
        candidate_range,
        reason_codes,
    }
}

pub fn human_capture(report: &CaptureReport) -> String {
    let mut text = String::new();
    text.push_str(&format!(
        "{}: {} via {} ({})\n",
        report.status,
        report.adapter.name(),
        report.exerciser,
        report.provenance_name()
    ));
    text.push_str(&format!("  evidence: {}\n", report.out));
    text.push_str(&format!("  samples: {}", report.samples));
    if let Some(statistic) = report.statistic {
        text.push_str(&format!("; mean {:.3} ns", display(statistic)));
    }
    text.push('\n');
    for reason in &report.invalid {
        text.push_str(&format!("  invalid: {reason:?}\n"));
    }
    for reason in &report.unavailable {
        text.push_str(&format!("  unavailable: {reason:?}\n"));
    }
    if let Some(failure) = &report.failure {
        text.push_str(&format!("  failure: {failure}\n"));
    }
    text.push_str(&format!("  scope: {}\n", report.scope));
    text
}

pub fn human_inspection(inspection: &Inspection) -> String {
    let mut text = String::new();
    text.push_str(&format!(
        "{} version {} adapter {}, provenance {}, plan verified {}\n",
        inspection.kind,
        inspection.version,
        inspection.adapter.name(),
        inspection.provenance_name(),
        inspection.plan_verified
    ));
    text.push_str(&format!(
        "  revision {}; samples {}; exerciser {}\n",
        inspection.revision, inspection.samples, inspection.exerciser
    ));
    if let Some(statistic) = inspection.statistic {
        text.push_str(&format!(
            "  mean duration {:.3} ns on clock {} ({})\n",
            display(statistic),
            inspection.clock.id,
            inspection.clock.units.name()
        ));
    }
    if let Some(derived) = &inspection.derived {
        text.push_str(&format!(
            "  {}: algorithm bandwidth {:.3} B/s\n",
            derived.definition, derived.algorithm_bandwidth_display
        ));
        text.push_str(&format!(
            "  bus normalization {}/{} = {:.3} ({})\n",
            derived.bus_normalization.numerator,
            derived.bus_normalization.denominator,
            derived.bus_normalization_display,
            derived.bus_scope
        ));
    }
    for reason in &inspection.invalid {
        text.push_str(&format!("  invalid: {reason:?}\n"));
    }
    for reason in &inspection.unavailable {
        text.push_str(&format!("  unavailable: {reason:?}\n"));
    }
    for failure in &inspection.failures {
        text.push_str(&format!(
            "  retained failure {:?}: {}\n",
            failure.kind, failure.detail
        ));
    }
    text.push_str(&format!("  scope: {}\n", inspection.scope));
    text
}

pub fn human_decision(decision: &Decision) -> String {
    let mut text = String::new();
    text.push_str(&format!(
        "microbench comparison ({}): {:?}\n",
        decision.statistic, decision.decision
    ));
    for (name, role) in [
        ("baseline", &decision.roles.baseline),
        ("candidate", &decision.roles.candidate),
        ("reference", &decision.roles.reference),
    ] {
        match role.statistic {
            Some(statistic) => text.push_str(&format!(
                "  {name}: {} samples, mean {:.3} ns, {}, eligible {}\n",
                role.samples,
                display(statistic),
                role.exerciser,
                role.eligible
            )),
            None => text.push_str(&format!("  {name}: unavailable\n")),
        }
    }
    if let Some(bounds) = decision.adverse_bounds {
        text.push_str(&format!(
            "  adverse bounds: {:+.4} .. {:+.4}\n",
            bounds[0], bounds[1]
        ));
    }
    for reason in &decision.reason_codes {
        text.push_str(&format!("  reason: {reason:?}\n"));
    }
    text.push_str(&format!("  scope: {}\n", decision.scope));
    text
}

#[cfg(test)]
#[path = "../tests/support/microbench_model.rs"]
mod microbench_model;

#[cfg(test)]
#[path = "../tests/support/microbench.rs"]
mod microbench;
