//! Synthetic CPU-only fixtures. Never open the NVIDIA soname or a device.
use super::*;
use super::super::{self as resources, Capture, Clock, ClockKind, ClockUnit, Gate, MeasuredInterval,
    Observation, Ownership, Phase, Provenance, ResourcesConfig, Role, Snapshot, Source, Statistic,
    Synchronization, Value, load, summarize, validate_observation};
use std::path::PathBuf;

const UUID: &str = "GPU-00000000-0000-0000-0000-000000000001";
const OTHER_UUID: &str = "GPU-00000000-0000-0000-0000-000000000002";
fn buffer(s: &str, n: usize) -> Payload {
    let mut bytes = vec![0; n]; bytes[..s.len()].copy_from_slice(s.as_bytes()); Payload::Text(bytes)
}
fn fixture(memory: bool) -> Trace {
    let mut trace = Trace::new();
    trace.push(Api::Init, 0, None);
    trace.push(Api::Version, 0, Some(buffer("fixture-nvml", 80)));
    trace.push(Api::Driver, 0, Some(buffer("fixture-driver", 80)));
    trace.push(Api::Handle, 0, Some(Payload::Handle(true)));
    trace.push(Api::UuidBefore, 0, Some(buffer(UUID, 96)));
    if memory {
        trace.push(Api::Memory, 0, Some(Payload::Memory(MemoryInfo { total: 8192, free: 6144, used: 2048 })));
    } else {
        trace.push(Api::Power, 0, Some(Payload::Power(PowerValue { field_id: 186, scope_id: 0,
            timestamp_us: 1_000_000, latency_us: 10, value_type: 1, code: 0,
            value: Some(Scalar::UnsignedInt(125123)) })));
    }
    trace.push(Api::UuidAfter, 0, Some(buffer(UUID, 96)));
    trace.push(Api::Shutdown, 0, None);
    trace
}
fn decode(trace: &Trace, memory: bool) -> std::result::Result<(Option<String>, Vec<Reading>), Failure> {
    parse(&serde_json::to_vec(trace).unwrap(), UUID, memory)
}
fn source(memory: bool) -> Source {
    Source { id: if memory { "memory" } else { "power" }.into(),
        target: if memory { Target::NvidiaMemory { uuid: UUID.into(), rank: Some(0) } }
            else { Target::NvidiaPower { uuid: UUID.into(), rank: Some(0) } },
        ownership: Ownership::SharedMemory { group: "synthetic-shared-device".into() } }
}
fn snapshot(source: &Source, trace: &Trace, time: u64) -> Snapshot {
    let raw = Raw { name: "nvml.json".into(), bytes: serde_json::to_vec(trace).unwrap(), error: None };
    let parsed = resources::parse_snapshot(source, std::slice::from_ref(&raw), None);
    let (incarnation, readings, failure) = match parsed {
        Ok((identity, readings)) => (identity, readings, None), Err(error) => (None, vec![], Some(error)),
    };
    Snapshot { source: source.id.clone(), clock: "fixture-clock".into(), started_us: time,
        observed_us: time, incarnation, raw: vec![raw], readings, failure }
}
fn observed(sources: Vec<Source>, traces: &[Trace]) -> Observation {
    let snapshots: Vec<_> = [1000, 2000].into_iter().flat_map(|t|
        sources.iter().zip(traces).map(move |(s, trace)| snapshot(s, trace, t))).collect();
    let raw_bytes = snapshots.iter().flat_map(|s| &s.raw).map(|r| r.bytes.len() as u64).sum();
    let config = ResourcesConfig { version: 1, sources, cadence_us: 1000, max_gap_us: 1000,
        max_read_us: 100, max_samples: 10, deadline_us: 10_000, per_sample_bytes: 32768, raw_total_bytes: 131072 };
    Observation { version: 1, kind: "resource-observation-v1".into(), provenance: Provenance::Imported,
        adapter: resources::native_adapter(&config).into(), binary_sha256: "1".repeat(64), config,
        clock: Clock { id: "fixture-clock".into(), kind: ClockKind::LinuxMonotonic,
            unit: ClockUnit::Microseconds, resolution_ns: 1000, synchronization: Synchronization::LocalOrigin },
        clk_tck: 100, observer_started_us: 0, observer_settled_us: 2000,
        measured: MeasuredInterval { clock: "fixture-clock".into(), started_us: 1000, settled_us: 2000 },
        snapshots, failures: vec![], raw_bytes, read_overhead_us: 0 }
}
fn gate(memory: bool) -> Gate {
    Gate { source: if memory { "memory" } else { "power" }.into(),
        metric: if memory { Metric::MemoryUsed } else { Metric::Power },
        statistic: if memory { Statistic::SampledMaximum } else { Statistic::SampledEnergyEstimate },
        max_regression_bps: 0, max_reference_spread_bps: 0 }
}

#[test]
fn exact_memory_bytes_and_instant_power_milliwatt_conversion() {
    let (_, memory) = decode(&fixture(true), true).unwrap();
    assert_eq!(memory, vec![reading(Metric::MemoryUsed, Unit::Bytes, Ok(2048)),
        reading(Metric::MemoryFree, Unit::Bytes, Ok(6144)), reading(Metric::MemoryLimit, Unit::Bytes, Ok(8192))]);
    let o = observed(vec![source(false)], &[fixture(false)]);
    validate_observation(&o).unwrap();
    assert_eq!(summarize(&o, &gate(false)).value, Some((125123, 1).into()));
    let mut t = fixture(false);
    let Returned::Called { value: Some(Payload::Power(p)), .. } = &mut t.calls[5].result else { panic!() };
    p.value_type = 3; p.value = Some(Scalar::UnsignedLongLong(u64::MAX));
    assert_eq!(decode(&t, false), Err(Failure::Overflow));
}

#[test]
fn unsupported_memory_does_not_falsify_power_or_missing_ranks() {
    let mut memory = fixture(true);
    memory.calls.truncate(6);
    memory.calls[5].result = Returned::Called { code: 3, value: None };
    memory.push(Api::Shutdown, 0, None);
    let o = observed(vec![source(true), source(false)], &[memory, fixture(false)]);
    validate_observation(&o).unwrap();
    assert_eq!(summarize(&o, &gate(true)).unavailable, Some(Failure::UnsupportedMetric));
    assert_eq!(summarize(&o, &gate(false)).value, Some((125123, 1).into()));
    let mut missing = gate(false); missing.source = "rank-not-observed".into();
    assert_eq!(summarize(&o, &missing).unavailable, Some(Failure::Missing));
    let mut unknown = o; unknown.config.sources[1].ownership = Ownership::Unknown;
    assert_eq!(summarize(&unknown, &gate(false)).unavailable, Some(Failure::Ownership));
}

#[test]
fn native_admission_requires_full_uuid_and_device_scope() {
    let mut config = observed(vec![source(false)], &[fixture(false)]).config;
    config.validate().unwrap();
    for bad in ["0", "GPU-0", "MIG-00000000-0000-0000-0000-000000000001", "GPU-00000000-0000-0000-0000-00000000000z"] {
        config.sources[0].target = Target::NvidiaPower { uuid: bad.into(), rank: None };
        assert!(config.validate().is_err());
    }
    config.sources[0] = source(false);
    config.sources[0].ownership = Ownership::ProcessAddressSpace;
    assert!(config.validate().is_err());
    let parsed: Target = serde_json::from_str(&format!(r#"{{"kind":"nvidia_power","uuid":"{UUID}"}}"#)).unwrap();
    assert_eq!(parsed, Target::NvidiaPower { uuid: UUID.into(), rank: None });
    assert!(serde_json::from_str::<Target>(r#"{"kind":"nvidia_power","rank":0}"#).is_err());
    assert!(serde_json::from_str::<Target>(&format!(r#"{{"kind":"nvidia_power","uuid":"{UUID}","index":0}}"#)).is_err());
}

#[test]
fn raw_api_errors_have_no_zero_substitution_and_replay_requires_shutdown() {
    for (api, code, expected) in [(Api::Init, 9, Failure::Missing), (Api::Handle, 4, Failure::Permission),
        (Api::Power, 15, Failure::SourceChanged), (Api::Power, 999, Failure::Io), (Api::Shutdown, 10, Failure::Deadline)] {
        let mut trace = fixture(false);
        let i = trace.calls.iter().position(|c| c.api == api).unwrap();
        trace.calls.truncate(i + 1);
        trace.calls[i].result = Returned::Called { code, value: None };
        if api != Api::Init && api != Api::Shutdown { trace.push(Api::Shutdown, 0, None); }
        assert_eq!(decode(&trace, false), Err(expected));
    }
    let mut trace = fixture(false);
    trace.calls.pop();
    assert_eq!(decode(&trace, false), Err(Failure::Malformed));
    let mut trace = fixture(false);
    trace.calls.truncate(6); trace.calls[5].result = Returned::MissingSymbol;
    trace.push(Api::Shutdown, 0, None);
    assert_eq!(decode(&trace, false), Err(Failure::UnsupportedMetric));
    trace.calls.insert(6, fixture(false).calls[6].clone());
    assert_eq!(decode(&trace, false), Err(Failure::Malformed));
}

#[test]
fn field_status_type_scope_and_memory_consistency_are_independently_replayed() {
    let mutate = |update: fn(&mut PowerValue)| {
        let mut t = fixture(false);
        let Returned::Called { value: Some(Payload::Power(p)), .. } = &mut t.calls[5].result else { panic!() };
        update(p); decode(&t, false)
    };
    assert_eq!(mutate(|p| { p.code = 3; p.value = None; }), Err(Failure::UnsupportedMetric));
    assert_eq!(mutate(|p| p.code = 3), Err(Failure::Malformed));
    assert_eq!(mutate(|p| p.field_id = 185), Err(Failure::Malformed));
    assert_eq!(mutate(|p| p.scope_id = 1), Err(Failure::Malformed));
    assert_eq!(mutate(|p| p.value_type = 0), Err(Failure::Malformed));
    assert_eq!(mutate(|p| { p.value_type = 0; p.value = Some(Scalar::DoubleBits(f64::NAN.to_bits())); }), Err(Failure::UnsupportedMetric));
    assert_eq!(mutate(|p| p.timestamp_us = 0), Err(Failure::Malformed));
    let mut t = fixture(true);
    t.calls[5].result = Returned::Called { code: 0, value: Some(Payload::Memory(MemoryInfo { total: 8192, free: 8192, used: 1 })) };
    assert_eq!(decode(&t, true), Err(Failure::Malformed));
    t.calls[5].result = Returned::Called { code: 0, value: Some(Payload::Memory(MemoryInfo { total: 0, free: 0, used: 0 })) };
    assert_eq!(decode(&t, true), Err(Failure::Malformed));
}

#[test]
fn observed_uuid_and_runtime_changes_cannot_replay_as_same_source() {
    let mut t = fixture(false);
    t.calls[6].result = Returned::Called { code: 0, value: Some(buffer(OTHER_UUID, 96)) };
    assert_eq!(decode(&t, false), Err(Failure::SourceChanged));
    let base = fixture(false);
    let mut changed = fixture(false);
    changed.calls[2].result = Returned::Called { code: 0, value: Some(buffer("different-driver", 80)) };
    let mut o = observed(vec![source(false)], std::slice::from_ref(&base));
    o.snapshots[1] = snapshot(&source(false), &changed, 2000);
    o.raw_bytes = o.snapshots.iter().flat_map(|s| &s.raw).map(|r| r.bytes.len() as u64).sum();
    validate_observation(&o).unwrap();
    assert_eq!(summarize(&o, &gate(false)).unavailable, Some(Failure::SourceChanged));
    let mut forged = observed(vec![source(false)], &[base]);
    forged.snapshots[0].readings[0].value = Value::Observed { value: 0 };
    assert!(validate_observation(&forged).is_err());
    let mut t = fixture(false); t.abi_sha256 = "0".repeat(64);
    assert_eq!(decode(&t, false), Err(Failure::Malformed));
}

#[test]
fn bounded_missing_library_and_budget_paths_never_load_nvidia() {
    // Absolute private nonexistent filename: no search-path fallback to NVIDIA.
    let temp = Temp::new();
    let path = CString::new(temp.0.join("absent-fixture.so").as_os_str().as_encoded_bytes()).unwrap();
    let error = match Library::open(&path) { Err(error) => error, Ok(_) => panic!("nonexistent library opened") };
    let mut source = NvmlSource { library: Some(Err(error)) };
    let raw = source.read(UUID, false, TRACE_CAP);
    assert!(raw.error.is_none());
    assert_eq!(parse(&raw.bytes, UUID, false), Err(Failure::Missing));
    let mut source = NvmlSource::default();
    let raw = source.read(UUID, false, TRACE_CAP - 1);
    assert_eq!(raw.error, Some(Failure::ByteBudget));
    assert!(source.library.is_none());
}

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!("grill-nvml-fixture-{}-{}", std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
        std::fs::create_dir(&path).unwrap(); Self(path)
    }
}
impl Drop for Temp { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }

#[test]
fn cpu_shared_library_exercises_c_abi_without_driver_or_devices() {
    let temp = Temp::new();
    let so = temp.0.join("synthetic-nvml.so");
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/nvml.c");
    let output = std::process::Command::new("cc").args(["-std=c11", "-shared", "-fPIC", "-o"])
        .arg(&so).arg(fixture).output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let path = CString::new(so.as_os_str().as_encoded_bytes()).unwrap();
    let library = Library::open(&path).unwrap_or_else(|e| panic!("fixture load: {e:?}"));
    // The sampler moves between executor threads; initialization must remain
    // balanced and the owned library usable after a cross-thread read.
    let (library, memory) = std::thread::spawn(move || {
        let mut memory = Trace::new();
        library.collect(UUID, true, &mut memory);
        (library, memory)
    }).join().unwrap();
    assert_eq!(decode(&memory, true), Err(Failure::UnsupportedMetric));
    let mut power = Trace::new(); library.collect(UUID, false, &mut power);
    assert_eq!(decode(&power, false).unwrap().1,
        vec![reading(Metric::Power, Unit::Microwatts, Ok(125123000))]);
    let mut wrong_uuid = Trace::new(); library.collect(OTHER_UUID, false, &mut wrong_uuid);
    assert_eq!(decode(&wrong_uuid, false), Err(Failure::Missing));
}

#[test]
fn gpu_import_stays_imported_and_adapter_identity_is_source_specific() {
    let temp = Temp::new();
    let mut observation = observed(vec![source(false)], &[fixture(false)]);
    // A caller-supplied native claim is synthetic; import must downgrade it.
    observation.provenance = Provenance::NativeObserved;
    validate_observation(&observation).unwrap();
    observation.adapter = resources::ADAPTER.into();
    assert!(validate_observation(&observation).is_err());
    observation.adapter = ADAPTER.into();
    let capture = Capture { version: 1, study_sha256: "2".repeat(64), acquisition_id: "fixture".into(),
        role: Role::A, phase: Phase::Measured, index: 0, started_unix_ms: 0, settled_unix_ms: 1, observation };
    let input = temp.0.join("input.json");
    std::fs::write(&input, serde_json::to_vec(&capture).unwrap()).unwrap();
    let out = temp.0.join("imported");
    resources::import(&input, &out).unwrap();
    assert_eq!(load(&out).unwrap().observation.provenance, Provenance::Imported);
    let mut config = capture.observation.config;
    config.sources.push(Source { id: "host".into(), target: Target::HostMemory, ownership: Ownership::HostShared });
    assert_eq!(resources::native_adapter(&config), "linux-proc-cgroup-nvml-resource-v1");
    config.sources.remove(0);
    assert_eq!(resources::native_adapter(&config), resources::ADAPTER);
}
