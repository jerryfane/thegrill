use super::*;

fn config(source: Source) -> ResourcesConfig {
    ResourcesConfig {
        version: 1,
        sources: vec![source],
        cadence_us: 1000,
        max_gap_us: 1_000_000,
        max_read_us: 10_000,
        max_samples: 100,
        deadline_us: 10_000_000,
        per_sample_bytes: 16_384,
        raw_total_bytes: 1_048_576,
    }
}
fn process_source() -> Source {
    Source {
        id: "process".into(),
        target: Target::Process { pid: 7 },
        ownership: Ownership::ProcessAddressSpace,
    }
}
fn clock() -> Clock {
    Clock {
        id: "fixture-clock".into(),
        kind: ClockKind::LinuxMonotonic,
        unit: ClockUnit::Microseconds,
        resolution_ns: 1000,
        synchronization: Synchronization::LocalOrigin,
    }
}
fn stat(start: u64, cpu: u64) -> Vec<u8> {
    let mut fields = vec!["0".to_owned(); 50];
    fields[0] = "R".into();
    fields[11] = cpu.to_string();
    fields[19] = start.to_string();
    // Guest is already included in utime, and children are not process CPU time.
    fields[13] = "9000".into();
    fields[14] = "8000".into();
    fields[40] = "777".into();
    format!("7 (name ) with spaces)) {}\n", fields.join(" ")).into_bytes()
}
fn raw(name: &str, bytes: Vec<u8>) -> Raw {
    Raw {
        name: name.into(),
        bytes,
        error: None,
    }
}
fn process_snapshot(t: u64, start: u64, cpu: u64, rss_kb: u64) -> Snapshot {
    let source = process_source();
    let raw = vec![
        raw("stat", stat(start, cpu)),
        raw(
            "status",
            format!("VmRSS:\t{rss_kb} kB\nVmHWM:\t{rss_kb} kB\n").into_bytes(),
        ),
        raw("stat_after", stat(start, cpu)),
    ];
    let (incarnation, readings) = parse_snapshot(&source, &raw, None).unwrap();
    Snapshot {
        source: source.id,
        clock: clock().id,
        started_us: t,
        observed_us: t,
        incarnation,
        raw,
        readings,
        failure: None,
    }
}
fn observation(source: Source, snapshots: Vec<Snapshot>) -> Observation {
    let end = snapshots.last().unwrap().observed_us;
    let start = snapshots[0].observed_us;
    let raw_bytes = snapshots
        .iter()
        .flat_map(|s| &s.raw)
        .map(|r| r.bytes.len() as u64)
        .sum();
    Observation {
        version: 1,
        kind: "resource-observation-v1".into(),
        provenance: Provenance::Imported,
        adapter: ADAPTER.into(),
        binary_sha256: "1".repeat(64),
        config: config(source),
        clock: clock(),
        clk_tck: 100,
        observer_started_us: 0,
        observer_settled_us: end,
        measured: MeasuredInterval {
            clock: clock().id,
            started_us: start,
            settled_us: end,
        },
        snapshots,
        failures: Vec::new(),
        raw_bytes,
        read_overhead_us: 0,
    }
}
fn gate(metric: Metric, statistic: Statistic) -> Gate {
    Gate {
        source: "process".into(),
        metric,
        statistic,
        max_regression_bps: 100,
        max_reference_spread_bps: 0,
    }
}
fn process_observation() -> Observation {
    observation(
        process_source(),
        vec![
            process_snapshot(1000, 100, 10, 10),
            process_snapshot(2000, 100, 13, 20),
        ],
    )
}
#[test]
fn proc_final_parenthesis_guest_children_and_overflow() {
    assert_eq!(proc_stat(&stat(123, 41), 7), Ok((123, 41)));
    assert_eq!(proc_stat(&stat(123, 41), 8), Err(Failure::SourceChanged));
    assert_eq!(proc_stat(b"7 (bad) R 0\n", 7), Err(Failure::Malformed));
    assert_eq!(number("18446744073709551616"), Err(Failure::Overflow));
    assert_eq!(
        field(b"VmRSS: 18446744073709551615 kB\n", "VmRSS:", true),
        Err(Failure::Overflow)
    );
    assert_eq!(
        field(b"VmRSS: 1 MB\n", "VmRSS:", true),
        Err(Failure::Malformed)
    );
    assert_eq!(
        field(b"VmRSS: 1 kB\nVmRSS: 2 kB\n", "VmRSS:", true),
        Err(Failure::Malformed)
    );
    let source = Source {
        id: "host".into(),
        target: Target::HostCpu,
        ownership: Ownership::HostShared,
    };
    let (_, readings) = parse_snapshot(
        &source,
        &[raw(
            "stat",
            b"cpu 10 20 30 400 500 60 70 80 900 1000\nbtime 42\n".to_vec(),
        )],
        None,
    )
    .unwrap();
    assert_eq!(readings[0].value, Value::Observed { value: 270 });
}
#[test]
fn exact_cpu_denominator_and_source_specific_boundaries() {
    let mut o = process_observation();
    validate_observation(&o).unwrap();
    assert_eq!(
        summary_value(&o, &gate(Metric::CpuTime, Statistic::CpuTime)),
        Ok((30_000, 1).into())
    );
    // 30 logical CPUs is not clamped to one or divided by machine cores.
    assert_eq!(
        summary_value(&o, &gate(Metric::CpuTime, Statistic::CpuUtilizationOneCpu)),
        Ok((30, 1).into())
    );
    assert_eq!(
        summary_value(&o, &gate(Metric::ResidentMemory, Statistic::SampledMaximum)),
        Ok((20 * 1024, 1).into())
    );
    o.measured.started_us += 1;
    assert_eq!(
        summary_value(&o, &gate(Metric::CpuTime, Statistic::CpuTime)),
        Err(Failure::UnsupportedBoundary)
    );
    assert_eq!(
        summary_value(
            &o,
            &gate(Metric::LifetimePeakMemory, Statistic::SampledMaximum)
        ),
        Err(Failure::UnsupportedMetric)
    );
    o.measured.started_us = 0;
    assert_eq!(
        summary_value(&o, &gate(Metric::ResidentMemory, Statistic::SampledMaximum)),
        Err(Failure::IncompleteExposure)
    );
}
#[test]
fn pid_reuse_between_and_within_samples_and_counter_reset() {
    let changed = observation(
        process_source(),
        vec![
            process_snapshot(1000, 100, 10, 10),
            process_snapshot(2000, 101, 11, 20),
        ],
    );
    assert_eq!(
        summary_value(
            &changed,
            &gate(Metric::ResidentMemory, Statistic::SampledMaximum)
        ),
        Err(Failure::SourceChanged)
    );
    let reset = observation(
        process_source(),
        vec![
            process_snapshot(1000, 100, 10, 10),
            process_snapshot(2000, 100, 9, 20),
        ],
    );
    assert_eq!(
        summary_value(
            &reset,
            &gate(Metric::ResidentMemory, Statistic::SampledMaximum)
        ),
        Err(Failure::CounterReset)
    );
    let mut raw = process_snapshot(1000, 100, 10, 10).raw;
    raw[2].bytes = stat(101, 10);
    assert_eq!(
        parse_snapshot(&process_source(), &raw, None),
        Err(Failure::SourceChanged)
    );
    raw[2].bytes = stat(100, 9);
    assert_eq!(
        parse_snapshot(&process_source(), &raw, None),
        Err(Failure::CounterReset)
    );
}
#[test]
fn source_gaps_cancellation_permission_and_budget_never_qualify() {
    let mut o = process_observation();
    let g = gate(Metric::ResidentMemory, Statistic::SampledMaximum);
    o.config.max_gap_us = 1000;
    assert!(summary_value(&o, &g).is_ok());
    o.snapshots[1].observed_us += 1;
    o.observer_settled_us += 1;
    assert_eq!(summary_value(&o, &g), Err(Failure::Gap));
    let mut o = process_observation();
    o.snapshots[1].raw[1].error = Some(Failure::Permission);
    o.snapshots[1].readings = parse_snapshot(&process_source(), &o.snapshots[1].raw, None)
        .unwrap()
        .1;
    validate_observation(&o).unwrap();
    assert_eq!(summary_value(&o, &g), Err(Failure::Permission));
    for failure in [
        Failure::Cancelled,
        Failure::SampleBudget,
        Failure::ByteBudget,
        Failure::Deadline,
    ] {
        let mut o = process_observation();
        o.failures.push(failure);
        assert_eq!(summary_value(&o, &g), Err(failure));
    }
}
#[test]
fn clock_and_raw_replay_reject_forged_observations() {
    let mut o = process_observation();
    o.snapshots[1].clock = "other-origin".into();
    assert!(validate_observation(&o).is_err());
    let mut o = process_observation();
    o.snapshots[1].readings[0].unit = Unit::Bytes;
    assert!(validate_observation(&o).is_err());
    let mut o = process_observation();
    o.snapshots[1].readings[1].value = Value::Observed { value: 1 };
    assert!(validate_observation(&o).is_err());
    let mut c = clock();
    c.id = "another-acquisition".into();
    assert!(compatible_clocks(&clock(), &c));
    c.resolution_ns = 2000;
    assert!(!compatible_clocks(&clock(), &c));
    c = clock();
    c.synchronization = Synchronization::Unrelated;
    assert!(!compatible_clocks(&clock(), &c));
}
#[test]
fn cgroup_unlimited_missing_and_microsecond_cpu_are_distinct() {
    let source = Source {
        id: "group".into(),
        target: Target::CgroupV2 {
            path: "/sys/fs/cgroup/fixture".into(),
        },
        ownership: Ownership::CgroupMembers,
    };
    let mut raw = vec![
        raw("memory.current", b"1024\n".to_vec()),
        raw("memory.peak", b"2048\n".to_vec()),
        raw("memory.max", b"max\n".to_vec()),
        raw(
            "cpu.stat",
            b"usage_usec 1500\nuser_usec 1000\nsystem_usec 500\n".to_vec(),
        ),
    ];
    let (_, readings) = parse_snapshot(&source, &raw, Some("directory:1:2")).unwrap();
    assert_eq!(readings[2].value, Value::Unlimited);
    assert_eq!(
        readings[3],
        reading(Metric::CpuTime, Unit::Microseconds, Ok(1500))
    );
    raw[2].bytes.clear();
    raw[2].error = Some(Failure::Missing);
    let (_, readings) = parse_snapshot(&source, &raw, Some("directory:1:2")).unwrap();
    assert_eq!(
        readings[2].value,
        Value::Unavailable {
            reason: Failure::Missing
        }
    );
    raw[2].error = None;
    raw[2].bytes = b"4096\n".to_vec();
    assert_eq!(
        parse_snapshot(&source, &raw, Some("directory:1:2"))
            .unwrap()
            .1[2]
            .value,
        Value::Observed { value: 4096 }
    );
}

#[test]
fn cgroup_integral_and_reset_boundaries_replay_independently() {
    let source = Source {
        id: "process".into(),
        target: Target::CgroupV2 {
            path: "/sys/fs/cgroup/fixture".into(),
        },
        ownership: Ownership::CgroupMembers,
    };
    let make = |t: u64, cpu: u64, peak: u64| {
        let raw = vec![
            raw("memory.current", b"1024\n".to_vec()),
            raw("memory.peak", format!("{peak}\n").into_bytes()),
            raw("memory.max", b"max\n".to_vec()),
            raw("cpu.stat", format!("usage_usec {cpu}\n").into_bytes()),
        ];
        let (incarnation, readings) = parse_snapshot(&source, &raw, Some("directory:1:2")).unwrap();
        Snapshot {
            source: source.id.clone(),
            clock: clock().id,
            started_us: t,
            observed_us: t,
            incarnation,
            readings,
            raw,
            failure: None,
        }
    };
    let o = observation(
        source.clone(),
        vec![make(1000, 1000, 2048), make(2000, 2501, 2048)],
    );
    validate_observation(&o).unwrap();
    assert_eq!(
        summary_value(&o, &gate(Metric::CpuTime, Statistic::CpuTime)),
        Ok((1501, 1).into())
    );
    assert_eq!(
        summary_value(&o, &gate(Metric::CpuTime, Statistic::CpuUtilizationOneCpu)),
        Ok((1501, 1000).into())
    );
    let o = observation(
        source.clone(),
        vec![make(1000, 1000, 2048), make(2000, 2501, 1024)],
    );
    assert_eq!(
        summary_value(&o, &gate(Metric::MemoryUsed, Statistic::SampledMaximum)),
        Err(Failure::CounterReset)
    );
    let mut o = observation(
        source.clone(),
        vec![make(1000, 1000, 2048), make(2000, 2501, 2048)],
    );
    o.snapshots[1].incarnation = Some("directory:1:3".into());
    validate_observation(&o).unwrap();
    assert_eq!(
        summary_value(&o, &gate(Metric::MemoryUsed, Statistic::SampledMaximum)),
        Err(Failure::SourceChanged)
    );
}
fn power_source() -> Source {
    Source {
        id: "process".into(),
        target: Target::Imported {
            adapter: "fixture-power-v1".into(),
            device: Some("synthetic-device".into()),
            rank: Some(0),
        },
        ownership: Ownership::SharedMemory {
            group: "shared-allocation".into(),
        },
    }
}
fn power_snapshot(t: u64, microwatts: u64) -> Snapshot {
    let readings = vec![reading(Metric::Power, Unit::Microwatts, Ok(microwatts))];
    Snapshot {
        source: "process".into(),
        clock: clock().id,
        started_us: t,
        observed_us: t,
        incarnation: Some("source-epoch-1".into()),
        raw: vec![raw("readings.json", serde_json::to_vec(&readings).unwrap())],
        readings,
        failure: None,
    }
}
#[test]
fn sampled_energy_clips_linear_segments_with_exact_rational_arithmetic() {
    let mut o = observation(
        power_source(),
        vec![
            power_snapshot(0, 2_000_000),
            power_snapshot(1_000_000, 4_000_000),
        ],
    );
    let g = gate(Metric::Power, Statistic::SampledEnergyEstimate);
    validate_observation(&o).unwrap();
    assert_eq!(summary_value(&o, &g), Ok((3_000_000, 1).into()));
    o.measured.started_us = 250_000;
    o.measured.settled_us = 750_000;
    assert_eq!(summary_value(&o, &g), Ok((1_500_000, 1).into()));
    let o = observation(
        power_source(),
        vec![power_snapshot(0, 0), power_snapshot(1000, 1)],
    );
    assert_eq!(summary_value(&o, &g), Ok((1, 2000).into()));
    let mut o = o;
    o.snapshots.pop();
    assert_eq!(summary_value(&o, &g), Err(Failure::IncompleteExposure));
}
#[test]
fn energy_overflow_unknown_ownership_and_partial_visibility_are_unavailable() {
    let mut o = observation(
        power_source(),
        vec![
            power_snapshot(0, u64::MAX),
            power_snapshot(2_000_000, u64::MAX),
        ],
    );
    o.config.max_gap_us = 2_000_000;
    let g = gate(Metric::Power, Statistic::SampledEnergyEstimate);
    assert_eq!(summary_value(&o, &g), Err(Failure::Overflow));
    let mut o = observation(
        power_source(),
        vec![power_snapshot(0, 1), power_snapshot(1000, 1)],
    );
    o.config.sources[0].ownership = Ownership::Unknown;
    assert_eq!(summary_value(&o, &g), Err(Failure::Ownership));
    o.config.sources[0].ownership = Ownership::DedicatedDevice;
    o.snapshots[1].failure = Some(Failure::Missing);
    assert_eq!(summary_value(&o, &g), Err(Failure::Missing));
    let mut g = g;
    g.source = "unobserved-rank".into();
    assert_eq!(summary_value(&o, &g), Err(Failure::Missing));
    let mut c = config(process_source());
    c.sources[0].ownership = Ownership::HostShared;
    assert!(c.validate().is_err());
}

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "grill-resource-fixture-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn fixture_study() -> Study {
    Study {
        version: 1,
        id: "resource-envelope-fixture".into(),
        exposure_pin: "2".repeat(64),
        collector_sha256: "1".repeat(64),
        candidate_change: "declared-fixture-memory".into(),
        duration_us: 1000,
        arms: [Role::A, Role::B, Role::A2]
            .into_iter()
            .map(|role| Arm {
                role,
                deployment_pin: if role == Role::B { "4" } else { "3" }.repeat(64),
                config: config(process_source()),
                warmup_ids: vec![format!("{role:?}-warmup")],
                measured_ids: (0..3).map(|i| format!("{role:?}-measured-{i}")).collect(),
            })
            .collect(),
        gates: vec![gate(Metric::ResidentMemory, Statistic::SampledMaximum)],
    }
}
fn write_comparison(
    root: &Path,
    candidate_kb: u64,
    failed_candidate: bool,
) -> (PathBuf, [Vec<PathBuf>; 3]) {
    let plan = fixture_study();
    plan.validate().unwrap();
    let plan_path = root.join("study.json");
    let bytes = serde_json::to_vec(&plan).unwrap();
    evidence::write(&plan_path, &bytes).unwrap();
    let hash = evidence::digest(&bytes);
    let mut groups: [Vec<PathBuf>; 3] = std::array::from_fn(|_| Vec::new());
    for (arm_index, arm) in plan.arms.iter().enumerate() {
        for (ordinal, id) in arm.warmup_ids.iter().chain(&arm.measured_ids).enumerate() {
            let phase = if ordinal == 0 {
                Phase::Warmup
            } else {
                Phase::Measured
            };
            let kb = if arm.role == Role::B {
                candidate_kb
            } else {
                10000
            };
            let mut observation = observation(
                process_source(),
                vec![
                    process_snapshot(1000, 100, 10, kb),
                    process_snapshot(2000, 100, 13, kb),
                ],
            );
            observation.clock.id = id.clone();
            observation.measured.clock = id.clone();
            for s in &mut observation.snapshots {
                s.clock = id.clone();
            }
            if failed_candidate && arm.role == Role::B && ordinal == 2 {
                observation.failures.push(Failure::Cancelled);
            }
            let capture = Capture {
                version: 1,
                study_sha256: hash.clone(),
                acquisition_id: id.clone(),
                role: arm.role,
                phase,
                index: ordinal.saturating_sub(1) as u32,
                started_unix_ms: (arm_index * 8 + ordinal * 2) as u64,
                settled_unix_ms: (arm_index * 8 + ordinal * 2 + 1) as u64,
                observation,
            };
            let out = root.join(id);
            let bytes = serde_json::to_vec(&capture).unwrap();
            save(&out, &capture, Some(&bytes)).unwrap();
            groups[arm_index].push(out);
        }
    }
    (plan_path, groups)
}
#[test]
fn exact_aba2_gate_boundary_and_failure_population() {
    for (candidate, failed, expected) in [
        (10100, false, Outcome::Pass),
        (10101, false, Outcome::Regression),
        (9000, true, Outcome::Inconclusive),
    ] {
        let temp = Temp::new();
        let (plan, groups) = write_comparison(&temp.0, candidate, failed);
        let decision = compare(&plan, [&groups[0], &groups[1], &groups[2]]);
        assert_eq!(decision.decision, expected, "{:?}", decision.errors);
        assert_eq!(
            decision.claim,
            "imported-or-declared-resource-comparison-not-native-execution"
        );
        if !failed {
            assert_eq!(decision.gates[0].counts, [3; 3]);
            assert_eq!(decision.gates[0].warmup_counts, [1; 3]);
        }
    }
    let temp = Temp::new();
    let (plan, mut groups) = write_comparison(&temp.0, 10000, false);
    groups[2].pop();
    assert_eq!(
        compare(&plan, [&groups[0], &groups[1], &groups[2]]).decision,
        Outcome::Inconclusive
    );
    groups[0].swap(1, 2);
    assert_eq!(
        compare(&plan, [&groups[0], &groups[1], &groups[2]]).decision,
        Outcome::Error
    );
}
#[test]
fn import_cannot_promote_a_native_claim_and_digest_corruption_is_rejected() {
    let temp = Temp::new();
    let (plan, groups) = write_comparison(&temp.0, 10000, false);
    let mut captured = load(&groups[0][0]).unwrap();
    captured.observation.provenance = Provenance::NativeObserved;
    let input = temp.0.join("claimed-native.json");
    evidence::json(&input, &captured).unwrap();
    let out = temp.0.join("reimport");
    import(&input, &out).unwrap();
    assert_eq!(
        load(&out).unwrap().observation.provenance,
        Provenance::Imported
    );
    let file = out.join("observation.json");
    std::fs::write(file, b"{}").unwrap();
    assert!(load(&out).is_err());
    assert_eq!(
        compare(&plan, [&groups[0], &groups[1], &groups[2]]).decision,
        Outcome::Pass
    );
}
