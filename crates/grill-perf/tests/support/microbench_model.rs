use super::*;

fn sample(index: u32, rank: Option<u32>, repetition: Option<u32>, nanos: u64) -> Sample {
    Sample {
        index,
        rank,
        repetition,
        raw: nanos.to_string(),
        duration_ns: nanos,
    }
}

fn cpu_artifact(samples: Vec<Sample>) -> Artifact {
    Artifact {
        kind: KIND.into(),
        version: VERSION,
        adapter: AdapterId::CpuSumU64Reference,
        revision: "cpu-sum-u64-reference-v1".into(),
        operation: frozen_operation(AdapterId::CpuSumU64Reference),
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
        clock: required_clock(AdapterId::CpuSumU64Reference),
        execution: Execution {
            warmups: 2,
            iterations: 5,
            deadline_ms: 5000,
            work_units: 28672,
            memory_bytes: 32768,
            completed: true,
            timed_out: false,
            observed_fallback: None,
        },
        samples,
        correctness: Correctness::ExactSum {
            reference: SUM_REFERENCE_ID.into(),
            computed: reference_sum(4096).to_string(),
            bound: reference_sum(4096).to_string(),
            passed: true,
        },
        failures: Vec::new(),
    }
}

fn rank_artifact(values: &[[u64; 2]; 5]) -> Artifact {
    let mut artifact = cpu_artifact(Vec::new());
    artifact.adapter = AdapterId::NcclAllreduceSum;
    artifact.operation = frozen_operation(AdapterId::NcclAllreduceSum);
    artifact.clock = required_clock(AdapterId::NcclAllreduceSum);
    artifact.topology = Topology {
        scope: Scope::Ranks,
        world: Some(2),
        ranks: vec![
            RankRef {
                index: 0,
                device: Some(0),
            },
            RankRef {
                index: 1,
                device: Some(1),
            },
        ],
    };
    artifact.execution.warmups = 1;
    artifact.execution.work_units = 1_572_864;
    artifact.execution.memory_bytes = 6_291_456;
    artifact.program_sha256 = Some("a".repeat(64));
    artifact.kernel_revision = Some("fixture".into());
    artifact.correctness = Correctness::ExactReduction {
        reference: REDUCE_REFERENCE_ID.into(),
        mismatches: 0,
        tolerance_elems: 0,
        finite: true,
        passed: true,
    };
    let mut index = 0;
    for (repetition, pair) in values.iter().enumerate() {
        for (rank, nanos) in pair.iter().enumerate() {
            artifact.samples.push(sample(
                index,
                Some(rank as u32),
                Some(repetition as u32),
                *nanos,
            ));
            index += 1;
        }
    }
    artifact
}

#[test]
fn exact_decimal_is_bounded_and_never_invents_precision() {
    assert_eq!(
        exact_decimal("4096"),
        Some(Rational {
            numerator: 4096,
            denominator: 1
        })
    );
    assert_eq!(
        exact_decimal("1.25"),
        Some(Rational {
            numerator: 125,
            denominator: 100
        })
    );
    assert_eq!(
        exact_decimal("1."),
        Some(Rational {
            numerator: 1,
            denominator: 1
        })
    );
    for rejected in ["", ".5", "-1", "+1", "1e3", "1.2.3", "1.0000000000", " 1", "0x10"] {
        assert_eq!(exact_decimal(rejected), None, "{rejected}");
    }
    assert_eq!(duration_ns("12.345", Units::Milliseconds), Some(12_345_000));
    // Finer than the declared nanosecond scale is rejected, never rounded.
    assert_eq!(duration_ns("0.0000001", Units::Milliseconds), None);
    assert_eq!(duration_ns("1.2345678901", Units::Nanoseconds), None);
    assert_eq!(duration_ns("0", Units::Nanoseconds), Some(0));
    assert_eq!(
        duration_ns("1", Units::Nanoseconds),
        Some(1)
    );
}

#[test]
fn operation_admission_pins_frozen_cells_and_declared_collective_world() {
    assert!(operation_admitted(
        AdapterId::CpuSumU64Reference,
        &frozen_operation(AdapterId::CpuSumU64Reference)
    ));
    let mut cpu = frozen_operation(AdapterId::CpuSumU64Reference);
    if let Operation::SumU64 { bound, .. } = &mut cpu {
        *bound = 1024;
    }
    assert!(!operation_admitted(AdapterId::CpuSumU64Reference, &cpu));

    let mut e3 = frozen_operation(AdapterId::Exl3E3Grouped);
    if let Operation::Exl3Experts { cap, .. } = &mut e3 {
        *cap = 64;
    }
    assert!(!operation_admitted(AdapterId::Exl3E3Grouped, &e3));

    let declared = Operation::AllReduce {
        op: ReduceOp::Sum,
        dtype: Dtype::Int64,
        numel: 262_144,
        world: 3,
        ranks: vec![0, 1, 2],
        input: REDUCE_INPUT_ID.into(),
        reference: REDUCE_REFERENCE_ID.into(),
    };
    assert!(operation_admitted(AdapterId::NcclAllreduceSum, &declared));
    if let Operation::AllReduce { ranks, .. } = &declared {
        let mut wrong = declared.clone();
        if let Operation::AllReduce { ranks: slot, .. } = &mut wrong {
            *slot = ranks.iter().rev().copied().collect();
        }
        assert!(!operation_admitted(AdapterId::NcclAllreduceSum, &wrong));
    }
    let single = Operation::AllReduce {
        op: ReduceOp::Sum,
        dtype: Dtype::Int64,
        numel: 262_144,
        world: 1,
        ranks: vec![0],
        input: REDUCE_INPUT_ID.into(),
        reference: REDUCE_REFERENCE_ID.into(),
    };
    assert!(!operation_admitted(AdapterId::NcclAllreduceSum, &single));
}

#[test]
fn byte_definitions_keep_payload_and_bus_normalization_distinct() {
    let operation = frozen_operation(AdapterId::NcclAllreduceSum);
    assert_eq!(algorithm_payload_bytes(&operation), Some(2_097_152));
    assert_eq!(
        bus_normalization(2),
        Some(Rational {
            numerator: 1,
            denominator: 1
        })
    );
    assert_eq!(
        bus_normalization(3),
        Some(Rational {
            numerator: 4,
            denominator: 3
        })
    );
    assert_eq!(bus_normalization(1), None);
    let derived = derived(
        &operation,
        &required_clock(AdapterId::NcclAllreduceSum),
        Rational {
            numerator: 1_048_576,
            denominator: 1,
        },
    )
    .expect("rank operation derives bandwidth");
    // 2097152 bytes over 1048576 ns is exactly 2e9 bytes per second.
    assert_eq!(derived.payload_bytes, 2_097_152);
    assert_eq!(
        derived.algorithm_bandwidth_bps,
        Rational {
            numerator: 2_000_000_000,
            denominator: 1
        }
    );
    assert_eq!(
        derived.bus_normalization,
        Rational {
            numerator: 1,
            denominator: 1
        }
    );
    // The bus bandwidth is the labelled normalization applied to the algorithm
    // bandwidth, not a separately measured link number.
    assert_eq!(
        derived.bus_bandwidth_bps,
        derived.algorithm_bandwidth_bps
    );
    assert!(derived.bus_scope.contains("not measured physical link traffic"));
}

#[test]
fn rank_statistic_is_the_mean_of_complete_per_repetition_maxima() {
    let artifact = rank_artifact(&[
        [100, 400],
        [200, 300],
        [100, 100],
        [500, 100],
        [100, 100],
    ]);
    // (400 + 300 + 100 + 500 + 100) / 5 reduces to 280.
    assert_eq!(
        statistic(&artifact),
        Some(Rational {
            numerator: 280,
            denominator: 1
        })
    );
    let mut short = artifact.clone();
    short.samples.pop();
    assert_eq!(statistic(&short), None);
    let mut duplicated = artifact.clone();
    duplicated.samples[0].rank = Some(1);
    assert!(!check_artifact(None, &duplicated).invalid.is_empty());
}

#[test]
fn e3_bounds_are_recomputed_from_the_frozen_tolerances() {
    let ref_max = exact_decimal("2.0").unwrap();
    let e2 = [
        exact_decimal("0.001000000").unwrap(),
        exact_decimal("0.001000000").unwrap(),
        exact_decimal("0.001000000").unwrap(),
        exact_decimal("0.010000000").unwrap(),
    ];
    let bounds = e3_bounds(E3_TOLERANCE, ref_max, &e2).expect("bounds");
    // floor = 1e-3 * ref_max = 0.002; maxabs bound = 1.5 * 0.001 + 0.002 = 0.0035
    assert_eq!(display(bounds[0]), 0.0035);
    // nrmse bound = 1.5 * 0.01 + 1e-4 = 0.0151
    assert_eq!(display(bounds[3]), 0.0151);
    // coarse = max(0.15, 0.08 * 2.0) = 0.16
    assert_eq!(display(bounds[4]), 0.16);
    // Coarse uses max(1.0, ref_max): a sub-unit reference keeps the 0.15 floor.
    let tiny = e3_bounds(E3_TOLERANCE, exact_decimal("0.5").unwrap(), &e2).expect("bounds");
    assert_eq!(display(tiny[4]), 0.15);
    // A widened tolerance changes the recomputed bound, so retained evidence
    // cannot quietly carry a relaxed contract.
    let widened = Tolerance::E3Rel {
        factor_milli: 2000,
        abs_rel_micro: 1000,
        nrmse_abs_micro: 100,
        coarse_abs_milli: 150,
        coarse_factor_milli: 80,
    };
    let wide = e3_bounds(widened, ref_max, &e2).expect("bounds");
    assert_eq!(display(wide[0]), 0.004);
}

#[test]
fn cpu_artifact_validation_accepts_the_reference_and_rejects_drift() {
    let samples = vec![
        sample(0, None, None, 160),
        sample(1, None, None, 100),
        sample(2, None, None, 120),
        sample(3, None, None, 140),
        sample(4, None, None, 180),
    ];
    let artifact = cpu_artifact(samples);
    assert!(check_artifact(None, &artifact).clean());
    // (160 + 100 + 120 + 140 + 180) / 5 reduces to 140.
    assert_eq!(
        statistic(&artifact),
        Some(Rational {
            numerator: 140,
            denominator: 1
        })
    );

    let mut drifted = artifact.clone();
    drifted.correctness = Correctness::ExactSum {
        reference: SUM_REFERENCE_ID.into(),
        computed: "8386561".into(),
        bound: "8386560".into(),
        passed: false,
    };
    assert!(
        check_artifact(None, &drifted)
            .invalid
            .contains(&Reason::CorrectnessFailed)
    );

    let mut unsynchronized = artifact.clone();
    unsynchronized.clock.synchronization = Synchronization::CollectiveBarrierBeforeAfter;
    assert!(
        check_artifact(None, &unsynchronized)
            .invalid
            .contains(&Reason::ClockMismatch)
    );

    let mut rounded = artifact.clone();
    rounded.clock.units = Units::Milliseconds;
    rounded.clock.resolution_ns = 1000;
    rounded.samples[0].raw = "0.00016".into();
    assert!(
        check_artifact(None, &rounded)
            .invalid
            .contains(&Reason::SamplePrecision)
    );

    let mut declared = artifact.clone();
    declared.provenance = Provenance::Declared;
    assert!(
        check_artifact(None, &declared)
            .invalid
            .contains(&Reason::InvalidArtifact)
    );
}
