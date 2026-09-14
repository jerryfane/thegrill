use super::*;

fn ratio(numerator: u64, denominator: u64) -> Rational {
    Rational {
        numerator,
        denominator,
    }
}

/// A sample whose raw text is the decimal in the clock's units and whose
/// duration is the exact nanosecond rational.
fn sample(index: u32, rank: Option<u32>, repetition: Option<u32>, raw: &str, ns: Rational) -> Sample {
    Sample {
        index,
        rank,
        repetition,
        raw: raw.to_string(),
        duration: ns,
    }
}

fn integer_sample(index: u32, raw: &str, nanos: u64) -> Sample {
    sample(index, None, None, raw, ratio(nanos, 1))
}

fn cpu_artifact(samples: Vec<Sample>) -> Artifact {
    Artifact {
        kind: KIND.into(),
        version: VERSION,
        adapter: AdapterId::CpuSumU64Reference,
        revision: evidence::digest(b"cpu-fixture"),
        acquisition: Acquisition { id: "fixture".into(), started_unix_ms: 1 },
        operation: frozen_operation(AdapterId::CpuSumU64Reference),
        provenance: Provenance::NativeObserved,
        submitted_provenance: None,
        program_sha256: None,
        kernel_revision: Some("cpu-fixture".into()),
        sources: Vec::new(),
        observations: Vec::new(),
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

fn cpu_samples() -> Vec<Sample> {
    vec![
        integer_sample(0, "160", 160),
        integer_sample(1, "100", 100),
        integer_sample(2, "120", 120),
        integer_sample(3, "140", 140),
        integer_sample(4, "180", 180),
    ]
}

/// A device artifact is the easiest way to exercise the millisecond clock and
/// the sub-resolution / fractional-nanosecond sample rules.
fn device_artifact() -> Artifact {
    let mut artifact = cpu_artifact(cpu_samples());
    artifact.adapter = AdapterId::Exl3E3Grouped;
    artifact.operation = frozen_operation(AdapterId::Exl3E3Grouped);
    artifact.clock = required_clock(AdapterId::Exl3E3Grouped);
    artifact.topology = Topology {
        scope: Scope::Device,
        world: None,
        ranks: vec![RankRef {
            index: 0,
            device: Some(0),
        }],
    };
    artifact.execution.warmups = 2;
    // The effective tier must match the declared tier or the measurement is
    // withheld, so a device fixture always reports the grouped fallback.
    artifact.execution.observed_fallback = Some(Tier::E3Grouped);
    artifact.execution.work_units = 206_158_430_208;
    artifact.execution.memory_bytes = 1_073_741_824;
    artifact.program_sha256 = Some("b".repeat(64));
    artifact.kernel_revision = Some("exl3-module:fixture".into());
    artifact.revision = evidence::digest(b"exl3-module:fixture");
    artifact.sources = vec![Source {
        path: "/observed/tests/test_exl3_overlay.py".into(),
        sha256: E3_PARITY_SOURCE_SHA256.into(),
    }];
    artifact.observations = e3_observations("e3-grouped");
    artifact.correctness = Correctness::E3Parity {
        reference: E3_REFERENCE_ID.into(),
        finite: true,
        ref_max: "2.000000000".into(),
        e2: parity("0.001000000", "0.001000000", "0.001000000", "0.010000000"),
        e3: parity("0.003000000", "0.003000000", "0.003000000", "0.015000000"),
        tolerance: E3_TOLERANCE,
        passed: true,
        outcome: CheckOutcome::Pass,
        detail: None,
    };
    artifact.samples = (0..5)
        .map(|index| sample(index, None, None, "31.234", ratio(31_234_000, 1)))
        .collect();
    artifact
}

fn parity(maxabs: &str, per_token_max: &str, per_token_p99: &str, nrmse: &str) -> ParityStats {
    ParityStats {
        maxabs: maxabs.into(),
        per_token_max: per_token_max.into(),
        per_token_p99: per_token_p99.into(),
        nrmse: nrmse.into(),
    }
}

fn e3_observations(fallback: &str) -> Vec<Observation> {
    let fallback = match fallback {
        "e3-grouped" => "grouped",
        "e2-kernel" => "kernel",
        other => other,
    };
    let observed = |name: ObservationName, value: &str| Observation {
        name,
        value: value.into(),
    };
    vec![
        observed(ObservationName::TorchVersion, "2.9.0"),
        observed(ObservationName::NcclVersion, "2.27.7"),
        observed(ObservationName::Exl3ModuleSha256, &"c".repeat(64)),
        observed(ObservationName::Device, "fixture-device"),
        observed(ObservationName::Experts, "288"),
        observed(ObservationName::Topk, "8"),
        observed(ObservationName::Tokens, "1024"),
        observed(ObservationName::Hidden, "4096"),
        observed(ObservationName::Intermediate, "1024"),
        observed(ObservationName::Cap, "32"),
        observed(ObservationName::SkewMilli, "1000"),
        observed(ObservationName::RoutingSeed, "1"),
        observed(ObservationName::LayerSeed, "0"),
        observed(ObservationName::ParityRoutingSeed, "3"),
        observed(ObservationName::ActivationSeed, "3"),
        observed(ObservationName::FallbackTier, fallback),
        observed(ObservationName::SourcesVerified, "1"),
        observed(ObservationName::SourcesDeclared, "0"),
    ]
}

fn rank_artifact(values: &[[u64; 2]; 5]) -> Artifact {
    let mut artifact = device_artifact();
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
    artifact.execution.observed_fallback = None;
    artifact.sources = vec![Source {
        path: "/observed/doc/PERFORMANCE.md".into(),
        sha256: NCCL_BYTE_SOURCE_SHA256.into(),
    }];
    artifact.observations = vec![
        Observation {
            name: ObservationName::TorchVersion,
            value: "2.9.0".into(),
        },
        Observation {
            name: ObservationName::NcclVersion,
            value: "2.27.7".into(),
        },
        Observation {
            name: ObservationName::Device,
            value: "fixture-device".into(),
        },
        Observation {
            name: ObservationName::World,
            value: "2".into(),
        },
        Observation {
            name: ObservationName::Numel,
            value: "262144".into(),
        },
        Observation {
            name: ObservationName::Dtype,
            value: "int64".into(),
        },
        Observation {
            name: ObservationName::Op,
            value: "sum".into(),
        },
        Observation {
            name: ObservationName::SourcesVerified,
            value: "1".into(),
        },
        Observation {
            name: ObservationName::SourcesDeclared,
            value: "0".into(),
        },
    ];
    artifact.correctness = Correctness::ExactReduction {
        reference: REDUCE_REFERENCE_ID.into(),
        mismatches: 0,
        tolerance_elems: 0,
        finite: true,
        passed: true,
    };
    artifact.samples = Vec::new();
    let mut index = 0;
    for (repetition, pair) in values.iter().enumerate() {
        for (rank, nanos) in pair.iter().enumerate() {
            artifact.samples.push(sample(
                index,
                Some(rank as u32),
                Some(repetition as u32),
                &nanos.to_string(),
                ratio(*nanos, 1),
            ));
            index += 1;
        }
    }
    artifact
}

#[test]
fn exact_decimal_accepts_scientific_notation_and_rejects_the_rest() {
    for (text, expected) in [
        ("4096", ratio(4096, 1)),
        ("1.25", ratio(125, 100)),
        ("1.", ratio(1, 1)),
        ("1.234e-05", ratio(1234, 100_000_000)),
        ("1E+3", ratio(1000, 1)),
        ("0.123456789012345", ratio(123_456_789_012_345, 1_000_000_000_000_000)),
    ] {
        let actual = exact_decimal(text).unwrap();
        assert_eq!(
            u128::from(actual.numerator) * u128::from(expected.denominator),
            u128::from(expected.numerator) * u128::from(actual.denominator),
            "{text}"
        );
    }
    for rejected in [
        "", ".5", "-1", "+1", "nan", "inf", "1.2.3", " 1", "0x10", "1e", "1e31", "1e-31",
    ] {
        assert_eq!(exact_decimal(rejected), None, "{rejected}");
    }
    // A long but bounded digit string stays exact; an over-long one is refused.
    assert!(exact_decimal(&format!("1.{}", "0".repeat(30))).is_some());
    assert_eq!(exact_decimal(&format!("1.{}", "0".repeat(32))), None);
}

#[test]
fn durations_keep_fractional_nanoseconds_and_never_round_to_a_resolution() {
    // 31.2345678 ms is not a whole number of nanoseconds: 156172839/5 ns.
    assert_eq!(
        duration_ns("31.2345678", Units::Milliseconds),
        Ok(ratio(156_172_839, 5))
    );
    // The acceptance case: an exact rational from a long decimal.
    assert_eq!(
        duration_ns("0.123456789012345", Units::Milliseconds),
        Ok(ratio(24_691_357_802_469, 200_000_000))
    );
    // Scientific notation from a float representation.
    assert_eq!(duration_ns("1.234e-05", Units::Milliseconds), Ok(ratio(617, 50)));
    assert_eq!(
        duration_ns("0.0000001", Units::Milliseconds),
        Ok(ratio(1, 10))
    );
    // The raw fraction exceeds u64, but conversion to nanoseconds fits.
    assert_eq!(
        duration_ns("0.00000000000000000001", Units::Milliseconds),
        Ok(ratio(1, 100_000_000_000_000))
    );
    // Legacy three-decimal device samples still parse exactly.
    assert_eq!(
        duration_ns("31.234", Units::Milliseconds),
        Ok(ratio(31_234_000, 1))
    );
    assert_eq!(duration_ns("7", Units::Nanoseconds), Ok(ratio(7, 1)));
    assert_eq!(
        duration_ns("0.0", Units::Milliseconds),
        Err(DurationError::Zero)
    );
    assert_eq!(
        duration_ns("-1.5", Units::Milliseconds),
        Err(DurationError::Malformed)
    );
    assert_eq!(
        duration_ns("nan", Units::Nanoseconds),
        Err(DurationError::Malformed)
    );
    assert_eq!(
        duration_ns("inf", Units::Nanoseconds),
        Err(DurationError::Malformed)
    );
    assert_eq!(
        duration_ns("1e30", Units::Milliseconds),
        Err(DurationError::Overflow)
    );
}

#[test]
fn samples_are_checked_against_their_own_raw_representation() {
    // A sub-resolution digit is retained, not rejected: the declared clock
    // resolution never quantizes a value.
    let mut artifact = device_artifact();
    artifact.clock.resolution_ns = 1000;
    let findings = check_artifact(None, &artifact);
    assert!(findings.clean(), "invalid={:?}; unavailable={:?}", findings.invalid, findings.unavailable);
    artifact.samples[0] = sample(0, None, None, "31.2345678", ratio(156_172_839, 5));
    assert!(check_artifact(None, &artifact).clean());

    let mut mismatched = artifact.clone();
    mismatched.samples[0].duration = ratio(156_172_839, 50);
    assert!(
        check_artifact(None, &mismatched)
            .invalid
            .contains(&Reason::SamplePrecision)
    );

    let mut zero = artifact.clone();
    zero.samples[0] = sample(0, None, None, "0.0", ratio(0, 1));
    assert!(
        check_artifact(None, &zero)
            .invalid
            .contains(&Reason::DurationOutOfRange)
    );

    let mut overflow = artifact.clone();
    overflow.samples[0] = sample(0, None, None, "1e30", ratio(1, 1));
    assert!(
        check_artifact(None, &overflow)
            .invalid
            .contains(&Reason::SampleOverflow)
    );

    let mut denominator = artifact.clone();
    denominator.samples[0].duration = ratio(1, 0);
    assert!(
        check_artifact(None, &denominator)
            .invalid
            .contains(&Reason::SamplePrecision)
    );
}

#[test]
fn mean_and_acquisition_statistics_are_exact_rationals() {
    assert_eq!(mean(&[]), None);
    assert_eq!(mean(&[ratio(1, 3), ratio(2, 3)]), Some(ratio(1, 2)));
    assert_eq!(
        mean(&[ratio(1, 10), ratio(3, 10), ratio(1, 1)]),
        Some(ratio(7, 15))
    );
    let artifact = cpu_artifact(cpu_samples());
    assert_eq!(statistic(&artifact), Some(ratio(140, 1)));

    // Rank scope: per-repetition maxima first, then the mean of those rationals.
    let ranks = rank_artifact(&[[100, 400], [200, 300], [100, 100], [500, 100], [100, 100]]);
    assert_eq!(statistic(&ranks), Some(ratio(280, 1)));
    let mut short = ranks.clone();
    short.samples.pop();
    assert_eq!(statistic(&short), None);

    // A fractional rank duration survives the maximum and the mean:
    // repetition 0 peaks at 3/2, the rest at 100 → (3/2 + 400) / 5 = 803/10.
    let mut fractional = rank_artifact(&[[100, 100], [100, 100], [100, 100], [100, 100], [100, 100]]);
    fractional.samples[0].duration = ratio(3, 2);
    fractional.samples[1].duration = ratio(1, 1);
    assert_eq!(statistic(&fractional), Some(ratio(803, 10)));
}

#[test]
fn operation_admission_and_byte_definitions_are_unchanged() {
    assert!(operation_admitted(
        AdapterId::CpuSumU64Reference,
        &frozen_operation(AdapterId::CpuSumU64Reference)
    ));
    let mut cpu = frozen_operation(AdapterId::CpuSumU64Reference);
    if let Operation::SumU64 { bound, .. } = &mut cpu {
        *bound = 1024;
    }
    assert!(!operation_admitted(AdapterId::CpuSumU64Reference, &cpu));

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
    assert_eq!(algorithm_payload_bytes(&declared), Some(2_097_152));
    assert_eq!(bus_normalization(2), Some(ratio(1, 1)));
    assert_eq!(bus_normalization(3), Some(ratio(4, 3)));
    assert_eq!(bus_normalization(1), None);
    // Inspection still derives fields from rejected input; it must not panic.
    let mut oversized = declared;
    if let Operation::AllReduce { numel, .. } = &mut oversized {
        *numel = u32::MAX;
    }
    assert!(derived(&oversized, &required_clock(AdapterId::NcclAllreduceSum),
        ratio(1, u64::MAX)).is_none());
}

#[test]
fn e3_parity_matches_public_binary64_boundaries() {
    let mut artifact = device_artifact();
    // Exactly on the recomputed maxabs bound: 1.5 * 0.001 + 1e-3 * 2 = 0.0035.
    artifact.correctness = Correctness::E3Parity {
        reference: E3_REFERENCE_ID.into(),
        finite: true,
        ref_max: "2.000000000".into(),
        e2: parity("0.001000000", "0.001000000", "0.001000000", "0.010000000"),
        e3: parity("0.003500000", "0.003500000", "0.003500000", "0.015099999999999999"),
        tolerance: E3_TOLERANCE,
        passed: true,
        outcome: CheckOutcome::Pass,
        detail: None,
    };
    assert!(check_artifact(None, &artifact).clean());

    // Python computes 1.5 * 0.01 + 0.0001 as 0.015099999999999999.
    // The next binary64 value, 0.0151, must fail rather than pass a decimal-rational gate.
    let Correctness::E3Parity { e3, .. } = &mut artifact.correctness else {
        panic!("expected E3 fixture");
    };
    e3.nrmse = "0.0151".into();
    assert!(check_artifact(None, &artifact).invalid.contains(&Reason::CorrectnessFailed));

    // One micro-unit beyond the maxabs bound fails.
    artifact.correctness = Correctness::E3Parity {
        reference: E3_REFERENCE_ID.into(),
        finite: true,
        ref_max: "2.000000000".into(),
        e2: parity("0.001000000", "0.001000000", "0.001000000", "0.010000000"),
        e3: parity("0.003600000", "0.003500000", "0.003500000", "0.015099999999999999"),
        tolerance: E3_TOLERANCE,
        passed: true,
        outcome: CheckOutcome::Pass,
        detail: None,
    };
    assert!(
        check_artifact(None, &artifact)
            .invalid
            .contains(&Reason::CorrectnessFailed)
    );

    // The coarse bound is strict: exactly 0.16 does not pass.
    artifact.correctness = Correctness::E3Parity {
        reference: E3_REFERENCE_ID.into(),
        finite: true,
        ref_max: "2.000000000".into(),
        e2: parity("0.200000000", "0.001000000", "0.001000000", "0.010000000"),
        e3: parity("0.160000000", "0.003500000", "0.003500000", "0.015099999999999999"),
        tolerance: E3_TOLERANCE,
        passed: true,
        outcome: CheckOutcome::Pass,
        detail: None,
    };
    assert!(
        check_artifact(None, &artifact)
            .invalid
            .contains(&Reason::CorrectnessFailed)
    );
    let Correctness::E3Parity { e3, .. } = &mut artifact.correctness else {
        panic!("expected E3 fixture");
    };
    e3.maxabs = "0.15999999999999998".into();
    assert!(check_artifact(None, &artifact).clean());
}

#[test]
fn e3_parity_accepts_finite_source_precision() {
    let mut artifact = device_artifact();
    let Correctness::E3Parity { e2, e3, .. } = &mut artifact.correctness else {
        panic!("expected E3 fixture");
    };
    // A normal binary64 nRMSE representation needs a decimal denominator wider
    // than u64. It is still a finite source statistic, not a timing rational.
    e2.nrmse = "0.00012345678901234567".into();
    e3.nrmse = "0.00012345678901234567".into();
    let findings = check_artifact(None, &artifact);
    assert!(findings.clean(), "{:?}", findings.invalid);
}

#[test]
fn observed_sources_and_runtime_observations_are_cross_checked() {
    let artifact = device_artifact();
    assert!(check_artifact(None, &artifact).clean());

    // A pinned source hash present in the plan but absent from the observation
    // set is identity drift, even when the paths look plausible.
    let plan_sources = vec![Source {
        path: "tests/test_exl3_overlay.py".into(),
        sha256: E3_PARITY_SOURCE_SHA256.into(),
    }];
    let mut findings = Findings::new();
    check_sources(Some(&plan_sources), &artifact.sources, &mut findings);
    assert!(findings.clean());

    let mut other_hash = artifact.sources.clone();
    other_hash[0].sha256 = E3_OPERATION_SOURCE_SHA256.into();
    let mut findings = Findings::new();
    check_sources(Some(&plan_sources), &other_hash, &mut findings);
    assert!(findings.invalid.contains(&Reason::IdentityDrift));

    // A missing required observation never passes.
    let mut missing = artifact.clone();
    missing
        .observations
        .retain(|observation| observation.name != ObservationName::Cap);
    assert!(
        check_artifact(None, &missing)
            .invalid
            .contains(&Reason::ObservationMissing)
    );

    // A reported parameter that is not the frozen one is drift, not noise.
    let mut wrong_cap = artifact.clone();
    for observation in &mut wrong_cap.observations {
        if observation.name == ObservationName::Cap {
            observation.value = "64".into();
        }
    }
    assert!(
        check_artifact(None, &wrong_cap)
            .invalid
            .contains(&Reason::ObservationDrift)
    );

    // A source echoed as a declaration instead of verified from bytes is drift.
    let mut echoed = artifact.clone();
    for observation in &mut echoed.observations {
        if observation.name == ObservationName::SourcesDeclared {
            observation.value = "1".into();
        }
    }
    assert!(
        check_artifact(None, &echoed)
            .invalid
            .contains(&Reason::ObservationDrift)
    );

    // The CPU reference carries no runtime observations at all.
    let mut cpu = cpu_artifact(cpu_samples());
    cpu.observations = e3_observations("e3-grouped");
    assert!(
        check_artifact(None, &cpu)
            .invalid
            .contains(&Reason::InvalidArtifact)
    );
}

#[test]
fn study_declaration_requires_complete_roles_and_observed_revision_pins() {
    let study = |minimum: u32, baseline: &str, candidate: &str, slots: usize| Study {
        kind: STUDY_KIND.into(),
        version: VERSION,
        adapter: AdapterId::CpuSumU64Reference,
        revision_axis: RevisionAxis {
            baseline: evidence::digest(baseline.as_bytes()),
            candidate: evidence::digest(candidate.as_bytes()),
        },
        thresholds: Thresholds {
            adverse_bps: 500,
            spread_bps: 2000,
        },
        minimum_acquisitions: minimum,
        roles: Roles {
            baseline: (0..slots).map(|i| format!("a{i:02}")).collect(),
            candidate: (0..slots).map(|i| format!("b{i:02}")).collect(),
            reference: (0..slots).map(|i| format!("r{i:02}")).collect(),
        },
        started_unix_ms: 1_757_000_000_000,
    };
    assert!(check_study(&study(3, "rev-a", "rev-b", 3)).is_empty());
    assert!(
        check_study(&study(2, "rev-a", "rev-b", 3))
            .contains(&Reason::RoleMinimum)
    );
    assert!(
        check_study(&study(3, "rev-a", "rev-b", 2))
            .contains(&Reason::RoleMinimum)
    );
    assert!(check_study(&study(3, "rev-a", "rev-a", 3)).is_empty());
    let mut unobserved = study(3, "rev-a", "rev-b", 3);
    unobserved.revision_axis.candidate = "operator-label".into();
    assert!(check_study(&unobserved).contains(&Reason::InvalidStudy));
    let mut duplicated = study(3, "rev-a", "rev-b", 3);
    duplicated.roles.reference[0] = "a00".into();
    assert!(check_study(&duplicated).contains(&Reason::RoleReuse));

    let mut bad_thresholds = study(3, "rev-a", "rev-b", 3);
    bad_thresholds.thresholds.adverse_bps = 10_000;
    assert!(check_study(&bad_thresholds).contains(&Reason::InvalidBounds));
}
