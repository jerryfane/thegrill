use super::*;
use serde_json::{Value, json};

fn flat(trials: u32, concurrency: u32) -> Workload {
    serde_json::from_value(json!({"version":6,"name":"population-fixture",
        "request":{"profile":"portable-chat-v1","stream":true,"output":{"tokens":8,"mode":"cap"},"cache":"observe"},
        "limits":{"total_ms":3000,"idle_ms":1000,"response_bytes":1024,"wave_buffer_bytes":33554432},
        "cases":[{"id":"one","messages":[{"role":"user","content":"fixture"}]}],
        "cells":[{"id":"cell","case":"one","concurrency":concurrency,"warmup_trials":1,"trials":trials}],
        "acquisition":{"kind":"flat","input_bytes":65536}})).unwrap()
}
fn loaded(workload: Workload, latency: u64) -> evidence::Loaded {
    let specs = workload.waves();
    let waves = specs
        .iter()
        .map(|spec| {
            let attempts = (0..spec.concurrency)
                .map(|lane| {
                    let mut a =
                        crate::schedule::undispatched(lane, crate::schedule::Reason::Cancelled)
                            .attempt;
                    a.dispatched = true;
                    a.status = Status::Complete;
                    a.http_status = Some(200);
                    a.usage.completion_tokens = Some(8);
                    a.eligibility_errors.clear();
                    a.timing = Timing {
                        first_generated_text_us: Some(latency / 4),
                        first_answer_text_us: Some(latency / 3),
                        last_generated_text_us: Some(latency / 2),
                        settle_us: latency,
                        ..Timing::default()
                    };
                    a
                })
                .collect();
            Some(Wave {
                version: 2,
                plan_sha256: String::new(),
                reservation_sha256: String::new(),
                spec: spec.clone(),
                attempts,
                elapsed_us: latency,
                dispatch_spread_us: 0,
                preparation_us: 0,
                reservation_publication_us: 0,
                body_publication_us: 0,
                completion_tokens: Some(8 * u64::from(spec.concurrency)),
                achieved_completion_tokens_per_second: None,
                eligible: true,
                metrics: None,
                schedule: None,
                acquisition_clock: Some(crate::acquisition::StepClock {
                    clock_id: String::new(),
                    kind: crate::acquisition::ClockKind::StdInstantMonotonic,
                    units: crate::acquisition::ClockUnits::Microseconds,
                    started_offset_us: u64::from(spec.index) * 1000,
                    settled_offset_us: u64::from(spec.index) * 1000 + latency,
                }),
            })
        })
        .collect();
    let plan=serde_json::from_value(json!({"version":5,"metric_contract":METRIC_CONTRACT,"kind":"performance-run-v1",
        "tool_version":"fixture","collector_sha256":"0".repeat(64),"workload":workload,"workload_sha256":"1".repeat(64),
        "source_sha256":"2".repeat(64),"model":"fixture","cache_evidence_source":"provider-reported",
        "endpoint":"http://127.0.0.1:9/v1/chat/completions","local_http":true,"pool_max_idle_per_host":2,
        "started_unix_ms":1,"waves":specs})).unwrap();
    evidence::Loaded {
        plan,
        waves,
        states: Vec::new(),
        metrics: None,
        policy: None,
        resources: Vec::new(),
        plan_sha256: String::new(),
        evidence_sha256: String::new(),
        lineage_sha256: String::new(),
        history: crate::lifecycle::History {
            count: 1,
            next_wave: 0,
            last_status: Some("completed".into()),
            open: false,
            evidence_sha256: String::new(),
            lineage_sha256: String::new(),
        },
    }
}
fn policy(workload: &Workload, version: u32) -> Value {
    json!({"version":version,"method":format!("observed-envelope-v{version}"),"id":"fixture",
        "collector_sha256":"0".repeat(64),"workload_source_sha256":"1".repeat(64),"min_trials":3,
        "cells":workload.cells.iter().filter(|c| workload.acquisition.as_ref().is_none_or(|p| p.measured(&c.case))).map(|c|
            json!({"cell":c.id,"metrics":[{"metric":"completion_latency_us","max_regression_bps":500,"max_reference_spread_bps":1000}]})).collect::<Vec<_>>()})
}
fn admitted(workload: &Workload, p: &Value) -> Result<Policy, Reason> {
    parse(
        &serde_json::to_vec(p).unwrap(),
        &"0".repeat(64),
        &"1".repeat(64),
        workload,
        None,
    )
}

#[test]
fn acquisition_population_has_one_worst_lane_per_repetition_and_exact_tail_floors() {
    let mut run = loaded(flat(200, 2), 100);
    for wave in run
        .waves
        .iter_mut()
        .flatten()
        .filter(|w| w.spec.phase == Phase::Measured)
    {
        wave.attempts[0].timing.settle_us = u64::from(wave.spec.trial) + 1;
        wave.attempts[1].timing.settle_us = 2 * (u64::from(wave.spec.trial) + 1);
    }
    let mut samples = completion_population(&run, "cell", Phase::Measured).unwrap();
    assert_eq!(samples.len(), 200);
    assert_eq!(
        crate::acquisition::Percentile::P95.nearest_rank(&mut samples),
        Some(380)
    );
    assert_eq!(
        crate::acquisition::Percentile::P99.nearest_rank(&mut samples),
        None
    );
    run.waves[100] = None;
    assert!(completion_population(&run, "cell", Phase::Measured).is_none());
    let mut values: Vec<_> = (1..=1000).collect();
    assert_eq!(
        crate::acquisition::Percentile::P99.nearest_rank(&mut values),
        Some(990)
    );
    assert_eq!(
        crate::acquisition::Percentile::P95.nearest_rank(&mut values[..199]),
        None
    );
}

#[test]
fn acquisition_tail_envelope_retains_both_reference_points_and_never_drops_failures() {
    let a = loaded(flat(200, 2), 100);
    let mut b = loaded(flat(200, 2), 120);
    let r = loaded(flat(200, 2), 100);
    let threshold = Thresholds {
        max_regression_bps: 500,
        max_reference_spread_bps: 1000,
    };
    let gate = acquisition_gate(
        [Some(&a), Some(&b), Some(&r)],
        TailTarget::Completion {
            cell: "cell".into(),
        },
        Some(crate::acquisition::Percentile::P95),
        &threshold,
        &[],
    );
    assert_eq!(gate.decision, Outcome::Regression);
    assert_eq!(gate.ranges.baseline, Some([(100, 1).into(); 2]));
    assert_eq!(gate.ranges.reference, Some([(100, 1).into(); 2]));
    assert_eq!(gate.coverage.candidate.observed_observations, 200);
    b.waves[1].as_mut().unwrap().eligible = false;
    let gate = acquisition_gate(
        [Some(&a), Some(&b), Some(&r)],
        TailTarget::Completion {
            cell: "cell".into(),
        },
        Some(crate::acquisition::Percentile::P95),
        &threshold,
        &[],
    );
    assert_eq!(gate.decision, Outcome::Inconclusive);
    assert!(gate.ranges.candidate.is_none());
}

#[test]
fn acquisition_admission_preserves_old_limits_and_checks_actual_seed_corner() {
    let mut w = flat(1000, 1);
    w.validate().unwrap();
    w.request.seed = Some(i64::MAX - 999 * 64);
    w.validate().unwrap();
    w.request.seed = Some(i64::MAX - 999 * 64 + 1);
    assert!(w.validate().is_err());
    w.request.seed = None;
    w.cases[0].messages[0].content = "x".repeat(130 * 1024);
    w.acquisition = Some(crate::acquisition::Protocol::Flat {
        input_bytes: 200 * 1024,
    });
    w.validate().unwrap();
    w.version = 3;
    w.acquisition = None;
    w.cells[0].trials = 3;
    assert!(w.validate().is_err());
}

#[test]
fn acquisition_policy_requires_warmup_and_closed_nonempty_tail_targets() {
    let mut w = flat(3, 1);
    let mut p = policy(&w, 3);
    assert!(admitted(&w, &p).is_ok());
    w.cells[0].warmup_trials = 0;
    assert!(admitted(&w, &p).is_err());
    w.cells[0].warmup_trials = 1;
    p["whole_conversation"] = json!({"max_regression_bps":500,"max_reference_spread_bps":1000});
    assert!(admitted(&w, &p).is_err());
    p.as_object_mut().unwrap().remove("whole_conversation");
    p["tail"] = json!([]);
    assert!(admitted(&w, &p).is_err());
    let t = json!({"target":{"kind":"completion","cell":"cell"},"percentile":"p95","max_regression_bps":500,"max_reference_spread_bps":1000});
    p["tail"] = json!([t.clone(), t]);
    assert!(admitted(&w, &p).is_err());
    p["tail"] = json!([{"target":{"kind":"first_answer","cell":"cell"},"percentile":"p95","max_regression_bps":500,"max_reference_spread_bps":1000}]);
    assert!(admitted(&w, &p).is_err());
    assert!(admitted(&w, &policy(&w, 2)).is_err());
}

#[test]
fn acquisition_tool_gates_never_borrow_text_or_terminal_and_require_real_tool_step() {
    let mut a = crate::schedule::undispatched(0, crate::schedule::Reason::Cancelled).attempt;
    a.dispatched = true;
    a.status = Status::Complete;
    a.eligibility_errors.clear();
    a.timing = Timing {
        first_answer_text_us: Some(2),
        terminal_us: Some(40),
        settle_us: 50,
        first_tool_delta_us: Some(10),
        first_validated_tool_call_us: Some(30),
        ..Timing::default()
    };
    assert_eq!(
        latency_sample(&a, Metric::FirstToolDeltaUs),
        Some((10, 1).into())
    );
    assert_eq!(
        latency_sample(&a, Metric::FirstValidatedToolCallUs),
        Some((30, 1).into())
    );
    a.timing.first_validated_tool_call_us = None;
    assert!(latency_sample(&a, Metric::FirstValidatedToolCallUs).is_none());
    let w = flat(3, 1);
    let mut p = policy(&w, 3);
    p["cells"][0]["metrics"][0]["metric"] = json!("first_tool_delta_us");
    assert!(admitted(&w, &p).is_err());
}

fn mixed() -> Workload {
    let mut w = flat(3, 1);
    w.version = 4;
    w.acquisition = None;
    w.cells.clear();
    w.schedule=Some(serde_json::from_value(json!([
        {"id":"solo-a","kind":"solo","warmup_trials":1,"trials":3,"lanes":[{"id":"a","case":"one","arrival":{"kind":"fixed_offset","offset_us":0}}]},
        {"id":"solo-b","kind":"solo","warmup_trials":1,"trials":3,"lanes":[{"id":"b","case":"one","arrival":{"kind":"fixed_offset","offset_us":0}}]},
        {"id":"mixed","kind":"overlap","warmup_trials":1,"trials":3,"lanes":[
            {"id":"decode","case":"one","arrival":{"kind":"fixed_offset","offset_us":0},"control":{"scenario":"solo-a","lane":"a"}},
            {"id":"prefill","case":"one","arrival":{"kind":"after_first_generated","lane":"decode","offset_us":0},"control":{"scenario":"solo-b","lane":"b"}}
        ]}])).unwrap());
    w
}
#[test]
fn mixed_policy_requires_named_selectors_and_never_accepts_empty_legacy_cells() {
    let w = mixed();
    w.validate().unwrap();
    let mut p = policy(&w, 2);
    assert!(admitted(&w, &p).is_err());
    p["cells"] = json!([
        {"cell":"solo-a","metrics":[{"metric":"first_answer_text_us","lane":"a","max_regression_bps":500,"max_reference_spread_bps":1000}]},
        {"cell":"solo-b","metrics":[{"metric":"first_answer_text_us","lane":"b","max_regression_bps":500,"max_reference_spread_bps":1000}]},
        {"cell":"mixed","metrics":[{"metric":"first_answer_text_us","lane":"decode","max_regression_bps":500,"max_reference_spread_bps":1000},{"metric":"first_answer_text_us","lane":"prefill","max_regression_bps":500,"max_reference_spread_bps":1000}]}
    ]);
    assert!(admitted(&w, &p).is_ok());
    p["cells"][2]["metrics"][0]
        .as_object_mut()
        .unwrap()
        .remove("lane");
    assert!(admitted(&w, &p).is_err());
}
#[test]
fn mixed_policy_withholds_false_overlap_and_missing_matched_control() {
    let mut run = loaded(mixed(), 100);
    for wave in run.waves.iter_mut().flatten() {
        if wave.spec.cell == "mixed" {
            wave.attempts[1].timing.dispatch_offset_us = 30;
        }
        wave.schedule = Some(
            crate::schedule::observation(
                &wave.spec,
                &wave.attempts,
                &vec![None; wave.attempts.len()],
                None,
                200,
            )
            .unwrap(),
        );
    }
    assert!(mixed_qualified(&run, "mixed"));
    let pop = population(&run.plan.workload, "mixed").unwrap();
    let (coverage, samples) =
        observations(Some(&run), &pop, Metric::FirstAnswerTextUs, Some("prefill"));
    assert_eq!(coverage.expected_observations, 3);
    assert_eq!(samples.len(), 3);
    let wave = run
        .waves
        .iter_mut()
        .flatten()
        .find(|w| w.spec.cell == "mixed")
        .unwrap();
    wave.attempts[1].timing.dispatch_offset_us = 110;
    wave.schedule = Some(
        crate::schedule::observation(&wave.spec, &wave.attempts, &[None, None], None, 210).unwrap(),
    );
    assert!(!mixed_qualified(&run, "mixed"));
    let mut run = loaded(mixed(), 100);
    run.waves[0] = None;
    assert!(!mixed_qualified(&run, "mixed"));
}

#[test]
fn acquisition_whole_membership_excludes_warmups_but_requires_all_controls() {
    let mut w: Workload =
        serde_json::from_str(include_str!("../../examples/conversation-tools-v3.json")).unwrap();
    w.version = 6;
    w.limits.wave_buffer_bytes = 16 * 1024 * 1024;
    w.acquisition = Some(crate::acquisition::Protocol::Conversation {
        repetitions: 3,
        warmup_repetitions: 1,
        measured_steps: vec!["followup".into()],
        input_bytes: 1024 * 1024,
        retained_history_bytes: 1024 * 1024,
    });
    w.validate().unwrap();
    let p = policy(&w, 3);
    assert!(admitted(&w, &p).is_ok());
    let mut run = loaded(w, 100);
    assert_eq!(
        crate::acquisition::whole_samples(&run.plan, &run.waves, Phase::Measured).unwrap(),
        vec![1100; 3]
    );
    assert_eq!(
        crate::acquisition::whole_samples(&run.plan, &run.waves, Phase::Warmup).unwrap(),
        vec![1100]
    );
    let pop = population(&run.plan.workload, "followup").unwrap();
    assert_eq!(pop.trials, 3);
    assert_eq!(pop.warmup_trials, 1);
    run.waves[0].as_mut().unwrap().eligible = false;
    assert!(!conversation_qualified(&run));
    run.waves[0].as_mut().unwrap().eligible = true;
    run.waves[2].as_mut().unwrap().eligible = false;
    assert!(crate::acquisition::whole_samples(&run.plan, &run.waves, Phase::Measured).is_err());
    run.waves[2].as_mut().unwrap().eligible = true;
    // A wrong final answer still has usable timing, but cannot count as a
    // complete acquisition even when every required step is present.
    run.waves[3].as_mut().unwrap().attempts[0].sequence = Some(crate::sequence::Check {
        history: "h".into(),
        parent: None,
        correct: false,
        canonical_match: Some(false),
        strict_match: None,
        error: None,
    });
    assert!(run.waves[3].as_ref().unwrap().eligible);
    assert!(crate::acquisition::whole_samples(&run.plan, &run.waves, Phase::Measured).is_err());
    run.waves[3].as_mut().unwrap().attempts[0].sequence = None;
    run.waves[3] = None;
    assert!(crate::acquisition::whole_samples(&run.plan, &run.waves, Phase::Measured).is_err());
}

#[test]
fn acquisition_tool_policy_accepts_only_named_measured_tool_exposure() {
    let mut w: Workload =
        serde_json::from_str(include_str!("../../examples/conversation-tools-v3.json")).unwrap();
    w.version = 6;
    w.limits.wave_buffer_bytes = 16 * 1024 * 1024;
    w.acquisition = Some(crate::acquisition::Protocol::Conversation {
        repetitions: 3,
        warmup_repetitions: 1,
        measured_steps: vec!["lookup".into(), "followup".into()],
        input_bytes: 1024 * 1024,
        retained_history_bytes: 1024 * 1024,
    });
    let mut p = policy(&w, 3);
    p["cells"][0]["metrics"] = json!([
        {"metric":"first_tool_delta_us","max_regression_bps":500,"max_reference_spread_bps":1000},
        {"metric":"first_validated_tool_call_us","max_regression_bps":500,"max_reference_spread_bps":1000}
    ]);
    assert!(admitted(&w, &p).is_ok());
    p["cells"][1]["metrics"][0]["metric"] = json!("first_tool_delta_us");
    assert!(admitted(&w, &p).is_err());
}

#[test]
fn acquisition_tail_oracle_preserves_improvement_drift_warmup_and_zero_outcomes() {
    let threshold = Thresholds {
        max_regression_bps: 500,
        max_reference_spread_bps: 1000,
    };
    for (candidate, reference, expected) in [
        (100, 100, Outcome::Pass),
        (80, 100, Outcome::Pass),
        (120, 100, Outcome::Regression),
        (100, 130, Outcome::Inconclusive),
    ] {
        let a = loaded(flat(200, 1), 100);
        let b = loaded(flat(200, 1), candidate);
        let r = loaded(flat(200, 1), reference);
        let gate = acquisition_gate(
            [Some(&a), Some(&b), Some(&r)],
            TailTarget::Completion {
                cell: "cell".into(),
            },
            Some(crate::acquisition::Percentile::P95),
            &threshold,
            &[],
        );
        assert_eq!(gate.decision, expected);
    }
    let a = loaded(flat(200, 1), 100);
    let mut b = loaded(flat(200, 1), 100);
    let r = loaded(flat(200, 1), 100);
    b.waves[0].as_mut().unwrap().eligible = false;
    let gate = acquisition_gate(
        [Some(&a), Some(&b), Some(&r)],
        TailTarget::Completion {
            cell: "cell".into(),
        },
        Some(crate::acquisition::Percentile::P95),
        &threshold,
        &[],
    );
    assert_eq!(gate.decision, Outcome::Inconclusive);
    assert!(gate.reason_codes.contains(&Reason::WarmupIncomplete));
    b.waves[0].as_mut().unwrap().eligible = true;
    b.waves[1].as_mut().unwrap().attempts[0].timing.settle_us = 0;
    let gate = acquisition_gate(
        [Some(&a), Some(&b), Some(&r)],
        TailTarget::Completion {
            cell: "cell".into(),
        },
        Some(crate::acquisition::Percentile::P95),
        &threshold,
        &[],
    );
    assert_eq!(gate.decision, Outcome::Inconclusive);
    assert_eq!(gate.coverage.candidate.eligible_waves, 199);
    assert!(gate.ranges.candidate.is_none());
    b.waves[1].as_mut().unwrap().attempts[0].timing.settle_us = 100;
    b.waves[1].as_mut().unwrap().attempts[0]
        .usage
        .completion_tokens = Some(7);
    let gate = acquisition_gate(
        [Some(&a), Some(&b), Some(&r)],
        TailTarget::Completion {
            cell: "cell".into(),
        },
        Some(crate::acquisition::Percentile::P95),
        &threshold,
        &[],
    );
    assert_eq!(gate.decision, Outcome::Inconclusive);
    assert!(gate.reason_codes.contains(&Reason::OutputAmountsMismatch));
}
