use super::*;
use serde_json::json;

fn spec() -> WaveSpec {
    let request: RequestSettings = serde_json::from_value(json!({"profile":"portable-chat-v1","stream":true,"output":{"tokens":8,"mode":"cap"},"cache":"observe"})).unwrap();
    WaveSpec {
        index: 0, phase: Phase::Measured, cell: "mixed".into(), case: None, trial: 0, concurrency: 2,
        lanes: Some(vec![
            ResolvedLane { id: "a".into(), case: "a".into(), arrival: Arrival::FixedOffset { offset_us: 0 }, request: request.clone(), control: None },
            ResolvedLane { id: "b".into(), case: "b".into(), arrival: Arrival::AfterFirstGenerated { lane: "a".into(), offset_us: 3 }, request, control: None },
        ]),
    }
}

fn attempt(lane: u32, dispatch: u64, first: u64, last: u64, settle: u64) -> Attempt {
    let mut a = undispatched(lane, Reason::Cancelled).attempt;
    a.dispatched = true;
    a.status = Status::Complete;
    a.timing = Timing {
        dispatch_offset_us: dispatch, first_generated_text_us: Some(first), last_generated_text_us: Some(last), settle_us: settle,
        ..Timing::default()
    };
    a.usage.completion_tokens = Some(8);
    a
}

#[test]
fn interval_oracle_distinguishes_text_mixed_and_terminal_tail() {
    let spec = spec();
    // A text=[5,10], B prefill=[15,25]: only request intervals overlap.
    let mut attempts = vec![attempt(0, 0, 5, 10, 100), attempt(1, 15, 10, 20, 90)];
    let observed = observation(&spec, &attempts, &[None, None], None, 105).unwrap();
    assert!(observed.overlap[0].request_inflight);
    assert!(!observed.overlap[0].generated_text);
    assert!(!observed.overlap[0].left_decode_right_prefill);
    // Extend actual text, not terminal: mixed now positive, text remains disjoint.
    attempts[0].timing.last_generated_text_us = Some(20);
    let observed = observation(&spec, &attempts, &[None, None], None, 105).unwrap();
    assert!(observed.overlap[0].left_decode_right_prefill);
    assert!(!observed.overlap[0].generated_text);
    // One coalesced text event is a zero-span interval, never overlap proof.
    attempts[0].timing.first_generated_text_us = Some(20);
    let observed = observation(&spec, &attempts, &[None, None], None, 105).unwrap();
    assert!(!observed.overlap[0].left_decode_right_prefill);
    // Schedule-origin conversions are checked, not wrapping/saturating.
    attempts[1].timing.dispatch_offset_us = u64::MAX - 5;
    assert!(observation(&spec, &attempts, &[None, None], None, u64::MAX).is_err());
}

#[test]
fn finite_admission_requires_one_real_notification_and_stops_after_failure() {
    let spec = spec();
    let lanes = spec.lanes.as_deref().unwrap();
    let mut state = State::new(lanes);
    assert_eq!(state.ready(0), Some(0));
    assert_eq!(state.ready(100), None);
    state.notify(FirstGenerated { lane: 0, offset_us: 7 });
    assert_eq!(state.ready(9), None);
    assert_eq!(state.ready(10), Some(1));
    assert_eq!(state.ready(100), None);
    state.notify(FirstGenerated { lane: 0, offset_us: 8 });
    assert_eq!(state.fatal, Some(Reason::NotificationFailed));
    assert_eq!(state.ready(u64::MAX), None);
    let mut state = State::new(lanes);
    state.notify(FirstGenerated { lane: 5, offset_us: 0 });
    assert_eq!(state.fatal, Some(Reason::NotificationFailed));
    assert_eq!(state.ready(u64::MAX), None);
}

#[test]
fn replay_rejects_dispatch_before_trigger_and_missing_admitted_positions() {
    let spec = spec();
    let mut attempts = vec![attempt(0, 0, 5, 30, 50), attempt(1, 7, 10, 20, 30)];
    let observed = observation(&spec, &attempts, &[None, None], None, 50).unwrap();
    assert!(verify(&spec, &attempts, &observed, 1).is_err());
    attempts[1].timing.dispatch_offset_us = 8;
    let observed = observation(&spec, &attempts, &[None, None], None, 50).unwrap();
    verify(&spec, &attempts, &observed, 1).unwrap();
    assert!(verify(&spec, &attempts[..1], &observed, 1).is_err());
    let mut corrupt = observed;
    corrupt.overlap[0].left_decode_right_prefill = false;
    assert!(verify(&spec, &attempts, &corrupt, 1).is_err());
}

#[test]
fn barrier_counts_initial_waits_but_never_fabricates_undispatched_service() {
    let a = attempt(0, 100, 5, 10, 20);
    let missing = undispatched(1, Reason::Deadline).attempt;
    assert_eq!(bounds([&a, &missing].into_iter(), Some(200)).unwrap(), (200, 0));
    assert!(bounds([&a].into_iter(), Some(119)).is_err());
    assert_eq!(bounds([&missing].into_iter(), Some(200)).unwrap(), (200, 0));
    // Legacy makespan stays first-dispatch to last-settlement.
    assert_eq!(bounds([&a].into_iter(), None).unwrap(), (20, 0));
}
