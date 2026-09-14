use super::*;
use serde_json::{Value, json};

fn expected() -> crate::sequence::ToolExpectation {
    crate::sequence::ToolExpectation {
        key: "harbor".into(), prior_ids: Vec::new(),
        result: "sapphire".into(), history_bytes: 2, history_messages: 0,
        history_cap: 128 * 1024, message_cap: 64,
    }
}

fn event(delta: Value, finish: Option<&str>) -> Vec<u8> {
    format!("data: {}\n\n", json!({"choices":[{"index":0,"delta":delta,"finish_reason":finish}]})).into_bytes()
}

fn fragments() -> Vec<(u64, Vec<u8>)> {
    vec![
        (10, event(json!({"tool_calls":[{"index":0,"id":"call_1","type":"function"}]}), None)),
        (20, event(json!({"tool_calls":[{"index":0,"function":{"name":"look"}}]}), None)),
        (30, event(json!({"tool_calls":[{"index":0,"function":{"name":"up_fact","arguments":"{\"key\":"}}]}), None)),
        (40, event(json!({"tool_calls":[{"index":0,"function":{"arguments":"\"harbor\"}"}}]}), None)),
        (50, event(json!({}), Some("tool_calls"))),
        (90, b"data: [DONE]\n\n".to_vec()),
    ]
}

fn capture(chunks: Vec<(u64, Vec<u8>)>, interrupted: bool) -> (Attempt, Vec<u8>) {
    let mut semantic = Semantic { tool: Some(crate::sequence::ToolStream::new(expected())), ..Semantic::default() };
    let mut parser = Parser::new(SseLimits { line_bytes: FRAME_CAP, event_bytes: FRAME_CAP }).unwrap();
    let mut timing = Timing { headers_us: Some(0), settle_us: 100, tool_stream_arrivals: Some(Vec::new()), ..Timing::default() };
    let mut body = Vec::new();
    let mut status = Status::Incomplete;
    let mut terminal_offset = None;
    for (observed_us, chunk) in chunks {
        let offset = body.len();
        body.extend_from_slice(&chunk);
        timing.first_body_us.get_or_insert(observed_us);
        timing.tool_stream_arrivals.as_mut().unwrap().push(ToolArrival { end_offset: body.len(), observed_us });
        match parser.feed(&chunk, |bytes| semantic.event(bytes, true, observed_us, &mut timing)) {
            Ok(Some(consumed)) => { terminal_offset = Some(offset + consumed); status = Status::Complete; break; }
            Ok(None) => (),
            Err(grill_sse::Error::Handler((error, _))) => { status = error; break; }
            Err(_) => { status = Status::Malformed; break; }
        }
    }
    if interrupted { status = Status::Interrupted; }
    if status != Status::Complete { timing.first_validated_tool_call_us = None; }
    (Attempt {
        lane: 0, dispatched: true, status, detail: String::new(), http_status: Some(200),
        finish_reason: semantic.finish, usage: semantic.usage, timing,
        response_bytes: body.len(), response_sha256: crate::evidence::digest(&body),
        terminal_offset, surplus_observed_bytes: 0, eligibility_errors: Vec::new(), sequence: None,
    }, body)
}

#[test]
fn tool_stream_delayed_fragments_and_terminal_replay_exact_distinct_boundaries() {
    let (attempt, body) = capture(fragments(), false);
    assert_eq!(attempt.status, Status::Complete);
    assert_eq!(attempt.timing.first_tool_delta_us, Some(20));
    assert_eq!(attempt.timing.first_validated_tool_call_us, Some(40));
    assert_eq!(attempt.timing.terminal_us, Some(90));
    assert_eq!(attempt.timing.first_generated_text_us, None);
    assert_eq!(attempt.timing.first_answer_text_us, None);
    let message = sequence_tool(&attempt, &body, expected()).unwrap().unwrap();
    assert_eq!(message["tool_calls"][0]["id"], "call_1");
    assert_eq!(message["tool_calls"][0]["function"]["arguments"], "{\"key\":\"harbor\"}");
    for field in ["delta", "validated", "terminal", "offset", "clock", "coverage"] {
        let mut changed = attempt.clone();
        match field {
            "delta" => changed.timing.first_tool_delta_us = Some(10),
            "validated" => changed.timing.first_validated_tool_call_us = Some(30),
            "terminal" => changed.timing.terminal_us = Some(50),
            "offset" => changed.timing.tool_stream_arrivals.as_mut().unwrap()[1].end_offset -= 1,
            "clock" => changed.timing.tool_stream_arrivals.as_mut().unwrap()[2].observed_us = 10,
            _ => { changed.timing.tool_stream_arrivals.as_mut().unwrap().pop(); }
        }
        assert!(sequence_tool(&changed, &body, expected()).is_err(), "{field}");
    }
}

#[test]
fn tool_stream_split_event_uses_chunk_completing_frame_not_initial_byte() {
    let mut chunks = fragments();
    let (_, name) = chunks.remove(1);
    chunks.insert(1, (15, name[..name.len()-1].to_vec()));
    chunks.insert(2, (25, name[name.len()-1..].to_vec()));
    let (attempt, body) = capture(chunks, false);
    assert_eq!(attempt.timing.first_tool_delta_us, Some(25));
    assert!(sequence_tool(&attempt, &body, expected()).unwrap().is_some());
}

#[test]
fn tool_stream_incomplete_invalid_and_cancelled_candidates_never_qualify() {
    for kind in ["missing-done", "incomplete-arguments", "trailing-value", "duplicate-id", "cancelled", "cancelled-after-done"] {
        let mut chunks = fragments();
        match kind {
            "missing-done" | "cancelled" => { chunks.pop(); }
            "incomplete-arguments" => { chunks.remove(3); }
            "trailing-value" => chunks.insert(4, (45, event(json!({"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}), None))),
            "duplicate-id" => chunks.insert(4, (45, event(json!({"tool_calls":[{"index":0,"id":"call_1"}]}), None))),
            _ => (),
        }
        let (attempt, body) = capture(chunks, kind.starts_with("cancelled"));
        assert_ne!(attempt.status, Status::Complete, "{kind}");
        assert_eq!(attempt.timing.first_tool_delta_us, Some(20), "{kind}");
        assert_eq!(attempt.timing.first_validated_tool_call_us, None, "{kind}");
        assert!(sequence_tool(&attempt, &body, expected()).unwrap().is_none(), "{kind}");
        let mut forged = attempt;
        forged.timing.first_validated_tool_call_us = Some(40);
        assert!(sequence_tool(&forged, &body, expected()).is_err(), "{kind}");
    }
}

#[test]
fn tool_stream_terminal_boundary_excludes_coread_postterminal_fragments() {
    let mut chunks = fragments();
    chunks.last_mut().unwrap().1.extend(event(json!({"tool_calls":[{"index":0,"function":{"arguments":"junk"}}]}), None));
    let (attempt, body) = capture(chunks, false);
    assert!(attempt.terminal_offset.unwrap() < body.len());
    assert!(sequence_tool(&attempt, &body, expected()).unwrap().is_some());
    let mut forged = attempt;
    forged.terminal_offset = Some(body.len());
    assert!(sequence_tool(&forged, &body, expected()).is_err());
}

#[test]
fn tool_stream_trace_contract_rejects_null_oversize_and_historical_fields() {
    let (attempt, _) = capture(fragments(), false);
    let timing = serde_json::to_value(&attempt.timing).unwrap();
    for field in ["first_tool_delta_us", "first_validated_tool_call_us", "tool_stream_arrivals"] {
        let mut null = timing.clone();
        null[field] = Value::Null;
        assert!(serde_json::from_value::<Timing>(null).is_err(), "{field}");
    }
    assert!(attempt.timing.validate_tools(false).is_err());
    let mut oversized = timing;
    oversized["tool_stream_arrivals"] = json!(vec![json!({"end_offset":1,"observed_us":1}); TOOL_ARRIVAL_CAP + 1]);
    assert!(serde_json::from_value::<Timing>(oversized).is_err());
    let mut invalid = attempt.timing;
    invalid.first_tool_delta_us = None;
    assert!(invalid.validate_tools(true).is_err());
}

#[test]
fn tool_stream_retained_history_overflow_withholds_validated_call() {
    let mut context = expected();
    context.history_messages = 63;
    context.history_bytes = 128 * 1024;
    let mut semantic = Semantic {
        tool: Some(crate::sequence::ToolStream::new(context)),
        ..Semantic::default()
    };
    let mut timing = Timing::default();
    let raw = json!({"choices":[{"delta":{"tool_calls":[{
        "index":0,"id":"call_1","type":"function","function":{
            "name":"lookup_fact","arguments":"{\"key\":\"harbor\"}"
        }
    }]},"finish_reason":"tool_calls"}]}).to_string();
    assert!(semantic.event(raw.as_bytes(), true, 10, &mut timing).is_err());
    assert_eq!(timing.first_tool_delta_us, Some(10));
    assert_eq!(timing.first_validated_tool_call_us, None);
}
