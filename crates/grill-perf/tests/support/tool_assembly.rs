use super::*;

fn stream(prior_ids: &[&str]) -> ToolStream {
    ToolStream::new(ToolExpectation {
        key: "harbor".into(),
        prior_ids: prior_ids.iter().map(|id| (*id).to_owned()).collect(),
        result: "sapphire".into(),
        history_bytes: 2,
        history_messages: 0,
        history_cap: 128 * 1024,
        message_cap: 64,
    })
}

#[test]
fn fragmented_fixed_call_keeps_first_delta_distinct_from_complete_call() {
    let mut tool = stream(&[]);
    tool.delta(r#"[{"index":0,"id":"call_1","type":"function"}]"#, 10).unwrap();
    assert_eq!(tool.first_delta_us, None);
    tool.delta(r#"[{"index":0,"function":{"name":"look"}}]"#, 20).unwrap();
    assert_eq!(tool.first_delta_us, Some(20));
    assert!(tool.validated_us().is_err());
    tool.delta(r#"[{"index":0,"function":{"name":"up_fact","arguments":"{\"key\":"}}]"#, 30).unwrap();
    assert!(tool.validated_us().is_err());
    tool.delta(r#"[{"index":0,"function":{"arguments":"\"harbor\"}"}}]"#, 40).unwrap();
    assert_eq!(tool.validated_us().unwrap(), 40);
    tool.delta(r#"[{"index":0,"function":{"arguments":"\n"}}]"#, 50).unwrap();
    assert_eq!(tool.validated_us().unwrap(), 40);
    assert_eq!(tool.message().unwrap()["tool_calls"][0]["function"]["arguments"], "{\"key\":\"harbor\"}\n");
}

#[test]
fn valid_json_prefix_does_not_rescue_contradictory_final_arguments() {
    let mut tool = stream(&[]);
    tool.delta(r#"[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup_fact","arguments":"{\"key\":\"harbor\"}"}}]"#, 10).unwrap();
    assert_eq!(tool.validated_us().unwrap(), 10);
    tool.delta(r#"[{"index":0,"function":{"arguments":"{}"}}]"#, 20).unwrap();
    assert!(tool.validated_us().is_err());
    assert!(tool.message().is_err());
}

#[test]
fn fixed_calls_reject_duplicate_ids_indices_names_and_argument_semantics() {
    for raw in [
        r#"[{"index":1,"id":"call_1"}]"#,
        r#"[{"index":0,"id":"call_1"},{"index":1,"id":"call_1"}]"#,
        r#"[{"index":0,"id":"bad id"}]"#,
        r#"[{"index":0,"id":"call_1","id":"call_2"}]"#,
        r#"[{"index":0,"function":{"name":"other_tool"}}]"#,
        r#"[{"index":0,"type":"code"}]"#,
    ] {
        assert!(stream(&[]).delta(raw, 10).is_err(), "{raw}");
    }
    let mut reused = stream(&["call_1"]);
    assert!(reused.delta(r#"[{"index":0,"id":"call_1"}]"#, 10).is_err());
    let mut duplicate = stream(&[]);
    duplicate.delta(r#"[{"index":0,"id":"call_1"}]"#, 10).unwrap();
    assert!(duplicate.delta(r#"[{"index":0,"id":"call_1"}]"#, 20).is_err());
    for arguments in [
        r#"{"key":"other"}"#,
        r#"{"key":"other","key":"harbor"}"#,
        r#"{"key":"harbor","extra":true}"#,
        r#"{"key":"harbor""#,
    ] {
        let raw = json!([{"index":0,"id":"call_1","type":"function","function":{"name":"lookup_fact","arguments":arguments}}]).to_string();
        let mut tool = stream(&[]);
        tool.delta(&raw, 10).unwrap();
        assert!(tool.validated_us().is_err(), "{arguments}");
    }
}

#[test]
fn assembly_bound_is_prospective_and_rejected_fragment_is_not_observed() {
    let mut tool = stream(&[]);
    let raw = json!([{"index":0,"function":{"arguments":" ".repeat(4097)}}]).to_string();
    assert!(tool.delta(&raw, 10).is_err());
    assert_eq!(tool.first_delta_us, None);
    assert!(tool.message().is_err());
}
