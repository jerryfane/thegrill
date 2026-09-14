use super::*;

fn workload() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/conversation-tools-v3.json")
}

fn run(temp: &Temp, server: &Server, source: &Path) -> Output {
    command().arg("run").arg(source)
        .args(["--endpoint", &server.endpoint, "--model", "fixture", "--local-http", "--json", "--out"])
        .arg(temp.path("run")).output().unwrap()
}

fn replay(temp: &Temp) -> Output {
    command().arg("compare").arg(temp.path("run")).arg(temp.path("run"))
        .arg("--json").output().unwrap()
}

fn delta(stream: &mut TcpStream, value: Value, finish: Option<&str>) -> bool {
    write!(stream, "data: {}\n\n", json!({"choices":[{"index":0,"delta":value,"finish_reason":finish}]})).is_ok()
        && stream.flush().is_ok()
}

fn fixture(fault: Option<&'static str>) -> Server {
    Server::new(move |mut stream, index, request| {
        assert_eq!(request["stream"], true);
        assert_eq!(request["stream_options"], json!({"include_usage":true}));
        write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n").unwrap();
        if request.get("tools").is_none() {
            let messages = request["messages"].as_array().unwrap();
            let calls = messages.iter().find_map(|message| message.get("tool_calls")).unwrap();
            let result = messages.iter().find(|message| message["role"] == "tool").unwrap();
            assert_eq!(calls[0]["id"], result["tool_call_id"]);
            assert_eq!(calls[0]["function"]["name"], "lookup_fact");
            assert_eq!(calls[0]["function"]["arguments"], "{ \"key\": \"harbor\" }\n");
            assert_eq!(result["content"], "sapphire");
            assert!(delta(&mut stream, json!({"content":"{\"fact\":\"sapphire\"}"}), Some("stop")));
        } else {
            let id = if fault == Some("reuse-id") { "call_shared".to_owned() } else { format!("call_{index}") };
            if !delta(&mut stream, json!({"tool_calls":[{"index":0,"id":id,"type":"function"}]}), None) { return; }
            thread::sleep(Duration::from_millis(30));
            if !delta(&mut stream, json!({"tool_calls":[{"index":0,"function":{"name":"look"}}]}), None) { return; }
            thread::sleep(Duration::from_millis(30));
            let name = if fault == Some("name") { "up_other" } else { "up_fact" };
            if !delta(&mut stream, json!({"tool_calls":[{"index":0,"function":{"name":name,"arguments":"{ \"key\": "}}]}), None) { return; }
            if fault == Some("name") { return; }
            thread::sleep(Duration::from_millis(30));
            let arguments = match fault {
                Some("arguments") => "\"other\",\"key\":\"harbor\" }\n",
                Some("incomplete") => "\"harbor\"",
                _ => "\"harbor\" }\n",
            };
            if !delta(&mut stream, json!({"tool_calls":[{"index":0,"function":{"arguments":arguments}}]}), None) { return; }
            if fault == Some("duplicate-id") && !delta(&mut stream, json!({"tool_calls":[{"index":0,"id":id}]}), None) { return; }
            if fault == Some("trailing") && !delta(&mut stream, json!({"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}), None) { return; }
            if !delta(&mut stream, json!({}), Some("tool_calls")) { return; }
        }
        if write!(stream, "data: {}\n\n", json!({"choices":[],"usage":{"prompt_tokens":32,"completion_tokens":12}})).is_err() { return; }
        if stream.flush().is_err() { return; }
        if fault == Some("missing-done") { return; }
        thread::sleep(Duration::from_millis(70));
        let _ = write!(stream, "data: [DONE]\n\n");
    })
}

#[test]
fn tool_conversation_streams_fragments_links_actual_history_and_replays() {
    let temp = Temp::new();
    let server = fixture(None);
    let result = decoded(&run(&temp, &server, &workload()));
    assert_eq!(result["status"], "completed", "{result}");
    assert_eq!(server.count.load(Ordering::SeqCst), 2);
    let plan = read_json(&temp.path("run/plan.json"));
    assert_eq!(plan["version"], 4);
    let wave = read_json(&temp.path("run/wave-000000/wave.json"));
    assert_eq!(wave["version"], 2);
    assert_eq!(wave["attempts"][0]["sequence"]["correct"], true);
    let timing = &wave["attempts"][0]["timing"];
    let first = timing["first_tool_delta_us"].as_u64().unwrap();
    let valid = timing["first_validated_tool_call_us"].as_u64().unwrap();
    let terminal = timing["terminal_us"].as_u64().unwrap();
    assert!(first < valid && valid < terminal, "{timing}");
    assert!(timing["first_generated_text_us"].is_null());
    let followup = read_json(&temp.path("run/wave-000001/wave.json"));
    assert_eq!(followup["attempts"][0]["sequence"]["parent"], "lookup");
    assert_eq!(followup["attempts"][0]["sequence"]["correct"], true);
    let offline = decoded(&replay(&temp));
    assert_eq!(offline["baseline"][0]["sequence_check"], wave["attempts"][0]["sequence"]);
    assert_eq!(server.count.load(Ordering::SeqCst), 2);
    let mut forged = wave;
    forged["attempts"][0]["timing"]["first_validated_tool_call_us"] = json!(first);
    write_json(&temp.path("run/wave-000000/wave.json"), &forged);
    assert!(!replay(&temp).status.success());
}

#[test]
fn tool_conversation_invalid_and_partial_streams_stop_without_followup() {
    for fault in ["name", "arguments", "incomplete", "duplicate-id", "trailing", "missing-done"] {
        let temp = Temp::new();
        let server = fixture(Some(fault));
        let report = decoded(&run(&temp, &server, &workload()));
        assert_ne!(report["status"], "completed", "{fault}: {report}");
        assert_eq!(server.count.load(Ordering::SeqCst), 1, "{fault}");
        assert!(!temp.path("run/wave-000001").exists());
        let wave = read_json(&temp.path("run/wave-000000/wave.json"));
        assert_eq!(wave["attempts"][0]["sequence"]["correct"], false, "{fault}");
        assert!(wave["attempts"][0]["timing"]["first_tool_delta_us"].is_u64(), "{fault}");
        assert!(wave["attempts"][0]["timing"]["first_validated_tool_call_us"].is_null(), "{fault}");
        let offline = decoded(&replay(&temp));
        assert_eq!(offline["baseline"][0]["sequence_check"], wave["attempts"][0]["sequence"], "{fault}");
    }
}

#[test]
fn tool_conversation_reused_parent_call_id_is_rejected_and_replayed() {
    let temp = Temp::new();
    let server = fixture(Some("reuse-id"));
    let mut source = read_json(&workload());
    let mut lookup = source["cases"][0].clone();
    lookup["id"] = json!("lookup-again");
    lookup["messages"] = json!([{"role":"user","content":"Call lookup_fact with key harbor again."}]);
    lookup["step"]["parent"] = json!("followup");
    let mut followup = source["cases"][1].clone();
    followup["id"] = json!("followup-again");
    followup["step"]["parent"] = json!("lookup-again");
    source["cases"].as_array_mut().unwrap().extend([lookup, followup]);
    for id in ["lookup-again", "followup-again"] {
        source["cells"].as_array_mut().unwrap().push(json!({"id":id,"case":id,"concurrency":1,"warmup_trials":0,"trials":1}));
    }
    write_json(&temp.path("workload.json"), &source);
    let report = decoded(&run(&temp, &server, &temp.path("workload.json")));
    assert_ne!(report["status"], "completed");
    assert_eq!(server.count.load(Ordering::SeqCst), 3);
    let wave = read_json(&temp.path("run/wave-000002/wave.json"));
    assert_eq!(wave["attempts"][0]["status"], "malformed");
    assert!(wave["attempts"][0]["timing"]["first_validated_tool_call_us"].is_null());
    assert!(!temp.path("run/wave-000003").exists());
    let offline = decoded(&replay(&temp));
    assert_eq!(offline["baseline"][2]["sequence_check"]["correct"], false);
}

#[test]
fn tool_conversation_cancel_retains_partial_without_resume_or_replacement() {
    let temp = Temp::new();
    let (entered, ready) = std::sync::mpsc::channel();
    let (release, gate) = std::sync::mpsc::channel();
    let server = Server::new(move |mut stream, _, _| {
        write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n").unwrap();
        assert!(delta(&mut stream, json!({"tool_calls":[{"index":0,"id":"partial","type":"function","function":{"name":"lookup_fact","arguments":"{"}}]}), None));
        entered.send(()).unwrap();
        let _ = gate.recv_timeout(Duration::from_secs(10));
    });
    let child = command().arg("run").arg(workload())
        .args(["--endpoint", &server.endpoint, "--model", "fixture", "--local-http", "--json", "--out"])
        .arg(temp.path("run")).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    ready.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(unsafe { libc::kill(child.id() as i32, libc::SIGINT) }, 0);
    let output = child.wait_with_output().unwrap();
    let _ = release.send(());
    assert_eq!(decoded(&output)["status"], "interrupted");
    let wave = read_json(&temp.path("run/wave-000000/wave.json"));
    assert_eq!(wave["attempts"][0]["status"], "interrupted");
    assert!(wave["attempts"][0]["timing"]["first_validated_tool_call_us"].is_null());
    assert!(!temp.path("run/wave-000001").exists());
    assert_eq!(decoded(&replay(&temp))["baseline"][0]["sequence_check"]["correct"], false);
    let resume = command().arg("resume").arg(temp.path("run")).arg("--json").output().unwrap();
    assert!(!resume.status.success());
    assert_eq!(server.count.load(Ordering::SeqCst), 1);
}
