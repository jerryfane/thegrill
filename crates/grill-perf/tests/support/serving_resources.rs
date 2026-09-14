use super::*;

fn work() -> Value {
    let mut w = workload(1, 0, 1);
    w["version"] = json!(6);
    w["request"]["profile"] = json!("vllm-conversation-v3");
    w["cases"] = json!([
        {"id":"prime","messages":[{"role":"user","content":"Remember sapphire."}],"step":{"history":"h","parent":null,"cache":"observe","expect":{"kind":"json","value":{"fact":"sapphire"},"strict":null}}},
        {"id":"probe","messages":[{"role":"user","content":"Repeat the fact."}],"step":{"history":"h","parent":"prime","cache":"observe","expect":{"kind":"json","value":{"fact":"sapphire"},"strict":null}}}
    ]);
    w["cells"] = json!([
        {"id":"prime","case":"prime","concurrency":1,"warmup_trials":0,"trials":1},
        {"id":"probe","case":"probe","concurrency":1,"warmup_trials":0,"trials":1}
    ]);
    w["acquisition"] = json!({"kind":"conversation","repetitions":1,"warmup_repetitions":1,
        "measured_steps":["probe"],"input_bytes":65536,"retained_history_bytes":1048576});
    w["resources"] = json!({"version":1,"mode":"review_only","retained_bytes":134217728,
        "observer":{"version":1,"sources":[{"id":"fixture-process","target":{"kind":"process","pid":std::process::id()},"ownership":{"kind":"process_address_space"}}],
            "cadence_us":1000,"max_gap_us":1000000,"max_read_us":100000,"max_samples":1000,"deadline_us":10000000,
            "per_sample_bytes":16384,"raw_total_bytes":1048576},
        "summaries":[{"source":"fixture-process","metric":"resident_memory","statistic":"sampled_maximum","max_regression_bps":0,"max_reference_spread_bps":0},
            {"source":"fixture-process","metric":"cpu_time","statistic":"cpu_time","max_regression_bps":0,"max_reference_spread_bps":0}]});
    w
}
fn read_json(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}
fn replay(temp: &Temp) -> Output {
    cli()
        .arg("compare")
        .arg(temp.path("run"))
        .arg(temp.path("run"))
        .arg("--json")
        .output()
        .unwrap()
}

#[test]
fn serving_resources_cover_complete_warmup_and_measured_spans_without_raw_step_copies() {
    let temp = Temp::new();
    let server = Server::new(|mut stream, _, request| {
        if request["messages"].as_array().unwrap().len() > 1 {
            assert_eq!(request["messages"][1]["content"], "{\"fact\":\"sapphire\"}");
        }
        header(&mut stream, "text/event-stream");
        thread::sleep(Duration::from_millis(30));
        frame(
            &mut stream,
            json!({"choices":[{"delta":{"content":"{\"fact\":\"sapphire\"}"}}]}),
        );
        finish(&mut stream, Some(8), Some(0));
    });
    successful(&run(&temp, &server, "run", &work()));
    assert_eq!(server.count.load(Ordering::SeqCst), 4);
    let output = replay(&temp);
    assert_eq!(output.status.code(), Some(2)); // review-only never throughput PASS
    let comparison: Value = serde_json::from_slice(&output.stdout).unwrap();
    let resources = &comparison["acquisition"]["baseline"]["resources"];
    assert_eq!(resources["mode"], "review_only");
    for first in [0, 2] {
        let capture = read_json(&temp.path(&format!("run/resources-{first:06}.json")));
        let a = wave(&temp, "run", first);
        let b = wave(&temp, "run", first + 1);
        let observed = &capture["observation"];
        assert_eq!(capture["complete_membership"], true);
        assert_eq!(capture["required_waves"], json!([first, first + 1]));
        assert_eq!(
            observed["measured"]["started_us"],
            a["acquisition_clock"]["started_offset_us"]
        );
        assert_eq!(
            observed["measured"]["settled_us"],
            b["acquisition_clock"]["settled_offset_us"]
        );
        let start = observed["measured"]["started_us"].as_u64().unwrap();
        let end = observed["measured"]["settled_us"].as_u64().unwrap();
        let snapshots = observed["snapshots"].as_array().unwrap();
        assert!(snapshots.first().unwrap()["observed_us"].as_u64().unwrap() <= start);
        assert!(snapshots.last().unwrap()["observed_us"].as_u64().unwrap() >= end);
        assert!(
            end - start >= a["elapsed_us"].as_u64().unwrap() + b["elapsed_us"].as_u64().unwrap()
        );
        assert!(a.get("resources").is_none() && b.get("resources").is_none());
    }
    let reports = resources["acquisitions"].as_array().unwrap();
    assert!(
        reports
            .iter()
            .all(|r| r["summaries"][0]["value"].is_object())
    );
    assert!(
        reports
            .iter()
            .all(|r| r["summaries"][1]["unavailable"] == "unsupported_boundary")
    );
    let path = temp.path("run/resources-000000.json");
    let mut forged = read_json(&path);
    forged["observation"]["measured"]["settled_us"] =
        forged["members"][0]["clock"]["settled_offset_us"].clone();
    fs::write(path, serde_json::to_vec(&forged).unwrap()).unwrap();
    assert!(!replay(&temp).status.success());
}

#[test]
fn serving_resource_cancellation_drains_raw_samples_and_retains_missing_members() {
    let temp = Temp::new();
    let (ready, receive) = std::sync::mpsc::sync_channel(1);
    let server = Server::new(move |mut stream, _, _| {
        header(&mut stream, "text/event-stream");
        frame(
            &mut stream,
            json!({"choices":[{"delta":{"content":"partial"}}]}),
        );
        let (release, wait) = std::sync::mpsc::sync_channel(0);
        ready.send(release).unwrap();
        wait.recv_timeout(Duration::from_secs(10)).unwrap();
    });
    fs::write(
        temp.path("input.json"),
        serde_json::to_vec(&work()).unwrap(),
    )
    .unwrap();
    let mut child = cli()
        .arg("run")
        .arg(temp.path("input.json"))
        .args([
            "--endpoint",
            &server.endpoint,
            "--model",
            "fixture-model",
            "--local-http",
            "--json",
            "--out",
        ])
        .arg(temp.path("run"))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    server.wait_for_request(&mut child);
    let release = receive.recv_timeout(Duration::from_secs(3)).unwrap();
    thread::sleep(Duration::from_millis(10));
    assert_eq!(unsafe { libc::kill(child.id() as i32, libc::SIGINT) }, 0);
    let output = child.wait_with_output().unwrap();
    release.send(()).unwrap();
    assert_eq!(output.status.code(), Some(2));
    let capture = read_json(&temp.path("run/resources-000000.json"));
    assert_eq!(capture["cancelled"], true);
    assert_eq!(capture["complete_membership"], false);
    assert_eq!(capture["members"].as_array().unwrap().len(), 1);
    assert!(
        capture["observation"]["failures"]
            .as_array()
            .unwrap()
            .contains(&json!("cancelled"))
    );
    assert_eq!(server.count.load(Ordering::SeqCst), 1);
    let output = replay(&temp);
    assert_eq!(output.status.code(), Some(2));
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        report["acquisition"]["baseline"]["resources"]["acquisitions"][1]["failure"],
        "resource acquisition not retained"
    );
}

#[test]
fn old_workload_rejects_new_resource_options_before_dispatch() {
    let temp = Temp::new();
    let server = Server::new(normal);
    let mut w = workload(1, 0, 1);
    w["resources"] = work()["resources"].clone();
    assert!(!run(&temp, &server, "old", &w).status.success());
    assert_eq!(server.count.load(Ordering::SeqCst), 0);
    assert!(!temp.path("old").exists());
}
