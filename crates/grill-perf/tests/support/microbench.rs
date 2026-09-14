//! CPU protocol fixtures for the #54 microbenchmark commands.
//!
//! Every case here is a synthetic protocol fixture: the device adapters are
//! never executed, and no device, collective, model or service is touched.
//! Only the in-process CPU reference runs for real. The external-program cases
//! launch a trivial local shell script so that the collector-side launch,
//! hashing, receipt and bounded-timeout paths are exercised; that script
//! performs no kernel or collective work, and nothing here demonstrates device
//! or fabric execution.

use super::*;
use sha2::{Digest, Sha256};
use std::os::unix::fs::PermissionsExt;

const NCCL_BYTE_SOURCE_SHA256: &str =
    "2242493053a6f9d5db1c6542a8ce7740be919778645125b94eeae440efd31edf";
const E3_ANCHOR_SHA256: &str = "f57e4fa6726b0110c44ce56b0b0968ba425e566ace6d89f2db3872aae13889ec";
const E3_PARITY_SHA256: &str = "8777254d47cdbb89186ba3bd1ac2437cbebc7987e0791d36c68e7bccb3df2b35";

const START: u64 = 1_757_000_000_000;
const CPU_BOUND_REFERENCE: &str = "8386560";

fn mb_sha(bytes: &[u8]) -> String {
    let mut hash = Sha256::new();
    hash.update(bytes);
    hash.finalize().iter().map(|byte| format!("{byte:02x}")).collect()
}

fn mb_write(path: &Path, value: &Value) {
    fs::write(path, serde_json::to_vec_pretty(value).unwrap()).unwrap();
}

fn mb_write_bytes(path: &Path, bytes: &[u8]) {
    fs::write(path, bytes).unwrap();
}

fn mb_cpu_plan(id: &str, started: u64) -> Value {
    json!({
        "kind": "microbench-plan-v1",
        "version": 1,
        "adapter": "cpu-sum-u64-reference",
        "revision": "cpu-sum-u64-reference-v1",
        "program_sha256": null,
        "sources": [],
        "operation": {"kind": "sum-u64", "bound": 4096, "dtype": "u64"},
        "clock": {
            "id": "process-monotonic-ns",
            "kind": "process_monotonic",
            "units": "nanoseconds",
            "resolution_ns": 1,
            "synchronization": "process-clock-read-before-after"
        },
        "warmups": 2,
        "iterations": 5,
        "allowance": {"work_units": 28672, "memory_bytes": 65536, "deadline_ms": 5000},
        "correctness": {"reference": "n*(n-1)/2", "tolerance": {"kind": "exact-u64"}},
        "thresholds": {"adverse_bps": 500, "spread_bps": 2000},
        "acquisition": {"id": id, "started_unix_ms": started}
    })
}

fn mb_reduce_sources(program_sha256: &str) -> Value {
    json!([
        {"path": "tools/microbench/nccl_allreduce_sum.py", "sha256": program_sha256},
        {"path": "doc/PERFORMANCE.md", "sha256": NCCL_BYTE_SOURCE_SHA256}
    ])
}

fn mb_reduce_plan(id: &str, started: u64, world: u32, program_sha256: &str) -> Value {
    json!({
        "kind": "microbench-plan-v1",
        "version": 1,
        "adapter": "nccl-allreduce-sum",
        "revision": "nccl-allreduce-sum-world2-v1",
        "program_sha256": program_sha256,
        "sources": mb_reduce_sources(program_sha256),
        "operation": {
            "kind": "all-reduce",
            "op": "sum",
            "dtype": "int64",
            "numel": 262144,
            "world": world,
            "ranks": (0..world).collect::<Vec<u32>>(),
            "input": "input[i]=(i%251)+rank",
            "reference": "world*(i%251)+world*(world-1)/2"
        },
        "clock": {
            "id": "rank-monotonic-ns",
            "kind": "host_monotonic",
            "units": "nanoseconds",
            "resolution_ns": 1,
            "synchronization": "collective-barrier-before-after"
        },
        "warmups": 1,
        "iterations": 5,
        "allowance": {"work_units": 2000000, "memory_bytes": 8388608, "deadline_ms": 120000},
        "correctness": {
            "reference": "world*(i%251)+world*(world-1)/2",
            "tolerance": {"kind": "exact-int64"}
        },
        "thresholds": {"adverse_bps": 500, "spread_bps": 2000},
        "acquisition": {"id": id, "started_unix_ms": started}
    })
}

fn mb_e3_plan(id: &str, started: u64) -> Value {
    json!({
        "kind": "microbench-plan-v1",
        "version": 1,
        "adapter": "exl3-e3-grouped",
        "revision": "exl3-e3-grouped-cap32-v1",
        "program_sha256": "b".repeat(64),
        "sources": [
            {"path": "tools/microbench/exl3_e3_grouped.py", "sha256": "b".repeat(64)},
            {"path": "tests/bench_e3_microbench.py", "sha256": E3_ANCHOR_SHA256},
            {"path": "tests/test_exl3_overlay.py", "sha256": E3_PARITY_SHA256}
        ],
        "operation": {
            "kind": "exl3-experts",
            "hidden": 4096,
            "intermediate": 1024,
            "experts": 288,
            "topk": 8,
            "tokens": 1024,
            "layer_seed": 0,
            "routing_seed": 1,
            "parity_routing_seed": 3,
            "activation_seed": 3,
            "skew_milli": 1000,
            "tier": "e3-grouped",
            "cap": 32,
            "dtype": "float16",
            "routing": "synthetic-zipf-permuted",
            "activation": "randn-fp16-activation-seed-3",
            "weights": "random-k4-trellis-shared-gate-up-suh"
        },
        "clock": {
            "id": "cuda-event-ms",
            "kind": "device_event",
            "units": "milliseconds",
            "resolution_ns": 1000,
            "synchronization": "device-event-timing-after-synchronize"
        },
        "warmups": 2,
        "iterations": 5,
        "allowance": {"work_units": 300000000000u64, "memory_bytes": 68719476736u64, "deadline_ms": 600000},
        "correctness": {
            "reference": "apply-exl3-python-loop",
            "tolerance": {
                "kind": "e3-rel",
                "factor_milli": 1500,
                "abs_rel_micro": 1000,
                "nrmse_abs_micro": 100,
                "coarse_abs_milli": 150,
                "coarse_factor_milli": 80
            }
        },
        "thresholds": {"adverse_bps": 500, "spread_bps": 2000},
        "acquisition": {"id": id, "started_unix_ms": started}
    })
}

fn mb_cpu_artifact(plan: &Value, durations: &[u64]) -> Value {
    let samples: Vec<Value> = durations
        .iter()
        .enumerate()
        .map(|(index, nanos)| {
            json!({"index": index, "rank": null, "repetition": null,
                   "raw": nanos.to_string(), "duration_ns": nanos})
        })
        .collect();
    json!({
        "kind": "microbench-artifact-v1",
        "version": 1,
        "adapter": plan["adapter"],
        "revision": plan["revision"],
        "operation": plan["operation"],
        "provenance": "native_observed",
        "submitted_provenance": null,
        "program_sha256": null,
        "kernel_revision": null,
        "sources": [],
        "topology": {"scope": "cpu", "world": null, "ranks": []},
        "clock": plan["clock"],
        "execution": {
            "warmups": 2, "iterations": 5, "deadline_ms": 5000,
            "work_units": 28672, "memory_bytes": 32768,
            "completed": true, "timed_out": false, "observed_fallback": null
        },
        "samples": samples,
        "correctness": {
            "kind": "exact-sum", "reference": "n*(n-1)/2",
            "computed": CPU_BOUND_REFERENCE, "bound": CPU_BOUND_REFERENCE, "passed": true
        },
        "failures": []
    })
}

fn mb_reduce_artifact(plan: &Value, per_repetition: &[Vec<u64>], mismatches: u64) -> Value {
    let world = plan["operation"]["world"].as_u64().unwrap() as u32;
    let mut samples = Vec::new();
    let mut index = 0u32;
    for (repetition, ranks) in per_repetition.iter().enumerate() {
        for (rank, nanos) in ranks.iter().enumerate() {
            samples.push(json!({
                "index": index, "rank": rank, "repetition": repetition,
                "raw": nanos.to_string(), "duration_ns": nanos
            }));
            index += 1;
        }
    }
    let complete = per_repetition.len() == 5
        && per_repetition.iter().all(|ranks| ranks.len() == world as usize);
    json!({
        "kind": "microbench-artifact-v1",
        "version": 1,
        "adapter": plan["adapter"],
        "revision": plan["revision"],
        "operation": plan["operation"],
        "provenance": "native_observed",
        "submitted_provenance": null,
        "program_sha256": plan["program_sha256"],
        "kernel_revision": "fixture-synthetic-protocol-only",
        "sources": plan["sources"],
        "topology": {
            "scope": "ranks",
            "world": world,
            "ranks": (0..world).map(|r| json!({"index": r, "device": r})).collect::<Vec<Value>>()
        },
        "clock": plan["clock"],
        "execution": {
            "warmups": 1, "iterations": 5, "deadline_ms": 120000,
            "work_units": 1572864, "memory_bytes": 6291456,
            "completed": complete, "timed_out": false, "observed_fallback": null
        },
        "samples": samples,
        "correctness": {
            "kind": "exact-reduction",
            "reference": "world*(i%251)+world*(world-1)/2",
            "mismatches": mismatches, "tolerance_elems": 0,
            "finite": true, "passed": mismatches == 0
        },
        "failures": []
    })
}

fn mb_e3_artifact(plan: &Value, fallback: &str, e3_maxabs: &str) -> Value {
    let samples: Vec<Value> = (0..5)
        .map(|index| {
            json!({"index": index, "rank": null, "repetition": null,
                   "raw": "31.234", "duration_ns": 31234000})
        })
        .collect();
    json!({
        "kind": "microbench-artifact-v1",
        "version": 1,
        "adapter": plan["adapter"],
        "revision": plan["revision"],
        "operation": plan["operation"],
        "provenance": "native_observed",
        "submitted_provenance": null,
        "program_sha256": plan["program_sha256"],
        "kernel_revision": "fixture-synthetic-protocol-only",
        "sources": plan["sources"],
        "topology": {"scope": "device", "world": null, "ranks": [{"index": 0, "device": 0}]},
        "clock": plan["clock"],
        "execution": {
            "warmups": 2, "iterations": 5, "deadline_ms": 600000,
            "work_units": 206158430208u64, "memory_bytes": 1073741824,
            "completed": true, "timed_out": false, "observed_fallback": fallback
        },
        "samples": samples,
        "correctness": {
            "kind": "e3-parity",
            "reference": "apply-exl3-python-loop",
            "finite": true,
            "ref_max": "2.000000000",
            "e2": {"maxabs": "0.001000000", "per_token_max": "0.001000000",
                   "per_token_p99": "0.001000000", "nrmse": "0.010000000"},
            "e3": {"maxabs": e3_maxabs, "per_token_max": "0.003000000",
                   "per_token_p99": "0.003000000", "nrmse": "0.015000000"},
            "tolerance": {"kind": "e3-rel", "factor_milli": 1500, "abs_rel_micro": 1000,
                          "nrmse_abs_micro": 100, "coarse_abs_milli": 150,
                          "coarse_factor_milli": 80},
            "passed": true,
            "outcome": "pass",
            "detail": null
        },
        "failures": []
    })
}

fn mb_evidence(temp: &Temp, name: &str, plan: &Value, artifact: &Value) -> PathBuf {
    let dir = temp.path(name);
    fs::create_dir_all(&dir).unwrap();
    mb_write(&dir.join("plan.json"), plan);
    mb_write(&dir.join("artifact.json"), artifact);
    dir
}

fn mb_stdout(output: &Output) -> Value {
    serde_json::from_slice(&output.stdout).unwrap_or(Value::Null)
}

fn mb_code(output: &Output) -> i32 {
    output.status.code().unwrap_or(-1)
}

fn mb_reasons(report: &Value, key: &str) -> Vec<String> {
    report[key]
        .as_array()
        .map(|values| {
            values
                .iter()
                .filter_map(|value| value.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

fn mb_has(report: &Value, key: &str, reason: &str) -> bool {
    mb_reasons(report, key).iter().any(|value| value == reason)
}

fn mb_inspect(temp: &Temp, name: &str, plan: &Value, artifact: &Value) -> (i32, Value) {
    let dir = mb_evidence(temp, name, plan, artifact);
    let output = cli()
        .args(["microbench", "inspect"])
        .arg(&dir)
        .arg("--json")
        .output()
        .unwrap();
    (mb_code(&output), mb_stdout(&output))
}

fn mb_capture(temp: &Temp, name: &str, adapter: &str, plan: &Value) -> (i32, Value, String, PathBuf) {
    let dir = temp.path(name);
    let plan_path = temp.path(&format!("{name}-plan.json"));
    mb_write(&plan_path, plan);
    let output = cli()
        .args(["microbench", "capture", "--adapter", adapter])
        .arg("--plan")
        .arg(&plan_path)
        .arg("--out")
        .arg(&dir)
        .arg("--json")
        .output()
        .unwrap();
    (
        mb_code(&output),
        mb_stdout(&output),
        String::from_utf8_lossy(&output.stderr).into_owned(),
        dir,
    )
}

fn mb_run(args: &[&str], paths: &[&Path]) -> (i32, Value, String) {
    let mut command = cli();
    command.args(["microbench"]).args(args);
    for path in paths {
        command.arg(path);
    }
    let output = command.arg("--json").output().unwrap();
    (
        mb_code(&output),
        mb_stdout(&output),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

#[test]
fn cpu_reference_capture_is_native_exact_and_inspectable() {
    let temp = Temp::new();
    let plan = mb_cpu_plan("a01", START);
    let (code, report, stderr, dir) = mb_capture(&temp, "cpu", "cpu-sum-u64-reference", &plan);
    assert_eq!(code, 0, "stderr={stderr}");
    assert_eq!(report["provenance"], "native_observed");
    assert_eq!(report["exerciser"], "native-cpu-reference");
    assert_eq!(report["samples"], 5);
    assert!(report["invalid"] == json!([]));
    assert!(report["scope"].as_str().unwrap().contains("never sets a serving-speed claim"));

    let artifact: Value = serde_json::from_slice(&fs::read(dir.join("artifact.json")).unwrap()).unwrap();
    assert_eq!(artifact["adapter"], "cpu-sum-u64-reference");
    assert_eq!(artifact["correctness"]["computed"], CPU_BOUND_REFERENCE);
    assert_eq!(artifact["correctness"]["bound"], CPU_BOUND_REFERENCE);
    assert_eq!(artifact["correctness"]["passed"], true);
    assert_eq!(artifact["execution"]["completed"], true);
    assert_eq!(artifact["execution"]["observed_fallback"], Value::Null);
    assert_eq!(artifact["samples"].as_array().unwrap().len(), 5);
    let nanos = artifact["samples"][0]["duration_ns"].as_u64().unwrap();
    assert!(nanos > 0);
    // Raw text is the exact nanosecond decimal, never a re-rounded float.
    assert_eq!(
        artifact["samples"][0]["raw"].as_str().unwrap().parse::<u64>().unwrap(),
        nanos
    );
    assert!(artifact["execution"]["work_units"].as_u64().unwrap() <= 28672);
    assert!(artifact["execution"]["memory_bytes"].as_u64().unwrap() <= 65536);
    let receipt: Value = serde_json::from_slice(&fs::read(dir.join("receipt.json")).unwrap()).unwrap();
    assert_eq!(receipt["status"], "captured");
    assert_eq!(receipt["provenance"], "native_observed");
    assert_eq!(receipt["claim"], "native-in-process-cpu-reference");

    let (code, report, stderr) = mb_run(&["inspect"], &[&dir]);
    assert_eq!(code, 0, "stderr={stderr}");
    assert_eq!(report["plan_verified"], true);
    assert_eq!(report["invalid"], json!([]));
    assert_eq!(report["unavailable"], json!([]));
    assert!(report["statistic"]["numerator"].as_u64().unwrap() > 0);
    assert_eq!(report["derived"], Value::Null);
}

#[test]
fn capture_rejects_drifted_declarations_before_running() {
    let temp = Temp::new();
    let mutations: [(&str, fn(&mut Value)); 4] = [
        ("clock", |plan| plan["clock"]["units"] = json!("milliseconds")),
        ("warmups", |plan| plan["warmups"] = json!(3)),
        ("bound", |plan| plan["operation"]["bound"] = json!(1024)),
        ("allowance", |plan| plan["allowance"]["work_units"] = json!(64)),
    ];
    for (name, mutate) in mutations {
        let mut plan = mb_cpu_plan("a01", START);
        mutate(&mut plan);
        let (code, _report, stderr, dir) = mb_capture(&temp, name, "cpu-sum-u64-reference", &plan);
        assert_eq!(code, 1, "{name} stderr={stderr}");
        assert!(stderr.contains("rejected"), "{name} stderr={stderr}");
        assert!(!dir.join("artifact.json").exists(), "{name}");
    }

    // A plan declared for another adapter cannot be captured as the CPU reference.
    let reduce = mb_reduce_plan("a01", START, 2, &"a".repeat(64));
    let (code, _report, stderr, _) = mb_capture(&temp, "adapter-mismatch", "cpu-sum-u64-reference", &reduce);
    assert_eq!(code, 1);
    assert!(stderr.contains("adapter"), "{stderr}");

    // The in-process reference refuses an operator program.
    let plan_path = temp.path("cpu-inproc-plan.json");
    mb_write(&plan_path, &mb_cpu_plan("a01", START));
    let output = cli()
        .args(["microbench", "capture", "--adapter", "cpu-sum-u64-reference"])
        .arg("--plan")
        .arg(&plan_path)
        .arg("--out")
        .arg(temp.path("cpu-inproc"))
        .arg("--program")
        .arg("/bin/true")
        .output()
        .unwrap();
    assert_eq!(mb_code(&output), 1);
    assert!(String::from_utf8_lossy(&output.stderr).contains("rejects --program"));

    // Device and collective capture needs a separately authorized window.
    let (code, _report, stderr, dir) = mb_capture(&temp, "gated", "nccl-allreduce-sum", &reduce);
    assert_eq!(code, 1);
    assert!(stderr.contains("CPU-only authorization"), "{stderr}");
    assert!(!dir.exists());
}

#[test]
fn inspect_rejects_iteration_clock_units_and_correctness_drift() {
    let temp = Temp::new();
    let plan = mb_cpu_plan("a01", START);
    let base = mb_cpu_artifact(&plan, &[160, 100, 120, 140, 180]);

    let mut missing = base.clone();
    missing["samples"].as_array_mut().unwrap().pop();
    let (code, report) = mb_inspect(&temp, "missing-iteration", &plan, &missing);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "SampleGridIncomplete"), "{report}");

    let mut extra = base.clone();
    extra["execution"]["iterations"] = json!(4);
    let (code, report) = mb_inspect(&temp, "iteration-count", &plan, &extra);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "IterationCountMismatch"), "{report}");

    let mut units = base.clone();
    units["clock"]["units"] = json!("milliseconds");
    units["clock"]["resolution_ns"] = json!(1000);
    let (code, report) = mb_inspect(&temp, "units", &plan, &units);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "ClockMismatch"), "{report}");

    let mut synchronization = base.clone();
    synchronization["clock"]["kind"] = json!("device_event");
    let (code, report) = mb_inspect(&temp, "sync", &plan, &synchronization);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "ClockMismatch"), "{report}");

    let mut precision = base.clone();
    precision["samples"][0]["raw"] = json!("1.2345");
    precision["samples"][0]["duration_ns"] = json!(12345);
    let (code, report) = mb_inspect(&temp, "precision", &plan, &precision);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "SamplePrecision"), "{report}");

    let mut zero = base.clone();
    zero["samples"][1]["raw"] = json!("0");
    zero["samples"][1]["duration_ns"] = json!(0);
    let (code, report) = mb_inspect(&temp, "zero-duration", &plan, &zero);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "DurationOutOfRange"), "{report}");

    let mut inverted = base.clone();
    inverted["samples"][2]["rank"] = json!(0);
    let (code, report) = mb_inspect(&temp, "grid", &plan, &inverted);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "SampleGridIncomplete"), "{report}");

    let mut incorrect = base.clone();
    incorrect["correctness"]["computed"] = json!("8386561");
    incorrect["correctness"]["passed"] = json!(false);
    let (code, report) = mb_inspect(&temp, "correctness", &plan, &incorrect);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "CorrectnessFailed"), "{report}");

    let mut drifted = base.clone();
    drifted["operation"]["bound"] = json!(2048);
    let (code, report) = mb_inspect(&temp, "operation-drift", &plan, &drifted);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "IdentityDrift"), "{report}");

    let mut sourced = base.clone();
    sourced["sources"] = json!([{"path": "extra.py", "sha256": "d".repeat(64)}]);
    let (code, report) = mb_inspect(&temp, "source-drift", &plan, &sourced);
    assert_eq!(code, 1);
    assert!(!mb_reasons(&report, "invalid").is_empty(), "{report}");

    let mut partial = base.clone();
    partial["execution"]["completed"] = json!(false);
    partial["execution"]["timed_out"] = json!(true);
    let (code, report) = mb_inspect(&temp, "timeout", &plan, &partial);
    assert_eq!(code, 2);
    assert!(mb_has(&report, "unavailable", "DeadlineExceeded"), "{report}");
    assert!(mb_has(&report, "unavailable", "SampleGridIncomplete"), "{report}");

    let mut retained = base.clone();
    retained["failures"] = json!([{"kind": "adapter-error", "detail": "fixture retained failure"}]);
    let (code, report) = mb_inspect(&temp, "retained", &plan, &retained);
    assert_eq!(code, 2);
    assert!(mb_has(&report, "unavailable", "RetainedFailure"), "{report}");
}

#[test]
fn import_never_upgrades_a_native_provenance_claim() {
    let temp = Temp::new();
    let plan = mb_cpu_plan("a01", START);
    let artifact = mb_cpu_artifact(&plan, &[1000, 1000, 1000, 1000, 1000]);
    let artifact_bytes = serde_json::to_vec_pretty(&artifact).unwrap();
    let artifact_path = temp.path("imported-artifact.json");
    mb_write_bytes(&artifact_path, &artifact_bytes);
    let plan_path = temp.path("imported-plan.json");
    mb_write(&plan_path, &plan);
    let out = temp.path("imported");

    let output = cli()
        .args(["microbench", "import", "--artifact"])
        .arg(&artifact_path)
        .arg("--plan")
        .arg(&plan_path)
        .arg("--out")
        .arg(&out)
        .arg("--json")
        .output()
        .unwrap();
    assert_eq!(mb_code(&output), 0, "stderr={}", String::from_utf8_lossy(&output.stderr));
    let report = mb_stdout(&output);
    assert_eq!(report["provenance"], "imported");
    assert_eq!(report["exerciser"], "imported-bytes");
    assert!(report["scope"].as_str().unwrap().contains("never upgraded"));
    let stored: Value = serde_json::from_slice(&fs::read(out.join("artifact.json")).unwrap()).unwrap();
    assert_eq!(stored["provenance"], "imported");
    assert_eq!(stored["submitted_provenance"], "native_observed");
    let receipt: Value = serde_json::from_slice(&fs::read(out.join("receipt.json")).unwrap()).unwrap();
    assert_eq!(receipt["status"], "imported");
    assert_eq!(receipt["source_sha256"], json!(mb_sha(&artifact_bytes)));

    let (code, inspection, _) = mb_run(&["inspect"], &[&out]);
    assert_eq!(code, 0);
    assert_eq!(inspection["provenance"], "imported");
    assert_eq!(inspection["submitted_provenance"], "native_observed");
}

#[test]
fn compare_uses_exact_envelope_arithmetic_at_the_declared_boundary() {
    let temp = Temp::new();
    let a = mb_cpu_plan("a01", START);
    let b = mb_cpu_plan("b01", START + 60_000);
    let a2 = mb_cpu_plan("a02", START + 120_000);
    let evidence = |name: &str, plan: &Value, durations: &[u64]| {
        mb_evidence(&temp, name, plan, &mb_cpu_artifact(plan, durations))
    };
    let base = evidence("a", &a, &[1000; 5]);
    let repeat = evidence("a2", &a2, &[1000; 5]);
    // 1050 ns is exactly 5% above the 1000 ns reference: equality passes.
    let edge = evidence("b", &b, &[1050; 5]);
    let compare = |baseline: &Path, candidate: &Path, reference: &Path| {
        let output = cli()
            .args(["microbench", "compare"])
            .arg(baseline)
            .arg(candidate)
            .arg("--reference")
            .arg(reference)
            .arg("--json")
            .output()
            .unwrap();
        (mb_code(&output), mb_stdout(&output))
    };

    let (code, report) = compare(&base, &edge, &repeat);
    assert_eq!(code, 0, "{report}");
    assert_eq!(report["decision"], "PASS");
    assert_eq!(report["statistic"], "mean-duration-ns");
    assert_eq!(report["direction"], "lower_better");
    assert_eq!(report["axis"], "adapter-revision");
    assert_eq!(report["candidate_range"][0]["numerator"], 1050);
    assert_eq!(report["candidate_range"][0]["denominator"], 1);
    assert_eq!(report["reference_range"][1]["numerator"], 1000);
    assert!(report["scope"].as_str().unwrap().contains("Imported comparison only"));
    assert!(report["scope"].as_str().unwrap().contains("never establish serving-speed"));

    // One nanosecond beyond the same boundary regresses.
    let worse = evidence("b2", &b, &[1051; 5]);
    let (code, report) = compare(&base, &worse, &repeat);
    assert_eq!(code, 3, "{report}");
    assert_eq!(report["decision"], "REGRESSION");

    // A reference spread wider than declared withholds the comparison.
    let mut wide_plan = a2.clone();
    wide_plan["acquisition"]["id"] = json!("a03");
    let wide = evidence("a3", &wide_plan, &[1300; 5]);
    let (code, report) = compare(&base, &worse, &wide);
    assert_eq!(code, 2, "{report}");
    assert_eq!(report["decision"], "INCONCLUSIVE");
    assert!(mb_has(&report, "reason_codes", "ReferenceSpreadExceeded"), "{report}");

    // The reference role is required before any favorable verdict.
    let output = cli()
        .args(["microbench", "compare"])
        .arg(&base)
        .arg(&edge)
        .arg("--json")
        .output()
        .unwrap();
    assert_eq!(mb_code(&output), 2);
    assert!(mb_has(&mb_stdout(&output), "reason_codes", "MissingReference"));
}

#[test]
fn compare_rejects_role_reuse_order_and_pin_drift() {
    let temp = Temp::new();
    let a = mb_cpu_plan("a01", START);
    let b = mb_cpu_plan("b01", START + 60_000);
    let a2 = mb_cpu_plan("a02", START + 120_000);
    let durations = [1000; 5];
    let base = mb_evidence(&temp, "a", &a, &mb_cpu_artifact(&a, &durations));
    let candidate = mb_evidence(&temp, "b", &b, &mb_cpu_artifact(&b, &durations));
    let compare = |baseline: &Path, candidate: &Path, reference: &Path| {
        let output = cli()
            .args(["microbench", "compare"])
            .arg(baseline)
            .arg(candidate)
            .arg("--reference")
            .arg(reference)
            .arg("--json")
            .output()
            .unwrap();
        (mb_code(&output), mb_stdout(&output))
    };

    // The same acquisition identity in two roles is role reuse.
    let reused = mb_evidence(&temp, "reuse", &a, &mb_cpu_artifact(&a, &durations));
    let (code, report) = compare(&base, &candidate, &reused);
    assert_eq!(code, 1, "{report}");
    assert!(mb_has(&report, "reason_codes", "RoleReuse"), "{report}");

    // Declared starts must be strictly ordered A < B < A2.
    let mut early = a2.clone();
    early["acquisition"]["started_unix_ms"] = json!(START + 1_000);
    let out_of_order = mb_evidence(&temp, "early", &early, &mb_cpu_artifact(&early, &durations));
    let (code, report) = compare(&base, &candidate, &out_of_order);
    assert_eq!(code, 1, "{report}");
    assert!(mb_has(&report, "reason_codes", "DeclaredStartsOutOfOrder"), "{report}");

    // Any pin other than the declared revision axis must match across roles.
    let mut thresholds = a2.clone();
    thresholds["thresholds"]["adverse_bps"] = json!(400);
    let drifted = mb_evidence(&temp, "thresholds", &thresholds, &mb_cpu_artifact(&thresholds, &durations));
    let (code, report) = compare(&base, &candidate, &drifted);
    assert_eq!(code, 1, "{report}");
    assert!(mb_has(&report, "reason_codes", "IdentityDrift"), "{report}");

    let repeat = mb_evidence(&temp, "a2", &a2, &mb_cpu_artifact(&a2, &durations));

    // A faster candidate whose correctness failed cannot pass.
    let mut failed = mb_cpu_artifact(&b, &[500; 5]);
    failed["correctness"]["computed"] = json!("8386559");
    failed["correctness"]["passed"] = json!(false);
    let broken = mb_evidence(&temp, "broken", &b, &failed);
    let (code, report) = compare(&base, &broken, &repeat);
    assert_eq!(code, 1, "{report}");
    assert_ne!(report["decision"], "PASS");
    assert!(mb_has(&report, "reason_codes", "CorrectnessFailed"), "{report}");
    assert_eq!(report["roles"]["candidate"]["eligible"], false);

    // A timed-out candidate acquisition is unavailable, not a fast run.
    let mut partial = mb_cpu_artifact(&b, &[500; 5]);
    partial["execution"]["timed_out"] = json!(true);
    partial["execution"]["completed"] = json!(false);
    let unavailable = mb_evidence(&temp, "unavailable", &b, &partial);
    let (code, report) = compare(&base, &unavailable, &repeat);
    assert_eq!(code, 2, "{report}");
    assert_eq!(report["decision"], "INCONCLUSIVE");
    assert!(mb_has(&report, "reason_codes", "DeadlineExceeded"), "{report}");
    assert_eq!(report["roles"]["candidate"]["statistic"], Value::Null);
}

#[test]
fn collective_evidence_keeps_rank_scope_bytes_and_clock_contract() {
    let temp = Temp::new();
    let plan = mb_reduce_plan("a01", START, 2, &"a".repeat(64));
    // Rank durations differ per repetition; one sample is the per-repetition max.
    let per_repetition: Vec<Vec<u64>> = (0..5).map(|index| vec![1000 + index, 2000 + index]).collect();
    let artifact = mb_reduce_artifact(&plan, &per_repetition, 0);
    let (code, report) = mb_inspect(&temp, "collective", &plan, &artifact);
    assert_eq!(code, 0, "{report}");
    assert_eq!(report["plan_verified"], true);
    assert_eq!(report["samples"], 10);
    // mean of per-repetition maxima (2000..2004) = 2002
    assert_eq!(report["statistic"]["numerator"], 2002);
    assert_eq!(report["statistic"]["denominator"], 1);
    assert_eq!(report["derived"]["payload_bytes"], 2097152);
    assert_eq!(report["derived"]["bus_normalization"]["numerator"], 1);
    assert_eq!(report["derived"]["bus_normalization"]["denominator"], 1);
    assert!(
        report["derived"]["definition"]
            .as_str()
            .unwrap()
            .contains("numel * sizeof(dtype)")
    );
    assert!(
        report["derived"]["bus_scope"]
            .as_str()
            .unwrap()
            .contains("not measured physical link traffic")
    );

    // A trailing world larger than two keeps the same payload bytes and a
    // different bus normalization; the two are never conflated.
    let three = mb_reduce_plan("a01", START, 3, &"a".repeat(64));
    let wide = mb_reduce_artifact(
        &three,
        &(0..5).map(|_| vec![1000, 1000, 1000]).collect::<Vec<Vec<u64>>>(),
        0,
    );
    let (code, report) = mb_inspect(&temp, "collective-3", &three, &wide);
    assert_eq!(code, 0, "{report}");
    assert_eq!(report["derived"]["payload_bytes"], 2097152);
    assert_eq!(report["derived"]["bus_normalization"]["numerator"], 4);
    assert_eq!(report["derived"]["bus_normalization"]["denominator"], 3);

    // Missing rank evidence is a rejected grid, never a silent survivor set.
    let kept: Vec<Value> = artifact["samples"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|sample| sample["rank"] == json!(0))
        .cloned()
        .collect();
    let mut missing = artifact.clone();
    missing["samples"] = json!(kept);
    let (code, report) = mb_inspect(&temp, "missing-rank", &plan, &missing);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "SampleGridIncomplete"), "{report}");

    let mismatched = mb_reduce_artifact(&plan, &per_repetition, 3);
    let (code, report) = mb_inspect(&temp, "mismatch", &plan, &mismatched);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "CorrectnessFailed"), "{report}");

    let mut unsynchronized = artifact.clone();
    unsynchronized["clock"]["synchronization"] = json!("process-clock-read-before-after");
    let (code, report) = mb_inspect(&temp, "collective-sync", &plan, &unsynchronized);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "ClockMismatch"), "{report}");

    let mut world_drift = artifact.clone();
    world_drift["topology"]["world"] = json!(3);
    world_drift["topology"]["ranks"] =
        json!([{"index": 0, "device": 0}, {"index": 1, "device": 1}, {"index": 2, "device": 2}]);
    let (code, report) = mb_inspect(&temp, "world-drift", &plan, &world_drift);
    assert_eq!(code, 1);
    assert!(!mb_reasons(&report, "invalid").is_empty(), "{report}");

    let mut timed_out = artifact.clone();
    timed_out["execution"]["timed_out"] = json!(true);
    let (code, report) = mb_inspect(&temp, "collective-timeout", &plan, &timed_out);
    assert_eq!(code, 2);
    assert!(mb_has(&report, "unavailable", "DeadlineExceeded"), "{report}");
}

#[test]
fn e3_parity_recomputation_rejects_widened_tolerances_and_tier_fallbacks() {
    let temp = Temp::new();
    let plan = mb_e3_plan("a01", START);
    let valid = mb_e3_artifact(&plan, "e3-grouped", "0.003000000");
    let (code, report) = mb_inspect(&temp, "e3-valid", &plan, &valid);
    assert_eq!(code, 0, "{report}");
    assert_eq!(report["samples"], 5);
    assert_eq!(report["provenance"], "native_observed");
    assert_eq!(report["exerciser"], "native-device-program-observation");

    // Importing a payload that claims native execution keeps the imported label.
    let artifact_path = temp.path("e3-artifact.json");
    mb_write(&artifact_path, &valid);
    let plan_path = temp.path("e3-plan.json");
    mb_write(&plan_path, &plan);
    let out = temp.path("e3-imported");
    let output = cli()
        .args(["microbench", "import", "--artifact"])
        .arg(&artifact_path)
        .arg("--plan")
        .arg(&plan_path)
        .arg("--out")
        .arg(&out)
        .arg("--json")
        .output()
        .unwrap();
    assert_eq!(mb_code(&output), 0, "{}", String::from_utf8_lossy(&output.stderr));
    let imported = mb_stdout(&output);
    assert_eq!(imported["provenance"], "imported");
    assert_eq!(imported["exerciser"], "imported-bytes");
    assert!(imported["scope"].as_str().unwrap().contains("cannot establish"));

    // A silent lower-tier fallback withholds the measurement.
    let fallback = mb_e3_artifact(&plan, "e2-kernel", "0.003000000");
    let (code, report) = mb_inspect(&temp, "e3-fallback", &plan, &fallback);
    assert_eq!(code, 2);
    assert!(mb_has(&report, "unavailable", "TierFallback"), "{report}");

    // Parity beyond the recomputed bound is a correctness failure.
    let wide = mb_e3_artifact(&plan, "e3-grouped", "0.004000000");
    let (code, report) = mb_inspect(&temp, "e3-parity", &plan, &wide);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "CorrectnessFailed"), "{report}");

    // A relaxed tolerance is an identity change, not a tolerable difference.
    let mut relaxed = mb_e3_artifact(&plan, "e3-grouped", "0.005000000");
    relaxed["correctness"]["tolerance"]["factor_milli"] = json!(2000);
    let (code, report) = mb_inspect(&temp, "e3-relaxed", &plan, &relaxed);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "IdentityDrift"), "{report}");
    assert!(mb_has(&report, "invalid", "CorrectnessFailed"), "{report}");

    let mut nonfinite = mb_e3_artifact(&plan, "e3-grouped", "0.003000000");
    nonfinite["correctness"]["finite"] = json!(false);
    let (code, report) = mb_inspect(&temp, "e3-nonfinite", &plan, &nonfinite);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "NonFinite"), "{report}");

    let mut drifted = mb_e3_artifact(&plan, "e3-grouped", "0.003000000");
    drifted["operation"]["cap"] = json!(64);
    let (code, report) = mb_inspect(&temp, "e3-cap-drift", &plan, &drifted);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "IdentityDrift"), "{report}");
}

#[test]
fn external_program_launch_pins_the_program_and_retains_failures() {
    let temp = Temp::new();

    // A local program that prints a prepared body. It performs no kernel or
    // collective work: this case exercises the collector's launch, hashing,
    // receipt and bounded-timeout paths only.
    let fixture = temp.path("fixture-adapter.sh");
    let body_path = temp.path("body.json");
    fs::write(
        &fixture,
        format!("#!/bin/sh\n# fixture: prints a prepared body, no device work\nexec cat '{}'\n", body_path.display()),
    )
    .unwrap();
    fs::set_permissions(&fixture, fs::Permissions::from_mode(0o755)).unwrap();
    let program_sha = mb_sha(&fs::read(&fixture).unwrap());

    let plan = mb_reduce_plan("a01", START, 2, &program_sha);
    let body = mb_reduce_artifact(&plan, &[vec![1000, 1000]; 5], 0);
    mb_write(&body_path, &body);
    let plan_path = temp.path("launch-plan.json");
    mb_write(&plan_path, &plan);
    let out = temp.path("launch");
    let output = cli()
        .args(["microbench", "capture", "--adapter", "nccl-allreduce-sum"])
        .arg("--plan")
        .arg(&plan_path)
        .arg("--out")
        .arg(&out)
        .arg("--program")
        .arg(&fixture)
        .arg("--authorize-device-window")
        .arg("--json")
        .output()
        .unwrap();
    assert_eq!(mb_code(&output), 0, "{}", String::from_utf8_lossy(&output.stderr));
    let report = mb_stdout(&output);
    assert_eq!(report["provenance"], "native_observed");
    assert_eq!(report["exerciser"], "native-collective-program-observation");
    assert_eq!(report["program_sha256"], program_sha);
    assert!(report["scope"].as_str().unwrap().contains("never sets a serving-speed claim"));
    let receipt: Value = serde_json::from_slice(&fs::read(out.join("receipt.json")).unwrap()).unwrap();
    assert_eq!(receipt["program_sha256"], program_sha);
    assert_eq!(receipt["status"], "captured");

    // A program that does not match the pinned hash never runs.
    let mut wrong = plan.clone();
    wrong["program_sha256"] = json!("e".repeat(64));
    let wrong_path = temp.path("wrong-plan.json");
    mb_write(&wrong_path, &wrong);
    let rejected = cli()
        .args(["microbench", "capture", "--adapter", "nccl-allreduce-sum"])
        .arg("--plan")
        .arg(&wrong_path)
        .arg("--out")
        .arg(temp.path("wrong"))
        .arg("--program")
        .arg(&fixture)
        .arg("--authorize-device-window")
        .output()
        .unwrap();
    assert_eq!(mb_code(&rejected), 1);
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("program_sha256"));
    assert!(!temp.path("wrong").join("artifact.json").exists());

    // Nonzero exit, malformed output and a missed deadline are retained as
    // failed receipts with no artifact, never as empty successful evidence.
    for (name, script, deadline) in [
        ("exit", "#!/bin/sh\nexit 1\n".to_string(), 120000),
        ("malformed", "#!/bin/sh\necho not-json\n".to_string(), 120000),
        ("timeout", "#!/bin/sh\nsleep 5\n".to_string(), 400),
    ] {
        let path = temp.path(&format!("{name}.sh"));
        fs::write(&path, script).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
        let mut scripted = plan.clone();
        scripted["program_sha256"] = json!(mb_sha(&fs::read(&path).unwrap()));
        scripted["allowance"]["deadline_ms"] = json!(deadline);
        let scripted_path = temp.path(&format!("{name}-plan.json"));
        mb_write(&scripted_path, &scripted);
        let dir = temp.path(name);
        let output = cli()
            .args(["microbench", "capture", "--adapter", "nccl-allreduce-sum"])
            .arg("--plan")
            .arg(&scripted_path)
            .arg("--out")
            .arg(&dir)
            .arg("--program")
            .arg(&path)
            .arg("--authorize-device-window")
            .arg("--json")
            .output()
            .unwrap();
        assert_eq!(mb_code(&output), 1, "{name}");
        assert!(!dir.join("artifact.json").exists(), "{name}");
        let receipt: Value =
            serde_json::from_slice(&fs::read(dir.join("receipt.json")).unwrap()).unwrap();
        assert_eq!(receipt["status"], "failed", "{name}");
        assert!(receipt["failure"].is_string(), "{name}");
        if name == "timeout" {
            assert!(
                receipt["failure"].as_str().unwrap().contains("deadline"),
                "{name}: {}",
                receipt["failure"]
            );
        }
        let inspected = cli()
            .args(["microbench", "inspect"])
            .arg(&dir)
            .arg("--json")
            .output()
            .unwrap();
        assert_eq!(mb_code(&inspected), 1, "{name}");
        assert!(
            String::from_utf8_lossy(&inspected.stderr).contains("MissingArtifact"),
            "{name}"
        );
    }
}
