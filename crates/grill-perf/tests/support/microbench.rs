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
const E3_SCRIPT_SHA256: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

const START: u64 = 1_757_000_000_000;
const COLLECTOR: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
const COLLECTOR_OTHER: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
const CPU_BOUND_REFERENCE: &str = "8386560";
const CPU_FIXTURE_KERNEL: &str = "cpu-sum-u64-reference-v1;collector:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
const CPU_FIXTURE_REVISION: &str = "188b240187fb83628d7a32f34da598ec6d305dce66ca251ca300dae47d0ddbf3";

fn mb_native_cpu_plan(id: &str, started: u64) -> Value {
    let binary_sha = mb_sha(&fs::read(cli().get_program()).unwrap());
    let kernel = format!("cpu-sum-u64-reference-v1;collector:{binary_sha}");
    mb_cpu_plan(id, started, &mb_sha(kernel.as_bytes()))
}

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

fn mb_dir(path: &Path) -> PathBuf {
    fs::create_dir_all(path).unwrap();
    path.to_path_buf()
}

fn mb_cpu_plan(id: &str, started: u64, revision: &str) -> Value {
    json!({
        "kind": "microbench-plan-v1",
        "version": 1,
        "adapter": "cpu-sum-u64-reference",
        "revision": revision,
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

fn mb_reduce_plan(id: &str, started: u64, world: u32) -> Value {
    let program_sha256 = E3_SCRIPT_SHA256;
    json!({
        "kind": "microbench-plan-v1",
        "version": 1,
        "adapter": "nccl-allreduce-sum",
        "revision": mb_sha(b"torch:2.9.0;nccl:2.27.7"),
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
        "revision": mb_sha(b"torch:2.9.0;nccl:2.27.7;exl3-module:eeee"),
        "program_sha256": E3_SCRIPT_SHA256,
        "sources": [
            {"path": "tools/microbench/exl3_e3_grouped.py", "sha256": E3_SCRIPT_SHA256},
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

/// One retained sample: the timer's own text plus the exact nanosecond rational.
fn mb_sample(index: u32, rank: Option<u32>, repetition: Option<u32>, raw: &str, num: u64, den: u64) -> Value {
    json!({
        "index": index,
        "rank": rank,
        "repetition": repetition,
        "raw": raw,
        "duration": {"numerator": num, "denominator": den}
    })
}

fn mb_cpu_artifact(plan: &Value, samples: &[(&str, u64, u64)]) -> Value {
    let values: Vec<Value> = samples
        .iter()
        .enumerate()
        .map(|(index, (raw, num, den))| mb_sample(index as u32, None, None, raw, *num, *den))
        .collect();
    json!({
        "kind": "microbench-artifact-v1",
        "version": 1,
        "adapter": plan["adapter"],
        "revision": plan["revision"],
        "acquisition": plan["acquisition"],
        "operation": plan["operation"],
        "provenance": "native_observed",
        "submitted_provenance": null,
        "program_sha256": null,
        "kernel_revision": CPU_FIXTURE_KERNEL,
        "sources": [],
        "observations": [],
        "topology": {"scope": "cpu", "world": null, "ranks": []},
        "clock": plan["clock"],
        "execution": {
            "warmups": 2, "iterations": 5, "deadline_ms": 5000,
            "work_units": 28672, "memory_bytes": 32768,
            "completed": true, "timed_out": false, "observed_fallback": null
        },
        "samples": values,
        "correctness": {
            "kind": "exact-sum", "reference": "n*(n-1)/2",
            "computed": CPU_BOUND_REFERENCE, "bound": CPU_BOUND_REFERENCE, "passed": true
        },
        "failures": []
    })
}

fn mb_cpu_uniform(plan: &Value, nanos: u64) -> Value {
    let raw = nanos.to_string();
    let samples: Vec<(&str, u64, u64)> = (0..5).map(|_| (raw.as_str(), nanos, 1)).collect();
    mb_cpu_artifact(plan, &samples)
}

fn mb_observations(entries: &[(&str, &str)]) -> Value {
    Value::Array(
        entries
            .iter()
            .map(|(name, value)| json!({"name": name, "value": value}))
            .collect(),
    )
}

fn mb_e3_observations(fallback: &str) -> Value {
    let fallback = match fallback {
        "e3-grouped" => "grouped",
        "e2-kernel" => "kernel",
        other => other,
    };
    mb_observations(&[
        ("torch-version", "2.9.0"),
        ("nccl-version", "2.27.7"),
        ("exl3-module-sha256", &"e".repeat(64)),
        ("device", "fixture-device"),
        ("experts", "288"),
        ("topk", "8"),
        ("tokens", "1024"),
        ("hidden", "4096"),
        ("intermediate", "1024"),
        ("cap", "32"),
        ("skew-milli", "1000"),
        ("routing-seed", "1"),
        ("layer-seed", "0"),
        ("parity-routing-seed", "3"),
        ("activation-seed", "3"),
        ("fallback-tier", fallback),
        ("sources-verified", "3"),
        ("sources-declared", "0"),
    ])
}

fn mb_e3_artifact(plan: &Value, fallback: &str, e3_maxabs: &str) -> Value {
    let samples: Vec<Value> = (0..5)
        .map(|index| mb_sample(index, None, None, "31.2345678", 156_172_839, 5))
        .collect();
    json!({
        "kind": "microbench-artifact-v1",
        "version": 1,
        "adapter": plan["adapter"],
        "acquisition": plan["acquisition"],
        "revision": plan["revision"],
        "operation": plan["operation"],
        "provenance": "native_observed",
        "submitted_provenance": null,
        "program_sha256": plan["program_sha256"],
        "kernel_revision": "torch:2.9.0;nccl:2.27.7;exl3-module:eeee",
        "sources": [
            {"path": "/observed/tools/microbench/exl3_e3_grouped.py", "sha256": E3_SCRIPT_SHA256},
            {"path": "/observed/tests/bench_e3_microbench.py", "sha256": E3_ANCHOR_SHA256},
            {"path": "/observed/tests/test_exl3_overlay.py", "sha256": E3_PARITY_SHA256}
        ],
        "observations": mb_e3_observations(fallback),
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

fn mb_reduce_artifact(plan: &Value, per_repetition: &[Vec<u64>], mismatches: u64) -> Value {
    let world = plan["operation"]["world"].as_u64().unwrap() as u32;
    let mut samples = Vec::new();
    for (repetition, ranks) in per_repetition.iter().enumerate() {
        for (rank, nanos) in ranks.iter().enumerate() {
            samples.push(mb_sample(
                samples.len() as u32,
                Some(rank as u32),
                Some(repetition as u32),
                &nanos.to_string(),
                *nanos,
                1,
            ));
        }
    }
    json!({
        "kind": "microbench-artifact-v1",
        "version": 1,
        "adapter": plan["adapter"],
        "revision": plan["revision"],
        "acquisition": plan["acquisition"],
        "operation": plan["operation"],
        "provenance": "native_observed",
        "submitted_provenance": null,
        "program_sha256": plan["program_sha256"],
        "kernel_revision": "torch:2.9.0;nccl:2.27.7",
        "sources": [
            {"path": "/observed/tools/microbench/nccl_allreduce_sum.py", "sha256": plan["program_sha256"]},
            {"path": "doc/PERFORMANCE.md", "sha256": NCCL_BYTE_SOURCE_SHA256}
        ],
        "observations": mb_observations(&[
            ("torch-version", "2.9.0"),
            ("nccl-version", "2.27.7"),
            ("device", "fixture-device"),
            ("world", &world.to_string()),
            ("numel", "262144"),
            ("dtype", "int64"),
            ("op", "sum"),
            ("sources-verified", "1"),
            ("sources-declared", "1")
        ]),
        "topology": {
            "scope": "ranks",
            "world": world,
            "ranks": (0..world).map(|r| json!({"index": r, "device": r})).collect::<Vec<Value>>()
        },
        "clock": plan["clock"],
        "execution": {
            "warmups": 1, "iterations": 5, "deadline_ms": 120000,
            "work_units": 1572864, "memory_bytes": 6291456,
            "completed": true, "timed_out": false, "observed_fallback": null
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

fn mb_receipt(collector: &str) -> Value {
    json!({
        "kind": "microbench-receipt-v1",
        "version": 1,
        "adapter": "cpu-sum-u64-reference",
        "provenance": "native_observed",
        "status": "captured",
        "plan_sha256": "0".repeat(64),
        "artifact_sha256": "1".repeat(64),
        "collector_sha256": collector,
        "program": null,
        "program_sha256": null,
        "source_sha256": null,
        "started_unix_ms": START,
        "observation_unix_ms": START,
        "duration_ms": 0,
        "command": [],
        "failure": null,
        "claim": "fixture"
    })
}

/// Write one acquisition directory: `plan.json`, `artifact.json` and a receipt.
fn mb_member(role: &Path, slot: &str, plan: &Value, artifact: &Value, collector: &str) {
    let dir = mb_dir(&role.join(slot));
    mb_write(&dir.join("plan.json"), plan);
    mb_write(&dir.join("artifact.json"), artifact);
    let mut receipt = mb_receipt(collector);
    receipt["adapter"] = artifact["adapter"].clone();
    receipt["provenance"] = artifact["provenance"].clone();
    receipt["program_sha256"] = artifact["program_sha256"].clone();
    receipt["plan_sha256"] = json!(mb_sha(&fs::read(dir.join("plan.json")).unwrap()));
    receipt["artifact_sha256"] = json!(mb_sha(&fs::read(dir.join("artifact.json")).unwrap()));
    receipt["started_unix_ms"] = plan["acquisition"]["started_unix_ms"].clone();
    receipt["observation_unix_ms"] = plan["acquisition"]["started_unix_ms"].clone();
    mb_write(&dir.join("receipt.json"), &receipt);
}

/// A complete CPU role. `slot` is the declared slot prefix, which is
/// independent of the directory name so a role can be relocated in a fixture.
fn mb_cpu_role(temp: &Temp, name: &str, slot: &str, revisions: &[(&str, u64, u64)]) -> PathBuf {
    let role = mb_dir(&temp.path(name));
    for (index, (revision, started, nanos)) in revisions.iter().enumerate() {
        let id = format!("{slot}{index:02}");
        let plan = mb_cpu_plan(&id, *started, revision);
        mb_member(&role, &id, &plan, &mb_cpu_uniform(&plan, *nanos), COLLECTOR);
    }
    role
}

fn mb_study(baseline: &str, candidate: &str, slots: usize, minimum: usize) -> Value {
    let ids = |prefix: &str| -> Vec<Value> {
        (0..slots)
            .map(|index| json!(format!("{prefix}{index:02}")))
            .collect()
    };
    json!({
        "kind": "microbench-study-v1",
        "version": 1,
        "adapter": "cpu-sum-u64-reference",
        "revision_axis": {"baseline": baseline, "candidate": candidate},
        "thresholds": {"adverse_bps": 500, "spread_bps": 2000},
        "minimum_acquisitions": minimum,
        "roles": {
            "baseline": ids("baseline"),
            "candidate": ids("candidate"),
            "reference": ids("reference")
        },
        "started_unix_ms": START - 1000
    })
}

fn mb_study_file(temp: &Temp, study: &Value) -> PathBuf {
    let path = temp.path("study.json");
    mb_write(&path, study);
    path
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
    let dir = mb_dir(&temp.path(name));
    mb_write(&dir.join("plan.json"), plan);
    mb_write(&dir.join("artifact.json"), artifact);
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

fn mb_compare(
    temp: &Temp,
    name: &str,
    study: &Value,
    baseline: &Path,
    candidate: &Path,
    reference: &Path,
) -> (i32, Value, String) {
    let study_path = temp.path(&format!("{name}-study.json"));
    mb_write(&study_path, study);
    let output = cli()
        .args(["microbench", "compare", "--study"])
        .arg(&study_path)
        .arg("--baseline")
        .arg(baseline)
        .arg("--candidate")
        .arg(candidate)
        .arg("--reference")
        .arg(reference)
        .arg("--json")
        .output()
        .unwrap();
    (
        mb_code(&output),
        mb_stdout(&output),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

#[test]
fn cpu_reference_capture_records_exact_rational_samples() {
    let temp = Temp::new();
    let plan = mb_native_cpu_plan("a01", START);
    let (code, report, stderr, dir) = mb_capture(&temp, "cpu", "cpu-sum-u64-reference", &plan);
    assert_eq!(code, 0, "stderr={stderr}");
    assert_eq!(report["provenance"], "native_observed");
    assert_eq!(report["exerciser"], "native-cpu-reference");
    assert_eq!(report["samples"], 5);
    assert_eq!(report["invalid"], json!([]));

    let artifact: Value = serde_json::from_slice(&fs::read(dir.join("artifact.json")).unwrap()).unwrap();
    assert_eq!(artifact["observations"], json!([]));
    assert_eq!(artifact["sources"], json!([]));
    assert_eq!(artifact["correctness"]["computed"], CPU_BOUND_REFERENCE);
    assert_eq!(artifact["execution"]["completed"], true);
    let sample = &artifact["samples"][0];
    let raw: u64 = sample["raw"].as_str().unwrap().parse().unwrap();
    assert!(raw > 0);
    // Integer nanosecond readings are retained as the exact rational value/1.
    assert_eq!(sample["duration"]["numerator"], json!(raw));
    assert_eq!(sample["duration"]["denominator"], json!(1));
    assert!(report["scope"].as_str().unwrap().contains("never sets a serving-speed claim"));

    let inspected = cli()
        .args(["microbench", "inspect"])
        .arg(&dir)
        .arg("--json")
        .output()
        .unwrap();
    assert_eq!(mb_code(&inspected), 0);
    let report = mb_stdout(&inspected);
    assert_eq!(report["plan_verified"], true);
    assert_eq!(report["invalid"], json!([]));
    assert!(report["statistic"]["numerator"].as_u64().unwrap() > 0);
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
        let mut plan = mb_cpu_plan("a01", START, CPU_FIXTURE_REVISION);
        mutate(&mut plan);
        let (code, _report, stderr, dir) = mb_capture(&temp, name, "cpu-sum-u64-reference", &plan);
        assert_eq!(code, 1, "{name} stderr={stderr}");
        assert!(stderr.contains("rejected"), "{name} stderr={stderr}");
        assert!(!dir.join("artifact.json").exists(), "{name}");
    }

    // A plan declared for another adapter cannot be captured as the CPU reference.
    let reduce = mb_reduce_plan("a01", START, 2);
    let (code, _report, stderr, _) = mb_capture(&temp, "mismatch", "cpu-sum-u64-reference", &reduce);
    assert_eq!(code, 1);
    assert!(stderr.contains("adapter"), "{stderr}");

    // The in-process reference refuses an operator program.
    let plan_path = temp.path("cpu-inproc-plan.json");
    mb_write(&plan_path, &mb_native_cpu_plan("a01", START));
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
fn inspect_accepts_fractional_and_scientific_durations() {
    let temp = Temp::new();
    let plan = mb_e3_plan("a01", START);

    // The acceptance case: 0.123456789012345 ms is not a whole nanosecond.
    let mut artifact = mb_e3_artifact(&plan, "e3-grouped", "0.003000000");
    for (index, sample) in artifact["samples"].as_array_mut().unwrap().iter_mut().enumerate() {
        *sample = mb_sample(index as u32, None, None, "0.123456789012345", 24_691_357_802_469, 200_000_000);
    }
    let (code, report) = mb_inspect(&temp, "long-decimal", &plan, &artifact);
    assert_eq!(code, 0, "{report}");
    assert_eq!(report["invalid"], json!([]));

    // Scientific notation and a sub-resolution digit are both preserved.
    let mut scientific = mb_e3_artifact(&plan, "e3-grouped", "0.003000000");
    for (index, sample) in scientific["samples"].as_array_mut().unwrap().iter_mut().enumerate() {
        *sample = mb_sample(index as u32, None, None, "1.234e-05", 617, 50);
    }
    let (code, report) = mb_inspect(&temp, "scientific", &plan, &scientific);
    assert_eq!(code, 0, "{report}");

    // 31.2345678 ms is 156172839/5 ns: finer than the declared 1000 ns
    // resolution and still retained exactly.
    let mut fractional = mb_e3_artifact(&plan, "e3-grouped", "0.003000000");
    for (index, sample) in fractional["samples"].as_array_mut().unwrap().iter_mut().enumerate() {
        *sample = mb_sample(index as u32, None, None, "31.2345678", 156_172_839, 5);
    }
    let (code, report) = mb_inspect(&temp, "fractional", &plan, &fractional);
    assert_eq!(code, 0, "{report}");

    // A fractional-nanosecond digit is part of the acceptance surface, not a
    // rounding target: claiming a rounded value is rejected.
    let mut rounded = fractional.clone();
    rounded["samples"][0]["duration"] = json!({"numerator": 31234568, "denominator": 1});
    let (code, report) = mb_inspect(&temp, "rounded", &plan, &rounded);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "SAMPLE_PRECISION"), "{report}");

    // A zero reading is not a usable duration for this cell.
    let mut zero = fractional.clone();
    zero["samples"][0] = mb_sample(0, None, None, "0.0", 0, 1);
    let (code, report) = mb_inspect(&temp, "zero", &plan, &zero);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "DURATION_OUT_OF_RANGE"), "{report}");

    // Negative, non-finite and out-of-range values stay fail-closed.
    for (name, raw, num, den, reason) in [
        ("negative", "-1.5", 3, 2, "SAMPLE_PRECISION"),
        ("nan", "nan", 1, 1, "SAMPLE_PRECISION"),
        ("inf", "inf", 1, 1, "SAMPLE_PRECISION"),
        ("overflow", "1e30", 1, 1, "SAMPLE_OVERFLOW"),
        ("zero-denominator", "31.0", 31, 0, "SAMPLE_PRECISION"),
    ] {
        let mut broken = fractional.clone();
        broken["samples"][0] = mb_sample(0, None, None, raw, num, den);
        let (code, report) = mb_inspect(&temp, name, &plan, &broken);
        assert_eq!(code, 1, "{name}");
        assert!(mb_has(&report, "invalid", reason), "{name}: {report}");
    }
}

#[test]
fn inspect_rejects_iteration_clock_units_and_correctness_drift() {
    let temp = Temp::new();
    let plan = mb_cpu_plan("a01", START, CPU_FIXTURE_REVISION);
    let base = mb_cpu_uniform(&plan, 1000);

    let mut missing = base.clone();
    missing["samples"].as_array_mut().unwrap().pop();
    let (code, report) = mb_inspect(&temp, "missing-iteration", &plan, &missing);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "SAMPLE_GRID_INCOMPLETE"), "{report}");

    let mut warmups = base.clone();
    warmups["execution"]["warmups"] = json!(0);
    let (code, report) = mb_inspect(&temp, "warmups", &plan, &warmups);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "WARMUP_COUNT_MISMATCH"), "{report}");

    let mut iterations = base.clone();
    iterations["execution"]["iterations"] = json!(4);
    let (code, report) = mb_inspect(&temp, "iterations", &plan, &iterations);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "ITERATION_COUNT_MISMATCH"), "{report}");

    let mut units = base.clone();
    units["clock"]["units"] = json!("milliseconds");
    let (code, report) = mb_inspect(&temp, "units", &plan, &units);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "CLOCK_MISMATCH"), "{report}");

    let mut synchronization = base.clone();
    synchronization["clock"]["kind"] = json!("device_event");
    let (code, report) = mb_inspect(&temp, "sync", &plan, &synchronization);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "CLOCK_MISMATCH"), "{report}");

    let mut inverted = base.clone();
    inverted["samples"][2]["rank"] = json!(0);
    let (code, report) = mb_inspect(&temp, "grid", &plan, &inverted);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "SAMPLE_GRID_INCOMPLETE"), "{report}");

    let mut incorrect = base.clone();
    incorrect["correctness"]["computed"] = json!("8386561");
    incorrect["correctness"]["passed"] = json!(false);
    let (code, report) = mb_inspect(&temp, "correctness", &plan, &incorrect);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "CORRECTNESS_FAILED"), "{report}");

    let mut drifted = base.clone();
    drifted["operation"]["bound"] = json!(2048);
    let (code, report) = mb_inspect(&temp, "operation-drift", &plan, &drifted);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "IDENTITY_DRIFT"), "{report}");

    let mut partial = base.clone();
    partial["execution"]["completed"] = json!(false);
    partial["execution"]["timed_out"] = json!(true);
    let (code, report) = mb_inspect(&temp, "timeout", &plan, &partial);
    assert_eq!(code, 2);
    assert!(mb_has(&report, "unavailable", "DEADLINE_EXCEEDED"), "{report}");
    assert!(mb_has(&report, "unavailable", "SAMPLE_GRID_INCOMPLETE"), "{report}");

    let mut retained = base.clone();
    retained["failures"] = json!([{"kind": "adapter-error", "detail": "fixture retained failure"}]);
    let (code, report) = mb_inspect(&temp, "retained", &plan, &retained);
    assert_eq!(code, 2);
    assert!(mb_has(&report, "unavailable", "RETAINED_FAILURE"), "{report}");
}

#[test]
fn inspect_checks_observed_sources_and_runtime_observations() {
    let temp = Temp::new();
    let plan = mb_e3_plan("a01", START);
    let base = mb_e3_artifact(&plan, "e3-grouped", "0.003000000");
    let (code, report) = mb_inspect(&temp, "e3", &plan, &base);
    assert_eq!(code, 0, "{report}");
    assert_eq!(report["samples"], 5);

    // Observed source paths may differ from the declared hints; the hash set is
    // what must match.
    let mut renamed = base.clone();
    renamed["sources"][2]["path"] = json!("/elsewhere/test_exl3_overlay.py");
    let (code, report) = mb_inspect(&temp, "renamed-source", &plan, &renamed);
    assert_eq!(code, 0, "{report}");

    // A substituted source hash is identity drift even though the path matches.
    let mut substituted = base.clone();
    substituted["sources"][2]["sha256"] = json!(E3_ANCHOR_SHA256);
    let (code, report) = mb_inspect(&temp, "substituted-source", &plan, &substituted);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "IDENTITY_DRIFT"), "{report}");

    // A pinned source the producer never observed is drift.
    let mut dropped = base.clone();
    dropped["sources"].as_array_mut().unwrap().pop();
    let (code, report) = mb_inspect(&temp, "dropped-source", &plan, &dropped);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "IDENTITY_DRIFT"), "{report}");

    // Missing and drifted runtime observations are rejected, never ignored.
    let mut missing = base.clone();
    missing["observations"] = Value::Array(
        missing["observations"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|observation| observation["name"] != json!("topk"))
            .cloned()
            .collect(),
    );
    let (code, report) = mb_inspect(&temp, "missing-observation", &plan, &missing);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "OBSERVATION_MISSING"), "{report}");

    let mut drifted = base.clone();
    for observation in drifted["observations"].as_array_mut().unwrap() {
        if observation["name"] == json!("cap") {
            observation["value"] = json!("64");
        }
    }
    let (code, report) = mb_inspect(&temp, "drifted-observation", &plan, &drifted);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "OBSERVATION_DRIFT"), "{report}");

    // An unavailable fallback tier is reported as unavailable and rejected,
    // never accepted as a declared value.
    let mut unavailable = base.clone();
    unavailable["observations"] = mb_e3_observations("unavailable");
    unavailable["execution"]["observed_fallback"] = Value::Null;
    let (code, report) = mb_inspect(&temp, "fallback-unavailable", &plan, &unavailable);
    assert_eq!(code, 2);
    assert_eq!(report["invalid"], json!([]));
    assert!(mb_has(&report, "unavailable", "TIER_FALLBACK"), "{report}");

    // A device adapter never reports the CPU reference's empty observation set.
    let mut cpu = mb_cpu_uniform(&mb_cpu_plan("a01", START, CPU_FIXTURE_REVISION), 1000);
    cpu["observations"] = mb_e3_observations("e3-grouped");
    let (code, report) = mb_inspect(
        &temp,
        "cpu-observations",
        &mb_cpu_plan("a01", START, CPU_FIXTURE_REVISION),
        &cpu,
    );
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "INVALID_ARTIFACT"), "{report}");
}

#[test]
fn import_never_upgrades_a_native_provenance_claim() {
    let temp = Temp::new();
    let plan = mb_cpu_plan("a01", START, CPU_FIXTURE_REVISION);
    let artifact = mb_cpu_uniform(&plan, 1000);
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
}

#[test]
fn group_compare_requires_three_complete_acquisitions_per_role() {
    let temp = Temp::new();
    let study = mb_study(CPU_FIXTURE_REVISION, CPU_FIXTURE_REVISION, 3, 3);
    let baseline = mb_cpu_role(
        &temp,
        "baseline",
        "baseline",
        &[(CPU_FIXTURE_REVISION, START, 1000), (CPU_FIXTURE_REVISION, START + 1_000, 1000), (CPU_FIXTURE_REVISION, START + 2_000, 1000)],
    );
    let candidate = mb_cpu_role(
        &temp,
        "candidate",
        "candidate",
        &[(CPU_FIXTURE_REVISION, START + 60_000, 1000), (CPU_FIXTURE_REVISION, START + 61_000, 1000), (CPU_FIXTURE_REVISION, START + 62_000, 1000)],
    );
    let reference = mb_cpu_role(
        &temp,
        "reference",
        "reference",
        &[(CPU_FIXTURE_REVISION, START + 120_000, 1000), (CPU_FIXTURE_REVISION, START + 121_000, 1000), (CPU_FIXTURE_REVISION, START + 122_000, 1000)],
    );

    let (code, report, stderr) = mb_compare(&temp, "complete", &study, &baseline, &candidate, &reference);
    assert_eq!(code, 0, "stderr={stderr} {report}");
    assert_eq!(report["decision"], "PASS");
    assert_eq!(report["study"]["declared_acquisitions"], 9);
    assert_eq!(report["study"]["complete_acquisitions"], 9);
    assert_eq!(report["roles"]["baseline"]["complete"], 3);
    assert_eq!(report["roles"]["candidate"]["complete"], 3);
    assert_eq!(report["roles"]["reference"]["complete"], 3);
    assert_eq!(report["axis_baseline"], CPU_FIXTURE_REVISION);
    assert_eq!(report["axis_candidate"], CPU_FIXTURE_REVISION);
    assert_eq!(report["reason_codes"], json!([]));

    // A role that declares fewer than the minimum is rejected outright.
    let short_study = mb_study(CPU_FIXTURE_REVISION, CPU_FIXTURE_REVISION, 2, 2);
    let (code, report, _) = mb_compare(&temp, "short", &short_study, &baseline, &candidate, &reference);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "reason_codes", "ROLE_MINIMUM"), "{report}");

    // A declared acquisition that is not present withholds the verdict; it is
    // never replaced or filtered into a survivor set.
    let missing_study = mb_study(CPU_FIXTURE_REVISION, CPU_FIXTURE_REVISION, 4, 3);
    let (code, report, _) = mb_compare(&temp, "missing", &missing_study, &baseline, &candidate, &reference);
    assert_eq!(code, 2);
    assert_eq!(report["decision"], "INCONCLUSIVE");
    assert!(mb_has(&report, "reason_codes", "MISSING_ACQUISITION"), "{report}");

    // An unexpected acquisition directory is rejected, not ignored.
    let extra_plan = mb_cpu_plan("candidate99", START + 90_000, CPU_FIXTURE_REVISION);
    let extra = mb_dir(&candidate.join("candidate99"));
    mb_write(&extra.join("plan.json"), &extra_plan);
    mb_write(&extra.join("artifact.json"), &mb_cpu_uniform(&extra_plan, 1000));
    let (code, report, _) = mb_compare(&temp, "extra", &study, &baseline, &candidate, &reference);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "reason_codes", "UNEXPECTED_ACQUISITION"), "{report}");
    fs::remove_dir_all(&extra).unwrap();

    // A member that did not finish is unavailable, not a smaller sample.
    let partial_role = mb_dir(&temp.path("partial"));
    for (index, started) in [START + 60_000u64, START + 61_000, START + 62_000].into_iter().enumerate() {
        let id = format!("candidate{index:02}");
        let plan = mb_cpu_plan(&id, started, CPU_FIXTURE_REVISION);
        let mut artifact = mb_cpu_uniform(&plan, 1000);
        if index == 0 {
            artifact["execution"]["timed_out"] = json!(true);
            artifact["execution"]["completed"] = json!(false);
        }
        mb_member(&partial_role, &id, &plan, &artifact, COLLECTOR);
    }
    let (code, report, _) = mb_compare(&temp, "partial", &study, &baseline, &partial_role, &reference);
    assert_eq!(code, 2);
    assert_eq!(report["roles"]["candidate"]["eligible"], false);
    assert!(mb_has(&report, "reason_codes", "DEADLINE_EXCEEDED"), "{report}");

    // A rejected member (failed numerical correctness) is an error, and a
    // faster candidate can never outweigh it.
    let broken_role = mb_dir(&temp.path("broken"));
    for (index, started) in [START + 60_000u64, START + 61_000, START + 62_000].into_iter().enumerate() {
        let id = format!("candidate{index:02}");
        let plan = mb_cpu_plan(&id, started, CPU_FIXTURE_REVISION);
        let mut artifact = mb_cpu_uniform(&plan, 1);
        if index == 0 {
            artifact["correctness"]["computed"] = json!("8386559");
            artifact["correctness"]["passed"] = json!(false);
        }
        mb_member(&broken_role, &id, &plan, &artifact, COLLECTOR);
    }
    let (code, report, _) = mb_compare(&temp, "broken", &study, &baseline, &broken_role, &reference);
    assert_eq!(code, 1);
    assert_ne!(report["decision"], "PASS");
    assert!(mb_has(&report, "reason_codes", "CORRECTNESS_FAILED"), "{report}");
}

#[test]
fn group_compare_envelope_boundaries_and_pooled_reference() {
    let temp = Temp::new();
    let study = mb_study(CPU_FIXTURE_REVISION, CPU_FIXTURE_REVISION, 3, 3);
    let baseline = mb_cpu_role(
        &temp,
        "baseline",
        "baseline",
        &[(CPU_FIXTURE_REVISION, START, 1000), (CPU_FIXTURE_REVISION, START + 1_000, 1000), (CPU_FIXTURE_REVISION, START + 2_000, 1000)],
    );
    let reference = mb_cpu_role(
        &temp,
        "reference",
        "reference",
        &[(CPU_FIXTURE_REVISION, START + 120_000, 1010), (CPU_FIXTURE_REVISION, START + 121_000, 1010), (CPU_FIXTURE_REVISION, START + 122_000, 1010)],
    );
    let edge = mb_cpu_role(
        &temp,
        "edge",
        "candidate",
        &[(CPU_FIXTURE_REVISION, START + 60_000, 1050), (CPU_FIXTURE_REVISION, START + 61_000, 1050), (CPU_FIXTURE_REVISION, START + 62_000, 1050)],
    );
    let worse = mb_cpu_role(
        &temp,
        "worse",
        "candidate",
        &[(CPU_FIXTURE_REVISION, START + 60_000, 1061), (CPU_FIXTURE_REVISION, START + 61_000, 1061), (CPU_FIXTURE_REVISION, START + 62_000, 1061)],
    );

    // The pooled A/A2 envelope is [1000, 1010]. 1050 is exactly the
    // admissible upper boundary; 1061 exceeds 5% even against the high end.
    let (code, report, stderr) = mb_compare(&temp, "edge", &study, &baseline, &edge, &reference);
    assert_eq!(code, 0, "stderr={stderr} {report}");
    assert_eq!(report["reference_range"][0]["numerator"], 1000);
    assert_eq!(report["reference_range"][1]["numerator"], 1010);
    assert_eq!(report["candidate_range"][0]["numerator"], 1050);
    assert_eq!(report["roles"]["candidate"]["range"][0]["numerator"], 1050);

    let (code, report, _) = mb_compare(&temp, "worse", &study, &baseline, &worse, &reference);
    assert_eq!(code, 3);
    assert_eq!(report["decision"], "REGRESSION");

    // A reference spread wider than declared withholds the comparison.
    let wide = mb_cpu_role(
        &temp,
        "wide",
        "reference",
        &[(CPU_FIXTURE_REVISION, START + 120_000, 1300), (CPU_FIXTURE_REVISION, START + 121_000, 1300), (CPU_FIXTURE_REVISION, START + 122_000, 1300)],
    );
    let (code, report, _) = mb_compare(&temp, "spread", &study, &baseline, &edge, &wide);
    assert_eq!(code, 2);
    assert_eq!(report["decision"], "INCONCLUSIVE");
    assert!(mb_has(&report, "reason_codes", "REFERENCE_SPREAD_EXCEEDED"), "{report}");
}

#[test]
fn group_compare_rejects_membership_axis_and_collector_drift() {
    let temp = Temp::new();
    let study = mb_study(CPU_FIXTURE_REVISION, CPU_FIXTURE_REVISION, 3, 3);
    let revisions = |revision: &'static str, started: u64| {
        vec![
            (revision, started, 1000u64),
            (revision, started + 1000, 1000),
            (revision, started + 2000, 1000),
        ]
    };
    let baseline = mb_cpu_role(&temp, "baseline", "baseline", &revisions(CPU_FIXTURE_REVISION, START));
    let candidate = mb_cpu_role(&temp, "candidate", "candidate", &revisions(CPU_FIXTURE_REVISION, START + 60_000));
    let reference = mb_cpu_role(&temp, "reference", "reference", &revisions(CPU_FIXTURE_REVISION, START + 120_000));

    // A declaration cannot claim a different implementation than was observed.
    let mut wrong_study = study.clone();
    wrong_study["revision_axis"]["candidate"] = json!(mb_sha(b"different-implementation"));
    let (code, report, _) = mb_compare(&temp, "axis", &wrong_study, &baseline, &candidate, &reference);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "reason_codes", "OBSERVATION_DRIFT"), "{report}");

    // Role starts must be strictly ordered A < B < A2.
    let early = mb_cpu_role(&temp, "early", "reference", &revisions(CPU_FIXTURE_REVISION, START + 1_000));
    let (code, report, _) = mb_compare(&temp, "order", &study, &baseline, &candidate, &early);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "reason_codes", "OBSERVED_STARTS_OUT_OF_ORDER"), "{report}");

    // Within one role the observed order must also be strictly increasing.
    let shuffled = mb_cpu_role(
        &temp,
        "shuffled",
        "baseline",
        &[(CPU_FIXTURE_REVISION, START, 1000), (CPU_FIXTURE_REVISION, START, 1000), (CPU_FIXTURE_REVISION, START + 2000, 1000)],
    );
    let (code, report, _) = mb_compare(&temp, "shuffled", &study, &shuffled, &candidate, &reference);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "reason_codes", "OBSERVED_STARTS_OUT_OF_ORDER"), "{report}");

    // A different collector for one member breaks the collector contract.
    let mixed = mb_dir(&temp.path("mixed"));
    for (index, started) in [START + 60_000u64, START + 61_000, START + 62_000].into_iter().enumerate() {
        let id = format!("candidate{index:02}");
        let plan = mb_cpu_plan(&id, started, CPU_FIXTURE_REVISION);
        let collector = if index == 1 { COLLECTOR_OTHER } else { COLLECTOR };
        mb_member(&mixed, &id, &plan, &mb_cpu_uniform(&plan, 1000), collector);
    }
    let (code, report, _) = mb_compare(&temp, "collector", &study, &baseline, &mixed, &reference);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "reason_codes", "COLLECTOR_MISMATCH"), "{report}");

    // The same acquisition reused in two roles is role reuse.
    let reused = mb_dir(&temp.path("reused"));
    for (index, started) in [START, START + 1000, START + 2000].into_iter().enumerate() {
        let id = format!("baseline{index:02}");
        let plan = mb_cpu_plan(&id, started, CPU_FIXTURE_REVISION);
        mb_member(&reused, &id, &plan, &mb_cpu_uniform(&plan, 1000), COLLECTOR);
    }
    let (code, report, _) = mb_compare(&temp, "reuse", &study, &baseline, &candidate, &reused);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "reason_codes", "ROLE_REUSE"), "{report}");

    // A member whose pins differ from the rest of the study is identity drift.
    let drifted = mb_dir(&temp.path("drifted"));
    for (index, started) in [START + 60_000u64, START + 61_000, START + 62_000].into_iter().enumerate() {
        let id = format!("candidate{index:02}");
        let mut plan = mb_cpu_plan(&id, started, CPU_FIXTURE_REVISION);
        let artifact = if index == 0 {
            plan["allowance"]["work_units"] = json!(30000);
            mb_cpu_uniform(&plan, 1000)
        } else {
            mb_cpu_uniform(&plan, 1000)
        };
        mb_member(&drifted, &id, &plan, &artifact, COLLECTOR);
    }
    let (code, report, _) = mb_compare(&temp, "pins", &study, &baseline, &drifted, &reference);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "reason_codes", "IDENTITY_DRIFT"), "{report}");

    // An invalid study declaration is rejected before any comparison.
    let (code, report, _) = mb_compare(
        &temp,
        "incomplete-study",
        &json!({"kind": "microbench-study-v1", "version": 1}),
        &baseline,
        &candidate,
        &reference,
    );
    assert_eq!(code, 1);
    assert!(mb_has(&report, "reason_codes", "INVALID_STUDY"), "{report}");
}

#[test]
fn collective_world_samples_use_per_repetition_maxima() {
    let temp = Temp::new();
    let plan = mb_reduce_plan("a01", START, 2);
    let per_repetition: Vec<Vec<u64>> = (0..5).map(|index| vec![1000 + index, 2000 + index]).collect();
    let artifact = mb_reduce_artifact(&plan, &per_repetition, 0);
    let (code, report) = mb_inspect(&temp, "collective", &plan, &artifact);
    assert_eq!(code, 0, "{report}");
    assert_eq!(report["samples"], 10);
    // mean of per-repetition maxima (2000..2004) = 2002
    assert_eq!(report["statistic"]["numerator"], 2002);
    assert_eq!(report["statistic"]["denominator"], 1);
    assert_eq!(report["derived"]["payload_bytes"], 2097152);
    assert_eq!(report["derived"]["bus_normalization"]["numerator"], 1);
    assert_eq!(report["derived"]["bus_normalization"]["denominator"], 1);
    assert!(
        report["derived"]["bus_scope"]
            .as_str()
            .unwrap()
            .contains("not measured physical link traffic")
    );

    // A world larger than two keeps identical payload bytes and a different
    // bus normalization; the two are never conflated.
    let three = mb_reduce_plan("a01", START, 3);
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
    assert_eq!(report["statistic"]["numerator"], 1000);

    // A missing rank is a rejected grid, never a survivor set.
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
    assert!(mb_has(&report, "invalid", "SAMPLE_GRID_INCOMPLETE"), "{report}");

    // A world observation that disagrees with the topology is rejected.
    let mut world_drift = artifact.clone();
    for observation in world_drift["observations"].as_array_mut().unwrap() {
        if observation["name"] == json!("world") {
            observation["value"] = json!("3");
        }
    }
    let (code, report) = mb_inspect(&temp, "world-observation", &plan, &world_drift);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "OBSERVATION_DRIFT"), "{report}");

    // A collective that could not verify any of its own bytes is rejected.
    let mut unverified = artifact.clone();
    for observation in unverified["observations"].as_array_mut().unwrap() {
        if observation["name"] == json!("sources-verified") {
            observation["value"] = json!("0");
        }
    }
    let (code, report) = mb_inspect(&temp, "unverified", &plan, &unverified);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "OBSERVATION_MISSING"), "{report}");

    let mismatched = mb_reduce_artifact(&plan, &per_repetition, 3);
    let (code, report) = mb_inspect(&temp, "mismatch", &plan, &mismatched);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "CORRECTNESS_FAILED"), "{report}");
}

#[test]
fn e3_parity_tolerance_boundaries_and_tier_fallbacks() {
    let temp = Temp::new();
    let plan = mb_e3_plan("a01", START);

    // Exactly on the recomputed maxabs and nRMSE bounds passes.
    let mut boundary = mb_e3_artifact(&plan, "e3-grouped", "0.003500000");
    boundary["correctness"]["e3"]["nrmse"] = json!("0.015100000");
    let (code, report) = mb_inspect(&temp, "e3-boundary", &plan, &boundary);
    assert_eq!(code, 0, "{report}");

    // One micro-unit beyond the bound fails.
    let (code, report) = mb_inspect(
        &temp,
        "e3-over",
        &plan,
        &mb_e3_artifact(&plan, "e3-grouped", "0.003600000"),
    );
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "CORRECTNESS_FAILED"), "{report}");

    // A silent lower-tier fallback withholds the measurement.
    let fallback = mb_e3_artifact(&plan, "e2-kernel", "0.003000000");
    let (code, report) = mb_inspect(&temp, "e3-fallback", &plan, &fallback);
    assert_eq!(code, 2);
    assert!(mb_has(&report, "unavailable", "TIER_FALLBACK"), "{report}");

    // A relaxed tolerance is an identity change, not a tolerable difference.
    let mut relaxed = mb_e3_artifact(&plan, "e3-grouped", "0.005000000");
    relaxed["correctness"]["tolerance"]["factor_milli"] = json!(2000);
    let (code, report) = mb_inspect(&temp, "e3-relaxed", &plan, &relaxed);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "IDENTITY_DRIFT"), "{report}");
    assert!(mb_has(&report, "invalid", "CORRECTNESS_FAILED"), "{report}");

    let mut nonfinite = mb_e3_artifact(&plan, "e3-grouped", "0.003000000");
    nonfinite["correctness"]["finite"] = json!(false);
    let (code, report) = mb_inspect(&temp, "e3-nonfinite", &plan, &nonfinite);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "NON_FINITE"), "{report}");

    let mut cap_drift = mb_e3_artifact(&plan, "e3-grouped", "0.003000000");
    cap_drift["operation"]["cap"] = json!(64);
    let (code, report) = mb_inspect(&temp, "e3-cap", &plan, &cap_drift);
    assert_eq!(code, 1);
    assert!(mb_has(&report, "invalid", "IDENTITY_DRIFT"), "{report}");
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

    let mut plan = mb_reduce_plan("a01", START, 2);
    plan["program_sha256"] = json!(program_sha);
    plan["sources"] = json!([
        {"path": "tools/microbench/nccl_allreduce_sum.py", "sha256": program_sha},
        {"path": "doc/PERFORMANCE.md", "sha256": NCCL_BYTE_SOURCE_SHA256}
    ]);
    let body = mb_reduce_artifact(&plan, &std::array::from_fn::<_, 5, _>(|_| vec![1000, 1000]), 0);
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
    assert_eq!(mb_code(&output), 0, "stdout={} stderr={}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    let report = mb_stdout(&output);
    assert_eq!(report["provenance"], "native_observed");
    assert_eq!(report["exerciser"], "native-collective-program-observation");
    assert_eq!(report["program_sha256"], program_sha);
    assert!(report["scope"].as_str().unwrap().contains("not authenticated execution"));
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
            assert!(receipt["duration_ms"].as_u64().unwrap() >= deadline, "{name}: {receipt}");
        }
        let inspected = cli()
            .args(["microbench", "inspect"])
            .arg(&dir)
            .arg("--json")
            .output()
            .unwrap();
        assert_eq!(mb_code(&inspected), 1, "{name}");
    }
}
