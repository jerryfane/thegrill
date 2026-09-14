#!/usr/bin/env python3
"""CPU-only startup/restart domain behavioral fixture, never a model endpoint.

Run only after writers settle: python3 tools/smoke-startup.py --binary PATH --case all
All process launches, termination and restarts belong to this test harness.
The production CLI is only an observer/finite request collector. No GPU imports.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time

SOURCE = Path(__file__).with_name("startup-fixture.py").resolve()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n")


def port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def config(endpoint, count):
    names = ["vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:num_preemptions_total",
             "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
             "vllm:external_prefix_cache_queries_total", "vllm:external_prefix_cache_hits_total",
             "vllm:prompt_tokens_total", "vllm:prompt_tokens_by_source_total", "vllm:prompt_tokens_cached_total",
             "vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
             "vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_accepted_tokens_per_pos_total"]
    return {"version": 2, "endpoint": endpoint, "allowlist": names,
            "epochs": [name.removesuffix("_total") + "_created" for name in names[2:]],
            "scope": "server-wide-full-labels-not-workload-attributed", "source": "vllm:prometheus-exporter",
            "isolation": "shared-server-unattributed", "auth_env": None,
            "cadence": "before-finishes-before-measured-origin-after-starts-after-settlement",
            "deadline_us": 2000000, "body_bytes": 1048576, "line_bytes": 65536, "series": 256,
            "labels_per_series": 16, "label_bytes_per_series": 4096, "run_budget_us": 30000000,
            "retained_raw_bytes": 16777216, "max_requests": count * 2, "max_series": count * 512,
            "max_overhead_us": 1000000}


def plan(binary, endpoint, restart=False, count=1):
    request = {"model": "neutral", "prompt": "Return the decimal sum of integers zero through six.",
               "max_tokens": 8, "expected_answer": "21"}
    study = {"kind": "startup"}
    if restart:
        study = {"kind": "restart_cache",
                 "identity": {"version": 1, "prompt_sha256": digest(request["prompt"].encode()),
                              "source_sha256": digest(request["prompt"].encode()), "cache_salt": "fixed-study-key",
                              "hash_seed_contract": "PYTHONHASHSEED=0:engine+cache_server"},
                 "post_restart": [{"id": "first_reload"}] + [{"id": f"followup-{i}"} for i in range(1, count)],
                 "classes": [{"class": "engine", "transition": "changed"},
                             {"class": "cache_server", "transition": "persisted"}],
                 "no_warming_attestation": True}
    slots = []
    for arm in ["a", "b", "a2"]:
        slots.append({"id": f"{arm}-warmup", "arm": arm, "warmup": True, "index": 0})
        slots.extend({"id": f"{arm}-{i}", "arm": arm, "warmup": False, "index": i} for i in range(3))
    targets = ["listening", "ready", "first_valid_inference", "communication"]
    if restart:
        targets.append("reload")
    return {"version": 1, "kind": "startup-study-v1", "study_id": "cpu-neutral-study",
            "collector_sha256": digest(binary.read_bytes()), "adapter": "neutral_journal_v1",
            "adapter_sha256": digest(SOURCE.read_bytes()), "setup_axis": "synthetic-delay",
            "setup_sha256": ["1" * 64, "2" * 64, "1" * 64], "slots": slots, "study": study,
            "states": [{"class": name, "declared": "unknown", "required_observed": "unknown"}
                       for name in ["engine", "filesystem", "compiled_artifact_cache", "weights", "prefix_offload"]
                       + (["cache_server"] if restart else [])],
            "request": request, "endpoint": endpoint, "local_http": True, "auth_env": None,
            "metrics": config(endpoint.replace("/v1/chat/completions", "/metrics"), count) if restart else None,
            "max_events": 128, "event_bytes": 65536, "deadline_us": max(12000000, count * 7000000),
            "limits": {"total_ms": 2000, "idle_ms": 1500, "response_bytes": 65536, "wave_buffer_bytes": 1048576},
            "gates": [{"target": {"kind": target}, "max_regression_bps": 9999,
                       "max_reference_spread_bps": 1000000} for target in targets]}


def cli(binary, *args, expect=None):
    result = subprocess.run([str(binary), "startup", *map(str, args)], capture_output=True, text=True, timeout=90)
    if expect is not None:
        assert result.returncode == expect, (args, result.returncode, result.stdout, result.stderr)
    report = json.loads(result.stdout) if result.stdout else None
    return result, report


@contextmanager
def child(*args):
    process = subprocess.Popen([sys.executable, str(SOURCE), *map(str, args)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                               env={**os.environ, "PYTHONHASHSEED": "0"})
    try:
        yield process
    finally:
        # Only the harness-owned child; no production/service process discovery.
        if process.poll() is None:
            process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)


def run_capture(binary, root, plan_path, engine_port, slot=0, stage="startup", store=None,
                previous=None, cache=None, fault="none", delays=(80, 100, 120, 140), name="capture"):
    events = root / f"{name}.events"
    out = root / name
    args = ["engine", "--port", engine_port, "--plan", plan_path, "--events", events,
            "--stage", stage, "--fault", fault, "--listen-ms", delays[0],
            "--communication-ms", delays[1], "--ready-ms", delays[2], "--inference-ms", delays[3]]
    if cache:
        args += ["--cache-port", cache[0], "--cache-pid", cache[1]]
    with child(*args) as process:
        cmd = ["observe" if stage == "startup" else stage, plan_path, "--slot", slot,
               "--pid", process.pid, "--events", events, "--producer-source", SOURCE, "--out", out]
        if store:
            cmd += ["--store", store]
        if previous:
            cmd += ["--previous", previous]
        if cache:
            cmd += ["--cache-pid", cache[1]]
        result, report = cli(binary, *cmd)
        assert (out / "capture.json").exists(), (result.stderr, result.stdout)
        replay, offline = cli(binary, "inspect", out)
        assert result.returncode == replay.returncode and report == offline, (result, replay)
    return out, report


def durations(report):
    return {value["target"]["kind"]: value["duration_us"] for value in report["durations"]}


def rewrite_events(root, transform):
    events = [json.loads(line) for line in (root / "events.bin").read_text().splitlines()]
    transform(events)
    for i, event in enumerate(events):
        event["sequence"] = i
    raw = b"".join(json.dumps(event, separators=(",", ":")).encode() + b"\n" for event in events)
    (root / "events.bin").write_bytes(raw)
    receipt = json.loads((root / "capture.json").read_text())
    receipt["events_sha256"] = digest(raw)
    save(root / "capture.json", receipt)


def startup(binary, root):
    engine_port = port()
    p = plan(binary, f"http://127.0.0.1:{engine_port}/v1/chat/completions")
    plan_path = root / "startup-plan.json"
    save(plan_path, p)
    base, report = run_capture(binary, root, plan_path, engine_port, name="startup-base")
    assert report["outcome"] == "PASS", report
    values = durations(report)
    assert values["listening"] >= 80000 and values["communication"] >= 100000, values
    assert values["ready"] >= values["listening"] + values["communication"] + 120000, values
    assert values["first_valid_inference"] >= values["ready"] + 140000, values
    for name, delays, minimum in [("listen", (280, 100, 120, 140), ("listening", 280000)),
                                  ("comm", (80, 300, 120, 140), ("communication", 300000)),
                                  ("ready", (80, 100, 320, 140), ("ready", 500000)),
                                  ("inference", (80, 100, 120, 340), ("first_valid_inference", 640000))]:
        _, delayed = run_capture(binary, root, plan_path, engine_port, name=name, delays=delays)
        assert delayed["outcome"] == "PASS" and durations(delayed)[minimum[0]] >= minimum[1], delayed
    for fault in ["crash", "malformed", "bad_ready", "inference", "timeout", "clock", "stale", "missing_substage"]:
        out, failed = run_capture(binary, root, plan_path, engine_port, name=f"startup-{fault}", fault=fault)
        assert failed["outcome"] != "PASS", failed
        assert (out / "events.bin").exists()
    # Missing client response is not rescued by a producer's successful event.
    missing = root / "missing-client"
    shutil.copytree(base, missing)
    receipt = json.loads((missing / "capture.json").read_text())
    receipt["requests"] = []
    save(missing / "capture.json", receipt)
    _, report = cli(binary, "inspect", missing)
    assert report["outcome"] != "PASS", report
    print("startup: independent delays, exact replay, failed/malformed/clock/stale/missing boundaries covered")
    return base, p


def restart(binary, root):
    engine_port, cache_port = port(), port()
    p = plan(binary, f"http://127.0.0.1:{engine_port}/v1/chat/completions", restart=True, count=2)
    path = root / "restart-plan.json"
    save(path, p)
    with child("cache", "--port", cache_port) as cache_process:
        # Readiness is only harness synchronization, not a measured startup milestone.
        end = time.monotonic() + 5
        while True:
            try:
                with socket.create_connection(("127.0.0.1", cache_port), timeout=0.1):
                    break
            except OSError:
                assert time.monotonic() < end, "fixture cache did not listen"
                time.sleep(0.01)
        cache = (cache_port, cache_process.pid)
        stored, report = run_capture(binary, root, path, engine_port, stage="store", cache=cache, name="store")
        assert report["outcome"] == "PASS", report
        reloaded, report = run_capture(binary, root, path, engine_port, stage="reload", cache=cache,
                                       store=stored, name="reload")
        assert report["outcome"] == "PASS" and report["cache_result"] == "persisted_hit", report
        assert report["metrics"]["budget"]["requests"] == 4, report
        for fault in ["identity", "hidden", "extra", "reordered", "missing_reload", "missing_hit",
                      "missing_class", "missing_metrics", "unsealed"]:
            _, failed = run_capture(binary, root, path, engine_port, stage="reload", store=stored,
                                    cache=cache, name=f"reload-{fault}", fault=fault)
            assert failed["outcome"] != "PASS", failed
        # Correlated stale identity with all hashes repaired still cannot pass a restart.
        same = root / "same-engine"
        shutil.copytree(reloaded, same)
        old = json.loads((same / "store/capture.json").read_text())["engine"]
        receipt = json.loads((same / "capture.json").read_text())
        receipt["engine"] = old
        save(same / "capture.json", receipt)
        def same_identity(events):
            for event in events:
                event["engine"] = old
                if event["event"]["kind"] == "state" and event["event"]["state"]["class"] == "engine":
                    event["event"]["state"]["identity"]["identity"] = old
        rewrite_events(same, same_identity)
        _, report = cli(binary, "inspect", same)
        assert report["outcome"] == "ERROR" and any("Engine incarnation" in r for r in report["reasons"]), report
        imported = root / "imported-reload"
        _, report = cli(binary, "import", reloaded, "--out", imported)
        assert report["provenance"] == "imported" and report["outcome"] != "PASS", report
    # Full recomputation must remain distinguishable even with the same promised controls.
    with child("cache", "--port", cache_port, "--recompute") as cache_process:
        time.sleep(0.1)
        cache = (cache_port, cache_process.pid)
        stored, report = run_capture(binary, root, path, engine_port, stage="store", cache=cache, name="recompute-store")
        assert report["outcome"] == "PASS", report
        _, report = run_capture(binary, root, path, engine_port, stage="reload", cache=cache,
                                store=stored, name="recomputed")
        assert report["outcome"] != "PASS" and report["cache_result"] == "recomputed", report
    print("restart: actual store/process restart/reload, cache attribution, hidden requests and fail-closed imports covered")


def comparison(binary, root, base, original_plan):
    native = json.loads(json.dumps(original_plan))
    native["gates"] = [g for g in native["gates"] if g["target"]["kind"] == "communication"]
    native_path = root / "native-plan.json"
    save(native_path, native)
    engine_port = int(native["endpoint"].split(":")[2].split("/")[0])
    paths = []
    previous = None
    for index in range(len(native["slots"])):
        current, report = run_capture(binary, root, native_path, engine_port, slot=index,
                                      previous=previous, name=f"native-{index}")
        assert report["outcome"] == "PASS", report
        paths.append(str(current))
        previous = current
    manifest = root / "native-manifest.json"
    save(manifest, paths)
    _, report = cli(binary, "compare", native_path, manifest, expect=0)
    assert report["scope"] == "native-unauthenticated-domain-comparison-not-live-qualification", report
    assert report["gates"][0]["counts"] == [3, 3, 3], report
    # Exact arithmetic oracle: copied captures are deliberately marked IMPORTED by the CLI.
    # No claim of native execution is made for these synthetic event timings.
    p = json.loads(json.dumps(original_plan))
    p["gates"] = [{"target": {"kind": "ready"}, "max_regression_bps": 500, "max_reference_spread_bps": 0}]
    plan_path = root / "oracle-plan.json"
    save(plan_path, p)
    for candidate, expected in [(100000, "PASS"), (105000, "PASS"), (106000, "REGRESSION")]:
        paths = []
        previous = None
        for index, slot in enumerate(p["slots"]):
            source = root / f"oracle-source-{candidate}-{index}"
            shutil.copytree(base, source)
            shutil.copyfile(plan_path, source / "plan.json")
            receipt = json.loads((source / "capture.json").read_text())
            receipt.update(slot=index, plan_sha256=digest(plan_path.read_bytes()), previous_sha256=previous)
            receipt["engine"]["start_ticks"] += index
            save(source / "capture.json", receipt)
            ready = candidate if slot["arm"] == "b" else 100000
            def timestamps(events):
                for event in events:
                    event["engine"] = receipt["engine"]
                    if event["event"]["kind"] == "state" and event["event"]["state"]["class"] == "engine":
                        event["event"]["state"]["identity"]["identity"] = receipt["engine"]
                    kind = event["event"]["kind"]
                    event["offset_us"] = ({"launched": 0, "listening": 10000, "communication_start": 20000,
                                           "communication_end": 30000, "ready": ready}.get(kind, 1)
                                          if event["sequence"] <= 11 else ready + event["sequence"] * 1000)
                    if kind in ["request_start", "inference_complete", "sealed"]:
                        event["offset_us"] = ready + event["sequence"] * 1000
            rewrite_events(source, timestamps)
            previous = digest((source / "capture.json").read_bytes())
            imported = root / f"oracle-import-{candidate}-{index}"
            _, report = cli(binary, "import", source, "--out", imported, expect=0)
            assert report["provenance"] == "imported"
            paths.append(str(imported))
        manifest = root / f"oracle-{candidate}.json"
        save(manifest, paths)
        _, report = cli(binary, "compare", plan_path, manifest)
        assert report["outcome"] == expected, report
        assert report["scope"] == "imported-comparison-only-not-real-adapter-exercised", report
        assert report["gates"][0]["counts"] == [3, 3, 3], report
        save(manifest, paths[:-1])
        _, report = cli(binary, "compare", plan_path, manifest)
        assert report["outcome"] != "PASS", report
        save(manifest, [paths[1], paths[0], *paths[2:]])
        result, _ = cli(binary, "compare", plan_path, manifest)
        assert result.returncode == 1, result
    print("comparison: independent 5% exact boundary oracle, all-role counts, missing and reordered acquisitions covered")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--case", choices=["all", "startup", "restart", "comparison"], default="all")
    args = parser.parse_args()
    binary = args.binary.resolve()
    with tempfile.TemporaryDirectory(prefix="grill-startup-cpu-") as temp:
        root = Path(temp)
        if args.case in ["all", "startup", "comparison"]:
            base, p = startup(binary, root)
            if args.case in ["all", "comparison"]:
                comparison(binary, root, base, p)
        if args.case in ["all", "restart"]:
            restart(binary, root)
    print("CPU startup/restart smoke passed; no model/device or recipe qualification claimed")


if __name__ == "__main__":
    main()
