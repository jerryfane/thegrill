#!/usr/bin/env python3
"""Main-owned CPU verification: finite ordinary fixture, no live model endpoint.

Every invocation retains stdout/stderr and all native evidence in a fresh root.
No reruns, replacements, fallback assertions or real-backend qualification.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("retention_journal", ROOT / "retention-journal.py")
journal_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(journal_module)


def workload():
    steps = [
        ("prime", "h", None, "Remember sapphire.", "sapphire"),
        ("pressure", "p", None, "Remember amber.", "amber"),
        ("probe", "h", "prime", "Probe the remembered fact.", "sapphire"),
        ("recover", "h", "prime", "Repeat the remembered fact.", "sapphire"),
    ]
    return {
        "version": 6, "name": "finite-retention-fixture",
        "request": {"profile": "vllm-conversation-v3", "stream": True,
                    "output": {"tokens": 8, "mode": "cap"}, "cache": "observe", "seed": 0},
        "limits": {"total_ms": 3000, "idle_ms": 1000, "response_bytes": 65536, "wave_buffer_bytes": 16777216},
        "cases": [{"id": id, "messages": [{"role": "user", "content": content}],
                   "step": {"history": history, "parent": parent, "cache": "observe",
                            "expect": {"kind": "json", "value": {"fact": fact}, "strict": None}}}
                  for id, history, parent, content, fact in steps],
        "cells": [{"id": id, "case": id, "concurrency": 1, "warmup_trials": 0, "trials": 1} for id, *_ in steps],
        "acquisition": {"kind": "conversation", "repetitions": 1, "warmup_repetitions": 0,
                        "measured_steps": ["probe", "recover"], "input_bytes": 65536, "retained_history_bytes": 1048576},
    }


def capacity_plan(cells):
    return {"version": 1, "kind": "capacity-study-v1", "cells": cells,
            "max_requests": 100, "max_output_tokens": 1000, "max_wall_us": 60000000,
            "retained_bytes": 2147483648, "stop_condition": "first_unsuccessful_cell",
            "oom_response": "stop_operator_recovery"}


def cell(name, context, w):
    return {"id": name, "context_token_ceiling": context, "workload": w,
            "success": {"quality": "all_sequence_checks", "require_prompt_usage": True,
                        "resources": {"kind": "not_required_no_resource_claim"}}}


def command(binary, root, name, args, expected):
    full = [str(binary), *map(str, args)]
    output = subprocess.run(full, capture_output=True, timeout=90)
    (root / (name + ".stdout")).write_bytes(output.stdout)
    (root / (name + ".stderr")).write_bytes(output.stderr)
    (root / (name + ".command.json")).write_text(json.dumps(full))
    if output.returncode != expected:
        raise AssertionError((name, output.returncode, expected, output.stderr.decode(errors="replace")))
    return json.loads(output.stdout)


def fixture(root, args):
    journal = root / "producer.jsonl"
    ready = root / "ready.json"
    log = (root / "fixture.stderr").open("wb")
    process = subprocess.Popen([sys.executable, str(ROOT / "retention-fixture.py"), "--journal", str(journal),
                                "--ready", str(ready), "--max-seconds", "120", *args], stdout=log, stderr=log)
    deadline = time.monotonic() + 10
    while not ready.exists():
        if process.poll() is not None or time.monotonic() >= deadline:
            raise AssertionError("ordinary fixture failed to become ready")
        time.sleep(0.01)
    port = json.loads(ready.read_text())["port"]
    return process, log, journal, f"http://127.0.0.1:{port}"


def stop(process, log):
    # Only this harness-owned ordinary fixture, never a serving process.
    process.terminate()
    process.wait(timeout=5)
    log.close()


def retention_case(binary, root, name, fixture_args, condition, satisfied, eviction, hit, counters,
                   isolation="exclusive-single-acquisition"):
    target = root / name
    target.mkdir()
    process, log, journal, endpoint = fixture(target, fixture_args)
    try:
        c = cell(name, 512, workload())
        c["retention"] = {"version": 1, "prime": "prime", "pressure": ["pressure"], "probe": "probe", "recovery": ["recover"],
                          "intended_pressure": "One independent root history, finite and declared before requests",
                          "condition": condition, "source": "ordinary-lru-fixture-v1",
                          "producer_sha256": hashlib.sha256((ROOT / "retention-journal.py").read_bytes()).hexdigest(),
                          "metrics_labels": {"model_name": "fixture-model", "engine": "0"},
                          "journal_bytes": 8388608, "max_events": 10000, "max_window_us": 300000000}
        plan = target / "plan.json"
        plan.write_text(json.dumps(capacity_plan([c])))
        command(binary, target, "preflight", ["capacity", "preflight", plan], 0)
        isolation_args = [] if isolation is None else ["--metrics-isolation", isolation]
        report = command(binary, target, "run", ["capacity", "run", plan, "--endpoint", endpoint + "/v1/chat/completions",
                         "--model", "fixture-model", "--local-http", "--metrics-url", endpoint + "/metrics",
                         *isolation_args, "--eviction-journal", journal, "--out", target / "capture"],
                         0 if satisfied or condition == "observe_only" else 2)
        replay = command(binary, target, "inspect", ["capacity", "inspect", target / "capture"], 0)
        assert replay == report
        retained = report["cells"][0]["retention"]
        assert retained["condition_satisfied"] is satisfied
        assert retained["counter_coverage_complete"] is counters
        execution = json.loads((target / "capture/execution.json").read_text())
        native_plan = json.loads((target / "capture/cell-000/plan.json").read_text())
        assert execution["metrics_isolation"] == isolation
        assert native_plan["metrics"]["isolation"] == (isolation or "shared-server-unattributed")
        if condition != "observe_only" and isolation != "exclusive-single-acquisition":
            assert report["cells"][0]["state"] == "unsuccessful_tested"
            assert any("exclusive-single-acquisition" in reason for reason in retained["unavailable"])
        group = retained["acquisitions"][0]
        assert group["probe_hit"] is hit
        if eviction is None:
            assert group["observed_evicted_target_keys"] is None
        else:
            assert bool(group["observed_evicted_target_keys"]) is eviction
        assert retained["fixture_not_real_backend"] is True
        assert retained["live_qualified"] is False
        # Actual acquired parent remains in the native reservation.
        reservation = json.loads((target / "capture/cell-000/wave-000002/reservation.json").read_text())
        parent = json.loads(reservation["requests"][0])["messages"][1]
        assert parent["role"] == "assistant" and json.loads(parent["content"])["fact"] == "sapphire"
    finally:
        stop(process, log)


def capacity_failure(binary, root):
    target = root / "successful-then-failed"
    target.mkdir()
    process, log, _, endpoint = fixture(target, ["--capacity", "8", "--fail-after", "4"])
    try:
        cells = [cell("small", 128, workload()), cell("larger", 256, workload()), cell("never-started", 512, workload())]
        plan = target / "plan.json"
        plan.write_text(json.dumps(capacity_plan(cells)))
        report = command(binary, target, "run", ["capacity", "run", plan, "--endpoint", endpoint + "/v1/chat/completions",
                         "--model", "fixture-model", "--local-http", "--out", target / "capture"], 2)
        assert [c["state"] for c in report["cells"]] == ["successful_tested", "unsuccessful_tested", "undispatched"]
        assert [c["dispatched_requests"] for c in report["cells"]] == [4, 1, 0]
        assert [c["undispatched_requests"] for c in report["cells"]] == [0, 3, 4]
        assert report["largest_successful_tested_cell"] == "small"
        assert not (target / "capture/cell-002").exists()
        assert b"out_of_memory" in (target / "capture/cell-001/wave-000000/response-0000.bin").read_bytes()
        assert command(binary, target, "inspect", ["capacity", "inspect", target / "capture"], 0) == report
    finally:
        stop(process, log)


def source_sets(root):
    # Synthetic files exercise whole-set admission without fetching/importing vLLM.
    target = root / "source-sets"
    target.mkdir()
    names = tuple(journal_module.PINS)
    sources, pins = [], []
    for revision in ["old", "new"]:
        directory = target / revision
        directory.mkdir()
        paths, expected = {}, {}
        for index, name in enumerate(names):
            raw = f"{revision}-{name}\n".encode()
            path = directory / f"source-{index}.py"
            path.write_bytes(raw)
            paths[name] = path
            expected[name] = hashlib.sha256(raw).hexdigest()
        sources.append(paths)
        pins.append(expected)
    contracts = [journal_module.SOURCE, journal_module.SOURCE_487]

    def rejected(paths, source):
        try:
            journal_module.verify_sources(paths, source)
        except ValueError:
            return
        raise AssertionError(("unreviewed source set admitted", source))

    with patch.object(journal_module, "PINS", pins[0]), patch.object(journal_module, "PINS_487", pins[1]):
        for index, source in enumerate(contracts):
            assert journal_module.verify_sources(sources[index], source) == source
            rejected(sources[1 - index], source)
        for mask in range(1, (1 << len(names)) - 1):
            mixed = {name: sources[(mask >> index) & 1][name] for index, name in enumerate(names)}
            for source in contracts:
                rejected(mixed, source)
        for index, source in enumerate(contracts):
            rejected({name: path for name, path in sources[index].items() if name != names[0]}, source)
            rejected({**sources[index], "unreviewed.py": sources[index][names[0]]}, source)
            rejected(sources[index], journal_module.FIXTURE)
            rejected(sources[index], "vllm-latest")
            sources[index][names[0]].write_bytes(b"drift after selection\n")
            rejected(sources[index], source)
    for source in contracts:
        rejected(sources[0], source)  # Real public pins reject synthetic bytes.
    # Refuse absent, unknown and mismatched explicit selections before any import.
    for selected in [None, journal_module.FIXTURE, "vllm-latest", journal_module.SOURCE_487]:
        with patch.object(journal_module.importlib, "import_module", side_effect=AssertionError("unexpected runtime import")):
            try:
                journal = SimpleNamespace(source=journal_module.SOURCE)
                if selected is None:
                    journal_module.install_vllm(journal)
                else:
                    journal_module.install_vllm(journal, source=selected)
            except (ValueError, TypeError):
                pass
            else:
                raise AssertionError(("invalid installer selection admitted", selected))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(mode=0o700)  # Fresh only: failures are retained, never rerun.
    binary = args.binary.resolve()
    source_sets(args.out)
    capacity_failure(binary, args.out)
    retention_case(binary, args.out, "eviction-recovery", ["--capacity", "1"], "eviction_and_recovery", True, True, False, True)
    retention_case(binary, args.out, "retained-hit", ["--capacity", "8"], "retained_hit", True, False, True, True)
    retention_case(binary, args.out, "missing-counters", ["--capacity", "1", "--missing-counters"], "eviction_and_recovery", False, None, False, False)
    retention_case(binary, args.out, "non-eviction-miss", ["--capacity", "8", "--miss-on-probe"], "observe_only", False, False, False, True)
    retention_case(binary, args.out, "shared-isolation", ["--capacity", "1"], "eviction_and_recovery",
                   False, None, False, True, isolation="shared-server-unattributed")
    retention_case(binary, args.out, "missing-isolation", ["--capacity", "8"], "retained_hit",
                   False, None, True, True, isolation=None)
    retention_case(binary, args.out, "shared-observe-only", ["--capacity", "8"], "observe_only",
                   False, False, True, True, isolation="shared-server-unattributed")
    print(json.dumps({"status": "all-finite-ordinary-fixtures-passed", "real_backend_qualification": False, "out": str(args.out)}))


if __name__ == "__main__":
    main()
