#!/usr/bin/env python3
"""Later CPU verification of the real bridge with an ordinary ASGI process.

No vLLM/Uvicorn/device imports or live endpoints. Legacy startup/restart cases
remain in smoke-startup.py; run both scripts at the integrated head.
"""
import argparse
from contextlib import contextmanager, nullcontext
import http.client
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


HERE = Path(__file__).resolve().parent
old = module("startup_smoke", HERE / "smoke-startup.py")
bridge = module("startup_runtime", HERE / "startup-runtime.py")


@contextmanager
def child(*args):
    process = subprocess.Popen([sys.executable, str(HERE / "startup-asgi-fixture.py"), *map(str, args)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                               env={**os.environ, "PYTHONHASHSEED": "0"})
    try:
        yield process
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)


def events(path):
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return []
    return [json.loads(line) for line in raw.splitlines(keepends=True) if line.endswith(b"\n")]


def wait_event(path, kind, deadline=5):
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        if any(row["event"]["kind"] == kind for row in events(path)):
            return
        time.sleep(0.01)
    raise AssertionError(("missing fixture event", kind, path))


def external(port):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        connection.request("POST", "/v1/chat/completions", b'{"messages":[{"role":"user","content":"hidden"}]}',
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        response.read(65537)
        assert response.status == 200, response.status
    finally:
        connection.close()


def run(binary, root, fault="none", extra=None, delays=(80, 100, 120), bootstrap_ms=None):
    port = old.port()
    plan = old.plan(binary, f"http://127.0.0.1:{port}/v1/chat/completions")
    plan["adapter"] = "runtime_asgi_fixture_v1" if bootstrap_ms is None else "runtime_asgi_fixture_bootstrap_v1"
    plan["adapter_sha256"] = old.digest((HERE / "startup-runtime.py").read_bytes())
    plan["runtime_window_us"] = 2000000
    plan["gates"] = [gate for gate in plan["gates"] if gate["target"]["kind"] != "communication"]
    if bootstrap_ms is not None:
        plan["gates"].append({"target": {"kind": "bootstrap"}, "max_regression_bps": 500,
                              "max_reference_spread_bps": 500})
    plan_path, journal, out = root / "plan.json", root / "journal.ndjson", root / "capture"
    old.save(plan_path, plan)
    with child("--plan", plan_path, "--events", journal, "--port", port, "--fault", fault,
               "--ready-ms", delays[0], "--listen-ms", delays[1], "--inference-ms", delays[2],
               "--bootstrap-ms", bootstrap_ms if bootstrap_ms is not None else 0) as server:
        if extra == "before":
            wait_event(journal, "listening")
            external(port)
        command = [str(binary), "startup", "observe", str(plan_path), "--slot", "0", "--pid", str(server.pid),
                   "--events", str(journal), "--producer-source", str(HERE / "startup-runtime.py"), "--out", str(out)]
        collector = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            if extra == "after":
                wait_event(journal, "inference_complete")
                external(port)  # Must be seen after the planned response, before the interval seal.
            stdout, stderr = collector.communicate(timeout=30)
        finally:
            if collector.poll() is None:
                collector.kill()
                collector.communicate(timeout=5)
        assert (out / "capture.json").exists(), (stdout, stderr)
        report = json.loads(stdout)
        result, offline = old.cli(binary, "inspect", out)
        assert offline == report and result.returncode == collector.returncode, (report, offline)
        if fault == "none" and extra is None:
            assert report["outcome"] == "PASS", report
            assert report["launch_anchor"] == "bridge-entrypoint-not-os-launch", report
            samples = old.durations(report)
            assert samples["ready"] >= delays[0] * 1000, samples
            assert samples["listening"] >= samples["ready"] + delays[1] * 1000, samples
            assert samples["first_valid_inference"] >= samples["listening"] + delays[2] * 1000, samples
            rows = events(out / "events.bin")
            ready = next(e for e in rows if e["event"]["kind"] == "ready")
            if bootstrap_ms is not None:
                start = next(e for e in rows if e["event"]["kind"] == "bootstrap_start")
                end = next(e for e in rows if e["event"]["kind"] == "bootstrap_end")
                assert samples["bootstrap"] == end["offset_us"] - start["offset_us"]
                assert samples["bootstrap"] >= bootstrap_ms * 1000
                assert ready["offset_us"] - end["offset_us"] >= delays[0] * 1000
                assert start["clock"] == end["clock"] == ready["clock"] == rows[0]["clock"]
                assert start["engine"] == end["engine"] == ready["engine"] == rows[0]["engine"]
                assert rows[0]["offset_us"] < start["offset_us"] < end["offset_us"] < ready["offset_us"]
            coverage = next(e for e in rows if e["event"]["kind"] == "runtime_coverage")
            assert coverage["offset_us"] - ready["offset_us"] >= plan["runtime_window_us"]
            # An early count-based seal would finish before this required exposure.
            assert rows[-1]["event"] == {"kind": "sealed", "requests": 1}, rows[-1]
            assert report["cache_result"] == "unavailable", report
        else:
            assert report["outcome"] != "PASS", report
            if bootstrap_ms is not None:
                assert "bootstrap" not in old.durations(report), report
                if fault == "bootstrap_failed":
                    assert not any(e["event"]["kind"] == "bootstrap_end" for e in events(out / "events.bin"))
            if extra is not None:
                ids = [e["event"]["id"] for e in events(out / "events.bin") if e["event"]["kind"] == "request_start"]
                assert any(value.startswith("external-") for value in ids), ids
            if fault == "hidden_engine":
                assert any("internal AsyncLLM" in e["event"].get("detail", "")
                           for e in events(out / "events.bin"))
        return report


def restart_unavailable(binary, root):
    port, cache_port = old.port(), old.port()
    plan = old.plan(binary, f"http://127.0.0.1:{port}/v1/chat/completions", restart=True)
    plan["adapter"] = "runtime_asgi_fixture_v1"
    plan["adapter_sha256"] = old.digest((HERE / "startup-runtime.py").read_bytes())
    plan["runtime_window_us"] = 2000000
    plan["gates"] = [gate for gate in plan["gates"] if gate["target"]["kind"] == "reload"]
    plan_path = root / "restart-plan.json"
    old.save(plan_path, plan)
    identities = []
    with old.child("cache", "--port", cache_port) as cache:
        for stage in ["store", "reload"]:
            journal, out = root / f"{stage}.events", root / stage
            with child("--plan", plan_path, "--events", journal, "--port", port, "--stage", stage) as server:
                cmd = [stage, plan_path, "--slot", 0, "--pid", server.pid, "--cache-pid", cache.pid,
                       "--events", journal, "--producer-source", HERE / "startup-runtime.py", "--out", out]
                if stage == "reload":
                    cmd += ["--store", root / "store"]
                result, report = old.cli(binary, *cmd)
                replay, offline = old.cli(binary, "inspect", out)
                assert result.returncode == replay.returncode and report == offline
                assert report["outcome"] == "INCONCLUSIVE", report
                assert report["cache_result"] == "unavailable", report
                assert any("per-request persisted KV transfer" in reason for reason in report["reasons"]), report
                receipt = json.loads((out / "capture.json").read_text())
                identities.append(receipt["engine"])
                expected = "store" if stage == "store" else "first_reload"
                assert [record["id"] for record in receipt["requests"]] == [expected], receipt
                starts = [row["event"]["id"] for row in events(out / "events.bin")
                          if row["event"]["kind"] == "request_start"]
                assert starts == [expected], starts
                assert all(state["identity"] is None for state in report["states"]
                           if state["class"] in ["weights", "prefix_offload", "compiled_artifact_cache"]), report
        assert identities[0] != identities[1], identities


def bootstrap_replay(binary, root, base):
    _, original = old.cli(binary, "inspect", base, expect=0)
    imported = root / "bootstrap-import"
    _, report = old.cli(binary, "import", base, "--out", imported, expect=0)
    assert report["provenance"] == "imported"
    assert old.durations(report) == old.durations(original)
    _, replayed = old.cli(binary, "inspect", imported, expect=0)
    assert replayed == report

    for fault in ["end-clock", "frontend-clock", "incarnation", "before-runtime", "failed-with-end",
                  "legacy-contract", "communication-unavailable", "duration-oracle"]:
        source = root / f"bootstrap-replay-{fault}"
        shutil.copytree(base, source)

        def transform(rows):
            start = next(e for e in rows if e["event"]["kind"] == "bootstrap_start")
            end = next(e for e in rows if e["event"]["kind"] == "bootstrap_end")
            if fault in ("end-clock", "frontend-clock"):
                end["clock"]["id"] = "other-frontend-clock"
                if fault == "frontend-clock":
                    start["clock"]["id"] = "other-frontend-clock"
            elif fault == "incarnation":
                start["engine"]["start_ticks"] += 1
                end["engine"]["start_ticks"] += 1
            elif fault == "before-runtime":
                start["offset_us"] = rows[0]["offset_us"]
                rows.remove(start)
                rows.insert(0, start)
            elif fault == "failed-with-end":
                failed = json.loads(json.dumps(start))
                failed["event"] = {"kind": "failure", "detail": "synthetic construction exception"}
                rows.insert(rows.index(end), failed)
            elif fault == "legacy-contract":
                rows[0]["event"]["source_contract"] = "ordinary-asgi-fixture-v1"
            elif fault == "duration-oracle":
                delta = start["offset_us"] + 100000 - end["offset_us"]
                for row in rows[rows.index(end):]:
                    row["offset_us"] += delta

        old.rewrite_events(source, transform)
        if fault in ("legacy-contract", "communication-unavailable"):
            plan = json.loads((source / "plan.json").read_text())
            if fault == "legacy-contract":
                plan["adapter"] = "runtime_asgi_fixture_v1"
            else:
                plan["gates"].append({"target": {"kind": "communication"}, "max_regression_bps": 500,
                                      "max_reference_spread_bps": 500})
            old.save(source / "plan.json", plan)
            receipt = json.loads((source / "capture.json").read_text())
            receipt["plan_sha256"] = old.digest((source / "plan.json").read_bytes())
            old.save(source / "capture.json", receipt)
        destination = root / f"bootstrap-replay-import-{fault}"
        _, report = old.cli(binary, "import", source, "--out", destination)
        if fault == "duration-oracle":
            assert report["outcome"] == "PASS", report
            assert old.durations(report)["bootstrap"] == 100000, report
        else:
            assert report["outcome"] != "PASS", report
            if fault == "communication-unavailable":
                assert "communication" not in old.durations(report), report
                assert old.durations(report)["bootstrap"] == old.durations(original)["bootstrap"]
            else:
                assert "bootstrap" not in old.durations(report), report
        _, replayed = old.cli(binary, "inspect", destination)
        assert replayed == report


def source_sets(root):
    # Synthetic public-source stand-ins; no serving import or local pin promotion.
    names = tuple(bridge.SOURCE_PINS)
    bootstrap_names = tuple(bridge.SOURCE_PINS_487_BOOTSTRAP)
    shared = "vllm/entrypoints/openai/api_server.py"
    sources, pins = [], []
    for revision in ["old", "new", "bootstrap"]:
        directory = root / revision
        directory.mkdir()
        paths, expected = {}, {}
        for index, name in enumerate(bootstrap_names if revision == "bootstrap" else names):
            source_revision = "new" if revision == "bootstrap" else revision
            raw = ("shared-api\n" if name == shared else f"{source_revision}-{name}\n").encode()
            path = directory / f"source-{index}.py"
            path.write_bytes(raw)
            paths[name] = path
            expected[name] = old.digest(raw)
        sources.append(paths)
        pins.append(expected)

    adapters = ["runtime_vllm_v1", "runtime_vllm_487ecf187_v1", "runtime_vllm_487ecf187_bootstrap_v1"]
    contracts = ["vllm-0.27.0-uvicorn-0.34.0-sha256-v1",
                 "vllm-487ecf187-uvicorn-0.52.4-sha256-v1",
                 "vllm-487ecf187-uvicorn-0.52.4-sha256-bootstrap-v1"]

    def rejected(paths, adapter):
        try:
            bridge.verify_sources(paths, adapter)
        except ValueError:
            return
        raise AssertionError(("unreviewed source set admitted", adapter, paths))

    with (patch.object(bridge, "SOURCE_PINS", pins[0]),
          patch.object(bridge, "SOURCE_PINS_487", pins[1]),
          patch.object(bridge, "SOURCE_PINS_487_BOOTSTRAP", pins[2])):
        for index, adapter in enumerate(adapters):
            assert bridge.verify_sources(sources[index], adapter) == contracts[index]
            for other in range(len(adapters)):
                if other != index:
                    rejected(sources[other], adapter)
        changed = tuple(name for name in names if name != shared)
        for mask in range(1, (1 << len(changed)) - 1):
            mixed = dict(sources[0])
            for index, name in enumerate(changed):
                mixed[name] = sources[(mask >> index) & 1][name]
            for adapter in adapters:
                rejected(mixed, adapter)
            rejected({**sources[2], **mixed}, adapters[2])
        for index, adapter in enumerate(adapters):
            for missing in sources[index]:
                rejected({name: path for name, path in sources[index].items() if name != missing}, adapter)
            rejected({**sources[index], "unreviewed.py": sources[index][shared]}, adapter)
            for unsupported in ["runtime_asgi_fixture_v1", "runtime_asgi_fixture_bootstrap_v1", "runtime_vllm_latest"]:
                rejected(sources[index], unsupported)
            assert bridge.verify_sources(sources[index], adapter) == contracts[index]
            for path in sources[index].values():
                original = path.read_bytes()
                path.write_bytes(b"drift after initial admission\n")
                rejected(sources[index], adapter)
                path.write_bytes(original)
    for index, adapter in enumerate(adapters):
        rejected(sources[index], adapter)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="Fresh directory retaining every case, including failures")
    args = parser.parse_args()
    binary = args.binary.resolve()
    if args.out:
        args.out.mkdir(parents=True, exist_ok=False)
    with (nullcontext(args.out) if args.out else tempfile.TemporaryDirectory(prefix="grill-startup-runtime-")) as directory:
        root = Path(directory)
        cases = [("base", "none", None, (80, 100, 120)),
                 ("ready-delay", "none", None, (280, 100, 120)),
                 ("listen-delay", "none", None, (80, 300, 120)),
                 ("inference-delay", "none", None, (80, 100, 320)),
                 ("external-before", "none", "before", (80, 100, 120)),
                 ("external-after", "none", "after", (80, 100, 120))]
        cases += [(fault, fault, None, (80, 100, 120)) for fault in
                  ["invalid_ready", "namespace", "source_drift", "missing_seal", "hidden_engine", "hidden_inline"]]
        for name, fault, extra, delays in cases:
            case = root / name
            case.mkdir()
            run(binary, case, fault, extra, delays)
        for name, fault, delay, ready_delay in [
                ("bootstrap-base", "none", 60, 80),
                ("bootstrap-delay", "none", 260, 80),
                ("bootstrap-ready-delay", "none", 60, 280),
                *[(fault, fault, 60, 80) for fault in [
                    "missing_bootstrap", "missing_bootstrap_start", "missing_bootstrap_end",
                    "bootstrap_failed", "duplicate_bootstrap_start", "duplicate_bootstrap_end",
                    "reordered_bootstrap", "late_bootstrap"]]]:
            case = root / name
            case.mkdir()
            run(binary, case, fault, delays=(ready_delay, 100, 120), bootstrap_ms=delay)
        bootstrap_replay(binary, root, root / "bootstrap-base" / "capture")
        restart_unavailable(binary, root)
        source_sets(root)
    print("runtime ASGI: native/replay, independent bootstrap/ready/listen/inference delays, malformed bootstrap boundaries, exposure failures and closed source sets")


if __name__ == "__main__":
    main()
