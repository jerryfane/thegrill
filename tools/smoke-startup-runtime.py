#!/usr/bin/env python3
"""Later CPU verification of the real bridge with an ordinary ASGI process.

No vLLM/Uvicorn/device imports or live endpoints. Legacy startup/restart cases
remain in smoke-startup.py; run both scripts at the integrated head.
"""
import argparse
from contextlib import contextmanager
import http.client
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


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


def run(binary, root, fault="none", extra=None, delays=(80, 100, 120)):
    port = old.port()
    plan = old.plan(binary, f"http://127.0.0.1:{port}/v1/chat/completions")
    plan["adapter"] = "runtime_asgi_fixture_v1"
    plan["adapter_sha256"] = old.digest((HERE / "startup-runtime.py").read_bytes())
    plan["runtime_window_us"] = 2000000
    plan["gates"] = [gate for gate in plan["gates"] if gate["target"]["kind"] != "communication"]
    plan_path, journal, out = root / "plan.json", root / "journal.ndjson", root / "capture"
    old.save(plan_path, plan)
    with child("--plan", plan_path, "--events", journal, "--port", port, "--fault", fault,
               "--ready-ms", delays[0], "--listen-ms", delays[1], "--inference-ms", delays[2]) as server:
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
            coverage = next(e for e in rows if e["event"]["kind"] == "runtime_coverage")
            assert coverage["offset_us"] - ready["offset_us"] >= plan["runtime_window_us"]
            # An early count-based seal would finish before this required exposure.
            assert rows[-1]["event"] == {"kind": "sealed", "requests": 1}, rows[-1]
            assert report["cache_result"] == "unavailable", report
        else:
            assert report["outcome"] != "PASS", report
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve()
    with tempfile.TemporaryDirectory(prefix="grill-startup-runtime-") as directory:
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
        restart_unavailable(binary, root)
        # Execute the actual source validator against wrong installed-file bytes,
        # without importing vLLM or mocking its hash function.
        wrong = root / "wrong-source.py"
        wrong.write_bytes(b"# synthetic source drift\n")
        try:
            bridge.verify_sources({name: wrong for name in bridge.SOURCE_PINS})
        except ValueError as error:
            assert "runtime source drift" in str(error)
        else:
            raise AssertionError("source drift accepted")
    print("runtime ASGI: native/replay, three delays, external before/after, internal admissions, bad readiness, namespace mismatch, source drift and missing seal")


if __name__ == "__main__":
    main()
