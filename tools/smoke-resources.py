#!/usr/bin/env python3
"""CPU-only resource smoke. Run only after Main's stable-head validation authorization.
Launches its own finite 8 MiB ordinary fixture, never a serving process or device tool.
Keeps all plans, commands, stdout/stderr and evidence under an explicitly new --out.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def fixture():
    memory = bytearray(8 * 1024 * 1024)
    for i in range(0, len(memory), 4096):
        memory[i] = 1
    print("ready", flush=True)
    end = time.monotonic() + 90
    value = 1
    while time.monotonic() < end:
        for i in range(20000):
            value = (value + i) & 0xFFFF
        memory[value % len(memory)] = value & 0xFF
        time.sleep(0.01)


def run(binary, out, sources=1):
    binary = binary.resolve(strict=True)
    os.mkdir(out, 0o700)
    with binary.open("rb") as stream:
        collector = hashlib.file_digest(stream, "sha256").hexdigest()
    children = []
    try:
        import selectors
        for _ in range(sources):
            child = subprocess.Popen([sys.executable, __file__, "--fixture"], stdout=subprocess.PIPE, text=True)
            children.append(child)
            selector = selectors.DefaultSelector()
            try:
                selector.register(child.stdout, selectors.EVENT_READ)
                if not selector.select(5) or child.stdout.readline().strip() != "ready":
                    raise RuntimeError("ordinary fixture did not become ready")
            finally:
                selector.close()
        config = {
            "version": 1,
            "sources": [{"id": "fixture-process" if i == 0 else f"fixture-process-{i}", "target": {"kind": "process", "pid": children[i].pid},
                         "ownership": {"kind": "process_address_space"}} for i in range(sources)],
            "cadence_us": 10000, "max_gap_us": 100000, "max_read_us": 50000,
            "max_samples": 32, "deadline_us": 1000000,
            "per_sample_bytes": 16384, "raw_total_bytes": 524288,
        }
        plan = {
            "version": 1, "id": "ordinary-process-resource-smoke",
            "exposure_pin": hashlib.sha256(b"8MiB fixed ordinary process; 100ms measured observation v1").hexdigest(),
            "collector_sha256": collector, "candidate_change": "unchanged-control",
            "duration_us": 100000,
            "arms": [{"role": role, "deployment_pin": hashlib.sha256(b"same-ordinary-fixture").hexdigest(),
                      "config": config, "warmup_ids": [role + "-warmup"],
                      "measured_ids": [role + "-" + str(i) for i in range(3)]} for role in ("a", "b", "a2")],
            "gates": [{"source": source["id"], "metric": "resident_memory", "statistic": "sampled_maximum",
                       "max_regression_bps": 500, "max_reference_spread_bps": 500} for source in config["sources"]],
        }
        plan_path = out / "study.json"
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
        groups = {role: [] for role in ("a", "b", "a2")}
        commands = []

        def invoke(name, arguments, allowed=(0,)):
            command = [str(binary), "resource", *map(str, arguments)]
            commands.append(command)
            (out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
            result = subprocess.run(command, capture_output=True, text=True, timeout=15)
            (out / (name + ".stdout.json")).write_text(result.stdout)
            (out / (name + ".stderr.txt")).write_text(result.stderr)
            if result.returncode not in allowed:
                raise RuntimeError(f"{name}: exit {result.returncode}; retained stderr/evidence in {out}")
            return json.loads(result.stdout)

        for arm in plan["arms"]:
            for phase, identities in (("warmup", arm["warmup_ids"]), ("measured", arm["measured_ids"])):
                for index, identity in enumerate(identities):
                    destination = out / identity
                    groups[arm["role"]].append(destination)
                    captured = invoke(identity, ["capture", "--plan", plan_path, "--role", arm["role"],
                                               "--phase", phase, "--index", index, "--out", destination], (0, 2))
                    replay = invoke(identity + "-inspect", ["inspect", destination, "--plan", plan_path])
                    assert captured == replay, "offline replay changed the capture summary"
                    assert replay["provenance"] == "native_observed"
                    assert replay["summaries"][0]["unavailable"] is None, "source exposure was incomplete; never retry"
                    assert replay["summaries"][0]["value"]["numerator"] >= 8 * 1024 * 1024
                    assert (destination / "observation.json").stat().st_mode & 0o777 == 0o600
                    retained = json.loads((destination / "observation.json").read_text())
                    observation = retained["observation"]
                    for source in config["sources"]:
                        ticks = [s["readings"][0]["value"]["value"] for s in observation["snapshots"] if s["source"] == source["id"]]
                        assert all(a <= b for a, b in zip(ticks, ticks[1:])), "native process counter reset"
        decision = invoke("comparison", ["compare", "--plan", plan_path, "--a", *groups["a"],
                                         "--b", *groups["b"], "--a2", *groups["a2"]], (0, 2, 3))
        assert decision["decision"] == "PASS", "unchanged fixture did not qualify; retain, do not replace"
        assert decision["gates"][0]["counts"] == [3, 3, 3]
        imported = out / "imported"
        invoke("import", ["import", groups["a"][0] / "observation.json", "--out", imported])
        imported_view = invoke("import-inspect", ["inspect", imported, "--plan", plan_path])
        assert imported_view["provenance"] == "imported", "import promoted native provenance"
        print(json.dumps({"result": "CPU_PROTOCOL_SMOKE_PASSED", "collector_sha256": collector,
                          "evidence": str(out), "scope": "ordinary Linux process only; no GPU, capacity, retention, or serving qualification"}))
    finally:
        # Only these explicitly created finite fixtures; never service PIDs.
        for child in children:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", nargs="?", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--sources", type=int, choices=(1, 2), default=1,
                        help="Number of owned ordinary processes; two exercises shared interval coverage")
    parser.add_argument("--fixture", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.fixture:
        fixture()
    elif args.binary and args.out:
        os.umask(0o077)
        run(args.binary, args.out.resolve(), args.sources)
    else:
        parser.error("binary and a new --out directory are required")
