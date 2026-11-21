"""Record Prime in a headless PTY and verify the complete capture."""

import fcntl
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone

from swarm_common import (
    ADAPTERS, ARTIFACTS, LOGS, ROOT, TRACE_TARGET, WORKSPACE, adapter_fingerprint,
    digest, read_json, read_jsonl, stage_environment, write_json,
)


def assert_idle_gpu():
    result = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
         "--format=csv,noheader,nounits"], text=True,
    )
    users = [line for line in result.splitlines() if "python" in line.lower()]
    if users:
        raise RuntimeError(f"Python GPU processes are still active: {'; '.join(users)}")


def validate_run(cast):
    metrics = read_json(ARTIFACTS / "metrics.json")
    rows = read_jsonl(ARTIFACTS / "verified_gen1.jsonl")
    if len(rows) != TRACE_TARGET or metrics["verified_traces"] != TRACE_TARGET:
        raise ValueError("Verified trace count mismatch.")
    if metrics["trace_sha256"] != digest(ARTIFACTS / "verified_gen1.jsonl"):
        raise ValueError("Trace checksum mismatch.")
    if metrics["adapters"]["gen1"] != adapter_fingerprint(ADAPTERS / "gen1"):
        raise ValueError("Gen-1 checkpoint checksum mismatch.")
    before, after = metrics["benchmark"]["gen0"], metrics["benchmark"]["gen1"]
    if len(before) != 10 or len(after) != 10 or [
        (row["task_id"], row["port"], row["seed"]) for row in before
    ] != [(row["task_id"], row["port"], row["seed"]) for row in after]:
        raise ValueError("Benchmark pairing mismatch.")
    boot, swap = metrics["worker_boot"], metrics["worker_swap"]
    if len(boot) != 3 or len({row["pid"] for row in boot}) != 3 or len(swap) != 3:
        raise ValueError("There must be three independent worker processes.")
    if any(left["pid"] != right["pid"] for left, right in zip(boot, swap)):
        raise ValueError("Worker process identity changed during replacement.")
    if not metrics["children_reaped"] or not metrics["trainer"]["probe_completions"]:
        raise ValueError("Missing live inference or process cleanup evidence.")
    with cast.open(encoding="utf-8") as stream:
        header = json.loads(next(stream))
        if (header.get("version"), header.get("width"), header.get("height")) != (2, 160, 44):
            raise ValueError("Capture was not recorded at 160×44.")
        labels = {"DECENTRALIZED EXPERT-ITERATION SWARM", "Starting three independent",
                  "All owned GPU processes exited", "Held-out change"}
        labels.update(f"ENTER {phase['phase']}" for phase in metrics["phases"])
        markers, tail = set(), ""
        overlap = max(map(len, labels)) - 1
        for line in stream:
            _, kind, content = json.loads(line)
            if kind == "o":
                chunk = tail + content
                markers.update(marker for marker in labels - markers if marker in chunk)
                tail = chunk[-overlap:]
        if markers != labels:
            raise ValueError(f"Capture is missing dashboard events: {sorted(labels - markers)}.")
    return metrics


def archive_partial():
    sources = [ARTIFACTS / name for name in (
        "demo.cast", "metrics.json", "verified_gen1.jsonl", "benchmark_scorecard.png", "demo_video.mp4",
    )] + [ADAPTERS / "gen1", WORKSPACE / "partial_verified.jsonl"]
    existing = [path for path in sources if path.exists()]
    if existing:
        previous = WORKSPACE / "attempts" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        previous.mkdir(parents=True)
        for path in existing:
            path.rename(previous / path.name)
        print(f"Previous incomplete outputs kept in {previous}.")


def main():
    import signal

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    LOGS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    cast = ARTIFACTS / "demo.cast"
    checkpoint = WORKSPACE / "recording.json"
    with (WORKSPACE / "recording.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_idle_gpu()
        if checkpoint.exists():
            existing = read_json(checkpoint)
            if existing != {"cast_sha256": digest(cast), "metrics_sha256": digest(ARTIFACTS / "metrics.json")}:
                raise ValueError("Recorded run checksum mismatch. Use a fresh run directory.")
        else:
            archive_partial()
            command = f"stty cols 160 rows 44; exec {shlex.quote(sys.executable)} prime_orchestrator.py"
            config_dir = WORKSPACE / "asciinema-config"
            config_dir.mkdir(exist_ok=True)
            env = {**stage_environment(), "ASCIINEMA_CONFIG_HOME": str(config_dir),
                   "TERM": "xterm-256color", "COLORTERM": "truecolor"}
            with (LOGS / "recorder.log").open("wb") as output:
                process = subprocess.Popen(
                    [str(ROOT / ".tools/asciinema"), "rec", "--headless", "--window-size", "160x44",
                     "--return", "--output-format", "asciicast-v2", "--capture-env", "TERM,COLORTERM",
                     "--title", "Prime Expert Iteration", "-c", command, str(cast)],
                    cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                )
                try:
                    status = process.wait()
                except KeyboardInterrupt:
                    process.terminate()
                    process.wait(timeout=30)
                    raise
                if status:
                    raise RuntimeError(f"Recorded run exited {status}; inspect workspace/logs.")
            validate_run(cast)
            assert_idle_gpu()
            write_json(checkpoint, {"cast_sha256": digest(cast), "metrics_sha256": digest(ARTIFACTS / "metrics.json")})
        metrics = validate_run(cast)
        assert_idle_gpu()
    print(f"Recorded benchmark: {metrics['gen0_pass']}/10 → {metrics['gen1_pass']}/10.")
    if not metrics["improvement_gate"]:
        print("Improvement gate unmet. The measured results will still be charted and packaged.")


if __name__ == "__main__":
    main()
