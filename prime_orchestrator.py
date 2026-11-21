"""Run the paired benchmark, local expert iteration, and 160×44 Rich HUD."""

import ast
import asyncio
import json
import shlex
import signal
import subprocess
import sys
import time
import traceback
from collections import deque
from contextlib import suppress
from datetime import datetime, timezone
from hashlib import sha256

import httpx
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from swarm_common import (
    ADAPTERS, ARTIFACTS, LOGS, MODEL, MODEL_REVISION, PLAN_SYSTEM, PORTS, ROOT, SEED,
    SOLVE_SYSTEM, TRACE_TARGET, WORKSPACE, adapter_fingerprint, clean_line, code_source,
    digest, load_datasets, plan_request, read_json, solve_request, validate_plan,
    write_json, write_jsonl, write_text,
)
from swarm_runtime import generate, request, start_worker, swap, swarm_session, wait_health
from verifier import verify

PHASES = ("GEN-0 EVAL", "SWARM EXPLORE", "SELF-TRAIN", "HOT-SWAP", "GEN-1 EVAL")
RED, CYAN, GREEN, MUTED = "#ff5979", "#6ad9f5", "#53e6aa", "#9cabbe"


def gpu_status():
    query = ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]
    line = subprocess.check_output(query, text=True, stderr=subprocess.DEVNULL, timeout=5).splitlines()[0]
    name, utilization, used, total = (part.strip() for part in line.split(","))
    return {"name": name, "utilization": int(utilization), "used_mib": int(used), "total_mib": int(total)}


class Dashboard:
    def __init__(self):
        self.started = time.monotonic()
        self.events = deque(maxlen=8)
        self.phase = -1
        self.phase_events = []
        self.nodes = {port: {"status": "LOADING", "version": "seed_v0", "solved": 0} for port in PORTS}
        self.decision = {"thought": "Loading the worker mesh and validating the seed adapter."}
        self.scores = {0: [], 1: []}
        self.valid = self.rejected = self.trace_count = 0
        self.gpu, self.samples = None, []
        self.result = ""
        self.present = lambda: None

    def elapsed(self):
        return round(time.monotonic() - self.started, 3)

    def log(self, message):
        self.events.append((self.elapsed(), clean_line(message, 147)))

    def move(self, phase):
        self.phase = phase
        self.phase_events.append({"phase": PHASES[phase], "seconds": self.elapsed()})
        self.log(f"ENTER  {PHASES[phase]}")
        self.present()

    def score(self, generation):
        rows = self.scores[generation]
        return f"{sum(row['passed'] for row in rows):02}/{len(rows):02}" if rows else "--/--"

    def frame(self):
        header = Text()
        header.append(" PRIME-0 // DECENTRALIZED EXPERT-ITERATION SWARM", f"bold {RED}")
        header.append(f"   {self.elapsed() // 60:02.0f}:{self.elapsed() % 60:02.0f}", f"bold {MUTED}")
        header.append("\n ")
        for i, name in enumerate(PHASES):
            header.append(f"[{i+1}. {name}]", f"bold {CYAN}" if i == self.phase else GREEN if i < self.phase else MUTED)
            if i < 4:
                header.append("  >  ", MUTED)
        header.append("\n ")
        if self.gpu:
            gpu = self.gpu
            slots = int(22 * gpu["used_mib"] / gpu["total_mib"])
            header.append(f"{gpu['name']}   GPU {gpu['utilization']:3}%   VRAM [", MUTED)
            header.append("=" * slots + "." * (22 - slots), CYAN)
            header.append(f"]   {gpu['used_mib'] / 1024:4.1f}/{gpu['total_mib'] / 1024:.0f} GiB", MUTED)
        else:
            header.append("Waiting for GPU telemetry", MUTED)

        table = Table(expand=True, padding=(0, 1), show_edge=False, show_lines=False, border_style=MUTED)
        for name in ("Worker", "Status", "Adapter", "Solved"):
            table.add_column(name, justify="left", header_style=f"bold {CYAN}", no_wrap=True)
        for port, node in self.nodes.items():
            status = node["status"]
            table.add_row(f"EDGE-NODE-{port}", Text(f" {status} ", style=f"bold black on {GREEN if status == 'ONLINE' else CYAN}"),
                          str(node["version"]), str(node["solved"]))
        decision = Text(json.dumps(self.decision, indent=2, ensure_ascii=True), style=CYAN, overflow="fold")

        stats = Table(expand=True, show_header=False, box=None)
        stats.add_column()
        stats.add_column(justify="right")
        stats.add_row("Accepted traces", f"{self.trace_count:02} / {TRACE_TARGET}")
        stats.add_row("Sandbox passed", Text(str(self.valid), style=GREEN))
        stats.add_row("Sandbox rejected", Text(str(self.rejected), style=RED))
        bar = ProgressBar(total=TRACE_TARGET, completed=self.trace_count, complete_style=GREEN, style="#293447")
        benchmark = Table(expand=True, show_edge=False, border_style=MUTED)
        for label, color in (("Gen-0 pass@1", CYAN), ("Gen-1 pass@1", GREEN)):
            benchmark.add_column(label, justify="center", header_style=f"bold {color}")
        benchmark.add_row(self.score(0), self.score(1))
        result = Text(self.result or "10 paired held-out tasks  /  one greedy completion per task", style=MUTED)

        layout = Layout()
        layout.split_column(Layout(header, size=3), Layout(name="body"), Layout(name="log", size=12))
        layout["body"].split_row(Layout(name="left", ratio=55), Layout(name="right", ratio=45))
        layout["left"].split_column(
            Layout(Panel(table, title="Decentralized edge mesh", border_style=CYAN), size=10),
            Layout(Panel(decision, title="Prime decision / JSON", border_style=CYAN)),
        )
        layout["right"].split_column(
            Layout(Panel(Group(stats, bar), title="Deterministic verifier gate", border_style=RED), size=10),
            Layout(Panel(Group(benchmark, Text(""), result), title="Paired benchmark", border_style=GREEN)),
        )
        lines = [Text.assemble((f"[{elapsed:07.1f}s]  ", MUTED), message) for elapsed, message in self.events]
        layout["log"].update(Panel(Group(*lines), title="Prime event stream", border_style=MUTED))
        return layout


async def refresh(live, hud):
    while True:
        live.update(hud.frame(), refresh=True)
        await asyncio.sleep(1.0)


async def sample_gpu(hud):
    while True:
        try:
            hud.gpu = await asyncio.to_thread(gpu_status)
            hud.samples.append({"seconds": hud.elapsed(), "phase": hud.phase, **hud.gpu})
        except (subprocess.SubprocessError, OSError, ValueError, IndexError):
            pass
        await asyncio.sleep(2)


async def watch_nodes(hud, client):
    while True:
        async def update(port):
            try:
                data = await request(client, port, "health", timeout=1.0)
                hud.nodes[port].update({key: data[key] for key in ("status", "version")})
            except httpx.HTTPError:
                pass

        await asyncio.gather(*(update(port) for port in PORTS))
        await asyncio.sleep(1)


async def checked_response(hud, client, port, task, fingerprint, focus="", seed=SEED, temperature=0.0):
    response = await generate(client, port, SOLVE_SYSTEM, solve_request(task["problem"], focus), seed, temperature)
    if response["sha256"] != fingerprint:
        raise RuntimeError(f"Unexpected adapter returned by worker {port}.")
    code = code_source(response["text"])
    passed, reason = await asyncio.to_thread(verify, code, task["cases"])
    if passed:
        hud.valid += 1
        hud.nodes[port]["solved"] += 1
    else:
        hud.rejected += 1
    return {
        "task_id": task["id"], "port": port, "focus": focus, "seed": seed,
        "code": code, "adapter_sha256": fingerprint, "origin": "local_worker",
        "passed": passed, "verifier": reason,
    }


async def evaluate(hud, client, tasks, generation, fingerprint):
    for offset in range(0, len(tasks), len(PORTS)):
        batch = tasks[offset:offset+len(PORTS)]
        results = await asyncio.gather(*(
            checked_response(hud, client, port, task, fingerprint)
            for port, task in zip(PORTS, batch)
        ))
        for row in results:
            hud.log(f"{row['task_id']}  |  EDGE-NODE-{row['port']}  |  {row['verifier']}")
            code_hash = sha256(row.pop("code").encode()).hexdigest()
            hud.scores[generation].append({**row, "completion_sha256": code_hash})


async def local_plan(hud, client, task, attempt, fingerprint):
    port = PORTS[attempt % len(PORTS)]
    result = await generate(
        client, port, PLAN_SYSTEM, plan_request(task["problem"]), SEED + attempt, 0.35, 220,
    )
    if result["sha256"] != fingerprint:
        raise RuntimeError("Planner returned an unexpected adapter.")
    try:
        text = result["text"]
        candidate = json.loads(text[text.index("{"):text.rindex("}") + 1])
        plan = validate_plan(candidate)
        hud.log(f"{task['id']}  |  local plan accepted on EDGE-NODE-{port}")
        source = "local_worker"
    except (ValueError, KeyError, TypeError):
        plan = {
            "thought": "Use independent strategies after plan validation failed.",
            "workers": list(PORTS),
            "focus": ["Implement the simplest direct approach.", "Work through edge conditions.",
                      "Use a distinct algorithm and check ordering."],
        }
        source = "prime_default"
        hud.log(f"{task['id']}  |  bounded default route")
    hud.decision = plan
    return plan, source


async def explore(hud, client, tasks, fingerprint):
    traces, distinct, per_task = [], set(), {task["id"]: 0 for task in tasks}
    for round_number, temperature in enumerate((0.42, 0.55, 0.67, 0.78)):
        for index, task in enumerate(tasks):
            if per_task[task["id"]] >= 3:
                continue
            plan, source = await local_plan(hud, client, task, index + 20 * round_number, fingerprint)
            results = await asyncio.gather(*(
                checked_response(hud, client, port, task, fingerprint, focus,
                                 SEED + 1000 * round_number + 10 * index + i, temperature)
                for i, (port, focus) in enumerate(zip(plan["workers"], plan["focus"]))
            ))
            ordered = sorted(
                (row for row in results if row["passed"]), key=lambda row: (len(row["code"]), row["port"]),
            )
            winner = None
            for winner_candidate in ordered:
                normalized = ast.dump(ast.parse(winner_candidate["code"]), include_attributes=False)
                key = (task["id"], normalized)
                if key not in distinct:
                    distinct.add(key)
                    winner = winner_candidate
                    break
            if winner:
                per_task[task["id"]] += 1
                traces.append({**winner, "round": round_number + 1, "plan": plan, "plan_source": source})
                hud.trace_count = len(traces)
                hud.log(f"{task['id']}  |  {len(ordered)}/3 passed  |  EDGE-NODE-{winner['port']} accepted")
            else:
                hud.log(f"{task['id']}  |  no new passing completion")
            if len(traces) == TRACE_TARGET:
                return traces
    write_jsonl(WORKSPACE / "partial_verified.jsonl", traces)
    raise RuntimeError(f"Collected {len(traces)}/{TRACE_TARGET} unique passing traces within 4 rounds.")


async def train_live(hud, group, client, seed_fingerprint):
    script = WORKSPACE / "train_next_gen.sh"
    write_text(
        script, "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {shlex.quote(str(ROOT))}\nexec {shlex.quote(sys.executable)} run_rsi_train.py\n",
    )
    script.chmod(0o755)
    proc = group.start("gen1-trainer", ["bash", str(script)])
    started = time.monotonic()
    probes = 0
    while proc.poll() is None:
        if time.monotonic() - started > 3600:
            raise TimeoutError("Concurrent LoRA trainer exceeded one hour.")
        responses = await asyncio.gather(*(
            generate(client, port, SOLVE_SYSTEM, "Implement solve(x) returning x + 1.", max_tokens=32)
            for port in PORTS
        ))
        if any(row["sha256"] != seed_fingerprint for row in responses):
            raise RuntimeError("A worker changed adapters during training.")
        if proc.poll() is None:
            probes += len(responses)
            if probes == len(PORTS):
                hud.log("All three edge nodes served inference while the trainer was active")
        await asyncio.sleep(2)
    if proc.returncode:
        raise RuntimeError("Concurrent trainer failed. See workspace/logs/gen1-trainer.log.")
    if not probes:
        raise RuntimeError("No inference completed while the trainer was active.")
    checkpoint = read_json(ADAPTERS / "gen1" / "training.json")
    if checkpoint["pid"] != proc.pid:
        raise RuntimeError("Concurrent trainer did not produce this run's checkpoint.")
    hud.log(f"Gen-1 checkpoint ready  |  {probes} concurrent worker responses")
    return {"pid": proc.pid, "probe_completions": probes, **checkpoint}


async def hot_swap(hud, client, identities, before, after):
    async def broadcast(version, value):
        return await asyncio.gather(
            *(swap(client, port, version, value) for port in PORTS), return_exceptions=True,
        )

    for node in hud.nodes.values():
        node["status"] = "SYNCING"
    hud.present()
    replies = await broadcast("gen1", after)
    same_process = all(
        isinstance(row, dict) and row["pid"] == identities[port]["pid"]
        for port, row in zip(PORTS, replies)
    )
    if not same_process:
        recovery = await broadcast("seed_v0", before)
        if any(isinstance(item, Exception) for item in recovery):
            hud.log("A rollback failed. Shutting down the worker mesh.")
        raise RuntimeError("Hot-swap did not complete in place on all three workers.")
    responses = await asyncio.gather(*(
        generate(client, port, SOLVE_SYSTEM, "Implement solve(x) returning x + 1.", max_tokens=32)
        for port in PORTS
    ))
    for port, reply in zip(PORTS, responses):
        if reply["pid"] != identities[port]["pid"] or reply["sha256"] != after:
            raise RuntimeError("A worker failed the post-swap inference probe.")
        hud.nodes[port].update(version="gen1", status="ONLINE")
        hud.log(f"EDGE-NODE-{port}  |  Gen-1 live  |  same process {reply['pid']}")
    hud.present()
    return replies


async def run(hud):
    _, tasks = load_datasets()
    holdout = [task for task in tasks if task["split"] == "held_out"]
    training = [task for task in tasks if task["split"] == "explore"]
    seed_hash = adapter_fingerprint(ADAPTERS / "seed_v0")
    smoke = read_json(WORKSPACE / "smoke.json")
    if smoke["seed_sha256"] != seed_hash:
        raise RuntimeError("Run the worker/trainer smoke checkpoint for this seed adapter.")
    started = datetime.now(timezone.utc).isoformat()
    hud.log("Starting three independent FastAPI worker processes")
    async with swarm_session() as (group, client):
        jobs = {port: start_worker(group, port) for port in PORTS}
        ready = await asyncio.gather(*(wait_health(client, port, proc, seed_hash) for port, proc in jobs.items()))
        identity = dict(zip(PORTS, ready))
        observer = asyncio.create_task(watch_nodes(hud, client))
        try:
            hud.log("Mesh online  |  base weights and seed adapter verified")
            hud.move(0)
            await evaluate(hud, client, holdout, 0, seed_hash)
            hud.move(1)
            traces = await explore(hud, client, training, seed_hash)
            write_jsonl(ARTIFACTS / "verified_gen1.jsonl", traces)
            hud.move(2)
            trainer = await train_live(hud, group, client, seed_hash)
            gen1_hash = adapter_fingerprint(ADAPTERS / "gen1")
            if gen1_hash == seed_hash:
                raise RuntimeError("Training produced identical adapter weights.")
            hud.move(3)
            swaps = await hot_swap(hud, client, identity, seed_hash, gen1_hash)
            hud.move(4)
            await evaluate(hud, client, holdout, 1, gen1_hash)
        finally:
            observer.cancel()
            with suppress(asyncio.CancelledError):
                await observer
    if not any(sample["phase"] == 2 for sample in hud.samples):
        raise RuntimeError("No GPU telemetry was captured during concurrent training.")
    if any(proc.poll() is None for proc in jobs.values()):
        raise RuntimeError("Worker processes still running after shutdown.")
    for node in hud.nodes.values():
        node["status"] = "STOPPED"
    counts = [sum(row["passed"] for row in hud.scores[i]) for i in (0, 1)]
    delta = (counts[1] - counts[0]) * 100 / len(holdout)
    hud.result = f"Held-out change {delta:+.1f} pp  /  {'improvement observed' if delta > 0 else 'improvement gate unmet'}"
    hud.log(hud.result)
    hud.log("All owned GPU processes exited")
    return {
        "schema": 1, "start_utc": started, "end_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": hud.elapsed(), "phases": hud.phase_events,
        "base_model": MODEL, "base_revision": MODEL_REVISION,
        "dataset_sha256": digest(WORKSPACE / "dataset_manifest.json"),
        "held_out_count": len(holdout), "gen0_pass": counts[0], "gen1_pass": counts[1],
        "gen0_pass_at_1": counts[0] / len(holdout), "gen1_pass_at_1": counts[1] / len(holdout),
        "delta_percentage_points": delta, "improvement_gate": delta > 0,
        "benchmark": {"gen0": hud.scores[0], "gen1": hud.scores[1]},
        "verified_traces": len(traces), "sandbox_passed": hud.valid, "sandbox_rejected": hud.rejected,
        "trace_sha256": digest(ARTIFACTS / "verified_gen1.jsonl"),
        "adapters": {"seed_v0": seed_hash, "gen1": gen1_hash},
        "worker_boot": ready, "worker_swap": swaps, "trainer": trainer,
        "gpu_samples": hud.samples, "children_reaped": True,
    }


async def main():
    hud = Dashboard()
    LOGS.mkdir(parents=True, exist_ok=True)
    console = Console(width=160, height=44, force_terminal=True, color_system="truecolor",
                      style="on #0d1117", no_color=False)
    job = asyncio.create_task(run(hud))
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        loop.add_signal_handler(sig, job.cancel)
    with Live(hud.frame(), console=console, auto_refresh=False, vertical_overflow="crop") as live:
        hud.present = lambda: live.update(hud.frame(), refresh=True)
        tasks = [asyncio.create_task(refresh(live, hud)), asyncio.create_task(sample_gpu(hud))]
        try:
            metrics = await job
        except (Exception, asyncio.CancelledError) as error:
            with (LOGS / "prime-error.log").open("w", encoding="utf-8") as log:
                traceback.print_exc(file=log)
            hud.result = "Run interrupted" if isinstance(error, asyncio.CancelledError) else "Checkpoint failed"
            hud.log(hud.result + "  |  see workspace/logs/prime-error.log")
            exit_code = 1
        else:
            write_json(ARTIFACTS / "metrics.json", metrics)
            exit_code = 0
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            live.update(hud.frame(), refresh=True)
            await asyncio.sleep(1.2)
    return exit_code


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
