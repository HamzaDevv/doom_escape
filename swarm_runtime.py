"""GPU process ownership, worker HTTP protocol, and smoke checkpoint."""

import asyncio
import os
import secrets
import signal
import subprocess
import sys
from contextlib import asynccontextmanager

import httpx

from swarm_common import (
    ADAPTERS, LOGS, PORTS, ROOT, SEED, SOLVE_SYSTEM, WORKSPACE, adapter_fingerprint,
    stage_environment, write_json,
)


class ProcessGroup:
    def __init__(self):
        self.items = []
        self.env = stage_environment()
        self.token = secrets.token_hex(32)
        self.env["SWARM_TOKEN"] = self.token

    def start(self, log, command):
        LOGS.mkdir(parents=True, exist_ok=True)
        with (LOGS / f"{log}.log").open("wb") as output:
            proc = subprocess.Popen(
                command, cwd=ROOT, env=self.env, stdout=output, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
        self.items.append(proc)
        return proc

    async def close(self):
        for proc in reversed(self.items):
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for proc in reversed(self.items):
            try:
                await asyncio.to_thread(proc.wait, 12)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await asyncio.to_thread(proc.wait)


@asynccontextmanager
async def swarm_session():
    group = ProcessGroup()
    try:
        async with httpx.AsyncClient(
            headers={"X-Swarm-Token": group.token}, timeout=150.0, trust_env=False,
        ) as client:
            yield group, client
    finally:
        await group.close()


async def request(client, port, route, payload=None, timeout=150.0):
    url = f"http://127.0.0.1:{port}/{route}"
    if payload is None:
        response = await client.get(url, timeout=timeout)
    else:
        response = await client.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


async def wait_health(client, port, process, fingerprint):
    for _ in range(300):
        if process.poll() is not None:
            raise RuntimeError(f"Worker {port} exited; see workspace/logs.")
        try:
            info = await request(client, port, "health", timeout=2)
        except httpx.TransportError:
            await asyncio.sleep(1)
            continue
        if info["pid"] != process.pid or info["sha256"] != fingerprint:
            raise RuntimeError(f"Worker {port} identity or adapter mismatch.")
        return info
    raise TimeoutError(f"Worker {port} startup timed out.")


def start_worker(group, port, smoke=False):
    command = [sys.executable, "worker_node.py", "--port", str(port)]
    if smoke:
        command.append("--smoke")
    return group.start(f"{'smoke-' if smoke else ''}worker-{port}", command)


async def generate(client, port, system, user, seed=SEED, temperature=0.0, max_tokens=420):
    return await request(client, port, "infer", {
        "system": system, "user": user, "seed": seed,
        "temperature": temperature, "max_tokens": max_tokens,
    })


async def swap(client, port, version, sha256):
    info = await request(client, port, "hot_swap", {"version": version, "sha256": sha256})
    if info["version"] != version or info["sha256"] != sha256:
        raise RuntimeError(f"Worker {port} failed to acknowledge {version}.")
    return info


async def smoke():
    seed = adapter_fingerprint(ADAPTERS / "seed_v0")
    port = PORTS[0]
    async with swarm_session() as (group, client):
        trainer = group.start("smoke-trainer", [sys.executable, "run_rsi_train.py", "--smoke"])
        if await asyncio.to_thread(trainer.wait, 1800):
            raise RuntimeError("Smoke trainer failed; see workspace/logs/smoke-trainer.log.")
        successor = adapter_fingerprint(WORKSPACE / "smoke_adapter")
        if seed == successor:
            raise AssertionError("Smoke adapter did not change the weights.")
        proc = start_worker(group, port, smoke=True)
        await wait_health(client, port, proc, seed)
        prompt = "Implement solve(x) returning x plus 1."
        before = await generate(client, port, SOLVE_SYSTEM, prompt, max_tokens=96)
        updated = await swap(client, port, "gen1", successor)
        after = await generate(client, port, SOLVE_SYSTEM, prompt, max_tokens=96)
        if (not before["text"].strip() or not after["text"].strip()
                or before["sha256"] != seed or after["sha256"] != successor
                or updated["pid"] != proc.pid or after["pid"] != proc.pid):
            raise AssertionError("Smoke inference or in-place weight replacement failed.")
    if proc.poll() is None:
        raise AssertionError("Smoke worker was not cleaned up.")
    write_json(WORKSPACE / "smoke.json", {"seed_sha256": seed, "successor_sha256": successor,
                                        "worker_pid": proc.pid, "port": port})
    print("Smoke trainer, /health, /infer, /hot_swap, and worker shutdown passed.")


if __name__ == "__main__":
    asyncio.run(smoke())
