"""Stage 3: continue seed LoRA with verified local worker completions."""

import argparse

from lora_training import train, verify_checkpoint
from swarm_common import (
    ADAPTERS, ARTIFACTS, MODEL_REVISION, PORTS, SOLVE_SYSTEM, TRACE_TARGET, WORKSPACE,
    adapter_fingerprint, digest, load_datasets, read_jsonl, solve_request,
)
from verifier import verify


def examples_from_workers():
    _, tasks = load_datasets()
    pool = {task["id"]: task for task in tasks if task["split"] == "explore"}
    traces = read_jsonl(ARTIFACTS / "verified_gen1.jsonl")
    if len(traces) != TRACE_TARGET:
        raise ValueError(f"Training requires exactly {TRACE_TARGET} verified traces.")
    seeds = adapter_fingerprint(ADAPTERS / "seed_v0")
    examples, distinct = [], set()
    for trace in traces:
        task = pool[trace["task_id"]]
        code = trace["code"]
        key = (task["id"], code)
        if (trace["origin"] != "local_worker" or trace["port"] not in PORTS
                or trace["adapter_sha256"] != seeds or key in distinct
                or not verify(code, task["cases"])[0]):
            raise ValueError(f"Invalid verified trace for {task['id']}")
        distinct.add(key)
        examples.append((SOLVE_SYSTEM, solve_request(task["problem"]), code))
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    parent = ADAPTERS / "seed_v0"
    if args.smoke:
        destination = WORKSPACE / "smoke_adapter"
        examples = [(SOLVE_SYSTEM, "Implement solve(x) returning x plus 1.",
                     "def solve(x):\n    return x + 1\n")]
        steps, source = 1, "smoke"
    else:
        destination = ADAPTERS / "gen1"
        examples = examples_from_workers()
        steps, source = 20, digest(ARTIFACTS / "verified_gen1.jsonl")
    identity = {
        "parent_sha256": adapter_fingerprint(parent), "training_source": source,
        "steps": steps, "base_revision": MODEL_REVISION,
    }
    if not destination.exists():
        train(examples, destination, steps=steps, learning_rate=8e-5, fraction=0.40,
              parent=parent, provenance=identity)
    info = verify_checkpoint(destination, identity)
    if args.smoke and info["peak_allocator_mib"] >= 14_305:
        raise RuntimeError("Smoke trainer used 15 GB or more of allocator-reserved VRAM.")
    print(f"Adapter ready: {destination}, peak allocator {info['peak_allocator_mib']:.0f} MiB.")


if __name__ == "__main__":
    main()
