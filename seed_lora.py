"""Stage 1: train the initial coordinator adapter."""

from lora_training import train, verify_checkpoint
from swarm_common import (
    ADAPTERS, MODEL_REVISION, PLAN_SYSTEM, WORKSPACE, digest, json_text, load_datasets,
    plan_request,
)


def main():
    seeds, _ = load_datasets()
    destination = ADAPTERS / "seed_v0"
    identity = {
        "dataset_sha256": digest(WORKSPACE / "dataset_manifest.json"),
        "base_revision": MODEL_REVISION, "steps": 40, "kind": "coordinator_seed",
    }
    if not destination.exists():
        examples = [(PLAN_SYSTEM, plan_request(row["problem"]), json_text(row["plan"])) for row in seeds]
        train(examples, destination, steps=40, learning_rate=1e-4, fraction=0.90,
              parent=None, provenance=identity)
    info = verify_checkpoint(destination, identity)
    print(f"Seed LoRA ready: {info['rows']} rows over {info['steps']} steps.")


if __name__ == "__main__":
    main()
