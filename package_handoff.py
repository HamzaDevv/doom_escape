"""Generate the proof-of-work README, commit the run, and archive the tracked tree."""

import os
import subprocess
from zipfile import ZipFile

from record_run import assert_idle_gpu, validate_run
from swarm_common import ARTIFACTS, ROOT, WORKSPACE, digest, read_json, write_text

SOURCES = (
    ".gitignore", "README.md", "ARCHITECTURE.md", "INIT_SPEC.md", "requirements.txt",
    "setup_env.sh", "swarm_expert_iteration_colab.ipynb", "generate_seed_data.py", "seed_lora.py", "worker_node.py",
    "run_rsi_train.py", "prime_orchestrator.py", "swarm_common.py", "swarm_runtime.py",
    "lora_training.py", "sandbox_runner.py", "verifier.py", "record_run.py", "plot_results.py",
    "retime_cast.py", "render_demo_video.sh", "package_handoff.py",
)
OUTPUTS = (
    "artifacts/demo.cast", "artifacts/metrics.json", "artifacts/verified_gen1.jsonl",
    "artifacts/benchmark_scorecard.png", "artifacts/demo_video.mp4",
    "adapters/seed_v0", "adapters/gen1", "workspace/seed_dataset.jsonl",
    "workspace/synthetic_tasks.jsonl", "workspace/dataset_manifest.json", "workspace/dependencies.txt",
)


def readme(metrics):
    count, before, after = (metrics[k] for k in ("held_out_count", "gen0_pass", "gen1_pass"))
    delta = metrics["delta_percentage_points"]
    result = "Improvement observed" if metrics["improvement_gate"] else "Improvement gate unmet"
    return f"""# Bounded swarm expert iteration

Three local Qwen2.5-3B workers run verifier-gated expert iteration on one A100.
A coordinator LoRA plans exploration across the mesh. Only worker completions that pass
the subprocess tests train the successor adapter, which is hot-swapped into the live mesh.

| Benchmark | Passed | Pass@1 |
| --- | ---: | ---: |
| Gen-0 seed | {before}/{count} | {before/count:.1%} |
| Gen-1 post-swap | {after}/{count} | {after/count:.1%} |

**{delta:+.1f} percentage points. {result}.** Across both evaluations, the same
{count} held-out tasks each receive one greedy completion. Exploration contributed
{metrics['verified_traces']} verified worker traces. The workers served
{metrics['trainer']['probe_completions']} probe responses while the trainer was live.

![Measured benchmark scorecard](artifacts/benchmark_scorecard.png)

- [1080p terminal demo and scorecard](artifacts/demo_video.mp4)
- [Run metrics and paired task outcomes](artifacts/metrics.json)
- [Technical architecture](ARCHITECTURE.md)

```mermaid
flowchart LR
    DS[DeepSeek upstream synthesis] --> Seed[80 coordinator traces]
    DS --> Tasks[30 Python tasks]
    Seed --> L0[Seed LoRA]
    L0 --> Edge[3 local FastAPI workers]
    Tasks --> Prime[Prime coordinator]
    Prime --> Edge
    Edge --> Gate[Restricted Python verifier]
    Gate --> Traces[24 passing exploration traces]
    Traces --> Train[Concurrent LoRA training]
    Train -->|In-place adapter update| Edge
    Gate --> Scores[Paired Gen-0 and Gen-1 pass@1]
```

## Reproduce on an A100

With the Colab notebook, connect to an A100 runtime, use the hidden Deploy secret key prompt,
and run all cells in order. The notebook clones this repository and checks every stage.
For CLI reproduction, use the tracked source in a fresh directory and run these commands.
The generator also accepts the tracked, checksummed data when it is present.

```bash
bash setup_env.sh
read -rsp 'DeepSeek API key: ' DEEPSEEK_API_KEY
printf '\\n'
DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" .venv/bin/python generate_seed_data.py
unset DEEPSEEK_API_KEY
.venv/bin/python seed_lora.py
.venv/bin/python swarm_runtime.py
.venv/bin/python record_run.py
.venv/bin/python plot_results.py
bash render_demo_video.sh
.venv/bin/python package_handoff.py
```

The notebook uses Deploy’s `deepseek-ai/deepseek-v4.1-flash` model. The CLI
defaults to the direct DeepSeek API. To override the CLI model and endpoint, set
`DEEPSEEK_MODEL` and `DEEPSEEK_BASE_URL` in the environment.
The recorded adapter checkpoint incorporates stochastic local exploration.
For a fresh run, use a new run directory. The archive includes the tracked source,
datasets, adapter checkpoints and outputs. `workspace/dependencies.txt` records
the exact installed Python packages.
"""


def git(*args, check=True, capture=False):
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=check, text=True, capture_output=capture,
    )
    return result.stdout.strip() if capture else result.returncode


def main():
    assert_idle_gpu()
    cast = ARTIFACTS / "demo.cast"
    checkpoint = read_json(WORKSPACE / "recording.json")
    if checkpoint != {"cast_sha256": digest(cast), "metrics_sha256": digest(ARTIFACTS / "metrics.json")}:
        raise RuntimeError("The record checkpoint does not match the capture or metrics.")
    metrics = validate_run(cast)
    for name in ("demo_video.mp4", "benchmark_scorecard.png"):
        if not (ARTIFACTS / name).is_file() or not (ARTIFACTS / name).stat().st_size:
            raise FileNotFoundError(ARTIFACTS / name)
    write_text(ROOT / "README.md", readme(metrics))
    if not (ROOT / ".git").is_dir():
        git("init", "-b", "main")
    if git("rev-parse", "--show-toplevel", capture=True) != str(ROOT):
        raise RuntimeError("Git root differs from the generated repository.")
    git("add", "--", *SOURCES, *OUTPUTS)
    if git("diff", "--cached", "--quiet", check=False) == 1:
        git("-c", "user.name=Swarm Expert Iteration", "-c", "user.email=swarm@example.invalid",
            "commit", "-m", "Record bounded swarm expert iteration")
    if git("status", "--porcelain", capture=True):
        raise RuntimeError("Generated Git repository has outstanding changes.")
    temporary = ROOT / ".handoff.tmp.zip"
    archive = ROOT / "handoff_bundle.zip"
    git("archive", "--format=zip", f"--output={temporary}", "HEAD")
    with ZipFile(temporary) as bundle:
        required = [name for name in (*SOURCES, *OUTPUTS) if not name.startswith("adapters/")]
        required += [f"adapters/{stage}/adapter_model.safetensors" for stage in ("seed_v0", "gen1")]
        missing = set(required).difference(bundle.namelist())
        if missing or bundle.testzip() is not None:
            raise RuntimeError(f"Invalid handoff archive, missing {sorted(missing)}.")
    os.replace(temporary, archive)
    assert_idle_gpu()

    print("\nComplete handoff")
    for item in (ARTIFACTS / "demo_video.mp4", ARTIFACTS / "benchmark_scorecard.png", ROOT / "README.md", archive):
        print(item)
    print(f"Held-out pass@1 {metrics['gen0_pass_at_1']:.1%} → {metrics['gen1_pass_at_1']:.1%} "
          f"({metrics['delta_percentage_points']:+.1f} pp).")
    if not metrics["improvement_gate"]:
        print("Improvement gate unmet.")


if __name__ == "__main__":
    main()
