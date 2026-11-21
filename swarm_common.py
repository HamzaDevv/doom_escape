"""Shared protocol, paths, and atomic artifact helpers."""

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT / "workspace"
LOGS = WORKSPACE / "logs"
ARTIFACTS = ROOT / "artifacts"
ADAPTERS = ROOT / "adapters"
PORTS = (8001, 8002, 8003)
MODEL = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
MODEL_REVISION = "2672b588848527e456d320c11926c794539f47d5"
SEED = 3407
CONTEXT = 2048
MAX_CODE_TOKENS = 420
TRACE_TARGET = 24
CORE_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

PLAN_SYSTEM = (
    'You are Prime, the coordinator of a bounded Python problem-solving swarm. '
    'Return one JSON object with keys "thought", "workers", and "focus". '
    '"thought" is a brief operational decision summary; "workers" is a permutation '
    'of [8001, 8002, 8003]; "focus" is an array of three short, distinct strategy hints '
    'in corresponding order. Each hint must help implement the public Python task. '
    'Never include code, commands, or tool calls in the plan.'
)
SOLVE_SYSTEM = (
    "Implement the given pure Python function solve. Return only Python source. "
    "Use no filesystem, network, input, print, or subprocesses. "
    "Return the requested JSON-compatible value. Keep the implementation concise."
)


def stage_environment():
    env = {key: value for key, value in os.environ.items() if not key.startswith("DEEPSEEK_")}
    env.update({
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TRANSFORMERS_VERBOSITY": "error",
        "TQDM_DISABLE": "1",
        "WANDB_MODE": "disabled",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
    })
    return env


def write_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(content)
            handle.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def json_text(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True)


def write_json(path, value):
    write_text(path, json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path, rows):
    write_text(path, "".join(json_text(row) + "\n" for row in rows))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def clean_line(value, limit):
    return " ".join(value.split()).encode("ascii", "replace").decode()[:limit]


def plan_request(problem):
    return f"Public task:\n{problem}"


def validate_plan(value):
    if not isinstance(value, dict) or set(value) != {"thought", "workers", "focus"}:
        raise ValueError("unexpected plan keys")
    workers, focus, thought = value["workers"], value["focus"], value["thought"]
    if not isinstance(workers, list) or len(workers) != 3 or any(type(x) is not int for x in workers):
        raise ValueError("invalid worker routing")
    if set(workers) != set(PORTS) or not isinstance(focus, list) or len(focus) != 3:
        raise ValueError("invalid plan routes")
    if not isinstance(thought, str) or not thought.strip() or len(thought) > 200:
        raise ValueError("invalid decision summary")
    if any(not isinstance(x, str) or not x.strip() or len(x) > 160 for x in focus):
        raise ValueError("invalid strategy")
    if len({x.strip().lower() for x in focus}) != 3:
        raise ValueError("duplicate strategies")
    return {
        "thought": clean_line(thought, 200),
        "workers": workers,
        "focus": [clean_line(x, 160) for x in focus],
    }


def solve_request(problem, focus=""):
    return problem + (f"\n\nImplementation hint: {focus}" if focus else "")


def code_source(text):
    text = text.strip()
    fenced = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.I | re.S)
    if fenced:
        text = fenced.group(1).strip()
    return text + "\n"


def adapter_fingerprint(path):
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (path / name).is_file() or not (path / name).stat().st_size:
            raise ValueError(f"Incomplete adapter at {path}")
    return digest(path / "adapter_model.safetensors")


def load_datasets():
    manifest_path = WORKSPACE / "dataset_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema") != 1:
        raise ValueError("Unsupported dataset manifest. Use a fresh run directory.")
    paths = [WORKSPACE / name for name in ("seed_dataset.jsonl", "synthetic_tasks.jsonl")]
    for path in paths:
        if manifest["sha256"].get(path.name) != digest(path):
            raise ValueError(f"Dataset checksum mismatch: {path}")
    seeds, tasks = (read_jsonl(path) for path in paths)
    if len(seeds) != 80 or len(tasks) != 30 or len({t["id"] for t in tasks}) != 30:
        raise ValueError("Expected 80 seed rows and 30 unique tasks.")
    if sum(t["split"] == "held_out" for t in tasks) != 10:
        raise ValueError("Expected ten held-out tasks.")
    return seeds, tasks
