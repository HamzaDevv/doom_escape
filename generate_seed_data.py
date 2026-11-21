"""DeepSeek-only upstream synthesis and executable reference validation."""

import asyncio
import json
import os
import re
from urllib.parse import urlsplit

import httpx

from swarm_common import (
    PLAN_SYSTEM, WORKSPACE, clean_line, code_source, digest, json_text, load_datasets,
    validate_plan, write_json, write_jsonl,
)
from verifier import verify

TOPICS = (
    "run-length encoding and empty strings", "balanced brackets with nesting",
    "stable unique elements", "merging overlapping closed intervals",
    "longest distinct-character substring length", "small integer prime factors",
    "case-insensitive word histogram and ordering", "matrix transpose including empty input",
    "first missing positive integer", "rotate a list with signed offsets",
    "peak concurrent half-open intervals", "flatten small nested lists",
    "palindrome checks after filtering", "count index pairs with a given sum",
    "shortest path length on an unweighted grid", "fully ordered frequency ranking",
    "parse and combine simple integer ranges", "filter a list of short records",
    "small integer Roman numeral conversion", "two-pointer sorted array intersection",
    "apply a queue of ledger transactions", "multiset equality after normalization",
    "reconstruct a path from parent links", "count ways to climb small staircases",
    "merge sorted timestamps from multiple sources", "stable sort by multiple record keys",
    "subtotal a small shopping basket with rule-based pricing",
    "find connected components of a small undirected graph",
    "evaluate postfix arithmetic with integer operators", "cyclic permutation validation",
)
HOLDOUT = {2, 5, 8, 11, 14, 17, 20, 23, 26, 29}
SEED_TOPICS = tuple(topic for index, topic in enumerate(TOPICS) if index not in HOLDOUT)
CONSTRAINTS = (
    "emphasize empty input", "prefer simple operations", "handle repeated values", "guard boundaries",
    "separate parsing and transformation", "describe a validation angle", "prioritize readable code",
    "compare asymptotic costs",
)


def seed_prompt(slot):
    variation, topic_index = divmod(slot, len(SEED_TOPICS))
    topic = SEED_TOPICS[topic_index]
    constraint = CONSTRAINTS[(slot + 2 * variation) % len(CONSTRAINTS)]
    return (
        f"Seed example {slot + 1}/80. Invent a short pure-Python exercise about {topic}; "
        f"planning emphasis: {constraint}. Produce a single JSON object with keys "
        '"problem" (a brief public task requiring def solve(...)) and "plan". The plan follows '
        f"this contract: {PLAN_SYSTEM} Vary port ordering and use task-specific hints. "
        "The thought must be at most 150 characters, and each focus hint at most 120 characters. "
        "Include no solution, answers, hidden tests, or tools. Use plain English and ASCII."
    )


def task_prompt(slot, revision):
    return (
        f"Create exactly one distinct and approachable pure-Python exercise for topic {TOPICS[slot]}. "
        f"Variation index {slot + 1}, regeneration attempt {revision}. "
        'Return a JSON object with keys "title", "problem", "reference", and "cases". '
        'The public "problem" is standalone, declares the exact solve(...) signature and '
        'return type, and defines ordering, tie-breaking and edge conditions. Do not include '
        'hidden test values in the description. "reference" contains executable Python defining '
        'solve, with no file, network, randomness, or side effects. "cases" is an array of 6 to 8 '
        'distinct { "args": [positional arguments], "expected": exact JSON return } objects. '
        'Use bounded integers, strings, booleans, lists or string-keyed dicts and include edge '
        'cases. No floats. Keep each case under 750 characters and the reference under 50 lines. '
        "Use only builtins or collections, itertools, functools, heapq, bisect, math, re, string. "
        "Use plain English and ASCII."
    )


def no_floats(value):
    if isinstance(value, list):
        return all(no_floats(child) for child in value)
    if isinstance(value, dict):
        return all(no_floats(child) for child in value.values())
    return value is None or type(value) in (int, bool, str)


def checked_problem(value):
    if not isinstance(value, str) or not 35 <= len(value.strip()) <= 1600 or not value.isascii():
        raise ValueError("invalid public task")
    if not re.search(r"\bsolve\s*\(", value):
        raise ValueError("missing solve signature")
    return value.strip()


def checked_seed(value, slot):
    return {"id": f"seed-{slot+1:03}", "problem": checked_problem(value["problem"]),
            "plan": validate_plan(value["plan"])}


async def checked_task(value, slot):
    problem = checked_problem(value["problem"])
    title, cases = value["title"], value["cases"]
    if not isinstance(title, str) or not 3 <= len(title.strip()) <= 90:
        raise ValueError("invalid title")
    if not isinstance(cases, list) or not 6 <= len(cases) <= 8 or not no_floats(cases):
        raise ValueError("invalid test cases")
    if any(not isinstance(case, dict) or set(case) != {"args", "expected"}
           or not isinstance(case["args"], list) or len(json_text(case)) > 900 for case in cases):
        raise ValueError("invalid case schema")
    signatures = {json_text(case["args"]) for case in cases}
    if len(signatures) != len(cases) or len({json_text(c["expected"]) for c in cases}) < 2:
        raise ValueError("insufficient variety in tests")
    if not isinstance(value["reference"], str):
        raise ValueError("reference must be Python source")
    passed, reason = await asyncio.to_thread(verify, code_source(value["reference"]), cases)
    if not passed:
        raise ValueError(f"reference failed {reason}")
    return {
        "id": f"task-{slot+1:02}", "title": clean_line(title, 90), "problem": problem,
        "split": "held_out" if slot in HOLDOUT else "explore", "cases": cases,
    }


class APIError(RuntimeError):
    pass


class Retryable(RuntimeError):
    pass


class Synthesis:
    def __init__(self, client, model, base_url):
        self.client, self.model, self.base_url = client, model, base_url
        self.limit = asyncio.Semaphore(5)
        self.fingerprints = {"seed": set(), "task": set()}

    async def ask(self, prompt, tokens):
        payload = {
            "model": self.model, "messages": [
                {"role": "system", "content": "Generate structured data. Respond with a JSON object."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"}, "max_tokens": tokens,
        }
        host = (urlsplit(self.base_url).hostname or "").lower()
        if host == "deploy.com" or host.endswith(".deploy.com"):
            payload.pop("response_format")
            payload["max_completion_tokens"] = payload.pop("max_tokens")
            payload["reasoning_effort"] = "none"
        elif self.model in ("deepseek-flash", "deepseek-v4-pro", "deepseek-v4-flash"):
            payload["thinking"] = {"type": "disabled"}
        async with self.limit:
            response = await self.client.post(self.base_url + "/chat/completions", json=payload)
        if response.status_code in (408, 425, 429) or response.status_code >= 500:
            raise Retryable(f"HTTP {response.status_code}")
        if not response.is_success:
            raise APIError(f"Upstream API rejected the request (HTTP {response.status_code}). Check key and model.")
        choice = response.json()["choices"][0]
        if choice["finish_reason"] != "stop":
            raise ValueError("incomplete JSON response")
        content = choice["message"]["content"].strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, flags=re.DOTALL | re.IGNORECASE)
        return json.loads(fenced.group(1) if fenced else content)

    async def item(self, kind, slot):
        for attempt in range(7):
            try:
                if kind == "seed":
                    row = checked_seed(await self.ask(seed_prompt(slot), 700), slot)
                else:
                    row = await checked_task(await self.ask(task_prompt(slot, attempt), 3300), slot)
                key = " ".join(row["problem"].lower().split())
                if key in self.fingerprints[kind] or (
                    kind == "task" and key in self.fingerprints["seed"]
                ):
                    raise ValueError("duplicate task text")
                self.fingerprints[kind].add(key)
                print(f"Verified {row['id']}", flush=True)
                return row
            except (httpx.TransportError, Retryable, ValueError, KeyError, IndexError, TypeError) as error:
                if attempt == 6:
                    raise RuntimeError(f"{kind} slot {slot+1} exhausted retries ({type(error).__name__}).") from None
                await asyncio.sleep(min(1.5 * 2**attempt, 24) + (slot % 5) / 10)

    async def batch(self, kind, count):
        async with asyncio.TaskGroup() as group:
            jobs = [group.create_task(self.item(kind, i)) for i in range(count)]
        return [job.result() for job in jobs]


async def main():
    endpoint = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
    manifest = WORKSPACE / "dataset_manifest.json"
    if manifest.exists():
        load_datasets()
        print("Reusing verified dataset in this run directory.")
        return
    token = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not token:
        raise SystemExit("Set DEEPSEEK_API_KEY to synthesize the initial dataset.")
    uri = urlsplit(endpoint)
    if uri.scheme != "https" or not uri.hostname or uri.username or uri.password or uri.query or uri.fragment:
        raise SystemExit("DEEPSEEK_BASE_URL must be an HTTPS API base URL.")
    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {token}"}, timeout=75, trust_env=False
    ) as http:
        service = Synthesis(http, model, endpoint)
        seeds = await service.batch("seed", 80)
        tasks = await service.batch("task", 30)
    seed_path, task_path = (WORKSPACE / x for x in ("seed_dataset.jsonl", "synthetic_tasks.jsonl"))
    write_jsonl(seed_path, seeds)
    write_jsonl(task_path, tasks)
    write_json(manifest, {
        "schema": 1, "upstream_model": model,
        "sha256": {path.name: digest(path) for path in (seed_path, task_path)},
        "seed_rows": 80, "explore_tasks": 20, "held_out_tasks": 10,
    })


if __name__ == "__main__":
    asyncio.run(main())
