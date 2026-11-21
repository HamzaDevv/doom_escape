"""Parent-side, three-second verifier for structured JSON test cases."""

import argparse
import ast
import json
import subprocess
import sys
import tempfile
import time

from swarm_common import ROOT, json_text

RUNNER = ROOT / "sandbox_runner.py"
TIMEOUT = 3.0
MAX_SOURCE = 8000


def run_case(request, timeout):
    with tempfile.TemporaryFile() as output:
        try:
            result = subprocess.run(
                [sys.executable, "-S", "-P", "-B", str(RUNNER)],
                input=json_text(request).encode(),
                stdout=output,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
                env={"LANG": "C.UTF-8", "PYTHONHASHSEED": "0"},
                cwd="/",
                start_new_session=True,
            )
        except subprocess.TimeoutExpired:
            return "timeout", b""
        if result.returncode == 78:
            raise RuntimeError("The verifier subprocess could not establish its seccomp filter.")
        output.seek(0)
        return ("ok" if result.returncode == 0 else "runtime"), output.read(32769)


def verify(code, cases):
    if len(code) > MAX_SOURCE or not cases:
        return False, "input_limits"
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return False, "syntax"
    if not any(isinstance(node, ast.FunctionDef) and node.name == "solve" for node in tree.body):
        return False, "missing_solve"
    deadline = time.monotonic() + TIMEOUT
    for index, case in enumerate(cases, 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False, "timeout"
        status, stdout = run_case({"code": code, "args": case["args"]}, remaining)
        if status != "ok":
            return False, status
        try:
            actual = json.loads(stdout)
            equal = json_text(actual) == json_text(case["expected"])
        except (ValueError, RecursionError):
            return False, "invalid_output"
        if not equal:
            return False, f"wrong_case_{index}"
    return True, "passed"


def self_test():
    status, result = run_case({"probe": True}, TIMEOUT)
    if status != "ok" or result.strip() != b"isolated":
        raise RuntimeError("Linux verifier isolation probe failed.")
    cases = [{"args": [[3, 1, 3]], "expected": [1, 3]}, {"args": [[]], "expected": []}]
    good = "from collections import Counter\ndef solve(xs): return sorted(Counter(xs))"
    bad = "def solve(xs): return xs"
    loop = "def solve(xs):\n    while True: pass"
    if verify(good, cases) != (True, "passed") or verify(bad, cases)[0] or verify(loop, cases)[0]:
        raise RuntimeError("Verifier did not separate valid, invalid, and resource-limited code.")
    print("Verifier self-test passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    self_test()
