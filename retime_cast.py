"""Retime the recorded phases while preserving the original ANSI event sequence."""

import bisect
import json

from swarm_common import ARTIFACTS, WORKSPACE, read_json


def main():
    original = ARTIFACTS / "demo.cast"
    with original.open(encoding="utf-8") as source:
        header = json.loads(next(source))
        first, *_ = json.loads(next(source))
        last = first
        for line in source:
            last, *_ = json.loads(line)
    # The first HUD appears at Python boot. Anchors are seconds relative to that HUD.
    data = read_json(ARTIFACTS / "metrics.json")
    positions = [0.0] + [entry["seconds"] for entry in data["phases"]] + [last - first]
    for index in range(1, len(positions)):
        positions[index] = max(positions[index], positions[index-1] + 0.001)
    times = (0, 5, 12, 24, 34, 39, 44)
    if len(positions) != len(times):
        raise ValueError("Expected all five run phases in the capture.")
    header.pop("idle_time_limit", None)
    with original.open(encoding="utf-8") as source, (WORKSPACE / "timelapse.cast").open(
        "w", encoding="utf-8"
    ) as output:
        next(source)
        output.write(json.dumps(header) + "\n")
        for line in source:
            stamp, kind, content = json.loads(line)
            elapsed = stamp - first
            slot = min(len(times) - 2, bisect.bisect_right(positions, elapsed) - 1)
            ratio = (elapsed - positions[slot]) / (positions[slot+1] - positions[slot])
            changed = times[slot] + ratio * (times[slot+1] - times[slot])
            output.write(json.dumps([round(changed * 6, 4), kind, content]) + "\n")


if __name__ == "__main__":
    main()
