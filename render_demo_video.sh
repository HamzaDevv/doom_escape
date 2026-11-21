#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
runner="${SWARM_PYTHON:-.venv/bin/python}"
agg="${SWARM_AGG:-.tools/agg}"

"$runner" retime_cast.py
font_dir="$("$runner" -c 'from pathlib import Path; import matplotlib; print(Path(matplotlib.get_data_path()) / "fonts/ttf")')"
"$agg" --cols 160 --rows 44 --font-size 16 --font-dir "$font_dir" --font-family "DejaVu Sans Mono" \
  --speed 6 --idle-time-limit 300 --fps-cap 12 --last-frame-duration 1 --theme monokai \
  workspace/timelapse.cast workspace/raw_demo.gif

ffmpeg -nostdin -y -hide_banner -loglevel error -filter_threads 2 -ignore_loop 1 \
  -i workspace/raw_demo.gif \
  -vf "fps=30,scale=1920:1080:force_original_aspect_ratio=decrease:force_divisible_by=2:flags=lanczos,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0x0d1117,setsar=1" \
  -an -c:v libx264 -preset fast -crf 15 -pix_fmt yuv420p -threads 4 workspace/terminal_part.mp4
ffmpeg -nostdin -y -hide_banner -loglevel error -filter_complex_threads 2 \
  -i workspace/terminal_part.mp4 -loop 1 -framerate 30 -i artifacts/benchmark_scorecard.png \
  -filter_complex '[1:v]trim=duration=4,setpts=PTS-STARTPTS,setsar=1,format=yuv420p[outro];[0:v][outro]concat=n=2:v=1:a=0[v]' \
  -map '[v]' -an -c:v libx264 -preset medium -b:v 1M -minrate 1M -maxrate 1M -bufsize 2M \
  -x264-params 'nal-hrd=cbr:force-cfr=1' -pix_fmt yuv420p -r 30 -threads 4 -movflags +faststart \
  workspace/demo_video.mp4
ffmpeg -nostdin -hide_banner -v error -xerror -i workspace/demo_video.mp4 -map 0:v:0 -f null -
"$runner" - <<'PY'
import json
import subprocess
from pathlib import Path

video = Path("workspace/demo_video.mp4")
info = json.loads(subprocess.check_output([
    "ffprobe", "-v", "error", "-show_entries",
    "stream=width,height,pix_fmt,r_frame_rate:format=duration,size", "-of", "json", str(video),
]))
stream = info["streams"][0]
duration = float(info["format"]["duration"])
assert (stream["width"], stream["height"], stream["pix_fmt"], stream["r_frame_rate"]) == (
    1920, 1080, "yuv420p", "30/1"
), info
assert 45 <= duration <= 60 and int(info["format"]["size"]) > 500_000, info
video.replace("artifacts/demo_video.mp4")
print(f"Verified 1920×1080 / 30 fps / {duration:.1f} s.")
PY
