#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

if [[ "$(uname -sm)" != "Linux x86_64" ]]; then
  echo "Use a Linux x86_64 Colab runtime with one A100." >&2
  exit 1
fi
bootstrap_python="${SWARM_BOOTSTRAP_PYTHON:-python3}"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
if [[ "$gpu_name" != *A100* || "$gpu_name" == *$'\n'* || "$gpu_name" == *MIG* ]]; then
  echo "Expected one full A100. Found: $gpu_name" >&2
  exit 1
fi

mkdir -p .tools workspace/logs artifacts adapters
if [[ "$(id -u)" == "0" ]]; then
  APT=(apt-get)
else
  APT=(sudo apt-get)
fi
"${APT[@]}" update -qq
"${APT[@]}" install -y -qq ffmpeg libseccomp2 fonts-dejavu-core curl git

if [[ ! -x .venv/bin/python ]]; then
  "$bootstrap_python" -m venv --without-pip .venv
fi
runner=.venv/bin/python
if ! "$runner" -m pip --version >/dev/null 2>&1; then
  "$bootstrap_python" -m pip --python "$runner" install --disable-pip-version-check pip
fi
"$runner" -m pip install --disable-pip-version-check -r requirements.txt
"$runner" -m pip check

install_release() {
  local repo="$1" version="$2" asset="$3" hash="$4" name="$5"
  local target=".tools/$name"
  if ! printf '%s  %s\n' "$hash" "$target" | sha256sum --check --status; then
    curl --fail --location --silent --show-error --retry 3 \
      "https://github.com/asciinema/$repo/releases/download/$version/$asset" -o "$target.download"
    printf '%s  %s\n' "$hash" "$target.download" | sha256sum --check
    chmod +x "$target.download"
    mv "$target.download" "$target"
  fi
  "$target" --version
}

install_release asciinema v3.2.1 asciinema-x86_64-unknown-linux-gnu \
  1b405bbda565b33c3c4718de67fedc3535580603c0694b1ff3fb04f363430a20 asciinema
install_release agg v1.9.0 agg-x86_64-unknown-linux-gnu \
  f111e315cd71056b116302342553dd765b7297579ed511f111d0cedb442aeda6 agg
ffmpeg -version
ffprobe -version

"$runner" - <<'PY'
import unsloth
import torch
import fastapi
import rich
from peft.utils.hotswap import hotswap_adapter

assert torch.cuda.is_available() and torch.cuda.device_count() == 1
gpu = torch.cuda.get_device_properties(0)
assert "A100" in gpu.name and 38 <= gpu.total_memory / 2**30 <= 82, gpu
assert torch.cuda.is_bf16_supported()
print("CUDA, Unsloth, FastAPI, Rich, and PEFT loaded.")
PY
"$runner" verifier.py --self-test
"$runner" -m pip freeze > workspace/dependencies.txt
