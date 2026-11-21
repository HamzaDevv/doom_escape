# INIT_SPEC.md — Bounded Swarm Expert-Iteration (STaR) Prototype & Proof-of-Work Handoff

## 1. Objective & Final Handoff Deliverable
Build, execute, record, and package a self-contained, single-GPU (Google Colab A100 40GB/80GB) demonstration of **Multi-Agent Orchestration + Verifier-Gated Expert Iteration**:
1. Synthesize high-diversity seed orchestration examples and unit-tested coding tasks via the **DeepSeek API** (`DEEPSEEK_API_KEY`).
2. Fine-tune a lightweight seed LoRA (`adapters/seed_v0`) on `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` to unlock the autonomous JSON coordinator persona.
3. Run **1 Prime Coordinator** and **3 independent FastAPI Worker Processes** (`127.0.0.1:8001–8003`) rendered through a **16:9 cinema-grade `rich` terminal UI** with zero background log bleed.
4. Execute the full cycle (**Gen-0 Eval** $\rightarrow$ **Best-of-3 Swarm Sampling + Subprocess Verification** $\rightarrow$ **Concurrent LoRA Self-Training** $\rightarrow$ **Zero-Downtime Hot-Swap** $\rightarrow$ **Gen-1 Eval**) while recording the terminal session headlessly.
5. **Final Handoff:** Produce a polished `artifacts/demo_video.mp4` (time-lapsed to ~45–60s with the final benchmark scorecard appended) alongside a fully documented, reproducible Git repository and `handoff_bundle.zip`.

---

## 2. Hard Architectural & UI Constraints
- **DeepSeek API for Upstream Seed Synthesis:**
  - Read `DEEPSEEK_API_KEY` from `os.environ` (optional `DEEPSEEK_BASE_URL="https://api.deepseek.com"`, `DEEPSEEK_MODEL="deepseek-chat"`). Fail fast if `DEEPSEEK_API_KEY` is missing.
  - Use async `httpx` / `openai` (`Semaphore(5)`) to generate **80 unique JSON orchestration traces** (`workspace/seed_dataset.jsonl`) and **30 subprocess-verified Python coding tasks** (`workspace/synthetic_tasks.jsonl`).
  - **Scientific Integrity Rule:** DeepSeek API is used *only* upstream. All `Gen-0 -> Gen-1` solutions in `artifacts/verified_gen1.jsonl` MUST be generated locally by the 3B worker swarm (`:8001–8003`) and verified by the Python subprocess sandbox.
- **Single-A100 VRAM Partitioning (40GB Total Budget):**
  - Workers `8001`, `8002`, `8003`: Each process MUST set `torch.cuda.set_per_process_memory_fraction(0.18, device=0)` before model load (~54% total VRAM).
  - Concurrent Trainer (`run_rsi_train.py`): MUST set `torch.cuda.set_per_process_memory_fraction(0.40, device=0)` so training runs concurrently with live worker inference without OOM.
- **Video-Ready Terminal UI (`prime_orchestrator.py`):**
  - Target a fixed `160x44` terminal grid with a clean dark palette (obsidian/crimson/cyan/emerald) optimized for 1080p video rendering.
  - **Strict Stderr Isolation:** Redirect all child process `stdout`/`stderr`, Hugging Face progress bars, and `transformers`/`uvicorn` warnings to `workspace/logs/*.log` so zero raw stack traces or progress bars corrupt the `rich.live.Live` frame.
  - **HUD Layout:**
    - **Header (3 rows):** Title (`PRIME-0 // DECENTRALIZED EXPERT-ITERATION SWARM`), animated phase pipeline (`[1. GEN-0 EVAL] -> [2. SWARM EXPLORE] -> [3. SELF-TRAIN] -> [4. HOT-SWAP] -> [5. GEN-1 EVAL]`), elapsed clock, and live `nvidia-smi` GPU utilization + VRAM meter.
    - **Left Column (55% width):** Decentralized Edge Mesh table (`EDGE-NODE-8001..8003`, live status pill `ONLINE` / `INFERRING` / `SYNCING`, active LoRA adapter version `seed_v0` $\rightarrow$ `gen1`, tasks solved) + Prime's live JSON `"thought"` reasoning box.
    - **Right Column (45% width):** Verified trace progress bar (`0/24`), deterministic sandbox gate counter (passed vs rejected), and live side-by-side benchmark scorecard (`Gen-0 pass@1` vs `Gen-1 pass@1`).
    - **Footer (12 rows):** Clean, timestamped Prime orchestration event stream.

---

## 3. Final Proof-of-Work Repository Layout
The agent MUST organize the workspace into a clean, publication-ready repository structure before handoff:

```text
swarm-expert-iter/
├── README.md                  # Executive overview, Mermaid architecture diagram, live results table & repro guide
├── ARCHITECTURE.md            # Technical deep-dive: VRAM partitioning, deterministic verifier gate, LoRA hot-swap
├── INIT_SPEC.md               # This specification
├── setup_env.sh               # Installs Python deps + asciinema, agg, and ffmpeg for headless video rendering
├── generate_seed_data.py      # Stage 0: Async DeepSeek API seed & task generator
├── seed_lora.py               # Stage 1: Trains adapters/seed_v0 on workspace/seed_dataset.jsonl
├── worker_node.py             # Stage 2: FastAPI edge worker (:8001-8003) with /health, /infer, /hot_swap
├── run_rsi_train.py           # Stage 3: Concurrent Unsloth SFT trainer on verified traces
├── prime_orchestrator.py      # Stage 4: Async coordinator, subprocess verifier, bash trigger & 160x44 Rich HUD
├── plot_results.py            # Stage 5: High-DPI 16:9 dark-mode benchmark scorecard generator
├── render_demo_video.sh       # Stage 6: Headless .cast -> .gif -> 1080p .mp4 (with scorecard outro) pipeline
├── adapters/
│   ├── seed_v0/               # Initial seed LoRA checkpoint
│   └── gen1/                  # Self-trained successor LoRA checkpoint
├── artifacts/                 # Curated proof-of-work outputs for LinkedIn & GitHub
│   ├── demo_video.mp4         # Final 1080p time-lapsed terminal demo + scorecard outro (~45-60s)
│   ├── demo.cast              # Raw asciinema terminal recording
│   ├── benchmark_scorecard.png# High-DPI Gen-0 vs Gen-1 comparison chart
│   ├── metrics.json           # Full run telemetry, timestamps, and pass@1 benchmark scores
│   └── verified_gen1.jsonl    # Self-generated, unit-test-verified training traces
└── workspace/
    └── logs/                  # Isolated worker & trainer logs (excluded from git via .gitignore)
4. Sequential Execution CheckpointsExecute and verify every checkpoint in order. Do not skip checkpoints.Checkpoint 1: Environment & Headless Video Toolchain (setup_env.sh)Action: Install Python dependencies (unsloth, trl, peft, bitsandbytes, fastapi, uvicorn, httpx, datasets, rich, openai, matplotlib) and system video tools (asciinema, ffmpeg, and the agg binary from asciinema/agg GitHub releases).Verification Gate: Verify python -c "import unsloth, fastapi, rich", asciinema --version, agg --version, and ffmpeg -version all exit 0.Checkpoint 2: DeepSeek Seed Data & Seed LoRA (generate_seed_data.py + seed_lora.py)Action:Run generate_seed_data.py using DEEPSEEK_API_KEY to create workspace/seed_dataset.jsonl (80 diverse coordinator JSON examples) and workspace/synthetic_tasks.jsonl (30 subprocess-verified Python coding tasks).Run seed_lora.py (r=16, lora_alpha=16, max_steps=40) to produce adapters/seed_v0.Verification Gate: Confirm adapters/seed_v0/adapter_model.safetensors exists and is non-empty.Checkpoint 3: Hot-Swappable Worker Node (worker_node.py) & Concurrent Trainer (run_rsi_train.py)Action:Implement worker_node.py (127.0.0.1:<PORT>, 0.18 VRAM cap) with GET /health, POST /infer, and POST /hot_swap using peft.utils.hotswap.hotswap_adapter (in-place weight replacement with zero downtime).Implement run_rsi_train.py (0.40 VRAM cap) to fine-tune adapters/seed_v0 on verified traces and export adapters/gen1.Verification Gate: Smoke-test 1 worker on :8001 (/health $\rightarrow$ /infer $\rightarrow$ /hot_swap $\rightarrow$ clean terminate) and verify run_rsi_train.py compiles a dummy adapter under <15GB VRAM.Checkpoint 4: Recorded Full Run (prime_orchestrator.py + asciinema)Action: Implement prime_orchestrator.py with the 160x44 video-optimized rich HUD, subprocess sandbox verifier (timeout=3s), autonomous train_next_gen.sh generation/execution, live POST /hot_swap broadcast across :8001–8003, and artifacts/metrics.json export. Execute the full run inside a fixed-dimension PTY recorded by asciinema:Bashstty cols 160 rows 44 && asciinema rec artifacts/demo.cast --cols 160 --rows 44 -c "python prime_orchestrator.py" --overwrite
Verification Gate: Confirm artifacts/demo.cast, artifacts/verified_gen1.jsonl (24 verified traces), adapters/gen1/adapter_model.safetensors, and artifacts/metrics.json (gen1_pass > gen0_pass) all exist with zero orphan GPU processes.Checkpoint 5: Benchmark Chart & Video Rendering (plot_results.py + render_demo_video.sh)Action:Run plot_results.py to render artifacts/benchmark_scorecard.png (1920x1080 dark-mode chart showing Gen-0 vs Gen-1 pass@1 accuracy, verified trace yield, and concurrent GPU VRAM telemetry).Implement and run render_demo_video.sh to convert artifacts/demo.cast into a crisp 1080p LinkedIn-ready MP4:Render artifacts/demo.cast via agg --cols 160 --rows 44 --font-size 16 --speed 6 --theme monokai artifacts/demo.cast workspace/raw_demo.gif (compressing idle waits and speeding up long inference loops into ~45 seconds).Convert workspace/raw_demo.gif to workspace/terminal_part.mp4 (1920x1080, yuv420p, 30fps, centered on #0d1117 dark canvas via ffmpeg).Append a 4-second still frame of artifacts/benchmark_scorecard.png at the end via ffmpeg concat to produce the final artifacts/demo_video.mp4.Verification Gate: Verify artifacts/demo_video.mp4 exists, is non-empty (>500KB), has 1920x1080 resolution via ffprobe, and plays cleanly from terminal boot through the benchmark scorecard outro.Checkpoint 6: Documentation, Git Repo Initialization & Final HandoffAction:Write README.md (with Mermaid architecture diagram, actual Gen-0 vs Gen-1 numbers pulled from artifacts/metrics.json, embedded artifacts/benchmark_scorecard.png, and reproduction commands) and ARCHITECTURE.md.Initialize a clean Git repository (git init, .gitignore excluding workspace/logs/ and temporary GIFs, clean commit history) and package the entire repository + artifacts/demo_video.mp4 into handoff_bundle.zip.Print a concise Final Handoff Summary listing the exact paths to artifacts/demo_video.mp4, artifacts/benchmark_scorecard.png, README.md, and handoff_bundle.zip, along with the recorded Gen-0 -> Gen-1 accuracy jump.Verification Gate: Confirm git status is clean, handoff_bundle.zip is created, and all GPU memory is released (nvidia-smi shows 0MB active python processes).
---

### What Your Agent Will Hand Back to You at the End
Once Checkpoint 6 finishes, you will have four ready-to-use deliverables waiting in your Colab workspace:
1. **`artifacts/demo_video.mp4`:** A 1080p, ~50-second time-lapsed terminal recording showing the 3 edge nodes running live, the concurrent LoRA self-training, the zero-downtime weight hot-swap, and a 4-second outro of your benchmark scorecard—ready to upload straight to LinkedIn.
2. **`artifacts/benchmark_scorecard.png`:** A dark-mode 16:9 results chart (`Gen-0 Seed` vs. `Gen-1 Post-Swap` pass@1 accuracy) to attach as a second slide or standalone image.
3. **A Complete Git Proof-of-Work Repo (`README.md` + `ARCHITECTURE.md` + Source + `artifacts/`):** Ready to push to GitHub and link in the first comment of your LinkedIn post.
4. **`handoff_bundle.zip`:** A single archive containing the entire repository and video for one-click download from Colab.