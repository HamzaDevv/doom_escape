# Bounded swarm expert iteration

A single-GPU experiment that trains a coordinator LoRA, runs three local workers, collects verifier-gated traces, and measures a second generation after an adapter hot-swap. The repository includes the supporting Python scripts, video pipeline, and one notebook that runs the training and evaluation stages in sequence.

## Run on Google Colab through VS Code

1. Open `swarm_expert_iteration_colab.ipynb` in VS Code and connect it to a Google Colab A100 40 GB or 80 GB runtime.
2. Run the cells in order. Enter your DeepSeek API key at the hidden prompt in the first cell. You can also configure `DEEPSEEK_API_KEY` as an environment variable or enable the Colab secret option in that cell.
3. Wait for the notebook to finish. It clones this repository, installs the environment, generates data, trains both adapters, runs the evaluation, renders the video, and packages the results.

The notebook defaults to the `deepseek-flash` model and provides settings for the DeepSeek model and base URL. API calls can incur charges. Training and local model inference run on the Colab GPU.

The generated files appear under `/content/swarm-expert-iter`, including `artifacts/demo_video.mp4`, `artifacts/benchmark_scorecard.png`, `artifacts/metrics.json`, and `handoff_bundle.zip`. Use the Colab Contents view in VS Code to download the archive. The run records the observed scores even when Gen-1 does not improve over Gen-0.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and [INIT_SPEC.md](INIT_SPEC.md) for the original requirements. Real adapters and measured results are produced when you run the notebook.
