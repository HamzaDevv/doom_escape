# Architecture

## Model and GPU ownership

One Qwen2.5-3B 4-bit base model revision is used throughout. The seed process trains rank-16, alpha-16 LoRA for 40 optimizer steps on coordinator completions. It exits before the workers start. Each worker loads the pinned model in its own process and applies a CUDA allocator fraction of 0.18 before loading. The second trainer applies a 0.40 fraction and resumes the same LoRA configuration from the seed weights. The 0.94 total leaves room for CUDA context and non-allocator overhead. These are allocator limits on the physical GPU, so the absolute ceilings scale with the card. Whole-device usage is sampled with `nvidia-smi`. The smoke stage checks the trainer's allocator peak separately.

During self-training, Prime sends short local generation probes to each worker and counts replies completed while the trainer process is still alive. The benchmark consistently uses greedy generation, the same public prompts, the same worker assignment, and the same seed before and after the swap.

## Data and verifier

DeepSeek supplies 80 JSON planning examples and proposes 30 pure-Python tasks with a reference and structured input/output cases. The verifier executes each reference before accepting a task. The reference is discarded from the persisted dataset. A fixed subset of 10 task slots is reserved for both evaluations. The other 20 slots supply candidates for training. Seed plans use topics from the exploration pool. The paired score measures performance on these API-generated cases, whose coverage depends on the generated tests.

The model receives the public task and, during exploration, a local routing hint. It never receives the verifier's input/output pairs. The trusted parent makes exact JSON comparisons against stored expected values. Each case gets an isolated subprocess and fresh Python globals. The subprocess sees a scrubbed environment, only standard file descriptors, a small preloaded library allowlist, a syscall allowlist enforced by Linux seccomp, and CPU, output, and memory limits. The verifier grants at most three seconds across the cases for one candidate and aborts if isolation setup fails. Supplied code runs only through this verifier.

After a local planning call, all three loopback workers sample candidates. An invalid plan triggers a bounded default route. Passing candidates are ranked by source length and worker port. At most one new AST-distinct winner per task per round and three per task overall can enter the 24-trace dataset. The continuation trainer validates the task split, worker provenance, seed adapter digest and tests again before fitting.

## Adapter replacement and cleanup

The FastAPI endpoints require a per-run token and only bind to loopback. The swap accepts a known adapter version and the expected SHA-256 digest. Inference and adapter replacement share one lock per node. PEFT copies compatible LoRA weights in place, preserving the worker PID and resident base. HTTP health stays available while generation waits for the lock. Prime broadcasts the swap, attempts seed rollback if acknowledgment fails, and requires an inference reply on the new adapter from all three original processes before post-swap evaluation.

Child process groups are reaped on completion or a handled interrupt. Their streams go to files under `workspace/logs/`. Asciinema only captures the Prime HUD in a 160×44 PTY. The original capture retains real elapsed times. A derivative capture assigns screen time to each phase for the video, whose last four seconds display the scorecard.

## References

- [Unsloth Colab installation](https://unsloth.ai/docs/get-started/install/google-colab)
- [PEFT adapter hot-swapping](https://huggingface.co/docs/peft/en/package_reference/hotswap)
- [DeepSeek model and API reference](https://api-docs.deepseek.com/quick_start/pricing/)
- [asciinema agg usage](https://docs.asciinema.org/manual/agg/usage/)
