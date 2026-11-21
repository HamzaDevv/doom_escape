"""Render a 1920×1080 benchmark scorecard from measured telemetry."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from swarm_common import ARTIFACTS, read_json

BACKGROUND, PANEL = "#0d1117", "#151e2b"
INK, MUTED = "#eef5ff", "#abbacf"
CYAN, GREEN, RED = "#6ad9f5", "#53e6aa", "#ff5979"


def main():
    data = read_json(ARTIFACTS / "metrics.json")
    with plt.rc_context({
        "font.family": "DejaVu Sans", "text.color": INK,
        "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.facecolor": PANEL, "figure.facecolor": BACKGROUND,
    }):
        fig = plt.figure(figsize=(16, 9), dpi=120)
        fig.text(0.066, 0.910, "EXPERT ITERATION  /  PAIRED BENCHMARK",
                 fontsize=12, weight="bold", color=CYAN)
        fig.text(0.064, 0.837, "Seed LoRA to self-trained LoRA", fontsize=36, weight="bold")
        delta = data["delta_percentage_points"]
        score_label = f"{delta:+.0f} pp on {data['held_out_count']} held-out Python tasks"
        fig.text(0.067, 0.766, score_label, fontsize=17, color=GREEN if delta > 0 else RED)
        fig.text(0.067, 0.713, "PASS@1  /  ONE GREEDY COMPLETION PER TASK", fontsize=11, color=MUTED)
        fig.text(0.533, 0.713, "A100 VRAM  /  CONCURRENT SELF-TRAIN", fontsize=11, color=MUTED)

        score = fig.add_axes((0.067, 0.300, 0.410, 0.377))
        values = [data["gen0_pass_at_1"], data["gen1_pass_at_1"]]
        bars = score.bar([0, 1], values, width=0.48, color=[CYAN, GREEN], zorder=3)
        score.set(xlim=(-0.6, 1.6), ylim=(0, 1.18))
        score.set_yticks([0, 0.25, 0.5, 0.75, 1])
        score.yaxis.set_major_formatter(PercentFormatter(1))
        score.set_xticks([0, 1], ["Gen-0   Seed", "Gen-1   Post-swap"], fontsize=13)
        for index, bar in enumerate(bars):
            score.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.035,
                       f"{data[f'gen{index}_pass']}/{data['held_out_count']}",
                       ha="center", fontsize=18, weight="bold")

        memory = fig.add_axes((0.535, 0.300, 0.393, 0.377))
        overlap = [row for row in data["gpu_samples"] if row["phase"] == 2]
        if overlap:
            start = data["phases"][2]["seconds"]
            x = [(row["seconds"] - start) for row in overlap]
            y = [row["used_mib"] / 1024 for row in overlap]
            total = overlap[0]["total_mib"] / 1024
            memory.plot(x, y, color=CYAN, linewidth=2.4, marker="." if len(x) < 2 else None)
            memory.fill_between(x, y, color=CYAN, alpha=0.15)
            memory.set_ylim(0, total * 1.10)
            memory.axhline(total, color=MUTED, linestyle=":", linewidth=1)
            memory.set_xlabel("Seconds into self-training", labelpad=10)
            memory.set_ylabel("VRAM in use (GiB)", labelpad=8)
            fig.text(0.557, 0.610, f"Peak {max(y):.1f} / {total:.1f} GiB",
                     fontsize=17, weight="bold")
        else:
            memory.text(0.5, 0.52, "GPU telemetry unavailable", transform=memory.transAxes,
                        ha="center", fontsize=15)
            memory.set_xticks([])
            memory.set_yticks([])
        for axis in (score, memory):
            axis.tick_params(length=0, pad=6)
            axis.grid(axis="y", alpha=0.16, zorder=0)
            for spine in axis.spines.values():
                spine.set_visible(False)

        fig.text(0.068, 0.158, f"{data['verified_traces']} verified local traces", color=GREEN,
                 fontsize=19, weight="bold")
        fig.text(0.068, 0.115, f"{data['sandbox_passed']} passed  /  {data['sandbox_rejected']} rejected across all phases",
                 color=MUTED, fontsize=11)
        fig.text(0.535, 0.158, f"{data['trainer']['probe_completions']} worker responses during training",
                 color=CYAN, fontsize=19, weight="bold")
        fig.text(0.535, 0.115, "Qwen2.5-3B Instruct  /  4-bit base and rank-16 LoRA",
                 color=MUTED, fontsize=11)
        fig.savefig(ARTIFACTS / "benchmark_scorecard.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
