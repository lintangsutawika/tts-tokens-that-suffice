#!/usr/bin/env python3
"""
Plot reward and grad_norm overlaid across multiple runs.

Usage:
    uv run scripts/analysis/plot_overlay.py
    uv run scripts/analysis/plot_overlay.py --checkpoints checkpoints --out outputs/metrics_overlay.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUN_LABELS = {
    "summarizer-rl-OLDEST": "OLDEST (lr=1e-4, std-norm)",
    "summarizer-rl-OLDER":  "OLDER (lr=4e-5, std-norm)",
    "summarizer-rl-OLD":    "OLD (lr=1e-4, no std-norm)",
    "summarizer-rl-OLDISH": "OLDISH (lr=1e-4, temp=1.5)",
    "summarizer-rl":        "current (round-robin, clip=1)",
}


def load_metrics(path: Path) -> dict[str, list]:
    data = [json.loads(l) for l in path.open() if l.strip()]
    return {
        "steps":   [d["step"] for d in data],
        "rewards": [d["reward/mean"] for d in data],
        "grads":   [d.get("skyrl.ai/grad_norm", 0) for d in data],
    }


def plot_overlay(checkpoints_dir: Path, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    for run_name, label in RUN_LABELS.items():
        metrics_file = checkpoints_dir / run_name / "metrics.jsonl"
        if not metrics_file.exists():
            continue
        m = load_metrics(metrics_file)
        lw = 2.5 if run_name == "summarizer-rl" else 1.2
        ax1.plot(m["steps"], m["rewards"], marker="o", markersize=3, linewidth=lw, label=label)
        ax2.plot(m["steps"], m["grads"],   marker="o", markersize=3, linewidth=lw, label=label)

    ax1.set_ylabel("Mean Reward")
    ax1.set_title("Reward over Steps (all runs)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("Grad Norm (log scale)")
    ax2.set_xlabel("Step")
    ax2.set_title("Grad Norm over Steps (all runs)")
    ax2.set_yscale("log")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", default="checkpoints")
    parser.add_argument("--out", default="outputs/metrics_overlay.png")
    args = parser.parse_args()

    plot_overlay(Path(args.checkpoints), Path(args.out))


if __name__ == "__main__":
    main()
