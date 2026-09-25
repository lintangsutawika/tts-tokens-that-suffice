#!/usr/bin/env python3
"""
Plot training reward and the held-out eval reward from a metrics.jsonl file.

Train reward/mean is confounded by per-batch data ordering; eval/reward_mean
(fixed held-out set, greedy decoding) is the data-controlled learning signal.

Usage:
    uv run scripts/analysis/plot_metrics.py [--metrics checkpoints/summarizer-rl/metrics.jsonl]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="checkpoints/summarizer-rl/metrics.jsonl")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    steps, means, maxs = [], [], []
    eval_steps, eval_means, eval_mins, eval_maxs = [], [], [], []
    with open(metrics_path) as f:
        for line in f:
            d = json.loads(line)
            step = d.get("step", d.get("progress/batch"))
            # Train reward: summarizer logs "reward/mean"; test_rl_loop logs "reward/total".
            train_reward = d.get("reward/mean", d.get("reward/total"))
            if train_reward is not None:
                steps.append(step)
                means.append(train_reward)
                maxs.append(d.get("reward/max", np.nan))
            # Eval metrics are only logged on eval steps (every eval_every batches).
            if "eval/reward_mean" in d:
                eval_steps.append(step)
                eval_means.append(d["eval/reward_mean"])
                eval_mins.append(d.get("eval/reward_min", np.nan))
                eval_maxs.append(d.get("eval/reward_max", np.nan))

    steps = np.array(steps)
    means = np.array(means)
    maxs = np.array(maxs)
    have_eval = len(eval_steps) > 0
    have_max = maxs.size > 0 and not np.all(np.isnan(maxs))

    n_panels = 2 if have_eval else 1
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(9, 4 * n_panels), sharex=True, squeeze=False
    )
    ax = axes[0, 0]

    # --- Train reward ---
    ax.plot(steps, means, label="train reward (mean)", marker="o", markersize=4, alpha=0.7)
    if have_max:
        ax.plot(steps, maxs, label="train reward/max", marker="s", markersize=4, alpha=0.7)
    if len(steps) >= 2:
        ax.plot(steps, np.poly1d(np.polyfit(steps, means, 1))(steps),
                linestyle="--", color="C0", label="mean trend")
        if have_max:
            ax.plot(steps, np.poly1d(np.polyfit(steps, maxs, 1))(steps),
                    linestyle="--", color="C1", label="max trend")
    ax.set_ylabel("Reward")
    ax.set_title("RL train reward")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Held-out eval reward (the learning signal) ---
    if have_eval:
        es = np.array(eval_steps)
        em = np.array(eval_means)
        ax2 = axes[1, 0]
        ax2.fill_between(es, np.array(eval_mins), np.array(eval_maxs),
                         color="C2", alpha=0.15, label="eval min–max")
        ax2.plot(es, em, label="eval/reward_mean", marker="D", markersize=4, color="C2")
        if len(es) >= 2:
            ax2.plot(es, np.poly1d(np.polyfit(es, em, 1))(es),
                     linestyle="--", color="C2", label="eval trend")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Eval reward")
        ax2.set_title("Held-out eval reward (fixed set, greedy — measures policy improvement)")
        ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax.set_xlabel("Step")

    fig.tight_layout()
    out_path = metrics_path.parent / "reward_plot.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}"
          + (f"  ({len(eval_steps)} eval points)" if have_eval else "  (no eval data found)"))


if __name__ == "__main__":
    main()
