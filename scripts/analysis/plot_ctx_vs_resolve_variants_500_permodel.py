#!/usr/bin/env python3
"""Context length (stored-trajectory max, crude //4 tokens) vs resolve rate,
per model. Semantics match plot_ctx_vs_resolve_500.py EXACTLY:
x = per-instance max cumulative token count over stored messages, sorted ascending.
y = cumulative resolved count over the FIXED /500 denominator, i.e. resolves with
    max_ctx <= x are counted, divided by 500. Same denominator at every x-position
    so all variants (and the baseline) are directly comparable at a given ctx length.
Each curve ends at the run's overall /500 resolve rate.

Generates one fig_ctx_vs_resolve_variants_500.png per model:
  fig_ctx_vs_resolve_variants_500_9b.png
  fig_ctx_vs_resolve_variants_500_35A3B.png
  fig_ctx_vs_resolve_variants_500_27BFP8.png
"""
import json, glob, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.expanduser("~/tts-tokens-that-suffice/jobs")
TOTAL = 500

# tag -> job name per model. 27B-FP8 parts (32k, 48k) are partial; 64k & run-0 full.
MODELS = {
    "9b": {
        "suffix": "_9b",
        "title": "9B (Qwen3.5-9B)",
        "jobs": {
            "run3-base": "verified-Qwen--Qwen3.5-9B-run-3",
            "32k": "verified-Qwen--Qwen3.5-9B-sum-model-sectioned-32k-run-0",
            "48k": "verified-Qwen--Qwen3.5-9B-sum-model-sectioned-48k-run-0",
            "64k": "verified-Qwen--Qwen3.5-9B-sum-model-sectioned-64k-run-0",
        },
    },
    "35A3B": {
        "suffix": "_35A3B",
        "title": "35B-A3B (Qwen3.6-35B-A3B)",
        "jobs": {
            "run3-base": "verified-Qwen--Qwen3.6-35B-A3B-run-3",
            "32k": "verified-Qwen--Qwen3.6-35B-A3B-sum-model-sectioned-32k-run-0",
            "48k": "verified-Qwen--Qwen3.6-35B-A3B-sum-model-sectioned-48k-run-0",
            "64k": "verified-Qwen--Qwen3.6-35B-A3B-sum-model-sectioned-64k-run-0",
        },
    },
    "27BFP8": {
        "suffix": "_27BFP8",
        "title": "27B-FP8 (Qwen3.8-27B-FP8)",
        "jobs": {
            "run0-base": "verified-Qwen--Qwen3.8-27B-FP8-run-0",
            "32k": "verified-Qwen--Qwen3.8-27B-FP8-sum-model-sectioned-32k-run-0",
            "48k": "verified-Qwen--Qwen3.8-27B-FP8-sum-model-sectioned-48k-run-0",
            "64k": "verified-Qwen--Qwen3.8-27B-FP8-sum-model-sectioned-64k-run-0",
        },
    },
}


def msg_text(m):
    parts = [m.get("content") or ""]
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function", {})
        args = fn.get("arguments", "")
        parts.append(fn.get("name", ""))
        parts.append(args if isinstance(args, str) else json.dumps(args))
    return "\n".join(p for p in parts if p)


def max_ctx(j):
    cum = 0
    mx = 0
    for m in j.get("messages", []):
        cum += len(msg_text(m)) // 4
        mx = max(mx, cum)
    return mx


def results_matrix(tag_jobs):
    res = {}
    for tag, jobname in tag_jobs.items():
        job = os.path.join(BASE, jobname)
        for tj in glob.glob(os.path.join(job, "*/agent/mini-swe-agent.trajectory.json")):
            iid = os.path.basename(os.path.dirname(os.path.dirname(tj))).rsplit("__", 1)[0]
            try:
                j = json.load(open(tj))
            except Exception:
                d = {"res": None, "ctx": None}
            else:
                vr = None
                rj = os.path.join(os.path.dirname(os.path.dirname(tj)), "result.json")
                try:
                    rr = json.load(open(rj))
                    rew = (rr.get("verifier_result") or {}).get("rewards", {}).get("reward")
                    vr = (1 if rew >= 0.5 else 0) if rew is not None else None
                except Exception:
                    vr = None
                d = {"res": vr, "ctx": max_ctx(j)}
            res.setdefault(iid, {})[tag] = d
    return res


def plot_model(tag, spec):
    matrix = results_matrix(spec["jobs"])
    fig, ax = plt.subplots(figsize=(10, 6))
    for t, jobname in spec["jobs"].items():
        vals = [(m.get(t, {}).get("ctx"), m.get(t, {}).get("res")) for m in matrix.values()]
        vals = [(c, r) for (c, r) in vals if c is not None]
        vals.sort()
        ctxs = [v[0] for v in vals]
        cum_res = 0
        ys = []
        for _, r in vals:
            if r == 1:
                cum_res += 1
            ys.append(cum_res / TOTAL)
        n = len(vals)
        label = f"{t} ({cum_res}/{n}=... resolved={cum_res/TOTAL*100:.1f}% of 500)"
        ax.plot(ctxs, ys, lw=2, label=label)
    ax.set_ylim(0, 1)
    ax.set_xlabel("per-instance max context tokens (crude //4, stored trajectory)")
    ax.set_ylabel("cumulative resolved count / 500")
    ax.set_title(f"SWE-bench-verified: cumulative resolved /500 vs max context length — {spec['title']}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(BASE, "fig_ctx_vs_resolve_variants_500" + spec["suffix"] + ".png")
    fig.savefig(out, dpi=150)
    print("saved", out, f"({tag})")


def main():
    for tag, spec in MODELS.items():
        plot_model(tag, spec)


if __name__ == "__main__":
    main()