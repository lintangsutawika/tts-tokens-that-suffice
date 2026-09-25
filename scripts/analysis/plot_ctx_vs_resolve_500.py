#!/usr/bin/env python3
"""Context length (stored-trajectory max, crude //4 tokens) vs resolve rate.
x = per-instance max cumulative token count over stored messages, sorted ascending.
y = cumulative resolved count over the FIXED /500 denominator, i.e. resolves with
    max_ctx <= x are counted, divided by 500. Same denominator at every x-position
    so all variants (and run-3) are directly comparable at a given context length.
Each curve ends at the run's overall /500 resolve rate.
"""
import json, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.expanduser("~/tts-tokens-that-suffice/jobs")
PREFIX = "verified-Qwen--Qwen3.6-35B-A3B-sum-model-sectioned"
VARIANT_JOBS = {
    "run3-base": "verified-Qwen--Qwen3.6-35B-A3B-run-3",
    "32k": f"{PREFIX}-32k-run-0",
    "48k": f"{PREFIX}-48k-run-0",
    "64k": f"{PREFIX}-64k-run-0",
}
TOTAL = 500

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

def main():
    matrix = results_matrix(VARIANT_JOBS)
    fig, ax = plt.subplots(figsize=(10, 6))
    for tag, jobname in VARIANT_JOBS.items():
        # treat None-resolution as not-resolved for the /500 denominator? No: /500 means
        # resolved/500; ungraded counts as unresolved (consistent with rate-over-500).
        vals = [(m[tag]["ctx"], m[tag]["res"]) for m in matrix.values()
                if m[tag]["ctx"] is not None]
        vals.sort()
        ctxs = [v[0] for v in vals]
        cum_res = 0
        ys = []
        for i, (_, r) in enumerate(vals):
            if r == 1:
                cum_res += 1
            ys.append(cum_res / TOTAL)
        label = f"{tag} (resolved={cum_res}/500={cum_res/TOTAL*100:.1f}%)"
        ax.plot(ctxs, ys, lw=2, label=label)
    ax.set_ylim(0, 1)
    ax.set_xlabel("per-instance max context tokens (crude //4, stored trajectory)")
    ax.set_ylabel("cumulative resolved count / 500")
    ax.set_title("SWE-bench-verified: cumulative resolved /500 vs max context length per variant")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(BASE, "fig_ctx_vs_resolve_variants_500.png")
    fig.savefig(out, dpi=150)
    print("saved", out)

if __name__ == "__main__":
    main()