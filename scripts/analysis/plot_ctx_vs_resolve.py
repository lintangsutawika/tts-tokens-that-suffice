#!/usr/bin/env python3
"""Context length (stored-trajectory max, crude //4 tokens) vs resolve rate.
One cumulative-sorted line per variant + the run-3 baseline.
x = per-instance max cumulative token count over stored messages, sorted ascending.
y = running (cumulative) resolve rate up to that point.
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
        vals = [(m[tag]["ctx"], m[tag]["res"]) for m in matrix.values()
                if m[tag]["res"] is not None and m[tag]["ctx"] is not None]
        vals.sort()
        ctxs = [v[0] for v in vals]
        n = len(vals)
        cum_res = 0
        ys = []
        for i, (_, r) in enumerate(vals):
            cum_res += r
            ys.append(cum_res / (i + 1))
        label = f"{tag} (n={n}, rate={cum_res/max(n,1)*100:.1f}%)"
        ax.plot(ctxs, ys, lw=2, label=label)
    ax.axhline(0.5, color="0.8", lw=0.8, ls="--")
    ax.set_xlabel("per-instance max context tokens (crude //4, stored trajectory)")
    ax.set_ylabel("cumulative (running) resolve rate")
    ax.set_title("SWE-bench-verified: resolve rate vs sorted max context length per variant")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(BASE, "fig_ctx_vs_resolve_variants.png")
    fig.savefig(out, dpi=150)
    print("saved", out)
    dump = {iid: {t: matrix[iid][t] for t in VARIANT_JOBS} for iid in matrix}
    jp = os.path.join(BASE, "ctx_vs_resolve_matrix.json")
    json.dump(dump, open(jp, "w"), indent=2)
    print("saved", jp)

if __name__ == "__main__":
    main()