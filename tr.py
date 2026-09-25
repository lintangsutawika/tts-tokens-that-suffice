import json, glob, os
import statistics
BASE = os.path.expanduser("~/tts-tokens-that-suffice/jobs")

def msg_text(m):
    parts = [m.get("content") or ""]
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function", {}); args = fn.get("arguments", "")
        parts.append(fn.get("name", "")); parts.append(args if isinstance(args, str) else json.dumps(args))
    return "\n".join(p for p in parts if p)

def per_inst(job, T, base_frac):
    out = []
    for tj in glob.glob(os.path.join(BASE, job, "*/agent/mini-swe-agent.trajectory.json")):
        tdir = os.path.dirname(os.path.dirname(tj))
        iid = os.path.basename(tdir); task = iid.rsplit("__", 1)[0]
        rj = os.path.join(tdir, "result.json")
        try:
            rr = json.load(open(rj)); rew = (rr.get("verifier_result") or {}).get("rewards", {}).get("reward")
            r = (1 if rew is not None and rew >= 0.5 else 0) if rew is not None else None
        except Exception: r = None
        # actual prompt tokens
        try:
            h = json.load(open(os.path.join(tdir, "agent", "trajectory.json")))
            actual = h["final_metrics"]["total_prompt_tokens"]
        except Exception: continue
        try: d = json.load(open(tj))
        except Exception: continue
        cum = 0; mono = 0
        for m in d.get("messages", []):
            cum += len(msg_text(m)) // 4
            if m.get("role") == "assistant": mono += cum
        if not T:
            n_trig = 0
        else:
            crude_gap = mono * (1/(1-base_frac) - 1)
            excess = (actual - mono) - crude_gap
            n_trig = max(0, round(excess / T))
        out.append({"task": task, "r": r, "trig": n_trig, "ctx": cum})
    return out

# baseline excess frac per model (from no-summ runs)
MODELS = {
    "Qwen3.5-9B": ("verified-Qwen--Qwen3.5-9B-run-3","verified-Qwen--Qwen3.5-9B-sum-model-sectioned-32k-run-0","verified-Qwen--Qwen3.5-9B-sum-model-sectioned-48k-run-0","verified-Qwen--Qwen3.5-9B-sum-model-sectioned-64k-run-0"),
    "Qwen3.6-35B-A3B": ("verified-Qwen--Qwen3.6-35B-A3B-run-3","verified-Qwen--Qwen3.6-35B-A3B-sum-model-sectioned-32k-run-0","verified-Qwen--Qwen3.6-35B-A3B-sum-model-sectioned-48k-run-0","verified-Qwen--Qwen3.6-35B-A3B-sum-model-sectioned-64k-run-0"),
    "Qwen3.6-27B-FP8": ("verified-Qwen--Qwen3.6-27B-FP8-run-0","verified-Qwen--Qwen3.6-27B-FP8-sum-model-sectioned-32k-run-0","verified-Qwen--Qwen3.6-27B-FP8-sum-model-sectioned-48k-run-0","verified-Qwen--Qwen3.6-27B-FP8-sum-model-sectioned-64k-run-0"),
    "Qwen3.8-27B-FP8": ("verified-Qwen--Qwen3.8-27B-FP8-run-0","verified-Qwen--Qwen3.8-27B-FP8-sum-model-sectioned-32k-run-0","verified-Qwen--Qwen3.8-27B-FP8-sum-model-sectioned-48k-run-0","verified-Qwen--Qwen3.8-27B-FP8-sum-model-sectioned-64k-run-0"),
    "Nemotron": ("verified-nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-run-0","verified-nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-sum-model-sectioned-32k-run-0","verified-nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-sum-model-sectioned-48k-run-0","verified-nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-sum-model-sectioned-64k-run-0"),
}

out = {}
for model, (bj, j32, j48, j64) in MODELS.items():
    base = per_inst(bj, None, 0)
    # compute baseline frac from actual vs mono in no-summ
    actuals = []; monos = []
    # reuse per_inst but need actual/mono; compute frac differently:
    tot_a = 0; tot_m = 0
    for tj in glob.glob(os.path.join(BASE, bj, "*/agent/mini-swe-agent.trajectory.json")):
        tdir = os.path.dirname(os.path.dirname(tj))
        try:
            h = json.load(open(os.path.join(tdir, "agent", "trajectory.json")))
            a = h["final_metrics"]["total_prompt_tokens"]; d=json.load(open(tj))
        except Exception: continue
        cum=0;mo=0
        for m in d.get("messages", []):
            cum += len(msg_text(m))//4
            if m.get("role")=="assistant": mo+=cum
        tot_a+=a; tot_m+=mo
    bf = (tot_a - tot_m)/tot_a if tot_a else 0.2
    out[model] = {
        "no-summ": per_inst(bj, None, bf),
        "32k": per_inst(j32, 32000, bf),
        "48k": per_inst(j48, 48000, bf),
        "64k": per_inst(j64, 64000, bf),
    }
    print(f"{model}: baseline_frac={bf:.3f}")
    for tag in ["32k","48k","64k"]:
        trigs=[x["trig"] for x in out[model][tag] if x["r"] is not None]
        r1=statistics.median([x["trig"] for x in out[model][tag] if x["r"]==1])
        r0=statistics.median([x["trig"] for x in out[model][tag] if x["r"]==0])
        print(f"  {tag}: med_trig_resolved={r1:.1f} med_trig_unresolved={r0:.1f}")

json.dump(out, open(os.path.join(BASE,"per_inst_trigger.json"),"w"), indent=1)
print("saved per_inst_trigger.json")