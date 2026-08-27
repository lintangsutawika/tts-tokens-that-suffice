#!/usr/bin/env python3
"""
How far past the compaction point does a compaction keep hurting?

compare_summary_prompts / scaled measure the *next* turn only. But a compaction
made at turn T also sits under turns T+1, T+2, ... . This measures the decay:
for depths k = 0..N, how well does the compacted context still predict the k-th
real subsequent turn, given the real turns in between?

    z, t_0, t_1, t_2, t_3  ->  t_4

The t_i between z and the scored turn are the ACTUAL ground-truth turns, not
generated — teacher forcing. That is what keeps arms comparable at depth > 0: a
free rollout would send each arm down its own branch, so any gap would conflate
"the compaction hurt" with "the arms diverged." Here every arm is conditioned on
the identical recent history, so the only difference is the compacted head, and
the fidelity gap at depth k is the compaction's residual influence k turns later.

Per depth k the reference context is just msgs[:y_index] (the natural full
prefix) and each arm's context is compacted(msgs[:T]) + msgs[T:y_index] — the
compacted head with the real turns glued back on. As k grows the shared
real-turn tail grows, so influence should fade; depth 0 should be worst.

    uv run scripts/analysis/compaction_decay.py \\
        --full-dir outputs/swe-bench__Qwen3.6-35B-A3B__full \\
        --limit 20 --max-depth 4 --workers 8 \\
        --summarizer-api-base http://babel-t9-24:8001/v1 \\
        --scoring-base-url http://0.0.0.0:8080/v1
"""

from __future__ import annotations

import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer
from tqdm import tqdm
from transformers import AutoTokenizer

from minisweagent.models.utils.actions_toolcall import BASH_TOOL
from tts.reward.distortion_reward import distortion_reward_messages, precompute_x_context
from tts.summarization.mask_based import build_maskenv_scoring_messages
from tts.summarization.model_based import build_z_scoring_messages
from tts.summarization.truncation_based import build_truncation_scoring_messages

# Reuse the prompt set and summarize() (single source of truth; see scaled).
_csp_path = Path(__file__).parent / "compare_summary_prompts.py"
_spec = importlib.util.spec_from_file_location("compare_summary_prompts", _csp_path)
_csp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_csp)
PROMPTS = _csp.PROMPTS
summarize = _csp.summarize

# Reuse find_compaction_turn from the scaled script.
_scp_path = Path(__file__).parent / "scaled_prompt_comparison.py"
_scp_spec = importlib.util.spec_from_file_location("scaled_prompt_comparison", _scp_path)
_scp = importlib.util.module_from_spec(_scp_spec)
_scp_spec.loader.exec_module(_scp)
find_compaction_turn = _scp.find_compaction_turn

app = typer.Typer(add_completion=False)


def arm_bases(context: list[dict], *, keep_first, keep_last_turns, max_size,
              summarizer_model, summarizer_api_base, summarizer_max_tokens) -> dict[str, list[dict]]:
    """The compacted head for each arm, built once from msgs[:T].

    Each is later extended by the real subsequent turns at every depth, so the
    compression is done here (including the summarizer calls) exactly once.
    """
    bases = {
        # full = the uncompacted context, so full+history == the reference prefix
        # and its fidelity is ~0 at every depth (the sanity baseline / ceiling).
        "full": list(context),
        "truncation": build_truncation_scoring_messages(context, keep_first, keep_last_turns),
        "mask": build_maskenv_scoring_messages(context, keep_first, keep_last_turns),
    }
    for name, prompt in PROMPTS.items():
        z = summarize(context, prompt, summarizer_model, summarizer_api_base, summarizer_max_tokens)
        bases[f"summary:{name}"] = build_z_scoring_messages(
            z, context, max_size=max_size, keep_first=keep_first)
    return bases


def score_instance(
    tj: Path, *, tok, tools, trigger, keep_first, keep_last_turns, max_size, max_depth,
    summarizer_model, summarizer_api_base, summarizer_max_tokens,
    scoring_model, scoring_base_url, beta,
) -> dict | None:
    """Fidelity of every arm at depths 0..max_depth for one instance."""
    inst = tj.parent.name
    try:
        msgs = [m for m in json.loads(tj.read_text())["messages"] if m.get("role") != "exit"]
    except Exception as exc:
        return {"_skip": f"unreadable ({exc})", "instance": inst}

    T = find_compaction_turn(msgs, tok, trigger, tools)
    if T is None:
        return None  # never crosses the trigger

    # The real assistant turns at and after the compaction point are the y's.
    asst = [i for i in range(T, len(msgs)) if msgs[i].get("role") == "assistant"]
    depths = list(range(min(max_depth + 1, len(asst))))
    if not depths:
        return {"_skip": "no assistant turns at/after compaction point", "instance": inst}

    bases = arm_bases(
        msgs[:T], keep_first=keep_first, keep_last_turns=keep_last_turns, max_size=max_size,
        summarizer_model=summarizer_model, summarizer_api_base=summarizer_api_base,
        summarizer_max_tokens=summarizer_max_tokens)
    arm_names = list(bases)

    rec = {"instance": inst, "compaction_turn": T, "depths": depths,
           "y_turns": [], "n_x_tokens": [],
           "arms": {a: {"fidelity_by_depth": []} for a in arm_names}}

    for k in depths:
        y_index = asst[k]
        history = msgs[T:y_index]  # the real turns t_0..t_{k-1} between z and the scored turn
        x_ctx = precompute_x_context(
            partial_messages=msgs[:y_index], next_action=msgs[y_index], model=scoring_model,
            api_base=scoring_base_url, tokenizer=tok, tools=tools,
        )
        rec["y_turns"].append(y_index)
        if x_ctx is None:  # context too long / scoring failed at this depth: stop the curve
            rec["n_x_tokens"].append(None)
            for a in arm_names:
                rec["arms"][a]["fidelity_by_depth"].append(None)
            break
        rec["n_x_tokens"].append(x_ctx.x_logprobs.n_x_ctx)
        for a in arm_names:
            r = distortion_reward_messages(
                x_ctx=x_ctx, z_messages=bases[a] + history, summary="",
                model=scoring_model, api_base=scoring_base_url, tokenizer=tok, beta=beta, tools=tools,
            )
            rec["arms"][a]["fidelity_by_depth"].append(r.get("fidelity_bounded"))
    return rec


def build_aggregate(per_instance: list[dict], arm_names: list[str], max_depth: int) -> dict:
    """Mean fidelity per arm per depth, over instances that reached that depth."""
    agg = {}
    for a in arm_names:
        by_depth = []
        for k in range(max_depth + 1):
            vals = []
            for r in per_instance:
                fbd = r["arms"].get(a, {}).get("fidelity_by_depth", [])
                if k < len(fbd) and fbd[k] is not None:
                    vals.append(fbd[k])
            by_depth.append({"depth": k, "n": len(vals),
                             "mean_fidelity": sum(vals) / len(vals) if vals else None})
        agg[a] = by_depth
    return agg


def save_output(out: Path, meta: dict, per_instance: list[dict], arm_names: list[str],
                max_depth: int) -> dict:
    agg = build_aggregate(per_instance, arm_names, max_depth)
    payload = {**meta, "n_instances": len(per_instance), "aggregate": agg, "instances": per_instance}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    tmp.replace(out)
    return agg


@app.command(help=__doc__)
def main(
    full_dir: Path = typer.Option(..., "--full-dir"),
    limit: int = typer.Option(20, "--limit"),
    max_depth: int = typer.Option(4, "--max-depth", help="Score subsequent turns 0..N"),
    trigger: int = typer.Option(16384, "--trigger"),
    keep_first: int = typer.Option(2, "--keep-first"),
    keep_last_turns: int = typer.Option(4, "--keep-last-turns"),
    max_size: int = typer.Option(20, "--max-size"),
    summarizer_model: str = typer.Option("hosted_vllm/Qwen/Qwen3.5-9B", "--summarizer-model"),
    summarizer_api_base: str = typer.Option(..., "--summarizer-api-base"),
    summarizer_max_tokens: int = typer.Option(512, "--summarizer-max-tokens"),
    scoring_model: str = typer.Option("litellm_proxy/Qwen/Qwen3.6-35B-A3B", "--scoring-model"),
    scoring_base_url: str = typer.Option(..., "--scoring-base-url"),
    scoring_tokenizer: str = typer.Option("Qwen/Qwen3.6-35B-A3B", "--scoring-tokenizer"),
    beta: float = typer.Option(1.0, "--beta"),
    workers: int = typer.Option(4, "--workers"),
    out: Path = typer.Option(Path("outputs/compaction_decay.json"), "-o", "--out"),
) -> None:
    tok = AutoTokenizer.from_pretrained(scoring_tokenizer)
    tools = [BASH_TOOL]
    arm_names = ["full", "truncation", "mask"] + [f"summary:{k}" for k in PROMPTS]
    meta = {"full_dir": str(full_dir), "trigger": trigger, "max_depth": max_depth,
            "scoring_model": scoring_model, "summarizer_model": summarizer_model}

    per_instance: list[dict] = []
    done: set[str] = set()
    if out.exists():
        prev = json.loads(out.read_text())
        per_instance = prev.get("instances", [])
        done = {r["instance"] for r in per_instance}
        print(f"resuming from {out}: {len(done)} instances already scored")

    trajs = sorted(full_dir.glob("*/*.traj.json"))
    print(f"{len(trajs)} trajectories; target {limit} crossing {trigger} tok, depths 0..{max_depth}\n")

    scored = len(per_instance)
    candidates = iter(tj for tj in trajs if tj.parent.name not in done)

    def work(tj: Path):
        return score_instance(
            tj, tok=tok, tools=tools, trigger=trigger, keep_first=keep_first,
            keep_last_turns=keep_last_turns, max_size=max_size, max_depth=max_depth,
            summarizer_model=summarizer_model, summarizer_api_base=summarizer_api_base,
            summarizer_max_tokens=summarizer_max_tokens, scoring_model=scoring_model,
            scoring_base_url=scoring_base_url, beta=beta,
        )

    bar = tqdm(total=limit, initial=scored, unit="inst", desc="scoring")
    in_flight: set = set()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        def refill():
            while len(in_flight) < workers and scored < limit:
                tj = next(candidates, None)
                if tj is None:
                    break
                in_flight.add(ex.submit(work, tj))

        refill()
        while in_flight:
            fut = next(as_completed(in_flight))
            in_flight.discard(fut)
            rec = fut.result()
            if rec is not None and "_skip" in rec:
                tqdm.write(f"  skip {rec['instance']}: {rec['_skip']}")
            elif rec is not None and scored < limit:
                per_instance.append(rec)
                done.add(rec["instance"])
                scored += 1
                d0 = " ".join(f"{rec['arms'][a]['fidelity_by_depth'][0]:+.3f}"
                              if rec['arms'][a]['fidelity_by_depth'] and
                              rec['arms'][a]['fidelity_by_depth'][0] is not None else "  n/a"
                              for a in arm_names)
                tqdm.write(f"  [{scored:>2}] {rec['instance']:<28} T={rec['compaction_turn']:>3} "
                           f"depths={len(rec['depths'])}  fid@0({'/'.join(arm_names)})={d0}")
                save_output(out, meta, per_instance, arm_names, max_depth)
                bar.update(1)
            if scored >= limit:
                for f in in_flight:
                    f.cancel()
                break
            refill()
    bar.close()

    agg = save_output(out, meta, per_instance, arm_names, max_depth)

    print(f"\n=== mean fidelity by depth over {len(per_instance)} instances ===")
    hdr = "  ".join(f"d{k}" for k in range(max_depth + 1))
    print(f"{'arm':>24}  {hdr}")
    for a in arm_names:
        cells = []
        for k in range(max_depth + 1):
            m = agg[a][k]["mean_fidelity"]
            cells.append(f"{m:+.3f}" if m is not None else "  n/a")
        print(f"{a:>24}  {'  '.join(cells)}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    app()
