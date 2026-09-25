#!/usr/bin/env python3
"""
Scaled version of compare_summary_prompts.py: run the five-arm compaction
comparison across many instances instead of one, with no container.

For every instance in a `full`-arm run:
  1. find the compaction turn = the first assistant turn whose *preceding*
     context has already crossed the token trigger (where the eval would fire);
  2. that assistant turn is y (ground truth); the context before it is x;
  3. compact x five ways (truncation, mask, and three summarizer prompts) and,
     per arm, score fidelity and generate a single next action [0];
  4. compare [0] to ground truth by exact import-line match and difflib ratio.

No rollout, no env — only the summarizer and scorer endpoints are needed, so
this scales to the whole pool. `[0]` is the only step that is a fair comparison
to ground truth anyway (after it, an executed rollout would branch), so a single
non-executed step is exactly what the [0]-match metric wants.

The summarizer runs on one of two backends (exactly one required):
  * --summarizer-api-base   an OpenAI/vLLM endpoint (litellm), or
  * --tinker-base-url [+ --summarizer-checkpoint]  the tinker server (:9123),
                            which is how a trained LoRA checkpoint is sampled.

    # litellm/vLLM backend
    uv run scripts/eval/scaled_prompt_comparison.py \\
        --full-dir outputs/swe-bench__Qwen3.6-35B-A3B__full \\
        --limit 20 \\
        --summarizer-api-base http://localhost:8001/v1 \\
        --scoring-base-url http://localhost:1235/v1

    # tinker backend, trained checkpoint (thinking enabled + stripped)
    uv run scripts/eval/scaled_prompt_comparison.py \\
        --full-dir outputs/swe-bench__Qwen3.6-35B-A3B__full \\
        --limit 20 --summarizer-max-tokens 2048 \\
        --tinker-base-url http://<amd-host>:9123 \\
        --summarizer-checkpoint tinker://model_329cf39d/weights/000120 \\
        --scoring-base-url http://localhost:1235/v1
"""

from __future__ import annotations

import difflib
import importlib.util
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer
from tqdm import tqdm
from transformers import AutoTokenizer

from minisweagent.models.utils.actions_toolcall import BASH_TOOL
from tts.agent.step import step_once
from tts.reward.copy_penalty import ngram_overlap
from tts.reward.distortion_reward import distortion_reward_messages, precompute_x_context
from tts.reward.utils import to_template_tool_calls
from tts.summarization.mask_based import build_maskenv_scoring_messages
from tts.summarization.model_based import build_z_scoring_messages
from tts.summarization.truncation_based import build_truncation_scoring_messages

# Single source of truth for the prompt set and the summarize() call: reuse the
# per-instance script rather than copy its PROMPTS (which would drift).
_csp_path = Path(__file__).parent / "compare_summary_prompts.py"
_spec = importlib.util.spec_from_file_location("compare_summary_prompts", _csp_path)
_csp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_csp)
PROMPTS = _csp.PROMPTS
summarize = _csp.summarize

app = typer.Typer(add_completion=False)


def command_of(message: dict) -> str:
    """The bash command string from an assistant tool call ('' if none)."""
    tc = message.get("tool_calls") or []
    if not tc:
        return ""
    args = tc[0].get("function", {}).get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, ValueError):
            args = {}
    return args.get("command", "") if isinstance(args, dict) else str(args)


def find_compaction_turn(msgs: list[dict], tok, trigger: int, tools: list) -> int | None:
    """First assistant index whose preceding context has crossed `trigger` tokens.

    Mirrors the eval: the trigger is checked on the accumulated context *before*
    the agent's next query, so the compacted context is msgs[:t] and y is
    msgs[t]. Returns None if the trajectory never crosses (too short to compact).

    Token count is monotonic in prefix length, so a binary search finds the
    crossing in ~log2(turns) renders instead of one per turn — the earlier
    linear scan re-rendered the whole growing prefix at every turn, and that
    render holds the GIL, so it also serialized across worker threads.
    """
    def n_tokens(prefix_len: int) -> int:
        return len(tok.apply_chat_template(
            to_template_tool_calls(msgs[:prefix_len]),
            add_generation_prompt=True, tokenize=True, tools=tools,
        ))

    if n_tokens(len(msgs)) < trigger:
        return None  # never crosses

    # Smallest prefix length whose render is >= trigger.
    lo, hi = 0, len(msgs)
    while lo < hi:
        mid = (lo + hi) // 2
        if n_tokens(mid) >= trigger:
            hi = mid
        else:
            lo = mid + 1
    # The compaction turn is the first assistant turn at or after that crossing.
    for t in range(lo, len(msgs)):
        if msgs[t].get("role") == "assistant":
            return t
    return None


def score_instance(
    tj: Path, *, tok, tools, trigger, keep_first, keep_last_turns, max_size,
    summarize_fn, scoring_model, scoring_base_url, beta, ngram_n,
) -> dict | None:
    """Score one instance end-to-end. Returns a result record, a {'_skip': reason}
    marker, or None when the trajectory never crosses the trigger (a silent skip).

    Pure: touches no shared state, so it is safe to run in a worker thread. All
    the network calls (summarizer + scorer) live here, which is what the thread
    pool overlaps across instances.
    """
    inst = tj.parent.name
    try:
        msgs = [m for m in json.loads(tj.read_text())["messages"] if m.get("role") != "exit"]
    except Exception as exc:
        return {"_skip": f"unreadable ({exc})", "instance": inst}

    turn = find_compaction_turn(msgs, tok, trigger, tools)
    if turn is None:
        return None  # never crosses the trigger

    context, next_action = msgs[:turn], msgs[turn]
    gt_cmd = command_of(next_action)

    x_ctx = precompute_x_context(
        partial_messages=context, next_action=next_action, model=scoring_model,
        api_base=scoring_base_url, tokenizer=tok, tools=tools,
    )
    if x_ctx is None:
        return {"_skip": "x-context scoring failed", "instance": inst}

    arms = [
        # Baseline: the uncompacted context. Its fidelity is ~0 by construction
        # (z == x), and its [0]-vs-GT overlap is the ceiling every compaction arm
        # is measured against — since [0] is a resample, even the full context
        # does not reproduce GT exactly, so a compaction arm near this ceiling
        # has lost almost nothing. Wire-format `context` so [0] generation works.
        ("full", context, ""),
        ("truncation", build_truncation_scoring_messages(context, keep_first, keep_last_turns), ""),
        ("mask", build_maskenv_scoring_messages(context, keep_first, keep_last_turns), ""),
    ]
    for name, prompt in PROMPTS.items():
        z = summarize_fn(context, prompt)
        arms.append((f"summary:{name}",
                     build_z_scoring_messages(z, context, max_size=max_size, keep_first=keep_first), z))

    rec = {"instance": inst, "turn": turn,
           "n_x_ctx_tokens": x_ctx.x_logprobs.n_x_ctx, "n_y_tokens": x_ctx.x_logprobs.n_completion,
           "gt_command": gt_cmd, "arms": {}}
    for name, z_messages, z in arms:
        r = distortion_reward_messages(
            x_ctx=x_ctx, z_messages=z_messages, summary="",
            model=scoring_model, api_base=scoring_base_url, tokenizer=tok, beta=beta, tools=tools,
        )
        try:
            step = step_once(z_messages, model_name=scoring_model, api_base=scoring_base_url,
                             execute=False, model_kwargs={"temperature": 0.0})
            a0 = (step.action or {}).get("command", "")
        except Exception as exc:
            a0 = f"__error__: {type(exc).__name__}"
        rec["arms"][name] = {
            "fidelity_bounded": r.get("fidelity_bounded"),
            "n_z_ctx_tokens": r.get("n_z_ctx_tokens"),
            "fraction_kept": (r["n_z_ctx_tokens"] / x_ctx.x_logprobs.n_x_ctx
                              if r.get("n_z_ctx_tokens") else None),
            "a0_command": a0,
            # Lexical agreement of the generated next action with ground truth:
            # n-gram overlap (fraction of a0's word n-grams found in GT) and the
            # difflib character ratio. Both are surface proxies for "same next
            # step"; fidelity is the principled signal and these complement it.
            "a0_ngram": ngram_overlap(a0, gt_cmd, n=ngram_n),
            "a0_difflib": difflib.SequenceMatcher(None, a0, gt_cmd).ratio(),
            "summary": z,
        }
    return rec


def build_aggregate(per_instance: list[dict], arm_names: list[str]) -> dict:
    agg = {}
    for name in arm_names:
        rows = [r["arms"][name] for r in per_instance]
        fids = [x["fidelity_bounded"] for x in rows if x["fidelity_bounded"] is not None]
        kept = [x["fraction_kept"] for x in rows if x["fraction_kept"] is not None]
        ngr = [x["a0_ngram"] for x in rows]
        dfl = [x["a0_difflib"] for x in rows]
        agg[name] = {
            "n": len(rows),
            "mean_fidelity": sum(fids) / len(fids) if fids else None,
            "mean_fraction_kept": sum(kept) / len(kept) if kept else None,
            "mean_ngram": sum(ngr) / len(ngr) if ngr else None,
            "mean_difflib": sum(dfl) / len(dfl) if dfl else None,
        }
    return agg


def save_output(out: Path, meta: dict, per_instance: list[dict], arm_names: list[str]) -> dict:
    """Write the full result + freshly-computed aggregate; return the aggregate.

    Called after every scored instance so a crash keeps everything up to the
    last completed one, and again at the end. An atomic replace via a temp file
    means a kill mid-write cannot corrupt the checkpoint.
    """
    agg = build_aggregate(per_instance, arm_names)
    payload = {**meta, "n_instances": len(per_instance),
               "aggregate": agg, "instances": per_instance}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    tmp.replace(out)
    return agg


@app.command(help=__doc__)
def main(
    full_dir: Path = typer.Option(..., "--full-dir", help="A *__full run directory of trajectories"),
    limit: int = typer.Option(20, "--limit", help="Max instances that cross the trigger to score"),
    trigger: int = typer.Option(16384, "--trigger", help="Token threshold that fires compaction"),
    keep_first: int = typer.Option(2, "--keep-first"),
    keep_last_turns: int = typer.Option(4, "--keep-last-turns"),
    max_size: int = typer.Option(20, "--max-size"),
    summarizer_model: str = typer.Option("hosted_vllm/Qwen/Qwen3.5-9B", "--summarizer-model"),
    summarizer_api_base: str = typer.Option(
        "", "--summarizer-api-base",
        help="OpenAI/vLLM endpoint (litellm backend). Mutually exclusive with the tinker flags."),
    tinker_base_url: str = typer.Option(
        "", "--tinker-base-url",
        help="tinker server URL, e.g. http://host:9123. Selects the tinker backend."),
    summarizer_checkpoint: str = typer.Option(
        "", "--summarizer-checkpoint",
        help="tinker:// checkpoint to sample (implies tinker backend); empty = tinker base model."),
    summarizer_renderer: str = typer.Option(
        "qwen3_5", "--summarizer-renderer",
        help="renderer for the tinker backend; must match the checkpoint's training renderer."),
    summarizer_max_tokens: int = typer.Option(512, "--summarizer-max-tokens"),
    scoring_model: str = typer.Option("litellm_proxy/Qwen/Qwen3.6-35B-A3B", "--scoring-model"),
    scoring_base_url: str = typer.Option(..., "--scoring-base-url"),
    scoring_tokenizer: str = typer.Option("Qwen/Qwen3.6-35B-A3B", "--scoring-tokenizer"),
    beta: float = typer.Option(1.0, "--beta"),
    ngram_n: int = typer.Option(3, "--ngram-n", help="n for [0]-vs-GT n-gram overlap"),
    workers: int = typer.Option(4, "--workers", help="Concurrent instances in flight"),
    out: Path = typer.Option(Path("outputs/scaled_prompt_comparison.json"), "-o", "--out"),
) -> None:
    tok = AutoTokenizer.from_pretrained(scoring_tokenizer)
    tools = [BASH_TOOL]
    arm_names = ["full", "truncation", "mask"] + [f"summary:{k}" for k in PROMPTS]

    # Pick the summarizer backend. Both are wrapped into one summarize(messages,
    # system_prompt) -> z closure so score_instance is backend-agnostic and the
    # tinker SamplingClient (built once here) is shared across worker threads.
    use_tinker = bool(tinker_base_url or summarizer_checkpoint)
    if use_tinker and summarizer_api_base:
        raise typer.BadParameter(
            "--summarizer-api-base is mutually exclusive with --tinker-base-url/--summarizer-checkpoint")
    if not use_tinker and not summarizer_api_base:
        raise typer.BadParameter(
            "need a summarizer backend: --summarizer-api-base OR --tinker-base-url [+ --summarizer-checkpoint]")

    if use_tinker:
        tinker_base_url = tinker_base_url or "http://localhost:9123"
        os.environ.setdefault("TINKER_API_KEY", "tml-dummy")
        from tts.summarization.model_based import build_summarizer, tinker_summarize
        _t = build_summarizer(
            summarizer_model=summarizer_model, renderer_name=summarizer_renderer,
            max_tokens=summarizer_max_tokens, tinker_base_url=tinker_base_url,
            checkpoint=summarizer_checkpoint,
        )
        _sc, _rd = _t.sampling_client, _t.renderer

        def summarize_fn(messages, system_prompt):
            return tinker_summarize(messages, system_prompt, _sc, _rd, summarizer_max_tokens)

        backend = f"tinker {tinker_base_url} ({summarizer_checkpoint or 'base'})"
    else:
        def summarize_fn(messages, system_prompt):
            return summarize(messages, system_prompt, summarizer_model,
                             summarizer_api_base, summarizer_max_tokens)

        backend = f"litellm {summarizer_api_base}"
    print(f"summarizer backend: {backend}")

    meta = {"full_dir": str(full_dir), "trigger": trigger,
            "scoring_model": scoring_model, "summarizer_model": summarizer_model,
            "summarizer_backend": backend}

    # Resume: reuse instances already scored in a prior run of the same --out.
    per_instance: list[dict] = []
    done: set[str] = set()
    if out.exists():
        prev = json.loads(out.read_text())
        per_instance = prev.get("instances", [])
        done = {r["instance"] for r in per_instance}
        print(f"resuming from {out}: {len(done)} instances already scored")

    trajs = sorted(full_dir.glob("*/*.traj.json"))
    print(f"{len(trajs)} trajectories in {full_dir}; target {limit} that cross {trigger} tok\n")

    scored = len(per_instance)
    candidates = iter(tj for tj in trajs if tj.parent.name not in done)

    def work(tj: Path):
        return score_instance(
            tj, tok=tok, tools=tools, trigger=trigger, keep_first=keep_first,
            keep_last_turns=keep_last_turns, max_size=max_size,
            summarize_fn=summarize_fn, scoring_model=scoring_model,
            scoring_base_url=scoring_base_url, beta=beta, ngram_n=ngram_n,
        )

    # Bounded submission: keep at most `workers` instances in flight and stop
    # feeding new ones once `limit` is reached, so total work is limit + workers,
    # NOT the whole 498-instance pool.
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
                ngr = " ".join(f"{rec['arms'][n]['a0_ngram']:.2f}" for n in arm_names)
                tqdm.write(f"  [{scored:>2}] {rec['instance']:<28} turn {rec['turn']:>3}  "
                           f"ngram({'/'.join(arm_names)})={ngr}")
                save_output(out, meta, per_instance, arm_names)  # checkpoint after each
                bar.update(1)
            if scored >= limit:
                for f in in_flight:
                    f.cancel()
                break
            refill()
    bar.close()

    agg = save_output(out, meta, per_instance, arm_names)

    print(f"\n=== aggregate over {len(per_instance)} instances ===")
    print(f"{'arm':>24} {'fidelity':>9} {'kept':>7} {'ngram[0]':>9} {'difflib[0]':>11}")
    for name in arm_names:
        a = agg[name]
        f = f"{a['mean_fidelity']:.4f}" if a["mean_fidelity"] is not None else "n/a"
        k = f"{a['mean_fraction_kept']:.1%}" if a["mean_fraction_kept"] is not None else "n/a"
        ng = f"{a['mean_ngram']:.3f}" if a["mean_ngram"] is not None else "n/a"
        d = f"{a['mean_difflib']:.3f}" if a["mean_difflib"] is not None else "n/a"
        print(f"{name:>24} {f:>9} {k:>7} {ng:>9} {d:>11}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    app()
