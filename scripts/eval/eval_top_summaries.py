#!/usr/bin/env python3
"""
Evaluate a set of *already-generated* summarizer outputs with the same
fidelity + [0]-vs-GT machinery as scaled_prompt_comparison.py.

scaled_prompt_comparison.py *generates* z from PROMPTS on a full-dir; this
script instead ingests fixed z texts (e.g. the highest-reward summaries pulled
from a training run's summaries.jsonl) and scores each one against the exact
(x, y) it was trained/rewarded on. For every (trajectory, z) it reports, per
arm, the scaled_prompt_comparison metrics:

    full / truncation / mask   — reference arms rebuilt on the same x
    trained                    — the fixed z text

x and y are reconstructed the way train_rl did: load the source collect-format
trajectory, threshold_split at split_at_tokens, then
  context   = steps_to_messages(traj.steps, ...)          # x
  y         = next assistant turn of the continuation      # GT next action

Needs a scoring endpoint (Qwen3.6-35B-A3B), the same one training used.

    uv run scripts/eval/eval_top_summaries.py \
        --top-summaries /tmp/top5.json \
        --source-trajectories /tmp/src5.jsonl \
        --scoring-base-url http://babel-u9-20:8080/v1
"""
from __future__ import annotations

import difflib
import json
from pathlib import Path

import typer
from transformers import AutoTokenizer

from minisweagent.models.utils.actions_toolcall import BASH_TOOL
from tts.agent.step import step_once
from tts.data.agent_trajectory import (
    AGENT_SYSTEM_PROMPT,
    AgentTrajectory,
    steps_to_messages,
)
from tts.reward.copy_penalty import ngram_overlap
from tts.reward.distortion_reward import (
    distortion_reward_messages,
    next_agent_action,
    precompute_x_context,
)
from tts.summarization.mask_based import build_maskenv_scoring_messages
from tts.summarization.model_based import build_z_scoring_messages
from tts.summarization.truncation_based import build_truncation_scoring_messages

app = typer.Typer(add_completion=False)


def to_wire_tool_calls(messages: list[dict]) -> list[dict]:
    """Serialize tool_call arguments dict -> JSON string (OpenAI wire format).

    steps_to_messages emits arguments as dicts, but the chat-completion API
    (step_once) rejects a dict — it wants the string the .traj.json trajectories
    that scaled_prompt_comparison reads already carry. The reward/logprob path
    re-parses to dicts via to_template_tool_calls, so this keeps both happy.
    """
    import copy
    out = copy.deepcopy(messages)
    for m in out:
        for tc in m.get("tool_calls") or []:
            args = tc.get("function", {}).get("arguments")
            if isinstance(args, dict):
                tc["function"]["arguments"] = json.dumps(args)
    return out


def command_of(message: dict | None) -> str:
    """The bash command string from an assistant tool call ('' if none)."""
    if not message:
        return ""
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


@app.command(help=__doc__)
def main(
    top_summaries: Path = typer.Option(..., "--top-summaries",
        help="JSON list of {trajectory_id, summary, reward, overlap} records"),
    source_trajectories: Path = typer.Option(..., "--source-trajectories",
        help="JSONL of collect-format source trajectories (from datasets/summarizer_train.jsonl)"),
    # Split knobs — must match the training run's config.json.
    split_at_tokens: int = typer.Option(16384, "--split-at-tokens"),
    compaction_token_budget: int = typer.Option(9000, "--compaction-token-budget"),
    min_split_prefix: int = typer.Option(6, "--min-split-prefix"),
    min_split_suffix: int = typer.Option(3, "--min-split-suffix"),
    keep_first: int = typer.Option(4, "--keep-first"),
    keep_last_turns: int = typer.Option(3, "--keep-last-turns"),
    max_size: int = typer.Option(20, "--max-size"),
    scoring_model: str = typer.Option("litellm_proxy/Qwen/Qwen3.6-35B-A3B", "--scoring-model"),
    scoring_base_url: str = typer.Option(..., "--scoring-base-url"),
    scoring_tokenizer: str = typer.Option("Qwen/Qwen3.6-35B-A3B", "--scoring-tokenizer"),
    beta: float = typer.Option(1.0, "--beta"),
    ngram_n: int = typer.Option(3, "--ngram-n"),
    a0_samples: int = typer.Option(1, "--a0-samples",
        help="Next-action resamples per arm; averaged, to steady the MoE/greedy noise"),
    a0_temperature: float = typer.Option(0.0, "--a0-temperature",
        help="Sampling temperature for the resampled next action (0.6 ~ deployment)"),
    out: Path = typer.Option(Path("outputs/eval_top_summaries.json"), "-o", "--out"),
) -> None:
    tok = AutoTokenizer.from_pretrained(scoring_tokenizer)
    tools = [BASH_TOOL]
    max_continuation = split_at_tokens - compaction_token_budget

    # z-texts keyed by trajectory_id.
    top = json.loads(top_summaries.read_text())
    z_by_id = {r["trajectory_id"]: r for r in top}

    # Source trajectories keyed by trajectory_id.
    src: dict[str, AgentTrajectory] = {}
    for line in source_trajectories.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        t = AgentTrajectory.from_collect_dict(json.loads(line))
        src[t.trajectory_id] = t

    arm_names = ["full", "truncation", "mask", "trained"]
    per_instance: list[dict] = []

    for tid, rec in z_by_id.items():
        traj = src.get(tid)
        if traj is None:
            print(f"  skip {tid}: no source trajectory")
            continue
        split = traj.threshold_split(
            tok, split_at_tokens=split_at_tokens,
            min_prefix=min_split_prefix, min_suffix=min_split_suffix,
            max_continuation_tokens=max_continuation,
        )
        if split is None:
            print(f"  skip {tid}: threshold_split returned None")
            continue

        sys_prompt = split.agent_system_prompt or AGENT_SYSTEM_PROMPT
        # Wire format (JSON-string tool args) throughout, matching the .traj.json
        # inputs scaled_prompt_comparison consumes; step_once forwards these to the
        # chat API, which rejects dict args.
        context = to_wire_tool_calls(
            steps_to_messages(split.steps, split.task, system_prompt=sys_prompt))
        cont = to_wire_tool_calls(
            steps_to_messages(split.continuation, split.task, system_prompt=sys_prompt))[2:]
        y = next_agent_action(cont)
        gt_cmd = command_of(y)
        if y is None:
            print(f"  skip {tid}: no assistant turn in continuation")
            continue

        x_ctx = precompute_x_context(
            partial_messages=context, next_action=y, model=scoring_model,
            api_base=scoring_base_url, tokenizer=tok, tools=tools,
        )
        if x_ctx is None:
            print(f"  skip {tid}: x-context scoring failed")
            continue

        z = rec["summary"]
        arms = [
            ("full", context, ""),
            ("truncation", build_truncation_scoring_messages(context, keep_first, keep_last_turns), ""),
            ("mask", build_maskenv_scoring_messages(context, keep_first, keep_last_turns), ""),
            ("trained", build_z_scoring_messages(z, context, max_size=max_size, keep_first=keep_first), z),
        ]

        entry = {
            "instance": tid, "n_prefix_steps": len(split.steps),
            "train_reward": rec.get("reward"), "train_overlap": rec.get("overlap"),
            "n_x_ctx_tokens": x_ctx.x_logprobs.n_x_ctx,
            "n_y_tokens": x_ctx.x_logprobs.n_completion,
            "gt_command": gt_cmd, "arms": {},
        }
        for name, z_messages, z_text in arms:
            r = distortion_reward_messages(
                x_ctx=x_ctx, z_messages=z_messages, summary="",
                model=scoring_model, api_base=scoring_base_url, tokenizer=tok,
                beta=beta, tools=tools,
            )
            # Resample the next action a0_samples times and score each against GT.
            # A single greedy draw is unstable on an MoE served with continuous
            # batching (temp-0 output still varies run to run); averaging K draws
            # at the deployment temperature gives a steady "how often does this
            # compacted context lead to GT's next action" estimate.
            # a0_samples=0 skips the next-action resampling entirely (fidelity-only
            # pass, e.g. when re-ranking a large pool by corrected fidelity).
            a0s: list[str] = []
            for _ in range(a0_samples):
                try:
                    step = step_once(z_messages, model_name=scoring_model, api_base=scoring_base_url,
                                     execute=False, model_kwargs={"temperature": a0_temperature})
                    a0s.append((step.action or {}).get("command", ""))
                except Exception as exc:
                    a0s.append(f"__error__: {type(exc).__name__}")
            ngrams = [ngram_overlap(a, gt_cmd, n=ngram_n) for a in a0s]
            difflibs = [difflib.SequenceMatcher(None, a, gt_cmd).ratio() for a in a0s]
            entry["arms"][name] = {
                "fidelity_bounded": r.get("fidelity_bounded"),
                "n_z_ctx_tokens": r.get("n_z_ctx_tokens"),
                "fraction_kept": (r["n_z_ctx_tokens"] / x_ctx.x_logprobs.n_x_ctx
                                  if r.get("n_z_ctx_tokens") else None),
                "a0_command": a0s[0] if a0s else None,
                "a0_commands": a0s,
                "a0_ngram": (sum(ngrams) / len(ngrams)) if ngrams else None,
                "a0_difflib": (sum(difflibs) / len(difflibs)) if difflibs else None,
                "a0_exact_match_rate": (sum(a.strip() == gt_cmd.strip() for a in a0s) / len(a0s)) if a0s else None,
            }
        per_instance.append(entry)
        f = entry["arms"]["trained"]["fidelity_bounded"]
        ng = entry["arms"]["trained"]["a0_ngram"]
        ng_s = f"ngram={ng:.2f}" if ng is not None else "ngram=skip"
        print(f"  scored {tid}  train_reward={rec.get('reward'):.4f}  "
              f"trained fidelity={f:.4f}  {ng_s}")

    # Aggregate.
    def agg(name, key):
        vals = [i["arms"][name][key] for i in per_instance if i["arms"][name][key] is not None]
        return sum(vals) / len(vals) if vals else None

    aggregate = {n: {
        "mean_fidelity": agg(n, "fidelity_bounded"),
        "mean_fraction_kept": agg(n, "fraction_kept"),
        "mean_ngram": agg(n, "a0_ngram"),
        "mean_difflib": agg(n, "a0_difflib"),
        "mean_exact_match": agg(n, "a0_exact_match_rate"),
    } for n in arm_names}

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"n_instances": len(per_instance), "a0_samples": a0_samples,
         "a0_temperature": a0_temperature, "aggregate": aggregate,
         "instances": per_instance},
        indent=2, ensure_ascii=False))

    print(f"\n=== aggregate over {len(per_instance)} instances "
          f"(a0: {a0_samples} samples @ T={a0_temperature}) ===")
    print(f"{'arm':>12} {'fidelity':>9} {'kept':>7} {'ngram':>7} {'difflib':>8} {'exact':>7}")
    for n in arm_names:
        a = aggregate[n]
        def fmt(v, pct=False):
            if v is None: return "n/a"
            return f"{v:.1%}" if pct else f"{v:.4f}"
        print(f"{n:>12} {fmt(a['mean_fidelity']):>9} {fmt(a['mean_fraction_kept'], True):>7} "
              f"{fmt(a['mean_ngram']):>7} {fmt(a['mean_difflib']):>8} {fmt(a['mean_exact_match'], True):>7}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    app()
