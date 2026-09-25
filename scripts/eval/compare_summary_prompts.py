#!/usr/bin/env python3
"""
Does the summarizer's output FORMAT change what the compacted context preserves?

The production prompt (tts.data.agent_trajectory.SYSTEM_PROMPT) asks for fixed
sections — USER_CONTEXT / COMPLETED / PENDING / CODE_STATE / TESTS / CHANGES /
DEPS / VERSION_CONTROL_STATUS. Every one of those is a slot for *state*. None is
a slot for a corrected belief, so a summarizer filling the template faithfully
has nowhere to put "I previously misread the test; the expectation itself was
wrong" — which in astropy-13398 was the turn the whole solve depended on.

This scores that. Pick one real (context, next-action) pair, compact the context
under several summarizer prompts, and measure how well each compacted context
predicts the action the agent actually took next. Truncation and env-masking run
alongside as reference points that need no summarizer at all.

    uv run scripts/eval/compare_summary_prompts.py \\
        --traj outputs/swe-bench__Qwen3.6-35B-A3B__full/astropy__astropy-13398/astropy__astropy-13398.traj.json \\
        --turn 57 \\
        --summarizer-api-base http://babel-u9-20:8001/v1 \\
        --summarizer-model hosted_vllm/Qwen/Qwen3.5-9B \\
        --scoring-base-url http://localhost:9999/v1 \\
        --scoring-model litellm_proxy/Qwen/Qwen3.6-35B-A3B \\
        --scoring-tokenizer Qwen/Qwen3.6-35B-A3B

--turn is the index of the assistant message to predict; it is y. Default 57 is
astropy-13398's self-correction turn ("my test expectations were wrong").
"""

from __future__ import annotations

import json
from pathlib import Path

import litellm
import typer

from minisweagent.models.utils.actions_toolcall import BASH_TOOL
from tts.data.agent_trajectory import SYSTEM_PROMPT, messages_to_steps
from tts.reward.distortion_reward import distortion_reward_messages, precompute_x_context
from tts.summarization.mask_based import build_maskenv_scoring_messages
from tts.summarization.model_based import build_z_scoring_messages, format_trajectory_text
from tts.summarization.truncation_based import build_truncation_scoring_messages

app = typer.Typer(add_completion=False)


# The production prompt is "sectioned"; the rest relax it by degrees, so any
# difference is attributable to the instruction and not to a different model.
PROMPTS: dict[str, str] = {
    "sectioned": SYSTEM_PROMPT,
    "free": (
        "You are maintaining a state summary for an interactive coding agent.\n"
        "You will be given a task description followed by a sequence of agent "
        "actions and observations. The agent has not yet finished.\n\n"
        "Write a concise summary of what has happened so far, in whatever form "
        "best serves the agent's next decision. There is no required format.\n"
        "Respond with plain text only — do not call any tools or emit tool call syntax."
    ),
    "free_reasoning": (
        "You are maintaining a state summary for an interactive coding agent.\n"
        "You will be given a task description followed by a sequence of agent "
        "actions and observations. The agent has not yet finished.\n\n"
        "Write a concise summary of what has happened so far, in whatever form "
        "best serves the agent's next decision. There is no required format.\n"
        "Include what the agent has come to UNDERSTAND, not only what it did: "
        "hypotheses it has ruled out, assumptions it has corrected, and why the "
        "current approach is the one it is taking. A fact the agent had to work "
        "to learn is worth more than a list of commands it ran.\n"
        "Respond with plain text only — do not call any tools or emit tool call syntax."
    ),
}


def load_instance(instance_id: str, dataset: str, split: str) -> dict:
    """The SWE-bench row for one instance — needed to build its container."""
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split)
    for row in ds:
        if row["instance_id"] == instance_id:
            return row
    raise typer.BadParameter(f"{instance_id} not in {dataset}[{split}]")


def rollout(messages: list[dict], n_steps: int, model, env=None) -> list[dict]:
    """
    Continue the agent from a compacted context.

    The point of compaction is what the agent does NEXT, which a one-shot
    fidelity score only proxies. With an env the actions actually run, so the
    agent sees real observations and can recover (or loop) exactly as it would
    in the eval. Without one, only a single step is meaningful: there are no
    tool results to condition step 2 on.
    """
    from tts.agent.step import step_once

    convo = [dict(m) for m in messages]
    steps = []
    for i in range(n_steps):
        try:
            r = step_once(convo, model=model, env=env, execute=env is not None)
        except Exception as exc:
            steps.append({"step": i, "error": f"{type(exc).__name__}: {exc}"})
            break
        action = (r.action or {}).get("command", "")
        steps.append({
            "step": i,
            "reasoning": r.reasoning,
            "action": action,
            "observation": (r.observations[0].get("content") if r.observations else None),
        })
        if env is None:
            break  # no observation to continue from
        convo.append(r.message)
        convo.extend(r.observations)
    return steps


def summarize(messages: list[dict], system_prompt: str, model: str, api_base: str,
              max_tokens: int) -> str:
    """Generate z from `messages` under an arbitrary system prompt (greedy)."""
    resp = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": format_trajectory_text(messages_to_steps(messages))},
        ],
        api_base=api_base,
        api_key="EMPTY",
        temperature=0.0,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return resp.choices[0].message.content or ""


@app.command(help=__doc__)
def main(
    traj: Path = typer.Option(..., "--traj", help="A *.traj.json from the `full` arm"),
    turn: int = typer.Option(57, "--turn", help="Index of the assistant message to predict (y)"),
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
    rollout_steps: int = typer.Option(
        0, "--rollout",
        help="Continue each arm this many steps after compacting (0 = fidelity only). "
             "Without --rollout-env only 1 step is meaningful.",
    ),
    rollout_env: bool = typer.Option(
        False, "--rollout-env/--no-rollout-env",
        help="Execute the rolled-out actions in the instance's container, so the agent "
             "sees real observations. Needs the SWE-bench dataset and singularity/docker.",
    ),
    config_spec: list[str] = typer.Option(
        ["swebench.yaml"], "-c", "--config",
        help="Same config chain as eval_swebench.sh, e.g. -c swebench.yaml "
             "-c model.model_kwargs.api_base=http://localhost:1235/v1 "
             "-c model.model_kwargs.temperature=0.6. The rollout deliberator is built "
             "from this, so it samples exactly as the eval does.",
    ),
    deliberator_model: str = typer.Option(
        "", "--deliberator-model",
        help="Overrides model.model_name from --config for the rollout.",
    ),
    environment_class: str = typer.Option(
        "singularity", "--environment-class",
        help="docker | singularity. mini-swe's builtin swebench.yaml says docker, which "
             "this cluster does not have; singularity also gets the --no-mount hostfs fix.",
    ),
    dataset: str = typer.Option("SWE-bench/SWE-bench_Verified", "--dataset"),
    split: str = typer.Option("test", "--split"),
    data_source: str = typer.Option("swe-bench", "--data-source"),
    out: Path = typer.Option(Path("outputs/summary_prompt_comparison.json"), "-o", "--out"),
) -> None:
    from transformers import AutoTokenizer

    if rollout_env and not rollout_steps:
        raise typer.BadParameter(
            "--rollout-env has no effect without --rollout N (it only chooses HOW to "
            "roll out, not whether to). Add e.g. --rollout 5."
        )

    tok = AutoTokenizer.from_pretrained(scoring_tokenizer)
    msgs = [m for m in json.loads(traj.read_text())["messages"] if m.get("role") != "exit"]
    if msgs[turn].get("role") != "assistant":
        raise typer.BadParameter(f"--turn {turn} is a {msgs[turn].get('role')!r}, not an assistant turn")

    # Kept in wire format (tool_call arguments as JSON strings). The scoring
    # paths call to_template_tool_calls themselves, but the rollout sends these
    # same messages back to the chat API, which rejects a dict there. Converting
    # up front broke the rollout; converting is the renderer's job, not ours.
    context = msgs[:turn]
    next_action = msgs[turn]
    print(f"instance   : {traj.parent.name}")
    print(f"context    : {len(context)} messages")
    print(f"y (turn {turn}): {len(next_action.get('reasoning_content') or '')} chars reasoning "
          f"+ {len(next_action.get('tool_calls') or [])} tool call(s)\n")

    # x is fixed across every arm, so score it once.
    x_ctx = precompute_x_context(
        partial_messages=context, next_action=next_action, model=scoring_model,
        api_base=scoring_base_url, tokenizer=tok, tools=[BASH_TOOL],
    )
    if x_ctx is None:
        raise typer.Exit("x-context scoring failed; is the scoring server up?")
    print(f"x = {x_ctx.x_logprobs.n_x_ctx} tok | y = {x_ctx.x_logprobs.n_completion} tok "
          f"| reconstruction verified: {x_ctx.generation.verified}\n")

    arms: list[tuple[str, list[dict], str]] = [
        ("truncation", build_truncation_scoring_messages(context, keep_first, keep_last_turns), ""),
        ("mask", build_maskenv_scoring_messages(context, keep_first, keep_last_turns), ""),
    ]
    for name, prompt in PROMPTS.items():
        z = summarize(context, prompt, summarizer_model, summarizer_api_base, summarizer_max_tokens)
        arms.append((
            f"summary:{name}",
            build_z_scoring_messages(z, context, max_size=max_size, keep_first=keep_first),
            z,
        ))

    env = deliberator = None
    if rollout_steps:
        from minisweagent.config import get_config_from_spec
        from minisweagent.models import get_model
        from minisweagent.utils.serialize import UNSET, recursive_merge

        # swebench.yaml pins environment_class: docker, which this cluster does not
        # have; eval_swebench overrides it the same way via --environment-class.
        config = recursive_merge(
            *[get_config_from_spec(c) for c in config_spec],
            {
                "environment": {"environment_class": environment_class or UNSET},
                "model": {"model_name": deliberator_model or UNSET},
            },
        )
        deliberator = get_model(config=config.get("model", {}))
        print(f"[rollout] deliberator={config.get('model', {}).get('model_name')} "
              f"env={config.get('environment', {}).get('environment_class')}")
        if rollout_env:
            from tts.utils.mini_swe import get_sb_environment

            env = get_sb_environment(
                config, load_instance(traj.parent.name, dataset, split), data_source
            )
            print(f"[rollout] container up for {traj.parent.name}")
        elif rollout_steps > 1:
            print("[rollout] no env: capping at 1 step (nothing to condition step 2 on)")
        print()

    results = {}
    print(f"{'arm':>22} {'z_ctx':>7} {'kept':>6} {'|z| chars':>10} {'fidelity':>10}")
    print("-" * 60)
    for name, z_messages, z in arms:
        r = distortion_reward_messages(
            x_ctx=x_ctx, z_messages=z_messages, summary="",  # compare fidelity, not penalty
            model=scoring_model, api_base=scoring_base_url, tokenizer=tok,
            beta=beta, tools=[BASH_TOOL],
        )
        fid = r.get("fidelity_bounded")
        kept = r["n_z_ctx_tokens"] / x_ctx.x_logprobs.n_x_ctx if r.get("n_z_ctx_tokens") else float("nan")
        print(f"{name:>22} {r.get('n_z_ctx_tokens', 0):>7} {kept:>5.1%} {len(z):>10} "
              f"{fid if fid is not None else float('nan'):>10.4f}")
        results[name] = {"fidelity_bounded": fid, "n_z_ctx_tokens": r.get("n_z_ctx_tokens"),
                         "fraction_kept": kept, "summary": z}
        if rollout_steps:
            results[name]["rollout"] = rollout(z_messages, rollout_steps, deliberator, env=env)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "traj": str(traj), "turn": turn, "instance": traj.parent.name,
        "keep_first": keep_first, "keep_last_turns": keep_last_turns,
        "n_x_ctx_tokens": x_ctx.x_logprobs.n_x_ctx,
        "n_y_tokens": x_ctx.x_logprobs.n_completion,
        "y_reasoning": next_action.get("reasoning_content"),
        "y_action": next_action.get("tool_calls"),
        "rollout_steps": rollout_steps,
        "rollout_env": rollout_env,
        "arms": results,
    }, indent=2, ensure_ascii=False))
    if rollout_steps:
        print(f"\n=== what each arm did next (ground truth at turn {turn} below) ===")
        for name in results:
            for st in results[name].get("rollout", []):
                if "error" in st:
                    print(f"  {name:>22} [{st['step']}] ERROR {st['error'][:70]}")
                    continue
                print(f"  {name:>22} [{st['step']}] {' '.join(st['action'].split())[:88]}")
        truth = ""
        for tc in next_action.get("tool_calls") or []:
            a = tc["function"]["arguments"]
            try:
                a = json.loads(a).get("command", a)
            except Exception:
                pass
            truth = " ".join(str(a).split())
        print(f"  {'GROUND TRUTH':>22} [-] {truth[:88]}")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    app()
