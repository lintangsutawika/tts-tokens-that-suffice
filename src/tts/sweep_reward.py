"""
Offline reward sweep over compression trigger points and hXtY tail budgets.

Answers "what does the distortion reward look like as a function of WHERE we
compress, and HOW MUCH verbatim context we keep around the summary?" without
running an agent. It replays already-collected trajectories (the `full` arm of
eval_swebench, which never compressed and so carries the whole uncompressed
history).

The sweep is TURN-BY-TURN, not a handful of coarse thresholds. For each
trajectory we walk every turn whose cumulative context falls inside
[--min-tokens, --max-tokens] and summarize at each one:

    a trajectory reaching 24k might hit 16k at turn 10, so we summarize at
    turn 10, 11, 12, ... up to the last turn under 24k, recording the actual
    context length at each

so every trajectory contributes a curve of (context tokens -> reward) rather
than a single point, and the x-axis is the real context length at that turn.

    x = trajectory prefix up to the split turn   (grows turn by turn)
    z = summary of that prefix                   (one per (traj, turn))
    y = the agent's actual continuation          (held fixed, see --max-continuation-tokens)

Output is one JSONL row per (trajectory, turn, hXtY spec) carrying the bounded
fidelity, the raw fidelity, every penalty component, and the token counts —
enough to plot reward vs. trigger point with one line per hXtY, and enough to
re-derive the reward under different penalty weights without re-running.

Cost note: z, the x-side logprobs, and the copy penalty all depend only on
(trajectory, turn) — not on hXtY, which changes only the z-context assembly. So
the hXtY sweep costs one extra scoring call per spec, not a full re-run:

    N_traj x N_turns_in_window x (1 summarize + 1 x-call + N_spec z-calls)

N_turns_in_window is the expensive dimension — a trajectory spanning 16k to 32k
can contribute 30+ turns. Use --turn-stride to subsample it and --limit to cap
the trajectory count before committing to a full run.

Needs a scoring server (the same one training uses) and a summarizer — either
the tinker server (--mode base/trained) or a vLLM endpoint
(--summarizer-api-base, see scripts/serve/serve_summarizer.sh).

    uv run python -m tts.sweep_reward \\
        --traj-dir outputs/swe-bench__North-Mini-Code-1.0__full \\
        --mode base --summarizer-model Qwen/Qwen3-8B \\
        --scoring-model litellm_proxy/Qwen/Qwen3.6-27B-FP8 \\
        --scoring-base-url http://localhost:8000/v1 \\
        --out sweeps/north-full.jsonl --plot sweeps/north-full.png
"""

from __future__ import annotations

import json
import os
import re
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer
from tqdm import tqdm

from minisweagent.models.utils.actions_toolcall import BASH_TOOL
from tts.data.agent_trajectory import (
    AGENT_SYSTEM_PROMPT,
    AgentTrajectory,
    _turn_token_lens,
    steps_to_messages,
)
from tts.summarization.model_based import build_summarizer, hf_id
from tts.reward.distortion_reward import distortion_reward_messages, precompute_x_context
from tts.summarization.mask_based import build_maskenv_scoring_messages
from tts.summarization.model_based import build_z_scoring_messages

app = typer.Typer(rich_markup_mode="rich", add_completion=False)


# ---------------------------------------------------------------------------
# hXtY specs
# ---------------------------------------------------------------------------

_SPEC_RE = re.compile(r"^(maskenv-)?h(\d+)t(\d+)$")


def parse_spec(spec: str) -> tuple[int, int, int]:
    """
    "h4t3" -> (keep_first=4, keep_last_turns=3, max_size=20).

    build_z_scoring_messages takes max_size and derives the tail as
    (max_size//2 - keep_first) messages = that many //2 turns, so the max_size
    that realizes hXtY is 2*(X + 2*Y). h4t3 -> 20, which is the default.

    A "maskenv-" prefix ("maskenv-h4t3") selects the summarizer-free compaction:
    same head/tail, but the middle keeps agent actions verbatim and elides only
    the tool-result payloads. Signalled downstream by max_size == 0.
    """
    m = _SPEC_RE.match(spec.strip())
    if not m:
        raise typer.BadParameter(f"spec must look like 'h4t3' or 'maskenv-h4t3', got {spec!r}")
    keep_first, keep_last_turns = int(m.group(2)), int(m.group(3))
    if m.group(1):
        return keep_first, keep_last_turns, 0
    return keep_first, keep_last_turns, 2 * (keep_first + 2 * keep_last_turns)


# ---------------------------------------------------------------------------
# Trajectory loading and turn enumeration
# ---------------------------------------------------------------------------


def load_tokenizer(model_id: str):
    """
    get_tokenizer, falling back to PreTrainedTokenizerFast for newer repos.

    Some repos (CohereLabs/North-Mini-Code-1.0) declare
    tokenizer_class="TokenizersBackend", which transformers 4.57 cannot resolve —
    AutoTokenizer raises "Tokenizer class TokenizersBackend does not exist".
    That name is just the fast-tokenizer backend under a newer alias, so loading
    the repo's tokenizer.json through PreTrainedTokenizerFast is equivalent; the
    chat template comes from chat_template.jinja either way.
    """
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    repo_id = hf_id(model_id)
    try:
        return get_tokenizer(repo_id)
    except ValueError as exc:
        if "does not exist or is not currently imported" not in str(exc):
            raise
        from transformers import PreTrainedTokenizerFast

        print(f"[tokenizer] {repo_id}: {exc}; falling back to PreTrainedTokenizerFast")
        return PreTrainedTokenizerFast.from_pretrained(repo_id)


def load_traj_dir(traj_dir: str | Path) -> list[AgentTrajectory]:
    """
    Load eval_swebench output: traj_dir/{instance_id}/{instance_id}.traj.json.

    Each file holds the raw OpenAI-style `messages` list, which is the same
    shape collect_trajectories.py writes, so from_collect_dict parses it as-is.
    Prefer the `full` arm: any compressed arm has already replaced part of its
    own history with a summary, which is not the x this sweep wants to measure.
    """
    files = sorted(Path(traj_dir).glob("*/*.traj.json"))
    if not files:
        raise typer.BadParameter(f"no */*.traj.json under {traj_dir}")
    trajectories = []
    for f in tqdm(files, desc=f"loading {Path(traj_dir).name}", unit="traj"):
        with open(f) as fh:
            d = json.load(fh)
        traj = AgentTrajectory.from_collect_dict(d)
        if not traj.trajectory_id:
            traj.trajectory_id = d.get("instance_id") or f.parent.name
        if traj.steps:
            trajectories.append(traj)
    return trajectories


def enumerate_split_turns(
    traj: AgentTrajectory,
    tokenizer,
    min_tokens: int,
    max_tokens: int,
    min_prefix: int,
    min_suffix: int,
    stride: int = 1,
) -> list[tuple[int, int]]:
    """
    Every turn index whose cumulative prefix lands in [min_tokens, max_tokens].

    Returns [(k_turns, context_tokens), ...]. Turn lengths come from one batched
    tokenizer call over the whole trajectory, so walking 30 turns costs the same
    tokenization as testing a single threshold did.

    Turns are measured the way threshold_split measures them (the summarizer-side
    EVENT rendering), so the context lengths here are directly comparable to
    split_at_tokens in training rather than to the agent's own context counter.

    A turn is skipped when it leaves fewer than min_prefix turns before it or
    min_suffix turns after it — those cannot form a valid (x, y) pair.
    """
    turn_lens = _turn_token_lens(traj.steps, tokenizer)
    n = len(turn_lens)
    out: list[tuple[int, int]] = []
    total = 0
    for k in range(1, n + 1):
        total += turn_lens[k - 1]
        if total < min_tokens:
            continue
        if total > max_tokens:
            break
        if k < min_prefix or (n - k) < min_suffix:
            continue
        out.append((k, total))
    return out[::stride] if stride > 1 else out


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def sweep_one(
    traj: AgentTrajectory,
    k_turns: int,
    context_tokens: int,
    specs: list[tuple[str, int, int, int]],
    summarizer,
    scoring_tokenizer,
    split_tokenizer,
    scoring_model: str,
    scoring_base_url: str,
    max_continuation_tokens: int,
    beta: float,
    lambda_len: float,
    lambda_copy: float,
    copy_threshold: float,
    marker_penalty: float,
) -> list[dict]:
    """One (trajectory, split turn) unit: split, summarize once, score per spec."""
    # The original agent system prompt lives on the unsplit trajectory; with_prefix
    # does not carry it onto the split copy, so capture it before splitting.
    sys_prompt = traj.agent_system_prompt or AGENT_SYSTEM_PROMPT

    base_row = {
        "trajectory_id": traj.trajectory_id,
        "n_prefix_turns": k_turns,
        "context_tokens": context_tokens,
    }

    split = traj.with_prefix(k_turns * 2)

    # y is the agent's next action — the single assistant message the model would
    # produce from this context (reasoning_content + tool_calls), nothing else.
    # A token cap is not needed: the boundary is semantic, so every unit scores
    # exactly one decision and units stay comparable across trigger points.
    next_action = next(
        (s for s in split.continuation if s.role == "assistant"), None
    )
    if next_action is None:
        return [{**base_row, "status": "no_next_action"}]
    base_row["n_continuation_turns"] = 1

    partial_messages = steps_to_messages(split.steps, split.task, system_prompt=sys_prompt)
    next_action_message = steps_to_messages(
        [next_action], split.task, system_prompt=sys_prompt
    )[2]

    # maskenv specs need no summary; skip the call entirely when none do.
    needs_summary = any(max_size != 0 for _, _, _, max_size in specs)
    summary = ""
    if needs_summary:
        try:
            summary = summarizer.summarize(split.steps)
        except Exception as exc:
            return [{**base_row, "status": "summarize_failed", "error": str(exc)}]

    x_ctx = precompute_x_context(
        partial_messages=partial_messages,
        next_action=next_action_message,
        model=scoring_model,
        api_base=scoring_base_url,
        tokenizer=scoring_tokenizer,
        tools=[BASH_TOOL],
    )
    if x_ctx is None:
        return [{**base_row, "status": "x_precompute_failed"}]

    base_row.update({
        "summary": summary,
        "n_y_tokens": x_ctx.x_logprobs.n_completion,
        "n_x_ctx_tokens": x_ctx.x_logprobs.n_x_ctx,
        "y_verified": x_ctx.generation.verified if x_ctx.generation else None,
    })

    rows = []
    for spec, keep_first, keep_last_turns, max_size in specs:
        if max_size == 0:  # maskenv: compaction by deleting env text, no summary
            z_messages = build_maskenv_scoring_messages(
                partial_messages, keep_first=keep_first, keep_last_turns=keep_last_turns
            )
            spec_summary = ""  # nothing generated -> no anti-copy penalty
        else:
            z_messages = build_z_scoring_messages(
                summary, partial_messages, max_size=max_size, keep_first=keep_first
            )
            spec_summary = summary

        result = distortion_reward_messages(
            x_ctx=x_ctx,
            z_messages=z_messages,
            summary=spec_summary,
            model=scoring_model,
            api_base=scoring_base_url,
            tokenizer=scoring_tokenizer,
            lambda_len=lambda_len,
            lambda_copy=lambda_copy,
            copy_threshold=copy_threshold,
            marker_penalty=marker_penalty,
            beta=beta,
            tools=[BASH_TOOL],
        )
        row = {
            **base_row,
            "spec": spec,
            "keep_first": keep_first,
            "keep_last_turns": keep_last_turns,
            "max_size": max_size,
            # What head+tail cost verbatim at this spec: the summary only gets
            # (compaction budget - this) tokens, so a spec can be unusable at a
            # trigger point even when its reward looks fine.
            "compaction_tokens": split.compaction_tokens(
                scoring_tokenizer, keep_first=keep_first, keep_last_turns=keep_last_turns
            ),
        }
        if result.get("reward") is None:
            rows.append({**row, "status": "score_failed", "error": result.get("error")})
            continue
        rows.append({**row, "status": "ok", **{k: v for k, v in result.items() if k != "error"}})
    return rows


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def load_rows(patterns: str) -> list[dict]:
    """Load sweep rows from a comma-separated list of paths or globs (shard merge)."""
    import glob as _glob

    rows: list[dict] = []
    files: list[str] = []
    for pat in patterns.split(","):
        files.extend(sorted(_glob.glob(pat.strip())))
    if not files:
        raise typer.BadParameter(f"no files match {patterns!r}")
    for path in files:
        with open(path) as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    # Shards are disjoint, but an earlier unsharded run can cover the same units.
    # Key on (trajectory, turn, spec) and keep the last row read.
    deduped = {
        (r.get("trajectory_id"), r.get("n_prefix_turns"), r.get("spec")): r for r in rows
    }
    n_dup = len(rows) - len(deduped)
    print(
        f"loaded {len(rows)} rows from {len(files)} file(s)"
        + (f" ({n_dup} duplicate rows dropped)" if n_dup else "")
    )
    return list(deduped.values())


def report(rows: list[dict], bin_tokens: int) -> None:
    """Binned aggregate table over the scored rows."""
    ok = [r for r in rows if r.get("status") == "ok"]
    statuses = {}
    for r in rows:
        statuses[r.get("status")] = statuses.get(r.get("status"), 0) + 1
    print(f"\nstatus: {statuses}")
    if not ok:
        return
    n_traj = len({r["trajectory_id"] for r in ok})
    n_units = len({(r["trajectory_id"], r["n_prefix_turns"]) for r in ok})
    print(f"scored {len(ok)} rows | {n_traj} trajectories | {n_units} units")

    def mean_of(sel, key):
        """Mean over rows that carry `key` — strategies log different columns."""
        v = [r[key] for r in sel if r.get(key) is not None]
        return statistics.fmean(v) if v else float("nan")

    bins = sorted({(r["context_tokens"] // bin_tokens) * bin_tokens for r in ok})
    print(
        f"\n{'spec':>15} {'ctx bin':>9} {'n':>5} {'fid_bnd':>9} {'reward':>9} "
        f"{'z_ctx':>7} {'kept':>6}"
    )
    for spec in sorted({r["spec"] for r in ok}):
        for b in bins:
            sel = [
                r for r in ok
                if r["spec"] == spec and (r["context_tokens"] // bin_tokens) * bin_tokens == b
            ]
            if not sel:
                continue
            kept = [
                r["n_z_ctx_tokens"] / r["n_x_ctx_tokens"]
                for r in sel
                if r.get("n_z_ctx_tokens") and r.get("n_x_ctx_tokens")
            ]
            print(
                f"{spec:>15} {b:>9} {len(sel):>5} "
                f"{mean_of(sel, 'fidelity_bounded'):>9.4f} "
                f"{mean_of(sel, 'reward'):>9.4f} "
                f"{mean_of(sel, 'n_z_ctx_tokens'):>7.0f} "
                f"{statistics.fmean(kept) if kept else float('nan'):>5.1%}"
            )


def plot_sweep(rows: list[dict], out_png: str, y_key: str, bin_tokens: int) -> None:
    """
    Three views of a sweep, because no single one answers the question.

      1. y_key vs. trigger point — where compression fires
      2. y_key vs. what the compacted context actually costs — the frontier
      3. paired deltas within a family — the only composition-free comparison

    Panel 1 alone is misleading: strategies sit at very different budgets, so a
    gap there can be "kept more context" rather than "compressed better". Panel 2
    is where that is visible and panel 3 is where it is controlled for.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [r for r in rows if r.get("status") == "ok" and r.get(y_key) is not None]
    if not ok:
        print(f"[plot] no scored rows with {y_key}; skipping")
        return

    specs = sorted({r["spec"] for r in ok})
    # Family is the first thing to read off the chart: summarization arms in one
    # hue ramp, env-masking in another, darker within a family = bigger tail.
    summ = [s for s in specs if not s.startswith("maskenv")]
    mask = [s for s in specs if s.startswith("maskenv")]
    colors = {}
    for group, cmap in ((summ, plt.cm.Blues), (mask, plt.cm.Oranges)):
        for i, s in enumerate(group):
            colors[s] = cmap(0.45 + 0.45 * i / max(1, len(group) - 1))

    def stderr(v):
        return statistics.stdev(v) / len(v) ** 0.5 if len(v) > 1 else 0.0

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))

    # -- 1. vs. trigger point --------------------------------------------------
    ax = axes[0]
    all_means = []
    for spec in specs:
        acc: dict[int, list[float]] = {}
        for r in ok:
            if r["spec"] == spec:
                b = (r["context_tokens"] // bin_tokens) * bin_tokens + bin_tokens // 2
                acc.setdefault(b, []).append(r[y_key])
        xs = sorted(acc)
        ys = [statistics.fmean(acc[b]) for b in xs]
        all_means += ys
        ax.errorbar(xs, ys, yerr=[stderr(acc[b]) for b in xs], marker="o", capsize=3,
                    color=colors[spec], label=spec, lw=2)
    ax.axhline(0.0, color="0.6", lw=0.8, ls="--")
    # y_key may be bounded to (-1, 1) but in practice occupies a sliver of it;
    # clamping to the theoretical range hides every difference worth seeing.
    lo, hi = min(all_means), max(all_means)
    pad = max((hi - lo) * 0.25, 0.005)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("context tokens at compression trigger")
    ax.set_ylabel(y_key)
    ax.set_title(f"{y_key} vs. where we compress ({bin_tokens} token bins)")
    ax.legend(fontsize=8, title="head/tail kept", title_fontsize=8)
    ax.grid(alpha=0.3)

    # -- 2. the frontier -------------------------------------------------------
    ax = axes[1]
    have_budget = [r for r in ok if r.get("n_z_ctx_tokens") and r.get("n_x_ctx_tokens")]
    if have_budget:
        for spec in specs:
            g = [r for r in have_budget if r["spec"] == spec]
            if not g:
                continue
            fx = [r["n_z_ctx_tokens"] / r["n_x_ctx_tokens"] for r in g]
            fy = [r[y_key] for r in g]
            ax.scatter(fx, fy, s=5, alpha=0.07, color=colors[spec], linewidths=0)
            ax.errorbar([statistics.fmean(fx)], [statistics.fmean(fy)],
                        yerr=[stderr(fy)], xerr=[stderr(fx)], marker="D", ms=10,
                        capsize=4, color=colors[spec], markeredgecolor="black",
                        zorder=5, label=f"{spec} ({statistics.fmean(fx):.0%} of x)")
        ax.axhline(0.0, color="0.6", lw=0.8, ls="--")
        ax.set_ylim(lo - pad * 4, max(hi + pad * 4, 0.02))
        ax.set_xlabel("compacted context as fraction of x")
        ax.set_ylabel(y_key)
        ax.set_title("the frontier: fidelity vs. budget\n(compare arms only at equal x)")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.3)
    else:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no n_z_ctx_tokens in rows", ha="center", va="center")

    # -- 3. paired deltas, same family only ------------------------------------
    ax = axes[2]
    byunit: dict[tuple, dict[str, dict]] = {}
    for r in ok:
        byunit.setdefault((r["trajectory_id"], r["n_prefix_turns"]), {})[r["spec"]] = r
    bars, labels, bcolors = [], [], []
    for family in (summ, mask):
        for i in range(len(family)):
            for j in range(i + 1, len(family)):
                a, b = family[i], family[j]
                d = [v[b][y_key] - v[a][y_key] for v in byunit.values() if a in v and b in v]
                if len(d) < 2:
                    continue
                dz = [
                    v[b].get("n_z_ctx_tokens", 0) - v[a].get("n_z_ctx_tokens", 0)
                    for v in byunit.values() if a in v and b in v
                ]
                bars.append((statistics.fmean(d), stderr(d), statistics.fmean(dz), len(d)))
                labels.append(f"{b} − {a}")
                bcolors.append(colors[b])
    if bars:
        ys = range(len(bars))
        ax.barh(list(ys), [x[0] for x in bars], xerr=[x[1] for x in bars],
                color=bcolors, capsize=3, height=0.62, edgecolor="white")
        for k, (m, e, dz, n) in enumerate(bars):
            ax.text(m + (0.0015 if m >= 0 else -0.0015), k, f"{dz:+,.0f} tok  n={n}",
                    va="center", ha="left" if m >= 0 else "right", fontsize=7)
        ax.set_yticks(list(ys))
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color="0.4", lw=0.8)
        ax.margins(x=0.3)
    ax.set_xlabel(f"Δ {y_key} (paired, same trajectory+turn)")
    ax.set_title("within-family paired deltas\n(cross-family omitted: budgets differ)")
    ax.grid(alpha=0.3, axis="x")

    n_traj = len({r["trajectory_id"] for r in ok})
    med_y = sorted(r["n_y_tokens"] for r in ok if r.get("n_y_tokens"))
    suffix = f" | y median {med_y[len(med_y) // 2]} tok" if med_y else ""
    fig.suptitle(f"n={len(ok)} rows | {n_traj} trajectories | {len(byunit)} units{suffix}")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    print(f"[plot] wrote {out_png}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command(help=__doc__)
def main(
    traj_dir: str = typer.Option(..., "--traj-dir", help="eval_swebench output dir (use the `full` arm)"),
    out: str = typer.Option(..., "-o", "--out", help="Output JSONL path (appended; see --resume)"),
    limit: int = typer.Option(0, "--limit", help="Only sweep the first N trajectories (0 = all)"),
    shard: str = typer.Option("", "--shard", help="'i/N' — sweep only trajectory shard i of N (round-robin). Give each shard its own --out, then merge with --plot-from"),
    min_tokens: int = typer.Option(16384, "--min-tokens", help="Sweep turns from the first one reaching this context length"),
    max_tokens: int = typer.Option(32768, "--max-tokens", help="...up to the last turn at or below this length"),
    turn_stride: int = typer.Option(1, "--turn-stride", help="Take every Nth turn in the window (1 = every turn)"),
    specs: str = typer.Option("h4t3", "--specs", help="Comma-separated hXtY tail budgets, e.g. h4t3,h4t5,h8t3"),
    concurrency: int = typer.Option(8, "-j", "--concurrency", help="Parallel (trajectory, turn) units"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Skip (trajectory, turn) pairs already in --out"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Enumerate turns and report the unit/call count, then exit"),
    # Summarizer
    mode: str = typer.Option("base", "--mode", help="base | trained"),
    checkpoint: str = typer.Option("", "--checkpoint", help="tinker:// state path for --mode trained"),
    summarizer_model: str = typer.Option("Qwen/Qwen3-8B", "--summarizer-model", help="HF id for tinker, or litellm string for --summarizer-api-base"),
    summarizer_api_base: str = typer.Option("", "--summarizer-api-base", help="vLLM endpoint; unset = sample via the tinker server"),
    summarizer_tokenizer: str = typer.Option("", "--summarizer-tokenizer", help="Tokenizer for the turn/context token count (default: --summarizer-model)"),
    tinker_base_url: str = typer.Option("", "--tinker-base-url", help="Tinker server URL (default: TINKER_BASE_URL)"),
    renderer: str = typer.Option("qwen3_disable_thinking", "--renderer", help="Renderer name for the tinker summarizer"),
    max_tokens_gen: int = typer.Option(512, "--max-gen-tokens", help="Summary generation budget"),
    # Scoring
    scoring_model: str = typer.Option("litellm_proxy/Qwen/Qwen3.6-27B-FP8", "--scoring-model"),
    scoring_base_url: str = typer.Option("http://localhost:8080/v1", "--scoring-base-url", help="Scoring server (scripts/serve/serve_local.sh serves :8080)"),
    scoring_tokenizer: str = typer.Option("", "--scoring-tokenizer", help="Tokenizer for logprob offsets (default: --scoring-model; must match the scoring server)"),
    # Split
    min_prefix: int = typer.Option(3, "--min-prefix", help="Min turns before the split"),
    min_suffix: int = typer.Option(3, "--min-suffix", help="Min turns after the split"),
    max_continuation_tokens: int = typer.Option(4096, "--max-continuation-tokens", help="Fixed cap on y, so y stays comparable as the x-axis grows. 0 = no cap"),
    # Reward knobs (defaults mirror scripts/train/train_summarizer.sh)
    beta: float = typer.Option(1.0, "--beta", help="tanh temperature; >0 bounds the fidelity base to (-1, 1)"),
    lambda_len: float = typer.Option(0.5, "--lambda-len"),
    lambda_copy: float = typer.Option(1.0, "--lambda-copy"),
    copy_threshold: float = typer.Option(0.3, "--copy-threshold"),
    marker_penalty: float = typer.Option(1.0, "--marker-penalty"),
    # Plot
    plot: str = typer.Option("", "--plot", help="Write a PNG of --plot-y vs. context tokens, one line per spec"),
    plot_y: str = typer.Option("fidelity_bounded", "--plot-y", help="fidelity_bounded | reward | fidelity"),
    plot_bin: int = typer.Option(2048, "--plot-bin", help="Token bin width for the plotted mean"),
    plot_from: str = typer.Option("", "--plot-from", help="Comma-separated paths/globs to report and plot from (default: --out). Use to merge shards"),
    plot_only: bool = typer.Option(False, "--plot-only", help="Report and plot from existing JSONL without sweeping"),
) -> None:
    # Reporting/plotting reads only the JSONL, so it needs no tokenizer, no
    # trajectories, and no servers — safe to run against a sweep still in flight.
    if plot_only:
        rows = load_rows(plot_from or out)
        report(rows, plot_bin)
        if plot:
            plot_sweep(rows, plot, plot_y, plot_bin)
        return

    spec_list = [(s.strip(), *parse_spec(s)) for s in specs.split(",") if s.strip()]

    split_tok = load_tokenizer(summarizer_tokenizer or summarizer_model)
    score_tok = load_tokenizer(scoring_tokenizer or scoring_model)

    trajectories = load_traj_dir(traj_dir)
    if limit > 0:
        trajectories = trajectories[:limit]
    if shard:
        try:
            idx, n_shards = (int(v) for v in shard.split("/"))
        except ValueError:
            raise typer.BadParameter(f"--shard must look like '0/4', got {shard!r}")
        if not 0 <= idx < n_shards:
            raise typer.BadParameter(f"--shard index {idx} out of range for {n_shards} shards")
        # Round-robin, not contiguous blocks: trajectory cost varies a lot with
        # length, and neighbouring instances in the sorted listing are correlated
        # (same repo), so striping keeps the shards evenly sized in work.
        trajectories = trajectories[idx::n_shards]
        print(f"[shard] {idx}/{n_shards}: {len(trajectories)} trajectories")

    # Enumerate every (trajectory, turn) unit up front so the call count is known
    # before any server work starts — the turn dimension is what makes this
    # expensive, and it is easy to underestimate.
    units: list[tuple[AgentTrajectory, int, int]] = []
    n_curves = 0
    for t in tqdm(trajectories, desc="enumerating turns", unit="traj"):
        turns = enumerate_split_turns(
            t, split_tok, min_tokens, max_tokens, min_prefix, min_suffix, turn_stride
        )
        if turns:
            n_curves += 1
        units.extend((t, k, ctx) for k, ctx in turns)

    if not units:
        print(f"no trajectory has a turn in [{min_tokens}, {max_tokens}]; nothing to sweep")
        return
    print(
        f"{n_curves}/{len(trajectories)} trajectories reach [{min_tokens}, {max_tokens}]; "
        f"{len(units)} (trajectory, turn) units, {len(units) / n_curves:.1f} turns/curve avg"
    )
    print(
        f"cost: {len(units)} summarize + {len(units)} x-calls + "
        f"{len(units) * len(spec_list)} z-calls"
    )
    if dry_run:
        return

    # Resume: a (trajectory, turn) unit is atomic — all its spec rows are written
    # together — so skipping on that key never loses a partial unit.
    out_path = Path(out)
    if resume and out_path.exists():
        done: set[tuple[str, int]] = set()
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["trajectory_id"], r["n_prefix_turns"]))
        before = len(units)
        units = [u for u in units if (u[0].trajectory_id, u[1]) not in done]
        print(f"[resume] skipping {before - len(units)} units already in {out}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()

    summarizer = build_summarizer(
        tinker_base_url=tinker_base_url or os.getenv("TINKER_BASE_URL"),
        checkpoint=checkpoint,
        summarizer_model=summarizer_model,
        renderer_name=renderer,
        max_tokens=max_tokens_gen,
        summarizer_api_base=summarizer_api_base,
    )
    if summarizer is None:
        raise typer.BadParameter(f"--mode {mode!r} produces no summarizer; use base or trained")

    def _run(unit) -> list[dict]:
        traj, k, ctx = unit
        return sweep_one(
            traj=traj,
            k_turns=k,
            context_tokens=ctx,
            specs=spec_list,
            summarizer=summarizer,
            scoring_tokenizer=score_tok,
            split_tokenizer=split_tok,
            scoring_model=scoring_model,
            scoring_base_url=scoring_base_url,
            max_continuation_tokens=max_continuation_tokens,
            beta=beta,
            lambda_len=lambda_len,
            lambda_copy=lambda_copy,
            copy_threshold=copy_threshold,
            marker_penalty=marker_penalty,
        )

    with open(out_path, "a") as fh, ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(_run, u): u for u in units}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="sweep", unit="unit"):
            traj, k, ctx = futures[fut]
            try:
                rows = fut.result()
            except Exception as exc:
                rows = [{
                    "trajectory_id": traj.trajectory_id,
                    "n_prefix_turns": k,
                    "context_tokens": ctx,
                    "status": "unit_failed",
                    "error": str(exc),
                }]
            with write_lock:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
                fh.flush()

    # Reload for the summary and plot: --plot-from when merging shards, else the
    # shard's own file (which also picks up rows from prior resumed runs).
    all_rows = load_rows(plot_from or out)
    report(all_rows, plot_bin)

    if plot:
        plot_sweep(all_rows, plot, plot_y, plot_bin)


if __name__ == "__main__":
    app()
