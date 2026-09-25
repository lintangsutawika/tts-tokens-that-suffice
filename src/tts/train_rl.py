"""
RL recipe: teach a model to summarize partial coding agent trajectories via
GRPO-style reward centering.

For each trajectory in the batch the model generates `group_size` candidate
summaries. Each candidate is scored by the selected reward; advantages are
computed as reward – group_mean (reward centering). Policy gradients are
applied via the importance-sampling loss.

Two reward functions are available (set via Config.reward_fn):

  "coverage"   (default)
    Fraction of tool names and file names from the full trajectory
    (steps + continuation) that appear in the generated summary.
    Reference-free, cheap, and a useful proxy for distortion.

  "distortion"
    KL-distortion fidelity from "Tokens That Suffice" / Readable Context
    Distillation.  Measures how much predictive information the summary z
    preserves relative to the original steps x, with respect to predicting
    the continuation y:

        r(x, z) = (1/|y|) Σ_t [log p(y_t | y<t, z) − log p(y_t | y<t, x)] − λ·|z|

    x  = partial trajectory steps (the "seen" context)
    z  = generated summary (the compression)
    y  = continuation steps (what the agent does next, stored in trajectory)
    λ  = distortion_lambda length-penalty coefficient

    Forward KL is used (mass-covering): the summary must preserve *all* likely
    next-step distributions, not just the mode.  The fidelity term is
    non-positive; GRPO centering removes the per-trajectory constant
    −H(p(y|x)) so advantage signals reflect only relative group quality.

    Requires the serving backend to support returning logprobs for prompt
    tokens (SamplingParams(prompt_logprobs=1)).  Samples whose scoring call
    fails are dropped from their group (never given a numeric fallback reward).

Variable naming convention (mirrors rl_loop.py):
    _P  Problem dimension  (different trajectories in a batch)
    _G  Group dimension    (multiple sampled summaries per trajectory)
    _T  Token/time dimension
    _D  Datum dimension    (P × G after flattening)

Usage:
    TINKER_API_KEY=tml-dummy uv run -m tts.train_rl \\
        dataset_path=/path/to/partial_trajectories.jsonl

Or via the test script:
    TINKER_API_KEY=tml-dummy uv run tests/train_agent_summarization_rl.py
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed

import chz
import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData
from tqdm import tqdm

from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.git_rev import recipe_user_metadata

from tts.data.agent_trajectory import (
    AgentTrajectory,
    AGENT_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    TrajectoryStep,
    format_trajectory_text,
    load_collect_trajectories,
    load_trajectories,
    steps_to_messages,
)
from minisweagent.models.utils.actions_toolcall import BASH_TOOL

from tts.reward.copy_penalty import copy_penalty
from tts.reward.distortion_reward import (
    XContext,
    distortion_reward_z_multi,
    precompute_x_contexts,
)
from tts.summarization.model_based import build_z_scoring_messages, strip_thinking

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    base_url: str | None = None
    log_path: str = "/tmp/tinker-agent-summarization-rl"
    model_name: str = "Qwen/Qwen3.5-9B"
    dataset_path: str = ""
    batch_size: int = 32       # trajectories per gradient step
    group_size: int = 8        # candidate summaries sampled per trajectory
    learning_rate: float = 4e-5
    # Generation budget for the summarizer (non-thinking renderer, so this is the summary
    # itself). It doubles as a hard anti-copy bound: a 16k-token context cannot be
    # transcribed into 512 tokens, so the ceiling forecloses verbatim copying outright,
    # where the reward penalties only make it unprofitable.
    max_tokens: int = 512
    lora_rank: int = 32
    save_every: int = 20
    ttl_seconds: int | None = 604800
    num_epochs: int = 1
    # Reward function: "coverage" or "distortion"
    reward_fn: str = "coverage"
    # Anti-copy terms for the distortion reward. Fidelity alone peaks at z = x, so with
    # all of these at 0 the policy's optimal move is to transcribe its input verbatim —
    # which is exactly how the first summarizer-rl run collapsed. Keep distortion_lambda
    # > 0 at minimum.
    # λ: length penalty, applied to |z|/|x| in tokens (relative, not absolute word count).
    distortion_lambda: float = 0.0
    # Penalty for verbatim n-gram overlap with x beyond distortion_copy_threshold.
    distortion_lambda_copy: float = 0.0
    # Overlap fraction tolerated before the copy penalty starts (paraphrase reuses some
    # identifiers and paths, so a small overlap is expected and not punished).
    distortion_copy_threshold: float = 0.3
    # Flat penalty when z echoes the <EVENT>/tool-call scaffolding of its input.
    distortion_marker_penalty: float = 0.0
    # Temperature for the bounded per-token tanh contrast. When > 0 the fidelity base
    # becomes (1/|y|) Σ_t tanh(Δ_t/β), Δ_t = log p(y_t|z) − log p(y_t|x): bounded to
    # (-1, 1), 0 at parity, +1 z far better / −1 far worse. Absolute logprobs aren't
    # comparable across prompts — only this paired same-y contrast is — and per-token
    # bounding caps an outlier token from dominating the mean (see bounded_fidelity).
    # β sets how large a per-token log-ratio counts as decisive (~1–2 nats). 0 = raw fidelity.
    distortion_beta: float = 0.0
    # URL and model name of the vLLM scoring server (used only when reward_fn="distortion").
    scoring_base_url: str = "http://localhost:8000/v1"
    scoring_model: str = "Qwen/Qwen3.5-27B"
    # Target message budget for the z-context. Tail = max_size//2 - keep_first messages.
    distortion_max_size: int = 20
    # Messages to keep verbatim from the start of the trajectory in z-context.
    distortion_keep_first: int = 4
    # How many succeeding agent turns to score the summary against. The reward is
    # the mean fidelity over R_0..R_{n-1}, each teacher-forced on the real turns
    # before it (see distortion_reward_z_multi). 1 = the original next-action-only
    # reward. >1 rewards a summary for supporting a run of decisions.
    distortion_n_turns: int = 1
    # Min steps in prefix / suffix when sampling a random split from full trajectories.
    min_split_prefix: int = 3
    min_split_suffix: int = 3
    # Split each trajectory where its context first reaches this many tokens, instead
    # of at a random turn — set it to the eval's compress_at_tokens so the summarizer
    # trains on the context lengths it is actually invoked at. 0 = random split.
    # Trajectories that never reach the threshold are dropped, so raising it shrinks
    # the usable dataset (at 16384 ≈54% of trajectories survive; at 24000 ≈29%).
    split_at_tokens: int = 0
    # Budget for the whole COMPACTED CONTEXT — head(keep_first) + summary + tail
    # (keep_last_turns), i.e. h4t3 plus the summary, not the summary alone. The
    # continuation is capped at the remainder (split_at_tokens - this) so that
    # compaction + continuation stays inside one trigger window.
    compaction_token_budget: int = 9000
    # Head and tail are verbatim and not under the summarizer's control, so a bloated
    # tool output in the tail can eat the whole budget. Drop a trajectory when h4t3
    # leaves less than this much room for the summary to occupy.
    min_summary_tokens: int = 512
    # Renderer name override. Defaults to the model's recommended renderer.
    # Use "qwen3_instruct" to disable thinking for Qwen3/Qwen3.5 instruct models.
    renderer_name: str | None = None
    # W&B logging (disabled by default).
    wandb_project: str | None = None
    wandb_name: str | None = None
    # Sampling temperature for rollout generation.
    sampling_temperature: float = 1.0
    # Minimum within-group reward spread to use a group for training.
    # Groups below this threshold are skipped as low-signal noise.
    # Measured over 941 groups of the lr=1e-4 run: median spread is 0.073, falling to
    # ~0.048 late in training as the policy converges — so 0.1 dropped ~59% of groups
    # overall and ~78% by the last quarter, which is what stalled that run. Keep this
    # below the median spread; advantage_eps already attenuates low-variance groups
    # smoothly, so this hard filter only needs to catch degenerate (spread≈0) groups.
    # Keep it off a reward_snap multiple: snapped spreads land exactly on the
    # threshold and float error (0.09999...) then drops groups that should qualify.
    min_reward_spread: float = 0.02
    # Snap rewards to this grid before computing advantages; differences smaller
    # than this value are treated as ties (advantage 0). Set to 0 to disable.
    reward_snap: float = 0.05
    # GRPO advantage normalization: divide centered rewards by (std + eps).
    # When False, advantages are raw mean-centered rewards.
    normalize_advantage: bool = False
    # Epsilon floor added to the group std before dividing (prevents blow-up on
    # low-variance groups). Only used when normalize_advantage is True.
    advantage_eps: float = 0.1
    # Save sampled summaries and their rewards to log_path/summaries.jsonl.
    save_summaries: bool = True
    # --- Held-out eval (data-controlled learning signal) ---
    # Trajectories reserved from the END of the dataset for a fixed eval set that
    # is never trained on. Their splits are pinned once (eval_seed), so eval/reward
    # moves only when the policy moves. Set 0 to disable eval.
    eval_size: int = 16
    # Run the held-out eval every N batches (also at batch 0 for a baseline).
    eval_every: int = 10
    # Decoding temperature for eval generation. 0.0 = greedy/deterministic so eval
    # reward reflects weight changes, not sampling noise.
    eval_temperature: float = 0.0
    # Seed for pinning eval-set splits (fixed across the whole run).
    eval_seed: int = 12345
    # Diagnostic/overfit mode: eval on the SAME trajectories used for training
    # (fixed splits), instead of holding them out. Lets the eval curve answer
    # "can the setup fit the training set at all?" Use with a tiny dataset.
    eval_on_train: bool = False


# ---------------------------------------------------------------------------
# Coverage reward
# ---------------------------------------------------------------------------

def _extract_entities(steps: list[TrajectoryStep]) -> set[str]:
    """Extract tool names and file basenames from a list of TrajectoryStep."""
    entities: set[str] = set()
    for step in steps:
        if step.name:
            entities.add(step.name.lower())
        for tc in step.tool_calls:
            entities.add(tc.name.lower())
            for v in tc.arguments.values():
                if isinstance(v, str) and ("/" in v or "." in v):
                    entities.add(v.split("/")[-1].lower())
    return entities


def coverage_reward(summary: str, trajectory: AgentTrajectory) -> float:
    """
    Fraction of key entities mentioned in the summary.

    Entities are tool names and file basenames drawn from both the partial
    trajectory (steps) and the continuation — the full entity set represents
    everything the agent interacted with across the complete task.
    """
    entities = _extract_entities(trajectory.steps) | _extract_entities(trajectory.continuation)
    if not entities:
        return 0.5  # no named entities to check — neutral reward
    summary_lower = summary.lower()
    return sum(1 for e in entities if e in summary_lower) / len(entities)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main(config: Config) -> None:
    if not config.dataset_path:
        raise ValueError("dataset_path must be set to a JSONL file of trajectories")

    if config.reward_fn not in ("coverage", "distortion"):
        raise ValueError(
            f"reward_fn must be 'coverage' or 'distortion', got {config.reward_fn!r}"
        )

    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        config=config,
        do_configure_logging_module=True,
    )

    tokenizer = get_tokenizer(config.model_name)
    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info("Using renderer: %s", renderer_name)
    logger.info("Using reward_fn: %s", config.reward_fn)

    logger.info("Loading trajectories from %s", config.dataset_path)
    # Try collect_trajectories format first (full unsplit); fall back to pre-split format.
    try:
        trajectories = load_collect_trajectories(config.dataset_path)
        logger.info("Loaded %d full trajectories (will split per-batch)", len(trajectories))
        split_at_train_time = True
    except (KeyError, TypeError):
        trajectories = load_trajectories(config.dataset_path)
        logger.info("Loaded %d pre-split trajectories", len(trajectories))
        split_at_train_time = False

    # Match the eval-time compression trigger: split each trajectory where its
    # context first reaches split_at_tokens, rather than at a random turn. The
    # split point is deterministic, so we take it once here instead of per-batch;
    # trajectories that never reach the threshold (or leave too little
    # prefix/suffix around it) cannot form a valid pair and are dropped.
    if config.split_at_tokens > 0:
        if not split_at_train_time:
            raise ValueError("split_at_tokens requires full (unsplit) trajectories")
        max_continuation = config.split_at_tokens - config.compaction_token_budget
        if max_continuation <= 0:
            raise ValueError(
                f"compaction_token_budget={config.compaction_token_budget} leaves no room for "
                f"the continuation within split_at_tokens={config.split_at_tokens}"
            )
        before = len(trajectories)
        split = [
            s
            for t in tqdm(trajectories, desc=f"threshold split @{config.split_at_tokens}tok", unit="traj")
            if (
                s := t.threshold_split(
                    tokenizer,
                    split_at_tokens=config.split_at_tokens,
                    min_prefix=config.min_split_prefix,
                    min_suffix=config.min_split_suffix,
                    max_continuation_tokens=max_continuation,
                )
            )
            is not None
        ]
        # h4t3 is verbatim; if it already fills the compaction budget the summary has
        # nowhere to go, so those trajectories cannot be trained on at this budget.
        max_h4t3 = config.compaction_token_budget - config.min_summary_tokens
        keep_last_turns = (
            # Same derivation as build_z_scoring_messages: the tail message budget is
            # max_size//2 - keep_first, and each turn is 2 messages.
            config.distortion_max_size // 2 - config.distortion_keep_first
        ) // 2
        trajectories = [
            s
            for s in tqdm(split, desc=f"compaction filter (h4t{keep_last_turns})", unit="traj")
            if s.compaction_tokens(
                tokenizer,
                keep_first=config.distortion_keep_first,
                keep_last_turns=keep_last_turns,
            )
            <= max_h4t3
        ]
        if not trajectories:
            raise ValueError(
                f"No trajectory reaches split_at_tokens={config.split_at_tokens} with "
                f"min_prefix={config.min_split_prefix}/min_suffix={config.min_split_suffix} "
                f"and h4t3 <= {max_h4t3}"
            )
        logger.info(
            "Threshold split @%d tok (compaction budget %d = h4t3+summary, continuation cap %d): "
            "kept %d/%d (%.0f%%); %d dropped for h4t3 > %d",
            config.split_at_tokens, config.compaction_token_budget, max_continuation,
            len(trajectories), before, 100 * len(trajectories) / before,
            len(split) - len(trajectories), max_h4t3,
        )
        split_at_train_time = False  # already split; do not re-split per batch

    # Reserve a fixed held-out eval set from the END of the dataset (never trained
    # on). Splits are pinned with eval_seed and x-contexts precomputed once, so the
    # eval reward isolates policy improvement from the per-batch data-ordering noise
    # that dominates train reward/mean.
    eval_set: list[tuple[AgentTrajectory, XContext | None]] = []
    if config.eval_size > 0 and config.eval_every > 0:
        eval_rng = random.Random(config.eval_seed)
        if config.eval_on_train:
            # Overfit/diagnostic: eval on the same trajectories we train on
            # (fixed splits), kept in the training pool.
            eval_raw = trajectories[: config.eval_size]
        else:
            if len(trajectories) <= config.eval_size:
                raise ValueError(
                    f"eval_size={config.eval_size} >= dataset size {len(trajectories)}"
                )
            eval_raw = trajectories[-config.eval_size:]
            trajectories = trajectories[: -config.eval_size]
        eval_trajs = [
            t.sample_split(
                eval_rng,
                min_prefix=config.min_split_prefix,
                min_suffix=config.min_split_suffix,
            )
            if split_at_train_time
            else t
            for t in eval_raw
        ]

        def _precompute_eval_x(et):
            if config.reward_fn != "distortion":
                return (et, None)
            sys_prompt = et.agent_system_prompt or AGENT_SYSTEM_PROMPT
            partial = steps_to_messages(et.steps, et.task, system_prompt=sys_prompt)
            cont = steps_to_messages(et.continuation, et.task, system_prompt=sys_prompt)[2:]
            x_ctxs = precompute_x_contexts(
                partial_messages=partial, continuation_messages=cont,
                n_turns=config.distortion_n_turns,
                model=config.scoring_model, api_base=config.scoring_base_url,
                tokenizer=tokenizer, tools=[BASH_TOOL],
            )
            return (et, x_ctxs or None)

        with ThreadPoolExecutor(max_workers=len(eval_trajs)) as ex:
            eval_set = list(ex.map(_precompute_eval_x, eval_trajs))
        logger.info(
            "Reserved %d held-out eval trajectories; %d remain for training",
            len(eval_set), len(trajectories),
        )

    if len(trajectories) < config.batch_size:
        raise ValueError(
            f"Dataset has {len(trajectories)} trajectories but "
            f"batch_size={config.batch_size}. Reduce batch_size or add more data."
        )

    n_batches_per_epoch = len(trajectories) // config.batch_size
    total_batches = n_batches_per_epoch * config.num_epochs

    service_client = tinker.ServiceClient(
        base_url=config.base_url,
        user_metadata=recipe_user_metadata("recipe_agent_summarization_rl"),
    )

    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        training_client = (
            service_client.create_training_client_from_state_with_optimizer(
                resume_info.state_path
            )
        )
        start_batch = resume_info.batch
        logger.info("Resuming from batch %d", start_batch)
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
        start_batch = 0

    sampling_params = types.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.sampling_temperature,
        stop=renderer.get_stop_sequences(),
    )
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8,
        grad_clip_norm=1.0,
    )

    eval_sampling_params = types.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.eval_temperature,
        stop=renderer.get_stop_sequences(),
    )

    def run_eval(sampling_client) -> dict[str, float]:
        """Score the frozen held-out eval set with the current weights (no gradient).

        Reuses the existing training client's sampling snapshot, so it needs no
        second LoRA adapter (the FSDP backend allows only one). Greedy decoding
        makes the result depend only on the weights, isolating policy improvement.

        All generations are submitted up front (sample() returns a future) and the
        per-trajectory scoring runs in a thread pool, so the eval set is processed
        concurrently rather than one trajectory at a time.
        """
        # Submit every generation first; sample() is non-blocking and returns a future.
        sample_futures = []
        for et, _ in eval_set:
            convo = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": format_trajectory_text(et.steps)},
            ]
            sample_futures.append(
                sampling_client.sample(
                    prompt=renderer.build_generation_prompt(convo),
                    num_samples=1,
                    sampling_params=eval_sampling_params,
                )
            )

        def _eval_one(args) -> float | None:
            (et, x_ctxs), future = args
            try:
                result = future.result()
            except Exception as exc:
                logger.warning("Eval sampling failed for %s: %s", et.trajectory_id, exc)
                return None
            parsed_message, _ = renderer.parse_response(result.sequences[0].tokens)
            summary = strip_thinking(renderers.get_text_content(parsed_message))
            if config.reward_fn == "distortion":
                if x_ctxs is None:
                    return None
                partial = steps_to_messages(
                    et.steps, et.task,
                    system_prompt=et.agent_system_prompt or AGENT_SYSTEM_PROMPT,
                )
                r = distortion_reward_z_multi(
                    x_ctxs=x_ctxs, summary=summary, partial_messages=partial,
                    model=config.scoring_model, api_base=config.scoring_base_url,
                    tokenizer=tokenizer, max_size=config.distortion_max_size,
                    keep_first=config.distortion_keep_first,
                    lambda_len=config.distortion_lambda,
                    lambda_copy=config.distortion_lambda_copy,
                    copy_threshold=config.distortion_copy_threshold,
                    marker_penalty=config.distortion_marker_penalty,
                    beta=config.distortion_beta,
                    tools=[BASH_TOOL],
                )
            else:
                r = coverage_reward(summary, et)
            return r if r is not None and math.isfinite(r) else None

        rewards: list[float] = []
        with ThreadPoolExecutor(max_workers=len(eval_set)) as ex:
            for r in ex.map(_eval_one, zip(eval_set, sample_futures)):
                if r is not None:
                    rewards.append(r)
        if not rewards:
            return {}
        return {
            "eval/reward_mean": sum(rewards) / len(rewards),
            "eval/reward_min": min(rewards),
            "eval/reward_max": max(rewards),
            "eval/n": float(len(rewards)),
        }

    global_batch_idx = start_batch
    for epoch in range(config.num_epochs):
        epoch_rng = random.Random(epoch)
        shuffled = list(trajectories)
        epoch_rng.shuffle(shuffled)

        for batch_in_epoch in range(n_batches_per_epoch):
            batch_rng = random.Random(global_batch_idx)
            if global_batch_idx < start_batch:
                global_batch_idx += 1
                continue

            t_start = time.time()
            metrics: dict[str, float] = {
                "progress/batch": global_batch_idx,
                "optim/lr": config.learning_rate,
                "progress/done_frac": (global_batch_idx + 1) / total_batches,
                "progress/epoch": epoch,
            }

            if (
                config.save_every > 0
                and global_batch_idx % config.save_every == 0
                and global_batch_idx > 0
            ):
                checkpoint_utils.save_checkpoint(
                    training_client=training_client,
                    name=f"{global_batch_idx:06d}",
                    log_path=config.log_path,
                    kind="state",
                    loop_state={"batch": global_batch_idx},
                    ttl_seconds=config.ttl_seconds,
                )

            batch_start = batch_in_epoch * config.batch_size
            raw_batch: list[AgentTrajectory] = shuffled[
                batch_start : batch_start + config.batch_size
            ]
            if split_at_train_time:
                batch_trajectories = [
                    t.sample_split(
                        batch_rng,
                        min_prefix=config.min_split_prefix,
                        min_suffix=config.min_split_suffix,
                    )
                    for t in raw_batch
                ]
            else:
                batch_trajectories = raw_batch

            # Snapshot weights so sampling is consistent within the batch.
            # For the distortion reward this same snapshot is used for the
            # prompt_logprobs scoring calls, so teacher and decoder are the
            # same frozen checkpoint throughout the reward computation.
            sampling_client = training_client.save_weights_and_get_sampling_client()

            # --- Held-out eval (data-controlled learning signal) ---
            # Uses this batch's pre-gradient-step snapshot = weights after
            # global_batch_idx updates. Runs at batch 0 (baseline) and every
            # eval_every batches thereafter.
            if eval_set and global_batch_idx % config.eval_every == 0:
                eval_metrics = run_eval(sampling_client)
                metrics.update(eval_metrics)
                if eval_metrics:
                    logger.info(
                        "Batch %d eval/reward_mean=%.4f (n=%d)",
                        global_batch_idx, eval_metrics["eval/reward_mean"],
                        int(eval_metrics["eval/n"]),
                    )

            # --- Rollout phase ---
            datums_D: list[types.Datum] = []
            rewards_P: list[float] = []
            futures_P: list[Future[types.SampleResponse]] = []
            prompts_P: list[types.ModelInput] = []
            convos_P: list[list[dict]] = []

            for traj in batch_trajectories:
                convo = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": format_trajectory_text(traj.steps)},
                ]
                model_input = renderer.build_generation_prompt(convo)
                future = sampling_client.sample(
                    prompt=model_input,
                    num_samples=config.group_size,
                    sampling_params=sampling_params,
                )
                futures_P.append(future)
                prompts_P.append(model_input)
                convos_P.append(convo)

            # --- Reward computation ---
            # For distortion reward: precompute x-context logprobs once per trajectory
            # (2 API calls), then score all group_size summaries with only z calls (2 each).
            # This reduces API calls from 4*group_size to 2 + 2*group_size per trajectory.
            #
            # Step 1: resolve sampling futures and collect work items.
            WorkItem = tuple  # (traj_idx, seq_idx, traj, convo, prompt, tokens, logprobs, content, x_ctxs)
            work_items: list[WorkItem] = []

            if config.reward_fn == "distortion":
                def _precompute_x_for_traj(args):
                    traj_idx, traj = args
                    sys_prompt = traj.agent_system_prompt or AGENT_SYSTEM_PROMPT
                    partial_messages = steps_to_messages(traj.steps, traj.task, system_prompt=sys_prompt)
                    continuation_messages = steps_to_messages(traj.continuation, traj.task, system_prompt=sys_prompt)[2:]
                    x_ctxs = precompute_x_contexts(
                        partial_messages=partial_messages,
                        continuation_messages=continuation_messages,
                        n_turns=config.distortion_n_turns,
                        model=config.scoring_model,
                        api_base=config.scoring_base_url,
                        tokenizer=tokenizer,
                        tools=[BASH_TOOL],
                    )
                    return traj_idx, (x_ctxs or None)

                x_ctx_map: dict[int, list | None] = {}
                with ThreadPoolExecutor(max_workers=len(batch_trajectories)) as ex:
                    x_futures = {
                        ex.submit(_precompute_x_for_traj, (i, t)): i
                        for i, t in enumerate(batch_trajectories)
                    }
                    for xf in tqdm(
                        as_completed(x_futures),
                        total=len(x_futures),
                        desc=f"Precompute x batch {global_batch_idx}",
                    ):
                        traj_idx, x_ctxs = xf.result()
                        x_ctx_map[traj_idx] = x_ctxs
            else:
                x_ctx_map = {}

            for traj_idx, (future, prompt, traj, convo) in enumerate(
                zip(futures_P, prompts_P, batch_trajectories, convos_P)
            ):
                # The sampling server can 400 with "Out of range float ... nan" when
                # the policy emits a non-finite logit for a given prompt. Skip that
                # trajectory's samples instead of crashing the whole run; the batch
                # proceeds with the trajectories that did sample cleanly.
                try:
                    sample_result = future.result()
                except Exception as exc:
                    logger.warning(
                        "Batch %d traj %d: sampling failed (%s), skipping trajectory",
                        global_batch_idx, traj_idx, exc,
                    )
                    continue
                x_ctxs = x_ctx_map.get(traj_idx)
                for seq_idx, sequence in enumerate(sample_result.sequences):
                    assert sequence.logprobs is not None
                    parsed_message, _ = renderer.parse_response(sequence.tokens)
                    content = strip_thinking(renderers.get_text_content(parsed_message))
                    work_items.append(
                        (traj_idx, seq_idx, traj, convo, prompt, sequence.tokens, sequence.logprobs, content, x_ctxs)
                    )

            # Step 2: score all summaries in parallel (z-only for distortion).
            def _score_item(item: WorkItem) -> tuple[int, int, float | None, list[int], list[float], list[dict]]:
                traj_idx, seq_idx, traj, convo, prompt, tokens, logprobs, content, x_ctxs = item
                if config.reward_fn == "distortion":
                    partial_messages = steps_to_messages(traj.steps, traj.task, system_prompt=traj.agent_system_prompt or AGENT_SYSTEM_PROMPT)
                    if x_ctxs is not None:
                        reward = distortion_reward_z_multi(
                            x_ctxs=x_ctxs,
                            summary=content,
                            partial_messages=partial_messages,
                            model=config.scoring_model,
                            api_base=config.scoring_base_url,
                            tokenizer=tokenizer,
                            max_size=config.distortion_max_size,
                            keep_first=config.distortion_keep_first,
                            lambda_len=config.distortion_lambda,
                            lambda_copy=config.distortion_lambda_copy,
                            copy_threshold=config.distortion_copy_threshold,
                            marker_penalty=config.distortion_marker_penalty,
                            beta=config.distortion_beta,
                            tools=[BASH_TOOL],
                        )
                    else:
                        reward = None
                else:
                    reward = coverage_reward(content, traj)
                return traj_idx, seq_idx, reward, tokens, logprobs, convo

            n_workers = config.group_size * len(batch_trajectories)
            scored: dict[int, list[tuple[int, float | None, list[int], list[float], list[dict]]]] = {}
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                reward_futures = {ex.submit(_score_item, item): item for item in work_items}
                for rf in tqdm(
                    as_completed(reward_futures),
                    total=len(reward_futures),
                    desc=f"Scoring batch {global_batch_idx}",
                ):
                    traj_idx, seq_idx, reward, tokens, logprobs, convo = rf.result()
                    scored.setdefault(traj_idx, []).append((seq_idx, reward, tokens, logprobs, convo))

            # --- Save sampled summaries ---
            if config.save_summaries:
                summaries_path = os.path.join(config.log_path, "summaries.jsonl")
                with open(summaries_path, "a") as f:
                    for item in work_items:
                        traj_idx, seq_idx, traj, convo, prompt, tokens, logprobs, content, x_ctxs = item
                        group_scores = scored.get(traj_idx, [])
                        reward = next((r for si, r, _, _, _ in group_scores if si == seq_idx), None)
                        partial_messages = steps_to_messages(
                            traj.steps, traj.task,
                            system_prompt=traj.agent_system_prompt or AGENT_SYSTEM_PROMPT,
                        )
                        z_messages = build_z_scoring_messages(
                            content, partial_messages,
                            max_size=config.distortion_max_size,
                            keep_first=config.distortion_keep_first,
                        )
                        # Copy metrics travel with each summary so the collapse into
                        # transcription is visible in the log (overlap -> 1) rather than
                        # only showing up as a mystery reward plateau.
                        _, copy_info = copy_penalty(
                            content, partial_messages, tokenizer,
                            lambda_len=config.distortion_lambda,
                            lambda_copy=config.distortion_lambda_copy,
                            copy_threshold=config.distortion_copy_threshold,
                            marker_penalty=config.distortion_marker_penalty,
                        )
                        f.write(json.dumps({
                            "batch": global_batch_idx,
                            "traj_idx": traj_idx,
                            "seq_idx": seq_idx,
                            "trajectory_id": traj.trajectory_id,
                            "n_prefix_steps": len(traj.steps),
                            "n_continuation_steps": len(traj.continuation),
                            "reward": reward,
                            "summary": content,
                            "copy": copy_info,
                            "summarizer_input": convo,
                            "deliberator_input": z_messages,
                        }) + "\n")

            # --- Advantage computation and datum assembly ---
            for traj_idx, (prompt, traj) in enumerate(
                zip(prompts_P, batch_trajectories)
            ):
                group = sorted(scored.get(traj_idx, []), key=lambda x: x[0])
                # Drop samples whose reward failed to compute (None/NaN/inf).
                # Fidelity is typically negative, so mapping failures to a
                # numeric 0.0 would put them at the top of the group and
                # reinforce whatever made the scorer fail.
                n_failed = sum(
                    1 for _, r, _, _, _ in group
                    if r is None or not math.isfinite(r)
                )
                if n_failed:
                    logger.warning(
                        "Batch %d traj %d: dropping %d/%d samples with failed reward",
                        global_batch_idx, traj_idx, n_failed, len(group),
                    )
                    group = [
                        g for g in group
                        if g[1] is not None and math.isfinite(g[1])
                    ]
                if len(group) < 2:
                    continue
                rewards_G = [r for _, r, _, _, _ in group]
                mean_reward = sum(rewards_G) / len(rewards_G)
                rewards_P.append(mean_reward)

                # Snap rewards to the reward_snap grid so sub-grid differences are ties.
                snap = config.reward_snap
                snapped_G = [round(r / snap) * snap for r in rewards_G] if snap > 0 else rewards_G

                spread = max(snapped_G) - min(snapped_G)
                if spread < config.min_reward_spread:
                    logger.debug(
                        "Batch %d traj %d: spread=%.4f < threshold=%.4f, skipping",
                        global_batch_idx, traj_idx, spread, config.min_reward_spread,
                    )
                    continue

                snapped_mean = sum(snapped_G) / len(snapped_G)
                advantages_G = [r - snapped_mean for r in snapped_G]
                if config.normalize_advantage:
                    var = sum(a * a for a in advantages_G) / len(advantages_G)
                    std = var ** 0.5
                    advantages_G = [a / (std + config.advantage_eps) for a in advantages_G]

                ob_len = prompt.length - 1
                for (_, _, sampled_tokens, logprobs, _), advantage in zip(group, advantages_G):
                    # Guard the optimizer: a single non-finite advantage or sampled-token
                    # logprob (e.g. a -inf logprob from the sampler) NaNs the PPO loss,
                    # which NaNs the LoRA weights, after which every sample 400s and the
                    # run dies. Drop such datums instead of poisoning the gradient step.
                    if not math.isfinite(advantage) or any(
                        lp is None or not math.isfinite(lp) for lp in logprobs
                    ):
                        logger.warning(
                            "Batch %d traj %d: non-finite advantage/logprob, skipping datum",
                            global_batch_idx, traj_idx,
                        )
                        continue
                    model_input = prompt.append(
                        types.EncodedTextChunk(tokens=sampled_tokens[:-1])
                    )
                    target_tokens = [0] * ob_len + sampled_tokens
                    padded_logprobs = [0.0] * ob_len + logprobs
                    padded_advantages = (
                        [0.0] * ob_len + [advantage] * (model_input.length - ob_len)
                    )
                    datum = types.Datum(
                        model_input=model_input,
                        loss_fn_inputs={
                            "target_tokens": TensorData.from_torch(
                                torch.tensor(target_tokens)
                            ),
                            "logprobs": TensorData.from_torch(
                                torch.tensor(padded_logprobs)
                            ),
                            "advantages": TensorData.from_torch(
                                torch.tensor(padded_advantages)
                            ),
                        },
                    )
                    datums_D.append(datum)

            # --- Gradient step ---
            if not datums_D:
                logger.warning(
                    "Batch %d: all advantages zero, skipping gradient step", global_batch_idx
                )
            else:
                # NOTE: the "ppo" loss reads its clip bounds from per-token
                # loss_fn_inputs tensors (clip_low_threshold/clip_high_threshold on
                # each Datum), NOT from loss_fn_config. Passing them via config left
                # the clip undefined (≈0), which zeroed the gradient on positive-
                # advantage tokens and prevented any learning. importance_sampling is
                # the reference recipe's default; it needs no clip config and uses the
                # target_tokens/logprobs/advantages we already provide. Near-on-policy
                # here (weights re-snapshot every batch, one optim step), so clipping
                # is not required.
                fwd_bwd_future = training_client.forward_backward(
                    datums_D,
                    loss_fn="importance_sampling",
                )
                optim_step_future = training_client.optim_step(adam_params)
                fwd_bwd_future.result()
                optim_result = optim_step_future.result()
                if optim_result.metrics:
                    metrics.update(optim_result.metrics)

            metrics["time/total"] = time.time() - t_start
            metrics["reward/mean"] = sum(rewards_P) / len(rewards_P) if rewards_P else 0.0
            metrics["reward/max"] = max(rewards_P) if rewards_P else 0.0
            metrics["train/datums"] = len(datums_D)
            ml_logger.log_metrics(metrics, step=global_batch_idx)
            global_batch_idx += 1

    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": global_batch_idx},
        ttl_seconds=None,
    )
    ml_logger.close()
    logger.info("Training complete")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
