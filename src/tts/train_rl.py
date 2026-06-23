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
    tokens (SamplingParams(prompt_logprobs=1)).  Falls back to 0.0 and emits
    a warning if the backend does not support this.

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

import logging
import time
from concurrent.futures import Future

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
    SYSTEM_PROMPT,
    TrajectoryStep,
    format_trajectory_text,
    load_trajectories,
    steps_to_messages,
)
from tts.reward.reverse_kl_reward import distortion_reward

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    base_url: str | None = None
    log_path: str = "/tmp/tinker-agent-summarization-rl"
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    dataset_path: str = ""
    batch_size: int = 32       # trajectories per gradient step
    group_size: int = 8        # candidate summaries sampled per trajectory
    learning_rate: float = 4e-5
    max_tokens: int = 512      # max generated summary length
    lora_rank: int = 32
    save_every: int = 20
    ttl_seconds: int | None = 604800
    num_epochs: int = 1
    # Reward function: "coverage" or "distortion"
    reward_fn: str = "coverage"
    # Length-penalty coefficient λ for the distortion reward.
    # Positive values encourage shorter summaries.
    distortion_lambda: float = 0.0
    # URL of the vLLM scoring server (used only when reward_fn="distortion").
    scoring_base_url: str = "http://localhost:8000/v1"
    # Target message budget for the z-context. Tail = max_size//2 - keep_first messages.
    distortion_max_size: int = 20
    # Messages to keep verbatim from the start of the trajectory in z-context.
    distortion_keep_first: int = 4


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
        wandb_project=None,
        wandb_name=None,
        config=config,
        do_configure_logging_module=True,
    )

    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info("Using renderer: %s", renderer_name)
    logger.info("Using reward_fn: %s", config.reward_fn)

    logger.info("Loading trajectories from %s", config.dataset_path)
    trajectories = load_trajectories(config.dataset_path)
    logger.info("Loaded %d trajectories", len(trajectories))

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
        stop=renderer.get_stop_sequences(),
    )
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    import random
    global_batch_idx = start_batch
    for epoch in range(config.num_epochs):
        epoch_rng = random.Random(epoch)
        shuffled = list(trajectories)
        epoch_rng.shuffle(shuffled)

        for batch_in_epoch in range(n_batches_per_epoch):
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
            batch_trajectories: list[AgentTrajectory] = shuffled[
                batch_start : batch_start + config.batch_size
            ]

            # Snapshot weights so sampling is consistent within the batch.
            # For the distortion reward this same snapshot is used for the
            # prompt_logprobs scoring calls, so teacher and decoder are the
            # same frozen checkpoint throughout the reward computation.
            sampling_client = training_client.save_weights_and_get_sampling_client()

            # --- Rollout phase ---
            datums_D: list[types.Datum] = []
            rewards_P: list[float] = []
            futures_P: list[Future[types.SampleResponse]] = []
            prompts_P: list[types.ModelInput] = []

            for traj in batch_trajectories:
                convo = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": format_trajectory_text(traj.steps, traj.task)},
                ]
                model_input = renderer.build_generation_prompt(convo)
                future = sampling_client.sample(
                    prompt=model_input,
                    num_samples=config.group_size,
                    sampling_params=sampling_params,
                )
                futures_P.append(future)
                prompts_P.append(model_input)

            # --- Reward + advantage computation ---
            for future, prompt, traj in tqdm(
                zip(futures_P, prompts_P, batch_trajectories),
                total=len(futures_P),
                desc=f"Scoring batch {global_batch_idx}",
            ):
                sample_result = future.result()

                rewards_G: list[float] = []
                sampled_tokens_G_T: list[list[int]] = []
                logprobs_G_T: list[list[float]] = []

                for sequence in sample_result.sequences:
                    sampled_tokens = sequence.tokens
                    sampled_logprobs = sequence.logprobs
                    assert sampled_logprobs is not None

                    parsed_message, _ = renderer.parse_response(sampled_tokens)
                    content = renderers.get_text_content(parsed_message)

                    if config.reward_fn == "distortion":
                        partial_messages = steps_to_messages(traj.steps, traj.task)
                        continuation_messages = steps_to_messages(traj.continuation, traj.task)[2:]
                        reward = distortion_reward(
                            partial_messages=partial_messages,
                            summary=content,
                            continuation_messages=continuation_messages,
                            model=config.model_name,
                            api_base=config.scoring_base_url,
                            tokenizer=tokenizer,
                            max_size=config.distortion_max_size,
                            keep_first=config.distortion_keep_first,
                            lambda_len=config.distortion_lambda,
                        )
                    else:
                        reward = coverage_reward(content, traj)

                    sampled_tokens_G_T.append(sampled_tokens)
                    logprobs_G_T.append(sampled_logprobs)
                    rewards_G.append(reward)

                mean_reward = sum(rewards_G) / len(rewards_G)
                advantages_G = [r - mean_reward for r in rewards_G]
                rewards_P.append(mean_reward)

                # Skip if all completions got the same reward (no learning signal)
                if all(a == 0.0 for a in advantages_G):
                    continue

                ob_len = prompt.length - 1
                for sampled_tokens, logprobs, advantage in zip(
                    sampled_tokens_G_T, logprobs_G_T, advantages_G
                ):
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
                fwd_bwd_future = training_client.forward_backward(
                    datums_D, loss_fn="importance_sampling"
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
