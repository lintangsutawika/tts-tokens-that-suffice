"""
Supervised fine-tuning recipe: teach a model to summarize partial coding agent
trajectories ("what has the agent done so far?").

Each training record is a (partial_trajectory, summary) pair from a JSONL file.
The model sees the trajectory up to some cutpoint and is trained to generate a
concise progress summary. Only summary tokens receive loss weight.

Intended data source: mini-SWE trajectories sampled at random cutpoints, with
teacher-generated summaries attached (see data.py for the JSONL schema and the
from_swe_agent_dict() loader).

Optional augmentation (augment_with_prefixes=True): each stored trajectory also
trains on random sub-prefixes of its steps. This multiplies effective dataset
size without requiring additional teacher summaries — the stored summary is
treated as approximately valid for shorter prefixes too.

Usage:
    TINKER_API_KEY=tml-dummy uv run -m tts.recipes.agent_summarization.recipe \\
        dataset_path=/path/to/partial_trajectories.jsonl

Or via the test script:
    TINKER_API_KEY=tml-dummy uv run tests/train_agent_summarization.py
"""

from __future__ import annotations

import logging
import random
import time

import chz
import tinker
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.git_rev import recipe_user_metadata

from tts.recipes.agent_summarization.data import (
    AgentTrajectory,
    load_trajectories,
    trajectory_to_conversation,
)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    base_url: str | None = None
    log_path: str = "/tmp/tinker-agent-summarization"
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    dataset_path: str = ""
    batch_size: int = 16
    learning_rate: float = 1e-4
    # Generous context: partial trajectories can still be long
    max_length: int = 32768
    lora_rank: int = 32
    save_every: int = 20
    ttl_seconds: int | None = 604800
    num_epochs: int = 3
    # When True, also train on random sub-prefixes of each stored trajectory.
    # This multiplies data diversity at the cost of label noise (the stored
    # summary is not perfectly accurate for a shorter prefix).
    augment_with_prefixes: bool = False
    # Minimum fraction of steps to keep when augmenting with prefixes.
    min_prefix_fraction: float = 0.3


def _sample_training_trajectories(
    trajectories: list[AgentTrajectory],
    augment: bool,
    min_fraction: float,
    rng: random.Random,
) -> list[AgentTrajectory]:
    """
    Build the list of (trajectory, summary) pairs for one epoch.

    With augment=False: returns the trajectories as-is (each is already a
    partial trajectory at its recorded cutpoint).

    With augment=True: for each trajectory, also adds one randomly-truncated
    sub-prefix.  The stored summary is reused as an approximate label.
    """
    result = list(trajectories)
    if not augment:
        return result

    for traj in trajectories:
        n = len(traj.steps)
        if n < 2:
            continue
        min_steps = max(1, int(n * min_fraction))
        k = rng.randint(min_steps, n - 1)
        result.append(traj.with_prefix(k))

    return result


def main(config: Config) -> None:
    if not config.dataset_path:
        raise ValueError("dataset_path must be set to a JSONL file of trajectories")

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

    logger.info("Loading trajectories from %s", config.dataset_path)
    base_trajectories = load_trajectories(config.dataset_path)
    logger.info("Loaded %d base trajectories", len(base_trajectories))

    if len(base_trajectories) < config.batch_size:
        raise ValueError(
            f"Dataset has {len(base_trajectories)} trajectories but "
            f"batch_size={config.batch_size}. Reduce batch_size or add more data."
        )

    service_client = tinker.ServiceClient(
        base_url=config.base_url,
        user_metadata=recipe_user_metadata("recipe_agent_summarization"),
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

    global_batch_idx = start_batch
    for epoch in range(config.num_epochs):
        epoch_rng = random.Random(epoch)

        # Build this epoch's training set (base + optional prefix augmentations)
        epoch_trajectories = _sample_training_trajectories(
            base_trajectories,
            augment=config.augment_with_prefixes,
            min_fraction=config.min_prefix_fraction,
            rng=epoch_rng,
        )
        epoch_rng.shuffle(epoch_trajectories)

        n_batches = len(epoch_trajectories) // config.batch_size
        n_dropped = len(epoch_trajectories) % config.batch_size
        if n_dropped:
            logger.info(
                "Epoch %d: dropping %d examples to keep batch size uniform at %d",
                epoch,
                n_dropped,
                config.batch_size,
            )
        logger.info(
            "Epoch %d: %d examples → %d batches", epoch, len(epoch_trajectories), n_batches
        )

        for batch_in_epoch in range(n_batches):
            if global_batch_idx < start_batch:
                global_batch_idx += 1
                continue

            start_time = time.time()

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

            total_batches_estimate = config.num_epochs * (
                len(base_trajectories) // config.batch_size
            )
            lr_mult = max(0.0, 1.0 - global_batch_idx / max(1, total_batches_estimate))
            current_lr = config.learning_rate * lr_mult
            adam_params = tinker.AdamParams(
                learning_rate=current_lr, beta1=0.9, beta2=0.95, eps=1e-8
            )

            batch_start = batch_in_epoch * config.batch_size
            batch_trajectories = epoch_trajectories[batch_start : batch_start + config.batch_size]

            # Only the summary tokens (last assistant turn) receive loss weight.
            batch = [
                conversation_to_datum(
                    trajectory_to_conversation(traj),
                    renderer,
                    config.max_length,
                    renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE,
                )
                for traj in batch_trajectories
            ]

            fwd_bwd_future = training_client.forward_backward(batch, loss_fn="cross_entropy")
            optim_step_future = training_client.optim_step(adam_params)
            fwd_bwd_result = fwd_bwd_future.result()
            optim_result = optim_step_future.result()

            metrics: dict = {}
            if optim_result.metrics:
                metrics.update(optim_result.metrics)

            train_logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
            train_weights = [d.loss_fn_inputs["weights"] for d in batch]
            train_nll = compute_mean_nll(train_logprobs, train_weights)

            avg_steps = sum(len(t.steps) for t in batch_trajectories) / len(batch_trajectories)
            metrics.update(
                epoch=epoch,
                num_sequences=len(batch),
                num_tokens=sum(d.model_input.length for d in batch),
                avg_trajectory_steps=avg_steps,
                learning_rate=current_lr,
                train_mean_nll=train_nll,
                time_total=time.time() - start_time,
            )
            ml_logger.log_metrics(metrics=metrics, step=global_batch_idx)
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
