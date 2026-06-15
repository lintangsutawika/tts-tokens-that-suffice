"""
Supervised fine-tuning recipe: teach a model to summarize coding agent trajectories.

Each training example is a (trajectory, summary) pair loaded from a JSONL file.
The model sees the full agent trajectory (task + steps) as context and is trained
to generate the summary token-by-token (only summary tokens receive loss weight).

Data format: see data.py for the ADP-inspired JSONL schema.

Usage:
    TINKER_API_KEY=tml-dummy uv run -m tts.recipes.agent_summarization.recipe \\
        dataset_path=/path/to/trajectories.jsonl

Or via the test script:
    TINKER_API_KEY=tml-dummy uv run tests/train_agent_summarization.py
"""

from __future__ import annotations

import logging
import time

import chz
import tinker
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.git_rev import recipe_user_metadata

from tts.recipes.agent_summarization.data import load_trajectories, trajectory_to_conversation

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    base_url: str | None = None
    log_path: str = "/tmp/tinker-agent-summarization"
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    # Path to a JSONL file of AgentTrajectory records
    dataset_path: str = ""
    batch_size: int = 16
    learning_rate: float = 1e-4
    # Generous context to fit long trajectories
    max_length: int = 32768
    lora_rank: int = 32
    save_every: int = 20
    ttl_seconds: int | None = 604800
    num_epochs: int = 3


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
    trajectories = load_trajectories(config.dataset_path)
    logger.info("Loaded %d trajectories", len(trajectories))

    if len(trajectories) < config.batch_size:
        raise ValueError(
            f"Dataset has only {len(trajectories)} trajectories but batch_size="
            f"{config.batch_size}. Reduce batch_size or add more data."
        )

    n_batches_per_epoch = len(trajectories) // config.batch_size
    n_dropped = len(trajectories) % config.batch_size
    if n_dropped:
        logger.info(
            "Dropping last %d trajectories per epoch to keep batch size "
            "uniform at %d",
            n_dropped,
            config.batch_size,
        )

    total_batches = n_batches_per_epoch * config.num_epochs
    logger.info(
        "%d trajectories × %d epochs → %d total batches",
        len(trajectories),
        config.num_epochs,
        total_batches,
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

    for global_batch_idx in range(start_batch, total_batches):
        start_time = time.time()
        epoch = global_batch_idx // n_batches_per_epoch
        batch_in_epoch = global_batch_idx % n_batches_per_epoch

        # Shuffle deterministically per epoch so order differs each epoch
        import random
        rng = random.Random(epoch)
        shuffled = list(trajectories)
        rng.shuffle(shuffled)

        if config.save_every > 0 and global_batch_idx % config.save_every == 0 and global_batch_idx > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{global_batch_idx:06d}",
                log_path=config.log_path,
                kind="state",
                loop_state={"batch": global_batch_idx},
                ttl_seconds=config.ttl_seconds,
            )

        lr_mult = max(0.0, 1.0 - global_batch_idx / total_batches)
        current_lr = config.learning_rate * lr_mult
        adam_params = tinker.AdamParams(
            learning_rate=current_lr, beta1=0.9, beta2=0.95, eps=1e-8
        )

        batch_start = batch_in_epoch * config.batch_size
        batch_end = batch_start + config.batch_size
        batch_trajectories = shuffled[batch_start:batch_end]

        # Convert each trajectory to a conversation; train only on the final
        # assistant turn (the summary). LAST_ASSISTANT_MESSAGE masks the full
        # trajectory context from the loss so gradients come only from summary tokens.
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

        metrics.update(
            epoch=epoch,
            num_sequences=len(batch),
            num_tokens=sum(d.model_input.length for d in batch),
            learning_rate=current_lr,
            train_mean_nll=train_nll,
            progress=global_batch_idx / total_batches,
            time_total=time.time() - start_time,
        )
        ml_logger.log_metrics(metrics=metrics, step=global_batch_idx)

    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": total_batches},
        ttl_seconds=None,
    )

    ml_logger.close()
    logger.info("Training complete")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
