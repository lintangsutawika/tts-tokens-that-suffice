"""
test_rl_loop.py — minimal known-good GRPO loop on GSM8K, used to validate that
the tinker training machinery actually moves the policy:

    sample → reward → group-centered advantage → importance_sampling
    forward_backward → optim_step → (repeat)

Adapted from tinker_cookbook/recipes/rl_loop.py with two changes:
  * the few-shot ``convo_prefix`` is removed — our model is instruction-tuned, so
    a plain user turn suffices (no in-context example needed);
  * defaults target the instruction-tuned Qwen/Qwen3.5-9B, with an overridable
    renderer.

Reward is verifiable (boxed GSM8K answer correct → 1.0 else 0.0). If
``reward/total`` climbs here, the trainer + model + renderer are sound, which
isolates any flat-learning in the summarizer run to our reward/data — not the
machinery. The Datum assembly below is byte-for-byte the reference's, matching
what train_rl.py does.

Run (needs the tinker adapter slot free):
    TINKER_API_KEY=tml-dummy uv run scripts/debug/test_rl_loop.py \
        base_url=http://localhost:9123 \
        model_name=Qwen/Qwen3.5-9B \
        renderer_name=qwen3_5_disable_thinking \
        log_path=checkpoints/test-rl-loop
"""

import json
import logging
import os
import time
from concurrent.futures import Future

import chz
import datasets
import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData
from tqdm import tqdm

from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.recipes.math_rl.math_env import extract_gsm8k_final_answer
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.git_rev import recipe_user_metadata

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    base_url: str | None = None
    log_path: str = "checkpoints/test-rl-loop"
    model_name: str = "Qwen/Qwen3-8B"
    renderer_name: str | None = None  # default: model's recommended renderer
    batch_size: int = 32
    group_size: int = 16
    learning_rate: float = 4e-5
    lora_rank: int = 32
    save_every: int = 0  # 0 = disabled (throwaway validation run)
    max_tokens: int = 256
    ttl_seconds: int | None = 604800
    # Save each rollout's generated response + correctness to {log_path}/samples.jsonl
    # for inspecting what the model produces and sanity-checking the grader.
    save_samples: bool = True


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception:
        return None


def get_reward(response: str, answer: str) -> float:
    try:
        given_answer = extract_boxed(response)
        ground_truth = extract_gsm8k_final_answer(answer)
        reward = grade_answer(given_answer, ground_truth)
        # lambda_ = 0.5
        # length_bonus = 64/(len(response.split()) + 1)
        # reward += lambda_ * length_bonus
        return reward
    except ValueError:
        return 0.0


def main(config: Config):
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=None,
        wandb_name=None,
        config=config,
        do_configure_logging_module=True,
    )

    tokenizer = get_tokenizer(config.model_name)
    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(
        config.model_name
    )
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    logger.info("Loading dataset...")
    dataset = datasets.load_dataset("openai/gsm8k", "main")
    assert isinstance(dataset, datasets.DatasetDict)
    train_dataset = dataset["train"]

    question_suffix = (
        " Provide a numerical answer without units, written inside \\boxed{}."
    )

    n_train_batches = len(train_dataset) // config.batch_size

    service_client = tinker.ServiceClient(
        base_url=config.base_url,
        user_metadata=recipe_user_metadata("test_rl_loop"),
    )

    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_info.state_path
        )
        start_batch = resume_info.batch
        logger.info(f"Resuming from batch {start_batch}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
        start_batch = 0

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
    )
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    logger.info(f"Training for {n_train_batches} batches")

    for batch_idx in range(start_batch, n_train_batches):
        t_start = time.time()
        metrics: dict[str, float] = {
            "progress/batch": batch_idx,
            "optim/lr": config.learning_rate,
            "progress/done_frac": (batch_idx + 1) / n_train_batches,
        }

        if config.save_every > 0 and batch_idx % config.save_every == 0 and batch_idx > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{batch_idx:06d}",
                log_path=config.log_path,
                kind="state",
                loop_state={"batch": batch_idx},
                ttl_seconds=config.ttl_seconds,
            )

        batch_start = batch_idx * config.batch_size
        batch_end = min((batch_idx + 1) * config.batch_size, len(train_dataset))
        batch_rows = train_dataset.select(range(batch_start, batch_end))

        sampling_client = training_client.save_weights_and_get_sampling_client()

        datums_D: list[types.Datum] = []
        rewards_P: list[float] = []
        futures_P: list[Future[types.SampleResponse]] = []
        prompts_P: list[types.ModelInput] = []
        for question in batch_rows["question"]:
            # Instruction-tuned model: a plain user turn, no few-shot prefix.
            convo = [{"role": "user", "content": question + question_suffix}]
            model_input = renderer.build_generation_prompt(convo)
            future = sampling_client.sample(
                prompt=model_input,
                num_samples=config.group_size,
                sampling_params=sampling_params,
            )
            futures_P.append(future)
            prompts_P.append(model_input)

        sample_records: list[dict] = []
        for q_idx, (future, prompt, question, answer) in enumerate(tqdm(
            zip(futures_P, prompts_P, batch_rows["question"], batch_rows["answer"]),
            total=len(futures_P),
            desc=f"Sampling batch {batch_idx}",
        )):
            sample_result = future.result()
            rewards_G: list[float] = []
            sampled_tokens_G_T: list[list[int]] = []
            logprobs_G_T: list[list[float]] = []
            gold = _safe(extract_gsm8k_final_answer, answer)
            for s_idx, sequence in enumerate(sample_result.sequences):
                sampled_tokens = sequence.tokens
                sampled_logprobs = sequence.logprobs
                assert sampled_logprobs is not None
                sampled_tokens_G_T.append(sampled_tokens)
                logprobs_G_T.append(sampled_logprobs)
                parsed_message, _ = renderer.parse_response(sampled_tokens)
                content = renderers.get_text_content(parsed_message)
                reward = get_reward(content, answer)
                rewards_G.append(reward)
                if config.save_samples:
                    sample_records.append({
                        "batch": batch_idx,
                        "q_idx": q_idx,
                        "sample_idx": s_idx,
                        "question": question,
                        "response": content,
                        "pred": _safe(extract_boxed, content),
                        "gold": gold,
                        "correct": reward,
                    })

            mean_reward = sum(rewards_G) / len(rewards_G)
            advantages_G = [r - mean_reward for r in rewards_G]
            rewards_P.append(mean_reward)

            if all(a == 0.0 for a in advantages_G):
                continue

            for sampled_tokens, logprobs, advantage in zip(
                sampled_tokens_G_T, logprobs_G_T, advantages_G
            ):
                ob_len = prompt.length - 1
                model_input = prompt.append(types.EncodedTextChunk(tokens=sampled_tokens[:-1]))
                target_tokens = [0] * ob_len + sampled_tokens
                padded_logprobs = [0.0] * ob_len + logprobs
                padded_advantages = [0.0] * ob_len + [advantage] * (model_input.length - ob_len)
                assert (
                    model_input.length
                    == len(target_tokens)
                    == len(padded_logprobs)
                    == len(padded_advantages)
                )
                datum = types.Datum(
                    model_input=model_input,
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                        "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                        "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
                    },
                )
                datums_D.append(datum)

        # Save this batch's rollout outputs + correctness for inspection.
        if config.save_samples and sample_records:
            with open(os.path.join(config.log_path, "samples.jsonl"), "a") as f:
                for rec in sample_records:
                    f.write(json.dumps(rec) + "\n")

        if len(datums_D) == 0:
            logger.warning("Batch %d: all advantages zero, skipping training step", batch_idx)
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
        metrics["reward/total"] = sum(rewards_P) / len(rewards_P) if rewards_P else 0.0
        metrics["train/datums"] = len(datums_D)
        ml_logger.log_metrics(metrics, step=batch_idx)
        logger.info(
            "batch %d  reward/total=%.4f  datums=%d",
            batch_idx, metrics["reward/total"], len(datums_D),
        )

    ml_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
