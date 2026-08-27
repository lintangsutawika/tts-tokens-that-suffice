#!/usr/bin/env bash
#SBATCH --job-name=tts-train
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --gres=gpu:4
#SBATCH --mem-per-gpu=64G
#SBATCH --constraint=nvlink
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --time=0-48:00:00
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1

set -euo pipefail

[ -f .env ] && . .env

DATASET=${DATASET:-datasets/summarizer_train.jsonl}
LOG_PATH=${LOG_PATH:-checkpoints/summarizer-rl}
MODEL=${MODEL:-Qwen/Qwen3.5-9B}
TINKER_BASE_URL=${TINKER_BASE_URL:-http://localhost:9123}
SCORING_MODEL=${SCORING_MODEL:-litellm_proxy/Qwen/Qwen3.6-27B-FP8}
# Split trajectories where the context first reaches this many tokens, matching the
# eval's COMPRESS_AT_TOKENS so the summarizer trains at the length it is invoked at.
# Keep the two in lockstep when changing either.
SPLIT_AT_TOKENS=${SPLIT_AT_TOKENS:-16384}

TINKER_API_KEY=${TINKER_API_KEY:-tml-dummy} \
uv run -m tts.train_rl \
    dataset_path=${DATASET} \
    model_name=${MODEL} \
    renderer_name=${RENDERER:-qwen3_5} \
    max_tokens=${MAX_TOKENS:-2048} \
    log_path=${LOG_PATH} \
    base_url=${TINKER_BASE_URL} \
    reward_fn=distortion \
    scoring_base_url=${SCORING_BASE_URL:-http://localhost:8080/v1} \
    scoring_model=${SCORING_MODEL} \
    batch_size=16 \
    group_size=16 \
    num_epochs=3 \
    min_split_prefix=6 \
    split_at_tokens=${SPLIT_AT_TOKENS} \
    compaction_token_budget=${COMPACTION_BUDGET:-9000} \
    min_summary_tokens=${MIN_SUMMARY_TOKENS:-512} \
    distortion_lambda=${DISTORTION_LAMBDA:-0.5} \
    distortion_lambda_copy=${DISTORTION_LAMBDA_COPY:-1.0} \
    distortion_copy_threshold=${DISTORTION_COPY_THRESHOLD:-0.3} \
    distortion_marker_penalty=${DISTORTION_MARKER_PENALTY:-1.0} \
    distortion_beta=${DISTORTION_BETA:-1.0} \
    lora_rank=16 \
    learning_rate=${LEARNING_RATE:-4e-5} \
    sampling_temperature=1.0 \
    normalize_advantage=true \
    reward_snap=${REWARD_SNAP:-0.05} \
    min_reward_spread=${MIN_REWARD_SPREAD:-0.02} \
    eval_size=32 \
    eval_every=5 \
    save_every=5
