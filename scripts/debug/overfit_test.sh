#!/usr/bin/env bash
#SBATCH --job-name=tts-overfit
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --gres=gpu:4
#SBATCH --mem-per-gpu=64G
#SBATCH --constraint=nvlink
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1

# Overfit sanity check: can the setup improve reward on a handful of trajectories
# it sees every epoch? Trains on 8 trajectories (= 1 batch/epoch) and evals on
# those SAME 8 with fixed splits (eval_on_train=true). If eval/reward_mean climbs,
# optimization + reward signal work and the real-run flatness is a variance/
# generalization problem. If it stays flat even here, the gradient genuinely
# doesn't help — an optimization/loss issue, not a data problem.

set -euo pipefail

[ -f .env ] && . .env

DATASET=${DATASET:-datasets/overfit8.jsonl}
LOG_PATH=${LOG_PATH:-checkpoints/overfit8}
MODEL=${MODEL:-Qwen/Qwen3.5-9B}
TINKER_BASE_URL=${TINKER_BASE_URL:-http://localhost:9123}
SCORING_MODEL=${SCORING_MODEL:-litellm_proxy/Qwen/Qwen3.6-35B-A3B}

TINKER_API_KEY=${TINKER_API_KEY:-tml-dummy} \
uv run -m tts.train_rl \
    dataset_path=${DATASET} \
    model_name=${MODEL} \
    renderer_name=qwen3_5_disable_thinking \
    log_path=${LOG_PATH} \
    base_url=${TINKER_BASE_URL} \
    reward_fn=distortion \
    scoring_base_url=${SCORING_BASE_URL:-http://localhost:8000/v1} \
    scoring_model=${SCORING_MODEL} \
    batch_size=8 \
    group_size=8 \
    num_epochs=40 \
    lora_rank=16 \
    learning_rate=3e-4 \
    sampling_temperature=1.0 \
    distortion_max_size=8 \
    normalize_advantage=false \
    eval_size=8 \
    eval_every=2 \
    eval_on_train=true \
    save_every=20
