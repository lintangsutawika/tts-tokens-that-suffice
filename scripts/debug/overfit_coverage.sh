#!/usr/bin/env bash
#SBATCH --job-name=tts-overfit-cov
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

# Plumbing bisection: same overfit setup as overfit_test.sh, but with the dense,
# reference-free COVERAGE reward instead of distortion. Coverage is trivially
# learnable (the policy just has to mention the tool/file names), and needs no
# scoring server. If eval/reward_mean climbs here, the optimizer/Datum/PPO
# machinery is sound and the distortion reward is the problem. If it stays flat
# even here, the gradient plumbing itself is broken.

set -euo pipefail

[ -f .env ] && . .env

DATASET=${DATASET:-datasets/overfit8.jsonl}
LOG_PATH=${LOG_PATH:-checkpoints/overfit8_cov}
MODEL=${MODEL:-Qwen/Qwen3.5-9B}
TINKER_BASE_URL=${TINKER_BASE_URL:-http://localhost:9123}

TINKER_API_KEY=${TINKER_API_KEY:-tml-dummy} \
uv run -m tts.train_rl \
    dataset_path=${DATASET} \
    model_name=${MODEL} \
    renderer_name=qwen3_5_disable_thinking \
    log_path=${LOG_PATH} \
    base_url=${TINKER_BASE_URL} \
    reward_fn=coverage \
    batch_size=8 \
    group_size=8 \
    num_epochs=40 \
    lora_rank=16 \
    learning_rate=3e-4 \
    sampling_temperature=1.0 \
    reward_snap=0.0 \
    min_reward_spread=0.0 \
    eval_size=8 \
    eval_every=2 \
    eval_on_train=true \
    save_every=20
