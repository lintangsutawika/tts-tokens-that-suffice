#!/usr/bin/env bash
#SBATCH --job-name=tts-collect
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --mem=64G
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --time=0-48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1

set -euo pipefail

[ -f .env ] && . .env

MODEL=${MODEL:-litellm_proxy/Qwen/Qwen3.6-35B-A3B}
DATASET=${DATASET:-swe-smith}
SLICE=${SLICE:-}
MODEL_BASENAME=${MODEL##*/}
OUTPUT=${OUTPUT:-outputs/${DATASET}-${MODEL_BASENAME}}
WORKERS=${WORKERS:-4}
PORT=${PORT:-8000}

SLICE_ARG=${SLICE:+--slice ${SLICE}}
export MSWEA_COST_TRACKING='ignore_errors'
uv run -m tts.collect_trajectories \
    ${SLICE_ARG} \
    --dataset ${DATASET} \
    --output ${OUTPUT} \
    --workers ${WORKERS} \
    --model ${MODEL} \
    -c swebench.yaml \
    -c model.model_kwargs.api_base=http://0.0.0.0:${PORT}/v1 \
    -c model.model_kwargs.temperature=1.0 \
    -c model.model_kwargs.top_k=20 \
    -c model.model_kwargs.top_p=0.95 \
    -c model.model_kwargs.min_p=0.0 \
    -c model.model_kwargs.presence_penalty=0.0 \
    -c model.model_kwargs.repetition_penalty=1.0 \
    -c 'model.model_kwargs.extra_body={"chat_template_kwargs": {"enable_thinking": true}}'
