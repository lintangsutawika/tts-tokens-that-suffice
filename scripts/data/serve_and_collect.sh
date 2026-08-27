#!/usr/bin/env bash
#SBATCH --job-name=tts-serve
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --gres=gpu:2
#SBATCH --mem-per-gpu=64G
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --time=0-48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
##SBATCH --constraint=nvlink

set -euo pipefail

[ -f .env ] && . .env

MODEL=${1:-litellm_proxy/Qwen/Qwen3.5-27B}
DATASET=${DATASET:-swe-smith}
SLICE=${2:-}
MODEL_BASENAME=${MODEL##*/}
OUTPUT=${OUTPUT:-outputs/${DATASET}-${MODEL_BASENAME}}
WORKERS=${WORKERS:-4}
PORT=${PORT:-$((RANDOM % 16384 + 49152))}

# Start vLLM server in background
uv run vllm serve Qwen/Qwen3.6-27B-FP8 \
    --tensor-parallel-size 2 \
    --data-parallel-size 1 \
    --max-model-len 65536 \
    --disable-custom-all-reduce \
    --kv-cache-dtype auto \
    --gpu-memory-utilization 0.85 \
    --port ${PORT} \
    --enable-prefix-caching \
    --enforce-eager \
    --enable-auto-tool-choice \
    --language-model-only \
    --tool-call-parser qwen3_coder \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --reasoning-parser qwen3 &
VLLM_PID=$!

trap 'kill ${VLLM_PID} 2>/dev/null; wait ${VLLM_PID} 2>/dev/null' EXIT SIGTERM SIGINT

# Wait for server to be ready
echo "Waiting for vLLM server on port ${PORT}..."
READY=0
for i in $(seq 1 120); do
    if curl -sf http://0.0.0.0:${PORT}/health > /dev/null 2>&1; then
        READY=1
        break
    fi
    sleep 5
done
if [ "${READY}" -eq 0 ]; then
    echo "ERROR: vLLM server did not become ready after 600s" >&2
    exit 1
fi
echo "vLLM server ready."

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

echo "Killing vLLM server (pid=${VLLM_PID})..."
kill ${VLLM_PID} 2>/dev/null
wait ${VLLM_PID} 2>/dev/null
echo "vLLM server stopped."
