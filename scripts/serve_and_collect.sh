#!/usr/bin/env bash
#SBATCH --job-name=tts-serve
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --gres=gpu:2
#SBATCH --mem-per-gpu=64G
#SBATCH --constraint=nvlink
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --time=0-48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1

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
VLLM_ATTENTION_BACKEND=XFORMERS \
uv run vllm serve Qwen/Qwen3.5-27B \
    --tensor-parallel-size 2 \
    --data-parallel-size 1 \
    --max-model-len 65536 \
    --disable-custom-all-reduce \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.7 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 2000 \
    --port ${PORT} \
    --no-enable-prefix-caching \
    --long-prefill-token-threshold 0 \
    --enable-auto-tool-choice \
    --language-model-only \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' &
VLLM_PID=$!

trap 'kill ${VLLM_PID} 2>/dev/null; wait ${VLLM_PID} 2>/dev/null' EXIT SIGTERM SIGINT

# Wait for server to be ready
echo "Waiting for vLLM server on port ${PORT}..."
for i in $(seq 1 120); do
    curl -sf http://0.0.0.0:${PORT}/health && break
    sleep 5
done
echo "vLLM server ready."

SLICE_ARG=${SLICE:+--slice ${SLICE}}

BASE_URL=http://0.0.0.0:${PORT}/v1 \
uv run -m tts.collect_trajectories \
    ${SLICE_ARG} \
    --dataset ${DATASET} \
    --output ${OUTPUT} \
    --workers ${WORKERS} \
    --model ${MODEL} \
    -c swebench.yaml \
    -c model.model_kwargs.temperature=1.0 \
    -c model.model_kwargs.top_k=20

echo "Killing vLLM server (pid=${VLLM_PID})..."
kill ${VLLM_PID} 2>/dev/null
wait ${VLLM_PID} 2>/dev/null
echo "vLLM server stopped."
