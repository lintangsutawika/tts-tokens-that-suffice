#!/usr/bin/env bash
# Serve the summarizer model (Qwen3-8B) via vLLM on an OpenAI-compatible endpoint,
# as the sibling of scripts/serve/serve_local.sh (which serves the deliberator at :8000).
#
# Point the eval at it with:
#   uv run -m tts.eval_swebench --mode base \
#       --summarizer-api-base http://0.0.0.0:8001/v1 \
#       --summarizer-model openai/Qwen/Qwen3-8B ...
#
# The trained arm needs the RL LoRA adapter served too. Export the tinker
# checkpoint to an HF adapter dir, then attach it here via LORA_MODULES, e.g.:
#   LORA_MODULES="summarizer-rl=/path/to/exported_adapter" bash scripts/serve/serve_summarizer.sh
# and call it with --summarizer-model openai/summarizer-rl --mode trained.
#
# Run on an allocated GPU node (like serve_local.sh), or wrap with sbatch.

set -euo pipefail

[ -f .env ] && . .env

MODEL=${SUMMARIZER_MODEL:-Qwen/Qwen3-8B}
PORT=${SUMMARIZER_PORT:-8001}
TP=${TP:-1}
DP=${DP:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}

# Optional LoRA adapters: "name1=/path1,name2=/path2" (served alongside the base).
LORA_MODULES=${LORA_MODULES:-}

LORA_ARGS=()
if [ -n "${LORA_MODULES}" ]; then
    LORA_ARGS+=(--enable-lora --max-lora-rank 32)
    IFS=',' read -ra _mods <<< "${LORA_MODULES}"
    for m in "${_mods[@]}"; do
        LORA_ARGS+=(--lora-modules "${m}")
    done
fi

uv run vllm serve "${MODEL}" \
    --tensor-parallel-size "${TP}" \
    --data-parallel-size "${DP}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --disable-custom-all-reduce \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --enable-chunked-prefill \
    --port "${PORT}" \
    --enable-prefix-caching \
    --enforce-eager \
    --language-model-only \
    --reasoning-parser qwen3 \
    "${LORA_ARGS[@]}"
