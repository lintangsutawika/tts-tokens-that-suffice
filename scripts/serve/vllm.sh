#!/bin/bash
# Serve a model with vLLM directly out of the pre-built tts-vllm.sif.
# That image has vllm 0.20.2 built for ROCm (gfx90a/gfx942) plus the fastapi
# pin; this just launches `vllm serve` inside it with the ROCm/HF/Triton env.
# Build the image first: bash server/scripts/build-vllm.sh
# Usage: bash scripts/serve/vllm.sh

set -e
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts/serve -> repo -> parent of repo (holds the .sif images + caches)
BASE_DIR="$(dirname "$(dirname "$(dirname "$SELF_DIR")")")"

SIF="${SIF:-docker://vllm/vllm-openai-rocm:latest}"
MODEL="${MODEL:-Qwen/Qwen3.6-27B-FP8}"
PORT="${PORT:-8000}"
DP="${DP:-8}"
TP="${TP:-1}"

#if [ ! -f "$SIF" ]; then
#    echo "Error: $SIF not found. Run bash server/scripts/build-vllm.sh first."
#    exit 1
#fi

mkdir -p "$BASE_DIR/hf_cache" "$BASE_DIR/triton_cache" "$BASE_DIR/tmp"
#    --env VLLM_USE_TRITON_FLASH_ATTN=TRUE \

echo "==> Serving $MODEL with vLLM on port $PORT ..."
apptainer exec \
    --rocm \
    --writable-tmpfs \
    --bind "$BASE_DIR/hf_cache:/root/.cache/huggingface" \
    --bind "$BASE_DIR/triton_cache:/triton_cache" \
    --bind "$BASE_DIR/tmp:/tmp_work" \
    --env TMPDIR=/tmp_work \
    --env TRITON_CACHE_DIR=/triton_cache \
    --env OMP_NUM_THREADS=1 \
    --env FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
    --env ROCM_PATH=/opt/rocm \
    --env "HF_TOKEN=${HF_TOKEN:-}" \
    ${SIF} \
    vllm serve "$MODEL" \
        --tensor-parallel-size "${TP}" \
        --data-parallel-size "${DP}" \
        --max-model-len 131072 \
        --disable-custom-all-reduce \
        --gpu-memory-utilization 0.9 \
        --enable-chunked-prefill \
        --port "${PORT}" \
        --enable-prefix-caching \
        --enforce-eager \
        --block-size 16 \
        --enable-auto-tool-choice \
        --language-model-only \
        --tool-call-parser qwen3_coder \
        --reasoning-parser qwen3 \
        --max-num-batched-tokens 8192 \
        --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":3}'
