#!/bin/bash
# Serve a model with vLLM on MI300X (gfx942).
#
# This is the gfx942 counterpart to vllm.sh. Do NOT run it on the gfx90a
# (MI250X) nodes: every VLLM_ROCM_USE_AITER* var below and the ROCM_AITER_FA
# attention backend require gfx942, and the FP8 checkpoint needs native FP8
# MFMA that gfx90a does not have. Use vllm.sh there.
#
# Tuned per
# https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
#
# Build the image first: bash server/scripts/build-vllm.sh
# Usage: bash scripts/serve/vllm-MI300.sh

set -e
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts/serve -> repo -> parent of repo (holds the .sif images + caches)
BASE_DIR="$(dirname "$(dirname "$(dirname "$SELF_DIR")")")"

SIF="${SIF:-docker://vllm/vllm-openai-rocm:latest}"
MODEL="${MODEL:-Qwen/Qwen3.6-27B-FP8}"
PORT="${PORT:-8000}"
# FP8 weights are ~28GB and MI300X has 192GB, so a full replica fits on one
# GPU: 8 independent replicas, no tensor-parallel all-reduce.
DP="${DP:-8}"
TP="${TP:-1}"
# Set to 1 when serving >=32 concurrent requests per replica.
SHUFFLE_KV="${SHUFFLE_KV:-0}"
# Measured at 64 concurrent SWE-agent workers: 8192 -> 775 gen tok/s, 32768 ->
# 616, back to 8192 -> 751. Raising this drains the wait queue but costs ~20%
# throughput: large prefill chunks stall decode, and decode is ~73% of engine
# time here. A few queued requests are healthy -- they keep the GPU fed.
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"

mkdir -p "$BASE_DIR/hf_cache" "$BASE_DIR/triton_cache" "$BASE_DIR/tmp" "$BASE_DIR/.aiter/jit" \
         "$BASE_DIR/vllm_cache"

echo "==> Serving $MODEL with vLLM (MI300X/gfx942) on port $PORT ..."
apptainer exec \
    --rocm \
    --writable-tmpfs \
    --bind "$BASE_DIR/hf_cache:/root/.cache/huggingface" \
    --bind "$BASE_DIR/triton_cache:/triton_cache" \
    --bind "$BASE_DIR/tmp:/tmp_work" \
    --bind "$BASE_DIR/.aiter/jit:/aiter_jit" \
    --bind "$BASE_DIR/vllm_cache:/vllm_cache" \
    --env TMPDIR=/tmp_work \
    --env AITER_JIT_DIR=/aiter_jit \
    --env TRITON_CACHE_DIR=/triton_cache \
    --env VLLM_CACHE_ROOT=/vllm_cache \
    --env OMP_NUM_THREADS=1 \
    --env ROCM_PATH=/opt/rocm \
    --env HIP_FORCE_DEV_KERNARG=1 \
    --env TORCH_BLAS_PREFER_HIPBLASLT=1 \
    --env SAFETENSORS_FAST_GPU=1 \
    --env NCCL_MIN_NCHANNELS=112 \
    --env VLLM_ROCM_USE_AITER=1 \
    --env VLLM_ROCM_USE_AITER_LINEAR=1 \
    --env VLLM_ROCM_USE_AITER_MHA=1 \
    --env VLLM_ROCM_USE_AITER_RMSNORM=1 \
    --env VLLM_ROCM_USE_SKINNY_GEMM=1 \
    --env VLLM_ROCM_FP8_PADDING=1 \
    --env "VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=${SHUFFLE_KV}" \
    --env "HF_TOKEN=${HF_TOKEN:-}" \
    ${SIF} \
    vllm serve "$MODEL" \
        --tensor-parallel-size "${TP}" \
        --data-parallel-size "${DP}" \
        --disable-nccl-for-dp-synchronization \
        --max-model-len 131072 \
        --gpu-memory-utilization 0.9 \
        --dtype auto \
        --kv-cache-dtype fp8 \
        --attention-backend ROCM_AITER_FA \
        --port "${PORT}" \
        --enable-prefix-caching \
        --block-size 16 \
        --enable-auto-tool-choice \
        --language-model-only \
        --tool-call-parser qwen3_coder \
        --reasoning-parser qwen3 \
        --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
        --max-num-batched-tokens "${MAX_BATCHED_TOKENS}"
