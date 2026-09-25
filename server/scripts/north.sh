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
# IMAGE: the official vLLM ROCm *nightly*, which is built from main (the
# nightly-<sha> tags are literally main commits, tracking within ~an hour) and
# already ships everything North needs, prebuilt:
#   cohere2_moe (the model arch), aiter (AITER kernels), cohere_melody (the
#   reasoning parser's dependency).
# That is why AITER works here but NOT in our from-source north-vllm.sif, which
# has no aiter -- forcing ROCM_AITER_FA there dies at the first forward with
# ModuleNotFoundError: No module named 'aiter'. Keep the AITER settings below in
# sync with the image you point SIF at.
#
# Pull/refresh the image (re-pull to move to a newer main):
#   apptainer pull --force north-vllm-nightly.sif docker://vllm/vllm-openai-rocm:nightly
# For a reproducible pin, use the sha tag instead of the floating one:
#   docker://vllm/vllm-openai-rocm:nightly-<full-sha>
#
# Usage: bash server/scripts/north.sh

#SBATCH --job-name=north-vllm
#SBATCH --account=grahamneubig
#SBATCH --partition=mi3001x
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=0-04:00:00
# Absolute: a relative path here resolves against the *submit* cwd, so logs land
# somewhere different depending on where you ran sbatch from.
#SBATCH --output=/work1/grahamneubig/lsutawik/logs/%j.out
#SBATCH --error=/work1/grahamneubig/lsutawik/logs/%j.out

set -euo pipefail

MODE="${1:-rl}"

PROJECT_DIR=/work1/grahamneubig/lsutawik/tts-tokens-that-suffice
BASE_DIR="$PROJECT_DIR"
SERVER_DIR="$PROJECT_DIR/server"
SELF_DIR="$SERVER_DIR/scripts"

# BASE_DIR is the PARENT of the checkout, not the checkout itself: that is where
# all the heavy shared state lives -- the .sif images, hf_cache (~143G), and the
# triton/aiter/vllm caches. Resolve it the same way in both modes. Deriving it as
# $PROJECT_DIR under sbatch (as an earlier version did) points the job at an
# empty cache root, so it re-downloads the model and recompiles the AITER JIT
# objects instead of reusing the warm ones.
BASE_DIR="$(dirname "$PROJECT_DIR")"

# Export secrets/env (HF_TOKEN, etc.) to run.sh if present. set -a so every
# `export`-ed var in .env reaches the apptainer --env passthrough in run.sh.
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.env"
    set +a
fi

echo "Base Dir $BASE_DIR"

SIF="${SIF:-$BASE_DIR/north-vllm-nightly.sif}"
MODEL="${MODEL:-CohereLabs/North-Mini-Code-1.0}"
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

# AITER JIT cache, PER IMAGE. These .so files are compiled against the running
# container's libstdc++/ROCm, so a cache shared across images with different
# toolchains breaks: an object built in an Ubuntu 24.04 image (GCC 13) records a
# GLIBCXX_3.4.31 dependency that an Ubuntu 22.04 image (libstdc++ 6.0.30, GCC 12)
# cannot load, giving
#   ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.31'
#   not found (required by /aiter_jit/module_aiter_core.so)
# Keying the dir on the image name keeps each image's objects separate. A fresh
# dir means the first run recompiles (slow); that is intended and correct.
AITER_JIT_HOST="${AITER_JIT_HOST:-$BASE_DIR/.aiter/jit-$(basename "$SIF" .sif | tr -d '\n' | tr -c 'A-Za-z0-9_.-' '_')}"

mkdir -p "$BASE_DIR/hf_cache" "$BASE_DIR/triton_cache" "$BASE_DIR/tmp" "$AITER_JIT_HOST" \
         "$BASE_DIR/vllm_cache"
echo "==> AITER JIT cache: $AITER_JIT_HOST"

# Defaulted so the script still runs interactively, where these are unset and
# `set -u` would otherwise abort here.
echo "=== JOB ${SLURM_JOB_ID:-none} on ${SLURMD_NODENAME:-$(hostname)} (partition ${SLURM_JOB_PARTITION:-none}) ==="
echo "==> Serving $MODEL with vLLM (MI300X/gfx942) on ${SLURMD_NODENAME:-$(hostname)}:$PORT ..."
apptainer exec \
    --rocm \
    --bind "$BASE_DIR/hf_cache:/root/.cache/huggingface" \
    --bind "$BASE_DIR/triton_cache:/triton_cache" \
    --bind "$BASE_DIR/tmp:/tmp_work" \
    --bind "$AITER_JIT_HOST:/aiter_jit" \
    --bind "$BASE_DIR/vllm_cache:/vllm_cache" \
    --env TMPDIR=/tmp_work \
    --env AITER_JIT_DIR=/aiter_jit \
    --env TRITON_CACHE_DIR=/triton_cache \
    --env VLLM_CACHE_ROOT=/vllm_cache \
    --env OMP_NUM_THREADS=1 \
    --env VLLM_NO_USAGE_STATS=1 \
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
        --tool-call-parser cohere_command4 \
        --reasoning-parser cohere_command4 \
        --max-num-batched-tokens "${MAX_BATCHED_TOKENS}"
