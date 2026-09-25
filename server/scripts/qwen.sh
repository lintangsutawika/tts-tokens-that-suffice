#!/bin/bash
# Serve a Qwen3.6 model with vLLM on MI300X (gfx942).
#
# Sibling of north.sh -- same image, same cache layout, same AITER settings;
# only the model and its parsers differ. Do NOT run it on the gfx90a (MI250X)
# nodes: every VLLM_ROCM_USE_AITER* var below requires gfx942, and the 27B FP8
# checkpoint needs native FP8 MFMA that gfx90a does not have.
#
# Tuned per
# https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
#
# IMAGE: the official vLLM ROCm *nightly*, which is built from main (the
# nightly-<sha> tags are literally main commits, tracking within ~an hour) and
# ships aiter prebuilt. Our from-source north-vllm.sif does NOT have aiter --
# pointing SIF there with the AITER vars set below dies during model load with
#   ValueError: Unquantized MoE backend ROCm AITER does not support ...
# Keep the AITER settings in sync with the image you point SIF at.
#
# Pull/refresh the image (re-pull to move to a newer main):
#   apptainer pull --force north-vllm-nightly.sif docker://vllm/vllm-openai-rocm:nightly
# For a reproducible pin, use the sha tag instead of the floating one:
#   docker://vllm/vllm-openai-rocm:nightly-<full-sha>
#
# Usage:
#   sbatch server/scripts/qwen.sh 27b     # Qwen3.6-27B-FP8  (default)
#   sbatch server/scripts/qwen.sh 35b     # Qwen3.6-35B-A3B  (bf16)
#   bash   server/scripts/qwen.sh 35b     # interactive, on an salloc'd node

#SBATCH --job-name=qwen-vllm
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

SIZE="${1:-27b}"

PROJECT_DIR=/work1/grahamneubig/lsutawik/tts-tokens-that-suffice
SERVER_DIR="$PROJECT_DIR/server"

# BASE_DIR is the PARENT of the checkout, not the checkout itself: that is where
# the heavy shared state lives -- the .sif images, hf_cache (~143G), and the
# triton/aiter/vllm caches. Resolve it the same way under sbatch and
# interactively, or the job lands on an empty cache root and silently
# re-downloads the model and recompiles the AITER JIT objects.
BASE_DIR="$(dirname "$PROJECT_DIR")"

# Export secrets/env (HF_TOKEN, etc.) if present. set -a so every `export`-ed
# var in .env reaches the apptainer --env passthrough below.
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.env"
    set +a
fi

# Per-size settings. Both are Qwen3.6 multimodal MoE checkpoints with an MTP
# head (config.json: mtp_num_hidden_layers=1), so both get speculative decoding
# and both get --language-model-only to skip the vision tower.
case "$SIZE" in
    27b)
        MODEL="${MODEL:-Qwen/Qwen3.6-27B-FP8}"
        ;;
    35b)
        # bf16, NOT FP8: ~70GB of weights vs ~28GB for the 27B. Still one full
        # replica per 192GB MI300X, but it leaves markedly less room for KV
        # cache at the same --gpu-memory-utilization.
        MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B}"
        ;;
    *)
        echo "error: unknown size '$SIZE' (expected 27b or 35b)" >&2
        echo "       or set MODEL=<hf-repo-id> and pass any size to skip the preset" >&2
        exit 1
        ;;
esac

SIF="${SIF:-$BASE_DIR/north-vllm-nightly.sif}"
PORT="${PORT:-8000}"
# Weights fit in one GPU's 192GB, so run 8 independent replicas rather than
# tensor-parallel -- no all-reduce on the critical path.
DP="${DP:-8}"
TP="${TP:-1}"
# Set to 1 when serving >=32 concurrent requests per replica.
SHUFFLE_KV="${SHUFFLE_KV:-0}"
# Measured at 64 concurrent SWE-agent workers: 8192 -> 775 gen tok/s, 32768 ->
# 616, back to 8192 -> 751. Raising this drains the wait queue but costs ~20%
# throughput: large prefill chunks stall decode, and decode is ~73% of engine
# time here. A few queued requests are healthy -- they keep the GPU fed.
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"
# Both models declare max_position_embeddings=262144; 131072 is the tested
# working point and halves the KV reservation. Raise if you need the full window.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
# Set SPEC_DECODE=0 to disable MTP speculative decoding (useful when bisecting
# correctness problems -- it changes which sampling path runs).
SPEC_DECODE="${SPEC_DECODE:-1}"

# Attention backend: forced, and it has to be. Auto-selection does NOT pick
# AITER FlashAttention -- get_valid_backends() in vllm/platforms/rocm.py appends
# ROCM_ATTN *before* ROCM_AITER_FA, and ROCM_ATTN wins whenever you are not
# using a KV connector. So without this flag you silently get ROCM_ATTN even
# with every VLLM_ROCM_USE_AITER* var set.
#
# The catch: passing --attention-backend bypasses both the env check and the
# is-aiter-actually-installed check, so an image WITHOUT aiter starts up fine
# and then dies at the first forward with
#   ModuleNotFoundError: No module named 'aiter'
# That is safe here only because the default SIF (the nightly) ships aiter. If
# you point SIF at north-vllm.sif, which does not, set ATTN_BACKEND="" as well
# as clearing the VLLM_ROCM_USE_AITER vars below.
ATTN_BACKEND="${ATTN_BACKEND:-ROCM_AITER_FA}"

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

# Assembled as an array so the empty/optional flags simply vanish rather than
# expanding to an empty string that vLLM would parse as a positional arg.
EXTRA_ARGS=()
if [ -n "$ATTN_BACKEND" ]; then
    EXTRA_ARGS+=(--attention-backend "$ATTN_BACKEND")
fi
if [ "$SPEC_DECODE" = "1" ]; then
    EXTRA_ARGS+=(--speculative-config '{"method":"mtp","num_speculative_tokens":3}')
fi

# Defaulted so the script still runs interactively, where the SLURM vars are
# unset and `set -u` would otherwise abort here.
echo "=== JOB ${SLURM_JOB_ID:-none} on ${SLURMD_NODENAME:-$(hostname)} (partition ${SLURM_JOB_PARTITION:-none}) ==="
echo "==> Base dir:         $BASE_DIR"
echo "==> Image:            $SIF"
echo "==> AITER JIT cache:  $AITER_JIT_HOST"
echo "==> Serving $MODEL ($SIZE) with vLLM (MI300X/gfx942) on ${SLURMD_NODENAME:-$(hostname)}:$PORT ..."
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
        --max-model-len "${MAX_MODEL_LEN}" \
        --gpu-memory-utilization 0.9 \
        --dtype auto \
        --kv-cache-dtype fp8 \
        --port "${PORT}" \
        --enable-prefix-caching \
        --block-size 16 \
        --enable-auto-tool-choice \
        --language-model-only \
        --tool-call-parser "${TOOL_PARSER}" \
        --reasoning-parser "${REASONING_PARSER}" \
        --max-num-batched-tokens "${MAX_BATCHED_TOKENS}" \
        "${EXTRA_ARGS[@]}"
