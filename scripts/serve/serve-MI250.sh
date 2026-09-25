#!/bin/bash
# Serve a model with vLLM on MI250X (gfx90a).
#
# This is the gfx90a counterpart to vllm-MI300.sh. The two differ only in the
# accelerator-specific layer: everything AITER (VLLM_ROCM_USE_AITER*, the
# ROCM_AITER_FA backend), FP8 weight padding, and the KV-cache shuffle are
# gfx942-only and are absent here. gfx90a has no native FP8 MFMA, so the FP8
# checkpoint runs dequantized -- it loads and it saves memory, but it does not
# buy the math speedup it buys on MI300X.
#
# Runs either interactively or as a batch job -- the #SBATCH block below is
# inert when the script is invoked with bash.
#
#   bash scripts/serve/vllm-MI250.sh          # interactive (on an salloc'd node)
#   sbatch scripts/serve/vllm-MI250.sh        # batch; survives disconnects
#   sbatch scripts/serve/vllm-MI250.sh 9000   # ... on a different port
#   PORT=9000 MODEL=... sbatch scripts/serve/vllm-MI250.sh    # same, via env
#
# Port matters here because two of these can share a node only if they differ;
# the job log prints the node:port to point clients at.
#
# Watch:  tail -f /work1/grahamneubig/lsutawik/logs/tts-vllm-mi250-<jobid>.out
# Node:   squeue -u $USER -n tts-vllm-mi250 -o '%N'
# Stop:   scancel <jobid>
#
# mi2508x is the MI250X partition (k004-001..010), 8 GCDs/node. --exclusive
# because we take all 8. 12h is the site max -- the partition advertises
# 4-00:00:00 but anything over 12h will not run, so do not raise it.

#SBATCH --job-name=tts-vllm-mi250
#SBATCH --account=grahamneubig
#SBATCH --partition=mi2508x
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=/work1/grahamneubig/lsutawik/logs/%x-%j.out
#SBATCH --error=/work1/grahamneubig/lsutawik/logs/%x-%j.out

set -e
set -o pipefail   # without this, tee's exit status would mask a vllm crash

# sbatch executes a *spooled copy* of this file out of /var/spool, so BASH_SOURCE
# points somewhere useless and every bind path derived from it would be wrong.
# Under SLURM, hardcode; interactively, derive as before.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    BASE_DIR=/work1/grahamneubig/lsutawik
    PROJECT_DIR="$BASE_DIR/tts-tokens-that-suffice"
    echo "=== JOB $SLURM_JOB_ID on $SLURMD_NODENAME (partition $SLURM_JOB_PARTITION) ==="
    cd "$PROJECT_DIR"
    # Pick up HF_TOKEN etc. so the apptainer --env passthrough below sees them.
    if [ -f "$PROJECT_DIR/.env" ]; then
        set -a
        # shellcheck disable=SC1091
        source "$PROJECT_DIR/.env"
        set +a
    fi
else
    SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    # scripts/serve -> repo -> parent of repo (holds the .sif images + caches)
    BASE_DIR="$(dirname "$(dirname "$(dirname "$SELF_DIR")")")"
fi

# Local SIF, not docker://. A batch node may have no registry access and there is
# nobody around to answer an auth prompt; pulling at job start also burns minutes.
SIF="${SIF:-/work1/grahamneubig/lsutawik/vllm-openai-rocm-0.20.2.sif}"
MODEL="${MODEL:-Qwen/Qwen3.6-27B-FP8}"
# Port: first positional arg wins, else $PORT, else 8000. sbatch forwards script
# args and (via the default --export=ALL) the submit-time environment, so both
# `sbatch vllm-MI250.sh 9000` and `PORT=9000 sbatch vllm-MI250.sh` work.
PORT="${1:-${PORT:-8000}}"
# FP8 weights are ~28GB and each MI250X GCD has 64GB, so a replica still fits on
# one GCD: 8 independent replicas, no tensor-parallel all-reduce. A bf16
# checkpoint (~54GB) would NOT leave room for KV -- that needs TP=2, DP=4.
DP="${DP:-8}"
TP="${TP:-1}"
# Measured on MI300X: 8192 beats 32768 by ~20% on generation throughput. Large
# prefill chunks stall decode, and decode is ~73% of engine time on SWE-agent
# traffic. A few queued requests are healthy -- they keep the GPU fed.
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-4096}"
# vllm.sh runs eager. HIP graphs should help decode here too, but that is
# untested on gfx90a with this model -- flip to 0 and measure before trusting it.
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
# Leave at the vLLM default (= data_parallel_size). Setting this to 1 was tried
# and is WORSE: measured over 7 min with 64 mini-swe workers, api-server-count=8
# used 3 of 8 engines, and api-server-count=1 used only 2 of 8 (35 gen tok/s,
# 21 queued on e0, 6 preemptions, engines 3-7 at literally zero). The multiple
# API servers were *mitigating* the pile-up, not causing it: each holds its own
# 100ms-stale engine load table and anchors its scan at eng_start_index =
# (len(engines) * client_index) // client_count, so 8 servers spread bursts
# across 8 starting points. Collapse to one and every burst stacks from engine 0.
#
# The DP balancer is broken either way and has no knob. If you care about using
# all 8 GCDs, don't run this script -- run vllm-MI250-fleet.sh, which drops DP
# for 8 single-GPU servers behind router.py and measured 8/8 engines busy.
API_SERVER_COUNT="${API_SERVER_COUNT:-$DP}"

LOG_DIR="${LOG_DIR:-$BASE_DIR/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/vllm-MI250-$(hostname -s)-$(date +%Y%m%d-%H%M%S).log}"

mkdir -p "$BASE_DIR/hf_cache" "$BASE_DIR/triton_cache" "$BASE_DIR/tmp" "$LOG_DIR" \
         "$BASE_DIR/vllm_cache"

EAGER_FLAG=()
if [ "$ENFORCE_EAGER" = "1" ]; then
    EAGER_FLAG=(--enforce-eager)
fi

echo "==> Serving $MODEL with vLLM (MI250X/gfx90a) on port $PORT ..."
echo "==> Logging to $LOG_FILE"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "==> Point clients at http://$(hostname -s):$PORT/v1"
    echo "==> (or: ssh -N -L $PORT:$(hostname -s):$PORT <login-host>)"
fi
apptainer exec \
    --rocm \
    --writable-tmpfs \
    --bind "$BASE_DIR/hf_cache:/root/.cache/huggingface" \
    --bind "$BASE_DIR/triton_cache:/triton_cache" \
    --bind "$BASE_DIR/tmp:/tmp_work" \
    --bind "$BASE_DIR/vllm_cache:/vllm_cache" \
    --env TMPDIR=/tmp_work \
    --env TRITON_CACHE_DIR=/triton_cache \
    --env VLLM_CACHE_ROOT=/vllm_cache \
    --env HF_HOME=/root/.cache/huggingface \
    --env OMP_NUM_THREADS=1 \
    --env ROCM_PATH=/opt/rocm \
    --env FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
    --env TORCH_BLAS_PREFER_HIPBLASLT=1 \
    --env SAFETENSORS_FAST_GPU=1 \
    --env "HF_TOKEN=${HF_TOKEN:-}" \
    ${SIF} \
    vllm serve "$MODEL" \
        --tensor-parallel-size "${TP}" \
        --data-parallel-size "${DP}" \
        --api-server-count "${API_SERVER_COUNT}" \
        --disable-nccl-for-dp-synchronization \
        --disable-custom-all-reduce \
        --max-model-len 32000 \
        --gpu-memory-utilization 0.75 \
        --dtype auto \
        --port "${PORT}" \
        --enable-prefix-caching \
        --enable-chunked-prefill \
        "${EAGER_FLAG[@]}" \
        --enable-auto-tool-choice \
        --language-model-only \
        --tool-call-parser qwen3_coder \
        --reasoning-parser qwen3 \
        --max-num-batched-tokens "${MAX_BATCHED_TOKENS}" \
    2>&1 | tee "$LOG_FILE"
