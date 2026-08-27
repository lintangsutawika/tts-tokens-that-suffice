#!/usr/bin/env bash
#SBATCH --job-name=tts-harbor-eval
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --gres=gpu:4
#SBATCH --mem-per-gpu=64G
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --time=0-24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=nvlink

# One Slurm job that (optionally) serves the deliberator with vLLM on this node's
# GPUs and runs the Harbor + Modal + mini-swe-agent eval against it, on the SAME
# node. Delegates the eval itself to scripts/eval/harbor_eval.sh.
#
# Two modes ("allow both"):
#   * DELIBERATOR_BASE_URL set  -> use that endpoint (e.g. the cmu.litellm.ai
#     proxy); no local server. GPUs aren't needed — submit with
#     `sbatch --gres=gpu:0 -p <cpu-partition>` to avoid wasting them.
#   * DELIBERATOR_BASE_URL unset -> spawn vLLM here, expose it, eval against it.
#
# REACHABILITY: the agent runs inside a Modal sandbox and calls the deliberator
# back over the network, so the served vLLM must be reachable from Modal at
# VLLM_PUBLIC_HOST (defaults to this node's FQDN). On a cluster whose compute
# nodes are not publicly routable this is the piece a tunnel used to solve —
# set VLLM_PUBLIC_HOST to a routable address, or front the vLLM with your proxy
# and pass DELIBERATOR_BASE_URL instead.
#
# Serving backend (SERVE_BACKEND): cuda (NVIDIA, default) | rocm-mi300 (gfx942) |
# rocm-mi250 (gfx90a). The ROCm variants serve inside the apptainer --rocm vLLM
# container; submit them with the AMD cluster's own SBATCH flags (they override
# the NVIDIA header below) and point VLLM_ROCM_SIF at a local .sif on batch nodes.
#
# Resume: reruns pick up completed trials only when JOB_NAME (and JOBS_DIR) match a
# prior run's job dir -- Harbor keeps trials with a valid result.json and re-runs the
# rest, but refuses to resume a dir whose JobConfig differs. Pass a STABLE JOB_NAME
# to resume; the default is timestamped, so each plain rerun starts fresh.
#
# Usage:
#   # serve Qwen locally on this node's GPUs (NVIDIA), then eval Verified
#   sbatch scripts/eval/harbor_eval.sbatch verified
#   # AMD MI300X: serve in the ROCm container, DP=8/TP=1, on hpcfund
#   SERVE_BACKEND=rocm-mi300 DATA_PARALLEL=8 TENSOR_PARALLEL=1 \
#       VLLM_ROCM_SIF=/work1/.../vllm-openai-rocm.sif VLLM_MODEL=Qwen/Qwen3.6-27B-FP8 \
#       sbatch --account=grahamneubig --partition=mi3008x --exclusive \
#           scripts/eval/harbor_eval.sbatch verified
#   # AMD MI250X (gfx90a): same, rocm-mi250 + the mi2508x partition
#   SERVE_BACKEND=rocm-mi250 DATA_PARALLEL=8 TENSOR_PARALLEL=1 \
#       VLLM_ROCM_SIF=/work1/.../vllm-openai-rocm-0.20.2.sif \
#       sbatch --account=grahamneubig --partition=mi2508x --exclusive \
#           scripts/eval/harbor_eval.sbatch verified
#   # resume a partially-completed run (same JOB_NAME + JOBS_DIR)
#   JOB_NAME=verified-qwen3.5-9b-run1 sbatch scripts/eval/harbor_eval.sbatch verified
#   # use the hosted proxy instead (no GPUs); Pro, 50 tasks
#   DELIBERATOR_BASE_URL=https://cmu.litellm.ai/v1 N_TASKS=50 \
#       sbatch --gres=gpu:0 scripts/eval/harbor_eval.sbatch pro
#   # Senior needs a judge model + creds (forwarded from env by harbor_eval.sh)
#   JUDGE_MODEL=anthropic/claude-opus-4-1 \
#       sbatch scripts/eval/harbor_eval.sbatch senior

set -euo pipefail

# A Slurm batch script's $0 is the spooled copy, not this file, so resolve the
# repo from the submit dir (where `sbatch` was run — expected to be the repo root).
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

[ -f .env ] && . .env

BENCH="${1:-${BENCH:-verified}}"

# --- deliberator / serving decision ---------------------------------------------
# Provide DELIBERATOR_BASE_URL to use an existing endpoint; leave it empty to
# serve vLLM here. SERVE_VLLM=1/0 forces the choice.
DELIBERATOR_BASE_URL="${DELIBERATOR_BASE_URL:-}"
if [ -n "${DELIBERATOR_BASE_URL}" ]; then
    SERVE_DEFAULT=0
else
    SERVE_DEFAULT=1
fi
SERVE_VLLM="${SERVE_VLLM:-${SERVE_DEFAULT}}"

# Pick a random ephemeral TCP port that nothing is currently listening on, so
# concurrent jobs on a shared node don't collide. Retries a fresh random port
# until a free one is found.
pick_free_port() {
    local p
    for _ in $(seq 1 64); do
        p=$(( (RANDOM % 16384) + 49152 ))
        ss -Htln 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${p}\$" && continue
        echo "${p}"; return 0
    done
    echo "${p}"  # give up gracefully; caller may still race, but very unlikely
}

# Model served by vLLM (deliberator). Match the eval's MODEL string to it below.
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3.6-35B-A3B}"
VLLM_PORT="${VLLM_PORT:-$(pick_free_port)}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
# Address at which Modal reaches this node's vLLM. MUST be routable from Modal.
VLLM_PUBLIC_HOST="${VLLM_PUBLIC_HOST:-$(hostname -f)}"
# vLLM ignores the key value, but mini-swe-agent requires a non-empty key.
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
# Serving backend on THIS node's GPUs:
#   cuda        -> `uv run vllm serve` (NVIDIA, the local .venv/conda vLLM)
#   rocm-mi300  -> apptainer --rocm on MI300X/gfx942 (AITER + FP8 MFMA + ROCM_AITER_FA)
#   rocm-mi250  -> apptainer --rocm on MI250X/gfx90a (Triton FA; no AITER/FP8-MFMA)
# The two ROCm variants mirror server/scripts/vllm-MI300.sh / vllm-MI250.sh.
SERVE_BACKEND="${SERVE_BACKEND:-cuda}"
DATA_PARALLEL="${DATA_PARALLEL:-1}"   # ROCm FP8 fits per-GPU: set DP=8 TP=1 there
# ROCm-only knobs. On batch nodes point VLLM_ROCM_SIF at a LOCAL .sif (a compute
# node may have no registry access); ROCM_CACHE_BASE holds the container binds.
APPTAINER_BIN="${APPTAINER_BIN:-apptainer}"
VLLM_ROCM_SIF="${VLLM_ROCM_SIF:-docker://vllm/vllm-openai-rocm:latest}"
ROCM_CACHE_BASE="${ROCM_CACHE_BASE:-$(pwd)}"
SHUFFLE_KV="${SHUFFLE_KV:-0}"         # MI300 only; set 1 for >=32 concurrent/replica
# Optional speculative decoding, only for checkpoints that support it (e.g. an
# FP8 MTP model): SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":3}'.
SPEC_CONFIG="${SPEC_CONFIG:-}"
# Optional Modal reverse relay (harbor/relay/) so this private node's vLLM is
# reachable from Modal sandboxes without inbound access. Set RELAY_URL to the
# deployed relay base (https://<workspace>--tts-vllm-relay-web.modal.run) and
# RELAY_SECRET to the shared secret (Modal 'tts-relay'). When set, the node dials
# out to the relay and the eval talks to the relay instead of the node address.
RELAY_URL="${RELAY_URL:-}"
RELAY_SECRET="${RELAY_SECRET:-}"
# Extra vLLM flags (parsers etc.). Default mirrors serve/serve_local.sh. Keep
# --tool-call-parser qwen3_coder for Qwen (hermes silently drops the <function=..>
# XML tool calls -> RepeatedFormatError). --language-model-only skips the vision
# tower on multimodal Qwen3.5 (9B/27B), which we don't use.
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---enable-chunked-prefill --max-num-batched-tokens 4096 --enable-prefix-caching --long-prefill-token-threshold 0 --enforce-eager --enable-auto-tool-choice --language-model-only --tool-call-parser qwen3_coder --reasoning-parser qwen3}"

VLLM_PID=""
BRIDGE_PID=""
cleanup() {
    [ -n "${BRIDGE_PID}" ] && { echo "[cleanup] stopping relay bridge (pid=${BRIDGE_PID})";
        kill "${BRIDGE_PID}" 2>/dev/null; }
    [ -n "${VLLM_PID}" ] && { echo "[cleanup] stopping vLLM (pid=${VLLM_PID})";
        kill "${VLLM_PID}" 2>/dev/null; wait "${VLLM_PID}" 2>/dev/null; }
}
trap cleanup EXIT SIGTERM SIGINT

if [ "${SERVE_VLLM}" = "1" ]; then
    echo "[serve] launching vLLM ${VLLM_MODEL} on ${VLLM_PUBLIC_HOST}:${VLLM_PORT}" \
         "(backend=${SERVE_BACKEND} TP=${TENSOR_PARALLEL} DP=${DATA_PARALLEL})"
    # Speculative decoding is opt-in (model-specific); off unless SPEC_CONFIG is set.
    if [ -n "${SPEC_CONFIG}" ]; then SPEC_ARGS=( --speculative-config "${SPEC_CONFIG}" ); else SPEC_ARGS=(); fi

    case "${SERVE_BACKEND}" in
      cuda)
        # vLLM runs from the conda/uv env whose libzmq needs GLIBCXX_3.4.30; some older
        # compute nodes (e.g. neulab babel-t9-*) ship a system libstdc++ without it.
        # Put conda's own libstdc++ ahead of the system one so the loader finds it.
        CONDA_LIB="${CONDA_PREFIX:-$HOME/miniconda3}/lib"
        [ -f "${CONDA_LIB}/libstdc++.so.6" ] && export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH:-}"
        # Mirrors scripts/serve/serve_local.sh (Qwen3 deliberator, tool + reasoning parsers).
        uv run vllm serve "${VLLM_MODEL}" \
            --host 0.0.0.0 \
            --port "${VLLM_PORT}" \
            --tensor-parallel-size "${TENSOR_PARALLEL}" \
            --data-parallel-size "${DATA_PARALLEL}" \
            --max-model-len "${MAX_MODEL_LEN}" \
            --disable-custom-all-reduce \
            --gpu-memory-utilization "${GPU_MEM_UTIL}" \
            "${SPEC_ARGS[@]}" ${VLLM_EXTRA_ARGS} &
        VLLM_PID=$!
        ;;
      rocm-mi300|rocm-mi250)
        # Serve inside the ROCm vLLM container (apptainer --rocm). Mirrors
        # server/scripts/vllm-MI300.sh (gfx942) / vllm-MI250.sh (gfx90a): the two
        # differ only in the accelerator env block below -- AITER (VLLM_ROCM_USE_AITER*,
        # the ROCM_AITER_FA backend) and FP8 padding are gfx942-only; gfx90a uses the
        # Triton FlashAttention path and runs the FP8 checkpoint dequantized.
        mkdir -p "${ROCM_CACHE_BASE}/hf_cache" "${ROCM_CACHE_BASE}/triton_cache" \
                 "${ROCM_CACHE_BASE}/tmp" "${ROCM_CACHE_BASE}/vllm_cache"
        ROCM_BINDS=(
            --bind "${ROCM_CACHE_BASE}/hf_cache:/root/.cache/huggingface"
            --bind "${ROCM_CACHE_BASE}/triton_cache:/triton_cache"
            --bind "${ROCM_CACHE_BASE}/tmp:/tmp_work"
            --bind "${ROCM_CACHE_BASE}/vllm_cache:/vllm_cache"
        )
        ROCM_ENV=(
            --env TMPDIR=/tmp_work
            --env TRITON_CACHE_DIR=/triton_cache
            --env VLLM_CACHE_ROOT=/vllm_cache
            --env HF_HOME=/root/.cache/huggingface
            --env OMP_NUM_THREADS=1
            --env ROCM_PATH=/opt/rocm
            --env TORCH_BLAS_PREFER_HIPBLASLT=1
            --env SAFETENSORS_FAST_GPU=1
            --env "HF_TOKEN=${HF_TOKEN:-}"
        )
        # Shared serve flags; accelerator-specific ones appended per variant.
        ROCM_SERVE_EXTRA=( --dtype auto --block-size 16 --disable-nccl-for-dp-synchronization )
        if [ "${SERVE_BACKEND}" = "rocm-mi300" ]; then
            mkdir -p "${ROCM_CACHE_BASE}/.aiter/jit"
            ROCM_BINDS+=( --bind "${ROCM_CACHE_BASE}/.aiter/jit:/aiter_jit" )
            ROCM_ENV+=(
                --env AITER_JIT_DIR=/aiter_jit
                --env HIP_FORCE_DEV_KERNARG=1
                --env NCCL_MIN_NCHANNELS=112
                --env VLLM_ROCM_USE_AITER=1
                --env VLLM_ROCM_USE_AITER_LINEAR=1
                --env VLLM_ROCM_USE_AITER_MHA=1
                --env VLLM_ROCM_USE_AITER_RMSNORM=1
                --env VLLM_ROCM_USE_SKINNY_GEMM=1
                --env VLLM_ROCM_FP8_PADDING=1
                --env "VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=${SHUFFLE_KV}"
            )
            ROCM_SERVE_EXTRA+=( --kv-cache-dtype fp8 --attention-backend ROCM_AITER_FA )
        else  # rocm-mi250 (gfx90a): Triton FA, no AITER/FP8-MFMA
            ROCM_ENV+=( --env FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE )
            ROCM_SERVE_EXTRA+=( --disable-custom-all-reduce )
        fi
        "${APPTAINER_BIN}" exec --rocm --writable-tmpfs \
            "${ROCM_BINDS[@]}" "${ROCM_ENV[@]}" \
            "${VLLM_ROCM_SIF}" \
            vllm serve "${VLLM_MODEL}" \
                --host 0.0.0.0 \
                --port "${VLLM_PORT}" \
                --tensor-parallel-size "${TENSOR_PARALLEL}" \
                --data-parallel-size "${DATA_PARALLEL}" \
                --max-model-len "${MAX_MODEL_LEN}" \
                --gpu-memory-utilization "${GPU_MEM_UTIL}" \
                "${ROCM_SERVE_EXTRA[@]}" "${SPEC_ARGS[@]}" ${VLLM_EXTRA_ARGS} &
        VLLM_PID=$!
        ;;
      *)
        echo "ERROR: unknown SERVE_BACKEND '${SERVE_BACKEND}' (want cuda|rocm-mi300|rocm-mi250)" >&2
        exit 2
        ;;
    esac

    echo "[serve] waiting for vLLM /health on :${VLLM_PORT} ..."
    READY=0
    for _ in $(seq 1 180); do
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "ERROR: vLLM process exited during startup" >&2; exit 1
        fi
        if curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
            READY=1; break
        fi
        sleep 5
    done
    [ "${READY}" = "1" ] || { echo "ERROR: vLLM not ready after 900s" >&2; exit 1; }
    echo "[serve] vLLM ready."

    # Point the eval at the locally-served model.
    export MODEL="${MODEL:-openai/${VLLM_MODEL}}"
    if [ -n "${RELAY_URL}" ]; then
        # Reach vLLM through the Modal relay: start the node bridge (dials out to
        # the relay and forwards to local vLLM), and have the agent call the relay.
        [ -n "${RELAY_SECRET}" ] || { echo "ERROR: RELAY_URL set but RELAY_SECRET empty" >&2; exit 2; }
        echo "[relay] starting node bridge -> ${RELAY_URL}"
        BRIDGE_WS="${RELAY_URL%/}/bridge"; BRIDGE_WS="${BRIDGE_WS/https:/wss:}"
        RELAY_WS_URL="${BRIDGE_WS}" RELAY_SECRET="${RELAY_SECRET}" \
        VLLM_LOCAL_URL="http://localhost:${VLLM_PORT}" \
            uv run --with websockets --with httpx python harbor/relay/bridge.py &
        BRIDGE_PID=$!
        export DELIBERATOR_BASE_URL="${RELAY_URL%/}/v1"
        export DELIBERATOR_API_KEY="${RELAY_SECRET}"
        echo "[relay] waiting for bridge to register with relay ..."
        for _ in $(seq 1 24); do
            curl -sf "${RELAY_URL%/}/health" 2>/dev/null | grep -q '"bridge_connected":true' \
                && { echo "[relay] bridge connected"; break; }
            sleep 5
        done
    else
        # Direct node address (only works where Modal can route to it, e.g. -e singularity on-node).
        export DELIBERATOR_BASE_URL="http://${VLLM_PUBLIC_HOST}:${VLLM_PORT}/v1"
        export DELIBERATOR_API_KEY="${VLLM_API_KEY}"
    fi
else
    echo "[serve] using provided endpoint ${DELIBERATOR_BASE_URL} (no local vLLM)"
    export DELIBERATOR_BASE_URL
fi

echo "=================================================================="
echo "Harbor eval: BENCH=${BENCH}  deliberator=${DELIBERATOR_BASE_URL}"
echo "=================================================================="

# Delegate to the runner (it derives --allow-agent-host from the URL, wires Modal,
# and forwards judge creds for senior). Not `exec`, so cleanup runs afterward.
bash scripts/eval/harbor_eval.sh "${BENCH}"
