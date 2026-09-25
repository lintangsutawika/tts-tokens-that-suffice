#!/bin/bash
# Serve a model on MI250X (gfx90a) as 8 independent single-GPU vLLM servers
# behind a prefix-affinity router.
#
# Why not just --data-parallel-size 8 (see vllm-MI250.sh)? Because vLLM's
# internal DP load balancer does not balance. Measured with 64 mini-swe workers:
# it pinned every request to 3 of 8 GCDs (--api-server-count 8) or 2 of 8
# (--api-server-count 1) and left the rest at literally zero for 7 minutes,
# while the loaded ones queued and preempted. Its load table is a 100ms-stale
# snapshot and there is no knob to fix it.
#
# So: don't use DP at all. Eight standalone servers, each pinned to one GCD, and
# router.py in front doing exact-in-flight least-loaded assignment with sticky
# conversation affinity (a SWE agent resends its whole history every step, so
# keeping a trajectory on one GCD turns that history into a prefix-cache hit).
#
# mini-swe still points at a single URL -- the router owns ROUTER_PORT.
#
# Usage: bash scripts/serve/vllm-MI250-fleet.sh
#        bash scripts/serve/vllm-MI250-fleet.sh stop

set -e
set -o pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts/serve -> repo -> parent of repo (holds the .sif images + caches)
BASE_DIR="$(dirname "$(dirname "$(dirname "$SELF_DIR")")")"

SIF="${SIF:-/work1/grahamneubig/lsutawik/vllm-openai-rocm-0.20.2.sif}"
MODEL="${MODEL:-Qwen/Qwen3.6-27B-FP8}"
NGPU="${NGPU:-8}"
ROUTER_PORT="${ROUTER_PORT:-9142}"   # what mini-swe talks to
BACKEND_PORT0="${BACKEND_PORT0:-9150}"
# This workload is PREFILL-bound on gfx90a: measured 1826 prompt tok/s against
# 57 generation tok/s, a 32:1 ratio (SWE agents resend the whole history every
# step). So the two knobs below both target prefill work, and note that (2) is
# the *opposite* of the right answer on MI300 -- there decode was 73% of engine
# time and big prefill chunks stalled it, costing 20%. Here prefill IS the job.
#
# (1) DO NOT set this to fp8 on gfx90a. It halves KV memory (~2x the prefix
#     blocks retained), which is exactly the right lever for a prefill-bound
#     workload -- but it wrecked generation quality: mini-swe started throwing
#     RepeatedFormatError, i.e. the model was emitting degenerate repeated text.
#     gfx90a has no native FP8, so the KV cache takes a different conversion
#     path (e4m3fnuz) than MI300, and this model is a hybrid SSM/linear-attention
#     architecture (48 linear + 16 full attention layers). It works on MI300;
#     it does not work here. Left as a knob so the finding isn't lost.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
# (2) Bigger prefill chunks => far more efficient prefill GEMMs.
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-16384}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
LOG_DIR="${LOG_DIR:-$BASE_DIR/logs/fleet-$(date +%Y%m%d-%H%M%S)}"

PIDFILE="$BASE_DIR/tmp/mi250-fleet.pids"

if [ "${1:-}" = "stop" ]; then
    if [ -f "$PIDFILE" ]; then
        echo "==> Stopping fleet ..."
        xargs -r kill < "$PIDFILE" 2>/dev/null || true
        rm -f "$PIDFILE"
    fi
    pkill -u "$USER" -f "vllm serve" 2>/dev/null || true
    pkill -u "$USER" -f "router.py" 2>/dev/null || true
    echo "==> Stopped."
    exit 0
fi

mkdir -p "$BASE_DIR/hf_cache" "$BASE_DIR/triton_cache" "$BASE_DIR/tmp" "$LOG_DIR"
: > "$PIDFILE"

EAGER_FLAG=()
[ "$ENFORCE_EAGER" = "1" ] && EAGER_FLAG=(--enforce-eager)

echo "==> Launching $NGPU backends, logs in $LOG_DIR"
for i in $(seq 0 $((NGPU - 1))); do
    PORT=$((BACKEND_PORT0 + i))
    # Per-rank torch.compile cache: 8 processes compiling the same graph into
    # one directory race each other. Disk is cheap here, corrupted caches aren't.
    mkdir -p "$BASE_DIR/vllm_cache/rank$i"
    apptainer exec \
        --rocm \
        --writable-tmpfs \
        --bind "$BASE_DIR/hf_cache:/root/.cache/huggingface" \
        --bind "$BASE_DIR/triton_cache:/triton_cache" \
        --bind "$BASE_DIR/tmp:/tmp_work" \
        --bind "$BASE_DIR/vllm_cache/rank$i:/vllm_cache" \
        --env TMPDIR=/tmp_work \
        --env TRITON_CACHE_DIR=/triton_cache \
        --env VLLM_CACHE_ROOT=/vllm_cache \
        --env "HIP_VISIBLE_DEVICES=${i}" \
        --env OMP_NUM_THREADS=1 \
        --env ROCM_PATH=/opt/rocm \
        --env FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
        --env TORCH_BLAS_PREFER_HIPBLASLT=1 \
        --env SAFETENSORS_FAST_GPU=1 \
        --env "HF_TOKEN=${HF_TOKEN:-}" \
        "$SIF" \
        vllm serve "$MODEL" \
            --tensor-parallel-size 1 \
            --max-model-len 131072 \
            --gpu-memory-utilization 0.9 \
            --dtype auto \
            --kv-cache-dtype "$KV_CACHE_DTYPE" \
            --port "$PORT" \
            --enable-prefix-caching \
            --enable-chunked-prefill \
            "${EAGER_FLAG[@]}" \
            --block-size 16 \
            --enable-auto-tool-choice \
            --language-model-only \
            --tool-call-parser qwen3_coder \
            --reasoning-parser qwen3 \
            --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
            --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
        > "$LOG_DIR/backend-$i.log" 2>&1 &
    echo $! >> "$PIDFILE"
    echo "    gcd $i -> port $PORT (pid $!)"
done

BACKENDS=""
for i in $(seq 0 $((NGPU - 1))); do
    BACKENDS="${BACKENDS}${BACKENDS:+,}http://127.0.0.1:$((BACKEND_PORT0 + i))"
done

echo "==> Waiting for backends to come up (first run compiles; this is slow) ..."
for i in $(seq 0 $((NGPU - 1))); do
    PORT=$((BACKEND_PORT0 + i))
    until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; do
        if ! kill -0 "$(sed -n "$((i + 1))p" "$PIDFILE")" 2>/dev/null; then
            echo "!!! backend $i died -- see $LOG_DIR/backend-$i.log"
            exit 1
        fi
        sleep 5
    done
    echo "    backend $i ready"
done

echo "==> Starting router on port $ROUTER_PORT"
apptainer exec \
    --bind "$SELF_DIR:/scripts" \
    --env "BACKENDS=$BACKENDS" \
    --env "ROUTER_PORT=$ROUTER_PORT" \
    "$SIF" \
    python3 /scripts/router.py \
    > "$LOG_DIR/router.log" 2>&1 &
echo $! >> "$PIDFILE"

until curl -sf "http://127.0.0.1:$ROUTER_PORT/health" > /dev/null 2>&1; do sleep 2; done

echo
echo "==> Fleet up. Point mini-swe at http://$(hostname -s):$ROUTER_PORT/v1"
echo "    per-backend load:  curl -s http://$(hostname -s):$ROUTER_PORT/router/stats"
echo "    stop:              bash scripts/serve/vllm-MI250-fleet.sh stop"
wait
