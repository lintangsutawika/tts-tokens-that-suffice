#!/bin/bash
# Launch the tinker API server using a pre-built SkyRL image.
# Usage: BACKEND=fsdp|megatron bash server/scripts/run.sh [sft|rl]
#   sft  (default) — SFT / supervised learning backend config
#   rl              — RL backend config (enables vLLM inference engines)
#   BACKEND=fsdp     (default) FSDP training       -> tts-server.sif
#   BACKEND=megatron           Megatron training    -> tts-server-megatron.sif
#   MODEL=...        base model (default Qwen/Qwen3-8B)
#   GPU=mi210|mi300|mi325  hardware-tuned config (default: generic <mode>.json)
# Build the matching image first: build.sh (fsdp) or build-megatron.sh (megatron).

set -e
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$(dirname "$SELF_DIR")"
BASE_DIR="$(dirname "$(dirname "$SERVER_DIR")")"

# Backend selects both the image and the uv extra (the extra name matches the
# backend name). SkyRL's fsdp/megatron extras conflict, so each lives in its own
# image. SIF can be overridden to point anywhere (e.g. tts-server-alt.sif).
BACKEND="${BACKEND:-fsdp}"
case "$BACKEND" in
    fsdp)     DEFAULT_SIF="$BASE_DIR/tts-server.sif" ;;
    megatron) DEFAULT_SIF="$BASE_DIR/tts-server-megatron.sif" ;;
    *) echo "Error: BACKEND must be 'fsdp' or 'megatron' (got '$BACKEND')." >&2; exit 1 ;;
esac
SIF="${SIF:-$DEFAULT_SIF}"
MODE="${1:-rl}"
# Base model to serve/train. Override with MODEL=... (must match the checkpoints
# and the client's expectation).
MODEL="${MODEL:-Qwen/Qwen3-8B}"

if [ ! -f "$SIF" ]; then
    echo "Error: $SIF not found (BACKEND=$BACKEND)." >&2
    if [ "$BACKEND" = megatron ]; then
        echo "Build it first: bash server/scripts/build-megatron.sh" >&2
    else
        echo "Build it first: bash server/scripts/build.sh" >&2
    fi
    exit 1
fi

# GPU selects a hardware-tuned config variant. mi210 (64GB) is memory-tight
# (TP sharding, low gpu_memory_utilization); mi300/mi325 (192-256GB) relax to
# TP=1 + high utilization. Unset -> the generic ${MODE}.json (unchanged behavior).
GPU="${GPU:-}"
case "$GPU" in
    mi210|mi210x)              CONFIG_NAME="${MODE}-mi210x" ;;
    mi300|mi300x|mi325|mi325x) CONFIG_NAME="${MODE}-mi300x" ;;
    "")                        CONFIG_NAME="${MODE}" ;;
    *) echo "Error: GPU must be mi210|mi300|mi325 (got '$GPU')." >&2; exit 1 ;;
esac
CONFIG_FILE="$SERVER_DIR/config/${CONFIG_NAME}.json"
# Per-GPU variants exist only where they matter (rl). Fall back to the generic
# mode config otherwise (e.g. sft has no inference engines to tune).
[ -f "$CONFIG_FILE" ] || CONFIG_FILE="$SERVER_DIR/config/${MODE}.json"
echo "==> Config: $CONFIG_FILE (GPU=${GPU:-generic})"

# Compact the JSON config into a single-line string for --backend-config
BACKEND_CONFIG=$(python3 -c "
import json
with open('$CONFIG_FILE') as f:
    print(json.dumps(json.load(f)))
")

ROOT="$(dirname "$SERVER_DIR")"
mkdir -p "$SERVER_DIR/checkpoints" "$BASE_DIR/hf_cache" "$BASE_DIR/triton_cache" "$BASE_DIR/tmp" "$BASE_DIR/tinker_state" \
         "$BASE_DIR/torch_extensions"

# Private node-local /tmp for this run, bound over the container's /tmp. This
# avoids three failure modes that all surface as SQLite "disk I/O error" on the
# tinker.db (symlinked to /tmp/tinker.db below):
#   1. --no-mount tmp + --writable-tmpfs caps /tmp at 64MB (sessiondir max size)
#      -> the DB + WAL overflow -> ENOSPC -> "disk I/O error".
#   2. The shared host /tmp may hold a stale /tmp/tinker.db owned by another
#      user (sticky bit blocks our `rm -f`), so SQLite opens an unwritable file.
#   3. Lustre (/work1) does not provide reliable POSIX fcntl locking for SQLite.
# A fresh mktemp dir on the node-local disk (~25G, ext4 -> real locking) is
# unique per run, owned by us, and has ample space.
HOST_TMP="$(mktemp -d "$TMPDIR/tts-${SLURM_JOB_ID:-$$}-XXXXXX")"

# The tinker DB runs on this same node-local disk, NOT on /work1 (wekafs). SQLite
# WAL needs POSIX locking + mmap shm that network filesystems don't provide, and
# running the DB on /work1 corrupted it repeatedly ("database disk image is
# malformed" -- a fresh DB went bad within one run). LIVE_DB lives here; a
# background daemon snapshots it back to the durable copy on /work1 every
# SNAPSHOT_INTERVAL_SEC and once on exit. Inside the container HOST_TMP is bound
# at /tmp, so LIVE_DB is /tmp/tinker.db there (see SKYRL_DATABASE_URL below).
DURABLE_DB="$BASE_DIR/tinker_state/tinker.db"
LIVE_DB="$HOST_TMP/tinker.db"
SNAPSHOT_INTERVAL_SEC="${SNAPSHOT_INTERVAL_SEC:-300}"

# Seed the live DB from the durable copy so a restart resumes prior state.
if [ -f "$DURABLE_DB" ]; then
    echo "==> Seeding live DB from $DURABLE_DB ($(du -h "$DURABLE_DB" | cut -f1))"
    cp "$DURABLE_DB" "$LIVE_DB"
fi

# Final snapshot + cleanup on exit. Runs on normal exit and on SIGINT/SIGTERM,
# but NOT on SIGKILL (scancel/OOM) -- that's why the daemon snapshots periodically.
SNAP_PID=""
cleanup() {
    [ -n "$SNAP_PID" ] && kill "$SNAP_PID" 2>/dev/null || true
    bash "$SELF_DIR/db_snapshot.sh" once "$LIVE_DB" "$DURABLE_DB" || true
    rm -rf "$HOST_TMP"
}
trap cleanup EXIT

# Background daemon: consistent snapshot of the node-local DB back to /work1.
bash "$SELF_DIR/db_snapshot.sh" loop "$LIVE_DB" "$DURABLE_DB" "$SNAPSHOT_INTERVAL_SEC" &
SNAP_PID=$!
echo "==> DB snapshot daemon pid $SNAP_PID ($LIVE_DB -> $DURABLE_DB every ${SNAPSHOT_INTERVAL_SEC}s)"

#    --bind "$SERVER_DIR/patches/vllm_server_actor.py:/skyrl/skyrl/backends/skyrl_train/inference_servers/vllm_server_actor.py" \
# Patch bind: SkyRL's vllm_router.py:127 reads RouterArgs.pd_disaggregation, a
# field vllm-router 0.1.15 renamed to vllm_pd_disaggregation. SkyRL's `or` compat
# still evaluates the missing attr and AttributeErrors on the first sampler save.
# Our copy guards both with getattr. Drop this once SkyRL/vllm-router realign.
ROUTER_PATCH=""
if [ -f "$SERVER_DIR/patches/vllm_router.py" ]; then
    ROUTER_PATCH="--bind $SERVER_DIR/patches/vllm_router.py:/skyrl/skyrl/backends/skyrl_train/inference_servers/vllm_router.py"
fi

# Patch bind: neutralize SkyRL's auto-disable of enforce_eager for LoRA weight
# sync (config.py). That flip is performance-only, but the forced HIP-graph path
# over Triton attention is a suspected source of NaN logprobs on gfx90a, so we
# honor inference_engine.enforce_eager instead. Remove once the NaN is resolved.
EAGER_PATCH=""
if [ -f "$SERVER_DIR/patches/config.py" ]; then
    EAGER_PATCH="--bind $SERVER_DIR/patches/config.py:/skyrl/skyrl/train/config/config.py"
fi
echo "==> Starting server: backend=$BACKEND mode=$MODE model=$MODEL"
echo "==> (image: $SIF; private tmp: $HOST_TMP)"
apptainer exec \
    --rocm \
    --writable-tmpfs \
    $ROUTER_PATCH \
    $EAGER_PATCH \
    --bind "$HOST_TMP:/tmp" \
    --bind "$BASE_DIR/checkpoints:/checkpoints" \
    --bind "$BASE_DIR/hf_cache:/root/.cache/huggingface" \
    --bind "$BASE_DIR/triton_cache:/triton_cache" \
    --bind "$ROOT/src:/tts/src" \
    --bind "$BASE_DIR/tmp:/ray_tmp" \
    --bind "$BASE_DIR/tinker_state:/tinker_state" \
    --bind "$BASE_DIR/torch_extensions:/torch_extensions" \
    --env PYTHONPATH=/tts/src \
    --env TORCH_EXTENSIONS_DIR=/torch_extensions \
    --env SKYRL_DATABASE_URL=sqlite:////tmp/tinker.db \
    --env RAY_TMPDIR=/tmp/ray \
    --env TMPDIR=/tmp \
    --env RAY_local_fs_capacity_threshold=0.99 \
    --env TRITON_CACHE_DIR=/triton_cache \
    --env FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
    --env PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
    --env HF_HOME=/root/.cache/huggingface \
    --env MEGATRON_CONFIG_LOCK_DIR=/tmp \
    --env HF_HUB_OFFLINE=1 \
    --env HF_DATASETS_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env _SKYRL_USE_NEW_INFERENCE=1 \
    --env RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1 \
    --env RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1 \
    --env ROCM_PATH=/opt/rocm \
    --env SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1 \
    --env RAY_worker_maximum_startup_concurrency=8 \
    --env UV_PROJECT_ENVIRONMENT=/opt/venv \
    --env UV_NO_SYNC=1 \
    --env "TINKER_API_KEY=${TINKER_API_KEY:-tml-dummy}" \
    --env "WANDB_MODE=${WANDB_MODE:-disabled}" \
    "$SIF" \
    bash -c "
        /opt/venv/bin/ray stop --force 2>/dev/null || true &&
        cd /skyrl &&
        uv run --no-sync --extra tinker --extra $BACKEND -m skyrl.tinker.api \
            --base-model '$MODEL' \
            --backend $BACKEND \
            --port 9123 \
            --checkpoints-base /checkpoints \
            --backend-config '$BACKEND_CONFIG'
    "
