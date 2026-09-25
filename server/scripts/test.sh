#!/bin/bash
# Launch the tinker API server using the pre-built tts-server.sif.
# Build the image first: bash server/scripts/build.sh
# Usage: bash server/scripts/run.sh [sft|rl]
#   sft  (default) — SFT / supervised learning backend config
#   rl              — RL backend config (enables vLLM inference engines)

set -e
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$(dirname "$SELF_DIR")"
BASE_DIR="$(dirname "$(dirname "$SERVER_DIR")")"

SIF="$BASE_DIR/tts-server.sif"
MODE="${1:-sft}"

if [ ! -f "$SIF" ]; then
    echo "Error: $SIF not found. Run bash server/scripts/build.sh first."
    exit 1
fi

# Compact the JSON config into a single-line string for --backend-config
BACKEND_CONFIG=$(python3 -c "
import json
with open('$SERVER_DIR/config/${MODE}.json') as f:
    print(json.dumps(json.load(f)))
")

ROOT="$(dirname "$SERVER_DIR")"
mkdir -p "$SERVER_DIR/checkpoints" "$BASE_DIR/hf_cache" "$BASE_DIR/triton_cache" "$BASE_DIR/tmp" "$BASE_DIR/tinker_state"

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
HOST_TMP="$(mktemp -d "/tmp/tts-${SLURM_JOB_ID:-$$}-XXXXXX")"
trap 'rm -rf "$HOST_TMP"' EXIT

echo "==> Starting server in $MODE mode (private tmp: $HOST_TMP)..."
apptainer exec \
    --rocm \
    --writable-tmpfs \
    --bind "$HOST_TMP:/tmp" \
    --bind "$BASE_DIR/checkpoints:/checkpoints" \
    --bind "$BASE_DIR/hf_cache:/root/.cache/huggingface" \
    --bind "$BASE_DIR/triton_cache:/triton_cache" \
    --bind "$ROOT/src:/tts/src" \
    --bind "$BASE_DIR/tmp:/ray_tmp" \
    --bind "$BASE_DIR/tinker_state:/tinker_state" \
    --bind "$SERVER_DIR/patches/vllm_server_actor.py:/skyrl/skyrl/backends/skyrl_train/inference_servers/vllm_server_actor.py" \
    --env PYTHONPATH=/tts/src \
    --env SKYRL_DATABASE_URL=sqlite:////tinker_state/_tinker.db \
    --env RAY_TMPDIR=/tmp/ray \
    --env TMPDIR=/tmp \
    --env RAY_local_fs_capacity_threshold=0.99 \
    --env TRITON_CACHE_DIR=/triton_cache \
    --env FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
    --env PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
    --env HF_HOME=/root/.cache/huggingface \
    --env HF_HUB_OFFLINE=1 \
    --env HF_DATASETS_OFFLINE=1 \
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
    cat /skyrl/skyrl/backends/skyrl_train/inference_servers/vllm_server_actor.py
