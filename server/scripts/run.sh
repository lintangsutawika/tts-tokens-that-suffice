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
mkdir -p "$SERVER_DIR/checkpoints" "$BASE_DIR/hf_cache" "$BASE_DIR/triton_cache" "$BASE_DIR/tmp"

echo "==> Starting server in $MODE mode..."
apptainer exec \
    --rocm \
    --writable-tmpfs \
    --bind "$SERVER_DIR/checkpoints:/checkpoints" \
    --bind "$BASE_DIR/hf_cache:/root/.cache/huggingface" \
    --bind "$BASE_DIR/triton_cache:/triton_cache" \
    --bind "$ROOT/src:/tts/src" \
    --bind "$BASE_DIR/tmp:/ray_tmp" \
    --env PYTHONPATH=/tts/src \
    --env RAY_TMPDIR=/ray_tmp \
    --env TMPDIR=/ray_tmp \
    --env TRITON_CACHE_DIR=/triton_cache \
    --env FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
    --env _SKYRL_USE_NEW_INFERENCE=0 \
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
        rm -f /tmp/tinker.db /skyrl/skyrl/tinker/tinker.db &&
        ln -sf /tmp/tinker.db /skyrl/skyrl/tinker/tinker.db &&
        cd /skyrl &&
        uv run --no-sync --extra tinker --extra fsdp -m skyrl.tinker.api \
            --base-model 'Qwen/Qwen3-8B' \
            --backend fsdp \
            --port 9123 \
            --checkpoints-base /checkpoints \
            --backend-config '$BACKEND_CONFIG'
    "
