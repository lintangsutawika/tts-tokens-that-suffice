#!/bin/bash
# Build tts-vllm.sif for standalone `vllm serve` (see vllm.sh).
# This is a thin layer on top of the already-built tts-server.sif: it reuses the
# ROCm vllm 0.20.2 compiled there and only bumps prometheus-fastapi-instrumentator.
# Kept separate from build.sh so the tinker/SkyRL training image is untouched.
# Build the base image first: bash server/scripts/build.sh
# Usage: bash server/scripts/build-vllm.sh

set -e
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$(dirname "$SELF_DIR")"
BASE_DIR="$(dirname "$(dirname "$SERVER_DIR")")"

BASE_SIF="$BASE_DIR/tts-server.sif"
BUILT_SIF="$BASE_DIR/tts-vllm.sif"
DEF_FILE="$SELF_DIR/tts-vllm.def"

if [ ! -f "$BASE_SIF" ]; then
    echo "Error: $BASE_SIF not found. Run bash server/scripts/build.sh first."
    exit 1
fi

cat > "$DEF_FILE" <<EOF
Bootstrap: localimage
From: $BASE_SIF

%post
    set -e
    PY=/opt/venv/bin/python

    # fastapi 0.116+ wraps routers from app.include_router in an internal
    # _IncludedRouter object that has no .path attribute. The base image's
    # prometheus-fastapi-instrumentator 8.0.0 iterates app.routes reading
    # route.path, so every request 500s with
    # "'_IncludedRouter' object has no attribute 'path'". 8.0.2 added explicit
    # _IncludedRouter handling; bump only that package (keeps fastapi/starlette,
    # so sse-starlette etc. stay satisfied).
    \$PY -m pip install -q "prometheus-fastapi-instrumentator==8.0.2"

%environment
    export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
    export ROCM_PATH=/opt/rocm
EOF

echo "==> Building $BUILT_SIF (incremental on tts-server.sif) ..."
apptainer build --fakeroot "$BUILT_SIF" "$DEF_FILE"
echo "==> Done: $BUILT_SIF"
