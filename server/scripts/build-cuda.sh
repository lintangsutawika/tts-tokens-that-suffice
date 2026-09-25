#!/bin/bash
# Build tts-vllm-cuda.sif: a standalone `vllm serve` image for NVIDIA GPUs
# (H100/H200 Hopper sm_90, A100 Ampere sm_80, ...). This is the CUDA analog of
# server/scripts/build-vllm.sh -- a thin serving layer, NOT the SkyRL training image.
# Usage: bash server/scripts/build-cuda.sh
#
# On CUDA, vllm/vllm-openai:v<ver> already IS a complete vLLM serving image
# (torch + CUDA + vllm prebuilt), so unlike the ROCm build there is nothing to
# compile. Version is VLLM_VERSION (default below); bump it to move vLLM forward.
# We only:
#   1. bump prometheus-fastapi-instrumentator (the base's 8.0.0 500s every request),
#   2. add causal_conv1d + fla, the kernels Qwen3.5's Gated-DeltaNet layers need
#      (without them transformers falls back to slow torch paths). Harmless for
#      non-Qwen3.5 models (GLM, gpt-oss); drop step 2 if you never serve Qwen3.5.
#
# H200 note: Hopper is sm_90, covered by the base's CUDA 12.x wheels -- no arch flag,
# nothing H200-specific. The same .sif serves on H100/A100 too.

set -e
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$(dirname "$SELF_DIR")"
ROOT="$(dirname "$SERVER_DIR")"
BASE_DIR="$(dirname "$ROOT")"

# vLLM release to build on (matches docker tag vllm/vllm-openai:v<ver>).
VLLM_VERSION="${VLLM_VERSION:-0.28.0}"

BASE_SIF="$BASE_DIR/vllm-openai-cuda-${VLLM_VERSION}.sif"
BUILT_SIF="$BASE_DIR/tts-vllm-cuda.sif"
DEF_FILE="$SELF_DIR/tts-vllm-cuda.def"

# Pull the CUDA vLLM base (vllm ${VLLM_VERSION} + matching torch + CUDA runtime), one-time.
if [ ! -f "$BASE_SIF" ]; then
    echo "==> Pulling CUDA base image vllm/vllm-openai:v${VLLM_VERSION} (one-time)..."
    apptainer pull "$BASE_SIF" "docker://vllm/vllm-openai:v${VLLM_VERSION}"
fi

cat > "$DEF_FILE" <<EOF
Bootstrap: localimage
From: $BASE_SIF

%post
    set -e
    PY=\$(command -v python3)

    # fastapi 0.116+ wraps routers from app.include_router in an internal
    # _IncludedRouter object that has no .path attribute. The base image's
    # prometheus-fastapi-instrumentator 8.0.0 iterates app.routes reading
    # route.path, so every request 500s with
    # "'_IncludedRouter' object has no attribute 'path'". 8.0.2 added explicit
    # _IncludedRouter handling; bump only that package. (Same fix as build-vllm.sh.)
    \$PY -m pip install -q "prometheus-fastapi-instrumentator==8.0.2"

    # Qwen3.5 (Gated-DeltaNet hybrid) needs causal_conv1d + fla kernels; on CUDA
    # these are prebuilt wheels (contrast build.sh's HIP source build). fla is pure
    # Python + Triton -- --no-deps so it doesn't pull a triton that shadows vllm's.
    \$PY -m pip install -q "causal-conv1d>=1.5.0"
    \$PY -m pip install -q --no-deps "fla-core==0.5.1" "flash-linear-attention==0.5.1"

    # Guard: torch/vllm untouched.
    \$PY -c "import torch, vllm; print('torch', torch.__version__, '| vllm', vllm.__version__)"

%environment
    # vLLM serving needs no ROCm/HIP env on CUDA; leave the base image's defaults.
    :
EOF

echo "==> Building $BUILT_SIF ..."
apptainer build --fakeroot "$BUILT_SIF" "$DEF_FILE"
echo "==> Done: $BUILT_SIF"
