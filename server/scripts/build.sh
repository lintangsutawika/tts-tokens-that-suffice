#!/bin/bash
# Build a custom .sif with SkyRL pre-installed on top of the ROCm base image.
# SkyRL is cloned at a pinned commit during the build (no git submodule).
# Run this once (or after bumping SKYRL_COMMIT).
# Usage: bash server/scripts/build.sh
#
# Stack (version-aligned with SkyRL's own pins, so no source patching/shims):
#   base   : rocm/pytorch 7.2.4 (ROCm 7.2.4, py3.12, venv at /opt/venv)
#   torch  : 2.11.0+rocm7.2  (official PyTorch ROCm wheel; matches vllm 0.20.2)
#   ray    : 2.51.1          (plain PyPI)
#   vllm   : 0.20.2          (built from source for gfx90a=MI250, gfx942=MI300X)
# Everything installs into the base image's /opt/venv (used directly as the
# project environment; run.sh points UV_PROJECT_ENVIRONMENT at it).

set -e
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$(dirname "$SELF_DIR")"
ROOT="$(dirname "$SERVER_DIR")"
BASE_DIR="$(dirname "$ROOT")"

BASE_SIF="$BASE_DIR/rocm-pytorch-7.2.4.sif"
BUILT_SIF="$BASE_DIR/tts-server.sif"
DEF_FILE="$SELF_DIR/tts-server.def"

ROCM_INDEX="https://download.pytorch.org/whl/rocm7.2"
SKYRL_REPO="https://github.com/NovaSky-AI/SkyRL.git"
SKYRL_COMMIT="6c300c755178527e2c797277f2c48a47ed451626"

# Pull the base image if needed
if [ ! -f "$BASE_SIF" ]; then
    echo "==> Pulling base image (one-time)..."
    apptainer pull "$BASE_SIF" docker://rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0
fi

# Write the definition file
cat > "$DEF_FILE" <<EOF
Bootstrap: localimage
From: $BASE_SIF

%post
    set -e
    # Use the base image's pytorch venv directly as the project environment.
    export VENV=/opt/venv
    export PATH=\$VENV/bin:\$PATH
    PY=\$VENV/bin/python

    # Clone SkyRL at the pinned commit (no submodule). The patches we used to
    # carry (LoRA->bf16 cast, gradient_checkpointing_use_reentrant=True, the
    # _SKYRL_USE_NEW_INFERENCE import guard) were old torch-2.9/ROCm-7.0
    # workarounds and are unnecessary on this torch-2.11/ROCm-7.2 stack.
    rm -rf /skyrl
    mkdir -p /skyrl && cd /skyrl
    git init -q
    git remote add origin $SKYRL_REPO
    git fetch --depth 1 -q origin $SKYRL_COMMIT
    git checkout -q FETCH_HEAD

    \$PY -m pip install -q uv
    # unsafe-best-match: consider all indexes for each package (the ROCm index
    # carries stale copies of common deps like requests; without this uv pins
    # them to the old ROCm-index version and resolution fails).
    PIP="uv pip install --python \$PY --index-strategy unsafe-best-match"

    # Pin the ROCm torch so nothing downgrades it to a CUDA build. The
    # +rocm7.2 local version only exists on the PyTorch ROCm index, so every
    # install below carries the extra index + this constraint.
    printf 'torch==2.11.0+rocm7.2\ntorchvision==0.26.0+rocm7.2\ncausal-conv1d==1.4.0\ntinker==0.22.2\n' > /tmp/constraints.txt
    EXTRA="--extra-index-url $ROCM_INDEX --constraint /tmp/constraints.txt"

    # Upgrade the base torch 2.10 -> 2.11 to match vllm 0.20.2's pin.
    \$PY -m pip install -q torch==2.11.0 torchvision==0.26.0 --index-url $ROCM_INDEX

    # Ray (plain PyPI package; pinned to match SkyRL).
    \$PIP -q "ray==2.51.1" \$EXTRA

    # causal_conv1d is a transitive dep that tries to build a CUDA extension.
    # Install a no-op stub so the dependency is satisfied without compiling.
    python3 -c "
import os
d = '/tmp/causal_conv1d_stub/causal_conv1d'
os.makedirs(d, exist_ok=True)
open(d + '/__init__.py', 'w').close()
with open('/tmp/causal_conv1d_stub/setup.py', 'w') as f:
    f.write(\"from setuptools import setup; setup(name='causal-conv1d', version='1.4.0', packages=['causal_conv1d'])\")
"
    \$PIP /tmp/causal_conv1d_stub --no-deps -q

    # Install SkyRL with the tinker extra only — [fsdp] pulls in CUDA-only
    # packages (flashinfer, vllm, nixl, flash-attn) that don't exist on ROCm.
    \$PIP -e /skyrl[tinker] -q \$EXTRA

    # Install the ROCm-compatible subset of [skyrl-train] deps manually,
    # excluding: vllm-router, nixl (CUDA/glibc-specific).
    \$PIP -q \$EXTRA \\
        loguru tqdm ninja tensorboard func_timeout \\
        "hydra-core==1.3.2" accelerate torchdata omegaconf "ray==2.51.1" \\
        "peft==0.18.1" "debugpy==1.8.0" hf_transfer wandb "datasets>=4.0.0" \\
        tensordict jaxtyping skyrl-gym polars s3fs uvicorn pybind11 setuptools \\
        "transformers>=4.51.0" "tokenizers>=0.21"

    # SkyRL's model_wrapper unconditionally imports flash_attn.bert_padding, so
    # flash-attn must be present even when using sdpa. The Triton-AMD backend
    # builds a pure-Python wheel (no CUDA/CK compile) that works on ROCm.
    FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \\
        \$PY -m pip install flash-attn==2.8.3 --no-build-isolation -q

    # vllm 0.20.2 detects the ROCm platform via \`import amdsmi\`; the AMD SMI
    # python bindings ship with ROCm but aren't in the base venv. Without them
    # vllm falls back to UnspecifiedPlatform (empty device_type) and crashes.
    \$PY -m pip install -q /opt/rocm/share/amd_smi

    # Build vllm 0.20.2 from source for ROCm (gfx90a = MI250, gfx942 = MI300X).
    # --no-build-isolation lets the build see the torch 2.11 already in /opt/venv,
    # but means build-time deps must be installed explicitly (setuptools_scm is
    # imported by vllm's setup.py during metadata generation).
    # vllm 0.20.2 build-system requires setuptools>=77.0.3,<81.0.0; the deps
    # install above pulls a newer setuptools whose PEP 639 handling rejects
    # vllm's `license = "Apache-2.0"`. Pin it back into the supported range.
    \$PY -m pip install -q ninja cmake setuptools-scm packaging wheel jinja2 \\
        "setuptools>=77.0.3,<81.0.0"
    rm -rf /tmp/vllm-src
    git clone --depth 1 --branch v0.20.2 https://github.com/vllm-project/vllm.git /tmp/vllm-src

    # Single source build (compiles once) + installs vllm's runtime deps. The
    # constraint keeps torch pinned to the ROCm build during dep resolution.
    PYTORCH_ROCM_ARCH="gfx90a;gfx942" \\
        VLLM_TARGET_DEVICE=rocm \\
        ROCM_HOME=/opt/rocm \\
        SETUPTOOLS_SCM_PRETEND_VERSION=0.20.2 \\
        MAX_JOBS=32 \\
        \$PY -m pip install /tmp/vllm-src --no-build-isolation \$EXTRA -q
    rm -rf /tmp/vllm-src

    # vllm depends on upstream CUDA 'triton', which shares the 'triton' import
    # with torch's 'pytorch-triton-rocm' and would shadow it. Make the ROCm
    # triton win so torch.compile / vllm kernels work.
    \$PY -m pip uninstall -y triton 2>/dev/null || true
    \$PY -m pip install --force-reinstall --no-deps pytorch-triton-rocm --index-url $ROCM_INDEX -q

    # Final guard: ensure the ROCm torch is the one installed (deps above may
    # have pulled a CUDA build). Reinstall it without touching anything else.
    \$PY -m pip install --force-reinstall --no-deps \\
        torch==2.11.0 torchvision==0.26.0 --index-url $ROCM_INDEX -q

%environment
    export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
    export _SKYRL_USE_NEW_INFERENCE=0
    export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
    export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
    # Base image sets ROCM_PATH=/opt/rocm-7.2.0 (a path that doesn't exist; the
    # real install is /opt/rocm-7.2.4 via the /opt/rocm symlink). Ray's pyamdsmi
    # uses ROCM_PATH to find librocm_smi64.so for GPU detection, so point it at
    # the symlink — otherwise Ray detects 0 GPUs.
    export ROCM_PATH=/opt/rocm
    export SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1
EOF

echo "==> Building $BUILT_SIF ..."
apptainer build --fakeroot "$BUILT_SIF" "$DEF_FILE"
echo "==> Done: $BUILT_SIF"
