#!/bin/bash
# Build a STANDALONE vllm-serving .sif (no SkyRL / Megatron) from vllm main,
# on the SAME ROCm base as build-megatron.sh.
#
# Why this exists: serving CohereLabs/North-Mini-Code-1.0 (arch `cohere2_moe`,
# a 30B MoE transformer) needs a vllm recent enough to know that architecture,
# so we build vllm MAIN from source. The prebuilt vllm-openai.sif can't be used
# for two independent reasons:
#   1. It's too old to have `cohere2_moe`.
#   2. It's on an Ubuntu-22.04 base whose libstdc++ tops out below GLIBCXX_3.4.31
#      (`strings .../libstdc++.so.6 | grep GLIBCXX_3.4.31` -> 0). Newer vllm /
#      torch extensions reference that symbol and fail to load.
# The rocm/pytorch 7.2.4 base (Ubuntu 24.04) that build-megatron.sh uses DOES
# ship GLIBCXX_3.4.31 (grep -> 1), so we build on it and inherit the symbol.
#
# This is NOT a hybrid model, so unlike the megatron image we do NOT build
# causal_conv1d / fla / flash-attn / TransformerEngine. It's just torch + vllm.
#
# Usage: bash server/scripts/build-north-vllm.sh
#   VLLM_REF=<branch|tag|sha>   which vllm to build (default: main)
#   BUILT_SIF=<path>            output image (default: <BASE_DIR>/north-vllm.sif)
# Run on a host in the arch family you'll serve on; the compile targets BOTH
# gfx90a (MI250) and gfx942 (MI300X) so one image runs on either.

set -e
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$(dirname "$SELF_DIR")"
ROOT="$(dirname "$SERVER_DIR")"
BASE_DIR="$(dirname "$ROOT")"

BASE_SIF="$BASE_DIR/rocm-pytorch-7.2.4.sif"
BUILT_SIF="${BUILT_SIF:-$BASE_DIR/north-vllm.sif}"
DEF_FILE="$SELF_DIR/north-vllm.def"

ROCM_INDEX="https://download.pytorch.org/whl/rocm7.2"
VLLM_REPO="https://github.com/vllm-project/vllm.git"
# Build vllm main by default (cohere2_moe support is recent). Pin to a tag/sha
# for reproducibility once a known-good commit is found.
VLLM_REF="${VLLM_REF:-main}"

# Pull the base image if needed (same base as build-megatron.sh; reuse if pulled).
if [ ! -f "$BASE_SIF" ]; then
    echo "==> Pulling base image (one-time)..."
    apptainer pull "$BASE_SIF" docker://rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0
fi

cat > "$DEF_FILE" <<EOF
Bootstrap: localimage
From: $BASE_SIF

%post
    set -e
    # Use the base image's pytorch venv directly (matches build-megatron.sh).
    export VENV=/opt/venv
    export PATH=\$VENV/bin:\$PATH
    PY=\$VENV/bin/python

    \$PY -m pip install -q uv
    # unsafe-best-match: the ROCm index carries stale copies of common deps;
    # without this uv can pin them to the old ROCm-index version and fail.
    PIP="uv pip install --python \$PY --index-strategy unsafe-best-match"

    # Pin the ROCm torch so no dep downgrades it to a CUDA build. The +rocm7.2
    # local version only exists on the PyTorch ROCm index, so every install below
    # carries the extra index + this constraint. torch 2.11 matches what vllm
    # 0.23 pinned; if vllm main has moved to a newer torch and you hit an ABI
    # mismatch at runtime, this pin (and the ROCm index tag) is the thing to bump.
    printf 'torch==2.11.0+rocm7.2\ntorchvision==0.26.0+rocm7.2\n' > /tmp/constraints.txt
    EXTRA="--extra-index-url $ROCM_INDEX --constraint /tmp/constraints.txt"

    # Upgrade the base torch 2.10 -> 2.11 (vllm's expected torch).
    \$PY -m pip install -q torch==2.11.0 torchvision==0.26.0 --index-url $ROCM_INDEX

    # vllm detects the ROCm platform via 'import amdsmi'; the AMD SMI python
    # bindings ship with ROCm but aren't in the base venv. Without them vllm
    # falls back to UnspecifiedPlatform (empty device_type) and crashes.
    \$PY -m pip install -q /opt/rocm/share/amd_smi

    # --- vllm from source (ROCm, gfx90a=MI250 + gfx942=MI300X) ---
    # Build-time deps must be explicit under --no-build-isolation (so the build
    # sees the torch 2.11 already in the venv instead of pulling a CUDA torch).
    # setuptools range: vllm's build-system pins setuptools>=77.0.3,<81.0.0; a
    # newer setuptools' PEP 639 handling rejects vllm's license metadata.
    # setuptools-rust: recent vllm setup.py imports it at top level (the new
    # vllm/vllm-rs Rust frontend). Its RustExtension is optional=True by default
    # (VLLM_REQUIRE_RUST_FRONTEND unset), so with no cargo present setuptools-rust
    # SKIPS it cleanly -- we serve with vllm's Python/OpenAI server, not vllm-rs,
    # so no Rust toolchain is needed; the package just has to import.
    \$PY -m pip install -q ninja cmake setuptools-scm packaging wheel jinja2 \\
        "setuptools>=77.0.3,<81.0.0" "setuptools-rust>=1.9.0"

    rm -rf /tmp/vllm-src
    # Partial clone (blob:none) keeps the FULL commit graph + tags so
    # setuptools-scm can derive the dev version, while downloading blobs lazily
    # -- a plain --depth 1 clone has no nearest tag and setuptools-scm fails.
    git clone --filter=blob:none $VLLM_REPO /tmp/vllm-src
    git -C /tmp/vllm-src checkout -q $VLLM_REF
    echo "==> vllm at: \$(git -C /tmp/vllm-src describe --tags --always) (\$(git -C /tmp/vllm-src rev-parse --short HEAD))"

    # Single source build (compiles once) + installs vllm's runtime deps
    # (including a transformers new enough for cohere2_moe). The constraint keeps
    # torch pinned to the ROCm build during that dep resolution.
    PYTORCH_ROCM_ARCH="gfx90a;gfx942" \\
        VLLM_TARGET_DEVICE=rocm \\
        ROCM_HOME=/opt/rocm \\
        MAX_JOBS=32 \\
        \$PY -m pip install /tmp/vllm-src --no-build-isolation \$EXTRA -q
    rm -rf /tmp/vllm-src

    # Cohere reasoning parser (vllm --reasoning-parser cohere / the North chat
    # template) imports the `cohere_melody` package at request time; without it
    # vllm raises ImportError at serve. Install under the torch constraint so it
    # can't pull a CUDA torch. --index-strategy unsafe-best-match via \$PIP.
    \$PIP -q \$EXTRA cohere_melody

    # vllm depends on upstream CUDA 'triton', which shares the 'triton' import
    # namespace with torch's 'pytorch-triton-rocm' and would shadow it. Make the
    # ROCm triton win so vllm's Triton kernels dispatch to ROCm.
    \$PY -m pip uninstall -y triton 2>/dev/null || true
    \$PY -m pip install --force-reinstall --no-deps pytorch-triton-rocm --index-url $ROCM_INDEX -q

    # Final guard: keep the ROCm torch (vllm's deps may have pulled a CUDA build).
    # Reinstall it without touching anything else.
    \$PY -m pip install --force-reinstall --no-deps \\
        torch==2.11.0 torchvision==0.26.0 --index-url $ROCM_INDEX -q

    # Sanity-check the import path + confirm the GLIBCXX symbol that is the whole
    # reason we're on this base is present. Non-fatal: this build is expensive
    # (vllm compile), so we never discard a good image over a diagnostic.
    strings /lib/x86_64-linux-gnu/libstdc++.so.6 | grep -q GLIBCXX_3.4.31 \\
        && echo "GLIBCXX_3.4.31 present" \\
        || echo "WARNING: GLIBCXX_3.4.31 missing -- wrong base image?"
    \$PY -c "import vllm; print('vllm', vllm.__version__, 'import OK')" \\
        || echo "WARNING: build-time vllm import failed -- verify at runtime before trusting this image"

%environment
    export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
    export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
    export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
    # Base image sets ROCM_PATH=/opt/rocm-7.2.0 (nonexistent); point it at the
    # /opt/rocm symlink so amdsmi/rocm_smi find their libs.
    export ROCM_PATH=/opt/rocm
EOF

echo "==> Building $BUILT_SIF (vllm $VLLM_REF, standalone) ..."
echo "    base: $BASE_SIF (Ubuntu 24.04 -> GLIBCXX_3.4.31)"
apptainer build --fakeroot "$BUILT_SIF" "$DEF_FILE"
echo "==> Done: $BUILT_SIF"
echo "==> Serve, e.g.:"
echo "    apptainer exec --rocm --env ROCM_PATH=/opt/rocm \\"
echo "        --bind \$PWD/hf_cache:/root/.cache/huggingface \\"
echo "        $BUILT_SIF \\"
echo "        vllm serve CohereLabs/North-Mini-Code-1.0 --tensor-parallel-size <ngpus>"
