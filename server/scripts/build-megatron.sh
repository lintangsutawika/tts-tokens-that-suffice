#!/bin/bash
# Build a SEPARATE .sif with SkyRL's MEGATRON backend on top of the ROCm base.
#
# Why a separate image from build.sh (which builds the FSDP image tts-server.sif):
#   - SkyRL's `megatron` and `fsdp` extras are declared CONFLICTING in its uv
#     config, so they can't share one environment.
#   - The Megatron backend needs megatron-core + megatron-bridge + a working
#     TransformerEngine, none of which are in tts-server.sif.
# Use tts-server.sif for --backend fsdp, and THIS image for --backend megatron.
#
# Reason we're on Megatron at all: SkyRL's FSDP LoRA-adapter loading path broke
# for our model (Qwen3.5 hybrid Gated-DeltaNet); the Megatron adapter path works.
#
# Stack is identical to build.sh (base rocm/pytorch 7.2.4, torch 2.11.0+rocm7.2,
# ray 2.51.1, vllm 0.23.0 from source, causal_conv1d + fla for the hybrid model)
# PLUS the megatron trio. The only genuinely risky step is TransformerEngine on
# ROCm -- see the TE block. Expect to iterate there.
#
# Usage: bash server/scripts/build-megatron.sh
# Run on a build host with the SAME arch family you'll run on; the compiles below
# target BOTH gfx90a (MI250) and gfx942 (MI300X) so one image runs on either.

set -e
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$(dirname "$SELF_DIR")"
ROOT="$(dirname "$SERVER_DIR")"
BASE_DIR="$(dirname "$ROOT")"

BASE_SIF="$BASE_DIR/rocm-pytorch-7.2.4.sif"
BUILT_SIF="$BASE_DIR/tts-server-megatron.sif"
DEF_FILE="$SELF_DIR/tts-server-megatron.def"

ROCM_INDEX="https://download.pytorch.org/whl/rocm7.2"
SKYRL_REPO="https://github.com/NovaSky-AI/SkyRL.git"
# Jul 8 2026 (matches the NVIDIA setup's HEAD). This is the first SkyRL that
# honors language_model_only for Megatron VLMs (PR #1867, "fix
# language_model_only=True case for megatron is_vlm", 2026-07-06): it routes
# Qwen3.5 to a native text-only GPTModel+GDN, so the vision tower is never built
# and no lora.target_modules/exclude_modules workaround is needed. Older
# 6c300c7 (2026-06-16) predated the fix and forced the lora.* hacks in rl.json.
# NOTE: build.sh (fsdp image) is still on 6c300c7 -- these are now out of lockstep.
SKYRL_COMMIT="1ab51f1b965c0a3a6fa7dfd4e965758525bfd89b"

# Megatron pins, copied verbatim from SkyRL/pyproject.toml [tool.uv.sources] so
# this image matches what SkyRL was written against. Bump these only together
# with SKYRL_COMMIT.
MEGATRON_CORE_REV="71e418ea7d7b3a6c9a53238c543c3e0b43e11026"   # NVIDIA/Megatron-LM
MEGATRON_BRIDGE_REV="91a15142a4b4442a8d46ab539d1b923bd08570d0"  # NVIDIA-NeMo/Megatron-Bridge

# TransformerEngine ROCm fork. SkyRL pins TE 2.11.0 (a CUDA PyPI wheel that does
# NOT exist for ROCm); AMD maintains a source fork instead. The ROCm fork has NO
# 2.11 tag -- its release tags are v2.1_rocm, v2.2_rocm, v2.4_rocm, v2.6_rocm,
# v2.8_rocm, v2.10_rocm, plus a release_v2.15_rocm branch. v2.10_rocm is the
# closest below SkyRL's 2.11 pin; if megatron-core hits a TE API mismatch, try
# TE_REF=release_v2.15_rocm instead. This is the step most likely to need
# adjustment.
TE_REPO="${TE_REPO:-https://github.com/ROCm/TransformerEngine.git}"
TE_REF="${TE_REF:-v2.10_rocm}"

# Pull the base image if needed (same base as build.sh; reuse if already pulled).
if [ ! -f "$BASE_SIF" ]; then
    echo "==> Pulling base image (one-time)..."
    apptainer pull "$BASE_SIF" docker://rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0
fi

cat > "$DEF_FILE" <<EOF
Bootstrap: localimage
From: $BASE_SIF

%post
    set -e
    export VENV=/opt/venv
    export PATH=\$VENV/bin:\$PATH
    PY=\$VENV/bin/python

    # --- SkyRL at the pinned commit (same as build.sh) ---
    rm -rf /skyrl
    mkdir -p /skyrl && cd /skyrl
    git init -q
    git remote add origin $SKYRL_REPO
    git fetch --depth 1 -q origin $SKYRL_COMMIT
    git checkout -q FETCH_HEAD

    \$PY -m pip install -q uv
    PIP="uv pip install --python \$PY --index-strategy unsafe-best-match"

    printf 'torch==2.11.0+rocm7.2\ntorchvision==0.26.0+rocm7.2\ntinker==0.22.2\n' > /tmp/constraints.txt
    EXTRA="--extra-index-url $ROCM_INDEX --constraint /tmp/constraints.txt"

    \$PY -m pip install -q torch==2.11.0 torchvision==0.26.0 --index-url $ROCM_INDEX
    \$PIP -q "ray==2.51.1" \$EXTRA

    # SkyRL with the tinker extra only. As in build.sh we do NOT install the
    # [megatron] extra directly -- it resolves CUDA-only transformer-engine and
    # nvidia-* wheels. We hand-install the ROCm-safe megatron pieces below.
    \$PIP -e /skyrl[tinker] -q \$EXTRA

    # Shared skyrl-train runtime deps (the Megatron backend is the same
    # SkyRLTrainBackend class, so it needs all of these).
    # NOTE: vllm-router is NOT the PyPI package. SkyRL pins a custom fork wheel
    # (SumanthRH/router 0.1.14.post1) whose RouterArgs has pd_disaggregation.
    # PyPI's 0.1.15 dropped that field, so installing plain vllm-router gives a
    # RouterArgs without it and vllm_router.py:127 (... or pd_disaggregation)
    # AttributeErrors on the first sampler save. Install SkyRL's exact wheel so
    # the field is present and no runtime patch is needed (matches NVIDIA).
    \$PIP -q \$EXTRA \\
        loguru tqdm ninja tensorboard func_timeout \\
        "prometheus-fastapi-instrumentator==8.0.2" \\
        "hydra-core==1.3.2" accelerate torchdata omegaconf "ray==2.51.1" \\
        "peft==0.18.1" "debugpy==1.8.0" hf_transfer wandb "datasets>=4.0.0" \\
        tensordict jaxtyping skyrl-gym polars s3fs uvicorn pybind11 setuptools \\
        "transformers>=4.51.0" "tokenizers>=0.21"
    \$PY -m pip install -q --no-deps \\
        "https://github.com/SumanthRH/router/releases/download/0.1.14.post1/vllm_router-0.1.14.post1-cp38-abi3-manylinux_2_35_x86_64.whl"

    FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \\
        \$PY -m pip install flash-attn==2.8.3 --no-build-isolation -q

    # Hybrid Gated-DeltaNet kernels for Qwen3.5 (see build.sh for the gfx arch
    # gotcha: HIP_ARCHITECTURES, comma list, ROCM_PATH=/opt/rocm).
    CAUSAL_CONV1D_FORCE_BUILD=TRUE \\
        ROCM_PATH=/opt/rocm HIP_HOME=/opt/rocm \\
        HIP_ARCHITECTURES="gfx90a,gfx942" MAX_JOBS=32 \\
        \$PY -m pip install --no-build-isolation --no-deps -q \\
        "git+https://github.com/Dao-AILab/causal-conv1d.git@v1.5.0.post8"
    \$PY -m pip install --no-deps -q "fla-core==0.5.1" "flash-linear-attention==0.5.1"

    # =========================== MEGATRON STACK ===========================
    # TE builds a CMake extension under --no-build-isolation, so its build tools
    # must already be in the venv. build.sh installs cmake in its vllm block,
    # which runs AFTER this section -- so install the build tools now.
    # pybind11[global]: the [global] extra drops the CMake config files so TE's
    # find_package(pybind11) resolves (plain pybind11 from the shared deps does
    # not). jax/flax appear in TE's build-system.requires but are NOT needed:
    # NVTE_FRAMEWORK=pytorch (set below) makes get_frameworks() skip the jax
    # auto-detect import, so we don't drag CUDA jax onto ROCm.
    \$PY -m pip install -q cmake ninja wheel packaging setuptools-scm "pybind11[global]"

    # TransformerEngine (ROCm fork). This is the compile most likely to fail /
    # need a different TE_REF. --no-deps so it doesn't drag in nvidia-* wheels;
    # --no-build-isolation so it builds against the venv's torch 2.11. Targets
    # both gfx90a and gfx942. NVTE_FRAMEWORK=pytorch skips the JAX/paddle bits.
    #
    # NVTE_FUSED_ATTN_CK=0 disables TE's CK/AITER fused-attention backend. Its
    # bundled AITER passes '-mllvm -amdgpu-coerce-illegal-types=1', an LLVM flag
    # the ROCm 7.2.4 clang in this base image doesn't recognize, so the CK build
    # dies. CK is gfx942-only anyway; we keep the AOTriton backend (default ON),
    # which builds for gfx90a+gfx942 and doesn't need that flag. If AOTriton also
    # fails to build, disable all fused attention with NVTE_FUSED_ATTN=0 (TE then
    # uses unfused attention -- correct, just slower).
    NVTE_FRAMEWORK=pytorch \\
        NVTE_FUSED_ATTN_CK=0 \\
        NVTE_ROCM_ARCH="gfx90a;gfx942" \\
        PYTORCH_ROCM_ARCH="gfx90a;gfx942" \\
        ROCM_PATH=/opt/rocm ROCM_HOME=/opt/rocm HIP_HOME=/opt/rocm \\
        MAX_JOBS=32 \\
        \$PY -m pip install --no-build-isolation --no-deps -v \\
        "git+$TE_REPO@$TE_REF"

    # megatron-core + megatron-bridge at SkyRL's exact pins. --no-deps: their
    # dependency closures pull CUDA apex/nvidia wheels and would try to downgrade
    # torch. Anything genuinely missing at import time is a plain PyPI package we
    # add explicitly below.
    \$PY -m pip install --no-deps -q \\
        "git+https://github.com/NVIDIA/Megatron-LM@$MEGATRON_CORE_REV" \\
        "git+https://github.com/NVIDIA-NeMo/Megatron-Bridge@$MEGATRON_BRIDGE_REV"

    # Pure-python deps megatron-core/bridge import but which --no-deps skipped.
    # These are CPU/framework-only (no CUDA), safe on ROCm. If a later ImportError
    # names another, add it here rather than dropping --no-deps.
    \$PIP -q \$EXTRA \\
        einops "onnxscript>=0.5.4" "onnx>=1.19.0" tqdm regex

    # SkyRL's megatron LoRA worker does 'from megatron.bridge import AutoBridge'
    # and '.peft.lora'. megatron.bridge/__init__ eagerly imports (a) its HF
    # conversion AutoBridge, which needs modelopt (NVIDIA Model Optimizer), and
    # (b) its diffusion bridges, which need diffusers + flashinfer -- flashinfer
    # is CUDA-only and cannot install on ROCm. We're doing LLM LoRA, not
    # diffusion, so install modelopt and strip the eager diffusion import (the
    # torch guard at the end re-pins torch in case modelopt nudged it).
    \$PIP -q \$EXTRA nvidia-modelopt
    BRIDGE_INIT=/opt/venv/lib/python3.12/site-packages/megatron/bridge/__init__.py
    sed -i '/^import megatron.bridge.diffusion.models/d' "\$BRIDGE_INIT"
    # Prove the bridge LoRA path imports (mirror runtime env; non-fatal like the
    # assert below).
    FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \$PY -c \\
        "from megatron.bridge import AutoBridge; from megatron.bridge.peft.lora import LoRA; print('megatron.bridge LoRA import OK')" \\
        || echo "WARNING: megatron.bridge LoRA import failed -- verify before trusting this image"

    # apex: megatron-core's fused kernels reference apex.* when available but fall
    # back to torch when absent. The base image already ships an apex build (we
    # verified 'import apex' works), so we do NOT rebuild it here. If megatron
    # hard-requires a fused apex op at runtime, that's the next thing to build.
    # ======================================================================

    # amdsmi so vllm detects the ROCm platform (same as build.sh).
    \$PY -m pip install -q /opt/rocm/share/amd_smi

    # vllm 0.23.0 from source for ROCm. SkyRL 1ab51f1b pins vllm==0.23.0 (bumped
    # from 0.20.2 before the megatron language_model_only fix landed, so matching
    # the Jul-8 SkyRL requires this jump). If the build-system setuptools range
    # below is wrong for 0.23.0, pip will say so -- adjust the pin then.
    # setuptools-rust: vllm 0.23.0's setup.py imports it at top level (new Rust
    # frontend vllm/vllm-rs). Its RustExtension is optional=True by default
    # (VLLM_REQUIRE_RUST_FRONTEND unset), so with no cargo present setuptools-rust
    # SKIPS it cleanly (command.py: all_optional -> no raise). We use vllm's
    # Python/OpenAI server, not the vllm-rs binary, so skipping it is fine and we
    # avoid needing a Rust toolchain. The package itself is still required for the
    # import to succeed under --no-build-isolation.
    \$PY -m pip install -q ninja cmake setuptools-scm packaging wheel jinja2 \\
        "setuptools>=77.0.3,<81.0.0" "setuptools-rust>=1.9.0"
    rm -rf /tmp/vllm-src
    git clone --depth 1 --branch v0.23.0 https://github.com/vllm-project/vllm.git /tmp/vllm-src
    PYTORCH_ROCM_ARCH="gfx90a;gfx942" \\
        VLLM_TARGET_DEVICE=rocm ROCM_HOME=/opt/rocm \\
        SETUPTOOLS_SCM_PRETEND_VERSION=0.23.0 MAX_JOBS=32 \\
        \$PY -m pip install /tmp/vllm-src --no-build-isolation \$EXTRA -q
    rm -rf /tmp/vllm-src

    # Make ROCm triton win over vllm's CUDA triton (same as build.sh).
    \$PY -m pip uninstall -y triton 2>/dev/null || true
    \$PY -m pip install --force-reinstall --no-deps pytorch-triton-rocm --index-url $ROCM_INDEX -q

    # Final guard: keep the ROCm torch (TE / megatron / vllm deps may have pulled
    # a CUDA build). Reinstall without touching anything else.
    \$PY -m pip install --force-reinstall --no-deps \\
        torch==2.11.0 torchvision==0.26.0 --index-url $ROCM_INDEX -q

    # Sanity-check the import path. flash_attn here is the Triton-AMD wheel with
    # no compiled flash_attn_2_cuda; it only takes the Triton path when
    # FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE. That's set in %environment (runtime)
    # but NOT during %post, so set it here to mirror runtime -- otherwise the
    # import falls back to the absent CUDA extension and fails spuriously.
    # Non-fatal: this build is expensive (TE + vllm compiles), so we never
    # discard a good image over a diagnostic. If it warns, verify at runtime:
    #   apptainer exec tts-server-megatron.sif \\
    #     python -c "import megatron.core, transformer_engine.pytorch"
    FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \$PY -c \\
        "import megatron.core; import transformer_engine.pytorch; print('megatron+TE import OK')" \\
        || echo "WARNING: build-time megatron/TE import failed -- verify at runtime before trusting this image"

%environment
    export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
    export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
    export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
    export ROCM_PATH=/opt/rocm
    export SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1
    # Megatron-core probes these; harmless on the FSDP image but set here so the
    # backend doesn't warn about missing distributed-optimizer env.
    export CUDA_DEVICE_MAX_CONNECTIONS=1
EOF

echo "==> Building $BUILT_SIF (megatron) ..."
echo "    TE fork: $TE_REPO@$TE_REF (override with TE_REF=... if the compile fails)"
apptainer build --fakeroot "$BUILT_SIF" "$DEF_FILE"
echo "==> Done: $BUILT_SIF"
echo "==> Launch with the megatron backend by pointing run.sh's SIF + --backend at it."
