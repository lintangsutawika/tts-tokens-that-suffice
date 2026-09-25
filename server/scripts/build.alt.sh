#!/bin/bash
# ALTERNATIVE build: bake a .sif from the prebuilt vLLM ROCm image + the AMD
# tinker SkyRL fork, instead of building vLLM from source on rocm/pytorch.
# This mirrors adityasoni9998/SkyRL@skyrl_amd_tinker setup.md, adapted from the
# interactive "venv on /tmp at runtime" flow into a baked, reusable image so it
# drops into our existing run.sbatch automation.
#
#   Compare against build.sh (the current working stack). Outputs a SEPARATE
#   image (tts-server-alt.sif) and def (tts-server-alt.def) so nothing here
#   touches the working tts-server.sif.
#
# Usage: bash server/scripts/build.alt.sh
#
# Stack (per the fork's pyproject.other.toml + setup.md):
#   base   : vllm/vllm-openai-rocm:v0.20.2 (vLLM 0.20.2 + ROCm + torch prebuilt)
#   vllm   : 0.20.2            (FROM THE BASE IMAGE — no source compile)
#   skyrl  : adityasoni9998 fork @ skyrl_amd_tinker (pinned below)
#   tinker : 0.16.1           (fork pins this; NOTE: downgrade from our 0.22.2)
#   transformers : >=5.6.1,<=5.8.0  (fork pin; major jump from our 4.51)
#   ray    : 2.51.1           (ray[all])
#   extras : flash-linear-attention[rocm], orjson, torchdata
# The venv is created with --system-site-packages so it inherits the base
# image's ROCm torch + vllm rather than reinstalling them.

set -e
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$(dirname "$SELF_DIR")"
ROOT="$(dirname "$SERVER_DIR")"
BASE_DIR="$(dirname "$ROOT")"

BASE_SIF="$BASE_DIR/vllm-openai-rocm-0.20.2.sif"
BUILT_SIF="$BASE_DIR/tts-server-alt.sif"
DEF_FILE="$SELF_DIR/tts-server-alt.def"

SKYRL_REPO="https://github.com/adityasoni9998/SkyRL.git"
SKYRL_BRANCH="skyrl_amd_tinker"
# HEAD of skyrl_amd_tinker as of 2026-06-26; pin for reproducibility.
SKYRL_COMMIT="8b7d87161d3fd1d2f9458a595b73570db5345d17"

# The baked venv lives at a persistent path inside the image (not /tmp, which
# setup.md uses because it builds at runtime). run.alt.sh activates this.
VENV="/opt/skyrl-venv"

# Pull the prebuilt vLLM ROCm base image if needed (one-time).
if [ ! -f "$BASE_SIF" ]; then
    echo "==> Pulling base image (one-time): vllm/vllm-openai-rocm:v0.20.2 ..."
    apptainer pull "$BASE_SIF" docker://docker.io/vllm/vllm-openai-rocm:v0.20.2
fi

# Write the definition file
cat > "$DEF_FILE" <<EOF
Bootstrap: localimage
From: $BASE_SIF

%post
    set -e

    # The vLLM ROCm image ships vLLM + ROCm torch in the default python's
    # site-packages. Create a venv that INHERITS them (--system-site-packages)
    # so we never reinstall (and never risk pulling a CUDA torch over the
    # ROCm one). All SkyRL deps install on top of that base.
    python3 -m venv --system-site-packages $VENV
    export PATH=$VENV/bin:\$PATH
    PY=$VENV/bin/python
    \$PY -m pip install -q --upgrade pip uv

    # Record the base image's torch so we can verify nothing clobbered it.
    BASE_TORCH="\$(\$PY -c 'import torch; print(torch.__version__)')"
    echo "==> Base image torch: \$BASE_TORCH"

    # Clone the AMD tinker fork at the pinned commit.
    rm -rf /skyrl
    mkdir -p /skyrl && cd /skyrl
    git init -q
    git remote add origin $SKYRL_REPO
    git fetch --depth 1 -q origin $SKYRL_COMMIT
    git checkout -q FETCH_HEAD

    # setup.md swaps in pyproject.other.toml (the AMD/ROCm packaging that omits
    # CUDA-only deps and relies on the image's vllm) before installing.
    cp pyproject.toml pyproject.toml.bak
    cp pyproject.other.toml pyproject.toml

    # Install SkyRL (editable) with fsdp + tinker extras, then the extra runtime
    # deps setup.md adds. Faithful to the fork's recipe.
    \$PY -m pip install -e '.[fsdp,tinker]'
    \$PY -m pip install -U 'ray[all]==2.51.1'
    \$PY -m pip install 'flash-linear-attention[rocm]'
    \$PY -m pip install orjson torchdata fastapi psutil

    # Restore the original pyproject so the checkout matches upstream.
    cp pyproject.toml.bak pyproject.toml
    rm -f pyproject.toml.bak

    # Guard: make sure the ROCm torch from the base image is still the one
    # installed (a transitive dep can silently pull a CUDA build). If this
    # changed, the install needs a torch constraint like build.sh has.
    NOW_TORCH="\$(\$PY -c 'import torch; print(torch.__version__)')"
    echo "==> torch after install: \$NOW_TORCH (base was \$BASE_TORCH)"
    if [ "\$NOW_TORCH" != "\$BASE_TORCH" ]; then
        echo "WARNING: torch changed from \$BASE_TORCH to \$NOW_TORCH — verify it is still a ROCm/HIP build before trusting this image." >&2
    fi
    \$PY -c 'import torch; assert torch.version.hip is not None, "torch is not a ROCm/HIP build!"; print("HIP:", torch.version.hip)'

    # Sanity: vLLM importable from the baked venv.
    \$PY -c 'import vllm; print("vllm:", vllm.__version__)'

%environment
    export PATH=$VENV/bin:\$PATH
    export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
    export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
    export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
    export ROCM_PATH=/opt/rocm
    export SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1
EOF

echo "==> Building $BUILT_SIF (alternative stack) ..."
apptainer build --fakeroot "$BUILT_SIF" "$DEF_FILE"
echo "==> Done: $BUILT_SIF"
echo
echo "Next: run it with a matching launcher (base-model Qwen/Qwen3.5-9B, layout 2/2)."
echo "This image bakes the venv at $VENV; a run.alt.sh should 'source $VENV/bin/activate'"
echo "then 'uv run --active --no-sync --extra tinker --extra fsdp -m skyrl.tinker.api ...'."
