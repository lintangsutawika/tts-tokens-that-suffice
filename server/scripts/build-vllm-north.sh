#!/bin/bash
# Build a STANDALONE CUDA vllm-serving .sif for North-Mini-Code-1.0 (and other
# models needing vllm MAIN), mirroring server/scripts/build-cuda.sh but building
# vllm from source for the `cohere2_moe` architecture + the cohere_melody parser.
#
# Why this exists: CohereLabs/North-Mini-Code-1.0 (arch `cohere2_moe`, a 30B MoE)
# needs a vllm recent enough to know that architecture, plus the cohere_melody
# package for its cohere_command4 response/reasoning parser. The stock
# vllm/vllm-openai:v0.28.0 SIF (the repo's cuda-nv default) is too old to have
# cohere2_moe and lacks cohere_melody. This builds vllm MAIN on top of the
# existing CUDA SIF so it serves on H100/H200 (sm_90, CUDA 12.x).
#
# Usage: bash server/scripts/build-vllm-north.sh
#   VLLM_REF=<branch|tag|sha>    which vllm to build (default: main)
#   BASE_SIF=<path>              CUDA base (default: <BASE_DIR>/vllm-openai:v0.28.0.sif)
#   BUILT_SIF=<path>             output image (default: <BASE_DIR>/north-vllm-cuda.sif)
#   MAX_JOBS=<int>               compile parallelism (default 8)
#   BUILD_TMP=<dir>              host scratch for the build (default: $PBS_LOCALDIR,
#                                else <pwd>/vllm-build-tmp). Binds into the container
#                                at /var/tmp and is where TMPDIR + the source/build live.
#
# DISK SPACE: a vLLM-main CUDA compile is enormous (hundreds of .cu kernels, each
# writing multi-GB nvcc intermediates to $TMPDIR / nvcc tmpxft files) and easily
# fills the container's /tmp -> "No space left on device" / "nvFatbin error: empty
# input". We bind a LARGE node-local scratch ($PBS_LOCALDIR by default) into the
# container at /var/tmp (a path that EXISTS in the base; singularity build --bind
# requires the destination to exist), point TMPDIR/TMP/NVCC_TMPDIR there, and build
# the vllm source tree there too.
#
# Run on an x86_64 NVIDIA host (sm_90). Container tool: singularity (default) or
# apptainer; override with BUILD_BIN.
BUILD_BIN="${BUILD_BIN:-singularity}"
set -e
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$(dirname "$SELF_DIR")"
ROOT="$(dirname "$SERVER_DIR")"
BASE_DIR="$(dirname "$ROOT")"
BASE_SIF="${BASE_SIF:-$BASE_DIR/vllm-openai:v0.28.0.sif}"
BUILT_SIF="${BUILT_SIF:-$BASE_DIR/north-vllm-cuda.sif}"
DEF_FILE="$SELF_DIR/north-vllm-cuda.def"
VLLM_REPO="https://github.com/vllm-project/vllm.git"
VLLM_REF="${VLLM_REF:-main}"
# vllm:v0.28.0 base installs python3.12 at /usr/bin (NOT /usr/local/bin); vllm + its
# deps live in /usr/local/lib/python3.12/dist-packages (the Debian system dist-packages).
PY=/usr/bin/python3
MAX_JOBS="${MAX_JOBS:-8}"
# Host scratch: prefer PBS node-local (big, fast), else a dir under the current dir.
BUILD_TMP="${BUILD_TMP:-${PBS_LOCALDIR:$(dirname "$0")/vllm-build-tmp}}"
mkdir -p "$BUILD_TMP"

if [ ! -f "$BASE_SIF" ]; then
    echo "==> Base CUDA SIF not found; pulling vllm/vllm-openai:v0.28.0 (one-time)..."
    "$BUILD_BIN" pull "$BASE_SIF" docker://vllm/vllm-openai:v0.28.0
fi

cat > "$DEF_FILE" <<EOF
Bootstrap: localimage
From: $BASE_SIF
%post
    set -e
    export PATH=/usr/local/bin:/usr/bin:\\\$PATH
    PY=$PY
    # Big scratch is bind-mounted at /var/tmp by the build command; make nvcc and
    # CMake write there (the container's own /tmp is small and overflows).
    mkdir -p /var/tmp/vllm-tmp /var/tmp/vllm-tmp/nvcc
    export TMPDIR=/var/tmp/vllm-tmp
    export TMP=/var/tmp/vllm-tmp
    export NVCC_TMPDIR=/var/tmp/vllm-tmp/nvcc
    # git is not in the vllm base image; install it (Debian base) + libdw-dev/elfutils
    # (deep_gemm's _C ext includes elfutils/libdwfl.h; missing it -> fatal error at
    # the DeepGEMM build step, ~458/460). The base's NVIDIA CUDA repo occasionally
    # hits a mirror-sync race on apt-get update (E: Failed to fetch ...Packages.gz
    # unexpected size); drop it (not needed for these packages) and retry update so
    # install is never skipped by a transient update failure.
    rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/cuda-*.list 2>/dev/null || true
    apt-get update -q -o Acquire::Retries=3 || apt-get update -q
    apt-get install -y --no-install-recommends git ca-certificates libdw-dev libelf-dev libdw1
    rm -rf /var/lib/apt/lists/*
    # The vllm base ships only VERSIONED nvrtc symlinks (libnvrtc.so.13); CUDA13
    # removes the unversioned libnvrtc.so that CMake's find_library(CUDA_nvrtc_LIBRARY)
    # requires, so the vllm build fails with CUDA_nvrtc_LIBRARY NOTFOUND. Recreate the
    # unversioned symlinks so the C++ ext (spinloop/cumem_allocator/fs_io_C) link.
    ln -sf /usr/local/cuda/lib64/libnvrtc.so.13 /usr/local/cuda/lib64/libnvrtc.so
    ln -sf /usr/local/cuda/lib64/libnvrtc-builtins.so.13.0 /usr/local/cuda/lib64/libnvrtc-builtins.so
    # The base's /usr/local/cuda/include is PARTIAL -- it lacks cusparse.h (only
    # cublas/cudart are present). torch's ATen/cuda/CUDAContextLight.h does
    # '#include <cusparse.h>' (pulled in by DeepGEMM's deep_jit), so DeepGEMM's _C
    # build fails with 'fatal error: cusparse.h: No such file or directory'. The
    # full CUDA headers ship in the nvidia/cu13 pip package -- link the missing ones
    # into /usr/local/cuda/include so `-I/usr/local/cuda/include` finds them.
    # Link ALL CUDA headers from the nvidia/cu13 pip package into /usr/local/cuda/include.
    # The base's cuda include is PARTIAL (only ~105 of 156 headers) and torch's
    # ATen/cuda/CUDAContextLight.h pulls in several that are missing (cusparse.h,
    # cusolverDn.h, ...), so DeepGEMM's deep_jit _C build fails header-by-header
    # ('fatal error: <h>: No such file or directory'). Link every header once up
    # front (overwriting the present ones with identical content) to stop the
    # whack-a-mole.
    for _h in /usr/local/lib/python3.12/dist-packages/nvidia/cu13/include/*.h; do
        ln -sf "\$_h" /usr/local/cuda/include/\$(basename "\$_h") 2>/dev/null || true
    done
    # Build toolchain + vllm build deps (CUDA torch already in the base).
    # setuptools: vllm's build-system pins setuptools>=77.0.3,<81.0.0 (PEP 639).
    /usr/bin/python3 -m pip install -q --no-cache-dir \\
        ninja cmake setuptools-scm packaging wheel jinja2 \\
        "setuptools>=77.0.3,<81.0.0" "setuptools-rust>=1.9.0"
    rm -rf /var/tmp/vllm-src
    # Partial clone (blob:none) keeps full commit graph + tags so setuptools-scm
    # derives the dev version; a --depth 1 clone has no nearest tag and fails.
    # Cloning into /var/tmp (big scratch) keeps the tree + build dir on the big disk.
    git clone --filter=blob:none $VLLM_REPO /var/tmp/vllm-src
    git -C /var/tmp/vllm-src checkout -q $VLLM_REF
    echo "==> vllm at: \\\$(git -C /var/tmp/vllm-src describe --tags --always) (\\\$(git -C /var/tmp/vllm-src rev-parse --short HEAD))"
    # Build + install vllm MAIN on the CUDA torch already present. Prebuilt CUDA
    # kernels (flash-attn etc.) come from vllm's nightly index; only in-tree
    # kernels compile. sm_90 (H100/H200); add sm_80 for A100 if needed.
    MAX_JOBS=$MAX_JOBS \\
        VLLM_TARGET_DEVICE=cuda \\
        TORCH_CUDA_ARCH_LIST="9.0" \\
        /usr/bin/python3 -m pip install /var/tmp/vllm-src --no-build-isolation -q \\
            --extra-index-url https://wheels.vllm.ai/nightly/cu129
    rm -rf /var/tmp/vllm-src /var/tmp/vllm-tmp
    # cohere_melody: required by vllm's --reasoning-parser cohere_command4 /
    # --tool-call-parser cohere_command4 at request time.
    # ALIGN flashinfer + flashinfer-cubin. The vLLM-main build upgrades `flashinfer`
    # (pip dep) but leaves the base's precompiled `flashinfer-cubin` at an OLD version;
    # flashinfer's own check then fails at serve ("flashinfer-cubin version X does not
    # match flashinfer version Y -- install the same version of both" -> EngineCore init
    # fails for every worker). Upgrade flashinfer-cubin to the SAME version as the
    # flashinfer python lib, and fail loudly if they still don't match.
    /usr/bin/python3 - <<'PYONCE'
import importlib.metadata as md, subprocess, sys
try:
    fi = md.version("flashinfer")
except Exception:
    fi = "?"
try:
    cub = md.version("flashinfer-cubin")
except Exception:
    cub = None
if cub is not None and cub != fi:
    print(f"[build] aligning flashinfer-cubin {cub} -> flashinfer {fi}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           f"flashinfer-cubin=={fi}"])
else:
    print(f"[build] flashinfer {fi} / flashinfer-cubin {cub} aligned")
PYONCE
    /usr/bin/python3 -m pip install -q --no-cache-dir "cohere_melody>=0.9.0"
    # Sanity check (non-fatal): vllm imports.
    /usr/bin/python3 -c "import vllm; print('vllm', vllm.__version__, 'import OK')" \\
        || echo "WARNING: vllm import failed at build time -- verify before trusting"
%environment
    export PYTHONUNBUFFERED=1
    export NVIDIA_VISIBLE_DEVICES=all
%runscript
    exec /usr/local/bin/vllm "\$@"
EOF

echo "==> Building $BUILT_SIF from $BASE_SIF (vllm $VLLM_REF, MAX_JOBS=$MAX_JOBS)"
echo "    build scratch: $BUILD_TMP -> /var/tmp inside the container"
"$BUILD_BIN" build --fakeroot \
    --bind "$BUILD_TMP:/var/tmp" \
    "$BUILT_SIF" "$DEF_FILE"
echo "==> Done: $BUILT_SIF"
echo "    Serve North with: VLLM_CUDA_SIF=$BUILT_SIF SERVE_BACKEND=cuda-nv MODEL=CohereLabs/North-Mini-Code-1.0"