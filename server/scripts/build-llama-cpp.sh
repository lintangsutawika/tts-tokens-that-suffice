#!/bin/bash
# Pull the llama.cpp OpenAI-compatible server (llama-server) as a .sif, to serve GGUF
# models -- e.g. unsloth/GLM-5.3-Flash-GGUF -- as a harbor deliberator. llama-server
# exposes /v1/chat/completions, so it plugs in like any OpenAI-compatible endpoint
# (DELIBERATOR_BASE_URL=http://<node>:<port>/v1, MODEL=openai/<served-name>).
# Usage: bash server/scripts/build-llama-cpp.sh
#
# Unlike the vLLM builds there is nothing to compile -- ggml-org ships a prebuilt
# server image, so this just pulls it into a .sif.
#
# GLM-5.3-Flash = 320B-A18B hybrid (sparse+linear attention) MoE. The unsloth 1-bit
# dynamic quants -- UD-IQ1_S (93 GB) / UD-IQ1_M (98 GB) -- fit a single H200 (141 GB)
# with full GPU offload (-ngl 99).
#
# !! TWO THINGS TO CHECK before trusting a run:
#   1. llama.cpp VERSION: GLM-5.3-Flash support landed in llama.cpp PR #27754. The
#      stock ':server' tag may predate it -> the model won't load. Use a recent tag
#      (or a build off that PR) once it's in a release.
#   2. GPU vs CPU tag: ':server' is CPU-only. For H200 offload you want the CUDA
#      image -- set LLAMA_CPP_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda. (ROCm/MI300
#      has no official server image; build from source or use a ROCm tag if available.)

set -e
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$(dirname "$SELF_DIR")"
ROOT="$(dirname "$SERVER_DIR")"
BASE_DIR="$(dirname "$ROOT")"

# Override for the CUDA variant (H200): LLAMA_CPP_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda
LLAMA_CPP_IMAGE="${LLAMA_CPP_IMAGE:-ghcr.io/ggml-org/llama.cpp:server}"
BUILT_SIF="${LLAMA_CPP_SIF:-$BASE_DIR/llama-cpp-server.sif}"

if [ -f "$BUILT_SIF" ]; then
    echo "==> $BUILT_SIF already exists (delete it to re-pull)."
else
    echo "==> Pulling ${LLAMA_CPP_IMAGE} -> $BUILT_SIF ..."
    APPTAINER_BIN="${APPTAINER_BIN:-$(command -v apptainer || command -v singularity || echo apptainer)}"
    "${APPTAINER_BIN}" pull "$BUILT_SIF" "docker://${LLAMA_CPP_IMAGE}"
fi
echo "==> Done: $BUILT_SIF"

cat <<EOF

To serve GLM-5.3-Flash-GGUF (manual smoke test; --nv for GPU, drop it for CPU):
  \${APPTAINER_BIN:-apptainer} exec --nv "$BUILT_SIF" \\
    llama-server -hf unsloth/GLM-5.3-Flash-GGUF:UD-IQ1_S \\
      -ngl 99 -c 131072 --host 0.0.0.0 --port 8000 \\
      --jinja --temp 0.95 --top-p 1.0
Then: curl -s http://localhost:8000/v1/models
  * -hf downloads the (sharded) GGUF to the HF cache on first run; bind a persistent
    HF_HOME so 93 GB isn't re-fetched. Swap :UD-IQ1_S for :UD-IQ1_M for the 98 GB quant.
  * --jinja enables the model's chat template so TOOL/function calls work (required
    for mini-swe-agent). Confirm GLM-5.3 tool calls actually parse before a full run --
    the llama.cpp harness analogue of the vLLM tool-parser gotcha.
EOF
