#!/bin/bash
# Pull Ollama as a .sif, to serve GGUF models (e.g. unsloth/GLM-5.3-Flash-GGUF) as a
# harbor deliberator. Ollama exposes an OpenAI-compatible API at :11434/v1, so it plugs
# in like any OpenAI endpoint (DELIBERATOR_BASE_URL=http://<node>:11434/v1).
# Usage: bash server/scripts/build-ollama.sh
#
# Ollama vs raw llama.cpp (server/scripts/build-llama-cpp.sh): Ollama IS llama.cpp under
# the hood, but adds model management + GPU auto-detect + direct HF-GGUF pull, so it's
# less setup. llama-server gives finer control (context, batching, --jinja tool template).
# Pick Ollama for "just run a GGUF", llama-server when you need to tune the serve.
#
# GLM-5.3-Flash = 320B-A18B hybrid MoE; the unsloth UD-IQ1_S (93 GB) / UD-IQ1_M (98 GB)
# 1-bit quants fit a single H200 (141 GB) with full GPU offload.
#
# !! CHECK before trusting a run:
#   1. Ollama VERSION: GLM-5.3-Flash needs Ollama built on a recent llama.cpp (support
#      in PR #27754). An older Ollama won't load it -- pull a current image.
#   2. GPU tag: docker.io/ollama/ollama auto-detects NVIDIA (--nv). For AMD/MI300 use
#      OLLAMA_IMAGE=docker.io/ollama/ollama:rocm (and apptainer --rocm).
#   3. TOOL CALLS: mini-swe-agent needs function calling. Ollama's OpenAI endpoint
#      supports `tools` only when the model's template declares them -- verify GLM-5.3
#      tool calls actually parse (the Ollama analogue of the vLLM tool-parser gotcha).

set -e
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$(dirname "$SELF_DIR")"
ROOT="$(dirname "$SERVER_DIR")"
BASE_DIR="$(dirname "$ROOT")"

# Override for AMD: OLLAMA_IMAGE=docker.io/ollama/ollama:rocm
OLLAMA_IMAGE="${OLLAMA_IMAGE:-docker.io/ollama/ollama:latest}"
BUILT_SIF="${OLLAMA_SIF:-$BASE_DIR/ollama.sif}"

if [ -f "$BUILT_SIF" ]; then
    echo "==> $BUILT_SIF already exists (delete it to re-pull)."
else
    echo "==> Pulling ${OLLAMA_IMAGE} -> $BUILT_SIF ..."
    APPTAINER_BIN="${APPTAINER_BIN:-$(command -v apptainer || command -v singularity || echo apptainer)}"
    "${APPTAINER_BIN}" pull "$BUILT_SIF" "docker://${OLLAMA_IMAGE}"
fi
echo "==> Done: $BUILT_SIF"

cat <<EOF

To serve GLM-5.3-Flash-GGUF (manual smoke test; --nv for GPU):
  # start the server (writable model store on the shared FS; --nv for NVIDIA)
  OLLAMA_MODELS=\${OLLAMA_MODELS:-$BASE_DIR/ollama_models}
  mkdir -p "\$OLLAMA_MODELS"
  \${APPTAINER_BIN:-apptainer} exec --nv --writable-tmpfs \\
    --env OLLAMA_HOST=0.0.0.0:11434 --env OLLAMA_MODELS="\$OLLAMA_MODELS" \\
    "$BUILT_SIF" ollama serve &
  # pull the GGUF straight from HF (Ollama's hf.co shortcut), then it's ready:
  \${APPTAINER_BIN:-apptainer} exec --nv --env OLLAMA_HOST=0.0.0.0:11434 \\
    --env OLLAMA_MODELS="\$OLLAMA_MODELS" "$BUILT_SIF" \\
    ollama pull hf.co/unsloth/GLM-5.3-Flash-GGUF:UD-IQ1_S
  # OpenAI API check:
  curl -s http://localhost:11434/v1/models
  * The served model name is 'hf.co/unsloth/GLM-5.3-Flash-GGUF:UD-IQ1_S'; point the
    eval at it with MODEL=openai/hf.co/unsloth/GLM-5.3-Flash-GGUF:UD-IQ1_S .
  * Bind OLLAMA_MODELS to a persistent path so the 93 GB isn't re-pulled each run.
  * Ollama auto-offloads to GPU; it splits across visible GPUs if the quant exceeds one.
EOF
