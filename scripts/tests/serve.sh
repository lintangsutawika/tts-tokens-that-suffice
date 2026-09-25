#!/usr/bin/env bash
# Usage: ./scripts/tests/serve.sh [sft|rl]
# Defaults to sft if no argument given.

set -euo pipefail

MODE="${1:-sft}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/config/${MODE}.json"

if [[ ! -f "$CONFIG" ]]; then
    echo "Unknown config '${MODE}'. Available: $(ls "${SCRIPT_DIR}/config/" | sed 's/\.json//' | tr '\n' ' ')" >&2
    exit 1
fi

export TINKER_API_KEY=tml-dummy
export _SKYRL_USE_NEW_INFERENCE=0

uv run -m tts.tinker.api \
    --base-model "Qwen/Qwen3-8B" \
    --backend fsdp \
    --backend-config "$(cat "$CONFIG")"
