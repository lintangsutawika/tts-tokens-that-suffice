#!/usr/bin/env bash
# Usage: bash scripts/serve/portforward.sh <node> <remote_port> [local_port]
# Example: bash scripts/serve/portforward.sh gpu042 8000
# Example: bash scripts/serve/portforward.sh gpu042 9123 9123

set -euo pipefail

NODE=${1:?Usage: $0 <node> <remote_port> [local_port]}
REMOTE_PORT=${2:?Usage: $0 <node> <remote_port> [local_port]}
LOCAL_PORT=${3:-${REMOTE_PORT}}

JUMP=lsutawik@hpcfund.amd.com

echo "Forwarding localhost:${LOCAL_PORT} -> ${JUMP} -> ${NODE}:${REMOTE_PORT}"
echo "Press Ctrl-C to stop."

ssh -N \
    -J "${JUMP}" \
    -L "${LOCAL_PORT}:localhost:${REMOTE_PORT}" \
    "${JUMP%%@*}@${NODE}"
