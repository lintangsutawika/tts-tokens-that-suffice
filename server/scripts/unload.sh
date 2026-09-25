#!/bin/bash
# Free the live model on the running tinker server so a new client run can start
# without restarting the server. The FSDP backend hosts ONE LoRA adapter at a
# time; a second create_model -> "register_adapter is not implemented: multi-
# tenant LoRA". This unloads the current adapter (server tears down + rebuilds
# on the next create_model). The sbatch server process / DB stay up.
#
# Usage:
#   bash server/scripts/unload.sh                 # auto-detect model on localhost:9123
#   bash server/scripts/unload.sh <model_id>      # unload a specific model
#   HOST=k006-004 bash server/scripts/unload.sh   # target a remote server node
set -euo pipefail

DB=/work1/grahamneubig/lsutawik/tinker_state/tinker.db
HOST="${HOST:-localhost}"
PORT="${PORT:-9123}"
URL="http://${HOST}:${PORT}"

MODEL_ID="${1:-}"
if [ -z "$MODEL_ID" ]; then
    MODEL_ID=$(sqlite3 "$DB" "select model_id from models where status != 'unloaded' order by rowid desc limit 1;")
    [ -z "$MODEL_ID" ] && { echo "No loaded model found in $DB"; exit 1; }
    echo "Auto-detected loaded model: $MODEL_ID"
fi

echo "Unloading $MODEL_ID via $URL ..."
curl -sS -X POST "$URL/api/v1/unload_model" \
    -H 'Content-Type: application/json' \
    -d "{\"model_id\":\"$MODEL_ID\"}"
echo

# Wait for the engine to finish the async teardown.
for _ in $(seq 1 60); do
    st=$(sqlite3 "$DB" "select status from models where model_id='$MODEL_ID';")
    echo "  status=$st"
    [ "$st" = "unloaded" ] && { echo "Done. Slot is free — relaunch your client."; exit 0; }
    sleep 5
done
echo "Timed out waiting for unload; check the server log." >&2
exit 1
