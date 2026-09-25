#!/bin/bash
# Load a previously-trained adapter back onto the running tinker server.
#
# NOTE: this is NOT the inverse of unload.sh. There is no "re-load this
# model_id" endpoint -- load_weights refuses a model the backend isn't already
# holding (engine.py: `if not self.backend.has_model(model_id)`), and model ids
# are bound to a session that is long gone. An unloaded model id survives only
# as a *path namespace on disk*: checkpoints/<model_id>/<ckpt>.tar.gz.
#
# So what this does is: create a NEW model with the same base + LoRA rank, then
# pull the old model's latest TRAINING checkpoint into it. You get a new
# model_id. Weights and optimizer state are restored, so training resumes.
#
# Sessions are heartbeat-driven: the server reaps any session silent for
# session_timeout_sec (default 300) and unloads its models -- that is the
# "Auto-unloaded stale model" line in the server log. A shell script can't hold
# a session open, so we background a heartbeat keepalive and print its pid.
# Kill it (or run unload.sh) when you're done, or the slot stays occupied.
#
# Usage:
#   bash server/scripts/load.sh                          # newest model, newest ckpt
#   bash server/scripts/load.sh model_876401be           # that model, newest ckpt
#   bash server/scripts/load.sh model_876401be 000020    # exact checkpoint
#   HOST=k006-004 bash server/scripts/load.sh
set -euo pipefail

DB=/work1/grahamneubig/lsutawik/tinker_state/tinker.db
HOST="${HOST:-localhost}"
PORT="${PORT:-9123}"
URL="http://${HOST}:${PORT}"
KEEPALIVE_PIDFILE="${KEEPALIVE_PIDFILE:-/work1/grahamneubig/lsutawik/tinker_state/keepalive.pid}"

if ! curl -sS --max-time 5 "$URL/api/v1/healthz" >/dev/null 2>&1; then
    echo "Server not reachable at $URL (healthz failed)." >&2
    exit 1
fi

SRC="${1:-}"
CKPT="${2:-}"

# The FSDP backend holds ONE adapter at a time; a second create_model dies with
# "register_adapter is not implemented: multi-tenant LoRA". Fail early and loud.
LIVE=$(sqlite3 "$DB" "select model_id from models where status != 'unloaded' order by rowid desc limit 1;")
if [ -n "$LIVE" ]; then
    echo "A model is already loaded: $LIVE" >&2
    echo "The backend holds one adapter at a time. Free it first:" >&2
    echo "    bash server/scripts/unload.sh" >&2
    exit 1
fi

# Newest model that actually has a completed TRAINING checkpoint. Models whose
# run died before the first save have only SAMPLER checkpoints (or none) and
# cannot be resumed -- load_weights validates checkpoint_type == TRAINING.
if [ -z "$SRC" ]; then
    SRC=$(sqlite3 "$DB" "
        select m.model_id from models m
        join checkpoints c on c.model_id = m.model_id
        where c.checkpoint_type = 'TRAINING' and c.status = 'COMPLETED'
        group by m.model_id order by max(c.completed_at) desc limit 1;")
    [ -z "$SRC" ] && { echo "No model with a completed TRAINING checkpoint in $DB" >&2; exit 1; }
    echo "Auto-detected source model: $SRC"
fi

BASE=$(sqlite3 "$DB" "select base_model from models where model_id='$SRC';")
[ -z "$BASE" ] && { echo "Unknown model: $SRC" >&2; exit 1; }
RANK=$(sqlite3 "$DB" "select json_extract(lora_config,'\$.rank') from models where model_id='$SRC';")

if [ -z "$CKPT" ]; then
    CKPT=$(sqlite3 "$DB" "
        select checkpoint_id from checkpoints
        where model_id='$SRC' and checkpoint_type='TRAINING' and status='COMPLETED'
        order by completed_at desc limit 1;")
    [ -z "$CKPT" ] && { echo "$SRC has no completed TRAINING checkpoint (sampler-only?)" >&2; exit 1; }
fi

TARBALL="/work1/grahamneubig/lsutawik/checkpoints/$SRC/$CKPT.tar.gz"
[ -f "$TARBALL" ] || { echo "Checkpoint missing on disk: $TARBALL" >&2; exit 1; }

echo "  base model: $BASE"
echo "  lora rank:  $RANK      (alpha is hardcoded to 32.0 server-side)"
echo "  checkpoint: tinker://$SRC/weights/$CKPT  ($(du -h "$TARBALL" | cut -f1))"
echo

SESSION_ID=$(curl -sS -X POST "$URL/api/v1/create_session" \
    -H 'Content-Type: application/json' \
    -d '{"tags":["load.sh"],"sdk_version":"load.sh"}' \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["session_id"])')
echo "Created session: $SESSION_ID"

# Start the heartbeat BEFORE the slow load_weights (a 33GB restore can outrun
# the 300s reaper on its own).
if [ -f "$KEEPALIVE_PIDFILE" ]; then
    kill "$(cat "$KEEPALIVE_PIDFILE")" 2>/dev/null || true
fi
setsid bash -c "
    while true; do
        curl -sS -X POST '$URL/api/v1/session_heartbeat' \
            -H 'Content-Type: application/json' \
            -d '{\"session_id\":\"$SESSION_ID\"}' >/dev/null 2>&1 || true
        sleep 10
    done
" >/dev/null 2>&1 &
KEEPALIVE_PID=$!
echo "$KEEPALIVE_PID" > "$KEEPALIVE_PIDFILE"
echo "Heartbeat keepalive: pid $KEEPALIVE_PID (pidfile $KEEPALIVE_PIDFILE)"

MODEL_ID=$(curl -sS -X POST "$URL/api/v1/create_model" \
    -H 'Content-Type: application/json' \
    -d "{\"session_id\":\"$SESSION_ID\",\"base_model\":\"$BASE\",\"lora_config\":{\"rank\":$RANK}}" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["model_id"])')
echo "Created model:   $MODEL_ID"

echo "Loading weights (restores optimizer state too; this takes a while) ..."
REQ=$(curl -sS -X POST "$URL/api/v1/load_weights" \
    -H 'Content-Type: application/json' \
    -d "{\"model_id\":\"$MODEL_ID\",\"path\":\"tinker://$SRC/weights/$CKPT\"}" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["request_id"])')

# retrieve_future long-polls but gives up at 300s; a big restore needs more.
for _ in $(seq 1 40); do
    OUT=$(curl -sS -X POST "$URL/api/v1/retrieve_future" \
        -H 'Content-Type: application/json' \
        -d "{\"request_id\":\"$REQ\"}" 2>/dev/null || true)
    case "$OUT" in
        *'"failed"'*|*'"error"'*) echo "load_weights failed: $OUT" >&2; exit 1 ;;
        *load_weights*)           echo; echo "Loaded."; break ;;
    esac
    echo "  ... still loading"
done

echo
echo "  $SRC/$CKPT  ->  $MODEL_ID   (LIVE)"
echo
echo "Point your client at model_id=$MODEL_ID, or verify with:"
echo "    HOST=$HOST bash server/scripts/models.sh"
echo
echo "When done, stop the keepalive or the slot stays taken:"
echo "    kill \$(cat $KEEPALIVE_PIDFILE) && bash server/scripts/unload.sh"
