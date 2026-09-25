#!/bin/bash
# Show every model the running tinker server knows about, with its live status.
# The FSDP backend holds ONE adapter at a time: the row with status "created"
# (or "ready") is the live one; "unloading" rows are usually stale DB entries
# whose teardown never flipped to "unloaded".
#
# Usage:
#   bash server/scripts/models.sh                 # auto-detect server node via squeue
#   HOST=k002-002 bash server/scripts/models.sh   # target a node explicitly
#   HOST=localhost PORT=9123 bash server/scripts/models.sh
#
# Auto-detection looks for a SLURM job named "tts-rl-server" (see run.sbatch).
set -euo pipefail

PORT="${PORT:-9123}"
HOST="${HOST:-}"
if [ -z "$HOST" ]; then
    HOST=$(squeue -u "$USER" -h -n tts-rl-server -t RUNNING -o '%N' 2>/dev/null | head -1)
    [ -z "$HOST" ] && { echo "No running 'tts-rl-server' job found; set HOST=<node> or HOST=localhost." >&2; exit 1; }
fi
SRV="http://${HOST}:${PORT}"

# Health check first so we fail fast with a clear message.
if ! curl -sS --max-time 5 "$SRV/api/v1/healthz" >/dev/null 2>&1; then
    echo "Server not reachable at $SRV (healthz failed)." >&2
    exit 1
fi
echo "Server: $SRV  ($(curl -sS --max-time 5 "$SRV/api/v1/healthz"))"
echo

SRV="$SRV" python3 - <<'PY'
import json, os, subprocess

SRV = os.environ["SRV"]

def curl(path, payload=None):
    cmd = ["curl", "-sS", "--max-time", "8", f"{SRV}{path}"]
    if payload is not None:
        cmd += ["-H", "Content-Type: application/json", "-d", json.dumps(payload)]
    return subprocess.run(cmd, capture_output=True, text=True).stdout

runs = json.loads(curl("/api/v1/training_runs?limit=100")).get("training_runs", [])
runs.sort(key=lambda r: str(r.get("last_request_time")))

rows, loaded = [], []
for r in runs:
    mid = r["training_run_id"]
    out = curl("/api/v1/get_info", {"model_id": mid})
    try:
        status = json.loads(out).get("status", "?")
    except Exception:
        status = f"(err {out[:40]})"
    rows.append((mid, r.get("lora_rank"), str(r.get("last_request_time")), status))
    if status not in ("unloaded",) and not status.startswith("(err"):
        loaded.append((mid, status))

w = max([len(r[0]) for r in rows] + [5])
print(f'{"MODEL":<{w}}  {"RANK":>4}  {"LAST REQUEST":<26}  STATUS')
print("-" * (w + 4 + 4 + 28 + 12))
for mid, rank, last, status in rows:
    tag = ""
    if status in ("created", "ready"):
        tag = "  <== LIVE"
    elif status == "unloading":
        tag = "  (stale?)"
    print(f'{mid:<{w}}  {str(rank):>4}  {last:<26}  {status}{tag}')

print()
live = [m for m, s in loaded if s in ("created", "ready")]
if live:
    print("Live adapter(s):", ", ".join(live))
else:
    print("No live adapter (nothing 'created'/'ready') -- backend likely idle.")
if any(s == "unloading" for _, s in loaded):
    print("Note: 'unloading' rows are usually stale; the real backend holds <=1 adapter.")
PY
