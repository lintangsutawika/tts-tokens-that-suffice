#!/bin/bash
# Batch wrapper around run.sh so the tinker/SkyRL server survives terminal
# disconnects and interactive-job timeouts (no tmux needed).
#
# Submit:   sbatch server/scripts/run.sbatch rl        # rl (default) or sft
# Override: sbatch --partition=mi3258x --time=12:00:00 server/scripts/run.sbatch rl
# Watch:    tail -f tts-rl-server-<jobid>.out          (written to submit dir)
# Stop:     scancel <jobid>
#
# These partitions allocate a whole node (all 8 GPUs, gfx942 = MI300X/MI325),
# which the rl.json placement (6 policy + 2 engines) requires. The trailing
# digit in the partition name is GPUs/node, so use an *8* partition for RL.

#SBATCH --job-name=tts-rl-server
#SBATCH --account=grahamneubig
#SBATCH --partition=mi3008x
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=1-00:00:00
#SBATCH --output=../logs/%x-%j.out
#SBATCH --error=../logs/%x-%j.out

set -euo pipefail

# Absolute path (sbatch runs a spooled copy, so BASH_SOURCE is unreliable here).
PROJECT_DIR=/work1/grahamneubig/lsutawik/tts-tokens-that-suffice
MODE="${1:-rl}"

echo "=== JOB $SLURM_JOB_ID on $SLURMD_NODENAME (partition $SLURM_JOB_PARTITION) ==="
echo "=== server will listen on $SLURMD_NODENAME:9123 (mode=$MODE) ==="

cd "$PROJECT_DIR"

# Export secrets/env (HF_TOKEN, etc.) to run.sh if present. set -a so every
# `export`-ed var in .env reaches the apptainer --env passthrough in run.sh.
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.env"
    set +a
fi

# run.sh holds the apptainer/server process in the foreground, so the job
# stays alive until the server exits, scancel, or the time limit.
exec bash server/scripts/run.sh "$MODE"
