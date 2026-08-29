#!/usr/bin/env bash
#SBATCH --job-name=tts-harbor-eval
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --nodes=1
#SBATCH --time=0-24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
# Send SIGTERM to the batch script (B:) 300s BEFORE the time limit, so the cleanup
# trap has a generous window to stop the ephemeral relay + vLLM and flush partial
# results before Slurm's hard SIGKILL. Without this, on TIMEOUT the trap only gets
# the default KillWait window (~30s) and may be killed mid-teardown, orphaning the
# relay app. NOTE: this signal fires 300s before the limit regardless, so size
# --time long enough that setup (multi-minute vLLM load) finishes well before then;
# a too-short --time makes this TERM land during setup and tear the job down early.
#SBATCH --signal=B:TERM@300
# NOTE: GPU/partition/memory are NOT hardcoded here because they differ per
# cluster and cannot be conditionalized in a static #SBATCH header. Pass them at
# submit time (CLI flags override #SBATCH):
#   babel (NVIDIA, gpu gres):  --partition=neulab --qos=neulab_qos \
#                              --constraint=nvlink --gres=gpu:4 --mem-per-gpu=64G
#   hpcfund (AMD, whole-node): --account=grahamneubig --partition=mi3008x --exclusive
# hpcfund MI nodes define no `gpu` gres; requesting --gres=gpu:N there fails with
# "Invalid generic resource (gres) specification" — use --exclusive instead.

# One Slurm job that (optionally) serves the deliberator with vLLM on this node's
# GPUs and runs the Harbor + Modal + mini-swe-agent eval against it, on the SAME
# node. Delegates the eval itself to scripts/eval/harbor_eval.sh.
#
# Two modes ("allow both"):
#   * DELIBERATOR_BASE_URL set  -> use that endpoint (e.g. the cmu.litellm.ai
#     proxy); no local server. GPUs aren't needed — submit with
#     `sbatch --gres=gpu:0 -p <cpu-partition>` to avoid wasting them.
#   * DELIBERATOR_BASE_URL unset -> spawn vLLM here, expose it, eval against it.
#
# REACHABILITY: the agent runs inside a Modal sandbox and calls the deliberator
# back over the network, so the served vLLM must be reachable from Modal at
# VLLM_PUBLIC_HOST (defaults to this node's FQDN). On a cluster whose compute
# nodes are not publicly routable this is the piece a tunnel used to solve —
# set VLLM_PUBLIC_HOST to a routable address, or front the vLLM with your proxy
# and pass DELIBERATOR_BASE_URL instead.
#
# Serving backend (SERVE_BACKEND): cuda (NVIDIA, default) | rocm-mi300 (gfx942) |
# rocm-mi250 (gfx90a). The ROCm variants serve inside the apptainer --rocm vLLM
# container; submit them with the AMD cluster's own SBATCH flags (they override
# the NVIDIA header below) and point VLLM_ROCM_SIF at a local .sif on batch nodes.
#
# Resume: reruns pick up completed trials only when JOB_NAME (and JOBS_DIR) match a
# prior run's job dir -- Harbor keeps trials with a valid result.json and re-runs the
# rest, but refuses to resume a dir whose JobConfig differs. Pass a STABLE JOB_NAME
# to resume; the default is timestamped, so each plain rerun starts fresh.
#
# Usage (GPU/partition/mem flags are passed at submit time — see header note):
#   # babel (NVIDIA): serve Qwen locally on this node's GPUs, then eval Verified
#   sbatch --partition=neulab --qos=neulab_qos --constraint=nvlink \
#       --gres=gpu:4 --mem-per-gpu=64G scripts/eval/harbor_eval.sbatch verified
#   # AMD MI300X: serve in the ROCm container, DP=8/TP=1, on hpcfund (whole node)
#   SERVE_BACKEND=rocm-mi300 DATA_PARALLEL=8 TENSOR_PARALLEL=1 \
#       VLLM_ROCM_SIF=/work1/.../vllm-openai-rocm-0.20.2.sif VLLM_MODEL=Qwen/Qwen3.5-9B \
#       sbatch --account=grahamneubig --partition=mi3008x --exclusive \
#           scripts/eval/harbor_eval.sbatch verified
#   # AMD MI250X (gfx90a): same, rocm-mi250 + the mi2508x partition
#   SERVE_BACKEND=rocm-mi250 DATA_PARALLEL=8 TENSOR_PARALLEL=1 \
#       VLLM_ROCM_SIF=/work1/.../vllm-openai-rocm-0.20.2.sif \
#       sbatch --account=grahamneubig --partition=mi2508x --exclusive \
#           scripts/eval/harbor_eval.sbatch verified
#   # resume a partially-completed run (same JOB_NAME + JOBS_DIR)
#   JOB_NAME=verified-qwen3.5-9b-run1 sbatch scripts/eval/harbor_eval.sbatch verified
#   # use the hosted proxy instead (no GPUs); Pro, 50 tasks
#   DELIBERATOR_BASE_URL=https://cmu.litellm.ai/v1 N_TASKS=50 \
#       sbatch --gres=gpu:0 scripts/eval/harbor_eval.sbatch pro
#   # Senior needs a judge model + creds (forwarded from env by harbor_eval.sh)
#   JUDGE_MODEL=anthropic/claude-opus-4-1 \
#       sbatch scripts/eval/harbor_eval.sbatch senior

set -euo pipefail

# A batch script's $0 is the spooled copy, not this file, so resolve the repo from
# the submit dir (where the job was submitted — expected to be the repo root).
# Slurm: $SLURM_SUBMIT_DIR. PBS/Torque (e.g. ABCI `qsub`): $PBS_O_WORKDIR -- crucial
# there, since PBS starts the job in $HOME, not the submit dir. Fall back to $(pwd).
cd "${SLURM_SUBMIT_DIR:-${PBS_O_WORKDIR:-$(pwd)}}"

# Live log for tail -f DURING the run. Slurm writes logs/%j.out live already (see the
# #SBATCH --output above), but PBS/Torque (ABCI) SPOOLS stdout and only copies it back
# when the job ENDS -- so there's nothing to tail while it runs. Under PBS, tee all
# output to logs/<PBS_JOBID>.out on the shared FS so you can `tail -f` it live.
if [ -n "${PBS_JOBID:-}" ] && [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p logs
    exec > >(tee -a "logs/${PBS_JOBID}.out") 2>&1
    echo "[log] teeing output to logs/${PBS_JOBID}.out  (tail -f it)"
fi

[ -f .env ] && . .env

# BENCH: Slurm passes it positionally (`sbatch script verified` -> $1); PBS/qsub does
# NOT take script args, so pass it as an env var (BENCH=verified, exported via `-V`).
# Defaults to verified when neither is set.
BENCH="${1:-${BENCH:-verified}}"

# --- deliberator / serving decision ---------------------------------------------
# Provide DELIBERATOR_BASE_URL to use an existing endpoint; leave it empty to
# serve vLLM here. SERVE_VLLM=1/0 forces the choice.
DELIBERATOR_BASE_URL="${DELIBERATOR_BASE_URL:-}"
if [ -n "${DELIBERATOR_BASE_URL}" ]; then
    SERVE_DEFAULT=0
else
    SERVE_DEFAULT=1
fi
SERVE_VLLM="${SERVE_VLLM:-${SERVE_DEFAULT}}"

# Pick a random ephemeral TCP port that nothing is currently listening on, so
# concurrent jobs on a shared node don't collide. Retries a fresh random port
# until a free one is found.
pick_free_port() {
    local p
    for _ in $(seq 1 64); do
        p=$(( (RANDOM % 16384) + 49152 ))
        ss -Htln 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${p}\$" && continue
        echo "${p}"; return 0
    done
    echo "${p}"  # give up gracefully; caller may still race, but very unlikely
}

# Model: set MODEL to the HF repo you're evaluating, e.g. MODEL=zai-org/GLM-4.7-Flash.
# That single knob drives serving, the agent, both per-model config files, and JOB_NAME.
# vLLM serves the bare repo (VLLM_MODEL, below); litellm's -m gets openai/<repo> (added
# in harbor_eval.sh -- do NOT put openai/ in MODEL). A leading openai/ is tolerated
# (stripped), except the openai-org repo openai/gpt-oss-120b (kept). VLLM_MODEL still
# works as an alias if you'd rather set that.
MODEL="${MODEL:-${VLLM_MODEL:-Qwen/Qwen3.6-35B-A3B}}"
case "${MODEL}" in openai/*/*) MODEL="${MODEL#openai/}" ;; esac   # tolerate litellm prefix
VLLM_MODEL="${MODEL}"          # what `vllm serve` gets (bare HF repo)
export MODEL                   # harbor_eval.sh re-derives litellm/config paths identically

# Deterministic default JOB_NAME: <bench>-<model repo, / -> -->-run-<RUN> -- e.g.
# verified-zai-org--GLM-4.7-Flash-run-0. RUN defaults to 0; bump RUN=1,2,... for repeat
# runs of the same data+model. It MUST be deterministic: a resumed 4h chunk reproduces
# the SAME name -> same relay (identity derives from JOB_NAME) and same job dir (resume
# keys off it); a timestamped name would start a brand-new run every chunk and never
# resume. Set JOB_NAME explicitly to override. Exported so harbor_eval.sh matches.
RUN="${RUN:-0}"
export JOB_NAME="${JOB_NAME:-${BENCH}-${VLLM_MODEL//\//--}-run-${RUN}}"

# --- Per-model vLLM serve params (serve/models/<repo>.json) ----------------------
# Model-INTRINSIC serving flags -- tool/reasoning parsers, speculative decoding,
# context length -- live in serve/models/<VLLM_MODEL>.json, auto-selected here. This
# is the SERVING analogue of harbor/configs/<repo>.yaml (agent sampling): the correct
# flags travel WITH the model, because the wrong tool-call parser silently zeroes the
# eval (memory: vllm-tool-parser-gotcha). Node/accelerator knobs (TP/DP, backend,
# gpu-mem, sif) stay OUT of this file -- the same model serves on different hardware.
# The loader only fills VLLM_EXTRA_ARGS / SPEC_CONFIG / MAX_MODEL_LEN when they are
# not already in the environment (explicit override always wins) and only for keys
# present in the JSON; the `:-` defaults below remain the fallback when no file
# matches (so an unconfigured Qwen run behaves exactly as before). Override the path
# with VLLM_SERVE_CONFIG.
VLLM_SERVE_CONFIG="${VLLM_SERVE_CONFIG:-serve/models/${VLLM_MODEL}.json}"
if [ -f "${VLLM_SERVE_CONFIG}" ]; then
    echo "[serve] loading vLLM serve params from ${VLLM_SERVE_CONFIG}"
    eval "$(python3 - "${VLLM_SERVE_CONFIG}" <<'PY'
import json, os, shlex, sys
c = json.load(open(sys.argv[1]))
def emit(var, val):
    if os.environ.get(var):            # explicit env/CLI override wins
        return
    print(f"{var}={shlex.quote(val)}")
if isinstance(c.get("extra_args"), list):
    emit("VLLM_EXTRA_ARGS", " ".join(str(a) for a in c["extra_args"]))
if c.get("spec_config"):
    emit("SPEC_CONFIG", json.dumps(c["spec_config"], separators=(",", ":")))
if c.get("max_model_len"):
    emit("MAX_MODEL_LEN", str(c["max_model_len"]))
PY
)"
elif [ "${SERVE_VLLM:-}" = "1" ] && [ -z "${VLLM_EXTRA_ARGS:-}" ]; then
    echo "WARNING: no serve config ${VLLM_SERVE_CONFIG} and no VLLM_EXTRA_ARGS set;" >&2
    echo "         falling back to built-in Qwen defaults -- WRONG for a non-Qwen model." >&2
fi

VLLM_PORT="${VLLM_PORT:-$(pick_free_port)}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
# Address at which Modal reaches this node's vLLM. MUST be routable from Modal.
VLLM_PUBLIC_HOST="${VLLM_PUBLIC_HOST:-$(hostname -f)}"
# vLLM ignores the key value, but mini-swe-agent requires a non-empty key.
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
# Serving backend on THIS node's GPUs:
#   cuda        -> `uv run vllm serve` (NVIDIA bare-metal, the local .venv/conda vLLM)
#   cuda-nv     -> apptainer --nv from VLLM_CUDA_SIF (NVIDIA container, e.g. H100/H200)
#   rocm-mi300  -> apptainer --rocm on MI300X/gfx942 (AITER + FP8 MFMA + ROCM_AITER_FA)
#   rocm-mi250  -> apptainer --rocm on MI250X/gfx90a (Triton FA; no AITER/FP8-MFMA)
# The two ROCm variants mirror scripts/serve/vllm-MI300.sh / vllm-MI250.sh.
SERVE_BACKEND="${SERVE_BACKEND:-cuda}"
DATA_PARALLEL="${DATA_PARALLEL:-1}"   # ROCm FP8 fits per-GPU: set DP=8 TP=1 there
# ROCm-only knobs. On batch nodes point VLLM_ROCM_SIF at a LOCAL .sif (a compute
# node may have no registry access); ROCM_CACHE_BASE holds the container binds.
# apptainer and singularity are the same tool (apptainer is the renamed fork; the
# exec --nv/--rocm/--bind/--env flags are identical). Nodes ship one or the other, so
# auto-pick whichever is on PATH (prefer apptainer). Falls back to the name "apptainer"
# for a clear "command not found" if neither exists. Override with APPTAINER_BIN.
APPTAINER_BIN="${APPTAINER_BIN:-$(command -v apptainer || command -v singularity || echo apptainer)}"
VLLM_ROCM_SIF="${VLLM_ROCM_SIF:-docker://vllm/vllm-openai-rocm:latest}"
# CUDA/NVIDIA SIF for SERVE_BACKEND=cuda-nv (apptainer --nv). Build with
# server/scripts/build-cuda.sh, or point at docker://vllm/vllm-openai:<ver>.
VLLM_CUDA_SIF="${VLLM_CUDA_SIF:-docker://vllm/vllm-openai:v0.28.0}"
ROCM_CACHE_BASE="${ROCM_CACHE_BASE:-$(pwd)}"
# Back apptainer's session/overlay on a WRITABLE /tmp dir. hpcfund's apptainer.conf
# declares system bind paths (e.g. /scratch) that this user can't create -> the
# container start dies with "mkdir: cannot create directory '/scratch': Permission
# denied". We provide every bind we need explicitly below, so we also pass
# `--no-mount bind-paths` to skip those config binds. TMPDIR must be a real, owned,
# writable dir (compute nodes have a local /tmp); override APPTAINER_TMPDIR to point
# elsewhere if this node's /tmp is tiny.
# Default derives from BASE_DIR (set in .env, e.g. hpcfund BASE_DIR=/scratch/) ->
# ${BASE_DIR}/lsutawik/apptainer_tmp; falls back to a per-job /tmp dir when BASE_DIR
# is unset (e.g. babel). An explicit APPTAINER_TMPDIR in the env always wins.
APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${BASE_DIR:+${BASE_DIR%/}/lsutawik/apptainer_tmp}}"
APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/tmp/apptainer-${SLURM_JOB_ID:-$$}}"
SHUFFLE_KV="${SHUFFLE_KV:-0}"         # MI300 only; set 1 for >=32 concurrent/replica
# Optional speculative decoding, only for checkpoints that support it (e.g. an
# FP8 MTP model): SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":3}'.
SPEC_CONFIG="${SPEC_CONFIG:-}"
# Optional Modal reverse relay (harbor/relay/) so this private node's vLLM is
# reachable from Modal sandboxes without inbound access. Set RELAY_URL to the
# deployed relay base (https://<workspace>--tts-vllm-relay-web.modal.run) and
# RELAY_SECRET to the shared secret (Modal 'tts-relay'). When set, the node dials
# out to the relay and the eval talks to the relay instead of the node address.
RELAY_URL="${RELAY_URL:-}"
RELAY_SECRET="${RELAY_SECRET:-}"
# EPHEMERAL_RELAY=1: this run deploys its OWN relay app, named + secreted
# DETERMINISTICALLY from JOB_NAME (tts-vllm-relay-<job-slug>-<hash>), then stops it
# on exit. This is the safe way to run multiple models at once: the shared relay is
# a singleton (one bridge socket), so two concurrent jobs on it steal each other's
# traffic; distinct JOB_NAMEs get distinct relays, so there is nothing to collide
# over. Because identity is a function of JOB_NAME (not the Slurm attempt), a resume
# of the same job reproduces the SAME relay URL+secret -> the endpoint recorded in
# the job config comes back live and `harbor run --job-name <same>` / `harbor job
# resume -p` both work. AUTHORITATIVE: overrides any inherited RELAY_URL.
EPHEMERAL_RELAY="${EPHEMERAL_RELAY:-0}"
# Extra vLLM flags (parsers etc.). Keep --tool-call-parser qwen3_coder for Qwen
# (hermes silently drops the <function=..> XML tool calls -> RepeatedFormatError).
# DO NOT add --reasoning-parser qwen3 for the AGENT server. Proven at the raw-
# response level (job smoke-relay-qwen3.5-9b-005431, django-16938): with thinking
# on + function-calling mini (-c mini), the 9B closes </think> only ~half the time;
# on the ~50% of turns it omits </think>, the reasoning parser sweeps the valid
# <tool_call> into reasoning_content, leaving content/tool_calls empty -> "No tool
# calls found" FormatError (72 clean vs 75 swallowed turns, 1 AgentTimeoutError,
# Mean 0.000). With the reasoning parser OFF, the <tool_call> (emitted before the
# often-omitted </think>) stays in content where --tool-call-parser qwen3_coder
# extracts it; thinking still happens, just inline in content. (The reward-scoring
# server scripts/serve/serve_local.sh keeps --reasoning-parser qwen3 on purpose.)
# --language-model-only skips the vision tower on multimodal Qwen3.5 (9B/27B).
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---enable-chunked-prefill --max-num-batched-tokens 4096 --enable-prefix-caching --long-prefill-token-threshold 0 --enforce-eager --enable-auto-tool-choice --language-model-only --tool-call-parser qwen3_coder}"

VLLM_PID=""
BRIDGE_PID=""
HARBOR_PID=""       # the backgrounded harbor_eval.sh (execs into harbor)
RELAY_APP_NAME=""   # set only when EPHEMERAL_RELAY deploys a per-job relay app
_CLEANED_UP=0
cleanup() {
    # Idempotent: SIGTERM-then-EXIT would otherwise run this twice.
    [ "${_CLEANED_UP}" = "1" ] && return; _CLEANED_UP=1
    # Stop harbor FIRST, gracefully (SIGINT), so in-flight trials cancel cleanly and
    # write CancelledError result.jsons -- which `harbor job resume` re-runs next
    # chunk -- BEFORE we pull vLLM/relay out from under them. If harbor were left
    # running against a dead model it would instead mark trials with real errors that
    # a plain resume would KEEP as failures. The --signal=B:TERM@300 lead time is what
    # makes this graceful window possible on a timeout. Cancelling doesn't need the
    # model, so keep vLLM/relay up during this wait.
    if [ -n "${HARBOR_PID}" ] && kill -0 "${HARBOR_PID}" 2>/dev/null; then
        echo "[cleanup] stopping harbor (pid=${HARBOR_PID}); cancelling in-flight trials"
        kill -INT "${HARBOR_PID}" 2>/dev/null || true
        for _ in $(seq 1 30); do kill -0 "${HARBOR_PID}" 2>/dev/null || break; sleep 1; done
        kill -TERM "${HARBOR_PID}" 2>/dev/null || true
    fi
    # Then stop the ephemeral relay: the outward, billable, collision-prone resource.
    # Prefer a direct `modal` (on the venv PATH via harbor[modal]) over a cold
    # `uv run --with modal` env-resolve so it finishes promptly.
    if [ -n "${RELAY_APP_NAME}" ]; then
        echo "[cleanup] stopping ephemeral relay app ${RELAY_APP_NAME}"
        # -y: `modal app stop` prompts for confirmation; in a batch job stdin is not
        # a TTY, so without --yes it stalls and the relay never actually stops.
        # </dev/null belt-and-suspenders so it can never block on input.
        if command -v modal >/dev/null 2>&1; then
            modal app stop -y "${RELAY_APP_NAME}" </dev/null 2>/dev/null || true
        elif [ -x ".venv/bin/modal" ]; then
            .venv/bin/modal app stop -y "${RELAY_APP_NAME}" </dev/null 2>/dev/null || true
        else
            uv run --with modal modal app stop -y "${RELAY_APP_NAME}" </dev/null 2>/dev/null || true
        fi
    fi
    [ -n "${BRIDGE_PID}" ] && { echo "[cleanup] stopping relay bridge (pid=${BRIDGE_PID})";
        kill "${BRIDGE_PID}" 2>/dev/null; }
    [ -n "${VLLM_PID}" ] && { echo "[cleanup] stopping vLLM (pid=${VLLM_PID})";
        kill "${VLLM_PID}" 2>/dev/null; wait "${VLLM_PID}" 2>/dev/null; }
}
trap cleanup EXIT SIGTERM SIGINT

if [ "${SERVE_VLLM}" = "1" ]; then
    echo "[serve] launching vLLM ${VLLM_MODEL} on ${VLLM_PUBLIC_HOST}:${VLLM_PORT}" \
         "(backend=${SERVE_BACKEND} TP=${TENSOR_PARALLEL} DP=${DATA_PARALLEL})"
    # Speculative decoding is opt-in (model-specific); off unless SPEC_CONFIG is set.
    if [ -n "${SPEC_CONFIG}" ]; then SPEC_ARGS=( --speculative-config "${SPEC_CONFIG}" ); else SPEC_ARGS=(); fi

    # ABCI/PBS (and some cgroup setups) hand GPUs to the job by UUID in
    # CUDA_VISIBLE_DEVICES; vLLM's device_id_to_physical_device_id does int() on each
    # entry and dies on a 'GPU-<uuid>'. For the NVIDIA backends, rewrite it to integer
    # indices 0..N-1 (correct on a full-node allocation). No-op if already numeric/unset.
    case "${SERVE_BACKEND}" in
      cuda|cuda-nv)
        if printf '%s' "${CUDA_VISIBLE_DEVICES:-}" | grep -q '[^0-9,]'; then
            _NGPU=$(printf '%s' "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)
            export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((_NGPU - 1)))"
            echo "[serve] normalized CUDA_VISIBLE_DEVICES (UUIDs) -> ${CUDA_VISIBLE_DEVICES}"
        fi ;;
    esac

    case "${SERVE_BACKEND}" in
      cuda)
        # vLLM runs from the conda/uv env whose libzmq needs GLIBCXX_3.4.30; some older
        # compute nodes (e.g. neulab babel-t9-*) ship a system libstdc++ without it.
        # Put conda's own libstdc++ ahead of the system one so the loader finds it.
        CONDA_LIB="${CONDA_PREFIX:-$HOME/miniconda3}/lib"
        [ -f "${CONDA_LIB}/libstdc++.so.6" ] && export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH:-}"
        # Mirrors scripts/serve/serve_local.sh (Qwen3 deliberator, tool + reasoning parsers).
        uv run vllm serve "${VLLM_MODEL}" \
            --host 0.0.0.0 \
            --port "${VLLM_PORT}" \
            --tensor-parallel-size "${TENSOR_PARALLEL}" \
            --data-parallel-size "${DATA_PARALLEL}" \
            --max-model-len "${MAX_MODEL_LEN}" \
            --disable-custom-all-reduce \
            --gpu-memory-utilization "${GPU_MEM_UTIL}" \
            "${SPEC_ARGS[@]}" ${VLLM_EXTRA_ARGS} &
        VLLM_PID=$!
        ;;
      cuda-nv)
        # NVIDIA analog of the rocm-mi300 path: serve inside the CUDA vLLM container
        # (apptainer --nv) from VLLM_CUDA_SIF (server/scripts/build-cuda.sh). No ROCm/
        # AITER env; --nv exposes the GPUs + driver. Model-intrinsic flags (parsers,
        # spec, attention backend) ride in VLLM_EXTRA_ARGS from serve/models/<repo>.json
        # -- NOT the rocm-only ROCM_SERVE_EXTRA. ROCM_CACHE_BASE is reused as a generic
        # host cache dir so HF weights persist. If the site's apptainer needs its
        # default binds stripped (e.g. a failing /scratch bind), pass extra flags via
        # APPTAINER_ARGS (e.g. APPTAINER_ARGS="--no-mount bind-paths").
        mkdir -p "${ROCM_CACHE_BASE}/hf_cache" "${ROCM_CACHE_BASE}/tmp" "${APPTAINER_TMPDIR}"
        # Set both prefixes: apptainer reads APPTAINER_TMPDIR, singularity SINGULARITY_TMPDIR.
        export APPTAINER_TMPDIR SINGULARITY_TMPDIR="${APPTAINER_TMPDIR}"
        "${APPTAINER_BIN}" exec --nv --writable-tmpfs \
            --workdir "${APPTAINER_TMPDIR}" \
            --bind "${ROCM_CACHE_BASE}/hf_cache:/root/.cache/huggingface" \
            --bind "${ROCM_CACHE_BASE}/tmp:/tmp_work" \
            --env TMPDIR=/tmp_work \
            --env HF_HOME=/root/.cache/huggingface \
            --env "HF_TOKEN=${HF_TOKEN:-}" \
            --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" \
            ${APPTAINER_ARGS:-} \
            "${VLLM_CUDA_SIF}" \
            vllm serve "${VLLM_MODEL}" \
                --host 0.0.0.0 \
                --port "${VLLM_PORT}" \
                --tensor-parallel-size "${TENSOR_PARALLEL}" \
                --data-parallel-size "${DATA_PARALLEL}" \
                --max-model-len "${MAX_MODEL_LEN}" \
                --gpu-memory-utilization "${GPU_MEM_UTIL}" \
                "${SPEC_ARGS[@]}" ${VLLM_EXTRA_ARGS} &
        VLLM_PID=$!
        ;;
      rocm-mi300|rocm-mi250)
        # Serve inside the ROCm vLLM container (apptainer --rocm). Mirrors
        # scripts/serve/vllm-MI300.sh (gfx942) / vllm-MI250.sh (gfx90a): the two
        # differ only in the accelerator env block below -- AITER (VLLM_ROCM_USE_AITER*,
        # the ROCM_AITER_FA backend) and FP8 padding are gfx942-only; gfx90a uses the
        # Triton FlashAttention path and runs the FP8 checkpoint dequantized.
        mkdir -p "${ROCM_CACHE_BASE}/hf_cache" "${ROCM_CACHE_BASE}/triton_cache" \
                 "${ROCM_CACHE_BASE}/tmp" "${ROCM_CACHE_BASE}/vllm_cache"
        # Writable /tmp-backed apptainer session dir (avoids the /scratch config bind).
        mkdir -p "${APPTAINER_TMPDIR}"
        # Set both prefixes: apptainer reads APPTAINER_TMPDIR, singularity SINGULARITY_TMPDIR.
        export APPTAINER_TMPDIR SINGULARITY_TMPDIR="${APPTAINER_TMPDIR}"
        ROCM_BINDS=(
            --bind "${ROCM_CACHE_BASE}/hf_cache:/root/.cache/huggingface"
            --bind "${ROCM_CACHE_BASE}/triton_cache:/triton_cache"
            --bind "${ROCM_CACHE_BASE}/tmp:/tmp_work"
            --bind "${ROCM_CACHE_BASE}/vllm_cache:/vllm_cache"
        )
        ROCM_ENV=(
            --env TMPDIR=/tmp_work
            --env TRITON_CACHE_DIR=/triton_cache
            --env VLLM_CACHE_ROOT=/vllm_cache
            --env HF_HOME=/root/.cache/huggingface
            --env OMP_NUM_THREADS=1
            --env ROCM_PATH=/opt/rocm
            --env TORCH_BLAS_PREFER_HIPBLASLT=1
            --env SAFETENSORS_FAST_GPU=1
            --env "HF_TOKEN=${HF_TOKEN:-}"
        )
        # Shared serve flags; accelerator-specific ones appended per variant.
        ROCM_SERVE_EXTRA=( --dtype auto --block-size 16 --disable-nccl-for-dp-synchronization )
        if [ "${SERVE_BACKEND}" = "rocm-mi300" ]; then
            mkdir -p "${ROCM_CACHE_BASE}/.aiter/jit"
            ROCM_BINDS+=( --bind "${ROCM_CACHE_BASE}/.aiter/jit:/aiter_jit" )
            ROCM_ENV+=(
                --env AITER_JIT_DIR=/aiter_jit
                --env HIP_FORCE_DEV_KERNARG=1
                --env NCCL_MIN_NCHANNELS=112
                --env VLLM_ROCM_USE_AITER=1
                --env VLLM_ROCM_USE_AITER_LINEAR=1
                --env VLLM_ROCM_USE_AITER_MHA=1
                --env VLLM_ROCM_USE_AITER_RMSNORM=1
                --env VLLM_ROCM_USE_SKINNY_GEMM=1
                --env VLLM_ROCM_FP8_PADDING=1
                --env "VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=${SHUFFLE_KV}"
            )
            ROCM_SERVE_EXTRA+=( --kv-cache-dtype fp8 --attention-backend ROCM_AITER_FA )
        else  # rocm-mi250 (gfx90a): Triton FA, no AITER/FP8-MFMA
            ROCM_ENV+=( --env FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE )
            ROCM_SERVE_EXTRA+=( --disable-custom-all-reduce )
        fi
        "${APPTAINER_BIN}" exec --rocm --writable-tmpfs --no-mount bind-paths \
            --workdir "${APPTAINER_TMPDIR}" \
            "${ROCM_BINDS[@]}" "${ROCM_ENV[@]}" \
            "${VLLM_ROCM_SIF}" \
            vllm serve "${VLLM_MODEL}" \
                --host 0.0.0.0 \
                --port "${VLLM_PORT}" \
                --tensor-parallel-size "${TENSOR_PARALLEL}" \
                --data-parallel-size "${DATA_PARALLEL}" \
                --max-model-len "${MAX_MODEL_LEN}" \
                --gpu-memory-utilization "${GPU_MEM_UTIL}" \
                "${ROCM_SERVE_EXTRA[@]}" "${SPEC_ARGS[@]}" ${VLLM_EXTRA_ARGS} &
        VLLM_PID=$!
        ;;
      *)
        echo "ERROR: unknown SERVE_BACKEND '${SERVE_BACKEND}' (want cuda|cuda-nv|rocm-mi300|rocm-mi250)" >&2
        exit 2
        ;;
    esac

    echo "[serve] waiting for vLLM /health on :${VLLM_PORT} ..."
    READY=0
    for _ in $(seq 1 180); do
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "ERROR: vLLM process exited during startup" >&2; exit 1
        fi
        if curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
            READY=1; break
        fi
        sleep 5
    done
    [ "${READY}" = "1" ] || { echo "ERROR: vLLM not ready after 900s" >&2; exit 1; }
    echo "[serve] vLLM ready."

    # MODEL is already the bare HF repo (model block above) and exported; harbor_eval.sh
    # adds the openai/ litellm prefix. Kept here as a no-op guard for a bare invocation.
    export MODEL="${MODEL:-${VLLM_MODEL}}"

    # EPHEMERAL_RELAY: deploy a per-run relay app, then let the normal relay path
    # below use it. This is AUTHORITATIVE: it overrides any inherited RELAY_URL
    # (e.g. .env's shared-relay default), because asking for an ephemeral relay means
    # you explicitly do NOT want the shared singleton. To target a specific relay
    # instead, just don't set EPHEMERAL_RELAY=1.
    #
    # Identity is DETERMINISTIC in JOB_NAME (not the Slurm job-id), so a resume of the
    # same job reproduces the SAME relay URL + secret: redeploying brings that exact
    # endpoint back live, and the URL/key already recorded in the job's agent config
    # still match -> `harbor run --job-name <same>` auto-resumes, and `harbor job
    # resume -p` reaches a live server. The relay is pure transport to the same model,
    # so making its identity a function of the run (not the attempt) is what lets a
    # resume repoint at a fresh-but-identical endpoint. Distinct runs use distinct
    # JOB_NAMEs -> distinct relays, so concurrent different-model runs stay isolated.
    if [ "${EPHEMERAL_RELAY}" = "1" ]; then
        RELAY_URL=""   # discard any inherited value; we deploy our own just below
        # Sanitize JOB_NAME into a valid modal app slug (lowercase alnum + dashes),
        # cap length, and append a short hash of the full name so names that collapse
        # to the same slug after sanitizing don't collide.
        _JOB_SLUG="$(printf '%s' "${JOB_NAME}" | tr '[:upper:]' '[:lower:]' \
            | tr -c 'a-z0-9' '-' | tr -s '-')"
        # Cap the slug at 26: the modal web URL label is <workspace>--<app>-web, a
        # DNS label limited to 63 chars. The fixed parts sum to 36 (neulab=6, '--'=2,
        # 'tts-vllm-relay-'=15, '-<hash8>'=9, '-web'=4), so slug<=27 keeps the label
        # <=63; 26 leaves a char of margin, and modal won't truncate+hash it so our
        # CONSTRUCTED URL matches. Distinct JOB_NAMEs whose slugs collide after the cut
        # still get distinct app names via the hash of the full name.
        _JOB_SLUG="${_JOB_SLUG#-}"; _JOB_SLUG="${_JOB_SLUG%-}"; _JOB_SLUG="${_JOB_SLUG:0:26}"
        _JOB_SLUG="${_JOB_SLUG%-}"   # re-strip in case the cut left a trailing dash
        _JOB_HASH="$(printf '%s' "${JOB_NAME}" | sha256sum | cut -c1-8)"
        RELAY_APP_NAME="tts-vllm-relay-${_JOB_SLUG}-${_JOB_HASH}"
        # Secret derived deterministically from JOB_NAME so a resume reproduces it,
        # salted with the shared .env RELAY_SECRET (stable + private) so the token
        # isn't guessable from the public job name/URL. Baked into the app via
        # Secret.from_dict in the deploy env; the same value flows to the bridge auth
        # and the eval's DELIBERATOR_API_KEY below.
        _RELAY_SALT="${RELAY_SECRET:-tts-relay-default-salt}"
        RELAY_SECRET="$(printf '%s' "${JOB_NAME}:${_RELAY_SALT}" | sha256sum | cut -c1-32)"
        # Close any pre-existing relay app of this deterministic name, then rebuild
        # fresh. The name derives from JOB_NAME, so prior (cancelled/orphaned) runs of
        # the same job leave a stale app; a bare `modal deploy` redeploys ONTO it while
        # it is mid-stop, and Modal then routes the web URL but keeps no container ready
        # -- /health accepts the TCP connection and hangs forever, wedging the bridge
        # registration wait below. The relay is stateless (a pure bridge forwarder), so
        # tearing it down and rebuilding every chunk is free and guarantees a healthy
        # container. Safe because same-JOB_NAME chunks are SEQUENTIAL (never concurrent).
        echo "[relay] stopping any existing relay app ${RELAY_APP_NAME} before rebuild ..."
        uv run --with modal modal app stop -y "${RELAY_APP_NAME}" </dev/null >/dev/null 2>&1 || true
        sleep 10   # let the stop propagate before deploying a fresh instance
        echo "[relay] deploying per-run relay app ${RELAY_APP_NAME} (resumable) ..."
        # NOTE: keep the deploy OUT of a pipeline and never post-process its output
        # with `grep ... | head` under `set -e -o pipefail` -- head closing the pipe
        # early makes grep die on SIGPIPE (141), which set -e treats as fatal and
        # exits the whole script SILENTLY (the job tore down ~8s after vLLM came up,
        # no error). We construct the URL deterministically instead of parsing.
        _DEPLOY_LOG="$(RELAY_APP_NAME="${RELAY_APP_NAME}" RELAY_SECRET="${RELAY_SECRET}" \
            uv run --with modal modal deploy harbor/relay/modal_relay.py 2>&1)" || {
            echo "${_DEPLOY_LOG}" >&2
            echo "ERROR: ephemeral relay deploy failed (is modal authenticated?)" >&2; exit 2; }
        # Construct the web URL from workspace + app name (modal's URL scheme is
        # https://<workspace>--<app>-web.modal.run) rather than scraping deploy output
        # -- robust to modal's output formatting across versions. RELAY_WORKSPACE
        # defaults to the workspace in .env's shared RELAY_URL (neulab). We also try to
        # read a modal.run URL back from the deploy log purely as a cross-check.
        RELAY_WORKSPACE="${RELAY_WORKSPACE:-neulab}"
        RELAY_URL="https://${RELAY_WORKSPACE}--${RELAY_APP_NAME}-web.modal.run"
        _PARSED_URL="$(printf '%s\n' "${_DEPLOY_LOG}" \
            | grep -m1 -oE 'https://[A-Za-z0-9._-]+\.modal\.run')" || true
        if [ -n "${_PARSED_URL}" ] && [ "${_PARSED_URL}" != "${RELAY_URL}" ]; then
            echo "[relay] NOTE: modal reported ${_PARSED_URL}, using that over constructed ${RELAY_URL}"
            RELAY_URL="${_PARSED_URL}"   # trust modal's actual URL if it differs
        fi
        echo "[relay] ephemeral relay live at ${RELAY_URL}"
    fi

    if [ -n "${RELAY_URL}" ]; then
        # Reach vLLM through the Modal relay: start the node bridge (dials out to
        # the relay and forwards to local vLLM), and have the agent call the relay.
        [ -n "${RELAY_SECRET}" ] || { echo "ERROR: RELAY_URL set but RELAY_SECRET empty" >&2; exit 2; }
        echo "[relay] starting node bridge -> ${RELAY_URL}"
        BRIDGE_WS="${RELAY_URL%/}/bridge"; BRIDGE_WS="${BRIDGE_WS/https:/wss:}"
        RELAY_WS_URL="${BRIDGE_WS}" RELAY_SECRET="${RELAY_SECRET}" \
        VLLM_LOCAL_URL="http://localhost:${VLLM_PORT}" \
            uv run --with websockets --with httpx python harbor/relay/bridge.py &
        BRIDGE_PID=$!
        export DELIBERATOR_BASE_URL="${RELAY_URL%/}/v1"
        export DELIBERATOR_API_KEY="${RELAY_SECRET}"
        echo "[relay] waiting for bridge to register with relay ..."
        # --max-time is REQUIRED: a wedged/half-started relay container answers the TCP
        # connect but never returns a body, and a timeout-less curl hangs forever (the
        # loop never advances, the job sits here past its wall clock). Bound each probe,
        # and if the bridge never registers, FAIL loudly -> cleanup fires -> the chunk
        # can be resubmitted, rather than silently launching harbor against a dead relay.
        _BRIDGE_OK=0
        for _ in $(seq 1 24); do
            if curl -sf --max-time 10 "${RELAY_URL%/}/health" 2>/dev/null \
                | grep -q '"bridge_connected":true'; then
                echo "[relay] bridge connected"; _BRIDGE_OK=1; break
            fi
            sleep 5
        done
        [ "${_BRIDGE_OK}" = "1" ] || {
            echo "ERROR: bridge never registered with relay ${RELAY_APP_NAME} (relay unhealthy)" >&2
            exit 2; }
    else
        # Direct node address (only works where Modal can route to it, e.g. -e singularity on-node).
        export DELIBERATOR_BASE_URL="http://${VLLM_PUBLIC_HOST}:${VLLM_PORT}/v1"
        export DELIBERATOR_API_KEY="${VLLM_API_KEY}"
    fi
else
    echo "[serve] using provided endpoint ${DELIBERATOR_BASE_URL} (no local vLLM)"
    export DELIBERATOR_BASE_URL
fi

echo "=================================================================="
echo "Harbor eval: BENCH=${BENCH}  deliberator=${DELIBERATOR_BASE_URL}"
echo "=================================================================="

# Delegate to the runner (it derives --allow-agent-host from the URL, wires Modal,
# and forwards judge creds for senior; it auto-RESUMEs if the job dir already exists).
# Run it in the BACKGROUND and wait, so a Slurm TERM (--signal=B:TERM@300) reaches our
# trap while harbor is still a killable child: cleanup SIGINTs it for a graceful,
# resumable checkpoint. `|| true` so harbor's non-zero exit (or our kill) doesn't trip
# `set -e` before the EXIT trap runs.
bash scripts/eval/harbor_eval.sh "${BENCH}" &
HARBOR_PID=$!
wait "${HARBOR_PID}" || true
