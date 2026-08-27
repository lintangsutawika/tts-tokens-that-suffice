#!/usr/bin/env bash
# Downstream agent eval on three SWE benchmarks via Harbor + Modal + mini-swe-agent.
#
#   1. SWE-bench Verified   -> swe-bench/swe-bench-verified          (500, test-graded)
#   2. SWE-bench Pro        -> scale-ai/swe-bench-pro                (731, test-graded)
#   3. Senior SWE-bench     -> snorkel-ai/senior-swe-bench-v2026.06  ( 50, LLM-judge)
#
# All three are pre-published Harbor datasets (no adapter needed). Harbor installs
# mini-swe-agent inside a Modal sandbox per task, the agent solves it by calling
# our deliberator model, and Harbor runs the task's verifier and records reward.
#
# This is the BASELINE arm: stock mini-swe-agent, full context, no summarizer.
# The in-loop summarizer compaction plugs in later as a custom Harbor agent
# (module.path:ClassName passed to `-a`); see harbor/README.md.
#
# The deliberator is a SELF-HOSTED Qwen served over an OpenAI-compatible endpoint.
# Because the agent runs inside Modal, that endpoint MUST be reachable from the
# public internet (e.g. an ngrok tunnel to your vLLM/litellm proxy) and its host
# is added to the sandbox's egress allowlist via --allow-agent-host.
#
# Usage:
#   DELIBERATOR_BASE_URL=https://<tunnel>/v1 scripts/eval/harbor_eval.sh verified
#   DELIBERATOR_BASE_URL=https://<tunnel>/v1 N_TASKS=20 N_CONCURRENT=8 \
#       scripts/eval/harbor_eval.sh pro
#   # Senior needs a judge model + its credentials (see JUDGE_* below):
#   DELIBERATOR_BASE_URL=https://<tunnel>/v1 JUDGE_MODEL=anthropic/claude-opus-4-1 \
#       ANTHROPIC_API_KEY=sk-... scripts/eval/harbor_eval.sh senior
#
# Prereqs: `uv tool install harbor`; `modal setup` (a ~/.modal.toml must exist);
# the deliberator endpoint publicly reachable.

set -euo pipefail

# Preserve deliberator/relay overrides passed by the caller (e.g.
# harbor_eval.sbatch's relay path) — .env also defines DELIBERATOR_* for the
# proxy path, and a plain `. .env` would clobber them, making the agent send the
# wrong key/URL (e.g. the proxy key to the relay -> 401 'bad relay token').
_PRESET_BASE_URL="${DELIBERATOR_BASE_URL:-}"
_PRESET_API_KEY="${DELIBERATOR_API_KEY:-}"
[ -f .env ] && . .env
[ -n "${_PRESET_BASE_URL}" ] && DELIBERATOR_BASE_URL="${_PRESET_BASE_URL}"
[ -n "${_PRESET_API_KEY}" ] && DELIBERATOR_API_KEY="${_PRESET_API_KEY}"

BENCH="${1:-${BENCH:-}}"
if [ -z "${BENCH}" ]; then
    echo "usage: scripts/eval/harbor_eval.sh <verified|pro|senior>" >&2
    exit 2
fi

case "${BENCH}" in
    verified) DATASET="swe-bench/swe-bench-verified" ;;
    pro)      DATASET="scale-ai/swe-bench-pro" ;;
    senior)   DATASET="snorkel-ai/senior-swe-bench-v2026.06" ;;
    *) echo "unknown benchmark '${BENCH}' (want verified|pro|senior)" >&2; exit 2 ;;
esac

# --- Deliberator (task-solving model) -------------------------------------------
# provider/model string; `openai/<name>` routes litellm at an OpenAI-compatible
# endpoint using OPENAI_BASE_URL, posting `<name>` as the model. Match <name> to
# what your server actually serves (raw vLLM: the HF path; litellm proxy: the alias).
MODEL="${MODEL:-openai/Qwen/Qwen3.6-35B-A3B}"
# PUBLIC OpenAI-compatible base URL for the deliberator (must end in /v1 for vLLM).
DELIBERATOR_BASE_URL="${DELIBERATOR_BASE_URL:-}"
# vLLM ignores the key's value but mini-swe-agent requires a non-empty key.
DELIBERATOR_API_KEY="${DELIBERATOR_API_KEY:-EMPTY}"
MINI_CONFIG="${MINI_CONFIG:-harbor/mini_qwen.yaml}"

if [ -z "${DELIBERATOR_BASE_URL}" ]; then
    echo "ERROR: set DELIBERATOR_BASE_URL to a PUBLIC OpenAI-compatible endpoint" >&2
    echo "       (the Modal sandbox calls it; a localhost URL is unreachable)." >&2
    exit 2
fi

# Host to add to the sandbox egress allowlist (strip scheme, path, and port).
ALLOW_HOST="${ALLOW_HOST:-$(printf '%s' "${DELIBERATOR_BASE_URL}" \
    | sed -E 's#^[a-zA-Z]+://##; s#/.*$##; s#:[0-9]+$##')}"

# --- Run knobs ------------------------------------------------------------------
ENVIRONMENT="${ENV:-modal}"        # docker|modal|daytona|... (default modal)
N_CONCURRENT="${N_CONCURRENT:-4}"  # parallel trials
N_TASKS="${N_TASKS:-}"             # limit; empty = whole dataset
JOBS_DIR="${JOBS_DIR:-jobs}"
JOB_NAME="${JOB_NAME:-${BENCH}-mini-baseline-$(date +%Y%m%d-%H%M%S)}"
INCLUDE="${INCLUDE:-}"             # -i glob (task-name filter), optional
EXCLUDE="${EXCLUDE:-}"             # -x glob, optional

# --- Senior judge (verifier) ----------------------------------------------------
# Senior grades with an LLM judge inside the verifier phase. Point it at a strong
# model and supply that provider's key. SSB_OVERRIDE_ALL_JUDGE_MODEL overrides
# every judge/classifier stage at once. The verifier runs on a `public` network,
# so it reaches Anthropic/OpenAI/Portkey directly.
JUDGE_MODEL="${JUDGE_MODEL:-}"

HARBOR_BIN="${HARBOR_BIN:-harbor}"
command -v "${HARBOR_BIN}" >/dev/null 2>&1 || {
    echo "ERROR: '${HARBOR_BIN}' not found. Install with: uv tool install harbor" >&2
    exit 127
}

# --- Assemble the command -------------------------------------------------------
ARGS=(
    run
    -d "${DATASET}"
    -a mini-swe-agent
    -m "${MODEL}"
    -e "${ENVIRONMENT}"
    -n "${N_CONCURRENT}"
    --job-name "${JOB_NAME}"
    -o "${JOBS_DIR}"
    # Deliberator credentials, injected into the agent env inside the sandbox.
    --ae "OPENAI_BASE_URL=${DELIBERATOR_BASE_URL}"
    --ae "OPENAI_API_KEY=${DELIBERATOR_API_KEY}"
    --ae "MSWEA_API_KEY=${DELIBERATOR_API_KEY}"
    # Qwen sampling / thinking (see harbor/mini_qwen.yaml).
    --ak "config_file=${MINI_CONFIG}"
    # Let the sandbox reach the (otherwise blocked) deliberator host.
    --allow-agent-host "${ALLOW_HOST}"
    -y
)

[ -n "${N_TASKS}" ] && ARGS+=( -l "${N_TASKS}" )
[ -n "${INCLUDE}" ] && ARGS+=( -i "${INCLUDE}" )
[ -n "${EXCLUDE}" ] && ARGS+=( -x "${EXCLUDE}" )

if [ "${BENCH}" = "senior" ]; then
    [ -n "${JUDGE_MODEL}" ] && ARGS+=( --ve "SSB_OVERRIDE_ALL_JUDGE_MODEL=${JUDGE_MODEL}" )
    # Forward whichever judge credentials are present to the verifier phase.
    for k in ANTHROPIC_API_KEY OPENAI_API_KEY PORTKEY_API_KEY; do
        v="${!k:-}"; [ -n "${v}" ] && ARGS+=( --ve "${k}=${v}" )
    done
    if [ -z "${JUDGE_MODEL}" ] \
       && [ -z "${ANTHROPIC_API_KEY:-}" ] \
       && [ -z "${OPENAI_API_KEY:-}" ] \
       && [ -z "${PORTKEY_API_KEY:-}" ]; then
        echo "WARNING: senior needs a judge model + credentials; set JUDGE_MODEL and" >&2
        echo "         ANTHROPIC_API_KEY / OPENAI_API_KEY / PORTKEY_API_KEY." >&2
    fi
fi

echo "=================================================================="
echo "BENCH=${BENCH}  DATASET=${DATASET}"
echo "MODEL=${MODEL}  ENV=${ENVIRONMENT}  n=${N_CONCURRENT}  tasks=${N_TASKS:-all}"
echo "deliberator=${DELIBERATOR_BASE_URL}  allow-host=${ALLOW_HOST}"
[ "${BENCH}" = "senior" ] && echo "judge=${JUDGE_MODEL:-<task default>}"
echo "job=${JOB_NAME} -> ${JOBS_DIR}/"
echo "=================================================================="

exec "${HARBOR_BIN}" "${ARGS[@]}"
