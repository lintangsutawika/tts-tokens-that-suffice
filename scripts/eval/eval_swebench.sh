#!/usr/bin/env bash
#SBATCH --job-name=tts-eval
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --mem=64G
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --time=0-24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1

# Downstream SWE-bench eval: run the agent with a summarizer compressing its
# context in-loop, for three arms (trained / base / truncation), and report the
# resolve rate of each.
#
# This job is CPU-only (like collect.sh) — it orchestrates agent runs and grades
# them in singularity containers.  It expects two servers to already be running:
#   * deliberator (task-solving) model, vLLM/litellm at PORT      (default :8000)
#   * tinker server for the summarizer sampling client at TINKER  (default :9123)
# Bring the deliberator up the same way as serve_and_collect.sh; tinker is the
# same server used for training.  The truncation arm needs no tinker server.
#
# Usage:
#   sbatch scripts/eval/eval_swebench.sh                 # all arms, slice 0:20
#   MODES=trained SLICE=0:50 sbatch scripts/eval/eval_swebench.sh
#   CHECKPOINT=tinker://model_7cf52d89/weights/000084 sbatch scripts/eval/eval_swebench.sh
#   PRESERVE_THINKING=true sbatch scripts/eval/eval_swebench.sh    # keep prior-turn <think> (off by default)

set -euo pipefail

[ -f .env ] && . .env

MODEL=${MODEL:-litellm_proxy/Qwen/Qwen3.6-35B-A3B}
DATASET=${DATASET:-swe-bench}
DATA_SOURCE=${DATA_SOURCE:-swe-bench}
SLICE=${SLICE:-0:20}
WORKERS=${WORKERS:-4}
PORT=${PORT:-8000}
# Deliberator (task-solving model) endpoint. Defaults to this node's :PORT; set
# the full URL when the deliberator runs on another node, e.g.
# DELIBERATOR_API_BASE=http://babel-w9-28:8000/v1
DELIBERATOR_API_BASE=${DELIBERATOR_API_BASE:-http://0.0.0.0:${PORT}/v1}
TINKER_BASE_URL=${TINKER_BASE_URL:-http://localhost:9123}

# Summarizer / compression knobs.
CHECKPOINT=${CHECKPOINT:-tinker://model_7cf52d89/weights/000084}
SUMMARIZER_MODEL=${SUMMARIZER_MODEL:-Qwen/Qwen3-8B}
# HF tokenizer for the compression trigger; keep as the base model even when
# SUMMARIZER_MODEL is a served LoRA name (e.g. openai/summarizer-rl).
SUMMARIZER_TOKENIZER=${SUMMARIZER_TOKENIZER:-Qwen/Qwen3-8B}
# Must match train_summarizer.sh RENDERER — sampling the LoRA under a different chat
# format than it trained on silently degrades it.
SUMMARIZER_RENDERER=${SUMMARIZER_RENDERER:-qwen3_disable_thinking}
# Must match train_summarizer.sh MAX_TOKENS.
SUMMARIZER_MAX_TOKENS=${SUMMARIZER_MAX_TOKENS:-512}
# Set to a vLLM endpoint (scripts/serve/serve_summarizer.sh) to summarize over litellm
# instead of tinker; leave empty to use the tinker server. When set, point
# SUMMARIZER_MODEL at the litellm string, e.g. openai/Qwen/Qwen3-8B.
SUMMARIZER_API_BASE=${SUMMARIZER_API_BASE:-}
# Must match the summarizer's training-time split (train_summarizer.sh SPLIT_AT_TOKENS),
# or the summarizer is invoked at context lengths it never trained on.
COMPRESS_AT_TOKENS=${COMPRESS_AT_TOKENS:-16384}
# If >0, trigger on complete-turn count instead of tokens (count-based, like
# training/OpenHands). Overrides COMPRESS_AT_TOKENS.
COMPRESS_AT_TURNS=${COMPRESS_AT_TURNS:-0}
KEEP_FIRST=${KEEP_FIRST:-4}
KEEP_LAST_TURNS=${KEEP_LAST_TURNS:-3}

# Deliberator sampling — Qwen3 non-thinking recommended settings. Greedy (temp=0)
# deterministically gets stuck when the model emits a no-tool-call response,
# tripping RepeatedFormatError; this stochasticity lets it recover. Control
# cross-arm variance with more instances/seeds, not by lowering temperature.
TEMP=${TEMP:-0.6}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:-20}
MIN_P=${MIN_P:-0.0}
PRESENCE_PENALTY=${PRESENCE_PENALTY:-0.0}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}

# Which arms to run (space-separated). `full` = no compression (true baseline);
# `truncation` = keep first+last, drop middle; `base`/`trained` = summarize.
# `MODE` (singular) is accepted as an alias for a single arm.
MODES=${MODES:-${MODE:-"full base trained"}}

# Re-run instances already present in preds.json (default: skip them).
REDO=${REDO:-false}
[ "${REDO}" = "true" ] && REDO_ARG="--redo-existing" || REDO_ARG=""

# Grade (score) with the SWE-bench harness in-process. Set GRADE=false to skip
# scoring and only write preds.json + trajectories (grade later / elsewhere).
GRADE=${GRADE:-true}
[ "${GRADE}" = "false" ] && GRADE_ARG="--no-grade" || GRADE_ARG="--grade"

# Keep prior-turn <think> reasoning in the deliberator's context across turns
# (Qwen3 strips it by default). Set PRESERVE_THINKING=false to disable; thinking
# itself stays enabled either way. The on/off runs write to different output
# dirs (…-preserve-thinking vs plain) so they don't collide.
PRESERVE_THINKING=${PRESERVE_THINKING:-false}
if [ "${PRESERVE_THINKING}" = "true" ]; then
    THINKING_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": true, "preserve_thinking": true}}'
    THINK_SUFFIX="-preserve-thinking"
else
    THINKING_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": true}}'
    THINK_SUFFIX=""
fi

# Output dir per arm: <task>__<deliberator basename>__<mode>, and for the arms
# that actually compress (base/trained) the compression trigger too, e.g.
#   swe-bench__Qwen3.6-35B-A3B__base__tok24000
#   swe-bench__Qwen3.6-35B-A3B__trained__turns32
MODEL_BASENAME=${MODEL##*/}
OUT_ROOT=${OUT_ROOT:-outputs}

# Compression-trigger label: turn-count wins when COMPRESS_AT_TURNS>0 (it
# overrides the token trigger in the eval), otherwise the token threshold.
if [ "${COMPRESS_AT_TURNS}" -gt 0 ] 2>/dev/null; then
    TRIGGER="turns${COMPRESS_AT_TURNS}"
else
    TRIGGER="tok${COMPRESS_AT_TOKENS}"
fi

# head/tail label: messages kept verbatim from the start + complete turns kept
# at the end. Applies to every compressing arm (base/trained/truncation); full
# keeps everything so it carries no head/tail.
HEADTAIL="h${KEEP_FIRST}t${KEEP_LAST_TURNS}"

# Output dir name for an arm:
#   full        -> <task>__<model>__full                       (no compression)
#   truncation  -> <task>__<model>__truncation__h<N>t<N>       (head/tail only)
#   base/trained-> <task>__<model>__<mode>__<trigger>__h<N>t<N> (trigger too)
arm_dir() {
    local mode="$1" name="${DATASET}__${MODEL_BASENAME}__$1"
    case "${mode}" in
        base|trained) name="${name}__${TRIGGER}__${HEADTAIL}" ;;
        truncation)   name="${name}__${HEADTAIL}" ;;
    esac
    echo "${OUT_ROOT}/${name}"
}

export MSWEA_COST_TRACKING='ignore_errors'
export TINKER_API_KEY=${TINKER_API_KEY:-tml-dummy}

# The FSDP tinker backend hosts ONE LoRA adapter per worker group; an adapter left
# loaded by a prior run makes this run's create_model collide ("register_adapter is
# not implemented: multi-tenant LoRA"). Free the slot before each tinker-summarizer
# arm (HTTP, best-effort). Skipped for the vLLM path (SUMMARIZER_API_BASE set) and
# non-summarizing arms (full/truncation), which never load the adapter.
UNLOAD_TIMEOUT=${UNLOAD_TIMEOUT:-30}
UNLOAD_SH="server/scripts/unload.sh"
[ -f "${UNLOAD_SH}" ] || UNLOAD_SH="$(dirname "$0")/../../server/scripts/unload.sh"
maybe_unload_adapter() {
    local mode="$1"
    [ -n "${SUMMARIZER_API_BASE}" ] && return 0          # vLLM path: no tinker slot
    case "${mode}" in base|trained) ;; *) return 0 ;; esac  # full/truncation/mask: compact without generating
    local hp=${TINKER_BASE_URL#http://}; hp=${hp#https://}
    local host=${hp%%:*} port=${hp##*:}; port=${port%%/*}
    echo "[slot] freeing FSDP adapter before ${mode} arm ..."
    HOST=${host:-localhost} PORT=${port:-9123} TIMEOUT=${UNLOAD_TIMEOUT} \
        bash "${UNLOAD_SH}" || true
}

for MODE in ${MODES}; do
    OUTPUT="$(arm_dir "${MODE}")"
    echo "=================================================================="
    echo "ARM: ${MODE}  ->  ${OUTPUT}"
    echo "=================================================================="
    maybe_unload_adapter "${MODE}"
    CKPT_ARG=""
    [ "${MODE}" = "trained" ] && CKPT_ARG="--checkpoint ${CHECKPOINT}"
    # `full` and `truncation` never touch the summarizer / tinker server.
    APIBASE_ARG=""
    [ -n "${SUMMARIZER_API_BASE}" ] && APIBASE_ARG="--summarizer-api-base ${SUMMARIZER_API_BASE}"

    uv run -m tts.eval_swebench \
        --dataset "${DATASET}" \
        --data-source "${DATA_SOURCE}" \
        --slice "${SLICE}" \
        --output "${OUTPUT}${THINK_SUFFIX}" \
        --workers "${WORKERS}" \
        --model "${MODEL}" \
        --mode "${MODE}" \
        ${GRADE_ARG} \
        ${REDO_ARG} \
        ${CKPT_ARG} \
        --summarizer-model "${SUMMARIZER_MODEL}" \
        --summarizer-tokenizer "${SUMMARIZER_TOKENIZER}" \
        --summarizer-renderer "${SUMMARIZER_RENDERER}" \
        --summarizer-max-tokens "${SUMMARIZER_MAX_TOKENS}" \
        --tinker-base-url "${TINKER_BASE_URL}" \
        ${APIBASE_ARG} \
        --compress-at-tokens "${COMPRESS_AT_TOKENS}" \
        --compress-at-turns "${COMPRESS_AT_TURNS}" \
        --keep-first "${KEEP_FIRST}" \
        --keep-last-turns "${KEEP_LAST_TURNS}" \
        -c swebench.yaml \
        -c model.model_kwargs.api_base=${DELIBERATOR_API_BASE} \
        -c model.model_kwargs.temperature=${TEMP} \
        -c model.model_kwargs.top_p=${TOP_P} \
        -c model.model_kwargs.top_k=${TOP_K} \
        -c model.model_kwargs.min_p=${MIN_P} \
        -c model.model_kwargs.presence_penalty=${PRESENCE_PENALTY} \
        -c model.model_kwargs.repetition_penalty=${REPETITION_PENALTY} \
        -c "model.model_kwargs.extra_body.drop_params=true" \
        -c "model.model_kwargs.extra_body=${THINKING_EXTRA_BODY}"
done

echo "=================================================================="
echo "SUMMARY"
for MODE in ${MODES}; do
    F="$(arm_dir "${MODE}")/results_summary.json"
    [ -f "${F}" ] && echo "${MODE}: $(cat "${F}")"
done
