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
# MODEL is the HF repo you're evaluating (org/name), e.g. MODEL=zai-org/GLM-4.7-Flash.
# It drives the agent model, the sampling config, and JOB_NAME. litellm needs an
# `openai/<name>` provider prefix to route at the OpenAI-compatible deliberator
# (OPENAI_BASE_URL, posting `<name>`); that prefix is added internally as
# LITELLM_MODEL -- you do NOT put it in MODEL. A leading openai/ is tolerated
# (stripped) for backward compat, EXCEPT the openai-org repo openai/gpt-oss-120b
# (its org really is "openai", so it's kept). When driven by harbor_eval.sbatch,
# MODEL already arrives as the bare repo.
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B}"
case "${MODEL}" in openai/*/*) MODEL="${MODEL#openai/}" ;; esac
LITELLM_MODEL="openai/${MODEL}"
# PUBLIC OpenAI-compatible base URL for the deliberator (must end in /v1 for vLLM).
DELIBERATOR_BASE_URL="${DELIBERATOR_BASE_URL:-}"
# vLLM ignores the key's value but mini-swe-agent requires a non-empty key.
DELIBERATOR_API_KEY="${DELIBERATOR_API_KEY:-EMPTY}"
# Per-model sampling config lives in harbor/configs/<MODEL>.yaml (MODEL is the bare
# repo). So MODEL=zai-org/GLM-4.7-Flash auto-selects
# harbor/configs/zai-org/GLM-4.7-Flash.yaml. Add a new model by dropping a yaml at that
# path (see the existing ones for the sampling + required serving flags). Override
# MINI_CONFIG to force a specific file (e.g. the legacy harbor/mini_qwen.yaml).
MINI_CONFIG="${MINI_CONFIG:-harbor/configs/${MODEL}.yaml}"

if [ -z "${DELIBERATOR_BASE_URL}" ]; then
    echo "ERROR: set DELIBERATOR_BASE_URL to a PUBLIC OpenAI-compatible endpoint" >&2
    echo "       (the Modal sandbox calls it; a localhost URL is unreachable)." >&2
    exit 2
fi

if [ ! -f "${MINI_CONFIG}" ]; then
    echo "ERROR: sampling config not found: ${MINI_CONFIG}" >&2
    echo "       (auto-derived from MODEL=${MODEL}). Add it under harbor/configs/," >&2
    echo "       or set MINI_CONFIG explicitly. Available configs:" >&2
    find harbor/configs -name '*.yaml' 2>/dev/null | sort | sed 's/^/         /' >&2
    exit 2
fi

# Per-benchmark agent step cap (max turns). mini's `-c mini` scaffold defaults to
# step_limit: 0 (UNLIMITED) -- so without this a weak model thrashes hundreds of turns
# into the wall-clock timeout. We cap PER BENCHMARK here (verified -> 250, the standard
# SWE-bench cap; pro/senior left uncapped). It's layered onto the per-model sampling
# config as agent.step_limit: mini merges its `-c` files with recursive_merge (deep),
# and model.* / agent.* are disjoint, so appending an agent block overrides only
# step_limit and keeps mini's system_template etc. Override with STEP_LIMIT= (empty
# disables). Only affects a FRESH run -- resume reads the step_limit already in the
# stored trial config.
case "${BENCH}" in
    verified) STEP_LIMIT="${STEP_LIMIT:-250}" ;;
    *)        STEP_LIMIT="${STEP_LIMIT:-}" ;;
esac
if [ -n "${STEP_LIMIT}" ]; then
    _GEN_DIR="harbor/configs/.generated"
    mkdir -p "${_GEN_DIR}"
    _EFF_CONFIG="${_GEN_DIR}/$(printf '%s' "${BENCH}-${MODEL}" | tr '/ ' '__').yaml"
    { cat "${MINI_CONFIG}"; printf '\n# step cap injected by harbor_eval.sh for BENCH=%s\nagent:\n  step_limit: %s\n' "${BENCH}" "${STEP_LIMIT}"; } > "${_EFF_CONFIG}"
    echo "[config] BENCH=${BENCH}: capping agent.step_limit=${STEP_LIMIT} -> ${_EFF_CONFIG}"
    MINI_CONFIG="${_EFF_CONFIG}"
fi

# Host to add to the sandbox egress allowlist (strip scheme, path, and port).
ALLOW_HOST="${ALLOW_HOST:-$(printf '%s' "${DELIBERATOR_BASE_URL}" \
    | sed -E 's#^[a-zA-Z]+://##; s#/.*$##; s#:[0-9]+$##')}"

# --- Run knobs ------------------------------------------------------------------
ENVIRONMENT="${ENV:-modal}"        # docker|modal|daytona|... (default modal)
N_CONCURRENT="${N_CONCURRENT:-4}"  # parallel trials
N_TASKS="${N_TASKS:-}"             # limit; empty = whole dataset
JOBS_DIR="${JOBS_DIR:-jobs}"
# Deterministic default JOB_NAME: <bench>-<model repo, / -> -->-run-<RUN> -- e.g.
# verified-zai-org--GLM-4.7-Flash-run-0. RUN defaults to 0 (bump for repeats). Must be
# deterministic so a re-run of the same data+model RESUMES (harbor keys resume off the
# job dir = JOBS_DIR/JOB_NAME); the old timestamped default never resumed. When driven
# by harbor_eval.sbatch this is already exported (same formula); set JOB_NAME to override.
RUN="${RUN:-0}"
JOB_NAME="${JOB_NAME:-${BENCH}-${MODEL//\//--}-run-${RUN}}"
# Task-name filters (globs). NAMESPACED on purpose: bare `INCLUDE`/`EXCLUDE`
# collide with env vars the OpenHPC/Lmod module stack exports on hpcfund
# (INCLUDE=/opt/ohpc/.../gcc/.../include), which would silently become a `-i`
# filter matching zero tasks. Accept the old bare names only as a fallback.
HARBOR_INCLUDE="${HARBOR_INCLUDE:-}"   # -i glob (task-name filter), optional
HARBOR_EXCLUDE="${HARBOR_EXCLUDE:-}"   # -x glob, optional
AGENT_SETUP_TIMEOUT_MULTIPLIER="${AGENT_SETUP_TIMEOUT_MULTIPLIER:-}"  # scales 360s default, optional

# --- Senior judge (verifier) ----------------------------------------------------
# Senior grades with an LLM judge inside the verifier phase. Point it at a strong
# model and supply that provider's key. SSB_OVERRIDE_ALL_JUDGE_MODEL overrides
# every judge/classifier stage at once. The verifier runs on a `public` network,
# so it reaches Anthropic/OpenAI/Portkey directly.
JUDGE_MODEL="${JUDGE_MODEL:-}"

# Resolve the Harbor CLI. Explicit HARBOR_BIN wins; else a `harbor` on PATH
# (standalone `uv tool install harbor[modal]`, as on babel); else the project
# venv's entrypoint or `uv run harbor` (Harbor is a core pyproject dependency now,
# so a plain `uv sync` installs .venv/bin/harbor).
if [ -n "${HARBOR_BIN:-}" ]; then
    HARBOR_CMD=( "${HARBOR_BIN}" )
elif command -v harbor >/dev/null 2>&1; then
    HARBOR_CMD=( harbor )
elif [ -x ".venv/bin/harbor" ]; then
    HARBOR_CMD=( .venv/bin/harbor )
elif command -v uv >/dev/null 2>&1; then
    HARBOR_CMD=( uv run harbor )
else
    echo "ERROR: harbor not found. Install: uv sync   (harbor is a core dep; needs Python >=3.12)" >&2
    echo "       (or standalone: uv tool install --python 3.12 'harbor[modal] @ git+https://github.com/harbor-framework/harbor.git@main')" >&2
    exit 127
fi

# --- Assemble the command -------------------------------------------------------
ARGS=(
    run
    -d "${DATASET}"
    -a mini-swe-agent
    -m "${LITELLM_MODEL}"
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
[ -n "${HARBOR_INCLUDE}" ] && ARGS+=( -i "${HARBOR_INCLUDE}" )
[ -n "${HARBOR_EXCLUDE}" ] && ARGS+=( -x "${HARBOR_EXCLUDE}" )
# Agent-setup timeout MULTIPLIER (scales harbor's 360s default; e.g. 5 -> 1800s,
# 10 -> 3600s). The in-sandbox toolchain install (apt build-essential + `uv tool
# install mini-swe-agent litellm`) blows past 360s under high N_CONCURRENT (mirror
# contention), so every trial dies with AgentSetupTimeoutError before the agent
# runs. Raise it for scale, and set it generously when benchmarking prep so slow
# setups are measured, not censored. NB: use the *multiplier* flag, not the absolute
# --agent-setup-timeout -- the latter exists only on newer harbor (hpcfund) and errors
# on the older CLI (babel); the multiplier is present in both.
[ -n "${AGENT_SETUP_TIMEOUT_MULTIPLIER}" ] && ARGS+=( --agent-setup-timeout-multiplier "${AGENT_SETUP_TIMEOUT_MULTIPLIER}" )

# Agent-EXECUTION timeout MULTIPLIER (scales the per-trial agent wall-clock, which is
# the task's timeout_sec -- 3000s for swe-bench-verified). e.g. 2.4 -> 7200s, 2 -> 6000s,
# 0.5 -> 1500s (fail faster). This is the "3000s wall" that produces AgentTimeoutError;
# distinct from the setup-timeout and from mini's step_limit. NB: it is baked into the
# JobConfig + lock, so changing it forces a FRESH run (a resume with a different value
# hits the same lock-mismatch as n_concurrent).
AGENT_TIMEOUT_MULTIPLIER="${AGENT_TIMEOUT_MULTIPLIER:-}"
[ -n "${AGENT_TIMEOUT_MULTIPLIER}" ] && ARGS+=( --agent-timeout-multiplier "${AGENT_TIMEOUT_MULTIPLIER}" )

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

# --- Make the deliberator token resume-safe -------------------------------------
# Harbor persists the agent env (the --ae values above) into every trial's
# config.json and reconciles it on resume. For a sensitive key (…KEY/…TOKEN/…),
# harbor's templatize_sensitive_env (harbor/utils/env.py) stores a ${NAME} TEMPLATE
# *only when the same value is present in harbor's own os.environ under that name*;
# otherwise it redacts the literal (-> `440f****962`, and _scrub_jobs_dir rewrites
# any surviving literal to `[REDACTED]`). A redacted value cannot be recovered on
# resume, which breaks BOTH: (a) reconciliation throws "Existing trial config does
# not match planned job config", and (b) any trial that does run authenticates with
# the dead string -> `401 bad relay token` -> NonZeroAgentExitCodeError.
# So export the token under the exact --ae names: harbor then writes
# ${OPENAI_API_KEY}/${MSWEA_API_KEY} templates that reconcile across chunks and
# re-resolve from this (deterministic) env every resume. Skip `senior`, which
# overloads OPENAI_API_KEY for the judge (forwarded to the verifier via --ve above).
if [ "${BENCH}" != "senior" ]; then
    export OPENAI_API_KEY="${DELIBERATOR_API_KEY}"
    export MSWEA_API_KEY="${DELIBERATOR_API_KEY}"
fi

# --- Resume vs fresh run --------------------------------------------------------
# If this job dir already has a config.json, the job was started before (e.g. a
# previous 4h allocation) -> RESUME it with `harbor job resume` instead of `harbor
# run`. Resume reads the stored config (dataset, model, agent env), so the SAME
# command works every chunk: first submit creates the job, later submits continue
# it. `harbor job resume` re-runs trials whose error type is in --filter-error-type
# (default CancelledError, which is what interrupted trials become when the batch
# script kills harbor at the 4h boundary), so boundary-truncated trials get retried
# rather than kept as failures (which a plain `harbor run` resume would do).
#   RESUME=auto (default) : resume iff <job_dir>/config.json exists; else fresh run.
#   RESUME=1              : force resume (errors if the job dir has no config).
#   RESUME=0              : force fresh run (Harbor still refuses a different config).
#   RESUME_FILTER_ERRORS  : space-separated error types to also retry, e.g.
#                           "CancelledError AgentTimeoutError NonZeroAgentExitCodeError".
#                           Passing any REPLACES harbor's default, so include
#                           CancelledError explicitly if you set this.
JOB_DIR="${JOBS_DIR}/${JOB_NAME}"
RESUME="${RESUME:-auto}"
RESUME_FILTER_ERRORS="${RESUME_FILTER_ERRORS:-}"
if [ "${RESUME}" = "1" ] || { [ "${RESUME}" = "auto" ] && [ -f "${JOB_DIR}/config.json" ]; }; then
    # `harbor job resume` has NO -n flag: it runs at the concurrency stored in the job
    # dir. That value lives in n_concurrent_trials, which the scheduler reads from
    # config.json (job.py) AND harbor lock-checks against lock.json (build_job_lock ==
    # stored lock; n_concurrent_trials is part of JobLock equality). So to honor
    # N_CONCURRENT on resume we patch BOTH files in place before resuming -- editing
    # config.json alone trips "lock.json does not match the resolved job lock", and
    # not editing either leaves resume stuck at the original value (default 4). Pass
    # the SAME N_CONCURRENT each chunk (like RESUME_FILTER_ERRORS); it's authoritative.
    if [ -n "${N_CONCURRENT}" ]; then
        python3 - "${JOB_DIR}" "${N_CONCURRENT}" <<'PY'
import json, sys, pathlib
job_dir, n = pathlib.Path(sys.argv[1]), int(sys.argv[2])
changed = []
for name in ("config.json", "lock.json"):
    p = job_dir / name
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    if d.get("n_concurrent_trials") != n:
        d["n_concurrent_trials"] = n
        p.write_text(json.dumps(d, indent=4) + "\n")
        changed.append(name)
if changed:
    print(f"[resume] set n_concurrent_trials={n} in {', '.join(changed)}")
PY
    fi
    RESUME_ARGS=( job resume -p "${JOB_DIR}" )
    for _et in ${RESUME_FILTER_ERRORS}; do RESUME_ARGS+=( -f "${_et}" ); done
    echo "=================================================================="
    echo "RESUME  job=${JOB_NAME}  dir=${JOB_DIR}  n=${N_CONCURRENT}"
    echo "deliberator=${DELIBERATOR_BASE_URL} (from stored config; must be live)"
    echo "retry-error-types=${RESUME_FILTER_ERRORS:-<harbor default: CancelledError>}"
    echo "=================================================================="
    exec "${HARBOR_CMD[@]}" "${RESUME_ARGS[@]}"
fi

echo "=================================================================="
echo "BENCH=${BENCH}  DATASET=${DATASET}"
echo "MODEL=${MODEL} (litellm: ${LITELLM_MODEL})  ENV=${ENVIRONMENT}  n=${N_CONCURRENT}  tasks=${N_TASKS:-all}"
echo "deliberator=${DELIBERATOR_BASE_URL}  allow-host=${ALLOW_HOST}"
[ "${BENCH}" = "senior" ] && echo "judge=${JUDGE_MODEL:-<task default>}"
echo "job=${JOB_NAME} -> ${JOBS_DIR}/"
echo "=================================================================="

exec "${HARBOR_CMD[@]}" "${ARGS[@]}"
