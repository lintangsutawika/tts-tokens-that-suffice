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
#   # Senior with a SELF-HOSTED litellm_proxy judge. JUDGE_BASE_URL + JUDGE_API_KEY
#   # point at an OpenAI-compatible proxy and are forwarded to the verifier as
#   # LITELLM_PROXY_API_BASE / LITELLM_PROXY_API_KEY; JUDGE_MODEL is the litellm_proxy/<name>.
#   DELIBERATOR_BASE_URL=https://<tunnel>/v1 \
#       JUDGE_MODEL=litellm_proxy/Qwen/Qwen3.6-35B-A3B \
#       JUDGE_BASE_URL=https://cmu.litellm.ai/v1 JUDGE_API_KEY=sk-... \
#       scripts/eval/harbor_eval.sh senior
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
# It drives the agent model, the sampling config, and JOB_NAME. litellm needs a provider
# prefix to route at the self-hosted deliberator; that prefix is added internally as
# LITELLM_MODEL -- you do NOT put it in MODEL. A leading openai/ is tolerated (stripped)
# for backward compat, EXCEPT the openai-org repo openai/gpt-oss-120b (its org really is
# "openai", so it's kept). When driven by harbor_eval.sbatch, MODEL arrives as the bare repo.
#
# LITELLM_PROVIDER (default litellm_proxy): the litellm provider prefix.
#   litellm_proxy = PASSTHROUGH to an OpenAI-compatible server (our vLLM), forwarding the
#     request as-is. Preferred: openai/ applies OpenAI-specific transforms -- notably
#     routing reasoning+tools to /v1/responses, which vLLM does NOT implement -> failures.
#   openai = the legacy behavior; keep it if you specifically want openai/ routing.
# The two read DIFFERENT endpoint envs: litellm_proxy -> LITELLM_PROXY_API_BASE/_KEY;
# openai -> OPENAI_BASE_URL/OPENAI_API_KEY. Both sets are injected below so either works.
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B}"
case "${MODEL}" in openai/*/*) MODEL="${MODEL#openai/}" ;; esac
LITELLM_PROVIDER="${LITELLM_PROVIDER:-litellm_proxy}"
LITELLM_MODEL="${LITELLM_PROVIDER}/${MODEL}"
# PUBLIC OpenAI-compatible base URL for the deliberator (must end in /v1 for vLLM).
DELIBERATOR_BASE_URL="${DELIBERATOR_BASE_URL:-}"
# vLLM ignores the key's value but mini-swe-agent requires a non-empty key.
DELIBERATOR_API_KEY="${DELIBERATOR_API_KEY:-EMPTY}"
# Per-model sampling config lives in configs/sampling/<MODEL>.yaml (MODEL is the bare
# repo). So MODEL=zai-org/GLM-4.7-Flash auto-selects
# configs/sampling/zai-org/GLM-4.7-Flash.yaml. Add a new model by dropping a yaml at that
# path (see the existing ones for the sampling + required serving flags). Override
# MINI_CONFIG to force a specific file (e.g. the legacy harbor/mini_qwen.yaml).
MINI_CONFIG="${MINI_CONFIG:-configs/sampling/${MODEL}.yaml}"

if [ -z "${DELIBERATOR_BASE_URL}" ]; then
    echo "ERROR: set DELIBERATOR_BASE_URL to a PUBLIC OpenAI-compatible endpoint" >&2
    echo "       (the Modal sandbox calls it; a localhost URL is unreachable)." >&2
    exit 2
fi

if [ ! -f "${MINI_CONFIG}" ]; then
    echo "ERROR: sampling config not found: ${MINI_CONFIG}" >&2
    echo "       (auto-derived from MODEL=${MODEL}). Add it under configs/sampling/," >&2
    echo "       or set MINI_CONFIG explicitly. Available configs:" >&2
    find configs/sampling -name '*.yaml' 2>/dev/null | sort | sed 's/^/         /' >&2
    exit 2
fi

# --- summarization (USE_SUMMARIZER) --------------------------------------------
# Route the deliberator's own context through tts.agent.summarization_agent.SummarizingAgent:
# once the agent's context passes SUMMARIZER_TRIGGER_TOKENS, compact the middle
# (mask|truncation|model; model reuses the agent's endpoint). In-container the agent
# imports tts from a BIND-MOUNT of this repo's src/ (no install of tts's heavy vllm/tinker
# deps) + PYTHONPATH -- so it is ENV=singularity only. Off by default.
_is_true() { case "${1:-}" in true|True|TRUE|1|yes|on) return 0 ;; *) return 1 ;; esac; }
USE_SUMMARIZER="${USE_SUMMARIZER:-false}"
SUMMARIZER_COMPACTOR="${SUMMARIZER_COMPACTOR:-mask}"            # mask | truncation | model | none
SUMMARIZER_STYLE="${SUMMARIZER_STYLE:-sectioned}"              # model: sectioned|unconstrained; subtask: subtask
SUMMARIZER_TRIGGER_TOKENS="${SUMMARIZER_TRIGGER_TOKENS:-64000}"
SUMMARIZER_KEEP_FIRST="${SUMMARIZER_KEEP_FIRST:-4}"
SUMMARIZER_KEEP_LAST_TURNS="${SUMMARIZER_KEEP_LAST_TURNS:-3}"
SUMMARIZER_MODEL="${SUMMARIZER_MODEL:-}"                        # empty => reuse the agent's own model
# Host path bound into the task container (this repo's src/) and the mountpoint it appears
# at (also the agent's PYTHONPATH). src/ lives on shared FS -> valid on every node, so it
# bakes cleanly into the resumed job config.
TTS_SRC="${TTS_SRC:-$(cd "$(dirname "$0")/../.." && pwd)/src}"
TTS_SRC_MOUNT="${TTS_SRC_MOUNT:-/opt/tts-src}"
if _is_true "${USE_SUMMARIZER}"; then
    case "${ENV:-modal}" in
        singularity*)
            [ -d "${TTS_SRC}/tts" ] || echo "WARNING: TTS_SRC=${TTS_SRC} has no tts/ package; the in-container import will fail. Point TTS_SRC at this repo's src/." >&2 ;;
        *) echo "WARNING: USE_SUMMARIZER is only wired for ENV=singularity (host bind-mount + PYTHONPATH); disabling for ENV=${ENV:-modal}." >&2; USE_SUMMARIZER=false ;;
    esac
fi
# Name tag so summarizer runs get their OWN job dir + wandb run (never colliding with the
# baseline, whose agent config differs -> otherwise same dir => conflated results + lock
# mismatch). Baseline (off) keeps the untagged name, so existing dirs are unaffected.
_SUM_TAG=""
if _is_true "${USE_SUMMARIZER}"; then
    _trig="${SUMMARIZER_TRIGGER_TOKENS}"; case "${_trig}" in *000) _trig="$((_trig/1000))k" ;; esac
    _cmp="${SUMMARIZER_COMPACTOR}"
    # For model/subtask, the summary STYLE is part of the identity (own dir/run);
    # mask/truncation have no style.
    if [ "${_cmp}" = "model" ]; then _cmp="model-${SUMMARIZER_STYLE}"
    elif [ "${_cmp}" = "subtask" ]; then _cmp="subtask"; fi
    _SUM_TAG="-sum-${_cmp}-${_trig}"
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
if [ -n "${STEP_LIMIT}" ] || _is_true "${USE_SUMMARIZER}"; then
    _GEN_DIR="configs/sampling/.generated"
    mkdir -p "${_GEN_DIR}"
    _EFF_CONFIG="${_GEN_DIR}/$(printf '%s' "${BENCH}-${MODEL}${_SUM_TAG}" | tr '/ ' '__').yaml"
    # ONE appended agent: block carries the step cap AND (when enabled) the summarizer
    # agent_class + compaction knobs. mini deep-merges -c files; model.* / agent.* are
    # disjoint, so this overrides only these keys and keeps mini's system_template etc.
    {
        cat "${MINI_CONFIG}"
        printf '\n# injected by harbor_eval.sh (BENCH=%s)\n' "${BENCH}"
        # Cap per-turn generation tokens for all models (bounded gen, prevents
        # unbounded output / context blow-up). Set MAX_TOKENS (e.g. 32768); empty
        # leaves the model/litellm default (unbounded). Merge-safe (mini deep-merges
        # model.* across -c files).
        if [ -n "${MAX_TOKENS:-}" ]; then
            printf 'model:\n  model_kwargs:\n    max_tokens: %s\n' "${MAX_TOKENS}"
        fi
        printf 'agent:\n'
        [ -n "${STEP_LIMIT}" ] && printf '  step_limit: %s\n' "${STEP_LIMIT}"
        if _is_true "${USE_SUMMARIZER}"; then
            printf '  agent_class: tts.agent.summarization_agent.SummarizingAgent\n'
            printf '  compactor: %s\n' "${SUMMARIZER_COMPACTOR}"
            [ "${SUMMARIZER_COMPACTOR}" = "model" ] && printf '  summarizer_style: %s\n' "${SUMMARIZER_STYLE}"
            [ "${SUMMARIZER_COMPACTOR}" = "subtask" ] && printf '  summarizer_style: subtask\n'
            printf '  compress_at_tokens: %s\n' "${SUMMARIZER_TRIGGER_TOKENS}"
            printf '  keep_first: %s\n' "${SUMMARIZER_KEEP_FIRST}"
            printf '  keep_last_turns: %s\n' "${SUMMARIZER_KEEP_LAST_TURNS}"
            [ -n "${SUMMARIZER_MODEL}" ] && printf '  summarizer_model: %s\n' "${SUMMARIZER_MODEL}"
            printf '  compressions_dir: /logs/agent\n'  # -> jobs/<run>/<trial>/agent/count.json (harbor syncs /logs/agent -> trial agent dir)
        fi
    } > "${_EFF_CONFIG}"
    echo "[config] BENCH=${BENCH}: step_limit=${STEP_LIMIT:-<none>}$(_is_true "${USE_SUMMARIZER}" && printf ' summarizer=%s@%stok' "${SUMMARIZER_COMPACTOR}" "${SUMMARIZER_TRIGGER_TOKENS}") -> ${_EFF_CONFIG}"
    MINI_CONFIG="${_EFF_CONFIG}"
fi

# Host to add to the sandbox egress allowlist (strip scheme, path, and port).
ALLOW_HOST="${ALLOW_HOST:-$(printf '%s' "${DELIBERATOR_BASE_URL}" \
    | sed -E 's#^[a-zA-Z]+://##; s#/.*$##; s#:[0-9]+$##')}"

# --- Run knobs ------------------------------------------------------------------
ENVIRONMENT="${ENV:-modal}"        # docker|modal|daytona|... (default modal)
# Modal App name for the sandboxes (harbor's default is "__harbor__"; passed via
# --ek app_name=... for the modal env only). All this run's Modal sandboxes group
# under it -- rename to keep them identifiable in the Modal dashboard.
MODAL_APP_NAME="${MODAL_APP_NAME:-harbor_sandbox}"
# Modal sandbox LIFETIME cap (harbor's default is 24h!) and IDLE-kill (harbor's default
# is None -- never). A trial whose sandbox isn't torn down cleanly, or that hangs, would
# otherwise bill for HOURS -- the main runaway-cost risk. Cap lifetime at 2h and kill
# idle sandboxes after 15 min. Both are modal env-kwargs (--ek). Set empty to fall back
# to harbor's defaults. Keep the lifetime >= agent+setup+verifier so it doesn't cut a
# live trial short; keep idle > the longest gap between agent actions (LLM waits) so a
# slow-but-active trial isn't killed.
MODAL_SANDBOX_TIMEOUT_SEC="${MODAL_SANDBOX_TIMEOUT_SEC:-7200}"         # 2h max lifetime
MODAL_SANDBOX_IDLE_TIMEOUT_SEC="${MODAL_SANDBOX_IDLE_TIMEOUT_SEC:-3600}"  # 1h idle kill
# Singularity/Apptainer env (ENV=singularity): run the SWE-bench task container ON THIS
# node instead of in Modal -- no relay needed (the on-node container shares the host net
# namespace, so it reaches local vLLM directly). SINGULARITY_CACHE_DIR is the dir for
# converted .sif images.
#   * LEAVE IT UNSET (default) for node-local: it is NOT passed to harbor, and the
#     harbor-singularity-hpc backend resolves $PBS_LOCALDIR/$SLURM_TMPDIR LIVE per process. Nothing
#     absolute is baked into the job config, so chunked RESUME on a new node works
#     (each chunk uses its own node-local dir). Cost: wiped per job => re-pull per chunk.
#   * SET a LITERAL shared-FS path (e.g. /home/aci18914wh/sif_cache) to PERSIST across
#     chunks and avoid re-pulls. It is then passed via --ek and baked into config.json --
#     which is fine ONLY because a shared path stays valid on every node. NEVER set it to
#     a $PBS_LOCALDIR-derived path: that path is empty at submit time and, if it did
#     resolve, would be a per-job node-local path that goes stale on the next resume.
# This is the big image cache (~tens of GB for verified), NOT apptainer's own
# SINGULARITY_CACHEDIR layer cache -- keep them separate.
SINGULARITY_CACHE_DIR="${SINGULARITY_CACHE_DIR:-}"
SINGULARITY_FORCE_PULL="${SINGULARITY_FORCE_PULL:-}"   # re-pull even if cached (default off)
# Mounts to suppress. Harbor's own default is "home,tmp,bind-paths", but on ABCI the
# `bind-paths` group carries /etc/resolv.conf -- dropping it leaves the container with
# NO DNS, so the in-container bootstrap pip can't reach PyPI and the harbor server dies
# ("ModuleNotFoundError: No module named 'uvicorn'"). Keep bind-paths (=> resolv.conf);
# only suppress home,tmp. Set to "" is not passable via --ek (use a real value).
SINGULARITY_NO_MOUNT="${SINGULARITY_NO_MOUNT:-home,tmp}"
# Writable-sandbox mode: on ABCI-class nodes that can't FUSE-mount SIFs under --fakeroot
# and lack overlayfs, extract to a per-session sandbox dir + `--fakeroot --writable`.
# The harbor-singularity-hpc class defaults this ON, so LEAVE UNSET to use it. Set
# "false" to force upstream --writable-tmpfs (e.g. a node where FUSE is configured).
SINGULARITY_WRITABLE_SANDBOX="${SINGULARITY_WRITABLE_SANDBOX:-}"   # empty=on (class default); false to disable
# Where per-session sandboxes are built. Empty => the class picks node-local scratch
# ($PBS_LOCALDIR / $SLURM_TMPDIR / $TMPDIR). Keep it OFF the shared FS.
SINGULARITY_SANDBOX_DIR="${SINGULARITY_SANDBOX_DIR:-}"
# The writable-rootfs environment class, selected via --environment-import-path (harbor's
# -e is a fixed enum and can't name a custom class). Override only to test a fork.
SINGULARITY_ENV_IMPORT_PATH="${SINGULARITY_ENV_IMPORT_PATH:-harbor_singularity_hpc.environment:SingularityWritableEnvironment}"
# Task-container memory. Harbor runs a soft watchdog on each task container (PSS of its
# process tree) and kills at 95% of the task's [environment].memory (8G for these SWE-bench
# datasets) -> MemoryLimitExceededError. Singularity uses no cgroup, so it's a monitor, not
# a hard cap: raise it for heavy repos (compilation / big test suites), or disable it.
#   * SINGULARITY_MEMORY_MB=24576  -> raise the per-task cap to 24 GB (keeps a guardrail).
#   * SINGULARITY_MEMORY_ENFORCEMENT=ignore -> turn the watchdog OFF (node RAM / PBS
#     allocation is then the only ceiling; size N_CONCURRENT so N x per-task peak fits RAM).
SINGULARITY_MEMORY_MB="${SINGULARITY_MEMORY_MB:-}"
SINGULARITY_MEMORY_ENFORCEMENT="${SINGULARITY_MEMORY_ENFORCEMENT:-}"
N_CONCURRENT="${N_CONCURRENT:-4}"  # parallel trials
N_TASKS="${N_TASKS:-}"             # limit; empty = whole dataset
JOBS_DIR="${JOBS_DIR:-jobs}"
# Deterministic default JOB_NAME: <bench>-<model repo, / -> -->-run-<RUN> -- e.g.
# verified-zai-org--GLM-4.7-Flash-run-0. RUN defaults to 0 (bump for repeats). Must be
# deterministic so a re-run of the same data+model RESUMES (harbor keys resume off the
# job dir = JOBS_DIR/JOB_NAME); the old timestamped default never resumed. When driven
# by harbor_eval.sbatch this is already exported (same formula); set JOB_NAME to override.
RUN="${RUN:-0}"
JOB_NAME="${JOB_NAME:-${BENCH}-${MODEL//\//--}${_SUM_TAG}-run-${RUN}}"
# Task-name filters (globs). NAMESPACED on purpose: bare `INCLUDE`/`EXCLUDE`
# collide with env vars the OpenHPC/Lmod module stack exports on hpcfund
# (INCLUDE=/opt/ohpc/.../gcc/.../include), which would silently become a `-i`
# filter matching zero tasks. Accept the old bare names only as a fallback.
HARBOR_INCLUDE="${HARBOR_INCLUDE:-}"   # -i glob (task-name filter), optional
HARBOR_EXCLUDE="${HARBOR_EXCLUDE:-}"   # -x glob, optional
# Agent-SETUP timeout, ABSOLUTE seconds (agent.override_setup_timeout_sec) -- the ONLY
# setup-timeout knob, symmetric with AGENT_TIMEOUT_SEC. `harbor run` exposes no absolute
# timeout flag (only multipliers), but its JobConfig DOES carry override_setup_timeout_sec,
# so both fresh and resume set it by patching the config: fresh via --print-config ->
# patch -> `run -c` (see the fresh-run block below), resume via the in-place patch. No
# multiplier. Empty -> harbor's 360s default.
AGENT_SETUP_TIMEOUT_SEC="${AGENT_SETUP_TIMEOUT_SEC:-}"

# --- Senior judge (verifier) ----------------------------------------------------
# Senior grades with an LLM judge inside the verifier phase. Point it at a strong
# model and supply that provider's key. SSB_OVERRIDE_ALL_JUDGE_MODEL overrides
# every judge/classifier stage at once. The verifier runs on a `public` network,
# so it reaches Anthropic/OpenAI/Portkey directly.
#
# For a SELF-HOSTED litellm_proxy judge, set JUDGE_BASE_URL (the OpenAI-compatible
# endpoint, ending /v1) and JUDGE_API_KEY; both are forwarded to the VERIFIER env as
# LITELLM_PROXY_API_BASE / LITELLM_PROXY_API_KEY (--ve only, not the agent env), and
# JUDGE_MODEL should be the litellm_proxy/<name> form. This lets the Senior verifier
# call a litellm_proxy judge the same way the deliberator uses litellm_proxy.
JUDGE_MODEL="${JUDGE_MODEL:-}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-}"
JUDGE_API_KEY="${JUDGE_API_KEY:-}"

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

# --- Singularity: writable-rootfs environment via the harbor-singularity-hpc package -
# The Singularity backend on ABCI-class nodes needs: (1) a Dockerfile FROM fallback
# (SWE-bench tasks set no [environment].docker_image), and (2) a writable rootfs --
# those nodes can't FUSE-mount SIFs under --fakeroot (no user_allow_other) and have no
# overlayfs, so --writable-tmpfs/--overlay give a read-only rootfs. Both (plus pull
# retry + a long exec timeout) now live in the `harbor-singularity-hpc` package, a
# subclass of harbor's SingularityEnvironment selected below via
# --environment-import-path. This replaces the old in-place source patching of harbor
# (patches/harbor_singularity_*.patch + the pristine-reset dance): it is a normal uv
# dependency, so a plain `uv sync` installs it and there is nothing to re-apply per job.
# Only for ENV=singularity; podman/modal build the Dockerfile natively.
case "${ENVIRONMENT}" in
  singularity*)
    # Pre-create the .sif cache (harbor also mkdir's it, but a stale/absent parent
    # -- e.g. a node-local path from a previous job -- otherwise crashes). The
    # sandbox dir defaults to the LIVE $PBS_LOCALDIR when SINGULARITY_SANDBOX_DIR is
    # unset; only mkdir an explicit one. NOTE: keep the cache on a SHARED FS, not
    # $PBS_LOCALDIR -- node-local is wiped per job (re-pull every run) and yields
    # stale-jobid paths.
    if [ -n "${SINGULARITY_CACHE_DIR}" ] && ! mkdir -p "${SINGULARITY_CACHE_DIR}" 2>/dev/null; then
        echo "WARNING: could not create SINGULARITY_CACHE_DIR=${SINGULARITY_CACHE_DIR} (bad path? stale \$PBS_LOCALDIR jobid? put the .sif cache on a shared FS)" >&2
    fi
    [ -n "${SINGULARITY_SANDBOX_DIR}" ] && mkdir -p "${SINGULARITY_SANDBOX_DIR}" 2>/dev/null
    # Verify the environment class is importable in the SAME interpreter harbor uses.
    # If missing, `uv sync` (it's a pinned dependency) usually fixes it.
    _CLS_MOD="${SINGULARITY_ENV_IMPORT_PATH%%:*}"
    _IMPORT_CHECK="import importlib; importlib.import_module('${_CLS_MOD}')"
    if command -v uv >/dev/null 2>&1; then
        uv run --quiet python -c "${_IMPORT_CHECK}" >/dev/null 2>&1 && _OK=1 || _OK=""
    else
        .venv/bin/python -c "${_IMPORT_CHECK}" >/dev/null 2>&1 && _OK=1 || _OK=""
    fi
    if [ -z "${_OK}" ]; then
        echo "WARNING: ENV=singularity but '${_CLS_MOD}' is not importable. Install it: uv sync" >&2
        echo "         (harbor-singularity-hpc is a pinned dependency; see pyproject.toml [tool.uv.sources].)" >&2
    else
        echo "[env] singularity: ${SINGULARITY_ENV_IMPORT_PATH}"
    fi
    ;;
esac

# --- Assemble the command -------------------------------------------------------
ARGS=(
    run
    -d "${DATASET}"
    -a mini-swe-agent
    -m "${LITELLM_MODEL}"
    -n "${N_CONCURRENT}"
    --job-name "${JOB_NAME}"
    -o "${JOBS_DIR}"
    # Deliberator credentials, injected into the agent env inside the sandbox.
    --ae "OPENAI_BASE_URL=${DELIBERATOR_BASE_URL}"
    --ae "OPENAI_API_KEY=${DELIBERATOR_API_KEY}"
    --ae "MSWEA_API_KEY=${DELIBERATOR_API_KEY}"
    # litellm_proxy/ (default LITELLM_PROVIDER) resolves its endpoint from these, NOT
    # OPENAI_BASE_URL. Inject both name-sets so the deliberator is reachable under either
    # provider prefix (same URL + key; harmless when using openai/).
    --ae "LITELLM_PROXY_API_BASE=${DELIBERATOR_BASE_URL}"
    --ae "LITELLM_PROXY_API_KEY=${DELIBERATOR_API_KEY}"
    # Qwen sampling / thinking (see harbor/mini_qwen.yaml).
    --ak "config_file=${MINI_CONFIG}"
    -y
)

# Environment selection. `singularity` is our custom writable-rootfs class from the
# harbor-singularity-hpc package -- harbor's -e is a fixed enum and can't name a custom
# class, so select it by import path (config.json then stores import_path, so resume
# reloads the same class as long as the package is installed on the node). Every other
# backend uses the built-in -e enum.
case "${ENVIRONMENT}" in
    singularity*) ARGS+=( --environment-import-path "${SINGULARITY_ENV_IMPORT_PATH}" ) ;;
    *)            ARGS+=( -e "${ENVIRONMENT}" ) ;;
esac

# mini-swe-agent is installed with an unpinned `uv tool install`, which builds the
# tool venv against the TASK IMAGE's system Python. Many SWE-bench images ship
# Python 3.10 (some 3.9), but current mini-swe-agent needs >=3.11 (imports
# typing.NotRequired) -> "ImportError: cannot import name 'NotRequired'" and the
# agent exits 1 before contacting the model. UV_PYTHON forces uv to use (and, if
# absent, download) a 3.11 interpreter for the agent tool venv, independent of the
# image's system Python. Set MSWEA_UV_PYTHON="" to disable.
MSWEA_UV_PYTHON="${MSWEA_UV_PYTHON:-3.11}"
[ -n "${MSWEA_UV_PYTHON}" ] && ARGS+=( --ae "UV_PYTHON=${MSWEA_UV_PYTHON}" )

# Egress allowlist for the deliberator host. Only meaningful when the task's
# effective network policy is restricted (Modal sandboxes with egress control).
# The SWE-bench tasks set allow_internet=true -> PUBLIC policy, under which harbor
# DISCARDS run-specific allowlist hosts and warns ("... ignored because the
# effective network policy is public"). On a local node (singularity/podman) the
# agent reaches the on-node deliberator directly, so pass it only for Modal.
case "${ENVIRONMENT}" in
    modal*) ARGS+=( --allow-agent-host "${ALLOW_HOST}" ) ;;
esac

[ -n "${N_TASKS}" ] && ARGS+=( -l "${N_TASKS}" )
[ -n "${HARBOR_INCLUDE}" ] && ARGS+=( -i "${HARBOR_INCLUDE}" )
[ -n "${HARBOR_EXCLUDE}" ] && ARGS+=( -x "${HARBOR_EXCLUDE}" )
# Per-environment kwargs (--ek): each backend's constructor takes different kwargs, and
# passing a kwarg the constructor doesn't accept ERRORS -- so gate by environment type.
case "${ENVIRONMENT}" in
    modal*)
        # Modal: app name + sandbox lifetime/idle caps.
        [ -n "${MODAL_APP_NAME}" ] && ARGS+=( --ek "app_name=${MODAL_APP_NAME}" )
        [ -n "${MODAL_SANDBOX_TIMEOUT_SEC}" ] && ARGS+=( --ek "sandbox_timeout_secs=${MODAL_SANDBOX_TIMEOUT_SEC}" )
        [ -n "${MODAL_SANDBOX_IDLE_TIMEOUT_SEC}" ] && ARGS+=( --ek "sandbox_idle_timeout_secs=${MODAL_SANDBOX_IDLE_TIMEOUT_SEC}" )
        ;;
    singularity*)
        # Singularity/Apptainer (on-node): a PERSISTENT .sif cache is essential -- without
        # it harbor pulls each task image into a throwaway temp dir and re-pulls every
        # trial (500x the same base). Point it at a big, shared path. force_pull re-pulls
        # even if cached; no_mount suppresses mount types (default home,tmp,bind-paths).
        [ -n "${SINGULARITY_CACHE_DIR}" ] && ARGS+=( --ek "singularity_image_cache_dir=${SINGULARITY_CACHE_DIR}" )
        [ -n "${SINGULARITY_FORCE_PULL}" ] && ARGS+=( --ek "singularity_force_pull=${SINGULARITY_FORCE_PULL}" )
        [ -n "${SINGULARITY_NO_MOUNT}" ] && ARGS+=( --ek "singularity_no_mount=${SINGULARITY_NO_MOUNT}" )
        [ -n "${SINGULARITY_WRITABLE_SANDBOX}" ] && ARGS+=( --ek "singularity_writable_sandbox=${SINGULARITY_WRITABLE_SANDBOX}" )
        [ -n "${SINGULARITY_SANDBOX_DIR}" ] && ARGS+=( --ek "singularity_sandbox_dir=${SINGULARITY_SANDBOX_DIR}" )
        # Memory watchdog (see the knobs above): raise the per-task cap, or disable it.
        [ -n "${SINGULARITY_MEMORY_MB}" ] && ARGS+=( --ek "override_memory_mb=${SINGULARITY_MEMORY_MB}" )
        [ -n "${SINGULARITY_MEMORY_ENFORCEMENT}" ] && ARGS+=( --ek "memory_enforcement_policy=${SINGULARITY_MEMORY_ENFORCEMENT}" )
        ;;
esac

# Summarization (USE_SUMMARIZER, singularity only): bind this repo's src/ into the task
# container and point the in-container agent's PYTHONPATH at it, so mini-swe imports
# tts.agent.summarization_agent (the agent_class in the generated config) WITHOUT
# installing tts. The writable-sandbox env pre-creates the bind destination.
if _is_true "${USE_SUMMARIZER}"; then
    # The CliffCompactor imports cliffcompaction (a pip dep reachable in the
    # project venv but NOT in the container's mini-swe tool venv). Install it into
    # a target dir inside the bind-mounted src/ so the agent venv can import it,
    # without forking harbor or vendoring the package. Idempotent; shared FS.
    if [ "${SUMMARIZER_COMPACTOR:-mask}" = "cliff" ]; then
        TTS_DEP_TARGET="${TTS_SRC}/.deps"
        if ! [ -d "${TTS_DEP_TARGET}/cliffcompaction" ]; then
            echo "[env] installing cliffcompaction -> ${TTS_DEP_TARGET} (for compactor=cliff in-container)"
            mkdir -p "${TTS_DEP_TARGET}"
            uv pip install --quiet --python "${TTS_SRC}/../.venv/bin/python" \
                --target "${TTS_DEP_TARGET}" "cliffcompaction>=0.1.0" >/dev/null 2>&1 \
              || echo "WARNING: could not install cliffcompaction into ${TTS_DEP_TARGET}; compactor=cliff will fail in-container" >&2
        fi
    fi
    # Point the agent at the bound src AND its .deps (cliffcompaction etc.), so the
    # container's mini-swe tool venv can import packages the project venv has.
    ARGS+=( --ae "PYTHONPATH=${TTS_SRC_MOUNT}:${TTS_SRC_MOUNT}/.deps" )
    # Use harbor's first-class --mounts (Docker-Compose long-form volume). Passing the
    # bind via --ek mounts= is the MANUAL `harbor task` route and does NOT forward to the
    # env on `harbor run` -> the mount silently never binds -> tts not importable ->
    # "Unknown agent type: tts.agent.summarization_agent.SummarizingAgent".
    ARGS+=( --mounts "[{\"type\":\"bind\",\"source\":\"${TTS_SRC}\",\"target\":\"${TTS_SRC_MOUNT}\"}]" )
fi
# (Agent-SETUP timeout is set absolutely via AGENT_SETUP_TIMEOUT_SEC ->
# override_setup_timeout_sec in the JobConfig, not a CLI flag -- see the fresh-run block.
# The in-sandbox toolchain install can blow past the 360s default under high N_CONCURRENT
# / on heavy repos; raise AGENT_SETUP_TIMEOUT_SEC generously for scale / prep.)

# Agent-EXECUTION timeout MULTIPLIER (scales the per-trial agent wall-clock, which is
# the task's timeout_sec -- 3000s for swe-bench-verified). e.g. 2.4 -> 7200s, 2 -> 6000s,
# 0.5 -> 1500s (fail faster). This is the "3000s wall" that produces AgentTimeoutError;
# distinct from the setup-timeout and from mini's step_limit. NB: it is baked into the
# JobConfig + lock, so changing it forces a FRESH run (a resume with a different value
# hits the same lock-mismatch as n_concurrent).
AGENT_TIMEOUT_MULTIPLIER="${AGENT_TIMEOUT_MULTIPLIER:-}"
[ -n "${AGENT_TIMEOUT_MULTIPLIER}" ] && ARGS+=( --agent-timeout-multiplier "${AGENT_TIMEOUT_MULTIPLIER}" )

# Agent-EXECUTION timeout, ABSOLUTE seconds (agent.override_timeout_sec) -- sets the
# per-trial wall directly, dataset-independent (e.g. AGENT_TIMEOUT_SEC=7200 -> 7200s on
# verified OR pro, no base math). harbor has no absolute CLI flag, but its JobConfig
# carries the field, so it works on BOTH fresh (via --print-config -> patch -> `run -c`,
# see the fresh-run block) and resume (in-place patch); it's not in the lock, so no lock
# touch. AGENT_TIMEOUT_MULTIPLIER still exists for relative scaling; if both are set,
# override_timeout_sec is the base the multiplier then scales.
AGENT_TIMEOUT_SEC="${AGENT_TIMEOUT_SEC:-}"

if [ "${BENCH}" = "senior" ]; then
    [ -n "${JUDGE_MODEL}" ] && ARGS+=( --ve "SSB_OVERRIDE_ALL_JUDGE_MODEL=${JUDGE_MODEL}" )
    # A SELF-HOSTED litellm_proxy judge: forward JUDGE_BASE_URL/JUDGE_API_KEY to the
    # verifier as LITELLM_PROXY_API_BASE / LITELLM_PROXY_API_KEY, so a
    # litellm_proxy/<name> JUDGE_MODEL can reach your proxy (same provider path the
    # deliberator uses). These are --ve (verifier env only), so they never collide
    # with the deliberator's agent env --ae values. Also pass them as OPENAI_BASE_URL/
    # OPENAI_API_KEY when no other OPENAI key is set, for litellm's openai-compatible
    # fallback (ssb_lib may read either pair).
    if [ -n "${JUDGE_BASE_URL}" ] && [ -n "${JUDGE_API_KEY}" ]; then
        ARGS+=( --ve "LITELLM_PROXY_API_BASE=${JUDGE_BASE_URL}" )
        ARGS+=( --ve "LITELLM_PROXY_API_KEY=${JUDGE_API_KEY}" )
        [ -z "${OPENAI_API_KEY:-}" ] && ARGS+=( --ve "OPENAI_BASE_URL=${JUDGE_BASE_URL}" --ve "OPENAI_API_KEY=${JUDGE_API_KEY}" )
        [ -z "${OPENAI_API_KEY:-}" ] && export OPENAI_API_KEY="${JUDGE_API_KEY}"  # resume-safe template
    fi
    # Forward whichever judge credentials are present to the verifier phase.
    for k in ANTHROPIC_API_KEY OPENAI_API_KEY PORTKEY_API_KEY; do
        v="${!k:-}"; [ -n "${v}" ] && ARGS+=( --ve "${k}=${v}" )
    done
    if [ -z "${JUDGE_MODEL}" ] \
       && [ -z "${ANTHROPIC_API_KEY:-}" ] \
       && [ -z "${OPENAI_API_KEY:-}" ] \
       && [ -z "${PORTKEY_API_KEY:-}" ] \
       && [ -z "${JUDGE_API_KEY:-}" ]; then
        echo "WARNING: senior needs a judge model + credentials; set JUDGE_MODEL and" >&2
        echo "         ANTHROPIC_API_KEY / OPENAI_API_KEY / PORTKEY_API_KEY /" >&2
        echo "         (JUDGE_BASE_URL + JUDGE_API_KEY for a litellm_proxy judge)." >&2
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
    # Same resume-safety for the litellm_proxy key: harbor stores ${LITELLM_PROXY_API_KEY}
    # as a template only if it matches this process's env, else it redacts it (breaking
    # resume). (LITELLM_PROXY_API_BASE is a URL, not sensitive, so it needs no export.)
    export LITELLM_PROXY_API_KEY="${DELIBERATOR_API_KEY}"
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
    # `harbor job resume` has no CLI for concurrency or the timeout multipliers, and it
    # lock-checks the stored lock.json against one rebuilt from config.json + the trial
    # configs (JobLock equality). So editing any of these on a resume must be done in
    # ALL the places harbor reads/hashes them, or resume either ignores the change or
    # dies with "lock.json does not match the resolved job lock". We patch them in place
    # here so a resume honors N_CONCURRENT / AGENT_TIMEOUT_MULTIPLIER /
    # AGENT_SETUP_TIMEOUT_SEC / AGENT_TIMEOUT_SEC WITHOUT forcing a fresh run
    # (AGENT_SETUP_TIMEOUT_SEC arrives here already converted to a multiplier). Pass the
    # SAME values each chunk (like
    # RESUME_FILTER_ERRORS); they're authoritative. Locations:
    #   n_concurrent_trials            -> job config.json + lock.json (top-level)
    #   agent_timeout_multiplier       -> job config.json + EVERY trial config.json
    #                                     + lock.json trials[] (per-trial, reconciled)
    #   override_timeout_sec           -> job config.json agents[] + EVERY trial
    #   override_setup_timeout_sec        config.json agent (NOT lock: not hashed)
    # (the multiplier sweep touches trial files only when a multiplier is set; the
    #  absolute overrides only when a *_SEC is set. Setup timeout is absolute-only now
    #  -- AGENT_SETUP_TIMEOUT_SEC -> override_setup_timeout_sec, no multiplier.)
    python3 - "${JOB_DIR}" "${N_CONCURRENT}" "${AGENT_TIMEOUT_MULTIPLIER}" "${AGENT_TIMEOUT_SEC}" "${AGENT_SETUP_TIMEOUT_SEC}" "${DELIBERATOR_BASE_URL}" <<'PY'
import json, sys, os, glob
job_dir, nconc, atm, atsec, assec, deliberator = sys.argv[1:7]
_EMPTY = object()
def _load_json(p):
    # Tolerate a missing or EMPTY lock.json/config.json (e.g. a fresh run that
    # crashed before Harbor wrote the lock -- RESUME=auto keys off config.json
    # presence, so an unwritten lock.json here would crash json.loads with
    # JSONDecodeError). Return {} for empty/missing so the patch is a no-op.
    try:
        t = open(p).read().strip()
    except OSError:
        return {}
    if not t:
        return {}
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return {}
load = _load_json
save = lambda p, d: open(p, "w").write(json.dumps(d, indent=4) + "\n")
changed = []

# n_concurrent_trials: JOB-level, in config.json + lock.json (top level).
if nconc:
    n = int(nconc)
    for name in ("config.json", "lock.json"):
        p = os.path.join(job_dir, name)
        if os.path.exists(p):
            d = load(p)
            if d.get("n_concurrent_trials") != n:
                d["n_concurrent_trials"] = n; save(p, d); changed.append(f"{name}:n_concurrent={n}")

# Agent-execution timeout MULTIPLIER (agent_timeout_multiplier): a per-trial field that
# IS in the lock -> patch job config (drives planned trials), every trial config.json
# (reconcile), and lock.json trials[] (lock check).
tmult = {}
if atm:  tmult["agent_timeout_multiplier"] = float(atm)
if tmult:
    def apply_mult(d):
        ch = False
        for k, v in tmult.items():
            if d.get(k) != v: d[k] = v; ch = True
        return ch
    jp = os.path.join(job_dir, "config.json")
    if os.path.exists(jp):
        d = load(jp)
        if apply_mult(d): save(jp, d); changed.append("config:" + ",".join(f"{k}={v}" for k, v in tmult.items()))
    n_tr = 0
    for tp in glob.glob(os.path.join(job_dir, "*", "config.json")):
        d = load(tp)
        if apply_mult(d): save(tp, d); n_tr += 1
    if n_tr: changed.append(f"{n_tr} trial-config multipliers")
    lp = os.path.join(job_dir, "lock.json")
    if os.path.exists(lp):
        d = load(lp)
        if any(apply_mult(t) for t in d.get("trials", [])): save(lp, d); changed.append("lock multipliers")

# Absolute timeouts: agent.override_timeout_sec (EXECUTION) + override_setup_timeout_sec
# (SETUP). AgentConfig fields set DIRECTLY (dataset-independent) and NOT part of the lock,
# so they need NO lock.json patch -- just job config agents[] (drives planned trials) +
# every trial config.json agent (reconcile).
overrides = {}
if atsec: overrides["override_timeout_sec"] = float(atsec)
if assec: overrides["override_setup_timeout_sec"] = float(assec)
if overrides:
    def set_overrides(agent):
        ch = False
        for k, v in overrides.items():
            if agent.get(k) != v: agent[k] = v; ch = True
        return ch
    jp = os.path.join(job_dir, "config.json")
    if os.path.exists(jp):
        d = load(jp)
        if any(set_overrides(a) for a in d.get("agents", [])):
            save(jp, d); changed.append("config:" + ",".join(f"{k}={v}" for k, v in overrides.items()))
    n_tr = 0
    for tp in glob.glob(os.path.join(job_dir, "*", "config.json")):
        d = load(tp); a = d.get("agent")
        if isinstance(a, dict) and set_overrides(a): save(tp, d); n_tr += 1
    if n_tr: changed.append(f"{n_tr} trial-config overrides")

# Deliberator endpoint (agent env). Each chunk serves vLLM on a NEW node:port, but the URL
# is baked into the stored config from the FIRST chunk -- so resume otherwise sends every
# agent at a dead host (NetworkConnectionError for ALL trials, zero forward progress).
# Rewrite the URL keys that ALREADY exist (never add new ones) to THIS chunk's
# DELIBERATOR_BASE_URL: job config.json agents[].env + every trial config.json agent.env.
# API-key keys are left alone (harbor stores them as ${NAME} templates, resolved from env
# at run time). No-op when the endpoint is unchanged (e.g. a stable relay/tunnel URL).
if deliberator:
    _URL_KEYS = ("OPENAI_BASE_URL", "OPENAI_API_BASE", "LITELLM_PROXY_API_BASE")
    def set_urls(env):
        ch = False
        if isinstance(env, dict):
            for k in _URL_KEYS:
                if k in env and env[k] != deliberator:
                    env[k] = deliberator; ch = True
        return ch
    # The agent env is part of the JobLock hash (TrialLock._equality_key ->
    # _frozen_value(self.agent) includes env), so the URL must be rewritten in BOTH the
    # configs AND lock.json/trials[].agent.env -- otherwise the lock harbor rebuilds from
    # the patched configs no longer matches the stored lock.json ("... does not match the
    # resolved job lock"). Patch every entry (no short-circuit).
    jp = os.path.join(job_dir, "config.json")
    if os.path.exists(jp):
        d = load(jp)
        n = sum(int(set_urls(a.get("env"))) for a in d.get("agents", []) if isinstance(a, dict))
        if n: save(jp, d); changed.append(f"config:deliberator={deliberator}")
    n_tr = 0
    for tp in glob.glob(os.path.join(job_dir, "*", "config.json")):
        d = load(tp); a = d.get("agent")
        if isinstance(a, dict) and set_urls(a.get("env")): save(tp, d); n_tr += 1
    if n_tr: changed.append(f"{n_tr} trial-config deliberator URLs")
    lp = os.path.join(job_dir, "lock.json")
    if os.path.exists(lp):
        d = load(lp); n = 0
        for t in d.get("trials", []):
            a = t.get("agent") if isinstance(t, dict) else None
            if isinstance(a, dict) and set_urls(a.get("env")): n += 1
        if n: save(lp, d); changed.append(f"lock:{n} trial deliberator URLs")

if changed:
    print("[resume] patched: " + "; ".join(changed))
PY
    RESUME_ARGS=( job resume -p "${JOB_DIR}" )
    for _et in ${RESUME_FILTER_ERRORS}; do RESUME_ARGS+=( -f "${_et}" ); done
    echo "=================================================================="
    echo "RESUME  job=${JOB_NAME}  dir=${JOB_DIR}  n=${N_CONCURRENT}"
    echo "deliberator=${DELIBERATOR_BASE_URL} (patched into stored config + trials this resume)"
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

# Fresh run. If an ABSOLUTE agent/setup timeout is requested, harbor's `run` CLI can't
# set it (only multipliers) -- but its JobConfig carries override_(setup_)timeout_sec. So
# resolve the JobConfig with --print-config, patch the AgentConfig override fields, and
# run from the patched config (`run -c`). This makes AGENT_TIMEOUT_SEC /
# AGENT_SETUP_TIMEOUT_SEC absolute on FRESH runs, symmetric with the resume patch, no
# multiplier. --print-config templatizes the --ae keys to ${OPENAI_API_KEY}/${MSWEA_API_KEY}
# (resume-safe, since we export them) and preserves extra_allowed_hosts + the config_file
# kwarg, so `run -c` reproduces the same job. Otherwise run directly.
if [ -n "${AGENT_TIMEOUT_SEC}" ] || [ -n "${AGENT_SETUP_TIMEOUT_SEC}" ]; then
    RESOLVED_CFG="${JOBS_DIR}/.${JOB_NAME}.resolved.json"
    mkdir -p "${JOBS_DIR}"
    echo "[fresh] resolving JobConfig (--print-config) to inject absolute timeouts ..."
    "${HARBOR_CMD[@]}" "${ARGS[@]}" --print-config > "${RESOLVED_CFG}"
    python3 - "${RESOLVED_CFG}" "${AGENT_TIMEOUT_SEC}" "${AGENT_SETUP_TIMEOUT_SEC}" <<'PY'
import json, sys
cfg, atsec, assec = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(cfg))
ch = []
for a in d.get("agents", []):
    if atsec: a["override_timeout_sec"] = float(atsec)
    if assec: a["override_setup_timeout_sec"] = float(assec)
if atsec: ch.append(f"override_timeout_sec={atsec}")
if assec: ch.append(f"override_setup_timeout_sec={assec}")
json.dump(d, open(cfg, "w"), indent=2)
print("[fresh] patched JobConfig: " + ", ".join(ch))
PY
    exec "${HARBOR_CMD[@]}" run -c "${RESOLVED_CFG}" -y
fi

exec "${HARBOR_CMD[@]}" "${ARGS[@]}"
