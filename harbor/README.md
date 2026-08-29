# Harbor evals: Verified / Pro / Senior SWE-bench

Downstream agent evaluation on three SWE benchmarks, run through
[Harbor](https://www.harborframework.com) with the **Modal** environment and the
**mini-swe-agent** harness. This is the baseline stack; the trained summarizer
plugs in later as a custom Harbor agent (see [Adding the summarizer arm](#adding-the-summarizer-arm)).

| Benchmark | Harbor dataset | Size | Grading |
|---|---|---|---|
| SWE-bench Verified | `swe-bench/swe-bench-verified` | 500 | deterministic tests (FAIL_TO_PASS / PASS_TO_PASS) |
| SWE-bench Pro | `scale-ai/swe-bench-pro` | 731 | deterministic tests |
| Senior SWE-bench | `snorkel-ai/senior-swe-bench-v2026.06` | 50 | **LLM judge** (`ssb_lib` validation judge) |

All three are pre-published Harbor datasets — no adapter to write. Harbor pulls
each task's prebuilt image (Verified: `swebench/sweb.eval.*`, Pro:
`jefzda/sweap-images:*`; Senior builds from a per-task Dockerfile) into a Modal
sandbox, installs mini-swe-agent inside it, has the agent solve the task by
calling our deliberator model, then runs the task's verifier and records reward.

## How the pieces fit

```
 host (this repo)                 Modal sandbox (per task)          your infra
 ────────────────                 ───────────────────────          ──────────
 harbor run  ── spins up ──►      mini-swe-agent (bash agent)  ──►  Qwen deliberator
   -e modal                         calls OPENAI_BASE_URL              (hosted LiteLLM proxy,
   -a mini-swe-agent                                                   or vLLM on a routable node)
   -d <dataset>                   verifier phase  ──────────────►  judge model (Senior only:
                                    tests / LLM judge                 Anthropic/OpenAI/Portkey)
```

The agent runs **inside** Modal, so:
- the **deliberator endpoint must be reachable from Modal**. The hosted LiteLLM
  proxy (`https://cmu.litellm.ai`) is; a `localhost` URL is not. A vLLM you serve
  yourself works only if its node is routable from Modal.
- the sandbox egress is allowlisted; the runner adds your deliberator host via
  `--allow-agent-host` so the agent can reach it.

## Prereqs

```bash
uv tool install harbor          # installs `harbor` (also `hb`, `hr`)
modal setup                     # once; writes ~/.modal.toml   (already configured here)
```

Point the deliberator at a Modal-reachable endpoint: the hosted LiteLLM proxy
(`DELIBERATOR_API_URL=https://cmu.litellm.ai` in `.env`; base URL + `/v1`), or a
vLLM you serve on a routable node via `scripts/eval/harbor_eval.sbatch`.

## Run (baseline: stock mini-swe-agent, full context)

Use `scripts/harbor_eval.sh <verified|pro|senior>` — **from the repo root** (the
mini config path is repo-relative). Everything is env-var driven:

```bash
# smoke test: 5 Verified tasks, 4 in parallel, against the hosted proxy
DELIBERATOR_BASE_URL=https://cmu.litellm.ai/v1 \
  N_TASKS=5 scripts/eval/harbor_eval.sh verified

# SWE-bench Pro, 50 tasks, 16 concurrent
DELIBERATOR_BASE_URL=https://cmu.litellm.ai/v1 \
  N_TASKS=50 N_CONCURRENT=16 scripts/eval/harbor_eval.sh pro

# Senior (needs a judge model + its credentials)
DELIBERATOR_BASE_URL=https://cmu.litellm.ai/v1 \
  JUDGE_MODEL=anthropic/claude-opus-4-1 ANTHROPIC_API_KEY=sk-... \
  scripts/eval/harbor_eval.sh senior
```

The deliberator endpoint must be reachable from inside Modal. The hosted LiteLLM
proxy (`https://cmu.litellm.ai`, in `.env` as `DELIBERATOR_API_URL`) already is;
a vLLM you serve yourself on a compute node is only reachable if that node is
routable from Modal (see the Slurm entry point below).

### Slurm: serve vLLM + eval on one node

`scripts/eval/harbor_eval.sbatch` runs the whole thing as a single Slurm job: it
optionally spins up the deliberator with vLLM on the node's GPUs, then runs the
Harbor eval (which still orchestrates Modal) against it — tearing the server down
on exit.

```bash
# serve Qwen locally on this node's GPUs, then eval Verified
sbatch scripts/eval/harbor_eval.sbatch verified

# use the hosted proxy instead — no GPUs needed
DELIBERATOR_BASE_URL=https://cmu.litellm.ai/v1 N_TASKS=50 \
  sbatch --gres=gpu:0 scripts/eval/harbor_eval.sbatch pro
```

Serve mode kicks in when `DELIBERATOR_BASE_URL` is unset (`SERVE_VLLM=1/0` forces
it). Extra knobs: `VLLM_MODEL` (default `Qwen/Qwen3.6-35B-A3B`), `TENSOR_PARALLEL`
(4), `MAX_MODEL_LEN` (48000), `VLLM_PORT`, and **`VLLM_PUBLIC_HOST`** — the
address Modal uses to reach the server (defaults to the node's FQDN; override it
if the node isn't routable from Modal, or front the vLLM with your proxy and pass
`DELIBERATOR_BASE_URL` instead).

### Knobs (env vars)

| Var | Default | Meaning |
|---|---|---|
| `DELIBERATOR_BASE_URL` | — (required) | public OpenAI-compatible endpoint, ending `/v1` |
| `MODEL` | `openai/Qwen/Qwen3.6-35B-A3B` | provider/model; `<name>` must match what the server serves |
| `DELIBERATOR_API_KEY` | `EMPTY` | key for the endpoint (vLLM ignores the value but a non-empty key is required) |
| `ENV` | `modal` | Harbor environment (`docker` for a local smoke test) |
| `N_CONCURRENT` | `4` | parallel trials |
| `N_TASKS` | all | cap the number of tasks (`-l`) |
| `INCLUDE` / `EXCLUDE` | — | task-name glob filters (`-i` / `-x`) |
| `MINI_CONFIG` | `harbor/mini_qwen.yaml` | mini-swe-agent sampling config |
| `JOB_NAME` / `JOBS_DIR` | ts / `jobs` | output naming |
| `ALLOW_HOST` | derived from URL | host added to the sandbox egress allowlist |
| `JUDGE_MODEL` | task default | Senior only: `SSB_OVERRIDE_ALL_JUDGE_MODEL` |

`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `PORTKEY_API_KEY` in the environment are
forwarded to the Senior **verifier** (judge) phase automatically.

## Deliberator sampling

`harbor/mini_qwen.yaml` sets Qwen3 recommended sampling (temp 0.6, top_p 0.95,
top_k 20, thinking on), layered on mini-swe-agent's `-c mini` scaffold. **Not
greedy** — temp 0 deterministically stalls Qwen into mini-swe's
`RepeatedFormatError` loop (see memory `swebench-downstream-eval`).

## Results

- Per-job output under `jobs/<job-name>/` — one trial dir per (task × attempt),
  each with the ATIF `trajectory.json`, the agent log, and the **verifier
  reward** (pass/fail for Verified/Pro, judge verdict for Senior).
- Browse everything: `harbor view jobs/` (web UI on :8080).
- LLM trajectory critique (reward-hacking / spec-following rubrics), optional:
  `harbor analyze jobs/<job-name>`.

## Adding the summarizer arm

The baseline uses stock `-a mini-swe-agent` (no compaction). To evaluate the
trained summarizer's in-loop compaction under the same harness, register a custom
Harbor agent that wraps mini-swe-agent with our `SummarizingAgent` and pass it as
an import path: `-a your_module:YourSummarizingMiniAgent`. It reuses the existing
compaction logic (`tts.summarization` + `tts.agent.summarization_agent`); the arm
comparison (full / truncation / trained) then runs across all three benchmarks on
the same Modal + judge infrastructure. Not built yet — this is the next step after
baseline numbers land.

## Security note

Credentials are passed to the agent/verifier via Harbor's `--ae` / `--ve`
(KEY=VALUE) flags, which places them in the host process's argv. Keep runs on a
trusted host. Do not `echo` the assembled command (it expands secrets); the
runner's own banner never prints keys.
