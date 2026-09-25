# tts-tokens-that-suffice

## Setup

Install dependencies:

```bash
uv sync --extra fsdp
```

flash-attn is installed automatically via the prebuilt wheel for torch 2.8 + cu12.

Install vllm separately (conflicts with skyrl on transformers; bypass with `--no-deps`):

```bash
uv pip install vllm==0.19.1 --no-deps
```

## Running

Start the Tinker API server:

```bash
TINKER_API_KEY=tml-dummy uv run -m tts.tinker.api \
    --base-model "Qwen/Qwen3-4B-Instruct-2507" --backend fsdp
```

Run the SL training loop test:

```bash
TINKER_API_KEY=tml-dummy uv run tests/train_sl_loop.py
```

### SkyRL server on AMD (ROCm), per-GPU config

On the AMD cluster the tinker server runs from a pre-built SkyRL image via
`server/scripts/run.sh`. `GPU=` selects a hardware-tuned config: MI210 (4×64 GB)
is memory-tight (`config/rl-mi210x.json`, TP=2, low `gpu_memory_utilization`);
MI300/MI325 (192–256 GB) relax to TP=1 + high utilization
(`config/rl-mi300x.json`). Unset `GPU` falls back to the generic `config/rl.json`.
Run these **on the GPU node** (inside the Slurm allocation), from the repo root;
the server listens on `:9123`.

```bash
# MI210 node (4×64 GB)
MODEL=Qwen/Qwen3.5-9B GPU=mi210 BACKEND=megatron bash server/scripts/run.sh rl

# MI300 / MI325 node (8 GPUs)
MODEL=Qwen/Qwen3.5-9B GPU=mi300 BACKEND=megatron bash server/scripts/run.sh rl
```

Notes:

- `MODEL=Qwen/Qwen3.5-9B` is required — `run.sh` defaults to `Qwen/Qwen3-8B`,
  the wrong base for the summarizer and its checkpoints.
- `rl` (not `sft`) is the mode that enables the vLLM inference engines.
- `SIF` defaults to `tts-server-megatron.sif`; override to pin a different image,
  e.g. the 0.20.2 rollback:
  ```bash
  SIF=/work1/grahamneubig/lsutawik/tts-server-megatron.working-0.20.2.sif \
  MODEL=Qwen/Qwen3.5-9B GPU=mi300 BACKEND=megatron bash server/scripts/run.sh rl
  ```
- The script manages the node-local SQLite DB, its snapshots, and the
  eager/router patches automatically.

## Evaluations

### Downstream SWE-bench

Run the mini-SWE-agent on SWE-bench with the summarizer compressing its context
in-loop, and report the resolve rate. Three arms:

- `truncation` — keep first + last turns, drop the middle (no summary; the do-nothing baseline)
- `base` — summarize with the untrained base model (isolates what training bought)
- `trained` — summarize with a trained tinker LoRA checkpoint

The deliberator (task-solving) model is identical across arms; only how the
context is compressed differs. Requires the deliberator served at `:8000`, and
(for `base`/`trained`) the tinker server at `:9123`.

Run all three arms on a slice via the launcher:

```bash
# defaults: SWE-bench Verified, slice 0:20, deliberator Qwen3.6-35B-A3B at :8000
SLICE=0:20 sbatch scripts/eval/eval_swebench.sh

# baseline only (needs no tinker server)
MODES=truncation SLICE=0:5 sbatch scripts/eval/eval_swebench.sh

# point at a specific checkpoint
CHECKPOINT=tinker://model_7cf52d89/weights/000084 sbatch scripts/eval/eval_swebench.sh
```

Knobs (env vars): `MODES`, `SLICE`, `WORKERS`, `MODEL` (deliberator), `CHECKPOINT`,
`COMPRESS_AT_TOKENS`, `KEEP_FIRST`, `KEEP_LAST_TURNS`, `TEMP` (deliberator sampling;
`0.0` greedy by default for low cross-arm variance).

Or run a single arm directly:

```bash
uv run -m tts.eval_swebench \
    --dataset swe-bench --data-source swe-bench --slice 0:20 \
    --mode trained --checkpoint tinker://model_7cf52d89/weights/000084 \
    --output outputs/eval-swebench-trained -w 4 \
    -m litellm_proxy/Qwen/Qwen3.6-35B-A3B \
    -c swebench.yaml -c model.model_kwargs.api_base=http://0.0.0.0:8000/v1
```

Each run writes per-instance trajectories to the output dir and a
`results_summary.json` with the resolve rate and compression stats.


for SUM_LEN in 16384 24576 32768
do
    export SUMMARIZER_API_BASE=http://0.0.0.0:8001/v1
    export PORT=9122
    export SUMMARIZER_MODEL=litellm_proxy/Qwen/Qwen3-8B
    export MODEL=litellm_proxy/Qwen/Qwen3.6-27B-FP8
    export MODES=base
    export SLICE=0:500
    export WORKERS=32
    export GRADE=false
    export COMPRESS_AT_TOKENS=${SUM_LEN}
    export DATASET=swe-bench
    bash scripts/eval/eval_swebench.sh
done

test