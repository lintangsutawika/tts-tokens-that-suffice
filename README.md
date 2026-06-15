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
