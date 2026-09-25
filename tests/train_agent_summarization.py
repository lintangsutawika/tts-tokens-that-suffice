"""
Smoke-test for the agent trajectory summarization training loop.

Start the server first:
    TINKER_API_KEY=tml-dummy uv run -m tts.tinker.api \\
        --base-model "Qwen/Qwen3-4B-Instruct-2507" --backend fsdp

Then run this script:
    TINKER_API_KEY=tml-dummy uv run tests/train_agent_summarization.py

Pass chz overrides to change any Config field, e.g.:
    TINKER_API_KEY=tml-dummy uv run tests/train_agent_summarization.py \\
        dataset_path=/path/to/my_trajectories.jsonl
"""

import sys

import chz
from tts.train_sft import Config, main

# Points at the bundled sample dataset in this repo
_SAMPLE_DATA = "tests/fixtures/sample_trajectories.jsonl"

DEFAULTS = [
    "base_url=http://localhost:8000",
    "model_name=Qwen/Qwen3-4B-Instruct-2507",
    f"dataset_path={_SAMPLE_DATA}",
    "batch_size=4",
    "num_epochs=1",
    "save_every=0",
    "log_path=/tmp/tinker-agent-summarization-test",
]

if __name__ == "__main__":
    sys.argv = [sys.argv[0]] + DEFAULTS + sys.argv[1:]
    chz.nested_entrypoint(main)
