"""
Smoke-test for the SL training loop against a local SkyRL Tinker server.

Start the server first:
    python -m skyrl.tinker.api --base-model "Qwen/Qwen3-4B-Instruct-2507" --backend fsdp

Then run this script:
    TINKER_API_KEY=tml-dummy python tests/train_sl_loop.py

Or pass chz overrides to change any Config field:
    TINKER_API_KEY=tml-dummy python tests/train_sl_loop.py base_url=http://localhost:9000
"""

import sys

import chz
from tinker_cookbook.recipes.sl_loop import Config, main

DEFAULTS = [
    "base_url=http://localhost:8000",
    "model_name=Qwen/Qwen3-4B-Instruct-2507",
    "batch_size=4",
    "save_every=0",
    "log_path=/tmp/tinker-sl-loop-test",
    "train_on_what=LAST_ASSISTANT_MESSAGE",
]

if __name__ == "__main__":
    # Prepend defaults so CLI args can still override them
    sys.argv = [sys.argv[0]] + DEFAULTS + sys.argv[1:]
    chz.nested_entrypoint(main)
