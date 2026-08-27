#!/usr/bin/env bash
set -euo pipefail

# Scoring server for the distortion reward (train_rl scoring_base_url -> :8000).
#
# max-model-len must fit the ENTIRE x-context scoring call, |x| + |y|:
#   |x| = trajectory prefix at the compression trigger. With split_at_tokens=16384 this
#         overshoots to ~19.4k, because the split lands on a turn boundary, not mid-turn.
#   |y| = continuation being scored, capped at split_at_tokens - compaction_token_budget
#         = 16384 - 9000 = 7384.
# So the call needs ~27k plus chat-template/tool overhead. At 20000 every score raised
# ContextWindowExceededError; a failed score returns None and the sample is DROPPED, so
# whole groups disappeared and training got no gradient at all.
#
# Invariant: max-model-len > (split_at_tokens * ~1.2) + (split_at_tokens - compaction_token_budget).
# Re-check this when changing split_at_tokens in train_summarizer.sh.
#    --kv-cache-dtype fp8 \
uv run --with git+https://github.com/vllm-project/vllm.git --with cohere_melody vllm serve CohereLabs/North-Mini-Code-1.0 \
    --tensor-parallel-size 2 \
    --data-parallel-size 2 \
    --max-model-len 65536 \
    --disable-custom-all-reduce \
    --gpu-memory-utilization 0.85 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 4096 \
    --port 8080 \
    --enable-prefix-caching \
    --long-prefill-token-threshold 0 \
    --enforce-eager \
    --enable-auto-tool-choice \
    --language-model-only \
    --tool-call-parser cohere_command4 \
    --reasoning-parser cohere_command4
