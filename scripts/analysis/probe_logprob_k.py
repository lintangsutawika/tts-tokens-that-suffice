"""
Verify that requesting k top-k alternatives is dead weight in the distortion reward.

The reward reads only token_logprobs (the realized continuation token); nothing in
the codebase touches top_logprobs. This probe confirms that empirically before we
rely on it: it scores the same trajectories at k=0 and k=20 and asserts the
returned token_logprobs are bit-identical, then reports the latency difference.

If the logprobs differ at all, k=0 is NOT safe and the defaults must be reverted.

Needs a live scoring server (the same one training uses):
    SCORING_BASE_URL=http://localhost:8000/v1 uv run python scripts/analysis/probe_logprob_k.py
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from minisweagent.models.utils.actions_toolcall import BASH_TOOL
from transformers import AutoTokenizer

from tts.data.agent_trajectory import load_collect_trajectories
from tts.train_rl import AGENT_SYSTEM_PROMPT, steps_to_messages
from tts.reward.utils import build_x_scoring_messages, format_continuation
from tts.utils.logprob import precompute_x

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="datasets/summarizer_train.jsonl")
p.add_argument("--model", default=os.environ.get("SCORING_MODEL", "litellm_proxy/Qwen/Qwen3.6-27B-FP8"))
p.add_argument("--api-base", default=os.environ.get("SCORING_BASE_URL", "http://localhost:8000/v1"))
p.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
p.add_argument("--n", type=int, default=4, help="trajectories to probe")
p.add_argument("--split-at-tokens", type=int, default=16384)
p.add_argument("--compaction-budget", type=int, default=9000)
p.add_argument("--concurrency", type=int, default=0,
               help="if >0, also time N concurrent precomputes at each k")
args = p.parse_args()

tok = AutoTokenizer.from_pretrained(args.tokenizer)

print(f"scoring server : {args.api_base}")
print(f"scoring model  : {args.model}")
print(f"loading {args.dataset} ...")
trajs, seen = [], 0
for t in load_collect_trajectories(args.dataset):
    seen += 1
    s = t.threshold_split(
        tok,
        split_at_tokens=args.split_at_tokens,
        min_prefix=6,
        min_suffix=2,
        max_continuation_tokens=args.split_at_tokens - args.compaction_budget,
    )
    if s is not None:
        trajs.append(s)
    if len(trajs) >= args.n:
        break
print(f"using {len(trajs)} split trajectories (from {seen} scanned)\n")


def build(traj):
    sys_p = traj.agent_system_prompt or AGENT_SYSTEM_PROMPT
    partial = steps_to_messages(traj.steps, traj.task, system_prompt=sys_p)
    cont = steps_to_messages(traj.continuation, traj.task, system_prompt=sys_p)[2:]
    x_messages = build_x_scoring_messages(partial)
    y_text = format_continuation(x_messages, cont, tok)
    return x_messages, y_text


def score(traj, k):
    x_messages, y_text = build(traj)
    t0 = time.perf_counter()
    xl = precompute_x(x_messages, y_text, args.model, args.api_base, tok, k=k, tools=[BASH_TOOL])
    return xl, time.perf_counter() - t0


failures, t_by_k = [], {0: [], 20: []}
for i, traj in enumerate(trajs):
    row = {}
    for k in (20, 0):
        xl, dt = score(traj, k)
        if xl is None:
            print(f"[{i}] k={k:<2} -> scoring FAILED (server error); aborting")
            sys.exit(2)
        row[k] = xl
        t_by_k[k].append(dt)

    a, b = row[20], row[0]
    same_len = len(a.x_tok) == len(b.x_tok)
    same_vals = same_len and all(
        (x is None and y is None) or (x is not None and y is not None and x == y)
        for x, y in zip(a.x_tok, b.x_tok)
    )
    ok = same_len and same_vals and a.n_completion == b.n_completion
    if not ok:
        failures.append(i)
    print(
        f"[{i}] ctx={a.n_x_ctx:>6} tok  completion={a.n_completion:>4} tok  "
        f"k=20 {t_by_k[20][-1]:6.2f}s  k=0 {t_by_k[0][-1]:6.2f}s  "
        f"identical={'YES' if ok else 'NO <<<'}"
    )
    if not ok:
        print(f"     len: k=20 {len(a.x_tok)} vs k=0 {len(b.x_tok)}")
        for j, (x, y) in enumerate(zip(a.x_tok, b.x_tok)):
            if x != y:
                print(f"     first mismatch at {j}: k=20 {x!r} vs k=0 {y!r}")
                break

print()
if failures:
    print(f"FAIL: {len(failures)}/{len(trajs)} trajectories differ between k=0 and k=20.")
    print("k=0 is NOT safe -- revert the k defaults to 20 in")
    print("  src/tts/utils/logprob.py (precompute_x, score_completion_z, score_completion)")
    print("  src/tts/reward/utils.py (precompute_x_context)")
    sys.exit(1)

m20, m0 = statistics.mean(t_by_k[20]), statistics.mean(t_by_k[0])
print(f"PASS: token_logprobs bit-identical at k=0 and k=20 across {len(trajs)} trajectories.")
print(f"  mean latency  k=20 {m20:6.2f}s   k=0 {m0:6.2f}s   speedup {m20 / m0:.2f}x")

if args.concurrency > 0:
    n = args.concurrency
    print(f"\nconcurrent: {n} precomputes at once (mirrors the batch precompute pool)")
    for k in (20, 0):
        batch = [trajs[i % len(trajs)] for i in range(n)]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n) as ex:
            list(ex.map(lambda t: score(t, k), batch))
        wall = time.perf_counter() - t0
        serial_est = statistics.mean(t_by_k[k]) * n
        print(f"  k={k:<2} wall {wall:6.2f}s   vs {serial_est:6.2f}s if fully serial "
              f"-> {serial_est / wall:.2f}x overlap")
