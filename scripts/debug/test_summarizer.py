#!/usr/bin/env python3
"""
Test summarizer output by sampling trajectories from the dataset and
generating summaries with the current prompt format.

Usage:
    BASE_URL=http://localhost:8000/v1 MODEL=Qwen/Qwen3.5-9B \
        uv run scripts/debug/test_summarizer.py \
        [--dataset datasets/summarizer_train.jsonl] \
        [--n 3] [--seed 0]
"""

import argparse
import os
import random
import sys
import textwrap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import litellm

from tts.data.agent_trajectory import (
    SYSTEM_PROMPT,
    format_trajectory_text,
    load_collect_trajectories,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/summarizer_train.jsonl")
    parser.add_argument("--n", type=int, default=3, help="Number of trajectories to test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    base_url = os.environ.get("BASE_URL", "http://localhost:8000/v1").rstrip("/")
    model = os.environ.get("MODEL", "Qwen/Qwen3.5-9B")
    vllm_model = f"hosted_vllm/{model}"

    trajectories = load_collect_trajectories(args.dataset)
    rng = random.Random(args.seed)
    rng.shuffle(trajectories)
    trajectories = trajectories[: args.n]
    # trajectories = [0]

    for i, traj in enumerate(trajectories):
        split = traj.sample_split(rng)

        convo = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_trajectory_text(split.steps)},
        ]

        print(f"\n{'='*80}")
        print(f"Trajectory {i+1} | id={traj.trajectory_id}")
        print(f"  n_prefix_steps={len(split.steps)}  n_continuation_steps={len(split.continuation)}")
        print(f"{'='*80}")

        print("\n--- SUMMARIZER INPUT ---")
        for msg in convo:
            role = msg["role"]
            content = msg["content"]
            print(f"\n[{role.upper()}]")
            print(textwrap.indent(content[:1000] + ("..." if len(content) > 1000 else ""), "  "))

        print("\n--- GENERATION ---")
        try:
            # convo = [
            #     {"role": "user", "content": "How many Rs in Strawberry?"},
            # ]
            response = litellm.completion(
                model=vllm_model,
                base_url=base_url,
                api_key="dummy",
                messages=convo,
                max_tokens=args.max_tokens,
                temperature=0.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            print(response)
            summary = response["choices"][0]["message"]["content"] or ""
            print(textwrap.indent(summary[:800] + ("..." if len(summary) > 800 else ""), "  "))
        except Exception as e:
            print(f"  [ERROR: {e}]")

    print(f"\n{'='*80}")
    print("Done.")


if __name__ == "__main__":
    main()
