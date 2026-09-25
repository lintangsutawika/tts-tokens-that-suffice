"""
What a served vLLM endpoint does with the same agent trajectory.

The headline: vLLM validates chat requests against the OpenAI message schema,
which has no reasoning_content field, so reasoning is dropped *before* the
chat template ever runs. Local apply_chat_template keeps it. The two paths
therefore render different prompts from identical messages, and any code that
renders locally and posts to /v1/completions is not reproducing what a chat
client would have gotten.

Requires a live server; skipped otherwise (see the vllm_server fixture).
"""

from __future__ import annotations

import json

import pytest

from .conftest import MODEL, to_api_tool_calls, to_template_tool_calls

requests = pytest.importorskip("requests")


def server_prompt(url, tokenizer, messages) -> str:
    """Round-trip messages through /tokenize and decode back to the prompt text."""
    r = requests.post(
        f"{url}/tokenize",
        json={
            "model": MODEL,
            "messages": to_api_tool_calls(messages),
            "add_generation_prompt": True,
            "return_token_strs": True,
        },
        timeout=60,
    )
    r.raise_for_status()
    return tokenizer.decode(r.json()["tokens"])


def test_server_requires_string_arguments(vllm_server, context_messages):
    """
    function.arguments must be a JSON string over the wire.

    The chat template needs a dict for the same field, so messages cannot be
    shared verbatim between the two paths.
    """
    assert any(m.get("tool_calls") for m in context_messages)
    r = requests.post(
        f"{vllm_server}/tokenize",
        json={
            "model": MODEL,
            "messages": to_template_tool_calls(context_messages),  # dict args
            "add_generation_prompt": True,
        },
        timeout=60,
    )
    assert r.status_code == 400
    assert "string_type" in r.text or "valid string" in r.text


def test_server_drops_reasoning_content(vllm_server, tokenizer, context_messages):
    """reasoning_content does not survive request validation."""
    prompt = server_prompt(vllm_server, tokenizer, context_messages)
    reasonings = [
        m["reasoning_content"]
        for m in context_messages
        if m["role"] == "assistant" and m.get("reasoning_content")
    ]
    assert reasonings, "fixture has no reasoning to drop"
    for r in reasonings:
        assert r not in prompt, "reasoning_content unexpectedly survived to the prompt"


def test_server_keeps_think_tags_embedded_in_content(vllm_server, tokenizer):
    """
    Reasoning inside `content` DOES survive — the drop is field-specific.

    So the loss is about where reasoning is carried, not about thinking per se.
    """
    tool_calls = [{
        "id": "c0", "type": "function",
        "function": {"name": "bash", "arguments": json.dumps({"command": "ls"})},
    }]
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "<think>SENTINEL</think>visible", "tool_calls": tool_calls},
        {"role": "tool", "content": "out"},
    ]
    assert "SENTINEL" in server_prompt(vllm_server, tokenizer, msgs)


def test_server_and_local_render_disagree(vllm_server, tokenizer, context_messages):
    """
    The divergence is exactly the reasoning tokens.

    Pinned as a test because it is easy to "verify" the local template in
    isolation and wrongly conclude the served path behaves the same way.
    """
    local = tokenizer.apply_chat_template(
        to_template_tool_calls(context_messages), tokenize=False, add_generation_prompt=True
    )
    served = server_prompt(vllm_server, tokenizer, context_messages)
    assert served != local

    n_local = len(tokenizer.encode(local))
    n_served = len(tokenizer.encode(served))
    assert n_served < n_local

    reasoning_tokens = sum(
        len(tokenizer.encode(m["reasoning_content"]))
        for m in context_messages
        if m["role"] == "assistant" and m.get("reasoning_content")
    )
    # Not exact: dropping the text also changes surrounding whitespace tokens.
    assert abs((n_local - n_served) - reasoning_tokens) <= 0.5 * reasoning_tokens


def test_stripping_reasoning_locally_matches_the_server(
    vllm_server, tokenizer, context_messages
):
    """
    Dropping reasoning_content locally reproduces the served prompt exactly.

    This is the construction the x/z contexts should use if they are to match
    what the agent actually conditioned on at rollout time.
    """
    stripped = [
        {k: v for k, v in m.items() if k != "reasoning_content"}
        for m in to_template_tool_calls(context_messages)
    ]
    local = tokenizer.apply_chat_template(
        stripped, tokenize=False, add_generation_prompt=True
    )
    assert local == server_prompt(vllm_server, tokenizer, context_messages)
