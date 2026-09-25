"""
What tokenizer.apply_chat_template does with an agent trajectory.

These pin the *local* rendering path — the one tts.utils.logprob uses to build
the exact prompt string it sends to /v1/completions. They deliberately say
nothing about what a served chat endpoint would do with the same messages;
test_vllm_render.py covers that, and the two do not agree.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from .conftest import MODEL, to_template_tool_calls

# Renderings are written here so the prompt can be eyeballed after a run.
OUT_DIR = Path(__file__).parent / "rendered"


@pytest.fixture(scope="module")
def tokenizer():
    """AutoTokenizer for the model under test, overriding the conftest fixture."""
    return AutoTokenizer.from_pretrained(MODEL)


def render(tokenizer, messages, **kwargs):
    """Render as the local scoring path does — arguments coerced to dicts first."""
    return tokenizer.apply_chat_template(
        to_template_tool_calls(messages), tokenize=False, add_generation_prompt=True, **kwargs
    )


def save_rendering(name: str, text: str, **extra) -> Path:
    """
    Write a rendering to OUT_DIR as both .txt and .json.

    The .txt is what you actually read — a prompt full of escaped newlines is
    unreviewable inside JSON. The .json carries the same text plus metadata for
    programmatic diffing.
    """
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / f"{name}.txt").write_text(text)
    payload = {"name": name, "n_chars": len(text), **extra, "rendered": text}
    (OUT_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return OUT_DIR / f"{name}.txt"


def test_fixture_shape(context_messages, next_action):
    """The fixture is a context ending in a tool result, plus one agent action."""
    assert context_messages[0]["role"] == "system"
    assert context_messages[1]["role"] == "user"
    assert context_messages[-1]["role"] == "tool"
    assert next_action["role"] == "assistant"


def test_save_rendering(tokenizer, context_messages, next_action):
    """
    Render the fixture and write it to OUT_DIR for inspection.

    Saves three variants, because the difference between them is the whole
    question: with reasoning_content (what apply_chat_template does), without
    it (what the served endpoint actually produces), and the continuation y.
    """
    from tts.summarization.utils import format_continuation

    with_reasoning = render(tokenizer, context_messages)
    stripped = [
        {k: v for k, v in m.items() if k != "reasoning_content"} for m in context_messages
    ]
    without_reasoning = render(tokenizer, stripped)
    y = format_continuation(
        to_template_tool_calls(context_messages),
        to_template_tool_calls([next_action]),
        tokenizer,
    )

    save_rendering(
        "context_with_reasoning",
        with_reasoning,
        n_tokens=len(tokenizer.encode(with_reasoning)),
        n_messages=len(context_messages),
        ends_with_open_think=with_reasoning.endswith("<think>\n"),
    )
    save_rendering(
        "context_without_reasoning",
        without_reasoning,
        n_tokens=len(tokenizer.encode(without_reasoning)),
        n_messages=len(stripped),
        ends_with_open_think=without_reasoning.endswith("<think>\n"),
    )
    save_rendering(
        "continuation_y",
        y,
        n_tokens=len(tokenizer.encode(y)),
        starts_with_dangling_close=y.lstrip().startswith("</think>"),
    )

    assert (OUT_DIR / "context_with_reasoning.txt").exists()
    assert (OUT_DIR / "context_without_reasoning.txt").exists()
    assert (OUT_DIR / "continuation_y.txt").exists()


