"""Shared fixtures for tokenization tests."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent.parent / "fixtures" / "partial_trajectory.json"

# The scoring model. Override to test another served model.
MODEL = os.getenv("TOKENIZATION_TEST_MODEL", "Qwen/Qwen3.6-35B-A3B")
VLLM_URL = os.getenv("TOKENIZATION_TEST_VLLM_URL", "http://localhost:9999")


@pytest.fixture(scope="session")
def trajectory() -> dict:
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="session")
def context_messages(trajectory) -> list[dict]:
    """Everything up to (not including) the next agent action."""
    return copy.deepcopy(trajectory["messages"][:-1])


@pytest.fixture(scope="session")
def next_action(trajectory) -> dict:
    """The agent action the context is meant to predict — the target y."""
    return copy.deepcopy(trajectory["messages"][-1])


@pytest.fixture(scope="session")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture(scope="session")
def vllm_server():
    """Skip the whole vLLM module unless a matching server is actually up."""
    requests = pytest.importorskip("requests")
    try:
        r = requests.get(f"{VLLM_URL}/v1/models", timeout=5)
        r.raise_for_status()
        served = [m["id"] for m in r.json()["data"]]
    except Exception as exc:
        pytest.skip(f"no vLLM server at {VLLM_URL}: {exc}")
    if MODEL not in served:
        pytest.skip(f"{VLLM_URL} serves {served}, not {MODEL}")
    return VLLM_URL


def to_template_tool_calls(messages: list[dict]) -> list[dict]:
    """
    Parse tool_call arguments from JSON strings into dicts.

    The inverse of to_api_tool_calls. Trajectories are stored in the wire format
    (a string, as model_dump produces), but the Jinja template calls .items() on
    the same field, so anything going to apply_chat_template needs this first.
    """
    out = copy.deepcopy(messages)
    for m in out:
        for tc in m.get("tool_calls") or []:
            args = tc["function"]["arguments"]
            if isinstance(args, str):
                try:
                    tc["function"]["arguments"] = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    tc["function"]["arguments"] = {}
    return out


def to_api_tool_calls(messages: list[dict]) -> list[dict]:
    """
    JSON-encode tool_call arguments.

    The OpenAI schema (and therefore vLLM's request validation) types
    `function.arguments` as a string, while the Jinja chat template calls
    .items() on it and needs a mapping. The two layers disagree, so which form
    a message must be in depends on where it is going.
    """
    out = copy.deepcopy(messages)
    for m in out:
        for tc in m.get("tool_calls") or []:
            args = tc["function"]["arguments"]
            if not isinstance(args, str):
                tc["function"]["arguments"] = json.dumps(args)
    return out
