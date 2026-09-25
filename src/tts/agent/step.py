"""
Reproduce one mini-swe step from a partial trajectory.

Answers "what would the agent do next, given this context?" — the question the
distortion reward is really asking, since y is the agent's next action. Running
it through a real DefaultAgent rather than calling litellm directly is the whole
point: the harness applies `_prepare_messages_for_api` (which drops the `extra`
key), passes `tools=[BASH_TOOL]`, and merges the config's model_kwargs. Bypass
any of that and the action you get back is not the one the agent would have
taken.

    result = step_once(messages, model_name="Qwen/Qwen3.6-35B-A3B",
                       api_base="http://localhost:9999/v1")
    result.action        # {"command": "ls /testbed"}
    result.message       # the raw assistant message (reasoning_content + tool_calls)

By default only the model is queried — producing the action but not running it,
which is all an offline trajectory can support. Pass an `env` to also execute it
and get the observation back, completing the step as the live agent would.

Note vLLM drops `reasoning_content` from inbound messages, so prior-turn
thinking in `messages` will not reach the model however it is passed. See
tests/tokenization/test_vllm_render.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from minisweagent.agents.default import DefaultAgent
from minisweagent.models import get_model


@dataclass
class StepResult:
    """One step's worth of agent output."""

    message: dict  # raw assistant message: content, reasoning_content, tool_calls, extra
    actions: list[dict] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)  # empty unless executed
    messages_sent: list[dict] = field(default_factory=list)  # exactly what went to the API

    @property
    def action(self) -> dict | None:
        """The first action, which is the only one for a single-tool agent."""
        return self.actions[0] if self.actions else None

    @property
    def reasoning(self) -> str:
        return self.message.get("reasoning_content") or ""

    @property
    def content(self) -> str:
        return self.message.get("content") or ""


class _NoEnvironment:
    """Placeholder env for query-only stepping; raises if anything tries to run."""

    def execute(self, *args, **kwargs):
        raise RuntimeError("step_once() was called without an env; pass execute=True with one")


def build_step_agent(
    model=None,
    model_name: str = "",
    api_base: str = "",
    env=None,
    model_kwargs: dict | None = None,
    **agent_kwargs,
) -> DefaultAgent:
    """
    A DefaultAgent wired for stepping, with no task templates required.

    AgentConfig demands system_template/instance_template because run() renders
    them to seed the conversation. We supply the history directly, so run() is
    never called and the templates stay empty.
    """
    if model is None:
        if not model_name:
            raise ValueError("pass either `model` or `model_name`")
        cfg: dict[str, Any] = {
            # Locally served models have no entry in litellm's price table, and
            # mini-swe raises rather than guessing. Replay does not bill anyone,
            # so default to tolerating it; override via model_kwargs.
            "cost_tracking": "ignore_errors",
            "model_name": model_name,
            **(model_kwargs or {}),
        }
        if api_base:
            cfg.setdefault("model_kwargs", {})["api_base"] = api_base
        model = get_model(config=cfg)
    agent_kwargs.setdefault("system_template", "")
    agent_kwargs.setdefault("instance_template", "")
    return DefaultAgent(model, env or _NoEnvironment(), **agent_kwargs)


def step_once(
    messages: list[dict],
    model=None,
    model_name: str = "",
    api_base: str = "",
    env=None,
    execute: bool = False,
    model_kwargs: dict | None = None,
    **agent_kwargs,
) -> StepResult:
    """
    Run one step of the mini-swe loop against `messages`.

    `messages` is the conversation so far (system, task, then assistant/tool
    pairs) — a partial trajectory, or a compacted one from tts.summarization.
    It is copied, so the caller's list is never mutated.

    With execute=False (default) the model is queried and the action parsed, but
    nothing runs; `observations` stays empty. With execute=True and an `env`, the
    action is executed and the observation messages returned, matching
    DefaultAgent.step() exactly.
    """
    if execute and env is None:
        raise ValueError("execute=True requires an env")

    agent = build_step_agent(
        model=model,
        model_name=model_name,
        api_base=api_base,
        env=env,
        model_kwargs=model_kwargs,
        **agent_kwargs,
    )
    agent.messages = [dict(m) for m in messages]
    n_before = len(agent.messages)

    # What the harness would actually put on the wire, captured before querying
    # so it is available even if the call fails.
    messages_sent = agent.model._prepare_messages_for_api(agent.messages)

    message = agent.query()  # appends the assistant message to agent.messages
    actions = message.get("extra", {}).get("actions", [])

    observations: list[dict] = []
    if execute:
        observations = agent.execute_actions(message)

    assert len(agent.messages) >= n_before + 1, "query() did not append a message"
    return StepResult(
        message=message,
        actions=actions,
        observations=observations,
        messages_sent=messages_sent,
    )

