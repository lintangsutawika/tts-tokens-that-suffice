"""
Agent trajectory data format and conversion utilities.

Trajectories follow a schema inspired by Agent Data Protocol (ADP):
each trajectory has a task description, an ordered list of steps (assistant
thoughts/tool calls and tool observations), and a target summary.

JSONL on disk: one JSON object per line, schema:
  {
    "trajectory_id": "<str>",          # optional unique id
    "task": "<str>",                   # what the agent was asked to do
    "steps": [
      {
        "role": "assistant" | "tool" | "user",
        "content": "<str>",            # text content / observation
        "tool_calls": [                # optional, for assistant role
          {"name": "<str>", "arguments": {<dict>}, "id": "<str>"}
        ],
        "tool_call_id": "<str>",       # optional, links tool result to call
        "name": "<str>"                # tool name, for role=tool
      }
    ],
    "summary": "<str>",               # target: what the model should learn to produce
    "metadata": {<dict>}               # optional, ignored during training
  }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = (
    "You are an expert at summarizing coding agent trajectories. "
    "Given a task description and the full sequence of agent actions and "
    "tool observations, produce a concise, accurate summary of what the agent "
    "did and what the outcome was. Focus on decisions, key findings, and the "
    "final result. Be specific about files changed, errors encountered, and "
    "solutions applied."
)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolCall:
        return cls(
            name=d["name"],
            arguments=d.get("arguments", {}),
            id=d.get("id", ""),
        )


@dataclass
class TrajectoryStep:
    role: str  # "assistant", "tool", "user"
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str = ""
    name: str = ""  # tool name when role == "tool"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrajectoryStep:
        return cls(
            role=d["role"],
            content=d.get("content", ""),
            tool_calls=[ToolCall.from_dict(tc) for tc in d.get("tool_calls", [])],
            tool_call_id=d.get("tool_call_id", ""),
            name=d.get("name", ""),
        )


@dataclass
class AgentTrajectory:
    task: str
    steps: list[TrajectoryStep]
    summary: str
    trajectory_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AgentTrajectory:
        return cls(
            task=d["task"],
            steps=[TrajectoryStep.from_dict(s) for s in d.get("steps", [])],
            summary=d["summary"],
            trajectory_id=d.get("trajectory_id", ""),
            metadata=d.get("metadata", {}),
        )


def load_trajectories(path: str | Path) -> list[AgentTrajectory]:
    """Load trajectories from a JSONL file (one JSON object per line)."""
    trajectories = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                trajectories.append(AgentTrajectory.from_dict(json.loads(line)))
    return trajectories


def _format_tool_call(tc: ToolCall) -> str:
    args_str = json.dumps(tc.arguments, indent=2) if tc.arguments else "{}"
    return f"<tool_call name={tc.name!r}>\n{args_str}\n</tool_call>"


def _format_step(step: TrajectoryStep, index: int) -> str:
    parts: list[str] = []

    if step.role == "assistant":
        header = f"[Step {index} — assistant]"
        parts.append(header)
        if step.content:
            parts.append(step.content)
        for tc in step.tool_calls:
            parts.append(_format_tool_call(tc))

    elif step.role == "tool":
        tool_label = step.name or "tool"
        header = f"[Step {index} — {tool_label} result]"
        parts.append(header)
        parts.append(step.content)

    else:  # user or other
        parts.append(f"[Step {index} — {step.role}]")
        parts.append(step.content)

    return "\n".join(parts)


def format_trajectory_text(trajectory: AgentTrajectory) -> str:
    """Render a trajectory as a flat text block for the user message."""
    sections: list[str] = []
    sections.append(f"[Task]\n{trajectory.task}")
    sections.append("[Trajectory]")
    for i, step in enumerate(trajectory.steps, start=1):
        sections.append(_format_step(step, i))
    return "\n\n".join(sections)


def trajectory_to_conversation(
    trajectory: AgentTrajectory,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict[str, Any]]:
    """
    Convert an AgentTrajectory to a list[Message] suitable for
    tinker_cookbook.supervised.data.conversation_to_datum.

    The returned conversation has three turns:
      system  → instructions for the summarization task
      user    → the rendered trajectory (task + steps)
      assistant → the target summary (the only tokens trained on)
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": format_trajectory_text(trajectory)},
        {"role": "assistant", "content": trajectory.summary},
    ]
