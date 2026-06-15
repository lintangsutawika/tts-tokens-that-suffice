"""
Agent trajectory data format and conversion utilities.

Each record represents a PARTIAL coding-agent trajectory — a prefix of actions
and observations captured at some point mid-task — paired with a teacher-generated
summary of what the agent has done so far.

The intended data pipeline:
  1. Run a coding agent (e.g. on mini-SWE tasks) and record the full trajectory.
  2. Sample cutpoints along each trajectory.
  3. For each (trajectory[:cutpoint], task) pair, call a teacher model to generate
     a "progress so far" summary.
  4. Write (partial_steps, summary) to JSONL using the schema below.

This recipe then trains a student model to reproduce those summaries given only
the partial trajectory — enabling context compression for long-horizon agents.

JSONL schema (one JSON object per line):
  {
    "trajectory_id": "<str>",       # optional; useful for grouping cutpoints
    "task": "<str>",                # the original task / problem statement
    "steps": [                      # partial trajectory up to the sample cutpoint
      {
        "role": "assistant" | "tool" | "user",
        "content": "<str>",
        "tool_calls": [             # optional; present when role == "assistant"
          {"name": "<str>", "arguments": {<dict>}, "id": "<str>"}
        ],
        "tool_call_id": "<str>",    # links a tool result back to its call
        "name": "<str>"             # tool name; present when role == "tool"
      },
      ...
    ],
    "summary": "<str>",            # teacher-generated summary of steps so far
    "metadata": {<dict>}            # optional; e.g. source repo, cutpoint index
  }

Mini-SWE loader:
  Use from_swe_agent_dict() to convert SWE-agent trajectory records (which use
  a flat "history" list of role/content dicts) into AgentTrajectory objects.
  The "task" field is taken from the problem_statement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = (
    "You are an expert at tracking and summarizing the progress of coding agents. "
    "You will be shown a task description followed by a partial sequence of agent "
    "actions and tool observations — the agent has not yet finished. "
    "Produce a concise summary of what the agent has done so far: "
    "which files it has read or modified, what it has discovered, "
    "what approaches it has tried, and what state it has left things in. "
    "Do not speculate about what it will do next."
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
    steps: list[TrajectoryStep]  # partial trajectory; may not be complete
    summary: str                 # teacher-generated summary of these steps
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

    @classmethod
    def from_swe_agent_dict(
        cls,
        d: dict[str, Any],
        summary: str,
        cutpoint: int | None = None,
    ) -> AgentTrajectory:
        """
        Convert a SWE-agent trajectory record to AgentTrajectory.

        SWE-agent records typically have:
          - instance_id: str
          - problem_statement: str
          - history: list of {"role": str, "content": str} dicts
            (roles are "system", "user", "assistant")

        The bash tool results appear as "user" messages in SWE-agent's format;
        we re-label them as role="tool" with name="bash" for clarity.

        Args:
            d: raw SWE-agent trajectory dict
            summary: teacher-generated summary to attach
            cutpoint: if given, only include history[:cutpoint]; otherwise all
        """
        history: list[dict[str, Any]] = d.get("history", [])
        if cutpoint is not None:
            history = history[:cutpoint]

        steps: list[TrajectoryStep] = []
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # Skip system prompt — it's static and doesn't belong in the trajectory
            if role == "system":
                continue
            # In SWE-agent, the environment responses come back as "user" messages
            # that follow an assistant action; re-label them as tool observations.
            if role == "user" and steps and steps[-1].role == "assistant":
                steps.append(TrajectoryStep(role="tool", name="bash", content=content))
            else:
                steps.append(TrajectoryStep(role=role, content=content))

        task = d.get("problem_statement", d.get("task", ""))
        instance_id = d.get("instance_id", d.get("trajectory_id", ""))

        return cls(
            task=task,
            steps=steps,
            summary=summary,
            trajectory_id=instance_id,
            metadata={"source": "swe_agent", "cutpoint": cutpoint},
        )

    def with_prefix(self, n_steps: int) -> AgentTrajectory:
        """Return a new trajectory with only the first n_steps steps."""
        return AgentTrajectory(
            task=self.task,
            steps=self.steps[:n_steps],
            summary=self.summary,
            trajectory_id=self.trajectory_id,
            metadata=self.metadata,
        )


def load_trajectories(path: str | Path) -> list[AgentTrajectory]:
    """Load AgentTrajectory records from a JSONL file."""
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
        parts.append(f"[Step {index} — assistant]")
        if step.content:
            parts.append(step.content)
        for tc in step.tool_calls:
            parts.append(_format_tool_call(tc))

    elif step.role == "tool":
        tool_label = step.name or "tool"
        parts.append(f"[Step {index} — {tool_label} result]")
        parts.append(step.content)

    else:
        parts.append(f"[Step {index} — {step.role}]")
        parts.append(step.content)

    return "\n".join(parts)


def format_trajectory_text(steps: list[TrajectoryStep], task: str) -> str:
    """Render a partial trajectory as a flat text block for the user message."""
    sections: list[str] = [
        f"[Task]\n{task}",
        f"[Partial Trajectory — {len(steps)} step(s) so far]",
    ]
    for i, step in enumerate(steps, start=1):
        sections.append(_format_step(step, i))
    return "\n\n".join(sections)


def trajectory_to_conversation(
    trajectory: AgentTrajectory,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict[str, Any]]:
    """
    Convert an AgentTrajectory to a list[Message] for tinker's conversation_to_datum.

    Three turns:
      system    → summarization instructions
      user      → rendered partial trajectory (task + steps so far)
      assistant → the target summary (only these tokens receive loss weight)
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": format_trajectory_text(trajectory.steps, trajectory.task)},
        {"role": "assistant", "content": trajectory.summary},
    ]
