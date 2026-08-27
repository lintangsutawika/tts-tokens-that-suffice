"""
Agent trajectory data format and conversion utilities.

Each record represents a PARTIAL coding-agent trajectory split at a cutpoint:
  - steps        — the prefix the model sees (history up to the cutpoint)
  - continuation — the suffix after the cutpoint (what the agent did next)
  - summary      — teacher-generated description of what steps contain

Storing both halves lets future reward functions measure whether a summary is
sufficient for the agent to continue without revisiting already-seen context.

The intended data pipeline:
  1. Run a coding agent (e.g. on mini-SWE tasks) and record the full trajectory.
  2. Sample cutpoints along each trajectory.
  3. For each cutpoint, call a teacher model to generate a "progress so far" summary.
  4. Write (steps, continuation, summary) to JSONL using the schema below.

JSONL schema (one JSON object per line):
  {
    "trajectory_id": "<str>",       # optional; useful for grouping cutpoints
    "task": "<str>",                # the original task / problem statement
    "steps": [ <step>, ... ],       # partial trajectory up to the cutpoint
    "continuation": [ <step>, ... ],# remaining steps after the cutpoint (may be [])
    "summary": "<str>",             # teacher-generated summary of steps so far
    "metadata": {<dict>}            # optional; e.g. source repo, cutpoint index
  }

  where each <step> is:
  {
    "role": "assistant" | "tool" | "user",
    "content": "<str>",
    "tool_calls": [{"name": "<str>", "arguments": {<dict>}, "id": "<str>"}],
    "tool_call_id": "<str>",
    "name": "<str>"
  }

Mini-SWE loader:
  Use from_swe_agent_dict() to convert SWE-agent trajectory records (which use
  a flat "history" list of role/content dicts) into AgentTrajectory objects.
  The "task" field is taken from the problem_statement.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

AGENT_SYSTEM_PROMPT = (
    "You are a helpful assistant that can interact with a computer shell to solve programming tasks."
)

SYSTEM_PROMPT = """\
You are maintaining a context-aware state summary for an interactive coding agent.
You will be given a task description followed by a sequence of agent actions and observations.
The agent has not yet finished. Produce a concise summary of what has happened so far.
Respond with plain text only — do not call any tools or emit tool call syntax.

Fill in each section with concrete details from the events. Skip sections with nothing to report.

Example output format:
USER_CONTEXT: Fix the signal decoding bug in cantools where decode_choices=False is ignored.
COMPLETED: Located the bug in conversion.py line 243. Applied a one-line fix. All tests pass.
PENDING: Open a PR with the fix.
CODE_STATE: cantools/database/conversion.py — raw_to_scaled() modified.
TESTS: test_conversion.py passed after fix.
CHANGES: Replaced str(val) with f"{val:.16G}" in raw_to_scaled().
DEPS: None modified.
VERSION_CONTROL_STATUS: Branch fix-float-precision, latest commit a1b2c3d.\
"""

# SYSTEM_PROMPT = """\
# You are maintaining a context-aware state summary for an interactive coding agent.
# You will be given a task description followed by a sequence of agent actions and observations.
# The agent has not yet finished. Produce a concise summary of what has happened so far.
# Respond with plain text only — do not call any tools or emit tool call syntax.

# Fill in each section with concrete details from the events. Skip sections with nothing to report.

# """


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
    # Thinking-model trajectories put the whole turn in reasoning_content and leave
    # content empty, so dropping this loses most of what the agent produced.
    reasoning_content: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrajectoryStep:
        return cls(
            role=d["role"],
            content=d.get("content", "") or "",
            tool_calls=[ToolCall.from_dict(tc) for tc in d.get("tool_calls", [])],
            tool_call_id=d.get("tool_call_id", ""),
            name=d.get("name", ""),
            reasoning_content=d.get("reasoning_content", "") or "",
        )


@dataclass
class AgentTrajectory:
    task: str
    steps: list[TrajectoryStep]          # partial trajectory up to the cutpoint
    continuation: list[TrajectoryStep]   # remaining steps after the cutpoint
    summary: str                         # teacher-generated summary of steps
    trajectory_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    agent_system_prompt: str = ""        # original system prompt from the collected trajectory

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AgentTrajectory:
        return cls(
            task=d["task"],
            steps=[TrajectoryStep.from_dict(s) for s in d.get("steps", [])],
            continuation=[TrajectoryStep.from_dict(s) for s in d.get("continuation", [])],
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
        System messages are skipped (static boilerplate, not part of the trajectory).

        Args:
            d: raw SWE-agent trajectory dict
            summary: teacher-generated summary of steps up to cutpoint
            cutpoint: index into the raw history list; steps before it become
                      `steps`, steps from it onwards become `continuation`.
                      If None, all history goes into steps and continuation=[].
        """
        raw_history: list[dict[str, Any]] = d.get("history", [])

        def _parse_messages(msgs: list[dict[str, Any]]) -> list[TrajectoryStep]:
            result: list[TrajectoryStep] = []
            for msg in msgs:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    continue
                if role == "user" and result and result[-1].role == "assistant":
                    result.append(TrajectoryStep(role="tool", name="bash", content=content))
                else:
                    result.append(TrajectoryStep(role=role, content=content))
            return result

        if cutpoint is not None:
            steps = _parse_messages(raw_history[:cutpoint])
            continuation = _parse_messages(raw_history[cutpoint:])
        else:
            steps = _parse_messages(raw_history)
            continuation = []

        task = d.get("problem_statement", d.get("task", ""))
        instance_id = d.get("instance_id", d.get("trajectory_id", ""))

        return cls(
            task=task,
            steps=steps,
            continuation=continuation,
            summary=summary,
            trajectory_id=instance_id,
            metadata={"source": "swe_agent", "cutpoint": cutpoint},
        )

    def with_prefix(self, n_steps: int) -> AgentTrajectory:
        """Return a view with only the first n_steps of steps; the rest move to continuation."""
        return AgentTrajectory(
            task=self.task,
            steps=self.steps[:n_steps],
            continuation=self.steps[n_steps:] + self.continuation,
            summary=self.summary,
            trajectory_id=self.trajectory_id,
            metadata=self.metadata,
        )

    def sample_split(
        self,
        rng: random.Random,
        min_prefix: int = 3,
        min_suffix: int = 3,
    ) -> AgentTrajectory:
        """Return a copy split at a uniformly random step index."""
        n = len(self.steps) // 2  # steps alternate assistant/tool, count pairs
        lo = min_prefix
        hi = n - min_suffix
        if lo > hi:
            # Trajectory too short — split in the middle as a fallback
            hi = max(lo, n // 2)
        k = rng.randint(lo, hi) * 2  # convert pairs back to step list indices
        return self.with_prefix(k)

    def threshold_split(
        self,
        tokenizer,
        split_at_tokens: int,
        min_prefix: int = 3,
        min_suffix: int = 3,
        max_continuation_tokens: int = 0,
    ) -> AgentTrajectory | None:
        """
        Split at the first turn whose cumulative prefix reaches `split_at_tokens`.

        Mirrors the eval-time compression trigger (eval_swebench's
        compress_at_tokens), so the summarizer trains on the context lengths it
        is actually invoked at. A uniformly random split instead teaches it to
        compress at a length distribution eval never uses.

        max_continuation_tokens caps the continuation (y), keeping whole turns.
        At eval the post-compression context is summary + kept turns, and it must
        stay under the trigger — otherwise compression re-fires immediately and
        the agent loops. Budget it as split_at_tokens - (max summary tokens), and
        the reward then scores a y that fits the same window eval leaves for it.

        Returns None when the trajectory never reaches the threshold, or does so
        with fewer than min_prefix turns before / min_suffix turns after — those
        cannot produce a valid (x, y) pair at this threshold and must be dropped
        rather than silently split somewhere else.
        """
        turn_lens = _turn_token_lens(self.steps, tokenizer)
        n = len(turn_lens)
        total = 0
        for k in range(1, n + 1):
            total += turn_lens[k - 1]
            if total >= split_at_tokens:
                if k < min_prefix or (n - k) < min_suffix:
                    return None
                split = self.with_prefix(k * 2)
                if max_continuation_tokens > 0:
                    split = split._truncate_continuation(tokenizer, max_continuation_tokens)
                    if split is None:
                        return None
                return split
        return None

    def compaction_tokens(self, tokenizer, keep_first: int = 4, keep_last_turns: int = 3) -> int:
        """
        Tokens the verbatim parts of the compacted context cost: head + tail, no summary.

        The compacted context the deliberator sees after compression is
        head(keep_first msgs) + summary + tail(keep_last_turns turns) — see
        build_z_scoring_messages. Head and tail are copied verbatim and are not
        under the summarizer's control, so they set a floor on the compacted
        context: the summary only gets (budget - this) tokens to live in.
        """
        from tts.reward.utils import messages_text
        from tts.summarization.base import last_n_turns

        msgs = steps_to_messages(self.steps, self.task)
        head = msgs[:keep_first]
        tail = last_n_turns(msgs[keep_first:], keep_last_turns)
        text = messages_text(head + tail)
        return len(tokenizer.encode(text)) if text else 0

    def _truncate_continuation(self, tokenizer, max_tokens: int) -> AgentTrajectory | None:
        """
        Drop whole turns off the end of the continuation until it fits max_tokens.

        Returns None if even the first continuation turn overflows the budget — a
        y that cannot be scored within the window eval leaves for it.
        """
        cont = self.continuation
        turn_lens = _turn_token_lens(cont, tokenizer)
        total, kept = 0, 0
        for i, tl in enumerate(turn_lens):
            total += tl
            if total > max_tokens:
                break
            kept = (i + 1) * 2
        if kept == 0:
            return None
        if kept == len(cont):
            return self
        return AgentTrajectory(
            task=self.task,
            steps=self.steps,
            continuation=cont[:kept],
            summary=self.summary,
            trajectory_id=self.trajectory_id,
            metadata=self.metadata,
        )

    @classmethod
    def from_collect_dict(cls, d: dict[str, Any]) -> AgentTrajectory:
        """
        Load from a collect_trajectories.py output record.

        Schema: {uid, instance_id, is_correct, num_calls, messages}
        where messages is the raw OpenAI-style list including system, user,
        alternating assistant (with tool_calls) / tool pairs, and a final exit.
        """
        messages = d.get("messages", [])
        task = next((m["content"] for m in messages if m.get("role") == "user"), "")
        agent_system_prompt = next((m["content"] for m in messages if m.get("role") == "system"), "")

        steps: list[TrajectoryStep] = []
        i = 0
        while i < len(messages):
            m = messages[i]
            role = m.get("role", "")
            if role == "assistant" and m.get("tool_calls"):
                tool_calls = []
                for tc in m.get("tool_calls", []):
                    args = tc.get("function", {}).get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except (json.JSONDecodeError, ValueError):
                            args = {}
                    tool_calls.append(ToolCall(
                        name=tc.get("function", {}).get("name", ""),
                        arguments=args,
                        id=tc.get("id", ""),
                    ))
                steps.append(TrajectoryStep(
                    role="assistant",
                    content=m.get("content") or "",
                    tool_calls=tool_calls,
                    reasoning_content=m.get("reasoning_content") or "",
                ))
                i += 1
                if i < len(messages) and messages[i].get("role") == "tool":
                    t = messages[i]
                    steps.append(TrajectoryStep(
                        role="tool",
                        content=t.get("content") or "",
                        tool_call_id=t.get("tool_call_id", ""),
                    ))
                    i += 1
            else:
                i += 1

        return cls(
            task=task,
            steps=steps,
            continuation=[],
            summary="",
            trajectory_id=d.get("uid", ""),
            metadata={
                "instance_id": d.get("instance_id"),
                "is_correct": d.get("is_correct"),
                "num_calls": d.get("num_calls"),
            },
            agent_system_prompt=agent_system_prompt,
        )


def _turn_token_lens(steps: list[TrajectoryStep], tokenizer) -> list[int]:
    """
    Token length of each (assistant, tool) turn, as one batched tokenizer call.

    Encoding turn-by-turn in a Python loop costs one round-trip per turn — ~40 per
    trajectory, ~230k across the dataset — and never lets the Rust tokenizer
    parallelize. Batching hands it every turn at once instead.
    """
    if not steps:
        return []
    texts = [format_trajectory_text(steps[i : i + 2]) for i in range(0, len(steps) - 1, 2)]
    if not texts:
        return []
    encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
    return [len(ids) for ids in encoded]


def parse_tool_calls(raw_tool_calls: list) -> list[ToolCall]:
    """Parse OpenAI-style assistant tool_calls into ToolCall (as in from_collect_dict)."""
    tool_calls: list[ToolCall] = []
    for tc in raw_tool_calls or []:
        fn = tc.get("function", {})
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                args = {}
        tool_calls.append(ToolCall(name=fn.get("name", ""), arguments=args, id=tc.get("id", "")))
    return tool_calls


def messages_to_steps(messages: list[dict]) -> list[TrajectoryStep]:
    """Convert live agent messages to TrajectorySteps for the summarizer input.

    Skips the system message and the initial task user message (both are always
    kept verbatim via keep_first, matching training, where the summarizer never
    sees the raw task).  Later user messages — notably a previous <summary> — ARE
    kept as events so recursive summarization does not lose already-compressed
    context.
    """
    steps: list[TrajectoryStep] = []
    seen_task = False
    for m in messages:
        role = m.get("role", "")
        if role == "system":
            continue
        if role == "user":
            if not seen_task:
                seen_task = True  # the task/problem statement; preserved via keep_first
                continue
            steps.append(TrajectoryStep(role="user", content=m.get("content") or ""))
        elif role == "assistant":
            steps.append(TrajectoryStep(
                role="assistant",
                content=m.get("content") or "",
                tool_calls=parse_tool_calls(m.get("tool_calls") or []),
            ))
        elif role == "tool":
            steps.append(TrajectoryStep(
                role="tool",
                content=m.get("content") or "",
                tool_call_id=m.get("tool_call_id", ""),
            ))
        # exit / other roles: ignore
    return steps


def steps_to_messages(
    steps: list[TrajectoryStep],
    task: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict[str, Any]]:
    """
    Convert TrajectoryStep list to a list[dict] message format compatible
    with tts.reward.utils (which expects standard OpenAI-style message dicts).
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]
    for step in steps:
        if step.role == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": step.content}
            if step.reasoning_content:
                msg["reasoning_content"] = step.reasoning_content
            if step.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id or f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments),
                        },
                    }
                    for i, tc in enumerate(step.tool_calls)
                ]
            messages.append(msg)
        elif step.role == "tool":
            messages.append({"role": "tool", "content": step.content})
        else:
            messages.append({"role": step.role, "content": step.content})
    return messages


def load_trajectories(path: str | Path) -> list[AgentTrajectory]:
    """Load AgentTrajectory records from a JSONL file."""
    trajectories = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                trajectories.append(AgentTrajectory.from_dict(json.loads(line)))
    return trajectories


def load_collect_trajectories(path: str | Path) -> list[AgentTrajectory]:
    """Load full (unsplit) trajectories from collect_trajectories.py output."""
    from tqdm import tqdm

    trajectories = []
    total = Path(path).stat().st_size
    with open(path) as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=f"reading {Path(path).name}"
    ) as bar:
        for line in f:
            bar.update(len(line))
            line = line.strip()
            if line:
                trajectories.append(AgentTrajectory.from_collect_dict(json.loads(line)))
    return trajectories


def _format_tool_call(tc: ToolCall) -> str:
    args_str = json.dumps(tc.arguments, indent=2) if tc.arguments else "{}"
    return f"Action: {tc.name}\n{args_str}"


def _format_step(step: TrajectoryStep) -> str:
    parts: list[str] = []

    if step.role == "assistant":
        if step.content:
            parts.append(step.content)
        for tc in step.tool_calls:
            parts.append(_format_tool_call(tc))
        label = "assistant"

    elif step.role == "tool":
        label = f"{step.name} observation" if step.name else "observation"
        parts.append(step.content)

    else:
        label = step.role
        parts.append(step.content)

    body = "\n".join(parts)
    return f"<EVENT type={label!r}>\n{body}\n</EVENT>"


def format_trajectory_text(
    steps: list[TrajectoryStep],
    start_idx: int = 0,
    end_idx: int | None = None,
) -> str:
    """Render a slice of a trajectory as EVENT blocks for the summarizer user message."""
    sections: list[str] = []
    for step in steps[start_idx:end_idx]:
        sections.append(_format_step(step))
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
        {"role": "user", "content": format_trajectory_text(trajectory.steps)},
        {"role": "assistant", "content": trajectory.summary},
    ]
