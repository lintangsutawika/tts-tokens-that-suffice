"""
tts.reward.utils — pieces shared across reward components.

The distortion computation lives in distortion_reward.py and the anti-copy
penalty in copy_penalty.py; what remains here is what both (and, for
messages_text, tts.data) reach for — chiefly recovering y, the exact token
stream an agent turn was generated as.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass


def messages_text(messages: list[dict]) -> str:
    """Flatten message contents to plain text, for overlap/length comparison against z."""
    return "\n".join(
        str(m.get("content") or "") for m in messages if m.get("role") != "system"
    )


def to_template_tool_calls(messages: list[dict]) -> list[dict]:
    """
    Parse tool_call arguments from JSON strings into dicts.

    Trajectories store the wire format (a string, as model_dump produces) but
    the Jinja template calls .items() on the same field and raises on a string.
    Anything headed for apply_chat_template needs this first.
    """
    out = copy.deepcopy(messages)
    for m in out:
        for tc in m.get("tool_calls") or []:
            args = tc.get("function", {}).get("arguments")
            if isinstance(args, str):
                try:
                    tc["function"]["arguments"] = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    tc["function"]["arguments"] = {}
    return out


def strip_reasoning(messages: list[dict]) -> list[dict]:
    """Drop `reasoning_content` from every message in a *context*.

    Inference goes through /v1/chat/completions, whose OpenAI-schema validation
    has no reasoning_content field, so prior-turn reasoning is dropped before the
    chat template runs (see tests/tokenization/test_vllm_render.py). Fidelity
    renders the context locally and posts text to /v1/completions, which would
    otherwise keep that reasoning — scoring a context the model never actually
    sees. Stripping here makes the scored prompt byte-identical to what inference
    tokenizes.

    Apply to the context only. The scored completion y keeps its reasoning: the
    model generates that live in the same turn, so it is part of the next action,
    not prior-turn history.
    """
    out = copy.deepcopy(messages)
    for m in out:
        m.pop("reasoning_content", None)
    return out


@dataclass
class Generation:
    """The token stream an assistant message was produced as."""

    text: str
    token_ids: list[int]
    recorded_n_tokens: int | None = None  # usage.completion_tokens, when recorded

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def verified(self) -> bool | None:
        """
        True when the reconstruction matches the recorded token count.

        None when the trajectory recorded no usage to check against. Note this
        verifies the *count*, not the ids — nothing in a standard trajectory
        records ids, so identity cannot be proven, only strongly evidenced.
        """
        if self.recorded_n_tokens is None:
            return None
        return self.n_tokens == self.recorded_n_tokens


def reconstruct_generation(
    context_messages: list[dict],
    assistant_message: dict,
    tokenizer,
    tools: list | None = None,
) -> Generation:
    """
    Recover the exact token stream an assistant turn was generated as.

    Servers rarely persist completion token ids (vLLM leaves prompt_token_ids
    and logprobs null unless asked), so the generation is re-derived: render the
    conversation with and without the turn and take the delta past the
    generation prompt.

    The template appends a '\\n' separator after every message, including the
    last. The model never generates it — it emits <|im_end|> (the eos token) and
    stops — so it is stripped. Keeping it would add one token of pure template
    artifact to every scored continuation.

    Verified count-exact on 514/514 assistant turns across 8 SWE-bench-lite
    Qwen3.6-35B-A3B trajectories; `Generation.verified` re-checks per call.
    """
    ctx = to_template_tool_calls(context_messages)
    act = to_template_tool_calls([assistant_message])
    kw = {"tokenize": False}
    if tools:
        kw["tools"] = tools

    full = tokenizer.apply_chat_template(ctx + act, add_generation_prompt=False, **kw)
    prompt = tokenizer.apply_chat_template(ctx, add_generation_prompt=True, **kw)
    if not full.startswith(prompt):
        raise ValueError(
            "generation prompt is not a prefix of the full render; "
            "context and assistant message may not be adjacent turns"
        )
    text = full[len(prompt):].rstrip("\n")

    usage = (assistant_message.get("extra") or {}).get("response", {}).get("usage") or {}
    return Generation(
        text=text,
        token_ids=tokenizer.encode(text),
        recorded_n_tokens=usage.get("completion_tokens"),
    )
