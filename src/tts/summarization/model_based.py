"""
Model-based compaction: replace the middle of the history with a generated summary.

This is the strategy under study — the summarizer policy trained by train_rl.py.
Two backends produce z from the same prompt:

  * TinkerSummarizer  — a tinker SamplingClient (base model or trained LoRA),
                        used during training and by --mode base/trained
  * LitellmSummarizer — an OpenAI-compatible vLLM endpoint
                        (scripts/serve_summarizer.sh)

Both are greedy so a given (trajectory, checkpoint) pair yields a fixed z.
See mask_based.py for the summarizer-free control.
"""

from __future__ import annotations

import os

import litellm

from tts.data.agent_trajectory import (
    SYSTEM_PROMPT,
    TrajectoryStep,
    format_trajectory_text,
    messages_to_steps,
)

from .base import CompactionResult, split_head_tail


def summarizer_messages(steps: list[TrajectoryStep]) -> list[dict]:
    """The prompt used to generate z (identical to training-time generation)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_trajectory_text(steps)},
    ]


def hf_id(model: str) -> str:
    """Strip a leading litellm provider prefix so the string is a plain HF repo id."""
    for prefix in ("openai/", "litellm_proxy/", "hosted_vllm/"):
        if model.startswith(prefix):
            return model[len(prefix):]
    return model


def strip_thinking(text: str) -> str:
    """Return z: the summary after the reasoning block, if any.

    A thinking-enabled summarizer emits <think>...</think> then the summary; z
    is only that summary, so the deliberator never sees the summarizer's chain
    of thought. Safe to apply unconditionally: with thinking disabled there is
    no </think>, so the text is returned unchanged (and any pre-filled empty
    think block the template left in is cleaned up either way).
    """
    _head, sep, tail = text.rpartition("</think>")
    return tail.lstrip("\n") if sep else text


RECALL_TOOL_NAME = "recall_context"


def summary_as_tool_turn(
    summary: str, tool_name: str = RECALL_TOOL_NAME, call_id: str = "summary_0"
) -> list[dict]:
    """
    Render z as a tool call the agent made and the result it got back.

    Not cosmetic. The chat template strips reasoning from every assistant turn
    *before the last genuine user query*, and it identifies those by scanning
    backwards for a `user` message that is not <tool_response>-wrapped. A
    summary inserted as `user` therefore became that boundary and silently
    deleted the head turns' thinking — in the summary arm only, so the arms were
    no longer identical apart from the middle.

    A tool result is exempt from that scan, so head reasoning survives and every
    strategy renders alike. It also reads correctly: z is something the agent
    *observed about its own state*, not a new instruction. Costs ~27 tokens of
    call scaffolding over a bare message.

    The paired assistant tool_call is required, not decoration — a tool message
    with no matching call breaks the alternation the chat API validates.
    """
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                # Wire format: the OpenAI schema types `arguments` as a string and
                # the chat API rejects a dict. The Jinja template wants the
                # opposite, so rendering paths run this through
                # tts.reward.utils.to_template_tool_calls first.
                "function": {"name": tool_name, "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": call_id, "content": summary},
    ]


def build_z_scoring_messages(
    summary: str,
    partial_messages: list[dict],
    max_size: int = 20,
    keep_first: int = 4,
) -> list[dict]:
    """
    z-context: [first keep_first msgs verbatim, z as a tool turn, tail turns verbatim].

    max_size:   target total message budget for the z-context (before summary insertion).
                Tail size = max_size // 2 - keep_first messages = that many // 2 turns.
    keep_first: messages preserved verbatim from the start (sys, user task, first turn(s)).
    """
    keep_turns = (max_size // 2 - keep_first) // 2
    head, _middle, tail = split_head_tail(partial_messages, keep_first, keep_turns)
    return head + summary_as_tool_turn(summary) + tail


class Summarizer:
    """Interface: .summarize(steps) -> z. Two backends below."""

    def summarize(self, steps: list[TrajectoryStep]) -> str:
        raise NotImplementedError


class TinkerSummarizer(Summarizer):
    """Summarize via a tinker SamplingClient (base model or a trained LoRA checkpoint).

    Mirrors the training-time generation: system=SYSTEM_PROMPT,
    user=format_trajectory_text(steps), greedy decode.
    """

    def __init__(self, sampling_client, renderer, max_tokens: int = 512):
        self.sampling_client = sampling_client
        self.renderer = renderer
        self.max_tokens = max_tokens

    def summarize(self, steps: list[TrajectoryStep]) -> str:
        from tinker import types
        from tinker_cookbook import renderers

        model_input = self.renderer.build_generation_prompt(summarizer_messages(steps))
        result = self.sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=types.SamplingParams(
                max_tokens=self.max_tokens,
                temperature=0.0,
                stop=self.renderer.get_stop_sequences(),
            ),
        ).result()
        parsed_message, _ = self.renderer.parse_response(result.sequences[0].tokens)
        return strip_thinking(renderers.get_text_content(parsed_message))


def tinker_summarize(
    messages: list[dict],
    system_prompt: str,
    sampling_client,
    renderer,
    max_tokens: int = 2048,
) -> str:
    """Generate z from `messages` under an arbitrary system prompt via a tinker
    SamplingClient (base model or a trained LoRA), greedy.

    Signature mirrors compare_summary_prompts.summarize (the litellm path) so the
    same prompt-sweep loop can drive either backend; the difference is only that
    this samples through the tinker server, with thinking enabled and stripped
    (see strip_thinking). Note the sweep only makes sense for a *base* model: a
    checkpoint trained under one fixed prompt is off-distribution under the others.
    """
    from tinker import types
    from tinker_cookbook import renderers

    prompt = renderer.build_generation_prompt([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": format_trajectory_text(messages_to_steps(messages))},
    ])
    result = sampling_client.sample(
        prompt=prompt,
        num_samples=1,
        sampling_params=types.SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            stop=renderer.get_stop_sequences(),
        ),
    ).result()
    parsed_message, _ = renderer.parse_response(result.sequences[0].tokens)
    return strip_thinking(renderers.get_text_content(parsed_message))


class LitellmSummarizer(Summarizer):
    """Summarize via an OpenAI-compatible vLLM endpoint (scripts/serve_summarizer.sh).

    `model` is the litellm model string (e.g. "openai/Qwen/Qwen3-8B" for base, or
    "openai/<lora-name>" for a served trained adapter). Greedy, thinking disabled
    to match the qwen3_disable_thinking renderer used in training.
    """

    def __init__(self, model: str, api_base: str, max_tokens: int = 512, api_key: str = "EMPTY"):
        self.model = model
        self.api_base = api_base
        self.max_tokens = max_tokens
        # litellm's openai/ provider requires a key even though vLLM ignores its value.
        self.api_key = api_key or "EMPTY"

    def summarize(self, steps: list[TrajectoryStep]) -> str:
        resp = litellm.completion(
            model=self.model,
            messages=summarizer_messages(steps),
            api_base=self.api_base,
            api_key=self.api_key,
            temperature=0.0,
            max_tokens=self.max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return resp.choices[0].message.content or ""


class ModelBasedSummarizer:
    """
    Adapts a Summarizer backend to the Compactor interface.

    Falls back to plain truncation when generation fails, so a summarizer
    outage degrades the run rather than killing it — the result's `kind`
    records which happened.
    """

    def __init__(self, summarizer: Summarizer, max_size: int = 20):
        self.summarizer = summarizer
        self.max_size = max_size

    def compact(
        self,
        messages: list[dict],
        keep_first: int = 4,
        keep_last_turns: int = 3,
    ) -> CompactionResult:
        head, _middle, tail = split_head_tail(messages, keep_first, keep_last_turns)
        try:
            z = self.summarizer.summarize(messages_to_steps(messages))
        except Exception as exc:
            return CompactionResult(
                messages=head + tail,
                kind="summary_failed",
                summary=None,
                metadata={"error": str(exc)},
            )
        return CompactionResult(
            messages=head + summary_as_tool_turn(z) + tail,
            kind="summary",
            summary=z,
        )


def build_summarizer(
    summarizer_model: str,
    renderer_name: str,
    max_tokens: int,
    tinker_base_url: str | None = None,
    checkpoint: str = "",
    summarizer_api_base: str = "",
) -> Summarizer:
    """Construct the summarizer backend that generates z.

    The caller has already decided that this arm generates, so there is no
    `mode` here: `checkpoint` is what actually distinguishes the two tinker
    paths — a trained LoRA is loaded from its state, otherwise the base model is
    sampled directly. Passing a checkpoint therefore always uses it, rather than
    being silently ignored when it disagreed with a mode string.

    If `summarizer_api_base` is set, the summarizer is served over an
    OpenAI-compatible vLLM endpoint (litellm backend); otherwise it is sampled
    through the tinker server (the default).
    """
    # vLLM / litellm backend (scripts/serve_summarizer.sh).
    if summarizer_api_base:
        api_key = os.getenv("SUMMARIZER_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
        return LitellmSummarizer(
            summarizer_model, summarizer_api_base, max_tokens=max_tokens, api_key=api_key
        )

    # tinker backend (default).
    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from tinker_cookbook.utils.git_rev import recipe_user_metadata

    tokenizer = get_tokenizer(hf_id(summarizer_model))
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    service_client = tinker.ServiceClient(
        base_url=tinker_base_url,
        user_metadata=recipe_user_metadata("eval_swebench"),
    )

    if checkpoint:
        # A TRAINING checkpoint (tinker://.../weights/NNNNNN) can only be loaded
        # via a training client — the server's create_sampling_client path
        # validates model_path as a SAMPLER checkpoint (api.py get_sampling_model),
        # so it 404s on training weights. The training-client load also pulls the
        # distributed optimizer, which is sharded to the checkpoint's *training*
        # TP×DP layout and does NOT reshard — so the serving server MUST run the
        # same TP×DP the checkpoint was trained at, or this fails with
        # "Missing key ... optimizer.distributed.dp_group_idx_N".
        tc = service_client.create_training_client_from_state(checkpoint)
        sampling_client = tc.save_weights_and_get_sampling_client()
    else:
        sampling_client = service_client.create_sampling_client(
            base_model=hf_id(summarizer_model)
        )

    return TinkerSummarizer(sampling_client, renderer, max_tokens=max_tokens)
