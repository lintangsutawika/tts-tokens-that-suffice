"""
tts.utils.logprob — per-token distortion scoring via a vLLM-compatible server.

The per-token distortion is the realized-token logprob difference

    d_t = log p(y_t | y<t, x) − log p(y_t | y<t, z)

i.e. how much less likely the *actual* continuation token y_t becomes when the
full context x is compressed to the summary z. This is a single-sample estimate
of the forward KL D_KL(p(·|x) ‖ p(·|z)); unlike a top-k-truncated KL it keeps the
high-divergence tokens (where bad summaries should be penalized most), giving a
much larger dynamic range across summaries.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import litellm

for _name in ("litellm", "LiteLLM", "litellm.utils", "litellm.proxy"):
    logging.getLogger(_name).setLevel(logging.WARNING)


_THINK_OPEN = "<think>\n"
_THINK_CLOSED = "<think>\n\n</think>\n\n"


def _text_completion(model, base, prompt, max_tokens, logprobs, echo):
    return litellm.text_completion(
        model=model, base_url=base, api_key="dummy",
        prompt=prompt, max_tokens=max_tokens, logprobs=logprobs, echo=echo,
    )


def _completion_opens_in_think(completion: str) -> bool:
    """
    True if `completion` begins *inside* a think block.

    apply_chat_template ends a generation prompt with an open "<think>\\n", so a
    continuation carrying reasoning_content starts with the reasoning text and
    closes with "</think>" before its tool call. Such a completion must be
    concatenated onto the still-open block, not onto a closed one.
    """
    close = completion.find("</think>")
    if close == -1:
        return False
    open_ = completion.find("<think>")
    return open_ == -1 or close < open_


def _close_think_block(text: str, completion: str = "") -> str:
    """
    Replace a trailing open <think> with an empty closed block.

    This aligns logprobs for thinking-*disabled* continuations, which begin at
    the assistant's visible content. When the completion supplies its own
    reasoning, closing the block here would push that reasoning outside <think>
    and leave its "</think>" unmatched, so leave the block open.
    """
    if text.endswith(_THINK_OPEN) and not _completion_opens_in_think(completion):
        return text[: -len(_THINK_OPEN)] + _THINK_CLOSED
    return text


@dataclass
class XLogprobs:
    n_x_ctx: int
    n_completion: int
    x_tok: list  # realized-token logprob log p(y_t|y<t,x), one per completion token


def precompute_x(
    x_messages: list[dict],
    completion: str,
    model: str,
    api_base: str,
    tokenizer,
    k: int = 0,
    tools: list | None = None,
) -> XLogprobs | None:
    """
    Pre-compute x-context logprobs for a fixed (trajectory, continuation) pair.

    k is the number of top-k alternatives requested per position; only
    token_logprobs (the realized token) is read, so k=0 is correct. It must not
    be None — that suppresses token_logprobs too.

    Makes 2 API calls (context length + full scoring). The result can be
    reused across all group_size summaries for the same trajectory, avoiding
    redundant x-side scoring.
    """
    if not completion.strip():
        return XLogprobs(n_x_ctx=0, n_completion=0, x_tok=[])

    base = api_base.rstrip("/")
    vllm_model = model.replace("litellm_proxy/", "hosted_vllm/")
    x_text = _close_think_block(tokenizer.apply_chat_template(x_messages, tokenize=False, add_generation_prompt=True, tools=tools), completion)

    n_x_ctx = len(tokenizer.encode(x_text))

    try:
        x_resp = _text_completion(vllm_model, base, x_text + completion, 1, k, True)
        n_x_full = x_resp["usage"]["prompt_tokens"]
        n_completion = n_x_full - n_x_ctx
        x_tok = x_resp["choices"][0]["logprobs"].token_logprobs[n_x_ctx:n_x_full]
    except Exception as exc:
        print(f"    [score] x-context scoring failed: {exc}")
        return None

    return XLogprobs(n_x_ctx=n_x_ctx, n_completion=n_completion, x_tok=x_tok)


def score_completion_z(
    x_logprobs: XLogprobs,
    z_messages: list[dict],
    completion: str,
    model: str,
    api_base: str,
    tokenizer,
    k: int = 0,
    tools: list | None = None,
) -> list[float] | None:
    """
    Score the z-context side and compute the per-token realized-token distortion
    d_t = log p_x(y_t) − log p_z(y_t) against pre-computed x logprobs.

    Makes 2 API calls (z context length + full z scoring).
    """
    if not completion.strip() or x_logprobs.n_completion == 0:
        return []

    base = api_base.rstrip("/")
    vllm_model = model.replace("litellm_proxy/", "hosted_vllm/")
    z_text = _close_think_block(tokenizer.apply_chat_template(z_messages, tokenize=False, add_generation_prompt=True, tools=tools), completion)

    n_z_ctx = len(tokenizer.encode(z_text))

    try:
        z_resp = _text_completion(vllm_model, base, z_text + completion, 1, k, True)
        z_tok = z_resp["choices"][0]["logprobs"].token_logprobs[
            n_z_ctx : n_z_ctx + x_logprobs.n_completion
        ]
    except Exception as exc:
        print(f"    [score] z-context scoring failed: {exc}")
        return None

    x_tok = x_logprobs.x_tok
    n = min(len(x_tok), len(z_tok))
    return [
        x_tok[t] - z_tok[t]
        for t in range(n)
        if x_tok[t] is not None and z_tok[t] is not None
    ]


def score_completion(
    x_messages: list[dict],
    z_messages: list[dict],
    completion: str,
    model: str,
    api_base: str,
    tokenizer,
    k: int = 0,
) -> list[float] | None:
    """
    Per-position realized-token distortion d_t = log p_x(y_t) − log p_z(y_t).

    At each completion position t, reads the logprob of the *actual* continuation
    token y_t under the x-context and the z-context and returns their difference.
    This is a single-sample estimate of the forward KL D_KL(p(·|x) ‖ p(·|z)).

    Context texts are produced via tokenizer.apply_chat_template so the
    tokenization matches exactly what the model sees during inference.

    2 parallel full-prompt calls (echo + logprobs) for the realized-token logprobs.

    Returns one distortion float per completion token, or None on error.
    """
    if not completion.strip():
        return []

    base = api_base.rstrip("/")
    vllm_model = model.replace("litellm_proxy/", "hosted_vllm/")
    x_text = _close_think_block(tokenizer.apply_chat_template(x_messages, tokenize=False, add_generation_prompt=True), completion)
    z_text = _close_think_block(tokenizer.apply_chat_template(z_messages, tokenize=False, add_generation_prompt=True), completion)

    n_x_ctx = len(tokenizer.encode(x_text))
    n_z_ctx = len(tokenizer.encode(z_text))

    # Full-prompt realized-token scoring (parallel)
    try:
        with ThreadPoolExecutor(max_workers=2) as ex:
            fx = ex.submit(_text_completion, vllm_model, base, x_text + completion, 1, k, True)
            fz = ex.submit(_text_completion, vllm_model, base, z_text + completion, 1, k, True)
            x_resp = fx.result()
            z_resp = fz.result()
        n_x_full = x_resp["usage"]["prompt_tokens"]
        n_completion = n_x_full - n_x_ctx
        x_tok = x_resp["choices"][0]["logprobs"].token_logprobs[n_x_ctx:n_x_full]
        z_tok = z_resp["choices"][0]["logprobs"].token_logprobs[n_z_ctx:n_z_ctx + n_completion]
    except Exception as exc:
        print(f"    [score] Full-prompt scoring failed: {exc}")
        return None

    # per-position realized-token distortion: log p_x(y_t) − log p_z(y_t)
    n = min(len(x_tok), len(z_tok))
    return [
        x_tok[t] - z_tok[t]
        for t in range(n)
        if x_tok[t] is not None and z_tok[t] is not None
    ]
