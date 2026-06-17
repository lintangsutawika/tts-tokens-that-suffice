"""
tts.utils.logprob — per-token log-probability scoring via a vLLM-compatible server.

Primary entry point: score_completion(context_messages, completion, model, api_base)
"""

from __future__ import annotations

import litellm


def score_completion(
    context_messages: list[dict],
    completion: str,
    model: str,
    api_base: str,
) -> list[float] | None:
    """
    Compute per-token log p(completion | context_messages).

    Uses echo=True which populates logprobs.token_logprobs with exactly one
    float per actual prompt token — no top-k ambiguity. The slice
    [n_ctx : full_prompt_tokens] isolates the completion tokens, excluding
    the one generated token that echo=True appends at the end.

    Returns one float per completion token, or None on error.
    """
    if not completion.strip():
        return []

    context_text = _messages_to_text(context_messages)
    full_prompt = context_text + completion
    base = api_base.rstrip("/")
    vllm_model = model.replace("litellm_proxy/", "hosted_vllm/")

    try:
        ctx_resp = litellm.text_completion(
            model=vllm_model,
            base_url=base,
            api_key="dummy",
            prompt=context_text,
            max_tokens=1,
            logprobs=1,
            echo=True,
            extra_body={"prompt_logprobs": 1},
        )
        n_ctx = ctx_resp["usage"]["prompt_tokens"]
    except Exception as exc:
        print(f"    [score] Context tokenisation failed: {exc}")
        return None

    try:
        full_resp = litellm.text_completion(
            model=vllm_model,
            base_url=base,
            api_key="dummy",
            prompt=full_prompt,
            max_tokens=1,
            logprobs=1,
            echo=True,
            extra_body={"prompt_logprobs": 1},
        )
        # logprobs is returned as a Logprobs object — use attribute access.
        # token_logprobs[i] = log p(token_i | token_0..i-1); position 0 is None.
        # With echo=True and max_tokens=1, the list is:
        #   [None, lp_1, ..., lp_{n_prompt-1}, lp_generated]
        # usage.prompt_tokens gives n_prompt, so we slice [n_ctx:n_prompt]
        # to get exactly the completion token logprobs without the generated token.
        logprobs_obj = full_resp["choices"][0]["logprobs"]
        token_logprobs = logprobs_obj.token_logprobs
        n_full = full_resp["usage"]["prompt_tokens"]
    except Exception as exc:
        print(f"    [score] Full-prompt scoring failed: {exc}")
        return None

    return [lp for lp in token_logprobs[n_ctx:n_full] if lp is not None]


def inspect_score_completion(
    context_messages: list[dict],
    completion: str,
    model: str,
    api_base: str,
) -> None:
    """
    Diagnostic helper: print token structure to verify KL alignment.
    """
    context_text = _messages_to_text(context_messages)
    full_prompt = context_text + completion
    base = api_base.rstrip("/")
    vllm_model = model.replace("litellm_proxy/", "hosted_vllm/")

    print(f"  base_url  : {base}")
    print(f"  model     : {vllm_model}")
    print(f"  ctx chars : {len(context_text)}")
    print(f"  y chars   : {len(completion)}")

    ctx_resp = litellm.text_completion(
        model=vllm_model, base_url=base, api_key="dummy",
        prompt=context_text, max_tokens=1, logprobs=1, echo=True,
        extra_body={"prompt_logprobs": 1},
    )
    n_ctx = ctx_resp["usage"]["prompt_tokens"]
    ctx_tokens = ctx_resp["choices"][0]["logprobs"].tokens
    print(f"  n_ctx tokens: {n_ctx}  (last 3: {ctx_tokens[-3:]})")

    full_resp = litellm.text_completion(
        model=vllm_model, base_url=base, api_key="dummy",
        prompt=full_prompt, max_tokens=1, logprobs=1, echo=True,
        extra_body={"prompt_logprobs": 1},
    )
    n_full = full_resp["usage"]["prompt_tokens"]
    full_tokens = full_resp["choices"][0]["logprobs"].tokens
    full_lps = full_resp["choices"][0]["logprobs"].token_logprobs
    n_completion = n_full - n_ctx
    print(f"  n_full tokens : {n_full}  (n_completion = {n_completion})")
    print(f"  First 3 completion tokens: {full_tokens[n_ctx:n_ctx+3]}")
    completion_lps = [lp for lp in full_lps[n_ctx:n_full] if lp is not None]
    print(f"  Extracted {len(completion_lps)} completion logprobs")
    print(f"  First 5: {completion_lps[:5]}")
    if completion_lps:
        print(f"  Mean logprob: {sum(completion_lps) / len(completion_lps):.4f}")


def _messages_to_text(messages: list[dict]) -> str:
    """Serialize chat messages to plain text for the completions endpoint."""
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content") or ""
        if role == "system":
            parts.append(content)
        elif role == "user":
            parts.append(f"User:\n{content}")
        elif role == "assistant":
            parts.append(f"Assistant:\n{content}")
        elif role == "tool":
            parts.append(f"Tool:\n{content}")
    return "\n\n".join(parts) + "\n\nAssistant:\n"
