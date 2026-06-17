"""
tts.utils.logprob — per-token log-probability scoring via a vLLM-compatible server.

Primary entry point: score_completion(context_messages, completion, model, api_base)
"""

from __future__ import annotations

import httpx
import litellm


def score_completion(
    context_messages: list[dict],
    completion: str,
    model: str,
    api_base: str,
) -> list[float] | None:
    """
    Compute per-token log p(completion | context_messages) using the
    /v1/completions endpoint with echo=True and logprobs=1.

    Converts context_messages to plain text (role-prefixed, no model-specific
    chat template) so the full prompt can be submitted to the text completions
    endpoint. The context boundary is found by tokenising the context alone
    with max_tokens=1, then the completion logprobs are extracted from the
    combined prompt+completion call.

    Returns a list of log-probs (one per completion token), or None on error.
    """
    if not completion.strip():
        return []

    context_text = _messages_to_text(context_messages)
    full_prompt = context_text + completion
    base = api_base.rstrip("/")

    try:
        ctx_resp = litellm.text_completion(
            model=model.replace("litellm_proxy/", "hosted_vllm/"),
            # base_url=f"{base}/v1",
            base_url=base,
            prompt=context_text,
            max_tokens=1,
            logprobs=1,
            extra_body={
                "prompt_logprobs": 1  # Tell vLLM to safely return input sequence logprobs
            },
        )
        n_ctx = ctx_resp['usage']['prompt_tokens']
    except Exception as exc:
        print(f"    [score] Context tokenisation failed: {exc}")
        return None

    try:
        full_resp = litellm.text_completion(
            model=model.replace("litellm_proxy/", "hosted_vllm/"),
            # base_url=f"{base}/v1",
            base_url=base,
            prompt=full_prompt,
            max_tokens=1,
            logprobs=1,
            extra_body={
                "prompt_logprobs": 1  # Tell vLLM to safely return input sequence logprobs
            },
        )
        all_logprobs = full_resp['choices'][0]['prompt_logprobs']
    except Exception as exc:
        print(f"    [score] Full-prompt scoring failed: {exc}")
        return None

    raw_lp_dict = [lp for lp in all_logprobs[n_ctx:] if lp is not None]
    lp = [v['logprob'] for z in raw_lp_dict for k,v in z.items()]
    return lp


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
