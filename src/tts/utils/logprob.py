"""
tts.utils.logprob — per-token KL scoring via a vLLM-compatible server.
"""

from __future__ import annotations

import math

import litellm


def score_completion(
    x_messages: list[dict],
    z_messages: list[dict],
    completion: str,
    model: str,
    api_base: str,
    tokenizer,
    k: int = 20,
) -> list[float] | None:
    """
    Estimate per-position KL D_KL(p(·|x) ‖ p(·|z)) using top-k distributions.

    At each completion position t, sums over tokens in the intersection of
    x-context and z-context top-k lists:

        KL_t ≈ Σ_{v ∈ top_k(x) ∩ top_k(z)} p_x(v) · (log p_x(v) − log p_z(v))

    Context texts are produced via tokenizer.apply_chat_template so the
    tokenization matches exactly what the model sees during inference.

    4 API calls total: 2 cheap context-length calls (logprobs=1),
    then 2 full-prompt calls with top-k distributions.

    Returns one KL float per completion token, or None on error.
    """
    if not completion.strip():
        return []

    base = api_base.rstrip("/")
    vllm_model = model.replace("litellm_proxy/", "hosted_vllm/")
    x_text = tokenizer.apply_chat_template(x_messages, tokenize=False, add_generation_prompt=True)
    z_text = tokenizer.apply_chat_template(z_messages, tokenize=False, add_generation_prompt=True)

    # Step 1: context token counts
    try:
        n_x_ctx = litellm.text_completion(
            model=vllm_model, base_url=base, api_key="dummy",
            prompt=x_text, max_tokens=1, logprobs=1, echo=True,
        )["usage"]["prompt_tokens"]
        n_z_ctx = litellm.text_completion(
            model=vllm_model, base_url=base, api_key="dummy",
            prompt=z_text, max_tokens=1, logprobs=1, echo=True,
        )["usage"]["prompt_tokens"]
    except Exception as exc:
        print(f"    [score] Context tokenisation failed: {exc}")
        return None

    # Step 2: full-prompt top-k scoring
    # n_completion is the same for x and z since the completion text is identical.
    try:
        x_resp = litellm.text_completion(
            model=vllm_model, base_url=base, api_key="dummy",
            prompt=x_text + completion, max_tokens=1, logprobs=k, echo=True,
        )
        n_x_full = x_resp["usage"]["prompt_tokens"]
        n_completion = n_x_full - n_x_ctx
        x_top = x_resp["choices"][0]["logprobs"].top_logprobs[n_x_ctx:n_x_full]
    except Exception as exc:
        print(f"    [score] x-context scoring failed: {exc}")
        return None

    try:
        z_resp = litellm.text_completion(
            model=vllm_model, base_url=base, api_key="dummy",
            prompt=z_text + completion, max_tokens=1, logprobs=k, echo=True,
        )
        z_top = z_resp["choices"][0]["logprobs"].top_logprobs[n_z_ctx:n_z_ctx + n_completion]
    except Exception as exc:
        print(f"    [score] z-context scoring failed: {exc}")
        return None

    # Step 3: per-position KL over intersection of top-k tokens (forward KL, teacher-weighted)
    n = min(len(x_top), len(z_top))
    kl_per_token: list[float] = []
    for t in range(n):
        x_dist: dict[str, float] = x_top[t] or {}
        z_dist: dict[str, float] = z_top[t] or {}
        kl_t = 0.0
        for token, lp_x in x_dist.items():
            if token in z_dist:
                kl_t += math.exp(lp_x) * (lp_x - z_dist[token])
        kl_per_token.append(kl_t)

    return kl_per_token
