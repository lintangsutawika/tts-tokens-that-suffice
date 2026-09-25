"""
tts.reward.distortion_reward — distortion-based fidelity reward.

Named for what it computes, not for a divergence it does not. The per-token
contrast is evaluated on the *recorded* continuation, i.e. y ~ p(·|x), making it
a single-sample estimate of the FORWARD KL D_KL(p(·|x) ‖ p(·|z)) — see
tts.utils.logprob. The reward proper is that contrast squashed per token and
netted against copy/length penalties, so it is not a KL either way.

    raw:      r(x, z) = (1/|y|) Σ_t [log p(y_t|y<t, z) − log p(y_t|y<t, x)] − λ·|z|
    bounded:  r(x, z) = (1/|y|) Σ_t tanh(Δ_t / β) − λ·|z|,  Δ_t = log p(y_t|z) − log p(y_t|x)

x  = partial trajectory steps (the "seen" context)
z  = generated summary (the compression)
y  = continuation steps (what the agent does next)
λ  = length-penalty coefficient (distortion_lambda)
β  = distortion_beta; when > 0, use the bounded (-1, 1) tanh contrast — comparable
     across prompts, 0 at parity, +1 z far better / −1 far worse. β = 0 keeps raw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from tts.summarization.model_based import build_z_scoring_messages
from tts.utils.logprob import XLogprobs, precompute_x, score_completion_z

from tts.reward.copy_penalty import copy_penalty
from tts.reward.utils import (
    Generation,
    reconstruct_generation,
    strip_reasoning,
    to_template_tool_calls,
)


def build_x_scoring_messages(partial_messages: list[dict]) -> list[dict]:
    """x-context: the full original message history, minus prior-turn reasoning.

    reasoning_content is dropped so the x the reward scores is byte-identical to
    what inference (/v1/chat/completions) tokenizes; see strip_reasoning. Only
    the context is stripped — the completion y is built from next_action, which
    is never routed through here, so its live reasoning is preserved.
    """
    return strip_reasoning(partial_messages)


def bounded_fidelity(kl_per_token: list[float], beta: float) -> float:
    """
    Mean per-token probability-space contrast between the z- and x-predictors,
    bounded to (-1, 1).

    Each element of kl_per_token is d_t = log p(y_t|x) − log p(y_t|z), so the
    signed log-ratio favouring z is Δ_t = −d_t. We squash each token with
    tanh(Δ_t / beta) and average:

        r = (1/|y|) Σ_t tanh(Δ_t / beta)

      * Δ_t = 0 (z as predictive as x on token t)      -> 0
      * Δ_t → +∞ (z a far better predictor)            -> +1
      * Δ_t → −∞ (z far worse)                         -> −1

    Raw logprob magnitudes differ wildly between prompts, so only this paired,
    same-y, same-model contrast is comparable across trajectories. Bounding PER
    TOKEN (before averaging) caps a single unpredictable token — e.g. an env
    byte neither context anticipates — from dominating the mean, while the mean
    keeps the score length-normalized and in (-1, 1). beta sets how large a
    per-token log-ratio counts as "decisively" better/worse.
    """
    n = len(kl_per_token)
    if n == 0:
        return 0.0
    return sum(math.tanh(-d / beta) for d in kl_per_token) / n


@dataclass
class XContext:
    """Pre-computed x-side data for a fixed (trajectory, next-action) pair."""
    x_messages: list[dict]
    y_text: str
    x_logprobs: XLogprobs
    generation: Generation | None = None  # y's provenance; .verified checks the token count


def next_agent_action(continuation_messages: list[dict]) -> dict | None:
    """The next assistant turn — the single decision y is made of."""
    return next((m for m in continuation_messages if m.get("role") == "assistant"), None)


def continuation_turns(
    continuation_messages: list[dict], n_turns: int
) -> list[tuple[list[dict], dict]]:
    """The first n_turns assistant turns, each with the history that precedes it.

    Yields (history, action) where action is the k-th assistant turn and history
    is the continuation messages before it — the real turns t_0..t_{k-1} and
    their observations. Scoring action given z + history is the depth-k term of
    the multi-turn reward: R_k from P(t_k | z, t_0, ..., t_{k-1}). Teacher forced
    on the real turns, so every summary in a group is scored on the same history.
    """
    out: list[tuple[list[dict], dict]] = []
    for i, m in enumerate(continuation_messages):
        if m.get("role") == "assistant":
            out.append((continuation_messages[:i], m))
            if len(out) >= n_turns:
                break
    return out


def precompute_x_contexts(
    partial_messages: list[dict],
    continuation_messages: list[dict],
    n_turns: int,
    model: str,
    api_base: str,
    tokenizer,
    k: int = 0,
    tools: list | None = None,
) -> list[tuple[list[dict], XContext]]:
    """Pre-compute the x-context for each of the first n_turns continuation turns.

    Returns [(history_k, x_ctx_k), ...]. The reference context at depth k is the
    full prefix partial + history_k, so x_ctx_k is exactly precompute_x_context
    on that prefix. Computed once per trajectory (x does not depend on the
    generated summary); distortion_reward_z_multi reuses the list for every
    summary in the group.

    Stops at the first depth whose x-context scoring fails (e.g. the prefix
    outgrew the scorer's window), so the returned list is a contiguous 0..M-1
    and all group members average over the same M depths.
    """
    ctxs: list[tuple[list[dict], XContext]] = []
    for history, action in continuation_turns(continuation_messages, n_turns):
        x_ctx = precompute_x_context(
            partial_messages=partial_messages + history,
            next_action=action, model=model, api_base=api_base,
            tokenizer=tokenizer, k=k, tools=tools,
        )
        if x_ctx is None:
            break
        ctxs.append((history, x_ctx))
    return ctxs


def precompute_x_context(
    partial_messages: list[dict],
    next_action: dict,
    model: str,
    api_base: str,
    tokenizer,
    k: int = 0,
    tools: list | None = None,
) -> XContext | None:
    """
    Pre-compute the x-context logprobs for one (trajectory, next-action) pair.

    y is the agent's NEXT ACTION — one assistant turn, reasoning included — not a
    token-capped slab of the remaining transcript. That boundary is semantic, so
    every unit scores exactly one decision and units stay comparable across
    trigger points; a token cap instead truncated 83% of units mid-continuation
    and made the sample composition drift with the trigger.

    Call this once per (trajectory, action); pass the result to
    compute_distortion_z for each of the group_size summaries so x is not
    rescored group_size times.
    """
    x_messages = to_template_tool_calls(build_x_scoring_messages(partial_messages))
    gen = reconstruct_generation(x_messages, next_action, tokenizer, tools=tools)
    x_logprobs = precompute_x(x_messages, gen.text, model, api_base, tokenizer, k=k, tools=tools)
    if x_logprobs is None:
        return None
    return XContext(
        x_messages=x_messages, y_text=gen.text, x_logprobs=x_logprobs, generation=gen
    )


def compute_distortion_z(
    x_ctx: XContext,
    summary: str,
    partial_messages: list[dict],
    model: str,
    api_base: str,
    tokenizer,
    max_size: int = 20,
    keep_first: int = 4,
    beta: float = 0.0,
    tools: list | None = None,
) -> dict:
    """
    Compute distortion using pre-computed x-context logprobs.

    Makes 2 API calls (z context length + full z scoring) instead of 4.

    When beta > 0, also returns fidelity_bounded — the mean per-token tanh
    contrast in (-1, 1); see bounded_fidelity. Raw fidelity is always returned
    for logging.
    """
    z_messages = build_z_scoring_messages(
        summary, partial_messages, max_size=max_size, keep_first=keep_first
    )
    return compute_distortion_messages(
        x_ctx, z_messages, model, api_base, tokenizer, beta=beta, tools=tools
    )


def compute_distortion_messages(
    x_ctx: XContext,
    z_messages: list[dict],
    model: str,
    api_base: str,
    tokenizer,
    beta: float = 0.0,
    tools: list | None = None,
) -> dict:
    """
    Distortion for an already-built compacted context.

    Shared by every compaction strategy — summarization (build_z_scoring_messages)
    and env-masking (build_maskenv_scoring_messages) differ only in how they
    construct z_messages, not in how they are scored against x.
    """
    # Trajectories carry tool_call arguments as JSON strings, the template needs
    # dicts. Normalise here so z built from raw messages renders like x does, and
    # drop prior-turn reasoning so the scored z-context matches what inference
    # tokenizes (chat validation drops it; /v1/completions would keep it). This
    # is the single choke point for every z-scoring path — base_z's head/tail and
    # any teacher-forced continuation turns appended to it — so it also keeps the
    # n_z_ctx_tokens count below on the inference-faithful render.
    z_messages = strip_reasoning(to_template_tool_calls(z_messages))
    kl_per_token = score_completion_z(
        x_ctx.x_logprobs, z_messages, x_ctx.y_text, model, api_base, tokenizer, tools=tools
    )
    if kl_per_token is None:
        return {"error": "score_completion_z failed", "fidelity": None}
    n = len(kl_per_token)
    if n == 0:
        return {"error": "No completion tokens scored", "fidelity": None}
    distortion = sum(kl_per_token) / n
    result = {
        "fidelity": -distortion,
        "distortion": distortion,
        "n_tokens": n,
        "n_x_messages": len(x_ctx.x_messages),
        "n_z_messages": len(z_messages),
        # Actual compacted-context cost. Strategies differ a lot here (env-masking
        # keeps every agent action, summarization keeps none), so fidelity is only
        # comparable across specs alongside this.
        "n_z_ctx_tokens": len(
            tokenizer.encode(
                tokenizer.apply_chat_template(
                    z_messages, tokenize=False, add_generation_prompt=True, tools=tools
                )
            )
        ),
    }
    if beta and beta > 0:
        result["fidelity_bounded"] = bounded_fidelity(kl_per_token, beta)
    return result


def distortion_reward_z(
    x_ctx: XContext,
    summary: str,
    partial_messages: list[dict],
    model: str,
    api_base: str,
    tokenizer,
    max_size: int = 20,
    keep_first: int = 4,
    lambda_len: float = 0.0,
    lambda_copy: float = 0.0,
    copy_threshold: float = 0.3,
    marker_penalty: float = 0.0,
    beta: float = 0.0,
    tools: list | None = None,
) -> float | None:
    """
    KL-distortion reward using pre-computed x-context logprobs.

    Use precompute_x_context() once per trajectory, then call this for each
    of the group_size summaries. Makes 2 API calls instead of 4.

    When beta > 0 the fidelity base is the bounded per-token tanh contrast in
    (-1, 1) — 0 at parity, +1 when z predicts y far better than x, −1 far worse
    (see bounded_fidelity) — instead of the raw unbounded mean log-ratio. This
    makes the reward comparable across prompts and caps outlier tokens; beta=0
    keeps the legacy raw fidelity.

    Fidelity is maximized at z = x, so the anti-copy penalty (see copy_penalty)
    is what stops the policy from transcribing its input. A copy is scored and
    penalized, never dropped: dropping it would remove the negative signal the
    group needs to learn that copying is bad.

    Returns None only when scoring fails, so the caller can drop the sample from
    its group. A numeric fallback (e.g. 0.0) would rank above the typically
    negative fidelities and reinforce whatever caused the failure.
    """
    z_messages = build_z_scoring_messages(
        summary, partial_messages, max_size=max_size, keep_first=keep_first
    )
    result = distortion_reward_messages(
        x_ctx=x_ctx,
        z_messages=z_messages,
        summary=summary,
        model=model,
        api_base=api_base,
        tokenizer=tokenizer,
        lambda_len=lambda_len,
        lambda_copy=lambda_copy,
        copy_threshold=copy_threshold,
        marker_penalty=marker_penalty,
        beta=beta,
        tools=tools,
    )
    return result["reward"]


def distortion_reward_z_multi(
    x_ctxs: list[tuple[list[dict], XContext]],
    summary: str,
    partial_messages: list[dict],
    model: str,
    api_base: str,
    tokenizer,
    max_size: int = 20,
    keep_first: int = 4,
    lambda_len: float = 0.0,
    lambda_copy: float = 0.0,
    copy_threshold: float = 0.3,
    marker_penalty: float = 0.0,
    beta: float = 0.0,
    tools: list | None = None,
) -> float | None:
    """Multi-turn distortion reward: average fidelity over N succeeding turns.

    R = mean_k R_k, where R_k is the fidelity of predicting the k-th real turn
    t_k from the compacted context plus the real turns before it:

        R_0 from P(t_0 | z)
        R_1 from P(t_1 | z, t_0)
        R_2 from P(t_2 | z, t_0, t_1)   ...

    The compacted head (z built once from partial_messages) is fixed; only the
    teacher-forced history t_0..t_{k-1} grows, which is exactly x_ctxs[k]'s
    history. This rewards a summary for supporting a run of decisions, not only
    the immediate next one — the single-turn reward is the N=1 special case and
    reproduces distortion_reward_z exactly.

    The anti-copy penalty applies to the summary text once (not per depth); it
    is about the generated z, not about any particular continuation. Returns
    None if x_ctxs is empty or any depth's z-scoring fails, so the caller drops
    the sample rather than average a fabricated term.
    """
    if not x_ctxs:
        return None
    base_z = build_z_scoring_messages(
        summary, partial_messages, max_size=max_size, keep_first=keep_first
    )
    r_ks: list[float] = []
    for history, x_ctx in x_ctxs:
        result = compute_distortion_messages(
            x_ctx, base_z + history, model, api_base, tokenizer, beta=beta, tools=tools
        )
        if "error" in result or result.get("fidelity") is None:
            return None
        r_ks.append(result["fidelity_bounded"] if beta and beta > 0 else result["fidelity"])
    base = sum(r_ks) / len(r_ks)

    # Penalty on the summary text, computed once against the trajectory it
    # summarized (depth 0's x-messages == partial, no history appended yet).
    if summary:
        penalty, _ = copy_penalty(
            summary, x_ctxs[0][1].x_messages, tokenizer,
            lambda_len=lambda_len, lambda_copy=lambda_copy,
            copy_threshold=copy_threshold, marker_penalty=marker_penalty,
        )
    else:
        penalty = 0.0
    return base - penalty


def distortion_reward_messages(
    x_ctx: XContext,
    z_messages: list[dict],
    summary: str = "",
    *,
    model: str,
    api_base: str,
    tokenizer,
    lambda_len: float = 0.0,
    lambda_copy: float = 0.0,
    copy_threshold: float = 0.3,
    marker_penalty: float = 0.0,
    beta: float = 0.0,
    tools: list | None = None,
) -> dict:
    """
    Reward for an already-built compacted context, with its components.

    The single place the reward is composed. Every compaction strategy scores
    the same way against x and differs only in how z_messages were built, so
    strategies stay comparable and callers cannot drift into their own
    arithmetic.

    `summary` is the generated text the anti-copy penalty applies to. Strategies
    that generate nothing (env-masking, truncation) pass "" and take no penalty:
    there is no policy output to transcribe its input, so charging them the
    summarizer's length/copy penalty would understate them against a summary
    arm for a cost they never incurred.

    Returns fidelity, penalty components, and `reward` — None when scoring
    failed, so the caller can drop the sample rather than feed a fabricated
    number into a group.
    """
    result = compute_distortion_messages(
        x_ctx, z_messages, model, api_base, tokenizer, beta=beta, tools=tools
    )
    if "error" in result or result.get("fidelity") is None:
        return {**result, "reward": None}

    base = result["fidelity_bounded"] if beta and beta > 0 else result["fidelity"]
    if summary:
        penalty, components = copy_penalty(
            summary,
            x_ctx.x_messages,
            tokenizer,
            lambda_len=lambda_len,
            lambda_copy=lambda_copy,
            copy_threshold=copy_threshold,
            marker_penalty=marker_penalty,
        )
    else:
        penalty, components = 0.0, {}
    return {**result, **components, "base": base, "penalty": penalty, "reward": base - penalty}
