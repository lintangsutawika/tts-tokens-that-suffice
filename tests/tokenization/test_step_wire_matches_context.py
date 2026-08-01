"""
Prove step_once puts *exactly the reconstructed context* on the wire, and pin
the one transform that still separates the fidelity-scored prompt from the
prompt the agent actually decodes from.

The context here is a REAL example: trajectory 6d2bcc9c from the summarizer
training pool, reconstructed exactly the way eval_top_summaries / the reward do
— from_collect_dict -> threshold_split(@16384 tok) -> steps_to_messages ->
wire tool-args — and the summary arm uses that trajectory's actual trained
summary. So the two arms under test are byte-identical to what fidelity scores.

Why this matters. The distortion reward scores a compacted context by rendering
it locally (tokenizer.apply_chat_template) and posting the text to
/v1/completions. The [0]-vs-GT proxy instead asks the agent for its next action
via step_once, which posts the same context as *messages* to
/v1/chat/completions. If step_once silently mutated the context, the two numbers
would describe different inputs. It does not — the only differences are:

  * client side: the mini-swe harness drops the bookkeeping `extra` key
    (_prepare_messages_for_api). Nothing else is reordered, injected, or dropped.
  * server side: vLLM validates the chat request against the OpenAI schema,
    which has no reasoning_content field, so prior-turn reasoning is dropped
    before the template runs. A naive local /completions render would keep it.

Net: the served (decoded-from) prompt equals the local render with
reasoning_content stripped, and differs from the reasoning-kept render by
precisely the reasoning tokens. The fidelity path used to score the reasoning-
kept render — a context inference never sees — so build_x_scoring_messages /
compute_distortion_messages now strip reasoning (tts.reward.utils.strip_reasoning)
to score exactly the inference render; test_fidelity_x_render_matches_inference
guards that.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from tts.data.agent_trajectory import (
    AGENT_SYSTEM_PROMPT,
    AgentTrajectory,
    steps_to_messages,
)
from tts.reward.distortion_reward import next_agent_action
from tts.reward.utils import reconstruct_generation, to_template_tool_calls
from tts.summarization.model_based import build_z_scoring_messages

from .conftest import MODEL, to_api_tool_calls

# The qwen assistant-generation grammar: an optional reasoning span, the think
# close, then the bash tool call. add_generation_prompt leaves "<think>\n" open,
# so a real continuation begins inside the reasoning and closes it before the
# call; reconstruct_generation strips the trailing <|im_end|> the model emits to
# stop, so `end` is optional. Whitespace is load-bearing — it is what the
# per-token logprobs of y are read against.
Y_GRAMMAR = re.compile(
    r"^(?P<reasoning>.*)\n</think>\n\n<tool_call>\n<function=bash>\n"
    r"<parameter=command>\n(?P<cmd>.*)\n</parameter>\n</function>\n</tool_call>"
    r"(?P<end><\|im_end\|>)?\Z",
    re.DOTALL,
)

BASH_TOOL = pytest.importorskip(
    "minisweagent.models.utils.actions_toolcall"
).BASH_TOOL

FIXTURE = Path(__file__).parent.parent / "fixtures" / "real_trajectory_6d2bcc9c.json"


@pytest.fixture(scope="session")
def real_fixture() -> dict:
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="session")
def real_context(real_fixture, tokenizer) -> list[dict]:
    """The real compaction-point context, rebuilt exactly as the reward does.

    from_collect_dict -> threshold_split @ split_at_tokens -> steps_to_messages,
    with tool-call args in wire (JSON-string) form. This is x — the input every
    arm compacts and the input fidelity scores.
    """
    cfg = real_fixture["split"]
    traj = AgentTrajectory.from_collect_dict(real_fixture["source"])
    split = traj.threshold_split(
        tokenizer,
        split_at_tokens=cfg["split_at_tokens"],
        min_prefix=cfg["min_split_prefix"],
        min_suffix=cfg["min_split_suffix"],
        max_continuation_tokens=cfg["split_at_tokens"] - cfg["compaction_token_budget"],
    )
    assert split is not None, "fixture trajectory no longer splits at the configured threshold"
    sys_prompt = split.agent_system_prompt or AGENT_SYSTEM_PROMPT
    return to_api_tool_calls(
        steps_to_messages(split.steps, split.task, system_prompt=sys_prompt)
    )


@pytest.fixture(scope="session")
def real_next_action(real_fixture, tokenizer) -> dict:
    """y — the GT next agent action fidelity scores log P(y|x) for.

    The first assistant turn of the continuation, rebuilt exactly as the reward
    does (steps_to_messages, dropping the re-emitted system+task pair).
    """
    cfg = real_fixture["split"]
    traj = AgentTrajectory.from_collect_dict(real_fixture["source"])
    split = traj.threshold_split(
        tokenizer,
        split_at_tokens=cfg["split_at_tokens"],
        min_prefix=cfg["min_split_prefix"],
        min_suffix=cfg["min_split_suffix"],
        max_continuation_tokens=cfg["split_at_tokens"] - cfg["compaction_token_budget"],
    )
    sys_prompt = split.agent_system_prompt or AGENT_SYSTEM_PROMPT
    cont = steps_to_messages(split.continuation, split.task, system_prompt=sys_prompt)[2:]
    y = next_agent_action(cont)
    assert y is not None, "fixture continuation has no assistant action"
    return y


# --- the two reconstructions the eval actually scores -----------------------

def full_arm(real_context, real_fixture):
    """The uncompacted context arm: scored and stepped verbatim."""
    return copy.deepcopy(real_context)


def summary_arm(real_context, real_fixture):
    """The trained/summary arm: head + real-z-as-tool-turn + tail (build_z_scoring_messages)."""
    cfg = real_fixture["split"]
    return build_z_scoring_messages(
        real_fixture["summary"], real_context,
        max_size=cfg["max_size"], keep_first=cfg["keep_first"],
    )


ARMS = {"full": full_arm, "summary": summary_arm}


def drop_extra(messages):
    """What _prepare_messages_for_api does to the payload, minus cache-control."""
    return [{k: v for k, v in m.items() if k != "extra"} for m in messages]


# --- 1. the API-call catcher: no server needed ------------------------------

def _canned_bash_response(model, command="ls /testbed"):
    import litellm
    return litellm.ModelResponse(
        model=model,
        choices=[{
            "index": 0, "finish_reason": "tool_calls",
            "message": {
                "role": "assistant", "content": "",
                "tool_calls": [{
                    "id": "call_0", "type": "function",
                    "function": {"name": "bash", "arguments": json.dumps({"command": command})},
                }],
            },
        }],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


@pytest.mark.parametrize("arm", list(ARMS))
def test_step_once_sends_exactly_the_reconstructed_context(
    arm, real_context, real_fixture, monkeypatch
):
    """
    The messages litellm.completion receives are the reconstructed context with
    only `extra` dropped — nothing reordered, injected, or otherwise mutated —
    and StepResult.messages_sent mirrors that payload exactly.
    """
    import litellm
    from tts.agent.step import step_once

    reconstructed = ARMS[arm](real_context, real_fixture)

    captured = {}

    def catcher(*, model, messages, tools=None, **kwargs):
        captured["messages"] = copy.deepcopy(messages)
        captured["tools"] = tools
        return _canned_bash_response(model)

    monkeypatch.setattr(litellm, "completion", catcher)

    result = step_once(
        reconstructed, model_name=f"hosted_vllm/{MODEL}",
        api_base="http://unused/v1", execute=False, model_kwargs={"temperature": 0.0},
    )

    # The agent asked for exactly one tool and got a parseable action back.
    assert captured["tools"] == [BASH_TOOL]
    assert result.action == {"command": "ls /testbed", "tool_call_id": "call_0"}

    # The wire payload is the reconstructed context, only `extra` removed.
    assert captured["messages"] == drop_extra(reconstructed)
    # StepResult.messages_sent is a faithful record of that payload.
    assert result.messages_sent == captured["messages"]


# --- 2. the processed text: needs a live server -----------------------------

def _served_prompt(url, tokenizer, messages) -> str:
    import requests
    r = requests.post(
        f"{url}/tokenize",
        json={
            "model": MODEL,
            "messages": to_api_tool_calls(messages),
            "tools": [BASH_TOOL],
            "add_generation_prompt": True,
            "return_token_strs": True,
        },
        timeout=60,
    )
    r.raise_for_status()
    return tokenizer.decode(r.json()["tokens"])


def _served_token_ids(url, messages) -> list[int]:
    """The exact token ids the chat endpoint renders these messages+tools to."""
    import requests
    r = requests.post(
        f"{url}/tokenize",
        json={
            "model": MODEL,
            "messages": to_api_tool_calls(messages),
            "tools": [BASH_TOOL],
            "add_generation_prompt": True,
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["tokens"]


def _strip_reasoning(messages):
    out = copy.deepcopy(messages)
    for m in out:
        m.pop("reasoning_content", None)
    return out


@pytest.mark.parametrize("arm", list(ARMS))
def test_served_prompt_equals_reasoning_stripped_reconstruction(
    arm, vllm_server, tokenizer, real_context, real_fixture
):
    """
    The text the server actually processes for step_once equals the local render
    of the reconstructed context with reasoning_content stripped — byte for byte
    — and is NOT equal to the reasoning-kept render (the render the fidelity path
    used before the reasoning-strip fix). The gap is exactly the reasoning tokens,
    none of which survive to the prompt.
    """
    reconstructed = ARMS[arm](real_context, real_fixture)

    served = _served_prompt(vllm_server, tokenizer, reconstructed)
    local_kept = tokenizer.apply_chat_template(
        to_template_tool_calls(reconstructed),
        tokenize=False, add_generation_prompt=True, tools=[BASH_TOOL],
    )
    local_stripped = tokenizer.apply_chat_template(
        to_template_tool_calls(_strip_reasoning(reconstructed)),
        tokenize=False, add_generation_prompt=True, tools=[BASH_TOOL],
    )

    reasonings = [m["reasoning_content"] for m in reconstructed
                  if m.get("role") == "assistant" and m.get("reasoning_content")]
    assert reasonings, f"{arm} arm unexpectedly carries no reasoning to distinguish the renders"

    assert served == local_stripped, "served prompt != reasoning-stripped reconstruction"
    assert served != local_kept, "expected the reasoning-kept (fidelity) render to differ"
    for r in reasonings:
        assert r not in served, "reasoning_content leaked into the served prompt"
    assert len(tokenizer.encode(local_kept)) > len(tokenizer.encode(served))


def test_fidelity_x_render_matches_inference(vllm_server, tokenizer, real_context):
    """
    The reward's actual x-context render is byte-identical to what inference
    tokenizes. This drives the production path — build_x_scoring_messages, which
    strip_reasoning made inference-faithful — rather than a re-derived strip, so
    it regresses if that stripping is ever removed.
    """
    from tts.reward.distortion_reward import build_x_scoring_messages

    x_messages = to_template_tool_calls(build_x_scoring_messages(real_context))
    assert not any(m.get("reasoning_content") for m in x_messages), \
        "build_x_scoring_messages left prior-turn reasoning in the x-context"

    fidelity_render = tokenizer.apply_chat_template(
        x_messages, tokenize=False, add_generation_prompt=True, tools=[BASH_TOOL],
    )
    assert fidelity_render == _served_prompt(vllm_server, tokenizer, real_context), \
        "reward x-render diverged from the inference (chat) render"


@pytest.mark.parametrize("arm", list(ARMS))
def test_litellm_and_renderer_produce_identical_tokens(
    arm, vllm_server, tokenizer, real_context, real_fixture
):
    """
    A real litellm chat call tokenizes the context to exactly our renderer output.

    The catcher above proves what we *hand* litellm; this proves litellm does not
    reshape it on the way to the model. Two live-server checks:

      * server /tokenize ids == our reasoning-stripped render ids (exact), so the
        chat template + tool rendering the server runs matches ours; and
      * a real step_once call reports usage.prompt_tokens == that length, so the
        end-to-end litellm path put exactly those tokens on the wire.

    Together: the tokens the model actually processes are byte-identical to what
    the fidelity renderer produces once reasoning_content is stripped.
    """
    from tts.agent.step import step_once

    reconstructed = ARMS[arm](real_context, real_fixture)

    ours = tokenizer.apply_chat_template(
        to_template_tool_calls(_strip_reasoning(reconstructed)),
        tokenize=True, add_generation_prompt=True, tools=[BASH_TOOL],
    )

    assert _served_token_ids(vllm_server, reconstructed) == ours, \
        "server /tokenize ids disagree with our renderer"

    result = step_once(
        reconstructed, model_name=f"hosted_vllm/{MODEL}",
        api_base=f"{vllm_server}/v1", execute=False,
        model_kwargs={"temperature": 0.0, "max_tokens": 1},
    )
    prompt_tokens = result.message["extra"]["response"]["usage"]["prompt_tokens"]
    assert prompt_tokens == len(ours), (prompt_tokens, len(ours))


# --- 3. the completion y: format fidelity for log P(y|x) --------------------

def _command_of(action: dict) -> str:
    tc = action["tool_calls"][0]["function"]["arguments"]
    if isinstance(tc, str):
        tc = json.loads(tc)
    return tc["command"]


def _real_generation_text(url, tokenizer, messages) -> str:
    """The raw token stream the model generates from `messages`, detokenized.

    Uses return_tokens_as_token_ids so the generation is read as the exact ids
    the model emitted — independent of whether the server splits reasoning into
    reasoning_content — then decoded back to text (which includes the trailing
    <|im_end|> the model stops on).
    """
    import requests
    r = requests.post(
        f"{url}/v1/chat/completions",
        json={
            "model": MODEL, "messages": to_api_tool_calls(messages),
            "tools": [BASH_TOOL], "temperature": 0.0, "max_tokens": 1024,
            "logprobs": True, "top_logprobs": 0, "return_tokens_as_token_ids": True,
        },
        timeout=120,
    )
    r.raise_for_status()
    content = r.json()["choices"][0]["logprobs"]["content"]
    ids = [int(t["token"].split("token_id:")[1]) for t in content]
    return tokenizer.decode(ids)


def test_reconstructed_y_follows_the_generation_grammar(tokenizer, real_context, real_next_action):
    """
    The y fidelity scores — reconstruct_generation(x, GT action).text — has the
    exact shape of a model generation: reasoning, </think>, the bash call, and the
    terminating <|im_end|>. reconstruct_generation drops only the template's
    trailing newline, so the eos token stays in y — the decision to stop is part
    of the next action, and it is scored consistently on the x and z sides. If y
    did not match this grammar, log P(y|x)'s per-token logprobs would be reading a
    malformed continuation.
    """
    gen = reconstruct_generation(real_context, real_next_action, tokenizer)
    m = Y_GRAMMAR.match(gen.text)
    assert m is not None, f"reconstructed y is not a well-formed generation:\n{gen.text!r}"
    assert m.group("end") == "<|im_end|>", "y should end at the eos stop token, not a bare newline"
    assert not gen.text.endswith("\n"), "the template's trailing newline must be stripped from y"
    assert m.group("cmd") == _command_of(real_next_action)


def test_real_generation_matches_the_same_grammar(vllm_server, tokenizer, real_context):
    """
    A real greedy generation from x detokenizes to the SAME grammar the
    reconstructed y follows — reasoning, </think>, bash call — differing only in
    the (variable) reasoning text and the trailing <|im_end|>. So the y whose
    logprobs we read is in the exact format the model actually produces: log
    P(y|x) scores an on-distribution continuation, not a re-serialized artifact.
    """
    text = _real_generation_text(vllm_server, tokenizer, real_context)
    m = Y_GRAMMAR.match(text)
    assert m is not None, f"real generation did not match the y grammar:\n{text!r}"
    assert m.group("cmd"), "real generation produced no bash command"
