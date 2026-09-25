from __future__ import annotations

import json

from jinja2 import StrictUndefined, Template


def render_template(template: str, template_vars: dict) -> str:
    return Template(template, undefined=StrictUndefined).render(**template_vars)


def parse_messages(messages: list[dict]) -> list[str]:
    """
    Convert a message list into event strings for the summarization template.

    Each event covers one (assistant action, tool result) pair. Supports two
    wire formats for tool messages:
      - ``msg["extra"]["raw_output"]``  (rllm/era internal format)
      - ``msg["content"]``              (standard OpenAI / example.json format)

    For assistant messages, ``tool_calls[0].function.arguments`` may be a
    pre-parsed dict (example.json) or a JSON string (OpenAI wire format).
    """
    parsed_messages = []
    msg_string = ""
    for msg in messages:
        role = msg.get("role", "")
        if role == "assistant":
            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                fn = tool_calls[0].get("function", {})
                fn_name = fn.get("name", "tool")
                args = fn.get("arguments", {})
                if isinstance(args, dict):
                    args_str = json.dumps(args)
                else:
                    args_str = args
                msg_string += f"{content}\nTool({fn_name}): {args_str}"
            else:
                msg_string += content
        elif role == "tool":
            if "extra" in msg:
                raw_output = msg["extra"]["raw_output"]
            else:
                raw_output = msg.get("content", "")
            if len(raw_output) > 5000:
                raw_output = raw_output[:2500] + "\n...[truncated]..." + raw_output[-2500:]
            msg_string += f"\nEnvironment:\n{raw_output}"
            parsed_messages.append(msg_string)
            msg_string = ""
    return parsed_messages


def format_continuation(
    x_messages: list[dict],
    continuation_messages: list[dict],
    tokenizer,
) -> str:
    """
    Return the continuation text as it appears in the model's token stream.

    Applies the model's chat template to both the full conversation and the
    context alone, then returns the delta — the exact bytes the model would
    generate (and observe from the environment) after the x-context.

    This ensures tokenization matches what the model sees during inference,
    so KL estimates in score_completion are computed over meaningful tokens.
    """
    full_text = tokenizer.apply_chat_template(
        x_messages + continuation_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    ctx_text = tokenizer.apply_chat_template(
        x_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return full_text[len(ctx_text):]

def parse_messages_mask_env(messages):
    parsed_messages = []
    msg_string = ""
    for msg in messages:
        role = msg.get("role", "")
        if role == "assistant":
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                tool_arg = tool_calls[0]["function"]["arguments"]

            msg_string += f"{content}\nTool: {tool_arg}"
        elif role == "tool":
            msg_string += "\nEnvironment:\n[ENVIRONMENT OUTPUT MASKED]"
            parsed_messages.append(msg_string)
            msg_string = ""
    return parsed_messages