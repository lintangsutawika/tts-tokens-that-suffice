from jinja2 import StrictUndefined, Template

def render_template(template: str, template_vars: dict) -> str:
    return Template(template, undefined=StrictUndefined).render(**template_vars)

def parse_messages(messages):
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
            raw_output = msg["extra"]["raw_output"]
            if len(raw_output) > 5000:
                raw_output = raw_output[:2500] + "\n...[truncated]..." + raw_output[-2500:]

            msg_string += f"\nEnvironment:\n{raw_output}"
            parsed_messages.append(msg_string)
            msg_string = ""
    return parsed_messages

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