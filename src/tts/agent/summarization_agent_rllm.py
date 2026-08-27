"""Single-turn summarization agent for rLLM RL training."""
import copy
from dataclasses import dataclass, field
from typing import Any

import litellm
from jinja2 import StrictUndefined, Template

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory

from era.utils.summary_utils import render_template, parse_messages

DEFAULT_INSTANCE_TEMPLATE = """\
Here are the past events so far.
<TASK>
{{ task }}
</TASK>
{% if previous_summary %}{% for summary in previous_summary %}
<PREVIOUS SUMMARY>
{{ summary }}
</PREVIOUS SUMMARY>
{% endfor %}{% endif %}{% for event in events %}
<EVENT>
{{ event }}
</EVENT>
{% endfor %}
Summarize the above to assist the agent in understanding its current state.
Suggest next steps clearly and concisely.
A good summary enables the agent to effectively choose its next actions.
Focus on key decisions, actions taken, and their outcomes.

Your summary will be directly provided to the agent to help it decide its next actions."""


@dataclass
class SummarizerConfig:
    system_template: str = ""
    instance_template: str = DEFAULT_INSTANCE_TEMPLATE


class SummarizationAgent(BaseAgent):
    """
    Single-turn agent that generates trajectory summaries for rLLM RL training.

    On each episode:
      - update_from_env() stores task/events/previous_summary from extra_info
      - chat_completions renders the config templates into the message list
        that rLLM's rollout engine uses to call the actor model
      - update_from_model() records the generated summary
    """

    def __init__(
        self,
        summarizer_model_name: str | None = None,
        summarizer_config: SummarizerConfig | None = None,
        summarizer_model_kwargs: dict | None = None,
    ):
        self.summarizer_model_name = summarizer_model_name
        self.summarizer_config = summarizer_config or SummarizerConfig()
        self.summarizer_model_kwargs = summarizer_model_kwargs or {}

        self._trajectory = Trajectory()
        self.messages = []
        self.task = ""
        self.events: list[dict] = []   # raw message dicts; parsed to strings in chat_completions
        self.summary_list: list[str] | None = None
        self.keep: int = 7

    def _render_template(self, template_str: str) -> str:
        return Template(template_str, undefined=StrictUndefined).render(
            task=self.task,
            events=self.events,
            previous_summary=self.previous_summary,
        )

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        if done:
            return

        # On reset, observation is the full task dict from the dataset row.
        # After a step, the task dict is in info.
        task_data = observation if (observation and "extra_info" in observation) else info
        extra = task_data.get("extra_info", {}) if isinstance(task_data, dict) else {}

        self.task = extra.get("task", "")
        self.events = extra.get("messages", [])   # raw dicts → parsed in chat_completions
        self.summary_list = extra.get("previous_summary")
        self.keep = extra.get("keep", 7)

    def update_from_model(self, response: str = None, **kwargs) -> Action:
        if response is None:
            response = self.generate_summary(**kwargs)

        return Action(action=response)

    def reset(self):
        self._trajectory = Trajectory()
        self.messages = []
        self.task = ""
        self.events = []
        self.summary_list = None
        self.keep = 7

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """
        Build the summarizer prompt messages from config templates.
        Mirrors DefaultAgentWithSummarizer.summarize_trajectory() message construction.
        Used by rLLM's rollout engine to call the actor model.
        """

        # target_length = 10
        # start_idx = 2
        # keep = target_length - start_idx - 1
        # event_list = messages[start_idx:-keep]
        # if messages[-keep].get("role") == "tool":
        #     event_list.append(messages[-keep])
        #     keep += 1

        template_vars = {
            "previous_summary": self.summary_list,
            "task": self.task, 
            "events": parse_messages(self.events),  # Exclude initial system message
            }

        user_message = {"role": "user", "content": render_template(self.summarizer_config.instance_template, template_vars=template_vars)}

        if self.summarizer_config.system_template != "":
            system_message = {"role": "system", "content": render_template(self.summarizer_config.system_template, template_vars=template_vars)}
            summary_message = [system_message, user_message]
        else:
            summary_message = [user_message]

        return summary_message

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    def get_current_state(self) -> Step:
        assert self._trajectory.steps, "Trajectory should not be empty."
        return self._trajectory.steps[-1]

    def generate_summary(self, **kwargs) -> str:
        """Call litellm with chat_completions and return the generated summary text."""
        summary_response = litellm.completion(
            model=self.summarizer_model_name,
            messages=self.chat_completions,
            **{**self.summarizer_model_kwargs, **kwargs},
        )
        return summary_response.choices[0].message.content
