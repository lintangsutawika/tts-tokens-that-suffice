from minisweagent.agents.default import DefaultAgent
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager

from era.agent.agent_with_summarizer import DefaultAgentWithSummarizer


class ProgressTrackingMixin:
    """Mixin that adds progress updates to any agent class."""

    def __init__(self, *args, progress_manager: RunBatchProgressManager, instance_id: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_manager: RunBatchProgressManager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        try:
            self.progress_manager.update_instance_status(self.instance_id, f"Step {self.n_calls + 1:3d} (${self.cost:.2f})")
        except KeyError:
            pass
        return super().step()


class ProgressTrackingAgent(ProgressTrackingMixin, DefaultAgent):
    pass


class ProgressTrackingAgentWithSummarizer(ProgressTrackingMixin, DefaultAgentWithSummarizer):
    pass