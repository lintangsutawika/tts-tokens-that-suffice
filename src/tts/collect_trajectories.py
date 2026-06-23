#!/usr/bin/env python3
"""
Collect mini-SWE-agent trajectories on SWE-bench/SWE-smith instances (phase 1 only).

Each instance is run once, the full message history is saved to a JSON file in
output_dir/{uuid}.json.  Instances with existing Submitted/LimitsExceeded
trajectories are skipped automatically.

Usage:
    uv run -m tts.collect_trajectories \\
        --dataset swe-smith --output /tmp/trajs -w 4 \\
        -c swebench.yaml -c model.model_name=Qwen/Qwen3-4B-Instruct-2507

    uv run -m tts.collect_trajectories \\
        --dataset swe-smith --slice 0:50 --output /tmp/trajs -w 8 \\
        -c swebench.yaml -m litellm_proxy/Qwen/Qwen3-8B
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import random
import re
import traceback
import uuid
from pathlib import Path

import litellm
import typer

litellm.drop_params = True
for _logger_name in ("litellm", "LiteLLM", "litellm.utils", "litellm.proxy"):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)

from rich.live import Live

from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.models import get_model
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.utils.log import add_file_handler, logger
from minisweagent.utils.serialize import UNSET, recursive_merge

from tts.utils.patch import apply_patch


DATASET_MAPPING = {
    "swe-smith": ("SWE-bench/SWE-smith-py", "train"),
    "swe-bench": ("SWE-bench/SWE-bench_Verified", "train"),
}

DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "swebench.yaml"

app = typer.Typer(rich_markup_mode="rich", add_completion=False)


# ---------------------------------------------------------------------------
# Progress-tracking agent (no summarizer needed for plain data collection)
# ---------------------------------------------------------------------------

class _ProgressAgent(DefaultAgent):
    def __init__(self, *args, progress_manager: RunBatchProgressManager, instance_id: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self._pm = progress_manager
        self._iid = instance_id

    def step(self) -> dict:
        try:
            self._pm.update_instance_status(self._iid, f"Step {self.n_calls + 1:3d} (${self.cost:.2f})")
        except KeyError:
            pass
        return super().step()


# ---------------------------------------------------------------------------
# Core per-instance worker
# ---------------------------------------------------------------------------

def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
    data_source: str = "swe-smith",
    capture_git_diffs: bool = True,
) -> str | None:
    """Run the agent on one instance and save the trajectory. Returns uid on success."""
    instance_id = instance["instance_id"]
    uid = str(uuid.uuid4())
    output_dir.mkdir(parents=True, exist_ok=True)

    cwd = config.get("environment", {}).get("cwd", "/testbed/")
    model = get_model(config=config.get("model", {}))
    task = instance["problem_statement"]

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Starting environment")

    agent = None
    exit_status = None
    result = None
    is_correct = 0.0
    evaluation = None
    error_info = {}

    agent_kwargs = dict(config.get("agent", {}))
    if capture_git_diffs:
        agent_kwargs["capture_git_diffs"] = True
        agent_kwargs["git_diff_cwd"] = cwd

    try:
        from tts.utils.mini_swe import evaluate_trajectory, get_sb_environment

        env = get_sb_environment(config, instance, data_source)
        if data_source == "swe-smith":
            env = apply_patch(env, instance["patch"], cwd)

        agent = _ProgressAgent(
            model, env,
            progress_manager=progress_manager,
            instance_id=instance_id,
            **agent_kwargs,
        )
        info = agent.run(task)
        exit_status = info.get("exit_status")
        result = info.get("submission")

        evaluation = evaluate_trajectory(
            instance=instance,
            model_patch=result or "",
            sweagent_config=config,
            data_source=data_source,
        )
        is_correct = float(evaluation.get("resolved", False))

    except Exception as e:
        logger.error(f"Error on {instance_id}: {e}", exc_info=True)
        exit_status = type(e).__name__
        result = ""
        error_info = {"traceback": traceback.format_exc(), "exception_str": str(e)}

    finally:
        if agent is not None:
            all_messages = agent.messages
            step_git_diffs = (
                agent.get_step_git_diffs() if hasattr(agent, "get_step_git_diffs") else []
            )
            traj_path = output_dir / f"{uid}.json"
            with open(traj_path, "w") as f:
                json.dump(
                    {
                        "uid": uid,
                        "instance_id": instance_id,
                        "exit_status": exit_status,
                        "is_correct": is_correct,
                        "evaluation": evaluation,
                        "num_calls": agent.n_calls,
                        "submission": result,
                        "messages": all_messages,
                        "step_git_diffs": step_git_diffs,
                        **error_info,
                    },
                    f,
                    indent=4,
                )
        progress_manager.on_instance_end(instance_id, exit_status)

    return uid if agent is not None else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_HELP_TEXT = """Collect mini-SWE-agent trajectories (phase 1 only — no branching)."""

_CONFIG_SPEC_HELP_TEXT = """Path to config files, filenames, or key-value pairs.

[bold red]IMPORTANT:[/bold red] The default config file is NOT added automatically.
Pass it explicitly, e.g. [bold green]-c swebench.yaml -c model.model_name=Qwen/Qwen3-4B[/bold green]
"""


@app.command(help=_HELP_TEXT)
def main(
    dataset: str = typer.Option("swe-smith", "--dataset", help="'swe-smith', 'swe-bench', or a HuggingFace path"),
    split: str = typer.Option("", "--split", help="Dataset split (inferred from --dataset if omitted)"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex"),
    slice_spec: str = typer.Option("", "--slice", help="Slice (e.g. '0:100')"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances before slicing"),
    output: str = typer.Option(..., "-o", "--output", help="Output directory"),
    workers: int = typer.Option(1, "-w", "--workers", help="Parallel worker threads"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model name"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class"),
    environment_class: str = typer.Option("singularity", "--environment-class", help="Environment type (docker/singularity)"),
    data_source: str = typer.Option("swe-smith", "--data-source", help="Evaluation harness: swe-smith or swe-bench"),
    capture_git_diffs: bool = typer.Option(True, "--capture-git-diffs/--no-capture-git-diffs", help="Capture git diff after each step"),
    runs_per_instance: int = typer.Option(1, "--runs-per-instance", help="Number of completed trajectories to collect per instance"),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", help=_CONFIG_SPEC_HELP_TEXT),
) -> None:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    add_file_handler(output_path / "collect.log")
    logger.info(f"Output: {output_path}")

    from datasets import load_dataset

    if dataset in DATASET_MAPPING:
        dataset_path, default_split = DATASET_MAPPING[dataset]
    else:
        dataset_path, default_split = dataset, "train"
    resolved_split = split or default_split

    logger.info(f"Loading {dataset_path} / {resolved_split} ...")
    instances = list(load_dataset(dataset_path, split=resolved_split))

    if shuffle:
        instances = sorted(instances, key=lambda x: x["instance_id"])
        random.seed(42)
        random.shuffle(instances)

    if slice_spec:
        parts = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*parts)]

    if filter_spec:
        before = len(instances)
        instances = [i for i in instances if re.match(filter_spec, i["instance_id"])]
        logger.info(f"Filter: {before} → {len(instances)} instances")

    # Count completed trajectories per instance; skip those that have enough
    _DONE_STATUSES = {"Submitted", "LimitsExceeded"}
    done_counts: dict[str, int] = {}
    for traj_file in output_path.glob("*.json"):
        if traj_file.stem in ("collect",):
            continue
        try:
            traj = json.loads(traj_file.read_text())
            msgs = traj.get("messages", [])
            if msgs and msgs[-1].get("role") == "exit":
                status = msgs[-1].get("extra", {}).get("exit_status", "")
                if status in _DONE_STATUSES:
                    iid = traj.get("instance_id", "")
                    done_counts[iid] = done_counts.get(iid, 0) + 1
        except Exception:
            pass

    before = len(instances)
    instances = [i for i in instances if done_counts.get(i["instance_id"], 0) < runs_per_instance]
    logger.info(f"Skipping {before - len(instances)} already-completed ({runs_per_instance} run(s) each); running {len(instances)}")

    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append({
        "environment": {"environment_class": environment_class or UNSET},
        "model": {"model_name": model or UNSET, "model_class": model_class or UNSET},
    })
    config = recursive_merge(*configs)

    progress_manager = RunBatchProgressManager(
        len(instances), output_path / "exit_statuses.yaml"
    )

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_instance,
                    instance, output_path, config, progress_manager,
                    data_source, capture_git_diffs,
                ): instance["instance_id"]
                for instance in instances
            }
            try:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except concurrent.futures.CancelledError:
                        pass
                    except Exception as e:
                        iid = futures[future]
                        logger.error(f"Uncaught error for {iid}: {e}", exc_info=True)
                        progress_manager.on_uncaught_exception(iid, e)
            except KeyboardInterrupt:
                logger.info("Cancelling pending jobs ...")
                for f in futures:
                    if not f.running() and not f.done():
                        f.cancel()
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception:
                        pass

    logger.info("Done.")


if __name__ == "__main__":
    app()
