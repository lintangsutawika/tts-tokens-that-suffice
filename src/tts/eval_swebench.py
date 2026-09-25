#!/usr/bin/env python3
"""
Downstream SWE-bench evaluation with a summarizer compressing the agent's
context *in the loop*.

The thesis of "Tokens That Suffice" is that a learned summary z can replace the
full context x while keeping the continuation just as likely.  This script tests
that end-to-end: run the mini-SWE-agent on SWE-bench instances and, whenever the
running context grows past a token budget, replace the middle of the message
history with a summary produced by the trained summarizer.  We then grade the
final patch and report the resolve rate.

Four arms (--mode):
  full        no compression at all; the agent runs with its complete growing
              context until it finishes or hits a step/cost/context limit
              (the true do-nothing baseline)
  trained     summarizer = a trained tinker LoRA checkpoint (--checkpoint)
  base        summarizer = the untrained base model, same prompt, no RL
  truncation  no summarizer; keep the first `keep_first` messages + the last
              `keep_last_turns` turns, dropping the middle (a cheap-compression baseline)

The *deliberator* (task-solving) model is identical across arms — only how the
context is compressed differs — so any resolve-rate gap is attributable to the
summarizer.  The deliberator is built via the exact same `get_model(config)`
path as tts.collect_trajectories, so it behaves identically to data collection.

Compression is faithful to training (see tts.reward.utils.build_z_scoring_messages):
  new_messages = messages[:keep_first]                       # sys, task, first turn
               + [user("<summary>\\n{z}\\n</summary>")]        # z replaces the middle
               + last_n_turns(messages[keep_first:], keep_last_turns)
and z is generated from the same prompt used in training
(SYSTEM_PROMPT + format_trajectory_text(steps)).  On a second compression the
previous <summary> message is fed back in as an event, so information already
compressed is not lost.

Output follows mini-swe-agent's SWE-bench runner:
  output_dir/preds.json                     SWE-bench predictions {iid: {model_name_or_path, instance_id, model_patch}}
  output_dir/{instance_id}/{iid}.traj.json  full trajectory (+ a `summarizer` block: mode, n_compressions, resolved)
  output_dir/results_summary.json           aggregate compression / resolve-rate summary
Grading is decoupled (mini-swe grades preds.json with the SWE-bench harness). The
official harness needs docker; on this singularity cluster we grade in-process via
tts.utils.mini_swe (default). Use --no-grade to emit predictions only, then grade
preds.json elsewhere, e.g.:
    python -m swebench.harness.run_evaluation --predictions_path preds.json \\
        --run_id eval --dataset_name SWE-bench/SWE-bench_Verified

Usage:
    uv run -m tts.eval_swebench \\
        --dataset swe-bench --data-source swe-bench --slice 0:20 \\
        --mode trained --checkpoint tinker://model_7cf52d89/weights/000084 \\
        --output outputs/eval-swebench-trained -w 4 \\
        -c swebench.yaml -m litellm_proxy/Qwen/Qwen3.6-35B-A3B \\
        -c model.model_kwargs.api_base=http://0.0.0.0:8000/v1
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import random
import re
import threading
import traceback
from pathlib import Path

import litellm
import typer

litellm.drop_params = True
for _logger_name in ("litellm", "LiteLLM", "litellm.utils", "litellm.proxy"):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)

from rich.live import Live

from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.models import get_model
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.utils.log import add_file_handler, logger
from minisweagent.utils.serialize import UNSET, recursive_merge

from tts.agent.summarization_agent import SummarizingAgent
from tts.data.agent_trajectory import TrajectoryStep
from tts.summarization import (
    MaskBasedSummarizer,
    ModelBasedSummarizer,
    TruncationBasedSummarizer,
)
from tts.summarization.model_based import build_summarizer, hf_id
from tts.utils.patch import apply_patch


DATASET_MAPPING = {
    "swe-smith": ("SWE-bench/SWE-smith-py", "train"),
    "swe-bench": ("SWE-bench/SWE-bench_Verified", "test"),
    "swe-bench-lite": ("SWE-bench/SWE-bench_Lite", "test"),
    # 50-instance subset of SWE-bench Verified (same schema & instance_ids), so
    # it grades with --data-source swe-bench like the full Verified set.
    "swe-bench-verified-mini": ("MariusHobbhahn/swe-bench-verified-mini", "test"),
}

DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "swebench.yaml"

app = typer.Typer(rich_markup_mode="rich", add_completion=False)


# ---------------------------------------------------------------------------
# Summarizer: generate z from the trajectory-so-far via a tinker sampling client
# ---------------------------------------------------------------------------

def preflight_check(config: dict, compactor, mode: str) -> None:
    """Ping the deliberator and summarizer endpoints before launching the run.

    Fails fast with a clear error if either is unreachable/misconfigured, rather
    than losing a 500-instance run to a server that was down or a missing
    credential. A compactor with no `.summarizer` compacts without generating
    (full/truncation/mask), so there is nothing to ping.
    """
    model_cfg = config.get("model", {})
    model_name = model_cfg.get("model_name")
    mk = dict(model_cfg.get("model_kwargs", {}))
    api_base = mk.get("api_base")

    logger.info(f"Preflight: pinging deliberator {model_name} @ {api_base} ...")
    try:
        litellm.completion(
            model=model_name,
            messages=[{"role": "user", "content": "ping"}],
            api_base=api_base,
            api_key=mk.get("api_key"),
            max_tokens=1,
        )
    except Exception as e:
        raise RuntimeError(
            f"Deliberator preflight FAILED for {model_name} @ {api_base}: {e}"
        ) from e
    logger.info("Preflight: deliberator OK")

    summarizer = getattr(compactor, "summarizer", None)
    if summarizer is None:
        logger.info(f"Preflight: mode={mode} compacts without generating — nothing to ping")
        return

    logger.info(f"Preflight: pinging summarizer (mode={mode}) ...")
    try:
        z = summarizer.summarize([TrajectoryStep(role="assistant", content="ping")])
    except Exception as e:
        raise RuntimeError(
            f"Summarizer preflight FAILED (mode={mode}): {e}"
        ) from e
    logger.info(f"Preflight: summarizer OK (returned {len(z)} chars)")


# ---------------------------------------------------------------------------
# Message <-> step conversion (live agent messages -> summarizer input)
# ---------------------------------------------------------------------------

def _message_text(m: dict) -> str:
    """Rough text of a message for token counting (content + tool-call args)."""
    parts = [m.get("content") or ""]
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function", {})
        args = fn.get("arguments", "")
        parts.append(fn.get("name", ""))
        parts.append(args if isinstance(args, str) else json.dumps(args))
    return "\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Summarizing agent
# ---------------------------------------------------------------------------

# Every arm the eval knows about. build_compactor is the only interpreter; the
# CLI validator and --mode help text are derived from this so they cannot drift.
MODES = ("full", "truncation", "mask", "base", "trained")


def build_compactor(mode: str, **kwargs):
    """Build the compaction strategy for an eval arm.

    Sole interpreter of `mode`: it decides both which strategy to use and, for
    the generating arms, constructs the summarizer that backs it. Keeping that
    in one place is deliberate — when `mode` was dispatched here *and* in
    build_summarizer, adding an arm to one and not the other silently produced
    a summarizer that was never used.

    `full` returns None (no compaction). kwargs are forwarded to build_summarizer
    and are only consulted by the `base`/`trained` arms.

    Build once and share across workers: the tinker arms open a sampling client.
    """
    if mode == "full":
        return None
    if mode == "truncation":
        return TruncationBasedSummarizer()
    if mode == "mask":
        return MaskBasedSummarizer()
    if mode in ("base", "trained"):
        # mode and checkpoint must agree; previously a checkpoint passed with
        # mode=base was silently discarded.
        checkpoint = kwargs.get("checkpoint", "")
        if mode == "trained" and not checkpoint:
            raise ValueError("--checkpoint is required for --mode trained")
        if mode == "base" and checkpoint:
            raise ValueError("--checkpoint is not valid for --mode base")
        return ModelBasedSummarizer(build_summarizer(**kwargs))
    raise ValueError(f"Unknown mode: {mode!r}")


# ---------------------------------------------------------------------------
# Predictions file (SWE-bench standard format, as mini-swe-agent writes it)
# ---------------------------------------------------------------------------

_OUTPUT_FILE_LOCK = threading.Lock()


def update_preds_file(preds_path: Path, instance_id: str, model_name: str, result: str) -> None:
    """Add/replace one instance in the SWE-bench predictions file (thread-safe)."""
    with _OUTPUT_FILE_LOCK:
        data = json.loads(preds_path.read_text()) if preds_path.exists() else {}
        data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": result,
        }
        preds_path.write_text(json.dumps(data, indent=2))


def remove_from_preds_file(preds_path: Path, instance_id: str) -> None:
    """Drop one instance from the predictions file (avoids stale state on retry)."""
    if not preds_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        data = json.loads(preds_path.read_text())
        if instance_id in data:
            del data[instance_id]
            preds_path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Per-instance worker
# ---------------------------------------------------------------------------

def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
    compactor,
    tokenizer,
    mode: str,
    compress_at_tokens: int,
    compress_at_turns: int,
    keep_first: int,
    keep_last_turns: int,
    run_name: str,
    grade: bool = True,
    data_source: str = "swe-bench",
) -> str | None:
    """Run the summarizing agent on one instance; write its prediction + trajectory.

    Output follows mini-swe-agent's SWE-bench runner:
      * output_dir/preds.json                       standard predictions file
      * output_dir/{instance_id}/{iid}.traj.json    full trajectory (+ summarizer info)
    Grading is optional (default on): the official SWE-bench harness needs docker,
    so on a singularity cluster we grade in-process via tts.utils.mini_swe; pass
    --no-grade to produce predictions only and grade preds.json elsewhere.
    """
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    preds_path = output_dir / "preds.json"

    # Clear any stale prediction / trajectory / compactions so a retry starts clean.
    remove_from_preds_file(preds_path, instance_id)
    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)
    import shutil
    shutil.rmtree(instance_dir / "compressions", ignore_errors=True)

    cwd = config.get("environment", {}).get("cwd", "/testbed/")
    model = get_model(config=config.get("model", {}))
    task = instance["problem_statement"]

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Starting environment")

    agent = None
    exit_status = None
    result = None
    resolved = None
    evaluation = None
    error_info = {}

    try:
        from tts.utils.mini_swe import evaluate_trajectory, get_sb_environment

        env = get_sb_environment(config, instance, data_source)
        if data_source == "swe-smith":
            env = apply_patch(env, instance["patch"], cwd)

        agent = SummarizingAgent(
            model, env,
            compactor=compactor,
            tokenizer=tokenizer,
            compress_at_tokens=compress_at_tokens,
            compress_at_turns=compress_at_turns,
            keep_first=keep_first,
            keep_last_turns=keep_last_turns,
            progress_manager=progress_manager,
            instance_id=instance_id,
            compressions_dir=instance_dir / "compressions",
            **dict(config.get("agent", {})),
        )
        info = agent.run(task)
        exit_status = info.get("exit_status")
        result = info.get("submission")

        if grade:
            progress_manager.update_instance_status(instance_id, "Grading")
            evaluation = evaluate_trajectory(
                instance=instance,
                model_patch=result or "",
                sweagent_config=config,
                data_source=data_source,
            )
            resolved = bool(evaluation.get("resolved", False))

    except Exception as e:
        logger.error(f"Error on {instance_id}: {e}", exc_info=True)
        exit_status = type(e).__name__
        result = ""
        error_info = {"traceback": traceback.format_exc(), "exception_str": str(e)}

    finally:
        if agent is not None:
            comps = getattr(agent, "compressions", [])
            # Each compaction's full record (incl. input_messages) is written to
            # its own file under {instance_dir}/compressions/ as it fires. The
            # .traj.json keeps only the lightweight per-compaction metadata.
            comps_light = [{k: v for k, v in c.items() if k != "input_messages"} for c in comps]

            traj_path = instance_dir / f"{instance_id}.traj.json"
            agent.save(
                traj_path,
                {
                    "info": {"exit_status": exit_status, "submission": result, **error_info},
                    "instance_id": instance_id,
                    "summarizer": {
                        "mode": mode,
                        "n_compressions": getattr(agent, "n_compressions", 0),
                        "n_summaries": sum(1 for c in comps if c["kind"] == "summary"),
                        "compressions": comps_light,
                        "context_tokens_per_turn": getattr(agent, "context_tokens", []),
                        "resolved": resolved,
                        "evaluation": evaluation,
                    },
                },
            )
        update_preds_file(preds_path, instance_id, run_name, result or "")
        progress_manager.on_instance_end(instance_id, exit_status)

    return instance_id if agent is not None else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_HELP_TEXT = """Downstream SWE-bench eval with a summarizer compressing context in-loop."""

_CONFIG_SPEC_HELP_TEXT = """Path to config files, filenames, or key-value pairs.

[bold red]IMPORTANT:[/bold red] The default config file is NOT added automatically.
Pass it explicitly, e.g. [bold green]-c swebench.yaml -c model.model_name=Qwen/Qwen3-8B[/bold green]
"""


@app.command(help=_HELP_TEXT)
def main(
    dataset: str = typer.Option("swe-bench", "--dataset", help="'swe-bench', 'swe-bench-lite', 'swe-bench-verified-mini', 'swe-smith', or a HuggingFace path"),
    split: str = typer.Option("", "--split", help="Dataset split (inferred from --dataset if omitted)"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex"),
    slice_spec: str = typer.Option("", "--slice", help="Slice (e.g. '0:20')"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances before slicing"),
    output: str = typer.Option(..., "-o", "--output", help="Output directory"),
    workers: int = typer.Option(1, "-w", "--workers", help="Parallel worker threads"),
    model: str | None = typer.Option(None, "-m", "--model", help="Deliberator model name"),
    model_class: str | None = typer.Option(None, "--model-class", help="Deliberator model class"),
    environment_class: str = typer.Option("singularity", "--environment-class", help="Environment type (docker/singularity)"),
    data_source: str = typer.Option("swe-bench", "--data-source", help="Evaluation harness: swe-bench or swe-smith"),
    grade: bool = typer.Option(True, "--grade/--no-grade", help="Grade in-process (singularity); --no-grade writes predictions only"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Re-run instances already present in preds.json"),
    skip_preflight: bool = typer.Option(False, "--skip-preflight", help="Skip the deliberator/summarizer connectivity check"),
    # --- summarizer / compression knobs ---
    mode: str = typer.Option("trained", "--mode", help="full | truncation | mask | base | trained"),
    checkpoint: str = typer.Option("", "--checkpoint", help="tinker:// state path for --mode trained"),
    summarizer_model: str = typer.Option("Qwen/Qwen3-8B", "--summarizer-model", help="Summarizer model: HF id for tinker, or litellm string (e.g. openai/Qwen/Qwen3-8B, openai/<lora-name>) for --summarizer-api-base"),
    summarizer_tokenizer: str = typer.Option("Qwen/Qwen3-8B", "--summarizer-tokenizer", help="HF tokenizer for the compression-trigger token count (base model; the served LoRA name is not a HF repo)"),
    summarizer_renderer: str = typer.Option("qwen3_disable_thinking", "--summarizer-renderer", help="Renderer for the summarizer (tinker backend only)"),
    tinker_base_url: str = typer.Option("http://localhost:9123", "--tinker-base-url", help="tinker server URL (tinker backend)"),
    summarizer_api_base: str = typer.Option("", "--summarizer-api-base", help="Serve the summarizer over this vLLM/OpenAI endpoint instead of tinker (scripts/serve/serve_summarizer.sh)"),
    summarizer_max_tokens: int = typer.Option(512, "--summarizer-max-tokens", help="Max tokens for a generated summary"),
    compress_at_tokens: int = typer.Option(24000, "--compress-at-tokens", help="Compress when context exceeds this many tokens (used when --compress-at-turns is 0)"),
    compress_at_turns: int = typer.Option(0, "--compress-at-turns", help="Compress when the context reaches this many complete turns; >0 overrides the token trigger (matches training/OpenHands count-based condensation)"),
    keep_first: int = typer.Option(4, "--keep-first", help="Messages kept verbatim from the start"),
    keep_last_turns: int = typer.Option(3, "--keep-last-turns", help="Complete turns kept verbatim at the end"),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", help=_CONFIG_SPEC_HELP_TEXT),
) -> None:
    if mode not in MODES:
        raise typer.BadParameter(f"--mode must be one of: {', '.join(MODES)}")

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    add_file_handler(output_path / "eval.log")
    logger.info(f"Output: {output_path}  | mode={mode}  checkpoint={checkpoint or '-'}")

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

    # Skip instances already present in preds.json (mini-swe-agent behaviour);
    # --redo-existing re-runs them.
    preds_path = output_path / "preds.json"
    if not redo_existing and preds_path.exists():
        existing = set(json.loads(preds_path.read_text()).keys())
        before = len(instances)
        instances = [i for i in instances if i["instance_id"] not in existing]
        logger.info(f"Skipping {before - len(instances)} in preds.json; running {len(instances)}")
    else:
        logger.info(f"Running {len(instances)} instances")

    run_name = f"{model or 'deliberator'}__summ-{mode}"

    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append({
        "environment": {"environment_class": environment_class or UNSET},
        "model": {"model_name": model or UNSET, "model_class": model_class or UNSET},
    })
    config = recursive_merge(*configs)

    # Build the compactor once and share it across worker threads (the tinker
    # arms open a sampling client, which must not be created per instance).
    compactor = build_compactor(
        mode,
        tinker_base_url=tinker_base_url,
        checkpoint=checkpoint,
        summarizer_model=summarizer_model,
        renderer_name=summarizer_renderer,
        max_tokens=summarizer_max_tokens,
        summarizer_api_base=summarizer_api_base,
    )
    # Tokenizer for the compression trigger (base-model vocab; the served LoRA name
    # like "summarizer-rl" is not a HF repo, so use the explicit base tokenizer).
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    tokenizer = get_tokenizer(hf_id(summarizer_tokenizer))

    # Fail fast if an endpoint is unreachable before spinning up any containers.
    if not skip_preflight:
        preflight_check(config, compactor, mode)

    progress_manager = RunBatchProgressManager(
        len(instances), output_path / "exit_statuses.yaml"
    )

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_instance,
                    instance, output_path, config, progress_manager,
                    compactor, tokenizer, mode, compress_at_tokens, compress_at_turns,
                    keep_first, keep_last_turns, run_name, grade, data_source,
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

    _write_summary(output_path, mode, graded=grade)
    logger.info("Done.")


def _write_summary(output_path: Path, mode: str, graded: bool) -> None:
    """Aggregate the per-instance trajectories into a compression/resolve summary.

    Reads output_dir/{instance_id}/{iid}.traj.json (the summarizer block carries
    mode / n_compressions / resolved). When graded, also reports the resolve rate.
    """
    records = []
    for traj_file in output_path.glob("*/*.traj.json"):
        try:
            rec = json.loads(traj_file.read_text())
        except Exception:
            continue
        records.append(rec.get("summarizer", {}))
    n = len(records)
    n_compress = [r.get("n_compressions", 0) for r in records]
    n_summ = [r.get("n_summaries", 0) for r in records]
    # Input context size at each compression trigger, across all instances.
    in_tokens = [c.get("tokens_before", 0)
                 for r in records for c in r.get("compressions", [])]
    summary = {
        "mode": mode,
        "n_instances": n,
        "mean_compressions": sum(n_compress) / n if n else 0.0,
        "instances_compressed": sum(1 for c in n_compress if c > 0),
        "instances_with_summary": sum(1 for c in n_summ if c > 0),
        "total_summaries": sum(n_summ),
        "mean_input_tokens": sum(in_tokens) / len(in_tokens) if in_tokens else 0.0,
        "max_input_tokens": max(in_tokens) if in_tokens else 0,
    }
    if graded:
        resolved = sum(1 for r in records if r.get("resolved"))
        summary["n_resolved"] = resolved
        summary["resolve_rate"] = resolved / n if n else 0.0
    (output_path / "results_summary.json").write_text(json.dumps(summary, indent=2))
    resolve_str = (
        f"resolved {summary['n_resolved']}/{n} = {summary['resolve_rate']:.3f} | "
        if graded else "predictions only (not graded) | "
    )
    logger.info(
        f"RESULTS [{mode}]: {resolve_str}"
        f"mean compressions {summary['mean_compressions']:.2f} "
        f"| {summary['instances_compressed']}/{n} instances compressed"
    )


if __name__ == "__main__":
    app()
