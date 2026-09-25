#!/usr/bin/env python3
"""Grade an existing preds.json separately (decoupled from the agent run).

`tts.eval_swebench --no-grade` writes predictions only; this script grades them
later. It reads `<output>/preds.json`, loads the dataset for the gold tests, and
for each instance runs the SWE-bench eval_script in a singularity container via
`tts.utils.mini_swe.evaluate_trajectory` — the same in-process grader the eval
uses (the official docker harness is unavailable on this cluster).

Results are written to `<output>/report.json` ({resolved: [...], unresolved: [...],
resolve_rate, per_instance: {...}}) and each instance's `.traj.json` is updated
in place (summarizer.resolved / summarizer.evaluation), so re-grading is idempotent.

Usage:
  python scripts/eval/grade_preds.py outputs/trained-Qwen3.6-35B-A3B
  python scripts/eval/grade_preds.py outputs/trained-Qwen3.6-35B-A3B -w 8
  python scripts/eval/grade_preds.py <outdir> --skip-graded   # only grade the unscored ones
  python scripts/eval/grade_preds.py <outdir> --dataset swe-bench-lite
  DELIBERATOR/summarizer servers are NOT needed — grading only touches containers.
"""
from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path

import typer

from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.utils.serialize import recursive_merge
from minisweagent.utils.log import logger

from tts.eval_swebench import DATASET_MAPPING
from tts.utils.mini_swe import evaluate_trajectory

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


def _grade_one(instance: dict, model_patch: str, config: dict, data_source: str) -> dict:
    iid = instance["instance_id"]
    try:
        ev = evaluate_trajectory(
            instance=instance,
            model_patch=model_patch or "",
            sweagent_config=config,
            data_source=data_source,
        )
        return {"instance_id": iid, "resolved": bool(ev.get("resolved", False)), "evaluation": ev}
    except Exception as e:
        logger.error(f"Grading {iid} failed: {e}")
        return {"instance_id": iid, "resolved": False, "evaluation": {"eval_error": str(e)}}


def _cached_grade(output_path: Path, iid: str) -> dict | None:
    """Return a previous grade for iid, or None if it was never successfully graded.

    A cached `resolved: false` is a real score and is honoured. An instance that
    was never graded (`resolved: null`) or whose grading itself errored (the
    container failed to start, the image pull timed out) has no score, so it is
    reported as ungraded and will be re-graded.

    Note `evaluation["eval_error"]` is overloaded by evaluate_trajectory: on
    success it holds the test report (a dict); only a string is an actual error.
    """
    traj = output_path / iid / f"{iid}.traj.json"
    if not traj.exists():
        return None
    try:
        summ = json.loads(traj.read_text()).get("summarizer", {})
    except Exception:
        return None
    resolved, evaluation = summ.get("resolved"), summ.get("evaluation") or {}
    if resolved is None or isinstance(evaluation.get("eval_error"), str):
        return None
    return {"instance_id": iid, "resolved": bool(resolved), "evaluation": evaluation}


def _update_traj(output_path: Path, iid: str, resolved: bool, evaluation: dict) -> None:
    traj = output_path / iid / f"{iid}.traj.json"
    if not traj.exists():
        return
    j = json.loads(traj.read_text())
    summ = j.setdefault("summarizer", {})
    summ["resolved"] = resolved
    summ["evaluation"] = evaluation
    traj.write_text(json.dumps(j))


@app.command()
def main(
    output: str = typer.Argument(..., help="Eval output dir containing preds.json"),
    dataset: str = typer.Option("swe-bench", "--dataset", help="'swe-bench', 'swe-smith', or a HF path"),
    split: str = typer.Option("", "--split", help="Dataset split (inferred if omitted)"),
    data_source: str = typer.Option("swe-bench", "--data-source", help="Grader: swe-bench or swe-smith"),
    environment_class: str = typer.Option("singularity", "--environment-class"),
    workers: int = typer.Option(4, "-w", "--workers", help="Parallel grading threads"),
    skip_graded: bool = typer.Option(
        False, "--skip-graded", help="Reuse existing scores; only grade instances not yet graded"
    ),
    config_spec: list[str] = typer.Option(
        [str(builtin_config_dir / "benchmarks" / "swebench.yaml")], "-c", "--config"
    ),
) -> None:
    output_path = Path(output)
    preds_path = output_path / "preds.json"
    if not preds_path.exists():
        raise typer.BadParameter(f"No preds.json in {output_path}")
    preds = json.loads(preds_path.read_text())

    if dataset in DATASET_MAPPING:
        dataset_path, default_split = DATASET_MAPPING[dataset]
    else:
        dataset_path, default_split = dataset, "train"

    from datasets import load_dataset
    logger.info(f"Loading {dataset_path} / {split or default_split} ...")
    by_id = {i["instance_id"]: i for i in load_dataset(dataset_path, split=split or default_split)}

    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append({"environment": {"environment_class": environment_class}})
    config = recursive_merge(*configs)

    todo = [iid for iid in preds if iid in by_id]
    missing = [iid for iid in preds if iid not in by_id]
    if missing:
        logger.warning(f"{len(missing)} preds not found in dataset (skipped): {missing[:5]}...")

    results: dict[str, dict] = {}
    if skip_graded:
        cached = {iid: g for iid in todo if (g := _cached_grade(output_path, iid))}
        results.update(cached)
        todo = [iid for iid in todo if iid not in cached]
        logger.info(f"Reusing {len(cached)} existing grade(s); {len(todo)} left to grade")
    logger.info(f"Grading {len(todo)} instance(s) with {workers} worker(s) ...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(_grade_one, by_id[iid], preds[iid].get("model_patch", ""), config, data_source): iid
            for iid in todo
        }
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result()
            iid = r["instance_id"]
            results[iid] = r
            _update_traj(output_path, iid, r["resolved"], r["evaluation"])
            logger.info(f"  {iid}: {'RESOLVED' if r['resolved'] else 'unresolved'}")

    resolved = sorted(i for i, r in results.items() if r["resolved"])
    unresolved = sorted(i for i, r in results.items() if not r["resolved"])
    n = len(results)
    report = {
        "n": n,
        "n_resolved": len(resolved),
        "resolve_rate": len(resolved) / n if n else 0.0,
        "resolved": resolved,
        "unresolved": unresolved,
        "per_instance": {i: {"resolved": r["resolved"]} for i, r in results.items()},
    }
    (output_path / "report.json").write_text(json.dumps(report, indent=2))
    logger.info(
        f"Done. resolved {report['n_resolved']}/{n} = {report['resolve_rate']:.3f} "
        f"-> {output_path / 'report.json'}"
    )


if __name__ == "__main__":
    app()
