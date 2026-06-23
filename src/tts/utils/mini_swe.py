from typing import TypedDict, Optional
import traceback
import uuid

from typing import Dict, Any
from loguru import logger


from jinja2 import Template

from swebench.harness.constants import DOCKER_WORKDIR, START_TEST_OUTPUT, END_TEST_OUTPUT
from swebench.harness.grading import (
    MAP_REPO_TO_PARSER,
    get_eval_tests_report as swebench_get_eval_tests_report,
    get_resolution_status as swebench_get_resolution_status,
)
from swebench.harness.test_spec.test_spec import make_test_spec
from swesmith.profiles import registry
from swesmith.constants import (
    TEST_OUTPUT_START,
    TEST_OUTPUT_END,
)

from swebench.harness.constants import (
    ResolvedStatus,
)
from swesmith.harness.grading import get_eval_tests_report, get_resolution_status
from minisweagent.environments import Environment, get_environment

from tts.utils.patch import apply_patch


class MiniSWEEvaluationResult(TypedDict):
    instance_id: str
    resolved: bool
    eval_error: Optional[str]


def _swesmith_sif_path(iid: str) -> str | None:
    """
    Return the path to a pre-cached Singularity .sif file for a SWE-smith instance,
    or None if no cache directory is configured or no matching file is found.

    SWE-smith instance IDs have the form: owner__repo.commit.functype__testid
    The cached .sif files are keyed on owner__repo.commit only, using the naming
    convention from lintangsutawika/agent-swe-smith:
        {sif_dir}/lintangsutawika_agent-swe-smith_{commit}-jyangballin_s_swesmith.x86_64.{base_compat}-source-minimal.sif
    where base_compat = "owner_1776_repo.commit" (dropping the .functype__testid suffix).

    Set SWESMITH_SIF_DIR to the cache directory and SWESMITH_IMAGE_COMMIT to the
    image commit tag (default: 62d35bb) to enable .sif lookup.
    """
    import os, re
    from pathlib import Path

    sif_dir = os.environ.get("SWESMITH_SIF_DIR", "")
    if not sif_dir:
        return None

    image_commit = os.environ.get("SWESMITH_IMAGE_COMMIT", "62d35bb")

    # Extract base instance: owner__repo.commit (drop .functype__testid)
    m = re.match(r"^(.+?\.[a-f0-9]{6,})", iid)
    base_iid = m.group(1) if m else iid
    base_compat = base_iid.replace("__", "_1776_").lower()

    sif_name = f"lintangsutawika_agent-swe-smith_{image_commit}-jyangballin_s_swesmith.x86_64.{base_compat}-source-minimal.sif"
    sif_path = Path(sif_dir) / sif_name
    return str(sif_path) if sif_path.exists() else None


def get_docker_image_name(instance: dict, data_source: str = "swe-bench") -> str:
    """Get the image name for a SWEBench instance."""
    image_name = instance.get("image_name", None)
    if image_name is None:
        # Docker doesn't allow double underscore, so we replace them with a magic token
        iid = instance["instance_id"]
        if "swe-gym" in data_source.lower():
            id_docker_compatible = iid.replace("__", "_s_")
            image_name = f"docker.io/xingyaoww/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
        elif "swe-bench" in data_source.lower():
            id_docker_compatible = iid.replace("__", "_1776_")
            image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
        elif "swe-smith" in data_source.lower():
            import re
            m = re.match(r"^(.+?\.[a-f0-9]{6,8})", iid)
            base_iid = m.group(1) if m else iid
            id_docker_compatible = base_iid.replace("__", "_1776_")
            image_name = f"docker.io/jyangballin/swesmith.x86_64.{id_docker_compatible}:latest".lower()
        else:
            raise NotImplementedError(f"Data source: {data_source} is not supported")
    # if not image_name.startswith("docker.io/"):
    #     image_name = f"docker.io/{image_name}"
    return image_name


def get_sb_environment(config: dict, instance: dict, data_source: str, max_retries: int = 3) -> Environment:
    import subprocess
    env_config = config.setdefault("environment", {})
    env_config["environment_class"] = env_config.get("environment_class", "singularity")

    # Prefer pre-cached .sif files (avoids Docker Hub rate limits)
    sif_path = None
    if "swe-smith" in data_source.lower():
        sif_path = _swesmith_sif_path(instance["instance_id"])

    if sif_path:
        env_config["image"] = sif_path
    else:
        image_name = get_docker_image_name(instance, data_source)
        if env_config["environment_class"] == "docker":
            env_config["image"] = image_name
        elif env_config["environment_class"] == "singularity":
            env_config["image"] = f"docker://{image_name}"

    for attempt in range(max_retries):
        try:
            env = get_environment(env_config)
            if startup_command := config.get("run", {}).get("env_startup_command"):
                startup_command = Template(startup_command).render(**instance)
                out = env.execute({"command": startup_command})
                if out["returncode"] != 0:
                    raise RuntimeError(f"Error executing startup command: {out}")
            return env
        except Exception as e:
            is_build_failure = "returned non-zero exit status 255" in str(e) or "stale NFS" in str(e)
            if is_build_failure and attempt < max_retries - 1:
                logger.warning(f"Singularity build failed on attempt {attempt + 1}, cleaning apptainer cache and retrying...")
                subprocess.run(["apptainer", "cache", "clean", "--force"], capture_output=True)
            else:
                raise


def _evaluate_swebench(env, instance, cwd):
    """Run swebench's eval_script in the container and return grading report."""
    test_spec = make_test_spec(instance)
    eval_cmd = f"bash <<'EVALEOF'\n{test_spec.eval_script}\nEVALEOF"
    obs = env.execute({"command": eval_cmd}, cwd=cwd, timeout=3600)
    content = obs["output"]

    start_idx = content.find(START_TEST_OUTPUT)
    end_idx = content.find(END_TEST_OUTPUT)
    if start_idx == -1 or end_idx == -1:
        test_content = content
    else:
        test_content = content[start_idx + len(START_TEST_OUTPUT):end_idx]

    log_parser = MAP_REPO_TO_PARSER.get(instance["repo"])
    if log_parser is None:
        logger.warning(f"No log parser for repo {instance['repo']}, assuming unresolved")
        return {"FAIL_TO_PASS": {"success": [], "failure": list(test_spec.FAIL_TO_PASS)}, "PASS_TO_PASS": {"success": [], "failure": list(test_spec.PASS_TO_PASS)}}

    status_map = log_parser(test_content, test_spec)
    gold = {"FAIL_TO_PASS": test_spec.FAIL_TO_PASS, "PASS_TO_PASS": test_spec.PASS_TO_PASS}
    return swebench_get_eval_tests_report(status_map, gold)


def _evaluate_swesmith(env, instance, cwd):
    """Run swesmith evaluation in the container and return grading report."""
    profile = registry.get_from_inst(instance)
    f2p_files, p2p_files = profile.get_test_files(instance)
    test_files = " ".join(f2p_files + p2p_files)
    if test_files:
        env.execute({"command": f"git checkout -- {test_files}"}, cwd=cwd)
    test_command, _ = profile.get_test_cmd(instance)
    eval_script = "\n".join([
        "#!/bin/bash",
        "set -uxo pipefail",
        f"cd {DOCKER_WORKDIR}",
        f": '{TEST_OUTPUT_START}'",
        test_command,
        f": '{TEST_OUTPUT_END}'",
    ]) + "\n"

    obs = env.execute({"command": f"bash <<'EOF'\n{eval_script}\nEOF"}, cwd=cwd, timeout=3600)
    content = obs["output"]
    start_sep = f"+ : '{TEST_OUTPUT_START}'"
    end_sep = f"+ : '{TEST_OUTPUT_END}'"
    start_idx = content.find(start_sep)
    end_idx = content.find(end_sep)
    test_logs = content[start_idx:end_idx][len(start_sep):]
    status_map = profile.log_parser(test_logs)
    return get_eval_tests_report(status_map, instance)


def evaluate_trajectory(
    instance: Dict[str, Any], model_patch: str, sweagent_config: dict, data_source: str
) -> MiniSWEEvaluationResult:
    ret = MiniSWEEvaluationResult(
        instance_id=instance["instance_id"], resolved=False, eval_error=None
    )

    cwd = sweagent_config["environment"].get("cwd", "/testbed/")
    env = None
    try:
        env = get_sb_environment(sweagent_config, instance, data_source)
    except Exception as e:
        ret["eval_error"] = f"Env creation failed with {e}"
        logger.info(
            f"Starting environment failed with exception: {e}\n, {traceback.format_exc()}"
        )
        return ret

    # For swe-smith the image starts clean; apply the bug patch first so the
    # model patch is evaluated against the same buggy state the agent saw.
    if data_source.lower() == "swe-smith":
        env = apply_patch(env, instance.get("patch", ""), cwd)

    # Apply model patch
    delimiter = f"PATCH_{uuid.uuid4().hex}"
    patch_string = f"git apply <<'{delimiter}'\n{model_patch}\n{delimiter}"
    env.execute({"command": patch_string}, cwd=cwd)

    if data_source.lower() == "swe-bench":
        report = _evaluate_swebench(env, instance, cwd)
    else:
        report = _evaluate_swesmith(env, instance, cwd)

    if swebench_get_resolution_status(report) == ResolvedStatus.FULL.value:
        ret["resolved"] = True
    else:
        ret["resolved"] = False

    ret["eval_error"] = report
    try:
        f2p_success = len(report["FAIL_TO_PASS"]["success"])
        f2p_failure = len(report["FAIL_TO_PASS"]["failure"])
        p2p_success = len(report["PASS_TO_PASS"]["success"])
        p2p_failure = len(report["PASS_TO_PASS"]["failure"])
        f2p_total = f2p_success + f2p_failure
        p2p_total = p2p_success + p2p_failure
        f2p_score = f2p_success / f2p_total if f2p_total > 0 else 1.0
        p2p_score = p2p_success / p2p_total if p2p_total > 0 else 1.0
        ret["partial_score"] = (f2p_score + p2p_score) / 2.0
    except Exception as e:
        logger.error(f"Error computing partial score: {e}")
        ret["partial_score"] = 0.0

    # truncate to last 1000 characters for brevity
    # ret["eval_error"] = (
    #     f"(truncated to last 1000 characters)\n{obs['output'][-1000:]}"
    #     if not ret["resolved"]
    #     else None
    # )
    return ret
