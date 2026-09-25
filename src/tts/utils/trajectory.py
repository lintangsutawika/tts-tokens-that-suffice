import os
import re
import json
import random
from collections import defaultdict
from itertools import chain


def validate_messages(messages):
    """
    Check that every assistant message with tool_calls has complete tool responses.

    Returns (ok, bad_index, missing_ids) where bad_index is the first offending
    assistant message index and missing_ids is the set of unresponded tool_call_ids.
    Returns (True, None, None) if valid.
    """
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            expected = {tc["id"] for tc in msg["tool_calls"]}
            responded = set()
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                responded.add(messages[j].get("tool_call_id"))
                j += 1
            missing = expected - responded
            if missing:
                return False, i, missing
        i += 1
    return True, None, None


def chunk_trajectory_by_assistant(trajectory):
    """
    Splits a trajectory (list of messages) into chunks, where each chunk is separated by a message with "role" == "assistant".
    Returns a list of chunks (each chunk is a list of messages).
    """
    chunks = []
    current_chunk = []
    for message in trajectory:
        current_chunk.append(message)
        if message.get("role") == "assistant" and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def get_uids_without_branches(output_dir):
    """
    Find all base trajectory UIDs that don't have any branch trajectories.

    Base trajectories have format: UUID:0.json
    Branch trajectories have format: UUID:0:N:0:0.json

    Returns:
        set: UIDs of base trajectories that have no associated branch trajectories
    """
    files = os.listdir(output_dir)

    base_pattern = re.compile(r'^([a-f0-9-]+):0\.json$')
    branch_pattern = re.compile(r'^([a-f0-9-]+):0:\d+:.+\.json$')

    base_uids = set()
    branch_base_uids = set()

    for f in files:
        base_match = base_pattern.match(f)
        if base_match:
            base_uids.add(base_match.group(1))

        branch_match = branch_pattern.match(f)
        if branch_match:
            branch_base_uids.add(branch_match.group(1))

    # Return base UIDs that have no branches
    return base_uids - branch_base_uids


def load_base_trajectory(output_dir, uid):
    """
    Load a base trajectory JSON file.

    Args:
        output_dir: Directory containing trajectory files
        uid: The UUID of the base trajectory

    Returns:
        dict: The loaded trajectory data
    """
    filepath = os.path.join(output_dir, f"{uid}:0.json")
    with open(filepath, 'r') as f:
        return json.load(f)


def create_partial_trajectories_from_messages(messages, uid, n_rollouts=1, max_branch_points=None, sample_limit=None):
    """
    Create partial trajectories from messages for branch generation.

    For each eligible branch point at step i two sets of N rollouts are created:

    - "before" (``{uid}:{i}:before:{n}``): trajectory up to just before action a_t.
      The agent will choose a new action, producing N alternative branches.

    - "after"  (``{uid}:{i}:after:{n}``): trajectory up to just before action a_{t+1},
      i.e. a_t has already been taken and its environment response is visible.
      The agent continues from there, producing N continuations of the original action.

    Args:
        messages: List of messages from the full trajectory
        uid: Base trajectory UID (format: UUID:0)
        n_rollouts: Number of rollouts per branch type per branch point
        max_branch_points: If set, randomly sample this many branch points
            from all eligible indices
        sample_limit: If set, eligible indices are capped at sample_limit - 5
            so branch runs have at least 5 steps of remaining budget.

    Returns:
        tuple: (partial_uids, partial_trajectories, step_indices)
            step_indices[k] is the chunk index at which trajectory k starts,
            used by callers to compute the remaining step budget.
    """
    chunked_trajectory = chunk_trajectory_by_assistant(messages)
    max_steps = len(chunked_trajectory)
    upper = (sample_limit - 5) if sample_limit is not None else (max_steps - 5)

    eligible_indices = [i for i in range(len(chunked_trajectory))
                        if (i >= 5) and (i < upper)]

    if max_branch_points is not None and len(eligible_indices) > max_branch_points:
        rng = random.Random(uid)
        eligible_indices = sorted(rng.sample(eligible_indices, max_branch_points))

    partial_uids = []
    partial_trajectories = []
    step_indices = []
    current_traj = []
    eligible_set = set(eligible_indices)

    for i, chunk in enumerate(chunked_trajectory):
        current_traj.extend(chunk)
        if i in eligible_set:
            # "before_init": single-step to sample one new a_t'.
            # n_rollouts continuations are then run from that a_t' (as before:0..n-1).
            before_traj = current_traj[:-1]
            partial_uids.append(f"{uid}:{i}:before_init")
            partial_trajectories.append(before_traj.copy())
            step_indices.append(i)

            # "after": state after a_t and its environment response — agent continues
            # from there. The env response is the non-assistant prefix of chunk i+1.
            if i + 1 < len(chunked_trajectory):
                next_chunk = chunked_trajectory[i + 1]
                # next_chunk ends with the next assistant action; drop it so the agent
                # decides what to do next (continuing from the original a_t path).
                after_traj = current_traj + next_chunk[:-1]
                for n in range(n_rollouts):
                    partial_uids.append(f"{uid}:{i}:after:{n}")
                    partial_trajectories.append(after_traj.copy())
                    step_indices.append(i + 1)

    return partial_uids, partial_trajectories, step_indices


def get_steps_missing_before(output_dir):
    """
    Scan output_dir and return, per base_uid, the set of step indices that have
    after:* files but no before:* files.

    Returns:
        dict: {base_uid: set of int step indices}
    """
    after_pattern = re.compile(r'^([a-f0-9-]+:0):(\d+):after:\d+(?::\d+)?\.json$')
    before_pattern = re.compile(r'^([a-f0-9-]+:0):(\d+):before:\d+(?::\d+)?\.json$')

    after_steps = defaultdict(set)
    before_steps = defaultdict(set)

    for f in os.listdir(output_dir):
        m = after_pattern.match(f)
        if m:
            after_steps[m.group(1)].add(int(m.group(2)))
            continue
        m = before_pattern.match(f)
        if m:
            before_steps[m.group(1)].add(int(m.group(2)))

    result = {}
    for base_uid, steps in after_steps.items():
        missing = steps - before_steps.get(base_uid, set())
        if missing:
            result[base_uid] = missing
    return result


def get_steps_incomplete_before(output_dir, n_rollouts):
    """
    Scan output_dir and return, per base_uid, steps that have after:* files but
    fewer than n_rollouts before:* files.

    Returns:
        dict: {base_uid: {step: {'existing_ks': set of int, 'has_before_init': bool}}}
    """
    after_pattern = re.compile(r'^([a-f0-9-]+:0):(\d+):after:\d+(?::\d+)?\.json$')
    before_pattern = re.compile(r'^([a-f0-9-]+:0):(\d+):before:(\d+)(?::\d+)?\.json$')
    before_init_pattern = re.compile(r'^([a-f0-9-]+:0):(\d+):before_init(?::\d+)?\.json$')

    after_steps = defaultdict(set)
    before_ks = defaultdict(lambda: defaultdict(set))
    before_init_steps = defaultdict(set)

    for f in os.listdir(output_dir):
        m = after_pattern.match(f)
        if m:
            after_steps[m.group(1)].add(int(m.group(2)))
            continue
        m = before_pattern.match(f)
        if m:
            before_ks[m.group(1)][int(m.group(2))].add(int(m.group(3)))
            continue
        m = before_init_pattern.match(f)
        if m:
            before_init_steps[m.group(1)].add(int(m.group(2)))

    result = {}
    for base_uid, steps in after_steps.items():
        for step in steps:
            existing_ks = before_ks.get(base_uid, {}).get(step, set())
            if len(existing_ks) < n_rollouts:
                result.setdefault(base_uid, {})[step] = {
                    'existing_ks': existing_ks,
                    'has_before_init': step in before_init_steps.get(base_uid, set()),
                }
    return result


def get_before_traj_for_step(messages, step):
    """Reconstruct the shared prefix (before_traj) for branch point at step i."""
    chunks = chunk_trajectory_by_assistant(messages)
    if step >= len(chunks):
        return None
    current = list(chain.from_iterable(chunks[:step + 1]))
    return current[:-1]  # drop the assistant message a_t at step i


def build_instance_id_to_task_map(dataset):
    """
    Build a mapping from instance_id to task dict.

    Args:
        dataset: List of task dictionaries

    Returns:
        dict: Mapping from instance_id to task dict
    """
    return {task['instance_id']: task for task in dataset}
