# Local dependency patches

Patches we apply to installed third-party packages in the `.venv`. Re-apply
these after any `uv sync` / reinstall that overwrites the affected package.

## `harbor_singularity_dockerfile_from.patch`

> **SUPERSEDED (2026-09-02).** This behavior now ships in the
> **`harbor-singularity-hpc`** package (a `SingularityEnvironment` subclass selected via
> `--environment-import-path`), pinned as a normal `uv` dependency. `harbor_eval.sh` no
> longer applies this patch. Kept here as the reference implementation that seeded the
> package; do not re-apply it to the venv (it would collide with the subclass override).

Lets the **Singularity** environment (`ENV=singularity`) run the SWE-bench datasets.
Harbor's Singularity backend only pulls a named image from `task.toml`
`[environment].docker_image`; it never builds a task's `environment/Dockerfile` (the
Modal/Docker/Podman backends do, via `Image.from_dockerfile`). SWE-bench tasks set no
`docker_image` — they declare the image via `FROM swebench/sweb.eval.x86_64.<inst>` in
the Dockerfile — so unpatched Singularity dies at validation:

```
ValueError: Singularity environment requires 'docker_image' in task.toml [environment].
```

Fix: `_docker_image` falls back to parsing the Dockerfile's effective (last, non-comment)
`FROM` line when `docker_image` is unset. Singularity then `singularity pull`s that base;
the Dockerfile's extra RUN layers (uv/git/tmux) are irrelevant because the backend's
`bootstrap.sh` installs python/pip and the harbor server at container runtime anyway.

Target: `harbor/environments/singularity/singularity.py` (harbor rev `c178c207`).
Only needed on the **no-podman** path — `ENV=podman` builds the Dockerfile natively and
needs no patch.

## `harbor_singularity_writable_sandbox.patch`

> **SUPERSEDED (2026-09-02).** Folded into the **`harbor-singularity-hpc`** package (see
> the note above). `harbor_eval.sh` selects that class via `--environment-import-path`
> instead of patching harbor's source; the `_HB_LOCAL_PATCHSET` upgrade dance is gone.
> Kept as reference only.

Makes `ENV=singularity` produce a **writable** container rootfs on HPC nodes where the
default path can't. On ABCI (and similar), `/etc/fuse.conf` lacks `user_allow_other`, so
squashfuse can't FUSE-mount the SIF under `--fakeroot` → singularity falls back to
extraction + **underlay**, which has no overlayfs → `--writable-tmpfs` (and `--overlay`,
even an ext3 image) silently degrade to a **read-only** rootfs. `bootstrap.sh` then can't
create its venv at `/opt/harbor-server` and the in-container server dies with "Server
failed to start on port …".

Fix: opt-in `singularity_writable_sandbox` kwarg. When set, `start()` runs
`singularity build --fix-perms --sandbox <dir>` to extract the image into a per-session
**directory** on node-local scratch, and `_start_server()` execs `--fakeroot --writable
<dir>` instead of `--writable-tmpfs <sif>`. A plain directory has **no squashfuse mount
and needs no overlayfs**, so it dodges both blockers and `--fakeroot` still gives
root-in-container (needed to write the root-owned `/testbed` and `/opt`). Because
`--writable` (unlike the overlay) **can't auto-create bind-mount destinations**, the patch
also pre-creates every `-B` target and the `--pwd` workdir (e.g. `/staging`,
`/staging/env_files`) inside the sandbox before exec -- otherwise container creation fails
with "destination /staging doesn't exist in container". `stop()` removes
the sandbox (`--fix-perms` makes it removable; node-local scratch is job-scoped anyway).

This patch also changes the **default image cache dir**: when no `singularity_image_cache_dir`
is passed, harbor now resolves it LIVE to `$PBS_LOCALDIR`/`$SLURM_TMPDIR/harbor_sif_cache`
(was a throwaway `mkdtemp` per env, which re-pulled every trial). Resolving live matters for
chunked resume: an absolute cache path passed via `--ek` gets baked into the job `config.json`
and, if node-local, goes stale on the next chunk (different node/jobid) -> `PermissionError`
building `/local/<old-jobid>/...`. So `harbor_eval.sh` leaves `SINGULARITY_CACHE_DIR` unset by
default (node-local, live, resume-safe); set a **literal shared-FS path** only if you want the
cache to persist across chunks (that value is safe to bake because it's valid on every node).

It also adds **retry-with-backoff** around `singularity pull` in `_convert_docker_to_sif`:
against Docker Hub under concurrency (`N_CONCURRENT=32`), the OCI pull intermittently fails with
"unexpected end of JSON input" / "conveyor failed to get" (rate limits + OCI-cache contention),
which otherwise fails the whole trial. Bounded retries (5/10/20/40s) recover it. Complementary
runtime mitigations: lower `N_CONCURRENT`, or authenticate (`SINGULARITY_DOCKER_USERNAME` +
`SINGULARITY_DOCKER_PASSWORD`) to raise the anon rate limit.

It also throttles concurrent `singularity pull`s process-wide via a class-level
`_PULL_SEMAPHORE` (default 4, override `HB_SINGULARITY_MAX_PULLS`): many simultaneous pulls
race the shared OCI cache and burst Docker Hub's limit. Agent concurrency stays at
`N_CONCURRENT`; only pulls are limited.

It also raises the exec HTTP-client timeout: mini-swe-agent runs its whole loop as one
`exec` (no `timeout_sec`), so the stock 600s default killed long agent runs well under the
trial's real AGENT budget (enforced by an outer `asyncio.wait_for`). Default is now 86400s
(override `HB_SINGULARITY_HTTP_TIMEOUT`), letting the outer budget govern.

The patch carries a version stamp `_HB_LOCAL_PATCHSET` (currently `"6"`). **Bump it here and in
`scripts/eval/harbor_eval.sh` together whenever this patch changes** — a persistent `$HOME` venv
would otherwise keep an older copy (the auto-apply keys "already current" on the stamp).

Applied **after** `harbor_singularity_dockerfile_from.patch` (it's cut against that patch);
`scripts/eval/harbor_eval.sh` applies both in order, and resets harbor to pristine first when a
persistent `$HOME` venv carries an older version (stamp mismatch) via `uv sync --reinstall-package harbor`.
Enable the sandbox mode at runtime with `SINGULARITY_WRITABLE_SANDBOX=true` (optionally
`SINGULARITY_SANDBOX_DIR=<node-local dir>`; default `$PBS_LOCALDIR`/`$SLURM_TMPDIR`/`$TMPDIR`).
Inert unless enabled.

Target: `harbor/environments/singularity/singularity.py` (harbor rev `c178c207`).
The proper fix is site-side (`user_allow_other` in `/etc/fuse.conf`), which restores the
faster RAM-overlay path and makes this patch unnecessary.

## `swebench_modal_build_pip_compat.patch`

Fixes persistent `error_instances` in `swebench.harness.run_evaluation --modal true`.
The Modal backend rebuilds each instance's test image from scratch with a *current*
pip/setuptools, which breaks older repos that the pre-built Docker Hub images handle
fine. Three failure modes, all in `make_repo_script_list_py`:

- **scikit-learn 1.3–1.6** (`25500/25570/25638/25747`): current pip removed the
  `--no-use-pep517` flag → `no such option: --no-use-pep517`. Fix pins
  `pip<21.3` + `setuptools<60` (old numpy.distutils build) before install.
- **pylint 2.15 / sympy 1.7** (`7114/7228/7993`, `20590`): current pip requires the
  PEP 660 `build_editable` hook for `pip install -e .`, which the repo's isolated
  build backend lacks. Fix installs `setuptools>=64` and adds `--no-build-isolation`
  so the editable build uses it.
- **sympy** (`REPO_BASE_COMMIT_BRANCH`): clones a branch (`1.7`) that no longer
  exists upstream (only tag `sympy-1.7`) → `Remote branch 1.7 not found`. Fix falls
  back to a full clone; the existing reset + tag-prune + leak-check keep it clean.

Target: `swebench/harness/test_spec/python.py` (swebench 4.1.0).

Apply:

```bash
cd .venv/lib/python3.*/site-packages
git apply --recount /path/to/patches/swebench_modal_build_pip_compat.patch
# check it's already applied:
git apply --reverse --check --recount /path/to/patches/swebench_modal_build_pip_compat.patch
```

## `swebench_modal_max_containers.patch`

Caps Modal eval concurrency. `--max_workers` is ignored on the `--modal true`
path (it only applies to the local Docker path); Modal otherwise autoscales
`run_instance_modal` up to the workspace container limit. This sets
`max_containers=32` on the `@app.function` so the fresh per-instance image builds
don't fan out unbounded. Adjust the number in
`swebench/harness/modal_eval/run_evaluation_modal.py` (swebench 4.1.0).

## `lora_adapter_checkpoint.patch`

Target: `skyrl/backends/skyrl_train_backend.py`. Persists the actually-synced LoRA
adapter instead of re-exporting a merged HF model (MoE LoRA merge doesn't round-trip).
