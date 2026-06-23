# 04 - SkyPilot YAML And Shell Rendering Hardening

Severity: High
Confidence: High
Subagent: Feynman

## Problem

`src/stable_retro_ppo/skypilot_launch.py` renders SkyPilot YAML by interpolating manifest and runner-profile fields directly into YAML scalars and shell blocks. The proof case is `venv_name`, which reaches `JOB_VENV="$HOME/{venv_name}"`, `UV_VENV="$HOME/{venv_name}-uv"`, and `PY="$HOME/{venv_name}/bin/python"` without shell-safe construction.

Affected surfaces:

- `src/stable_retro_ppo/skypilot_launch.py:93-114` validates presence, not safety.
- `src/stable_retro_ppo/skypilot_launch.py:348-361` interpolates manifest fields into setup.
- `src/stable_retro_ppo/skypilot_launch.py:424-430` interpolates fields into run assignments.
- `src/stable_retro_ppo/skypilot_launch.py:769-778` repeats the pattern for runner profiles.

## Desired state

Manifest/profile strings are either rejected as invalid structural identifiers or rendered as inert shell/YAML data. No manifest field can add a YAML key, terminate a shell assignment, trigger command substitution, append shell syntax, or change the SkyPilot task structure.

Keep intentional runtime expansion for trusted generated variables like `$HOME`, `${TS}`, `${GROUP}`, and `${RUN_NAME_n}`, but never by embedding untrusted text inside double quotes.

## Implementation plan

1. Add rendering helpers near `shell_quote()`:
   - `yaml_scalar(value: str) -> str`, using `json.dumps(str(value))` for YAML scalar values and file-mount keys/values.
   - `reject_control_chars(field, value)`, rejecting NUL, newline, and carriage return.
   - `safe_relative_name(field, value)`, requiring a conservative allowlist for `venv_name` and SkyPilot `name`, such as `^[A-Za-z0-9][A-Za-z0-9._-]*$`.
   - `home_path_expr(relative_path: str) -> str`, rendering shell paths as `"$HOME"/{shlex.quote(relative_path)}` after rejecting absolute paths and control chars.
2. Apply validation at render time in both `render_task_yaml()` and `render_runner_task_yaml()` so tests and direct callers are covered.
3. Replace raw YAML scalar interpolation:
   - `name: {sky_name}` -> `name: {yaml_scalar(sky_name)}`.
   - File mounts -> quoted key/value pairs through `yaml_scalar()`.
4. Replace shell assignments built with raw f-strings:
   - `JOB_VENV="$HOME/{venv_name}"` -> `JOB_VENV={home_path_expr(venv_name)}`.
   - `UV_VENV="$HOME/{venv_name}-uv"` -> `UV_VENV={home_path_expr(f"{venv_name}-uv")}`.
   - `PY="$HOME/{venv_name}/bin/python"` -> `PY={home_path_expr(f"{venv_name}/bin/python")}`.
   - `GROUP="{wandb_group_prefix}-${TS}"` -> `GROUP={shell_quote(wandb_group_prefix)}-"${TS}"`.
   - `LOG_DIR="{log_dir}"` -> `LOG_DIR={shell_quote(log_dir)}`.
   - `echo "train_profile={profile_id}"` -> `printf '%s\n' {shell_quote(f"train_profile={profile_id}")}`.
5. Fix dynamic run names safely:
   - Build per-run shell variables like `RUN_NAME_1={shell_quote(prefix_suffix)}_"${TS}"`.
   - Use `${RUN_NAME_1}` in `render_train_command()`.
   - Use `printf` and quoted variables for status/log messages.

## Tests/verification

Add regression tests in `tests/test_skypilot_launch.py`:

- Manifest `venv_name='bad"; touch /tmp/skypilot_pwned #'` is rejected or renders only as inert data.
- Same test for runner profile `venv_name`.
- `wandb_group_prefix`, `log_dir`, `profile_id`, and run `suffix` cannot produce raw `; touch`, `$(`, backticks, or broken assignments.
- `name` and `file_mounts` values containing newline plus `run:` cannot inject new YAML sections.
- Existing happy-path assertions are updated for quoted YAML, for example `name: "runner-test-4090"`.

Run:

```bash
uv run python -m unittest tests.test_skypilot_launch
uv run python -m stable_retro_ppo.skypilot_cli render /tmp/malicious_manifest.json
```

## Rollout notes

This is local launcher hardening and should not require remote SkyPilot access. Expect snapshot/assertion churn because YAML scalars will become quoted.

Prefer rejecting unsafe `venv_name` and `name` values over preserving exotic compatibility; these are infrastructure identifiers, not user-facing text.

Do not broaden the scope into changing train command construction unless a test proves a field still reaches shell syntax after render helpers are applied.

## Open questions

- Should invalid structural fields fail during `load_manifest()`/`load_runner_profile()` too, or only during render/preflight?
- Are slashes intentionally supported in `venv_name`?
- Should `extra_file_mounts` allow absolute local paths, or should sources be constrained under `repo_root` as a separate hardening follow-up?
