---
name: build-release
description: Build, publish, inspect, or verify GradLab Python releases. A bare $build-release invocation publishes the next release through trusted publishing and follows verification.
---

# Build Release

## When to use

Use for GradLab package releases. A bare `$build-release` means publish the next
release. Normal publication and validation build only in GitHub Actions.
Explicit local-build requests select the local candidate path. Status/diagnosis
requests inspect existing state.

## Prepare

Apply the repository specification review once for this task. Read the shared
[$release-workflow](/Users/tsilva/.codex/skills/release-workflow/SKILL.md), which
owns common preflight, authorization, monitoring, verification, and `$push` use.
If unavailable, report that dependency. Read the requested path in
[operations.md](references/operations.md) for project-specific arguments and gates.

## Act

For publication:

```bash
python3 scripts/release.py
```

Use `--to X.Y.Z` or `--part minor|major` only when requested. The launcher owns
version selection/surfaces, metadata checks, commit, annotated tag, and atomic
branch/tag push. The operator needs Python 3.11+, Git, `uv`, and `gh`; no local
package environment, web build, or source test suite is required. Tag-triggered
`release.yml` owns the complete source checks, web and Python builds, candidate
audits, installation smoke test, and trusted publication. Never manually upload
to PyPI.

For validation without publication:

```bash
python3 scripts/release.py --validate
```

This builds committed remote `main`, permits unrelated dirty local work, and
changes no version surfaces. Follow the printed full source SHA and require the
Actions build job and downloaded artifacts to pass; publication jobs must skip.

For a requested local candidate:

```bash
uv sync --frozen --group release
uv run python scripts/release_build.py build --auto-bump
```

This can update release version surfaces without publishing. Report those edits.

## Recover

Preserve candidate/failure evidence and inspect existing tags/workflows before
repeating publication. The launcher restores preparation changes on failure;
the shared skill owns publication recovery. `workflow_dispatch` validates only.

## Finish

Follow the tag-push workflow at the exact commit. Require the wheel and sdist on
PyPI and matching GitHub Release assets for the same version/tag. After a push,
use `$push` for default-branch README tagline reconciliation; report push and
repository-description synchronization separately. Local candidates report
version, changed sources, artifacts/hashes, and completed gates. Actions validation
reports the source SHA, run URL, artifact checks, and that nothing was published.
