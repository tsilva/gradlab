---
name: build-release
description: Cut, publish, monitor, verify, build, or inspect GradLab Python releases. A bare $build-release invocation launches the full trusted-publishing workflow and follows it until the version is live on PyPI.
---

# Build Release

Read and apply the shared `$release-workflow` skill at
`/Users/tsilva/.codex/skills/release-workflow/SKILL.md` before execution.
It owns common preflight, publication safeguards, `$push` integration,
workflow monitoring, verification, and reporting. The rules below are this
project's adapter; they retain its invocation default and required gates.
If the shared skill is unavailable, stop and report the missing dependency.

Use this skill to run the repo-owned GradLab release flow and monitor it until
the package is visible on PyPI. A bare `$build-release` invocation means
**publish the next release**, not merely build local artifacts. Use the
local-candidate path only when the user explicitly asks for a local build,
artifacts, validation, or a dry run. For status or diagnosis, inspect existing
state without mutating it.

The release launcher lives in `scripts/release.py`. It follows the same pattern
as the SuperMarioBros-Nes-turbo release flow: require a clean synchronized tree
and intended GitHub remote, select an unused version, update every version
surface, run the complete local source gates, build and audit a local candidate,
create an annotated tag, and atomically push the branch and tag. The tag triggers
`.github/workflows/release.yml`, which rebuilds and audits the distributions,
publishes with PyPI trusted publishing, and creates the GitHub Release.

## Publish the next release

1. Read `AGENTS.md`, use `$specs-author`, and confirm that publishing is within
   the user's request. A bare `$build-release` invocation is explicit release
   authorization under this skill.

2. Install the locked release environment:

   ```bash
   uv sync --frozen --group dev --group release
   ```

3. Launch the repo-owned release command:

   ```bash
   uv run python scripts/release.py
   ```

   With no version preference, an untagged unused project version is released;
   otherwise the helper advances to the first unused patch version. For an
   explicitly requested version or bump, use exactly one of:

   ```bash
   uv run python scripts/release.py --to X.Y.Z
   uv run python scripts/release.py --part minor
   uv run python scripts/release.py --part major
   ```

   The launcher updates `pyproject.toml`, `src/gradlab/__init__.py`, `uv.lock`,
   and the README's pinned one-command demo. It runs the workflow's Ruff, pytest,
   configuration-validation, and simulated lifecycle-certification gates, then
   builds, audits, checks, and dependency-free smoke-tests a local wheel and
   sdist before creating the release commit and annotated tag. Failed
   preparation restores changed release files and preserves candidate evidence.

4. Follow the shared monitoring and verification procedure for the `release.yml`
tag-push run at the full `vX.Y.Z` commit SHA. A `workflow_dispatch` run
validates artifacts but never publishes. Verify PyPI project `gradlab` and
the GitHub Release for the same tag.

Require `gradlab-X.Y.Z-py3-none-any.whl` and `gradlab-X.Y.Z.tar.gz`
on PyPI and as matching GitHub Release assets.

## Build a local candidate only

Use this path only when the user explicitly asks for a local candidate or
validation without publication:

```bash
uv sync --frozen --group release
uv run python .codex/skills/build-release/scripts/release_build.py build --auto-bump
```

This path may advance a used patch version and update the four release surfaces,
but it never commits, tags, pushes, or publishes. Report the selected version,
changed sources, artifacts, SHA-256 digests, and every completed gate.
