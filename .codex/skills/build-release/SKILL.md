---
name: build-release
description: Cut, publish, monitor, verify, build, or inspect GradLab Python releases. A bare $build-release invocation launches the full trusted-publishing workflow and follows it until the version is live on PyPI.
---

# Build Release

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

Do not manually upload with Twine, use a local PyPI token, republish an existing
version, move a release tag, bypass a failed gate, or create/switch branches.

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

4. Capture the pushed tag and release commit, then monitor the tag-triggered
   workflow:

   ```bash
   release_sha="$(git rev-list -n 1 vX.Y.Z)"
   gh run list --workflow release.yml --commit "$release_sha" --limit 5 \
     --json databaseId,status,conclusion,event,headBranch,headSha,displayTitle,url
   gh run watch <run-id> --exit-status
   ```

   If the commit-filtered query is initially empty, list the recent Release runs
   and select the tag-push run for `vX.Y.Z`. A `workflow_dispatch` run validates
   but never publishes.

5. After workflow success, poll
   `https://pypi.org/pypi/gradlab/X.Y.Z/json` until both expected files appear:

   - `gradlab-X.Y.Z-py3-none-any.whl`
   - `gradlab-X.Y.Z.tar.gz`

   Also verify the GitHub Release and its attached artifacts. If trusted
   publishing fails, report the run and failing step; do not attempt manual
   recovery unless the user explicitly requests it.

6. Report the PyPI version URL first, followed by the tag, release commit,
   workflow URL and conclusion, GitHub Release URL, published filenames, and
   digests when available.

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

## Useful inspection commands

```bash
gh run view <run-id> --log-failed
gh release view vX.Y.Z --json tagName,url,publishedAt,assets
git describe --tags --exact-match HEAD
```

The final package URLs are:

```text
https://pypi.org/project/gradlab/X.Y.Z/
https://pypi.org/project/gradlab/
```
