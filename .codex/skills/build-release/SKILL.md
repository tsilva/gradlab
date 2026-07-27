---
name: build-release
description: Build, audit, smoke-test, publish, or inspect GradLab Python releases. Use for local release candidates, PyPI release preparation, version/tag checks, trusted publishing, GitHub releases, or release-workflow diagnosis.
---

# Build Release

Build GradLab distributions locally or drive the tag-triggered trusted-publishing
workflow. Keep building and publishing separate: a local candidate is reversible;
pushing a release tag publishes externally.

## Choose the path

- For “build a release”, “make artifacts”, or release-candidate validation, use
  **Build a local candidate**. Do not commit, tag, push, or publish.
- For “release”, “publish”, “ship”, or “cut version”, use **Publish from a tag**.
- For status or diagnosis, inspect the existing workflow run and artifacts without
  rerunning or mutating it unless the user asks.

## Build a local candidate

1. Read the repository `AGENTS.md` and use `$specs-author` as required there.
2. Resolve the version from `pyproject.toml`; never infer a version bump.
3. Install the locked release tooling:

   ```bash
   uv sync --frozen --group release
   ```

4. Build into a version-scoped directory:

   ```bash
   uv run python .codex/skills/build-release/scripts/release_build.py build \
     --version X.Y.Z \
     --out-dir dist/release-vX.Y.Z
   ```

5. Report the wheel and sdist paths, SHA-256 digests, and every completed gate.
   Preserve failed artifacts and exact error output for diagnosis.

The helper requires matching project/import/lock versions, an unused PyPI version,
clean artifact names and metadata, no legacy `rlab` branding in published content,
`twine check`, and a dependency-free wheel smoke test.

## Publish from a tag

Only publish source that has already passed a local candidate build.

1. Require a clean worktree and a branch synchronized with its upstream.
2. Confirm the GitHub repository is the intended GradLab repository, not the old
   `rlab` remote.
3. Confirm `pyproject.toml`, `src/gradlab/__init__.py`, and the root package entry
   in `uv.lock` all equal `X.Y.Z`.
4. Confirm `https://pypi.org/pypi/gradlab/json` has no files for `X.Y.Z`.
5. Run the full source gates from `.github/workflows/release.yml`.
6. Create annotated tag `vX.Y.Z` only after explicitly confirming that pushing it
   will publish to PyPI.
7. Push the branch and tag atomically:

   ```bash
   git push --atomic origin HEAD vX.Y.Z
   ```

8. Monitor the `Release` GitHub Actions workflow through completion. Verify the
   PyPI release, its files, and the GitHub release before reporting success.

Do not upload with `twine`, use a local PyPI token, republish an existing version,
move a release tag, or bypass failed validation. Trusted publishing in
`.github/workflows/release.yml` is the only normal publication path.

## Workflow contract

- A `workflow_dispatch` run builds and audits artifacts but never publishes.
- A pushed `v*` tag builds, audits, publishes to the protected `pypi`
  environment, then creates a GitHub release.
- Distribution artifacts are `gradlab-X.Y.Z-py3-none-any.whl` and
  `gradlab-X.Y.Z.tar.gz`.
- The PyPI project URL is `https://pypi.org/project/gradlab/`.
