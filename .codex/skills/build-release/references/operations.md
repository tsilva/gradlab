# Build Release

The [entry skill](../SKILL.md) owns invocation defaults and loads the shared
release workflow. These procedures provide the project adapter's detailed gates
and arguments; follow only the path selected by the request.

The release launcher lives in `scripts/release.py`; its maintained build/audit
helper is `scripts/release_build.py`. The skill owns operating guidance only.
The launcher requires a clean synchronized tree
and intended GitHub remote, selects an unused version, updates every version
surface, checks metadata and lock consistency, creates an annotated tag, and
atomically pushes the branch and tag. The tag triggers
`.github/workflows/release.yml`, which runs source gates, builds and audits the distributions,
publishes with PyPI trusted publishing, and creates the GitHub Release.

## Publish the next release

1. Apply the entry skill's preparation and publication authorization.

2. Use Python 3.11+, Git, `uv`, and authenticated `gh` on clean synchronized
   `main`. Build tools and package dependencies are installed in Actions.

3. Launch the repo-owned release command:

   ```bash
   python3 scripts/release.py
   ```

   With no version preference, an untagged unused project version is released;
   otherwise the helper advances to the first unused patch version. For an
   explicitly requested version or bump, use exactly one of:

   ```bash
   python3 scripts/release.py --to X.Y.Z
   python3 scripts/release.py --part minor
   python3 scripts/release.py --part major
   ```

   The launcher updates `pyproject.toml`, `src/gradlab/__init__.py`, `uv.lock`,
   and the README's pinned one-command demo. It checks version consistency,
   unused PyPI version/tag, lock consistency, and patch formatting locally.
   Actions runs the locked web build, Ruff, pytest, configuration validation,
   simulated lifecycle certification, distribution audits, Twine metadata checks,
   and dependency-free installed-wheel smoke test before publication. Failed
   local preparation restores changed release files.

4. Follow the shared monitoring and verification procedure for the `release.yml`
tag-push run at the full `vX.Y.Z` commit SHA. A `workflow_dispatch` run
validates artifacts but never publishes. Verify PyPI project `gradlab` and
the GitHub Release for the same tag.

After a successful branch/tag push, follow `$push` to reconcile the destination
repository description with the marked tagline in its latest published
default-branch README. Report push and metadata synchronization separately.

Require `gradlab-X.Y.Z-py3-none-any.whl` and `gradlab-X.Y.Z.tar.gz`
on PyPI and as matching GitHub Release assets.

## Validate in Actions without publication

```bash
python3 scripts/release.py --validate
```

The launcher fetches the configured GradLab upstream and dispatches `release.yml`
on `main` with the exact remote-main SHA. It does not inspect or build dirty local
source, prepare versions, commit, tag, or publish. Require the workflow's verified
checkout SHA to equal the printed source SHA. Download `release-vX.Y.Z` from that
run, audit its exact wheel/sdist set, and record their SHA-256 hashes. `publish` and
`github-release` must skip. Existing published versions may be rebuilt in this
validation mode. Normal tag builds still require an unused PyPI version.

`--dry-run-push` retains local version preparation, commit, and annotated tag, but
only previews the atomic push. It does not build or certify artifacts; prefer
`--validate` for a build without publication.

## Build a local candidate only

Use this path only when the user explicitly asks for a local candidate:

```bash
uv sync --frozen --group release
uv run python scripts/release_build.py build --auto-bump
```

This path may advance a used patch version and update the four release surfaces,
but it never commits, tags, pushes, or publishes. Report the selected version,
changed sources, artifacts, SHA-256 digests, and every completed gate.
