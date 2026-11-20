# Source development

GradLab uses Python 3.14. Install [uv](https://docs.astral.sh/uv/getting-started/installation/),
Node.js 22.12 or later, and the pnpm version pinned in [package.json](../package.json).

```bash
git clone https://github.com/tsilva/gradlab.git
cd gradlab
./install.sh
gradlab validate
```

`install.sh` builds the player and installs the checkout as an editable uv tool
using the frozen Python lock. For work in the project's development environment,
run `uv sync --frozen`. Use `gradlab <command> --help` for CLI arguments, or
`uv run gradlab <command> --help` from the development environment.

## Player development

```bash
pnpm install --frozen-lockfile
pnpm check:web
pnpm build:web
uv run gradlab play
```

Add `--hotreload` to use Vite and reload changes under `frontend/` and
`src/gradlab/web_player/`. Restart GradLab after Python changes. The compiled
bundle in `src/gradlab/web_player/dist/` is generated and ignored. Build it for
compiled-mode tests and packaging. Wheels and source distributions include the
bundle, and building a wheel from the published source distribution reuses it
without Node.

## Checks and releases

Run `uv run pytest`, `uv run ruff check .`, and `pnpm test:web` for Python tests,
lint, and web tests. See the [player fixtures](../tests/web_player/fixtures/README.md)
for browser and performance workloads.

[Regression CI](../.github/workflows/gradlab-python-checks.yml) runs CPU tests and
replays the simulated lifecycle gate twice to verify determinism.
[Report validation](../.github/workflows/gradlab-reports-validate.yml) checks
portable experiment reports. [Secret scanning](../.github/workflows/secret-scanning.yml)
runs on pushes and pull requests, fails closed when required revisions are
missing, and reports finding locations without matched content. New branches and
rewritten pushes scan history reachable from the new head.

`python3 scripts/release.py --validate` builds the committed remote `main` in
GitHub Actions without publishing. The
[release procedure](../.codex/skills/gradlab_build-release/SKILL.md) covers
publication and candidate builds.

See the [product specifications](../SPECS.md), [scoped requirements](specs/README.md),
and [metric definitions](../METRICS.md) when changing a documented capability.
