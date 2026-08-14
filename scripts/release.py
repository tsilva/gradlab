#!/usr/bin/env python3
"""Prepare, validate, commit, tag, and push a GradLab release."""

from __future__ import annotations

import argparse
from itertools import count
import subprocess
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_HELPER = (
    REPO_ROOT / ".codex" / "skills" / "build-release" / "scripts" / "release_build.py"
)
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
RELEASE_FILES = (
    REPO_ROOT / "pyproject.toml",
    REPO_ROOT / "src" / "gradlab" / "__init__.py",
    REPO_ROOT / "uv.lock",
    REPO_ROOT / "README.md",
)
EXPECTED_REPOSITORY = "tsilva/gradlab"


def run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=REPO_ROOT, check=True)


def capture(args: list[str]) -> str:
    return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()


def ensure_clean() -> None:
    status = capture(["git", "status", "--short"])
    if status:
        raise SystemExit(f"release tree must be clean before preparation:\n{status}")


def upstream_ref() -> str:
    try:
        return capture(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    except subprocess.CalledProcessError as error:
        raise SystemExit("current branch must have an upstream before release") from error


def require_gradlab_remote(remote: str) -> None:
    url = capture(["git", "remote", "get-url", remote]).removesuffix(".git")
    normalized = url.replace("git@github.com:", "github.com/").replace(
        "https://github.com/", "github.com/"
    )
    if not normalized.endswith(f"github.com/{EXPECTED_REPOSITORY}"):
        raise SystemExit(
            f"release remote must be the GradLab repository, got {remote}={url}"
        )


def ensure_synced() -> tuple[str, str]:
    upstream = upstream_ref()
    if "/" not in upstream:
        raise SystemExit(f"unexpected upstream ref: {upstream}")
    remote, branch = upstream.split("/", 1)
    require_gradlab_remote(remote)
    run(["git", "fetch", "--prune", "--tags", remote])
    ahead_text, behind_text = capture(
        ["git", "rev-list", "--left-right", "--count", f"HEAD...{upstream}"]
    ).split()
    ahead, behind = int(ahead_text), int(behind_text)
    if ahead or behind:
        raise SystemExit(
            f"current branch must be synchronized with {upstream}; "
            f"ahead={ahead} behind={behind}"
        )
    return remote, branch


def helper(*args: str) -> None:
    run([str(PYTHON), str(RELEASE_HELPER), *args])


def current_version() -> str:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


def next_version(version: str, part: str) -> str:
    try:
        major, minor, patch = (int(value) for value in version.split("."))
    except ValueError as error:
        raise SystemExit(f"cannot {part}-bump non-final version {version!r}") from error
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(part)


def prepare_version(args: argparse.Namespace) -> str:
    if args.to:
        helper("prepare-version", "--version", args.to)
    elif args.part:
        helper("prepare-version", "--version", next_version(current_version(), args.part))
    else:
        helper("prepare-version")
    version = current_version()
    helper("check-version", "--version", version)
    return version


def tag_exists(tag: str) -> bool:
    return (
        subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", tag],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def candidate_directory(version: str) -> Path:
    base = REPO_ROOT / "dist" / f"release-v{version}"
    if not base.exists() or not any(base.iterdir()):
        return base
    for attempt in count(2):
        candidate = REPO_ROOT / "dist" / f"release-v{version}-candidate-{attempt}"
        if not candidate.exists():
            return candidate
    raise AssertionError("unreachable")


def run_release_gates(version: str) -> Path:
    run(["uv", "sync", "--frozen", "--group", "dev", "--group", "release"])
    run(["uv", "run", "ruff", "check", "."])
    run(["uv", "run", "pytest", "-q"])
    run(["uv", "run", "gradlab", "validate"])
    run(["uv", "run", "gradlab", "experiment", "certify", "--tier", "simulated", "--json"])
    out_dir = candidate_directory(version)
    helper("build", "--version", version, "--out-dir", str(out_dir))
    return out_dir


def create_commit_and_tag(version: str) -> str:
    tag = f"v{version}"
    if tag_exists(tag):
        raise SystemExit(f"release tag already exists: {tag}")
    run(["git", "add", *(str(path.relative_to(REPO_ROOT)) for path in RELEASE_FILES)])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO_ROOT).returncode != 0:
        run(["git", "commit", "-m", f"Release {tag}"])
    run(["git", "tag", "-a", tag, "-m", f"GradLab {tag}"])
    return tag


def push_release(remote: str, branch: str, tag: str, *, dry_run: bool) -> None:
    command = ["git", "push", "--atomic"]
    if dry_run:
        command.append("--dry-run")
    command.extend([remote, f"HEAD:{branch}", tag])
    run(command)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    version = parser.add_mutually_exclusive_group()
    version.add_argument("--to", help="exact release version, for example 0.1.2")
    version.add_argument("--part", choices=("patch", "minor", "major"))
    parser.add_argument("--dry-run-push", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if not PYTHON.exists():
        raise SystemExit(
            "release environment is missing; run `uv sync --frozen --group dev --group release`"
        )
    ensure_clean()
    remote, branch = ensure_synced()
    snapshots = {path: path.read_bytes() for path in RELEASE_FILES}
    try:
        version = prepare_version(args)
        tag = f"v{version}"
        if tag_exists(tag):
            raise SystemExit(f"release tag already exists: {tag}")
        out_dir = run_release_gates(version)
        tag = create_commit_and_tag(version)
    except BaseException:
        for path, contents in snapshots.items():
            path.write_bytes(contents)
        subprocess.run(["git", "reset", "--quiet"], cwd=REPO_ROOT, check=False)
        raise
    push_release(remote, branch, tag, dry_run=args.dry_run_push)
    print()
    print(f"Released {tag}: pushed {branch} and tag to {remote}.")
    print(f"Validated local candidate: {out_dir}")
    print("GitHub Actions will build, audit, publish to PyPI, and create the GitHub release.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error
