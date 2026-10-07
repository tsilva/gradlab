#!/usr/bin/env python3
"""Prepare, validate, commit, tag, and push a GradLab release."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_HELPER = REPO_ROOT / "scripts" / "release_build.py"
PYTHON = Path(sys.executable)
RELEASE_FILES = (
    REPO_ROOT / "pyproject.toml",
    REPO_ROOT / "src" / "gradlab" / "__init__.py",
    REPO_ROOT / "uv.lock",
    REPO_ROOT / "README.md",
)
EXPECTED_REPOSITORY = "tsilva/gradlab"


def run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    env = None
    if Path(args[0]).name == "uv":
        env = {**os.environ, "UV_CONFIG_FILE": str(REPO_ROOT / "uv-tool.toml")}
    subprocess.run(args, cwd=REPO_ROOT, check=True, env=env)


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
    if branch != "main" or capture(["git", "branch", "--show-current"]) != "main":
        raise SystemExit("publication requires main tracking the remote main branch")
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


def run_release_gates(version: str) -> None:
    """Validate metadata locally; Actions owns source tests and all builds."""
    helper("check-version", "--version", version)
    run(["uv", "lock", "--check"])
    run(["git", "diff", "--check"])


def validate_remote() -> None:
    """Build committed remote main without changing local release state."""
    upstream = upstream_ref()
    remote, branch = upstream.split("/", 1)
    require_gradlab_remote(remote)
    if branch != "main":
        raise SystemExit("validation requires an upstream main branch")
    run(["gh", "auth", "status"])
    run(["git", "fetch", remote, "main"])
    source_sha = capture(["git", "rev-parse", f"{remote}/main"])
    run([
        "gh", "workflow", "run", "release.yml", "--repo", EXPECTED_REPOSITORY,
        "--ref", "main", "-f", f"ref={source_sha}",
    ])
    print(f"Validation source SHA: {source_sha}")
    print("Monitor the workflow_dispatch run on main and verify its checked-out source SHA.")


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
    parser.add_argument("--validate", action="store_true", help="build remote main in Actions only")
    args = parser.parse_args(argv)
    if args.validate and (args.to or args.part or args.dry_run_push):
        parser.error("--validate cannot be combined with version preparation or --dry-run-push")
    return args


def main() -> None:
    args = parse_args()
    if args.validate:
        validate_remote()
        return
    ensure_clean()
    remote, branch = ensure_synced()
    snapshots = {path: path.read_bytes() for path in RELEASE_FILES}
    try:
        version = prepare_version(args)
        tag = f"v{version}"
        if tag_exists(tag):
            raise SystemExit(f"release tag already exists: {tag}")
        run_release_gates(version)
        tag = create_commit_and_tag(version)
    except BaseException:
        for path, contents in snapshots.items():
            path.write_bytes(contents)
        subprocess.run(["git", "reset", "--quiet"], cwd=REPO_ROOT, check=False)
        raise
    push_release(remote, branch, tag, dry_run=args.dry_run_push)
    print()
    source_sha = capture(["git", "rev-parse", "HEAD"])
    if args.dry_run_push:
        print(f"Prepared {tag} at {source_sha}; dry-run push only, no Actions build or publication.")
        return
    print(f"Pushed {branch} and {tag} at {source_sha} to {remote}.")
    print("GitHub Actions will build, audit, publish to PyPI, and create the GitHub release.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error
