#!/usr/bin/env python3
"""Build and audit GradLab release distributions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


PACKAGE_NAME = "gradlab"
IMPORT_NAME = "gradlab"
ENTRY_POINT = "gradlab = gradlab.main:main"
VERSION_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
CONTAMINATION_PARTS = {
    ".env",
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "logs",
    "models",
    "runs",
}


class ReleaseError(RuntimeError):
    """A release invariant failed."""


def fail(message: str) -> None:
    raise ReleaseError(message)


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def require_version(version: str) -> None:
    if not VERSION_RE.fullmatch(version):
        fail(f"release version must be strict X.Y.Z, got {version!r}")


def project_versions(root: Path) -> dict[str, str]:
    project = load_toml(root / "pyproject.toml")["project"]
    if project["name"] != PACKAGE_NAME:
        fail(f"project name must be {PACKAGE_NAME!r}, got {project['name']!r}")

    init_text = (root / "src" / IMPORT_NAME / "__init__.py").read_text()
    match = re.search(r'^__version__\s*=\s*"([^"]+)"$', init_text, re.MULTILINE)
    if match is None:
        fail("src/gradlab/__init__.py has no literal __version__")

    readme_text = (root / "README.md").read_text()
    readme_match = re.search(
        r"^uvx gradlab@([^ ]+) train gradlab__bandit/ppo$", readme_text, re.MULTILINE
    )
    if readme_match is None:
        fail("README.md has no pinned one-command GradLab demo version")

    lock_packages = load_toml(root / "uv.lock").get("package", [])
    lock_matches = [package for package in lock_packages if package.get("name") == PACKAGE_NAME]
    if len(lock_matches) != 1:
        fail(f"uv.lock must contain one {PACKAGE_NAME!r} package entry")

    return {
        "pyproject": str(project["version"]),
        "import": match.group(1),
        "lock": str(lock_matches[0]["version"]),
        "readme": readme_match.group(1),
    }


def check_version(root: Path, version: str) -> None:
    require_version(version)
    versions = project_versions(root)
    mismatches = {source: value for source, value in versions.items() if value != version}
    if mismatches:
        fail(f"version {version} does not match all sources: {mismatches}")
    print(json.dumps({"version": version, "sources": versions}, sort_keys=True))


def pypi_payload() -> dict | None:
    request = urllib.request.Request(
        f"https://pypi.org/pypi/{PACKAGE_NAME}/json",
        headers={"Accept": "application/json", "User-Agent": "gradlab-release-check"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def pypi_release_files(payload: dict | None, version: str) -> list[dict]:
    if payload is None:
        return []
    return list(payload.get("releases", {}).get(version, []))


def check_pypi(version: str) -> None:
    require_version(version)
    payload = pypi_payload()
    files = pypi_release_files(payload, version)
    if files:
        names = sorted(str(item.get("filename")) for item in files)
        fail(f"PyPI {PACKAGE_NAME} {version} already has files: {names}")
    state = "project-not-yet-public" if payload is None else "version-unused"
    print(json.dumps({"package": PACKAGE_NAME, "version": version, "state": state}))


def next_unused_patch(version: str, payload: dict | None) -> str:
    require_version(version)
    major, minor, patch = (int(part) for part in version.split("."))
    candidate = version
    while pypi_release_files(payload, candidate):
        patch += 1
        candidate = f"{major}.{minor}.{patch}"
    return candidate


def bump_version_sources(root: Path, current: str, target: str) -> None:
    require_version(current)
    require_version(target)
    if current == target:
        return

    pyproject_path = root / "pyproject.toml"
    import_path = root / "src" / IMPORT_NAME / "__init__.py"
    lock_path = root / "uv.lock"
    readme_path = root / "README.md"
    paths = (pyproject_path, import_path, lock_path, readme_path)
    snapshots = {path: path.read_bytes() for path in paths}
    import_text = snapshots[import_path].decode()
    updated_import, replacements = re.subn(
        rf'^__version__\s*=\s*"{re.escape(current)}"$',
        f'__version__ = "{target}"',
        import_text,
        count=1,
        flags=re.MULTILINE,
    )
    if replacements != 1:
        fail(f"could not replace import version {current!r}")

    readme_text = snapshots[readme_path].decode()
    updated_readme, readme_replacements = re.subn(
        rf"^uvx gradlab@{re.escape(current)} train gradlab__bandit/ppo$",
        f"uvx gradlab@{target} train gradlab__bandit/ppo",
        readme_text,
        count=1,
        flags=re.MULTILINE,
    )
    if readme_replacements != 1:
        fail(f"could not replace README version {current!r}")

    try:
        run(["uv", "version", target, "--no-sync"], cwd=root)
        import_path.write_text(updated_import, encoding="utf-8")
        readme_path.write_text(updated_readme, encoding="utf-8")
        check_version(root, target)
    except Exception:
        for path, contents in snapshots.items():
            path.write_bytes(contents)
        raise


def auto_bump_version(root: Path) -> str:
    versions = project_versions(root)
    unique_versions = set(versions.values())
    if len(unique_versions) != 1:
        fail(f"cannot auto-bump inconsistent version sources: {versions}")
    current = unique_versions.pop()
    require_version(current)
    target = next_unused_patch(current, pypi_payload())
    bump_version_sources(root, current, target)
    print(
        json.dumps(
            {
                "auto_bump": "applied" if target != current else "not-needed",
                "previous_version": current,
                "version": target,
            },
            sort_keys=True,
        )
    )
    return target


def prepare_version(root: Path, requested: str | None) -> str:
    versions = project_versions(root)
    unique_versions = set(versions.values())
    if len(unique_versions) != 1:
        fail(f"cannot prepare inconsistent version sources: {versions}")
    current = unique_versions.pop()
    require_version(current)
    target = requested or next_unused_patch(current, pypi_payload())
    require_version(target)
    bump_version_sources(root, current, target)
    check_pypi(target)
    print(
        json.dumps(
            {
                "prepared": "bumped" if target != current else "unchanged",
                "previous_version": current,
                "version": target,
            },
            sort_keys=True,
        )
    )
    return target


def expected_names(version: str) -> tuple[str, str]:
    return (
        f"{PACKAGE_NAME}-{version}-py3-none-any.whl",
        f"{PACKAGE_NAME}-{version}.tar.gz",
    )


def contaminated(member_name: str) -> bool:
    parts = set(PurePosixPath(member_name).parts)
    return bool(parts & CONTAMINATION_PARTS) or member_name.endswith((".pyc", ".pyo"))


def audit_wheel(path: Path, version: str) -> None:
    expected_wheel, _ = expected_names(version)
    if path.name != expected_wheel:
        fail(f"expected wheel {expected_wheel}, got {path.name}")

    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if any(contaminated(name) for name in names):
            fail(f"wheel contains generated or private paths: {path}")
        required = {
            f"{IMPORT_NAME}/__init__.py",
            f"{IMPORT_NAME}/METRICS.md",
        }
        missing = sorted(required - set(names))
        if missing:
            fail(f"wheel is missing required files: {missing}")

        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        entry_names = [name for name in names if name.endswith(".dist-info/entry_points.txt")]
        if len(metadata_names) != 1 or len(entry_names) != 1:
            fail("wheel must contain one METADATA and one entry_points.txt")
        metadata = archive.read(metadata_names[0]).decode()
        if f"Name: {PACKAGE_NAME}\n" not in metadata:
            fail("wheel METADATA has the wrong package name")
        if f"Version: {version}\n" not in metadata:
            fail("wheel METADATA has the wrong version")
        if ENTRY_POINT not in archive.read(entry_names[0]).decode():
            fail(f"wheel entry points do not contain {ENTRY_POINT!r}")

def audit_sdist(path: Path, version: str) -> None:
    _, expected_sdist = expected_names(version)
    if path.name != expected_sdist:
        fail(f"expected sdist {expected_sdist}, got {path.name}")

    prefix = f"{PACKAGE_NAME}-{version}/"
    with tarfile.open(path, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        names = [member.name for member in members]
        if not names or any(not name.startswith(prefix) for name in names):
            fail(f"sdist members must be rooted at {prefix}")
        if any(contaminated(name) for name in names):
            fail(f"sdist contains generated or private paths: {path}")
        required = {
            f"{prefix}pyproject.toml",
            f"{prefix}README.md",
            f"{prefix}METRICS.md",
            f"{prefix}src/{IMPORT_NAME}/__init__.py",
        }
        missing = sorted(required - set(names))
        if missing:
            fail(f"sdist is missing required files: {missing}")

def distribution_paths(dist_dir: Path, version: str) -> tuple[Path, Path]:
    wheel_name, sdist_name = expected_names(version)
    wheel = dist_dir / wheel_name
    sdist = dist_dir / sdist_name
    missing = [str(path) for path in (wheel, sdist) if not path.is_file()]
    if missing:
        fail(f"missing release distributions: {missing}")
    unexpected = sorted(
        path.name
        for path in dist_dir.iterdir()
        if path.is_file() and path not in {wheel, sdist} and path.name != ".gitignore"
    )
    if unexpected:
        fail(f"release directory contains unexpected files: {unexpected}")
    return wheel, sdist


def audit(dist_dir: Path, version: str) -> tuple[Path, Path]:
    wheel, sdist = distribution_paths(dist_dir, version)
    audit_wheel(wheel, version)
    audit_sdist(sdist, version)
    print(json.dumps({"audited": [str(wheel), str(sdist)]}, sort_keys=True))
    return wheel, sdist


def smoke_wheel(wheel: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        fail("uv is required for wheel smoke testing")
    with tempfile.TemporaryDirectory(prefix="gradlab-release-smoke-") as temporary:
        env_dir = Path(temporary) / "venv"
        run([uv, "venv", "--python", "3.14", str(env_dir)])
        python = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        run([uv, "pip", "install", "--python", str(python), "--no-deps", str(wheel)])
        code = (
            "import importlib.metadata as m, gradlab; "
            "assert gradlab.__version__ == m.version('gradlab'); "
            "eps=[e for e in m.entry_points(group='console_scripts') if e.name=='gradlab']; "
            "assert len(eps)==1 and eps[0].value=='gradlab.main:main'"
        )
        run([str(python), "-c", code])
        script = env_dir / ("Scripts/gradlab.exe" if os.name == "nt" else "bin/gradlab")
        run([str(script), "--help"])
    print(json.dumps({"smoke_tested": str(wheel)}))


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def build(root: Path, version: str, out_dir: Path) -> None:
    check_version(root, version)
    check_pypi(version)
    if out_dir.exists() and any(out_dir.iterdir()):
        fail(f"release output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    run(["uv", "build", "--out-dir", str(out_dir)], cwd=root)
    wheel, sdist = audit(out_dir, version)
    run(
        [
            "uv",
            "run",
            "--group",
            "release",
            "python",
            "-m",
            "twine",
            "check",
            str(wheel),
            str(sdist),
        ],
        cwd=root,
    )
    smoke_wheel(wheel)
    print(
        json.dumps(
            {
                "package": PACKAGE_NAME,
                "version": version,
                "artifacts": {
                    str(wheel): {"sha256": digest(wheel), "bytes": wheel.stat().st_size},
                    str(sdist): {"sha256": digest(sdist), "bytes": sdist.stat().st_size},
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    for name in ("check-version", "check-pypi"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--version", required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--version", required=True)
    audit_parser.add_argument("dist_dir", type=Path)
    smoke_parser = subparsers.add_parser("smoke-wheel")
    smoke_parser.add_argument("wheel", type=Path)
    prepare_parser = subparsers.add_parser("prepare-version")
    prepare_parser.add_argument("--version")
    build_parser = subparsers.add_parser("build")
    version_source = build_parser.add_mutually_exclusive_group(required=True)
    version_source.add_argument("--version")
    version_source.add_argument("--auto-bump", action="store_true")
    build_parser.add_argument("--out-dir", type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    root = Path(__file__).resolve().parents[4]
    try:
        if args.command == "check-version":
            check_version(root, args.version)
        elif args.command == "check-pypi":
            check_pypi(args.version)
        elif args.command == "audit":
            audit(args.dist_dir.resolve(), args.version)
        elif args.command == "smoke-wheel":
            smoke_wheel(args.wheel.resolve())
        elif args.command == "prepare-version":
            prepare_version(root, args.version)
        elif args.command == "build":
            if args.auto_bump:
                if args.out_dir is not None:
                    fail("--out-dir cannot be combined with --auto-bump")
                version = auto_bump_version(root)
                out_dir = root / "dist" / f"release-v{version}"
            else:
                if args.out_dir is None:
                    fail("--out-dir is required with --version")
                version = args.version
                out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
            build(root, version, out_dir.resolve())
        else:
            fail(f"unsupported command: {args.command}")
    except (ReleaseError, subprocess.CalledProcessError) as error:
        print(f"release error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
