from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".codex"
    / "skills"
    / "build-release"
    / "scripts"
    / "release_build.py"
)
SPEC = importlib.util.spec_from_file_location("gradlab_release_build", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
release_build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release_build)

LAUNCHER_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "release.py"
LAUNCHER_SPEC = importlib.util.spec_from_file_location("gradlab_release_launcher", LAUNCHER_SCRIPT)
assert LAUNCHER_SPEC is not None and LAUNCHER_SPEC.loader is not None
release_launcher = importlib.util.module_from_spec(LAUNCHER_SPEC)
LAUNCHER_SPEC.loader.exec_module(release_launcher)


def test_next_unused_patch_keeps_an_unused_version() -> None:
    payload = {"releases": {"0.1.1": []}}

    assert release_build.next_unused_patch("0.1.1", payload) == "0.1.1"


def test_next_unused_patch_skips_all_uploaded_versions() -> None:
    payload = {
        "releases": {
            "0.1.1": [{"filename": "gradlab-0.1.1.tar.gz"}],
            "0.1.2": [
                {
                    "filename": "gradlab-0.1.2.tar.gz",
                    "yanked": True,
                }
            ],
        }
    }

    assert release_build.next_unused_patch("0.1.1", payload) == "0.1.3"


def test_build_parser_supports_local_autobump() -> None:
    args = release_build.parser().parse_args(["build", "--auto-bump"])

    assert args.command == "build"
    assert args.auto_bump is True
    assert args.version is None
    assert args.out_dir is None


def test_prepare_version_parser_defaults_to_next_unused_patch() -> None:
    args = release_build.parser().parse_args(["prepare-version"])

    assert args.command == "prepare-version"
    assert args.version is None


def test_release_launcher_defaults_to_full_next_release() -> None:
    args = release_launcher.parse_args([])

    assert args.to is None
    assert args.part is None
    assert args.dry_run_push is False


def test_release_launcher_computes_requested_bumps() -> None:
    assert release_launcher.next_version("1.2.3", "patch") == "1.2.4"
    assert release_launcher.next_version("1.2.3", "minor") == "1.3.0"
    assert release_launcher.next_version("1.2.3", "major") == "2.0.0"
