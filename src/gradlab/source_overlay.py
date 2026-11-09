"""Verify a recorded source overlay without changing a saved recipe."""

import json
import re
from pathlib import Path, PurePosixPath

from gradlab.file_utils import file_sha256
from gradlab.json_utils import canonical_json_sha256


def _digest_map(value):
    if not isinstance(value, dict) or not value:
        raise ValueError("expected a nonempty source digest map")
    for name, digest in value.items():
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != name
            or not re.fullmatch(r"[0-9a-f]{64}", str(digest))
        ):
            raise ValueError(f"invalid source digest entry: {name}")
    return value


def verify_source_tree(root: Path, files: dict) -> None:
    root = root.resolve()
    for name, digest in _digest_map(files).items():
        path = root / name
        if not path.resolve().is_relative_to(root) or file_sha256(path) != digest:
            raise ValueError(f"source file digest mismatch: {name}")
    for path in (root / "src" / "gradlab").rglob("*.py"):
        if path.relative_to(root).as_posix() not in files:
            raise ValueError(f"unrecorded GradLab source file: {path.relative_to(root)}")


def validate_source_overlay(receipt: dict, *, base_commit: str) -> None:
    if set(receipt) != {
        "base_commit", "overlay_sha256", "overlay_files", "effective_files", "input_receipt_sha256",
    }:
        raise ValueError("invalid source overlay receipt fields")
    changes = _digest_map(receipt["overlay_files"])
    files = _digest_map(receipt["effective_files"])
    if (
        receipt["base_commit"] != base_commit
        or receipt["overlay_sha256"] != canonical_json_sha256(changes)
        or not re.fullmatch(r"[0-9a-f]{64}", str(receipt["input_receipt_sha256"]))
        or any(files.get(name) != digest for name, digest in changes.items())
    ):
        raise ValueError("source overlay receipt has inconsistent provenance")


def read_source_overlay(
    root: Path, baseline_receipt: Path, input_receipt: Path, *, recipe: dict
) -> dict:
    baseline = json.loads(baseline_receipt.read_bytes())
    inputs = _digest_map(json.loads(input_receipt.read_bytes()))
    if inputs.get(baseline_receipt.name) != file_sha256(baseline_receipt):
        raise ValueError("source baseline receipt is not bound by the recorded input receipt")
    commit = baseline["commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", str(commit)):
        raise ValueError("overlay base must identify a full Git commit")
    changes = _digest_map(
        {name.removeprefix("overlay/"): digest for name, digest in inputs.items()
         if name.startswith("overlay/")}
    )
    files = {**_digest_map(baseline["sha256"]), **changes}
    verify_source_tree(root, files)
    for entry in recipe["provenance"]["source_files"]:
        if files.get(entry["path"]) != entry["sha256"]:
            raise ValueError(f"overlay does not match recorded recipe source: {entry['path']}")
    return {
        "base_commit": commit,
        "overlay_sha256": canonical_json_sha256(changes),
        "overlay_files": changes,
        "effective_files": files,
        "input_receipt_sha256": file_sha256(input_receipt),
    }
