from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gradlab.json_utils import canonical_json_bytes


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_path(path: str | Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_tree(path: str | Path) -> None:
    root = Path(path)
    directories = [root]
    for entry in root.rglob("*"):
        if entry.is_file():
            fsync_path(entry)
        elif entry.is_dir():
            directories.append(entry)
    for directory in sorted(
        directories,
        key=lambda candidate: len(candidate.parts),
        reverse=True,
    ):
        fsync_path(directory)


def atomic_write_bytes(
    path: str | Path,
    payload: bytes | bytearray | memoryview,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        fsync_path(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: str | Path, document: Mapping[str, Any]) -> None:
    atomic_write_bytes(path, canonical_json_bytes(document, ensure_ascii=True))
