from __future__ import annotations

import fcntl
import json
import os
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from gradlab.json_utils import canonical_json_text


ENTRY_CACHE_SCHEMA_VERSION = 1
DEFAULT_MAX_ENTRIES = 128
DEFAULT_MAX_BYTES = 32 * 1024 * 1024


class CatalogEntryCache:
    """Bounded per-entry cache with process and inter-process slot locking."""

    def __init__(
        self,
        root: Path | str,
        *,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        self._guard = threading.Lock()
        self._thread_locks: dict[str, threading.RLock] = {}

    def _slot_name(self, namespace: str, key: str) -> str:
        safe_namespace = "".join(
            character
            for character in str(namespace)
            if character.isascii() and (character.isalnum() or character in "-_")
        )
        safe_key = "".join(
            character
            for character in str(key)
            if character.isascii() and (character.isalnum() or character in "-_")
        )
        if not safe_namespace or not safe_key:
            raise ValueError("catalog cache slot identity is invalid")
        return f"{safe_namespace}__{safe_key}"

    def _entry_path(self, namespace: str, key: str) -> Path:
        return self.root / "entries" / f"{self._slot_name(namespace, key)}.json"

    def _lock_path(self, namespace: str, key: str) -> Path:
        return self.root / "locks" / f"{self._slot_name(namespace, key)}.lock"

    def _thread_lock(self, slot_name: str) -> threading.RLock:
        with self._guard:
            return self._thread_locks.setdefault(slot_name, threading.RLock())

    @contextmanager
    def slot_lock(self, namespace: str, key: str) -> Iterator[None]:
        slot_name = self._slot_name(namespace, key)
        thread_lock = self._thread_lock(slot_name)
        with thread_lock:
            lock_path = self._lock_path(namespace, key)
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("a+b") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def read(self, namespace: str, key: str) -> dict[str, Any] | None:
        path = self._entry_path(namespace, key)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != ENTRY_CACHE_SCHEMA_VERSION
            or payload.get("namespace") != namespace
            or payload.get("key") != key
            or not isinstance(payload.get("value"), Mapping)
        ):
            return None
        try:
            os.utime(path, None)
        except OSError:
            pass
        return dict(payload["value"])

    def write(
        self,
        namespace: str,
        key: str,
        value: Mapping[str, Any],
    ) -> None:
        path = self._entry_path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": ENTRY_CACHE_SCHEMA_VERSION,
            "namespace": namespace,
            "key": key,
            "value": dict(value),
        }
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            temporary.write_text(
                canonical_json_text(payload, ensure_ascii=True),
                encoding="utf-8",
            )
            temporary.replace(path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        self.prune()

    def prune(self) -> None:
        entries_root = self.root / "entries"
        try:
            entries = [
                path
                for path in entries_root.glob("*.json")
                if path.is_file()
            ]
        except OSError:
            return
        sized: list[tuple[int, int, Path]] = []
        for path in entries:
            try:
                stat_result = path.stat()
            except OSError:
                continue
            sized.append((stat_result.st_mtime_ns, stat_result.st_size, path))
        sized.sort()
        count = len(sized)
        total = sum(size for _mtime, size, _path in sized)
        for _mtime, size, path in sized:
            if count <= self.max_entries and total <= self.max_bytes:
                break
            slot_name = path.stem
            namespace, separator, key = slot_name.partition("__")
            if not separator:
                continue
            lock = self._thread_lock(slot_name)
            if not lock.acquire(blocking=False):
                continue
            try:
                lock_path = self._lock_path(namespace, key)
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                with lock_path.open("a+b") as handle:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    try:
                        try:
                            current = path.stat()
                        except FileNotFoundError:
                            continue
                        if current.st_mtime_ns != _mtime or current.st_size != size:
                            continue
                        path.unlink()
                        count -= 1
                        total -= size
                    finally:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                lock.release()


__all__ = [
    "CatalogEntryCache",
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_ENTRIES",
    "ENTRY_CACHE_SCHEMA_VERSION",
]
