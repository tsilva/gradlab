from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from gradlab.file_utils import atomic_write_bytes, atomic_write_json
from gradlab.json_utils import canonical_json_bytes, canonical_json_sha256, json_safe


STATE_ARCHIVE_SEMANTIC_ID = "state-archive-v1"
SNAPSHOT_CODEC_API_VERSION = 1
STATE_ARCHIVE_CURRICULUM_SEMANTIC_ID = "archive-curriculum-v1"
RESTORE_SEMANTICS = frozenset({"episode_start", "continuation"})
STATE_ARCHIVE_PERSISTENCE = frozenset({"durable", "ephemeral"})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_VIEW_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")


def _require_sha256(value: str, *, label: str) -> str:
    normalized = str(value).strip().lower()
    if not _SHA256.fullmatch(normalized):
        raise ValueError(f"{label} must contain 64 lowercase hexadecimal characters")
    return normalized


def _immutable_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(json_safe(value)))


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} fields disagree; "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )


@dataclass(frozen=True)
class SnapshotRef:
    codec_id: str
    blob_sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not self.codec_id:
            raise ValueError("snapshot codec_id must not be empty")
        object.__setattr__(
            self,
            "blob_sha256",
            _require_sha256(self.blob_sha256, label="snapshot blob_sha256"),
        )
        if self.size_bytes < 1:
            raise ValueError("snapshot size_bytes must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "codec_id": self.codec_id,
            "blob_sha256": self.blob_sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SnapshotRef:
        _require_exact_fields(
            value,
            frozenset({"codec_id", "blob_sha256", "size_bytes"}),
            label="snapshot reference",
        )
        return cls(
            codec_id=str(value["codec_id"]),
            blob_sha256=str(value["blob_sha256"]),
            size_bytes=int(value["size_bytes"]),
        )


@dataclass(frozen=True)
class ProviderSnapshot:
    provider_id: str
    compatibility_id: str
    ref: SnapshotRef

    def __post_init__(self) -> None:
        if not self.provider_id or not self.compatibility_id:
            raise ValueError("provider snapshot identity fields must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "compatibility_id": self.compatibility_id,
            "ref": self.ref.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProviderSnapshot:
        _require_exact_fields(
            value,
            frozenset({"provider_id", "compatibility_id", "ref"}),
            label="provider snapshot",
        )
        return cls(
            provider_id=str(value["provider_id"]),
            compatibility_id=str(value["compatibility_id"]),
            ref=SnapshotRef.from_dict(value["ref"]),
        )


@dataclass(frozen=True)
class TaskLaneState:
    schema_id: str
    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.schema_id:
            raise ValueError("task lane state schema_id must not be empty")
        object.__setattr__(self, "values", _immutable_mapping(self.values))

    def to_dict(self) -> dict[str, Any]:
        return {"schema_id": self.schema_id, "values": dict(self.values)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TaskLaneState:
        _require_exact_fields(
            value,
            frozenset({"schema_id", "values"}),
            label="task lane state",
        )
        return cls(schema_id=str(value["schema_id"]), values=dict(value["values"]))


@dataclass(frozen=True)
class RuntimeLaneState:
    episode_index: int
    episode_return: float
    episode_length: int
    episode_seed: int | None
    start_id: str | None
    start_origin: str

    def __post_init__(self) -> None:
        if self.episode_index < 0 or self.episode_length < 0:
            raise ValueError("runtime lane counters must be non-negative")
        if not math.isfinite(self.episode_return):
            raise ValueError("runtime episode_return must be finite")
        if not self.start_origin:
            raise ValueError("runtime start_origin must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "episode_return": self.episode_return,
            "episode_length": self.episode_length,
            "episode_seed": self.episode_seed,
            "start_id": self.start_id,
            "start_origin": self.start_origin,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RuntimeLaneState:
        _require_exact_fields(
            value,
            frozenset(
                {
                    "episode_index",
                    "episode_return",
                    "episode_length",
                    "episode_seed",
                    "start_id",
                    "start_origin",
                }
            ),
            label="runtime lane state",
        )
        seed = value["episode_seed"]
        return cls(
            episode_index=int(value["episode_index"]),
            episode_return=float(value["episode_return"]),
            episode_length=int(value["episode_length"]),
            episode_seed=None if seed is None else int(seed),
            start_id=None if value["start_id"] is None else str(value["start_id"]),
            start_origin=str(value["start_origin"]),
        )


@dataclass(frozen=True)
class StateArchiveEntry:
    entry_id: str
    provider_snapshot: ProviderSnapshot
    task_state: TaskLaneState | None
    runtime_state: RuntimeLaneState | None
    restore_semantics: str
    created_step: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
    semantic_id: str = STATE_ARCHIVE_SEMANTIC_ID
    schema_version: int = 1

    def __post_init__(self) -> None:
        _require_sha256(self.entry_id, label="archive entry_id")
        if self.semantic_id != STATE_ARCHIVE_SEMANTIC_ID or self.schema_version != 1:
            raise ValueError("state archive entry schema is unsupported")
        if self.restore_semantics not in RESTORE_SEMANTICS:
            raise ValueError(f"restore_semantics must be one of {sorted(RESTORE_SEMANTICS)}")
        if self.created_step < 0:
            raise ValueError("created_step must be non-negative")
        if self.restore_semantics == "continuation" and (
            self.task_state is None or self.runtime_state is None
        ):
            raise ValueError("continuation entries require task and runtime lane state")
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))

    def identity_dict(self) -> dict[str, Any]:
        return {
            "semantic_id": self.semantic_id,
            "schema_version": self.schema_version,
            "provider_snapshot": self.provider_snapshot.to_dict(),
            "task_state": None if self.task_state is None else self.task_state.to_dict(),
            "runtime_state": (None if self.runtime_state is None else self.runtime_state.to_dict()),
            "restore_semantics": self.restore_semantics,
            "created_step": self.created_step,
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"entry_id": self.entry_id, **self.identity_dict()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StateArchiveEntry:
        _require_exact_fields(
            value,
            frozenset(
                {
                    "entry_id",
                    "semantic_id",
                    "schema_version",
                    "provider_snapshot",
                    "task_state",
                    "runtime_state",
                    "restore_semantics",
                    "created_step",
                    "metadata",
                }
            ),
            label="state archive entry",
        )
        task_state = value["task_state"]
        runtime_state = value["runtime_state"]
        entry = cls(
            entry_id=str(value["entry_id"]),
            provider_snapshot=ProviderSnapshot.from_dict(value["provider_snapshot"]),
            task_state=(None if task_state is None else TaskLaneState.from_dict(task_state)),
            runtime_state=(
                None if runtime_state is None else RuntimeLaneState.from_dict(runtime_state)
            ),
            restore_semantics=str(value["restore_semantics"]),
            created_step=int(value["created_step"]),
            metadata=dict(value["metadata"]),
            semantic_id=str(value["semantic_id"]),
            schema_version=int(value["schema_version"]),
        )
        expected = canonical_json_sha256(entry.identity_dict())
        if entry.entry_id != expected:
            raise ValueError(
                f"archive entry hash mismatch: expected {entry.entry_id}, got {expected}"
            )
        return entry


class ContentAddressedBlobStore:
    """Immutable local content-addressed bytes; no Python object serialization."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, sha256: str) -> Path:
        digest = _require_sha256(sha256, label="blob sha256")
        return self.root / digest[:2] / digest[2:]

    def put(self, payload: bytes | bytearray | memoryview) -> SnapshotRef:
        value = bytes(payload)
        if not value:
            raise ValueError("snapshot payload must not be empty")
        digest = hashlib.sha256(value).hexdigest()
        path = self.path_for(digest)
        if path.exists():
            existing = path.read_bytes()
            if existing != value:
                raise RuntimeError("content-addressed snapshot blob collision")
            return SnapshotRef(codec_id="unbound", blob_sha256=digest, size_bytes=len(value))
        atomic_write_bytes(path, value)
        return SnapshotRef(codec_id="unbound", blob_sha256=digest, size_bytes=len(value))

    def get(self, ref: SnapshotRef) -> bytes:
        payload = self.path_for(ref.blob_sha256).read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != ref.blob_sha256 or len(payload) != ref.size_bytes:
            raise ValueError("state archive snapshot blob failed integrity verification")
        return payload

    def discard(self, sha256: str) -> None:
        path = self.path_for(sha256)
        path.unlink(missing_ok=True)
        try:
            path.parent.rmdir()
        except OSError:
            pass


class SessionHandleStore:
    """Optional live-handle cache keyed by immutable provider payload digest."""

    def __init__(self) -> None:
        self._handles: dict[str, Any] = {}

    def put(self, ref: SnapshotRef, handle: Any) -> None:
        if handle is not None:
            self._handles[ref.blob_sha256] = handle

    def get(self, ref: SnapshotRef) -> Any | None:
        return self._handles.get(ref.blob_sha256)

    def retain(self, blob_sha256s: set[str]) -> None:
        self._handles = {
            digest: handle for digest, handle in self._handles.items() if digest in blob_sha256s
        }

    def clear(self) -> None:
        self._handles.clear()


class SnapshotCodec:
    api_version = SNAPSHOT_CODEC_API_VERSION
    codec_id: str
    provider_id: str

    def capture(
        self,
        provider: Any,
        mask: np.ndarray,
    ) -> tuple[tuple[bytes | None, ...], tuple[Any | None, ...]]:
        raise NotImplementedError

    def restore(
        self,
        provider: Any,
        payloads: Sequence[bytes | None],
        mask: np.ndarray,
    ) -> tuple[Any | None, ...]:
        raise NotImplementedError


class _MarioPortableCodec(SnapshotCodec):
    codec_id = "supermariobrosnes-turbo.portable-v1"
    provider_id = "supermariobrosnes-turbo"

    def capture(self, provider: Any, mask: np.ndarray):
        handles = tuple(provider.capture_snapshots(mask))
        payloads = tuple(provider.encode_snapshots(handles))
        return payloads, handles

    def restore(self, provider: Any, payloads: Sequence[bytes | None], mask: np.ndarray):
        if any(payloads[index] is not None and not bool(mask[index]) for index in range(len(mask))):
            raise ValueError("portable payloads may only select restored lanes")
        return tuple(provider.decode_snapshots(payloads))


class _BreakoutPortableCodec(SnapshotCodec):
    codec_id = "breakout-turbo-env.state-v1"
    provider_id = "breakout-turbo-env"

    def capture(self, provider: Any, mask: np.ndarray):
        handles = tuple(provider.capture_snapshots(mask))
        states = tuple(provider.get_state())
        payloads = tuple(states[index] if bool(mask[index]) else None for index in range(len(mask)))
        return payloads, handles

    def restore(self, provider: Any, payloads: Sequence[bytes | None], mask: np.ndarray):
        states = list(provider.get_state())
        for lane in np.flatnonzero(mask):
            payload = payloads[int(lane)]
            if payload is None:
                raise ValueError(f"archive restore lane {int(lane)} has no provider payload")
            states[int(lane)] = payload
        provider.set_state(states, reset_mask=mask)
        return tuple(provider.capture_snapshots(mask))


class SnapshotCodecRegistry:
    """Strict provider/codec dispatch. Registration is explicit; no probing."""

    def __init__(self, codecs: Sequence[SnapshotCodec] = ()):
        self._codecs: dict[str, SnapshotCodec] = {}
        for codec in codecs:
            self.register(codec)

    def register(self, codec: SnapshotCodec) -> None:
        if codec.api_version != SNAPSHOT_CODEC_API_VERSION:
            raise ValueError("snapshot codec API version is unsupported")
        if codec.codec_id in self._codecs:
            raise ValueError(f"duplicate snapshot codec id {codec.codec_id!r}")
        self._codecs[codec.codec_id] = codec

    def resolve(self, codec_id: str, *, provider_id: str) -> SnapshotCodec:
        try:
            codec = self._codecs[codec_id]
        except KeyError as exc:
            raise ValueError(f"unregistered snapshot codec {codec_id!r}") from exc
        if codec.provider_id != provider_id:
            raise ValueError(
                f"snapshot codec {codec_id!r} belongs to provider "
                f"{codec.provider_id!r}, not {provider_id!r}"
            )
        return codec


SNAPSHOT_CODECS = SnapshotCodecRegistry((_MarioPortableCodec(), _BreakoutPortableCodec()))


class StateArchive:
    """Immutable entry/blob store with mutable algorithm-owned views."""

    def __init__(
        self,
        root: str | Path,
        *,
        provider_id: str,
        codec_id: str,
        compatibility_id: str,
        persistence: str,
        codec_registry: SnapshotCodecRegistry = SNAPSHOT_CODECS,
    ) -> None:
        self.root = Path(root)
        self.entries_root = self.root / "entries"
        self.views_root = self.root / "views"
        self.blobs = ContentAddressedBlobStore(self.root / "blobs")
        self.handles = SessionHandleStore()
        self.provider_id = str(provider_id)
        self.codec_id = str(codec_id)
        self.compatibility_id = str(compatibility_id)
        self.persistence = str(persistence)
        if self.persistence not in STATE_ARCHIVE_PERSISTENCE:
            raise ValueError(
                f"state archive persistence must be one of {sorted(STATE_ARCHIVE_PERSISTENCE)}"
            )
        self.codec = codec_registry.resolve(codec_id, provider_id=provider_id)
        self._entries: dict[str, StateArchiveEntry] = {}
        self._views: dict[str, dict[str, Any]] = {}
        self._blob_ref_counts: dict[str, int] = {}
        self._blob_sizes: dict[str, int] = {}
        self._blob_bytes = 0
        self.entries_root.mkdir(parents=True, exist_ok=True)
        self.views_root.mkdir(parents=True, exist_ok=True)
        for path in sorted(self.entries_root.glob("*.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            entry = StateArchiveEntry.from_dict(raw)
            self._validate_compatibility(entry)
            self._entries[entry.entry_id] = entry
            self._track_entry_blob(entry)
        for path in sorted(self.views_root.glob("*.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            view_id = path.stem
            self._views[view_id] = self._validate_view(view_id, raw)

    def _validate_compatibility(self, entry: StateArchiveEntry) -> None:
        snapshot = entry.provider_snapshot
        if snapshot.provider_id != self.provider_id:
            raise ValueError("archive entry provider mismatch")
        if snapshot.compatibility_id != self.compatibility_id:
            raise ValueError("archive entry environment compatibility mismatch")
        if snapshot.ref.codec_id != self.codec_id:
            raise ValueError("archive entry snapshot codec mismatch")

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def is_closed(self) -> bool:
        path = self.root / "closure.json"
        if not path.is_file():
            return False
        value = json.loads(path.read_text(encoding="utf-8"))
        return value.get("status") == "closed"

    def _track_entry_blob(self, entry: StateArchiveEntry) -> None:
        ref = entry.provider_snapshot.ref
        digest = ref.blob_sha256
        size = int(ref.size_bytes)
        known_size = self._blob_sizes.get(digest)
        if known_size is not None and known_size != size:
            raise RuntimeError("state archive blob size changed for the same digest")
        references = self._blob_ref_counts.get(digest, 0)
        if references == 0:
            self._blob_sizes[digest] = size
            self._blob_bytes += size
        self._blob_ref_counts[digest] = references + 1

    def _untrack_entry_blob(self, entry: StateArchiveEntry) -> None:
        digest = entry.provider_snapshot.ref.blob_sha256
        references = self._blob_ref_counts[digest]
        if references > 1:
            self._blob_ref_counts[digest] = references - 1
            return
        del self._blob_ref_counts[digest]
        self._blob_bytes -= self._blob_sizes.pop(digest)

    def create_entry(
        self,
        *,
        payload: bytes,
        handle: Any,
        task_state: TaskLaneState | None,
        runtime_state: RuntimeLaneState | None,
        restore_semantics: str,
        created_step: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> StateArchiveEntry:
        unbound_ref = self.blobs.put(payload)
        ref = SnapshotRef(
            codec_id=self.codec_id,
            blob_sha256=unbound_ref.blob_sha256,
            size_bytes=unbound_ref.size_bytes,
        )
        provider_snapshot = ProviderSnapshot(
            provider_id=self.provider_id,
            compatibility_id=self.compatibility_id,
            ref=ref,
        )
        identity = {
            "semantic_id": STATE_ARCHIVE_SEMANTIC_ID,
            "schema_version": 1,
            "provider_snapshot": provider_snapshot.to_dict(),
            "task_state": None if task_state is None else task_state.to_dict(),
            "runtime_state": None if runtime_state is None else runtime_state.to_dict(),
            "restore_semantics": restore_semantics,
            "created_step": int(created_step),
            "metadata": dict(json_safe(metadata or {})),
        }
        entry_id = canonical_json_sha256(identity)
        entry = StateArchiveEntry(
            entry_id=entry_id,
            provider_snapshot=provider_snapshot,
            task_state=task_state,
            runtime_state=runtime_state,
            restore_semantics=restore_semantics,
            created_step=int(created_step),
            metadata=identity["metadata"],
        )
        path = self.entries_root / f"{entry_id}.json"
        if path.exists():
            existing = StateArchiveEntry.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if existing != entry:
                raise RuntimeError("state archive entry hash collision")
        else:
            atomic_write_json(path, entry.to_dict())
        if entry_id not in self._entries:
            self._track_entry_blob(entry)
        self._entries[entry_id] = entry
        self.handles.put(ref, handle)
        return entry

    def import_entry(
        self,
        value: Mapping[str, Any],
        payload: bytes,
    ) -> StateArchiveEntry:
        """Import one portable entry after verifying its identity and payload."""

        entry = StateArchiveEntry.from_dict(value)
        self._validate_compatibility(entry)
        ref = entry.provider_snapshot.ref
        raw_payload = bytes(payload)
        if (
            hashlib.sha256(raw_payload).hexdigest() != ref.blob_sha256
            or len(raw_payload) != ref.size_bytes
        ):
            raise ValueError("imported state archive payload failed integrity verification")
        stored = self.blobs.put(raw_payload)
        if stored.blob_sha256 != ref.blob_sha256 or stored.size_bytes != ref.size_bytes:
            raise ValueError("imported state archive payload identity mismatch")
        existing = self._entries.get(entry.entry_id)
        if existing is not None:
            if existing != entry:
                raise RuntimeError("state archive entry hash collision")
            return existing
        path = self.entries_root / f"{entry.entry_id}.json"
        if path.exists():
            on_disk = StateArchiveEntry.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if on_disk != entry:
                raise RuntimeError("state archive entry hash collision")
        else:
            atomic_write_json(path, entry.to_dict())
        self._entries[entry.entry_id] = entry
        self._track_entry_blob(entry)
        return entry

    def entry(self, entry_id: str) -> StateArchiveEntry:
        try:
            return self._entries[str(entry_id)]
        except KeyError as exc:
            raise KeyError(f"unknown state archive entry {entry_id!r}") from exc

    def payload(self, entry_id: str) -> bytes:
        return self.blobs.get(self.entry(entry_id).provider_snapshot.ref)

    def live_handle(self, entry_id: str) -> Any | None:
        return self.handles.get(self.entry(entry_id).provider_snapshot.ref)

    @staticmethod
    def _normalized_view_id(view_id: str) -> str:
        normalized = str(view_id).strip()
        if _VIEW_ID.fullmatch(normalized) is None:
            raise ValueError(
                "archive view_id must contain 1-64 lowercase letters, digits, or hyphens"
            )
        return normalized

    def _validate_view(self, view_id: str, value: Mapping[str, Any]) -> dict[str, Any]:
        normalized = self._normalized_view_id(view_id)
        _require_exact_fields(
            value,
            frozenset(
                {
                    "semantic_id",
                    "schema_version",
                    "view_id",
                    "document_sha256",
                    "referenced_entry_ids",
                    "document",
                }
            ),
            label=f"archive view {normalized!r}",
        )
        if value.get("semantic_id") != STATE_ARCHIVE_SEMANTIC_ID:
            raise ValueError(f"archive view {normalized!r} has an unsupported semantic_id")
        if int(value.get("schema_version", 0)) != 1:
            raise ValueError(f"archive view {normalized!r} has an unsupported schema_version")
        if value.get("view_id") != normalized:
            raise ValueError(f"archive view {normalized!r} has a mismatched view_id")
        document = value.get("document")
        if not isinstance(document, Mapping):
            raise ValueError(f"archive view {normalized!r} document must be an object")
        referenced = value.get("referenced_entry_ids")
        if (
            isinstance(referenced, str | bytes)
            or not isinstance(referenced, Sequence)
            or [str(entry_id) for entry_id in referenced]
            != sorted(set(str(entry_id) for entry_id in referenced))
        ):
            raise ValueError(
                f"archive view {normalized!r} referenced_entry_ids must be sorted and unique"
            )
        for entry_id in referenced:
            self.entry(str(entry_id))
        document_sha256 = canonical_json_sha256(document)
        if value.get("document_sha256") != document_sha256:
            raise ValueError(f"archive view {normalized!r} document hash mismatch")
        return dict(json_safe(value))

    def write_view(
        self,
        view_id: str,
        document: Mapping[str, Any],
        *,
        referenced_entry_ids: Sequence[str],
    ) -> dict[str, Any]:
        normalized = self._normalized_view_id(view_id)
        references = sorted(set(str(entry_id) for entry_id in referenced_entry_ids))
        for entry_id in references:
            self.entry(entry_id)
        safe_document = dict(json_safe(document))
        view = {
            "semantic_id": STATE_ARCHIVE_SEMANTIC_ID,
            "schema_version": 1,
            "view_id": normalized,
            "document_sha256": canonical_json_sha256(safe_document),
            "referenced_entry_ids": references,
            "document": safe_document,
        }
        validated = self._validate_view(normalized, view)
        atomic_write_json(self.views_root / f"{normalized}.json", validated)
        self._views[normalized] = validated
        return dict(validated)

    def retain_entries(self, entry_ids: Sequence[str]) -> dict[str, int]:
        retained = set(str(entry_id) for entry_id in entry_ids)
        for entry_id in retained:
            self.entry(entry_id)
        view_references = {
            str(entry_id)
            for view in self._views.values()
            for entry_id in view["referenced_entry_ids"]
        }
        missing_view_references = sorted(view_references - retained)
        if missing_view_references:
            raise ValueError(
                "cannot prune state archive entries referenced by a view: "
                f"{missing_view_references[:8]}"
            )
        removed_entries = set(self._entries) - retained
        retained_blobs = {
            self._entries[entry_id].provider_snapshot.ref.blob_sha256 for entry_id in retained
        }
        removed_blobs = {
            self._entries[entry_id].provider_snapshot.ref.blob_sha256
            for entry_id in removed_entries
        } - retained_blobs
        for entry_id in removed_entries:
            (self.entries_root / f"{entry_id}.json").unlink(missing_ok=True)
            self._untrack_entry_blob(self._entries[entry_id])
            del self._entries[entry_id]
        for blob_sha256 in removed_blobs:
            self.blobs.discard(blob_sha256)
        self.handles.retain(retained_blobs)
        (self.root / "closure.json").unlink(missing_ok=True)
        return {
            "removed_entries": len(removed_entries),
            "removed_blobs": len(removed_blobs),
            "retained_entries": len(retained),
            "retained_blobs": len(retained_blobs),
        }

    def view_document(self, view_id: str) -> Mapping[str, Any] | None:
        normalized = self._normalized_view_id(view_id)
        value = self._views.get(normalized)
        if value is None:
            return None
        return MappingProxyType(dict(value["document"]))

    def summary(self) -> dict[str, Any]:
        return {
            "semantic_id": STATE_ARCHIVE_SEMANTIC_ID,
            "schema_version": 1,
            "persistence": self.persistence,
            "provider_id": self.provider_id,
            "codec_id": self.codec_id,
            "compatibility_id": self.compatibility_id,
            "entry_count": len(self._entries),
            "blob_count": len(self._blob_ref_counts),
            "blob_bytes": self._blob_bytes,
            "view_ids": sorted(self._views),
        }

    def seal(
        self,
        *,
        step: int,
        status: str,
        referenced_entry_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        if int(step) < 0:
            raise ValueError("state archive closure step must be non-negative")
        if status not in {"recoverable", "closed"}:
            raise ValueError("state archive closure status must be recoverable or closed")
        retained = (
            set(self._entries)
            if referenced_entry_ids is None
            else set(str(entry_id) for entry_id in referenced_entry_ids)
        )
        for entry_id in retained:
            self.entry(entry_id)
        retained_blobs = {
            self.entry(entry_id).provider_snapshot.ref.blob_sha256 for entry_id in retained
        }
        paths = [
            *(self.entries_root / f"{entry_id}.json" for entry_id in sorted(retained)),
            *(self.blobs.path_for(blob_sha256) for blob_sha256 in sorted(retained_blobs)),
            *(candidate for candidate in self.views_root.glob("*.json")),
        ]
        files: list[dict[str, Any]] = []
        for path in sorted(paths):
            payload = path.read_bytes()
            files.append(
                {
                    "path": path.relative_to(self.root).as_posix(),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
            )
        closure = {
            "semantic_id": STATE_ARCHIVE_SEMANTIC_ID,
            "schema_version": 1,
            "status": status,
            "step": int(step),
            "archive": self.summary(),
            "files": files,
        }
        closure["inventory_sha256"] = canonical_json_sha256(files)
        atomic_write_json(self.root / "closure.json", closure)
        return closure

    def close(self) -> None:
        self.handles.clear()
        if self.persistence == "ephemeral":
            shutil.rmtree(self.root, ignore_errors=False)


_CURRICULUM_DEFAULTS: dict[str, Any] = {
    "restore_entries": False,
    "entries_per_cell": 4,
    "max_entries": 1024,
    "feedback_ema_alpha": 0.10,
    "staleness_weight": 0.30,
    "rank_temperature": 1.0,
    "max_cell_probability": 0.25,
}
_CURRICULUM_KEYS = frozenset(
    {
        "cell",
        "archive_share",
        "priority_metric",
        "semantic_id",
        "resolved_archive_lanes",
        *_CURRICULUM_DEFAULTS,
    }
)
_CELL_KEYS = frozenset({"dimensions"})
_CELL_DIMENSION_KEYS = frozenset({"signal", "source", "bucket_size", "clamp", "equals"})


def _finite_number(value: Any, *, label: str) -> float:
    if (
        not isinstance(value, int | float | np.number)
        or isinstance(value, bool | np.bool_)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def normalize_archive_cell_config(
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    """Normalize the shared YAML-defined archive cell detector."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    unexpected = sorted(set(value) - _CELL_KEYS)
    if unexpected:
        raise ValueError(f"{label} has unexpected fields: {unexpected}")
    dimensions = value.get("dimensions")
    if (
        isinstance(dimensions, str | bytes)
        or not isinstance(dimensions, Sequence)
        or not dimensions
    ):
        raise ValueError(f"{label}.dimensions must be a non-empty sequence")
    if len(dimensions) > 32:
        raise ValueError(f"{label}.dimensions must contain at most 32 entries")

    normalized_dimensions: list[dict[str, Any]] = []
    seen_selectors: set[tuple[str, str]] = set()
    for index, raw_dimension in enumerate(dimensions):
        dimension_label = f"{label}.dimensions[{index}]"
        if not isinstance(raw_dimension, Mapping):
            raise ValueError(f"{dimension_label} must be an object")
        unexpected_dimension = sorted(set(raw_dimension) - _CELL_DIMENSION_KEYS)
        if unexpected_dimension:
            raise ValueError(f"{dimension_label} has unexpected fields: {unexpected_dimension}")
        selector_fields = set(raw_dimension) & {"signal", "source"}
        if len(selector_fields) != 1:
            raise ValueError(f"{dimension_label} must define exactly one of signal or source")
        selector_kind = next(iter(selector_fields))
        selector_name = str(raw_dimension.get(selector_kind) or "").strip()
        if not selector_name:
            raise ValueError(f"{dimension_label}.{selector_kind} must be a non-empty string")
        selector = (selector_kind, selector_name)
        if selector in seen_selectors:
            raise ValueError(
                f"{label}.dimensions contains duplicate {selector_kind} {selector_name!r}"
            )
        seen_selectors.add(selector)

        has_equals = "equals" in raw_dimension
        has_bucket = "bucket_size" in raw_dimension
        has_clamp = "clamp" in raw_dimension
        if has_equals and (has_bucket or has_clamp):
            raise ValueError(
                f"{dimension_label}.equals cannot be combined with bucket_size or clamp"
            )
        normalized_dimension: dict[str, Any] = {selector_kind: selector_name}
        if has_equals:
            normalized_dimension["equals"] = _finite_number(
                raw_dimension["equals"],
                label=f"{dimension_label}.equals",
            )
        else:
            bucket_size = _finite_number(
                raw_dimension.get("bucket_size"),
                label=f"{dimension_label}.bucket_size",
            )
            if bucket_size <= 0.0:
                raise ValueError(f"{dimension_label}.bucket_size must be positive")
            normalized_dimension["bucket_size"] = bucket_size
            if has_clamp:
                clamp = raw_dimension["clamp"]
                if (
                    isinstance(clamp, str | bytes)
                    or not isinstance(clamp, Sequence)
                    or len(clamp) != 2
                ):
                    raise ValueError(f"{dimension_label}.clamp must be [minimum, maximum]")
                minimum = _finite_number(
                    clamp[0],
                    label=f"{dimension_label}.clamp[0]",
                )
                maximum = _finite_number(
                    clamp[1],
                    label=f"{dimension_label}.clamp[1]",
                )
                if minimum > maximum:
                    raise ValueError(f"{dimension_label}.clamp minimum must not exceed maximum")
                normalized_dimension["clamp"] = [minimum, maximum]
        normalized_dimensions.append(normalized_dimension)
    return {"dimensions": normalized_dimensions}


@dataclass(frozen=True)
class ArchiveCellDimension:
    signal: str | None = None
    source: str | None = None
    bucket_size: float | None = None
    clamp: tuple[float, float] | None = None
    equals: float | None = None

    @property
    def selector(self) -> tuple[str, str]:
        if self.signal is not None:
            return ("signal", self.signal)
        assert self.source is not None
        return ("source", self.source)

    def bucket(self, value: Any) -> int:
        kind, name = self.selector
        label = f"archive cell {kind} {name!r}"
        if isinstance(value, str | bytes):
            raise ValueError(f"{label} must be numeric")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be numeric") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"{label} must be finite")
        if self.equals is not None:
            return int(numeric == self.equals)
        if self.clamp is not None:
            numeric = min(max(numeric, self.clamp[0]), self.clamp[1])
        assert self.bucket_size is not None
        quotient = math.floor(numeric / self.bucket_size)
        if quotient < np.iinfo(np.int64).min or quotient > np.iinfo(np.int64).max:
            raise ValueError("state archive cell index exceeds signed int64")
        return int(quotient)


@dataclass(frozen=True)
class ArchiveCellConfig:
    dimensions: tuple[ArchiveCellDimension, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, label: str) -> "ArchiveCellConfig":
        normalized = normalize_archive_cell_config(value, label=label)
        return cls(
            dimensions=tuple(
                ArchiveCellDimension(
                    signal=(str(dimension["signal"]) if "signal" in dimension else None),
                    source=(str(dimension["source"]) if "source" in dimension else None),
                    bucket_size=(
                        float(dimension["bucket_size"]) if "bucket_size" in dimension else None
                    ),
                    clamp=(
                        (
                            float(dimension["clamp"][0]),
                            float(dimension["clamp"][1]),
                        )
                        if "clamp" in dimension
                        else None
                    ),
                    equals=(float(dimension["equals"]) if "equals" in dimension else None),
                )
                for dimension in normalized["dimensions"]
            )
        )

    @property
    def signals(self) -> tuple[str, ...]:
        return tuple(
            dimension.signal for dimension in self.dimensions if dimension.signal is not None
        )

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(
            dimension.source for dimension in self.dimensions if dimension.source is not None
        )


class ArchiveCellDetector:
    """Encode semantic signals or provider sources into deterministic cell keys."""

    def __init__(self, config: ArchiveCellConfig):
        self.config = config

    def keys(
        self,
        values_by_selector: Mapping[tuple[str, str], Any],
        *,
        n_envs: int,
    ) -> tuple[bytes, ...]:
        rows: list[list[int]] = [[] for _ in range(n_envs)]
        for dimension in self.config.dimensions:
            selector = dimension.selector
            kind, name = selector
            if selector not in values_by_selector:
                raise ValueError(f"archive cell {kind} {name!r} was not resolved")
            values = np.asarray(values_by_selector[selector])
            if values.shape != (n_envs,):
                raise ValueError(
                    f"archive cell {kind} {name!r} must have shape ({n_envs},), got {values.shape}"
                )
            for lane in range(n_envs):
                rows[lane].append(dimension.bucket(values[lane]))
        if len(self.config.dimensions) == 1:
            dimension = self.config.dimensions[0]
            if (
                dimension.signal is not None
                and dimension.clamp is None
                and dimension.equals is None
            ):
                return tuple(f"{dimension.signal}:{row[0]}".encode("ascii") for row in rows)
        return tuple(canonical_json_bytes(row) for row in rows)


def archive_lane_count(archive_share: float, n_envs: int) -> int:
    return int(math.floor(float(archive_share) * int(n_envs) + 0.5))


def normalize_archive_curriculum_config(
    value: Any,
    *,
    label: str = "state_archive.curriculum",
    n_envs: int | None = None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object or null")
    unexpected = sorted(set(value) - _CURRICULUM_KEYS)
    if unexpected:
        raise ValueError(f"{label} has unexpected fields: {unexpected}")
    semantic_id = value.get("semantic_id")
    if semantic_id is not None and semantic_id != STATE_ARCHIVE_CURRICULUM_SEMANTIC_ID:
        raise ValueError(f"{label}.semantic_id must be {STATE_ARCHIVE_CURRICULUM_SEMANTIC_ID!r}")
    cell = normalize_archive_cell_config(
        value.get("cell"),
        label=f"{label}.cell",
    )
    archive_share = value.get("archive_share")
    if (
        not isinstance(archive_share, int | float)
        or isinstance(archive_share, bool)
        or not math.isfinite(float(archive_share))
        or not 0.0 < float(archive_share) < 1.0
    ):
        raise ValueError(f"{label}.archive_share must be a finite number in (0, 1)")
    priority_metric = str(value.get("priority_metric") or "").strip()
    if not priority_metric:
        raise ValueError(f"{label}.priority_metric must be a non-empty string")
    normalized = {
        "semantic_id": STATE_ARCHIVE_CURRICULUM_SEMANTIC_ID,
        "cell": cell,
        "archive_share": float(archive_share),
        "priority_metric": priority_metric,
        **_CURRICULUM_DEFAULTS,
    }
    normalized.update({key: value[key] for key in _CURRICULUM_DEFAULTS if key in value})
    if not isinstance(normalized["restore_entries"], bool):
        raise ValueError(f"{label}.restore_entries must be a boolean")
    integer_fields = ("entries_per_cell", "max_entries")
    for key in integer_fields:
        item = normalized[key]
        if not isinstance(item, int) or isinstance(item, bool) or item < 1:
            raise ValueError(f"{label}.{key} must be a positive integer")
        normalized[key] = int(item)
    if normalized["max_entries"] < normalized["entries_per_cell"]:
        raise ValueError(f"{label}.max_entries must be >= entries_per_cell")
    if normalized["max_entries"] > 16384:
        raise ValueError(f"{label}.max_entries must be <= 16384")
    ranges = {
        "feedback_ema_alpha": (0.0, 1.0, False),
        "staleness_weight": (0.0, 1.0, True),
        "rank_temperature": (0.0, math.inf, False),
        "max_cell_probability": (0.0, 1.0, False),
    }
    for key, (minimum, maximum, include_minimum) in ranges.items():
        item = normalized[key]
        if not isinstance(item, int | float) or isinstance(item, bool):
            raise ValueError(f"{label}.{key} must be a number")
        numeric = float(item)
        lower_ok = numeric >= minimum if include_minimum else numeric > minimum
        if not math.isfinite(numeric) or not lower_ok or numeric > maximum:
            left = "[" if include_minimum else "("
            right = "]" if math.isfinite(maximum) else ")"
            upper = f"{maximum:g}" if math.isfinite(maximum) else "inf"
            raise ValueError(f"{label}.{key} must be in {left}{minimum:g}, {upper}{right}")
        normalized[key] = numeric
    if n_envs is not None:
        if not isinstance(n_envs, int) or isinstance(n_envs, bool) or n_envs < 2:
            raise ValueError(f"{label} requires n_envs >= 2")
        lanes = archive_lane_count(normalized["archive_share"], n_envs)
        if lanes < 1 or lanes >= n_envs:
            raise ValueError(
                f"{label}.archive_share resolves to {lanes} archive lanes for n_envs={n_envs}; "
                "it must resolve to at least one archive lane and one target lane"
            )
        normalized["resolved_archive_lanes"] = lanes
    return normalized


_STATE_ARCHIVE_KEYS = frozenset(
    {
        "semantic_id",
        "persistence",
        "restore_semantics",
        "recorder",
        "curriculum",
        "export",
    }
)
_RECORDER_KEYS = frozenset({"mode", "cell"})
_RECORDER_MODES = frozenset({"backend", "cell_transition"})
_EXPORT_KEYS = frozenset({"snapshots"})
_EXPORT_SNAPSHOT_MODES = frozenset({"none", "retained"})


def normalize_state_archive_config(
    value: Any,
    *,
    label: str = "state_archive",
    n_envs: int | None = None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object or null")
    unexpected = sorted(set(value) - _STATE_ARCHIVE_KEYS)
    if unexpected:
        raise ValueError(f"{label} has unexpected fields: {unexpected}")
    semantic_id = value.get("semantic_id", STATE_ARCHIVE_SEMANTIC_ID)
    if semantic_id != STATE_ARCHIVE_SEMANTIC_ID:
        raise ValueError(f"{label}.semantic_id must be {STATE_ARCHIVE_SEMANTIC_ID!r}")
    persistence = str(value.get("persistence", "durable"))
    if persistence not in STATE_ARCHIVE_PERSISTENCE:
        raise ValueError(f"{label}.persistence must be one of {sorted(STATE_ARCHIVE_PERSISTENCE)}")
    restore_semantics = str(value.get("restore_semantics", "continuation"))
    if restore_semantics not in RESTORE_SEMANTICS:
        raise ValueError(f"{label}.restore_semantics must be one of {sorted(RESTORE_SEMANTICS)}")
    recorder = value.get("recorder")
    if not isinstance(recorder, Mapping):
        raise ValueError(f"{label}.recorder must be an object")
    unexpected_recorder = sorted(set(recorder) - _RECORDER_KEYS)
    if unexpected_recorder:
        raise ValueError(f"{label}.recorder has unexpected fields: {unexpected_recorder}")
    mode = str(recorder.get("mode") or "").strip()
    if mode not in _RECORDER_MODES:
        raise ValueError(f"{label}.recorder.mode must be one of {sorted(_RECORDER_MODES)}")
    cell = recorder.get("cell")
    normalized_cell: dict[str, Any] | None = None
    if cell is not None:
        normalized_cell = normalize_archive_cell_config(
            cell,
            label=f"{label}.recorder.cell",
        )
    if mode == "cell_transition" and normalized_cell is None:
        raise ValueError(f"{label}.recorder.cell is required for cell_transition")

    curriculum = value.get("curriculum")
    normalized_curriculum: dict[str, Any] | None = None
    if curriculum is not None:
        if persistence != "durable":
            raise ValueError(f"{label}.curriculum requires persistence='durable'")
        if mode != "cell_transition" or normalized_cell is None:
            raise ValueError(f"{label}.curriculum requires recorder.mode='cell_transition'")
        if not isinstance(curriculum, Mapping):
            raise ValueError(f"{label}.curriculum must be an object or null")
        normalized_curriculum = normalize_archive_curriculum_config(
            {**dict(curriculum), "cell": normalized_cell},
            label=f"{label}.curriculum",
            n_envs=n_envs,
        )
    export = value.get("export", {})
    if not isinstance(export, Mapping):
        raise ValueError(f"{label}.export must be an object")
    unexpected_export = sorted(set(export) - _EXPORT_KEYS)
    if unexpected_export:
        raise ValueError(f"{label}.export has unexpected fields: {unexpected_export}")
    snapshot_mode = str(export.get("snapshots", "none")).strip()
    if snapshot_mode not in _EXPORT_SNAPSHOT_MODES:
        raise ValueError(
            f"{label}.export.snapshots must be one of {sorted(_EXPORT_SNAPSHOT_MODES)}"
        )
    return {
        "semantic_id": STATE_ARCHIVE_SEMANTIC_ID,
        "persistence": persistence,
        "restore_semantics": restore_semantics,
        "recorder": {
            "mode": mode,
            **({"cell": normalized_cell} if normalized_cell is not None else {}),
        },
        "curriculum": normalized_curriculum,
        "export": {"snapshots": snapshot_mode},
    }


def validate_state_archive_runtime_contract(
    common_config: Mapping[str, Any],
    *,
    backend_id: str,
    supported_priority_metrics: Sequence[str],
) -> None:
    value = common_config.get("state_archive")
    if value is None:
        return
    n_envs = int(common_config.get("n_envs", 0))
    normalized = normalize_state_archive_config(value, n_envs=n_envs)
    assert normalized is not None
    cell = normalized["recorder"].get("cell")
    if isinstance(cell, Mapping):
        task = common_config.get("task")
        signals = task.get("signals") if isinstance(task, Mapping) else None
        declared = set(signals) if isinstance(signals, Mapping) else set()
        required = {
            str(dimension["signal"])
            for dimension in cell.get("dimensions", ())
            if isinstance(dimension, Mapping) and "signal" in dimension
        }
        missing = sorted(required - declared)
        if missing:
            raise ValueError(
                "state_archive cell requires declared task signal(s): " + ", ".join(missing)
            )
    curriculum = normalized["curriculum"]
    if curriculum is None:
        return
    supported = frozenset(str(metric).strip() for metric in supported_priority_metrics)
    if curriculum["priority_metric"] not in supported:
        raise ValueError(
            f"training backend {backend_id!r} does not provide archive priority "
            f"{curriculum['priority_metric']!r}; supported priorities: {sorted(supported)}"
        )


@dataclass(frozen=True)
class ArchiveCurriculumConfig:
    archive_share: float
    priority_metric: str
    restore_entries: bool
    entries_per_cell: int
    max_entries: int
    feedback_ema_alpha: float
    staleness_weight: float
    rank_temperature: float
    max_cell_probability: float
    resolved_archive_lanes: int
    semantic_id: str = STATE_ARCHIVE_CURRICULUM_SEMANTIC_ID

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, n_envs: int) -> "ArchiveCurriculumConfig":
        archive = normalize_state_archive_config(value, n_envs=n_envs)
        assert archive is not None
        normalized = archive["curriculum"]
        if normalized is None:
            raise ValueError("state_archive.curriculum must be configured")
        return cls(
            archive_share=float(normalized["archive_share"]),
            priority_metric=str(normalized["priority_metric"]),
            restore_entries=bool(normalized["restore_entries"]),
            entries_per_cell=int(normalized["entries_per_cell"]),
            max_entries=int(normalized["max_entries"]),
            feedback_ema_alpha=float(normalized["feedback_ema_alpha"]),
            staleness_weight=float(normalized["staleness_weight"]),
            rank_temperature=float(normalized["rank_temperature"]),
            max_cell_probability=float(normalized["max_cell_probability"]),
            resolved_archive_lanes=int(normalized["resolved_archive_lanes"]),
        )


@dataclass(frozen=True)
class ArchiveSelection:
    cell_id: str
    entry_id: str
    generation: int


@dataclass
class _Cell:
    cell_id: str
    admission_index: int
    representatives: list[str] = field(default_factory=list)
    seen: int = 0
    feedback_score: float | None = None
    last_sample_rollout: int = 0
    cold_dispatched: bool = False
    active_count: int = 0
    pending_feedback: int = 0


def _counter_uint64(*parts: object) -> int:
    encoded = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(encoded, digest_size=8).digest(), "little")


class ArchiveCurriculum:
    """Bounded deterministic curriculum view over immutable state-archive entries."""

    def __init__(
        self,
        config: ArchiveCurriculumConfig,
        *,
        n_envs: int,
        run_seed: int,
        global_lane_ids: Sequence[int],
    ) -> None:
        self.config = config
        self.n_envs = int(n_envs)
        self.run_seed = int(run_seed)
        self.global_lane_ids = tuple(int(value) for value in global_lane_ids)
        self.generation = 1
        self.completed_rollout = 0
        self.activated = False
        self.activation_scheduled = False
        self._cells: dict[str, _Cell] = {}
        self._admission_counter = 0
        self._probabilities: dict[str, float] = {}
        self._sampled_this_rollout: set[str] = set()
        self._metrics: dict[str, float] = {}
        self.begin_rollout()

    @property
    def archive_lane_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_envs, dtype=np.bool_)
        if self.config.restore_entries:
            mask[: self.config.resolved_archive_lanes] = True
        return mask

    @property
    def cell_count(self) -> int:
        return len(self._cells)

    @property
    def entry_count(self) -> int:
        return sum(len(cell.representatives) for cell in self._cells.values())

    @property
    def ready(self) -> bool:
        return self.entry_count > 0

    def begin_rollout(self) -> None:
        self._metrics = {
            "admission_candidate_count": 0.0,
            "admission_accepted_count": 0.0,
            "evicted_count": 0.0,
            "capture_call_count": 0.0,
            "archive_reset_count": 0.0,
            "forced_boundary_count": 0.0,
            "feedback_trajectory_count": 0.0,
            "curriculum_transition_count": 0.0,
            "transition_count": 0.0,
            "capture_seconds": 0.0,
            "reset_seconds": 0.0,
        }
        self._rebuild_probabilities()

    def note_transition_batch(self, curriculum_count: int) -> None:
        self._metrics["transition_count"] += float(self.n_envs)
        self._metrics["curriculum_transition_count"] += float(curriculum_count)

    def note_candidates(self, count: int) -> None:
        self._metrics["admission_candidate_count"] += float(count)

    def note_capture(self, seconds: float) -> None:
        self._metrics["capture_call_count"] += 1.0
        self._metrics["capture_seconds"] += float(seconds)

    def note_reset(self, count: int, seconds: float, *, forced_boundaries: int = 0) -> None:
        self._metrics["archive_reset_count"] += float(count)
        self._metrics["forced_boundary_count"] += float(forced_boundaries)
        self._metrics["reset_seconds"] += float(seconds)

    def _representative_index(self, cell_id: str, seen: int) -> int:
        return (
            _counter_uint64(
                self.run_seed,
                self.generation,
                "reservoir",
                cell_id,
                seen,
            )
            % seen
        )

    def _evictable_cell(self, *, excluding: str | None = None) -> _Cell | None:
        candidates = [
            cell
            for cell in self._cells.values()
            if cell.cell_id != excluding
            and cell.feedback_score is not None
            and cell.active_count == 0
            and cell.pending_feedback == 0
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda cell: (self._probabilities.get(cell.cell_id, 0.0), cell.cell_id),
        )

    def _make_room(self, *, excluding: str | None = None) -> bool:
        if self.entry_count < self.config.max_entries:
            return True
        evicted = self._evictable_cell(excluding=excluding)
        if evicted is None:
            return False
        del self._cells[evicted.cell_id]
        self._probabilities.pop(evicted.cell_id, None)
        self._metrics["evicted_count"] += 1.0
        return self.entry_count < self.config.max_entries

    def admit(self, cell_id: str, entry_id: str) -> bool:
        if not entry_id:
            raise ValueError("archive admission requires an entry id")
        cell = self._cells.get(cell_id)
        if cell is None:
            if not self._make_room():
                return False
            self._admission_counter += 1
            cell = _Cell(cell_id=cell_id, admission_index=self._admission_counter)
            self._cells[cell_id] = cell
        cell.seen += 1
        if len(cell.representatives) < self.config.entries_per_cell:
            if not self._make_room(excluding=cell_id):
                return False
            cell.representatives.append(entry_id)
            self._metrics["admission_accepted_count"] += 1.0
            self._rebuild_probabilities()
            return True
        replacement = self._representative_index(cell_id, cell.seen)
        if replacement >= self.config.entries_per_cell:
            return False
        cell.representatives[int(replacement)] = entry_id
        self._metrics["admission_accepted_count"] += 1.0
        return True

    def schedule_activation(self) -> bool:
        if (
            not self.config.restore_entries
            or self.activated
            or self.activation_scheduled
            or not self.ready
        ):
            return False
        self.activation_scheduled = True
        return True

    def activate(self) -> None:
        if not self.ready:
            raise RuntimeError("state archive curriculum cannot activate with an empty archive")
        self.activated = True
        self.activation_scheduled = False

    def _draw_index(self, size: int, *, lane: int, episode_index: int, domain: str) -> int:
        if size <= 0:
            raise RuntimeError("cannot sample an empty archive set")
        value = _counter_uint64(
            self.run_seed,
            self.generation,
            self.completed_rollout,
            self.global_lane_ids[lane],
            episode_index,
            domain,
        )
        return int(value % size)

    def _draw_probability_cell(self, *, lane: int, episode_index: int) -> _Cell:
        ordered = sorted(self._probabilities)
        if not ordered:
            ordered = sorted(self._cells)
            return self._cells[
                ordered[
                    self._draw_index(
                        len(ordered), lane=lane, episode_index=episode_index, domain="unscored"
                    )
                ]
            ]
        raw = _counter_uint64(
            self.run_seed,
            self.generation,
            self.completed_rollout,
            self.global_lane_ids[lane],
            episode_index,
            "cell",
        )
        point = (raw + 0.5) / float(2**64)
        cumulative = 0.0
        for cell_id in ordered:
            cumulative += self._probabilities[cell_id]
            if point <= cumulative:
                return self._cells[cell_id]
        return self._cells[ordered[-1]]

    def sample(self, *, lane: int, episode_index: int) -> ArchiveSelection:
        cold = sorted(
            (cell for cell in self._cells.values() if not cell.cold_dispatched),
            key=lambda cell: (cell.admission_index, cell.cell_id),
        )
        if cold:
            cell = cold[0]
            cell.cold_dispatched = True
        else:
            cell = self._draw_probability_cell(lane=lane, episode_index=episode_index)
        representative_index = self._draw_index(
            len(cell.representatives),
            lane=lane,
            episode_index=episode_index,
            domain=f"representative:{cell.cell_id}",
        )
        self._sampled_this_rollout.add(cell.cell_id)
        cell.active_count += 1
        return ArchiveSelection(
            cell_id=cell.cell_id,
            entry_id=cell.representatives[representative_index],
            generation=self.generation,
        )

    def close_episode(self, cell_id: str) -> None:
        cell = self._cells.get(cell_id)
        if cell is None:
            return
        if cell.active_count <= 0:
            raise RuntimeError(f"archive cell {cell_id!r} has no active episode to close")
        cell.active_count -= 1
        cell.pending_feedback += 1

    def submit_feedback(self, cell_id: str, value_error: float) -> None:
        cell = self._cells.get(cell_id)
        if cell is None:
            return
        if cell.pending_feedback <= 0:
            raise RuntimeError(f"archive cell {cell_id!r} has no pending feedback")
        value = float(value_error)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("archive value_error feedback must be finite and non-negative")
        cell.pending_feedback -= 1
        alpha = self.config.feedback_ema_alpha
        cell.feedback_score = (
            value
            if cell.feedback_score is None
            else (1.0 - alpha) * cell.feedback_score + alpha * value
        )
        self._metrics["feedback_trajectory_count"] += 1.0
        self._rebuild_probabilities()

    @staticmethod
    def _rank_weights(
        values: Mapping[str, float], *, largest_first: bool, temperature: float
    ) -> dict[str, float]:
        ordered = sorted(
            values,
            key=lambda cell_id: (
                -values[cell_id] if largest_first else values[cell_id],
                cell_id,
            ),
        )
        weights = {
            cell_id: float((rank + 1) ** (-1.0 / temperature))
            for rank, cell_id in enumerate(ordered)
        }
        total = sum(weights.values())
        return {cell_id: weight / total for cell_id, weight in weights.items()}

    @staticmethod
    def _cap_probabilities(probabilities: Mapping[str, float], cap: float) -> dict[str, float]:
        if not probabilities:
            return {}
        result = {key: float(value) for key, value in probabilities.items()}
        active = set(result)
        fixed_mass = 0.0
        original = dict(result)
        while active:
            active_weight = sum(original[key] for key in active)
            if active_weight <= 0.0:
                share = (1.0 - fixed_mass) / len(active)
                for key in active:
                    result[key] = share
                break
            changed = False
            for key in sorted(tuple(active)):
                projected = (1.0 - fixed_mass) * original[key] / active_weight
                if projected > cap:
                    result[key] = cap
                    fixed_mass += cap
                    active.remove(key)
                    changed = True
            if not changed:
                active_weight = sum(original[key] for key in active)
                for key in active:
                    result[key] = (1.0 - fixed_mass) * original[key] / active_weight
                break
        residual = 1.0 - sum(result.values())
        if abs(residual) > 1e-15:
            for key in sorted(result):
                candidate = result[key] + residual
                if 0.0 <= candidate <= cap + 1e-12:
                    result[key] = candidate
                    break
        return result

    def _rebuild_probabilities(self) -> None:
        scored = {
            cell.cell_id: float(cell.feedback_score)
            for cell in self._cells.values()
            if cell.feedback_score is not None and cell.representatives
        }
        if not scored:
            self._probabilities = {}
            return
        score_weights = self._rank_weights(
            scored,
            largest_first=True,
            temperature=self.config.rank_temperature,
        )
        ages = {
            cell_id: float(self.completed_rollout - self._cells[cell_id].last_sample_rollout)
            for cell_id in scored
        }
        stale_weights = self._rank_weights(
            ages,
            largest_first=True,
            temperature=self.config.rank_temperature,
        )
        rho = self.config.staleness_weight
        mixed = {
            cell_id: (1.0 - rho) * score_weights[cell_id] + rho * stale_weights[cell_id]
            for cell_id in scored
        }
        effective_cap = max(self.config.max_cell_probability, 1.0 / len(mixed))
        self._probabilities = self._cap_probabilities(mixed, effective_cap)

    def complete_rollout(self) -> dict[str, float]:
        for cell_id in self._sampled_this_rollout:
            cell = self._cells.get(cell_id)
            if cell is not None:
                cell.last_sample_rollout = self.completed_rollout
        self._sampled_this_rollout.clear()
        transition_count = self._metrics["transition_count"]
        probabilities = tuple(self._probabilities.values())
        payload = {
            **self._metrics,
            "archive_cell_count": float(self.cell_count),
            "archive_entry_count": float(self.entry_count),
            "transition_share": (
                self._metrics["curriculum_transition_count"] / transition_count
                if transition_count > 0.0
                else 0.0
            ),
            "sampling_probability_max": max(probabilities, default=0.0),
            "sampling_effective_cell_count": (
                1.0 / sum(value * value for value in probabilities) if probabilities else 0.0
            ),
        }
        self.completed_rollout += 1
        self._rebuild_probabilities()
        return payload

    def artifact_summary(self) -> dict[str, Any]:
        return {
            "semantic_id": self.config.semantic_id,
            "generation": self.generation,
            "persistence": "durable",
            "resume_behavior": "restore_archive",
            "archive_cell_count": self.cell_count,
            "archive_entry_count": self.entry_count,
            "completed_rollout": self.completed_rollout,
        }

    def close(self) -> None:
        self._cells.clear()
        self._probabilities.clear()
        self._sampled_this_rollout.clear()


def state_archive_artifact_summary(source: Any) -> Mapping[str, Any] | None:
    """Find the neutral runtime summary through common environment wrappers."""

    seen: set[int] = set()
    current = source
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        summary = getattr(current, "state_archive_summary", None)
        if callable(summary):
            value = summary()
            if value is None or value.get("persistence") != "durable":
                return None
            return dict(value)
        current = getattr(current, "venv", None) or getattr(current, "env", None)
    return None
