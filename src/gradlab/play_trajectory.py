"""Bounded episode capture and the versioned, data-only trajectory dataset format.

Runtime storage is an append-only byte stream with a fixed-width seek index.
Exports pin those inodes, read only their fixed prefix, and write typed Parquet.
No import path calls a model loader, environment factory, or remote resolver.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from functools import lru_cache
import json
import math
import re
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import struct
import tempfile
import threading
from typing import Any
import uuid
import zipfile

import numpy as np

from gradlab.file_utils import file_sha256
from gradlab.policy_bundle import load_policy_bundle, write_canonical_json
from gradlab.validation import is_secret_like_key
from gradlab.play_timeline import EventOverview

FORMAT_VERSION = 1
MAX_RECORD_BYTES = 32 * 1024**2
BUFFER_BYTES = 64 * 1024**2
MAX_ARCHIVE_BYTES = 32 * 1024**3
MAX_RECORDING_BYTES = 32 * 1024**3
INDEX = struct.Struct("<QQ")
HEADER = struct.Struct("<Q")
TREE_COLUMNS = (
    "observation",
    "next_observation",
    "selected_action",
    "executed_action",
    "before_image",
    "after_image",
    "observation_frames",
    "policy_outputs",
    "facts",
)


@lru_cache(maxsize=4096)
def _portable_metadata_key(key: str) -> bool:
    # Cache field-name classification only, never values or mutable snapshots.
    return not is_secret_like_key(key) and key not in {
        "hostname",
        "host",
        "ssh",
        "operator",
        "endpoint",
        "object_uri",
    }


def portable_metadata(value: Any) -> Any:
    """Remove private configuration fields without changing scientific numeric data."""
    kind = type(value)
    if value is None or kind in (bool, int, float):
        return value
    if isinstance(value, str):
        if value.startswith(("/", "~", "s3://", "r2://", "file://")) or (
            "://" in value and any(part in value for part in ("?", "@"))
        ):
            return "[private location omitted]"
        return value
    if isinstance(value, Mapping):
        return {
            str(key): portable_metadata(item)
            for key, item in value.items()
            if _portable_metadata_key(str(key))
        }
    if isinstance(value, tuple | list):
        return [portable_metadata(item) for item in value]
    if isinstance(value, Path):
        return "[local path omitted]"
    if is_dataclass(value) and not isinstance(value, type):
        return portable_metadata(asdict(value))
    return value


def encode_tree(value: Any) -> dict[str, Any]:
    """Encode a structured value as JSON topology plus exact numeric array leaves."""
    arrays: list[dict[str, Any]] = []

    def node(item: Any, depth: int = 0) -> Any:
        if depth > 32:
            raise ValueError("trajectory structure exceeds 32 levels")
        # Most snapshot leaves are native JSON scalars. Avoid the NumPy and ABC
        # dispatch for each leaf, but keep NumPy scalars on the exact-array path.
        kind = type(item)
        if item is None or kind in (str, bool, int):
            return ["scalar", item]
        if kind is float and math.isfinite(item):
            return ["scalar", item]
        if isinstance(item, np.ndarray | np.generic):
            array = np.asarray(item)
            if array.dtype.kind not in "buifc" or array.ndim > 16:
                raise ValueError("trajectory arrays must have a numeric, non-object dtype")
            index = len(arrays)
            arrays.append(
                {
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                    "data": array.tobytes(order="C"),
                }
            )
            return ["array", index]
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("trajectory structures require string mapping keys")
            return ["dict", [[key, node(nested, depth + 1)] for key, nested in item.items()]]
        if isinstance(item, tuple | list):
            return [
                "tuple" if isinstance(item, tuple) else "list",
                [node(nested, depth + 1) for nested in item],
            ]
        if item is None or isinstance(item, str | bool | int | float):
            if isinstance(item, float) and not math.isfinite(item):
                return node(np.asarray(item), depth + 1)
            return ["scalar", item]
        raise ValueError(f"unsupported trajectory value type: {type(item).__name__}")

    structure = json.dumps(node(value), separators=(",", ":"), allow_nan=False)
    return {"structure": structure, "arrays": arrays}


def decode_tree(tree: Mapping[str, Any]) -> Any:
    arrays = []
    total = 0
    for leaf in tree["arrays"]:
        dtype = np.dtype(leaf["dtype"])
        shape = leaf["shape"]
        if (
            dtype.kind not in "buifc"
            or len(shape) > 16
            or any(type(size) is not int or size < 0 for size in shape)
        ):
            raise ValueError("invalid trajectory array dtype or shape")
        size = math.prod(shape) * dtype.itemsize
        total += size
        if size != len(leaf["data"]) or total > MAX_RECORD_BYTES:
            raise ValueError("invalid or excessive trajectory array bytes")
        arrays.append(np.frombuffer(leaf["data"], dtype=dtype).reshape(shape).copy())

    def node(value: Any, depth: int = 0) -> Any:
        if depth > 32 or not isinstance(value, list) or len(value) != 2:
            raise ValueError("invalid trajectory structure")
        kind, data = value
        if kind == "array" and type(data) is int and 0 <= data < len(arrays):
            return arrays[data]
        if kind in {"tuple", "list"} and isinstance(data, list):
            values = [node(item, depth + 1) for item in data]
            return tuple(values) if kind == "tuple" else values
        if kind == "dict" and isinstance(data, list):
            result = {}
            for key, item in data:
                if not isinstance(key, str) or key in result:
                    raise ValueError("invalid trajectory mapping key")
                result[key] = node(item, depth + 1)
            return result
        if kind == "scalar" and (data is None or isinstance(data, str | bool | int | float)):
            return data
        raise ValueError("invalid trajectory node")

    return node(json.loads(tree["structure"]))


def pack_record(value: Any) -> bytes:
    tree = encode_tree(value)
    buffers = []
    for leaf in tree["arrays"]:
        data = leaf.pop("data")
        leaf["size"] = len(data)
        buffers.append(data)
    header = json.dumps(tree, separators=(",", ":")).encode()
    if len(header) + sum(map(len, buffers)) > MAX_RECORD_BYTES:
        raise ValueError("transition exceeds the 32 MiB recording limit")
    return HEADER.pack(len(header)) + header + b"".join(buffers)


def unpack_record(data: bytes) -> Any:
    (size,) = HEADER.unpack_from(data)
    if size > MAX_RECORD_BYTES:
        raise ValueError("invalid trajectory record header")
    tree = json.loads(data[HEADER.size : HEADER.size + size])
    offset = HEADER.size + size
    for leaf in tree["arrays"]:
        length = leaf.pop("size")
        leaf["data"] = data[offset : offset + length]
        offset += length
    if offset != len(data):
        raise ValueError("invalid trajectory record length")
    return decode_tree(tree)


class EpisodeRecording:
    """One episode, one bounded queue, one background disk writer."""

    def __init__(
        self,
        metadata: Mapping[str, Any],
        *,
        root: Path | None = None,
        buffer_bytes: int = BUFFER_BYTES,
        max_bytes: int = MAX_RECORDING_BYTES,
        write_record=None,
    ):
        self.root = Path(tempfile.mkdtemp(prefix="gradlab-episode-", dir=root))
        self.metadata = deepcopy(dict(metadata))
        self.metadata.update(
            episode_id=uuid.uuid4().hex,
            format_version=FORMAT_VERSION,
            scientific_evidence=False,
            complete=False,
        )
        self.buffer_bytes = buffer_bytes
        self._event_overview = EventOverview(self.metadata["episode_id"])
        self.max_bytes = max_bytes
        self._accepted_bytes = 0
        self._write_record = write_record or self._append
        self._condition = threading.Condition()
        self._pending: deque[bytes] = deque()
        self._pending_bytes = 0
        self._accepted = 0
        self._written = 0
        self._error: str | None = None
        self._capture_error: str | None = None
        self._rejected_row: Any = None
        self._closed = False
        self._pins = 0
        self._retired = False
        (self.root / "records.bin").touch()
        (self.root / "records.idx").touch()
        self._thread = threading.Thread(target=self._run, name="gradlab-episode-writer")
        self._thread.start()

    def status(self) -> dict[str, Any]:
        with self._condition:
            return {
                "transitions": self._accepted,
                "written": self._written,
                "backlog_bytes": self._pending_bytes,
                "buffer_bytes": self.buffer_bytes,
                "storage_bytes": self._accepted_bytes,
                "storage_limit_bytes": self.max_bytes,
                "error": self._capture_error or self._error,
                "complete": self.metadata["complete"],
                "first_step": self.metadata["first_step"],
                "last_step": self.metadata["first_step"] + self._accepted - 1,
                "episode_id": self.metadata["episode_id"],
            }

    def event_overview(self) -> dict[str, Any]:
        with self._condition:
            return self._event_overview.payload()

    def check_capacity(self) -> None:
        with self._condition:
            # Reserve room for the largest possible next decision before advancing
            # the environment. The captured prefix is never evicted to make space.
            if self._accepted_bytes + MAX_RECORD_BYTES + HEADER.size + INDEX.size > self.max_bytes:
                raise OSError(
                    "Episode storage limit reached. Playback paused; captured steps retained. "
                    "Download the episode before starting another episode."
                )
            if self._capture_error:
                raise ValueError(self._capture_error)
            if self._error:
                raise OSError(
                    f"Recording storage failed: {self._error}. Retry recording to resume."
                )
            if self._pending_bytes >= self.buffer_bytes:
                raise OSError(
                    "Recording storage cannot keep up. Playback paused; captured data retained."
                )

    def append(self, row: Mapping[str, Any]) -> None:
        # This owns all arrays before another step can reuse a provider buffer.
        try:
            data = pack_record(row)
        except ValueError as exc:
            # Never advance over an unencodable decision. Keep its owned inputs until
            # the user explicitly replaces the episode; the accepted prefix is exportable.
            with self._condition:
                self._rejected_row = row
                self._capture_error = f"Recording stopped at transition {row['step']}: {exc}"
            raise
        with self._condition:
            if self._closed:
                raise OSError("recording is closed")
            self._pending.append(data)
            self._pending_bytes += len(data)
            self._accepted += 1
            self._accepted_bytes += len(data) + INDEX.size
            self._event_overview.append(
                {
                    "step": row["step"],
                    "boundary": row["boundary"],
                    "events": (row.get("presentation") or {}).get("events", []),
                }
            )
            self.metadata["complete"] = bool(row["boundary"])
            if self.metadata["classification"] != "counterfactual":
                self.metadata["classification"] = row["classification"]
            self._condition.notify_all()

    def transition(self, step: int) -> dict[str, Any]:
        """Read a captured step without flushing or scanning the episode.

        Pending records are readable even when the disk writer has failed. Holding
        the condition pins the files against close and writer retry/truncation.
        """
        with self._condition:
            if type(step) is not int:
                raise ValueError("step must be an integer")
            index = step - int(self.metadata["first_step"])
            if index < 0 or index >= self._accepted or self._retired:
                raise ValueError("step is outside the recorded episode")
            if index >= self._written:
                return unpack_record(self._pending[index - self._written])
            return read_record(self.root, index)

    @staticmethod
    def _append(root: Path, data: bytes) -> None:
        with (root / "records.bin").open("ab") as stream:
            offset = stream.tell()
            stream.write(data)
            stream.flush()
        with (root / "records.idx").open("ab") as index:
            index.write(INDEX.pack(offset, len(data)))
            index.flush()

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._pending and not self._error)
                if self._closed and (not self._pending or self._error):
                    return
                data = self._pending[0]
            try:
                self._write_record(self.root, data)
            except Exception as exc:
                with self._condition:
                    self._error = str(exc)
                    self._condition.notify_all()
                continue
            with self._condition:
                self._pending.popleft()
                self._pending_bytes -= len(data)
                self._written += 1
                self._condition.notify_all()

    def retry(self) -> None:
        with self._condition:
            if self._error is None:
                return
            # Roll back a partial failed append before retrying the same transition.
            with (self.root / "records.idx").open("r+b") as index:
                index.truncate(self._written * INDEX.size)
                end = 0
                if self._written:
                    index.seek((self._written - 1) * INDEX.size)
                    offset, size = INDEX.unpack(index.read(INDEX.size))
                    end = offset + size
            with (self.root / "records.bin").open("r+b") as stream:
                stream.truncate(end)
            self._error = None
            self._condition.notify_all()

    def reserve_read(self):
        """Pin immutable written records and own references to the bounded pending tail."""
        with self._condition:
            if self._retired:
                raise ValueError("the recorded episode has been replaced")
            self._pins += 1
            metadata = deepcopy(self.metadata)
            metadata["transition_count"] = self._accepted
            return dict(root=str(self.root), metadata=metadata, written=self._written,
                        pending=tuple(self._pending))

    def release_read(self):
        with self._condition:
            self._pins -= 1
            if self._retired and not self._pins:
                shutil.rmtree(self.root, ignore_errors=True)

    def reserve_prefix(self):
        with self._condition:
            cutoff = self._accepted
            metadata = deepcopy(self.metadata)
            self._pins += 1
            return cutoff, metadata

    @contextmanager
    def prefix(self, reservation):
        cutoff, metadata = reservation
        try:
            with self._condition:
                if not self._condition.wait_for(
                    lambda: self._written >= cutoff or self._error is not None, timeout=25
                ):
                    raise OSError("Recording is still flushing; retry the download shortly")
                if self._written < cutoff:
                    # Preserve the entire accepted prefix, including the failed write.
                    pending = tuple(self._pending)[: cutoff - self._written]
                else:
                    pending = ()
                written = cutoff - len(pending)
            metadata["transition_count"] = cutoff
            yield metadata, written, pending
        finally:
            with self._condition:
                self._pins -= 1
                if self._retired and not self._pins:
                    shutil.rmtree(self.root, ignore_errors=True)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join()
        with self._condition:
            self._retired = True
            if not self._pins:
                shutil.rmtree(self.root, ignore_errors=True)


def read_record(root: Path, index: int) -> dict[str, Any]:
    with (root / "records.idx").open("rb") as stream:
        stream.seek(index * INDEX.size)
        offset, size = INDEX.unpack(stream.read(INDEX.size))
    if size > MAX_RECORD_BYTES + HEADER.size:
        raise ValueError("excessive trajectory record size")
    with (root / "records.bin").open("rb") as stream:
        stream.seek(offset)
        return unpack_record(stream.read(size))


def freeze_recording(recording: EpisodeRecording, destination: Path, reservation) -> str:
    """Pin an immutable prefix into a temporary directory owned by the downloader."""
    try:
        with recording.prefix(reservation) as (metadata, written, pending):
            if not metadata["transition_count"]:
                raise ValueError("record at least one transition before downloading")
            # Copy the small fixed-width index. The data stream only ever appends.
            os.link(recording.root / "records.bin", destination / "records.bin")
            with (
                (recording.root / "records.idx").open("rb") as source,
                (destination / "records.idx").open("wb") as target,
            ):
                remaining = written * INDEX.size
                while remaining:
                    chunk = source.read(min(remaining, 1024**2))
                    if not chunk:
                        raise OSError("recording index was truncated while downloading")
                    target.write(chunk)
                    remaining -= len(chunk)
            if pending:
                # A failed sink leaves at most the bounded queue in RAM. Export it too.
                shutil.copyfile(destination / "records.bin", destination / "pending.bin")
                (destination / "records.bin").unlink()
                (destination / "pending.bin").rename(destination / "records.bin")
                for data in pending:
                    EpisodeRecording._append(destination, data)
            write_canonical_json(destination / "metadata.json", metadata)
        return str(destination)
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def parquet_schema():
    import pyarrow as pa

    tensor = pa.struct(
        [
            ("dtype", pa.string()),
            ("shape", pa.list_(pa.field("element", pa.int64()))),
            ("data", pa.binary()),
        ]
    )
    tree = pa.struct(
        [("structure", pa.string()), ("arrays", pa.list_(pa.field("element", tensor)))]
    )
    return pa.schema(
        [
            ("episode_id", pa.string()),
            ("sequence", pa.int64()),
            ("step", pa.int64()),
            ("seed", pa.int64()),
            ("start_id", pa.string()),
            ("action_source", pa.string()),
            ("reward", pa.float64()),
            ("return", pa.float64()),
            ("terminated", pa.bool_()),
            ("truncated", pa.bool_()),
            ("boundary", pa.bool_()),
            ("classification", pa.string()),
            ("next_observation_status", pa.string()),
            ("after_image_status", pa.string()),
            *[(name, tree) for name in TREE_COLUMNS],
            ("presentation", pa.string()),
        ],
        metadata={b"gradlab.trajectory.version": b"1"},
    )


def export_trajectory(frozen: str | Path, destination: Path) -> Path:
    """Finalize in bounded batches while the live runner continues on another thread."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(frozen)
    try:
        metadata = json.loads((root / "metadata.json").read_bytes())
        data_dir = root / "data"
        data_dir.mkdir()
        path = data_dir / "train-00000-of-00001.parquet"
        schema = parquet_schema()
        with pq.ParquetWriter(path, schema, compression="zstd", write_page_index=True) as writer:
            for index in range(metadata["transition_count"]):
                row = read_record(root, index)
                row["episode_id"] = metadata["episode_id"]
                for name in TREE_COLUMNS:
                    row[name] = encode_tree(row[name])
                row["presentation"] = json.dumps(row["presentation"], allow_nan=False)
                # One transition per row group bounds reads and enables exact seeking.
                writer.write_table(pa.Table.from_pylist([row], schema=schema))
        paths = [root / "metadata.json", path, *sorted((root / "checkpoint").iterdir())]
        manifest = {
            str(p.relative_to(root)): {"size": p.stat().st_size, "sha256": file_sha256(p)}
            for p in paths
        }
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr(
                zipfile.ZipInfo("manifest.json"),
                json.dumps({"format_version": FORMAT_VERSION, "files": manifest}),
            )
            for path in paths:
                info = zipfile.ZipInfo(str(path.relative_to(root)))
                with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
                    shutil.copyfileobj(source, target, length=1024**2)
        return destination
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(root, ignore_errors=True)


class ImportedTrajectory:
    """Validate and index one archive without constructing its Policy or provider."""

    def __init__(self, archive_path: Path):
        import pyarrow.parquet as pq

        self._read_lock = threading.Lock()
        self._read_pins = 0
        self._retired = False
        self.root = Path(tempfile.mkdtemp(prefix="gradlab-imported-episode-"))
        try:
            self._extract(archive_path)
            self.metadata = json.loads((self.root / "metadata.json").read_bytes())
            if (
                self.metadata.get("format_version") != FORMAT_VERSION
                or self.metadata.get("scientific_evidence") is not False
            ):
                raise ValueError("unsupported trajectory metadata or scientific classification")
            self.bundle = load_policy_bundle(self.root / "checkpoint")
            self._validate_metadata()
            parquet_path = self.root / "data/train-00000-of-00001.parquet"
            with parquet_path.open("rb") as stream:
                stream.seek(-8, 2)
                footer = stream.read(8)
                if footer[4:] != b"PAR1" or int.from_bytes(footer[:4], "little") > 128 * 1024**2:
                    raise ValueError("invalid or excessive trajectory Parquet footer")
            parquet = pq.ParquetFile(
                parquet_path,
                thrift_string_size_limit=16 * 1024**2,
                thrift_container_size_limit=1_000_000,
            )
            if not parquet.schema_arrow.equals(parquet_schema(), check_metadata=True):
                raise ValueError("incompatible trajectory Parquet features")
            count = self.metadata.get("transition_count")
            if (
                type(count) is not int
                or not 1 <= count <= 10_000_000
                or parquet.metadata.num_rows != count
            ):
                raise ValueError("trajectory transition count mismatch")
            expanded_size = 0
            for index in range(parquet.num_row_groups):
                group = parquet.metadata.row_group(index)
                expanded_size += group.total_byte_size
                if group.num_rows != 1 or group.total_byte_size > 2 * MAX_RECORD_BYTES:
                    raise ValueError("excessive or incompatible trajectory row group")
            if expanded_size > MAX_ARCHIVE_BYTES:
                raise ValueError("excessive expanded trajectory data")
            from gradlab.play_reward_summary import EpisodeRewardSummary

            rewards = EpisodeRewardSummary()
            reward_contract = self.metadata["initial_snapshot"]["session"].get(
                "reward_accounting", {}
            )
            previous = None
            classifications = set()
            for batch in parquet.iter_batches(batch_size=1):
                row = batch.to_pylist()[0]
                for name in TREE_COLUMNS:
                    row[name] = decode_tree(row[name])
                row["presentation"] = json.loads(row["presentation"])
                self._validate_row(row, previous)
                # Rebuild derived totals from validated transition evidence during
                # the existing streaming import, including archives without totals.
                rewards.append(row["presentation"], reward_contract)
                row["presentation"]["episode_rewards"] = rewards.payload(row["presentation"])
                classifications.add(row["classification"])
                EpisodeRecording._append(self.root, pack_record(row))
                previous = row
            if previous is None or bool(previous["boundary"]) != self.metadata["complete"]:
                raise ValueError("trajectory completion disagrees with its last transition")
            if (
                "counterfactual" in classifications
                and self.metadata["classification"] != "counterfactual"
            ):
                raise ValueError("trajectory classification hides a Counterfactual transition")
        except Exception as exc:
            self.close()
            raise ValueError(f"Invalid trajectory archive: {exc}") from exc
        except BaseException:
            self.close()
            raise

    def _validate_metadata(self) -> None:
        metadata = self.metadata
        if (
            not re.fullmatch(r"[0-9a-f]{32}", str(metadata.get("episode_id", "")))
            or type(metadata.get("complete")) is not bool
            or any(
                type(metadata.get(key)) is not int or metadata[key] < 1
                for key in ("first_step", "episode")
            )
            or metadata.get("classification")
            not in {"faithful", "evaluation_reproduction", "counterfactual"}
        ):
            raise ValueError("invalid trajectory episode metadata")
        snapshot = metadata["initial_snapshot"]
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get("session"), dict):
            raise ValueError("invalid trajectory initial presentation")
        if not isinstance(snapshot["session"].get("env_id"), str):
            raise ValueError("invalid trajectory environment identity")

    def _validate_row(self, row: Mapping[str, Any], previous: Mapping[str, Any] | None) -> None:
        if row["episode_id"] != self.metadata["episode_id"]:
            raise ValueError("trajectory contains another episode")
        expected_step = previous["step"] + 1 if previous else self.metadata["first_step"]
        if (
            row["step"] != expected_step
            or previous
            and (row["sequence"] != previous["sequence"] + 1 or previous["boundary"])
        ):
            raise ValueError("trajectory ordering or episode boundary is invalid")
        if row["classification"] not in {"faithful", "evaluation_reproduction", "counterfactual"}:
            raise ValueError("invalid recorded Playback classification")
        if any(type(row[name]) is not bool for name in ("terminated", "truncated", "boundary")):
            raise ValueError("invalid trajectory boundary flags")
        if (row["terminated"] or row["truncated"]) and not row["boundary"]:
            raise ValueError("terminal transition has no episode boundary")
        for field, status_field in (
            ("next_observation", "next_observation_status"),
            ("after_image", "after_image_status"),
        ):
            status = row[status_field]
            if (
                status not in {"available", "terminal_missing", "unavailable"}
                or (status == "available") != (row[field] is not None)
                or status == "terminal_missing"
                and not row["boundary"]
            ):
                raise ValueError(f"invalid trajectory {status_field}")
        images = [row["before_image"], row["after_image"], *row["observation_frames"]]
        for frame in images:
            if frame is not None and (
                not isinstance(frame, np.ndarray)
                or frame.dtype != np.uint8
                or frame.ndim not in (2, 3)
                or not frame.size
                or frame.ndim == 3
                and frame.shape[-1] not in (1, 3, 4)
            ):
                raise ValueError("invalid lossless trajectory image")
        presentation = row["presentation"]
        for field in (
            "sequence",
            "step",
            "seed",
            "start_id",
            "action_source",
            "terminated",
            "truncated",
            "boundary",
        ):
            if presentation.get(field) != row[field]:
                raise ValueError(f"trajectory presentation disagrees on {field}")
        if presentation["reward"]["step"] != row["reward"]:
            raise ValueError("trajectory presentation disagrees on reward")
        if presentation["reward"]["return"] != row["return"]:
            raise ValueError("trajectory presentation disagrees on return")
        for key in ("attribution", "cnn"):
            if presentation.get(key, {}).get("status") != "not-recorded":
                raise ValueError("trajectory contains unsupported heavy diagnostics")
        if not all(math.isfinite(row[name]) for name in ("reward", "return")):
            raise ValueError("non-finite trajectory reward")

    def _extract(self, path: Path) -> None:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) > 68 or sum(item.file_size for item in members) > MAX_ARCHIVE_BYTES:
                raise ValueError("excessive trajectory archive size")
            names = set()
            for item in members:
                name = item.filename
                mode = item.external_attr >> 16
                if (
                    name in names
                    or "\\" in name
                    or PurePosixPath(name).is_absolute()
                    or ".." in PurePosixPath(name).parts
                    or item.is_dir()
                    or stat.S_ISLNK(mode)
                    or item.flag_bits & 1
                    or name
                    not in {"manifest.json", "metadata.json", "data/train-00000-of-00001.parquet"}
                    and not (
                        name.startswith("checkpoint/")
                        and len(PurePosixPath(name).parts) == 2
                        and name.rsplit("/", 1)[1] in {"model.zip", "model.json", "recipe.json"}
                    )
                ):
                    raise ValueError(f"unsafe or unexpected trajectory member: {name}")
                names.add(name)
            if archive.getinfo("manifest.json").file_size > 64 * 1024:
                raise ValueError("excessive trajectory manifest")
            manifest = json.loads(archive.read("manifest.json"))
            if manifest.get("format_version") != FORMAT_VERSION:
                raise ValueError("unsupported trajectory archive version")
            files = manifest.get("files", {})
            if set(files) != names - {"manifest.json"}:
                raise ValueError("trajectory manifest inventory mismatch")
            for name, binding in files.items():
                info = archive.getinfo(name)
                if info.file_size != binding["size"]:
                    raise ValueError("trajectory manifest size mismatch")
                if name.endswith(".json") and info.file_size > 8 * 1024**2:
                    raise ValueError("excessive trajectory metadata")
                destination = self.root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target, 1024**2)
                if file_sha256(destination) != binding["sha256"]:
                    raise ValueError(f"trajectory integrity hash mismatch: {name}")

    def transition(self, step: int) -> dict[str, Any]:
        index = step - int(self.metadata["first_step"])
        if not 0 <= index < self.metadata["transition_count"]:
            raise ValueError("transition is outside the recorded range")
        return read_record(self.root, index)

    def reserve_read(self):
        with self._read_lock:
            if self._retired:
                raise ValueError("the recorded episode has been replaced")
            self._read_pins += 1
            return dict(root=str(self.root), metadata=deepcopy(self.metadata),
                        written=self.metadata["transition_count"], pending=())

    def release_read(self):
        with self._read_lock:
            self._read_pins -= 1
            if self._retired and not self._read_pins:
                shutil.rmtree(self.root, ignore_errors=True)

    def close(self) -> None:
        with self._read_lock:
            self._retired = True
            if not self._read_pins:
                shutil.rmtree(self.root, ignore_errors=True)
