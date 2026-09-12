"""Experimental, single-writer RGB trajectory collection. See the adjacent README."""

from __future__ import annotations

from collections import deque
from contextlib import AbstractContextManager
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import base64
import io
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

# Reuse the data-only codec, not Player's recording, session, or UI machinery.
from gradlab.play_trajectory import encode_tree, decode_tree, portable_metadata
from gradlab.seeds import (
    EVAL_SEED_START,
    validate_eval_seed,
    validate_playback_seed,
    validate_training_seed,
)

VERSION = 2
MAX_BATCH_BYTES = 8 * 1024**2
MAX_IMAGE_BYTES = 4 * 1024**2
DISK_RESERVE = 2 * 1024**2  # SQLite rollback journal, index growth, and progress replacement.


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def schema(fields):
    return pa.schema([(name, getattr(pa, kind)()) for name, kind in fields])


transition_schema = schema(
    [
        *[
            (name, "int64")
            for name in (
                "episode_id",
                "step",
                "source_frame_id",
                "successor_frame_id",
                "policy_decision_id",
                "configured_frame_skip",
                "elapsed_native_frames",
            )
        ],
        *[
            (name, "bool_")
            for name in (
                "successor_frame_new",
                "terminated",
                "truncated",
                "native_game_over",
                "native_truncated",
                "task_terminated",
                "task_truncated",
            )
        ],
        *[
            (name, "float64")
            for name in ("policy_reward", "native_reward", "task_reward", "temperature")
        ],
        *[
            (name, "string")
            for name in (
                "session_id",
                "action_selection_mode",
                "action_override_rule_id",
                "selected_action_json",
                "effective_action_json",
                "native_action_json",
                "record_json",
            )
        ],
    ]
)
episode_schema = schema(
    [
        *[(name, "int64") for name in ("episode_id", "seed", "initial_frame_id", "length")],
        *[(name, "string") for name in ("session_id", "split", "status", "end_reason")],
        ("initial_frame_new", "bool_"),
    ]
)
session_schema = schema([("session_id", "string"), ("record_json", "string")])
image_features = {
    "frame_id": {"dtype": "int64", "_type": "Value"},
    "sha256": {"dtype": "string", "_type": "Value"},
    "image": {"_type": "Image"},
}
frame_schema = pa.schema(
    [
        ("frame_id", pa.int64()),
        ("sha256", pa.string()),
        ("image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
    ],
    metadata={b"huggingface": canonical({"info": {"features": image_features}})},
)


TABLE_SCHEMAS = {
    "frames": frame_schema,
    "transitions": transition_schema,
    "episodes": episode_schema,
    "sessions": session_schema,
}


def record_json(value):
    tree = encode_tree(value)
    for leaf in tree["arrays"]:
        leaf["data"] = base64.b64encode(leaf["data"]).decode("ascii")
    return canonical(tree).decode()


def read_record(value):
    tree = json.loads(value)
    for leaf in tree["arrays"]:
        leaf["data"] = base64.b64decode(leaf["data"], validate=True)
    return decode_tree(tree)


def transition_record(row):
    result = {key: row.get(key) for key in transition_schema.names}
    for key in ("selected_action", "effective_action", "native_action"):
        value = row.get(key)
        if isinstance(value, np.ndarray | np.generic):
            value = value.tolist()
        result[f"{key}_json"] = json.dumps(value, allow_nan=False, separators=(",", ":"))
    result["record_json"] = record_json(row)
    return result


def parquet_bytes(kind, rows):
    stream = pa.BufferOutputStream()
    pq.write_table(
        pa.Table.from_pylist(rows, schema=TABLE_SCHEMAS[kind]),
        stream,
        compression="zstd",
        row_group_size=64,
        write_page_index=True,
    )
    return stream.getvalue().to_pybytes()


def decode_rgb(data, shape):
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (shape[1], shape[0]):
            raise ValueError("stored PNG disagrees with RGB contract")
        return owned_rgb(np.asarray(image))


def owned_rgb(image):
    value = np.asarray(image)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("rendered image must be uint8 HWC RGB")
    if min(value.shape) < 1 or value.nbytes > MAX_IMAGE_BYTES:
        raise ValueError("invalid or excessive provider image dimensions")
    return value.copy(order="C")


def image_header(shape):
    return canonical(
        {
            "version": VERSION,
            "shape": list(shape),
            "dtype": "uint8",
            "order": "HWC",
            "channels": "RGB",
        }
    )


def image_hash(image):
    return hashlib.sha256(image_header(image.shape) + image.tobytes()).hexdigest()


def existing_size(path):
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def disk_bytes(root):
    total = 0
    for parent, _, files in os.walk(root):
        for name in files:
            try:
                total += (Path(parent) / name).stat().st_size
            except FileNotFoundError:
                pass
    return total


def sync_directory(root):
    fd = os.open(root, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_synced(path, data):
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


@dataclass(frozen=True)
class Limits:
    max_steps: int = 10000
    max_bytes: int = 10 * 1024**3
    max_seconds: float = 3600
    batch_steps: int = 128

    def __post_init__(self):
        for name in ("max_steps", "max_bytes", "batch_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError("max_seconds must be positive and finite")
        if self.batch_steps > 4096:
            raise ValueError("batch_steps must be <= 4096")


@dataclass(frozen=True)
class TemperatureSchedule:
    enabled: bool = False
    values: tuple[float, ...] = (0.75, 1.0, 1.25)
    probabilities: tuple[float, ...] = (0.20, 0.60, 0.20)
    block_decisions: int = 256
    seed: int = 0

    def __post_init__(self):
        validate_playback_seed(self.seed)
        if not self.values or any(not math.isfinite(v) or v <= 0 for v in self.values):
            raise ValueError("temperatures must be positive and finite")
        if (
            len(self.values) != len(self.probabilities)
            or any(not math.isfinite(p) or p < 0 for p in self.probabilities)
            or not math.isclose(sum(self.probabilities), 1.0, abs_tol=1e-12, rel_tol=0)
        ):
            raise ValueError("temperature probabilities must be nonnegative and sum to one")
        if type(self.block_decisions) is not int or self.block_decisions < 1:
            raise ValueError("temperature block length must be a positive integer")

    def validate_execution(self, execution):
        if self.enabled and (
            execution.action_selection_mode != "stochastic" or not execution.supports_temperature
        ):
            raise ValueError("exploration requires a temperature-capable stochastic Policy")
        if self.enabled and not any(
            v != 1 and p > 0 for v, p in zip(self.values, self.probabilities)
        ):
            raise ValueError("exploration schedule cannot change the action distribution")


class DatasetReader(AbstractContextManager):
    """Read only SQLite-committed files. No model or environment is instantiated."""

    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_bytes())
        if self.manifest.get("format_version") != VERSION:
            raise ValueError("unsupported dataset version; preserve it and use a new directory")
        self.db = sqlite3.connect(
            f"{(self.root / 'index.sqlite').resolve().as_uri()}?mode=ro", uri=True
        )
        self.db.row_factory = sqlite3.Row
        identity = self.db.execute("SELECT sha256 FROM identity").fetchone()
        if identity is None or identity[0] != hashlib.sha256(canonical(self.manifest)).hexdigest():
            self.db.close()
            raise ValueError("dataset manifest integrity failure")

    def close(self):
        self.db.close()

    def __exit__(self, *args):
        self.close()

    def _data(self, name):
        # Paths are generated flat names. Never follow an imported arbitrary path.
        if Path(name).name != name or (self.root / name).is_symlink():
            raise ValueError("invalid dataset file path")
        return self.root / name

    def _table(self, name):
        binding = self.db.execute("SELECT sha256, size FROM files WHERE name=?", (name,)).fetchone()
        if binding is None:
            raise ValueError("uncommitted record file")
        path = self._data(name)
        if binding["size"] > 4 * MAX_BATCH_BYTES or path.stat().st_size != binding["size"]:
            raise ValueError("record file integrity failure: size disagrees with bounded contract")
        data = path.read_bytes()
        if len(data) != binding["size"] or hashlib.sha256(data).hexdigest() != binding["sha256"]:
            raise ValueError(f"record file integrity failure: {name}")
        return pq.read_table(pa.BufferReader(data))

    def _records(self, name):
        return [
            read_record(row["record_json"]) if "record_json" in row else row
            for row in self._table(name).to_pylist()
        ]

    def episodes(self):
        for row in self.db.execute("SELECT record_file FROM episodes ORDER BY episode_id"):
            yield self._records(row[0])[0]

    def episode(self, episode_id):
        row = self.db.execute(
            "SELECT record_file FROM episodes WHERE episode_id=?", (episode_id,)
        ).fetchone()
        if row is None:
            raise ValueError("unknown episode")
        return self._records(row[0])[0]

    def session(self, session_id):
        row = self.db.execute(
            "SELECT record_file FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        if row is None:
            raise ValueError("unknown session")
        return self._records(row[0])[0]

    def transitions(self, episode_id):
        for row in self.db.execute(
            "SELECT name FROM batches WHERE episode_id=? ORDER BY first_step", (episode_id,)
        ):
            yield from self._records(row[0])

    def transition(self, episode_id, step):
        row = self.db.execute(
            "SELECT name, first_step FROM batches WHERE episode_id=? AND first_step<=? AND last_step>=?",
            (episode_id, step, step),
        ).fetchone()
        if row is None:
            raise ValueError("unknown committed transition")
        return self._records(row["name"])[step - row["first_step"]]

    def frame(self, frame_id):
        row = self.db.execute("SELECT * FROM frames WHERE frame_id=?", (frame_id,)).fetchone()
        if row is None:
            raise ValueError(f"missing frame {frame_id}")
        shape = json.loads(row["shape"])
        size = math.prod(shape)
        if len(shape) != 3 or shape[2] != 3 or not 0 < size <= MAX_IMAGE_BYTES:
            raise ValueError("invalid stored image dimensions")
        frame = self._table(row["shard"]).slice(row["row_index"], 1).to_pylist()[0]
        if frame["frame_id"] != frame_id or frame["sha256"] != row["sha256"]:
            raise ValueError("frame index disagrees with Parquet row")
        image = decode_rgb(frame["image"]["bytes"], shape)
        if image_hash(image) != row["sha256"]:
            raise ValueError("RGB hash mismatch")
        return image

    def progress(self):
        captures, transitions, complete, episodes, unique, raw, compressed = self.db.execute(
            "SELECT coalesce((SELECT value FROM counters WHERE name='captured_occurrences'),0), "
            "coalesce((SELECT value FROM counters WHERE name='transitions'),0), "
            "(SELECT count(*) FROM episodes WHERE status='complete'), "
            "(SELECT count(*) FROM episodes), count(*), coalesce(sum(raw_size),0), "
            "(SELECT coalesce(sum(size),0) FROM files WHERE name IN "
            "(SELECT DISTINCT shard FROM frames)) FROM frames"
        ).fetchone()
        occurrences = captures
        actual = disk_bytes(self.root)
        return {
            "transitions": transitions,
            "unique_frames": unique,
            "complete_episodes": complete,
            "incomplete_episodes": episodes - complete,
            "captured_occurrences": occurrences,
            "reuse_fraction": None if not occurrences else 1 - unique / occurrences,
            "unique_raw_bytes": raw,
            "compressed_rgb_bytes": compressed,
            "compression_fraction_saved": None if not raw else 1 - compressed / raw,
            "actual_bytes": actual,
            "non_rgb_bytes": actual - compressed,
            "index_bytes": sum(existing_size(p) for p in self.root.glob("index.sqlite*")),
            "pending_transitions": 0,
        }


@dataclass
class WriterLane:
    episode: dict | None = None
    frame: int | None = None
    rows: list = field(default_factory=list)
    dirty: bool = False


class DatasetWriter(AbstractContextManager):
    """Bounded synchronous batches. File durability precedes SQLite visibility."""

    def __init__(
        self,
        root,
        contract,
        limits,
        *,
        seed_start=0,
        heldout_seed_start=EVAL_SEED_START,
        heldout_every=5,
    ):
        validate_training_seed(seed_start)
        validate_eval_seed(heldout_seed_start)
        validate_playback_seed(heldout_seed_start)
        if type(heldout_every) is not int or heldout_every < 2:
            raise ValueError("heldout_every must be an integer >= 2")
        self.root, self.limits = Path(root), limits
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / ".writer.lock").open("a+b")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise ValueError("dataset already has an active writer") from None
        self.db = None
        self.reader = None
        self.frames = {}
        self.lanes = {}
        self.select_lane(0)
        self.pending_bytes = 0
        self.delta_captures = 0
        self.episode_record = None
        self.failed = False
        try:
            manifest_path = self.root / "manifest.json"
            manifest = {
                "format_version": VERSION,
                "contract": portable_metadata(contract),
                "image_format": {"dtype": "uint8", "order": "HWC", "channels": "RGB"},
                "seed_start": seed_start,
                "heldout_seed_start": heldout_seed_start,
                "heldout_every": heldout_every,
                "scientific_evidence": False,
            }
            if manifest_path.exists():
                if json.loads(manifest_path.read_bytes()) != manifest:
                    raise ValueError(
                        "incompatible append contract, seed allocation, or dataset version"
                    )
            else:
                if any(p.name != ".writer.lock" for p in self.root.iterdir()):
                    raise ValueError("refusing to initialize a nonempty dataset directory")
                self._budget(len(canonical(manifest)))
                write_synced(manifest_path, canonical(manifest))
                sync_directory(self.root)
            self.manifest = manifest
            self.db = sqlite3.connect(self.root / "index.sqlite")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute("PRAGMA cache_size=-2048")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS identity(sha256 TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS files(name TEXT PRIMARY KEY, sha256 TEXT, size INTEGER);
                CREATE TABLE IF NOT EXISTS frames(frame_id INTEGER PRIMARY KEY, sha256 TEXT UNIQUE,
                    shape TEXT, shard TEXT, row_index INTEGER, size INTEGER, raw_size INTEGER);
                CREATE TABLE IF NOT EXISTS batches(name TEXT PRIMARY KEY, episode_id INTEGER,
                    first_step INTEGER, last_step INTEGER);
                CREATE INDEX IF NOT EXISTS batch_episode ON batches(episode_id, first_step);
                CREATE TABLE IF NOT EXISTS episodes(episode_id INTEGER PRIMARY KEY, seed INTEGER UNIQUE,
                    split TEXT, status TEXT, record_file TEXT);
                CREATE TABLE IF NOT EXISTS sessions(session_id TEXT PRIMARY KEY, record_file TEXT);
                CREATE TABLE IF NOT EXISTS counters(name TEXT PRIMARY KEY, value INTEGER);
            """)
            if not self.db.execute("SELECT count(*) FROM identity").fetchone()[0]:
                self.db.execute(
                    "INSERT INTO identity VALUES(?)",
                    (hashlib.sha256(canonical(manifest)).hexdigest(),),
                )
            self.db.commit()
            self.reader = DatasetReader(self.root)
            shape_row = self.db.execute("SELECT shape FROM frames LIMIT 1").fetchone()
            self.shape = None if shape_row is None else tuple(json.loads(shape_row[0]))
            self.next_frame = self.db.execute(
                "SELECT coalesce(max(frame_id),0)+1 FROM frames"
            ).fetchone()[0]
            # An interrupted allocation remains visible, with the last durable prefix.
            unfinished = list(
                self.db.execute("SELECT episode_id FROM episodes WHERE status='active'")
            )
            for (episode_id,) in unfinished:
                self.episode_record = self.reader.episode(episode_id)
                self.episode_record.update(status="incomplete", end_reason="interrupted")
                self.flush()
            self.episode_record = None
        except BaseException:
            self.close()
            raise

    def _budget(self, additional):
        index = self.root / "index.sqlite"
        journal_reserve = index.stat().st_size * 2 if index.exists() else 0
        if (
            disk_bytes(self.root) + additional + journal_reserve + DISK_RESERVE
            > self.limits.max_bytes
        ):
            raise ValueError(
                "dataset disk limit reached, including pending bytes and index reserve"
            )

    def select_lane(self, lane):
        self.lane = self.lanes.setdefault(lane, WriterLane())

    @property
    def episode_record(self):
        return self.lane.episode

    @episode_record.setter
    def episode_record(self, value):
        self.lane.episode = value
        self.lane.dirty = True

    @property
    def current_frame(self):
        return self.lane.frame

    @current_frame.setter
    def current_frame(self, value):
        self.lane.frame = value

    @property
    def rows(self):
        return self.lane.rows

    @property
    def pending_transitions(self):
        return sum(len(lane.rows) for lane in self.lanes.values())

    def session(self, provenance, settings):
        self.session_id = uuid.uuid4().hex
        data = {
            "session_id": self.session_id,
            "record_json": record_json(
                {
                    "session_id": self.session_id,
                    "provenance": portable_metadata(provenance),
                    "settings": settings,
                }
            ),
        }
        name = f"session-{self.session_id}.parquet"
        self._commit_files(
            {name: parquet_bytes("sessions", [data])},
            lambda: self.db.execute("INSERT INTO sessions VALUES(?,?)", (self.session_id, name)),
        )

    def _commit_files(self, outputs, update):
        self._budget(sum(map(len, outputs.values())))
        index_size = (self.root / "index.sqlite").stat().st_size
        page_size = self.db.execute("PRAGMA page_size").fetchone()[0]
        remaining = (
            self.limits.max_bytes
            - disk_bytes(self.root)
            - sum(map(len, outputs.values()))
            - 2 * index_size
            - DISK_RESERVE
        )
        self.db.execute(f"PRAGMA max_page_count={max(1, (index_size + remaining) // page_size)}")
        try:
            for name, data in outputs.items():
                write_synced(self.root / name, data)
            sync_directory(self.root)
            with self.db:
                for name, data in outputs.items():
                    self.db.execute(
                        "INSERT INTO files VALUES(?,?,?)",
                        (name, hashlib.sha256(data).hexdigest(), len(data)),
                    )
                update()
        except BaseException:
            # Do not retry a write with an uncertain visibility outcome during cleanup.
            self.failed = True
            raise

    def reserve_episode(self):
        self.flush()
        index = self.db.execute("SELECT coalesce(max(episode_id),0)+1 FROM episodes").fetchone()[0]
        split = "heldout" if index % self.manifest["heldout_every"] == 0 else "train"
        last = self.db.execute("SELECT max(seed) FROM episodes WHERE split=?", (split,)).fetchone()[
            0
        ]
        seed = (
            self.manifest["seed_start" if split == "train" else "heldout_seed_start"]
            if last is None
            else last + 1
        )
        (validate_training_seed if split == "train" else validate_eval_seed)(seed)
        validate_playback_seed(seed)
        self.episode_record = {
            "episode_id": index,
            "session_id": self.session_id,
            "seed": seed,
            "split": split,
            "status": "active",
            "end_reason": None,
            "initial_frame_id": None,
            "initial_frame_new": None,
            "length": 0,
        }
        self.flush()  # Persist identity and seed before the environment or Policy uses them.
        return deepcopy(self.episode_record)

    def _frame(self, image):
        image = owned_rgb(image)
        if self.shape is None:
            self.shape = image.shape
        if image.shape != self.shape:
            raise ValueError("provider RGB dimensions changed within the dataset")
        digest = image_hash(image)
        pending = self.frames.get(digest)
        existing = self.db.execute(
            "SELECT frame_id FROM frames WHERE sha256=?", (digest,)
        ).fetchone()
        if pending is not None or existing is not None:
            previous = (
                decode_rgb(pending[1], self.shape)
                if pending is not None
                else self.reader.frame(existing[0])
            )
            if not np.array_equal(previous, image):
                raise ValueError("RGB hash collision: matching hash has different bytes")
            return (pending[0] if pending is not None else existing[0]), False
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="PNG")
        encoded = buffer.getvalue()
        frame_id = self.next_frame
        self.next_frame += 1
        self.frames[digest] = (frame_id, encoded)
        self.pending_bytes += len(encoded)
        return frame_id, True

    def initial(self, image):
        frame_id, new = self._frame(image)
        self.episode_record.update(initial_frame_id=frame_id, initial_frame_new=new)
        self.lane.dirty = True
        self.current_frame = frame_id
        self.delta_captures += 1
        self.flush()

    def append(self, image, facts):
        # Flush before admitting another bounded item. No asynchronous queue can drop it.
        if self.pending_bytes >= MAX_BATCH_BYTES:
            self.flush()
        row = {
            **deepcopy(facts),
            "episode_id": self.episode_record["episode_id"],
            "session_id": self.episode_record["session_id"],
            "step": self.episode_record["length"],
            "source_frame_id": self.current_frame,
            # Upper-bound ID width and boolean size before staging this capture.
            "successor_frame_id": self.next_frame,
            "successor_frame_new": False,
        }
        candidate = transition_record(row)
        if pa.Table.from_pylist([candidate], schema=transition_schema).nbytes > MAX_BATCH_BYTES:
            raise ValueError("transition metadata exceeds bounded batch size")
        frame_id, new = self._frame(image)
        row.update(successor_frame_id=frame_id, successor_frame_new=new)
        encoded = transition_record(row)
        self.rows.append(encoded)
        self.pending_bytes += pa.Table.from_pylist([encoded], schema=transition_schema).nbytes
        self.current_frame = frame_id
        self.delta_captures += 1
        self.episode_record["length"] += 1
        self.lane.dirty = True
        if len(self.rows) >= self.limits.batch_steps or self.pending_bytes >= MAX_BATCH_BYTES:
            self.flush()
        return row

    def end_episode(self, reason, *, complete):
        self.episode_record.update(
            status="complete" if complete else "incomplete", end_reason=reason
        )
        self.lane.dirty = True
        self.flush()
        self.episode_record = None

    def flush(self):
        active = [lane for lane in self.lanes.values() if lane.episode is not None and lane.dirty]
        if not active:
            return
        token = uuid.uuid4().hex
        shard_name = f"frames-{token}.parquet"
        outputs = {}
        batches, episodes = [], []
        for lane in active:
            episode, rows = lane.episode, lane.rows
            episode_id = episode["episode_id"]
            episode_name = f"episode-{episode_id}-{token}.parquet"
            outputs[episode_name] = parquet_bytes("episodes", [episode])
            episodes.append(
                (episode_id, episode["seed"], episode["split"], episode["status"], episode_name)
            )
            if rows:
                rows_name = f"steps-{episode_id}-{token}.parquet"
                outputs[rows_name] = parquet_bytes("transitions", rows)
                batches.append((rows_name, episode_id, rows[0]["step"], rows[-1]["step"]))
        frame_records, images = [], []
        for index, (digest, (frame_id, png)) in enumerate(self.frames.items()):
            frame_records.append(
                (
                    frame_id,
                    digest,
                    json.dumps(self.shape),
                    shard_name,
                    index,
                    len(png),
                    math.prod(self.shape),
                )
            )
            images.append(
                {"frame_id": frame_id, "sha256": digest, "image": {"bytes": png, "path": None}}
            )
        if images:
            outputs[shard_name] = parquet_bytes("frames", images)

        def update():
            self.db.executemany("INSERT INTO frames VALUES(?,?,?,?,?,?,?)", frame_records)
            self.db.executemany("INSERT INTO batches VALUES(?,?,?,?)", batches)
            self.db.executemany("INSERT OR REPLACE INTO episodes VALUES(?,?,?,?,?)", episodes)
            for name, count in (
                ("captured_occurrences", self.delta_captures),
                ("transitions", self.pending_transitions),
            ):
                self.db.execute(
                    "INSERT INTO counters VALUES(?,?) ON CONFLICT(name) DO UPDATE SET value=value+excluded.value",
                    (name, count),
                )

        self._commit_files(outputs, update)
        self.frames.clear()
        for lane in active:
            lane.rows.clear()
            lane.dirty = False
        self.delta_captures = self.pending_bytes = 0

    def close(self):
        if self.reader is not None:
            self.reader.close()
        if self.db is not None:
            self.db.close()
        self.lock.close()

    def __exit__(self, *args):
        self.close()


class Collection(AbstractContextManager):
    """Collection and debug controls share this sole step entry point."""

    def __init__(
        self, root, execution, *, limits=Limits(), schedule=TemperatureSchedule(), **store_options
    ):
        schedule.validate_execution(execution)
        self.schedule = schedule
        self.execution, self.limits = execution, limits
        self.writer = None
        try:
            self.writer = DatasetWriter(root, execution.contract, limits, **store_options)
            self.writer.session(
                execution.provenance,
                {
                    "limits": asdict(limits),
                    "n_envs": getattr(execution, "n_envs", 1),
                    "schedule": asdict(schedule),
                    "classification": "Counterfactual Playback"
                    if schedule.enabled or execution.provenance.get("overrides")
                    else "Faithful Playback",
                    "scientific_evidence": False,
                },
            )
        except BaseException:
            if self.writer is not None:
                self.writer.close()
            execution.close()
            raise
        self.session = self.writer.reader.session(self.writer.session_id)
        self.steps = 0
        self.started = time.monotonic()
        self.n_envs = getattr(execution, "n_envs", 1)
        self.active = set()
        self.schedule_rngs, self.temperatures = {}, {}
        self.finished = False
        self.latest_source = None
        self.latest_row = None
        self.progress_samples = deque(maxlen=120)
        self.initial_progress = self.writer.reader.progress()
        self.progress_samples.append((self.started, self.initial_progress))
        self.next_report = self.started + 5

    def progress(self):
        import resource
        import sys

        now = time.monotonic()
        current = self.writer.reader.progress()
        oldest_time, oldest = self.progress_samples[0]
        interval = max(now - oldest_time, 1e-9)
        captures = current["captured_occurrences"] - oldest["captured_occurrences"]
        unique = current["unique_frames"] - oldest["unique_frames"]
        elapsed = max(now - self.started, 1e-9)
        growth = max(0, current["actual_bytes"] - self.initial_progress["actual_bytes"])
        current.update(
            elapsed_seconds=elapsed,
            recent_window_seconds=interval,
            recent_new_images_per_second=unique / interval,
            recent_new_image_fraction=None if not captures else unique / captures,
            transitions_per_second=(
                current["transitions"] - self.initial_progress.get("transitions", 0)
            )
            / elapsed,
            bytes_per_second=growth / elapsed,
            projected_bytes_per_hour=growth / elapsed * 3600,
            projected_bytes_per_million_transitions=None
            if not self.steps
            else growth / self.steps * 1_000_000,
            pending_transitions=self.writer.pending_transitions,
            pending_captured_occurrences=self.writer.delta_captures,
            pending_buffer_bytes=self.writer.pending_bytes,
            n_envs=self.n_envs,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            measured_at_unix=time.time(),
        )
        if now - self.progress_samples[-1][0] >= 1:
            self.progress_samples.append((now, dict(current)))
        return current

    def publish_progress(self):
        current = self.progress()
        data = canonical(current)
        self.writer._budget(len(data))
        temporary = self.writer.root / f"progress-{uuid.uuid4().hex}.tmp"
        write_synced(temporary, data)
        os.replace(temporary, self.writer.root / "progress.json")
        sync_directory(self.writer.root)
        return current

    def step(self):
        if self.finished:
            return None
        if (
            self.steps >= self.limits.max_steps
            or time.monotonic() - self.started >= self.limits.max_seconds
        ):
            self.stop("collection_limit")
            return None
        # A partial final batch never executes unrecorded extra transitions.
        lanes = range(min(self.n_envs, self.limits.max_steps - self.steps))
        temperatures = {}
        for lane in lanes:
            self.writer.select_lane(lane)
            if lane not in self.active:
                episode = self.writer.reserve_episode()
                self.active.add(lane)
                self.schedule_rngs[lane] = np.random.default_rng(
                    np.random.SeedSequence([self.schedule.seed, episode["episode_id"]])
                )
                image = (
                    self.execution.reset(episode["seed"])
                    if self.n_envs == 1
                    else self.execution.reset_lane(lane, episode["seed"])
                )
                self.writer.initial(image)
            decision_id = self.writer.episode_record["length"]
            if decision_id % self.schedule.block_decisions == 0:
                self.temperatures[lane] = (
                    float(
                        self.schedule_rngs[lane].choice(
                            self.schedule.values, p=self.schedule.probabilities
                        )
                    )
                    if self.schedule.enabled
                    else 1.0
                )
            temperatures[lane] = self.temperatures[lane]
        results = (
            {0: self.execution.step(temperatures[0])}
            if self.n_envs == 1
            else self.execution.step_batch(temperatures)
        )
        for lane, (image, facts) in results.items():
            self.writer.select_lane(lane)
            self.latest_source = owned_rgb(image)
            facts = {
                **facts,
                "policy_decision_id": self.writer.episode_record["length"],
                "temperature": temperatures[lane],
                "action_selection_mode": self.execution.action_selection_mode,
                "configured_frame_skip": self.execution.contract["frame_skip"],
                "lane_id": lane,
            }
            self.latest_row = self.writer.append(image, facts)
            self.steps += 1
            if facts["terminated"] or facts["truncated"]:
                self.writer.end_episode("environment_boundary", complete=True)
                self.active.remove(lane)
        return self.latest_row

    def run(self, progress_callback=None):
        while not self.finished:
            self.step()
            if time.monotonic() >= self.next_report:
                current = self.publish_progress()
                if progress_callback is not None:
                    progress_callback(current)
                self.next_report = time.monotonic() + 5

    def stop(self, reason="collection_cutoff"):
        if self.finished or self.writer.failed:
            return
        for lane in sorted(self.active):
            self.writer.select_lane(lane)
            self.writer.end_episode(reason, complete=False)
        self.active.clear()
        self.finished = True
        self.publish_progress()

    def __exit__(self, exc_type, exc, tb):
        try:
            try:
                self.stop("recording_failure" if exc is not None else "collection_cutoff")
            except Exception:
                if exc is None:
                    raise
                # Preserve the original recording error; recovery marks the durable prefix.
        finally:
            self.writer.close()
            self.execution.close()


def validate_dataset(root):
    with DatasetReader(root) as reader:
        reader.db.execute("PRAGMA temp_store=FILE")
        reader.db.execute(
            "CREATE TEMP TABLE seen(frame_id INTEGER PRIMARY KEY, discoveries INTEGER NOT NULL DEFAULT 0)"
        )
        reader.db.execute("BEGIN")  # One committed dataset prefix, even with a concurrent writer.
        transitions, occurrences, unique = 0, 0, 0
        shape = None

        def capture(frame_id, claimed_new):
            nonlocal occurrences, unique, shape
            first = (
                reader.db.execute(
                    "INSERT OR IGNORE INTO seen(frame_id) VALUES(?)", (frame_id,)
                ).rowcount
                == 1
            )
            if first:
                image = reader.frame(frame_id)
                if shape is not None and shape != image.shape:
                    raise ValueError("dataset image dimensions disagree")
                shape = image.shape
                unique += 1
            if type(claimed_new) is not bool:
                raise ValueError("image discovery attribution mismatch")
            reader.db.execute(
                "UPDATE seen SET discoveries=discoveries+? WHERE frame_id=?",
                (int(claimed_new), frame_id),
            )
            occurrences += 1

        for episode in reader.episodes():
            seed = episode["seed"]
            (validate_training_seed if episode["split"] == "train" else validate_eval_seed)(seed)
            validate_playback_seed(seed)
            if episode["split"] not in {"train", "heldout"}:
                raise ValueError("invalid episode split")
            source = episode["initial_frame_id"]
            if source is not None:
                capture(source, episode["initial_frame_new"])
            session = reader.db.execute(
                "SELECT record_file FROM sessions WHERE session_id=?", (episode["session_id"],)
            ).fetchone()
            if session is None:
                raise ValueError("missing episode session provenance")
            reader._records(session[0])
            length, last_boundary = 0, False
            for row in reader.transitions(episode["episode_id"]):
                if (
                    row["episode_id"] != episode["episode_id"]
                    or row["step"] != length
                    or row["policy_decision_id"] != length
                    or row["source_frame_id"] != source
                    or last_boundary
                ):
                    raise ValueError("transition continuity failure")
                if not math.isfinite(row["temperature"]) or row["temperature"] <= 0:
                    raise ValueError("invalid recorded temperature")
                capture(row["successor_frame_id"], row["successor_frame_new"])
                source = row["successor_frame_id"]
                last_boundary = bool(row["terminated"] or row["truncated"])
                length += 1
            if length != episode["length"] or (
                episode["status"] == "complete" and not last_boundary
            ):
                raise ValueError("episode length or boundary mismatch")
            transitions += length
        # Episode traversal differs from capture order when lanes interleave.
        # Each global image must still have exactly one discovering occurrence.
        if reader.db.execute("SELECT 1 FROM seen WHERE discoveries != 1 LIMIT 1").fetchone():
            raise ValueError("image discovery attribution mismatch")
        progress = reader.progress()
        if (
            transitions != progress["transitions"]
            or occurrences != progress["captured_occurrences"]
            or unique != progress["unique_frames"]
        ):
            raise ValueError("committed counter or frame inventory mismatch")
        return {"valid": True, **progress}


class PolicyExecution:
    """Single-lane adapter over GradLab's existing Policy and environment APIs."""

    def __init__(self, runtime, env, config, *, contract, provenance):
        self.runtime, self.env, self.config = runtime, env, config
        self.contract, self.provenance = contract, provenance
        self.action_selection_mode = runtime.capabilities.default_action_selection_mode
        self.supports_temperature = runtime.supports_sampling_temperature
        self.info = {}

    def _observation(self, obs):
        from collections.abc import Mapping
        from gradlab.policy_observation import model_observation, task_info_value_from_info

        # Native structured inputs are already prepared by the contracted task kernel.
        if isinstance(obs, Mapping):
            return obs
        image = np.asarray(obs)
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("Policy observations must be single-lane channel-first batches")
        spaces = getattr(getattr(self.runtime.model, "observation_space", None), "spaces", None)
        if isinstance(spaces, dict) and {"image", "task"}.issubset(spaces):
            return model_observation(
                self.runtime.model,
                image,
                self.config,
                active_info_value=task_info_value_from_info(self.info, self.config),
            )
        return image

    def reset(self, seed, *, reset_policy=True):
        import torch

        if reset_policy:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if bool(getattr(self.runtime.model, "use_sde", False)):
                self.runtime.model.policy.reset_noise()
        self.env.seed(seed)
        obs = self.env.reset()
        if reset_policy:
            self.runtime.reset()
        self.info = dict(self.env.reset_infos[0])
        self.obs = self._observation(obs)
        return owned_rgb(self.env.get_images()[0])

    def step(self, temperature):
        context = (
            self.env.policy_execution_context(self.runtime.model)
            if hasattr(self.env, "policy_execution_context")
            else None
        )
        decision = self.runtime.decide(
            self.obs,
            action_selection_mode=self.action_selection_mode,
            sampling_temperature=temperature,
            include_diagnostics=False,
            execution_context=context,
        )
        if len(decision.decisions) != 1:
            raise ValueError("collection requires one Policy decision per environment step")
        return self.apply(decision.actions, decision.decisions[0].raw_action)

    def apply(self, actions, raw_action):
        obs, rewards, dones, infos = self.env.step(actions)
        diagnostics = self.env.take_step_diagnostics()
        self.env.drain_records()  # Do not retain training telemetry across a long collection.
        if diagnostics is None:
            raise ValueError("required single-lane step diagnostics are unavailable")
        boundary = bool(dones[0])
        if boundary and diagnostics.terminal_frame is None:
            raise ValueError("terminal RGB unavailable; recording preserves an incomplete prefix")
        rgb = owned_rgb(diagnostics.terminal_frame if boundary else self.env.get_images()[0])
        if bool(diagnostics.terminated or diagnostics.truncated) != boundary:
            raise ValueError("step diagnostics disagree with the environment boundary")
        self.info = dict(infos[0].get("reset_info", {}) if boundary else diagnostics.provider_info)
        self.obs = self._observation(obs)
        return rgb, {
            "selected_action": deepcopy(raw_action),
            "policy_action": deepcopy(diagnostics.policy_action),
            "effective_action": deepcopy(diagnostics.effective_policy_action),
            "native_action": deepcopy(diagnostics.native_action),
            "action_override_rule_id": diagnostics.action_override_rule_id,
            "native_reward": diagnostics.provider_reward,
            "task_reward": diagnostics.task_reward,
            "policy_reward": float(rewards[0]),
            "native_game_over": diagnostics.provider_terminated,
            "native_truncated": diagnostics.provider_truncated,
            "task_terminated": diagnostics.task_terminated,
            "task_truncated": diagnostics.task_truncated,
            "terminated": diagnostics.terminated,
            "truncated": diagnostics.truncated,
            "outcome": int(diagnostics.outcome),
            "events": diagnostics.events,
            "event_transitions": deepcopy(diagnostics.event_transitions),
            "labels": deepcopy(dict(diagnostics.provider_info)) or None,
            "task_metrics": deepcopy(dict(diagnostics.task_metrics)) or None,
            # The pinned provider does not expose per-transition native frame duration.
            "elapsed_native_frames": None,
            "episode_seed": diagnostics.episode_seed,
            "start_id": diagnostics.start_id,
        }

    def close(self):
        self.env.close()


def validate_vector_runtime(runtime):
    if runtime.capabilities.algorithm_id not in {"ppo", "a2c"} or bool(
        getattr(runtime.model, "use_sde", False)
    ):
        raise ValueError(
            "vector collection requires stateless PPO/A2C without state-dependent exploration"
        )


class VectorPolicyExecution:
    """Batch one stateless actor across independently seeded environment instances.

    Native environment steps run in lane order in this process. The expensive
    actor forward pass runs once per distinct temperature, without model copies.
    """

    def __init__(self, lanes):
        self.lanes = lanes
        self.n_envs = len(lanes)
        first = lanes[0]
        self.runtime = first.runtime
        validate_vector_runtime(self.runtime)
        self.contract = first.contract
        self.provenance = {
            **first.provenance,
            "vectorization": {
                "n_envs": self.n_envs,
                "environment_execution": "sequential_independent_instances",
                "inference": "batched_by_temperature",
                "policy_rng": "session_stream_seeded_by_first_episode_seed",
            },
        }
        self.action_selection_mode = first.action_selection_mode
        self.supports_temperature = first.supports_temperature
        self.seeded = False

    def reset_lane(self, lane, seed):
        image = self.lanes[lane].reset(seed, reset_policy=not self.seeded)
        self.seeded = True
        return image

    def step_batch(self, temperatures):
        def concatenate(observations):
            if isinstance(observations[0], dict):
                return {
                    key: concatenate([obs[key] for obs in observations]) for key in observations[0]
                }
            return np.concatenate(observations, axis=0)

        groups = {}
        for lane, temperature in temperatures.items():
            groups.setdefault(temperature, []).append(lane)
        decisions = {}
        for temperature, lanes in groups.items():
            batch = self.runtime.decide(
                concatenate([self.lanes[lane].obs for lane in lanes]),
                action_selection_mode=self.action_selection_mode,
                sampling_temperature=temperature,
                include_diagnostics=False,
            )
            if len(batch.decisions) != len(lanes):
                raise ValueError("Policy decision batch does not match active collection lanes")
            for index, lane in enumerate(lanes):
                decisions[lane] = (
                    batch.actions[index : index + 1],
                    batch.decisions[index].raw_action,
                )
        return {lane: self.lanes[lane].apply(*decisions[lane]) for lane in temperatures}

    def close(self):
        for lane in self.lanes:
            lane.close()


class DebugController:
    def __init__(self, collection):
        self.collection = collection
        self.playing = False
        self.single_step = False

    def command(self, command):
        if command == "play":
            self.playing = True
        elif command == "pause":
            self.playing = False
        elif command == "step":
            self.playing, self.single_step = False, True
        else:
            raise ValueError("unknown debug command")

    def tick(self):
        run = self.collection
        if self.playing or self.single_step:
            run.step()
            run.writer.flush()
            self.single_step = False
        if run.latest_row is None:
            return None
        # Read again on redraw, so persistence corruption cannot be hidden by a source cache.
        row = run.writer.reader.transition(run.latest_row["episode_id"], run.latest_row["step"])
        decoded = run.writer.reader.frame(row["successor_frame_id"])
        equal = np.array_equal(run.latest_source, decoded)
        if not equal:
            self.playing = False
        return {
            "classification": run.session["settings"]["classification"],
            "overrides": {
                **run.session["provenance"].get("overrides", {}),
                **(
                    {"temperature_schedule": run.session["settings"]["schedule"]}
                    if run.schedule.enabled
                    else {}
                ),
            },
            "source": run.latest_source,
            "decoded": decoded,
            "row": row,
            "pixel_equal": equal,
            "progress": run.writer.reader.progress(),
            "persistence_lag": 0,
        }


class Inspector(AbstractContextManager):
    def __init__(self, root, episode_id=None):
        self.reader = DatasetReader(root)
        first = self.reader.db.execute("SELECT min(episode_id) FROM episodes").fetchone()[0]
        if first is None:
            self.reader.close()
            raise ValueError("dataset has no allocated episodes")
        self.select(first if episode_id is None else episode_id)
        self.playing = False

    def select(self, episode_id):
        self.episode = self.reader.episode(episode_id)
        self.session = self.reader.session(self.episode["session_id"])
        self.position = -1

    def move_episode(self, forward):
        comparison, order = (">", "ASC") if forward else ("<", "DESC")
        row = self.reader.db.execute(
            f"SELECT episode_id FROM episodes WHERE episode_id{comparison}? ORDER BY episode_id {order} LIMIT 1",
            (self.episode["episode_id"],),
        ).fetchone()
        if row:
            self.select(row[0])
        self.playing = False

    def move(self, delta):
        self.position = max(-1, min(self.episode["length"] - 1, self.position + delta))
        if self.position >= self.episode["length"] - 1:
            self.playing = False

    def snapshot(self):
        row = (
            None
            if self.position < 0
            else self.reader.transition(self.episode["episode_id"], self.position)
        )
        frame_id = self.episode["initial_frame_id"] if row is None else row["successor_frame_id"]
        return {
            "classification": self.session["settings"]["classification"],
            "overrides": {
                **self.session["provenance"].get("overrides", {}),
                **(
                    {"temperature_schedule": self.session["settings"]["schedule"]}
                    if self.session["settings"]["schedule"]["enabled"]
                    else {}
                ),
            },
            "source": None if row is None else self.reader.frame(row["source_frame_id"]),
            "decoded": None if frame_id is None else self.reader.frame(frame_id),
            "row": row,
            "episode": self.episode,
            "pixel_equal": None,
            "progress": self.reader.progress(),
            "persistence_lag": 0,
        }

    def __exit__(self, *args):
        self.reader.close()


def display_text(value):
    if isinstance(value, np.ndarray):
        return np.array2string(value, threshold=8, max_line_width=90)
    return str(value)


class Renderer(AbstractContextManager):
    """The same local Pygame renderer serves live readback and stored navigation."""

    def __init__(self):
        import pygame

        self.pg = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((1000, 790))
        pygame.display.set_caption("GradLab experimental dataset collector")
        self.font = pygame.font.Font(None, 23)
        self.clock = pygame.time.Clock()

    def draw(self, snapshot, *, live, playing):
        self.screen.fill((20, 23, 28))
        lines = [
            "Space: play/pause   Right: one step   Left: previous   PgUp/PgDn: episode   Esc: quit",
            f"{'Live collection' if live else 'Stored dataset'} | {'playing' if playing else 'paused'} | no scientific evidence",
        ]
        if snapshot:
            lines[1] = (
                f"{snapshot['classification']} | {'playing' if playing else 'paused'} | no scientific evidence"
            )
            overrides = snapshot["overrides"]
            details = []
            if overrides.get("full_game"):
                details.append(f"full game, cap {overrides['episode_steps']} steps")
            if "temperature_schedule" in overrides:
                details.append("temperature schedule enabled")
            lines.append(
                f"RGB equality: {snapshot['pixel_equal'] if live else 'unavailable offline'} | lag: {snapshot['persistence_lag']} | "
                + "; ".join(details)
            )
            for column, key in enumerate(("source", "decoded")):
                image = snapshot[key]
                x = 20 + column * 490
                self.screen.blit(
                    self.font.render(
                        ("Copied live RGB" if live else "Stored source RGB")
                        if key == "source"
                        else "Decoded committed RGB",
                        True,
                        (220, 225, 232),
                    ),
                    (x, 92),
                )
                if image is not None:
                    height, width, _ = image.shape
                    scale = min(460 / width, 360 / height)
                    surface = self.pg.surfarray.make_surface(image.transpose(1, 0, 2))
                    self.screen.blit(
                        self.pg.transform.scale(surface, (int(width * scale), int(height * scale))),
                        (x, 120),
                    )
            row = snapshot["row"] or snapshot.get("episode", {})
            for index, keys in enumerate(
                (
                    ("lane_id", "episode_id", "step", "policy_decision_id"),
                    ("source_frame_id", "successor_frame_id", "successor_frame_new"),
                    ("selected_action", "effective_action", "native_action"),
                    ("action_override_rule_id", "temperature", "action_selection_mode"),
                    ("policy_reward", "native_reward", "native_game_over"),
                    ("terminated", "truncated", "task_terminated", "task_truncated"),
                )
            ):
                text = " | ".join(f"{key}: {display_text(row.get(key))}" for key in keys)
                self.screen.blit(
                    self.font.render(text, True, (220, 225, 232)), (20, 500 + 29 * index)
                )
            progress = snapshot["progress"]
            for index, keys in enumerate(
                (
                    ("transitions", "unique_frames", "captured_occurrences"),
                    ("reuse_fraction", "actual_bytes", "complete_episodes", "incomplete_episodes"),
                )
            ):
                text = " | ".join(f"{key}: {progress[key]}" for key in keys)
                self.screen.blit(
                    self.font.render(text, True, (150, 205, 170)), (20, 695 + index * 28)
                )
        for index, line in enumerate(lines):
            self.screen.blit(self.font.render(line, True, (230, 235, 240)), (20, 15 + index * 26))
        self.pg.display.flip()

    def __exit__(self, *args):
        self.pg.quit()


def run_window(controller, *, live, fps=30):
    with Renderer() as renderer:
        pg = renderer.pg
        running = True
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                    running = False
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_SPACE:
                        if live:
                            controller.command("pause" if controller.playing else "play")
                        else:
                            controller.playing = not controller.playing
                    elif event.key == pg.K_RIGHT:
                        if live:
                            controller.command("step")
                        else:
                            controller.move(1)
                    elif not live and event.key == pg.K_LEFT:
                        controller.playing = False
                        controller.move(-1)
                    elif not live and event.key in (pg.K_PAGEUP, pg.K_PAGEDOWN):
                        controller.move_episode(event.key == pg.K_PAGEDOWN)
            if not running:
                break
            if live:
                snapshot = controller.tick()
                if controller.collection.finished:
                    controller.playing = False
            else:
                if controller.playing:
                    controller.move(1)
                snapshot = controller.snapshot()
            renderer.draw(snapshot, live=live, playing=controller.playing)
            renderer.clock.tick(fps)


def collection_environment(original, *, full_game, episode_steps):
    effective = deepcopy(original)
    if full_game:
        if type(episode_steps) is not int or episode_steps < 1:
            raise ValueError(
                "full-game collection requires a positive episode-duration cap in contracted steps"
            )
        effective["task"]["termination"] = {"max_episode_steps": episode_steps}
    elif episode_steps is not None:
        raise ValueError("episode-duration override requires --full-game")
    return effective


def load_execution(
    checkpoint,
    *,
    full_game=False,
    episode_steps=None,
    device="cpu",
    schedule=TemperatureSchedule(),
    n_envs=1,
):
    from importlib.metadata import version
    import subprocess
    from gradlab.action_contract import assert_action_contract_compatible
    from gradlab.env import make_eval_vec_env, resolve_env_config
    from gradlab.env_metadata import env_config_from_config_dict, runtime_versions_metadata
    from gradlab.policy_bundle import load_policy_bundle_from_checkpoint, playback_contract
    from gradlab.policy_execution import verify_policy_execution_contract
    from gradlab.policy_models import load_policy_model
    from gradlab.policy_registry import resolve_policy_algorithm
    from gradlab.policy_runtime import PolicyRuntime
    from gradlab.play_trajectory import portable_metadata
    from gradlab.trusted_inputs import stage_model_input, verify_staged_model

    if type(n_envs) is not int or not 1 <= n_envs <= 64:
        raise ValueError("n_envs must be an integer from 1 to 64")
    checkpoint = Path(checkpoint).expanduser()
    if checkpoint.is_dir():
        checkpoint = checkpoint / "model.zip"
    # Staging owns immutable copies of the Checkpoint and both bound sidecars.
    with stage_model_input(checkpoint) as staged:
        bundle = load_policy_bundle_from_checkpoint(staged.model_path)
        if bundle is None:
            raise ValueError(
                "an immutable Checkpoint bundle with model/recipe sidecars is required"
            )
        algorithm = resolve_policy_algorithm(bundle.model["policy"])
        original = playback_contract(bundle.recipe, mode="training")["environment"]
        effective = collection_environment(
            original, full_game=full_game, episode_steps=episode_steps
        )
        config = env_config_from_config_dict(effective)
        if (
            config is None
            or config.env_provider != "env-breakoutatari2600-turbo-native"
            or config.game != "Breakout-Atari2600-v0"
        ):
            raise ValueError("this experiment supports only native Breakout Turbo")
        config = resolve_env_config(config)
        provider_version = version(config.env_provider)
        if provider_version != "0.5.12":
            raise ValueError("native Breakout runtime must match the pinned 0.5.12 provider")
        runtime = PolicyRuntime(
            load_policy_model(verify_staged_model(staged), device=device, algorithm_id=algorithm),
            algorithm_id=algorithm,
        )
        schedule.validate_execution(SimpleExecutionCapabilities(runtime))
        if n_envs > 1:
            validate_vector_runtime(runtime)
        env = make_eval_vec_env(config=config, n_envs=1, seed=0, capture_step_diagnostics=True)
        environments = [env]
        try:
            training = bundle.model["provenance"].get("training_metadata") or {}
            assert_action_contract_compatible(
                training.get("action_contract"), env.runtime.action_contract
            )
            runtime.bind_action_space(env.action_space, env.runtime.action_contract)
            verify_policy_execution_contract(runtime.model, env)
            repo = Path(__file__).resolve().parents[2]
            source = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
            ).stdout.strip()
            provenance = {
                "checkpoint": deepcopy(bundle.model["checkpoint"]),
                "model_sha256": hashlib.sha256(bundle.model_path.read_bytes()).hexdigest(),
                "recipe_sha256": hashlib.sha256(bundle.recipe_path.read_bytes()).hexdigest(),
                "model": portable_metadata(bundle.model),
                "recipe": portable_metadata(bundle.recipe),
                "original_environment": original,
                "effective_environment": effective,
                "overrides": {"full_game": True, "episode_steps": episode_steps}
                if full_game
                else {},
                "source_commit": source,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "runtime_versions": runtime_versions_metadata(),
                "device": device,
                "provider_version": provider_version,
                "scientific_evidence": False,
            }
            contract = {
                "environment": asdict(config),
                "provider_version": provider_version,
                "action_contract": portable_metadata(env.runtime.action_contract),
                "frame_skip": config.frame_skip,
            }
            # Normalize tuples once so JSON reload comparison remains exact.
            contract = json.loads(canonical(contract))
            for _ in range(n_envs - 1):
                environments.append(
                    make_eval_vec_env(
                        config=config, n_envs=1, seed=0, capture_step_diagnostics=True
                    )
                )
            lanes = [
                PolicyExecution(runtime, item, config, contract=contract, provenance=provenance)
                for item in environments
            ]
            return lanes[0] if n_envs == 1 else VectorPolicyExecution(lanes)
        except BaseException:
            for item in environments:
                item.close()
            raise


class SimpleExecutionCapabilities:
    def __init__(self, runtime):
        self.action_selection_mode = runtime.capabilities.default_action_selection_mode
        self.supports_temperature = runtime.supports_sampling_temperature


def export_preview(root, destination, *, episode_id=1, max_steps=600, fps=30):
    """Reconstruct a bounded MP4 from committed images, with no Policy execution."""
    import cv2

    if max_steps < 1 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("preview limits must be positive and finite")
    destination = Path(destination)
    if destination.exists():
        raise ValueError("preview destination already exists")
    with DatasetReader(root) as reader:
        episode = reader.episode(episode_id)
        if episode["initial_frame_id"] is None:
            raise ValueError("episode has no captured image")
        initial = reader.frame(episode["initial_frame_id"])
        height, width, _ = initial.shape
        writer = cv2.VideoWriter(
            str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise ValueError("could not open preview encoder")
        try:
            writer.write(initial[:, :, ::-1])
            for index, row in enumerate(reader.transitions(episode_id)):
                if index >= max_steps:
                    break
                writer.write(reader.frame(row["successor_frame_id"])[:, :, ::-1])
        finally:
            writer.release()
    return destination


def prepare_huggingface(root):
    """Describe an exact committed snapshot; never convert or copy data files."""
    import yaml

    root = Path(root)
    with (root / ".writer.lock").open("rb") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("stop the dataset writer before preparing upload") from error
        summary = validate_dataset(root)
        if not summary["transitions"]:
            raise ValueError("preparation requires at least one committed transition")
        with DatasetReader(root) as reader:
            reader.db.execute("BEGIN")
            tables = {
                "frames": {
                    "assets": [
                        r[0]
                        for r in reader.db.execute(
                            "SELECT shard FROM frames GROUP BY shard ORDER BY min(frame_id)"
                        )
                    ]
                },
                "transitions": {},
                "episodes": {},
                "sessions": {
                    "metadata": [
                        r[0]
                        for r in reader.db.execute(
                            "SELECT record_file FROM sessions ORDER BY session_id"
                        )
                    ]
                },
            }
            for split in ("train", "heldout"):
                tables["transitions"][split] = [
                    r[0]
                    for r in reader.db.execute(
                        "SELECT b.name FROM batches b JOIN episodes e USING(episode_id) "
                        "WHERE e.split=? ORDER BY b.episode_id,b.first_step",
                        (split,),
                    )
                ]
                tables["episodes"][split] = [
                    r[0]
                    for r in reader.db.execute(
                        "SELECT record_file FROM episodes WHERE split=? ORDER BY episode_id",
                        (split,),
                    )
                ]
            configs, names = [], {"manifest.json"}
            for kind, splits in tables.items():
                files = [
                    {"split": split, "path": paths} for split, paths in splits.items() if paths
                ]
                configs.append(
                    {
                        "config_name": kind,
                        "data_files": files,
                        **({"default": True} if kind == "transitions" else {}),
                    }
                )
                names.update(name for paths in splits.values() for name in paths)
            card = {
                "pretty_name": "GradLab RGB trajectories",
                "tags": ["reinforcement-learning", "image"],
                "configs": configs,
            }
            body = Path(__file__).with_name("dataset-card.md").read_text()
            readme = ("---\n" + yaml.safe_dump(card, sort_keys=False) + "---\n" + body).encode()
            report = {
                "format_version": VERSION,
                "transitions": summary["transitions"],
                "unique_frames": summary["unique_frames"],
                "complete_episodes": summary["complete_episodes"],
                "incomplete_episodes": summary["incomplete_episodes"],
                "source_manifest_sha256": hashlib.sha256(canonical(reader.manifest)).hexdigest(),
                "files": [],
            }
            for name in sorted(names):
                path = reader._data(name)
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                report["files"].append(
                    {"path": name, "sha256": digest, "bytes": path.stat().st_size}
                )
            report["files"].append(
                {
                    "path": "README.md",
                    "sha256": hashlib.sha256(readme).hexdigest(),
                    "bytes": len(readme),
                }
            )
            metadata = canonical(report)
            if len(readme) + len(metadata) > MAX_BATCH_BYTES:
                raise ValueError("upload metadata exceeds bounded metadata size")
            # The receipt is the readiness marker. Invalidate it durably before
            # changing the card, then publish the new receipt last.
            (root / "upload.json").unlink(missing_ok=True)
            sync_directory(root)
            for name, content in (("README.md", readme), ("upload.json", metadata)):
                temporary = root / f".{name}-{uuid.uuid4().hex}"
                try:
                    write_synced(temporary, content)
                    os.replace(temporary, root / name)
                    sync_directory(root)  # Card durability must precede the new receipt.
                finally:
                    temporary.unlink(missing_ok=True)
            return report


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    collect = modes.add_parser(
        "collect", help="collect one local Checkpoint, or resume at a fresh episode"
    )
    collect.add_argument("dataset", type=Path)
    collect.add_argument("checkpoint", type=Path)
    collect.add_argument("--max-steps", type=int, default=10000)
    collect.add_argument("--max-gib", type=float, default=10)
    collect.add_argument("--max-seconds", type=float, default=3600)
    collect.add_argument("--batch-steps", type=int, default=128)
    collect.add_argument("--seed-start", type=int, default=0)
    collect.add_argument("--heldout-seed-start", type=int, default=EVAL_SEED_START)
    collect.add_argument("--heldout-every", type=int, default=5)
    collect.add_argument("--full-game", action="store_true")
    collect.add_argument("--episode-steps", type=int)
    collect.add_argument("--explore", action="store_true")
    collect.add_argument("--temperatures", type=float, nargs="+", default=[0.75, 1.0, 1.25])
    collect.add_argument("--probabilities", type=float, nargs="+", default=[0.20, 0.60, 0.20])
    collect.add_argument("--temperature-block", type=int, default=256)
    collect.add_argument("--schedule-seed", type=int, default=0)
    collect.add_argument("--device", default="cpu")
    collect.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="independent environments sharing batched PPO/A2C inference (1–64)",
    )
    collect.add_argument("--debug", action="store_true", help="start the local debugger paused")
    for name in ("validate", "inspect", "progress", "preview", "prepare-hf"):
        mode = modes.add_parser(name)
        mode.add_argument("dataset", type=Path)
        if name in {"inspect", "preview"}:
            mode.add_argument("--episode", type=int, default=1)
        if name == "preview":
            mode.add_argument("output", type=Path)
            mode.add_argument("--max-steps", type=int, default=600)
            mode.add_argument("--fps", type=float, default=30)
    args = parser.parse_args(argv)
    if args.mode == "prepare-hf":
        print(json.dumps(prepare_huggingface(args.dataset), sort_keys=True))
    elif args.mode == "validate":
        print(json.dumps(validate_dataset(args.dataset), sort_keys=True))
    elif args.mode == "progress":
        with DatasetReader(args.dataset) as reader:
            print(json.dumps(reader.progress(), sort_keys=True))
    elif args.mode == "inspect":
        with Inspector(args.dataset, args.episode) as inspector:
            run_window(inspector, live=False)
    elif args.mode == "preview":
        print(
            export_preview(
                args.dataset,
                args.output,
                episode_id=args.episode,
                max_steps=args.max_steps,
                fps=args.fps,
            )
        )
    else:
        if not math.isfinite(args.max_gib) or args.max_gib <= 0:
            parser.error("--max-gib must be positive and finite")
        limits = Limits(
            args.max_steps, int(args.max_gib * 1024**3), args.max_seconds, args.batch_steps
        )
        schedule = TemperatureSchedule(
            args.explore,
            tuple(args.temperatures),
            tuple(args.probabilities),
            args.temperature_block,
            args.schedule_seed,
        )
        execution = load_execution(
            args.checkpoint,
            full_game=args.full_game,
            episode_steps=args.episode_steps,
            device=args.device,
            schedule=schedule,
            n_envs=args.n_envs,
        )
        with Collection(
            args.dataset,
            execution,
            limits=limits,
            schedule=schedule,
            seed_start=args.seed_start,
            heldout_seed_start=args.heldout_seed_start,
            heldout_every=args.heldout_every,
        ) as run:
            try:
                if args.debug:
                    run_window(DebugController(run), live=True)
                else:
                    run.run(
                        lambda progress: print(json.dumps(progress, sort_keys=True), flush=True)
                    )
            except KeyboardInterrupt:
                run.stop("interrupted")
            print(json.dumps(run.writer.reader.progress(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
