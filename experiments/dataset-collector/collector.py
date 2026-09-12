"""Experimental, single-writer RGB trajectory collection. See the adjacent README."""

from __future__ import annotations

from collections import deque
from contextlib import AbstractContextManager
from copy import deepcopy
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time
import uuid
import zlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Reuse the data-only codec, not Player's recording, session, or UI machinery.
from gradlab.play_trajectory import pack_record, unpack_record
from gradlab.seeds import (
    EVAL_SEED_START,
    validate_eval_seed,
    validate_playback_seed,
    validate_training_seed,
)

VERSION = 1
MAX_BATCH_BYTES = 8 * 1024**2
MAX_IMAGE_BYTES = 4 * 1024**2
DISK_RESERVE = 2 * 1024**2  # SQLite rollback journal, index growth, and progress replacement.


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


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

    def _records(self, name):
        binding = self.db.execute("SELECT sha256, size FROM files WHERE name=?", (name,)).fetchone()
        if binding is None:
            raise ValueError("uncommitted record file")
        path = self._data(name)
        if binding["size"] > 4 * MAX_BATCH_BYTES or path.stat().st_size != binding["size"]:
            raise ValueError("record file integrity failure: size disagrees with bounded contract")
        data = path.read_bytes()
        if len(data) != binding["size"] or hashlib.sha256(data).hexdigest() != binding["sha256"]:
            raise ValueError(f"record file integrity failure: {name}")
        return [
            unpack_record(row["record"]) for row in pq.read_table(pa.BufferReader(data)).to_pylist()
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
        with self._data(row["shard"]).open("rb") as stream:
            stream.seek(row["offset"])
            compressed = stream.read(row["size"])
        decoder = zlib.decompressobj()
        raw = decoder.decompress(compressed, size + 1)
        if len(raw) != size or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError("invalid compressed RGB frame")
        image = np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()
        if image_hash(image) != row["sha256"]:
            raise ValueError("RGB hash mismatch")
        return image

    def progress(self):
        captures, transitions, complete, episodes, unique, raw, compressed = self.db.execute(
            "SELECT coalesce((SELECT value FROM counters WHERE name='captured_occurrences'),0), "
            "coalesce((SELECT value FROM counters WHERE name='transitions'),0), "
            "(SELECT count(*) FROM episodes WHERE status='complete'), "
            "(SELECT count(*) FROM episodes), count(*), coalesce(sum(raw_size),0), "
            "coalesce(sum(size),0) FROM frames"
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
        self.frames, self.rows = {}, []
        self.pending_bytes = 0
        self.delta_captures = 0
        self.episode_record = None
        self.failed = False
        try:
            manifest_path = self.root / "manifest.json"
            manifest = {
                "format_version": VERSION,
                "contract": contract,
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
                    shape TEXT, shard TEXT, offset INTEGER, size INTEGER, raw_size INTEGER);
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

    def session(self, provenance, settings):
        self.session_id = uuid.uuid4().hex
        data = pack_record(
            {"session_id": self.session_id, "provenance": provenance, "settings": settings}
        )
        name = f"session-{self.session_id}.parquet"
        self._commit_files(
            {name: self._parquet([data])},
            lambda: self.db.execute("INSERT INTO sessions VALUES(?,?)", (self.session_id, name)),
        )

    @staticmethod
    def _parquet(records):
        stream = pa.BufferOutputStream()
        table = pa.table({"record": pa.array(records, type=pa.binary())})
        pq.write_table(table, stream, compression="zstd")
        return stream.getvalue().to_pybytes()

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
                np.frombuffer(zlib.decompress(pending[1]), dtype=np.uint8).reshape(self.shape)
                if pending is not None
                else self.reader.frame(existing[0])
            )
            if not np.array_equal(previous, image):
                raise ValueError("RGB hash collision: matching hash has different bytes")
            return (pending[0] if pending is not None else existing[0]), False
        encoded = zlib.compress(image.tobytes())
        frame_id = self.next_frame
        self.next_frame += 1
        self.frames[digest] = (frame_id, encoded)
        self.pending_bytes += len(encoded)
        return frame_id, True

    def initial(self, image):
        frame_id, new = self._frame(image)
        self.episode_record.update(initial_frame_id=frame_id, initial_frame_new=new)
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
            "step": self.episode_record["length"],
            "source_frame_id": self.current_frame,
            # Upper-bound ID width and boolean size before staging this capture.
            "successor_frame_id": self.next_frame,
            "successor_frame_new": False,
        }
        if len(pack_record(row)) > MAX_BATCH_BYTES:
            raise ValueError("transition metadata exceeds bounded batch size")
        frame_id, new = self._frame(image)
        row.update(successor_frame_id=frame_id, successor_frame_new=new)
        encoded = pack_record(row)
        self.rows.append(encoded)
        self.pending_bytes += len(encoded)
        self.current_frame = frame_id
        self.delta_captures += 1
        self.episode_record["length"] += 1
        if len(self.rows) >= self.limits.batch_steps or self.pending_bytes >= MAX_BATCH_BYTES:
            self.flush()
        return row

    def end_episode(self, reason, *, complete):
        self.episode_record.update(
            status="complete" if complete else "incomplete", end_reason=reason
        )
        self.flush()
        self.episode_record = None

    def flush(self):
        if self.episode_record is None:
            return
        token = uuid.uuid4().hex
        shard_name, rows_name, episode_name = (
            f"{kind}-{token}.{ext}"
            for kind, ext in (("rgb", "bin"), ("steps", "parquet"), ("episode", "parquet"))
        )
        outputs = {episode_name: self._parquet([pack_record(self.episode_record)])}
        frame_records, chunks, offset = [], [], 0
        for digest, (frame_id, compressed) in self.frames.items():
            frame_records.append(
                (
                    frame_id,
                    digest,
                    json.dumps(self.shape),
                    shard_name,
                    offset,
                    len(compressed),
                    math.prod(self.shape),
                )
            )
            chunks.append(compressed)
            offset += len(compressed)
        if chunks:
            outputs[shard_name] = b"".join(chunks)
        if self.rows:
            outputs[rows_name] = self._parquet(self.rows)
        episode = self.episode_record

        def update():
            self.db.executemany("INSERT INTO frames VALUES(?,?,?,?,?,?,?)", frame_records)
            if self.rows:
                self.db.execute(
                    "INSERT INTO batches VALUES(?,?,?,?)",
                    (
                        rows_name,
                        episode["episode_id"],
                        episode["length"] - len(self.rows),
                        episode["length"] - 1,
                    ),
                )
            self.db.execute(
                "INSERT OR REPLACE INTO episodes VALUES(?,?,?,?,?)",
                (
                    episode["episode_id"],
                    episode["seed"],
                    episode["split"],
                    episode["status"],
                    episode_name,
                ),
            )
            for name, count in (
                ("captured_occurrences", self.delta_captures),
                ("transitions", len(self.rows)),
            ):
                self.db.execute(
                    "INSERT INTO counters VALUES(?,?) ON CONFLICT(name) DO UPDATE SET value=value+excluded.value",
                    (name, count),
                )

        self._commit_files(outputs, update)
        self.frames.clear()
        self.rows.clear()
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
        self.active = False
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
            pending_transitions=len(self.writer.rows),
            pending_captured_occurrences=self.writer.delta_captures,
            pending_buffer_bytes=self.writer.pending_bytes,
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
        if not self.active:
            episode = self.writer.reserve_episode()
            self.active = True
            self.schedule_rng = np.random.default_rng(
                np.random.SeedSequence([self.schedule.seed, episode["episode_id"]])
            )
            self.writer.initial(self.execution.reset(episode["seed"]))
        decision_id = self.writer.episode_record["length"]
        if decision_id % self.schedule.block_decisions == 0:
            self.temperature = (
                float(self.schedule_rng.choice(self.schedule.values, p=self.schedule.probabilities))
                if self.schedule.enabled
                else 1.0
            )
        image, facts = self.execution.step(self.temperature)
        self.latest_source = owned_rgb(image)
        facts = {
            **facts,
            "policy_decision_id": self.writer.episode_record["length"],
            "temperature": self.temperature,
            "action_selection_mode": self.execution.action_selection_mode,
            "configured_frame_skip": self.execution.contract["frame_skip"],
        }
        self.latest_row = self.writer.append(image, facts)
        self.steps += 1
        if facts["terminated"] or facts["truncated"]:
            self.writer.end_episode("environment_boundary", complete=True)
            self.active = False
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
        if self.active:
            self.writer.end_episode(reason, complete=False)
            self.active = False
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
        reader.db.execute("CREATE TEMP TABLE seen(frame_id INTEGER PRIMARY KEY)")
        reader.db.execute("BEGIN")  # One committed dataset prefix, even with a concurrent writer.
        transitions, occurrences, unique = 0, 0, 0
        shape = None

        def capture(frame_id, claimed_new):
            nonlocal occurrences, unique, shape
            first = (
                reader.db.execute("INSERT OR IGNORE INTO seen VALUES(?)", (frame_id,)).rowcount == 1
            )
            if first:
                image = reader.frame(frame_id)
                if shape is not None and shape != image.shape:
                    raise ValueError("dataset image dimensions disagree")
                shape = image.shape
                unique += 1
            if type(claimed_new) is not bool or first != claimed_new:
                raise ValueError("image discovery attribution mismatch")
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

    def reset(self, seed):
        import torch

        torch.manual_seed(seed)
        np.random.seed(seed)
        if bool(getattr(self.runtime.model, "use_sde", False)):
            self.runtime.model.policy.reset_noise()
        self.env.seed(seed)
        obs = self.env.reset()
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
        obs, rewards, dones, infos = self.env.step(decision.actions)
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
            "selected_action": deepcopy(decision.decisions[0].raw_action),
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
                    ("episode_id", "step", "policy_decision_id"),
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
    checkpoint, *, full_game=False, episode_steps=None, device="cpu", schedule=TemperatureSchedule()
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
        env = make_eval_vec_env(config=config, n_envs=1, seed=0, capture_step_diagnostics=True)
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
            return PolicyExecution(runtime, env, config, contract=contract, provenance=provenance)
        except BaseException:
            env.close()
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


def export_huggingface(root, destination, *, shard_rows=4096, max_bytes=10 * 1024**3):
    """Export a quiescent, validated dataset as standalone Hugging Face Parquet tables."""
    import base64
    import io
    import tempfile
    import shutil
    import yaml
    from PIL import Image
    from gradlab.play_trajectory import encode_tree, portable_metadata

    root, destination = Path(root).resolve(), Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("export destination already exists")
    if destination.resolve().is_relative_to(root):
        raise ValueError("export destination cannot be inside the source dataset")
    if type(shard_rows) is not int or not 1 <= shard_rows <= 1000000:
        raise ValueError("shard_rows must be between 1 and 1000000")
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("export byte budget must be positive")

    def readable(value):
        if isinstance(value, np.ndarray | np.generic):
            return value.tolist()
        raise TypeError(f"unsupported JSON value: {type(value).__name__}")

    def exact_record(value):
        tree = encode_tree(value)
        for leaf in tree["arrays"]:
            leaf["data"] = base64.b64encode(leaf["data"]).decode("ascii")
        return canonical(tree).decode()

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

    with (root / ".writer.lock").open("rb") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("stop the dataset writer before exporting") from error
        summary = validate_dataset(root)
        if not summary["transitions"]:
            raise ValueError("export requires at least one committed transition")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
        try:

            def budget():
                if disk_bytes(staging) > max_bytes:
                    raise ValueError("export exceeds byte budget")

            def write_table(config, split, rows, arrow_schema):
                folder = staging / config / split
                folder.mkdir(parents=True)
                writer, pending, pending_bytes = None, [], 0
                count, in_shard, shard = 0, 0, 0

                def flush():
                    nonlocal writer, pending, pending_bytes, in_shard, shard
                    if not pending:
                        return
                    if writer is None:
                        writer = pq.ParquetWriter(
                            folder / f"{shard:05d}.parquet",
                            arrow_schema,
                            compression="zstd",
                            write_page_index=True,
                        )
                    writer.write_table(
                        pa.Table.from_pylist(pending, schema=arrow_schema),
                        row_group_size=len(pending),
                    )
                    in_shard += len(pending)
                    pending, pending_bytes = [], 0
                    if in_shard == shard_rows:
                        writer.close()
                        writer, in_shard, shard = None, 0, shard + 1
                    budget()

                try:
                    for row in rows:
                        size = pa.Table.from_pylist([row], schema=arrow_schema).nbytes
                        if size > 4 * MAX_BATCH_BYTES:
                            raise ValueError("export record exceeds bounded row size")
                        if pending and pending_bytes + size > MAX_BATCH_BYTES:
                            flush()
                        pending.append(row)
                        pending_bytes += size
                        count += 1
                        if len(pending) >= min(128, shard_rows - in_shard):
                            flush()
                    flush()
                finally:
                    if writer is not None:
                        writer.close()
                budget()
                return count

            with DatasetReader(root) as reader:
                reader.db.execute("BEGIN")

                def frames():
                    for frame in reader.db.execute(
                        "SELECT frame_id, sha256 FROM frames ORDER BY frame_id"
                    ):
                        buffer = io.BytesIO()
                        Image.fromarray(reader.frame(frame["frame_id"])).save(buffer, format="PNG")
                        yield {
                            "frame_id": frame["frame_id"],
                            "sha256": frame["sha256"],
                            "image": {"bytes": buffer.getvalue(), "path": None},
                        }

                def transitions(split):
                    for episode in reader.episodes():
                        if episode["split"] != split:
                            continue
                        for row in reader.transitions(episode["episode_id"]):
                            result = {key: row.get(key) for key in transition_schema.names}
                            result["session_id"] = episode["session_id"]
                            for key in ("selected_action", "effective_action", "native_action"):
                                result[f"{key}_json"] = json.dumps(
                                    row.get(key),
                                    default=readable,
                                    allow_nan=False,
                                    separators=(",", ":"),
                                )
                            result["record_json"] = exact_record(row)
                            yield result

                configs = []
                write_table("frames", "assets", frames(), frame_schema)
                configs.append(
                    {
                        "config_name": "frames",
                        "data_files": [{"split": "assets", "path": "frames/assets/*.parquet"}],
                    }
                )
                for name, arrow_schema in [
                    ("transitions", transition_schema),
                    ("episodes", episode_schema),
                ]:
                    files = []
                    for split in ("train", "heldout"):
                        rows = (
                            transitions(split)
                            if name == "transitions"
                            else (e for e in reader.episodes() if e["split"] == split)
                        )
                        if write_table(name, split, rows, arrow_schema):
                            files.append({"split": split, "path": f"{name}/{split}/*.parquet"})
                    configs.append(
                        {
                            "config_name": name,
                            "data_files": files,
                            **({"default": True} if name == "transitions" else {}),
                        }
                    )
                sessions = (
                    {
                        "session_id": row[0],
                        "record_json": exact_record(portable_metadata(reader.session(row[0]))),
                    }
                    for row in reader.db.execute(
                        "SELECT session_id FROM sessions ORDER BY session_id"
                    )
                )
                write_table("sessions", "metadata", sessions, session_schema)
                configs.append(
                    {
                        "config_name": "sessions",
                        "data_files": [
                            {"split": "metadata", "path": "sessions/metadata/*.parquet"}
                        ],
                    }
                )
                (staging / "manifest.json").write_bytes(
                    canonical(portable_metadata(reader.manifest))
                )
                report = {
                    "export_version": 1,
                    "source_format_version": VERSION,
                    "source_manifest_sha256": hashlib.sha256(
                        canonical(reader.manifest)
                    ).hexdigest(),
                    "transitions": summary["transitions"],
                    "unique_frames": summary["unique_frames"],
                    "complete_episodes": summary["complete_episodes"],
                    "incomplete_episodes": summary["incomplete_episodes"],
                }
                card = {
                    "pretty_name": "GradLab RGB trajectories",
                    "tags": ["reinforcement-learning", "image"],
                    "configs": configs,
                }
                (staging / "README.md").write_text(
                    "---\n"
                    + yaml.safe_dump(card, sort_keys=False)
                    + """---
# GradLab RGB trajectories

A snapshot of recorded Policy execution for trajectory and neural-emulator research.
Full RGB, including HUD pixels, is preserved as lossless PNG. Each image is stored
once in `frames`; ordered `transitions` reference source/successor frame IDs.
`episodes` includes initial frames, seeds, completion status and session IDs.
`sessions` contains collection settings and portable checkpoint/runtime provenance.
`manifest.json` records the collection contract. No checkpoint is needed to read it.

## Loading

```python
from datasets import load_dataset
repo = "YOUR_NAMESPACE/YOUR_DATASET"  # Or this local export directory.
steps = load_dataset(repo, "transitions", split="train")
frames = load_dataset(repo, "frames", split="assets")
episodes = load_dataset(repo, "episodes", split="train")
# Join source_frame_id / successor_frame_id to frames.frame_id.
# Frame IDs are identifiers, not zero-based row offsets.
```

Each configuration is a separate table in the Hugging Face Dataset Viewer.
The `frames` table previews images; the transition table shows IDs and numerical
facts. The Hub does not automatically render frame-ID joins as image columns.
Use `(episode_id, step)` for ordering and `session_id` for provenance.
Action JSON columns preserve scalar/vector shape. Nullable fields mean unavailable.
`record_json` retains every original fact and exact NumPy dtype/shape: parse its
JSON, parse `structure` for the tagged tree, and base64-decode each `arrays[].data`
using the declared NumPy dtype/shape. Array references index that array list.
The tree tags are scalar, array, dict (key/value pairs), list, and tuple.

## Splits and limitations

Episode assignments are preserved as `train` and `heldout`; absent splits are
omitted. `frames/assets` is a shared image pool, NOT an independent training split.
Only join images referenced by your chosen episode split; sharing image bytes does
not permit fitting on held-out trajectories, labels, or histories. Incomplete
prefixes remain marked in `episodes` and must not be treated as game-over events.
Boundaries and temperature changes are recorded in session settings; Counterfactual
Playback remains ineligible as Acceptance or Promotion evidence. Capture cadence
comes from the manifest, and omitted native intermediate frames are unavailable.
Image uniqueness does not measure hidden simulator-state coverage.

This exporter does not assign a dataset license. Set the applicable license and
source attribution in this card before publication. `export.json` identifies the
snapshot and hashes its files; this is collection evidence, not model performance.
"""
                )
            report["files"] = []
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    with path.open("rb") as stream:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                    report["files"].append(
                        {
                            "path": path.relative_to(staging).as_posix(),
                            "sha256": digest,
                            "bytes": path.stat().st_size,
                        }
                    )
            (staging / "export.json").write_bytes(canonical(report))
            budget()
            # Destination must still be absent; preserve any concurrently created directory.
            if destination.exists() or destination.is_symlink():
                raise ValueError("export destination already exists")
            os.rename(staging, destination)
            return report
        finally:
            if staging.exists():
                shutil.rmtree(staging)


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
    collect.add_argument("--debug", action="store_true", help="start the local debugger paused")
    for name in ("validate", "inspect", "progress", "preview", "export-hf"):
        mode = modes.add_parser(name)
        mode.add_argument("dataset", type=Path)
        if name == "export-hf":
            mode.add_argument("output", type=Path)
            mode.add_argument("--shard-rows", type=int, default=4096)
            mode.add_argument("--max-gib", type=float, default=10)
        if name in {"inspect", "preview"}:
            mode.add_argument("--episode", type=int, default=1)
        if name == "preview":
            mode.add_argument("output", type=Path)
            mode.add_argument("--max-steps", type=int, default=600)
            mode.add_argument("--fps", type=float, default=30)
    args = parser.parse_args(argv)
    if args.mode == "export-hf":
        if not math.isfinite(args.max_gib) or args.max_gib <= 0:
            parser.error("--max-gib must be positive and finite")
        print(
            json.dumps(
                export_huggingface(
                    args.dataset,
                    args.output,
                    shard_rows=args.shard_rows,
                    max_bytes=int(args.max_gib * 1024**3),
                ),
                sort_keys=True,
            )
        )
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
