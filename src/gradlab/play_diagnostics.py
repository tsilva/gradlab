"""Bounded diagnostic computation in a process independent of Policy execution."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol
import multiprocessing
from pathlib import Path
import threading
from types import SimpleNamespace

from gradlab.play_trajectory import read_record, unpack_record


class RecordedPrefix:
    def __init__(self, descriptor):
        self.root = Path(descriptor["root"])
        self.metadata = descriptor["metadata"]
        self.written = descriptor["written"]
        self.pending = descriptor["pending"]

    def transition(self, step):
        index = step - self.metadata["first_step"]
        if not 0 <= index < self.metadata["transition_count"]:
            raise ValueError("step is outside the recorded prefix")
        return (
            read_record(self.root, index)
            if index < self.written
            else unpack_record(self.pending[index - self.written])
        )


# One process serves one runner; retain at most two episode/range results.
_READERS = OrderedDict()


def query_prefix(descriptor, history, kind, first, last):
    from gradlab.play_chart_history import chart_history
    from gradlab.play_event_history import event_history
    from gradlab.play_reward_history import reward_history

    key = (descriptor["root"], kind, first, last)
    reader = _READERS.pop(key, SimpleNamespace())
    reader.recording = RecordedPrefix(descriptor)
    reader.history = history
    try:
        result = {"chart": chart_history, "event": event_history, "reward": reward_history}[kind](
            reader, descriptor["metadata"]["episode_id"], first, last
        )
        _READERS[key] = reader
        while len(_READERS) > 2:
            _READERS.popitem(last=False)
        return result
    finally:
        # Cached summaries contain no images or pending recording buffers.
        reader.recording = None
        reader.history = []


class DiagnosticQueries:
    def __init__(self):
        self._lock = threading.Lock()
        self._pool = None
        self._closed = False
        self._slots = threading.BoundedSemaphore(4)

    def query(self, descriptor, history, kind, first, last):
        if not self._slots.acquire(blocking=False):
            raise ValueError("too many pending diagnostic reads")
        try:
            with self._lock:
                if self._closed:
                    raise ValueError("the Playback Session has been replaced")
                if self._pool is None:
                    self._pool = ProcessPoolExecutor(
                        max_workers=1, mp_context=multiprocessing.get_context("spawn")
                    )
                future = self._pool.submit(query_prefix, descriptor, history, kind, first, last)
            return future.result()
        finally:
            self._slots.release()

    def close(self):
        with self._lock:
            self._closed = True
            pool = self._pool
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)



class DiagnosticKind(StrEnum):
    CHART = "chart"
    REWARD = "reward"
    EVENT = "event"


@dataclass(frozen=True)
class DiagnosticRead:
    """Keep each calculation's query semantics, rather than normalizing a range."""

    kind: DiagnosticKind
    episode_id: str
    first: int | None = None
    last: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "kind", DiagnosticKind(self.kind))


class RecordingRead:
    """An owned pin. Callbacks and locks never cross the calculation process seam."""

    def __init__(self, descriptor, release: Callable[[], None], validate: Callable[[], None]):
        self.descriptor = descriptor
        self.history = []
        self._release = release
        self._validate = validate

    def validate(self):
        self._validate()

    def close(self):
        release, self._release = self._release, None
        if release is not None:
            release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def calibration_annotations(history):
    from gradlab.play_chart_history import ANNOTATIONS

    return [
        deepcopy({
            "step": point["step"],
            "episode": point["episode"],
            **{key: point[key] for key in ANNOTATIONS if key in point},
        })
        for point in list(history)
        if any(key in point for key in ANNOTATIONS)
    ]


class RecordingSource(Protocol):
    def reserve(self, episode_id: str): ...


class LiveRecordingSource:
    """Capture the live recording and amendments under its short identity lock."""

    def __init__(self, lock, recording: Callable, history: Callable):
        self._lock = lock
        self._recording = recording
        self._history = history

    @contextmanager
    def reserve(self, episode_id: str) -> Iterator[RecordingRead]:
        owned = None
        try:
            with self._lock:
                recording = self._recording()
                if recording is None or recording.metadata["episode_id"] != episode_id:
                    raise ValueError("the recorded episode has been replaced")

                def validate():
                    with self._lock:
                        if self._recording() is not recording:
                            raise ValueError("the recorded episode has been replaced")

                owned = RecordingRead(recording.reserve_read(), recording.release_read, validate)
                owned.history = calibration_annotations(self._history())
            yield owned
        finally:
            if owned is not None:
                owned.close()


class ImportedRecordingSource:
    """An imported recording is fixed for the runner's lifetime and stays data-only."""

    def __init__(self, lock, recording, history: Callable):
        self._lock = lock
        self._recording = recording
        self._history = history

    @contextmanager
    def reserve(self, episode_id: str) -> Iterator[RecordingRead]:
        owned = None
        try:
            with self._lock:
                recording = self._recording
                if recording.metadata["episode_id"] != episode_id:
                    raise ValueError("the recorded episode has been replaced")
                # reserve_read checks retirement; an admitted read may finish while
                # shutdown drains. Session validity remains the host's authority.
                owned = RecordingRead(recording.reserve_read(), recording.release_read, lambda: None)
                owned.history = calibration_annotations(self._history())
            yield owned
        finally:
            if owned is not None:
                owned.close()


class DiagnosticReads:
    """Own reservation, admission, isolated calculation, and final validity together."""

    def __init__(self, source: RecordingSource):
        self._source = source
        self._queries = DiagnosticQueries()

    def read(self, request: DiagnosticRead) -> dict[str, Any]:
        with self._source.reserve(request.episode_id) as owned:
            result = self._queries.query(
                owned.descriptor, owned.history, request.kind, request.first, request.last
            )
            owned.validate()
            return result

    def close(self):
        self._queries.close()


class DiagnosticReader(Protocol):
    def read_diagnostics(self, epoch: int, request: DiagnosticRead) -> dict[str, Any]: ...


class DirectDiagnosticReader:
    """Bind an embedded runner to the same epoch-aware interface as a host."""

    def __init__(self, diagnostics: DiagnosticReads | None, epoch: Callable[[], int]):
        self._diagnostics = diagnostics
        self._epoch = epoch

    def read_diagnostics(self, epoch: int, request: DiagnosticRead) -> dict[str, Any]:
        if epoch != self._epoch():
            raise ValueError("the Playback Session has been replaced")
        if self._diagnostics is None:
            raise ValueError(f"episode {request.kind} history is unavailable")
        result = self._diagnostics.read(request)
        if epoch != self._epoch():
            raise ValueError("the Playback Session has been replaced")
        return result
