"""Bounded diagnostic computation in a process independent of Policy execution."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from collections import OrderedDict
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


def read_diagnostics(runner, episode_id, kind, first=None, last=None):
    with runner._diagnostic_lock:
        recording = runner.recording
        if recording is None or recording.metadata["episode_id"] != episode_id:
            raise ValueError("the recorded episode has been replaced")
        descriptor = recording.reserve_read()
        # Only calibration amendments are needed in addition to recorded facts.
        from gradlab.play_chart_history import ANNOTATIONS

        history = [
            {
                "step": p["step"],
                "episode": p["episode"],
                **{key: p[key] for key in ANNOTATIONS if key in p},
            }
            for p in list(runner.history)
            if any(key in p for key in ANNOTATIONS)
        ]
    try:
        result = runner._diagnostics.query(descriptor, history, kind, first, last)
        with runner._diagnostic_lock:
            if runner.recording is not recording:
                raise ValueError("the recorded episode has been replaced")
        return result
    finally:
        recording.release_read()
