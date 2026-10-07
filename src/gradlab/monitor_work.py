"""Attempt-local episode dispatch; R2 remains the scientific completion authority."""

from contextlib import contextmanager
import fcntl
import json
from pathlib import Path

from gradlab.file_utils import atomic_write_json


class EpisodeWork:
    def __init__(self, root, count):
        self.root = Path(root) / "episodes"
        self.root.mkdir(parents=True, exist_ok=True)
        self.count = count

    @contextmanager
    def locked(self):
        with (self.root / "dispatch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def remaining(self):
        path = self.root / "next.json"
        return self.count - (json.loads(path.read_text()) if path.exists() else 0)

    def claim(self):
        with self.locked():
            ordinal = self.count - self.remaining()
            if ordinal == self.count:
                return None
            atomic_write_json(self.root / "next.json", ordinal + 1)
            return ordinal

    def complete(self, ordinal, result):
        # Called only after the episode's canonical manifest and chunks are verified.
        atomic_write_json(self.root / f"{ordinal}.json", result)

    def results(self):
        paths = [self.root / f"{i}.json" for i in range(self.count)]
        if not all(path.exists() for path in paths):
            return None
        return [json.loads(path.read_text()) for path in paths]
