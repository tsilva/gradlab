"""Compact, revisioned chart responses shared by viewers of the same server."""

from collections import OrderedDict
import hashlib
import json
import threading

ABSENT = {"$absent": True}


def flatten(value, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten(child, (*path, key))
    else:
        yield path, value


class ChartResponses:
    def __init__(self):
        self._lock = threading.Lock()
        self._versions = OrderedDict()

    def encode(self, result, base=None):
        points = result["points"]
        flat = [dict(flatten(point)) for point in points]
        paths = sorted({path for point in flat for path in point})
        constants = []
        fields = []
        for path in paths:
            values = [point.get(path, ABSENT) for point in flat]
            if values and all(value == values[0] for value in values):
                constants.append([path, values[0]])
            else:
                fields.append(path)
        rows = {
            point["step"]: [row.get(path, ABSENT) for path in fields]
            for point, row in zip(points, flat)
        }
        metadata = {key: value for key, value in result.items() if key != "points"}
        document = dict(
            format="chart-columns-v1",
            **metadata,
            fields=fields,
            constants=constants,
            rows=[[step, values] for step, values in rows.items()],
        )
        rendered = json.dumps(document, separators=(",", ":"), allow_nan=False)
        revision = hashlib.sha256(rendered.encode()).hexdigest()[:24]
        with self._lock:
            previous = self._versions.get(base)
            self._versions[revision] = (fields, rows, metadata["episode_id"])
            self._versions.move_to_end(revision)
            while len(self._versions) > 8:
                self._versions.popitem(last=False)
        document["revision"] = revision
        if previous and previous[0] == fields and previous[2] == metadata["episode_id"]:
            old = previous[1]
            document.update(
                base=base,
                removed=[step for step in old if step not in rows],
                rows=[
                    [step, values]
                    for step, values in rows.items()
                    if step not in old or values != old[step]
                ],
            )
        else:
            document.update(base=None, removed=[])
        return document
