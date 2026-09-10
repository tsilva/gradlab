"""Bounded chart views over a disk-backed episode diagnostic index."""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any


def scalars(value: Any, prefix: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from scalars(child, f"{prefix}/{key}")
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        yield prefix, value


def chart_history(runner, episode_id: str, first: int | None, last: int | None):
    from gradlab.play_web import history_point_payload

    recording = runner.recording
    if recording is None or recording.metadata["episode_id"] != episode_id:
        raise ValueError("the recorded episode has been replaced")
    start = int(recording.metadata["first_step"])
    end = (
        recording.status()["last_step"]
        if hasattr(recording, "status")
        else start + int(recording.metadata["transition_count"]) - 1
    )
    first = start if first is None else max(start, first)
    last = end if last is None else min(end, last)
    if first > last:
        raise ValueError("empty chart range")
    cache_key = (
        episode_id,
        first,
        last,
        end,
        len(runner.history),
        runner.history[-1].get("realized_return") if runner.history else None,
    )
    cached = getattr(runner, "_chart_history_cache", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1]
    # This disposable index lives beside the recording and is never exported.
    with closing(sqlite3.connect(Path(recording.root) / "chart-history.sqlite")) as db, db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS points (step INTEGER PRIMARY KEY, payload TEXT NOT NULL)"
        )
        indexed = db.execute("SELECT MAX(step) FROM points").fetchone()[0]
        for step in range(start if indexed is None else indexed + 1, end + 1):
            row = recording.transition(step)
            point = history_point_payload(
                row["inspection_snapshot"]["transition"]
                if "inspection_snapshot" in row
                else row["presentation"]
            )
            db.execute("INSERT INTO points VALUES (?, ?)", (step, json.dumps(point)))
        # Merge only calibration annotations; filtered live payloads must not
        # erase the original recorded signals or rewards in the chart index.
        for point in runner.history:
            if int(point.get("episode", -1)) != int(recording.metadata["episode"]):
                continue
            annotations = {
                key: point[key]
                for key in (
                    "realized_return",
                    "realized_return_bootstrapped",
                    "value_error",
                    "value_comparison_reasons",
                )
                if key in point
            }
            if not annotations:
                continue
            row = db.execute(
                "SELECT payload FROM points WHERE step = ?", (point["step"],)
            ).fetchone()
            if row is not None:
                stored = json.loads(row[0])
                stored.update(annotations)
                db.execute(
                    "UPDATE points SET payload = ? WHERE step = ?",
                    (json.dumps(stored), point["step"]),
                )
        points = []
        bucket = None
        retained = {}
        extrema = {}

        def flush():
            chosen = {point["step"]: point for point in retained.values()}
            for _, point in extrema.values():
                chosen[point["step"]] = point
            points.extend(chosen[key] for key in sorted(chosen))

        # Each bucket retains endpoints and extrema of each scalar signal.
        # Memory is bounded by bucket count and the diagnostic schema, not episode length.
        for step, payload in db.execute(
            "SELECT step, payload FROM points WHERE step BETWEEN ? AND ? ORDER BY step",
            (first, last),
        ):
            group = (step - first) * 64 // max(1, last - first + 1)
            if group != bucket:
                flush()
                retained, extrema, bucket = {}, {}, group
            point = json.loads(payload)
            retained.setdefault("first", point)
            retained["last"] = point
            for key, value in scalars(point):
                for kind, better in (("min", lambda a, b: a < b), ("max", lambda a, b: a > b)):
                    identity = (key, kind)
                    if identity not in extrema or better(value, extrema[identity][0]):
                        extrema[identity] = (value, point)
        flush()
    result = {
        "episode_id": episode_id,
        "first": first,
        "last": last,
        "episode_first": start,
        "episode_last": end,
        "points": points,
    }
    runner._chart_history_cache = (cache_key, result)
    return result
