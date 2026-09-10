"""Incremental, multiresolution chart views of recorded episode evidence."""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

ANNOTATIONS = (
    "realized_return",
    "realized_return_bootstrapped",
    "value_error",
    "value_comparison_reasons",
)


def scalars(value: Any, prefix: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from scalars(child, f"{prefix}/{key}")
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        yield prefix, value


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def comparable(point):
    return (
        not any(
            point.get(key)
            for key in ("boundary", "terminated", "truncated", "value_comparison_reasons")
        )
        and point.get("action_source") in (None, "policy")
        and point.get("policy_sampled") is not False
    )


def leaf(point, gamma):
    step = point["step"]
    return {
        "first": step,
        "last": step,
        "extrema": {key: [value, step, value, step] for key, value in scalars(point)},
        "reward": point.get("reward_shaped") if finite(point.get("reward_shaped")) else 0,
        "discount": gamma,
        "valid": comparable(point) and finite(point.get("reward_shaped")),
    }


def merge(left, right):
    extrema = {key: list(value) for key, value in left["extrema"].items()}
    for key, value in right["extrema"].items():
        if key not in extrema:
            extrema[key] = list(value)
        else:
            current = extrema[key]
            if value[0] < current[0]:
                current[:2] = value[:2]
            if value[2] > current[2]:
                current[2:] = value[2:]
    return dict(
        first=left["first"],
        last=right["last"],
        extrema=extrema,
        reward=left["reward"] + left["discount"] * right["reward"],
        discount=left["discount"] * right["discount"],
        valid=left["valid"] and right["valid"],
    )


def blocks(first, last, origin, maximum=None):
    """Disjoint aligned power-of-two blocks, including exact range edges."""
    while first <= last:
        offset = first - origin
        size = (offset & -offset) if offset else 1 << (last - first + 1).bit_length() - 1
        size = min(size, 1 << (last - first + 1).bit_length() - 1)
        if maximum is not None:
            size = min(size, maximum)
        yield first, size.bit_length() - 1
        first += size


def chart_history(runner, episode_id: str, first: int | None, last: int | None):
    from gradlab.play_web import history_point_payload

    recording = runner.recording
    if recording is None or recording.metadata["episode_id"] != episode_id:
        raise ValueError("the recorded episode has been replaced")
    metadata = recording.metadata
    start = int(metadata["first_step"])
    end = (
        recording.status()["last_step"]
        if hasattr(recording, "status")
        else start + int(metadata["transition_count"]) - 1
    )
    first = start if first is None else max(start, first)
    last = end if last is None else min(end, last)
    if first > last:
        raise ValueError("empty chart range")
    gamma = metadata.get("discount")
    has_discount = finite(gamma) and 0 <= gamma <= 1
    gamma = gamma if has_discount else 0.0
    annotations = {
        p["step"]: {key: p[key] for key in ANNOTATIONS if key in p}
        for p in runner.history
        if int(p.get("episode", -1)) == metadata["episode"]
        and first <= p["step"] <= end
        and any(key in p for key in ANNOTATIONS)
    }
    cache_key = (episode_id, first, last, end, annotations)
    cached = getattr(runner, "_chart_history_cache", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1]
    with closing(sqlite3.connect(Path(recording.root) / "chart-index-v2.sqlite")) as db, db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS points (step INTEGER PRIMARY KEY, payload TEXT, annotation TEXT)"
        )
        db.execute(
            "CREATE TABLE IF NOT EXISTS nodes (step INTEGER, level INTEGER, payload TEXT, PRIMARY KEY(step, level))"
        )
        indexed = db.execute("SELECT MAX(step) FROM points").fetchone()[0]
        nodes = {}

        def get(step, level):
            key = (step, level)
            if key not in nodes:
                nodes[key] = json.loads(
                    db.execute(
                        "SELECT payload FROM nodes WHERE step=? AND level=?", key
                    ).fetchone()[0]
                )
            return nodes[key]

        def put(step, level, node):
            nodes[step, level] = node
            db.execute(
                "INSERT OR REPLACE INTO nodes VALUES (?, ?, ?)",
                (step, level, json.dumps(node, separators=(",", ":"))),
            )

        def ancestors(step):
            offset = step - start
            level = 1
            while True:
                size = 1 << level
                base = start + offset // size * size
                if base + size - 1 > end:
                    break
                yield base, level
                level += 1

        changed = set()
        for step in range(start if indexed is None else indexed + 1, end + 1):
            row = recording.transition(step)
            point = history_point_payload(
                row["inspection_snapshot"]["transition"]
                if "inspection_snapshot" in row
                else row["presentation"]
            )
            db.execute(
                "INSERT INTO points VALUES (?, ?, NULL)",
                (step, json.dumps(point, separators=(",", ":"))),
            )
            put(step, 0, leaf(point, gamma))
            level = 1
            while (step - start + 1) % (1 << level) == 0:
                base = step - (1 << level) + 1
                put(
                    base,
                    level,
                    merge(get(base, level - 1), get(base + (1 << (level - 1)), level - 1)),
                )
                level += 1
            if step % 256 == 0:
                nodes.clear()
        for step, annotation in annotations.items():
            row = db.execute(
                "SELECT payload, annotation FROM points WHERE step=?", (step,)
            ).fetchone()
            rendered = json.dumps(annotation, sort_keys=True)
            if row is not None and row[1] != rendered:
                point = json.loads(row[0])
                point.update(annotation)
                db.execute(
                    "UPDATE points SET payload=?, annotation=? WHERE step=?",
                    (json.dumps(point, separators=(",", ":")), rendered, step),
                )
                put(step, 0, leaf(point, gamma))
                changed.update(ancestors(step))
        for step, level in sorted(changed, key=lambda key: key[1]):
            put(step, level, merge(get(step, level - 1), get(step + (1 << (level - 1)), level - 1)))
        # Bound the temporary working set after indexing a large imported prefix.
        nodes.clear()
        maximum = 1 << max(0, ((last - first + 1) // 64).bit_length() - 1)
        chosen = set()
        for step, level in blocks(first, last, start, maximum):
            node = get(step, level)
            chosen.update((node["first"], node["last"]))
            for value in node["extrema"].values():
                chosen.update((value[1], value[3]))
        points = [
            json.loads(db.execute("SELECT payload FROM points WHERE step=?", (step,)).fetchone()[0])
            for step in sorted(chosen)
        ]
        initial = metadata.get("initial_snapshot", {}).get("session", {})
        if has_discount and not initial.get("critic_comparison", {}).get("reasons"):
            tail = json.loads(
                db.execute("SELECT payload FROM points WHERE step=?", (end,)).fetchone()[0]
            )
            estimate = tail.get("value")
            if finite(estimate) and comparable(tail):
                cursor = end
                for point in reversed(points):
                    valid = True
                    for step, level in reversed(list(blocks(point["step"], cursor - 1, start))):
                        node = get(step, level)
                        if not node["valid"]:
                            valid = False
                            break
                        estimate = node["reward"] + node["discount"] * estimate
                    if not valid:
                        break
                    point.update(estimated_return=estimate, return_estimate_step=end)
                    cursor = point["step"]
    result = dict(
        episode_id=episode_id,
        first=first,
        last=last,
        episode_first=start,
        episode_last=end,
        points=points,
    )
    runner._chart_history_cache = (cache_key, result)
    return result
