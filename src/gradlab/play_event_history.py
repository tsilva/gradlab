"""Exact, paginated events from the recorded episode, independent of inspection."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path


def event_history(runner, episode_id, first=None, last=None):
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
    with closing(sqlite3.connect(Path(recording.root) / "event-history.sqlite")) as db, db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS events (step INTEGER PRIMARY KEY, payload TEXT NOT NULL)"
        )
        db.execute("CREATE TABLE IF NOT EXISTS progress (step INTEGER NOT NULL)")
        progress = db.execute("SELECT step FROM progress").fetchone()
        for step in range(start if progress is None else progress[0] + 1, end + 1):
            row = recording.transition(step)
            point = history_point_payload(
                row["inspection_snapshot"]["transition"]
                if "inspection_snapshot" in row
                else row["presentation"]
            )
            if point.get("boundary") or point.get("events"):
                point = {
                    key: point[key] for key in ("step", "episode", "sequence", "boundary", "events")
                }
                db.execute("INSERT INTO events VALUES (?, ?)", (step, json.dumps(point)))
        db.execute("DELETE FROM progress")
        db.execute("INSERT INTO progress VALUES (?)", (end,))
        rows = db.execute(
            "SELECT payload FROM events WHERE step >= ? AND step <= ? ORDER BY step DESC LIMIT 101",
            (start if first is None else first, end if last is None else min(last, end)),
        ).fetchall()
    points = [json.loads(row[0]) for row in rows[:100]]
    return {
        "episode_id": episode_id,
        "points": points,
        "next_last": points[-1]["step"] - 1 if len(rows) > 100 else None,
    }
