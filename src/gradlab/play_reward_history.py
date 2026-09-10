"""Exact reward contributions from recorded transitions; never execute a policy."""

import json
import math
import sqlite3
from contextlib import closing
from pathlib import Path


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def reward_history(runner, episode_id, first=None, last=None):
    """First is the selected step; last is an optional forward page cursor."""
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
    first = end if first is None else first
    if not start <= first <= end:
        raise ValueError("selected step is outside the recording")
    gamma = metadata.get("discount")
    if not finite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("training discount is unavailable")
    with closing(sqlite3.connect(Path(recording.root) / "reward-history.sqlite")) as db, db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS points (step INTEGER PRIMARY KEY, payload TEXT NOT NULL)"
        )
        indexed = db.execute("SELECT MAX(step) FROM points").fetchone()[0]
        for step in range(start if indexed is None else indexed + 1, end + 1):
            row = recording.transition(step)
            payload = row.get("inspection_snapshot", {}).get("transition", row.get("presentation"))
            full = history_point_payload(payload)
            point = {
                key: full.get(key)
                for key in (
                    "reward_shaped",
                    "events",
                    "terminated",
                    "truncated",
                    "return_bootstrap",
                    "action_source",
                    "policy_sampled",
                    "action_selection_mode",
                )
            }
            db.execute("INSERT INTO points VALUES (?, ?)", (step, json.dumps(point)))
        total = raw_total = 0.0
        points = []
        next_last = None
        reasons = set()
        boundary = {}
        cursor = first if last is None else max(first, last)
        for step, payload in db.execute(
            "SELECT step, payload FROM points WHERE step >= ? ORDER BY step", (first,)
        ):
            point = json.loads(payload)
            boundary = point
            reward = point.get("reward_shaped")
            if not finite(reward):
                raise ValueError("recorded policy reward is unavailable or non-finite")
            weight = gamma ** (step - first)
            contribution = reward * weight
            total += contribution
            raw_total += reward
            reasons.update(point.get("value_comparison_reasons") or [])
            if point.get("action_source") not in (None, "policy"):
                reasons.add("recorded future contains non-policy actions")
            if point.get("policy_sampled") is False:
                reasons.add("recorded future contains non-stochastic policy actions")
            if reward != 0 and step >= cursor:
                if len(points) < 100:
                    points.append(
                        dict(
                            step=step,
                            offset=step - first,
                            reward=reward,
                            weight=weight,
                            contribution=contribution,
                            events=point.get("events", []),
                        )
                    )
                elif next_last is None:
                    next_last = step
        bootstrap = None
        complete = bool(boundary.get("terminated")) and not boundary.get("truncated")
        if boundary.get("truncated"):
            recorded = boundary.get("return_bootstrap") or {}
            if recorded.get("source") == "terminal_state_value" and finite(recorded.get("value")):
                value = recorded["value"]
                bootstrap = dict(value=value, contribution=gamma ** (end - first + 1) * value)
            else:
                reasons.add("truncation continuation is unavailable")
        initial = (
            metadata.get("initial_snapshot", {}).get("session", {}).get("critic_comparison", {})
        )
        reasons.update(initial.get("reasons") or [])
    return dict(
        episode_id=episode_id,
        first=first,
        end=end,
        discount=gamma,
        points=points,
        next_last=next_last,
        reward_sum=raw_total,
        discounted_reward_sum=total,
        complete=complete,
        bootstrap=bootstrap,
        return_total=total
        if complete
        else total + bootstrap["contribution"]
        if bootstrap
        else None,
        comparison_reasons=sorted(reasons),
    )
