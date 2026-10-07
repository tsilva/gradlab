"""Run-owned checkpoint cadence, consumed only at trainer safe boundaries."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def normalize_checkpoint_config(config: dict[str, Any], *, metric_validator=None) -> None:
    steps = config.get("checkpoint_steps", [])
    if not isinstance(steps, list | tuple) or any(type(s) is not int or s <= 0 for s in steps):
        raise ValueError("checkpoint_steps must be a list of positive integers")
    if list(steps) != sorted(set(steps)):
        raise ValueError("checkpoint_steps must be strictly increasing")
    if "checkpoint_steps" in config:
        config["checkpoint_steps"] = list(steps)
    candidate = config.get("checkpoint_candidates")
    if candidate is None:
        return
    defaults = dict(
        direction="maximize",
        min_delta=0.05,
        delta_mode="relative",
        start_after_steps=0,
        min_interval_steps=1,
        max_checkpoints=3,
    )
    if not isinstance(candidate, Mapping) or set(candidate) - {*defaults, "metric"}:
        raise ValueError("checkpoint_candidates contains unknown settings")
    candidate = {**defaults, **candidate}
    metric = candidate.get("metric")
    if not isinstance(metric, str) or not metric.startswith("train/"):
        raise ValueError("checkpoint_candidates.metric must be a training metric")
    if metric_validator is None:
        from gradlab.metric_names import validate_metric_name

        metric_validator = validate_metric_name
    metric_validator(metric)
    if candidate["direction"] not in {"maximize", "minimize"}:
        raise ValueError("checkpoint_candidates.direction must be maximize or minimize")
    if candidate["delta_mode"] not in {"absolute", "relative"}:
        raise ValueError("checkpoint_candidates.delta_mode must be absolute or relative")
    delta = candidate["min_delta"]
    if type(delta) not in (int, float) or not math.isfinite(delta) or delta <= 0:
        raise ValueError("checkpoint_candidates.min_delta must be finite and positive")
    for key, minimum in (
        ("start_after_steps", 0),
        ("min_interval_steps", 1),
        ("max_checkpoints", 1),
    ):
        if type(candidate[key]) is not int or candidate[key] < minimum:
            raise ValueError(f"checkpoint_candidates.{key} must be an integer >= {minimum}")
    if candidate["max_checkpoints"] > 16:
        raise ValueError("checkpoint_candidates.max_checkpoints must be <= 16")
    envelope = config.get("training_backend") or {}
    backend = envelope.get("id") if isinstance(envelope, Mapping) else None
    if backend not in {"sb3.ppo", "sb3.a2c", "gradlab.ppo"}:
        raise ValueError("checkpoint_candidates requires a learned PPO/A2C policy")
    config["checkpoint_candidates"] = candidate


class CheckpointSchedule:
    """Absolute transition milestones; crossed thresholds coalesce into one save.

    Trainers call this after updates (or complete search batches), so a threshold
    rounds upward to the next safe boundary. Resume skips already crossed steps.
    Final is owned by the terminal path, never by this schedule.
    """

    def __init__(self, config: Mapping[str, Any], *, initial_step: int = 0):
        self.limit = int(config["timesteps"])
        self.interval = int(config.get("checkpoint_freq", 0))
        self.next_periodic = (
            (initial_step // self.interval + 1) * self.interval if self.interval else None
        )
        self.pending = [
            s for s in config.get("checkpoint_steps", ()) if initial_step < s < self.limit
        ]
        self.candidate = config.get("checkpoint_candidates")
        self._enabled = bool(self.interval or config.get("checkpoint_steps") or self.candidate)
        self.reference: float | None = None
        self.sample_step = -1
        self.candidate_count = 0
        self.last_candidate_step = initial_step

    @property
    def enabled(self) -> bool:
        return self._enabled

    def due(self, step: int) -> bool:
        due = False
        if self.next_periodic is not None and step >= self.next_periodic:
            due = True
            self.next_periodic = (step // self.interval + 1) * self.interval
        if self.pending and step >= self.pending[0]:
            due = True
            self.pending = [s for s in self.pending if s > step]
        return due and step < self.limit

    def candidate_due(self, step: int, sample: tuple[float, int] | None) -> bool:
        rule = self.candidate
        if not rule or sample is None or step >= self.limit:
            return False
        value, sample_step = sample
        if (
            not math.isfinite(value)
            or sample_step <= self.sample_step
            or sample_step > step
            or sample_step < rule["start_after_steps"]
        ):
            return False
        self.sample_step = sample_step
        if self.reference is None:
            self.reference = value
            return False
        delta = rule["min_delta"]
        if rule["delta_mode"] == "relative":
            delta *= max(abs(self.reference), 1.0)
        improvement = value - self.reference
        if rule["direction"] == "minimize":
            improvement = -improvement
        if (
            improvement < delta
            or self.candidate_count >= rule["max_checkpoints"]
            or step - self.last_candidate_step < rule["min_interval_steps"]
        ):
            return False
        self.reference = value
        self.candidate_count += 1
        self.last_candidate_step = step
        return True


def checkpoint_plan(config: Mapping[str, Any]) -> tuple[list[int], int]:
    """Nominal scheduled steps and maximum optional candidates, for calibration."""
    cap = int(config["timesteps"])
    interval = int(config.get("checkpoint_freq", 0))
    steps = set(s for s in config.get("checkpoint_steps", ()) if s < cap)
    if interval:
        steps.update(range(interval, cap, interval))
    # GradLab PPO's explicit backend branch points are part of the same workload.
    steps.update(
        s
        for s in config.get("training_backend", {})
        .get("config", {})
        .get("checkpoint_update_steps", ())
        if s < cap
    )
    steps.add(cap)
    return sorted(steps), int((config.get("checkpoint_candidates") or {}).get("max_checkpoints", 0))
