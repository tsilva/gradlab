from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
import time
from typing import Any, Callable

import numpy as np

from gradlab.eval_metrics import episode_reason_names
from gradlab.metric_names import (
    EPISODE_METRIC_WINDOW_SIZE,
    TRAIN_EPISODE_LENGTH_ORIGIN_ALL_ROLLING_MEAN,
    TRAIN_EPISODE_RETURN_SHAPED_ORIGIN_TARGET_ROLLING_MAX,
    TRAIN_EPISODE_RETURN_SHAPED_ORIGIN_TARGET_ROLLING_MEAN,
    TRAIN_OUTCOME_SUCCESS_STARTS_ALL_ROLLING_RATE_MEAN,
    TRAIN_OUTCOME_SUCCESS_STARTS_ALL_ROLLING_RATE_MIN,
    TRAIN_OUTCOME_SUCCESS_STARTS_OBSERVED_CUMULATIVE_RATE_MEAN,
    TRAIN_OUTCOME_SUCCESS_STARTS_OBSERVED_CUMULATIVE_RATE_MIN,
    TRAIN_EPISODE_COMPLETED_COUNT,
    TRAIN_EXPLORATION_CELL_UNIQUE_ORIGIN_TARGET_ROLLING_MEAN,
    TRAIN_THROUGHPUT_BETWEEN_ROLLOUTS_SECONDS,
    TRAIN_THROUGHPUT_LOOP_RATE,
    TRAIN_THROUGHPUT_PROVIDER_STEP_RATE,
    TRAIN_THROUGHPUT_ROLLOUT_OVERHEAD_SECONDS,
    metric_value_segment,
    train_outcome_reason_count_metric,
    train_outcome_reason_rolling_rate_metric,
    train_progress_origin_target_rolling_mean_metric,
    train_success_count_metric,
    train_success_rolling_rate_metric,
    validate_metric_payload,
)
from gradlab.task_kernels import CELL_NOVELTY_EPISODE_UNIQUE_CELLS


def _outcome_name(record: Any) -> str:
    outcome = getattr(record, "outcome", None)
    return str(getattr(outcome, "name", outcome)).lower()


def episode_succeeded(record: Any) -> bool:
    if _outcome_name(record) == "success":
        return True
    metrics = getattr(record, "metrics", None)
    return bool(isinstance(metrics, Mapping) and metrics.get("level_complete"))


def throughput_delta_metrics(
    *,
    steps: int,
    loop_seconds: float,
    provider_step_seconds: float | None = None,
    rollout_seconds: float | None = None,
    between_rollouts_seconds: float | None = None,
) -> dict[str, float]:
    """Return canonical rates and retained timing from one bounded step delta."""

    if steps <= 0 or loop_seconds <= 0.0:
        return {}
    payload = {TRAIN_THROUGHPUT_LOOP_RATE: steps / loop_seconds}
    if provider_step_seconds is not None and provider_step_seconds > 0.0:
        payload[TRAIN_THROUGHPUT_PROVIDER_STEP_RATE] = steps / provider_step_seconds
        if rollout_seconds is not None and rollout_seconds >= 0.0:
            payload[TRAIN_THROUGHPUT_ROLLOUT_OVERHEAD_SECONDS] = max(
                rollout_seconds - provider_step_seconds,
                0.0,
            )
    if between_rollouts_seconds is not None and between_rollouts_seconds >= 0.0:
        payload[TRAIN_THROUGHPUT_BETWEEN_ROLLOUTS_SECONDS] = between_rollouts_seconds
    validate_metric_payload(payload)
    return payload


def _native_step_stats(source: Any) -> Mapping[str, float | int] | None:
    seen: set[int] = set()
    current = source
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        native_step_stats = getattr(current, "native_step_stats", None)
        if callable(native_step_stats):
            value = native_step_stats()
            return value if isinstance(value, Mapping) else None
        current = getattr(current, "venv", None) or getattr(current, "env", None)
    return None


class DeltaThroughputTracker:
    """Measure report-to-report throughput without cumulative-run averaging."""

    def __init__(
        self,
        source: Any,
        *,
        initial_step: int = 0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.source = source
        self.clock = clock or time.perf_counter
        self.step = int(initial_step)
        self.started_at = self.clock()
        self.native_start = _native_step_stats(source)

    def snapshot(self, step: int) -> dict[str, float]:
        now = self.clock()
        current_step = int(step)
        steps = current_step - self.step
        loop_seconds = now - self.started_at
        native_end = _native_step_stats(self.source)
        provider_seconds: float | None = None
        if self.native_start is not None and native_end is not None:
            calls = int(native_end.get("calls_total", 0)) - int(
                self.native_start.get("calls_total", 0)
            )
            elapsed = float(native_end.get("seconds_total", 0.0)) - float(
                self.native_start.get("seconds_total", 0.0)
            )
            if calls > 0 and elapsed > 0.0:
                provider_seconds = elapsed
        self.step = current_step
        self.started_at = now
        self.native_start = native_end
        return throughput_delta_metrics(
            steps=steps,
            loop_seconds=loop_seconds,
            provider_step_seconds=provider_seconds,
        )


class EpisodeMetricsReducer:
    """Canonical reducer shared by every training backend."""

    window_size = EPISODE_METRIC_WINDOW_SIZE

    def __init__(
        self,
        *,
        event_names: Sequence[str] = (),
        configured_starts: Sequence[str] = (),
        progress_fields: Sequence[str] = (),
        track_success: bool = True,
    ) -> None:
        self.event_names = tuple(dict.fromkeys(str(name) for name in event_names))
        self.configured_starts = tuple(
            dict.fromkeys(metric_value_segment(start) for start in configured_starts)
        )
        self.track_success = bool(track_success)
        self.progress_fields = tuple(
            dict.fromkeys(metric_value_segment(field) for field in progress_fields)
        )
        self.target_returns: deque[float] = deque(maxlen=self.window_size)
        self.target_unique_cells: deque[int] = deque(maxlen=self.window_size)
        self.target_progress: dict[str, deque[float]] = {
            field: deque(maxlen=self.window_size) for field in self.progress_fields
        }
        self.lengths: deque[int] = deque(maxlen=self.window_size)
        self.terminal_count = 0
        self.reason_windows: dict[str, deque[bool]] = {
            name: deque(maxlen=self.window_size) for name in self.event_names
        }
        self.reason_counts: dict[str, int] = dict.fromkeys(self.event_names, 0)
        self.success_counts: dict[str, int] = {}
        self.attempt_counts: dict[str, int] = {}
        self.success_windows: dict[str, deque[bool]] = {}

    def consume(self, records: Iterable[Any]) -> dict[str, int | float]:
        for record in records:
            if not hasattr(record, "episode_return"):
                continue
            self._consume_episode(record)
        return self.snapshot()

    def _consume_episode(self, record: Any) -> None:
        self.lengths.append(int(getattr(record, "episode_length", 0)))
        target_origin = str(getattr(record, "start_origin", "target")) == "target"
        if target_origin:
            self.target_returns.append(float(record.episode_return))
            metrics = getattr(record, "metrics", None)
            if isinstance(metrics, Mapping):
                unique_cells = metrics.get(CELL_NOVELTY_EPISODE_UNIQUE_CELLS)
                if (
                    isinstance(unique_cells, int | float | np.number)
                    and not isinstance(unique_cells, bool | np.bool_)
                    and np.isfinite(float(unique_cells))
                    and float(unique_cells) >= 1.0
                    and float(unique_cells).is_integer()
                ):
                    self.target_unique_cells.append(int(unique_cells))
            for field, window in self.target_progress.items():
                value = metrics.get(field) if isinstance(metrics, Mapping) else None
                if (
                    not isinstance(value, int | float | np.number)
                    or isinstance(value, bool | np.bool_)
                    or not np.isfinite(float(value))
                ):
                    raise ValueError(
                        f"configured episode progress field {field!r} must be a finite number"
                    )
                window.append(float(value))

        succeeded = episode_succeeded(record)
        reasons = (
            set()
            if succeeded
            else episode_reason_names(
                getattr(record, "events", ()) or (),
                terminated=bool(getattr(record, "terminated", False)),
                truncated=bool(getattr(record, "truncated", False)),
            )
        )
        prior_count = self.terminal_count
        self.terminal_count += 1
        for reason in reasons:
            if reason not in self.reason_windows:
                prior = min(prior_count, self.window_size - 1)
                self.reason_windows[reason] = deque(
                    [False] * prior,
                    maxlen=self.window_size,
                )
                self.reason_counts[reason] = 0
            self.reason_counts[reason] += 1
        for reason, window in self.reason_windows.items():
            window.append(reason in reasons)

        if not self.track_success or not target_origin:
            return
        start_id = getattr(record, "start_id", None)
        if start_id is None:
            return
        start = metric_value_segment(start_id)
        window = self.success_windows.setdefault(start, deque(maxlen=self.window_size))
        window.append(succeeded)
        self.attempt_counts[start] = self.attempt_counts.get(start, 0) + 1
        if succeeded:
            self.success_counts[start] = self.success_counts.get(start, 0) + 1

    def snapshot(self) -> dict[str, int | float]:
        payload: dict[str, int | float] = {
            TRAIN_EPISODE_COMPLETED_COUNT: self.terminal_count,
        }
        if self.lengths:
            payload[TRAIN_EPISODE_LENGTH_ORIGIN_ALL_ROLLING_MEAN] = float(np.mean(self.lengths))
        if self.target_returns:
            payload[TRAIN_EPISODE_RETURN_SHAPED_ORIGIN_TARGET_ROLLING_MEAN] = float(
                np.mean(self.target_returns)
            )
            payload[TRAIN_EPISODE_RETURN_SHAPED_ORIGIN_TARGET_ROLLING_MAX] = float(
                np.max(self.target_returns)
            )
        if self.target_unique_cells:
            payload[TRAIN_EXPLORATION_CELL_UNIQUE_ORIGIN_TARGET_ROLLING_MEAN] = float(
                np.mean(self.target_unique_cells)
            )
        for field, window in self.target_progress.items():
            if window:
                payload[train_progress_origin_target_rolling_mean_metric(field)] = float(
                    np.mean(window)
                )
        for reason, window in sorted(self.reason_windows.items()):
            payload[train_outcome_reason_count_metric(reason)] = self.reason_counts[reason]
            payload[train_outcome_reason_rolling_rate_metric(reason)] = (
                sum(window) / len(window) if window else 0.0
            )

        if not self.track_success or not self.attempt_counts:
            return payload
        rates: dict[str, float] = {}
        for start, attempts in self.attempt_counts.items():
            successes = self.success_counts.get(start, 0)
            rates[start] = successes / attempts
            payload[train_success_count_metric(start)] = successes
            window = self.success_windows[start]
            if len(window) >= self.window_size:
                payload[train_success_rolling_rate_metric(start)] = sum(window) / len(window)

        expected = self.configured_starts or tuple(self.attempt_counts)
        payload[TRAIN_OUTCOME_SUCCESS_STARTS_OBSERVED_CUMULATIVE_RATE_MIN] = min(rates.values())
        payload[TRAIN_OUTCOME_SUCCESS_STARTS_OBSERVED_CUMULATIVE_RATE_MEAN] = float(
            np.mean(tuple(rates.values()))
        )
        if expected and all(
            len(self.success_windows.get(start, ())) >= self.window_size for start in expected
        ):
            window_rates = [
                sum(self.success_windows[start]) / len(self.success_windows[start])
                for start in expected
            ]
            payload[TRAIN_OUTCOME_SUCCESS_STARTS_ALL_ROLLING_RATE_MIN] = min(window_rates)
            payload[TRAIN_OUTCOME_SUCCESS_STARTS_ALL_ROLLING_RATE_MEAN] = float(
                np.mean(window_rates)
            )
        return payload
