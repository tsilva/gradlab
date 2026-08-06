from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from gradlab.eval_metrics import episode_reason_names
from gradlab.metric_names import (
    TRAIN_EPISODE_LENGTH_ACROSS_ORIGINS_ROLLING_UP_TO_100_MEAN,
    TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_ROLLING_UP_TO_100_MAX,
    TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_ROLLING_UP_TO_100_MEAN,
    TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_WINDOW_100_MEAN,
    TRAIN_EPISODE_RETURN_SHAPED_ACROSS_ORIGINS_ROLLING_UP_TO_100_MAX,
    TRAIN_EPISODE_RETURN_SHAPED_ACROSS_ORIGINS_ROLLING_UP_TO_100_MEAN,
    TRAIN_OUTCOME_SUCCESS_ACROSS_OBSERVED_STARTS_CUMULATIVE_RATE_MEAN,
    TRAIN_OUTCOME_SUCCESS_ACROSS_OBSERVED_STARTS_CUMULATIVE_RATE_MIN,
    TRAIN_OUTCOME_SUCCESS_ACROSS_STARTS_COVERAGE_RATE,
    TRAIN_OUTCOME_SUCCESS_ACROSS_STARTS_WINDOW_100_RATE_MEAN,
    TRAIN_OUTCOME_SUCCESS_ACROSS_STARTS_WINDOW_100_RATE_MIN,
    TRAIN_EPISODE_COMPLETED_COUNT,
    TRAIN_EXPLORATION_CELL_UNIQUE_FROM_TARGET_ROLLING_UP_TO_100_MEAN,
    metric_value_segment,
    train_outcome_reason_count_metric,
    train_outcome_reason_window_rate_metric,
    train_progress_from_target_rolling_mean_metric,
    train_success_attempts_metric,
    train_success_count_metric,
    train_success_window_rate_metric,
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


class EpisodeMetricsReducer:
    """Canonical reducer shared by every training backend."""

    window_size = 100

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
        self.returns: deque[float] = deque(maxlen=self.window_size)
        self.target_returns: deque[float] = deque(maxlen=self.window_size)
        self.target_unique_cells: deque[int] = deque(maxlen=self.window_size)
        self.target_progress: dict[str, deque[float]] = {
            field: deque(maxlen=self.window_size) for field in self.progress_fields
        }
        self.lengths: deque[int] = deque(maxlen=self.window_size)
        self.terminal_count = 0
        self.reason_counts: dict[str, int] = {name: 0 for name in self.event_names}
        self.reason_windows: dict[str, deque[bool]] = {
            name: deque(maxlen=self.window_size) for name in self.event_names
        }
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
        self.returns.append(float(record.episode_return))
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
            self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1
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
        if self.returns:
            payload[TRAIN_EPISODE_RETURN_SHAPED_ACROSS_ORIGINS_ROLLING_UP_TO_100_MEAN] = float(np.mean(self.returns))
            payload[TRAIN_EPISODE_RETURN_SHAPED_ACROSS_ORIGINS_ROLLING_UP_TO_100_MAX] = float(np.max(self.returns))
            payload[TRAIN_EPISODE_LENGTH_ACROSS_ORIGINS_ROLLING_UP_TO_100_MEAN] = float(np.mean(self.lengths))
        if self.target_returns:
            payload[TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_ROLLING_UP_TO_100_MEAN] = float(
                np.mean(self.target_returns)
            )
            payload[TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_ROLLING_UP_TO_100_MAX] = float(
                np.max(self.target_returns)
            )
            if len(self.target_returns) >= self.window_size:
                payload[TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_WINDOW_100_MEAN] = float(
                    np.mean(self.target_returns)
                )
        if self.target_unique_cells:
            payload[
                TRAIN_EXPLORATION_CELL_UNIQUE_FROM_TARGET_ROLLING_UP_TO_100_MEAN
            ] = float(np.mean(self.target_unique_cells))
        for field, window in self.target_progress.items():
            if window:
                payload[train_progress_from_target_rolling_mean_metric(field)] = float(
                    np.mean(window)
                )
        for reason, count in sorted(self.reason_counts.items()):
            window = self.reason_windows[reason]
            payload[train_outcome_reason_count_metric(reason)] = count
            payload[train_outcome_reason_window_rate_metric(reason)] = (
                sum(window) / len(window) if window else 0.0
            )

        if not self.track_success or not self.attempt_counts:
            return payload
        rates: dict[str, float] = {}
        for start, attempts in self.attempt_counts.items():
            successes = self.success_counts.get(start, 0)
            rates[start] = successes / attempts
            payload[train_success_count_metric(start)] = successes
            payload[train_success_attempts_metric(start)] = attempts
            window = self.success_windows[start]
            if len(window) >= self.window_size:
                payload[train_success_window_rate_metric(start)] = sum(window) / len(window)

        expected = self.configured_starts or tuple(self.attempt_counts)
        payload[TRAIN_OUTCOME_SUCCESS_ACROSS_OBSERVED_STARTS_CUMULATIVE_RATE_MIN] = min(rates.values())
        payload[TRAIN_OUTCOME_SUCCESS_ACROSS_OBSERVED_STARTS_CUMULATIVE_RATE_MEAN] = float(np.mean(tuple(rates.values())))
        payload[TRAIN_OUTCOME_SUCCESS_ACROSS_STARTS_COVERAGE_RATE] = sum(
            start in self.attempt_counts for start in expected
        ) / len(expected)
        if expected and all(
            len(self.success_windows.get(start, ())) >= self.window_size for start in expected
        ):
            window_rates = [
                sum(self.success_windows[start]) / len(self.success_windows[start])
                for start in expected
            ]
            payload[TRAIN_OUTCOME_SUCCESS_ACROSS_STARTS_WINDOW_100_RATE_MIN] = min(window_rates)
            payload[TRAIN_OUTCOME_SUCCESS_ACROSS_STARTS_WINDOW_100_RATE_MEAN] = float(np.mean(window_rates))
        return payload
