from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from gradlab.metric_names import (
    TRAIN_REWARD_ROOT,
    stat_metric,
    train_reward_component_metric,
    train_reward_event_metric,
    validate_metric_name,
)


@dataclass
class _RewardMoments:
    """Merge finite batch moments without retaining a rollout's samples."""

    size: int = 0
    mean: float = 0.0
    m2: float = 0.0
    nonzero: int = 0
    absolute_sum: float = 0.0

    def update(self, value: Any, *, reserve: int) -> np.ndarray:
        del reserve
        values = np.asarray(value, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        if not values.size:
            return values
        count = int(values.size)
        batch_mean = float(np.mean(values))
        delta = batch_mean - self.mean
        total = self.size + count
        self.m2 += (
            float(np.sum((values - batch_mean) ** 2)) + delta * delta * self.size * count / total
        )
        self.mean += delta * count / total
        self.size = total
        self.nonzero += int(np.count_nonzero(values))
        self.absolute_sum += float(np.sum(np.abs(values)))
        return values

    def flush(self) -> _RewardMoments:
        result = _RewardMoments(self.size, self.mean, self.m2, self.nonzero, self.absolute_sum)
        self.size = self.nonzero = 0
        self.mean = self.m2 = self.absolute_sum = 0.0
        return result

    @property
    def std(self) -> float:
        return math.sqrt(max(0.0, self.m2 / self.size)) if self.size else 0.0

    @property
    def nonzero_rate(self) -> float:
        return self.nonzero / self.size if self.size else 0.0


class RewardStatsAccumulator:
    component_info_keys = {
        "native": "native_reward_component",
        "cell_novelty": "cell_novelty_reward_component",
        "event": "event_reward_component",
        "progress": "progress_reward_component",
        "score": "score_reward_component",
        "completion": "completion_reward_component",
        "death": "death_penalty_component",
        "time": "time_penalty_component",
        "kill": "kill_reward_component",
        "hit": "hit_reward_component",
        "damage": "damage_reward_component",
        "health": "health_reward_component",
        "armor": "armor_reward_component",
        "weapon": "weapon_reward_component",
        "ammo": "ammo_reward_component",
        "weapon_hold": "weapon_hold_reward_component",
    }

    def __init__(
        self,
        *,
        active_components: Sequence[str] = (),
        task: Mapping[str, Any] | None = None,
        required_metrics: Sequence[str] = (),
    ) -> None:
        from gradlab.metric_inventory import redundant_reward_metrics

        self.omitted = (
            redundant_reward_metrics(task, required=frozenset(required_metrics))
            if task is not None
            else frozenset()
        )
        self.shaped = _RewardMoments()
        self.raw = _RewardMoments()
        self.active_components = tuple(
            component for component in active_components if component in self.component_info_keys
        )
        self.components = {component: _RewardMoments() for component in self.active_components}
        self.event_rewards: dict[str, _RewardMoments] = {}

    def consume(self, metrics: Mapping[str, Any], *, reserve: int) -> None:
        if (value := metrics.get("shaped_reward")) is not None:
            self.shaped.update(value, reserve=reserve)
        if (value := metrics.get("raw_reward")) is not None:
            self.raw.update(value, reserve=reserve)
        for component, accumulator in self.components.items():
            info_key = self.component_info_keys[component]
            value = metrics.get(info_key)
            if value is not None:
                accumulator.update(value, reserve=reserve)
        event_prefix = "event_reward_component/"
        for info_key, value in metrics.items():
            if not isinstance(info_key, str) or not info_key.startswith(event_prefix):
                continue
            event = info_key.removeprefix(event_prefix)
            train_reward_event_metric(event, "mean")
            accumulator = self.event_rewards.setdefault(event, _RewardMoments())
            accumulator.update(value, reserve=reserve)

    @staticmethod
    def _distribution(
        prefix: str, values: _RewardMoments, stats: Sequence[str]
    ) -> dict[str, float]:
        if values.size == 0:
            return {}
        calculations = {
            "mean": lambda: values.mean,
            "std": lambda: values.std,
            "nonzero_rate": lambda: values.nonzero_rate,
        }
        return {
            (
                validate_metric_name(f"{prefix}/nonzero/fraction")
                if stat == "nonzero_rate"
                else stat_metric(prefix, stat)
            ): calculations[stat]()
            for stat in stats
        }

    def flush(self) -> dict[str, float]:
        shaped = self.shaped.flush()
        raw = self.raw.flush()
        payload = self._distribution(
            TRAIN_REWARD_ROOT,
            shaped,
            ("mean", "std", "nonzero_rate"),
        )
        if raw.size > 0:
            payload.update(self._distribution(f"{TRAIN_REWARD_ROOT}/task", raw, ("mean", "std")))
        abs_sums: dict[str, float] = {}
        for component, accumulator in self.components.items():
            values = accumulator.flush()
            if values.size == 0:
                continue
            payload[train_reward_component_metric(component, "mean")] = values.mean
            payload[train_reward_component_metric(component, "nonzero_rate")] = values.nonzero_rate
            abs_sums[component] = values.absolute_sum
        total_abs_sum = sum(abs_sums.values())
        for component, abs_sum in abs_sums.items():
            payload[train_reward_component_metric(component, "share")] = (
                abs_sum / total_abs_sum if total_abs_sum > 0.0 else 0.0
            )
        for event, accumulator in self.event_rewards.items():
            values = accumulator.flush()
            if values.size == 0:
                continue
            payload[train_reward_event_metric(event, "mean")] = values.mean
            payload[train_reward_event_metric(event, "nonzero_rate")] = values.nonzero_rate
        return {name: value for name, value in payload.items() if name not in self.omitted}
