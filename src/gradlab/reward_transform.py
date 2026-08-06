from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Real
from typing import Any


REWARD_SCALE_KEY = "reward_scale"
REWARD_CLIP_KEY = "reward_clip"
COMMON_REWARD_KEYS = frozenset({REWARD_SCALE_KEY, REWARD_CLIP_KEY})
PROVIDER_REWARD_TRANSFORM_KEYS = frozenset(
    {
        "reward_clip",
        "reward_clipping",
        "normalize_reward",
        "norm_reward",
        "reward_normalization",
    }
)


@dataclass(frozen=True)
class RewardTransform:
    scale: float = 1.0
    clip_bounds: tuple[float, float] | None = None

    @property
    def active(self) -> bool:
        return self.scale != 1.0 or self.clip_bounds is not None


def _normalize_scale(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number between 0 and 1 inclusive")
    scale = float(value)
    if not math.isfinite(scale) or scale < 0.0 or scale > 1.0:
        raise ValueError(f"{label} must be a finite number between 0 and 1 inclusive")
    return scale


def _normalize_clip(
    value: Any,
    *,
    label: str,
) -> tuple[float, float] | None:
    if value is None or value is False:
        return None
    if value is True:
        return (-1.0, 1.0)
    if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be false, true, or a finite [low, high] pair")
    if len(value) != 2:
        raise ValueError(f"{label} must contain exactly [low, high]")
    low, high = value
    if (
        isinstance(low, bool)
        or isinstance(high, bool)
        or not isinstance(low, Real)
        or not isinstance(high, Real)
    ):
        raise ValueError(f"{label} bounds must be finite numbers")
    bounds = float(low), float(high)
    if not all(math.isfinite(item) for item in bounds) or bounds[0] > bounds[1]:
        raise ValueError(f"{label} bounds must be finite with low <= high")
    return bounds


def reward_transform_from_reward(
    reward: Mapping[str, Any],
    *,
    label: str = "task.reward",
) -> RewardTransform:
    scale = _normalize_scale(reward.get(REWARD_SCALE_KEY, 1.0), label=f"{label}.reward_scale")
    bounds = _normalize_clip(
        reward.get(REWARD_CLIP_KEY, False),
        label=f"{label}.reward_clip",
    )
    if scale == 0.0 and bounds is not None and not bounds[0] <= 0.0 <= bounds[1]:
        raise ValueError(f"{label}.reward_clip must include zero when {label}.reward_scale is zero")
    return RewardTransform(scale=scale, clip_bounds=bounds)


def normalize_reward_mapping(
    reward: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    normalized = deepcopy(dict(reward))
    transform = reward_transform_from_reward(normalized, label=label)
    normalized[REWARD_SCALE_KEY] = transform.scale
    if transform.clip_bounds is None:
        normalized[REWARD_CLIP_KEY] = False
    else:
        normalized[REWARD_CLIP_KEY] = list(transform.clip_bounds)
    return normalized


def normalize_task_reward(
    task: Mapping[str, Any],
    *,
    label: str = "task",
) -> dict[str, Any]:
    normalized = deepcopy(dict(task))
    reward = normalized.get("reward")
    if not isinstance(reward, Mapping):
        raise ValueError(f"{label}.reward must be an object")
    normalized["reward"] = normalize_reward_mapping(
        reward,
        label=f"{label}.reward",
    )
    return normalized


__all__ = [
    "COMMON_REWARD_KEYS",
    "PROVIDER_REWARD_TRANSFORM_KEYS",
    "RewardTransform",
    "normalize_reward_mapping",
    "normalize_task_reward",
    "reward_transform_from_reward",
]
