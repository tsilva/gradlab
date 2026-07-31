from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from stable_baselines3 import PPO

ADVANTAGE_NORMALIZATION_CHOICES = ("auto", "none", "global")


def normalize_advantage_normalization(
    value: Any,
    *,
    label: str,
) -> str | dict[str, str]:
    if isinstance(value, str):
        if value not in ADVANTAGE_NORMALIZATION_CHOICES:
            raise ValueError(
                f"{label} must be one of {', '.join(ADVANTAGE_NORMALIZATION_CHOICES)} "
                "or a grouped context object"
            )
        return value
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{label} must be a string or an object with mode and context"
        )
    unexpected = sorted(set(value) - {"mode", "context"})
    if unexpected:
        raise ValueError(f"{label} has unexpected fields: {unexpected}")
    if value.get("mode") != "grouped":
        raise ValueError(f"{label}.mode must be 'grouped'")
    context = value.get("context")
    if not isinstance(context, str) or not context.strip() or "/" in context:
        raise ValueError(f"{label}.context must be a context field name")
    return {"mode": "grouped", "context": context.strip()}


def resolve_advantage_normalization_mode(
    config: Mapping[str, Any],
) -> tuple[str, str | None]:
    value = config.get("advantage_normalization", "auto")
    if isinstance(value, Mapping):
        normalized = normalize_advantage_normalization(
            value,
            label="advantage_normalization",
        )
        assert isinstance(normalized, dict)
        return "grouped", normalized["context"]
    mode = value
    if mode == "auto":
        return (
            "global" if config.get("normalize_advantage", False) else "none",
            None,
        )
    if mode not in ADVANTAGE_NORMALIZATION_CHOICES:
        raise ValueError(f"unknown advantage normalization mode: {mode!r}")
    return str(mode), None


def normalize_advantages_by_context(
    advantages: np.ndarray,
    observations: Mapping[str, np.ndarray],
    context: str,
    *,
    eps: float = 1e-8,
) -> dict[int, dict[str, float]]:
    key = f"context/{context}"
    if key not in observations:
        raise ValueError(
            f"grouped advantage normalization requires observations with a {key!r} key"
        )
    task_ids = np.asarray(observations[key])
    if task_ids.shape == (*advantages.shape, 1):
        task_ids = task_ids[..., 0]
    if task_ids.shape != advantages.shape:
        raise ValueError(
            "categorical context shape must match advantages: "
            f"context={task_ids.shape}, advantages={advantages.shape}"
        )
    if not np.issubdtype(task_ids.dtype, np.integer):
        raise ValueError(f"grouped context {context!r} must contain integer category indices")
    stats: dict[int, dict[str, float]] = {}
    for task_id in np.unique(task_ids):
        normalized_id = int(task_id)
        if normalized_id < 0:
            raise ValueError(f"grouped context {context!r} contains a negative category index")
        mask = task_ids == task_id
        count = int(np.count_nonzero(mask))
        task_advantages = advantages[mask]
        mean = float(np.mean(task_advantages))
        std = float(np.std(task_advantages))
        stats[normalized_id] = {
            "count": float(count),
            "mean_pre": mean,
            "std_pre": std,
        }
        if count > 1:
            advantages[mask] = (task_advantages - mean) / (std + eps)
        stats[normalized_id]["mean_post"] = float(np.mean(advantages[mask]))
        stats[normalized_id]["std_post"] = float(np.std(advantages[mask]))
    return stats


def normalize_advantages_by_task(
    advantages: np.ndarray,
    observations: Mapping[str, np.ndarray],
    *,
    eps: float = 1e-8,
) -> dict[int, dict[str, float]]:
    if "task" not in observations:
        raise ValueError(
            "per-task advantage normalization requires dict observations with a 'task' key"
        )

    task_vectors = np.asarray(observations["task"])
    if task_vectors.ndim < 2:
        raise ValueError(f"expected one-hot task observations, got shape {task_vectors.shape}")
    if task_vectors.shape[:-1] != advantages.shape:
        raise ValueError(
            "task observation shape must match advantages except for task dimension: "
            f"task={task_vectors.shape}, advantages={advantages.shape}"
        )

    task_ids = np.argmax(task_vectors, axis=-1)
    task_count = int(task_vectors.shape[-1])
    stats: dict[int, dict[str, float]] = {}
    for task_id in range(task_count):
        mask = task_ids == task_id
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        task_advantages = advantages[mask]
        mean = float(np.mean(task_advantages))
        std = float(np.std(task_advantages))
        stats[task_id] = {
            "count": float(count),
            "mean_pre": mean,
            "std_pre": std,
        }
        if count > 1:
            advantages[mask] = (task_advantages - mean) / (std + eps)
        stats[task_id]["mean_post"] = float(np.mean(advantages[mask]))
        stats[task_id]["std_post"] = float(np.std(advantages[mask]))
    return stats


class PerTaskAdvantagePPO(PPO):
    """Legacy PPO class retained only for loading existing one-hot task artifacts."""

    def train(self) -> None:
        normalize_advantages_by_task(
            self.rollout_buffer.advantages,
            self.rollout_buffer.observations,
        )
        super().train()


class GroupedAdvantagePPO(PPO):
    """PPO variant that normalizes rollout advantages by a named context field."""

    def __init__(
        self,
        *args: Any,
        advantage_context: str = "",
        **kwargs: Any,
    ) -> None:
        loading = kwargs.get("_init_setup_model") is False
        if (
            not loading
            and (not isinstance(advantage_context, str) or not advantage_context)
        ):
            raise ValueError("advantage_context must be a non-empty context field name")
        self.advantage_context = advantage_context
        super().__init__(*args, **kwargs)

    def train(self) -> None:
        normalize_advantages_by_context(
            self.rollout_buffer.advantages,
            self.rollout_buffer.observations,
            self.advantage_context,
        )
        super().train()
