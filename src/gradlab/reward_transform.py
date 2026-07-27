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
LEGACY_SIGN_CLIP = "__gradlab_legacy_sign__"
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
    sign_clip: bool = False

    @property
    def active(self) -> bool:
        return self.scale != 1.0 or self.clip_bounds is not None or self.sign_clip


def _normalize_scale(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a positive finite number")
    scale = float(value)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{label} must be a positive finite number")
    return scale


def _normalize_clip(
    value: Any,
    *,
    label: str,
) -> tuple[tuple[float, float] | None, bool]:
    if value is None or value is False:
        return None, False
    if value is True:
        return (-1.0, 1.0), False
    if isinstance(value, str) and value == LEGACY_SIGN_CLIP:
        return None, True
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
    return bounds, False


def reward_transform_from_reward(
    reward: Mapping[str, Any],
    *,
    label: str = "task.reward",
) -> RewardTransform:
    scale = _normalize_scale(reward.get(REWARD_SCALE_KEY, 1.0), label=f"{label}.reward_scale")
    bounds, sign_clip = _normalize_clip(
        reward.get(REWARD_CLIP_KEY, False),
        label=f"{label}.reward_clip",
    )
    return RewardTransform(scale=scale, clip_bounds=bounds, sign_clip=sign_clip)


def normalize_reward_mapping(
    reward: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    normalized = deepcopy(dict(reward))
    transform = reward_transform_from_reward(normalized, label=label)
    normalized[REWARD_SCALE_KEY] = transform.scale
    if transform.sign_clip:
        normalized[REWARD_CLIP_KEY] = LEGACY_SIGN_CLIP
    elif transform.clip_bounds is None:
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


def migrate_legacy_artifact_reward_config(
    env_args: Mapping[str, Any] | None,
    task: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Translate old provider/Mario reward knobs into the gradlab-owned transform.

    Stored identities and hashes remain untouched; this translation is only for
    reconstructing an executable environment from legacy artifact metadata.
    """

    normalized_args = deepcopy(dict(env_args or {}))
    normalized_task = deepcopy(dict(task))
    reward = deepcopy(dict(normalized_task.get("reward") or {}))

    provider_clip: Any = None
    provider_clip_found = False
    for key in ("reward_clip", "reward_clipping"):
        if key not in normalized_args:
            continue
        value = normalized_args.pop(key)
        if provider_clip_found and value != provider_clip:
            raise ValueError("legacy artifact provider reward clipping settings disagree")
        provider_clip = value
        provider_clip_found = True
    for key in PROVIDER_REWARD_TRANSFORM_KEYS - {"reward_clip", "reward_clipping"}:
        normalized_args.pop(key, None)

    if "clip_rewards" in reward:
        if REWARD_CLIP_KEY in reward:
            raise ValueError(
                "legacy artifact defines both task.reward.clip_rewards and reward_clip"
            )
        legacy_sign = reward.pop("clip_rewards")
        if type(legacy_sign) is not bool:
            raise ValueError("legacy task.reward.clip_rewards must be a boolean")
        reward[REWARD_CLIP_KEY] = LEGACY_SIGN_CLIP if legacy_sign else False

        mode = str(reward.get("reward_mode") or "")
        old_scale = _normalize_scale(
            reward.get(REWARD_SCALE_KEY, 1.0),
            label="legacy task.reward.reward_scale",
        )
        if mode in {"native", "score", "additive"}:
            # These modes ignored reward_scale in the v2 Mario kernel.
            reward[REWARD_SCALE_KEY] = 1.0
        elif mode in {"bounded", "baseline"}:
            # V2 divided the shaped reward before subtracting the time penalty.
            reward["time_penalty"] = float(reward.get("time_penalty", 0.0)) * old_scale

    if provider_clip_found:
        provider_transform = _normalize_clip(
            provider_clip,
            label="legacy provider reward clipping",
        )
        if REWARD_CLIP_KEY in reward:
            configured = reward[REWARD_CLIP_KEY]
            configured_transform = _normalize_clip(
                configured,
                label="legacy task.reward.reward_clip",
            )
            configured_active = configured_transform[0] is not None or configured_transform[1]
            provider_active = provider_transform[0] is not None or provider_transform[1]
            if not configured_active and provider_active:
                reward[REWARD_CLIP_KEY] = provider_clip
            elif (
                provider_active
                and not configured_transform[1]
                and configured_transform != provider_transform
            ):
                raise ValueError(
                    "legacy artifact provider and task reward clipping settings disagree"
                )
        elif provider_transform[0] is not None or provider_transform[1]:
            reward[REWARD_CLIP_KEY] = provider_clip

    normalized_task["reward"] = normalize_reward_mapping(
        reward,
        label="legacy artifact task.reward",
    )
    return normalized_args, normalized_task


__all__ = [
    "COMMON_REWARD_KEYS",
    "LEGACY_SIGN_CLIP",
    "PROVIDER_REWARD_TRANSFORM_KEYS",
    "RewardTransform",
    "migrate_legacy_artifact_reward_config",
    "normalize_reward_mapping",
    "normalize_task_reward",
    "reward_transform_from_reward",
]
