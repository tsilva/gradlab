from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
)


ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")


@dataclass(frozen=True)
class PolicyDecision:
    distribution_kind: str
    raw_action: np.ndarray
    executed_action: np.ndarray
    value: float
    log_probability: float
    entropy: float | None
    mode: np.ndarray
    probabilities: np.ndarray | None = None
    component_probabilities: tuple[np.ndarray, ...] = ()
    mean: np.ndarray | None = None
    stddev: np.ndarray | None = None
    sampled: bool = True

    @property
    def selected_discrete_action(self) -> int | None:
        if self.probabilities is None or self.raw_action.size != 1:
            return None
        return int(self.raw_action.reshape(-1)[0])

    @property
    def selected_probability(self) -> float | None:
        action = self.selected_discrete_action
        if action is None:
            return None
        return float(self.probabilities[action])

    @property
    def selected_rank(self) -> int | None:
        probability = self.selected_probability
        if probability is None:
            return None
        return 1 + int(np.count_nonzero(self.probabilities > probability))


def _as_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _postprocess_action(policy: Any, raw_action: np.ndarray) -> np.ndarray:
    action = np.asarray(raw_action).reshape((-1, *policy.action_space.shape))
    if isinstance(policy.action_space, gym.spaces.Box):
        if policy.squash_output:
            action = policy.unscale_action(action)
        else:
            action = np.clip(action, policy.action_space.low, policy.action_space.high)
    return np.asarray(action)


def _decision_from_distribution(
    policy: Any,
    distribution: Any,
    *,
    raw_tensor: torch.Tensor,
    value_tensor: torch.Tensor,
    log_probability_tensor: torch.Tensor,
    sampled: bool,
) -> PolicyDecision:
    entropy_tensor = distribution.entropy()
    mode_tensor = distribution.mode()
    raw_action = _as_numpy(raw_tensor).reshape((-1, *policy.action_space.shape))
    executed_action = _postprocess_action(policy, raw_action)
    entropy = None if entropy_tensor is None else float(_as_numpy(entropy_tensor).reshape(-1)[0])
    common = {
        "raw_action": raw_action[0].copy(),
        "executed_action": executed_action[0].copy(),
        "value": float(_as_numpy(value_tensor).reshape(-1)[0]),
        "log_probability": float(_as_numpy(log_probability_tensor).reshape(-1)[0]),
        "entropy": entropy,
        "mode": _as_numpy(mode_tensor)[0].copy(),
        "sampled": sampled,
    }
    if isinstance(distribution, CategoricalDistribution):
        return PolicyDecision(
            **common,
            distribution_kind="categorical",
            probabilities=_as_numpy(distribution.distribution.probs)[0].copy(),
        )
    if isinstance(distribution, MultiCategoricalDistribution):
        return PolicyDecision(
            **common,
            distribution_kind="multi_categorical",
            component_probabilities=tuple(
                _as_numpy(component.probs)[0].copy() for component in distribution.distribution
            ),
        )
    if isinstance(distribution, BernoulliDistribution):
        return PolicyDecision(
            **common,
            distribution_kind="bernoulli",
            component_probabilities=(_as_numpy(distribution.distribution.probs)[0].copy(),),
        )
    if isinstance(distribution, (DiagGaussianDistribution, StateDependentNoiseDistribution)):
        return PolicyDecision(
            **common,
            distribution_kind="gaussian",
            mean=_as_numpy(distribution.distribution.mean)[0].copy(),
            stddev=_as_numpy(distribution.distribution.stddev)[0].copy(),
        )
    raise TypeError(f"unsupported PPO distribution {type(distribution).__name__}")


def sample_policy_decision(model: Any, model_obs: Any) -> PolicyDecision:
    """Sample once from PPO and describe that same state without another sample."""

    custom = getattr(model, "sample_policy_decision", None)
    if callable(custom):
        return custom(model_obs)

    policy = model.policy
    policy.set_training_mode(False)
    obs_tensor, _vectorized = policy.obs_to_tensor(model_obs)
    with torch.no_grad():
        raw_tensor, value_tensor, log_probability_tensor = policy.forward(
            obs_tensor,
            deterministic=False,
        )
        distribution = policy.get_distribution(obs_tensor)
    return _decision_from_distribution(
        policy,
        distribution,
        raw_tensor=raw_tensor,
        value_tensor=value_tensor,
        log_probability_tensor=log_probability_tensor,
        sampled=True,
    )


def inspect_policy(model: Any, model_obs: Any) -> PolicyDecision:
    """Inspect a state using its mode without sampling or changing policy RNG."""

    custom = getattr(model, "inspect_policy_decision", None)
    if callable(custom):
        return custom(model_obs)

    policy = model.policy
    policy.set_training_mode(False)
    obs_tensor, _vectorized = policy.obs_to_tensor(model_obs)
    with torch.no_grad():
        distribution = policy.get_distribution(obs_tensor)
        mode_tensor = distribution.mode()
        value_tensor = policy.predict_values(obs_tensor)
        log_probability_tensor = distribution.log_prob(mode_tensor)
    return _decision_from_distribution(
        policy,
        distribution,
        raw_tensor=mode_tensor,
        value_tensor=value_tensor,
        log_probability_tensor=log_probability_tensor,
        sampled=False,
    )


def _format_number(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.5g}"
    return str(value)


def format_action(value: Any) -> str:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return _format_number(arr.item())
    return np.array2string(arr, precision=4, separator=",", threshold=16)


def _input_leaf_lines(name: str, value: Any) -> list[str]:
    arr = np.asarray(value)
    if arr.dtype == object:
        return [f"{name}: {value!r}"]
    if arr.size <= 32:
        return [f"{name}: shape={arr.shape} dtype={arr.dtype} values={format_action(arr)}"]
    digest = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]
    finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.number) else np.array([])
    value_range = (
        f" min={_format_number(finite.min())} max={_format_number(finite.max())}"
        if finite.size
        else ""
    )
    return [f"{name}: shape={arr.shape} dtype={arr.dtype}{value_range} sha256={digest}"]


def model_input_lines(model_obs: Any) -> list[str]:
    if isinstance(model_obs, Mapping):
        lines: list[str] = []
        for name, value in model_obs.items():
            lines.extend(_input_leaf_lines(str(name), value))
        return lines
    return _input_leaf_lines("observation", model_obs)
