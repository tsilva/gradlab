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
    raw_action: np.ndarray
    executed_action: np.ndarray
    action_selection_mode: str
    requested_action_selection_mode: str | None = None
    distribution_kind: str | None = None
    value: float | None = None
    log_probability: float | None = None
    entropy: float | None = None
    mode: np.ndarray | None = None
    probabilities: np.ndarray | None = None
    component_probabilities: tuple[np.ndarray, ...] = ()
    mean: np.ndarray | None = None
    stddev: np.ndarray | None = None
    program: Mapping[str, Any] | None = None
    route: Mapping[str, Any] | None = None
    sampled: bool | None = None

    @property
    def selected_discrete_action(self) -> int | None:
        if self.raw_action.size != 1:
            return None
        return int(self.raw_action.reshape(-1)[0])

    @property
    def selected_probability(self) -> float | None:
        action = self.selected_discrete_action
        if (
            self.probabilities is None
            or action is None
            or not 0 <= action < self.probabilities.size
        ):
            return None
        return float(self.probabilities.reshape(-1)[action])

    @property
    def selected_rank(self) -> int | None:
        probability = self.selected_probability
        if self.probabilities is None or probability is None:
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


def _decisions_from_distribution(
    policy: Any,
    distribution: Any,
    *,
    raw_tensor: torch.Tensor,
    value_tensor: torch.Tensor | None,
    log_probability_tensor: torch.Tensor,
    sampled: bool,
    action_selection_mode: str,
) -> tuple[PolicyDecision, ...]:
    entropy_tensor = distribution.entropy()
    mode_tensor = distribution.mode()
    raw_action = _as_numpy(raw_tensor).reshape((-1, *policy.action_space.shape))
    executed_action = _postprocess_action(policy, raw_action)
    values = None if value_tensor is None else _as_numpy(value_tensor).reshape(-1)
    log_probabilities = _as_numpy(log_probability_tensor).reshape(-1)
    entropies = None if entropy_tensor is None else _as_numpy(entropy_tensor).reshape(-1)
    modes = _as_numpy(mode_tensor).reshape((-1, *policy.action_space.shape))

    distribution_kind: str
    probabilities: np.ndarray | None = None
    component_probabilities: tuple[np.ndarray, ...] = ()
    mean: np.ndarray | None = None
    stddev: np.ndarray | None = None
    if isinstance(distribution, CategoricalDistribution):
        distribution_kind = "categorical"
        probabilities = _as_numpy(distribution.distribution.probs)
    elif isinstance(distribution, MultiCategoricalDistribution):
        distribution_kind = "multi_categorical"
        component_probabilities = tuple(
            _as_numpy(component.probs) for component in distribution.distribution
        )
    elif isinstance(distribution, BernoulliDistribution):
        distribution_kind = "bernoulli"
        component_probabilities = (_as_numpy(distribution.distribution.probs),)
    elif isinstance(distribution, (DiagGaussianDistribution, StateDependentNoiseDistribution)):
        distribution_kind = "gaussian"
        mean = _as_numpy(distribution.distribution.mean)
        stddev = _as_numpy(distribution.distribution.stddev)
    else:
        raise TypeError(f"unsupported actor-critic distribution {type(distribution).__name__}")

    decisions: list[PolicyDecision] = []
    for lane in range(raw_action.shape[0]):
        decisions.append(
            PolicyDecision(
                raw_action=raw_action[lane].copy(),
                executed_action=executed_action[lane].copy(),
                action_selection_mode=action_selection_mode,
                distribution_kind=distribution_kind,
                value=None if values is None else float(values[lane]),
                log_probability=float(log_probabilities[lane]),
                entropy=None if entropies is None else float(entropies[lane]),
                mode=modes[lane].copy(),
                probabilities=(None if probabilities is None else probabilities[lane].copy()),
                component_probabilities=tuple(
                    component[lane].copy() for component in component_probabilities
                ),
                mean=None if mean is None else mean[lane].copy(),
                stddev=None if stddev is None else stddev[lane].copy(),
                sampled=sampled,
            )
        )
    return tuple(decisions)


def _actor_critic_policy_decisions(
    model: Any,
    model_obs: Any,
    *,
    deterministic: bool,
) -> tuple[PolicyDecision, ...]:
    """Run an SB3 actor-critic policy once and describe every lane."""

    policy = model.policy
    policy.set_training_mode(False)
    obs_tensor, _vectorized = policy.obs_to_tensor(model_obs)
    with torch.no_grad():
        role_decision = getattr(policy, "decision_distribution_and_value", None)
        if callable(role_decision):
            distribution, values = role_decision(obs_tensor)
        else:
            features = policy.extract_features(obs_tensor)
            if policy.share_features_extractor:
                latent_pi, latent_vf = policy.mlp_extractor(features)
            else:
                pi_features, vf_features = features
                latent_pi = policy.mlp_extractor.forward_actor(pi_features)
                latent_vf = policy.mlp_extractor.forward_critic(vf_features)
            values = policy.value_net(latent_vf)
            distribution = policy._get_action_dist_from_latent(latent_pi)
        raw_tensor = distribution.get_actions(deterministic=deterministic)
        log_probability_tensor = distribution.log_prob(raw_tensor)
    return _decisions_from_distribution(
        policy,
        distribution,
        raw_tensor=raw_tensor,
        value_tensor=values,
        log_probability_tensor=log_probability_tensor,
        sampled=not deterministic,
        action_selection_mode="deterministic" if deterministic else "stochastic",
    )


def actor_critic_policy_decisions(
    model: Any,
    model_obs: Any,
    *,
    deterministic: bool,
) -> tuple[PolicyDecision, ...]:
    return _actor_critic_policy_decisions(
        model,
        model_obs,
        deterministic=deterministic,
    )


def actor_critic_policy_actions(
    model: Any,
    model_obs: Any,
    *,
    deterministic: bool,
) -> tuple[PolicyDecision, ...]:
    """Choose actor actions without running critic or distribution diagnostics."""

    policy = model.policy
    policy.set_training_mode(False)
    obs_tensor, _vectorized = policy.obs_to_tensor(model_obs)
    with torch.no_grad():
        distribution = policy.get_distribution(obs_tensor)
        raw_tensor = distribution.get_actions(deterministic=deterministic)
    raw_actions = _as_numpy(raw_tensor).reshape((-1, *policy.action_space.shape))
    executed_actions = _postprocess_action(policy, raw_actions)
    mode = "deterministic" if deterministic else "stochastic"
    return tuple(
        PolicyDecision(
            raw_action=raw_actions[lane].copy(),
            executed_action=executed_actions[lane].copy(),
            action_selection_mode=mode,
            sampled=not deterministic,
        )
        for lane in range(raw_actions.shape[0])
    )


def sample_policy_decision(model: Any, model_obs: Any) -> PolicyDecision:
    """Sample once from an actor-critic policy and describe that same decision."""

    custom = getattr(model, "sample_policy_decision", None)
    if callable(custom):
        return custom(model_obs)

    return _actor_critic_policy_decisions(model, model_obs, deterministic=False)[0]


def inspect_policy(model: Any, model_obs: Any) -> PolicyDecision:
    """Inspect a state using its mode without sampling or changing policy RNG."""

    custom = getattr(model, "inspect_policy_decision", None)
    if callable(custom):
        return custom(model_obs)

    return _actor_critic_policy_decisions(model, model_obs, deterministic=True)[0]


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
