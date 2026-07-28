from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from gradlab.policy_registry import (
    MODEL_CLASS_ALGORITHMS,
    PolicyAlgorithmId,
    RUNTIME_POLICY_ALGORITHMS,
)


ACTOR_DISTRIBUTION = "actor_distribution"
STATE_VALUE = "state_value"
ACTION_VALUE = "action_value"
PROGRAM = "program"
SELECTED_ACTION_LOG_PROBABILITY = "selected_action_log_probability"
ENTROPY = "entropy"
ATTRIBUTION = "attribution"


@dataclass(frozen=True)
class PolicyCapabilities:
    algorithm_id: PolicyAlgorithmId
    action_selection_modes: tuple[str, ...]
    default_action_selection_mode: str
    introspection: frozenset[str]

    def payload(self, *, attribution_available: bool = False) -> dict[str, Any]:
        introspection = set(self.introspection)
        if attribution_available:
            introspection.add(ATTRIBUTION)
        return {
            "algorithm_id": self.algorithm_id,
            "action_selection": {
                "supported_modes": list(self.action_selection_modes),
                "default_mode": self.default_action_selection_mode,
            },
            "introspection": sorted(introspection),
        }


POLICY_CAPABILITIES: dict[PolicyAlgorithmId, PolicyCapabilities] = {
    "ppo": PolicyCapabilities(
        algorithm_id="ppo",
        action_selection_modes=("stochastic", "deterministic"),
        default_action_selection_mode="stochastic",
        introspection=frozenset(
            {
                ACTOR_DISTRIBUTION,
                STATE_VALUE,
                SELECTED_ACTION_LOG_PROBABILITY,
                ENTROPY,
            }
        ),
    ),
    "a2c": PolicyCapabilities(
        algorithm_id="a2c",
        action_selection_modes=("stochastic", "deterministic"),
        default_action_selection_mode="stochastic",
        introspection=frozenset(
            {
                ACTOR_DISTRIBUTION,
                STATE_VALUE,
                SELECTED_ACTION_LOG_PROBABILITY,
                ENTROPY,
            }
        ),
    ),
    "dqn": PolicyCapabilities(
        algorithm_id="dqn",
        action_selection_modes=("epsilon_greedy", "greedy"),
        default_action_selection_mode="epsilon_greedy",
        introspection=frozenset({ACTION_VALUE}),
    ),
    "jerk": PolicyCapabilities(
        algorithm_id="jerk",
        action_selection_modes=("program",),
        default_action_selection_mode="program",
        introspection=frozenset({PROGRAM}),
    ),
}


@dataclass(frozen=True)
class PolicyBatchDecision:
    requested_action_selection_mode: str
    effective_action_selection_mode: str
    actions: np.ndarray
    decisions: tuple[Any, ...]


def _model_class_name(model: Any) -> str:
    model_type = type(model)
    return f"{model_type.__module__}.{model_type.__qualname__}"


def infer_policy_algorithm(model: Any) -> PolicyAlgorithmId:
    algorithm = MODEL_CLASS_ALGORITHMS.get(_model_class_name(model))
    if algorithm in RUNTIME_POLICY_ALGORITHMS:
        return algorithm
    if callable(getattr(model, "policy_decisions", None)):
        return "jerk"
    policy = getattr(model, "policy", None)
    if policy is not None and hasattr(policy, "q_net"):
        return "dqn"
    if policy is not None and hasattr(policy, "value_net"):
        # Test doubles and compatible SB3 actor-critic subclasses can use the
        # common adapter even when their precise provenance is unavailable.
        return "ppo"
    raise ValueError(f"unsupported playback policy class: {_model_class_name(model)}")


def normalize_action_selection_mode(
    capabilities: PolicyCapabilities,
    requested_mode: str | None,
) -> tuple[str, str]:
    requested = str(requested_mode or capabilities.default_action_selection_mode).strip()
    effective = requested
    # Explicit compatibility interpretation for protocol-v3 and legacy recipe
    # readers, which represented all policy execution as a stochastic boolean.
    if capabilities.algorithm_id == "jerk" and requested in {"stochastic", "deterministic"}:
        effective = "program"
    elif capabilities.algorithm_id == "dqn" and requested == "stochastic":
        effective = "epsilon_greedy"
    elif capabilities.algorithm_id == "dqn" and requested == "deterministic":
        effective = "greedy"
    if effective not in capabilities.action_selection_modes:
        modes = ", ".join(capabilities.action_selection_modes)
        raise ValueError(
            f"unsupported action-selection mode {requested!r} for "
            f"{capabilities.algorithm_id}; supported: {modes}"
        )
    return requested, effective


class PolicyRuntime:
    """Validated action execution and optional policy introspection."""

    def __init__(
        self,
        model: Any,
        *,
        algorithm_id: PolicyAlgorithmId | None = None,
    ) -> None:
        self.model = model
        inferred = infer_policy_algorithm(model) if algorithm_id is None else algorithm_id
        if inferred not in POLICY_CAPABILITIES:
            raise ValueError(f"unsupported runtime policy algorithm: {inferred}")
        self.capabilities = POLICY_CAPABILITIES[inferred]

    def bind_action_space(self, action_space: Any) -> None:
        bind = getattr(self.model, "bind_action_space", None)
        if callable(bind):
            bind(action_space)
        model_action_space = getattr(self.model, "action_space", None)
        if model_action_space is None:
            try:
                self.model.action_space = action_space
            except Exception:
                pass

    def reset(self, lanes: Any | None = None) -> None:
        reset = getattr(
            self.model,
            "reset_episode" if lanes is None else "reset_lanes",
            None,
        )
        if callable(reset):
            reset() if lanes is None else reset(lanes)

    def decide(
        self,
        observation: Any,
        *,
        action_selection_mode: str | None = None,
    ) -> PolicyBatchDecision:
        requested, effective = normalize_action_selection_mode(
            self.capabilities,
            action_selection_mode,
        )
        if self.capabilities.algorithm_id in {"ppo", "a2c"}:
            from gradlab.play_debug import actor_critic_policy_decisions

            decisions = actor_critic_policy_decisions(
                self.model,
                observation,
                deterministic=effective == "deterministic",
            )
        elif self.capabilities.algorithm_id == "dqn":
            from gradlab.play_debug import dqn_policy_decisions

            decisions = dqn_policy_decisions(
                self.model,
                observation,
                epsilon_greedy=effective == "epsilon_greedy",
            )
        else:
            custom = getattr(self.model, "policy_decisions", None)
            if not callable(custom):
                raise RuntimeError("programmatic policy has no decision adapter")
            decisions = tuple(custom(observation, action_selection_mode=effective))
        if not decisions:
            raise RuntimeError("policy runtime produced no actions")
        decisions = tuple(
            replace(
                decision,
                requested_action_selection_mode=requested,
                action_selection_mode=effective,
            )
            for decision in decisions
        )
        actions = np.asarray(
            [np.asarray(decision.executed_action) for decision in decisions]
        )
        if all(np.asarray(decision.executed_action).ndim == 0 for decision in decisions):
            actions = actions.reshape(-1)
        return PolicyBatchDecision(
            requested_action_selection_mode=requested,
            effective_action_selection_mode=effective,
            actions=actions,
            decisions=decisions,
        )

    def inspect(self, observation: Any) -> PolicyBatchDecision:
        mode = {
            "ppo": "deterministic",
            "a2c": "deterministic",
            "dqn": "greedy",
            "jerk": "program",
        }[self.capabilities.algorithm_id]
        custom = getattr(self.model, "inspect_policy_decisions", None)
        if self.capabilities.algorithm_id == "jerk" and callable(custom):
            decisions = tuple(custom(observation, action_selection_mode=mode))
            return PolicyBatchDecision(mode, mode, np.asarray([]), decisions)
        return self.decide(observation, action_selection_mode=mode)


def bind_policy_action_space(model: Any, action_space: Any) -> None:
    bind = getattr(model, "bind_action_space", None)
    if callable(bind):
        bind(action_space)
    if getattr(model, "action_space", None) is None:
        try:
            model.action_space = action_space
        except Exception:
            pass


def reset_policy_state(model: Any, lanes: Any | None = None) -> None:
    reset = getattr(model, "reset_episode" if lanes is None else "reset_lanes", None)
    if callable(reset):
        reset() if lanes is None else reset(lanes)


def policy_capabilities(
    model: Any,
    *,
    attribution_available: bool = False,
) -> dict[str, Any]:
    return PolicyRuntime(model).capabilities.payload(
        attribution_available=attribution_available,
    )


def policy_action(
    model: Any,
    observation: Any,
    *,
    action_selection_mode: str | None = None,
) -> tuple[np.ndarray, PolicyBatchDecision]:
    result = PolicyRuntime(model).decide(
        observation,
        action_selection_mode=action_selection_mode,
    )
    return result.actions, result


__all__ = [
    "ACTION_VALUE",
    "ACTOR_DISTRIBUTION",
    "ATTRIBUTION",
    "ENTROPY",
    "PROGRAM",
    "POLICY_CAPABILITIES",
    "PolicyBatchDecision",
    "PolicyCapabilities",
    "PolicyRuntime",
    "SELECTED_ACTION_LOG_PROBABILITY",
    "STATE_VALUE",
    "bind_policy_action_space",
    "infer_policy_algorithm",
    "normalize_action_selection_mode",
    "policy_action",
    "policy_capabilities",
    "reset_policy_state",
]
