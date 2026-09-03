from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from gradlab.policy_registry import (
    ACTOR_DISTRIBUTION,
    ATTRIBUTION,
    ENTROPY,
    POLICY_ALGORITHM_SPECS,
    PROGRAM,
    ROUTE as ROUTE,
    SELECTED_ACTION_LOG_PROBABILITY,
    STATE_VALUE,
    PolicyAlgorithmId,
)


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
    algorithm_id: PolicyCapabilities(
        algorithm_id=algorithm_id,
        action_selection_modes=spec.action_selection_modes,
        default_action_selection_mode=spec.default_action_selection_mode,
        introspection=spec.introspection,
    )
    for algorithm_id, spec in POLICY_ALGORITHM_SPECS.items()
    if spec.runtime_family is not None
}


@dataclass(frozen=True)
class PolicyBatchDecision:
    requested_action_selection_mode: str
    effective_action_selection_mode: str
    actions: np.ndarray
    decisions: tuple[Any, ...]


def normalize_action_selection_mode(
    capabilities: PolicyCapabilities,
    requested_mode: str | None,
) -> tuple[str, str]:
    requested = str(requested_mode or capabilities.default_action_selection_mode).strip()
    if requested not in capabilities.action_selection_modes:
        modes = ", ".join(capabilities.action_selection_modes)
        raise ValueError(
            f"unsupported action-selection mode {requested!r} for "
            f"{capabilities.algorithm_id}; supported: {modes}"
        )
    return requested, requested


class PolicyRuntime:
    """Validated action execution and optional policy introspection."""

    def __init__(
        self,
        model: Any,
        *,
        algorithm_id: PolicyAlgorithmId,
    ) -> None:
        self.model = model
        if algorithm_id not in POLICY_CAPABILITIES:
            raise ValueError(f"unsupported runtime policy algorithm: {algorithm_id}")
        self.capabilities = POLICY_CAPABILITIES[algorithm_id]

    def bind_action_space(
        self,
        action_space: Any,
        action_contract: Mapping[str, Any] | None = None,
    ) -> None:
        bind_policy_action_space(self.model, action_space, action_contract)

    def reset(self, lanes: Any | None = None) -> None:
        reset_policy_state(self.model, lanes)

    def decide(
        self,
        observation: Any,
        *,
        action_selection_mode: str | None = None,
        execution_context: Any | None = None,
        include_diagnostics: bool = True,
    ) -> PolicyBatchDecision:
        requested, effective = normalize_action_selection_mode(
            self.capabilities,
            action_selection_mode,
        )
        if self.capabilities.algorithm_id in {"ppo", "a2c"}:
            from gradlab.play_debug import (
                actor_critic_policy_actions,
                actor_critic_policy_decisions,
            )

            decision_function = (
                actor_critic_policy_decisions
                if include_diagnostics
                else actor_critic_policy_actions
            )
            decisions = decision_function(
                self.model, observation, deterministic=effective == "deterministic"
            )
        else:
            custom = getattr(self.model, "policy_decisions", None)
            if not callable(custom):
                raise RuntimeError("programmatic policy has no decision adapter")
            decisions = tuple(
                custom(
                    observation,
                    action_selection_mode=effective,
                    **(
                        {"execution_context": execution_context}
                        if self.capabilities.algorithm_id == "cell-graph"
                        else {}
                    ),
                )
            )
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
        actions = np.asarray([np.asarray(decision.executed_action) for decision in decisions])
        if all(np.asarray(decision.executed_action).ndim == 0 for decision in decisions):
            actions = actions.reshape(-1)
        return PolicyBatchDecision(
            requested_action_selection_mode=requested,
            effective_action_selection_mode=effective,
            actions=actions,
            decisions=decisions,
        )

    def inspect(
        self,
        observation: Any,
        *,
        execution_context: Any | None = None,
    ) -> PolicyBatchDecision:
        mode = {
            "ppo": "deterministic",
            "a2c": "deterministic",
            "action-program": "program",
            "cell-graph": "route",
        }[self.capabilities.algorithm_id]
        custom = getattr(self.model, "inspect_policy_decisions", None)
        if self.capabilities.algorithm_id in {"action-program", "cell-graph"} and callable(custom):
            decisions = tuple(
                custom(
                    observation,
                    action_selection_mode=mode,
                    **(
                        {"execution_context": execution_context}
                        if self.capabilities.algorithm_id == "cell-graph"
                        else {}
                    ),
                )
            )
            return PolicyBatchDecision(mode, mode, np.asarray([]), decisions)
        return self.decide(
            observation,
            action_selection_mode=mode,
            execution_context=execution_context,
        )

    def state_values(self, observation: Any) -> np.ndarray:
        if STATE_VALUE not in self.capabilities.introspection:
            raise RuntimeError(
                f"{self.capabilities.algorithm_id} does not expose a state-value critic"
            )
        from gradlab.play_debug import actor_critic_state_values

        values = actor_critic_state_values(self.model, observation)
        if not np.isfinite(values).all():
            raise RuntimeError("policy runtime produced a non-finite state value")
        return values


def bind_policy_action_space(
    model: Any,
    action_space: Any,
    action_contract: Mapping[str, Any] | None = None,
) -> None:
    targets = [model]
    policy = getattr(model, "policy", None)
    if policy is not None and policy is not model:
        targets.append(policy)
    for target in targets:
        bind_contract = getattr(target, "bind_action_contract", None)
        if callable(bind_contract) and action_contract is not None:
            bind_contract(action_contract)
        bind = getattr(target, "bind_action_space", None)
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


__all__ = [
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
    "normalize_action_selection_mode",
    "reset_policy_state",
]
