"""Compile and execute provider-neutral conditional action overwrites."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import gymnasium as gym
import numpy as np

from gradlab.task_kernels import BoundTaskKernel, SignalBindings


CONDITIONAL_ACTION_OVERRIDE_KEY = "conditional_overrides"
SUPPORTED_CONDITIONAL_ACTION_OPERATIONS = frozenset({"equals"})


def normalize_conditional_action_overrides(
    action: Mapping[str, Any],
    signals: Mapping[str, Any],
    *,
    label: str = "task.action",
) -> list[dict[str, Any]]:
    """Return one canonical, bounded conditional-overwrite declaration."""

    raw_rules = action.get(CONDITIONAL_ACTION_OVERRIDE_KEY, [])
    if isinstance(raw_rules, str | bytes) or not isinstance(raw_rules, Sequence):
        raise ValueError(f"{label}.{CONDITIONAL_ACTION_OVERRIDE_KEY} must be a list")
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw_rule in enumerate(raw_rules):
        rule_label = f"{label}.{CONDITIONAL_ACTION_OVERRIDE_KEY}[{index}]"
        if not isinstance(raw_rule, Mapping):
            raise ValueError(f"{rule_label} must be an object")
        extra = sorted(set(raw_rule) - {"id", "when", "replace_with"})
        if extra:
            raise ValueError(f"{rule_label} has unexpected keys: {extra}")
        rule_id = raw_rule.get("id")
        if not isinstance(rule_id, str) or not rule_id.strip():
            raise ValueError(f"{rule_label}.id must be a non-empty string")
        rule_id = rule_id.strip()
        if rule_id in seen_ids:
            raise ValueError(f"{label}.{CONDITIONAL_ACTION_OVERRIDE_KEY} duplicates id {rule_id!r}")
        seen_ids.add(rule_id)

        condition = raw_rule.get("when")
        if not isinstance(condition, Mapping):
            raise ValueError(f"{rule_label}.when must be an object")
        extra = sorted(set(condition) - {"signal", "operation", "value"})
        if extra:
            raise ValueError(f"{rule_label}.when has unexpected keys: {extra}")
        signal = condition.get("signal")
        if not isinstance(signal, str) or signal not in signals:
            raise ValueError(f"{rule_label}.when.signal references unknown signal {signal!r}")
        operation = condition.get("operation")
        if operation not in SUPPORTED_CONDITIONAL_ACTION_OPERATIONS:
            raise ValueError(f"{rule_label}.when.operation is unsupported: {operation!r}")
        value = condition.get("value")
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ValueError(f"{rule_label}.when.value must be a number")
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f"{rule_label}.when.value must be finite")

        replacement = raw_rule.get("replace_with")
        if not isinstance(replacement, Mapping):
            raise ValueError(f"{rule_label}.replace_with must be an object")
        extra = sorted(set(replacement) - {"semantic_id"})
        if extra:
            raise ValueError(f"{rule_label}.replace_with has unexpected keys: {extra}")
        semantic_id = replacement.get("semantic_id")
        if not isinstance(semantic_id, str) or not semantic_id.strip():
            raise ValueError(f"{rule_label}.replace_with.semantic_id must be a non-empty string")

        normalized.append(
            {
                "id": rule_id,
                "when": {
                    "signal": signal,
                    "operation": operation,
                    "value": value,
                },
                "replace_with": {"semantic_id": semantic_id.strip()},
            }
        )
    return normalized


def _compiled_rules(action_contract: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    policy = action_contract.get("policy")
    raw = policy.get(CONDITIONAL_ACTION_OVERRIDE_KEY, ()) if isinstance(policy, Mapping) else ()
    if not isinstance(raw, list | tuple):
        raise ValueError("runtime action contract conditional overrides must be a list")
    return tuple(raw)


def _condition_value_for_dtype(value: int | float, dtype: np.dtype, *, rule_id: str) -> Any:
    if np.issubdtype(dtype, np.bool_):
        if value not in (0, 1):
            raise ValueError(
                f"conditional action override {rule_id!r} value {value!r} "
                "is not representable by a boolean signal"
            )
    elif np.issubdtype(dtype, np.integer):
        bounds = np.iinfo(dtype)
        if value < bounds.min or value > bounds.max or int(value) != value:
            raise ValueError(
                f"conditional action override {rule_id!r} value {value!r} "
                f"is not representable by signal dtype {dtype}"
            )
    elif not np.issubdtype(dtype, np.floating):
        raise ValueError(
            f"conditional action override {rule_id!r} requires a numeric signal, got {dtype}"
        )
    target = np.asarray(value, dtype=dtype).item()
    if isinstance(target, float) and not np.isfinite(target):
        raise ValueError(
            f"conditional action override {rule_id!r} value {value!r} "
            f"is not representable by signal dtype {dtype}"
        )
    return target


class ConditionalActionTaskKernel:
    """Apply ordered per-lane overwrites before the task action codec."""

    def __init__(
        self,
        kernel: BoundTaskKernel,
        descriptor: Any,
        signals: Mapping[str, Any],
        action_contract: Mapping[str, Any],
    ) -> None:
        self.kernel = kernel
        self.num_envs = int(kernel.num_envs)
        self._rules = _compiled_rules(action_contract)
        referenced = {
            str(rule["when"]["signal"]): signals[str(rule["when"]["signal"])]
            for rule in self._rules
        }
        self._bindings = SignalBindings(descriptor, referenced, self.num_envs)
        self._condition_values = []
        for rule in self._rules:
            signal = str(rule["when"]["signal"])
            dtypes = self._bindings.scalar_source_dtypes(signal, require_reset=True)
            if len(dtypes) != 1:
                raise ValueError(
                    f"conditional action override signal {signal!r} must use one provider source"
                )
            self._condition_values.append(
                _condition_value_for_dtype(
                    rule["when"]["value"],
                    dtypes[0],
                    rule_id=str(rule["id"]),
                )
            )
        self._active_rule = np.full(self.num_envs, -1, dtype=np.int32)
        self._last_rule = np.full(self.num_envs, -1, dtype=np.int32)
        self._last_effective_actions: Any = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.kernel, name)

    def _refresh(self, signals: Mapping[str, Any], mask: np.ndarray) -> None:
        selected = np.asarray(mask, dtype=np.bool_)
        if selected.shape != (self.num_envs,):
            raise ValueError(f"conditional action mask must have shape ({self.num_envs},)")
        self._active_rule[selected] = -1
        unmatched = selected.copy()
        for index, rule in enumerate(self._rules):
            if not np.any(unmatched):
                break
            condition = rule["when"]
            values = self._bindings.scalar(str(condition["signal"]), signals, mask=unmatched)
            target = self._condition_values[index]
            matches = unmatched & np.equal(values, target)
            self._active_rule[matches] = index
            unmatched[matches] = False

    def map_actions(self, actions: Any) -> Any:
        self._last_rule[:] = self._active_rule
        matched = self._active_rule >= 0
        if not np.any(matched):
            self._last_effective_actions = actions
            return self.kernel.map_actions(actions)
        effective = np.asarray(actions).copy()
        if isinstance(self.action_space, gym.spaces.Discrete):
            if effective.reshape(-1).shape != (self.num_envs,):
                raise ValueError(f"expected {self.num_envs} policy actions, got {effective.shape}")
            effective = effective.reshape(self.num_envs)
        elif isinstance(self.action_space, gym.spaces.MultiDiscrete):
            expected = (self.num_envs, *self.action_space.shape)
            if effective.shape != expected:
                raise ValueError(
                    f"expected policy actions with shape {expected}, got {effective.shape}"
                )
        else:
            raise ValueError(
                "conditional action overrides require Discrete or MultiDiscrete policy actions"
            )
        for index, rule in enumerate(self._rules):
            lanes = self._active_rule == index
            if not np.any(lanes):
                continue
            target = rule["replace_with"]["value"]
            effective[lanes] = target
        self._last_effective_actions = effective
        return self.kernel.map_actions(effective)

    def effective_action(self, lane: int) -> Any:
        if self._last_effective_actions is None:
            raise RuntimeError("conditional action attribution is unavailable before step")
        value = np.asarray(self._last_effective_actions)[int(lane)]
        return value.item() if isinstance(value, np.generic) else deepcopy(value)

    def action_override_rule_id(self, lane: int) -> str | None:
        index = int(self._last_rule[int(lane)])
        return None if index < 0 else str(self._rules[index]["id"])

    def encode_observations(self, observations: Any) -> Any:
        return self.kernel.encode_observations(observations)

    def process(
        self,
        native_rewards: np.ndarray,
        provider_terminated: np.ndarray,
        provider_truncated: np.ndarray,
        signals: Mapping[str, Any],
    ) -> Any:
        result = self.kernel.process(
            native_rewards,
            provider_terminated,
            provider_truncated,
            signals,
        )
        self._refresh(signals, np.ones(self.num_envs, dtype=np.bool_))
        return result

    def observe_step(self, signals: Mapping[str, Any]) -> None:
        """Refresh action conditions for an adapter that owns task processing."""

        self._refresh(signals, np.ones(self.num_envs, dtype=np.bool_))

    def on_reset(
        self,
        reset_observations: Any,
        reset_signals: Mapping[str, Any],
        mask: np.ndarray,
    ) -> None:
        self.kernel.on_reset(reset_observations, reset_signals, mask)
        self._refresh(reset_signals, mask)

    def validate_archive_signal(self, semantic_name: str) -> None:
        self.kernel.validate_archive_signal(semantic_name)

    def archive_signal_values(
        self,
        semantic_name: str,
        signals: Mapping[str, Any],
        *,
        mask: np.ndarray,
    ) -> np.ndarray:
        return self.kernel.archive_signal_values(semantic_name, signals, mask=mask)

    def capture_lane_states(self, mask: np.ndarray) -> Any:
        return self.kernel.capture_lane_states(mask)

    def restore_lane_states(self, states: Any, mask: np.ndarray) -> None:
        self.kernel.restore_lane_states(states, mask)


def with_conditional_action_overrides(
    kernel: BoundTaskKernel,
    descriptor: Any,
    signals: Mapping[str, Any],
    action_contract: Mapping[str, Any],
) -> BoundTaskKernel:
    if not _compiled_rules(action_contract):
        return kernel
    return ConditionalActionTaskKernel(kernel, descriptor, signals, action_contract)


class DeviceConditionalActionResolver:
    """Torch-resident overwrite resolver for the scalar discrete device runtime."""

    def __init__(
        self,
        *,
        action_space: gym.Space,
        signals: Mapping[str, Any],
        action_contract: Mapping[str, Any],
        device_signal_names: Sequence[str],
        num_envs: int,
        device: Any,
    ) -> None:
        import torch

        self._rules = _compiled_rules(action_contract)
        self.num_envs = int(num_envs)
        self.device = device
        if self._rules and not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError(
                "device conditional action overrides require a Discrete policy action space"
            )
        source_indices = {str(name): index for index, name in enumerate(device_signal_names)}
        self._signal_indices: list[int] = []
        self._targets: list[int] = []
        for rule in self._rules:
            semantic_name = str(rule["when"]["signal"])
            source = signals[semantic_name]
            if not isinstance(source, str):
                raise ValueError(
                    "device conditional action overrides require scalar single-source signals"
                )
            if source not in source_indices:
                raise ValueError(
                    f"device runtime does not expose conditional action signal {source!r}"
                )
            self._signal_indices.append(source_indices[source])
            self._targets.append(int(rule["replace_with"]["value"]))
        self._active_rule = torch.full(
            (self.num_envs,),
            -1,
            dtype=torch.int64,
            device=device,
        )

    def observe(self, signal_values: Any) -> None:
        import torch

        if not self._rules:
            return
        if not isinstance(signal_values, torch.Tensor):
            raise TypeError("device conditional action signals must remain torch tensors")
        if signal_values.ndim != 2 or signal_values.shape[0] != self.num_envs:
            raise ValueError("device conditional action signals have an invalid shape")
        self._active_rule.fill_(-1)
        unmatched = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        for index, (rule, signal_index) in enumerate(
            zip(self._rules, self._signal_indices, strict=True)
        ):
            matches = unmatched & (signal_values[:, signal_index] == rule["when"]["value"])
            self._active_rule.masked_fill_(matches, index)
            unmatched.logical_and_(~matches)

    def resolve(self, actions: Any) -> Any:
        import torch

        resolved = actions.to(device=self.device, dtype=torch.int64).reshape(self.num_envs)
        if not self._rules:
            return resolved
        resolved = resolved.clone()
        for index, target in enumerate(self._targets):
            resolved.masked_fill_(self._active_rule == index, target)
        return resolved


__all__ = [
    "CONDITIONAL_ACTION_OVERRIDE_KEY",
    "ConditionalActionTaskKernel",
    "DeviceConditionalActionResolver",
    "normalize_conditional_action_overrides",
    "with_conditional_action_overrides",
]
