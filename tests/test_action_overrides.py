from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from gradlab.action_contract import compile_runtime_action_contract
from gradlab.action_overrides import with_conditional_action_overrides
from gradlab.batch_runtime import ProviderDescriptor, SignalSpec
from gradlab.task_kernels import IdentityTaskDefinition


def _descriptor() -> ProviderDescriptor:
    return ProviderDescriptor(
        provider_id="env-breakoutatari2600-turbo-native",
        native_observation_space=gym.spaces.Box(0, 255, shape=(4, 84, 84), dtype=np.uint8),
        native_action_space=gym.spaces.Discrete(4),
        signal_schema={"ball_y": SignalSpec("ball_y", np.int32)},
        action_mode="custom_discrete",
        action_preset="simple",
        action_table=((), ("BUTTON",), ("RIGHT",), ("LEFT",)),
        action_meanings=("noop", "button", "right", "left"),
        action_table_hash="a" * 64,
        action_buttons=("BUTTON", "RIGHT", "LEFT"),
    )


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        env_provider="env-breakoutatari2600-turbo-native",
        game="Breakout-Atari2600-v0",
        env_args={"use_restricted_actions": "simple"},
        task={
            "action": {
                "set": "native",
                "conditional_overrides": [
                    {
                        "id": "auto_serve",
                        "when": {"signal": "ball_y", "operation": "equals", "value": 0},
                        "replace_with": {"semantic_id": "button"},
                    }
                ],
            },
            "signals": {"ball_y": "ball_y"},
        },
    )


def test_conditional_override_resolves_semantic_target_and_preserves_selected_actions() -> None:
    descriptor = _descriptor()
    config = _config()
    kernel = IdentityTaskDefinition(signals=config.task["signals"]).bind(descriptor, 2)
    contract = compile_runtime_action_contract(config, descriptor, kernel.action_space)
    kernel = with_conditional_action_overrides(
        kernel,
        descriptor,
        config.task["signals"],
        contract,
    )
    selected = np.asarray([3, 2], dtype=np.int64)

    kernel.on_reset(
        np.zeros((2, 4, 84, 84), dtype=np.uint8),
        {"ball_y": np.asarray([0, 54], dtype=np.int32)},
        np.ones(2, dtype=np.bool_),
    )
    native = kernel.map_actions(selected)

    np.testing.assert_array_equal(selected, [3, 2])
    np.testing.assert_array_equal(native, [1, 2])
    assert kernel.effective_action(0) == 1
    assert kernel.effective_action(1) == 2
    assert kernel.action_override_rule_id(0) == "auto_serve"
    assert kernel.action_override_rule_id(1) is None
    assert contract["policy"]["conditional_overrides"][0]["replace_with"] == {
        "semantic_id": "button",
        "value": 1,
    }


def test_conditional_override_refreshes_from_step_and_masked_reset_signals() -> None:
    descriptor = _descriptor()
    config = _config()
    kernel = IdentityTaskDefinition(signals=config.task["signals"]).bind(descriptor, 2)
    contract = compile_runtime_action_contract(config, descriptor, kernel.action_space)
    kernel = with_conditional_action_overrides(
        kernel,
        descriptor,
        config.task["signals"],
        contract,
    )
    observations = np.zeros((2, 4, 84, 84), dtype=np.uint8)
    kernel.on_reset(
        observations,
        {"ball_y": np.asarray([54, 54], dtype=np.int32)},
        np.ones(2, dtype=np.bool_),
    )
    kernel.process(
        np.zeros(2, dtype=np.float32),
        np.zeros(2, dtype=np.bool_),
        np.zeros(2, dtype=np.bool_),
        {"ball_y": np.asarray([0, 70], dtype=np.int32)},
    )
    kernel.on_reset(
        observations,
        {"ball_y": np.asarray([0, 0], dtype=np.int32)},
        np.asarray([False, True]),
    )

    np.testing.assert_array_equal(kernel.map_actions(np.asarray([3, 3])), [1, 1])


def test_conditional_override_changes_execution_not_base_action_semantics() -> None:
    descriptor = _descriptor()
    plain = _config()
    plain.task = {**plain.task, "action": {"set": "native"}}
    overridden = _config()

    plain_contract = compile_runtime_action_contract(
        plain,
        descriptor,
        descriptor.native_action_space,
    )
    overridden_contract = compile_runtime_action_contract(
        overridden,
        descriptor,
        descriptor.native_action_space,
    )

    assert plain_contract["semantic_hash"] == overridden_contract["semantic_hash"]
    assert plain_contract["execution_hash"] != overridden_contract["execution_hash"]


def test_conditional_override_rejects_unrepresentable_signal_value() -> None:
    descriptor = _descriptor()
    descriptor = ProviderDescriptor(
        **{
            **descriptor.__dict__,
            "signal_schema": {"ball_y": SignalSpec("ball_y", np.uint8)},
        }
    )
    config = _config()
    config.task["action"]["conditional_overrides"][0]["when"]["value"] = 256
    kernel = IdentityTaskDefinition(signals=config.task["signals"]).bind(descriptor, 1)
    contract = compile_runtime_action_contract(config, descriptor, kernel.action_space)

    with np.testing.assert_raises_regex(ValueError, "not representable"):
        with_conditional_action_overrides(
            kernel,
            descriptor,
            config.task["signals"],
            contract,
        )
