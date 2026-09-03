from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from gradlab.batch_runtime import ProviderDescriptor, SignalSpec
from gradlab.callbacks import RewardStatsAccumulator
from gradlab.env_identity import task_config_from_train_config, validate_task_config
from gradlab.play_web import _reward_accounting_payload
from gradlab.task_kernels import IdentityTaskDefinition, with_event_rewards, with_reward_transform
from gradlab.training.sb3_on_policy import active_reward_components


def task() -> dict:
    return {
        "id": "identity",
        "action": {"set": "native"},
        "signals": {"lives": "lives"},
        "events": {"life_loss": {"signal": "lives", "operation": "decrease"}},
        "termination": {},
        "reward": {
            "reward_mode": "native",
            "event_rewards": {"life_loss": -5},
            "reward_scale": 1.0,
            "reward_clip": False,
        },
    }


def descriptor() -> ProviderDescriptor:
    return ProviderDescriptor(
        provider_id="event-reward-test",
        native_observation_space=gym.spaces.Box(0, 255, shape=(1, 8, 8), dtype=np.uint8),
        native_action_space=gym.spaces.Discrete(2),
        signal_schema={"lives": SignalSpec("lives", np.int64)},
    )


def test_identity_event_rewards_are_strict_and_reference_declared_events() -> None:
    configured = task()
    validate_task_config(configured)
    normalized = task_config_from_train_config(
        {
            "env_provider": "gymnasium",
            "game": "CartPole-v1",
            "task": configured,
        }
    )
    assert normalized["reward"]["event_rewards"] == {"life_loss": -5.0}

    configured["reward"]["event_rewards"] = {"missing": -5}
    with pytest.raises(ValueError, match="references unknown events: missing"):
        validate_task_config(configured)

    configured["reward"]["event_rewards"] = {"life_loss": 0}
    with pytest.raises(ValueError, match="finite non-zero"):
        validate_task_config(configured)


def test_life_loss_event_subtracts_five_without_ending_the_episode() -> None:
    configured = task()
    base = IdentityTaskDefinition(
        signals=configured["signals"],
        events=configured["events"],
        termination=configured["termination"],
    ).bind(descriptor(), 2)
    kernel = with_event_rewards(base, configured["reward"]["event_rewards"])
    kernel.on_reset(
        np.zeros((2, 1, 8, 8), dtype=np.uint8),
        {"lives": np.asarray([5, 5], dtype=np.int64)},
        np.ones(2, dtype=np.bool_),
    )

    step = kernel.process(
        np.asarray([0.0, 4.0], dtype=np.float32),
        np.zeros(2, dtype=np.bool_),
        np.zeros(2, dtype=np.bool_),
        {"lives": np.asarray([4, 5], dtype=np.int64)},
    )

    np.testing.assert_allclose(step.rewards, [-5.0, 4.0])
    np.testing.assert_allclose(step.metrics["native_reward_component"], [0.0, 4.0])
    np.testing.assert_allclose(step.metrics["event_reward_component"], [-5.0, 0.0])
    np.testing.assert_array_equal(step.terminated, [False, False])
    np.testing.assert_array_equal(step.truncated, [False, False])


def test_event_reward_is_included_before_the_global_reward_transform_and_logging() -> None:
    configured = task()
    configured["reward"]["reward_scale"] = 0.5
    base = IdentityTaskDefinition(
        signals=configured["signals"],
        events=configured["events"],
    ).bind(descriptor(), 1)
    kernel = with_event_rewards(base, configured["reward"]["event_rewards"])
    kernel = with_reward_transform(kernel, configured["reward"])
    kernel.on_reset(
        np.zeros((1, 1, 8, 8), dtype=np.uint8),
        {"lives": np.asarray([5], dtype=np.int64)},
        np.ones(1, dtype=np.bool_),
    )

    step = kernel.process(
        np.asarray([1.0], dtype=np.float32),
        np.zeros(1, dtype=np.bool_),
        np.zeros(1, dtype=np.bool_),
        {"lives": np.asarray([4], dtype=np.int64)},
    )

    np.testing.assert_allclose(step.metrics["raw_reward"], [-4.0])
    np.testing.assert_allclose(step.rewards, [-2.0])
    components = active_reward_components(configured)
    assert components == ("native", "event")
    accumulator = RewardStatsAccumulator(active_components=components)
    accumulator.consume(step.metrics, reserve=1)
    payload = accumulator.flush()
    assert payload["train/reward/component/native/mean"] == 1.0
    assert payload["train/reward/component/event/mean"] == -5.0


def test_player_reward_accounting_exposes_the_event_component() -> None:
    raw, components, error = _reward_accounting_payload(
        final_reward=-5.0,
        task_metrics={
            "raw_reward": -5.0,
            "native_reward_component": 0.0,
            "event_reward_component": -5.0,
        },
        accounting={"status": "available", "reward_scale": 1.0, "clip_bounds": None},
    )

    assert raw == -5.0
    assert components == {"native_reward": 0.0, "event_reward": -5.0}
    assert error is None
