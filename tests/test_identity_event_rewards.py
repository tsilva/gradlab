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
        signal_schema={
            "ball_y": SignalSpec("ball_y", np.int64),
            "lives": SignalSpec("lives", np.int64),
        },
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


def test_identity_equals_event_rewards_every_matching_transition() -> None:
    configured = {
        "id": "identity",
        "action": {"set": "native"},
        "signals": {"ball_y": "ball_y"},
        "events": {
            "serve_wait": {
                "signal": "ball_y",
                "operation": "equals",
                "value": 0,
            }
        },
        "termination": {},
        "reward": {
            "reward_mode": "native",
            "event_rewards": {"serve_wait": -0.01},
            "reward_scale": 1.0,
            "reward_clip": False,
        },
    }
    validate_task_config(configured)
    base = IdentityTaskDefinition(
        signals=configured["signals"],
        events=configured["events"],
    ).bind(descriptor(), 2)
    kernel = with_event_rewards(base, configured["reward"]["event_rewards"])
    kernel.on_reset(
        np.zeros((2, 1, 8, 8), dtype=np.uint8),
        {"ball_y": np.asarray([0, 0], dtype=np.int64)},
        np.ones(2, dtype=np.bool_),
    )
    accumulator = RewardStatsAccumulator(active_components=("native", "event"))

    first = kernel.process(
        np.asarray([1.0, 1.0], dtype=np.float32),
        np.zeros(2, dtype=np.bool_),
        np.zeros(2, dtype=np.bool_),
        {"ball_y": np.asarray([0, 5], dtype=np.int64)},
    )
    np.testing.assert_allclose(first.rewards, [0.99, 1.0])
    assert first.event_bits.tolist() == [1, 0]
    accumulator.consume(first.metrics, reserve=4)

    second = kernel.process(
        np.asarray([1.0, 1.0], dtype=np.float32),
        np.zeros(2, dtype=np.bool_),
        np.zeros(2, dtype=np.bool_),
        {"ball_y": np.asarray([0, 0], dtype=np.int64)},
    )

    np.testing.assert_allclose(second.rewards, [0.99, 0.99])
    assert second.event_bits.tolist() == [1, 1]
    accumulator.consume(second.metrics, reserve=4)
    payload = accumulator.flush()
    assert payload["train/reward/event/serve_wait/mean"] == pytest.approx(-0.0075)
    assert payload["train/reward/event/serve_wait/nonzero/rate"] == 0.75


def test_previous_equals_attributes_serve_wait_to_the_transition_after_life_loss() -> None:
    configured = {
        "id": "identity",
        "action": {"set": "native"},
        "signals": {"ball_y": "ball_y", "lives": "lives"},
        "events": {
            "life_loss": {
                "signal": "lives",
                "operation": "decrease",
            },
            "serve_wait": {
                "signal": "ball_y",
                "operation": "previous_equals",
                "value": 0,
            },
        },
        "termination": {},
        "reward": {
            "reward_mode": "native",
            "event_rewards": {"life_loss": -5.0, "serve_wait": -0.01},
            "reward_scale": 1.0,
            "reward_clip": False,
        },
    }
    validate_task_config(configured)
    base = IdentityTaskDefinition(
        signals=configured["signals"],
        events=configured["events"],
    ).bind(descriptor(), 1)
    kernel = with_event_rewards(base, configured["reward"]["event_rewards"])
    kernel.on_reset(
        np.zeros((1, 1, 8, 8), dtype=np.uint8),
        {"ball_y": np.asarray([208]), "lives": np.asarray([5])},
        np.ones(1, dtype=np.bool_),
    )

    life_loss = kernel.process(
        np.asarray([0.0], dtype=np.float32),
        np.zeros(1, dtype=np.bool_),
        np.zeros(1, dtype=np.bool_),
        {"ball_y": np.asarray([0]), "lives": np.asarray([4])},
    )

    assert life_loss.event_bits.tolist() == [1]
    np.testing.assert_allclose(life_loss.rewards, [-5.0])

    auto_serve = kernel.process(
        np.asarray([0.0], dtype=np.float32),
        np.zeros(1, dtype=np.bool_),
        np.zeros(1, dtype=np.bool_),
        {"ball_y": np.asarray([116]), "lives": np.asarray([4])},
    )

    assert auto_serve.event_bits.tolist() == [2]
    np.testing.assert_allclose(auto_serve.rewards, [-0.01])
    assert auto_serve.event_transitions["serve_wait"][0].tolist() == [0]
    assert auto_serve.event_transitions["serve_wait"][1].tolist() == [116]


def test_equals_reward_and_equals_for_timeout_compose_on_threshold_transition() -> None:
    events = {
        "serve_wait": {
            "signal": "ball_y",
            "operation": "equals",
            "value": 0,
        },
        "serve_stall": {
            "signal": "ball_y",
            "operation": "equals_for",
            "value": 0,
            "steps": 2,
        },
    }
    base = IdentityTaskDefinition(
        signals={"ball_y": "ball_y"},
        events=events,
        termination={"timeout": ["serve_stall"]},
    ).bind(descriptor(), 1)
    kernel = with_event_rewards(
        base,
        {"serve_wait": -0.01, "serve_stall": -5.0},
    )
    kernel.on_reset(
        np.zeros((1, 1, 8, 8), dtype=np.uint8),
        {"ball_y": np.asarray([0], dtype=np.int64)},
        np.ones(1, dtype=np.bool_),
    )

    first = kernel.process(
        np.asarray([0.0], dtype=np.float32),
        np.zeros(1, dtype=np.bool_),
        np.zeros(1, dtype=np.bool_),
        {"ball_y": np.asarray([0], dtype=np.int64)},
    )
    np.testing.assert_allclose(first.rewards, [-0.01])
    np.testing.assert_array_equal(first.truncated, [False])

    threshold = kernel.process(
        np.asarray([0.0], dtype=np.float32),
        np.zeros(1, dtype=np.bool_),
        np.zeros(1, dtype=np.bool_),
        {"ball_y": np.asarray([0], dtype=np.int64)},
    )
    np.testing.assert_allclose(threshold.rewards, [-5.01])
    np.testing.assert_array_equal(threshold.truncated, [True])
    assert threshold.event_bits.tolist() == [3]


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
    np.testing.assert_allclose(step.metrics["event_reward_component/life_loss"], [-5.0])
    components = active_reward_components(configured)
    assert components == ("native", "event")
    accumulator = RewardStatsAccumulator(active_components=components)
    accumulator.consume(step.metrics, reserve=1)
    payload = accumulator.flush()
    assert payload["train/reward/component/native/mean"] == 1.0
    assert payload["train/reward/component/event/mean"] == -5.0
    assert payload["train/reward/event/life_loss/mean"] == -5.0
    assert payload["train/reward/event/life_loss/nonzero/rate"] == 1.0


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
