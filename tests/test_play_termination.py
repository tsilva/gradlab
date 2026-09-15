import gymnasium as gym
import numpy as np
import pytest

from gradlab.env import EnvConfig
from gradlab.env_identity import validate_task_config
from gradlab.play_termination import configured_termination_ids, with_enabled_termination_conditions
from gradlab.task_kernels import IdentityTaskDefinition, with_event_rewards


@pytest.mark.parametrize("continuous", [False, True])
def test_wall_clear_event_and_bonus_do_not_repeat_in_continued_playback(continuous):
    from gradlab.batch_runtime import ProviderDescriptor, SignalSpec

    config = EnvConfig(
        env_provider="gradlab",
        game="Bandit-v0",
        task={
            "id": "identity",
            "action": {"set": "native"},
            "signals": {"walls_cleared": "walls_cleared", "ball_y": "ball_y"},
            "events": {
                "one_wall_cleared": {
                    "signal": "walls_cleared", "operation": "equals", "value": 1,
                },
                "serve_wait": {"signal": "ball_y", "operation": "equals", "value": 0},
            },
            "termination": {"success": ["one_wall_cleared"], "max_episode_steps": 100},
            "reward": {
                "reward_mode": "events",
                "event_rewards": {"one_wall_cleared": 20.0, "serve_wait": -0.01},
            },
        },
    )
    faithful = with_enabled_termination_conditions(config, configured_termination_ids(config))
    assert faithful == config
    active = with_enabled_termination_conditions(
        config, [] if continuous else ["limit:max_episode_steps"],
    )
    validate_task_config(active.task)
    assert config.task["events"]["one_wall_cleared"]["operation"] == "equals"
    assert active.task["events"]["serve_wait"]["operation"] == "equals"

    descriptor = ProviderDescriptor(
        provider_id="playback-event-test",
        native_observation_space=gym.spaces.Box(0, 255, shape=(1, 8, 8), dtype=np.uint8),
        native_action_space=gym.spaces.Discrete(2),
        signal_schema={name: SignalSpec(name, np.int64) for name in active.task["signals"]},
    )
    kernel = with_event_rewards(
        IdentityTaskDefinition(
            signals=active.task["signals"],
            events=active.task["events"],
            termination=active.task["termination"],
        ).bind(descriptor, 2),
        active.task["reward"]["event_rewards"],
        include_native=False,
    )
    flags = np.zeros(2, dtype=np.bool_)

    def reset(mask):
        kernel.on_reset(
            np.zeros((2, 1, 8, 8), dtype=np.uint8),
            {"walls_cleared": np.array([0, 0]), "ball_y": np.array([0, 0])},
            np.array(mask),
        )

    def step(walls, expected_wall_events):
        result = kernel.process(
            np.zeros(2, dtype=np.float32), flags, flags,
            {"walls_cleared": np.array(walls), "ball_y": np.array([0, 0])},
        )
        assert result.event_bits.tolist() == [2 + int(fired) for fired in expected_wall_events]
        np.testing.assert_allclose(
            result.rewards, [20.0 * fired - 0.01 for fired in expected_wall_events], rtol=1e-6,
        )
        assert not result.terminated.any()
        assert not result.truncated.any()

    reset([True, True])
    step([0, 0], [False, False])
    step([1, 0], [True, False])
    step([1, 1], [False, True])
    for _ in range(5):
        step([1, 1], [False, False])
    reset([True, False])
    step([1, 1], [True, False])
    step([2, 1], [False, False])
