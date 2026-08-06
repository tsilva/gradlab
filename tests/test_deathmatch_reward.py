from __future__ import annotations

import gymnasium as gym
import numpy as np

from gradlab.batch_runtime import ProviderDescriptor, SignalSpec
from gradlab.callbacks import RewardStatsAccumulator
from gradlab.task_kernels import (
    IdentityTaskDefinition,
    with_deathmatch_reward,
    with_reward_transform,
)
from gradlab.training.sb3_on_policy import active_reward_components


SIGNALS = {
    "kills": "killcount",
    "deaths": "deathcount",
    "hits": "hitcount",
    "damage": "damagecount",
    "health": "health",
    "armor": "armor",
    "selected_weapon": "selected_weapon",
    "selected_weapon_ammo": "selected_weapon_ammo",
    "weapons_owned": [f"weapon{slot}" for slot in range(1, 7)],
    "weapon_ammo": [f"ammo{slot}" for slot in range(1, 7)],
    "player_dead": "player_dead",
    "native_timeout": "provider_truncated",
}


def reward_config(*, reward_scale: float = 1.0, reward_clip=False) -> dict[str, object]:
    return {
        "reward_mode": "sample-factory-v0",
        "reward_scale": reward_scale,
        "reward_clip": reward_clip,
        "kill_reward": 1.0,
        "kill_loss_penalty": 1.5,
        "death_penalty": 0.75,
        "death_count_decrease_reward": 0.75,
        "hit_reward": 0.01,
        "hit_count_decrease_penalty": 0.01,
        "damage_reward": 0.003,
        "damage_count_decrease_penalty": 0.003,
        "health_gain_reward": 0.005,
        "health_loss_penalty": 0.003,
        "armor_gain_reward": 0.005,
        "armor_loss_penalty": 0.001,
        "weapon_preferences": [1.0, 1.0, 5.0, 5.0, 5.0, 10.0],
        "weapon_gain_reward_scale": 0.02,
        "weapon_loss_penalty_scale": 0.01,
        "ammo_gain_reward_scale": 0.0002,
        "ammo_loss_penalty_scale": 0.0001,
        "selected_weapon_hold_reward_scale": 0.0002,
        "selected_weapon_hold_steps": 5,
        "hit_delta_cap": 5,
        "damage_delta_cap": 200,
    }


def descriptor(num_envs: int = 1) -> ProviderDescriptor:
    del num_envs
    sources = {
        source
        for binding in SIGNALS.values()
        for source in ([binding] if isinstance(binding, str) else binding)
        if source not in {"provider_terminated", "provider_truncated"}
    }
    return ProviderDescriptor(
        provider_id="fake-vizdoom",
        native_observation_space=gym.spaces.Box(0, 255, shape=(1,), dtype=np.uint8),
        native_action_space=gym.spaces.Discrete(2),
        signal_schema={source: SignalSpec(source, np.int64) for source in sources},
    )


def kernel(*, reward: dict[str, object] | None = None):
    source_descriptor = descriptor()
    inner = IdentityTaskDefinition(
        signals=SIGNALS,
        events={
            "player_died": {
                "signal": "player_dead",
                "operation": "equals_for",
                "value": 1,
                "steps": 1,
            },
            "time_limit_reached": {
                "signal": "native_timeout",
                "operation": "equals_for",
                "value": 1,
                "steps": 1,
            },
        },
        termination={
            "failure": ["player_died"],
            "timeout": ["time_limit_reached"],
        },
    ).bind(source_descriptor, 1)
    configured = reward or reward_config()
    shaped = with_deathmatch_reward(inner, source_descriptor, SIGNALS, configured)
    return with_reward_transform(shaped, configured)


def state(**updates: int) -> dict[str, np.ndarray]:
    values = {
        source: np.zeros(1, dtype=np.int64)
        for binding in SIGNALS.values()
        for source in ([binding] if isinstance(binding, str) else binding)
        if source not in {"provider_terminated", "provider_truncated"}
    }
    values.update(
        health=np.asarray([100], dtype=np.int64),
        selected_weapon=np.asarray([3], dtype=np.int64),
        selected_weapon_ammo=np.asarray([10], dtype=np.int64),
        weapon1=np.asarray([1], dtype=np.int64),
        weapon2=np.asarray([1], dtype=np.int64),
    )
    for name, value in updates.items():
        values[name] = np.asarray([value], dtype=np.int64)
    return values


FALSE = np.asarray([False])
TRUE = np.asarray([True])
NATIVE_REWARD = np.asarray([99.0], dtype=np.float32)


def reset(bound, initial: dict[str, np.ndarray]) -> None:
    bound.on_reset(None, initial, np.asarray([True]))


def test_sample_factory_v0_components_caps_and_native_reward_replacement() -> None:
    bound = kernel()
    initial = state()
    reset(bound, initial)

    baseline = bound.process(NATIVE_REWARD, FALSE, FALSE, initial)
    np.testing.assert_allclose(baseline.rewards, [0.0])

    changed = state(
        killcount=1,
        deathcount=1,
        hitcount=10,
        damagecount=300,
        health=90,
        armor=5,
        weapon3=1,
        ammo3=10,
    )
    step = bound.process(NATIVE_REWARD, FALSE, FALSE, changed)
    expected = {
        "kill_reward_component": 1.0,
        "death_penalty_component": -0.75,
        "hit_reward_component": 0.05,
        "damage_reward_component": 0.6,
        "health_reward_component": -0.03,
        "armor_reward_component": 0.025,
        "weapon_reward_component": 0.1,
        "ammo_reward_component": 0.01,
        "weapon_hold_reward_component": 0.0,
    }
    for name, value in expected.items():
        np.testing.assert_allclose(step.metrics[name], [value], atol=1e-7)
    np.testing.assert_allclose(step.rewards, [sum(expected.values())], atol=1e-7)
    np.testing.assert_array_equal(step.metrics["kills"], [1])

    reversed_step = bound.process(NATIVE_REWARD, FALSE, FALSE, initial)
    np.testing.assert_allclose(reversed_step.rewards, [-1.76], atol=1e-7)


def test_weapon_hold_uses_actual_selected_slot_on_the_fifth_decision() -> None:
    bound = kernel()
    plasma = state(selected_weapon=6, selected_weapon_ammo=50, weapon6=1, ammo6=50)
    reset(bound, plasma)

    for decision in range(1, 6):
        step = bound.process(NATIVE_REWARD, FALSE, FALSE, plasma)
        expected = 0.002 if decision == 5 else 0.0
        np.testing.assert_allclose(
            step.metrics["weapon_hold_reward_component"],
            [expected],
            atol=1e-8,
        )


def test_death_transition_is_shaped_but_pure_timeout_is_not() -> None:
    bound = kernel()
    initial = state(selected_weapon=0, selected_weapon_ammo=0)
    reset(bound, initial)
    bound.process(NATIVE_REWARD, FALSE, FALSE, initial)

    death = state(
        selected_weapon=0,
        selected_weapon_ammo=0,
        deathcount=1,
        player_dead=1,
    )
    death_step = bound.process(NATIVE_REWARD, FALSE, FALSE, death)
    assert bool(death_step.terminated[0])
    np.testing.assert_allclose(death_step.rewards, [-0.75])

    reset(bound, initial)
    bound.process(NATIVE_REWARD, FALSE, FALSE, initial)
    final = state(selected_weapon=0, selected_weapon_ammo=0, killcount=1)
    timeout_step = bound.process(NATIVE_REWARD, FALSE, TRUE, final)
    assert bool(timeout_step.truncated[0])
    np.testing.assert_allclose(timeout_step.rewards, [0.0])


def test_reward_state_restore_preserves_weapon_hold_history() -> None:
    first = kernel()
    second = kernel()
    plasma = state(selected_weapon=6, selected_weapon_ammo=50, weapon6=1, ammo6=50)
    for bound in (first, second):
        reset(bound, plasma)
    for _ in range(3):
        first.process(NATIVE_REWARD, FALSE, FALSE, plasma)

    saved = first.capture_lane_states(np.asarray([True]))
    second.restore_lane_states(saved, np.asarray([True]))
    for expected in (0.0, 0.002):
        first_step = first.process(NATIVE_REWARD, FALSE, FALSE, plasma)
        second_step = second.process(NATIVE_REWARD, FALSE, FALSE, plasma)
        np.testing.assert_allclose(first_step.rewards, [expected], atol=1e-8)
        np.testing.assert_allclose(second_step.rewards, first_step.rewards, atol=1e-8)


def test_global_reward_transform_runs_after_component_sum() -> None:
    configured = reward_config(reward_scale=0.5, reward_clip=[-1.0, 1.0])
    configured["kill_reward"] = 0.75
    bound = kernel(reward=configured)
    initial = state()
    reset(bound, initial)
    bound.process(NATIVE_REWARD, FALSE, FALSE, initial)

    killed = state(killcount=1)
    step = bound.process(NATIVE_REWARD, FALSE, FALSE, killed)
    np.testing.assert_allclose(step.metrics["kill_reward_component"], [0.75])
    np.testing.assert_allclose(step.metrics["raw_reward"], [0.75])
    np.testing.assert_allclose(step.metrics["shaped_reward"], [1.0])
    np.testing.assert_allclose(step.rewards, [1.0])


def test_training_telemetry_registers_every_deathmatch_component() -> None:
    configured = reward_config()
    components = active_reward_components({"reward": configured})
    assert components == (
        "kill",
        "death",
        "hit",
        "damage",
        "health",
        "armor",
        "weapon",
        "ammo",
        "weapon_hold",
    )
    accumulator = RewardStatsAccumulator(active_components=components)
    bound = kernel(reward=configured)
    initial = state()
    reset(bound, initial)
    bound.process(NATIVE_REWARD, FALSE, FALSE, initial)
    step = bound.process(NATIVE_REWARD, FALSE, FALSE, state(killcount=1))

    accumulator.consume(step.metrics, reserve=1)
    payload = accumulator.flush()

    assert payload["train/reward/component/kill/mean"] == 1.0
    assert payload["train/reward/component/kill/nonzero_rate"] == 1.0
    assert payload["train/reward/component/kill/share"] == 1.0
