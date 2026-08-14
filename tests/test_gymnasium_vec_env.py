from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from gradlab.action_contract import action_index_for_controls, compile_runtime_action_contract
from gradlab.env_providers import provider_descriptor
from gradlab.gymnasium_vec_env import (
    GYMNASIUM_ENV_CONTRACTS,
    GymnasiumTurboVecEnv,
)
from gradlab.turbo_api import validate_turbo_vector_env


def _env(game: str, num_envs: int = 2) -> GymnasiumTurboVecEnv:
    return GymnasiumTurboVecEnv(
        game,
        num_envs,
        autoreset_mode="disabled",
        vectorization_mode="async",
        multiprocessing_context="spawn",
        shared_memory=True,
        copy=True,
        daemon=True,
        observation_mode="same",
        render_mode="rgb_array",
    )


@pytest.mark.parametrize("game", tuple(GYMNASIUM_ENV_CONTRACTS))
def test_registered_envs_expose_strict_turbo_v2_contract(game: str) -> None:
    contract = GYMNASIUM_ENV_CONTRACTS[game]
    env = _env(game)
    try:
        turbo = validate_turbo_vector_env(env, "gymnasium")
        observations, infos = env.reset(
            seed=[101, 202],
            options={"reset_mask": np.ones(2, dtype=np.bool_)},
        )

        assert turbo.api_version == 2
        assert observations.shape == (2, *contract.observation_shape)
        assert observations.dtype == np.dtype(contract.observation_dtype)
        assert isinstance(env.single_action_space, gym.spaces.Discrete)
        assert env.single_action_space.n == contract.action_count
        assert env.action_mode == "discrete"
        assert env.action_meanings == contract.action_meanings
        assert env.action_table == contract.action_controls
        assert env.capabilities["supports_async_step"]
        assert env.capabilities["supports_per_lane_rgb"]
        assert env.state_catalog == ()
        assert env.signal_schema == {}
        assert env.active_state_indices().tolist() == [-1, -1]
        assert not env.active_state_indices().flags.writeable
        assert infos["state_index"].dtype == np.int32
        assert infos["start_source"].dtype == np.int8
        assert infos["noop_reset_count"].dtype == np.int64
        np.testing.assert_array_equal(infos["_state_index"], [True, True])
    finally:
        env.close()


def test_seeded_traces_are_reproducible_and_lanes_are_distinct() -> None:
    actions = (
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([1, 0], dtype=np.int64),
        np.asarray([0, 0], dtype=np.int64),
    )

    def trace() -> list[tuple[np.ndarray, ...]]:
        env = _env("CartPole-v1")
        try:
            observations, _infos = env.reset(
                seed=[1234, 5678],
                options={"reset_mask": np.ones(2, dtype=np.bool_)},
            )
            result = [(observations.copy(),)]
            for action in actions:
                step = env.step(action)
                result.append(tuple(np.asarray(value).copy() for value in step[:4]))
            return result
        finally:
            env.close()

    first = trace()
    second = trace()
    assert not np.array_equal(first[0][0][0], first[0][0][1])
    for first_step, second_step in zip(first, second, strict=True):
        for first_value, second_value in zip(first_step, second_step, strict=True):
            np.testing.assert_array_equal(first_value, second_value)


def test_masked_reset_preserves_unselected_lane() -> None:
    env = _env("CartPole-v1")
    try:
        env.reset(
            seed=[11, 22],
            options={"reset_mask": np.ones(2, dtype=np.bool_)},
        )
        observations, *_rest = env.step(np.asarray([0, 1], dtype=np.int64))
        lane_one = observations[1].copy()
        reset_observations, infos = env.reset(
            seed=[33, None],
            options={"reset_mask": np.asarray([True, False], dtype=np.bool_)},
        )

        np.testing.assert_array_equal(reset_observations[1], lane_one)
        np.testing.assert_array_equal(infos["_start_source"], [True, False])
    finally:
        env.close()


def test_async_step_terminal_lockout_and_lane_by_lane_reset() -> None:
    env = _env("MountainCar-v0")
    try:
        env.reset(
            seed=[7, 8],
            options={"reset_mask": np.ones(2, dtype=np.bool_)},
        )
        for _ in range(200):
            env.step_async(np.asarray([1, 1], dtype=np.int64))
            _observations, _rewards, terminated, truncated, _infos = env.step_wait()
        np.testing.assert_array_equal(terminated, [False, False])
        np.testing.assert_array_equal(truncated, [True, True])
        with pytest.raises(RuntimeError, match="explicitly reset"):
            env.step(np.asarray([1, 1], dtype=np.int64))

        env.reset(
            seed=[9, None],
            options={"reset_mask": np.asarray([True, False], dtype=np.bool_)},
        )
        with pytest.raises(RuntimeError, match="explicitly reset"):
            env.step(np.asarray([1, 1], dtype=np.int64))
        env.reset(
            seed=[None, 10],
            options={"reset_mask": np.asarray([False, True], dtype=np.bool_)},
        )
        env.step(np.asarray([1, 1], dtype=np.int64))
    finally:
        env.close()


def test_render_outputs_are_per_lane_owned_rgb_copies() -> None:
    env = _env("Acrobot-v1")
    try:
        env.reset(
            seed=[4, 5],
            options={"reset_mask": np.ones(2, dtype=np.bool_)},
        )
        frames = env.get_images()
        assert len(frames) == 2
        assert frames[0].dtype == np.uint8
        assert frames[0].ndim == 3 and frames[0].shape[-1] == 3
        original = frames[0].copy()
        frames[0].fill(0)
        np.testing.assert_array_equal(env.render_lane(0), original)
    finally:
        env.close()


@pytest.mark.parametrize(
    ("game", "expected"),
    (
        ("CartPole-v1", {("left",): 0, ("right",): 1}),
        ("MountainCar-v0", {("left",): 0, (): 1, ("right",): 2}),
        ("Acrobot-v1", {("left",): 0, (): 1, ("right",): 2}),
    ),
)
def test_action_metadata_compiles_to_browser_controls(
    game: str,
    expected: dict[tuple[str, ...], int],
) -> None:
    env = _env(game, num_envs=1)
    try:
        config = SimpleNamespace(
            env_provider="gymnasium",
            game=game,
            env_args={},
            state=None,
            states=(),
            state_probs=(),
            task={"action": {"set": "native"}},
        )
        descriptor = provider_descriptor(
            config,
            env,
            state_weight_mapping=lambda _config: {},
        )
        contract = compile_runtime_action_contract(
            config,
            descriptor,
            descriptor.native_action_space,
        )

        assert {
            controls: action_index_for_controls(contract, controls)
            for controls in expected
        } == expected
    finally:
        env.close()


def test_adapter_rejects_unknown_env_and_is_idempotently_closeable() -> None:
    with pytest.raises(ValueError, match="unsupported Gymnasium environment"):
        _env("Pendulum-v1")

    env = _env("CartPole-v1", num_envs=1)
    env.close()
    env.close()
    with pytest.raises(RuntimeError, match="closed"):
        env.active_state_indices()
