from __future__ import annotations

import importlib.metadata
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gymnasium as gym
import numpy as np
import env_stableretro_turbo as retro
from env_breakoutatari2600_turbo_native import FIXED_POINT_ONE, POLICY_INFO_KEYS, RAW_WIDTH

from gradlab.action_contract import MARIO_ACTION_TABLES
from gradlab.env import EnvConfig, _bound_task_kernel, make_vec_envs
from gradlab.env_providers import (
    _AleManualResetAdapter,
    _StartInfoAdapter,
    make_provider_vec_env,
    provider_descriptor,
    provider_native_vec_kwargs,
    super_mario_bros_nes_turbo_vec_env_type,
)
from gradlab.play_session import vector_env_frame
from gradlab.task_kernels import MarioTaskConfig, MarioTaskDefinition
from packaging.version import Version


BREAKOUT_NO_NOOP_ACTIONS = [["BUTTON"], ["RIGHT"], ["LEFT"]]
BREAKOUT_NO_NOOP_HASH = "a1f69721fbf7ef8a00084b9426767b0bce61f39ee0880b932a954c7d5789ee15"


class RegisteredNativeVectorEnv(gym.vector.VectorEnv):
    metadata = {"autoreset_mode": gym.vector.AutoresetMode.DISABLED}

    def __init__(self, num_envs: int, autoreset_mode, **kwargs):
        del kwargs
        if autoreset_mode is not gym.vector.AutoresetMode.DISABLED:
            raise ValueError("manual autoreset is required")
        self.num_envs = int(num_envs)
        self.autoreset_mode = autoreset_mode
        self.single_observation_space = gym.spaces.Box(-100, 100, shape=(2,), dtype=np.float32)
        self.single_action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, self.num_envs
        )
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self._observations = np.zeros((self.num_envs, 2), dtype=np.float32)
        self.reset_masks: list[np.ndarray] = []

    def reset(self, *, seed=None, options=None):
        del seed
        options = dict(options or {})
        mask = np.asarray(
            options.get("reset_mask", np.ones(self.num_envs, dtype=np.bool_)),
            dtype=np.bool_,
        )
        self.reset_masks.append(mask.copy())
        self._observations[mask] = 0
        return self._observations, {}

    def step(self, actions):
        del actions
        self._observations += 1
        return (
            self._observations,
            np.ones(self.num_envs, dtype=np.float32),
            np.zeros(self.num_envs, dtype=np.bool_),
            np.zeros(self.num_envs, dtype=np.bool_),
            {},
        )

    def close(self):
        return None


class GenericNativeProviderTests(unittest.TestCase):
    env_id = "GradLabRegisteredNativeVector-v0"
    scalar_env_id = "GradLabScalarOnly-v0"

    @classmethod
    def setUpClass(cls) -> None:
        gym.register(
            cls.env_id,
            entry_point=lambda: None,
            vector_entry_point=RegisteredNativeVectorEnv,
        )
        gym.register(cls.scalar_env_id, entry_point=lambda: None)

    @classmethod
    def tearDownClass(cls) -> None:
        gym.registry.pop(cls.env_id, None)
        gym.registry.pop(cls.scalar_env_id, None)

    def test_uses_strict_gymnasium_turbo_adapter(self) -> None:
        config = EnvConfig(
            env_provider="gymnasium",
            game="CartPole-v1",
            state="",
            env_args={
                "autoreset_mode": "disabled",
                "vectorization_mode": "async",
                "multiprocessing_context": "spawn",
                "shared_memory": True,
                "copy": True,
                "daemon": True,
                "observation_mode": "same",
                "render_mode": "rgb_array",
            },
            task={
                "id": "identity",
                "action": {"set": "native"},
                "signals": {},
                "events": {},
                "termination": {},
                "reward": {"reward_mode": "native"},
            },
        )
        kwargs = provider_native_vec_kwargs(
            config,
            n_envs=2,
            native_obs_crop=lambda _config: None,
            state_weight_mapping=lambda _config: {},
        )
        env = make_provider_vec_env(config, native_kwargs=kwargs)

        self.assertEqual(env.num_envs, 2)
        self.assertIs(env.autoreset_mode, gym.vector.AutoresetMode.DISABLED)
        observations, infos = env.reset(
            seed=[1, 2],
            options={"reset_mask": np.ones(2, dtype=np.bool_)},
        )
        self.assertEqual(observations.shape, (2, 4))
        np.testing.assert_array_equal(infos["state_index"], [-1, -1])
        env.close()

    def test_rejects_unregistered_gymnasium_environment(self) -> None:
        config = EnvConfig(
            env_provider="gymnasium",
            game=self.scalar_env_id,
            state="",
            task={
                "id": "identity",
                "action": {"set": "native"},
                "signals": {},
                "events": {},
                "termination": {},
                "reward": {"reward_mode": "native"},
            },
        )
        with self.assertRaisesRegex(ValueError, "unsupported Gymnasium environment"):
            make_provider_vec_env(
                config,
                native_kwargs={
                    "num_envs": 2,
                    "autoreset_mode": "disabled",
                    "vectorization_mode": "async",
                    "multiprocessing_context": "spawn",
                    "shared_memory": True,
                    "copy": True,
                    "daemon": True,
                    "observation_mode": "same",
                    "render_mode": "rgb_array",
                },
            )

    def test_bound_identity_task_applies_common_scale_then_clip(self) -> None:
        env = RegisteredNativeVectorEnv(3, gym.vector.AutoresetMode.DISABLED)
        config = EnvConfig(
            env_provider="gymnasium",
            game=self.env_id,
            state="",
            task={
                "id": "identity",
                "action": {"set": "native"},
                "signals": {},
                "events": {},
                "termination": {},
                "reward": {
                    "reward_mode": "native",
                    "reward_scale": 0.5,
                    "reward_clip": [0.0, 0.4],
                },
            },
        )
        descriptor = provider_descriptor(
            config,
            env,
            state_weight_mapping=lambda _config: {},
        )
        kernel = _bound_task_kernel(config, descriptor, 3)

        step = kernel.process(
            np.asarray([1.0, -2.0, 4.0], dtype=np.float32),
            np.zeros(3, dtype=np.bool_),
            np.zeros(3, dtype=np.bool_),
            {},
        )

        np.testing.assert_allclose(step.rewards, [0.4, 0.0, 0.4])
        np.testing.assert_allclose(step.metrics["raw_reward"], [1.0, -2.0, 4.0])
        env.close()

    def test_generic_provider_reward_transform_arguments_are_ignored(self) -> None:
        config = EnvConfig(
            env_provider="gymnasium",
            game=self.env_id,
            state="",
            env_args={
                "reward_clip": True,
                "reward_clipping": True,
                "normalize_reward": True,
                "norm_reward": True,
                "reward_normalization": True,
            },
            task={
                "id": "identity",
                "action": {"set": "native"},
                "signals": {},
                "events": {},
                "termination": {},
                "reward": {
                    "reward_mode": "native",
                    "reward_scale": 0.5,
                    "reward_clip": False,
                },
            },
        )

        kwargs = provider_native_vec_kwargs(
            config,
            n_envs=3,
            native_obs_crop=lambda _config: None,
            state_weight_mapping=lambda _config: {},
        )

        self.assertEqual(kwargs, {"num_envs": 3})

    def test_descriptor_discovers_step_only_configured_signal(self) -> None:
        class StepSignalEnv(RegisteredNativeVectorEnv):
            def step(self, actions):
                observations, rewards, terminated, truncated, _infos = super().step(actions)
                return (
                    observations,
                    rewards,
                    terminated,
                    truncated,
                    {"ball_y": np.arange(self.num_envs, dtype=np.int64)},
                )

        env = StepSignalEnv(2, gym.vector.AutoresetMode.DISABLED)
        config = EnvConfig(
            env_provider="gymnasium",
            game=self.env_id,
            state="",
            task={
                "id": "identity",
                "action": {"set": "native"},
                "signals": {"ball_y": "ball_y"},
                "events": {
                    "serve_stall": {
                        "signal": "ball_y",
                        "operation": "equals_for",
                        "value": 0,
                        "steps": 3,
                    }
                },
                "termination": {"failure": ["serve_stall"]},
                "reward": {"reward_mode": "native"},
            },
        )

        descriptor = provider_descriptor(config, env, state_weight_mapping=lambda _config: {})

        self.assertIn("ball_y", descriptor.signal_schema)
        self.assertFalse(descriptor.signal_schema["ball_y"].available_on_reset)
        self.assertTrue(descriptor.signal_schema["ball_y"].available_on_step)

    def test_descriptor_does_not_trust_safe_view_from_generic_provider(self) -> None:
        env = RegisteredNativeVectorEnv(2, gym.vector.AutoresetMode.DISABLED)
        env.obs_copy = "safe_view"
        config = EnvConfig(
            env_provider="gymnasium",
            game=self.env_id,
            state="",
            task={
                "id": "identity",
                "action": {"set": "native"},
                "signals": {},
                "events": {},
                "termination": {},
                "reward": {"reward_mode": "native"},
            },
        )

        descriptor = provider_descriptor(config, env, state_weight_mapping=lambda _config: {})

        self.assertEqual(descriptor.observation_buffer_depth, 1)

    def test_start_adapter_supports_snapshot_natural_and_unselected_lanes_together(self) -> None:
        class MixedNative:
            num_envs = 3
            state_catalog = ("full",)

            def __init__(self) -> None:
                self.options = None

            def reset(self, *, seed=None, options=None):
                self.seed = seed
                self.options = dict(options or {})
                state_indices = np.asarray(self.options["state_indices"], dtype=np.int32)
                mask = np.asarray(self.options["reset_mask"], dtype=np.bool_)
                snapshots = tuple(self.options["snapshots"])
                snapshot_mask = np.asarray(
                    [value is not None for value in snapshots], dtype=np.bool_
                )
                start_source = np.full(3, "environment", dtype=object)
                start_source[snapshot_mask] = "snapshot"
                return np.zeros((3, 1), dtype=np.uint8), {
                    "state_index": state_indices,
                    "start_source": start_source,
                    "_start_source": mask,
                }

        native = MixedNative()
        adapter = _StartInfoAdapter(native)
        handle = object()
        mask = np.asarray([True, True, False], dtype=np.bool_)
        _observations, infos = adapter.reset(
            seed=[None, 123, None],
            options={
                "reset_mask": mask,
                "start_ids": np.asarray([None, "full", None], dtype=object),
                "snapshots": (handle, None, None),
            },
        )

        np.testing.assert_array_equal(native.options["state_indices"], [-1, 0, -1])
        self.assertEqual(native.seed, [None, 123, None])
        self.assertEqual(
            infos["start_source"].tolist(),
            ["snapshot", "environment", "environment"],
        )
        np.testing.assert_array_equal(infos["_start_source"], mask)

    def test_start_adapter_preserves_torch_v2_reset_transport(self) -> None:
        import torch

        class TorchNative:
            num_envs = 2
            state_catalog = ("default",)
            transport = "torch"
            device = torch.device("cpu")

            def reset(self, *, seed=None, options=None):
                del seed
                self.options = dict(options or {})
                mask = self.options["reset_mask"]
                state_indices = self.options["state_indices"]
                self.assert_tensor(mask, torch.bool)
                self.assert_tensor(state_indices, torch.int32)
                return torch.zeros((2, 1), dtype=torch.uint8), {
                    "state_index": state_indices.clone(),
                    "start_source": torch.zeros(2, dtype=torch.int8),
                    "noop_reset_count": torch.zeros(2, dtype=torch.int64),
                    "_state_index": mask.clone(),
                    "_start_source": mask.clone(),
                    "_noop_reset_count": mask.clone(),
                }

            def assert_tensor(self, value, dtype):
                if not isinstance(value, torch.Tensor) or value.dtype != dtype:
                    raise AssertionError(f"expected {dtype} tensor")
                if value.device != self.device:
                    raise AssertionError("selector left env.device")

        native = TorchNative()
        adapter = _StartInfoAdapter(native, strict_v2=True)

        _observations, infos = adapter.reset(
            options={
                "reset_mask": np.asarray([True, False], dtype=np.bool_),
                "start_ids": np.asarray(["default", None], dtype=object),
            }
        )

        self.assertEqual(infos["state_index"].dtype, torch.int32)
        self.assertEqual(infos["start_source"].dtype, torch.int8)
        self.assertEqual(infos["start_id"].tolist(), ["default", None])


class BreakoutTurboProviderTests(unittest.TestCase):
    @staticmethod
    def config(**updates):
        values = {
            "env_provider": "env-breakoutatari2600-turbo-native",
            "game": "Breakout-Atari2600-v0",
            "state": "Start",
            "frame_skip": 4,
            "max_pool_frames": False,
            "sticky_action_prob": 0.0,
            "obs_resize": (84, 84),
            "obs_crop": (0, 0, 0, 0),
            "obs_crop_mode": "remove",
            "obs_crop_fill": 0,
            "obs_resize_algorithm": "area",
            "env_args": {
                "scenario": "scenario",
                "info": "data",
                "use_restricted_actions": "simple",
                "record": False,
                "players": 1,
                "inttype": "stable",
                "obs_type": "image",
                "num_threads": 1,
                "frame_stack": 4,
                "obs_layout": "chw",
                "obs_grayscale": True,
                "obs_copy": "safe_view",
                "info_filter": "all",
                "render_mode": "rgb_array",
                "rom_path": None,
                "noop_reset_max": 0,
                "use_fire_reset": False,
            },
            "task": {
                "id": "identity",
                "action": {"set": "native"},
                "signals": {
                    "bricks_remaining": "bricks_remaining",
                    "lives": "lives",
                },
                "events": {
                    "cleared": {
                        "signal": "bricks_remaining",
                        "operation": "equals_for",
                        "value": 0,
                        "steps": 1,
                    },
                    "game_over": {
                        "signal": "lives",
                        "operation": "equals_for",
                        "value": 0,
                        "steps": 1,
                    },
                },
                "termination": {
                    "success": ["cleared"],
                    "failure": ["game_over"],
                    "max_episode_steps": 54_000,
                },
                "reward": {"reward_mode": "native"},
            },
        }
        values.update(updates)
        return EnvConfig(**values)

    def test_runtime_matches_turbo_api_v2_release(self) -> None:
        installed = Version(importlib.metadata.version("env-breakoutatari2600-turbo-native"))
        self.assertEqual(installed, Version("0.5.11"))

    def test_policy_info_keys_expose_typed_normalized_state_on_every_boundary(self) -> None:
        raw_keys = (
            "ball_x",
            "ball_y",
            "ball_vx",
            "ball_vy",
            "paddle_x",
            "bricks_destroyed",
        )
        normalized_keys = tuple(f"{key}_normalized" for key in raw_keys)
        self.assertTrue(set(raw_keys + normalized_keys).issubset(POLICY_INFO_KEYS))

        config = self.config(
            env_args={
                **self.config().env_args,
                "info_filter": {"mode": "all", "keys": list(raw_keys + normalized_keys)},
                "use_restricted_actions": BREAKOUT_NO_NOOP_ACTIONS,
            },
            task={
                "id": "identity",
                "action": {"set": "native"},
                "signals": {key: key for key in raw_keys + normalized_keys},
                "events": {},
                "termination": {},
                "reward": {"reward_mode": "native"},
            },
        )
        kwargs = provider_native_vec_kwargs(
            config,
            n_envs=2,
            native_obs_crop=lambda value: value.obs_crop,
            state_weight_mapping=lambda _config: {},
        )
        env = make_provider_vec_env(config, native_kwargs=kwargs)
        try:
            descriptor = provider_descriptor(
                config,
                env,
                state_weight_mapping=lambda _config: {},
            )
            expected_ranges = {
                "ball_x_normalized": (0.0, 1.0),
                "ball_y_normalized": (0.0, 1.0),
                "ball_vx_normalized": (-1.0, 1.0),
                "ball_vy_normalized": (-1.0, 1.0),
                "paddle_x_normalized": (0.0, 1.0),
                "bricks_destroyed_normalized": (0.0, 1.0),
            }
            for key in normalized_keys:
                with self.subTest(key=key):
                    spec = descriptor.signal_schema[key]
                    self.assertEqual(spec.dtype, np.dtype(np.float32))
                    self.assertTrue(spec.available_on_reset)
                    self.assertTrue(spec.available_on_step)
                    self.assertEqual(env.signal_metadata[key]["units"], "ratio")
                    self.assertEqual(
                        env.signal_metadata[key]["nominal_range"],
                        expected_ranges[key],
                    )
                    self.assertIn("not clipped", env.signal_metadata[key]["normalization"])

            _observations, reset_infos = env.reset(seed=[1, 2])
            _observations, _rewards, _terminated, _truncated, step_infos = env.step(
                np.zeros(2, dtype=np.int64)
            )
            for infos in (reset_infos, step_infos):
                for key in normalized_keys:
                    with self.subTest(boundary="reset" if infos is reset_infos else "step", key=key):
                        self.assertEqual(infos[key].dtype, np.float32)
                        self.assertEqual(infos[f"_{key}"].dtype, np.bool_)
                        self.assertTrue(infos[f"_{key}"].all())

            np.testing.assert_allclose(
                reset_infos["ball_x_normalized"],
                reset_infos["ball_x"] / (RAW_WIDTH * FIXED_POINT_ONE),
            )
            np.testing.assert_allclose(
                reset_infos["ball_y_normalized"],
                reset_infos["ball_y"] / 255,
            )
            np.testing.assert_allclose(
                reset_infos["ball_vx_normalized"],
                reset_infos["ball_vx"] / (2 * FIXED_POINT_ONE),
            )
            np.testing.assert_allclose(
                reset_infos["ball_vy_normalized"],
                reset_infos["ball_vy"] / (27 * FIXED_POINT_ONE / 8),
            )
            np.testing.assert_allclose(
                reset_infos["paddle_x_normalized"],
                reset_infos["paddle_x"] / (RAW_WIDTH * FIXED_POINT_ONE),
            )
            np.testing.assert_allclose(
                reset_infos["bricks_destroyed_normalized"],
                reset_infos["bricks_destroyed"] / 216,
            )
        finally:
            env.close()

    def test_player_boundary_renders_canonical_stella_rgb(self) -> None:
        env = make_vec_envs(self.config(), 1, 17)
        try:
            env.reset()
            frame = vector_env_frame(env)
        finally:
            env.close()

        self.assertEqual(frame.shape, (210, 160, 3))
        self.assertEqual(frame.dtype, np.uint8)
        colors = {tuple(color) for color in np.unique(frame.reshape(-1, 3), axis=0)}
        self.assertEqual(
            colors,
            {
                (0, 0, 0),
                (136, 136, 136),
                (200, 72, 72),
                (192, 104, 56),
                (176, 120, 48),
                (160, 160, 40),
                (72, 160, 72),
                (64, 72, 200),
                (64, 152, 128),
            },
        )

    def test_constructs_and_preserves_native_manual_vector_contract(self) -> None:
        config = self.config()
        kwargs = provider_native_vec_kwargs(
            config,
            n_envs=2,
            native_obs_crop=lambda value: value.obs_crop,
            state_weight_mapping=lambda _config: {},
        )

        self.assertEqual(kwargs["num_envs"], 2)
        self.assertEqual(kwargs["obs_resize"], (84, 84))
        self.assertEqual(kwargs["obs_crop"], (0, 0, 0, 0))
        self.assertFalse(kwargs["maxpool_last_two"])
        self.assertNotIn("state", kwargs)
        self.assertNotIn("sticky_action_prob", kwargs)

        env = make_provider_vec_env(config, native_kwargs=kwargs)
        try:
            descriptor = provider_descriptor(
                config,
                env,
                state_weight_mapping=lambda _config: {},
            )
            self.assertIs(env.autoreset_mode, gym.vector.AutoresetMode.DISABLED)
            self.assertEqual(descriptor.start_catalog, ("Start", "checker", "tunnel", "sparse"))
            self.assertEqual(descriptor.lane_start_ids, ("Start", "Start"))
            self.assertEqual(descriptor.render_support, ("rgb_array",))
            self.assertEqual(descriptor.observation_buffer_depth, 2)
            self.assertEqual(env.single_observation_space.shape, (4, 84, 84))
            self.assertEqual(env.single_action_space.n, 4)
            self.assertTrue(descriptor.signal_schema["bricks_remaining"].available_on_reset)
            self.assertTrue(descriptor.signal_schema["bricks_remaining"].available_on_step)

            observations, infos = env.reset(
                seed=[1, 2],
                options={
                    "reset_mask": np.ones(2, dtype=np.bool_),
                    "start_ids": np.asarray(["checker", "sparse"], dtype=object),
                },
            )
            lane_one = observations[1].copy()
            self.assertEqual(infos["start_id"].tolist(), ["checker", "sparse"])
            observations, _rewards, terminated, truncated, infos = env.step(
                np.zeros(2, dtype=np.uint8)
            )
            self.assertFalse(terminated.any())
            self.assertFalse(truncated.any())
            self.assertTrue(infos["_bricks_remaining"].all())
            self.assertTrue(infos["_lives"].all())

            reset_observations, reset_infos = env.reset(
                seed=[3, None],
                options={
                    "reset_mask": np.asarray([True, False], dtype=np.bool_),
                    "start_ids": np.asarray(["Start", None], dtype=object),
                },
            )
            np.testing.assert_array_equal(reset_observations[1], observations[1])
            self.assertFalse(np.array_equal(reset_observations[1], lane_one))
            np.testing.assert_array_equal(reset_infos["_start_id"], [True, False])

            kernel = _bound_task_kernel(config, descriptor, 2)
            self.assertIsNone(kernel._observation_mask)
            self.assertEqual(kernel.action_space, gym.spaces.Discrete(4))
            self.assertTrue(kernel.observation_encoding_is_view)

            life_loss_kernel = _bound_task_kernel(
                self.config(
                    task={
                        "id": "identity",
                        "action": {"set": "native"},
                        "signals": {"lives": "lives"},
                        "events": {
                            "life_loss": {
                                "signal": "lives",
                                "operation": "decrease",
                            }
                        },
                        "termination": {"failure": ["life_loss"]},
                        "reward": {"reward_mode": "native"},
                    }
                ),
                descriptor,
                2,
            )
            self.assertEqual(life_loss_kernel.event_names, ("life_loss",))
        finally:
            env.close()

    def test_rl_zoo_noop_resets_are_seeded_and_keep_live_snapshots_deterministic(
        self,
    ) -> None:
        config = self.config(
            game="Breakout-Atari2600-v0",
            state="Start",
            env_args={
                **self.config().env_args,
                "noop_reset_max": 30,
            },
        )
        kwargs = provider_native_vec_kwargs(
            config,
            n_envs=2,
            native_obs_crop=lambda value: value.obs_crop,
            state_weight_mapping=lambda _config: {},
        )
        env = make_provider_vec_env(config, native_kwargs=kwargs)
        try:
            descriptor = provider_descriptor(
                config,
                env,
                state_weight_mapping=lambda _config: {},
            )
            _, infos = env.reset(seed=[7, 8])
            expected = np.asarray(
                [np.random.default_rng(seed).integers(1, 31, dtype=np.uint64) for seed in (7, 8)],
                dtype=np.uint32,
            )
            np.testing.assert_array_equal(infos["noop_reset_count"], expected)
            self.assertTrue(descriptor.supports_live_snapshots)
            self.assertTrue(descriptor.live_snapshots_deterministic)
        finally:
            env.close()

    def test_inline_table_without_noop_exposes_three_native_actions(self) -> None:
        config = self.config(
            game="Breakout-Atari2600-v0",
            state="Start",
            env_args={
                **self.config().env_args,
                "use_restricted_actions": BREAKOUT_NO_NOOP_ACTIONS,
            },
        )
        kwargs = provider_native_vec_kwargs(
            config,
            n_envs=2,
            native_obs_crop=lambda value: value.obs_crop,
            state_weight_mapping=lambda _config: {},
        )

        env = make_provider_vec_env(config, native_kwargs=kwargs)
        try:
            descriptor = provider_descriptor(
                config,
                env,
                state_weight_mapping=lambda _config: {},
            )
            kernel = _bound_task_kernel(config, descriptor, 2)

            self.assertEqual(env.single_action_space, gym.spaces.Discrete(3))
            self.assertEqual(descriptor.native_action_space, gym.spaces.Discrete(3))
            self.assertEqual(kernel.action_space, gym.spaces.Discrete(3))
            self.assertEqual(descriptor.action_mode, "custom_discrete")
            self.assertEqual(descriptor.action_meanings, ("button", "right", "left"))
            self.assertEqual(descriptor.action_table_hash, BREAKOUT_NO_NOOP_HASH)

            env.reset(
                seed=[1, 2],
                options={
                    "reset_mask": np.ones(2, dtype=np.bool_),
                    "start_ids": np.asarray(["Start", "Start"], dtype=object),
                },
            )
            for actions in (
                np.asarray([0, 0], dtype=np.int64),
                np.asarray([1, 1], dtype=np.int64),
                np.asarray([2, 2], dtype=np.int64),
            ):
                env.step(actions)
        finally:
            env.close()

    def test_rejects_unsupported_canonical_mechanics(self) -> None:
        for update, message in (
            ({"max_pool_frames": True}, "max_pool_frames=true"),
            ({"sticky_action_prob": 0.25}, "sticky_action_prob=0.0"),
        ):
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, message):
                provider_native_vec_kwargs(
                    self.config(**update),
                    n_envs=2,
                    native_obs_crop=lambda value: value.obs_crop,
                    state_weight_mapping=lambda _config: {},
                )


class MarioNativeProviderTests(unittest.TestCase):
    @staticmethod
    def config(**updates):
        values = {
            "env_provider": "env-supermariobrosnes-turbo-emu",
            "game": "SuperMarioBros-Nes-v0",
            "state": "Level1-1",
            "task": {
                "id": "mario",
                "action": {"set": "native"},
                "signals": {
                    "x": ["xscrollHi", "xscrollLo"],
                    "score": "score",
                    "lives": "lives",
                    "level": ["levelHi", "levelLo"],
                },
                "events": {
                    "life_loss": {"signal": "lives", "operation": "decrease"},
                    "level_change": {"signal": "level", "operation": "change"},
                },
                "termination": {"failure": ["life_loss"], "success": ["level_change"]},
                "reward": {"reward_mode": "native"},
            },
        }
        values.update(updates)
        return EnvConfig(**values)

    def test_runtime_matches_turbo_api_v2_releases(self) -> None:
        installed = Version(importlib.metadata.version("env-supermariobrosnes-turbo-emu"))
        self.assertEqual(installed, Version("0.7.3"))
        self.assertEqual(Version(retro.__version__), Version("1.0.1.post48"))
        env_type = super_mario_bros_nes_turbo_vec_env_type()
        self.assertIs(env_type.supports_live_snapshots, True)
        self.assertTrue(callable(getattr(env_type, "capture_snapshots", None)))

    def test_readable_goal_enum_args_normalize_to_provider_enums(self) -> None:
        config = self.config(
            env_args={
                "use_restricted_actions": "basic",
                "inttype": "stable",
                "obs_type": "image",
            }
        )

        kwargs = provider_native_vec_kwargs(
            config,
            n_envs=1,
            native_obs_crop=lambda _config: None,
            state_weight_mapping=lambda _config: {},
        )

        self.assertNotIn("action_set", kwargs)
        self.assertEqual(
            tuple(tuple(labels) for labels in kwargs["use_restricted_actions"]),
            MARIO_ACTION_TABLES["basic"],
        )
        self.assertIs(kwargs["inttype"], retro.data.Integrations.STABLE)
        self.assertIs(kwargs["obs_type"], retro.Observations.IMAGE)

    def test_string_all_info_filter_selects_required_extra_task_signals(self) -> None:
        task = self.config().task
        task["signals"]["game_mode"] = "game_mode"
        task["events"]["game_complete"] = {
            "signal": "game_mode",
            "operation": "equals",
            "value": 2,
            "when": {"signal": "level", "value": [7, 3]},
        }
        task["termination"]["success"] = ["game_complete"]
        config = self.config(task=task, env_args={"info_filter": "all"})

        kwargs = provider_native_vec_kwargs(
            config,
            n_envs=1,
            native_obs_crop=lambda _config: None,
            state_weight_mapping=lambda _config: {},
        )

        self.assertEqual(kwargs["info_filter"]["mode"], "all")
        self.assertIn("game_mode", kwargs["info_filter"]["keys"])
        self.assertIn("x_pos", kwargs["info_filter"]["keys"])

    def test_stable_retro_named_action_preset_is_owned_by_the_provider(self) -> None:
        config = self.config(
            env_provider="env-stableretro-turbo",
            env_args={
                "use_restricted_actions": "basic",
                "inttype": "stable",
                "obs_type": "image",
            },
        )

        kwargs = provider_native_vec_kwargs(
            config,
            n_envs=1,
            native_obs_crop=lambda _config: None,
            state_weight_mapping=lambda _config: {},
        )

        self.assertEqual(
            tuple(tuple(labels) for labels in kwargs["use_restricted_actions"]),
            MARIO_ACTION_TABLES["basic"],
        )

    def test_stable_retro_breakout_receives_named_and_inline_tables_directly(
        self,
    ) -> None:
        for action_request, expected_labels in (
            (
                "simple",
                ((), ("BUTTON",), ("RIGHT",), ("LEFT",)),
            ),
            (
                BREAKOUT_NO_NOOP_ACTIONS,
                (("BUTTON",), ("RIGHT",), ("LEFT",)),
            ),
        ):
            with self.subTest(action_request=action_request):
                config = EnvConfig(
                    env_provider="env-stableretro-turbo",
                    game="Breakout-Atari2600-v0",
                    state="Start",
                    max_pool_frames=False,
                    env_args={
                        "players": 1,
                        "use_restricted_actions": action_request,
                    },
                    task={
                        "id": "identity",
                        "action": {"set": "native"},
                        "signals": {},
                        "events": {},
                        "termination": {},
                        "reward": {"reward_mode": "native"},
                    },
                )
                kwargs = provider_native_vec_kwargs(
                    config,
                    n_envs=len(expected_labels),
                    native_obs_crop=lambda _config: None,
                    state_weight_mapping=lambda _config: {},
                )
                self.assertEqual(
                    tuple(tuple(labels) for labels in kwargs["use_restricted_actions"]),
                    expected_labels,
                )

    def test_constructs_with_disabled_autoreset_and_describes_starts_and_signals(self) -> None:
        class FakeMarioVectorEnv:
            supports_live_snapshots = True
            live_snapshots_deterministic = True
            metadata = {
                "autoreset_mode": gym.vector.AutoresetMode.DISABLED,
                "render_modes": ("rgb_array",),
            }

            def __init__(self, game, *, num_envs, **kwargs):
                self.game = game
                self.num_envs = num_envs
                self.autoreset_mode = gym.vector.AutoresetMode.DISABLED
                self.kwargs = kwargs
                self.single_observation_space = gym.spaces.Box(
                    0, 255, shape=(4, 84, 84), dtype=np.uint8
                )
                self.single_action_space = gym.spaces.MultiBinary(9)
                self.observation_space = gym.vector.utils.batch_space(
                    self.single_observation_space, num_envs
                )
                self.action_space = gym.vector.utils.batch_space(self.single_action_space, num_envs)
                self.state_catalog = ("Level1-1",)
                self._states = ["Level1-1" for _ in range(num_envs)]
                self._state_indices = np.zeros(num_envs, dtype=np.int32)

            def reset(self, *, seed=None, options=None):
                del seed
                options = dict(options or {})
                mask = np.asarray(
                    options.get("reset_mask", np.ones(self.num_envs, dtype=np.bool_)),
                    dtype=np.bool_,
                )
                starts = np.asarray(
                    options.get("state_indices", np.zeros(self.num_envs, dtype=np.int32))
                )
                for lane in np.flatnonzero(mask):
                    if starts[lane] >= 0:
                        self._state_indices[int(lane)] = int(starts[lane])
                        self._states[int(lane)] = self.state_catalog[int(starts[lane])]
                infos = {
                    "xscrollHi": np.zeros(self.num_envs, dtype=np.int64),
                    "xscrollLo": np.zeros(self.num_envs, dtype=np.int64),
                    "score": np.zeros(self.num_envs, dtype=np.int64),
                    "lives": np.full(self.num_envs, 3, dtype=np.int64),
                    "levelHi": np.zeros(self.num_envs, dtype=np.int64),
                    "levelLo": np.zeros(self.num_envs, dtype=np.int64),
                    "state_index": self._state_indices.copy(),
                    "_state_index": mask.copy(),
                    "start_source": np.full(self.num_envs, "environment", dtype=object),
                    "_start_source": mask.copy(),
                }
                return np.zeros((self.num_envs, 4, 84, 84), dtype=np.uint8), infos

            def active_state_indices(self):
                return self._state_indices

            def capture_snapshots(self, mask):
                return tuple(object() if selected else None for selected in mask)

        config = self.config(env_args={"rom_path": None})
        kwargs = provider_native_vec_kwargs(
            config,
            n_envs=2,
            native_obs_crop=lambda _config: None,
            state_weight_mapping=lambda _config: {},
        )
        env = make_provider_vec_env(
            config,
            native_kwargs=kwargs,
            super_mario_vec_env_type=lambda: FakeMarioVectorEnv,
        )
        descriptor = provider_descriptor(
            config,
            env,
            state_weight_mapping=lambda _config: {},
        )

        self.assertIs(env.autoreset_mode, gym.vector.AutoresetMode.DISABLED)
        self.assertNotIn("done_on", env.kwargs)
        self.assertNotIn("autoreset_mode", env.kwargs)
        self.assertEqual(env.kwargs["info_filter"]["mode"], "all")
        self.assertEqual(descriptor.start_catalog, ("Level1-1",))
        self.assertEqual(descriptor.lane_start_ids, ("Level1-1", "Level1-1"))
        self.assertEqual(descriptor.render_support, ("rgb_array",))
        self.assertEqual(descriptor.observation_buffer_depth, 1)
        self.assertTrue(descriptor.supports_live_snapshots)
        self.assertTrue(descriptor.live_snapshots_deterministic)
        self.assertEqual(descriptor.signal_schema["lives"].dtype, np.dtype(np.int64))
        _observations, reset_infos = env.reset(
            seed=[1, None],
            options={"reset_mask": np.asarray([True, False], dtype=np.bool_)},
        )
        np.testing.assert_array_equal(reset_infos["_start_id"], [True, False])
        self.assertEqual(reset_infos["start_id"].tolist(), ["Level1-1", "Level1-1"])

    def test_mario_provider_uses_the_verified_runtime_rom_path(self) -> None:
        class FakeMarioVectorEnv:
            metadata = {"autoreset_mode": gym.vector.AutoresetMode.DISABLED}

            def __init__(self, game, *, num_envs, **kwargs):
                self.game = game
                self.num_envs = num_envs
                self.autoreset_mode = gym.vector.AutoresetMode.DISABLED
                self.kwargs = kwargs

        config = self.config(env_args={"rom_path": None})
        with tempfile.TemporaryDirectory() as temporary:
            rom_path = Path(temporary) / "rom.nes"
            rom_path.write_bytes(b"rom")
            kwargs = provider_native_vec_kwargs(
                config,
                n_envs=2,
                native_obs_crop=lambda _config: None,
                state_weight_mapping=lambda _config: {},
                runtime_rom_path=str(rom_path),
            )
            env = make_provider_vec_env(
                config,
                native_kwargs=kwargs,
                super_mario_vec_env_type=lambda: FakeMarioVectorEnv,
            )

        self.assertEqual(env.kwargs["rom_path"], str(rom_path))

    def test_descriptor_does_not_invent_requested_signals(self) -> None:
        class Native:
            num_envs = 2
            metadata = {"autoreset_mode": gym.vector.AutoresetMode.DISABLED}
            single_observation_space = gym.spaces.Box(0, 255, shape=(4, 84, 84), dtype=np.uint8)
            single_action_space = gym.spaces.MultiBinary(9)
            observation_space = gym.vector.utils.batch_space(single_observation_space, 2)
            action_space = gym.vector.utils.batch_space(single_action_space, 2)
            state_catalog = ("Level1-1",)

            def reset(self, *, seed=None, options=None):
                del seed, options
                infos = {
                    name: np.zeros(2, dtype=np.int64)
                    for name in ("score", "lives", "levelHi", "levelLo")
                }
                return np.zeros((2, 4, 84, 84), dtype=np.uint8), infos

        config = self.config()
        config = EnvConfig(
            **{
                **config.__dict__,
                "task": {
                    **config.task,
                    "signals": {**config.task["signals"], "x": "missing_x"},
                },
            }
        )
        descriptor = provider_descriptor(
            config,
            Native(),
            state_weight_mapping=lambda _config: {},
        )

        self.assertNotIn("missing_x", descriptor.signal_schema)
        with self.assertRaisesRegex(ValueError, "does not expose task signals"):
            MarioTaskDefinition(MarioTaskConfig.from_env_config(config)).bind(descriptor, 2)

    def test_start_adapter_renders_every_native_lane(self) -> None:
        class Env:
            num_envs = 2

            def get_images(self):
                return [np.full((3, 4, 3), lane, dtype=np.uint8) for lane in range(self.num_envs)]

        env = Env()
        frames = _StartInfoAdapter(env).get_images()

        self.assertEqual([frame.shape for frame in frames], [(3, 4, 3), (3, 4, 3)])
        self.assertEqual(int(frames[1][0, 0, 0]), 1)

    def test_rejects_provider_without_disabled_autoreset(self) -> None:
        class OldMarioVectorEnv:
            metadata = {}

            def __init__(self, game, *, num_envs, **kwargs):
                del game, kwargs
                self.num_envs = num_envs

        with self.assertRaisesRegex(RuntimeError, "does not advertise disabled autoreset"):
            make_provider_vec_env(
                self.config(),
                native_kwargs={"num_envs": 2},
                super_mario_vec_env_type=lambda: OldMarioVectorEnv,
            )

    def test_stable_retro_release_without_manual_lifecycle_is_rejected(self) -> None:
        class OldRetroVectorEnv:
            metadata = {}

            def __init__(self, game, **kwargs):
                del game, kwargs

        config = self.config(env_provider="env-stableretro-turbo")
        with self.assertRaisesRegex(RuntimeError, "does not advertise disabled autoreset"):
            make_provider_vec_env(
                config,
                native_kwargs={"num_envs": 2},
                retro_vec_env_type=OldRetroVectorEnv,
            )

    def test_stable_retro_constructs_with_disabled_autoreset(self) -> None:
        class ManualRetroVectorEnv:
            metadata = {
                "autoreset_mode": gym.vector.AutoresetMode.DISABLED,
                "render_modes": ("rgb_array",),
            }

            def __init__(self, game, *, num_envs, **kwargs):
                self.game = game
                self.num_envs = num_envs
                self.autoreset_mode = gym.vector.AutoresetMode.DISABLED
                self.kwargs = kwargs

        config = self.config(env_provider="env-stableretro-turbo")
        kwargs = provider_native_vec_kwargs(
            config,
            n_envs=2,
            native_obs_crop=lambda _config: (32, 0, 0, 0),
            state_weight_mapping=lambda _config: {},
        )
        env = make_provider_vec_env(
            config,
            native_kwargs=kwargs,
            retro_vec_env_type=ManualRetroVectorEnv,
        )

        self.assertIs(env.autoreset_mode, gym.vector.AutoresetMode.DISABLED)
        self.assertEqual(env.kwargs["obs_crop_mode"], "remove")
        self.assertEqual(env.kwargs["obs_crop_fill"], 0)
        self.assertNotIn("done_on", env.kwargs)
        self.assertNotIn("autoreset_mode", env.kwargs)

    def test_stable_retro_receives_catalog_without_sampling_weights(self) -> None:
        config = self.config(
            env_provider="env-stableretro-turbo",
            state="",
            states=("Level1-1", "Level1-4"),
            state_probs=(0.25, 0.75),
        )

        kwargs = provider_native_vec_kwargs(
            config,
            n_envs=3,
            native_obs_crop=lambda _config: None,
            state_weight_mapping=lambda value: dict(
                zip(value.states, value.state_probs, strict=True)
            ),
        )

        self.assertEqual(
            tuple(Path(path).stem for path in kwargs["state_catalog"]),
            ("Level1-1", "Level1-4"),
        )
        self.assertTrue(all(Path(path).is_file() for path in kwargs["state_catalog"]))
        self.assertNotIn("state", kwargs)
        self.assertNotIn("state_probs", kwargs)

    def test_smb_turbo_receives_catalog_without_sampling_weights(self) -> None:
        config = self.config(
            state="",
            states=("Level1-1", "Level1-4"),
            state_probs=(0.25, 0.75),
        )

        kwargs = provider_native_vec_kwargs(
            config,
            n_envs=3,
            native_obs_crop=lambda _config: None,
            state_weight_mapping=lambda value: dict(
                zip(value.states, value.state_probs, strict=True)
            ),
        )

        self.assertEqual(kwargs["state_catalog"], ("Level1-1", "Level1-4"))
        self.assertNotIn("state", kwargs)
        self.assertNotIn("state_probs", kwargs)

    def test_stable_retro_adapter_translates_exact_start_ids_to_state_indices(self) -> None:
        class ManualRetroVectorEnv:
            metadata = {"autoreset_mode": gym.vector.AutoresetMode.DISABLED}

            def __init__(self, game, *, num_envs, **kwargs):
                del game, kwargs
                self.num_envs = num_envs
                self.autoreset_mode = gym.vector.AutoresetMode.DISABLED
                self.state_catalog = ("Level1-1", "Level1-4")
                self.indices = np.zeros(num_envs, dtype=np.int32)
                self.reset_options = None

            def reset(self, *, seed=None, options=None):
                del seed
                self.reset_options = dict(options or {})
                mask = np.asarray(self.reset_options["reset_mask"], dtype=np.bool_)
                requested = np.asarray(self.reset_options["state_indices"], dtype=np.int32)
                self.indices[mask] = requested[mask]
                infos = {
                    "state_index": self.indices.copy(),
                    "_state_index": mask.copy(),
                    "start_source": np.full(self.num_envs, "environment", dtype=object),
                    "_start_source": mask.copy(),
                }
                return np.zeros((self.num_envs, 1), dtype=np.uint8), infos

            def active_state_indices(self):
                return self.indices

        config = self.config(env_provider="env-stableretro-turbo")
        env = make_provider_vec_env(
            config,
            native_kwargs={"num_envs": 2},
            retro_vec_env_type=ManualRetroVectorEnv,
        )
        mask = np.asarray([True, False], dtype=np.bool_)

        _observations, infos = env.reset(
            options={
                "reset_mask": mask,
                "start_ids": np.asarray(["Level1-4", None], dtype=object),
            }
        )

        np.testing.assert_array_equal(env.env.reset_options["state_indices"], [1, -1])
        self.assertEqual(infos["start_id"].tolist(), ["Level1-4", "Level1-1"])
        np.testing.assert_array_equal(infos["_start_id"], mask)

    def test_stable_retro_atari_uses_retro_vec_env_contract(self) -> None:
        class ManualRetroVectorEnv:
            metadata = {"autoreset_mode": gym.vector.AutoresetMode.DISABLED}

            def __init__(self, game, *, num_envs, **kwargs):
                self.game = game
                self.num_envs = num_envs
                self.autoreset_mode = gym.vector.AutoresetMode.DISABLED
                self.kwargs = kwargs
                self.state_catalog = ("Start",)
                self.indices = np.zeros(num_envs, dtype=np.int32)
                self.reset_calls = 0
                self.reset_masks = []
                self.observations = np.zeros((num_envs, 4, 84, 84), dtype=np.uint8)
                self.single_observation_space = gym.spaces.Box(
                    0, 255, shape=(4, 84, 84), dtype=np.uint8
                )
                self.single_action_space = gym.spaces.MultiBinary(8)
                self.observation_space = gym.vector.utils.batch_space(
                    self.single_observation_space, num_envs
                )
                self.action_space = gym.vector.utils.batch_space(self.single_action_space, num_envs)

            def reset(self, *, seed=None, options=None):
                del seed
                self.reset_calls += 1
                options = dict(options or {})
                mask = np.asarray(
                    options.get("reset_mask", np.ones(self.num_envs, dtype=np.bool_)),
                    dtype=np.bool_,
                )
                self.reset_masks.append(mask.copy())
                self.observations[mask] = 0
                self.indices[mask] = 0
                return self.observations.copy(), {
                    "state_index": self.indices.copy(),
                    "_state_index": mask.copy(),
                    "start_source": np.full(self.num_envs, "environment", dtype=object),
                    "_start_source": mask.copy(),
                }

            def active_state_indices(self):
                values = self.indices.copy()
                values.setflags(write=False)
                return values

            def step(self, actions):
                del actions
                self.observations.fill(2)
                self.observations[0] = 9
                return (
                    self.observations.copy(),
                    np.zeros(self.num_envs, dtype=np.float32),
                    np.asarray([True] + [False] * (self.num_envs - 1)),
                    np.zeros(self.num_envs, dtype=np.bool_),
                    {},
                )

            def close(self):
                return None

        config = EnvConfig(
            env_provider="env-stableretro-turbo",
            game="Breakout-Atari2600-v0",
            state="Start",
            obs_crop=(17, 0, 0, 0),
            obs_crop_mode="mask",
            sticky_action_prob=0.25,
            env_args={"info_filter": "all", "num_threads": 8},
            task={
                "id": "identity",
                "action": {"set": "native"},
                "signals": {},
                "events": {},
                "termination": {"max_episode_steps": 54_000},
                "reward": {
                    "reward_mode": "native",
                    "reward_scale": 1.0,
                    "reward_clip": True,
                },
            },
        )
        kwargs = provider_native_vec_kwargs(
            config,
            n_envs=16,
            native_obs_crop=lambda value: value.obs_crop,
            state_weight_mapping=lambda _config: {},
        )
        env = make_provider_vec_env(
            config,
            native_kwargs=kwargs,
            retro_vec_env_type=ManualRetroVectorEnv,
        )

        self.assertEqual(env.game, "Breakout-Atari2600-v0")
        self.assertIs(env.autoreset_mode, gym.vector.AutoresetMode.DISABLED)
        self.assertEqual(env.kwargs["info_filter"], "all")
        self.assertEqual(env.kwargs["num_threads"], 8)
        self.assertEqual(
            Path(env.kwargs["state"]).name,
            "Start.state",
        )
        self.assertTrue(Path(env.kwargs["state"]).is_file())
        self.assertEqual(env.state_catalog, ("Start",))
        self.assertEqual(env.kwargs["obs_resize"], (84, 84))
        self.assertEqual(env.kwargs["obs_crop"], (17, 0, 0, 0))
        self.assertEqual(env.kwargs["obs_crop_mode"], "mask")
        self.assertEqual(env.kwargs["obs_layout"], "chw")
        self.assertEqual(env.kwargs["obs_copy"], "safe_view")
        self.assertEqual(env.kwargs["sticky_action_prob"], 0.25)
        self.assertIs(env.kwargs["reward_clip"], False)
        self.assertIs(env.kwargs["use_fire_reset"], False)
        self.assertNotIn("max_episode_steps", env.kwargs)
        env.reset(seed=123)
        observations, _rewards, terminated, _truncated, infos = env.step(
            np.zeros(16, dtype=np.int64)
        )
        self.assertTrue(terminated[0])
        self.assertEqual(int(observations[0, 0, 0, 0]), 9)
        self.assertEqual(int(observations[1, 0, 0, 0]), 2)
        self.assertEqual(infos, {})
        reset_observations, _reset_infos = env.reset(
            seed=[124] + [None] * 15,
            options={"reset_mask": np.asarray([True] + [False] * 15)},
        )
        self.assertEqual(int(reset_observations[0, 0, 0, 0]), 0)
        self.assertEqual(int(reset_observations[1, 0, 0, 0]), 2)
        self.assertEqual(env.reset_calls, 2)
        np.testing.assert_array_equal(env.reset_masks[-1], [True] + [False] * 15)

        descriptor = provider_descriptor(
            config,
            env,
            state_weight_mapping=lambda _config: {},
        )
        kernel = _bound_task_kernel(config, descriptor, 16)
        self.assertIsNone(kernel._observation_mask)
        self.assertEqual(descriptor.observation_buffer_depth, 1)
        self.assertTrue(kernel.observation_encoding_is_view)


class VizdoomTurboProviderTests(unittest.TestCase):
    class FakeVizdoomEnv:
        supports_live_snapshots = True
        live_snapshots_deterministic = True
        metadata = {
            "autoreset_mode": gym.vector.AutoresetMode.DISABLED,
            "render_modes": ["rgb_array"],
        }

        def __init__(self, game: str, **kwargs):
            self.game = game
            self.kwargs = kwargs
            self.num_envs = int(kwargs["num_envs"])
            self.autoreset_mode = gym.vector.AutoresetMode.DISABLED
            self.obs_copy = kwargs["obs_copy"]
            self.state_catalog = ("default",)
            self.single_observation_space = gym.spaces.Box(
                0, 255, shape=(4, 84, 84), dtype=np.uint8
            )
            self.observation_space = gym.vector.utils.batch_space(
                self.single_observation_space, self.num_envs
            )
            self.single_action_space = gym.spaces.Discrete(4)
            self.action_space = gym.vector.utils.batch_space(
                self.single_action_space, self.num_envs
            )
            self.signal_schema = {
                "health": {"dtype": np.float64, "shape": ()},
            }

        def capture_snapshots(self, mask):
            return tuple(object() if selected else None for selected in mask)

    def test_compiles_and_constructs_native_vector_provider(self) -> None:
        config = EnvConfig(
            env_provider="env-vizdoom-turbo",
            game="VizdoomBasic-v1",
            state="",
            sticky_action_prob=0.25,
            env_args={
                "use_restricted_actions": "discrete",
                "game_variables": ("HEALTH",),
            },
            task={
                "id": "identity",
                "action": {"set": "native"},
                "signals": {},
                "events": {},
                "termination": {},
                "reward": {"reward_mode": "native"},
            },
        )
        kwargs = provider_native_vec_kwargs(
            config,
            n_envs=6,
            native_obs_crop=lambda _config: (12, 0, 0, 0),
            state_weight_mapping=lambda _config: {},
        )

        env = make_provider_vec_env(
            config,
            native_kwargs=kwargs,
            vizdoom_vec_env_type=lambda: self.FakeVizdoomEnv,
        )
        descriptor = provider_descriptor(
            config,
            env,
            state_weight_mapping=lambda _config: {},
        )

        self.assertEqual(env.game, "VizdoomBasic-v1")
        self.assertEqual(env.num_envs, 6)
        self.assertEqual(env.kwargs["obs_resize"], (84, 84))
        self.assertEqual(env.kwargs["obs_crop"], (12, 0, 0, 0))
        self.assertEqual(env.kwargs["frame_stack"], 4)
        self.assertNotIn("info_frame_stack_keys", env.kwargs)
        self.assertEqual(env.kwargs["sticky_action_prob"], 0.25)
        self.assertEqual(env.kwargs["state"], None)
        self.assertEqual(env.kwargs["use_restricted_actions"], "discrete")
        self.assertEqual(env.kwargs["game_variables"], ("HEALTH",))
        self.assertEqual(descriptor.observation_buffer_depth, 1)
        self.assertTrue(descriptor.supports_live_snapshots)
        self.assertTrue(descriptor.live_snapshots_deterministic)
        self.assertIn("health", descriptor.signal_schema)

    def test_descriptor_accepts_runtime_boundary_but_rejects_missing_provider_signal(
        self,
    ) -> None:
        config = EnvConfig(
            env_provider="env-vizdoom-turbo",
            game="VizdoomHealthGathering-v1",
            state="",
            task={
                "id": "identity",
                "action": {"set": "native"},
                "signals": {"native_timeout": "provider_truncated"},
                "events": {},
                "termination": {},
                "reward": {"reward_mode": "native"},
            },
        )
        env = self.FakeVizdoomEnv(
            "VizdoomHealthGathering-v1",
            num_envs=2,
            obs_copy="safe_view",
        )
        env.metadata = {**env.metadata, "turbo_api_version": 2}
        turbo_contract = mock.Mock(
            observation_ownership="safe_view",
            observation_buffer_depth=2,
        )

        with mock.patch(
            "gradlab.env_providers.validate_turbo_vector_env",
            return_value=turbo_contract,
        ):
            descriptor = provider_descriptor(
                config,
                env,
                state_weight_mapping=lambda _config: {},
            )

        self.assertNotIn("provider_truncated", descriptor.signal_schema)

        invalid_config = EnvConfig(
            **{
                **config.__dict__,
                "task": {
                    **config.task,
                    "signals": {"missing": "missing_provider_signal"},
                },
            }
        )
        with (
            mock.patch(
                "gradlab.env_providers.validate_turbo_vector_env",
                return_value=turbo_contract,
            ),
            self.assertRaisesRegex(
                ValueError,
                r"does not declare task signal\(s\) \['missing_provider_signal'\]",
            ),
        ):
            provider_descriptor(
                invalid_config,
                env,
                state_weight_mapping=lambda _config: {},
            )

    def test_closes_native_environment_when_strict_validation_fails(self) -> None:
        constructed = []

        class InvalidVizdoomEnv:
            metadata = {
                "autoreset_mode": gym.vector.AutoresetMode.DISABLED,
            }

            def __init__(self, game: str, **kwargs):
                self.game = game
                self.kwargs = kwargs
                self.closed = False
                constructed.append(self)

            def close(self):
                self.closed = True

        config = EnvConfig(
            env_provider="env-vizdoom-turbo",
            game="VizdoomBasic-v1",
            state="",
            task={
                "id": "identity",
                "action": {"set": "native"},
                "signals": {},
                "events": {},
                "termination": {},
                "reward": {"reward_mode": "native"},
            },
        )
        with (
            mock.patch("env_vizdoom_turbo.EnvViZDoomTurboVecEnv", InvalidVizdoomEnv),
            mock.patch(
                "gradlab.env_providers.validate_turbo_vector_env",
                side_effect=RuntimeError("strict contract mismatch"),
            ),
            self.assertRaisesRegex(RuntimeError, "strict contract mismatch"),
        ):
            make_provider_vec_env(config, native_kwargs={"num_envs": 2})

        self.assertEqual(len(constructed), 1)
        self.assertTrue(constructed[0].closed)


class AleManualLifecycleTests(unittest.TestCase):
    def test_next_step_engine_cannot_autoreset_behind_runtime(self) -> None:
        class FakeAle:
            num_envs = 2
            metadata = {"autoreset_mode": gym.vector.AutoresetMode.NEXT_STEP}

            def __init__(self):
                self.steps = 0

            def reset(self, *, seed=None, options=None):
                del seed, options
                return np.zeros((2, 1), dtype=np.uint8), {}

            def step(self, actions):
                del actions
                self.steps += 1
                return (
                    np.zeros((2, 1), dtype=np.uint8),
                    np.zeros(2, dtype=np.float32),
                    np.asarray([self.steps == 1, False]),
                    np.zeros(2, dtype=np.bool_),
                    {},
                )

            def close(self):
                return None

        env = _AleManualResetAdapter(FakeAle())
        env.reset(seed=[1, 2])
        env.step(np.zeros(2, dtype=np.int64))
        with self.assertRaisesRegex(RuntimeError, "explicitly reset"):
            env.step(np.zeros(2, dtype=np.int64))
        env.reset(
            seed=[3, None],
            options={"reset_mask": np.asarray([True, False], dtype=np.bool_)},
        )
        env.step(np.zeros(2, dtype=np.int64))

    def test_cached_policy_frames_are_renderable_rgb(self) -> None:
        class FakeAle:
            num_envs = 2
            metadata = {"autoreset_mode": gym.vector.AutoresetMode.NEXT_STEP}

            def reset(self, *, seed=None, options=None):
                del seed, options
                observations = np.arange(2 * 4 * 3 * 5, dtype=np.uint8).reshape(2, 4, 3, 5)
                return observations, {}

            def close(self):
                return None

        env = _AleManualResetAdapter(FakeAle())
        env.reset(options={"reset_mask": np.ones(2, dtype=np.bool_)})
        frames = env.get_images()

        self.assertEqual([frame.shape for frame in frames], [(3, 5, 3), (3, 5, 3)])
        np.testing.assert_array_equal(frames[0][..., 0], frames[0][..., 1])


class GraDoomProviderTests(unittest.TestCase):
    def test_torch_transport_env_is_bridged_to_host_numpy_surface(self) -> None:
        import torch

        class FakeGraDoomEnv:
            transport = "torch"
            device = torch.device("cpu")
            state_catalog = ("default",)
            metadata = {"autoreset_mode": gym.vector.AutoresetMode.DISABLED}

            def __init__(self, game, **kwargs):
                self.game = game
                self.num_envs = int(kwargs.get("num_envs", 2))
                self.step_action_devices: list[torch.device] = []

            def active_state_indices(self):
                return torch.zeros(self.num_envs, dtype=torch.int32)

            def reset(self, *, seed=None, options=None):
                mask = torch.ones(self.num_envs, dtype=torch.bool)
                return torch.zeros((self.num_envs, 4, 84, 84), dtype=torch.uint8), {
                    "state_index": torch.zeros(self.num_envs, dtype=torch.int32),
                    "start_source": torch.zeros(self.num_envs, dtype=torch.int8),
                    "noop_reset_count": torch.zeros(self.num_envs, dtype=torch.int64),
                    "_state_index": mask.clone(),
                    "_start_source": mask.clone(),
                    "_noop_reset_count": mask.clone(),
                }

            def step(self, actions):
                if not isinstance(actions, torch.Tensor):
                    raise AssertionError("actions must be a torch tensor")
                self.step_action_devices.append(actions.device)
                mask = torch.zeros(self.num_envs, dtype=torch.bool)
                return (
                    torch.zeros((self.num_envs, 4, 84, 84), dtype=torch.uint8),
                    torch.zeros(self.num_envs, dtype=torch.float32),
                    mask.clone(),
                    mask.clone(),
                    {"killcount": torch.zeros(self.num_envs, dtype=torch.int64)},
                )

        config = EnvConfig(
            env_provider="env-gradoom-turbo-torch",
            game="VizdoomDeathmatch-v1",
            env_args={},
        )
        env = make_provider_vec_env(
            config,
            native_kwargs={"num_envs": 2},
            gradoom_env_type=lambda: FakeGraDoomEnv,
        )

        self.assertEqual(env.transport, "numpy")
        observations, infos = env.reset()
        self.assertIsInstance(observations, np.ndarray)
        self.assertEqual(observations.shape, (2, 4, 84, 84))
        self.assertIsInstance(infos["state_index"], np.ndarray)
        self.assertEqual(infos["start_id"].tolist(), ["default", "default"])
        observations, rewards, terminated, truncated, step_infos = env.step(
            np.zeros(2, dtype=np.int64)
        )
        self.assertIsInstance(observations, np.ndarray)
        self.assertIsInstance(rewards, np.ndarray)
        self.assertIsInstance(step_infos["killcount"], np.ndarray)
        self.assertIsInstance(env.active_state_indices(), np.ndarray)
        self.assertEqual(env.num_envs, 2)


if __name__ == "__main__":
    unittest.main()
