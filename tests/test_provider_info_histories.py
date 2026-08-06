from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from unittest import mock

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from gradlab.actor_critic_policy import SharedActorCriticFeatureExtractor, SharedActorCriticPolicy
from gradlab.batch_runtime import BatchRuntime, ProviderDescriptor, SignalSpec
from gradlab.env import EnvConfig
from gradlab.env_providers import provider_descriptor, provider_native_vec_kwargs
from gradlab.model_inputs import ContextTaskKernel, normalize_model_inputs
from gradlab.policy_execution import compile_policy_execution_contract
from gradlab.task_kernels import IdentityTaskDefinition


FRAME_STACK = 3


def _task(*, include_current_health: bool = False) -> dict:
    context = {
        "health": {
            "signal": "health",
            "update": "transition",
            "history": "provider_frame_stack",
            "encoding": {
                "kind": "continuous",
                "scale": 0.01,
                "offset": 0.0,
                "low": 0.0,
                "high": 3.0,
            },
        },
        "position": {
            "signal": "position",
            "update": "transition",
            "history": "provider_frame_stack",
            "encoding": {
                "kind": "continuous",
                "scale": [0.1, 0.01],
                "offset": 0.0,
                "low": [-10.0, -10.0],
                "high": [10.0, 10.0],
            },
        },
        "selected_weapon": {
            "signal": "selected_weapon",
            "update": "transition",
            "history": "provider_frame_stack",
            "encoding": {"kind": "categorical", "values": [1, 2, 3]},
        },
    }
    if include_current_health:
        context["health_current"] = {
            "signal": "health",
            "update": "transition",
            "encoding": {
                "kind": "continuous",
                "scale": 0.01,
                "offset": 0.0,
                "low": 0.0,
                "high": 3.0,
            },
        }
    return {
        "id": "identity",
        "action": {"set": "native"},
        "signals": {
            "health": "health",
            "position": "position",
            "selected_weapon": "selected_weapon",
        },
        "model_inputs": {"schema_version": 1, "context": context},
        "events": {},
        "termination": {},
        "reward": {"reward_mode": "native"},
    }


def _descriptor(*, frame_stack: int = FRAME_STACK) -> ProviderDescriptor:
    schemas = {
        "health": SignalSpec("health", np.float32),
        "position": SignalSpec("position", np.float32, shape=(2,)),
        "selected_weapon": SignalSpec("selected_weapon", np.int64),
        "health_frame_stack": SignalSpec("health_frame_stack", np.float32, shape=(frame_stack,)),
        "position_frame_stack": SignalSpec(
            "position_frame_stack", np.float32, shape=(frame_stack, 2)
        ),
        "selected_weapon_frame_stack": SignalSpec(
            "selected_weapon_frame_stack", np.int64, shape=(frame_stack,)
        ),
    }
    return ProviderDescriptor(
        provider_id="fake-history-provider",
        native_observation_space=gym.spaces.Box(
            -1000.0,
            1000.0,
            shape=(2,),
            dtype=np.float32,
        ),
        native_action_space=gym.spaces.Discrete(2),
        signal_schema=schemas,
        observation_ownership="owned",
        observation_buffer_depth=None,
    )


def _signals(
    health: np.ndarray,
    position: np.ndarray,
    selected_weapon: np.ndarray,
    *,
    presence: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    num_envs = int(health.shape[0])
    present = (
        np.ones(num_envs, dtype=np.bool_)
        if presence is None
        else np.asarray(presence, dtype=np.bool_)
    )
    return {
        "health": health[:, -1].astype(np.float32),
        "_health": present,
        "health_frame_stack": health.astype(np.float32),
        "_health_frame_stack": present,
        "position": position[:, -1].astype(np.float32),
        "_position": present,
        "position_frame_stack": position.astype(np.float32),
        "_position_frame_stack": present,
        "selected_weapon": selected_weapon[:, -1].astype(np.int64),
        "_selected_weapon": present,
        "selected_weapon_frame_stack": selected_weapon.astype(np.int64),
        "_selected_weapon_frame_stack": present,
    }


def _kernel(*, frame_stack: int = FRAME_STACK, include_current_health: bool = False):
    descriptor = _descriptor(frame_stack=frame_stack)
    task = _task(include_current_health=include_current_health)
    base = IdentityTaskDefinition(signals=task["signals"]).bind(descriptor, 2)
    return ContextTaskKernel(base, descriptor, task)


def test_history_configuration_is_explicit_and_rejects_non_transition_modes() -> None:
    normalized = normalize_model_inputs(_task()["model_inputs"])

    assert normalized["context"]["health"]["history"] == "provider_frame_stack"
    invalid = deepcopy(_task()["model_inputs"])
    invalid["context"]["health"]["update"] = "episode"
    with pytest.raises(ValueError, match="must update on every transition"):
        normalize_model_inputs(invalid)


def test_context_kernel_preserves_scalar_vector_and_categorical_provider_order() -> None:
    kernel = _kernel()
    observations = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    health = np.asarray([[100, 90, 80], [200, 210, 220]], dtype=np.float32)
    position = np.asarray(
        [
            [[1, 10], [2, 20], [3, 30]],
            [[4, 40], [5, 50], [6, 60]],
        ],
        dtype=np.float32,
    )
    weapons = np.asarray([[1, 2, 3], [3, 2, 1]], dtype=np.int64)

    kernel.on_reset(
        observations,
        _signals(health, position, weapons),
        np.ones(2, dtype=np.bool_),
    )
    encoded = kernel.encode_observations(observations)

    np.testing.assert_allclose(encoded["context/health"], health[..., None] * 0.01)
    np.testing.assert_allclose(
        encoded["context/position"],
        position * np.asarray([0.1, 0.01], dtype=np.float32),
    )
    np.testing.assert_array_equal(encoded["context/selected_weapon"], [[0, 1, 2], [2, 1, 0]])
    assert tuple(encoded) == (
        "observation",
        "context/health",
        "context/position",
        "context/selected_weapon",
    )
    history_contract = kernel.model_input_contract["context"]["health"]["history"]
    assert history_contract == {
        "kind": "provider_frame_stack",
        "depth": 3,
        "order": "oldest_to_newest",
        "flattening": "temporal_major",
        "base_sources": ["health"],
    }


def test_current_value_is_not_duplicated_unless_explicitly_declared() -> None:
    observations = np.zeros((2, 2), dtype=np.float32)
    health = np.asarray([[100, 90, 80], [200, 210, 220]], dtype=np.float32)
    position = np.zeros((2, FRAME_STACK, 2), dtype=np.float32)
    weapons = np.ones((2, FRAME_STACK), dtype=np.int64)

    history_only = _kernel()
    history_only.on_reset(
        observations,
        _signals(health, position, weapons),
        np.ones(2, dtype=np.bool_),
    )
    assert "context/health_current" not in history_only.encode_observations(observations)

    explicit_both = _kernel(include_current_health=True)
    explicit_both.on_reset(
        observations,
        _signals(health, position, weapons),
        np.ones(2, dtype=np.bool_),
    )
    encoded = explicit_both.encode_observations(observations)
    np.testing.assert_allclose(encoded["context/health_current"], [[0.8], [2.2]])
    np.testing.assert_allclose(
        encoded["context/health"][:, -1],
        encoded["context/health_current"],
    )


def test_masked_reset_replaces_only_selected_provider_history_lanes() -> None:
    kernel = _kernel()
    observations = np.zeros((2, 2), dtype=np.float32)
    initial_health = np.asarray([[100, 90, 80], [200, 210, 220]], dtype=np.float32)
    initial_position = np.zeros((2, FRAME_STACK, 2), dtype=np.float32)
    initial_weapons = np.ones((2, FRAME_STACK), dtype=np.int64)
    kernel.on_reset(
        observations,
        _signals(initial_health, initial_position, initial_weapons),
        np.ones(2, dtype=np.bool_),
    )

    reset_health = np.asarray([[50, 50, 50], [999, 999, 999]], dtype=np.float32)
    kernel.on_reset(
        observations,
        _signals(
            reset_health,
            np.ones((2, FRAME_STACK, 2), dtype=np.float32),
            np.full((2, FRAME_STACK), 2, dtype=np.int64),
            presence=np.asarray([True, False]),
        ),
        np.asarray([True, False]),
    )

    encoded = kernel.encode_observations(observations)
    np.testing.assert_allclose(encoded["context/health"][0, :, 0], [0.5, 0.5, 0.5])
    np.testing.assert_allclose(encoded["context/health"][1, :, 0], [2.0, 2.1, 2.2])


def _policy_model(*, history_hidden_sizes: list[int] | None = None) -> dict:
    result = {
        "schema_version": 2,
        "encoder": {"kind": "flatten"},
        "fusion": {"hidden_sizes": [], "activation": "tanh"},
        "normalize_images": False,
        "orthogonal_init": False,
    }
    if history_hidden_sizes is not None:
        result["info_history_encoder"] = {
            "hidden_sizes": history_hidden_sizes,
            "activation": "relu",
        }
    return result


def test_policy_fusion_is_temporal_major_after_the_image_encoder() -> None:
    kernel = _kernel()
    observations = np.asarray([[7.0, 8.0], [9.0, 10.0]], dtype=np.float32)
    health = np.asarray([[100, 90, 80], [200, 210, 220]], dtype=np.float32)
    position = np.asarray(
        [
            [[1, 10], [2, 20], [3, 30]],
            [[4, 40], [5, 50], [6, 60]],
        ],
        dtype=np.float32,
    )
    weapons = np.asarray([[1, 2, 3], [3, 2, 1]], dtype=np.int64)
    kernel.on_reset(
        observations,
        _signals(health, position, weapons),
        np.ones(2, dtype=np.bool_),
    )
    encoded = kernel.encode_observations(observations)
    policy = SharedActorCriticPolicy(
        kernel.observation_space,
        kernel.action_space,
        lambda _: 1e-3,
        policy_model=_policy_model(),
    )
    tensor, _ = policy.obs_to_tensor(encoded)

    features = policy.extract_features(tensor)

    expected_lane_zero = torch.tensor(
        [
            7.0,
            8.0,
            1.0,
            0.1,
            0.1,
            1.0,
            0.0,
            0.0,
            0.9,
            0.2,
            0.2,
            0.0,
            1.0,
            0.0,
            0.8,
            0.3,
            0.3,
            0.0,
            0.0,
            1.0,
        ]
    )
    torch.testing.assert_close(features[0], expected_lane_zero)

    env = SimpleNamespace(runtime=SimpleNamespace(kernel=kernel))
    model = SimpleNamespace(policy=policy)
    contract = compile_policy_execution_contract(model, env)
    assert contract is not None
    assert contract["model_inputs"]["context"]["health"]["history"]["depth"] == 3

    policy.features_extractor.provider_history_names = ()
    with pytest.raises(ValueError, match="provider histories disagree"):
        compile_policy_execution_contract(model, env)


def test_history_encoder_dimensions_frame_stack_one_and_state_round_trip() -> None:
    kernel = _kernel(frame_stack=1)
    observations = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    kernel.on_reset(
        observations,
        _signals(
            np.asarray([[100], [200]], dtype=np.float32),
            np.asarray([[[1, 10]], [[2, 20]]], dtype=np.float32),
            np.asarray([[1], [2]], dtype=np.int64),
        ),
        np.ones(2, dtype=np.bool_),
    )
    encoded = kernel.encode_observations(observations)
    policy_model = _policy_model(history_hidden_sizes=[5])
    policy = SharedActorCriticPolicy(
        kernel.observation_space,
        kernel.action_space,
        lambda _: 1e-3,
        policy_model=policy_model,
    )
    extractor = policy.features_extractor
    assert isinstance(extractor, SharedActorCriticFeatureExtractor)
    assert extractor.history_depth == 1
    assert isinstance(extractor.info_history_encoder, torch.nn.Sequential)
    history_linear = extractor.info_history_encoder[0]
    assert isinstance(history_linear, torch.nn.Linear)
    assert history_linear.in_features == 6
    assert history_linear.out_features == 5
    assert extractor.features_dim == 7

    tensor, _ = policy.obs_to_tensor(encoded)
    expected = policy.extract_features(tensor).detach().clone()
    restored = SharedActorCriticPolicy(
        kernel.observation_space,
        kernel.action_space,
        lambda _: 1e-3,
        policy_model=policy_model,
    )
    restored.load_state_dict(policy.state_dict())
    torch.testing.assert_close(restored.extract_features(tensor), expected)


def test_ppo_checkpoint_round_trips_history_spaces_and_encoder(tmp_path) -> None:
    kernel = _kernel()
    observations = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    kernel.on_reset(
        observations,
        _signals(
            np.asarray([[100, 90, 80], [200, 210, 220]], dtype=np.float32),
            np.asarray(
                [
                    [[1, 10], [2, 20], [3, 30]],
                    [[4, 40], [5, 50], [6, 60]],
                ],
                dtype=np.float32,
            ),
            np.asarray([[1, 2, 3], [3, 2, 1]], dtype=np.int64),
        ),
        np.ones(2, dtype=np.bool_),
    )
    sample = {
        key: np.asarray(value[0]).copy()
        for key, value in kernel.encode_observations(observations).items()
    }

    class StaticHistoryEnv(gym.Env):
        observation_space = kernel.observation_space
        action_space = kernel.action_space

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return deepcopy(sample), {}

        def step(self, action):
            return deepcopy(sample), float(action), True, False, {}

    env = DummyVecEnv([StaticHistoryEnv])
    policy_model = _policy_model(history_hidden_sizes=[5])
    try:
        model = PPO(
            SharedActorCriticPolicy,
            env,
            n_steps=2,
            batch_size=2,
            n_epochs=1,
            policy_kwargs={"policy_model": policy_model},
            verbose=0,
        )
        model.learn(total_timesteps=2)
        checkpoint = tmp_path / "provider-history.zip"
        model.save(checkpoint)

        loaded = PPO.load(checkpoint, env=env)

        assert loaded.policy.policy_model["info_history_encoder"] == {
            "hidden_sizes": [5],
            "activation": "relu",
        }
        original_history_state = {
            name: value
            for name, value in model.policy.state_dict().items()
            if "info_history_encoder" in name
        }
        loaded_history_state = {
            name: value
            for name, value in loaded.policy.state_dict().items()
            if "info_history_encoder" in name
        }
        assert original_history_state.keys() == loaded_history_state.keys()
        for name in original_history_state:
            torch.testing.assert_close(original_history_state[name], loaded_history_state[name])
    finally:
        env.close()


class _LifecycleProvider:
    def __init__(self) -> None:
        self.num_envs = 2
        self.single_observation_space = _descriptor().native_observation_space
        self.single_action_space = _descriptor().native_action_space
        self.frame_skip = 4
        self._observations = np.zeros((2, 2), dtype=np.float32)
        self._reset_count = np.zeros(2, dtype=np.int64)

    def reset(self, *, seed=None, options=None):
        del seed
        options = dict(options or {})
        mask = np.asarray(
            options.get("reset_mask", np.ones(2, dtype=np.bool_)),
            dtype=np.bool_,
        )
        self._reset_count[mask] += 1
        health = np.asarray([[50, 50, 50], [200, 200, 200]], dtype=np.float32)
        health[0] += 50 if self._reset_count[0] == 1 else 0
        self._observations[mask, 0] = health[mask, -1]
        signals = _signals(
            health,
            np.zeros((2, FRAME_STACK, 2), dtype=np.float32),
            np.ones((2, FRAME_STACK), dtype=np.int64),
            presence=mask,
        )
        return self._observations, signals

    def step(self, actions):
        del actions
        health = np.asarray([[100, 95, 90], [200, 205, 210]], dtype=np.float32)
        self._observations[:, 0] = health[:, -1]
        return (
            self._observations,
            np.zeros(2, dtype=np.float32),
            np.asarray([False, False]),
            np.asarray([True, False]),
            _signals(
                health,
                np.zeros((2, FRAME_STACK, 2), dtype=np.float32),
                np.ones((2, FRAME_STACK), dtype=np.int64),
            ),
        )

    def close(self) -> None:
        return None


def test_terminal_history_survives_masked_reset_and_matches_terminal_observation() -> None:
    descriptor = _descriptor()
    task = _task()
    task["events"] = {"health_decreased": {"signal": "health", "operation": "decrease"}}
    kernel = ContextTaskKernel(
        IdentityTaskDefinition(
            signals=task["signals"],
            events=task["events"],
        ).bind(descriptor, 2),
        descriptor,
        task,
    )
    runtime = BatchRuntime(_LifecycleProvider(), descriptor, kernel, run_seed=7)
    runtime.reset(seed=7)

    step = runtime.step(np.zeros(2, dtype=np.int64))

    assert step.final_observations is not None
    np.testing.assert_allclose(
        step.final_observations["context/health"][0, :, 0],
        [1.0, 0.95, 0.9],
    )
    np.testing.assert_allclose(step.observations["context/health"][0, :, 0], [0.5, 0.5, 0.5])
    np.testing.assert_allclose(step.observations["context/health"][1, :, 0], [2.0, 2.05, 2.1])
    assert step.transition_info["health"][0] == pytest.approx(90.0)
    assert step.transition_info["health_frame_stack"][0].tolist() == [100.0, 95.0, 90.0]
    assert any("health_decreased" in record.events for record in runtime.drain_records())


def _vizdoom_config() -> EnvConfig:
    task = _task()
    return EnvConfig(
        env_provider="vizdoom-turbo",
        game="VizdoomDeathmatch-v1",
        state="",
        env_args={
            "rom_path": "/fake/doom2.wad",
            "info_filter": {"mode": "all", "keys": ["killcount"]},
        },
        task=task,
    )


def test_vizdoom_constructor_receives_base_keys_and_augmented_info_filter() -> None:
    with mock.patch(
        "gradlab.vizdoom_assets.resolve_vizdoom_iwad_path",
        return_value="/fake/doom2.wad",
    ):
        kwargs = provider_native_vec_kwargs(
            _vizdoom_config(),
            n_envs=2,
            native_obs_crop=lambda _config: None,
            state_weight_mapping=lambda _config: {},
        )

    assert kwargs["info_frame_stack_keys"] == ("health", "position", "selected_weapon")
    assert kwargs["info_filter"] == {
        "mode": "all",
        "keys": ("killcount", "health", "position", "selected_weapon"),
    }


class _DescriptorEnv:
    metadata = {"turbo_api_version": 1, "render_modes": ("rgb_array",)}
    frame_stack = FRAME_STACK
    num_envs = 2
    single_observation_space = _descriptor().native_observation_space
    single_action_space = _descriptor().native_action_space
    observation_space = gym.vector.utils.batch_space(single_observation_space, num_envs)
    action_space = gym.vector.utils.batch_space(single_action_space, num_envs)
    state_catalog = ()
    signal_schema = {
        name: {
            "dtype": spec.dtype,
            "shape": spec.shape,
            "available_on_reset": spec.available_on_reset,
            "available_on_step": spec.available_on_step,
        }
        for name, spec in _descriptor().signal_schema.items()
    }


def _turbo_contract(*, supports_history: bool):
    return SimpleNamespace(
        capabilities={"supports_info_frame_stack": supports_history},
        observation_ownership="owned",
        observation_buffer_depth=None,
    )


def test_descriptor_validates_capability_and_declared_history_schema() -> None:
    with mock.patch(
        "gradlab.env_providers.validate_turbo_vector_env",
        return_value=_turbo_contract(supports_history=True),
    ):
        descriptor = provider_descriptor(
            _vizdoom_config(),
            _DescriptorEnv(),
            state_weight_mapping=lambda _config: {},
        )
    assert descriptor.signal_schema["health_frame_stack"].shape == (FRAME_STACK,)

    with (
        mock.patch(
            "gradlab.env_providers.validate_turbo_vector_env",
            return_value=_turbo_contract(supports_history=False),
        ),
        pytest.raises(
            ValueError, match="does not support requested provider-owned info frame stacks"
        ),
    ):
        provider_descriptor(
            _vizdoom_config(),
            _DescriptorEnv(),
            state_weight_mapping=lambda _config: {},
        )


def test_descriptor_rejects_history_shape_drift() -> None:
    env = _DescriptorEnv()
    env.signal_schema = deepcopy(env.signal_schema)
    env.signal_schema["health_frame_stack"]["shape"] = (FRAME_STACK + 1,)
    with (
        mock.patch(
            "gradlab.env_providers.validate_turbo_vector_env",
            return_value=_turbo_contract(supports_history=True),
        ),
        pytest.raises(ValueError, match="must have shape"),
    ):
        provider_descriptor(
            _vizdoom_config(),
            env,
            state_weight_mapping=lambda _config: {},
        )
