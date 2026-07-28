from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from gradlab.turbo_api import CAPABILITY_KEYS, TURBO_API_VERSION, validate_turbo_vector_env


def _contract_env():
    active = np.zeros(2, dtype=np.int32)
    active.setflags(write=False)
    capabilities = {
        "supported_action_modes": ("custom_discrete",),
        "supported_observation_layouts": ("chw",),
        "supported_resize_algorithms": ("area",),
        "supported_observation_copy_modes": ("copy", "safe_view", "unsafe_view"),
        "supports_maxpool_last_two": False,
        "supports_sticky_action_prob": True,
        "supports_reward_clipping": True,
        "supports_noop_reset": True,
        "supports_state_catalog": True,
        "supports_live_snapshots": True,
        "supports_per_lane_rgb": True,
    }
    assert set(capabilities) == CAPABILITY_KEYS
    signal_spec = MappingProxyType(
        {
            "dtype": np.dtype(np.int64),
            "shape": (),
            "available_on_reset": True,
            "available_on_step": True,
        }
    )
    return SimpleNamespace(
        metadata={
            "turbo_api_version": TURBO_API_VERSION,
            "render_modes": ("rgb_array",),
        },
        num_envs=2,
        num_threads=2,
        frame_skip=4,
        frame_stack=4,
        obs_layout="chw",
        obs_copy="safe_view",
        render_mode="rgb_array",
        closed=False,
        buttons=("NOOP", "FIRE"),
        action_mode="custom_discrete",
        action_preset="simple",
        action_table=(0, 1),
        action_meanings=("NOOP", "FIRE"),
        action_table_hash="sha256",
        single_action_space=gym.spaces.Discrete(2),
        state_catalog=("Start",),
        active_state_indices=lambda: active,
        capabilities=MappingProxyType(capabilities),
        observation_ownership="safe_view",
        observation_buffer_depth=2,
        supports_live_snapshots=True,
        live_snapshots_deterministic=True,
        signal_schema=MappingProxyType({"score": signal_spec}),
        render_lane=lambda lane: np.zeros((2, 2, 3), dtype=np.uint8),
        get_images=lambda: [
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.zeros((2, 2, 3), dtype=np.uint8),
        ],
        render=lambda: np.zeros((2, 2, 3), dtype=np.uint8),
    )


def test_validates_the_declarative_v1_surface_without_resetting_or_stepping() -> None:
    contract = validate_turbo_vector_env(_contract_env(), "test-turbo")

    assert contract.api_version == TURBO_API_VERSION
    assert contract.observation_ownership == "safe_view"
    assert contract.observation_buffer_depth == 2


def test_rejects_mutable_contract_declarations() -> None:
    env = _contract_env()
    env.capabilities = dict(env.capabilities)

    with pytest.raises(TypeError, match="capabilities must be immutable"):
        validate_turbo_vector_env(env, "test-turbo")


def test_rejects_action_table_cardinality_that_disagrees_with_the_space() -> None:
    env = _contract_env()
    env.single_action_space = gym.spaces.Discrete(3)

    with pytest.raises(ValueError, match="table length"):
        validate_turbo_vector_env(env, "test-turbo")
