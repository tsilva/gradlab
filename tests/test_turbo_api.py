from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from gradlab.turbo_api import CAPABILITY_KEYS, TURBO_API_VERSION, validate_turbo_vector_env


def _contract_env(*, capability_extensions=None):
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
    if capability_extensions is None:
        assert set(capabilities) == CAPABILITY_KEYS
    else:
        capabilities.update(capability_extensions)
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


@pytest.mark.parametrize(
    "provider_id",
    (
        "stable-retro-turbo",
        "supermariobrosnes-turbo",
        "breakout-turbo-env",
    ),
)
def test_pinned_common_providers_keep_the_exact_common_capability_contract(
    provider_id: str,
) -> None:
    contract = validate_turbo_vector_env(_contract_env(), provider_id)

    assert set(contract.capabilities) == CAPABILITY_KEYS


def test_accepts_the_strict_vizdoom_capability_extensions() -> None:
    env = _contract_env(
        capability_extensions={
            "supports_enemy_variants": False,
            "supports_surface_variants": False,
        }
    )

    contract = validate_turbo_vector_env(env, "vizdoom-turbo")

    assert contract.capabilities["supports_enemy_variants"] is False
    assert contract.capabilities["supports_surface_variants"] is False


def test_accepts_optional_vizdoom_info_frame_stack_capability() -> None:
    env = _contract_env(
        capability_extensions={
            "supports_enemy_variants": False,
            "supports_surface_variants": False,
            "supports_info_frame_stack": True,
        }
    )

    contract = validate_turbo_vector_env(env, "vizdoom-turbo")

    assert contract.capabilities["supports_info_frame_stack"] is True


def test_requires_boolean_optional_vizdoom_info_frame_stack_capability() -> None:
    env = _contract_env(
        capability_extensions={
            "supports_enemy_variants": False,
            "supports_surface_variants": False,
            "supports_info_frame_stack": "yes",
        }
    )

    with pytest.raises(TypeError, match="supports_info_frame_stack.*boolean"):
        validate_turbo_vector_env(env, "vizdoom-turbo")


def test_rejects_vizdoom_capability_extensions_from_other_providers() -> None:
    env = _contract_env(
        capability_extensions={
            "supports_enemy_variants": False,
            "supports_surface_variants": False,
        }
    )

    with pytest.raises(RuntimeError, match="extra=.*supports_enemy_variants"):
        validate_turbo_vector_env(env, "test-turbo")


def test_requires_the_complete_vizdoom_capability_extension_contract() -> None:
    env = _contract_env(
        capability_extensions={
            "supports_enemy_variants": False,
        }
    )

    with pytest.raises(RuntimeError, match="missing=.*supports_surface_variants"):
        validate_turbo_vector_env(env, "vizdoom-turbo")


def test_requires_boolean_vizdoom_capability_extensions() -> None:
    env = _contract_env(
        capability_extensions={
            "supports_enemy_variants": "no",
            "supports_surface_variants": False,
        }
    )

    with pytest.raises(TypeError, match="supports_enemy_variants.*boolean"):
        validate_turbo_vector_env(env, "vizdoom-turbo")


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
