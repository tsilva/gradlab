from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from gradlab.turbo_api import CAPABILITY_KEYS, TURBO_API_VERSION, validate_turbo_vector_env


def _capabilities() -> dict[str, object]:
    return {
        "supported_action_modes": ("custom_discrete",),
        "supported_observation_layouts": ("chw",),
        "supported_observation_color_modes": ("grayscale",),
        "supported_resize_algorithms": ("area",),
        "supported_crop_modes": ("remove", "mask"),
        "supported_observation_copy_modes": ("copy", "safe_view", "unsafe_view"),
        "supported_transition_transports": ("numpy",),
        "supports_async_step": False,
        "supports_branching": False,
        "supports_device_api": False,
        "supports_emulator_ram": False,
        "supports_enemy_variants": False,
        "supports_fire_reset": False,
        "supports_info_frame_stack": False,
        "supports_live_snapshots": True,
        "supports_maxpool_last_two": False,
        "supports_noop_reset": True,
        "supports_per_lane_rgb": True,
        "supports_reward_clipping": True,
        "supports_snapshot_codec": False,
        "supports_state_catalog": True,
        "supports_sticky_action_prob": True,
        "supports_surface_variants": False,
    }


def _contract_env(*, capabilities: dict[str, object] | None = None):
    active = np.zeros(2, dtype=np.int32)
    active.setflags(write=False)
    declared_capabilities = _capabilities() if capabilities is None else capabilities
    assert tuple(declared_capabilities) == CAPABILITY_KEYS
    signal_spec = MappingProxyType(
        {
            "dtype": "int64",
            "shape": (),
            "available_on_reset": True,
            "available_on_step": True,
        }
    )
    return SimpleNamespace(
        metadata={
            "turbo_api_version": TURBO_API_VERSION,
            "transition_transport": "numpy",
            "autoreset_mode": "disabled",
            "render_modes": ("rgb_array",),
        },
        transport="numpy",
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
        capabilities=MappingProxyType(declared_capabilities),
        observation_ownership="safe_view",
        observation_buffer_depth=2,
        supports_live_snapshots=True,
        live_snapshots_deterministic=True,
        signal_schema=MappingProxyType({"score": signal_spec}),
        capture_snapshots=lambda: (b"one", b"two"),
        render_lane=lambda lane: np.zeros((2, 2, 3), dtype=np.uint8),
        get_images=lambda: [
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.zeros((2, 2, 3), dtype=np.uint8),
        ],
        render=lambda: np.zeros((2, 2, 3), dtype=np.uint8),
    )


def test_validates_the_declarative_v2_surface_without_resetting_or_stepping() -> None:
    contract = validate_turbo_vector_env(_contract_env(), "test-turbo")

    assert contract.api_version == TURBO_API_VERSION
    assert tuple(contract.capabilities) == CAPABILITY_KEYS
    assert contract.observation_ownership == "safe_view"
    assert contract.observation_buffer_depth == 2


@pytest.mark.parametrize(
    "provider_id",
    (
        "stable-retro-turbo",
        "supermariobrosnes-turbo",
        "breakout-turbo-env",
        "vizdoom-turbo",
        "gradoom",
    ),
)
def test_all_pinned_providers_share_the_exact_capability_contract(provider_id: str) -> None:
    contract = validate_turbo_vector_env(_contract_env(), provider_id)

    assert tuple(contract.capabilities) == CAPABILITY_KEYS


def test_rejects_capability_key_order_drift() -> None:
    capabilities = _capabilities()
    capabilities["supported_action_modes"] = capabilities.pop("supported_action_modes")
    env = _contract_env()
    env.capabilities = MappingProxyType(capabilities)

    with pytest.raises(RuntimeError, match="order drift"):
        validate_turbo_vector_env(env, "test-turbo")


def test_requires_tuple_sequence_capabilities() -> None:
    capabilities = _capabilities()
    capabilities["supported_crop_modes"] = ["remove", "mask"]
    env = _contract_env(capabilities=capabilities)

    with pytest.raises(TypeError, match="supported_crop_modes.*immutable tuple"):
        validate_turbo_vector_env(env, "test-turbo")


def test_requires_boolean_feature_capabilities() -> None:
    capabilities = _capabilities()
    capabilities["supports_enemy_variants"] = "no"
    env = _contract_env(capabilities=capabilities)

    with pytest.raises(TypeError, match="supports_enemy_variants.*boolean"):
        validate_turbo_vector_env(env, "test-turbo")


def test_rejects_mutable_contract_declarations() -> None:
    env = _contract_env()
    env.capabilities = dict(env.capabilities)

    with pytest.raises(TypeError, match="capabilities must be immutable"):
        validate_turbo_vector_env(env, "test-turbo")


def test_rejects_transition_transport_drift() -> None:
    env = _contract_env()
    env.transport = "torch"

    with pytest.raises(RuntimeError, match="transport disagrees"):
        validate_turbo_vector_env(env, "test-turbo")


def test_validates_device_resident_torch_state_indices() -> None:
    import torch

    env = _contract_env()
    env.metadata = {**env.metadata, "transition_transport": "torch"}
    env.transport = "torch"
    env.device = torch.device("cpu")
    env.active_state_indices = lambda: torch.zeros(2, dtype=torch.int32)
    capabilities = _capabilities()
    capabilities["supported_transition_transports"] = ("torch",)
    env.capabilities = MappingProxyType(capabilities)

    contract = validate_turbo_vector_env(env, "gradoom")

    assert contract.api_version == 2


def test_rejects_nonportable_signal_dtype_declarations() -> None:
    env = _contract_env()
    env.signal_schema = MappingProxyType(
        {
            "score": MappingProxyType(
                {
                    "dtype": np.dtype(np.int64),
                    "shape": (),
                    "available_on_reset": True,
                    "available_on_step": True,
                }
            )
        }
    )

    with pytest.raises(TypeError, match="dtype must be a portable string"):
        validate_turbo_vector_env(env, "test-turbo")


def test_requires_methods_for_advertised_features() -> None:
    env = _contract_env()
    del env.capture_snapshots

    with pytest.raises(RuntimeError, match="supports_live_snapshots.*capture_snapshots"):
        validate_turbo_vector_env(env, "test-turbo")


def test_rejects_action_table_cardinality_that_disagrees_with_the_space() -> None:
    env = _contract_env()
    env.single_action_space = gym.spaces.Discrete(3)

    with pytest.raises(ValueError, match="table length"):
        validate_turbo_vector_env(env, "test-turbo")
