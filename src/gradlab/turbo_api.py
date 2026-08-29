"""Strict, side-effect-free validation for the Turbo Vector API v2 surface."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np


TURBO_API_VERSION = 2
ACTION_MODES = frozenset({"all", "filtered", "discrete", "multi_discrete", "custom_discrete"})
CAPABILITY_KEYS = (
    "supported_action_modes",
    "supported_observation_layouts",
    "supported_observation_color_modes",
    "supported_resize_algorithms",
    "supported_crop_modes",
    "supported_observation_copy_modes",
    "supported_transition_transports",
    "supports_async_step",
    "supports_branching",
    "supports_device_api",
    "supports_emulator_ram",
    "supports_enemy_variants",
    "supports_fire_reset",
    "supports_info_frame_stack",
    "supports_live_snapshots",
    "supports_maxpool_last_two",
    "supports_noop_reset",
    "supports_per_lane_rgb",
    "supports_reward_clipping",
    "supports_snapshot_codec",
    "supports_state_catalog",
    "supports_sticky_action_prob",
    "supports_surface_variants",
)
FILTERED_ACTION_CAPABILITY = "supported_filtered_actions"
FILTERED_ACTION_CAPABILITY_KEYS = (
    CAPABILITY_KEYS[0],
    FILTERED_ACTION_CAPABILITY,
    *CAPABILITY_KEYS[1:],
)
SEQUENCE_CAPABILITIES = CAPABILITY_KEYS[:7]
FEATURE_METHODS = MappingProxyType(
    {
        "supports_async_step": ("step_async", "step_wait"),
        "supports_branching": ("branch",),
        "supports_device_api": ("step_device", "reset_device"),
        "supports_emulator_ram": ("ram",),
        "supports_live_snapshots": ("capture_snapshots",),
        "supports_per_lane_rgb": ("render_lane", "get_images"),
        "supports_snapshot_codec": ("encode_snapshots", "decode_snapshots"),
        "supports_state_catalog": ("active_state_indices",),
    }
)
ACTION_FIELDS = (
    "buttons",
    "action_mode",
    "action_preset",
    "action_table",
    "action_meanings",
    "action_table_hash",
)
COMMON_FIELDS = (
    "num_envs",
    "num_threads",
    "frame_skip",
    "frame_stack",
    "obs_layout",
    "obs_copy",
    "render_mode",
    "transport",
    "closed",
)
SIGNAL_SPEC_FIELDS = frozenset({"dtype", "shape", "available_on_reset", "available_on_step"})
PORTABLE_DTYPES = frozenset(
    {
        "bool",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float16",
        "float32",
        "float64",
    }
)
IMMUTABLE_MAPPING_TYPE = type(MappingProxyType({}))


@dataclass(frozen=True)
class TurboApiContract:
    api_version: int
    capabilities: Mapping[str, Any]
    observation_ownership: str
    observation_buffer_depth: int | None


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _is_disabled_autoreset(value: Any) -> bool:
    name = getattr(value, "name", None)
    if isinstance(name, str) and name.casefold() == "disabled":
        return True
    raw_value = getattr(value, "value", value)
    return "disabled" in str(raw_value).casefold()


def _validate_active_state_indices(env: Any, provider_id: str, transport: str) -> None:
    active = env.active_state_indices()
    expected_shape = (int(env.num_envs),)
    if tuple(getattr(active, "shape", ())) != expected_shape:
        raise TypeError(
            f"{provider_id} active_state_indices must have shape {expected_shape}"
        )
    if transport == "numpy":
        if not isinstance(active, np.ndarray) or active.dtype != np.dtype(np.int32):
            raise TypeError(
                f"{provider_id} active_state_indices must be a read-only NumPy int32 array"
            )
        if active.flags.writeable:
            raise TypeError(f"{provider_id} active_state_indices must be read-only")
        return
    if str(getattr(active, "dtype", "")) != "torch.int32":
        raise TypeError(f"{provider_id} active_state_indices must be a torch.int32 tensor")
    if getattr(active, "device", None) != getattr(env, "device", None):
        raise TypeError(f"{provider_id} active_state_indices must remain on env.device")


def validate_turbo_vector_env(env: Any, provider_id: str) -> TurboApiContract:
    """Validate the declarative Turbo Vector API v2 surface without transitions."""

    metadata = _require_mapping(getattr(env, "metadata", None), "metadata")
    version = metadata.get("turbo_api_version")
    if version != TURBO_API_VERSION:
        raise RuntimeError(
            f"{provider_id} requires Turbo Vector API v{TURBO_API_VERSION}; "
            f"provider advertises {version!r}"
        )
    transport = metadata.get("transition_transport")
    if transport not in {"numpy", "torch"}:
        raise RuntimeError(
            f"{provider_id} must advertise transition_transport='numpy' or 'torch'"
        )
    if getattr(env, "transport", None) != transport:
        raise RuntimeError(
            f"{provider_id} transition transport disagrees between metadata and env.transport"
        )
    if not _is_disabled_autoreset(metadata.get("autoreset_mode")):
        raise RuntimeError(f"{provider_id} must advertise autoreset_mode=DISABLED")
    render_modes = tuple(str(value) for value in metadata.get("render_modes", ()))
    if "rgb_array" not in render_modes:
        raise RuntimeError(f"{provider_id} must advertise render_mode='rgb_array'")

    for name in (*COMMON_FIELDS, *ACTION_FIELDS):
        if not hasattr(env, name):
            raise RuntimeError(f"{provider_id} Turbo API v2 is missing {name!r}")
    for name in ("render_lane", "get_images", "render", "active_state_indices"):
        if not callable(getattr(env, name, None)):
            raise RuntimeError(f"{provider_id} Turbo API v2 is missing callable {name!r}")

    state_catalog = getattr(env, "state_catalog", None)
    if not isinstance(state_catalog, tuple):
        raise TypeError(f"{provider_id} state_catalog must be an immutable tuple")
    _validate_active_state_indices(env, provider_id, transport)

    capabilities = _require_mapping(getattr(env, "capabilities", None), "capabilities")
    if not isinstance(capabilities, IMMUTABLE_MAPPING_TYPE):
        raise TypeError(f"{provider_id} capabilities must be immutable")
    capability_keys = tuple(capabilities)
    if capability_keys not in {CAPABILITY_KEYS, FILTERED_ACTION_CAPABILITY_KEYS}:
        missing = set(CAPABILITY_KEYS) - set(capabilities)
        extra = set(capabilities) - set(FILTERED_ACTION_CAPABILITY_KEYS)
        raise RuntimeError(
            f"{provider_id} capabilities mismatch or order drift; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    for name in SEQUENCE_CAPABILITIES:
        if not isinstance(capabilities[name], tuple):
            raise TypeError(f"{provider_id} capability {name!r} must be an immutable tuple")
    for name in CAPABILITY_KEYS[len(SEQUENCE_CAPABILITIES) :]:
        if not isinstance(capabilities[name], bool):
            raise TypeError(f"{provider_id} capability {name!r} must be boolean")
    for feature, methods in FEATURE_METHODS.items():
        if capabilities[feature] and not all(
            callable(getattr(env, method, None)) for method in methods
        ):
            raise RuntimeError(
                f"{provider_id} capability {feature!r} requires callable methods {methods}"
            )
    if transport not in capabilities["supported_transition_transports"]:
        raise ValueError(f"{provider_id} resolved transport {transport!r} is not declared")
    if capabilities["supports_state_catalog"]:
        if not state_catalog or len(set(state_catalog)) != len(state_catalog):
            raise ValueError(f"{provider_id} state_catalog must be non-empty and unique")

    action_modes = tuple(str(value) for value in capabilities["supported_action_modes"])
    if (
        not action_modes
        or len(action_modes) != len(set(action_modes))
        or not set(action_modes) <= ACTION_MODES
    ):
        raise ValueError(f"{provider_id} declares invalid supported_action_modes")
    if str(env.action_mode) not in action_modes:
        raise ValueError(f"{provider_id} resolved action_mode {env.action_mode!r} is not declared")
    buttons = getattr(env, "buttons")
    if not isinstance(buttons, tuple) or any(
        value is not None and not isinstance(value, str) for value in buttons
    ):
        raise TypeError(f"{provider_id} buttons must be an immutable tuple of labels")
    if FILTERED_ACTION_CAPABILITY in capabilities:
        filtered_actions = capabilities[FILTERED_ACTION_CAPABILITY]
        if not isinstance(filtered_actions, tuple) or any(
            not isinstance(row, tuple) for row in filtered_actions
        ):
            raise TypeError(
                f"{provider_id} capability {FILTERED_ACTION_CAPABILITY!r} must be an "
                "immutable tuple of tuples"
            )
        if not filtered_actions or len(set(filtered_actions)) != len(filtered_actions):
            raise ValueError(
                f"{provider_id} capability {FILTERED_ACTION_CAPABILITY!r} must be "
                "non-empty and unique"
            )
        if any(len(row) != len(buttons) for row in filtered_actions):
            raise ValueError(
                f"{provider_id} capability {FILTERED_ACTION_CAPABILITY!r} rows must "
                "match the button transport width"
            )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value not in {0, 1}
            for row in filtered_actions
            for value in row
        ):
            raise TypeError(
                f"{provider_id} capability {FILTERED_ACTION_CAPABILITY!r} rows must "
                "contain integer zeros and ones"
            )
    table = getattr(env, "action_table")
    meanings = getattr(env, "action_meanings")
    table_hash = getattr(env, "action_table_hash")
    if table is not None and not isinstance(table, tuple):
        raise TypeError(f"{provider_id} action_table must be an immutable tuple or None")
    if meanings is not None and (
        not isinstance(meanings, tuple)
        or any(not isinstance(value, str) or not value for value in meanings)
    ):
        raise TypeError(
            f"{provider_id} action_meanings must be an immutable tuple of labels or None"
        )
    if meanings is not None and table is not None and len(meanings) != len(table):
        raise ValueError(f"{provider_id} action meanings and table lengths differ")
    if str(env.action_mode) == "custom_discrete":
        if not table or not meanings or not isinstance(table_hash, str) or not table_hash:
            raise ValueError(
                f"{provider_id} custom_discrete actions require table, meanings, and hash"
            )
        single_action_space = getattr(env, "single_action_space", None)
        expected_count = getattr(single_action_space, "n", None)
        if expected_count is not None and len(table) != int(expected_count):
            raise ValueError(
                f"{provider_id} custom_discrete table length does not match its action space"
            )
    elif table_hash is not None and not isinstance(table_hash, str):
        raise TypeError(f"{provider_id} action_table_hash must be a string or None")
    if bool(capabilities["supports_live_snapshots"]) != bool(
        getattr(env, "supports_live_snapshots", False)
    ):
        raise ValueError(f"{provider_id} snapshot capability declaration disagrees")
    if not isinstance(getattr(env, "live_snapshots_deterministic", None), bool):
        raise TypeError(f"{provider_id} must declare live_snapshots_deterministic")

    ownership = str(getattr(env, "observation_ownership", ""))
    depth = getattr(env, "observation_buffer_depth", object())
    expected_depth = {"owned": None, "safe_view": 2, "unsafe_view": 1}
    if ownership not in expected_depth or depth != expected_depth[ownership]:
        raise ValueError(
            f"{provider_id} observation ownership/depth must be one of "
            "owned/None, safe_view/2, or unsafe_view/1"
        )

    signal_schema = _require_mapping(getattr(env, "signal_schema", None), "signal_schema")
    if not isinstance(signal_schema, IMMUTABLE_MAPPING_TYPE):
        raise TypeError(f"{provider_id} signal_schema must be immutable")
    for name, raw_spec in signal_schema.items():
        if not isinstance(name, str):
            raise TypeError(f"{provider_id} signal names must be strings")
        spec = _require_mapping(raw_spec, f"signal_schema[{name!r}]")
        if not isinstance(spec, IMMUTABLE_MAPPING_TYPE):
            raise TypeError(f"{provider_id} signal {name!r} spec must be immutable")
        missing_fields = SIGNAL_SPEC_FIELDS - set(spec)
        extra_fields = set(spec) - SIGNAL_SPEC_FIELDS
        if missing_fields or extra_fields:
            raise RuntimeError(
                f"{provider_id} signal {name!r} schema mismatch; "
                f"missing={sorted(missing_fields)}, extra={sorted(extra_fields)}"
            )
        if not isinstance(spec["dtype"], str) or spec["dtype"] not in PORTABLE_DTYPES:
            raise TypeError(f"{provider_id} signal {name!r} dtype must be a portable string")
        shape = spec["shape"]
        if not isinstance(shape, tuple) or any(
            not isinstance(value, int) or value < 0 for value in shape
        ):
            raise TypeError(f"{provider_id} signal {name!r} shape must be an integer tuple")
        if not isinstance(spec["available_on_reset"], bool) or not isinstance(
            spec["available_on_step"], bool
        ):
            raise TypeError(f"{provider_id} signal availability must be boolean")

    return TurboApiContract(
        api_version=TURBO_API_VERSION,
        capabilities=MappingProxyType(dict(capabilities)),
        observation_ownership=ownership,
        observation_buffer_depth=depth,
    )
