from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np


TURBO_API_VERSION = 1
ACTION_MODES = frozenset(
    {"all", "filtered", "discrete", "multi_discrete", "custom_discrete"}
)
CAPABILITY_KEYS = frozenset(
    {
        "supported_action_modes",
        "supported_observation_layouts",
        "supported_resize_algorithms",
        "supported_observation_copy_modes",
        "supports_maxpool_last_two",
        "supports_sticky_action_prob",
        "supports_reward_clipping",
        "supports_noop_reset",
        "supports_state_catalog",
        "supports_live_snapshots",
        "supports_per_lane_rgb",
    }
)
PROVIDER_CAPABILITY_KEYS = MappingProxyType(
    {
        "vizdoom-turbo": frozenset(
            {
                "supports_enemy_variants",
                "supports_surface_variants",
            }
        ),
    }
)
OPTIONAL_PROVIDER_CAPABILITY_KEYS = MappingProxyType(
    {
        # Added within Turbo API v1. Older pinned builds remain valid when the
        # feature is not requested; construction validates it as mandatory for
        # every configured provider-owned policy history.
        "vizdoom-turbo": frozenset({"supports_info_frame_stack"}),
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
    "closed",
)
SIGNAL_SPEC_FIELDS = frozenset(
    {"dtype", "shape", "available_on_reset", "available_on_step"}
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


def validate_turbo_vector_env(env: Any, provider_id: str) -> TurboApiContract:
    """Validate the declarative, side-effect-free Turbo Vector API v1 surface."""

    metadata = _require_mapping(getattr(env, "metadata", None), "metadata")
    version = metadata.get("turbo_api_version")
    if version != TURBO_API_VERSION:
        raise RuntimeError(
            f"{provider_id} requires Turbo Vector API v{TURBO_API_VERSION}; "
            f"provider advertises {version!r}"
        )
    render_modes = tuple(str(value) for value in metadata.get("render_modes", ()))
    if "rgb_array" not in render_modes:
        raise RuntimeError(f"{provider_id} must advertise render_mode='rgb_array'")

    for name in (*COMMON_FIELDS, *ACTION_FIELDS):
        if not hasattr(env, name):
            raise RuntimeError(f"{provider_id} Turbo API v1 is missing {name!r}")
    for name in ("render_lane", "get_images", "render", "active_state_indices"):
        if not callable(getattr(env, name, None)):
            raise RuntimeError(f"{provider_id} Turbo API v1 is missing callable {name!r}")

    state_catalog = getattr(env, "state_catalog", None)
    if not isinstance(state_catalog, tuple):
        raise TypeError(f"{provider_id} state_catalog must be an immutable tuple")
    active = np.asarray(env.active_state_indices())
    expected_shape = (int(env.num_envs),)
    if active.dtype != np.dtype(np.int32) or active.shape != expected_shape:
        raise TypeError(
            f"{provider_id} active_state_indices must be read-only int32{expected_shape}"
        )
    if active.flags.writeable:
        raise TypeError(f"{provider_id} active_state_indices must be read-only")

    capabilities = _require_mapping(getattr(env, "capabilities", None), "capabilities")
    if not isinstance(capabilities, IMMUTABLE_MAPPING_TYPE):
        raise TypeError(f"{provider_id} capabilities must be immutable")
    provider_capability_keys = PROVIDER_CAPABILITY_KEYS.get(provider_id, frozenset())
    optional_capability_keys = OPTIONAL_PROVIDER_CAPABILITY_KEYS.get(
        provider_id,
        frozenset(),
    )
    expected_capability_keys = CAPABILITY_KEYS | provider_capability_keys
    missing = expected_capability_keys - set(capabilities)
    extra = set(capabilities) - expected_capability_keys - optional_capability_keys
    if missing or extra:
        raise RuntimeError(
            f"{provider_id} capabilities mismatch; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    for name in provider_capability_keys | (set(capabilities) & optional_capability_keys):
        if not isinstance(capabilities[name], bool):
            raise TypeError(f"{provider_id} capability {name!r} must be boolean")
    action_modes = tuple(str(value) for value in capabilities["supported_action_modes"])
    if (
        not action_modes
        or len(action_modes) != len(set(action_modes))
        or not set(action_modes) <= ACTION_MODES
    ):
        raise ValueError(f"{provider_id} declares invalid supported_action_modes")
    if str(env.action_mode) not in action_modes:
        raise ValueError(
            f"{provider_id} resolved action_mode {env.action_mode!r} is not declared"
        )
    buttons = getattr(env, "buttons")
    if not isinstance(buttons, tuple) or any(
        value is not None and not isinstance(value, str) for value in buttons
    ):
        raise TypeError(f"{provider_id} buttons must be an immutable tuple of labels")
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
        np.dtype(spec["dtype"])
        tuple(int(value) for value in spec["shape"])
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
