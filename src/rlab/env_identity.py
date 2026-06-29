from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from typing import Any


ENVIRONMENT_HASH_ALGORITHM = "rlab.environment.v1"

STATE_KEYS = ("state", "states", "state_probs")
ACTION_KEYS = ("action_set",)
PREPROCESSING_KEYS = (
    "frame_skip",
    "max_pool_frames",
    "sticky_action_prob",
    "observation_size",
    "hud_crop_top",
    "obs_resize_algorithm",
)
TASK_CONDITIONING_KEYS = (
    "task_conditioning",
    "task_conditioning_info_vars",
    "task_conditioning_info_values",
)
TERMINATION_KEYS = (
    "max_episode_steps",
    "completion_x_threshold",
    "no_progress_timeout_steps",
    "no_progress_min_delta",
    "info_events_json",
    "info_events",
    "done_on_events",
)
REWARD_KEYS = (
    "use_retro_reward",
    "clip_rewards",
    "reward_mode",
    "progress_reward_cap",
    "progress_reward_scale",
    "terminal_reward",
    "reward_scale",
    "time_penalty",
    "death_penalty",
    "completion_reward",
    "score_progress_clipped",
)


def _normalize_preprocessing(identity: dict[str, Any]) -> None:
    preprocessing = identity.setdefault("preprocessing", {})
    if not isinstance(preprocessing, dict):
        return
    preprocessing.setdefault("pipeline", "stable_retro_native_vec_env")
    preprocessing.setdefault("frame_skip", 4)
    preprocessing.setdefault("frame_stack", 4)
    preprocessing.setdefault("max_pool_frames", True)
    preprocessing.setdefault("sticky_action_prob", 0.0)
    preprocessing.setdefault("obs_grayscale", True)
    preprocessing.setdefault("obs_resize_algorithm", "area")
    preprocessing.setdefault("copy_observations", False)
    observation_size = preprocessing.get("observation_size", 84)
    preprocessing.setdefault("observation_size", observation_size)
    preprocessing.setdefault("obs_resize", [observation_size, observation_size])
    if "obs_crop" not in preprocessing:
        hud_crop_top = preprocessing.get("hud_crop_top")
        preprocessing["obs_crop"] = [hud_crop_top, 0, 0, 0] if hud_crop_top else None
    task_conditioning = identity.get("task_conditioning")
    if isinstance(task_conditioning, Mapping) and task_conditioning.get("task_conditioning"):
        layout = "dict_image_task"
    else:
        layout = "channel_first"
    preprocessing.setdefault("policy_observation_layout", layout)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def environment_hash(environment: Mapping[str, Any]) -> str:
    payload = f"{ENVIRONMENT_HASH_ALGORITHM}\n{canonical_json(environment)}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _copy_present(source: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: deepcopy(source[key]) for key in keys if key in source and source[key] is not None}


def _setdefault_section(
    environment: dict[str, Any],
    section: str,
    values: Mapping[str, Any],
) -> None:
    if not values:
        return
    existing = environment.get(section)
    if not isinstance(existing, dict):
        environment[section] = dict(values)
        return
    for key, value in values.items():
        existing.setdefault(key, value)


def environment_identity_from_train_config(
    train_config: Mapping[str, Any],
    *,
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical, hashable environment identity from launch config.

    The identity intentionally excludes optimizer, vectorization, scheduling, and
    logging knobs. It captures the interface and transition/reward semantics the
    policy actually acts within.
    """

    identity = deepcopy(dict(environment or {}))
    identity.setdefault("schema_version", 1)
    identity.setdefault("provider", "stable_retro")
    if "env_id" not in identity and train_config.get("game") is not None:
        identity["env_id"] = deepcopy(train_config["game"])
    if "provider_env_id" not in identity and train_config.get("game") is not None:
        identity["provider_env_id"] = deepcopy(train_config["game"])

    _setdefault_section(identity, "state", _copy_present(train_config, STATE_KEYS))
    _setdefault_section(identity, "action", _copy_present(train_config, ACTION_KEYS))
    _setdefault_section(
        identity,
        "preprocessing",
        _copy_present(train_config, PREPROCESSING_KEYS),
    )
    _setdefault_section(
        identity,
        "task_conditioning",
        _copy_present(train_config, TASK_CONDITIONING_KEYS),
    )
    _setdefault_section(
        identity,
        "termination",
        _copy_present(train_config, TERMINATION_KEYS),
    )
    _setdefault_section(identity, "reward", _copy_present(train_config, REWARD_KEYS))
    _normalize_preprocessing(identity)
    return identity


def train_config_from_environment(environment: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(environment, Mapping):
        return {}
    train_config: dict[str, Any] = {}
    env_id = environment.get("provider_env_id", environment.get("env_id"))
    if env_id is not None:
        train_config["game"] = deepcopy(env_id)
    for section in (
        "state",
        "action",
        "preprocessing",
        "task_conditioning",
        "termination",
        "reward",
    ):
        value = environment.get(section)
        if isinstance(value, Mapping):
            train_config.update(deepcopy(dict(value)))
    return train_config


def attach_environment_identity(document: Mapping[str, Any]) -> dict[str, Any]:
    materialized = deepcopy(dict(document))
    train_config = materialized.get("train_config")
    if not isinstance(train_config, Mapping):
        return materialized
    environment = environment_identity_from_train_config(
        train_config,
        environment=materialized.get("environment")
        if isinstance(materialized.get("environment"), Mapping)
        else None,
    )
    materialized["environment"] = environment
    materialized["environment_hash"] = environment_hash(environment)
    return materialized
