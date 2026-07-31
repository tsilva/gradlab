from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from gradlab.env_registry import environment_spec
from gradlab.json_utils import canonical_json_text
from gradlab.metric_names import metric_path_segment
from gradlab.provider_config import provider_env_id, provider_game, semantic_provider_args
from gradlab.preprocessing import preprocessing_contract
from gradlab.reward_programs import MARIO_REWARD_FIELD_SET
from gradlab.reward_transform import (
    COMMON_REWARD_KEYS,
    normalize_task_reward,
    reward_transform_from_reward,
)
from gradlab.rom_assets import manifest_from_train_config, portable_rom_asset_identity
from gradlab.task_kernels import default_task_document


ENVIRONMENT_HASH_ALGORITHM = "gradlab.environment.v4"
ENVIRONMENT_SCHEMA_VERSION = 4

STATE_KEYS = ("state", "states", "state_probs")
PREPROCESSING_KEYS = (
    "frame_skip",
    "max_pool_frames",
    "sticky_action_prob",
    "obs_resize",
    "obs_crop",
    "obs_crop_mode",
    "obs_crop_fill",
    "obs_resize_algorithm",
)
IDENTITY_REWARD_KEYS = frozenset({"reward_mode"}) | COMMON_REWARD_KEYS


def _normalize_preprocessing(identity: dict[str, Any]) -> None:
    preprocessing = identity.setdefault("preprocessing", {})
    if not isinstance(preprocessing, dict):
        return
    env_id = identity.get("env_id")
    provider_id = str(env_id).split(":", 1)[0] if isinstance(env_id, str) and ":" in env_id else ""
    task = identity.get("task")
    canonical = preprocessing_contract(
        preprocessing,
        provider_id=provider_id,
        task=task if isinstance(task, Mapping) else None,
    )
    preprocessing.clear()
    preprocessing.update(canonical)


def environment_hash(environment: Mapping[str, Any]) -> str:
    payload = (
        f"{ENVIRONMENT_HASH_ALGORITHM}\n"
        f"{canonical_json_text(environment, default=str, allow_nan=True)}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _copy_present(source: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: deepcopy(source[key]) for key in keys if key in source and source[key] is not None}


def task_config_from_train_config(
    train_config: Mapping[str, Any],
    *,
    task: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an explicit canonical task or the registered default for an environment."""

    if "provider" in train_config:
        raise ValueError("train config has unexpected key 'provider'; use 'env_provider'")
    provider_id = str(train_config.get("env_provider") or "")
    game = str(provider_game(train_config) or train_config.get("game") or "")
    inferred_id = environment_spec(provider_id, game).task_id
    canonical = default_task_document(inferred_id)

    embedded_task = train_config.get("task")
    if isinstance(embedded_task, Mapping) and embedded_task:
        canonical = deepcopy(dict(embedded_task))
    if isinstance(task, Mapping) and task:
        canonical = deepcopy(dict(task))
    validate_task_config(canonical)
    from gradlab.model_inputs import normalize_task_model_inputs

    canonical = normalize_task_model_inputs(canonical)
    return normalize_task_reward(canonical)


def validate_task_config(task: Mapping[str, Any], *, label: str = "task") -> None:
    allowed = {
        "id",
        "action",
        "signals",
        "events",
        "termination",
        "reward",
        "conditioning",
        "model_inputs",
    }
    extra = sorted(set(task) - allowed)
    if extra:
        raise ValueError(f"{label} has unexpected keys: {extra}")
    task_id = task.get("id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError(f"{label}.id must be a non-empty string")
    if task_id not in {"identity", "mario"}:
        raise ValueError(f"{label}.id has no registered task kernel: {task_id!r}")
    metric_path_segment(task_id)
    for section in ("action", "signals", "events", "termination", "reward"):
        if not isinstance(task.get(section), Mapping):
            raise ValueError(f"{label}.{section} must be an object")
    action = task["action"]
    action_set = action.get("set")
    if not isinstance(action_set, str) or not action_set.strip():
        raise ValueError(f"{label}.action.set must be a non-empty string")
    if task_id == "mario" and action_set != "native":
        raise ValueError(f"{label}.action.set must be 'native'")
    extra_action_keys = sorted(set(action) - {"set", "codec"})
    if extra_action_keys:
        raise ValueError(f"{label}.action has unexpected keys: {extra_action_keys}")
    codec = action.get("codec")
    if codec is not None:
        if task_id != "identity":
            raise ValueError(f"{label}.action.codec is only supported by the identity task")
        if not isinstance(codec, Mapping):
            raise ValueError(f"{label}.action.codec must be an object")
        extra_codec_keys = sorted(set(codec) - {"type", "values"})
        if extra_codec_keys:
            raise ValueError(f"{label}.action.codec has unexpected keys: {extra_codec_keys}")
        if codec.get("type") != "discrete_lookup":
            raise ValueError(f"{label}.action.codec.type must be 'discrete_lookup'")
        values = codec.get("values")
        if not isinstance(values, list | tuple) or not values:
            raise ValueError(f"{label}.action.codec.values must be a non-empty list")
    signals = task["signals"]
    for name, source in signals.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{label}.signals keys must be non-empty strings")
        metric_path_segment(name)
        if isinstance(source, str) and source.strip():
            continue
        if (
            isinstance(source, list | tuple)
            and source
            and all(isinstance(item, str) and item.strip() for item in source)
        ):
            continue
        raise ValueError(f"{label}.signals.{name} must be a signal name or non-empty list")
    events = task["events"]
    if len(events) > 64:
        raise ValueError(f"{label}.events supports at most 64 events")
    for name, raw_rule in events.items():
        metric_path_segment(name)
        if not isinstance(raw_rule, Mapping):
            raise ValueError(f"{label}.events.{name} must be an object")
        signal = raw_rule.get("signal")
        if signal not in signals:
            raise ValueError(f"{label}.events.{name}.signal references unknown signal {signal!r}")
        operation = raw_rule.get("operation")
        if operation not in {
            "change",
            "decrease",
            "increase",
            "unchanged_for",
            "equals_for",
            "equals",
        }:
            raise ValueError(f"{label}.events.{name}.operation is unsupported: {operation!r}")
        if operation in {"unchanged_for", "equals_for"}:
            steps = raw_rule.get("steps")
            if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
                raise ValueError(f"{label}.events.{name}.steps must be a positive integer")
        if operation in {"equals", "equals_for"}:
            value = raw_rule.get("value")
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise ValueError(f"{label}.events.{name}.value must be a number")
    if task_id == "identity":
        supported_operations = {"decrease", "increase", "equals_for"}
        unsupported_events = sorted(
            name
            for name, rule in events.items()
            if rule.get("operation") not in supported_operations
        )
        if unsupported_events:
            raise ValueError(
                f"{label} identity events support only operations "
                "'decrease', 'increase', and 'equals_for': " + ", ".join(unsupported_events)
            )
    if task_id == "mario":
        expected_events = {
            "life_loss": ("lives", "decrease"),
            "level_change": ("level", "change"),
            "game_complete": ("game_mode", "equals"),
            "stalled": ("x", "unchanged_for"),
        }
        unknown_events = sorted(set(events) - set(expected_events))
        if unknown_events:
            raise ValueError(f"{label} has unsupported Mario events: {', '.join(unknown_events)}")
        for name, rule in events.items():
            expected_signal, expected_operation = expected_events[name]
            if (rule.get("signal"), rule.get("operation")) != (
                expected_signal,
                expected_operation,
            ):
                raise ValueError(
                    f"{label}.events.{name} requires signal={expected_signal!r} "
                    f"and operation={expected_operation!r}"
                )
        game_complete = events.get("game_complete")
        if isinstance(game_complete, Mapping):
            when = game_complete.get("when")
            if not isinstance(when, Mapping) or when.get("signal") != "level":
                raise ValueError(f"{label}.events.game_complete.when.signal must be 'level'")
            value = when.get("value")
            if (
                not isinstance(value, list | tuple)
                or len(value) != 2
                or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
            ):
                raise ValueError(
                    f"{label}.events.game_complete.when.value must be a pair of integers"
                )
    termination = task["termination"]
    event_outcomes: dict[str, str] = {}
    for outcome in ("success", "failure", "timeout", "neutral"):
        if outcome not in termination:
            continue
        names = termination[outcome]
        if not isinstance(names, list | tuple):
            raise ValueError(f"{label}.termination.{outcome} must be a list")
        missing = sorted({str(name) for name in names} - set(events))
        if missing:
            raise ValueError(
                f"{label}.termination.{outcome} references unknown events: {', '.join(missing)}"
            )
        for name in names:
            event_name = str(name)
            previous = event_outcomes.get(event_name)
            if previous is not None:
                raise ValueError(
                    f"{label}.events.{event_name} cannot map to both {previous} and {outcome}"
                )
            event_outcomes[event_name] = outcome
    for key in ("max_episode_steps", "no_progress_min_delta"):
        if key not in termination:
            continue
        value = termination[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{label}.termination.{key} must be a non-negative integer")
    reward = task["reward"]
    allowed_reward_keys = (
        IDENTITY_REWARD_KEYS if task_id == "identity" else MARIO_REWARD_FIELD_SET
    )
    extra_reward_keys = sorted(set(reward) - allowed_reward_keys)
    if extra_reward_keys:
        raise ValueError(f"{label}.reward has unexpected keys: {extra_reward_keys}")
    reward_transform_from_reward(reward, label=f"{label}.reward")
    reward_mode = reward.get("reward_mode")
    if task_id == "identity" and reward_mode not in {None, "native"}:
        raise ValueError(f"{label}.reward.reward_mode must be 'native' for the identity task")
    if task_id == "mario" and reward_mode not in {
        None,
        "native",
        "bounded",
        "baseline",
        "score",
        "additive",
    }:
        raise ValueError(f"{label}.reward.reward_mode is unsupported: {reward_mode!r}")
    conditioning = task.get("conditioning")
    if conditioning is not None:
        if not isinstance(conditioning, Mapping):
            raise ValueError(f"{label}.conditioning must be an object")
        if conditioning.get("enabled"):
            signal = conditioning.get("signal")
            if signal not in signals:
                raise ValueError(
                    f"{label}.conditioning.signal references unknown signal {signal!r}"
                )
    model_inputs = task.get("model_inputs")
    if model_inputs is not None:
        from gradlab.model_inputs import RUNTIME_CONTEXT_SIGNALS, normalize_model_inputs

        normalized_inputs = normalize_model_inputs(
            model_inputs,
            label=f"{label}.model_inputs",
        )
        if conditioning is not None and conditioning.get("enabled"):
            raise ValueError(
                f"{label} cannot combine conditioning with model_inputs"
            )
        for name, field in normalized_inputs["context"].items():
            signal = field["signal"]
            if signal in RUNTIME_CONTEXT_SIGNALS:
                if signal in signals:
                    raise ValueError(
                        f"{label}.signals.{signal} is reserved for model-input runtime context"
                    )
                continue
            if signal not in signals:
                raise ValueError(
                    f"{label}.model_inputs.context.{name}.signal references "
                    f"unknown signal {signal!r}"
                )


def environment_identity_from_train_config(
    train_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a canonical, hashable environment identity from launch config.

    The identity intentionally excludes optimizer, vectorization, scheduling, and
    logging knobs. It captures the interface and transition/reward semantics the
    policy actually acts within.
    """

    normalized_train_config = deepcopy(dict(train_config))

    identity: dict[str, Any] = {"schema_version": ENVIRONMENT_SCHEMA_VERSION}
    resolved_env_id = provider_env_id(normalized_train_config)
    if resolved_env_id is not None:
        identity["env_id"] = resolved_env_id
    identity.update(_copy_present(normalized_train_config, STATE_KEYS))
    identity["preprocessing"] = _copy_present(normalized_train_config, PREPROCESSING_KEYS)
    identity["task"] = task_config_from_train_config(
        normalized_train_config,
    )
    provider_args = semantic_provider_args(normalized_train_config)
    if provider_args:
        identity.setdefault("provider_args", deepcopy(provider_args))
    rom_asset = manifest_from_train_config(
        normalized_train_config,
        expected_game=provider_game(normalized_train_config),
    )
    if rom_asset is not None:
        identity["rom_asset"] = portable_rom_asset_identity(rom_asset)
    _normalize_preprocessing(identity)
    return identity


def policy_environment_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return normalized policy-facing environment semantics."""

    identity = environment_identity_from_train_config(config)
    if identity.get("state") == "":
        identity.pop("state", None)
    for key in ("states", "state_probs"):
        value = identity.get(key)
        if isinstance(value, Sequence) and not isinstance(value, str | bytes) and not value:
            identity.pop(key, None)
    state_probs = identity.get("state_probs")
    if isinstance(state_probs, Sequence) and not isinstance(state_probs, str | bytes):
        identity["state_probs"] = [float(value) for value in state_probs]
    return identity


def policy_environment_hash(config: Mapping[str, Any]) -> str:
    """Hash only the normalized policy-facing environment semantics."""

    return environment_hash(policy_environment_identity(config))


def train_config_from_source_environment(
    environment: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(environment, Mapping):
        return {}
    unexpected = sorted(set(environment) - {"env_provider", "env_config", "preprocessing", "task"})
    if unexpected:
        raise ValueError("source environment has unexpected field(s): " + ", ".join(unexpected))
    env_config = environment.get("env_config")
    if not isinstance(env_config, Mapping):
        raise ValueError("source environment.env_config must be an object")
    train_config = deepcopy(dict(env_config))
    env_args = train_config.get("env_args")
    if isinstance(env_args, Mapping):
        aliases = sorted(set(env_args) & {"game", "num_envs"})
        if aliases:
            raise ValueError(
                "source environment.env_config.env_args uses canonical field(s) "
                f"{aliases}; put game and n_envs directly in env_config"
            )
    provider = environment.get("env_provider")
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("source environment.env_provider must be a non-empty string")
    train_config["env_provider"] = provider.strip()
    preprocessing = environment.get("preprocessing")
    if preprocessing is not None:
        if not isinstance(preprocessing, Mapping):
            raise ValueError("source environment.preprocessing must be an object")
        train_config.update(deepcopy(dict(preprocessing)))
    game = provider_game(train_config)
    if game is not None:
        train_config.setdefault("game", game)
    task = environment.get("task")
    train_config["task"] = task_config_from_train_config(
        train_config,
        task=task if isinstance(task, Mapping) else None,
    )
    return train_config


def attach_environment_identity(document: Mapping[str, Any]) -> dict[str, Any]:
    materialized = deepcopy(dict(document))
    train_config = materialized.get("train_config")
    if not isinstance(train_config, Mapping):
        return materialized
    environment = environment_identity_from_train_config(train_config)
    materialized["environment"] = environment
    materialized["environment_hash"] = environment_hash(environment)
    materialized["policy_environment_hash"] = policy_environment_hash(train_config)
    evaluation = train_config.get("checkpoint_eval_environment")
    if isinstance(evaluation, Mapping):
        evaluation_hash = policy_environment_hash(evaluation)
        materialized["evaluation_environment_hash"] = evaluation_hash
        if evaluation_hash != materialized["policy_environment_hash"]:
            raise ValueError(
                "training and evaluation policy environment semantics disagree: "
                f"training={materialized['policy_environment_hash']} "
                f"evaluation={evaluation_hash}"
            )
    return materialized
