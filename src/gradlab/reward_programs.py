from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

from gradlab.json_utils import canonical_json_sha256
from gradlab.reward_transform import normalize_reward_mapping


REWARD_PROGRAM_KIND_MARIO_V1 = "mario-v1"
MARIO_REWARD_KERNEL_REVISION = "mario-kernel-v3"
REWARD_PROGRAM_KIND_VIZDOOM_DEATHMATCH_V1 = "vizdoom-deathmatch-v1"
VIZDOOM_DEATHMATCH_REWARD_KERNEL_REVISION = "vizdoom-deathmatch-kernel-v1"
REWARD_SHAPE_KEY_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")

MARIO_REWARD_FIELDS = (
    "reward_mode",
    "use_native_reward",
    "reward_clip",
    "progress_reward_cap",
    "progress_reward_scale",
    "progress_reward_boost_start_x",
    "progress_reward_boost_scale",
    "terminal_reward",
    "reward_scale",
    "time_penalty",
    "death_penalty",
    "completion_reward",
    "score_progress_clipped",
)
MARIO_REWARD_FIELD_SET = frozenset(MARIO_REWARD_FIELDS)
MARIO_REWARD_MODES = frozenset({"native", "bounded", "baseline", "score", "additive"})
MARIO_BOOL_FIELDS = frozenset({"use_native_reward", "score_progress_clipped"})
MARIO_NUMBER_FIELDS = frozenset(
    MARIO_REWARD_FIELD_SET - MARIO_BOOL_FIELDS - {"reward_mode", "reward_clip"}
)

# Mario signal arithmetic is currently int64-backed and reward outputs are float32.
# Reserve headroom for several simultaneously active components before the output cast.
_FLOAT32_MAX = 3.4028234663852886e38
_MARIO_SAFE_COEFFICIENT_ABS_MAX = _FLOAT32_MAX / (8.0 * float(2**32))

VIZDOOM_DEATHMATCH_BASE_REWARD_FIELDS = (
    "reward_mode",
    "reward_scale",
    "reward_clip",
)
VIZDOOM_DEATHMATCH_SHAPED_REWARD_FIELDS = (
    "kill_reward",
    "kill_loss_penalty",
    "death_penalty",
    "death_count_decrease_reward",
    "hit_reward",
    "hit_count_decrease_penalty",
    "damage_reward",
    "damage_count_decrease_penalty",
    "health_gain_reward",
    "health_loss_penalty",
    "armor_gain_reward",
    "armor_loss_penalty",
    "weapon_preferences",
    "weapon_gain_reward_scale",
    "weapon_loss_penalty_scale",
    "ammo_gain_reward_scale",
    "ammo_loss_penalty_scale",
    "selected_weapon_hold_reward_scale",
    "selected_weapon_hold_steps",
    "hit_delta_cap",
    "damage_delta_cap",
)
VIZDOOM_DEATHMATCH_REWARD_FIELDS = (
    *VIZDOOM_DEATHMATCH_BASE_REWARD_FIELDS,
    *VIZDOOM_DEATHMATCH_SHAPED_REWARD_FIELDS,
)
VIZDOOM_DEATHMATCH_REWARD_FIELD_SET = frozenset(VIZDOOM_DEATHMATCH_REWARD_FIELDS)
VIZDOOM_DEATHMATCH_REWARD_MODES = frozenset({"native", "sample-factory-v0"})
VIZDOOM_DEATHMATCH_REQUIRED_SIGNALS = {
    "deaths": "deathcount",
    "hits": "hitcount",
    "damage": "damagecount",
}
_VIZDOOM_DEATHMATCH_SAFE_COEFFICIENT_MAX = _FLOAT32_MAX / (32.0 * float(2**32))
_VIZDOOM_DEATHMATCH_COUNTER_DELTA_MAX = 2**32


@dataclass(frozen=True)
class RewardShapeSelection:
    goal: dict[str, Any]
    key: str
    program_kind: str
    program_revision: str
    semantic_sha256: str
    is_default: bool
    reward: dict[str, Any]


def _sha256(value: Mapping[str, Any]) -> str:
    return f"sha256:{canonical_json_sha256(value)}"


def _normalize_zero(value: float) -> float:
    return 0.0 if value == 0.0 else value


def normalize_mario_reward(
    value: Mapping[str, Any],
    *,
    label: str,
    require_complete: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    unknown = sorted(str(key) for key in value if key not in MARIO_REWARD_FIELD_SET)
    if unknown:
        raise ValueError(f"{label} has unexpected field(s): {', '.join(unknown)}")
    if require_complete:
        missing = [key for key in MARIO_REWARD_FIELDS if key not in value]
        if missing:
            raise ValueError(f"{label} is missing required field(s): {', '.join(missing)}")

    normalized: dict[str, Any] = {}
    if "reward_mode" in value:
        mode = value["reward_mode"]
        if not isinstance(mode, str) or mode not in MARIO_REWARD_MODES:
            raise ValueError(
                f"{label}.reward_mode must be one of {sorted(MARIO_REWARD_MODES)}, got {mode!r}"
            )
        normalized["reward_mode"] = mode
    for key in MARIO_BOOL_FIELDS:
        if key not in value:
            continue
        item = value[key]
        if type(item) is not bool:
            raise ValueError(f"{label}.{key} must be a boolean")
        normalized[key] = item
    for key in MARIO_NUMBER_FIELDS:
        if key not in value:
            continue
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, Real):
            raise ValueError(f"{label}.{key} must be a finite number")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{label}.{key} must be finite")
        if abs(number) > _MARIO_SAFE_COEFFICIENT_ABS_MAX:
            raise ValueError(f"{label}.{key} exceeds the Mario float32 reward safety bound")
        normalized[key] = _normalize_zero(number)
    if "reward_clip" in value:
        normalized["reward_clip"] = normalize_reward_mapping(
            {
                "reward_scale": normalized.get(
                    "reward_scale",
                    value.get("reward_scale", 1.0),
                ),
                "reward_clip": value["reward_clip"],
            },
            label=label,
        )["reward_clip"]
    return {key: normalized[key] for key in MARIO_REWARD_FIELDS if key in normalized}


def normalize_vizdoom_deathmatch_reward(
    value: Mapping[str, Any],
    *,
    label: str,
    require_complete: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    unknown = sorted(str(key) for key in value if key not in VIZDOOM_DEATHMATCH_REWARD_FIELD_SET)
    if unknown:
        raise ValueError(f"{label} has unexpected field(s): {', '.join(unknown)}")

    mode = value.get("reward_mode")
    if not isinstance(mode, str) or mode not in VIZDOOM_DEATHMATCH_REWARD_MODES:
        raise ValueError(
            f"{label}.reward_mode must be one of "
            f"{sorted(VIZDOOM_DEATHMATCH_REWARD_MODES)}, got {mode!r}"
        )
    expected = set(VIZDOOM_DEATHMATCH_BASE_REWARD_FIELDS)
    if mode == "sample-factory-v0":
        expected.update(VIZDOOM_DEATHMATCH_SHAPED_REWARD_FIELDS)
    unexpected_for_mode = sorted(str(key) for key in value if key not in expected)
    if unexpected_for_mode:
        raise ValueError(
            f"{label} reward_mode={mode!r} does not use field(s): " + ", ".join(unexpected_for_mode)
        )
    if require_complete:
        missing = [
            key for key in VIZDOOM_DEATHMATCH_REWARD_FIELDS if key in expected and key not in value
        ]
        if missing:
            raise ValueError(f"{label} is missing required field(s): {', '.join(missing)}")

    transform = normalize_reward_mapping(
        {
            "reward_scale": value.get("reward_scale", 1.0),
            "reward_clip": value.get("reward_clip", False),
        },
        label=label,
    )
    normalized: dict[str, Any] = {
        "reward_mode": mode,
        "reward_scale": transform["reward_scale"],
        "reward_clip": transform["reward_clip"],
    }
    if mode == "native":
        return normalized

    preferences = value.get("weapon_preferences")
    if (
        isinstance(preferences, str | bytes)
        or not isinstance(preferences, list | tuple)
        or len(preferences) != 6
    ):
        raise ValueError(f"{label}.weapon_preferences must contain exactly six numbers")
    normalized_preferences: list[float] = []
    for index, item in enumerate(preferences):
        if isinstance(item, bool) or not isinstance(item, Real):
            raise ValueError(f"{label}.weapon_preferences[{index}] must be a finite number")
        number = float(item)
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"{label}.weapon_preferences[{index}] must be finite and positive")
        if number > _VIZDOOM_DEATHMATCH_SAFE_COEFFICIENT_MAX:
            raise ValueError(
                f"{label}.weapon_preferences[{index}] exceeds the Deathmatch float32 safety bound"
            )
        normalized_preferences.append(_normalize_zero(number))
    normalized["weapon_preferences"] = normalized_preferences

    integer_fields = ("selected_weapon_hold_steps", "hit_delta_cap", "damage_delta_cap")
    for key in integer_fields:
        item = value.get(key)
        if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
            raise ValueError(f"{label}.{key} must be a positive integer")
        if key != "selected_weapon_hold_steps" and item > _VIZDOOM_DEATHMATCH_COUNTER_DELTA_MAX:
            raise ValueError(
                f"{label}.{key} exceeds the Deathmatch counter-delta safety bound"
            )
        normalized[key] = int(item)

    number_fields = set(VIZDOOM_DEATHMATCH_SHAPED_REWARD_FIELDS) - {
        "weapon_preferences",
        *integer_fields,
    }
    for key in number_fields:
        item = value.get(key)
        if isinstance(item, bool) or not isinstance(item, Real):
            raise ValueError(f"{label}.{key} must be a finite non-negative number")
        number = float(item)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"{label}.{key} must be a finite non-negative number")
        if number > _VIZDOOM_DEATHMATCH_SAFE_COEFFICIENT_MAX:
            raise ValueError(f"{label}.{key} exceeds the Deathmatch float32 safety bound")
        normalized[key] = _normalize_zero(number)
    max_preference = max(normalized_preferences)
    for key in (
        "weapon_gain_reward_scale",
        "weapon_loss_penalty_scale",
        "ammo_gain_reward_scale",
        "ammo_loss_penalty_scale",
        "selected_weapon_hold_reward_scale",
    ):
        if max_preference * normalized[key] > _VIZDOOM_DEATHMATCH_SAFE_COEFFICIENT_MAX:
            raise ValueError(
                f"{label}.{key} combined with weapon_preferences exceeds the Deathmatch "
                "float32 safety bound"
            )
    return {key: normalized[key] for key in VIZDOOM_DEATHMATCH_REWARD_FIELDS if key in normalized}


def _mario_compiled_semantics(reward: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(reward["reward_mode"])
    semantics: dict[str, Any] = {
        "reward_mode": mode,
        "reward_scale": float(reward["reward_scale"]),
        "reward_clip": reward["reward_clip"],
        "time_penalty": float(reward["time_penalty"]),
    }
    if mode == "bounded":
        semantics.update(
            progress_reward_cap=float(reward["progress_reward_cap"]),
            terminal_reward=float(reward["terminal_reward"]),
        )
    elif mode == "baseline":
        semantics.update(
            terminal_reward=float(reward["terminal_reward"]),
        )
    elif mode in {"score", "additive"}:
        semantics.update(
            use_native_reward=bool(reward["use_native_reward"]),
            progress_reward_scale=float(reward["progress_reward_scale"]),
            progress_reward_boost_start_x=float(reward["progress_reward_boost_start_x"]),
            progress_reward_boost_scale=float(reward["progress_reward_boost_scale"]),
            completion_reward=float(reward["completion_reward"]),
            death_penalty=float(reward["death_penalty"]),
        )
        if mode == "score":
            clipped = bool(reward["score_progress_clipped"])
            semantics["score_progress_clipped"] = clipped
            if clipped:
                semantics["progress_reward_cap"] = float(reward["progress_reward_cap"])
    return semantics


def mario_reward_semantic_sha256(reward: Mapping[str, Any]) -> str:
    normalized = normalize_mario_reward(
        reward,
        label="Mario reward definition",
        require_complete=True,
    )
    return _sha256(
        {
            "task_id": "mario",
            "program_kind": REWARD_PROGRAM_KIND_MARIO_V1,
            "program_revision": MARIO_REWARD_KERNEL_REVISION,
            "compiled_semantics": _mario_compiled_semantics(normalized),
        }
    )


def vizdoom_deathmatch_reward_semantic_sha256(reward: Mapping[str, Any]) -> str:
    normalized = normalize_vizdoom_deathmatch_reward(
        reward,
        label="ViZDoom Deathmatch reward definition",
        require_complete=True,
    )
    return _sha256(
        {
            "task_id": "identity",
            "program_kind": REWARD_PROGRAM_KIND_VIZDOOM_DEATHMATCH_V1,
            "program_revision": VIZDOOM_DEATHMATCH_REWARD_KERNEL_REVISION,
            "compiled_semantics": normalized,
        }
    )


def reward_program_field_set(kind: object) -> frozenset[str]:
    if kind == REWARD_PROGRAM_KIND_MARIO_V1:
        return MARIO_REWARD_FIELD_SET
    if kind == REWARD_PROGRAM_KIND_VIZDOOM_DEATHMATCH_V1:
        return VIZDOOM_DEATHMATCH_REWARD_FIELD_SET
    raise ValueError(f"reward program kind has no registered compiler: {kind!r}")


def _reward_program_revision(kind: str) -> str:
    if kind == REWARD_PROGRAM_KIND_MARIO_V1:
        return MARIO_REWARD_KERNEL_REVISION
    if kind == REWARD_PROGRAM_KIND_VIZDOOM_DEATHMATCH_V1:
        return VIZDOOM_DEATHMATCH_REWARD_KERNEL_REVISION
    raise ValueError(f"reward program kind has no registered compiler: {kind!r}")


def _normalize_reward_program(
    kind: str,
    value: Mapping[str, Any],
    *,
    label: str,
    require_complete: bool,
) -> dict[str, Any]:
    if kind == REWARD_PROGRAM_KIND_MARIO_V1:
        return normalize_mario_reward(value, label=label, require_complete=require_complete)
    if kind == REWARD_PROGRAM_KIND_VIZDOOM_DEATHMATCH_V1:
        return normalize_vizdoom_deathmatch_reward(
            value,
            label=label,
            require_complete=require_complete,
        )
    raise ValueError(f"reward program kind has no registered compiler: {kind!r}")


def _reward_program_semantic_sha256(kind: str, reward: Mapping[str, Any]) -> str:
    if kind == REWARD_PROGRAM_KIND_MARIO_V1:
        return mario_reward_semantic_sha256(reward)
    if kind == REWARD_PROGRAM_KIND_VIZDOOM_DEATHMATCH_V1:
        return vizdoom_deathmatch_reward_semantic_sha256(reward)
    raise ValueError(f"reward program kind has no registered compiler: {kind!r}")


def _catalog(document: Mapping[str, Any], *, label: str) -> Mapping[str, Any] | None:
    value = document.get("reward_shapes")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}.reward_shapes must be an object")
    unknown = sorted(set(value) - {"program_kind", "default", "definitions"})
    if unknown:
        raise ValueError(
            f"{label}.reward_shapes has unknown field(s): {', '.join(str(x) for x in unknown)}"
        )
    return value


def validate_reward_shape_catalog(
    document: Mapping[str, Any],
    *,
    label: str = "goal",
) -> None:
    catalog = _catalog(document, label=label)
    if catalog is None:
        return
    kind = catalog.get("program_kind")
    reward_program_field_set(kind)
    assert isinstance(kind, str)
    default = catalog.get("default")
    if not isinstance(default, str) or not REWARD_SHAPE_KEY_PATTERN.fullmatch(default):
        raise ValueError(f"{label}.reward_shapes.default must be a lowercase kebab key")
    definitions = catalog.get("definitions")
    if not isinstance(definitions, Mapping) or not definitions:
        raise ValueError(f"{label}.reward_shapes.definitions must be a non-empty object")
    if default not in definitions:
        raise ValueError(f"{label}.reward_shapes.default references unknown key {default!r}")

    seen_hashes: dict[str, str] = {}
    for raw_key, raw_reward in definitions.items():
        key = str(raw_key)
        if not REWARD_SHAPE_KEY_PATTERN.fullmatch(key):
            raise ValueError(
                f"{label}.reward_shapes.definitions key {key!r} must be 1-64 lowercase kebab characters"
            )
        reward = _normalize_reward_program(
            kind,
            raw_reward,
            label=f"{label}.reward_shapes.definitions.{key}",
            require_complete=True,
        )
        semantic_hash = _reward_program_semantic_sha256(kind, reward)
        previous = seen_hashes.get(semantic_hash)
        if previous is not None:
            raise ValueError(
                f"{label}.reward_shapes definitions {previous!r} and {key!r} have identical executable semantics"
            )
        seen_hashes[semantic_hash] = key

    expected_task_id = "mario" if kind == REWARD_PROGRAM_KIND_MARIO_V1 else "identity"
    for phase in ("train", "eval"):
        section = document.get(phase)
        environment = section.get("environment") if isinstance(section, Mapping) else None
        task = environment.get("task") if isinstance(environment, Mapping) else None
        if not isinstance(task, Mapping) or task.get("id") != expected_task_id:
            raise ValueError(
                f"{label}.reward_shapes program {kind!r} requires "
                f"{phase}.environment.task.id={expected_task_id!r}"
            )
        if kind == REWARD_PROGRAM_KIND_MARIO_V1 and "reward" in task:
            raise ValueError(
                f"{label}.{phase}.environment.task.reward must be omitted when reward_shapes is declared"
            )
        if kind == REWARD_PROGRAM_KIND_VIZDOOM_DEATHMATCH_V1:
            base_reward = task.get("reward")
            if not isinstance(base_reward, Mapping):
                raise ValueError(
                    f"{label}.{phase}.environment.task.reward must declare the inherited native "
                    "Deathmatch reward when reward_shapes is declared"
                )
            normalized_base = normalize_vizdoom_deathmatch_reward(
                base_reward,
                label=f"{label}.{phase}.environment.task.reward",
                require_complete=True,
            )
            default_reward = normalize_vizdoom_deathmatch_reward(
                definitions[default],
                label=f"{label}.reward_shapes.definitions.{default}",
                require_complete=True,
            )
            if normalized_base != default_reward or normalized_base["reward_mode"] != "native":
                raise ValueError(
                    f"{label}.{phase}.environment.task.reward must match the native default "
                    "Deathmatch reward shape"
                )
    _validate_catalog_phase_semantics(document, label=label)
    if kind == REWARD_PROGRAM_KIND_MARIO_V1:
        _validate_mario_level_termination(document, label=label)


def _phase_semantic_projection(environment: Mapping[str, Any]) -> dict[str, Any]:
    projection = copy.deepcopy(dict(environment))
    env_config = projection.get("env_config")
    if isinstance(env_config, dict):
        for key in ("n_envs", "seed"):
            env_config.pop(key, None)
        env_args = env_config.get("env_args")
        if isinstance(env_args, dict):
            env_args.pop("num_threads", None)
    task = projection.get("task")
    if isinstance(task, dict):
        task.pop("reward", None)
    return projection


def _validate_catalog_phase_semantics(
    document: Mapping[str, Any],
    *,
    label: str,
) -> None:
    train = document.get("train")
    evaluation = document.get("eval")
    train_environment = train.get("environment") if isinstance(train, Mapping) else None
    eval_environment = evaluation.get("environment") if isinstance(evaluation, Mapping) else None
    if not isinstance(train_environment, Mapping) or not isinstance(eval_environment, Mapping):
        return
    if _phase_semantic_projection(train_environment) != _phase_semantic_projection(
        eval_environment
    ):
        raise ValueError(
            f"{label} catalog-backed train/eval environments must preserve task semantics, "
            "including termination; only execution settings may differ"
        )


def _validate_mario_level_termination(
    document: Mapping[str, Any],
    *,
    label: str,
) -> None:
    train_task = document["train"]["environment"]["task"]
    train_termination = train_task.get("termination")
    if not isinstance(train_termination, Mapping):
        return
    if train_termination.get("success") != ["level_change"]:
        return

    expected_failure = ["life_loss"]
    expected_success = ["level_change"]
    expected_timeout = ["stalled"]
    expected_stalled = {"signal": "x", "operation": "unchanged_for", "steps": 300}
    for phase in ("train", "eval"):
        task = document[phase]["environment"]["task"]
        termination = task.get("termination")
        events = task.get("events")
        if not isinstance(termination, Mapping):
            raise ValueError(f"{label}.{phase} Mario level task must declare termination")
        if termination.get("failure") != expected_failure:
            raise ValueError(
                f"{label}.{phase} Mario level task termination.failure must be {expected_failure!r}"
            )
        if termination.get("success") != expected_success:
            raise ValueError(
                f"{label}.{phase} Mario level task termination.success must be {expected_success!r}"
            )
        if termination.get("timeout") != expected_timeout:
            raise ValueError(
                f"{label}.{phase} Mario level task termination.timeout must be {expected_timeout!r}"
            )
        stalled = events.get("stalled") if isinstance(events, Mapping) else None
        if stalled != expected_stalled:
            raise ValueError(
                f"{label}.{phase} Mario level task must declare stalled={expected_stalled!r}"
            )


def _append_unique_string(value: object, item: str, *, label: str) -> None:
    if not isinstance(value, list) or any(not isinstance(entry, str) for entry in value):
        raise ValueError(f"{label} must be a list of strings")
    if item not in value:
        value.append(item)


def _materialize_vizdoom_deathmatch_reward(
    effective: dict[str, Any],
    reward: Mapping[str, Any],
    *,
    label: str,
) -> None:
    shaped = reward["reward_mode"] == "sample-factory-v0"
    for phase in ("train", "eval"):
        environment = effective[phase]["environment"]
        task = environment["task"]
        task["reward"] = copy.deepcopy(dict(reward))
        if not shaped:
            continue
        env_config = environment.get("env_config")
        env_args = env_config.get("env_args") if isinstance(env_config, Mapping) else None
        if not isinstance(env_args, dict):
            raise ValueError(f"{label}.{phase}.environment.env_config.env_args must be an object")
        variables = env_args.get("game_variables")
        info_filter = env_args.get("info_filter")
        info_keys = info_filter.get("keys") if isinstance(info_filter, Mapping) else None
        for semantic_name, provider_name in VIZDOOM_DEATHMATCH_REQUIRED_SIGNALS.items():
            _append_unique_string(
                variables,
                provider_name,
                label=f"{label}.{phase}.environment.env_config.env_args.game_variables",
            )
            _append_unique_string(
                info_keys,
                provider_name,
                label=f"{label}.{phase}.environment.env_config.env_args.info_filter.keys",
            )
            signals = task.get("signals")
            if not isinstance(signals, dict):
                raise ValueError(f"{label}.{phase}.environment.task.signals must be an object")
            existing = signals.get(semantic_name)
            if existing is not None and existing != provider_name:
                raise ValueError(
                    f"{label}.{phase}.environment.task.signals.{semantic_name} conflicts "
                    "with the Deathmatch reward program"
                )
            signals[semantic_name] = provider_name


def _materialize_reward_program(
    kind: str,
    effective: dict[str, Any],
    reward: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if kind == REWARD_PROGRAM_KIND_MARIO_V1:
        for phase in ("train", "eval"):
            effective[phase]["environment"]["task"]["reward"] = copy.deepcopy(dict(reward))
        return
    if kind == REWARD_PROGRAM_KIND_VIZDOOM_DEATHMATCH_V1:
        _materialize_vizdoom_deathmatch_reward(
            effective,
            reward,
            label=label,
        )
        return
    raise ValueError(f"reward program kind has no registered compiler: {kind!r}")


def select_goal_reward_shape(
    document: Mapping[str, Any],
    selector: str | None,
    *,
    label: str = "goal",
) -> RewardShapeSelection | None:
    catalog = _catalog(document, label=label)
    if catalog is None:
        if selector is not None:
            raise ValueError(f"{label} does not define reward_shapes; reward_shape is unsupported")
        return None
    validate_reward_shape_catalog(document, label=label)
    default = str(catalog["default"])
    key = str(selector).strip() if selector is not None else default
    if not REWARD_SHAPE_KEY_PATTERN.fullmatch(key):
        raise ValueError("reward_shape must be a 1-64 character lowercase kebab key")
    definitions = catalog["definitions"]
    if key not in definitions:
        available = ", ".join(sorted(str(item) for item in definitions))
        raise ValueError(f"unknown reward_shape {key!r}; available: {available}")
    kind = str(catalog["program_kind"])
    reward = _normalize_reward_program(
        kind,
        definitions[key],
        label=f"{label}.reward_shapes.definitions.{key}",
        require_complete=True,
    )
    effective = copy.deepcopy(dict(document))
    effective.pop("reward_shapes", None)
    _materialize_reward_program(kind, effective, reward, label=label)
    return RewardShapeSelection(
        goal=effective,
        key=key,
        program_kind=kind,
        program_revision=_reward_program_revision(kind),
        semantic_sha256=_reward_program_semantic_sha256(kind, reward),
        is_default=key == default,
        reward=reward,
    )


def goal_for_contract_validation(
    document: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    selected = select_goal_reward_shape(document, None, label=label)
    return selected.goal if selected is not None else copy.deepcopy(dict(document))
