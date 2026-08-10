from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from gradlab.environment_fields import (
    ENVIRONMENT_FIELD_SPECS,
    EnvConfig,
    EnvironmentFieldSpec,
    FieldKind,
    SequenceItemKind,
    TypeName,
)
from gradlab.local_paths import PORTABLE_DEFAULT_RUNS_DIR
from gradlab.metric_names import METRICS_SCHEMA_VERSION
from gradlab.modal_eval_protocol import SEED_PROTOCOL
from gradlab.seeds import DEFAULT_TRAIN_SEED, EVAL_SEED_START
from gradlab.validation import is_int as _is_int
from gradlab.validation import label_path as _label_path
from gradlab.validation import normalize_obs_crop, normalize_obs_resize


WANDB_MODE_CHOICES = ("online", "offline", "disabled")

FieldOwner = Literal["runtime", "goal_environment", "goal_objective"]
SourceSection = Literal["runtime", "train", "goal_train"]


def wandb_publication_enabled(train_config: Mapping[str, Any]) -> bool:
    """Return whether the canonical W&B mode enables metric publication."""

    return str(train_config.get("wandb_mode") or "online") != "disabled"


def checkpoint_eval_requires_acceptance(train_config: Mapping[str, Any]) -> bool:
    """Derive acceptance behavior from the evaluation backend and contract."""

    return (
        str(train_config.get("checkpoint_eval_backend") or "none") == "modal"
        and train_config.get("checkpoint_eval_acceptance") is not None
    )


@dataclass(frozen=True)
class TrainConfigField:
    dest: str
    flag: str | None = None
    kind: FieldKind = "value"
    type_name: TypeName = "str"
    default: Any = None
    env_default: str | None = None
    choices: tuple[str, ...] = ()
    help: str | None = None
    environment: bool = False
    non_empty: bool = False
    validation_min: float | None = None
    validation_max: float | None = None
    sequence_items: SequenceItemKind | None = None
    allow_empty_sequence: bool = False
    mapping_value: bool = False
    owner: FieldOwner = "runtime"
    source_section: SourceSection = "runtime"
    mixed_state: bool = False


def _field(dest: str, *, flag: str | None = None, **metadata: Any) -> TrainConfigField:
    cli_flag = flag or f"--{dest.replace('_', '-')}" if metadata.get("environment") else None
    return TrainConfigField(dest, flag=cli_flag, **metadata)


def _environment_field(spec: EnvironmentFieldSpec) -> TrainConfigField:
    return _field(
        spec.dest,
        flag=spec.flag,
        kind=spec.kind,
        type_name=spec.type_name,
        default=spec.cli_default,
        env_default=spec.dest if spec.use_runtime_default else None,
        choices=spec.choices,
        help=spec.help,
        environment=True,
        non_empty=spec.non_empty,
        validation_min=spec.validation_min,
        validation_max=spec.validation_max,
        sequence_items=spec.sequence_items,
        mapping_value=spec.mapping_value,
        mixed_state=spec.mixed_state,
    )


def _env_default(env_defaults: EnvConfig, field: TrainConfigField) -> Any:
    if field.env_default is None:
        return field.default
    return getattr(env_defaults, field.env_default)


def _type_callable(
    field: TrainConfigField,
    *,
    parse_json_value: Callable[[str], Any],
    parse_obs_crop: Callable[[Any], Any],
) -> Callable[[Any], Any] | type | None:
    if field.kind != "value":
        return None
    if field.type_name == "int":
        return int
    if field.type_name == "float":
        return float
    if field.type_name == "json":
        return parse_json_value
    if field.type_name == "obs_crop":
        return parse_obs_crop
    if field.type_name == "obs_resize":
        return normalize_obs_resize
    return None


def _add_config_field_argument(
    parser: argparse.ArgumentParser,
    field: TrainConfigField,
    *,
    default: Any,
    parse_json_value: Callable[[str], Any],
    parse_obs_crop: Callable[[Any], Any],
    dest: str | None = None,
) -> None:
    if field.flag is None:
        raise ValueError(f"{field.dest} is not exposed on the environment CLI")
    kwargs: dict[str, Any] = {"dest": dest or field.dest, "default": default}
    if field.help is not None:
        kwargs["help"] = field.help
    if field.choices:
        kwargs["choices"] = field.choices
    if field.kind == "bool_optional":
        kwargs["action"] = argparse.BooleanOptionalAction
    else:
        type_callable = _type_callable(
            field,
            parse_json_value=parse_json_value,
            parse_obs_crop=parse_obs_crop,
        )
        if type_callable is not None:
            kwargs["type"] = type_callable
    parser.add_argument(field.flag, **kwargs)


def add_env_config_args(
    parser: argparse.ArgumentParser,
    *,
    watchdog_steps_default: int,
    defaults: EnvConfig | None = None,
    parse_json_value: Callable[[str], Any],
    parse_obs_crop: Callable[[Any], Any],
) -> None:
    defaults = defaults or EnvConfig()
    for field in env_config_arg_fields():
        dest = field.dest
        default = _env_default(defaults, field)
        _add_config_field_argument(
            parser,
            field,
            default=default,
            parse_json_value=parse_json_value,
            parse_obs_crop=parse_obs_crop,
            dest=dest,
        )
    parser.add_argument(
        "--watchdog-steps",
        type=int,
        default=watchdog_steps_default,
        help="Abort an episode if its scientific boundary fails to produce a record.",
    )


def load_materialized_train_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"train config file must contain a JSON object: {path}")
    defaults = {field.dest: _env_default(EnvConfig(), field) for field in TRAIN_CONFIG_FIELDS}
    normalized = validate_and_normalize_train_config(
        payload,
        label=f"train config file {path}",
        required_keys=("training_backend",),
        enforce_early_stop_policy=True,
    )
    return {**defaults, **normalized}


def train_config_field_for_key(key: str) -> TrainConfigField | None:
    for field in TRAIN_CONFIG_FIELDS:
        if field.dest == key:
            return field
    return None


def env_config_arg_fields() -> tuple[TrainConfigField, ...]:
    return tuple(field for field in TRAIN_CONFIG_FIELDS if field.environment)


def env_config_allowed_keys() -> frozenset[str]:
    return frozenset(field.dest for field in env_config_arg_fields())


def train_config_keys_owned_by(owner: FieldOwner) -> frozenset[str]:
    keys: set[str] = set()
    for field in TRAIN_CONFIG_FIELDS:
        is_owned = field.owner == owner or (owner == "goal_environment" and field.environment)
        if not is_owned:
            continue
        keys.add(field.dest)
    return frozenset(keys)


def train_config_keys_in_source_section(section: SourceSection) -> frozenset[str]:
    return frozenset(field.dest for field in TRAIN_CONFIG_FIELDS if field.source_section == section)


def _validate_number_bounds(
    *,
    key: str,
    label: str,
    number: float,
    minimum: float | None,
    maximum: float | None,
) -> None:
    if minimum is not None and number < minimum:
        raise ValueError(f"{_label_path(label, key)} must be >= {minimum:g}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{_label_path(label, key)} must be <= {maximum:g}")


def _validate_string_sequence(
    *,
    key: str,
    label: str,
    value: Any,
    allow_empty: bool,
) -> None:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{_label_path(label, key)} must be a list")
    if not value and not allow_empty:
        raise ValueError(f"{_label_path(label, key)} must not be empty")
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{_label_path(label, key)}[{index}] must be a non-empty string")


def _validate_number_sequence(
    *,
    key: str,
    label: str,
    value: Any,
    allow_empty: bool,
) -> None:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{_label_path(label, key)} must be a list")
    if not value and not allow_empty:
        raise ValueError(f"{_label_path(label, key)} must not be empty")
    for index, item in enumerate(value):
        if not isinstance(item, int | float) or isinstance(item, bool):
            raise ValueError(f"{_label_path(label, key)}[{index}] must be a number")


def _validate_row_sequence(
    *,
    key: str,
    label: str,
    value: Any,
    allow_empty: bool,
) -> None:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{_label_path(label, key)} must be a list")
    if not value and not allow_empty:
        raise ValueError(f"{_label_path(label, key)} must not be empty")
    for row_index, row in enumerate(value):
        if not isinstance(row, Sequence) or isinstance(row, str | bytes) or not row:
            raise ValueError(f"{_label_path(label, key)}[{row_index}] must be a non-empty list")
        for column_index, item in enumerate(row):
            if not isinstance(item, int | str) or isinstance(item, bool):
                raise ValueError(
                    f"{_label_path(label, key)}[{row_index}][{column_index}] "
                    "must be an integer or string"
                )


def _validate_obs_crop_value(*, key: str, label: str, value: Any) -> None:
    normalize_obs_crop(value, label=_label_path(label, key))


def validate_train_config_value(
    key: str,
    value: Any,
    *,
    label: str = "train_config",
) -> None:
    field = train_config_field_for_key(key)
    if field is None:
        raise ValueError(f"{_label_path(label, key)} is not a known train config field")
    if field.sequence_items == "str":
        _validate_string_sequence(
            key=key,
            label=label,
            value=value,
            allow_empty=field.allow_empty_sequence,
        )
        return
    if field.sequence_items == "number":
        _validate_number_sequence(
            key=key,
            label=label,
            value=value,
            allow_empty=field.allow_empty_sequence,
        )
        return
    if field.sequence_items == "rows":
        _validate_row_sequence(
            key=key,
            label=label,
            value=value,
            allow_empty=field.allow_empty_sequence,
        )
        return
    if field.mapping_value:
        if not isinstance(value, Mapping):
            raise ValueError(f"{_label_path(label, key)} must be an object")
        return
    if field.kind == "bool_optional":
        if not isinstance(value, bool):
            raise ValueError(f"{_label_path(label, key)} must be a boolean")
        return
    if field.type_name == "int":
        if not _is_int(value):
            raise ValueError(f"{_label_path(label, key)} must be an integer")
        _validate_number_bounds(
            key=key,
            label=label,
            number=float(value),
            minimum=field.validation_min,
            maximum=field.validation_max,
        )
        return
    if field.type_name == "float":
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ValueError(f"{_label_path(label, key)} must be a number")
        _validate_number_bounds(
            key=key,
            label=label,
            number=float(value),
            minimum=field.validation_min,
            maximum=field.validation_max,
        )
        return
    if field.type_name == "obs_crop":
        _validate_obs_crop_value(key=key, label=label, value=value)
        return
    if field.type_name == "obs_resize":
        normalize_obs_resize(value, label=_label_path(label, key))
        return
    if field.type_name == "json":
        return
    if not isinstance(value, str):
        raise ValueError(f"{_label_path(label, key)} must be a string")
    if field.non_empty and not value.strip():
        raise ValueError(f"{_label_path(label, key)} must be a non-empty string")
    if field.choices and value not in field.choices:
        choices = ", ".join(field.choices)
        raise ValueError(f"{_label_path(label, key)} must be one of {choices}")


def validate_train_config_fields(
    train_config: Mapping[str, Any],
    *,
    label: str = "train_config",
    keys: Sequence[str] | None = None,
    required_keys: Sequence[str] = (),
) -> None:
    missing = [key for key in required_keys if key not in train_config]
    if missing:
        raise ValueError(f"{label} missing required field(s): {', '.join(missing)}")
    selected_keys = tuple(keys) if keys is not None else tuple(train_config)
    for key in selected_keys:
        if key in train_config:
            validate_train_config_value(key, train_config[key], label=label)


def validate_and_normalize_train_config(
    train_config: Mapping[str, Any],
    *,
    label: str = "train_config",
    required_keys: Sequence[str] = (),
    validate_backend_config: bool = True,
    enforce_early_stop_policy: bool = False,
    metric_validator: Callable[[str], object] | None = None,
) -> dict[str, Any]:
    """Validate one flat train config and normalize its structured rule fields."""

    from gradlab.early_stop import (
        normalize_metric_early_stop_config,
        normalize_metric_threshold_rules,
        validate_metric_early_stop_policy,
    )
    from gradlab.state_archive import normalize_state_archive_config

    normalized = dict(train_config)
    validate_train_config_fields(normalized, label=label, required_keys=required_keys)
    if "obs_resize" in normalized:
        normalized["obs_resize"] = normalize_obs_resize(
            normalized["obs_resize"],
            label=f"{label}.obs_resize",
        )
    if normalized.get("early_stop") is not None:
        early_stop_validator = (
            validate_metric_early_stop_policy
            if enforce_early_stop_policy
            else normalize_metric_early_stop_config
        )
        normalized["early_stop"] = early_stop_validator(
            normalized["early_stop"],
            label=f"{label}.early_stop",
            **({} if metric_validator is None else {"metric_validator": metric_validator}),
        )
    if normalized.get("checkpoint_eval_acceptance") is not None:
        normalized["checkpoint_eval_acceptance"] = normalize_metric_threshold_rules(
            normalized["checkpoint_eval_acceptance"],
            label=f"{label}.checkpoint_eval_acceptance",
            **({} if metric_validator is None else {"metric_validator": metric_validator}),
        )
    if normalized.get("state_archive") is not None:
        n_envs = normalized.get("n_envs")
        normalized["state_archive"] = normalize_state_archive_config(
            normalized["state_archive"],
            label=f"{label}.state_archive",
            n_envs=int(n_envs) if n_envs is not None else None,
        )
    policy_model = normalized.get("policy_model")
    if policy_model is not None:
        from gradlab.policy_model_config import (
            normalize_policy_model,
            validate_policy_model_context,
        )

        normalized["policy_model"] = normalize_policy_model(
            policy_model,
            label=f"{label}.policy_model",
        )
        task = normalized.get("task")
        if isinstance(task, Mapping):
            validate_policy_model_context(
                normalized["policy_model"],
                task,
                label=f"{label}.policy_model",
            )
    else:
        task = normalized.get("task")
        if isinstance(task, Mapping):
            from gradlab.policy_model_config import validate_policy_model_context

            validate_policy_model_context(
                None,
                task,
                label=f"{label}.policy_model",
            )
    episode_progress_fields = tuple(normalized.get("episode_progress_fields", ()))
    if episode_progress_fields:
        signals = task.get("signals") if isinstance(task, Mapping) else None
        if not isinstance(signals, Mapping):
            raise ValueError(f"{label}.episode_progress_fields requires task.signals")
        missing_progress_fields = sorted(set(episode_progress_fields) - set(signals))
        if missing_progress_fields:
            raise ValueError(
                f"{label}.episode_progress_fields references unknown task signal(s): "
                + ", ".join(missing_progress_fields)
            )
    early_stop = normalized.get("early_stop")
    conditions = early_stop.get("conditions") if isinstance(early_stop, Mapping) else None
    has_training_success_condition = isinstance(conditions, Mapping) and any(
        str(condition.get("outcome")) == "success"
        for condition in conditions.values()
        if isinstance(condition, Mapping)
    )
    if "training_backend" in normalized:
        common_config = {
            key: value for key, value in normalized.items() if key != "training_backend"
        }
        if validate_backend_config:
            from gradlab.training_backend import normalize_training_backend

            normalized["training_backend"] = normalize_training_backend(
                normalized["training_backend"],
                common_config=common_config,
                label=f"{label}.training_backend",
            )
            from gradlab.training_backend import accepts_first_training_success

            if accepts_first_training_success(normalized) and has_training_success_condition:
                raise ValueError(
                    f"{label}.early_stop success conditions are incompatible with "
                    "first-training-success backend acceptance"
                )
        else:
            from gradlab.training_backend import validate_training_backend_envelope

            backend_id, backend_config = validate_training_backend_envelope(
                normalized["training_backend"],
                label=f"{label}.training_backend",
            )
            normalized["training_backend"] = {
                "id": backend_id,
                "config": backend_config,
            }
    return normalized


TRAIN_CONFIG_FIELDS: tuple[TrainConfigField, ...] = (
    _field(
        "timesteps",
        type_name="int",
        default=1_000_000,
        validation_min=1,
        source_section="train",
    ),
    _field(
        "state_archive",
        type_name="json",
        default=None,
        mapping_value=True,
        source_section="train",
    ),
    _field(
        "training_backend",
        type_name="json",
        default=None,
        mapping_value=True,
    ),
    _field(
        "policy_model",
        type_name="json",
        default=None,
        source_section="train",
    ),
    _field(
        "episode_progress_fields",
        default=(),
        sequence_items="str",
        allow_empty_sequence=True,
        owner="goal_objective",
        source_section="goal_train",
    ),
    _field(
        "n_envs",
        type_name="int",
        default=8,
        validation_min=1,
        owner="goal_environment",
    ),
    _field(
        "seed",
        type_name="int",
        default=DEFAULT_TRAIN_SEED,
    ),
    _field("run_name", default="ppo_retro"),
    _field("run_description", default=""),
    _field("runs_dir", default=PORTABLE_DEFAULT_RUNS_DIR),
    *(_environment_field(spec) for spec in ENVIRONMENT_FIELD_SPECS),
    _field(
        "checkpoint_freq",
        type_name="int",
        default=500_000,
        validation_min=0,
        source_section="goal_train",
    ),
    _field(
        "post_train_eval_episodes",
        type_name="int",
        default=100,
        owner="goal_objective",
        source_section="goal_train",
    ),
    _field(
        "checkpoint_eval_environment",
        type_name="json",
        default=None,
        mapping_value=True,
        owner="goal_objective",
        source_section="goal_train",
    ),
    _field(
        "checkpoint_eval_n_envs",
        type_name="int",
        default=20,
        validation_min=1,
        owner="goal_objective",
        source_section="goal_train",
    ),
    _field(
        "checkpoint_eval_acceptance",
        type_name="json",
        default=None,
        owner="goal_objective",
        source_section="goal_train",
    ),
    _field(
        "checkpoint_eval_contract",
        type_name="json",
        default=None,
        mapping_value=True,
    ),
    _field(
        "checkpoint_eval_backend",
        default="modal",
        choices=("modal", "none"),
        non_empty=True,
        source_section="train",
    ),
    _field(
        "rom_asset_manifest",
        type_name="json",
        default=None,
        mapping_value=True,
    ),
    _field(
        "checkpoint_eval_seed_protocol",
        default=SEED_PROTOCOL,
        choices=(SEED_PROTOCOL,),
        non_empty=True,
    ),
    _field(
        "checkpoint_eval_seed",
        type_name="int",
        default=EVAL_SEED_START,
        validation_min=EVAL_SEED_START,
    ),
    _field(
        "checkpoint_eval_watchdog_steps",
        type_name="int",
        default=0,
        owner="goal_objective",
        source_section="goal_train",
    ),
    _field(
        "early_stop",
        type_name="json",
        default=None,
        source_section="train",
    ),
    _field(
        "selection_rank",
        type_name="json",
        default=(),
        sequence_items="str",
        owner="goal_objective",
    ),
    _field(
        "metrics_schema_version",
        type_name="int",
        default=METRICS_SCHEMA_VERSION,
        validation_min=METRICS_SCHEMA_VERSION,
        validation_max=METRICS_SCHEMA_VERSION,
    ),
    _field("wandb_project", default=None),
    _field("wandb_entity", default=None),
    _field("wandb_display_name", default=None),
    _field("wandb_group", default=None),
    _field("wandb_tags", default=""),
    _field(
        "wandb_mode",
        default="online",
        choices=WANDB_MODE_CHOICES,
        non_empty=True,
    ),
    _field("runtime_image_ref", default=""),
    _field("runtime_input_sha256", default=""),
    _field("runtime_build_source_sha", default=""),
    _field("source_sha", default=""),
    _field("compute_target", default=""),
    _field("dstack_coordinator_id", default=""),
    _field("dstack_project", default=""),
    _field("attempt_id", default=""),
    _field("dstack_task", default=""),
    _field("campaign_id", default=""),
    _field("game_family", default=""),
    _field("goal_slug", default=""),
    _field("goal_path", default=""),
    _field("goal_sha256", default=""),
    _field("goal_contract_sha256", default=""),
    _field("effective_goal_contract_sha256", default=""),
    _field("action_profile", default=""),
    _field("action_profile_revision", default=""),
    _field("action_profile_sha256", default=""),
    _field("action_profile_source_table_sha256", default=""),
    _field("reward_program_kind", default=""),
    _field("reward_program_revision", default=""),
    _field("reward_shape", default=""),
    _field("reward_shape_sha256", default=""),
    _field(
        "reward_shape_is_default",
        kind="bool_optional",
        default=False,
    ),
    _field("recipe_slug", default=""),
    _field("recipe_path", default=""),
    _field("recipe_json_path", default=""),
    _field("recipe_sha256", default=""),
    _field(
        "recipe_composition",
        type_name="json",
        default={},
        mapping_value=True,
    ),
    _field(
        "recipe_overrides",
        type_name="json",
        default=(),
    ),
    _field("recipe_variant_id", default=""),
    _field("wandb_run_id", default=""),
)
