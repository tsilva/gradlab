from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from gradlab.config_loader import (
    YAML_EXTENSIONS,
    ComposedDocument,
    RECIPE_TEMPLATE_FIELDS,
    TEMPLATE_VARS_KEY,
    apply_dotlist_overrides,
    deep_merge,
    dotlist_to_mapping,
    load_composed_mapping,
    load_mapping_document,
    render_template_vars,
    template_context_from_path,
)
from gradlab.env_identity import (
    attach_environment_identity,
    train_config_from_source_environment,
    validate_task_config,
)
from gradlab.experiment_contracts import validate_goal_contract_document
from gradlab.file_utils import file_sha256
from gradlab.provider_config import NON_SEMANTIC_ENV_ARG_KEYS
from gradlab.recipe_schema import (
    train_recipe_id,
    validate_materialized_train_recipe,
)
from gradlab.reward_programs import MARIO_REWARD_FIELD_SET, select_goal_reward_shape
from gradlab.train_config import train_config_keys_in_source_section, train_config_keys_owned_by


SECRET_KEY_FRAGMENTS = (
    "api_key",
    "access_key",
    "secret",
    "token",
    "password",
    "credential",
    "database_url",
)
TRAIN_CONFIG_SECTION_KEYS = ("train", "logging")
TRAIN_NESTED_SECTION_KEYS = frozenset({"backend"})
COMMON_TRAIN_CONFIG_KEYS = train_config_keys_in_source_section("train")
GOAL_TRAIN_CONFIG_KEYS = train_config_keys_in_source_section("goal_train")
SOURCE_RECIPE_FIELDS = frozenset(
    {
        "campaign_id",
        "defaults",
        "description",
        "logging",
        "max_attempts",
        "metadata",
        "notes",
        "recipe_id",
        "reward_shape",
        "schema_version",
        "seeds",
        TEMPLATE_VARS_KEY,
        "train",
    }
)
SOURCE_PRESET_FIELDS = frozenset({"defaults", "logging", TEMPLATE_VARS_KEY, "train"})
RECIPE_DEFERRED_TEMPLATE_FIELDS: dict[tuple[str, ...], frozenset[str]] = {
    ("description",): RECIPE_TEMPLATE_FIELDS,
    ("goal", "description"): RECIPE_TEMPLATE_FIELDS,
    ("goal", "tags", "2"): frozenset({"env_id"}),
}
GOAL_DEFERRED_TEMPLATE_FIELDS: dict[tuple[str, ...], frozenset[str]] = {
    **RECIPE_DEFERRED_TEMPLATE_FIELDS,
    ("tags", "1"): frozenset({"slug", "recipe_id", "recipe_slug"}),
    ("tags", "2"): frozenset({"env_id"}),
}
GOAL_OWNED_ENV_CONFIG_KEYS = train_config_keys_owned_by("goal_environment") | {
    "provider",
    "env_id",
}
GOAL_OWNED_OBJECTIVE_CONFIG_KEYS = train_config_keys_owned_by("goal_objective")
REWARD_DEFINITION_OVERRIDE_PREFIX = "reward_shapes.definitions."
_POLICY_ENVIRONMENT_PREFIXES = {
    "train": "train.environment.",
    "eval": "eval.environment.",
}
_PHASE_EXECUTION_ENV_PATHS = frozenset(
    {
        "env_config.max_steps",
        "env_config.n_envs",
        "env_config.seed",
    }
)
_LEGACY_PROVIDER_REWARD_PATHS = {
    "env_config.env_args.reward_clip": "task.reward.reward_clip",
    "env_config.env_args.reward_clipping": "task.reward.reward_clip",
}


def goal_contract_sha256(document: Mapping[str, Any]) -> str:
    """Hash the fully composed semantic goal contract, excluding source formatting."""

    payload = json.dumps(document, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _override_parts(value: str, *, label: str) -> tuple[str, str, Any]:
    if "=" not in value:
        raise ValueError(f"{label} must use path=value syntax: {value!r}")
    path, raw_value = value.split("=", 1)
    path = path.strip()
    if not path:
        raise ValueError(f"{label} path must be non-empty")
    parsed = dotlist_to_mapping(
        [f"value={raw_value}"],
        label=label,
    )
    return path, raw_value, parsed["value"]


def _phase_environment_override(path: str) -> tuple[str, str] | None:
    for phase, prefix in _POLICY_ENVIRONMENT_PREFIXES.items():
        if path.startswith(prefix):
            relative = path.removeprefix(prefix)
            if not relative:
                raise ValueError(f"{path} must target a field below {prefix.removesuffix('.')}")
            return phase, relative
    return None


def _is_execution_environment_path(relative: str) -> bool:
    if relative in _PHASE_EXECUTION_ENV_PATHS:
        return True
    prefix = "env_config.env_args."
    return relative.startswith(prefix) and relative.removeprefix(prefix) in (
        NON_SEMANTIC_ENV_ARG_KEYS
    )


def _canonical_policy_environment_path(relative: str) -> str:
    return _LEGACY_PROVIDER_REWARD_PATHS.get(relative, relative)


def _partition_policy_environment_overrides(
    overrides: Sequence[str],
    *,
    goal_document: Mapping[str, Any],
    label: str,
) -> tuple[list[str], list[str], list[str]]:
    """Separate recipe-local overrides from one mirrored policy environment contract."""

    source_overrides: list[str] = []
    by_path: dict[str, dict[str, tuple[str, Any]]] = {}
    for item in overrides:
        path, raw_value, parsed_value = _override_parts(item, label=label)
        phase_path = _phase_environment_override(path)
        if phase_path is None:
            source_overrides.append(item)
            continue
        phase, relative = phase_path
        canonical = _canonical_policy_environment_path(relative)
        if canonical == relative and _is_execution_environment_path(relative):
            if phase == "eval":
                raise ValueError(
                    f"{path} is goal-owned evaluation execution configuration and cannot be "
                    "changed by a recipe override"
                )
            source_overrides.append(item)
            continue
        existing = by_path.setdefault(canonical, {}).get(phase)
        if existing is not None and existing[1] != parsed_value:
            raise ValueError(
                f"conflicting {phase} policy environment overrides for "
                f"{canonical}: {existing[1]!r} != {parsed_value!r}"
            )
        by_path[canonical][phase] = (raw_value, parsed_value)

    has_eval = isinstance(goal_document.get("eval"), Mapping)
    goal_overrides: list[str] = []
    effective_overrides: list[str] = []
    catalog = goal_document.get("reward_shapes")
    for relative, phases in sorted(by_path.items()):
        training = phases.get("train")
        evaluation = phases.get("eval")
        if training is None:
            raise ValueError(
                f"eval.environment.{relative} cannot define policy semantics independently; "
                f"set train.environment.{relative} and gradlab will mirror it into evaluation"
            )
        if evaluation is not None and evaluation[1] != training[1]:
            raise ValueError(
                f"train/eval policy environment overrides disagree for {relative}: "
                f"{training[1]!r} != {evaluation[1]!r}"
            )
        if isinstance(catalog, Mapping) and relative.startswith("task.reward."):
            raise ValueError(
                "catalog goals reject raw reward overrides; select or override a named "
                f"reward_shape instead: train.environment.{relative}"
            )
        raw_value = training[0]
        goal_overrides.append(f"train.environment.{relative}={raw_value}")
        effective_overrides.append(f"train.environment.{relative}={raw_value}")
        if has_eval:
            goal_overrides.append(f"eval.environment.{relative}={raw_value}")
            effective_overrides.append(f"eval.environment.{relative}={raw_value}")
        elif evaluation is not None:
            raise ValueError("training-only goals do not support eval.environment overrides")
    return source_overrides, goal_overrides, effective_overrides


def _contains_secret_key(value: Any, path: str = "") -> str | None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            nested_path = f"{path}.{key}" if path else str(key)
            if any(fragment in key_text for fragment in SECRET_KEY_FRAGMENTS):
                return nested_path
            found = _contains_secret_key(nested, nested_path)
            if found:
                return found
    elif isinstance(value, list | tuple):
        for index, nested in enumerate(value):
            found = _contains_secret_key(nested, f"{path}[{index}]")
            if found:
                return found
    return None


def assert_no_secrets(value: Any, *, label: str) -> None:
    found = _contains_secret_key(value)
    if found:
        raise ValueError(f"{label} appears to contain a secret-like key: {found}")


def _document_train_environment(document: Mapping[str, Any]) -> Mapping[str, Any] | None:
    train_section = document.get("train")
    if isinstance(train_section, Mapping):
        train_environment = train_section.get("environment")
        if isinstance(train_environment, Mapping):
            return train_environment
    return None


def _without_keys(value: Mapping[str, Any], keys: frozenset[str]) -> dict[str, Any]:
    return {
        nested_key: copy.deepcopy(nested_value)
        for nested_key, nested_value in value.items()
        if nested_key not in keys
    }


def _goal_train_defaults(document: Mapping[str, Any]) -> dict[str, Any]:
    environment = _document_train_environment(document)
    config = (
        _train_environment_section_config(environment) if isinstance(environment, Mapping) else {}
    )
    train = document.get("train")
    if isinstance(train, Mapping):
        config = deep_merge(config, _train_config_from_train_section(train))
    config = deep_merge(config, _eval_train_defaults(document))
    objective = document.get("objective")
    if isinstance(objective, Mapping) and isinstance(objective.get("rank"), Sequence):
        config["selection_rank"] = copy.deepcopy(objective["rank"])
    return config


def _eval_train_defaults(document: Mapping[str, Any]) -> dict[str, Any]:
    eval_section = document.get("eval")
    if not isinstance(eval_section, Mapping):
        return {}
    episodes = eval_section.get("episodes")
    if episodes is None:
        return {}
    defaults: dict[str, Any] = {"post_train_eval_episodes": copy.deepcopy(episodes)}
    if "acceptance" in eval_section:
        defaults["checkpoint_eval_acceptance"] = copy.deepcopy(eval_section["acceptance"])
    environment = eval_section.get("environment")
    if not isinstance(environment, Mapping):
        return defaults
    eval_config = _train_environment_section_config(environment)
    if "n_envs" in eval_config:
        defaults["checkpoint_eval_n_envs"] = eval_config.pop("n_envs")
    if "max_steps" in eval_config:
        defaults["post_train_eval_max_steps"] = eval_config.pop("max_steps")
    defaults["checkpoint_eval_environment"] = eval_config
    return defaults


def _train_config_section_value(
    document: Mapping[str, Any],
    key: str,
    *,
    strip_goal_owned: bool = False,
) -> Mapping[str, Any] | None:
    value = document.get(key)
    if not isinstance(value, Mapping):
        return None
    if key != "train":
        section = dict(value)
    else:
        section = _train_config_from_train_section(value)
    if not strip_goal_owned:
        return section
    if key == "logging":
        return _without_keys(section, GOAL_OWNED_OBJECTIVE_CONFIG_KEYS)
    if key == "train":
        return _without_keys(section, GOAL_OWNED_ENV_CONFIG_KEYS | GOAL_OWNED_OBJECTIVE_CONFIG_KEYS)
    return section


def _train_environment_section_config(environment: Mapping[str, Any]) -> dict[str, Any]:
    return train_config_from_source_environment(environment)


def _normalized_train_section(section: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(section, Mapping):
        return {}
    nested_environment = section.get("environment")
    environment = (
        copy.deepcopy(dict(nested_environment)) if isinstance(nested_environment, Mapping) else {}
    )
    common = {
        key: copy.deepcopy(value)
        for key, value in section.items()
        if key in GOAL_TRAIN_CONFIG_KEYS | COMMON_TRAIN_CONFIG_KEYS
    }
    nested_backend = section.get("backend")
    backend = copy.deepcopy(dict(nested_backend)) if isinstance(nested_backend, Mapping) else {}
    normalized: dict[str, Any] = {}
    if environment:
        normalized["environment"] = environment
    normalized.update(common)
    if backend:
        normalized["backend"] = backend
    return normalized


def _train_config_from_train_section(section: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _normalized_train_section(section)
    config: dict[str, Any] = {}
    environment = normalized.get("environment")
    if isinstance(environment, Mapping):
        config = deep_merge(config, _train_environment_section_config(environment))
    common = {
        key: copy.deepcopy(value)
        for key, value in normalized.items()
        if key in GOAL_TRAIN_CONFIG_KEYS | COMMON_TRAIN_CONFIG_KEYS
    }
    config = deep_merge(config, common)
    backend = normalized.get("backend")
    if isinstance(backend, Mapping):
        config["training_backend"] = copy.deepcopy(dict(backend))
    return config


def _explicit_train_environment_config(document: Mapping[str, Any]) -> Mapping[str, Any] | None:
    train = document.get("train")
    if not isinstance(train, Mapping):
        return None
    environment = train.get("environment")
    if not isinstance(environment, Mapping):
        return None
    return _train_environment_section_config(environment)


def _merge_train_config_sections(
    document: Mapping[str, Any],
    *,
    goal_document: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    strip_goal_owned = goal_document is not None
    train_config: dict[str, Any] = _goal_train_defaults(goal_document or {})
    for key in TRAIN_CONFIG_SECTION_KEYS:
        value = _train_config_section_value(document, key, strip_goal_owned=strip_goal_owned)
        if isinstance(value, Mapping):
            train_config = deep_merge(train_config, value)
    if strip_goal_owned:
        explicit_environment = _explicit_train_environment_config(document)
        if isinstance(explicit_environment, Mapping):
            train_config = deep_merge(train_config, explicit_environment)

    return train_config


def _goal_with_environment_provider(
    document: Mapping[str, Any], provider: str | None
) -> dict[str, Any]:
    result = copy.deepcopy(dict(document))
    if provider is None:
        return result
    provider_id = str(provider).strip()
    if not provider_id:
        raise ValueError("environment provider override must be non-empty")
    from gradlab.env_registry import resolve_env_provider

    resolve_env_provider(provider_id)
    for section_name in ("train", "eval"):
        section = result.get(section_name)
        environment = section.get("environment") if isinstance(section, Mapping) else None
        if section_name == "eval" and section is None:
            continue
        if not isinstance(environment, dict):
            raise ValueError(f"goal.{section_name}.environment is required for provider override")
        environment["env_provider"] = provider_id
    return result


def _load_rendered_goal_composition(
    path: Path,
    *,
    label: str | None = None,
    env_provider: str | None = None,
) -> ComposedDocument:
    composition = load_composed_mapping(path, cycle_label="goal")
    _validate_reward_catalog_source_ownership(composition.sources)
    return ComposedDocument(
        document=render_template_vars(
            _goal_with_environment_provider(composition.document, env_provider),
            path=path,
            label=label or f"goal file {path}",
            deferred_fields_by_path=GOAL_DEFERRED_TEMPLATE_FIELDS,
        ),
        sources=composition.sources,
    )


def _validate_reward_catalog_source_ownership(sources: Sequence[Path]) -> None:
    program_owner: Path | None = None
    default_owner: Path | None = None
    definition_owners: dict[str, Path] = {}
    for source in sources:
        payload = load_mapping_document(source, label=f"goal source {source}")
        catalog = payload.get("reward_shapes")
        if not isinstance(catalog, Mapping):
            continue
        if "program_kind" in catalog:
            if program_owner is not None:
                raise ValueError(
                    "reward_shapes.program_kind cannot be shadowed across goal sources: "
                    f"{program_owner} and {source}"
                )
            program_owner = source
        if "default" in catalog:
            if default_owner is not None:
                raise ValueError(
                    "reward_shapes.default cannot be shadowed across goal sources: "
                    f"{default_owner} and {source}"
                )
            default_owner = source
        definitions = catalog.get("definitions")
        if not isinstance(definitions, Mapping):
            continue
        for raw_key in definitions:
            key = str(raw_key)
            previous = definition_owners.get(key)
            if previous is not None:
                raise ValueError(
                    f"reward shape {key!r} cannot be redefined across goal sources: "
                    f"{previous} and {source}"
                )
            definition_owners[key] = source


def _reject_active_specs_path(path: Path) -> None:
    if "specs" in path.parts:
        raise ValueError(f"{path} is under removed active specs/ layout; use recipes/ instead")


def _materialize_goal_owned_fields(
    materialized: dict[str, Any],
    *,
    path: Path | None = None,
    goal_composition: ComposedDocument | None = None,
) -> Mapping[str, Any] | None:
    if goal_composition is None:
        return None
    goal_document = goal_composition.document
    materialized["goal"] = copy.deepcopy(dict(goal_document))
    return goal_document


def _materialize_goal_train_environment(
    materialized: dict[str, Any],
    goal_document: Mapping[str, Any] | None,
) -> None:
    if goal_document is None:
        return
    goal_environment = _document_train_environment(goal_document)
    if not isinstance(goal_environment, Mapping):
        return
    train = _normalized_train_section(materialized.get("train"))
    train["environment"] = deep_merge(
        copy.deepcopy(dict(goal_environment)),
        train.get("environment") if isinstance(train.get("environment"), Mapping) else {},
    )
    materialized["train"] = train


def _materialize_goal_queue_defaults(
    materialized: dict[str, Any],
    goal_document: Mapping[str, Any] | None,
    *,
    path: Path | None = None,
) -> None:
    if goal_document is None:
        return
    for key in ("campaign_id",):
        if key in materialized:
            continue
        value = goal_document.get(key)
        if isinstance(value, str) and value.strip():
            materialized[key] = value
    if "tags" not in materialized:
        tags = goal_document.get("tags")
        if isinstance(tags, Sequence) and not isinstance(tags, str | bytes):
            materialized["tags"] = list(tags)
            if path is not None:
                tag_document = {
                    "tags": materialized["tags"],
                    "recipe_id": materialized.get("recipe_id"),
                    "train": goal_document.get("train"),
                }
                materialized["tags"] = render_template_vars(
                    tag_document,
                    path=path,
                    label=f"goal tags for recipe file {path}",
                )["tags"]


def materialize_train_recipe_document(
    document: Mapping[str, Any],
    *,
    path: Path | None = None,
    goal_composition: ComposedDocument | None = None,
) -> dict[str, Any]:
    materialized = copy.deepcopy(dict(document))
    source_sections = [key for key in TRAIN_CONFIG_SECTION_KEYS if key in materialized]
    if isinstance(materialized.get("train_config"), Mapping):
        if source_sections:
            raise ValueError(
                "recipe cannot mix compiled train_config with source section(s): "
                + ", ".join(source_sections)
            )
        return materialized
    normalized_train = _normalized_train_section(materialized.get("train"))
    if normalized_train:
        materialized["train"] = normalized_train
    goal_document = _materialize_goal_owned_fields(
        materialized,
        path=path,
        goal_composition=goal_composition,
    )
    _materialize_goal_queue_defaults(materialized, goal_document, path=path)
    _materialize_goal_train_environment(materialized, goal_document)
    train_config = _merge_train_config_sections(materialized, goal_document=goal_document)
    if train_config:
        from gradlab.training_backend import accepts_first_training_success

        if accepts_first_training_success(train_config):
            train_config["checkpoint_eval_backend"] = "none"
        if train_config.get("checkpoint_eval_backend") == "none":
            train_config["stop_on_acceptance"] = False
    if train_config:
        materialized["train_config"] = train_config
    return materialized


def _recipe_source_metadata(sources: Sequence[Path]) -> list[dict[str, str]]:
    return [
        {
            "path": str(source),
            "sha256": file_sha256(source),
        }
        for source in sources
    ]


def assert_no_template_vars(value: Any, *, label: str = "document") -> None:
    if isinstance(value, Mapping):
        if TEMPLATE_VARS_KEY in value:
            raise ValueError(f"{label} still contains {TEMPLATE_VARS_KEY}; render templates first")
        for key, nested in value.items():
            assert_no_template_vars(nested, label=f"{label}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for index, nested in enumerate(value):
            assert_no_template_vars(nested, label=f"{label}[{index}]")


def validate_source_recipe_shape(
    document: Mapping[str, Any],
    *,
    label: str,
    preset: bool = False,
) -> None:
    allowed_fields = SOURCE_PRESET_FIELDS if preset else SOURCE_RECIPE_FIELDS
    retired = sorted(
        set(document) & {"environment", "reward", "train_config", "group_id", "batch_id"}
    )
    if retired:
        raise ValueError(
            f"{label} uses compiled or retired source field(s): {', '.join(retired)}; "
            "author recipes with train.backend and logging"
        )
    unknown = sorted(str(key) for key in set(document) - allowed_fields)
    if unknown:
        kind = "recipe preset" if preset else "recipe"
        raise ValueError(
            f"{label} uses goal-owned or unsupported {kind} field(s): {', '.join(unknown)}"
        )
    train = document.get("train")
    if train is not None:
        if not isinstance(train, Mapping):
            raise ValueError(f"{label}.train must be an object")
        if "policy" in train:
            raise ValueError(
                f"{label}.train.policy is retired; use train.backend with an explicit id and config"
            )
        allowed = TRAIN_NESTED_SECTION_KEYS | COMMON_TRAIN_CONFIG_KEYS
        unexpected = sorted(set(train) - allowed)
        if unexpected:
            raise ValueError(
                f"{label}.train uses unsupported flat field(s): {', '.join(unexpected)}; "
                "put common fields directly under train and backend options under "
                "train.backend.config"
            )
    if not preset:
        recipe_id = train_recipe_id(document)
        if not recipe_id:
            raise ValueError(f"{label}.recipe_id is required")
        if recipe_id == "base":
            raise ValueError(f"{label}.recipe_id=base is unsupported; use an explicit recipe id")
        description = document.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"{label}.description is required")


def load_recipe_source_document(path: Path) -> ComposedDocument:
    _reject_active_specs_path(path)
    validate_source_recipe_shape(
        load_mapping_document(path, label=f"recipe file {path}"),
        label=f"recipe file {path}",
    )
    composed = load_composed_mapping(
        path,
        cycle_label="recipe",
    )
    validate_source_recipe_shape(
        composed.document,
        label=f"composed recipe file {path}",
    )
    resolved_path = path.resolve()
    experiments_root = next(
        (parent for parent in resolved_path.parents if parent.name == "experiments"),
        None,
    )
    if experiments_root is None:
        raise ValueError(f"recipe {path} is not under an experiments tree")
    presets_root = experiments_root / "recipes" / "_presets"
    if not resolved_path.is_relative_to(presets_root):
        for source in composed.sources[:-1]:
            if not source.resolve().is_relative_to(presets_root):
                raise ValueError(
                    f"launchable recipe {path} may inherit only shared presets under "
                    f"experiments/recipes/_presets, got {source}"
                )
    for source in composed.sources[:-1]:
        validate_source_recipe_shape(
            load_mapping_document(source, label=f"recipe preset {source}"),
            label=f"recipe preset {source}",
            preset=True,
        )
    return composed


def compose_train_document(
    goal_path: Path,
    recipe_path: Path,
    *,
    recipe_overrides: Sequence[str] = (),
    env_provider: str | None = None,
    prepare_materialized: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    expected_recipe_dir = goal_path.resolve().parent / "recipes"
    resolved_recipe_path = recipe_path.resolve()
    if resolved_recipe_path.parent != expected_recipe_dir:
        raise ValueError(
            f"recipe {recipe_path} does not belong to goal {goal_path}; "
            f"select a launchable recipe directly under {expected_recipe_dir}"
        )
    goal_composition = _load_rendered_goal_composition(
        goal_path,
        env_provider=env_provider,
    )
    authored_goal_document = goal_composition.document
    if goal_composition.sources:
        validate_goal_contract_document(
            goal_composition.document,
            goal_composition.sources[-1],
            Path(".").resolve(),
        )
    recipe_composition = load_recipe_source_document(recipe_path)
    recipe_override_list = [str(item).strip() for item in recipe_overrides if str(item).strip()]
    reward_definition_overrides = [
        item
        for item in recipe_override_list
        if item.split("=", 1)[0].strip().startswith(REWARD_DEFINITION_OVERRIDE_PREFIX)
    ]
    non_reward_definition_overrides = [
        item for item in recipe_override_list if item not in reward_definition_overrides
    ]
    source_overrides, policy_goal_overrides, effective_environment_overrides = (
        _partition_policy_environment_overrides(
            non_reward_definition_overrides,
            goal_document=goal_composition.document,
            label=f"recipe overrides for {recipe_path}",
        )
    )
    if policy_goal_overrides:
        goal_composition = ComposedDocument(
            document=apply_dotlist_overrides(
                goal_composition.document,
                policy_goal_overrides,
                label=f"policy environment overrides for {goal_path}",
            ),
            sources=goal_composition.sources,
        )
        validate_goal_contract_document(
            goal_composition.document,
            goal_composition.sources[-1] if goal_composition.sources else goal_path,
            Path(".").resolve(),
        )
    source_document = apply_dotlist_overrides(
        recipe_composition.document,
        source_overrides,
        label=f"recipe overrides for {recipe_path}",
    )
    selector_value = source_document.pop("reward_shape", None)
    selector = None
    if selector_value is not None:
        if not isinstance(selector_value, str) or not selector_value.strip():
            raise ValueError("reward_shape must be a non-empty string")
        selector = selector_value.strip()
    if reward_definition_overrides:
        catalog = goal_composition.document.get("reward_shapes")
        if not isinstance(catalog, Mapping):
            raise ValueError(
                f"goal file {goal_path} does not define reward_shapes; "
                "reward definition overrides are unsupported"
            )
        selected_key = selector or str(catalog.get("default") or "")
        for item in reward_definition_overrides:
            path = item.split("=", 1)[0].strip()
            parts = path.split(".")
            if len(parts) != 4 or parts[3] not in MARIO_REWARD_FIELD_SET:
                raise ValueError(
                    "reward definition overrides must target exactly "
                    "reward_shapes.definitions.<selected-shape>.<reward-field>: "
                    f"{path}"
                )
            if parts[2] != selected_key:
                raise ValueError(
                    "reward definition override targets an unselected shape; "
                    f"selected={selected_key!r} override={path!r}"
                )
        goal_composition = ComposedDocument(
            document=apply_dotlist_overrides(
                goal_composition.document,
                reward_definition_overrides,
                label=f"reward definition overrides for {goal_path}",
            ),
            sources=goal_composition.sources,
        )
    selected_reward = select_goal_reward_shape(
        goal_composition.document,
        selector,
        label=f"goal file {goal_path}",
    )
    if selected_reward is not None and reward_definition_overrides:
        selected_reward = replace(selected_reward, is_default=False)
    if selected_reward is not None:
        train_section = source_document.get("train")
        source_environment = (
            train_section.get("environment") if isinstance(train_section, Mapping) else None
        )
        source_task = (
            source_environment.get("task") if isinstance(source_environment, Mapping) else None
        )
        if isinstance(source_task, Mapping) and "reward" in source_task:
            raise ValueError(
                "catalog goals reject raw reward overrides and recipe-authored task.reward; "
                "select a named reward_shape"
            )
        raw_reward_override = next(
            (
                item
                for item in recipe_override_list
                if item.split("=", 1)[0].strip().startswith("train.environment.task.reward")
                or item.split("=", 1)[0].strip().startswith("train.task.reward")
                or (
                    item.split("=", 1)[0].strip().startswith("reward_shapes")
                    and item not in reward_definition_overrides
                )
            ),
            None,
        )
        if raw_reward_override is not None:
            raise ValueError(
                "catalog goals reject raw reward overrides; select a named reward_shape instead: "
                f"{raw_reward_override}"
            )
        goal_composition = ComposedDocument(
            document=selected_reward.goal,
            sources=goal_composition.sources,
        )
    recipe_id = train_recipe_id(source_document)
    goal_context = template_context_from_path(goal_path, goal_composition.document)
    document = render_template_vars(
        source_document,
        path=goal_path,
        label=f"recipe file {recipe_path} for goal file {goal_path}",
        extra_context={
            **goal_context,
            "recipe_id": recipe_id,
            "recipe_slug": recipe_id,
            "slug": recipe_id,
        },
        deferred_fields_by_path=RECIPE_DEFERRED_TEMPLATE_FIELDS,
    )
    sources = [*goal_composition.sources, *recipe_composition.sources]
    sources = list(dict.fromkeys(sources))
    document = materialize_train_recipe_document(
        document,
        path=goal_path,
        goal_composition=goal_composition,
    )
    document["train_config"]["goal_contract_sha256"] = goal_contract_sha256(authored_goal_document)
    document["train_config"]["effective_goal_contract_sha256"] = goal_contract_sha256(
        goal_composition.document
    )
    if selected_reward is not None:
        document["train_config"].update(
            {
                "reward_program_kind": selected_reward.program_kind,
                "reward_program_revision": selected_reward.program_revision,
                "reward_shape": selected_reward.key,
                "reward_shape_sha256": selected_reward.semantic_sha256,
                "reward_shape_is_default": selected_reward.is_default,
            }
        )
    document = attach_environment_identity(document)
    if recipe_override_list:
        document["recipe_overrides"] = recipe_override_list
    if effective_environment_overrides:
        document["effective_recipe_overrides"] = effective_environment_overrides
    if recipe_path.suffix.lower() in YAML_EXTENSIONS or len(sources) > 1:
        document["_composition"] = {
            "goal_root_path": str(goal_path.resolve()),
            "recipe_root_path": str(recipe_path.resolve()),
            "source_files": _recipe_source_metadata(sources),
        }
    if prepare_materialized is not None:
        prepare_materialized(document)
    label = f"goal file {goal_path} with recipe file {recipe_path}"
    validate_materialized_train_recipe(document, label=label)
    assert_no_template_vars(document, label=label)
    assert_no_secrets(document, label=label)
    validate_launch_event_config(
        document["train_config"],
        label=f"{label} train_config",
    )
    return document


def prepare_checkpoint_eval_mode(
    document: dict[str, Any],
    *,
    checkpoint_eval_backend: str | None,
) -> None:
    """Materialize the execution mode before recipe validation and hashing."""

    config = dict(document["train_config"])
    mode = str(
        checkpoint_eval_backend
        if checkpoint_eval_backend is not None
        else config.get("checkpoint_eval_backend") or "modal"
    ).strip()
    if mode not in {"modal", "none"}:
        raise ValueError(f"unsupported checkpoint eval backend: {mode}")
    config["checkpoint_eval_backend"] = mode
    if mode == "none":
        config["stop_on_acceptance"] = False
    document["train_config"] = config


def _git_text(args: Sequence[str], *, cwd: Path = Path(".")) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except OSError, subprocess.CalledProcessError:
        return None
    return result.stdout.strip() or None


def repo_git_commit(cwd: Path = Path(".")) -> str | None:
    return _git_text(("rev-parse", "HEAD"), cwd=cwd)


def recipe_slug(document: Mapping[str, Any]) -> str:
    return train_recipe_id(document)


def validate_launch_event_config(
    train_config: Mapping[str, Any], *, label: str = "train_config"
) -> None:
    task = train_config.get("task")
    if isinstance(task, Mapping):
        validate_task_config(task, label=f"{label}.task")


def recipe_tags(document: Mapping[str, Any]) -> list[str]:
    tags = []
    seen: set[str] = set()
    for raw_tag in document.get("tags") or []:
        tag = str(raw_tag).strip()
        if not tag or tag in seen:
            continue
        tags.append(tag)
        seen.add(tag)
    return tags


def load_goal_contract_document(path: Path, *, label: str | None = None) -> dict[str, Any]:
    return _load_rendered_goal_composition(path, label=label).document


def load_goal_contract(
    path: Path,
    repo_root: Path | None = None,
    *,
    validate: bool = True,
) -> dict[str, Any]:
    """Return a composed goal contract, validating it by default."""

    resolved_root = (repo_root or Path(".")).resolve()
    resolved_path = path.resolve()
    try:
        display_path = resolved_path.relative_to(resolved_root)
    except ValueError:
        display_path = resolved_path
    document = load_goal_contract_document(
        resolved_path,
        label=f"goal file {display_path}",
    )
    if validate:
        validate_goal_contract_document(document, resolved_path, resolved_root)
    return document


def validate_goal_contract(path: Path, repo_root: Path | None = None) -> None:
    load_goal_contract(path, repo_root)
