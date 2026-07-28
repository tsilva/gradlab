from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Annotated, Any, Callable, Literal

from pydantic import Field, StringConstraints, model_validator

from gradlab.boundary_schema import BoundaryModel, validate_boundary
from gradlab.file_utils import file_sha256 as sha256_file
from gradlab.json_utils import canonical_json_line_bytes


RECIPE_DOCUMENT_TYPE = "gradlab.recipe"
RECIPE_FORMAT_VERSION = 1
MODEL_DOCUMENT_TYPE = "gradlab.model"
MODEL_FORMAT_VERSION = 2

RECIPE_FILENAME = "recipe.json"
MODEL_FILENAME = "model.json"
CHECKPOINT_FILENAME = "model.zip"

_SECRET_FRAGMENTS = (
    "api_key",
    "access_key",
    "secret",
    "token",
    "password",
    "credential",
    "database_url",
)

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[
    str,
    StringConstraints(strip_whitespace=True, to_lower=True, pattern=r"^[0-9a-f]{64}$"),
]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]


class _EvaluationDocument(BoundaryModel):
    environment: dict[str, Any]
    action_sampling: Any = None
    episodes: Any = None
    n_envs: Any = None
    max_steps: Any = None
    seed: Any = None
    seed_protocol: Any = None
    protocol_version: Any = None
    deterministic: Any = None
    acceptance: Any = None
    manifest: Any = None
    evidence_policy: Any = None
    asset: Any = None


class _PlaybackDocument(BoundaryModel):
    environment: dict[str, Any]
    seed: Any = None
    asset: Any = None


class _RecipeValueDocument(BoundaryModel):
    goal: dict[str, Any]
    goal_variant: Any = None
    recipe_id: NonEmptyText
    description: NonEmptyText
    train: dict[str, Any]
    train_config: dict[str, Any]
    schema_version: Any = None
    tags: Any = None
    campaign_id: Any = None
    seeds: Any = None
    recipe_overrides: Any = None
    effective_recipe_overrides: Any = None
    environment: Any = None
    environment_hash: Any = None
    policy_environment_hash: Any = None
    evaluation_environment_hash: Any = None
    value_contract: Any = None
    eval: _EvaluationDocument | None = None
    playback: _PlaybackDocument | None = None

    @model_validator(mode="after")
    def validate_portable_contract(self) -> "_RecipeValueDocument":
        if self.eval is None and self.playback is None:
            raise ValueError("must define eval or playback")
        if self.eval is not None and self.playback is not None:
            raise ValueError("cannot define both eval and playback")
        if self.goal_variant is not None:
            if not isinstance(self.goal_variant, Mapping):
                raise ValueError("goal_variant must be a mapping")
            from gradlab.goal_variants import validate_goal_variant_descriptor

            descriptor = validate_goal_variant_descriptor(self.goal_variant)
            if (
                descriptor["goal_contract_sha256"] != self.train_config.get("goal_contract_sha256")
                or descriptor["effective_goal_contract_sha256"]
                != self.train_config.get("effective_goal_contract_sha256")
                or descriptor["variant_id"] != self.train_config.get("goal_variant_id")
            ):
                raise ValueError("goal_variant disagrees with train_config")
        return self


class _RecipeProvenanceDocument(BoundaryModel):
    source_commit: Any = None
    source_distribution: Any = None
    source_files: Any = None
    runtime: Any = None
    asset: Any = None


class _RecipeDocument(BoundaryModel):
    document_type: Literal[RECIPE_DOCUMENT_TYPE]
    format_version: Literal[RECIPE_FORMAT_VERSION]
    recipe: _RecipeValueDocument
    provenance: _RecipeProvenanceDocument


class _CheckpointDocument(BoundaryModel):
    filename: Literal[CHECKPOINT_FILENAME]
    sha256: Sha256
    size_bytes: PositiveInt
    kind: Literal["checkpoint", "best", "final", "interrupted"]
    step: NonNegativeInt | None
    algorithm_id: NonEmptyText
    model_class: NonEmptyText


class _RecipeBindingDocument(BoundaryModel):
    filename: Literal[RECIPE_FILENAME]
    document_type: Literal[RECIPE_DOCUMENT_TYPE]
    format_version: Literal[RECIPE_FORMAT_VERSION]
    sha256: Sha256
    size_bytes: PositiveInt


class _PolicyDocument(BoundaryModel):
    algorithm_id: NonEmptyText
    model_class: NonEmptyText
    training_backend_id: NonEmptyText
    training_backend_config_hash: Sha256


class _ModelProvenanceDocument(BoundaryModel):
    kind: Any = None
    run_name: Any = None
    run_description: Any = None
    wandb_run_id: Any = None
    wandb_project: Any = None
    wandb_run_path: Any = None
    campaign_id: Any = None
    game_family: Any = None
    goal_slug: Any = None
    goal_sha256: Any = None
    goal_contract_sha256: Any = None
    effective_goal_contract_sha256: Any = None
    goal_variant_id: Any = None
    goal_variant_label: Any = None
    goal_variant_source_relation: Any = None
    goal_variant_descriptor_sha256: Any = None
    reward_program_kind: Any = None
    reward_program_revision: Any = None
    reward_shape: Any = None
    reward_shape_sha256: Any = None
    reward_shape_is_default: Any = None
    recipe_slug: Any = None
    recipe_sha256: Any = None
    runtime_image_ref: Any = None
    seed: Any = None
    repo_git_commit: Any = None
    training_metadata: Any = None
    attempt_id: Any = None
    compute_target: Any = None
    dstack_task: Any = None
    search_algorithm_id: Any = None
    state_archive_preflight_sha256: Any = None
    state_archive_summary: Any = None


_MODEL_PROVENANCE_FIELDS = frozenset(_ModelProvenanceDocument.model_fields)


class _ModelDocument(BoundaryModel):
    document_type: Literal[MODEL_DOCUMENT_TYPE]
    format_version: Literal[MODEL_FORMAT_VERSION]
    checkpoint: _CheckpointDocument
    recipe: _RecipeBindingDocument
    policy: _PolicyDocument
    provenance: _ModelProvenanceDocument


_OPERATIONAL_TRAIN_FIELDS = frozenset(
    {
        "attempt_id",
        "campaign_id",
        "compute_target",
        "dstack_task",
        "game_family",
        "goal_path",
        "recipe_composition",
        "recipe_json_path",
        "recipe_path",
        "run_description",
        "run_name",
        "runs_dir",
        "runtime_build_source_sha",
        "runtime_image_ref",
        "runtime_input_sha256",
        "source_sha",
        "train_config_json",
        "wandb",
        "wandb_display_name",
        "wandb_entity",
        "wandb_group",
        "wandb_mode",
        "wandb_project",
        "wandb_run_id",
        "wandb_tags",
    }
)


class PolicyDocumentError(ValueError):
    """Base error for a policy document that cannot be interpreted safely."""


class UnsupportedPolicyDocumentVersion(PolicyDocumentError):
    def __init__(
        self,
        *,
        source: str,
        document_type: object,
        format_version: object,
        supported_versions: Sequence[int],
    ) -> None:
        supported = ", ".join(str(item) for item in supported_versions) or "none"
        super().__init__(
            f"Unsupported {document_type!r} format_version {format_version!r} in {source}. "
            f"This gradlab version supports: [{supported}]. Upgrade gradlab or install an "
            "explicit compatibility handler."
        )


@dataclass(frozen=True)
class PolicyBundle:
    checkpoint_path: Path
    model_path: Path
    recipe_path: Path
    model: dict[str, Any]
    recipe: dict[str, Any]
    source: str
    revision: str | None = None

    @property
    def checkpoint_sha256(self) -> str:
        return str(self.model["checkpoint"]["sha256"])

    @property
    def recipe_sha256(self) -> str:
        return str(self.model["recipe"]["sha256"])


def canonical_json_bytes(value: object) -> bytes:
    _assert_finite_json(value)
    try:
        return canonical_json_line_bytes(value)
    except (TypeError, ValueError) as exc:
        raise PolicyDocumentError(f"document is not canonical JSON: {exc}") from exc


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def write_canonical_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(dict(value)))
    return path


def _assert_finite_json(value: object, *, label: str = "document") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise PolicyDocumentError(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _assert_finite_json(nested, label=f"{label}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for index, nested in enumerate(value):
            _assert_finite_json(nested, label=f"{label}[{index}]")


def _required_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PolicyDocumentError(f"{label} must be an object")
    return value


def _required_text(value: object, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise PolicyDocumentError(f"{label} must be a non-empty string")
    return text


def _required_sha256(value: object, *, label: str) -> str:
    text = str(value or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise PolicyDocumentError(f"{label} must be a SHA-256 hex digest")
    return text


def _reject_unknown(mapping: Mapping[str, Any], allowed: frozenset[str], *, label: str) -> None:
    unknown = sorted(str(key) for key in mapping if key not in allowed)
    if unknown:
        raise PolicyDocumentError(f"{label} has unknown field(s): {', '.join(unknown)}")


def _assert_portable(value: object, *, label: str = "recipe") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            nested_label = f"{label}.{key}"
            if any(fragment in key_text for fragment in _SECRET_FRAGMENTS):
                raise PolicyDocumentError(f"{nested_label} is secret-like and is not portable")
            if key == "defaults":
                raise PolicyDocumentError(f"{nested_label} contains uncomposed defaults")
            _assert_portable(nested, label=nested_label)
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for index, nested in enumerate(value):
            _assert_portable(nested, label=f"{label}[{index}]")
        return
    if not isinstance(value, str) or not value:
        return
    if "${" in value or re.search(r"\{[A-Za-z_][A-Za-z0-9_]*\}", value):
        raise PolicyDocumentError(f"{label} contains unresolved interpolation: {value!r}")
    if value.startswith(("file://", "s3://", "r2://")):
        raise PolicyDocumentError(f"{label} contains a private or local URI")
    if "://" in value:
        raise PolicyDocumentError(f"{label} contains a URL, which is not portable")
    if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
        raise PolicyDocumentError(f"{label} contains an absolute local path")


def preflight_document(
    value: object,
    *,
    source: str,
    expected_type: str,
    handlers: Mapping[int, Callable[[Mapping[str, Any], str], dict[str, Any]]],
) -> dict[str, Any]:
    supported = sorted(handlers)
    document = _required_mapping(value, label=source)
    document_type = document.get("document_type")
    format_version = document.get("format_version")
    if document_type != expected_type:
        raise PolicyDocumentError(
            f"{source} document_type must be {expected_type!r}, got {document_type!r}; "
            f"supported versions are {supported}. Upgrade gradlab or use an explicit "
            "compatibility handler."
        )
    if isinstance(format_version, bool) or not isinstance(format_version, int):
        raise PolicyDocumentError(
            f"{source} {expected_type!r} format_version must be an integer, got "
            f"{format_version!r}; supported versions are {supported}. Upgrade gradlab or "
            "use an explicit compatibility handler."
        )
    handler = handlers.get(format_version)
    if handler is None:
        raise UnsupportedPolicyDocumentVersion(
            source=source,
            document_type=document_type,
            format_version=format_version,
            supported_versions=sorted(handlers),
        )
    try:
        return handler(document, source)
    except UnsupportedPolicyDocumentVersion:
        raise
    except PolicyDocumentError as exc:
        raise PolicyDocumentError(
            f"Invalid {expected_type} format_version {format_version} in {source}; "
            f"supported versions are {supported}: {exc}. Upgrade gradlab or use an "
            "explicit compatibility handler."
        ) from exc


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PolicyDocumentError(f"Could not read JSON document {path}: {exc}") from exc
    return dict(_required_mapping(value, label=str(path)))


def _validate_recipe_v1(document: Mapping[str, Any], source: str) -> dict[str, Any]:
    validate_boundary(
        _RecipeDocument,
        document,
        label=source,
        error_type=PolicyDocumentError,
    )
    recipe = document["recipe"]
    provenance = document["provenance"]
    assert isinstance(recipe, Mapping)
    assert isinstance(provenance, Mapping)
    goal = recipe["goal"]
    train_config = recipe["train_config"]
    assert isinstance(goal, Mapping)
    assert isinstance(train_config, Mapping)
    evaluation_value = recipe.get("eval")
    playback_value = recipe.get("playback")
    evaluation = evaluation_value if isinstance(evaluation_value, Mapping) else None
    playback = playback_value if isinstance(playback_value, Mapping) else None
    from gradlab.env_identity import validate_task_config
    from gradlab.env_registry import resolve_env_provider
    from gradlab.goal_schema import validate_goal_document_shape
    from gradlab.train_config import (
        TRAIN_CONFIG_FIELDS,
        env_config_allowed_keys,
        validate_and_normalize_train_config,
        validate_train_config_fields,
    )

    _reject_unknown(
        train_config,
        frozenset(field.dest for field in TRAIN_CONFIG_FIELDS),
        label=f"{source}.recipe.train_config",
    )
    try:
        backend_value = train_config.get("training_backend")
        backend = backend_value if isinstance(backend_value, Mapping) else {}
        backend_id = str(backend.get("id") or "").strip()
        from gradlab.policy_registry import (
            BACKEND_PROVENANCE_SPECS,
            TRAINING_BACKEND_SPECS,
        )

        if backend_id in TRAINING_BACKEND_SPECS:
            validate_and_normalize_train_config(
                train_config,
                label=f"{source}.recipe.train_config",
            )
        elif backend_id in BACKEND_PROVENANCE_SPECS:
            # Archived, non-launchable backends still get strict portable
            # field/environment validation without importing or inventing a
            # local learner implementation.
            validate_train_config_fields(
                train_config,
                label=f"{source}.recipe.train_config",
                required_keys=("training_backend",),
            )
            backend_config = backend.get("config")
            if not isinstance(backend_config, Mapping):
                raise ValueError(
                    f"{source}.recipe.train_config.training_backend.config must be an object"
                )
        else:
            raise ValueError(f"unknown training backend {backend_id!r}")
        validate_goal_document_shape(goal, label=f"{source}.recipe.goal")
    except ValueError as exc:
        raise PolicyDocumentError(str(exc)) from exc
    training_only = str(train_config.get("checkpoint_eval_backend") or "") == "none"
    if training_only and evaluation is not None:
        raise PolicyDocumentError(f"{source}.recipe training-only contract cannot define eval")
    if not training_only and evaluation is None:
        raise PolicyDocumentError(f"{source}.recipe evaluated contract must define eval")
    if evaluation is not None:
        portable_environment = evaluation.get("environment")
        portable_environment_label = "evaluation"
    else:
        assert playback is not None
        portable_environment = playback.get("environment")
        portable_environment_label = "playback"
    for label, environment in (
        ("training", train_config),
        (portable_environment_label, portable_environment),
    ):
        environment = _required_mapping(environment, label=f"{source} {label} environment")
        if label in {"evaluation", "playback"}:
            _reject_unknown(
                environment,
                env_config_allowed_keys(),
                label=f"{source} {label} environment",
            )
            try:
                validate_train_config_fields(
                    environment,
                    label=f"{source} {label} environment",
                )
            except ValueError as exc:
                raise PolicyDocumentError(str(exc)) from exc
        _required_text(environment.get("env_provider"), label=f"{source} {label} provider")
        _required_text(environment.get("game"), label=f"{source} {label} game")
        task = _required_mapping(environment.get("task"), label=f"{source} {label} task")
        try:
            validate_task_config(task, label=f"{source} {label} task")
        except ValueError as exc:
            raise PolicyDocumentError(str(exc)) from exc
    _required_mapping(train_config.get("training_backend"), label=f"{source} training backend")
    _required_mapping(goal.get("train"), label=f"{source}.recipe.goal.train")
    portable_seed = (evaluation or playback or {}).get("seed")
    if not isinstance(portable_seed, int) or isinstance(portable_seed, bool):
        raise PolicyDocumentError(f"{source} portable environment seed must be an integer")
    backend_value = train_config.get("training_backend")
    backend = backend_value if isinstance(backend_value, Mapping) else {}
    from gradlab.policy_registry import (
        backend_provenance_algorithm,
        default_action_selection_mode,
    )

    expected_action_sampling = default_action_selection_mode(
        backend_provenance_algorithm(str(backend.get("id") or ""))
    )
    if evaluation is not None and evaluation.get("action_sampling") not in {
        expected_action_sampling,
        # Version-1 recipes used "stochastic" as a universal boolean-shaped
        # marker. Readers canonically interpret it using backend provenance.
        "stochastic",
    }:
        raise PolicyDocumentError(
            f"{source}.recipe.eval.action_sampling must be {expected_action_sampling!r}"
        )
    if evaluation is not None and evaluation.get("deterministic", False) is not False:
        raise PolicyDocumentError(f"{source}.recipe.eval.deterministic must be false")
    if evaluation is not None:
        _required_text(evaluation.get("seed_protocol"), label=f"{source} eval seed_protocol")
    if evaluation is not None and (
        not isinstance(evaluation.get("episodes"), int) or int(evaluation["episodes"]) <= 0
    ):
        raise PolicyDocumentError(f"{source}.recipe.eval.episodes must be positive")
    if evaluation is not None and "manifest" in evaluation:
        from gradlab.checkpoint_acceptance import manifest_index

        try:
            manifest_index(evaluation)
        except ValueError as exc:
            raise PolicyDocumentError(f"{source}.recipe.eval manifest is invalid: {exc}") from exc
    if playback is not None and "asset" in playback:
        playback_environment = _required_mapping(
            playback.get("environment"),
            label=f"{source}.recipe.playback.environment",
        )
        provider = _required_text(
            playback_environment.get("env_provider"),
            label=f"{source}.recipe.playback.environment.env_provider",
        )
        requires_asset = resolve_env_provider(provider).requires_external_rom_asset
        asset = playback.get("asset")
        if requires_asset:
            if not isinstance(asset, Mapping):
                raise PolicyDocumentError(
                    f"{source}.recipe.playback.asset must contain portable ROM identity"
                )
            _required_sha256(
                asset.get("sha256"),
                label=f"{source}.recipe.playback.asset.sha256",
            )
            _required_text(
                asset.get("provider_rom_identity"),
                label=f"{source}.recipe.playback.asset.provider_rom_identity",
            )
        elif asset is not None:
            raise PolicyDocumentError(
                f"{source}.recipe.playback.asset must be null for a ROM-free provider"
            )
        if provenance.get("asset") != asset:
            raise PolicyDocumentError(
                f"{source}.recipe.playback.asset disagrees with provenance.asset"
            )
    source_commit = provenance.get("source_commit")
    source_distribution = provenance.get("source_distribution")
    if source_commit is not None:
        source_commit = _required_text(source_commit, label=f"{source} source_commit")
        if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
            raise PolicyDocumentError(f"{source} source_commit must be a full lowercase Git SHA")
    if source_distribution is not None:
        source_distribution = _required_mapping(
            source_distribution,
            label=f"{source}.provenance.source_distribution",
        )
        _reject_unknown(
            source_distribution,
            frozenset({"name", "version"}),
            label=f"{source}.provenance.source_distribution",
        )
        _required_text(
            source_distribution.get("name"),
            label=f"{source}.provenance.source_distribution.name",
        )
        _required_text(
            source_distribution.get("version"),
            label=f"{source}.provenance.source_distribution.version",
        )
    if source_commit is None and source_distribution is None:
        raise PolicyDocumentError(
            f"{source}.provenance must define source_commit or source_distribution"
        )
    runtime = _required_mapping(provenance.get("runtime"), label=f"{source}.provenance.runtime")
    _reject_unknown(runtime, frozenset({"image_ref", "packages"}), label=f"{source}.runtime")
    image_ref = runtime.get("image_ref")
    packages = runtime.get("packages")
    if image_ref is not None:
        image_ref = _required_text(image_ref, label=f"{source} runtime image_ref")
        if not re.fullmatch(r"docker:[^\s]+@sha256:[0-9a-f]{64}", image_ref):
            raise PolicyDocumentError(f"{source} runtime image_ref must be an immutable digest")
    if packages is not None:
        package_mapping = (
            isinstance(packages, Mapping)
            and bool(packages)
            and all(
                isinstance(name, str)
                and bool(name.strip())
                and isinstance(version, str)
                and bool(version.strip())
                for name, version in packages.items()
            )
        )
        package_list = (
            isinstance(packages, Sequence)
            and not isinstance(packages, str | bytes)
            and bool(packages)
            and all(isinstance(item, str) and "==" in item for item in packages)
        )
        if not package_mapping and not package_list:
            raise PolicyDocumentError(
                f"{source} runtime packages must be exact versions as a mapping or "
                "a non-empty list of name==version strings"
            )
    if image_ref is None and packages is None:
        raise PolicyDocumentError(
            f"{source} runtime must define an immutable image_ref or exact packages"
        )
    _assert_portable(recipe, label=f"{source}.recipe")
    _assert_portable(provenance, label=f"{source}.provenance")
    _assert_finite_json(document, label=source)
    return deepcopy(dict(document))


def load_recipe_document(path: Path) -> dict[str, Any]:
    value = load_json_object(path)
    return preflight_document(
        value,
        source=str(path),
        expected_type=RECIPE_DOCUMENT_TYPE,
        handlers={RECIPE_FORMAT_VERSION: _validate_recipe_v1},
    )


def validate_recipe_document(
    document: Mapping[str, Any], *, source: str = RECIPE_FILENAME
) -> dict[str, Any]:
    return preflight_document(
        document,
        source=source,
        expected_type=RECIPE_DOCUMENT_TYPE,
        handlers={RECIPE_FORMAT_VERSION: _validate_recipe_v1},
    )


def _validate_state_archive_summary(value: object, *, label: str) -> None:
    summary = _required_mapping(value, label=label)
    allowed = frozenset(
        {
            "semantic_id",
            "schema_version",
            "persistence",
            "provider_id",
            "codec_id",
            "compatibility_id",
            "entry_count",
            "blob_count",
            "blob_bytes",
            "view_ids",
            "curriculum",
        }
    )
    _reject_unknown(summary, allowed, label=label)
    if summary.get("semantic_id") != "state-archive-v1":
        raise PolicyDocumentError(f"{label}.semantic_id must be 'state-archive-v1'")
    if summary.get("schema_version") != 1:
        raise PolicyDocumentError(f"{label}.schema_version must be 1")
    if summary.get("persistence") != "durable":
        raise PolicyDocumentError(f"{label}.persistence must be 'durable'")
    for key in ("provider_id", "codec_id", "compatibility_id"):
        if not isinstance(summary.get(key), str) or not summary[key]:
            raise PolicyDocumentError(f"{label}.{key} must be a non-empty string")
    for key in ("entry_count", "blob_count", "blob_bytes"):
        item = summary.get(key)
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise PolicyDocumentError(f"{label}.{key} must be a non-negative integer")
    view_ids = summary.get("view_ids")
    if (
        not isinstance(view_ids, list)
        or any(not isinstance(item, str) or not item for item in view_ids)
        or view_ids != sorted(set(view_ids))
    ):
        raise PolicyDocumentError(
            f"{label}.view_ids must be a sorted list of unique non-empty strings"
        )
    curriculum = summary.get("curriculum")
    if curriculum is not None and not isinstance(curriculum, Mapping):
        raise PolicyDocumentError(f"{label}.curriculum must be an object")


def _validate_model(
    document: Mapping[str, Any],
    source: str,
) -> dict[str, Any]:
    validate_boundary(
        _ModelDocument,
        document,
        label=source,
        error_type=PolicyDocumentError,
    )
    provenance = document["provenance"]
    assert isinstance(provenance, Mapping)
    if "state_archive_preflight_sha256" in provenance:
        _required_sha256(
            provenance["state_archive_preflight_sha256"],
            label=f"{source}.provenance.state_archive_preflight_sha256",
        )
    if "state_archive_summary" in provenance:
        _validate_state_archive_summary(
            provenance["state_archive_summary"],
            label=f"{source}.provenance.state_archive_summary",
        )
    _assert_portable(provenance, label=f"{source}.provenance")
    _assert_finite_json(document, label=source)
    return deepcopy(dict(document))


def _validate_model_v2(document: Mapping[str, Any], source: str) -> dict[str, Any]:
    return _validate_model(document, source)


_MODEL_HANDLERS = {
    MODEL_FORMAT_VERSION: _validate_model_v2,
}


def load_model_document(path: Path) -> dict[str, Any]:
    value = load_json_object(path)
    return preflight_document(
        value,
        source=str(path),
        expected_type=MODEL_DOCUMENT_TYPE,
        handlers=_MODEL_HANDLERS,
    )


def model_document_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(".model.json")


def recipe_document_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(".recipe.json")


def _portable_source_path(path: object, *, repo_root: Path) -> str:
    candidate = Path(str(path))
    if not candidate.is_absolute():
        return candidate.as_posix()
    try:
        return candidate.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise PolicyDocumentError(
            f"recipe source path is outside the repository: {candidate}"
        ) from exc


def _resolve_recipe_templates(value: object, replacements: Mapping[str, object]) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _resolve_recipe_templates(nested, replacements)
            for key, nested in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_resolve_recipe_templates(nested, replacements) for nested in value]
    if not isinstance(value, str):
        return value
    rendered = value
    for key, replacement in replacements.items():
        rendered = rendered.replace("{" + key + "}", str(replacement))
    return rendered


def build_recipe_document(
    materialized_recipe: Mapping[str, Any],
    *,
    repo_root: Path,
    source_commit: str | None,
    source_distribution: Mapping[str, str] | None = None,
    run_description: str | None = None,
    seed: int | None = None,
    runtime_image_ref: str | None = None,
    runtime_packages: Sequence[str] | None = None,
) -> dict[str, Any]:
    recipe = deepcopy(dict(materialized_recipe))
    recipe.pop("logging", None)
    composition = recipe.pop("_composition", {})
    train_config = dict(_required_mapping(recipe.get("train_config"), label="train_config"))
    # recipe.json is the immutable policy contract consumed by evaluation. Materialize
    # backend defaults here, before the supervisor adds operational fields, so its backend
    # hash is identical to the normalized config executed by the learner.
    from gradlab.train_config import env_config_allowed_keys, validate_and_normalize_train_config

    train_config = validate_and_normalize_train_config(
        train_config,
        label="recipe.json train_config",
    )
    # Derive the recipe's environment identity from the same effective EnvConfig
    # path used by the learner. Sparse source config omits defaults such as empty
    # state vectors; hashing that sparse identity makes otherwise identical model
    # metadata fail the cross-document contract before evaluation can start.
    from gradlab.env import resolve_env_config
    from gradlab.env_config import env_config_from_mapping
    from gradlab.env_metadata import training_metadata

    effective_training_metadata = training_metadata(
        resolve_env_config(env_config_from_mapping(train_config)),
        rom_asset_manifest=train_config.get("rom_asset_manifest"),
    )
    recipe["environment"] = effective_training_metadata["environment"]
    recipe["environment_hash"] = effective_training_metadata["environment_hash"]
    from gradlab.env_identity import policy_environment_hash

    training_policy_environment_hash = policy_environment_hash(
        effective_training_metadata["env_config"]
    )
    recipe["policy_environment_hash"] = training_policy_environment_hash
    backend_value = train_config.get("training_backend")
    backend = backend_value if isinstance(backend_value, Mapping) else {}
    backend_config_value = backend.get("config")
    backend_config = backend_config_value if isinstance(backend_config_value, Mapping) else {}
    gamma_value = backend_config.get("gamma")
    discount = (
        float(gamma_value)
        if not isinstance(gamma_value, bool)
        and isinstance(gamma_value, int | float)
        and math.isfinite(float(gamma_value))
        and 0.0 <= float(gamma_value) <= 1.0
        else None
    )
    backend_id = str(backend.get("id") or "")
    from gradlab.policy_registry import (
        backend_provenance_algorithm,
        default_action_selection_mode,
    )

    algorithm_id = backend_provenance_algorithm(backend_id)
    action_sampling = default_action_selection_mode(algorithm_id)
    if algorithm_id in {"ppo", "a2c"}:
        recipe["value_contract"] = {
            "schema_version": 1,
            "policy_environment_hash": training_policy_environment_hash,
            "reward_stream": "task",
            "discount": discount,
            "action_sampling": action_sampling,
            "truncation_bootstrap": "terminal-value",
        }
    else:
        recipe.pop("value_contract", None)
    from gradlab.checkpoint_acceptance import (
        CheckpointEvalContractCompiler,
        portable_asset_from_train_config,
    )

    training_only = str(train_config.get("checkpoint_eval_backend") or "") == "none"
    eval_compiler = None
    portable_asset = None
    playback = None
    if training_only:
        from gradlab.seeds import EVAL_SEED_START

        playback_environment = {
            key: deepcopy(value)
            for key, value in train_config.items()
            if key in env_config_allowed_keys()
        }
        portable_asset = portable_asset_from_train_config(
            train_config,
            environment=playback_environment,
        )
        playback = {
            "environment": playback_environment,
            "seed": int(train_config.get("checkpoint_eval_seed", EVAL_SEED_START)),
            "asset": portable_asset,
        }
    else:
        eval_compiler = CheckpointEvalContractCompiler.from_train_config(
            train_config,
            portable_asset=True,
            require_asset=False,
            materialize_seed_defaults=True,
        )
        portable_asset = eval_compiler.asset
        from gradlab.env_metadata import sanitize_env_config_metadata

        evaluation_policy_environment_hash = policy_environment_hash(
            sanitize_env_config_metadata(dict(eval_compiler.environment))
        )
        recipe["evaluation_environment_hash"] = evaluation_policy_environment_hash
        if evaluation_policy_environment_hash != training_policy_environment_hash:
            raise ValueError(
                "training and evaluation policy environment semantics disagree while building "
                "recipe.json: "
                f"training={training_policy_environment_hash} "
                f"evaluation={evaluation_policy_environment_hash}"
            )
    stop_on_acceptance = bool(train_config.get("stop_on_acceptance"))
    for key in _OPERATIONAL_TRAIN_FIELDS:
        train_config.pop(key, None)
    train_config.pop("rom_asset_manifest", None)
    if seed is not None:
        train_config["seed"] = int(seed)
    recipe["train_config"] = train_config
    if run_description:
        recipe["description"] = str(run_description)
    recipe = dict(
        _resolve_recipe_templates(
            recipe,
            {
                "seed": "" if seed is None else int(seed),
                "recipe_id": recipe.get("recipe_id") or "",
                "env_id": train_config.get("game") or "",
            },
        )
    )
    recipe["schema_version"] = int(recipe.get("schema_version") or 2)
    if training_only:
        recipe.pop("eval", None)
        recipe["playback"] = playback
    else:
        assert eval_compiler is not None
        recipe.pop("playback", None)
        recipe["eval"] = eval_compiler.contract(require_acceptance=stop_on_acceptance)
    source_files = []
    if isinstance(composition, Mapping):
        for item in composition.get("source_files") or []:
            if not isinstance(item, Mapping):
                continue
            source_files.append(
                {
                    "path": _portable_source_path(item.get("path"), repo_root=repo_root),
                    "sha256": _required_sha256(
                        item.get("sha256"), label="recipe provenance source sha256"
                    ),
                }
            )
    runtime: dict[str, Any] = {}
    if runtime_image_ref is not None:
        runtime["image_ref"] = str(runtime_image_ref)
    if runtime_packages is not None:
        runtime["packages"] = [str(item) for item in runtime_packages]
    provenance: dict[str, Any] = {
        "source_files": source_files,
        "runtime": runtime,
        "asset": portable_asset,
    }
    if source_commit is not None:
        provenance["source_commit"] = _required_text(
            source_commit,
            label="recipe source_commit",
        )
        if not re.fullmatch(r"[0-9a-f]{40}", provenance["source_commit"]):
            raise PolicyDocumentError("recipe source_commit must be a full lowercase Git SHA")
    if source_distribution is not None:
        provenance["source_distribution"] = {
            "name": _required_text(
                source_distribution.get("name"),
                label="recipe source distribution name",
            ),
            "version": _required_text(
                source_distribution.get("version"),
                label="recipe source distribution version",
            ),
        }
    document = {
        "document_type": RECIPE_DOCUMENT_TYPE,
        "format_version": RECIPE_FORMAT_VERSION,
        "recipe": recipe,
        "provenance": provenance,
    }
    return _validate_recipe_v1(document, RECIPE_FILENAME)


def build_model_document(
    checkpoint_path: Path,
    recipe_path: Path,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    load_recipe_document(recipe_path)
    step = metadata.get("checkpoint_step")
    provenance = {
        key: deepcopy(value)
        for key, value in metadata.items()
        if key in _MODEL_PROVENANCE_FIELDS and value not in (None, "")
    }
    if not provenance.get("reward_shape"):
        provenance.pop("reward_shape_is_default", None)
    document = {
        "document_type": MODEL_DOCUMENT_TYPE,
        "format_version": MODEL_FORMAT_VERSION,
        "checkpoint": {
            "filename": CHECKPOINT_FILENAME,
            "sha256": sha256_file(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "kind": str(metadata.get("kind") or "checkpoint"),
            "step": int(step) if step is not None else None,
            "algorithm_id": str(metadata.get("algorithm_id") or ""),
            "model_class": str(metadata.get("model_class") or ""),
        },
        "recipe": {
            "filename": RECIPE_FILENAME,
            "document_type": RECIPE_DOCUMENT_TYPE,
            "format_version": RECIPE_FORMAT_VERSION,
            "sha256": sha256_file(recipe_path),
            "size_bytes": recipe_path.stat().st_size,
        },
        "policy": {
            "algorithm_id": str(metadata.get("algorithm_id") or ""),
            "model_class": str(metadata.get("model_class") or ""),
            "training_backend_id": str(metadata.get("training_backend_id") or ""),
            "training_backend_config_hash": str(metadata.get("training_backend_config_hash") or ""),
        },
        "provenance": provenance,
    }
    return _validate_model_v2(document, MODEL_FILENAME)


def model_document_as_metadata(document: Mapping[str, Any]) -> dict[str, Any]:
    validated = preflight_document(
        document,
        source=MODEL_FILENAME,
        expected_type=MODEL_DOCUMENT_TYPE,
        handlers=_MODEL_HANDLERS,
    )
    metadata = deepcopy(dict(validated["provenance"]))
    metadata.update(validated["policy"])
    metadata["checkpoint_step"] = validated["checkpoint"].get("step")
    metadata["kind"] = validated["checkpoint"].get("kind")
    metadata["filename"] = validated["checkpoint"]["filename"]
    metadata["checkpoint_sha256"] = validated["checkpoint"]["sha256"]
    metadata["recipe"] = deepcopy(validated["recipe"])
    return metadata


def policy_bundle_as_metadata(bundle: PolicyBundle) -> dict[str, Any]:
    """Return model metadata with its environment contract derived from recipe.json."""

    metadata = model_document_as_metadata(bundle.model)
    legacy_training = metadata.get("training_metadata")
    versions = (
        deepcopy(legacy_training.get("versions"))
        if isinstance(legacy_training, Mapping)
        and isinstance(legacy_training.get("versions"), Mapping)
        else {}
    )
    recipe = _required_mapping(
        bundle.recipe.get("recipe"),
        label=f"{bundle.recipe_path}.recipe",
    )
    environment = recipe.get("environment")
    if not isinstance(environment, Mapping):
        return metadata
    environment_hash_value = recipe.get("environment_hash")
    if not isinstance(environment_hash_value, str) or not environment_hash_value.strip():
        return metadata
    preprocessing = _required_mapping(
        environment.get("preprocessing"),
        label=f"{bundle.recipe_path}.recipe.environment.preprocessing",
    )
    env_config = _training_playback_environment(recipe)
    env_id = _required_text(
        environment.get("env_id"),
        label=f"{bundle.recipe_path}.recipe.environment.env_id",
    )
    provider, separator, game = env_id.partition(":")
    if not separator or not provider or not game:
        raise PolicyDocumentError(
            f"{bundle.recipe_path}.recipe.environment.env_id must be provider-qualified"
        )
    provider_args = environment.get("provider_args")
    task = _required_mapping(
        environment.get("task"),
        label=f"{bundle.recipe_path}.recipe.environment.task",
    )
    from gradlab.action_contract import declared_action_contract

    action = declared_action_contract(
        {
            "env_provider": provider,
            "game": game,
            "env_args": provider_args if isinstance(provider_args, Mapping) else {},
            "task": task,
        }
    )
    metadata["training_metadata"] = {
        "env_config": deepcopy(env_config),
        "environment": deepcopy(dict(environment)),
        "environment_hash": environment_hash_value,
        "preprocessing": deepcopy(dict(preprocessing)),
        "action": action,
        "versions": versions,
    }
    return metadata


def _validate_cross_document_contract(model: Mapping[str, Any], recipe: Mapping[str, Any]) -> None:
    checkpoint = model["checkpoint"]
    policy = model["policy"]
    if checkpoint["algorithm_id"] != policy["algorithm_id"]:
        raise PolicyDocumentError("model.json checkpoint and policy algorithm_id disagree")
    if checkpoint["model_class"] != policy["model_class"]:
        raise PolicyDocumentError("model.json checkpoint and policy model_class disagree")
    from gradlab.policy_registry import ALGORITHM_MODEL_CLASSES, resolve_policy_algorithm

    try:
        resolve_policy_algorithm(
            policy,
            allowed=frozenset(ALGORITHM_MODEL_CLASSES),
        )
    except ValueError as exc:
        raise PolicyDocumentError(str(exc)) from exc
    train_config = recipe["recipe"]["train_config"]
    for key in (
        "goal_contract_sha256",
        "effective_goal_contract_sha256",
        "goal_variant_id",
        "goal_variant_label",
        "goal_variant_source_relation",
        "goal_variant_descriptor_sha256",
        "reward_program_kind",
        "reward_program_revision",
        "reward_shape",
        "reward_shape_sha256",
        "reward_shape_is_default",
    ):
        model_value = model["provenance"].get(key)
        recipe_value = train_config.get(key)
        if key == "reward_shape_is_default" and model_value is False and recipe_value is None:
            continue
        if model_value not in (None, "") and model_value != recipe_value:
            raise PolicyDocumentError(f"model.json {key} disagrees with recipe.json")
    backend = _required_mapping(
        train_config.get("training_backend"),
        label="recipe.json training backend",
    )
    if str(backend.get("id") or "") != policy["training_backend_id"]:
        raise PolicyDocumentError("model.json training backend disagrees with recipe.json")
    from gradlab.training_backend import training_backend_config_hash

    expected_backend_hash = training_backend_config_hash(train_config)
    if expected_backend_hash != policy["training_backend_config_hash"]:
        raise PolicyDocumentError(
            "model.json training backend config hash disagrees with recipe.json"
        )
    training_metadata = model["provenance"].get("training_metadata")
    if not isinstance(training_metadata, Mapping):
        return
    environment = training_metadata.get("environment")
    qualified_env_id = (
        str(environment.get("env_id") or "") if isinstance(environment, Mapping) else ""
    )
    expected_env_id = f"{train_config['env_provider']}:{train_config['game']}"
    if qualified_env_id and qualified_env_id != expected_env_id:
        raise PolicyDocumentError("model.json training environment disagrees with recipe.json")
    recipe_environment = recipe["recipe"].get("environment")
    recipe_environment_hash = recipe["recipe"].get("environment_hash")
    if (
        environment is not None
        and recipe_environment is not None
        and environment != recipe_environment
    ):
        raise PolicyDocumentError(
            "model.json normalized training environment disagrees with recipe.json"
        )
    model_environment_hash = training_metadata.get("environment_hash")
    if (
        model_environment_hash is not None
        and recipe_environment_hash is not None
        and model_environment_hash != recipe_environment_hash
    ):
        raise PolicyDocumentError("model.json training environment hash disagrees with recipe.json")


def _load_policy_bundle_paths(
    checkpoint_path: Path,
    model_path: Path,
    recipe_path: Path,
    *,
    source: str,
    revision: str | None,
) -> PolicyBundle:
    model = load_model_document(model_path)
    recipe = load_recipe_document(recipe_path)
    for path, binding in (
        (checkpoint_path, model["checkpoint"]),
        (recipe_path, model["recipe"]),
    ):
        if sha256_file(path) != binding["sha256"]:
            raise PolicyDocumentError(f"{path} hash does not match {model_path}")
        if path.stat().st_size != binding["size_bytes"]:
            raise PolicyDocumentError(f"{path} size does not match {model_path}")
    _validate_cross_document_contract(model, recipe)
    return PolicyBundle(
        checkpoint_path=checkpoint_path,
        model_path=model_path,
        recipe_path=recipe_path,
        model=model,
        recipe=recipe,
        source=source,
        revision=revision,
    )


def load_policy_bundle(
    root: Path,
    *,
    source: str | None = None,
    revision: str | None = None,
) -> PolicyBundle:
    checkpoint_path = root / CHECKPOINT_FILENAME
    model_path = root / MODEL_FILENAME
    recipe_path = root / RECIPE_FILENAME
    missing = [
        path.name for path in (checkpoint_path, model_path, recipe_path) if not path.is_file()
    ]
    if missing:
        raise PolicyDocumentError(f"policy bundle {root} is missing: {', '.join(missing)}")
    return _load_policy_bundle_paths(
        checkpoint_path,
        model_path,
        recipe_path,
        source=source or str(root),
        revision=revision,
    )


def load_policy_bundle_from_checkpoint(
    checkpoint_path: Path,
    *,
    source: str | None = None,
    revision: str | None = None,
) -> PolicyBundle | None:
    if checkpoint_path.name == CHECKPOINT_FILENAME and (
        checkpoint_path.with_name(MODEL_FILENAME).is_file()
        or checkpoint_path.with_name(RECIPE_FILENAME).is_file()
    ):
        return load_policy_bundle(
            checkpoint_path.parent,
            source=source,
            revision=revision,
        )
    model_path = model_document_path(checkpoint_path)
    recipe_path = recipe_document_path(checkpoint_path)
    if not model_path.is_file() and not recipe_path.is_file():
        return None
    missing = [path.name for path in (model_path, recipe_path) if not path.is_file()]
    if missing:
        raise PolicyDocumentError(
            f"versioned policy checkpoint {checkpoint_path} is missing: {', '.join(missing)}"
        )
    return _load_policy_bundle_paths(
        checkpoint_path,
        model_path,
        recipe_path,
        source=source or str(checkpoint_path),
        revision=revision,
    )


def evaluation_contract(recipe_document: Mapping[str, Any]) -> dict[str, Any]:
    validated = preflight_document(
        recipe_document,
        source=RECIPE_FILENAME,
        expected_type=RECIPE_DOCUMENT_TYPE,
        handlers={RECIPE_FORMAT_VERSION: _validate_recipe_v1},
    )
    if "eval" not in validated["recipe"]:
        raise PolicyDocumentError("training-only policy bundle has no evaluation contract")
    contract = deepcopy(dict(validated["recipe"]["eval"]))
    train_config = _required_mapping(
        validated["recipe"].get("train_config"),
        label="recipe.json evaluation training config",
    )
    backend = _required_mapping(
        train_config.get("training_backend"),
        label="recipe.json evaluation training backend",
    )
    from gradlab.policy_registry import (
        backend_provenance_algorithm,
        default_action_selection_mode,
    )

    expected = default_action_selection_mode(
        backend_provenance_algorithm(str(backend.get("id") or ""))
    )
    if contract.get("action_sampling") == "stochastic" and expected != "stochastic":
        contract["action_sampling"] = expected
    return contract


def _training_playback_environment(recipe: Mapping[str, Any]) -> dict[str, Any]:
    from gradlab.env_metadata import sanitize_env_config_metadata
    from gradlab.train_config import env_config_allowed_keys

    train_config = _required_mapping(
        recipe.get("train_config"),
        label="recipe.json training playback environment",
    )
    environment = {
        key: deepcopy(value)
        for key, value in train_config.items()
        if key in env_config_allowed_keys()
    }
    return sanitize_env_config_metadata(environment)


def critic_value_contract(recipe_document: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the training-time inputs required to interpret critic diagnostics."""

    validated = preflight_document(
        recipe_document,
        source=RECIPE_FILENAME,
        expected_type=RECIPE_DOCUMENT_TYPE,
        handlers={RECIPE_FORMAT_VERSION: _validate_recipe_v1},
    )
    recipe = validated["recipe"]
    environment = _training_playback_environment(recipe)
    from gradlab.env_identity import policy_environment_hash

    derived_hash = policy_environment_hash(environment)
    stored_value = recipe.get("value_contract")
    stored = dict(stored_value) if isinstance(stored_value, Mapping) else {}
    train_config = _required_mapping(
        recipe.get("train_config"),
        label="recipe.json critic training config",
    )
    backend_value = train_config.get("training_backend")
    backend = backend_value if isinstance(backend_value, Mapping) else {}
    from gradlab.policy_registry import backend_provenance_algorithm

    algorithm_id = backend_provenance_algorithm(str(backend.get("id") or ""))
    if algorithm_id not in {"ppo", "a2c"}:
        return None
    backend_config_value = backend.get("config")
    backend_config = backend_config_value if isinstance(backend_config_value, Mapping) else {}
    gamma = backend_config.get("gamma")
    discount = (
        float(gamma)
        if not isinstance(gamma, bool)
        and isinstance(gamma, int | float)
        and math.isfinite(float(gamma))
        and 0.0 <= float(gamma) <= 1.0
        else None
    )
    expected = {
        "schema_version": 1,
        "policy_environment_hash": derived_hash,
        "reward_stream": "task",
        "discount": discount,
        "action_sampling": "stochastic",
        "truncation_bootstrap": (
            "terminal-value" if str(backend.get("id") or "").startswith("sb3.") else "unknown"
        ),
    }
    if stored:
        for key, value in expected.items():
            if stored.get(key) != value:
                raise PolicyDocumentError(
                    f"recipe.json value_contract.{key} disagrees with the derived training contract"
                )
    return expected


def _contract_difference_paths(
    training: Any,
    evaluation: Any,
    *,
    path: str = "environment",
) -> list[str]:
    if isinstance(training, Mapping) and isinstance(evaluation, Mapping):
        paths: list[str] = []
        for key in sorted(set(training) | set(evaluation), key=str):
            nested_path = f"{path}.{key}"
            if key not in training or key not in evaluation:
                paths.append(nested_path)
                continue
            paths.extend(
                _contract_difference_paths(
                    training[key],
                    evaluation[key],
                    path=nested_path,
                )
            )
        return paths
    if training == evaluation:
        return []
    return [path]


def playback_contract_audit(recipe_document: Mapping[str, Any]) -> dict[str, Any]:
    """Audit persisted train/eval policy semantics and legacy override provenance."""

    validated = preflight_document(
        recipe_document,
        source=RECIPE_FILENAME,
        expected_type=RECIPE_DOCUMENT_TYPE,
        handlers={RECIPE_FORMAT_VERSION: _validate_recipe_v1},
    )
    recipe = validated["recipe"]
    training_environment = _training_playback_environment(recipe)
    from gradlab.env_identity import policy_environment_hash, policy_environment_identity
    from gradlab.env_metadata import sanitize_env_config_metadata

    training_identity = policy_environment_identity(training_environment)
    training_hash = policy_environment_hash(training_environment)
    evaluation_value = recipe.get("eval")
    evaluation_hash: str | None = None
    mismatch_paths: list[str] = []
    if isinstance(evaluation_value, Mapping):
        evaluation_environment = sanitize_env_config_metadata(
            dict(
                _required_mapping(
                    evaluation_value.get("environment"),
                    label="recipe.json evaluation playback environment",
                )
            )
        )
        evaluation_identity = policy_environment_identity(evaluation_environment)
        evaluation_hash = policy_environment_hash(evaluation_environment)
        mismatch_paths = _contract_difference_paths(
            training_identity,
            evaluation_identity,
        )
    requested_value = recipe.get("recipe_overrides")
    requested_overrides = (
        [str(value) for value in requested_value]
        if isinstance(requested_value, Sequence) and not isinstance(requested_value, str | bytes)
        else []
    )
    effective_value = recipe.get("effective_recipe_overrides")
    effective_overrides = (
        [str(value) for value in effective_value]
        if isinstance(effective_value, Sequence) and not isinstance(effective_value, str | bytes)
        else []
    )
    policy_override_paths = sorted(
        {
            value.split("=", 1)[0].strip()
            for value in requested_overrides
            if value.split("=", 1)[0]
            .strip()
            .startswith(("train.environment.", "eval.environment."))
        }
    )
    return {
        "schema_version": 1,
        "training_policy_environment_hash": training_hash,
        "evaluation_policy_environment_hash": evaluation_hash,
        "evaluation_matches_training": (
            None if evaluation_hash is None else evaluation_hash == training_hash
        ),
        "mismatch_paths": mismatch_paths,
        "requested_policy_override_paths": policy_override_paths,
        "effective_recipe_overrides": effective_overrides,
        "legacy_override_provenance": bool(policy_override_paths and not effective_overrides),
    }


def playback_contract(
    recipe_document: Mapping[str, Any],
    *,
    mode: str = "training",
) -> dict[str, Any]:
    """Return explicit training-faithful or published-evaluation playback settings."""

    validated = preflight_document(
        recipe_document,
        source=RECIPE_FILENAME,
        expected_type=RECIPE_DOCUMENT_TYPE,
        handlers={RECIPE_FORMAT_VERSION: _validate_recipe_v1},
    )
    recipe = validated["recipe"]
    provenance = validated["provenance"]
    training_environment = _training_playback_environment(recipe)
    from gradlab.env_identity import policy_environment_hash
    from gradlab.env_metadata import sanitize_env_config_metadata

    training_hash = policy_environment_hash(training_environment)
    if mode == "training":
        train_config = _required_mapping(
            recipe.get("train_config"),
            label="recipe.json training playback config",
        )
        seed_value = train_config.get("seed")
        if not isinstance(seed_value, int) or isinstance(seed_value, bool):
            portable = recipe.get("eval") or recipe.get("playback") or {}
            seed_value = portable.get("seed") if isinstance(portable, Mapping) else None
        if not isinstance(seed_value, int) or isinstance(seed_value, bool):
            raise PolicyDocumentError("recipe.json training playback seed must be an integer")
        return {
            "mode": "training",
            "environment": training_environment,
            "seed": int(seed_value),
            "asset": deepcopy(provenance.get("asset")),
            "policy_environment_hash": training_hash,
            "training_policy_environment_hash": training_hash,
            "matches_training": True,
        }
    if mode != "evaluation":
        raise PolicyDocumentError(f"unsupported playback contract mode: {mode!r}")
    evaluation_value = recipe.get("eval")
    if not isinstance(evaluation_value, Mapping):
        raise PolicyDocumentError("training-only policy bundle has no evaluation contract")
    evaluation_environment = sanitize_env_config_metadata(
        dict(
            _required_mapping(
                evaluation_value.get("environment"),
                label="recipe.json evaluation playback environment",
            )
        )
    )
    evaluation_hash = policy_environment_hash(evaluation_environment)
    return {
        "mode": "evaluation",
        "environment": evaluation_environment,
        "seed": int(evaluation_value["seed"]),
        "asset": deepcopy(evaluation_value.get("asset")),
        "policy_environment_hash": evaluation_hash,
        "training_policy_environment_hash": training_hash,
        "matches_training": evaluation_hash == training_hash,
    }


def evaluation_contract_sha256(recipe_document: Mapping[str, Any]) -> str:
    return canonical_json_sha256(evaluation_contract(recipe_document))


def playback_contract_sha256(
    recipe_document: Mapping[str, Any],
    *,
    mode: str = "training",
) -> str:
    return canonical_json_sha256(playback_contract(recipe_document, mode=mode))
