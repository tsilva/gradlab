from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path, PureWindowsPath
from typing import Annotated, Any, Literal

from huggingface_hub import ModelCard
from huggingface_hub.utils import validate_repo_id
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import Field, StringConstraints

from gradlab.action_contract import action_contract_meanings, configured_action_meanings
from gradlab.boundary_schema import BoundaryModel, validate_boundary
from gradlab.env_registry import game_family_for_environment
from gradlab.file_utils import file_sha256 as sha256_file
from gradlab.metric_names import (
    EVAL_FULL_BY_START,
    EVAL_FULL_CHECKPOINT_ARTIFACT,
    EVAL_FULL_CHECKPOINT_STEP,
    EVAL_FULL_EPISODE_COUNT,
    EVAL_FULL_EPISODE_RETURN_MEAN,
    EVAL_FULL_PROGRESS_X_MAX,
    EVAL_FULL_SUCCESS_RATE_MEAN,
    EVAL_FULL_SUCCESS_RATE_MIN,
)
from gradlab.env_registry import environment_spec
from gradlab.policy_bundle import (
    PolicyBundle,
    PolicyDocumentError,
    evaluation_contract_sha256,
    load_policy_bundle,
    preflight_document,
)
from gradlab.policy_registry import ALGORITHM_MODEL_CLASSES


HUGGINGFACE_NAMESPACE = "tsilva"
REPO_NAMING_SCHEMA_VERSION = 1
RELEASE_MANIFEST_DOCUMENT_TYPE = "gradlab.release_manifest"
RELEASE_MANIFEST_VERSION = 1
HUGGINGFACE_RELEASE_FILES = frozenset(
    {
        ".gitattributes",
        "README.md",
        "LICENSE",
        "model.zip",
        "model.json",
        "recipe.json",
        "release_manifest.json",
        "replay.mp4",
    }
)
HASHED_RELEASE_FILES = HUGGINGFACE_RELEASE_FILES - {"release_manifest.json"}
GITATTRIBUTES_TEXT = """*.zip filter=lfs diff=lfs merge=lfs -text
*.mp4 filter=lfs diff=lfs merge=lfs -text
"""
MIT_LICENSE_TEXT = """MIT License

Copyright (c) 2026 Tiago Silva

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[
    str,
    StringConstraints(strip_whitespace=True, to_lower=True, pattern=r"^[0-9a-f]{64}$"),
]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]


class _ReleaseRepository(BoundaryModel):
    repo_id: NonEmptyText
    game_family: NonEmptyText
    goal: NonEmptyText
    policy_variant: NonEmptyText
    algorithm: NonEmptyText


class _ReleaseDetails(BoundaryModel):
    version: NonEmptyText
    published_at: NonEmptyText
    youtube_url: NonEmptyText | None = None


class _ReleaseModel(BoundaryModel):
    algorithm_id: Any
    model_class: Any
    qualified_env_id: Any
    environment_hash: Any
    preprocessing: Any
    action: Any


class _ReleaseSource(BoundaryModel):
    repository: Any
    commit: Any
    run_id: Any
    run_name: Any
    wandb_project: Any
    recipe: Any
    seed: Any
    checkpoint_step: Any
    checkpoint_artifact: Any


class _ReleaseEvaluation(BoundaryModel):
    action_sampling: Any
    protocol: Any
    checkpoint_step: Any
    checkpoint_artifact: Any
    episodes: Any
    success_rate_min: Any
    success_rate_mean: Any
    return_mean: Any
    by_start: Any
    checkpoint_sha256: Any
    recipe_sha256: Any
    recipe_format_version: Any
    evaluation_contract_sha256: Any
    exact_contract: Any
    progress_max: Any = None


class _ArtifactRecord(BoundaryModel):
    sha256: Sha256
    size_bytes: PositiveInt


class _ReleaseManifest(BoundaryModel):
    document_type: Literal[RELEASE_MANIFEST_DOCUMENT_TYPE]
    format_version: Literal[RELEASE_MANIFEST_VERSION]
    repo_naming_schema: Literal[REPO_NAMING_SCHEMA_VERSION]
    repository: _ReleaseRepository
    release: _ReleaseDetails
    model: _ReleaseModel
    source: _ReleaseSource
    evaluation: _ReleaseEvaluation
    artifacts: dict[str, _ArtifactRecord]


@dataclass(frozen=True)
class PublicationIdentity:
    game_family: str
    goal: str
    policy_variant: str
    algorithm: str

    @property
    def repo_name(self) -> str:
        return "_".join(asdict(self).values())


@dataclass(frozen=True)
class PublicationEvaluation:
    action_sampling: str
    protocol: str
    checkpoint_step: int
    checkpoint_artifact: str
    episodes: int
    success_rate_min: float
    success_rate_mean: float
    return_mean: float
    progress_max: float | None
    by_start: tuple[dict[str, Any], ...]

    def as_manifest_value(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "action_sampling": self.action_sampling,
            "protocol": self.protocol,
            "checkpoint_step": self.checkpoint_step,
            "checkpoint_artifact": self.checkpoint_artifact,
            "episodes": self.episodes,
            "success_rate_min": self.success_rate_min,
            "success_rate_mean": self.success_rate_mean,
            "return_mean": self.return_mean,
            "by_start": [dict(row) for row in self.by_start],
        }
        if self.progress_max is not None:
            result["progress_max"] = self.progress_max
        return result


def normalize_publication_component(value: object, *, label: str) -> str:
    text = str(value or "").strip()
    normalized = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")
    normalized = re.sub(r"-+", "-", normalized)
    if not normalized:
        raise ValueError(f"{label} does not contain a valid repository-name component")
    if "_" in normalized:
        raise AssertionError("publication component normalization retained an underscore")
    return normalized


def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _provider_and_environment(qualified_env_id: object) -> tuple[str, str]:
    value = str(qualified_env_id or "").strip()
    if ":" not in value:
        raise ValueError("model metadata environment.env_id must be provider-qualified")
    provider, environment = value.split(":", 1)
    if not provider or not environment:
        raise ValueError("model metadata environment.env_id must be provider-qualified")
    return provider, environment


def normalize_algorithm_id(value: object) -> str:
    algorithm = normalize_publication_component(value, label="algorithm").lower()
    if algorithm not in ALGORITHM_MODEL_CLASSES:
        known = ", ".join(sorted(ALGORITHM_MODEL_CLASSES))
        raise ValueError(f"unsupported publication algorithm {algorithm!r}; known: {known}")
    return algorithm


def validate_algorithm_model_class(algorithm: str, model_class: object) -> str:
    class_name = str(model_class or "").strip()
    if not class_name:
        raise ValueError("model metadata model_class is required for publication")
    allowed = ALGORITHM_MODEL_CLASSES[algorithm]
    if class_name not in allowed:
        raise ValueError(
            f"model class {class_name!r} is incompatible with algorithm {algorithm!r}; "
            f"expected one of {sorted(allowed)}"
        )
    return class_name


def _observation_shape(preprocessing: Mapping[str, Any]) -> tuple[int, int]:
    resize = preprocessing.get("obs_resize")
    if not isinstance(resize, Sequence) or isinstance(resize, str | bytes) or len(resize) != 2:
        raise ValueError("publication preprocessing.obs_resize must contain height and width")
    height, width = (int(resize[0]), int(resize[1]))
    if height <= 0 or width <= 0:
        raise ValueError("publication observation dimensions must be positive")
    return height, width


def _view_component(
    preprocessing: Mapping[str, Any],
    *,
    game: str,
    provider: str,
) -> str:
    raw_crop = preprocessing.get("obs_crop")
    if raw_crop is None:
        return "full"
    if (
        not isinstance(raw_crop, Sequence)
        or isinstance(raw_crop, str | bytes)
        or len(raw_crop) != 4
    ):
        raise ValueError("publication preprocessing.obs_crop must contain top,right,bottom,left")
    crop = tuple(int(value) for value in raw_crop)
    if any(value < 0 for value in crop):
        raise ValueError("publication crop values must be non-negative")
    if not any(crop):
        return "full"
    mode = str(preprocessing.get("obs_crop_mode") or "remove")
    if mode not in {"mask", "remove"}:
        raise ValueError(f"unsupported publication crop mode {mode!r}")
    default_crop = environment_spec(provider, game).default_obs_crop
    if default_crop is not None and crop == default_crop:
        return "hudmask" if mode == "mask" else "hudcrop"
    prefix = "mask" if mode == "mask" else "crop"
    top, right, bottom, left = crop
    return f"{prefix}-t{top}-r{right}-b{bottom}-l{left}"


def policy_variant_from_contract(
    preprocessing: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    game: str,
    provider: str = "stable-retro-turbo",
    action_contract: Mapping[str, Any] | None = None,
) -> str:
    height, width = _observation_shape(preprocessing)
    grayscale = preprocessing.get("obs_grayscale")
    if not isinstance(grayscale, bool):
        raise ValueError("publication preprocessing.obs_grayscale must be boolean")
    color = "gray" if grayscale else "rgb"
    dimensions = str(height) if height == width else f"{height}x{width}"
    components = [
        f"{color}{dimensions}",
        _view_component(preprocessing, game=game, provider=provider),
    ]

    frame_stack = int(preprocessing.get("frame_stack") or 0)
    if frame_stack <= 0:
        raise ValueError("publication preprocessing.frame_stack must be positive")
    components.append(f"stack{frame_stack}")

    layout = str(preprocessing.get("policy_observation_layout") or "")
    if layout == "dict_image_task":
        components.append("taskdict")
    elif layout != "channel_first":
        raise ValueError(f"unsupported policy observation layout {layout!r}")

    action = _require_mapping(task.get("action"), label="publication task.action")
    action_set = str(action.get("set") or "").strip()
    if isinstance(action_contract, Mapping) and action_contract.get("schema_version") is not None:
        provider_contract = _require_mapping(
            action_contract.get("provider"),
            label="publication runtime action contract provider",
        )
        action_set = str(
            provider_contract.get("preset") or provider_contract.get("mode") or action_set
        ).strip()
        meanings = action_contract_meanings(action_contract)
        if not meanings:
            raise ValueError(
                "publication runtime action contract requires semantic IDs "
                "for every discrete action"
            )
    elif isinstance(action_contract, Mapping) and action_contract.get("preset"):
        action_set = str(action_contract["preset"]).strip()
        meanings = action_contract.get("meanings")
        if not isinstance(meanings, Sequence) or isinstance(meanings, str | bytes) or not meanings:
            raise ValueError("publication action contract meanings must be a non-empty list")
    else:
        configured_action_meanings(
            {
                "env_provider": provider,
                "game": game,
                "env_args": {},
                "task": task,
            }
        )
    components.append(normalize_publication_component(action_set, label="publication action set"))
    return "-".join(components)


def publication_identity_from_policy_bundle(
    goal_id: object,
    bundle: PolicyBundle,
) -> PublicationIdentity:
    recipe = _require_mapping(bundle.recipe.get("recipe"), label="recipe.json recipe")
    environment = _require_mapping(
        recipe.get("environment"), label="recipe.json recipe.environment"
    )
    provider, game = _provider_and_environment(environment.get("env_id"))
    family = game_family_for_environment(provider, game, strict=True)
    preprocessing = _require_mapping(
        environment.get("preprocessing"),
        label="recipe.json recipe.environment.preprocessing",
    )
    task = _require_mapping(environment.get("task"), label="recipe.json recipe.environment.task")
    policy = _require_mapping(bundle.model.get("policy"), label="model.json policy")
    provenance = _require_mapping(bundle.model.get("provenance"), label="model.json provenance")
    training_metadata = provenance.get("training_metadata")
    action_contract = (
        training_metadata.get("action_contract")
        if isinstance(training_metadata, Mapping)
        and isinstance(training_metadata.get("action_contract"), Mapping)
        else None
    )
    if action_contract is None:
        from gradlab.action_contract import (
            declared_action_contract,
            migrate_legacy_artifact_action_configuration,
        )

        provider_args = environment.get("provider_args")
        migrated_args, migrated_task = migrate_legacy_artifact_action_configuration(
            provider_id=provider,
            game=game,
            env_args=(dict(provider_args) if isinstance(provider_args, Mapping) else {}),
            task=task,
        )
        task = migrated_task
        action_contract = declared_action_contract(
            {
                "env_provider": provider,
                "game": game,
                "env_args": migrated_args,
                "task": dict(task),
            }
        )
    algorithm = normalize_algorithm_id(policy.get("algorithm_id"))
    validate_algorithm_model_class(algorithm, policy.get("model_class"))
    game_family = normalize_publication_component(family, label="game family")
    goal_component = normalize_publication_component(goal_id, label="goal id")
    policy_variant = policy_variant_from_contract(
        preprocessing,
        task,
        game=game,
        provider=provider,
        action_contract=action_contract,
    )
    reward_shape = str(provenance.get("reward_shape") or "").strip()
    reward_shape_sha256 = str(provenance.get("reward_shape_sha256") or "").strip()
    if reward_shape and not bool(provenance.get("reward_shape_is_default", False)):
        raw_shape_component = normalize_publication_component(
            reward_shape, label="reward shape"
        ).lower()
        digest = reward_shape_sha256.removeprefix("sha256:")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("non-default publication reward_shape_sha256 must be a SHA-256")
        policy_budget = 96 - (len(game_family) + len(goal_component) + len(algorithm) + 3)
        shape_budget = policy_budget - len(policy_variant) - len("-shape--") - 8
        if shape_budget < 1:
            raise ValueError("publication identity leaves no room for reward-shape provenance")
        shape_component = raw_shape_component[:shape_budget].rstrip("-")
        policy_variant = f"{policy_variant}-shape-{shape_component}-{digest[:8]}"
    return PublicationIdentity(
        game_family=game_family,
        goal=goal_component,
        policy_variant=policy_variant,
        algorithm=algorithm,
    )


def build_model_repo_id(identity: PublicationIdentity) -> str:
    for field, value in asdict(identity).items():
        normalized = normalize_publication_component(value, label=field)
        if normalized != value:
            raise ValueError(
                f"publication identity {field} must already be canonical: "
                f"expected {normalized!r}, got {value!r}"
            )
    repo_id = f"{HUGGINGFACE_NAMESPACE}/{identity.repo_name}"
    validate_repo_id(repo_id)
    if len(identity.repo_name) > 96:
        raise ValueError("Hugging Face repository name exceeds 96 characters")
    return repo_id


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _required_text(value: object, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is required")
    return text


def _required_int(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _required_rate(value: object, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1")
    return result


def _required_float(value: object, *, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc


def _normalize_by_start_rows(value: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or not value:
        raise ValueError("evaluation by_start must be a non-empty list")
    by_start: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(value):
        if isinstance(raw, Mapping):
            row = dict(raw)
        elif isinstance(raw, Sequence) and not isinstance(raw, str | bytes) and len(raw) >= 7:
            row = {
                "start_id": raw[0],
                "episodes": raw[1],
                "success_count": raw[2],
                "success_rate": raw[3],
                "return_mean": raw[4],
                "return_std": raw[5],
                "return_median": raw[6],
            }
        else:
            raise ValueError(f"evaluation by_start row {index} has an unsupported shape")
        start_id = _required_text(row.get("start_id"), label=f"by_start[{index}].start_id")
        normalized = {
            "start_id": start_id,
            "episodes": _required_int(row.get("episodes"), label=f"by_start[{index}].episodes"),
            "success_count": _required_int(
                row.get("success_count"), label=f"by_start[{index}].success_count"
            ),
            "success_rate": _required_rate(
                row.get("success_rate"), label=f"by_start[{index}].success_rate"
            ),
            "return_mean": _required_float(
                row.get("return_mean"), label=f"by_start[{index}].return_mean"
            ),
        }
        if normalized["episodes"] <= 0:
            raise ValueError(f"by_start[{index}].episodes must be positive")
        if normalized["success_count"] > normalized["episodes"]:
            raise ValueError(f"by_start[{index}].success_count exceeds episodes")
        previous = by_start.get(start_id)
        if previous is not None and previous != normalized:
            raise ValueError(f"evaluation contains conflicting rows for start {start_id!r}")
        by_start[start_id] = normalized
    return tuple(by_start[key] for key in sorted(by_start))


def normalize_publication_evaluation(
    evaluation: Mapping[str, Any],
    *,
    allow_deterministic: bool = False,
    algorithm_id: str | None = None,
) -> PublicationEvaluation:
    action_sampling = str(evaluation.get("action_sampling") or "").strip().lower()
    if allow_deterministic and not action_sampling and evaluation.get("deterministic") is True:
        action_sampling = "deterministic"
    allowed_sampling = (
        {"program"}
        if algorithm_id == "action-program"
        else ({"stochastic", "deterministic"} if allow_deterministic else {"stochastic"})
    )
    if action_sampling not in allowed_sampling:
        expected = " or ".join(sorted(allowed_sampling))
        raise ValueError(f"release evaluation action_sampling must be {expected}")
    protocol = str(evaluation.get("protocol") or "full").strip().lower()
    if protocol != "full":
        raise ValueError("release evaluation protocol must be 'full'")
    checkpoint_step = _required_int(
        _first_present(evaluation, "checkpoint_step", EVAL_FULL_CHECKPOINT_STEP),
        label="evaluation checkpoint_step",
    )
    checkpoint_artifact = _required_text(
        _first_present(evaluation, "checkpoint_artifact", EVAL_FULL_CHECKPOINT_ARTIFACT),
        label="evaluation checkpoint_artifact",
    )
    episodes = _required_int(
        _first_present(evaluation, "episodes", EVAL_FULL_EPISODE_COUNT),
        label="evaluation episodes",
    )
    if episodes <= 0:
        raise ValueError("evaluation episodes must be positive")
    success_rate_min = _required_rate(
        _first_present(evaluation, "success_rate_min", EVAL_FULL_SUCCESS_RATE_MIN),
        label="evaluation success_rate_min",
    )
    success_rate_mean = _required_rate(
        _first_present(evaluation, "success_rate_mean", EVAL_FULL_SUCCESS_RATE_MEAN),
        label="evaluation success_rate_mean",
    )
    return_mean = _required_float(
        _first_present(evaluation, "return_mean", EVAL_FULL_EPISODE_RETURN_MEAN),
        label="evaluation return_mean",
    )
    progress_value = _first_present(evaluation, "progress_max", EVAL_FULL_PROGRESS_X_MAX)
    progress_max = (
        None
        if progress_value is None
        else _required_float(progress_value, label="evaluation progress_max")
    )
    by_start = _normalize_by_start_rows(
        _first_present(evaluation, "by_start", "_eval_by_start_rows", EVAL_FULL_BY_START)
    )
    if sum(int(row["episodes"]) for row in by_start) != episodes:
        raise ValueError("evaluation episodes must equal the sum of by_start episodes")
    observed_rates = [float(row["success_rate"]) for row in by_start]
    if abs(min(observed_rates) - success_rate_min) > 1e-9:
        raise ValueError("evaluation success_rate_min disagrees with by_start")
    observed_mean = sum(observed_rates) / len(observed_rates)
    if abs(observed_mean - success_rate_mean) > 1e-9:
        raise ValueError("evaluation success_rate_mean disagrees with by_start")
    return PublicationEvaluation(
        action_sampling=action_sampling,
        protocol=protocol,
        checkpoint_step=checkpoint_step,
        checkpoint_artifact=checkpoint_artifact,
        episodes=episodes,
        success_rate_min=success_rate_min,
        success_rate_mean=success_rate_mean,
        return_mean=return_mean,
        progress_max=progress_max,
        by_start=by_start,
    )


def publication_source_from_policy_bundle(
    bundle: PolicyBundle,
    evaluation: PublicationEvaluation,
) -> dict[str, Any]:
    provenance = _require_mapping(bundle.model.get("provenance"), label="model.json provenance")
    checkpoint = _require_mapping(bundle.model.get("checkpoint"), label="model.json checkpoint")
    seed = _required_int(provenance.get("seed"), label="model.json provenance.seed")
    commit = _required_text(
        provenance.get("repo_git_commit"),
        label="model.json provenance.repo_git_commit",
    ).lower()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("model.json provenance.repo_git_commit must be a full Git SHA")
    checkpoint_step = _required_int(checkpoint.get("step"), label="model.json checkpoint.step")
    if checkpoint_step != evaluation.checkpoint_step:
        raise ValueError("model.json checkpoint_step disagrees with evaluation")
    return {
        "repository": "https://github.com/tsilva/gradlab",
        "commit": commit,
        "run_id": _required_text(
            provenance.get("wandb_run_id"),
            label="model.json provenance.wandb_run_id",
        ),
        "run_name": _required_text(
            provenance.get("run_name"), label="model.json provenance.run_name"
        ),
        "wandb_project": _required_text(
            provenance.get("wandb_project"),
            label="model.json provenance.wandb_project",
        ),
        "recipe": _required_text(
            provenance.get("recipe_slug"), label="model.json provenance.recipe_slug"
        ),
        "seed": seed,
        "checkpoint_step": checkpoint_step,
        "checkpoint_artifact": evaluation.checkpoint_artifact,
    }


def publication_model_contract(bundle: PolicyBundle) -> dict[str, Any]:
    policy = _require_mapping(bundle.model.get("policy"), label="model.json policy")
    recipe = _require_mapping(bundle.recipe.get("recipe"), label="recipe.json recipe")
    environment = _require_mapping(
        recipe.get("environment"), label="recipe.json recipe.environment"
    )
    task = _require_mapping(environment.get("task"), label="recipe.json recipe.environment.task")
    return {
        "algorithm_id": policy.get("algorithm_id"),
        "model_class": policy.get("model_class"),
        "qualified_env_id": environment.get("env_id"),
        "environment_hash": recipe.get("environment_hash"),
        "preprocessing": environment.get("preprocessing"),
        "action": task.get("action"),
    }


def _markdown_value(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _percent(value: object) -> str:
    return f"{100.0 * float(value):.1f}%"


_MODEL_CARD_TEMPLATE_ENV = Environment(
    loader=FileSystemLoader(Path(__file__).with_name("templates")),
    undefined=StrictUndefined,
    autoescape=False,
    keep_trailing_newline=True,
)


def _render_model_card_template(context: Mapping[str, Any]) -> str:
    return _MODEL_CARD_TEMPLATE_ENV.get_template("model_card.md.j2").render(**context)


def render_model_card(
    manifest: Mapping[str, Any],
    bundle: PolicyBundle,
) -> str:
    repository = _require_mapping(manifest.get("repository"), label="manifest repository")
    release = _require_mapping(manifest.get("release"), label="manifest release")
    model = _require_mapping(manifest.get("model"), label="manifest model")
    source = _require_mapping(manifest.get("source"), label="manifest source")
    evaluation = _require_mapping(manifest.get("evaluation"), label="manifest evaluation")
    repo_id = _required_text(repository.get("repo_id"), label="manifest repository.repo_id")
    goal = _required_text(repository.get("goal"), label="manifest repository.goal")
    qualified_env_id = _required_text(
        model.get("qualified_env_id"), label="manifest model.qualified_env_id"
    )
    provider, game = _provider_and_environment(qualified_env_id)
    algorithm = _required_text(model.get("algorithm_id"), label="manifest model.algorithm_id")
    version = str(release.get("version") or "").strip()
    youtube_url = str(release.get("youtube_url") or "").strip()
    action_sampling = _required_text(
        evaluation.get("action_sampling"), label="manifest evaluation.action_sampling"
    )
    by_start = _normalize_by_start_rows(evaluation.get("by_start"))
    success_min = _required_rate(
        evaluation.get("success_rate_min"), label="manifest evaluation.success_rate_min"
    )
    success_mean = _required_rate(
        evaluation.get("success_rate_mean"), label="manifest evaluation.success_rate_mean"
    )
    checkpoint_step = _required_int(
        evaluation.get("checkpoint_step"), label="manifest evaluation.checkpoint_step"
    )
    episodes = _required_int(evaluation.get("episodes"), label="manifest evaluation.episodes")
    return_mean = _required_float(
        evaluation.get("return_mean"), label="manifest evaluation.return_mean"
    )
    preprocessing = _require_mapping(model.get("preprocessing"), label="manifest preprocessing")
    action = _require_mapping(model.get("action"), label="manifest action")
    run_id = str(source.get("run_id") or "").strip()
    project = str(source.get("wandb_project") or "").strip()
    wandb_url = f"https://wandb.ai/tsilva/{project}/runs/{run_id}" if project and run_id else ""
    if not re.fullmatch(r"v[1-9][0-9]*", version):
        raise ValueError("model cards require a sequential release version")
    commit = _required_text(source.get("commit"), label="manifest source.commit")
    model_ref = f"https://huggingface.co/{repo_id}/resolve/{version}/model.zip"
    install = "\n".join(
        (
            "```bash",
            "git clone https://github.com/tsilva/gradlab",
            "cd gradlab",
            f"git checkout {commit}",
            "uv sync --frozen",
            "```",
        )
    )
    youtube_value = f"[Watch on YouTube]({youtube_url})" if youtube_url else "Not available"
    manifest_purpose = "Release identity, evaluation evidence, and artifact hashes"
    rows = "\n".join(
        "| {start} | {episodes} | {success_count} | {success_rate} | {return_mean:.3f} |".format(
            start=_markdown_value(row["start_id"]),
            episodes=int(row["episodes"]),
            success_count=int(row["success_count"]),
            success_rate=_percent(row["success_rate"]),
            return_mean=float(row["return_mean"]),
        )
        for row in by_start
    )
    status = ""
    is_action_program = algorithm == "action-program"
    provenance = _require_mapping(bundle.model.get("provenance"), label="model.json provenance")
    producer = (
        _required_text(
            provenance.get("search_algorithm_id"),
            label="model provenance search_algorithm_id",
        )
        if is_action_program
        else ""
    )
    library_name = "gradlab" if is_action_program else "stable-baselines3"
    library_tag = "gradlab-policy" if is_action_program else "stable-baselines3"
    policy_description = (
        f"GradLab open-loop action program for `{game}` `{goal}`, produced by "
        f"`{producer}` and trained and evaluated with"
        if is_action_program
        else f"Stable-Baselines3 {algorithm.upper()} policy for `{game}` `{goal}`, "
        "trained and evaluated with"
    )
    model_file_description = (
        "Portable GradLab open-loop action program"
        if is_action_program
        else "Stable-Baselines3 policy checkpoint"
    )
    run_name = _required_text(source.get("run_name"), label="manifest source.run_name")
    run_value = f"[{_markdown_value(run_name)}]({wandb_url})" if wandb_url else run_name
    card = _render_model_card_template(
        {
            "library_name": library_name,
            "library_tag": library_tag,
            "algorithm": algorithm,
            "algorithm_upper": algorithm.upper(),
            "producer": producer,
            "provider": provider,
            "game": game,
            "goal": goal,
            "policy_description": policy_description,
            "checkpoint_step": checkpoint_step,
            "action_sampling": action_sampling,
            "episodes": episodes,
            "success_min": _percent(success_min),
            "success_mean": _percent(success_mean),
            "return_mean": f"{return_mean:.3f}",
            "version": version,
            "youtube_value": youtube_value,
            "install": install,
            "model_ref": model_ref,
            "rows": rows,
            "qualified_env_id": qualified_env_id,
            "environment_hash": _markdown_value(model.get("environment_hash")),
            "preprocessing": _markdown_value(
                json.dumps(preprocessing, sort_keys=True, separators=(",", ":"))
            ),
            "action": _markdown_value(json.dumps(action, sort_keys=True, separators=(",", ":"))),
            "run_value": run_value,
            "recipe": _markdown_value(
                _required_text(source.get("recipe"), label="manifest source.recipe")
            ),
            "seed": _required_int(source.get("seed"), label="manifest source.seed"),
            "source_commit": commit,
            "checkpoint_artifact": _markdown_value(
                _required_text(
                    source.get("checkpoint_artifact"),
                    label="manifest source.checkpoint_artifact",
                )
            ),
            "model_file_description": model_file_description,
            "manifest_purpose": manifest_purpose,
            "status": status,
        }
    )
    return card.strip() + "\n"


def validate_model_card(
    card_text: str,
    manifest: Mapping[str, Any],
    bundle: PolicyBundle,
) -> None:
    ModelCard(card_text).validate(repo_type="model")
    expected = render_model_card(manifest, bundle)
    if card_text != expected:
        raise ValueError("README.md does not match the generated model card")


def verify_replay(path: Path) -> dict[str, object]:
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe is required to validate replay.mp4")
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,codec_tag_string,pix_fmt,nb_read_frames:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    probe = json.loads(completed.stdout)
    streams = probe.get("streams")
    if not isinstance(streams, list) or not streams:
        raise ValueError("replay video does not contain a video stream")
    stream = streams[0]
    expected = {"codec_name": "h264", "codec_tag_string": "avc1", "pix_fmt": "yuv420p"}
    for key, value in expected.items():
        if stream.get(key) != value:
            raise ValueError(f"replay video {key} must be {value!r}, got {stream.get(key)!r}")
    duration = float(probe.get("format", {}).get("duration") or 0.0)
    frames = int(stream.get("nb_read_frames") or 0)
    if duration <= 0 or frames <= 0:
        raise ValueError("replay video must have a positive duration and frame count")
    data = path.read_bytes()
    moov = data.find(b"moov")
    mdat = data.find(b"mdat")
    if moov < 0 or mdat < 0 or moov > mdat:
        raise ValueError("replay video must use faststart with moov before mdat")
    return {"duration_seconds": duration, "frames": frames, **expected}


def release_artifact_records(root: Path) -> dict[str, dict[str, int | str]]:
    return {
        filename: {
            "sha256": sha256_file(root / filename),
            "size_bytes": (root / filename).stat().st_size,
        }
        for filename in sorted(HASHED_RELEASE_FILES)
    }


def _assert_no_absolute_paths(value: object, *, path: str = "manifest") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _assert_no_absolute_paths(nested, path=f"{path}.{key}")
        return
    if isinstance(value, list | tuple):
        for index, nested in enumerate(value):
            _assert_no_absolute_paths(nested, path=f"{path}[{index}]")
        return
    if not isinstance(value, str) or not value:
        return
    if value.startswith(("http://", "https://", "hf://", "s3://", "r2://")):
        return
    if (
        value.startswith("file://")
        or Path(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
    ):
        raise ValueError(f"{path} contains an absolute local path")


def build_release_manifest(
    identity: PublicationIdentity,
    bundle: PolicyBundle,
    *,
    release_version: str,
    published_at: str,
    source: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    youtube_url: str | None = None,
) -> dict[str, Any]:
    if not re.fullmatch(r"v[1-9][0-9]*", release_version):
        raise ValueError("release_version must be a sequential tag such as v1 or v2")
    expected_identity = publication_identity_from_policy_bundle(identity.goal, bundle)
    if expected_identity != identity:
        raise ValueError("release identity does not match model metadata")
    if identity.algorithm == "action-program" and evaluation.get("action_sampling") != "program":
        raise ValueError("action-program releases require program action sampling")
    manifest: dict[str, Any] = {
        "document_type": RELEASE_MANIFEST_DOCUMENT_TYPE,
        "format_version": RELEASE_MANIFEST_VERSION,
        "repo_naming_schema": REPO_NAMING_SCHEMA_VERSION,
        "repository": {"repo_id": build_model_repo_id(identity), **asdict(identity)},
        "release": {"version": release_version, "published_at": published_at},
        "model": publication_model_contract(bundle),
        "source": dict(source),
        "evaluation": dict(evaluation),
        "artifacts": dict(artifacts),
    }
    if youtube_url:
        manifest["release"]["youtube_url"] = youtube_url
    _assert_no_absolute_paths(manifest)
    return manifest


def _validate_release_manifest_v1(document: Mapping[str, Any], source: str) -> dict[str, Any]:
    manifest = validate_boundary(
        _ReleaseManifest,
        document,
        label=source,
        error_type=PolicyDocumentError,
    )
    if set(manifest.artifacts) != HASHED_RELEASE_FILES:
        raise PolicyDocumentError(
            f"{source}.artifacts must describe exactly: " + ", ".join(sorted(HASHED_RELEASE_FILES))
        )
    return deepcopy(dict(document))


def validate_release_bundle(root: Path) -> dict[str, Any]:
    actual_entries = {path.name for path in root.iterdir()}
    if actual_entries != HUGGINGFACE_RELEASE_FILES:
        missing = sorted(HUGGINGFACE_RELEASE_FILES - actual_entries)
        extra = sorted(actual_entries - HUGGINGFACE_RELEASE_FILES)
        raise ValueError(f"release file set mismatch; missing={missing}, extra={extra}")
    non_files = sorted(path.name for path in root.iterdir() if not path.is_file())
    if non_files:
        raise ValueError(f"release entries must all be regular files: {non_files}")
    manifest_path = root / "release_manifest.json"
    manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = preflight_document(
        manifest_value,
        source=str(manifest_path),
        expected_type=RELEASE_MANIFEST_DOCUMENT_TYPE,
        handlers={RELEASE_MANIFEST_VERSION: _validate_release_manifest_v1},
    )
    bundle = load_policy_bundle(root, source=str(root))
    card_text = (root / "README.md").read_text(encoding="utf-8")
    _assert_no_absolute_paths(manifest)
    repository = _require_mapping(manifest.get("repository"), label="manifest repository")
    identity = publication_identity_from_policy_bundle(repository.get("goal"), bundle)
    if repository.get("repo_id") != build_model_repo_id(identity):
        raise ValueError("release manifest repository id does not match model metadata")
    if int(manifest.get("repo_naming_schema") or 0) != REPO_NAMING_SCHEMA_VERSION:
        raise ValueError("release manifest has an unsupported repository naming schema")
    expected_model = publication_model_contract(bundle)
    if manifest.get("model") != expected_model:
        raise ValueError("release manifest model contract does not match model.json")
    expected_records = release_artifact_records(root)
    if manifest.get("artifacts") != expected_records:
        raise ValueError("release manifest artifact hashes or sizes do not match the bundle")
    evidence = _require_mapping(manifest.get("evaluation"), label="manifest evaluation")
    expected_evidence = {
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "recipe_sha256": bundle.recipe_sha256,
        "recipe_format_version": bundle.recipe["format_version"],
        "evaluation_contract_sha256": evaluation_contract_sha256(bundle.recipe),
    }
    for key, expected in expected_evidence.items():
        if evidence.get(key) != expected:
            raise ValueError(f"release evaluation {key} does not match the policy bundle")
    if evidence.get("exact_contract") is not True:
        raise ValueError("release evaluation evidence is not exact-contract")
    validate_model_card(card_text, manifest, bundle)
    return manifest
