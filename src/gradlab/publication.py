from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Literal

from huggingface_hub import ModelCard
from huggingface_hub.utils import validate_repo_id
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from gradlab.action_contract import action_contract_meanings
from gradlab.boundary_schema import (
    BoundaryModel,
    NonEmptyText,
    PositiveInt,
    Sha256,
    validate_boundary,
)
from gradlab.env_registry import game_family_for_environment
from gradlab.file_utils import file_sha256 as sha256_file
from gradlab.json_utils import canonical_json_text
from gradlab.metric_names import (
    EVAL_CHECKPOINT_STEP,
    EVAL_FULL_BY_START,
    EVAL_FULL_EPISODE_COMPLETED_COUNT,
    EVAL_FULL_EPISODE_RETURN_SHAPED_MEAN,
    EVAL_FULL_PROGRESS_X_MAX,
    EVAL_FULL_SUCCESS_ACROSS_STARTS_RATE_MEAN,
    EVAL_FULL_SUCCESS_ACROSS_STARTS_RATE_MIN,
)
from gradlab.env_registry import environment_spec
from gradlab.policy_bundle import (
    PolicyBundle,
    PolicyDocumentError,
    UnsupportedPolicyDocumentVersion,
    evaluation_contract_sha256,
    load_policy_bundle,
)
from gradlab.policy_registry import ALGORITHM_MODEL_CLASSES
from gradlab.validation import require_mapping as _require_mapping


HUGGINGFACE_NAMESPACE = "tsilva"
REPO_NAMING_SCHEMA_VERSION = 1
RELEASE_MANIFEST_DOCUMENT_TYPE = "gradlab.release_manifest"
RELEASE_MANIFEST_VERSION = 2
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


class _ReleaseManifestV1(BoundaryModel):
    document_type: Literal[RELEASE_MANIFEST_DOCUMENT_TYPE]
    format_version: Literal[1]
    repo_naming_schema: Literal[REPO_NAMING_SCHEMA_VERSION]
    repository: _ReleaseRepository
    release: _ReleaseDetails
    model: _ReleaseModel
    source: _ReleaseSource
    evaluation: _ReleaseEvaluation
    artifacts: dict[str, _ArtifactRecord]


class _ReplayExecution(BoundaryModel):
    source: Any
    qualified_environment_id: NonEmptyText
    provider_id: NonEmptyText
    provider_version: Any
    environment_hash: NonEmptyText
    runtime_versions: Any
    runtime_image_digest: Any
    asset: Any
    execution_target: NonEmptyText
    device_type: NonEmptyText
    contract_mode: NonEmptyText
    overrides: Any
    seed: Any


class _ReleaseReplay(BoundaryModel):
    capture_id: NonEmptyText
    capture_fence_sha256: Sha256
    run_id: NonEmptyText
    checkpoint_id: NonEmptyText
    checkpoint_sha256: Sha256
    recipe_sha256: Sha256
    episode: Any
    seed: Any
    start_id: Any
    sampling_mode: NonEmptyText
    steps: PositiveInt
    return_value: Any
    max_x_pos: Any
    outcome: NonEmptyText
    success: bool
    boundary_role: Literal["terminal_observation"]
    contract: Any
    execution: _ReplayExecution
    media: Any


class _ReleasePublisher(BoundaryModel):
    request_fingerprint: Sha256
    huggingface_username: NonEmptyText
    huggingface_namespace: NonEmptyText
    youtube_channel_id: NonEmptyText
    youtube_channel_title: NonEmptyText
    youtube_privacy: Literal["public", "unlisted", "private"]


class _ReleaseManifestV2(BoundaryModel):
    document_type: Literal[RELEASE_MANIFEST_DOCUMENT_TYPE]
    format_version: Literal[2]
    repo_naming_schema: Literal[REPO_NAMING_SCHEMA_VERSION]
    repository: _ReleaseRepository
    release: _ReleaseDetails
    model: _ReleaseModel
    source: _ReleaseSource
    evaluation: _ReleaseEvaluation
    replay: _ReleaseReplay
    publication: _ReleasePublisher
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
    if not isinstance(action_contract, Mapping) or action_contract.get("schema_version") is None:
        raise ValueError("publication requires a structured runtime action contract")
    provider_contract = _require_mapping(
        action_contract.get("provider"),
        label="publication runtime action contract provider",
    )
    policy_contract = _require_mapping(
        action_contract.get("policy"),
        label="publication runtime action contract policy",
    )
    raw_codec = policy_contract.get("codec")
    codec = dict(raw_codec) if isinstance(raw_codec, Mapping) else {}
    action_set = str(
        provider_contract.get("preset") or provider_contract.get("mode") or action_set
    ).strip()
    if codec.get("type") == "vizdoom_shared_multidiscrete_v1":
        action_set = "vizdoom-shared-multidiscrete-v1"
    if not action_contract_meanings(action_contract):
        raise ValueError(
            "publication runtime action contract requires semantic IDs "
            "for every legal policy action"
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
    family = game_family_for_environment(provider, game, require_registered=True)
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
        raise ValueError(
            "model.json provenance must contain a saved runtime action contract"
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
        else {"route"}
        if algorithm_id == "cell-graph"
        else ({"stochastic", "deterministic"} if allow_deterministic else {"stochastic"})
    )
    if action_sampling not in allowed_sampling:
        expected = " or ".join(sorted(allowed_sampling))
        raise ValueError(f"release evaluation action_sampling must be {expected}")
    protocol = str(evaluation.get("protocol") or "full").strip().lower()
    if protocol != "full":
        raise ValueError("release evaluation protocol must be 'full'")
    checkpoint_step = _required_int(
        _first_present(
            evaluation,
            "checkpoint_step",
            EVAL_CHECKPOINT_STEP,
        ),
        label="evaluation checkpoint_step",
    )
    checkpoint_artifact = _required_text(
        _first_present(
            evaluation,
            "checkpoint_artifact",
            "eval/full/checkpoint/artifact",
        ),
        label="evaluation checkpoint_artifact",
    )
    episodes = _required_int(
        _first_present(
            evaluation,
            "episodes",
            EVAL_FULL_EPISODE_COMPLETED_COUNT,
        ),
        label="evaluation episodes",
    )
    if episodes <= 0:
        raise ValueError("evaluation episodes must be positive")
    success_rate_min = _required_rate(
        _first_present(
            evaluation,
            "success_rate_min",
            EVAL_FULL_SUCCESS_ACROSS_STARTS_RATE_MIN,
        ),
        label="evaluation success_rate_min",
    )
    success_rate_mean = _required_rate(
        _first_present(
            evaluation,
            "success_rate_mean",
            EVAL_FULL_SUCCESS_ACROSS_STARTS_RATE_MEAN,
        ),
        label="evaluation success_rate_mean",
    )
    return_mean = _required_float(
        _first_present(
            evaluation,
            "return_mean",
            EVAL_FULL_EPISODE_RETURN_SHAPED_MEAN,
        ),
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
    is_cell_graph = algorithm == "cell-graph"
    is_gradlab_policy = is_action_program or is_cell_graph
    provenance = _require_mapping(bundle.model.get("provenance"), label="model.json provenance")
    producer = (
        _required_text(
            provenance.get("search_algorithm_id"),
            label="model provenance search_algorithm_id",
        )
        if is_gradlab_policy
        else ""
    )
    library_name = "gradlab" if is_gradlab_policy else "stable-baselines3"
    library_tag = "gradlab-policy" if is_gradlab_policy else "stable-baselines3"
    policy_description = (
        f"GradLab open-loop action program for `{game}` `{goal}`, produced by "
        f"`{producer}` and trained and evaluated with"
        if is_action_program
        else f"GradLab closed-loop semantic cell graph for `{game}` `{goal}`, "
        f"produced by `{producer}` and trained and evaluated with"
        if is_cell_graph
        else f"Stable-Baselines3 {algorithm.upper()} policy for `{game}` `{goal}`, "
        "trained and evaluated with"
    )
    model_file_description = (
        "Portable GradLab open-loop action program"
        if is_action_program
        else "Portable GradLab closed-loop semantic cell graph"
        if is_cell_graph
        else "Stable-Baselines3 policy checkpoint"
    )
    run_name = _required_text(source.get("run_name"), label="manifest source.run_name")
    run_value = f"[{_markdown_value(run_name)}]({wandb_url})" if wandb_url else run_name
    replay_value = manifest.get("replay")
    replay = replay_value if isinstance(replay_value, Mapping) else None
    replay_execution = (
        replay.get("execution") if isinstance(replay, Mapping) else None
    )
    template = (
        "model_card_v2.md.j2"
        if int(manifest.get("format_version") or 1) == 2
        else "model_card.md.j2"
    )
    card = _MODEL_CARD_TEMPLATE_ENV.get_template(template).render(
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
                canonical_json_text(preprocessing, ensure_ascii=True)
            ),
            "action": _markdown_value(canonical_json_text(action, ensure_ascii=True)),
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
            "replay_outcome": (
                _markdown_value(replay.get("outcome")) if replay is not None else ""
            ),
            "replay_success": bool(replay.get("success")) if replay is not None else False,
            "replay_seed": replay.get("seed") if replay is not None else None,
            "replay_start_id": (
                _markdown_value(replay.get("start_id")) if replay is not None else ""
            ),
            "replay_steps": replay.get("steps") if replay is not None else None,
            "replay_return": replay.get("return_value") if replay is not None else None,
            "replay_contract_mode": (
                _markdown_value((replay.get("contract") or {}).get("mode"))
                if replay is not None and isinstance(replay.get("contract"), Mapping)
                else ""
            ),
            "replay_runtime": (
                _markdown_value(canonical_json_text(replay_execution, ensure_ascii=True))
                if isinstance(replay_execution, Mapping)
                else ""
            ),
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
            "stream=codec_name,codec_tag_string,pix_fmt,nb_read_frames,width,height,r_frame_rate:format=duration",
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
    rate = str(stream.get("r_frame_rate") or "0/1")
    numerator, separator, denominator = rate.partition("/")
    fps = float(numerator) / float(denominator) if separator and float(denominator) else 0.0
    return {
        "duration_seconds": duration,
        "frames": frames,
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "fps": fps,
        **expected,
    }


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
    replay: Mapping[str, Any] | None = None,
    publication: Mapping[str, Any] | None = None,
    format_version: int = RELEASE_MANIFEST_VERSION,
) -> dict[str, Any]:
    if not re.fullmatch(r"v[1-9][0-9]*", release_version):
        raise ValueError("release_version must be a sequential tag such as v1 or v2")
    expected_identity = publication_identity_from_policy_bundle(identity.goal, bundle)
    if expected_identity != identity:
        raise ValueError("release identity does not match model metadata")
    if identity.algorithm == "action-program" and evaluation.get("action_sampling") != "program":
        raise ValueError("action-program releases require program action sampling")
    if identity.algorithm == "cell-graph" and evaluation.get("action_sampling") != "route":
        raise ValueError("cell-graph releases require route action sampling")
    if format_version not in {1, 2}:
        raise ValueError("release manifest format_version must be 1 or 2")
    if format_version == 2 and not isinstance(replay, Mapping):
        raise ValueError("release manifest v2 requires replay provenance")
    if format_version == 2 and not isinstance(publication, Mapping):
        raise ValueError("release manifest v2 requires publication provenance")
    manifest: dict[str, Any] = {
        "document_type": RELEASE_MANIFEST_DOCUMENT_TYPE,
        "format_version": format_version,
        "repo_naming_schema": REPO_NAMING_SCHEMA_VERSION,
        "repository": {"repo_id": build_model_repo_id(identity), **asdict(identity)},
        "release": {"version": release_version, "published_at": published_at},
        "model": publication_model_contract(bundle),
        "source": dict(source),
        "evaluation": dict(evaluation),
        "artifacts": dict(artifacts),
    }
    if format_version == 2:
        assert isinstance(replay, Mapping)
        assert isinstance(publication, Mapping)
        manifest["replay"] = dict(replay)
        manifest["publication"] = dict(publication)
    if youtube_url:
        manifest["release"]["youtube_url"] = youtube_url
    _assert_no_absolute_paths(manifest)
    return manifest


def release_replay_from_capture(capture: Mapping[str, Any]) -> dict[str, Any]:
    from gradlab.play_capture import validate_capture_document

    document = validate_capture_document(capture)
    replay = _require_mapping(document.get("replay"), label="capture replay")
    return {
        "capture_id": document["capture_id"],
        "capture_fence_sha256": document["capture_fence_sha256"],
        "run_id": document["run_id"],
        "checkpoint_id": document["checkpoint_id"],
        "checkpoint_sha256": document["checkpoint_sha256"],
        "recipe_sha256": document["recipe_sha256"],
        "episode": document["episode"],
        "seed": document["seed"],
        "start_id": document.get("start_id"),
        "sampling_mode": document["sampling_mode"],
        "steps": document["steps"],
        "return_value": document["return"],
        "max_x_pos": document["max_x_pos"],
        "outcome": document["outcome"],
        "success": document["success"],
        "boundary_role": document["boundary_role"],
        "contract": deepcopy(document["contract"]),
        "execution": deepcopy(document["execution"]),
        "media": deepcopy(dict(replay)),
    }


def _validate_release_manifest_v1(document: Mapping[str, Any], source: str) -> dict[str, Any]:
    manifest = validate_boundary(
        _ReleaseManifestV1,
        document,
        label=source,
        error_type=PolicyDocumentError,
    )
    if set(manifest.artifacts) != HASHED_RELEASE_FILES:
        raise PolicyDocumentError(
            f"{source}.artifacts must describe exactly: " + ", ".join(sorted(HASHED_RELEASE_FILES))
        )
    return deepcopy(dict(document))


def _validate_release_manifest_v2(document: Mapping[str, Any], source: str) -> dict[str, Any]:
    manifest = validate_boundary(
        _ReleaseManifestV2,
        document,
        label=source,
        error_type=PolicyDocumentError,
    )
    if set(manifest.artifacts) != HASHED_RELEASE_FILES:
        raise PolicyDocumentError(
            f"{source}.artifacts must describe exactly: " + ", ".join(sorted(HASHED_RELEASE_FILES))
        )
    if not str(manifest.replay.capture_id).startswith("capture-"):
        raise PolicyDocumentError(f"{source}.replay.capture_id is invalid")
    media = manifest.replay.media
    if not isinstance(media, Mapping) or int(media.get("frames") or 0) != manifest.replay.steps + 1:
        raise PolicyDocumentError(f"{source}.replay.media frames must equal replay steps + 1")
    return deepcopy(dict(document))


def validate_release_manifest_document(
    document: Mapping[str, Any],
    *,
    source: str = "release manifest",
) -> dict[str, Any]:
    if document.get("document_type") != RELEASE_MANIFEST_DOCUMENT_TYPE:
        raise PolicyDocumentError(f"{source} has an invalid document_type")
    version = document.get("format_version")
    if version == 1:
        return _validate_release_manifest_v1(document, source)
    if version == 2:
        return _validate_release_manifest_v2(document, source)
    raise UnsupportedPolicyDocumentVersion(
        source=source,
        document_type=RELEASE_MANIFEST_DOCUMENT_TYPE,
        format_version=version,
        supported_versions=[1, 2],
    )


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
    if not isinstance(manifest_value, Mapping):
        raise PolicyDocumentError(f"{manifest_path} must contain a JSON object")
    manifest = validate_release_manifest_document(manifest_value, source=str(manifest_path))
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
    if int(manifest.get("format_version") or 0) == 2:
        replay = _require_mapping(manifest.get("replay"), label="manifest replay")
        media = _require_mapping(replay.get("media"), label="manifest replay.media")
        replay_record = _require_mapping(
            expected_records.get("replay.mp4"),
            label="release replay.mp4 artifact",
        )
        if media.get("sha256") != replay_record.get("sha256"):
            raise ValueError("release replay sha256 does not match replay.mp4")
        if int(media.get("size_bytes") or 0) != int(replay_record.get("size_bytes") or 0):
            raise ValueError("release replay size does not match replay.mp4")
        replay_probe = verify_replay(root / "replay.mp4")
        for key in ("frames", "width", "height"):
            if int(media.get(key) or 0) != int(replay_probe.get(key) or 0):
                raise ValueError(f"release replay {key} does not match replay.mp4")
        if abs(float(media.get("fps") or 0.0) - float(replay_probe.get("fps") or 0.0)) > 1e-6:
            raise ValueError("release replay fps does not match replay.mp4")
    validate_model_card(card_text, manifest, bundle)
    return manifest
