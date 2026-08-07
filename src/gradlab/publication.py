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
from gradlab.file_utils import file_sha256 as sha256_file
from gradlab.json_utils import canonical_json_sha256, canonical_json_text
from gradlab.metric_names import (
    EVAL_CHECKPOINT_STEP,
    EVAL_FULL_EPISODE_RETURN_SHAPED_MEAN,
    EVAL_FULL_OUTCOME_SUCCESS_STARTS_RATE_MEAN,
    EVAL_FULL_OUTCOME_SUCCESS_STARTS_RATE_MIN,
    EVAL_FULL_PROGRESS_X_MAX,
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
REPO_NAMING_SCHEMA_VERSION = 3
RELEASE_MANIFEST_DOCUMENT_TYPE = "gradlab.release_manifest"
RELEASE_MANIFEST_VERSION = 3
EVALUATION_EVIDENCE_DOCUMENT_TYPE = "gradlab.evaluation_evidence"
EVALUATION_EVIDENCE_VERSION = 1
HUGGINGFACE_RELEASE_FILES = frozenset(
    {
        ".gitattributes",
        "README.md",
        "LICENSE",
        "model.zip",
        "model.json",
        "recipe.json",
        "evaluation_evidence.json",
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
    canonical_environment_id: NonEmptyText
    goal_id: NonEmptyText
    trainer: NonEmptyText
    trainer_slug: NonEmptyText
    algorithm: NonEmptyText
    lineage_digest: Sha256
    lineage_prefix: NonEmptyText


class _ReleaseDetails(BoundaryModel):
    version: NonEmptyText
    checkpoint_tag: NonEmptyText
    published_at: NonEmptyText
    youtube_url: NonEmptyText | None = None
    correction_note: Any = None


class _ReleaseModel(BoundaryModel):
    trainer: Any
    algorithm_id: Any
    model_class: Any
    compatibility: Any
    library_name: Any
    qualified_env_id: Any
    environment_hash: Any
    preprocessing: Any
    action: Any
    action_semantics: Any
    model_inputs: Any


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
    model_document_url: Any = None
    recipe_document_url: Any = None


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
    accepted: bool
    evidence_file: Literal["evaluation_evidence.json"]
    evidence_sha256: Sha256
    acceptance: Any
    ranking: Any


class _ArtifactRecord(BoundaryModel):
    sha256: Sha256
    size_bytes: PositiveInt


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


class _ReleaseManifestV3(BoundaryModel):
    document_type: Literal[RELEASE_MANIFEST_DOCUMENT_TYPE]
    format_version: Literal[3]
    repo_naming_schema: Literal[REPO_NAMING_SCHEMA_VERSION]
    repository: _ReleaseRepository
    release: _ReleaseDetails
    model: _ReleaseModel
    source: _ReleaseSource
    evaluation: _ReleaseEvaluation
    replay: _ReleaseReplay
    publication: _ReleasePublisher
    containers: Any
    comparison: Any
    featured: bool
    artifacts: dict[str, _ArtifactRecord]


@dataclass(frozen=True)
class PublicationIdentity:
    canonical_environment_id: str
    goal_id: str
    trainer: str
    trainer_slug: str
    algorithm: str
    lineage_digest: str

    @property
    def lineage_prefix(self) -> str:
        return self.lineage_digest[:8]

    @property
    def repo_name(self) -> str:
        return f"{self.goal_id}_{self.trainer_slug}-{self.algorithm}_{self.lineage_prefix}"


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


def publication_trainer(bundle: PolicyBundle) -> dict[str, str]:
    policy = _require_mapping(bundle.model.get("policy"), label="model.json policy")
    backend_id = str(policy.get("training_backend_id") or "").strip()
    model_class = str(policy.get("model_class") or "").strip()
    if backend_id.startswith("gradlab.") or model_class.startswith("gradlab."):
        return {
            "trainer": "GradLab",
            "trainer_slug": "gradlab",
            "compatibility": "SB3-compatible",
            "library_name": "gradlab",
        }
    if backend_id.startswith("sb3.") or model_class.startswith("stable_baselines3."):
        return {
            "trainer": "Stable-Baselines3",
            "trainer_slug": "stable-baselines3",
            "compatibility": "Stable-Baselines3",
            "library_name": "stable-baselines3",
        }
    raise ValueError("model policy does not identify a supported trainer")


_NON_LINEAGE_PROVIDER_ARGS = frozenset(
    {
        "n_envs",
        "num_threads",
        "record",
        "render_mode",
        "rom_path",
        "rom_asset",
    }
)


def _semantic_provider_args(value: object) -> dict[str, Any]:
    provider_args = _require_mapping(value, label="recipe environment.provider_args")
    return {
        str(key): deepcopy(nested)
        for key, nested in provider_args.items()
        if str(key) not in _NON_LINEAGE_PROVIDER_ARGS
    }


def policy_lineage_contract(
    goal_id: object,
    bundle: PolicyBundle,
) -> dict[str, Any]:
    recipe = _require_mapping(bundle.recipe.get("recipe"), label="recipe.json recipe")
    environment = _require_mapping(
        recipe.get("environment"), label="recipe.json recipe.environment"
    )
    provider, game = _provider_and_environment(environment.get("env_id"))
    spec = environment_spec(provider, game)
    task = _require_mapping(environment.get("task"), label="recipe environment.task")
    train = _require_mapping(recipe.get("train"), label="recipe train")
    policy = _require_mapping(bundle.model.get("policy"), label="model.json policy")
    provenance = _require_mapping(bundle.model.get("provenance"), label="model.json provenance")
    training_metadata = _require_mapping(
        provenance.get("training_metadata"), label="model provenance.training_metadata"
    )
    action_contract = _require_mapping(
        training_metadata.get("action_contract"),
        label="model provenance.training_metadata.action_contract",
    )
    if not action_contract_meanings(action_contract):
        raise ValueError("publication requires complete saved action semantics")
    provider_action = _require_mapping(
        action_contract.get("provider"), label="action contract provider"
    )
    semantic_provider_action = {
        key: deepcopy(value)
        for key, value in provider_action.items()
        if key != "provider_id"
    }
    policy_execution = training_metadata.get("policy_execution_contract")
    policy_execution_value = (
        dict(policy_execution) if isinstance(policy_execution, Mapping) else {}
    )
    trainer = publication_trainer(bundle)
    value_contract = _require_mapping(
        recipe.get("value_contract"), label="recipe value_contract"
    )
    return {
        "schema_version": 1,
        "canonical_environment_id": spec.spec_id,
        "goal_id": normalize_publication_component(goal_id, label="goal id"),
        "trainer": trainer["trainer"],
        "algorithm": normalize_algorithm_id(policy.get("algorithm_id")),
        "architecture": {
            "recipe": deepcopy(train.get("policy_model")),
            "saved_execution": deepcopy(policy_execution_value.get("policy_model")),
        },
        "observations": {
            "preprocessing": deepcopy(environment.get("preprocessing")),
            "model_inputs": deepcopy(task.get("model_inputs")),
            "policy_execution_model_inputs": deepcopy(
                policy_execution_value.get("model_inputs")
            ),
            "role_inputs": deepcopy(policy_execution_value.get("role_inputs")),
            "provider_args": _semantic_provider_args(environment.get("provider_args")),
        },
        "action_semantics": {
            "requested": deepcopy(action_contract.get("requested")),
            "provider": semantic_provider_action,
            "policy": deepcopy(action_contract.get("policy")),
        },
        "reward": deepcopy(task.get("reward")),
        "discount": deepcopy(value_contract.get("discount")),
        "signals": deepcopy(task.get("signals")),
        "events": deepcopy(task.get("events")),
        "starts": {
            "state": deepcopy(environment.get("state")),
            "start": deepcopy(task.get("start")),
            "starts": deepcopy(task.get("starts")),
        },
        "episode_boundaries": deepcopy(task.get("termination")),
    }


def publication_identity_from_policy_bundle(
    goal_id: object,
    bundle: PolicyBundle,
) -> PublicationIdentity:
    recipe = _require_mapping(bundle.recipe.get("recipe"), label="recipe.json recipe")
    environment = _require_mapping(
        recipe.get("environment"), label="recipe.json recipe.environment"
    )
    provider, game = _provider_and_environment(environment.get("env_id"))
    policy = _require_mapping(bundle.model.get("policy"), label="model.json policy")
    algorithm = normalize_algorithm_id(policy.get("algorithm_id"))
    validate_algorithm_model_class(algorithm, policy.get("model_class"))
    spec = environment_spec(provider, game)
    trainer = publication_trainer(bundle)
    lineage_contract = policy_lineage_contract(goal_id, bundle)
    return PublicationIdentity(
        canonical_environment_id=spec.spec_id,
        goal_id=normalize_publication_component(goal_id, label="goal id"),
        trainer=trainer["trainer"],
        trainer_slug=trainer["trainer_slug"],
        algorithm=algorithm,
        lineage_digest=canonical_json_sha256(lineage_contract),
    )


def build_model_repo_id(identity: PublicationIdentity) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", identity.lineage_digest):
        raise ValueError("publication lineage_digest must be a full lowercase SHA-256")
    for field, value in asdict(identity).items():
        normalized = normalize_publication_component(value, label=field)
        if normalized != value:
            raise ValueError(
                f"publication identity {field} must already be canonical: "
                f"expected {normalized!r}, got {value!r}"
            )
    repo_name = identity.repo_name
    repo_id = f"{HUGGINGFACE_NAMESPACE}/{repo_name}"
    validate_repo_id(repo_id)
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
        if not isinstance(raw, Mapping):
            raise ValueError(f"evaluation by_start row {index} has an unsupported shape")
        row = dict(raw)
        start_id = _required_text(row.get("start_id"), label=f"by_start[{index}].start_id")
        failure_reasons = row.get("failure_reasons")
        if not isinstance(failure_reasons, Mapping):
            raise ValueError(f"by_start[{index}].failure_reasons must be a mapping")
        normalized = {
            "start_id": start_id,
            "episode_count": _required_int(
                row.get("episode_count"), label=f"by_start[{index}].episode_count"
            ),
            "success_count": _required_int(
                row.get("success_count"), label=f"by_start[{index}].success_count"
            ),
            "success_rate": _required_rate(
                row.get("success_rate"), label=f"by_start[{index}].success_rate"
            ),
            "shaped_return_mean": _required_float(
                row.get("shaped_return_mean"),
                label=f"by_start[{index}].shaped_return_mean",
            ),
            "failure_reasons": {
                _required_text(reason, label=f"by_start[{index}].failure_reasons reason"): (
                    _required_int(count, label=f"by_start[{index}].failure_reasons count")
                )
                for reason, count in failure_reasons.items()
            },
        }
        if normalized["episode_count"] <= 0:
            raise ValueError(f"by_start[{index}].episode_count must be positive")
        if normalized["success_count"] > normalized["episode_count"]:
            raise ValueError(f"by_start[{index}].success_count exceeds episode_count")
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
        _first_present(evaluation, "episodes"),
        label="evaluation episodes",
    )
    if episodes <= 0:
        raise ValueError("evaluation episodes must be positive")
    success_rate_min = _required_rate(
        _first_present(
            evaluation,
            "success_rate_min",
            EVAL_FULL_OUTCOME_SUCCESS_STARTS_RATE_MIN,
        ),
        label="evaluation success_rate_min",
    )
    success_rate_mean = _required_rate(
        _first_present(
            evaluation,
            "success_rate_mean",
            EVAL_FULL_OUTCOME_SUCCESS_STARTS_RATE_MEAN,
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
    by_start = _normalize_by_start_rows(_first_present(evaluation, "by_start"))
    if sum(int(row["episode_count"]) for row in by_start) != episodes:
        raise ValueError("evaluation episodes must equal the sum of by_start episode_count")
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
    provenance = _require_mapping(bundle.model.get("provenance"), label="model provenance")
    training_metadata = _require_mapping(
        provenance.get("training_metadata"), label="model provenance.training_metadata"
    )
    action_contract = _require_mapping(
        training_metadata.get("action_contract"), label="saved action contract"
    )
    trainer = publication_trainer(bundle)
    return {
        "trainer": trainer["trainer"],
        "algorithm_id": policy.get("algorithm_id"),
        "model_class": policy.get("model_class"),
        "compatibility": trainer["compatibility"],
        "library_name": trainer["library_name"],
        "qualified_env_id": environment.get("env_id"),
        "environment_hash": recipe.get("environment_hash"),
        "preprocessing": environment.get("preprocessing"),
        "action": task.get("action"),
        "action_semantics": deepcopy(action_contract),
        "model_inputs": deepcopy(task.get("model_inputs")),
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


def _metric_value(value: object, unit: object) -> str:
    if str(unit) == "fraction":
        return _percent(value)
    if isinstance(value, bool):
        return "pass" if value else "fail"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def release_comparison(
    current: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if previous is None:
        return {"comparable": False, "reason": "no prior release selected"}
    checks = (
        (
            "lineage",
            (current.get("repository") or {}).get("lineage_digest"),
            (previous.get("repository") or {}).get("lineage_digest"),
        ),
        (
            "run",
            (current.get("source") or {}).get("run_id"),
            (previous.get("source") or {}).get("run_id"),
        ),
        (
            "seed",
            (current.get("source") or {}).get("seed"),
            (previous.get("source") or {}).get("seed"),
        ),
        (
            "evaluation contract",
            (current.get("evaluation") or {}).get("evaluation_contract_sha256"),
            (previous.get("evaluation") or {}).get("evaluation_contract_sha256"),
        ),
    )
    for label, left, right in checks:
        if left != right:
            return {"comparable": False, "reason": f"{label} differs"}
    current_ranking = (current.get("evaluation") or {}).get("ranking") or {}
    previous_ranking = (previous.get("evaluation") or {}).get("ranking") or {}
    return {
        "comparable": True,
        "reason": "lineage, run, seed, and evaluation contract match",
        "previous_version": (previous.get("release") or {}).get("version"),
        "ranking_before": deepcopy(previous_ranking.get("outcomes") or []),
        "ranking_after": deepcopy(current_ranking.get("outcomes") or []),
    }


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
    goal = _required_text(repository.get("goal_id"), label="manifest repository.goal_id")
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
    checkpoint_step = _required_int(
        evaluation.get("checkpoint_step"), label="manifest evaluation.checkpoint_step"
    )
    episodes = _required_int(evaluation.get("episodes"), label="manifest evaluation.episodes")
    preprocessing = _require_mapping(model.get("preprocessing"), label="manifest preprocessing")
    action = _require_mapping(
        model.get("action_semantics"), label="manifest action semantics"
    )
    run_id = str(source.get("run_id") or "").strip()
    project = str(source.get("wandb_project") or "").strip()
    wandb_url = f"https://wandb.ai/tsilva/{project}/runs/{run_id}" if project and run_id else ""
    if not re.fullmatch(r"v[1-9][0-9]*", version):
        raise ValueError("model cards require a sequential release version")
    commit = _required_text(source.get("commit"), label="manifest source.commit")
    model_ref = f"hf://{repo_id}@{version}"
    youtube_value = f"[Watch on YouTube]({youtube_url})" if youtube_url else "Not available"
    manifest_purpose = "Release identity, evaluation evidence, and artifact hashes"
    trainer = _required_text(model.get("trainer"), label="manifest model.trainer")
    library_name = _required_text(
        model.get("library_name"), label="manifest model.library_name"
    )
    compatibility = _required_text(
        model.get("compatibility"), label="manifest model.compatibility"
    )
    materialized_goal = _require_mapping(
        bundle.recipe.get("recipe"), label="recipe materialized recipe"
    ).get("goal")
    goal_title = (
        str(materialized_goal.get("title") or goal)
        if isinstance(materialized_goal, Mapping)
        else goal
    )
    acceptance = _require_mapping(
        evaluation.get("acceptance"), label="manifest evaluation.acceptance"
    )
    ranking = _require_mapping(
        evaluation.get("ranking"), label="manifest evaluation.ranking"
    )
    acceptance_rows = "\n".join(
        f"| `{_markdown_value(row['metric'])}` | {_markdown_value(row['label'])} | "
        f"{_metric_value(row['value'], row['unit'])} | `{row['operator']} {_markdown_value(row['threshold'])}` | "
        f"{'Pass' if row['passed'] else 'Fail'} |"
        for row in acceptance.get("outcomes") or ()
    )
    ranking_rows = "\n".join(
        f"| `{_markdown_value(row['metric'])}` | {_markdown_value(row['label'])} | "
        f"{_metric_value(row['value'], row['unit'])} | `{row['direction']}` |"
        for row in ranking.get("outcomes") or ()
    )
    quick_start_lines = [
        "```bash",
        "uvx --from gradlab gradlab play " + model_ref,
        "```",
    ]
    if provider == "stable-retro-turbo":
        quick_start_lines = [
            "Import a legally obtained compatible game image, then play the immutable release:",
            "",
            "```bash",
            f"uvx --from gradlab gradlab rom import ~/roms --game {game}",
            "uvx --from gradlab gradlab play " + model_ref,
            "```",
        ]
    quick_start = "\n".join(quick_start_lines)
    run_name = _required_text(source.get("run_name"), label="manifest source.run_name")
    run_value = f"[{_markdown_value(run_name)}]({wandb_url})" if wandb_url else run_name
    replay_value = manifest.get("replay")
    replay = replay_value if isinstance(replay_value, Mapping) else None
    replay_execution = (
        replay.get("execution") if isinstance(replay, Mapping) else None
    )
    comparison = _require_mapping(manifest.get("comparison"), label="manifest comparison")
    card = _MODEL_CARD_TEMPLATE_ENV.get_template("model_card_v3.md.j2").render(
        {
            "library_name": library_name,
            "algorithm": algorithm,
            "algorithm_upper": algorithm.upper(),
            "provider": provider,
            "game": game,
            "goal": goal,
            "goal_title": goal_title,
            "trainer": trainer,
            "model_class": _required_text(model.get("model_class"), label="model class"),
            "compatibility": compatibility,
            "checkpoint_step": checkpoint_step,
            "action_sampling": action_sampling,
            "episodes": episodes,
            "acceptance_rows": acceptance_rows,
            "ranking_rows": ranking_rows,
            "accepted": bool(evaluation.get("accepted")),
            "version": version,
            "checkpoint_tag": _required_text(
                release.get("checkpoint_tag"), label="release checkpoint_tag"
            ),
            "youtube_value": youtube_value,
            "quick_start": quick_start,
            "model_ref": model_ref,
            "qualified_env_id": qualified_env_id,
            "canonical_environment_id": repository.get("canonical_environment_id"),
            "environment_hash": _markdown_value(model.get("environment_hash")),
            "preprocessing": _markdown_value(
                canonical_json_text(
                    {
                        "preprocessing": preprocessing,
                        "model_inputs": model.get("model_inputs"),
                    },
                    ensure_ascii=True,
                )
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
            "manifest_purpose": manifest_purpose,
            "lineage_digest": repository.get("lineage_digest"),
            "comparison_status": (
                "Comparable" if comparison.get("comparable") else "Not compared"
            ),
            "comparison_reason": _markdown_value(comparison.get("reason")),
            "evidence_sha256": evaluation.get("evidence_sha256"),
            "environment_container": (manifest.get("containers") or {}).get("environment"),
            "featured": bool(manifest.get("featured")),
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
    evaluation_evidence: Mapping[str, Any] | None = None,
    comparison: Mapping[str, Any] | None = None,
    featured: bool = False,
    correction_note: str | None = None,
    format_version: int = RELEASE_MANIFEST_VERSION,
) -> dict[str, Any]:
    if not re.fullmatch(r"v[1-9][0-9]*", release_version):
        raise ValueError("release_version must be a sequential tag such as v1 or v2")
    expected_identity = publication_identity_from_policy_bundle(identity.goal_id, bundle)
    if expected_identity != identity:
        raise ValueError("release identity does not match model metadata")
    if identity.algorithm == "action-program" and evaluation.get("action_sampling") != "program":
        raise ValueError("action-program releases require program action sampling")
    if identity.algorithm == "cell-graph" and evaluation.get("action_sampling") != "route":
        raise ValueError("cell-graph releases require route action sampling")
    if format_version != RELEASE_MANIFEST_VERSION:
        raise ValueError("release manifest format_version must be 3")
    if not isinstance(replay, Mapping):
        raise ValueError("release manifest v3 requires replay provenance")
    if not isinstance(publication, Mapping):
        raise ValueError("release manifest v3 requires publication provenance")
    if not isinstance(evaluation_evidence, Mapping):
        raise ValueError("release manifest v3 requires evaluation_evidence.json")
    from gradlab.publication_evidence import validate_evaluation_evidence_document

    evidence_document = validate_evaluation_evidence_document(evaluation_evidence)
    evidence_hash = canonical_json_sha256(evidence_document)
    evidence_identity = _require_mapping(
        evidence_document.get("identity"), label="evaluation evidence identity"
    )
    if int(evidence_identity.get("checkpoint_step") or -1) != int(
        evaluation.get("checkpoint_step") or -2
    ):
        raise ValueError("evaluation evidence checkpoint step disagrees with release")
    evaluation_value = {
        **dict(evaluation),
        "accepted": True,
        "evidence_file": "evaluation_evidence.json",
        "evidence_sha256": evidence_hash,
        "acceptance": deepcopy(evidence_document["acceptance"]),
        "ranking": deepcopy(evidence_document["ranking"]),
    }
    checkpoint_step = int(evaluation_value["checkpoint_step"])
    environment_container = f"GradLab — {identity.canonical_environment_id}"
    manifest: dict[str, Any] = {
        "document_type": RELEASE_MANIFEST_DOCUMENT_TYPE,
        "format_version": format_version,
        "repo_naming_schema": REPO_NAMING_SCHEMA_VERSION,
        "repository": {
            "repo_id": build_model_repo_id(identity),
            **asdict(identity),
            "lineage_prefix": identity.lineage_prefix,
        },
        "release": {
            "version": release_version,
            "checkpoint_tag": f"checkpoint-{checkpoint_step}",
            "published_at": published_at,
            **({"correction_note": correction_note} if correction_note else {}),
        },
        "model": publication_model_contract(bundle),
        "source": dict(source),
        "evaluation": evaluation_value,
        "replay": dict(replay),
        "publication": dict(publication),
        "containers": {
            "environment": environment_container,
            "featured": "GradLab — Featured Research",
        },
        "comparison": dict(comparison or {"comparable": False, "reason": "no prior release selected"}),
        "featured": bool(featured),
        "artifacts": dict(artifacts),
    }
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


def _validate_release_manifest_v3(document: Mapping[str, Any], source: str) -> dict[str, Any]:
    manifest = validate_boundary(
        _ReleaseManifestV3,
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
    if version == RELEASE_MANIFEST_VERSION:
        return _validate_release_manifest_v3(document, source)
    raise UnsupportedPolicyDocumentVersion(
        source=source,
        document_type=RELEASE_MANIFEST_DOCUMENT_TYPE,
        format_version=version,
        supported_versions=[RELEASE_MANIFEST_VERSION],
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
    identity = publication_identity_from_policy_bundle(repository.get("goal_id"), bundle)
    if repository.get("repo_id") != build_model_repo_id(identity):
        raise ValueError("release manifest repository id does not match model metadata")
    if repository.get("lineage_digest") != identity.lineage_digest:
        raise ValueError("release manifest full lineage digest does not match model metadata")
    if repository.get("lineage_prefix") != identity.lineage_prefix:
        raise ValueError("release manifest lineage prefix does not match its full digest")
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
    from gradlab.publication_evidence import validate_evaluation_evidence_document

    evidence_document_value = json.loads(
        (root / "evaluation_evidence.json").read_text(encoding="utf-8")
    )
    if not isinstance(evidence_document_value, Mapping):
        raise ValueError("evaluation_evidence.json must contain an object")
    evidence_document = validate_evaluation_evidence_document(evidence_document_value)
    evidence_record = _require_mapping(
        expected_records.get("evaluation_evidence.json"),
        label="release evaluation evidence artifact",
    )
    if evidence.get("evidence_sha256") != evidence_record.get("sha256"):
        raise ValueError("release evaluation evidence hash does not match evaluation_evidence.json")
    if canonical_json_sha256(evidence_document) != evidence.get("evidence_sha256"):
        raise ValueError("release evaluation evidence canonical hash is inconsistent")
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
