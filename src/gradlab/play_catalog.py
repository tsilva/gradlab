from __future__ import annotations

import base64
import hmac
import json
import re
import secrets
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError, URLError

from gradlab.config_loader import load_mapping_document
from gradlab.catalog_errors import (
    CatalogIntegrityError,
    CatalogSnapshotChanged,
    CatalogUnavailable,
)
from gradlab.goal_catalog import (
    GOAL_CATALOG_SUCCESS_BADGES,
    GOAL_CATALOG_TERMINAL_STATES,
    goal_catalog_pointer_key,
    goal_catalog_run_success_badges,
    validate_goal_catalog_generation,
    validate_goal_catalog_page,
    validate_goal_catalog_pointer,
)
from gradlab.contract_inspection import inspection_document
from gradlab.early_stop import EARLY_STOP_OPERATORS, normalize_metric_threshold_rules
from gradlab.goal_variants import (
    build_goal_variant_descriptor,
    goal_contract_diff,
    goal_contract_diff_labels,
    goal_contract_structural_diff,
    validate_goal_variant_descriptor,
)
from gradlab.json_utils import (
    canonical_json_sha256 as compact_json_sha256,
    canonical_json_text,
)
from gradlab.evaluation_projection import validate_evaluation_scientific_metric
from gradlab.evaluation_fence import evaluation_selection_fence
from gradlab.metric_names import (
    EVAL_ACCEPTANCE_PASS,
    EVAL_ACCEPTANCE_EPISODE_COMPLETED_COUNT,
    EVAL_ACCEPTANCE_EPISODE_PLANNED_COUNT,
    EVAL_CHECKPOINT_STEP,
    EVAL_FULL_EPISODE_RETURN_SHAPED_MAX,
    EVAL_FULL_EPISODE_RETURN_SHAPED_MEAN,
    EVAL_FULL_OUTCOME_SUCCESS_STARTS_RATE_MIN,
    EVAL_FULL_OUTCOME_SUCCESS_STARTS_RATE_MEAN,
    LEADER_CHECKPOINT_STEP,
    TRAIN_EPISODE_RETURN_SHAPED_ORIGIN_TARGET_ROLLING_MAX,
    TRAIN_EPISODE_RETURN_SHAPED_ORIGIN_TARGET_ROLLING_MEAN,
    TRAIN_GLOBAL_STEP,
    TRAIN_OUTCOME_SUCCESS_STARTS_ALL_ROLLING_RATE_MIN,
    TRAIN_OUTCOME_SUCCESS_STARTS_ALL_ROLLING_RATE_MEAN,
    metric_display_label,
    metric_path_segment,
    require_current_metrics_schema,
    train_progress_origin_target_rolling_mean_metric,
)
from gradlab.model_sources import DEFAULT_PUBLIC_MODELS_BASE_URL, _public_json
from gradlab.policy_bundle import canonical_json_sha256, validate_recipe_document
from gradlab.ranking import (
    RankCriterion,
    objective_rank_strings,
    parse_objective_rank,
    require_objective_rank,
)
from gradlab.recipe_documents import (
    compose_resolved_train_documents,
    goal_contract_sha256,
    load_goal_contract,
    load_recipe_source_document,
)
from gradlab.reward_programs import goal_for_contract_validation
from gradlab.run_contracts import CheckpointManifest, RUN_ID_PATTERN, RunManifest
from gradlab.run_authority import RunAuthority
from gradlab.wandb_utils import load_wandb_env


WANDB_HOSTS = {"wandb.ai", "www.wandb.ai"}
CATALOG_PAGE_SIZE = 50
CATALOG_CURSOR_TTL_SECONDS = 300
CATALOG_INDEX_SCHEMA_VERSION = 2
CATALOG_INDEX_FILENAME = "_catalog.yaml"
_CONTROL_DOCUMENT_UNSET = object()
LIVE_TRAINING_METRICS = (
    (
        RankCriterion(
            direction="max",
            metric=TRAIN_OUTCOME_SUCCESS_STARTS_ALL_ROLLING_RATE_MIN,
        ),
        (TRAIN_OUTCOME_SUCCESS_STARTS_ALL_ROLLING_RATE_MIN,),
    ),
    (
        RankCriterion(
            direction="max",
            metric=TRAIN_EPISODE_RETURN_SHAPED_ORIGIN_TARGET_ROLLING_MEAN,
        ),
        (TRAIN_EPISODE_RETURN_SHAPED_ORIGIN_TARGET_ROLLING_MEAN,),
    ),
    (
        RankCriterion(
            direction="min",
            metric=TRAIN_GLOBAL_STEP,
        ),
        (TRAIN_GLOBAL_STEP,),
    ),
)
CHECKPOINT_STRUCTURAL_METRICS = frozenset({LEADER_CHECKPOINT_STEP, TRAIN_GLOBAL_STEP})
CHECKPOINT_COLUMN_ROLES = frozenset(
    {"objective", "tie_breaker", "acceptance", "training_proxy", "optimization"}
)
_EVAL_PROGRESS_METRIC_RE = re.compile(r"^eval/full/progress/([A-Za-z0-9_.-]+)/(mean|max)$")
_TRAIN_PROGRESS_METRIC_RE = re.compile(
    r"^train/progress/([A-Za-z0-9_.-]+)/origin/target/rolling/mean$"
)


@dataclass(frozen=True)
class WandbRunLocation:
    entity: str
    project: str
    run_id: str


@dataclass(frozen=True)
class CatalogPage:
    items: tuple[dict[str, Any], ...]
    next_cursor: str | None
    metric_columns: tuple[dict[str, str], ...] = ()
    fallback_metric_columns: tuple[dict[str, str], ...] = ()
    source: Mapping[str, Any] | None = None
    freshness: Literal["fresh", "stale", "partial"] = "fresh"
    warnings: tuple[Mapping[str, Any], ...] = ()
    generated_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "items": list(self.items),
            "next_cursor": self.next_cursor,
        }
        if self.metric_columns:
            payload["metric_columns"] = list(self.metric_columns)
        if self.fallback_metric_columns:
            payload["fallback_metric_columns"] = list(self.fallback_metric_columns)
        if self.source is not None:
            payload["source"] = dict(self.source)
        payload["freshness"] = self.freshness
        if self.warnings:
            payload["warnings"] = [dict(item) for item in self.warnings]
        if self.generated_at is not None:
            payload["generated_at"] = self.generated_at
        return payload


@dataclass(frozen=True)
class CheckpointMetricContract:
    metrics_schema_version: int
    evaluation_backend: Literal["modal", "none"]
    rank: tuple[RankCriterion, ...]
    acceptance: tuple[Mapping[str, Any], ...]
    columns: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class CheckpointPage:
    items: tuple[dict[str, Any], ...]
    metric_columns: tuple[Mapping[str, Any], ...]
    selection_fence: str
    freshness: Literal["fresh", "partial"] = "fresh"
    warnings: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class EnvironmentSummary:
    name: str
    goal_count: int
    run_count: int
    success_badges: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GoalSummary:
    environment_id: str
    goal_id: str
    goal_slug: str
    title: str
    recipe_count: int
    goal_path: str
    success_badges: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GoalVariantSummary:
    environment_id: str
    goal_id: str
    goal_slug: str
    variant_id: str
    label: str
    goal_contract_sha256: str
    effective_goal_contract_sha256: str
    source_sha: str
    source_relation: str
    status: str
    diff: tuple[Mapping[str, Any], ...]
    diff_truncated: bool
    configuration_kind: str
    display_label: str
    current_diff: tuple[Mapping[str, Any], ...]
    current_diff_truncated: bool
    current_diff_count: int | None
    current_diff_count_exact: bool
    comparison_available: bool
    run_count: int
    first_used_at: str
    last_activity_at: str
    success_badges: tuple[str, ...]
    exact_resolution_run_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunSummary:
    environment_id: str
    run_id: str
    name: str
    state: str
    stop_reason: str
    final_step: int | None
    early_stop: Mapping[str, Any] | None
    goal: str
    recipe: str
    recipe_sha256: str
    recipe_overrides: tuple[str, ...]
    recipe_variant_id: str
    goal_contract_sha256: str
    effective_goal_contract_sha256: str
    goal_variant_id: str
    goal_variant_label: str
    description: str
    seed: int | None
    created_at: str
    updated_at: str
    url: str
    metrics: Mapping[str, float | None]
    success_badges: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CheckpointSummary:
    run_id: str
    checkpoint_id: str
    step: int
    purpose: Literal["periodic", "final"]
    size_bytes: int
    created_at: str
    sha256: str
    manifest_url: str
    promoted: bool
    playback_seed: int | None
    playback_seed_source: Literal["evaluation", "training"] | None
    metrics: Mapping[str, float | None]
    evaluation: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _RepositoryGoal:
    environment_id: str
    goal_id: str
    goal_slug: str
    title: str
    recipe_count: int
    goal_path: str
    rank: tuple[RankCriterion, ...] | None


@dataclass(frozen=True)
class _RepositoryNamespace:
    directory: str
    environment_id: str
    title_template: str


@dataclass(frozen=True)
class _CheckpointEvaluationData:
    evaluations: Mapping[int, dict[str, Any]]
    training_seed: int | None
    evaluation_seed: int | None
    training_metric_history: Mapping[str, tuple[tuple[int, float], ...]]
    warning: Mapping[str, Any] | None = None


def parse_wandb_location(value: object) -> WandbRunLocation | None:
    parsed = urlparse(str(value or "").strip())
    if parsed.scheme not in {"http", "https"} or parsed.netloc.casefold() not in WANDB_HOSTS:
        return None
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if len(parts) != 4 or parts[2] != "runs":
        return None
    entity, project, _runs, run_id = parts
    if not entity or not project or RUN_ID_PATTERN.fullmatch(run_id) is None:
        return None
    return WandbRunLocation(entity=entity, project=project, run_id=run_id)


def checkpoint_manifest_url(model_url: object) -> str:
    value = str(model_url or "").strip()
    if not value.endswith("/model.zip"):
        raise ValueError("public checkpoint model URL is malformed")
    return f"{value.removesuffix('/model.zip')}/manifest.json"


def _cursor_offset(
    value: object,
    *,
    secret: bytes | None = None,
    snapshot_id: str = "",
) -> int:
    text = str(value or "").strip()
    if not text:
        return 0
    if secret is not None and snapshot_id:
        try:
            encoded_payload, encoded_signature = text.split(".", 1)
            payload_bytes = base64.urlsafe_b64decode(
                encoded_payload + "=" * (-len(encoded_payload) % 4)
            )
            signature = base64.urlsafe_b64decode(
                encoded_signature + "=" * (-len(encoded_signature) % 4)
            )
            expected = hmac.digest(secret, payload_bytes, "sha256")
            if not hmac.compare_digest(signature, expected):
                raise ValueError("catalog cursor signature mismatch")
            payload = json.loads(payload_bytes)
            if not isinstance(payload, Mapping):
                raise ValueError("catalog cursor payload is malformed")
            if payload.get("snapshot") != snapshot_id:
                raise CatalogSnapshotChanged()
            if float(payload.get("expires_at") or 0) < time.time():
                raise CatalogSnapshotChanged("catalog cursor expired; restart pagination")
            offset = int(payload["offset"])
        except CatalogSnapshotChanged:
            raise
        except Exception as exc:
            raise ValueError("invalid catalog cursor") from exc
        if offset < 0:
            raise ValueError("invalid catalog cursor")
        return offset
    try:
        decoded = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4)).decode("ascii")
        offset = int(decoded)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("invalid catalog cursor") from exc
    if offset < 0:
        raise ValueError("invalid catalog cursor")
    return offset


def _cursor_for(
    offset: int,
    *,
    secret: bytes | None = None,
    snapshot_id: str = "",
) -> str:
    if secret is not None and snapshot_id:
        payload = canonical_json_text(
            {
                "expires_at": time.time() + CATALOG_CURSOR_TTL_SECONDS,
                "offset": int(offset),
                "snapshot": snapshot_id,
            },
            ensure_ascii=True,
        ).encode("utf-8")
        signature = hmac.digest(secret, payload, "sha256")
        return (
            base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
            + "."
            + base64.urlsafe_b64encode(signature).decode("ascii").rstrip("=")
        )
    return base64.urlsafe_b64encode(str(int(offset)).encode("ascii")).decode("ascii").rstrip("=")


def _search_text(*values: object) -> str:
    return " ".join(str(value or "") for value in values).casefold()


def _projected_run_success_badges(run: Mapping[str, Any]) -> tuple[str, ...]:
    explicit = run.get("success_badges")
    explicit_badges = (
        {str(item) for item in explicit} if isinstance(explicit, list | tuple) else set()
    )
    evidenced = set(goal_catalog_run_success_badges(run))
    return tuple(
        badge
        for badge in GOAL_CATALOG_SUCCESS_BADGES
        if badge in explicit_badges or badge in evidenced
    )


def _projected_variant_success_badges(
    variant: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    explicit = variant.get("success_badges")
    badges = {str(item) for item in explicit} if isinstance(explicit, list | tuple) else set()
    variant_id = str(variant.get("variant_id") or "")
    for run in runs:
        if str(run.get("goal_variant_id") or "") == variant_id:
            badges.update(_projected_run_success_badges(run))
    return tuple(badge for badge in GOAL_CATALOG_SUCCESS_BADGES if badge in badges)


def _safe_int(value: object) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except TypeError, ValueError:
        return None


def _safe_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric = float(value)
    return numeric if numeric == numeric and abs(numeric) != float("inf") else None


def _run_metric_specs(
    rank: tuple[RankCriterion, ...],
) -> tuple[tuple[RankCriterion, tuple[str, ...]], ...]:
    return tuple((criterion, (criterion.metric,)) for criterion in rank)


def _run_fallback_metric_specs(
    rank: tuple[RankCriterion, ...],
) -> tuple[tuple[RankCriterion, tuple[str, ...]], ...]:
    if not rank or all(criterion.metric.startswith("train/") for criterion in rank):
        return ()
    return LIVE_TRAINING_METRICS


def _checkpoint_metric_label(metric: str) -> str:
    return metric_display_label(metric)


def _checkpoint_acceptance_direction(operator: str) -> Literal["min", "max"]:
    return "max" if operator in {">", ">="} else "min"


def _checkpoint_training_proxy(
    metric: str,
    *,
    progress_fields: frozenset[str],
) -> str | None:
    fixed = {
        EVAL_FULL_OUTCOME_SUCCESS_STARTS_RATE_MIN: (
            TRAIN_OUTCOME_SUCCESS_STARTS_ALL_ROLLING_RATE_MIN
        ),
        EVAL_FULL_OUTCOME_SUCCESS_STARTS_RATE_MEAN: (
            TRAIN_OUTCOME_SUCCESS_STARTS_ALL_ROLLING_RATE_MEAN
        ),
        EVAL_FULL_EPISODE_RETURN_SHAPED_MEAN: (
            TRAIN_EPISODE_RETURN_SHAPED_ORIGIN_TARGET_ROLLING_MEAN
        ),
        EVAL_FULL_EPISODE_RETURN_SHAPED_MAX: (
            TRAIN_EPISODE_RETURN_SHAPED_ORIGIN_TARGET_ROLLING_MAX
        ),
    }
    if metric in fixed:
        return fixed[metric]
    match = _EVAL_PROGRESS_METRIC_RE.fullmatch(metric)
    if match is None or match.group(2) != "mean" or match.group(1) not in progress_fields:
        return None
    return train_progress_origin_target_rolling_mean_metric(match.group(1))


def checkpoint_metric_contract(
    train_config: Mapping[str, Any],
) -> CheckpointMetricContract:
    schema_version = require_current_metrics_schema(train_config.get("metrics_schema_version"))
    raw_evaluation_backend = str(train_config.get("checkpoint_eval_backend") or "").strip()
    if raw_evaluation_backend == "modal":
        evaluation_backend: Literal["modal", "none"] = "modal"
    elif raw_evaluation_backend == "none":
        evaluation_backend = "none"
    else:
        raise ValueError("recipe.train_config.checkpoint_eval_backend is invalid")
    rank = require_objective_rank(
        train_config.get("selection_rank"),
        metrics_schema_version=schema_version,
    )
    raw_acceptance = train_config.get("checkpoint_eval_acceptance")
    acceptance = (
        tuple(
            normalize_metric_threshold_rules(
                raw_acceptance,
                label="recipe.train_config.checkpoint_eval_acceptance",
                metric_validator=lambda name: validate_evaluation_scientific_metric(
                    name,
                    schema_version=schema_version,
                ),
            )
        )
        if raw_acceptance is not None
        else ()
    )
    progress_fields = frozenset(
        metric_path_segment(field) for field in train_config.get("episode_progress_fields", ())
    )
    columns: list[dict[str, Any]] = []
    by_metric: dict[str, dict[str, Any]] = {}

    def add_column(
        metric: str,
        *,
        direction: Literal["min", "max"] | None,
        role: str,
        rank_index: int | None = None,
        acceptance_rule: Mapping[str, Any] | None = None,
        proxy_for: str | None = None,
    ) -> dict[str, Any]:
        if role not in CHECKPOINT_COLUMN_ROLES:
            raise ValueError(f"unsupported checkpoint metric role: {role}")
        column = by_metric.get(metric)
        if column is None:
            if metric.startswith("eval/"):
                evidence = "evaluation"
            elif metric.startswith("train/"):
                evidence = "training"
            else:
                raise ValueError(f"checkpoint list cannot project metric family: {metric}")
            column = {
                "metric": metric,
                "direction": direction,
                "label": _checkpoint_metric_label(metric),
                "evidence": evidence,
                "roles": [],
            }
            by_metric[metric] = column
            columns.append(column)
        if role not in column["roles"]:
            column["roles"].append(role)
        if rank_index is not None:
            previous_rank = column.get("rank_index")
            if previous_rank is None or rank_index < previous_rank:
                column["rank_index"] = rank_index
                column["direction"] = direction
        elif column.get("rank_index") is None:
            existing_direction = column.get("direction")
            if existing_direction is not None and direction != existing_direction:
                column["direction"] = None
        if acceptance_rule is not None:
            column.setdefault("acceptance", []).append(dict(acceptance_rule))
        if proxy_for is not None:
            column["proxy_for"] = proxy_for
        return column

    for rank_index, criterion in enumerate(rank):
        if criterion.metric in CHECKPOINT_STRUCTURAL_METRICS:
            continue
        role = "objective" if rank_index == 0 else "tie_breaker"
        add_column(
            criterion.metric,
            direction=criterion.direction,
            role=role,
            rank_index=rank_index,
        )
        proxy = _checkpoint_training_proxy(
            criterion.metric,
            progress_fields=progress_fields,
        )
        if proxy is not None:
            add_column(
                proxy,
                direction=criterion.direction,
                role="training_proxy",
                proxy_for=criterion.metric,
            )

    for rule in acceptance:
        metric = str(rule["metric"])
        direction = _checkpoint_acceptance_direction(str(rule["operator"]))
        add_column(
            metric,
            direction=direction,
            role="acceptance",
            acceptance_rule=rule,
        )
        proxy = _checkpoint_training_proxy(metric, progress_fields=progress_fields)
        if proxy is not None:
            add_column(
                proxy,
                direction=direction,
                role="training_proxy",
                proxy_for=metric,
            )

    add_column(
        TRAIN_EPISODE_RETURN_SHAPED_ORIGIN_TARGET_ROLLING_MEAN,
        direction="max",
        role="optimization",
    )
    return CheckpointMetricContract(
        metrics_schema_version=schema_version,
        evaluation_backend=evaluation_backend,
        rank=rank,
        acceptance=acceptance,
        columns=tuple(columns),
    )


def checkpoint_metric_columns(
    train_config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    return checkpoint_metric_contract(train_config).columns


def checkpoint_metric_leaders(
    items: Sequence[Mapping[str, Any]],
    columns: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    copied = [dict(item) for item in items]
    best_values: dict[str, float] = {}
    for column in columns:
        metric = str(column["metric"])
        direction = column.get("direction")
        if direction not in {"min", "max"}:
            continue
        values = [
            value
            for item in copied
            if isinstance(item.get("metrics"), Mapping)
            and (value := _safe_float(item["metrics"].get(metric))) is not None
        ]
        if values:
            best_values[metric] = min(values) if direction == "min" else max(values)
    return tuple(
        {
            **item,
            "best_metrics": [
                str(column["metric"])
                for column in columns
                if str(column["metric"]) in best_values
                and isinstance(item.get("metrics"), Mapping)
                and _safe_float(item["metrics"].get(str(column["metric"])))
                == best_values[str(column["metric"])]
            ],
        }
        for item in copied
    )


def filter_checkpoint_summaries(
    items: Sequence[Mapping[str, Any]],
    *,
    query: str = "",
) -> tuple[dict[str, Any], ...]:
    normalized = str(query or "").strip().casefold()
    return tuple(
        dict(item)
        for item in items
        if not normalized
        or normalized
        in _search_text(
            item.get("checkpoint_id"),
            item.get("step"),
            item.get("purpose"),
            item.get("sha256"),
            item.get("created_at"),
            "promoted" if item.get("promoted") else "",
            item.get("metrics"),
            item.get("evaluation"),
        )
    )


def _checkpoint_training_metric_history(
    run: Any,
    columns: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[tuple[int, float], ...]]:
    history: dict[str, tuple[tuple[int, float], ...]] = {}
    for column in columns:
        metric = str(column["metric"])
        if column.get("evidence") != "training" or metric == TRAIN_GLOBAL_STEP:
            continue
        samples: dict[int, float] = {}
        rows = run.scan_history(
            keys=[TRAIN_GLOBAL_STEP, metric],
            page_size=10_000,
        )
        for raw in rows:
            if not isinstance(raw, Mapping):
                continue
            step = _safe_int(raw.get(TRAIN_GLOBAL_STEP))
            value = _safe_float(raw.get(metric))
            if step is not None and value is not None:
                samples[step] = value
        if samples:
            history[metric] = tuple(sorted(samples.items()))
    return history


def _checkpoint_training_metrics(
    history: Mapping[str, tuple[tuple[int, float], ...]],
    *,
    checkpoint_step: int,
    columns: Sequence[Mapping[str, Any]],
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    for column in columns:
        metric = str(column["metric"])
        if column.get("evidence") != "training" or metric == TRAIN_GLOBAL_STEP:
            continue
        samples = history.get(metric, ())
        eligible = (sample for sample in samples if sample[0] <= checkpoint_step)
        latest = max(eligible, default=None, key=lambda sample: sample[0])
        metrics[metric] = latest[1] if latest is not None else None
    return metrics


def checkpoint_metric_values(
    training_metrics: Mapping[str, Any],
    evaluation: Mapping[str, Any] | None,
    columns: Sequence[Mapping[str, Any]],
) -> dict[str, float | None]:
    evaluation_metrics = evaluation.get("metrics") if isinstance(evaluation, Mapping) else None
    return {
        str(column["metric"]): (
            _safe_float(evaluation_metrics.get(str(column["metric"])))
            if column.get("evidence") == "evaluation" and isinstance(evaluation_metrics, Mapping)
            else _safe_float(training_metrics.get(str(column["metric"])))
            if column.get("evidence") == "training"
            else None
        )
        for column in columns
    }


def _complete_run_rank(
    item: Mapping[str, Any],
    metric_specs: tuple[tuple[RankCriterion, tuple[str, ...]], ...],
) -> tuple[float, ...] | None:
    if not metric_specs:
        return None
    metrics = item.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    score: list[float] = []
    for criterion, _sources in metric_specs:
        value = _safe_float(metrics.get(criterion.metric))
        if value is None:
            return None
        score.append(value if criterion.direction == "min" else -value)
    return tuple(score)


def _rank_run_summaries(
    items: list[dict[str, Any]],
    *,
    primary: tuple[tuple[RankCriterion, tuple[str, ...]], ...],
    fallback: tuple[tuple[RankCriterion, tuple[str, ...]], ...],
) -> None:
    active = (
        primary
        if any(_complete_run_rank(item, primary) is not None for item in items)
        else fallback
    )
    if not active:
        return
    items.sort(
        key=lambda item: (
            _complete_run_rank(item, active) is None,
            _complete_run_rank(item, active) or (),
        )
    )


def _page_items(
    items: list[dict[str, Any]],
    cursor: str | None,
    *,
    secret: bytes | None = None,
    snapshot_id: str = "",
) -> CatalogPage:
    offset = _cursor_offset(
        cursor,
        secret=secret,
        snapshot_id=snapshot_id,
    )
    selected = tuple(items[offset : offset + CATALOG_PAGE_SIZE])
    next_offset = offset + CATALOG_PAGE_SIZE
    return CatalogPage(
        items=selected,
        next_cursor=(
            _cursor_for(
                next_offset,
                secret=secret,
                snapshot_id=snapshot_id,
            )
            if next_offset < len(items)
            else None
        ),
    )


class PlayCatalog:
    """Repository catalog, lifecycle run metadata, and public-checkpoint discovery."""

    def __init__(
        self,
        *,
        public_models_base_url: str = DEFAULT_PUBLIC_MODELS_BASE_URL,
        repo_root: Path | str | None = None,
        control_bucket: Any | None = None,
        control_error: str = "",
        wandb_run_location: WandbRunLocation | None = None,
    ) -> None:
        self.public_models_base_url = str(public_models_base_url).rstrip("/")
        self.repo_root = (
            Path(repo_root).resolve()
            if repo_root is not None
            else Path(__file__).resolve().parents[2]
        )
        self.goals_root = self.repo_root / "experiments" / "goals"
        self.control_bucket = control_bucket
        self.control_error = str(control_error or "").strip()
        self.control_identity = str(
            getattr(self.control_bucket, "authority_identity", "")
            or ("explicit-control" if self.control_bucket is not None else "unavailable")
        )
        self._wandb_run_locations = (
            {wandb_run_location.run_id: wandb_run_location}
            if wandb_run_location is not None
            else {}
        )
        self._api: Any | None = None
        self._cursor_secret = secrets.token_bytes(32)
        self._cache_lock = threading.Lock()
        self._catalog_repairing: set[str] = set()
        self._repository_environment_cache: dict[
            str,
            tuple[
                tuple[tuple[str, int, int], ...],
                tuple[_RepositoryGoal, ...],
            ],
        ] = {}
        self._repository_details: dict[tuple[str, str], _RepositoryGoal] = {}
        self._namespace_cache: (
            tuple[
                tuple[tuple[str, int, int], ...],
                tuple[_RepositoryNamespace, ...],
            ]
            | None
        ) = None

    def _page(
        self,
        items: list[dict[str, Any]],
        cursor: str | None,
        *,
        identity: Mapping[str, Any],
    ) -> CatalogPage:
        snapshot_id = compact_json_sha256(
            {
                **dict(identity),
                "items": items,
            },
            ensure_ascii=True,
        )
        page = _page_items(
            items,
            cursor,
            secret=self._cursor_secret,
            snapshot_id=snapshot_id,
        )
        return CatalogPage(
            items=page.items,
            next_cursor=page.next_cursor,
            source={"snapshot_id": snapshot_id},
        )

    def _wandb_api(self):
        if self._api is None:
            load_wandb_env()
            import wandb

            self._api = wandb.Api(timeout=15)
        return self._api

    def _catalog_fingerprint(
        self,
        paths: Iterable[Path],
    ) -> tuple[tuple[str, int, int], ...]:
        entries: list[tuple[str, int, int]] = []
        for path in sorted(set(paths)):
            stat_result = path.stat()
            entries.append(
                (
                    path.relative_to(self.repo_root).as_posix(),
                    stat_result.st_mtime_ns,
                    stat_result.st_size,
                )
            )
        return tuple(entries)

    def _repository_namespaces(self) -> tuple[_RepositoryNamespace, ...]:
        index_path = self.goals_root / CATALOG_INDEX_FILENAME
        if not index_path.is_file():
            raise ValueError(f"repository goal catalog does not exist: {index_path}")
        goal_paths = tuple(self.goals_root.rglob("_goal.yaml"))
        fingerprint = self._catalog_fingerprint((index_path, *goal_paths))
        with self._cache_lock:
            cached = self._namespace_cache
            if cached is not None and cached[0] == fingerprint:
                return cached[1]

        document = load_mapping_document(index_path, label="repository goal catalog")
        if document.get("schema_version") != CATALOG_INDEX_SCHEMA_VERSION:
            raise ValueError(
                "repository goal catalog schema_version must be "
                f"{CATALOG_INDEX_SCHEMA_VERSION}: {index_path}"
            )
        raw_namespaces = document.get("namespaces")
        if not isinstance(raw_namespaces, Mapping):
            raise ValueError(f"repository goal catalog namespaces must be a mapping: {index_path}")

        namespaces: list[_RepositoryNamespace] = []
        for raw_directory, raw_metadata in raw_namespaces.items():
            directory = str(raw_directory or "").strip()
            if (
                not directory
                or Path(directory).is_absolute()
                or len(Path(directory).parts) != 1
                or directory in {".", ".."}
            ):
                raise ValueError(
                    f"repository goal catalog namespace must be one directory: {directory!r}"
                )
            if not isinstance(raw_metadata, Mapping):
                raise ValueError(
                    f"repository goal catalog namespace {directory!r} must be a mapping"
                )
            extra = set(raw_metadata) - {"environment_id", "title_template"}
            if extra:
                raise ValueError(
                    f"repository goal catalog namespace {directory!r} has unknown keys: "
                    + ", ".join(sorted(str(key) for key in extra))
                )
            environment_id = str(raw_metadata.get("environment_id") or "").strip()
            title_template = str(raw_metadata.get("title_template") or "").strip()
            if not environment_id:
                raise ValueError(
                    f"repository goal catalog namespace {directory!r} has no environment_id"
                )
            if title_template:
                try:
                    title_template.format(goal_id="example")
                except (IndexError, KeyError, ValueError) as exc:
                    raise ValueError(
                        f"repository goal catalog namespace {directory!r} has an invalid "
                        "title_template"
                    ) from exc
            namespace_root = self.goals_root / directory
            if not namespace_root.is_dir() or not any(
                path.is_relative_to(namespace_root) for path in goal_paths
            ):
                continue
            namespaces.append(
                _RepositoryNamespace(
                    directory=directory,
                    environment_id=environment_id,
                    title_template=title_template,
                )
            )

        result = tuple(sorted(namespaces, key=lambda item: item.directory))
        with self._cache_lock:
            self._namespace_cache = (fingerprint, result)
        return result

    def _environment_namespaces(
        self,
        environment_id: str,
        namespaces: tuple[_RepositoryNamespace, ...],
    ) -> tuple[_RepositoryNamespace, ...]:
        return tuple(
            namespace for namespace in namespaces if namespace.environment_id == environment_id
        )

    def _indexed_environment_fingerprint(
        self,
        namespaces: tuple[_RepositoryNamespace, ...],
    ) -> tuple[tuple[str, int, int], ...]:
        paths: list[Path] = [self.goals_root / CATALOG_INDEX_FILENAME]
        for namespace in namespaces:
            namespace_root = self.goals_root / namespace.directory
            paths.extend(namespace_root.rglob("*.yaml"))
            paths.extend(namespace_root.rglob("*.yml"))
        return self._catalog_fingerprint(paths)

    def _compose_repository_goal(
        self,
        path: Path,
        *,
        environment_id: str,
    ) -> _RepositoryGoal:
        document = load_goal_contract(path, self.repo_root)
        goal_id = str(document.get("goal_id") or "").strip()
        if not environment_id or not goal_id:
            raise ValueError(f"repository goal has no environment or goal identity: {path}")
        objective = document.get("objective")
        return _RepositoryGoal(
            environment_id=environment_id,
            goal_id=goal_id,
            goal_slug=path.parent.relative_to(self.goals_root).as_posix(),
            title=str(document.get("title") or goal_id).strip(),
            recipe_count=sum(1 for _ in path.parent.glob("recipes/*.yaml")),
            goal_path=path.relative_to(self.repo_root).as_posix(),
            rank=parse_objective_rank(
                objective.get("rank") if isinstance(objective, Mapping) else None
            ),
        )

    def _indexed_repository_goals(
        self,
        *,
        environment_id: str,
        namespaces: tuple[_RepositoryNamespace, ...],
    ) -> tuple[_RepositoryGoal, ...]:
        environment_namespaces = self._environment_namespaces(environment_id, namespaces)
        if not environment_namespaces:
            return ()
        fingerprint = self._indexed_environment_fingerprint(environment_namespaces)
        with self._cache_lock:
            cached = self._repository_environment_cache.get(environment_id)
            if cached is not None and cached[0] == fingerprint:
                return cached[1]

        goals: list[_RepositoryGoal] = []
        for namespace in environment_namespaces:
            namespace_root = self.goals_root / namespace.directory
            for path in sorted(namespace_root.rglob("_goal.yaml")):
                goal_id = path.parent.name
                raw_document = load_mapping_document(
                    path,
                    label=f"repository goal metadata {path}",
                )
                title = str(raw_document.get("title") or "").strip()
                if not title and namespace.title_template:
                    title = namespace.title_template.format(goal_id=goal_id)
                goals.append(
                    _RepositoryGoal(
                        environment_id=environment_id,
                        goal_id=goal_id,
                        goal_slug=path.parent.relative_to(self.goals_root).as_posix(),
                        title=title or goal_id,
                        recipe_count=sum(1 for _ in path.parent.glob("recipes/*.yaml")),
                        goal_path=path.relative_to(self.repo_root).as_posix(),
                        rank=None,
                    )
                )
        identities = [(goal.environment_id, goal.goal_id) for goal in goals]
        if len(identities) != len(set(identities)):
            raise ValueError("repository goals contain duplicate environment/goal identities")
        result = tuple(sorted(goals, key=lambda goal: goal.goal_id))
        with self._cache_lock:
            self._repository_environment_cache[environment_id] = (fingerprint, result)
            for key in tuple(self._repository_details):
                if key[0] == environment_id:
                    self._repository_details.pop(key, None)
        return result

    def _repository_goals(
        self,
        *,
        environment_id: str | None = None,
    ) -> tuple[_RepositoryGoal, ...]:
        if not self.goals_root.is_dir():
            raise ValueError(f"repository goals directory does not exist: {self.goals_root}")
        namespaces = self._repository_namespaces()
        environment_ids = sorted({namespace.environment_id for namespace in namespaces})
        selected_environment_ids = (
            [environment_id] if environment_id is not None else environment_ids
        )
        return tuple(
            goal
            for selected_environment_id in selected_environment_ids
            for goal in self._indexed_repository_goals(
                environment_id=selected_environment_id,
                namespaces=namespaces,
            )
        )

    def _repository_environments(self) -> dict[str, int]:
        namespaces = self._repository_namespaces()
        counts: dict[str, int] = {}
        for namespace in namespaces:
            count = sum(1 for _ in (self.goals_root / namespace.directory).rglob("_goal.yaml"))
            counts[namespace.environment_id] = counts.get(namespace.environment_id, 0) + count
        return counts

    def _repository_goal(
        self,
        *,
        environment_id: str,
        goal_id: str,
    ) -> _RepositoryGoal:
        for goal in self._repository_goals(environment_id=environment_id):
            if goal.environment_id == environment_id and goal.goal_id == goal_id:
                if goal.rank is not None:
                    return goal
                key = (environment_id, goal_id)
                with self._cache_lock:
                    detailed = self._repository_details.get(key)
                if detailed is not None:
                    return detailed
                path = self.repo_root / goal.goal_path
                detailed = self._compose_repository_goal(
                    path,
                    environment_id=goal.environment_id,
                )
                if (
                    detailed.environment_id != goal.environment_id
                    or detailed.goal_id != goal.goal_id
                    or detailed.goal_slug != goal.goal_slug
                    or detailed.title != goal.title
                ):
                    raise ValueError(
                        "repository goal browse metadata does not match the composed "
                        f"contract: {goal.goal_path}"
                    )
                with self._cache_lock:
                    self._repository_details[key] = detailed
                return detailed
        raise ValueError(f"repository has no goal {environment_id}/{goal_id}")

    def _current_goal_variant(
        self,
        repository_goal: _RepositoryGoal,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        authored = load_goal_contract(
            self.repo_root / repository_goal.goal_path,
            self.repo_root,
        )
        resolved = goal_for_contract_validation(
            authored,
            label=f"repository goal {repository_goal.goal_slug}",
        )
        descriptor = build_goal_variant_descriptor(
            goal_slug=repository_goal.goal_slug,
            source_sha="",
            authored_goal=authored,
            effective_goal=resolved,
        )
        return descriptor, resolved

    def _variant_summary(
        self,
        *,
        descriptor: Mapping[str, Any],
        repository_goal: _RepositoryGoal,
        current: Mapping[str, Any],
        current_goal: Mapping[str, Any],
        resolved_goal: Mapping[str, Any] | None = None,
        runs: Sequence[Mapping[str, Any]] = (),
        exact_resolution_run_id: str = "",
    ) -> dict[str, Any]:
        validated = validate_goal_variant_descriptor(descriptor)
        authored_current = validated["goal_contract_sha256"] == current["goal_contract_sha256"]
        effective_current = (
            validated["effective_goal_contract_sha256"] == current["effective_goal_contract_sha256"]
        )
        status = (
            "current"
            if authored_current and effective_current
            else "current changed"
            if authored_current
            else "historical"
        )
        configuration_kind = (
            "current_default"
            if authored_current and effective_current
            else "current_modified"
            if authored_current
            else "previous_default"
            if validated["source_relation"] == "canonical"
            else "previous_modified"
        )
        current_diff: list[dict[str, Any]] = []
        current_diff_truncated = False
        current_diff_count: int | None = None
        current_diff_count_exact = False
        comparison_available = configuration_kind == "current_default"
        if configuration_kind == "current_default":
            current_diff_count = 0
            current_diff_count_exact = True
        if resolved_goal is not None:
            if goal_contract_sha256(resolved_goal) != validated["effective_goal_contract_sha256"]:
                raise CatalogIntegrityError(
                    "goal variant resolved contract does not match its descriptor"
                )
            current_diff, current_diff_truncated = goal_contract_diff(
                current_goal,
                resolved_goal,
            )
            current_diff_count = len(goal_contract_structural_diff(current_goal, resolved_goal))
            current_diff_count_exact = True
            comparison_available = True
        elif configuration_kind == "current_modified":
            current_diff = [dict(item) for item in validated["diff"]]
            current_diff_truncated = bool(validated["diff_truncated"])
            current_diff_count = len(current_diff)
            comparison_available = True

        labels = goal_contract_diff_labels(
            current_diff,
            limit=2,
            truncated=current_diff_truncated,
        )
        display_label = (
            " · ".join(labels)
            if labels
            else "No behavioral changes"
            if comparison_available
            else "Behavioral difference unavailable — no exact goal proof"
        )
        variant_runs = {
            str(run.get("run_id") or ""): dict(run)
            for run in runs
            if str(run.get("goal_variant_id") or "") == validated["variant_id"]
            and str(run.get("run_id") or "")
        }
        first_used_values = [
            str(run.get("created_at") or "")
            for run in variant_runs.values()
            if str(run.get("created_at") or "")
        ]
        last_activity_values = [
            str(run.get("updated_at") or run.get("created_at") or "")
            for run in variant_runs.values()
            if str(run.get("updated_at") or run.get("created_at") or "")
        ]
        return GoalVariantSummary(
            environment_id=repository_goal.environment_id,
            goal_id=repository_goal.goal_id,
            goal_slug=repository_goal.goal_slug,
            variant_id=str(validated["variant_id"]),
            label=str(validated["label"]),
            goal_contract_sha256=str(validated["goal_contract_sha256"]),
            effective_goal_contract_sha256=str(validated["effective_goal_contract_sha256"]),
            source_sha=str(validated["source_sha"]),
            source_relation=str(validated["source_relation"]),
            status=status,
            diff=tuple(dict(item) for item in validated["diff"]),
            diff_truncated=bool(validated["diff_truncated"]),
            configuration_kind=configuration_kind,
            display_label=display_label,
            current_diff=tuple(current_diff),
            current_diff_truncated=current_diff_truncated,
            current_diff_count=current_diff_count,
            current_diff_count_exact=current_diff_count_exact,
            comparison_available=comparison_available,
            run_count=len(variant_runs),
            first_used_at=min(first_used_values) if first_used_values else "",
            last_activity_at=max(last_activity_values) if last_activity_values else "",
            success_badges=tuple(
                badge
                for badge in GOAL_CATALOG_SUCCESS_BADGES
                if any(badge in _projected_run_success_badges(run) for run in variant_runs.values())
            ),
            exact_resolution_run_id=str(exact_resolution_run_id or ""),
        ).to_dict()

    def _control_goal_variants(
        self,
        *,
        repository_goal: _RepositoryGoal,
        current: Mapping[str, Any],
        current_goal: Mapping[str, Any],
        generation_scope: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], ...] | None:
        if self.control_bucket is None:
            return None
        if generation_scope is None:
            generation_scope = self._control_generation_scope(goal_slug=repository_goal.goal_slug)
        if generation_scope is None:
            return ()
        raw_variants = generation_scope["variants"]
        raw_runs = generation_scope["runs"]
        if not isinstance(raw_variants, list):
            raise CatalogIntegrityError("goal variant index variants must be a list")
        if not isinstance(raw_runs, list) or any(not isinstance(run, Mapping) for run in raw_runs):
            raise CatalogIntegrityError("goal variant catalog runs must be a list")
        items = []
        for raw in raw_variants:
            if not isinstance(raw, Mapping):
                raise CatalogIntegrityError("goal variant index contains an invalid entry")
            descriptor = {
                key: value
                for key, value in raw.items()
                if key
                not in {
                    "descriptor_key",
                    "first_run_id",
                    "exact_resolution_run_id",
                    "resolved_goal",
                    "run_count",
                    "active_run_count",
                    "terminal_run_count",
                    "first_used_at",
                    "last_activity_at",
                    "success_badges",
                }
            }
            resolved_goal = raw.get("resolved_goal")
            if resolved_goal is not None and not isinstance(resolved_goal, Mapping):
                raise CatalogIntegrityError("goal variant resolved goal must be an object")
            validated_descriptor = validate_goal_variant_descriptor(descriptor)
            if (
                resolved_goal is None
                and validated_descriptor["goal_contract_sha256"] != current["goal_contract_sha256"]
            ):
                resolved_goal = self._variant_resolved_goal_from_exact_run(
                    variant_id=str(validated_descriptor["variant_id"]),
                    exact_run_id=str(raw.get("exact_resolution_run_id") or ""),
                )
            summary = self._variant_summary(
                descriptor=validated_descriptor,
                repository_goal=repository_goal,
                current=current,
                current_goal=current_goal,
                resolved_goal=resolved_goal,
                runs=raw_runs,
                exact_resolution_run_id=str(raw.get("exact_resolution_run_id") or ""),
            )
            summary.update(
                {
                    "run_count": int(raw.get("run_count") or 0),
                    "active_run_count": int(raw.get("active_run_count") or 0),
                    "terminal_run_count": int(raw.get("terminal_run_count") or 0),
                    "first_used_at": str(raw.get("first_used_at") or ""),
                    "last_activity_at": str(raw.get("last_activity_at") or ""),
                    "success_badges": list(_projected_variant_success_badges(raw, raw_runs)),
                    "recent_runs": [
                        dict(run)
                        for run in raw_runs
                        if run.get("goal_variant_id") == validated_descriptor["variant_id"]
                    ][:5],
                }
            )
            items.append(summary)
        return tuple(items)

    def _variant_resolved_goal_from_exact_run(
        self,
        *,
        variant_id: str,
        exact_run_id: str,
    ) -> dict[str, Any] | None:
        if not exact_run_id or self.control_bucket is None:
            return None
        try:
            raw_manifest = self.control_bucket.get_json_optional(
                f"runs/{exact_run_id}/manifest.json"
            )
        except TimeoutError, OSError:
            return None
        if raw_manifest is None:
            return None
        try:
            manifest = RunManifest.from_dict(raw_manifest)
        except Exception as exc:
            raise CatalogIntegrityError(
                "goal variant resolution run manifest is malformed"
            ) from exc
        if manifest.run_id != exact_run_id:
            raise CatalogIntegrityError("goal variant resolution run identity mismatch")
        try:
            document = self.control_bucket.get_json_optional(
                RunAuthority.recipe_document_key(manifest.recipe_sha256)
            )
        except TimeoutError, OSError:
            return None
        if document is None:
            return None
        if not isinstance(document, Mapping):
            raise CatalogIntegrityError("goal variant resolution recipe is malformed")
        if canonical_json_sha256(document) != manifest.recipe_sha256:
            raise CatalogIntegrityError("goal variant resolution recipe hash mismatch")
        recipe = document.get("recipe")
        if not isinstance(recipe, Mapping):
            raise CatalogIntegrityError("goal variant resolution recipe is malformed")
        raw_descriptor = recipe.get("goal_variant")
        if not isinstance(raw_descriptor, Mapping):
            raise CatalogIntegrityError("goal variant resolution recipe has no descriptor")
        descriptor = validate_goal_variant_descriptor(raw_descriptor)
        if descriptor["variant_id"] != variant_id:
            raise CatalogIntegrityError("goal variant resolution run proves a different variant")
        resolved_goal = recipe.get("goal")
        if not isinstance(resolved_goal, Mapping):
            raise CatalogIntegrityError("goal variant resolution recipe has no resolved goal")
        if goal_contract_sha256(resolved_goal) != descriptor["effective_goal_contract_sha256"]:
            raise CatalogIntegrityError(
                "goal variant resolution recipe goal does not match its descriptor"
            )
        return deepcopy(dict(resolved_goal))

    def _control_generation_scope(
        self,
        *,
        goal_slug: str,
        include_archives: bool = False,
        pointer_document: object = _CONTROL_DOCUMENT_UNSET,
        generation_document: object = _CONTROL_DOCUMENT_UNSET,
    ) -> dict[str, Any] | None:
        if self.control_bucket is None:
            return None
        digest = ""
        try:
            if pointer_document is _CONTROL_DOCUMENT_UNSET:
                pointer_document = self.control_bucket.get_json_optional(
                    goal_catalog_pointer_key(goal_slug)
                )
            if pointer_document is None:
                return None
            if not isinstance(pointer_document, Mapping):
                raise ValueError("goal catalog pointer is malformed")
            pointer = validate_goal_catalog_pointer(
                pointer_document,
                expected_goal_slug=goal_slug,
            )
            digest = str(pointer["generation_sha256"])
            if generation_document is _CONTROL_DOCUMENT_UNSET:
                generation_document = self.control_bucket.get_json_optional(
                    pointer["generation_key"]
                )
            if generation_document is None:
                raise ValueError("catalog pointer references a missing generation")
            if not isinstance(generation_document, Mapping):
                raise ValueError("goal catalog generation is malformed")
            generation = validate_goal_catalog_generation(
                generation_document,
                expected_digest=digest,
            )
            if generation["generated_at"] != pointer["generated_at"]:
                raise ValueError("catalog pointer timestamp disagrees with its generation")
        except ValueError as exc:
            raise CatalogIntegrityError(
                str(exc),
                source="control-catalog",
            ) from exc
        except Exception as exc:
            raise CatalogUnavailable(
                f"Goal activity is temporarily unavailable: {exc}",
                code="catalog_transient",
            ) from exc
        runs = []
        for raw in (*generation["active_runs"], *generation["terminal_runs"]):
            run = dict(raw)
            run["success_badges"] = list(_projected_run_success_badges(run))
            runs.append(run)
        if include_archives:
            for reference in generation["archive_pages"]:
                page_document = self.control_bucket.get_json_optional(str(reference["page_key"]))
                if page_document is None:
                    raise CatalogIntegrityError("catalog archive page is missing")
                page = validate_goal_catalog_page(
                    page_document,
                    expected_digest=str(reference["page_sha256"]),
                )
                if page["goal_slug"] != goal_slug:
                    raise CatalogIntegrityError("catalog archive page belongs to another goal")
                for raw in page["runs"]:
                    run = dict(raw)
                    run["success_badges"] = list(_projected_run_success_badges(run))
                    runs.append(run)
        return {
            "goal_slug": goal_slug,
            "variants": [dict(item) for item in generation["variants"]],
            "runs": runs,
            "generation_sha256": digest,
            "generated_at": str(generation["generated_at"]),
            "archive_pages": [dict(item) for item in generation["archive_pages"]],
        }

    def _control_generation_scopes(
        self,
        repository_goals: Sequence[_RepositoryGoal],
    ) -> tuple[dict[str, Any] | None, ...]:
        if self.control_bucket is None or not repository_goals:
            return tuple(None for _goal in repository_goals)
        bulk_reader = getattr(self.control_bucket, "get_json_many_optional", None)
        if not callable(bulk_reader):
            return tuple(
                self._control_generation_scope(goal_slug=goal.goal_slug)
                for goal in repository_goals
            )
        pointer_keys = tuple(goal_catalog_pointer_key(goal.goal_slug) for goal in repository_goals)
        try:
            pointer_documents = bulk_reader(pointer_keys)
            generation_keys: list[str] = []
            for goal, pointer_key in zip(repository_goals, pointer_keys, strict=True):
                pointer_document = pointer_documents[pointer_key]
                if pointer_document is None:
                    continue
                pointer = validate_goal_catalog_pointer(
                    pointer_document,
                    expected_goal_slug=goal.goal_slug,
                )
                generation_keys.append(str(pointer["generation_key"]))
            generation_documents = (
                bulk_reader(dict.fromkeys(generation_keys)) if generation_keys else {}
            )
        except CatalogUnavailable:
            return tuple(
                self._control_generation_scope(goal_slug=goal.goal_slug)
                for goal in repository_goals
            )
        scopes: list[dict[str, Any] | None] = []
        for goal, pointer_key in zip(repository_goals, pointer_keys, strict=True):
            pointer_document = pointer_documents[pointer_key]
            generation_document: object = _CONTROL_DOCUMENT_UNSET
            if pointer_document is not None:
                pointer = validate_goal_catalog_pointer(
                    pointer_document,
                    expected_goal_slug=goal.goal_slug,
                )
                generation_document = generation_documents.get(
                    str(pointer["generation_key"]),
                    _CONTROL_DOCUMENT_UNSET,
                )
            scopes.append(
                self._control_generation_scope(
                    goal_slug=goal.goal_slug,
                    pointer_document=pointer_document,
                    generation_document=generation_document,
                )
            )
        return tuple(scopes)

    def _load_goal_variants(
        self,
        *,
        repository_goal: _RepositoryGoal,
        generation_scope: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], ...]:
        current, current_goal = self._current_goal_variant(repository_goal)
        current_summary = self._variant_summary(
            descriptor=current,
            repository_goal=repository_goal,
            current=current,
            current_goal=current_goal,
        )
        history = self._control_goal_variants(
            repository_goal=repository_goal,
            current=current,
            current_goal=current_goal,
            generation_scope=generation_scope,
        )
        by_id = {str(item["variant_id"]): item for item in (history or ())}
        indexed_current = by_id.get(str(current_summary["variant_id"]))
        if indexed_current is None:
            by_id[str(current_summary["variant_id"])] = current_summary
        items = list(by_id.values())
        items.sort(key=lambda item: str(item.get("variant_id") or ""))
        items.sort(key=lambda item: str(item.get("last_activity_at") or ""), reverse=True)
        kind_order = {
            "current_default": 0,
            "current_modified": 1,
            "previous_default": 2,
            "previous_modified": 3,
        }
        items.sort(key=lambda item: kind_order.get(str(item.get("configuration_kind")), 4))
        return tuple(items)

    def _current_goal_evidence(
        self,
        repository_goal: _RepositoryGoal,
        *,
        generation_scope: object = _CONTROL_DOCUMENT_UNSET,
    ) -> tuple[tuple[str, ...], int]:
        if self.control_bucket is None:
            return (), 0
        try:
            scope = (
                self._control_generation_scope(goal_slug=repository_goal.goal_slug)
                if generation_scope is _CONTROL_DOCUMENT_UNSET
                else generation_scope
            )
        except CatalogUnavailable:
            return (), 0
        if scope is None:
            return (), 0
        if not isinstance(scope, Mapping):
            raise CatalogIntegrityError("goal catalog scope is malformed")
        current, _current_goal = self._current_goal_variant(repository_goal)
        current_variant_id = str(current["variant_id"])
        runs = [run for run in scope.get("runs", ()) if isinstance(run, Mapping)]
        for raw in scope.get("variants", ()):
            if isinstance(raw, Mapping) and str(raw.get("variant_id") or "") == current_variant_id:
                return (
                    _projected_variant_success_badges(raw, runs),
                    max(0, int(raw.get("run_count") or 0)),
                )
        return (), 0

    def _current_goal_success_badges(
        self,
        repository_goal: _RepositoryGoal,
        *,
        generation_scope: object = _CONTROL_DOCUMENT_UNSET,
    ) -> tuple[str, ...]:
        badges, _run_count = self._current_goal_evidence(
            repository_goal,
            generation_scope=generation_scope,
        )
        return badges

    def environments(
        self,
        *,
        query: str = "",
        cursor: str | None = None,
    ) -> CatalogPage:
        normalized = str(query or "").strip().casefold()
        goal_counts = self._repository_environments()
        environment_badges: dict[str, tuple[str, ...]] = {}
        environment_run_counts: dict[str, int] = {}
        if self.control_bucket is not None:
            badges_by_environment: dict[str, list[tuple[str, ...]]] = {}
            repository_goals = self._repository_goals()
            generation_scopes = self._control_generation_scopes(repository_goals)
            for goal, scope in zip(repository_goals, generation_scopes, strict=True):
                badges, run_count = self._current_goal_evidence(
                    goal,
                    generation_scope=scope,
                )
                badges_by_environment.setdefault(goal.environment_id, []).append(badges)
                environment_run_counts[goal.environment_id] = (
                    environment_run_counts.get(goal.environment_id, 0) + run_count
                )
            for environment_id, goal_badges in badges_by_environment.items():
                if not goal_badges:
                    continue
                environment_badges[environment_id] = tuple(
                    badge
                    for badge in GOAL_CATALOG_SUCCESS_BADGES
                    if all(badge in badges for badges in goal_badges)
                )
        items = []
        for environment_id, goal_count in sorted(goal_counts.items()):
            badges = environment_badges.get(environment_id, ())
            if normalized and normalized not in _search_text(environment_id, badges):
                continue
            items.append(
                EnvironmentSummary(
                    name=environment_id,
                    goal_count=goal_count,
                    run_count=environment_run_counts.get(environment_id, 0),
                    success_badges=badges,
                ).to_dict()
            )
        return self._page(
            items,
            cursor,
            identity={
                "level": "environments",
                "query": normalized,
                "repo_root": str(self.repo_root),
            },
        )

    def initial_environments(self) -> dict[str, Any]:
        return {
            "items": [
                EnvironmentSummary(
                    name=environment_id,
                    goal_count=goal_count,
                    run_count=0,
                    success_badges=(),
                ).to_dict()
                for environment_id, goal_count in sorted(self._repository_environments().items())
            ],
            "next_cursor": None,
            "freshness": "partial",
            "warnings": [
                {
                    "code": "control_enrichment_pending",
                    "message": "Live control-catalog status is loading.",
                    "retryable": True,
                    "source": "control-catalog",
                }
            ],
        }

    def _load_run_catalog(
        self,
        *,
        environment_id: str,
        selected_goal_slug: str,
        selected_goal_variant_id: str,
        metric_specs: tuple[tuple[RankCriterion, tuple[str, ...]], ...],
        fallback_metric_specs: tuple[tuple[RankCriterion, tuple[str, ...]], ...],
        generation_scope: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], ...]:
        control_summaries = self._control_run_catalog(
            environment_id=environment_id,
            selected_goal_slug=selected_goal_slug,
            selected_goal_variant_id=selected_goal_variant_id,
            metric_specs=metric_specs,
            fallback_metric_specs=fallback_metric_specs,
            generation_scope=generation_scope,
        )
        if control_summaries is None:
            raise CatalogUnavailable(
                self.control_error
                or (
                    "Run discovery requires control-catalog authority. Configure "
                    "GRADLAB_CONTROL_R2 in the private operator configuration and retry."
                ),
                code=("catalog_configuration" if self.control_error else "catalog_unavailable"),
            )
        return control_summaries

    def _control_run_catalog(
        self,
        *,
        environment_id: str,
        selected_goal_slug: str,
        selected_goal_variant_id: str,
        metric_specs: tuple[tuple[RankCriterion, tuple[str, ...]], ...],
        fallback_metric_specs: tuple[tuple[RankCriterion, tuple[str, ...]], ...],
        generation_scope: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], ...] | None:
        if self.control_bucket is None or not selected_goal_slug:
            return None
        if generation_scope is None:
            generation_scope = self._control_generation_scope(
                goal_slug=selected_goal_slug,
                include_archives=True,
            )
        else:
            generation_scope = {
                **dict(generation_scope),
                "variants": [dict(item) for item in generation_scope.get("variants", ())],
                "runs": [dict(item) for item in generation_scope.get("runs", ())],
                "archive_pages": [dict(item) for item in generation_scope.get("archive_pages", ())],
            }
            for reference in generation_scope["archive_pages"]:
                page_document = self.control_bucket.get_json_optional(str(reference["page_key"]))
                if page_document is None:
                    raise CatalogIntegrityError("catalog archive page is missing")
                page = validate_goal_catalog_page(
                    page_document,
                    expected_digest=str(reference["page_sha256"]),
                )
                if page["goal_slug"] != selected_goal_slug:
                    raise CatalogIntegrityError("catalog archive page belongs to another goal")
                generation_scope["runs"].extend(dict(run) for run in page["runs"])
        if generation_scope is None:
            return ()
        raw_variants = generation_scope["variants"]
        variant_ids = [
            str(item.get("variant_id") or "")
            for item in raw_variants
            if isinstance(item, Mapping)
            and (not selected_goal_variant_id or item.get("variant_id") == selected_goal_variant_id)
        ]
        if len(variant_ids) != sum(
            1
            for item in raw_variants
            if isinstance(item, Mapping)
            and (not selected_goal_variant_id or item.get("variant_id") == selected_goal_variant_id)
        ) or any(not value for value in variant_ids):
            raise CatalogIntegrityError("catalog generation contains an invalid variant")
        by_variant: dict[str, list[Any]] = {variant_id: [] for variant_id in variant_ids}
        for raw in generation_scope["runs"]:
            if not isinstance(raw, Mapping):
                raise CatalogIntegrityError("catalog generation contains an invalid run")
            variant_id = str(raw.get("goal_variant_id") or "")
            if variant_id in by_variant:
                by_variant[variant_id].append(raw)
        run_groups = list(by_variant.items())
        summaries: list[dict[str, Any]] = []
        for variant_id, raw_runs in run_groups:
            for raw in raw_runs:
                if not isinstance(raw, Mapping):
                    raise CatalogIntegrityError("goal variant run index contains an invalid entry")
                run_id = str(raw.get("run_id") or "")
                metrics = raw.get("metrics")
                authored_hash = str(raw.get("goal_contract_sha256") or "")
                effective_hash = str(raw.get("effective_goal_contract_sha256") or "")
                if (
                    RUN_ID_PATTERN.fullmatch(run_id) is None
                    or raw.get("goal_slug") != selected_goal_slug
                    or raw.get("goal_variant_id") != variant_id
                    or not isinstance(metrics, Mapping)
                    or (
                        raw.get("early_stop") is not None
                        and not isinstance(raw.get("early_stop"), Mapping)
                    )
                ):
                    raise CatalogIntegrityError("goal variant run index contains an invalid run")
                summaries.append(
                    RunSummary(
                        environment_id=environment_id,
                        run_id=run_id,
                        name=str(raw.get("name") or run_id),
                        state=str(raw.get("state") or ""),
                        stop_reason=str(raw.get("stop_reason") or ""),
                        final_step=_safe_int(raw.get("final_step")),
                        early_stop=(
                            dict(raw["early_stop"])
                            if isinstance(raw.get("early_stop"), Mapping)
                            else None
                        ),
                        goal=selected_goal_slug,
                        recipe=str(raw.get("recipe_slug") or ""),
                        recipe_sha256=str(raw.get("recipe_sha256") or ""),
                        recipe_overrides=tuple(
                            str(value)
                            for value in raw.get("recipe_overrides", ())
                            if str(value).strip()
                        ),
                        recipe_variant_id=str(raw.get("recipe_variant_id") or ""),
                        goal_contract_sha256=authored_hash,
                        effective_goal_contract_sha256=effective_hash,
                        goal_variant_id=variant_id,
                        goal_variant_label=str(raw.get("goal_variant_label") or ""),
                        description=str(raw.get("description") or ""),
                        seed=_safe_int(raw.get("seed")),
                        created_at=str(raw.get("created_at") or ""),
                        updated_at=str(raw.get("updated_at") or ""),
                        url=str(raw.get("url") or ""),
                        metrics={str(name): _safe_float(value) for name, value in metrics.items()},
                        success_badges=_projected_run_success_badges(raw),
                    ).to_dict()
                )
        summaries.sort(
            key=lambda item: (
                str(item.get("updated_at") or ""),
                str(item.get("run_id") or ""),
            ),
            reverse=True,
        )
        return tuple(summaries)

    def runs(
        self,
        *,
        environment_id: str,
        goal_id: str = "",
        goal_variant_id: str = "",
        query: str = "",
        cursor: str | None = None,
        refresh: bool = False,
    ) -> CatalogPage:
        del refresh
        normalized = str(query or "").strip().casefold()
        selected_goal = str(goal_id or "").strip()
        repository_goal = (
            self._repository_goal(
                environment_id=environment_id,
                goal_id=selected_goal,
            )
            if selected_goal
            else None
        )
        selected_goal_slug = repository_goal.goal_slug if repository_goal else ""
        selected_goal_variant = str(goal_variant_id or "").strip()
        if (
            selected_goal_variant
            and re.fullmatch(r"goal-variant-[0-9a-f]{24}", selected_goal_variant) is None
        ):
            raise ValueError("invalid goal variant id")
        rank = (repository_goal.rank or ()) if repository_goal else ()
        metric_specs = _run_metric_specs(rank) if repository_goal else ()
        fallback_metric_specs = _run_fallback_metric_specs(rank) if repository_goal else ()
        metric_columns = tuple(
            {"metric": criterion.metric, "direction": criterion.direction}
            for criterion, _sources in metric_specs
        )
        fallback_metric_columns = tuple(
            {"metric": criterion.metric, "direction": criterion.direction}
            for criterion, _sources in fallback_metric_specs
        )
        generation_scope = (
            self._control_generation_scope(goal_slug=selected_goal_slug)
            if selected_goal_slug
            else None
        )
        generation_sha256 = str((generation_scope or {}).get("generation_sha256") or "empty")
        summaries = self._load_run_catalog(
            environment_id=environment_id,
            selected_goal_slug=selected_goal_slug,
            selected_goal_variant_id=selected_goal_variant,
            metric_specs=metric_specs,
            fallback_metric_specs=fallback_metric_specs,
            generation_scope=generation_scope,
        )
        filtered = [
            summary
            for summary in summaries
            if not normalized
            or normalized
            in _search_text(
                summary.get("run_id"),
                summary.get("name"),
                summary.get("state"),
                summary.get("stop_reason"),
                summary.get("early_stop"),
                summary.get("goal"),
                summary.get("recipe"),
                summary.get("recipe_sha256"),
                summary.get("recipe_overrides"),
                summary.get("recipe_variant_id"),
                summary.get("goal_contract_sha256"),
                summary.get("effective_goal_contract_sha256"),
                summary.get("goal_variant_id"),
                summary.get("goal_variant_label"),
                summary.get("description"),
                summary.get("seed"),
                summary.get("success_badges"),
            )
        ]
        page = self._page(
            filtered,
            cursor,
            identity={
                "authority": self.control_identity,
                "environment_id": environment_id,
                "goal_id": selected_goal,
                "goal_slug": selected_goal_slug,
                "goal_variant_id": selected_goal_variant,
                "query": normalized,
                "generation_sha256": generation_sha256,
            },
        )
        return CatalogPage(
            items=page.items,
            next_cursor=page.next_cursor,
            metric_columns=metric_columns,
            fallback_metric_columns=fallback_metric_columns,
            source=page.source,
            freshness="fresh",
        )

    def goals(
        self,
        *,
        environment_id: str,
        query: str = "",
        cursor: str | None = None,
    ) -> CatalogPage:
        normalized = str(query or "").strip().casefold()
        items = []
        repository_goals = self._repository_goals(environment_id=environment_id)
        generation_scopes = self._control_generation_scopes(repository_goals)
        for goal, scope in zip(repository_goals, generation_scopes, strict=True):
            badges = self._current_goal_success_badges(
                goal,
                generation_scope=scope,
            )
            if normalized and normalized not in _search_text(
                goal.goal_id,
                goal.goal_slug,
                goal.title,
                goal.goal_path,
                badges,
            ):
                continue
            items.append(
                GoalSummary(
                    environment_id=environment_id,
                    goal_id=goal.goal_id,
                    goal_slug=goal.goal_slug,
                    title=goal.title,
                    recipe_count=goal.recipe_count,
                    goal_path=goal.goal_path,
                    success_badges=badges,
                ).to_dict()
            )
        return self._page(
            items,
            cursor,
            identity={
                "level": "goals",
                "environment_id": environment_id,
                "query": normalized,
            },
        )

    def goal_variants(
        self,
        *,
        environment_id: str,
        goal_id: str,
        query: str = "",
        cursor: str | None = None,
        refresh: bool = False,
    ) -> CatalogPage:
        del refresh
        repository_goal = self._repository_goal(
            environment_id=environment_id,
            goal_id=goal_id,
        )
        generation_scope = self._control_generation_scope(goal_slug=repository_goal.goal_slug)
        generation_sha256 = str((generation_scope or {}).get("generation_sha256") or "empty")
        items = self._load_goal_variants(
            repository_goal=repository_goal,
            generation_scope=generation_scope,
        )
        normalized = str(query or "").strip().casefold()
        filtered = [
            item
            for item in items
            if not normalized
            or normalized
            in _search_text(
                item.get("display_label"),
                item.get("configuration_kind"),
                item.get("label"),
                item.get("variant_id"),
                item.get("status"),
                item.get("source_relation"),
                item.get("current_diff"),
                item.get("diff"),
                item.get("first_used_at"),
                item.get("last_activity_at"),
                item.get("run_count"),
                item.get("goal_contract_sha256"),
                item.get("effective_goal_contract_sha256"),
                item.get("success_badges"),
            )
        ]
        page = self._page(
            filtered,
            cursor,
            identity={
                "authority": self.control_identity,
                "environment_id": environment_id,
                "goal_id": goal_id,
                "goal_slug": repository_goal.goal_slug,
                "query": normalized,
                "generation_sha256": generation_sha256,
            },
        )
        return CatalogPage(
            items=page.items,
            next_cursor=page.next_cursor,
            source=page.source,
            freshness=("partial" if self.control_bucket is None else "fresh"),
            warnings=(
                (
                    {
                        "code": "control_catalog_unavailable",
                        "message": (
                            self.control_error
                            or "Historical goal variants require control-catalog authority."
                        ),
                        "retryable": False,
                        "source": "control-catalog",
                    },
                )
                if self.control_bucket is None
                else ()
            ),
        )

    def _schedule_catalog_repair(self, goal_slug: str, generation_sha256: str) -> None:
        with self._cache_lock:
            if goal_slug in self._catalog_repairing:
                return
            self._catalog_repairing.add(goal_slug)

        def repair() -> None:
            try:
                from gradlab.catalog_jobs import enqueue_catalog_projection

                enqueue_catalog_projection(
                    repo_root=self.repo_root,
                    goal_slug=goal_slug,
                    request_id=(
                        f"browse-{generation_sha256}"
                        if generation_sha256
                        else f"browse-empty-{int(time.time() // 60)}"
                    ),
                )
            except Exception:
                pass
            finally:
                with self._cache_lock:
                    self._catalog_repairing.discard(goal_slug)

        threading.Thread(
            target=repair,
            name=f"gradlab-catalog-repair-{compact_json_sha256(goal_slug)[:12]}",
            daemon=True,
        ).start()

    def goal_activity(
        self,
        *,
        environment_id: str,
        goal_id: str,
        query: str = "",
        refresh: bool = False,
    ) -> dict[str, Any]:
        """Return one generation-consistent view of variants and their recent runs."""

        repository_goal = self._repository_goal(
            environment_id=environment_id,
            goal_id=goal_id,
        )
        scope = self._control_generation_scope(goal_slug=repository_goal.goal_slug)
        generation_sha256 = str((scope or {}).get("generation_sha256") or "")
        self._schedule_catalog_repair(repository_goal.goal_slug, generation_sha256)
        current, _current_goal = self._current_goal_variant(repository_goal)
        variants = list(
            self._load_goal_variants(
                repository_goal=repository_goal,
                generation_scope=scope,
            )
        )
        normalized_query = str(query or "").strip().casefold()
        if normalized_query:
            variants = [
                variant
                for variant in variants
                if normalized_query
                in _search_text(
                    variant.get("display_label"),
                    variant.get("configuration_kind"),
                    variant.get("label"),
                    variant.get("variant_id"),
                    variant.get("recent_runs"),
                    variant.get("success_badges"),
                )
            ]
        rank = repository_goal.rank or ()
        metric_specs = _run_metric_specs(rank)
        fallback_specs = _run_fallback_metric_specs(rank)
        hot_runs = [dict(run) for run in (scope or {}).get("runs", ())]
        for variant in variants:
            variant_id = str(variant["variant_id"])
            recent = [dict(run) for run in hot_runs if run.get("goal_variant_id") == variant_id]
            recent.sort(
                key=lambda run: (str(run.get("updated_at") or ""), str(run.get("run_id") or "")),
                reverse=True,
            )
            best = [dict(run) for run in recent]
            _rank_run_summaries(best, primary=metric_specs, fallback=fallback_specs)
            variant["recent_runs"] = recent[:5]
            variant["best_runs"] = best[:5]
            variant["has_more_runs"] = int(variant.get("run_count") or 0) > 5
        revision = compact_json_sha256(
            {
                "repository_goal_sha256": current["effective_goal_contract_sha256"],
                "generation_sha256": generation_sha256,
                "query": normalized_query,
            },
            ensure_ascii=True,
        )
        return {
            "schema_version": 1,
            "items": variants,
            "next_cursor": None,
            "environment_id": environment_id,
            "goal_id": goal_id,
            "goal_slug": repository_goal.goal_slug,
            "repository_goal_sha256": current["effective_goal_contract_sha256"],
            "generation_sha256": generation_sha256,
            "revision": revision,
            "freshness": "fresh",
            "has_active_runs": any(
                str(run.get("state") or "") not in GOAL_CATALOG_TERMINAL_STATES for run in hot_runs
            ),
            "metric_columns": [
                {"metric": criterion.metric, "direction": criterion.direction}
                for criterion, _sources in metric_specs
            ],
            "fallback_metric_columns": [
                {"metric": criterion.metric, "direction": criterion.direction}
                for criterion, _sources in fallback_specs
            ],
            "source": {
                "kind": "goal-catalog-v1",
                "generation_sha256": generation_sha256,
                "revision": revision,
            },
            "warnings": [],
        }

    @staticmethod
    def _inspection_envelope(
        *,
        source: Mapping[str, Any],
        goal: Mapping[str, Any] | None = None,
        recipe: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        documents = {}
        if goal is not None:
            documents["goal"] = dict(goal)
        if recipe is not None:
            documents["recipe"] = dict(recipe)
        return {
            "schema_version": 1,
            "source": dict(source),
            "documents": documents,
        }

    @staticmethod
    def _preview_recipe(document: Mapping[str, Any]) -> dict[str, Any]:
        preview = json.loads(json.dumps(document))
        preview.pop("_composition", None)
        return preview

    def recipes(
        self,
        *,
        environment_id: str,
        goal_id: str,
        query: str = "",
        cursor: str | None = None,
    ) -> CatalogPage:
        repository_goal = self._repository_goal(
            environment_id=environment_id,
            goal_id=goal_id,
        )
        recipes_root = (self.repo_root / repository_goal.goal_path).parent / "recipes"
        normalized = str(query or "").strip().casefold()
        items: list[dict[str, Any]] = []
        for path in sorted(recipes_root.glob("*.yaml")):
            composed = load_recipe_source_document(path).document
            recipe_id = str(composed.get("recipe_id") or path.stem).strip()
            description = str(composed.get("description") or "").strip()
            item = {
                "environment_id": environment_id,
                "goal_id": goal_id,
                "goal_slug": repository_goal.goal_slug,
                "recipe_id": recipe_id,
                "title": recipe_id,
                "description": description,
                "path": path.relative_to(self.repo_root).as_posix(),
                "availability": "static-preview",
            }
            if not normalized or normalized in _search_text(
                recipe_id,
                description,
                item["path"],
            ):
                items.append(item)
        return self._page(
            items,
            cursor,
            identity={
                "level": "recipes",
                "environment_id": environment_id,
                "goal_id": goal_id,
                "query": normalized,
            },
        )

    def inspect_goal(
        self,
        *,
        environment_id: str,
        goal_id: str,
    ) -> dict[str, Any]:
        repository_goal = self._repository_goal(
            environment_id=environment_id,
            goal_id=goal_id,
        )
        authored = load_goal_contract(
            self.repo_root / repository_goal.goal_path,
            self.repo_root,
        )
        canonical = goal_for_contract_validation(
            authored,
            label=f"repository goal {repository_goal.goal_slug}",
        )
        descriptor = build_goal_variant_descriptor(
            goal_slug=repository_goal.goal_slug,
            source_sha="",
            authored_goal=authored,
            effective_goal=canonical,
        )
        goal = inspection_document(
            kind="goal",
            title=repository_goal.title,
            availability="exact",
            resolved=canonical,
            base=canonical,
            variant_id=str(descriptor["variant_id"]),
            metadata={
                "environment_id": environment_id,
                "goal_id": goal_id,
                "goal_slug": repository_goal.goal_slug,
                "goal_contract_sha256": descriptor["goal_contract_sha256"],
                "effective_goal_contract_sha256": descriptor["effective_goal_contract_sha256"],
            },
            allow_placeholders=True,
        )
        return self._inspection_envelope(
            source={
                "kind": "repository-goal",
                "goal_path": repository_goal.goal_path,
            },
            goal=goal,
        )

    def inspect_recipe(
        self,
        *,
        environment_id: str,
        goal_id: str,
        recipe_id: str,
    ) -> dict[str, Any]:
        repository_goal = self._repository_goal(
            environment_id=environment_id,
            goal_id=goal_id,
        )
        normalized_recipe_id = str(recipe_id or "").strip()
        if (
            not normalized_recipe_id
            or re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]*",
                normalized_recipe_id,
            )
            is None
        ):
            raise ValueError("invalid recipe id")
        goal_path = self.repo_root / repository_goal.goal_path
        recipe_path = goal_path.parent / "recipes" / f"{normalized_recipe_id}.yaml"
        if not recipe_path.is_file():
            raise ValueError(
                f"repository has no recipe {environment_id}/{goal_id}/{normalized_recipe_id}"
            )
        resolved = compose_resolved_train_documents(goal_path, recipe_path)
        preview = self._preview_recipe(resolved.base)
        recipe = inspection_document(
            kind="recipe",
            title=str(preview.get("recipe_id") or normalized_recipe_id),
            availability="static-preview",
            resolved=preview,
            base=preview,
            variant_id="base",
            message=(
                "Launch-bound values remain as placeholders until a run supplies "
                "its seed, description, assets, and runtime."
            ),
            metadata={
                "environment_id": environment_id,
                "goal_id": goal_id,
                "goal_slug": repository_goal.goal_slug,
                "recipe_id": normalized_recipe_id,
                "recipe_path": recipe_path.relative_to(self.repo_root).as_posix(),
            },
            allow_placeholders=True,
        )
        return self._inspection_envelope(
            source={
                "kind": "repository-recipe",
                "goal_path": repository_goal.goal_path,
                "recipe_path": recipe_path.relative_to(self.repo_root).as_posix(),
            },
            recipe=recipe,
        )

    def _control_recipe_document(self, recipe_sha256: str) -> dict[str, Any] | None:
        if self.control_bucket is None:
            return None
        key = RunAuthority.recipe_document_key(recipe_sha256)
        document = self.control_bucket.get_json_optional(key)
        if document is None:
            return None
        validated = validate_recipe_document(
            document,
            source=f"control recipe {recipe_sha256}",
        )
        if canonical_json_sha256(validated) != recipe_sha256:
            raise ValueError("control recipe document hash mismatch")
        return validated

    @staticmethod
    def _recipe_document_inspections(
        document: Mapping[str, Any],
        *,
        title: str,
        metadata: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        validated = validate_recipe_document(document, source=title)
        recipe = dict(validated["recipe"])
        resolution = validated["resolution"]
        goal_resolution = resolution.get("goal") if isinstance(resolution, Mapping) else None
        recipe_resolution = resolution.get("recipe") if isinstance(resolution, Mapping) else None
        if not isinstance(goal_resolution, Mapping) or not isinstance(
            recipe_resolution,
            Mapping,
        ):
            raise ValueError("recipe resolution proof is malformed")
        goal_base = dict(goal_resolution["base"])
        recipe_base = dict(recipe_resolution["base"])
        message = ""
        goal_variant = recipe.get("goal_variant")
        goal_variant_id = (
            str(goal_variant.get("variant_id") or "") if isinstance(goal_variant, Mapping) else ""
        )
        recipe_variant_id_value = str(recipe_resolution.get("variant_id") or "")
        common_metadata = {
            **dict(metadata),
            "recipe_format_version": int(validated["format_version"]),
            "recipe_sha256": canonical_json_sha256(validated),
        }
        return (
            inspection_document(
                kind="goal",
                title=str(recipe.get("goal", {}).get("title") or title),
                availability="exact",
                resolved=dict(recipe["goal"]),
                base=goal_base,
                variant_id=goal_variant_id,
                message=message,
                metadata=common_metadata,
                allow_placeholders=True,
            ),
            inspection_document(
                kind="recipe",
                title=str(recipe.get("recipe_id") or title),
                availability="exact",
                resolved=recipe,
                base=recipe_base,
                variant_id=recipe_variant_id_value,
                message=message,
                metadata=common_metadata,
                allow_placeholders=True,
            ),
        )

    def inspect_portable_recipe(
        self,
        document: Mapping[str, Any],
        *,
        source: Mapping[str, Any],
    ) -> dict[str, Any]:
        goal, recipe = self._recipe_document_inspections(
            document,
            title=str(source.get("artifact_name") or source.get("run_id") or "Active playback"),
            metadata=dict(source),
        )
        return self._inspection_envelope(
            source=source,
            goal=goal,
            recipe=recipe,
        )

    def _public_run_recipe_document(self, run_id: str) -> dict[str, Any] | None:
        index_url = f"{self.public_models_base_url}/runs/{run_id}/index.json"
        try:
            index = _public_json(index_url)
        except HTTPError as exc:
            if exc.code == 404:
                return None
            if 500 <= exc.code < 600:
                raise CatalogUnavailable(
                    f"public run index is temporarily unavailable: HTTP {exc.code}",
                    code="public_catalog_transient",
                    retryable=True,
                    source="public-models",
                ) from exc
            raise CatalogIntegrityError(
                f"public run index request failed: HTTP {exc.code}",
                source="public-models",
            ) from exc
        except (TimeoutError, URLError, OSError) as exc:
            raise CatalogUnavailable(
                f"public run index is temporarily unavailable: {exc}",
                code="public_catalog_transient",
                retryable=True,
                source="public-models",
            ) from exc
        except Exception as exc:
            raise CatalogIntegrityError(
                f"public run index is malformed: {exc}",
                source="public-models",
            ) from exc
        if int(index.get("schema_version") or 0) != 1 or index.get("run_id") != run_id:
            raise CatalogIntegrityError(
                "public run index identity mismatch",
                source="public-models",
            )
        manifests = []
        for raw in index.get("checkpoints") or ():
            if not isinstance(raw, Mapping):
                raise CatalogIntegrityError(
                    "public run index contains an invalid checkpoint",
                    source="public-models",
                )
            try:
                manifest = CheckpointManifest.from_dict(raw)
            except Exception as exc:
                raise CatalogIntegrityError(
                    f"public checkpoint manifest is invalid: {exc}",
                    source="public-models",
                ) from exc
            manifests.append(manifest)
        if not manifests:
            return None
        manifest = max(manifests, key=lambda item: (item.step, item.checkpoint_id))
        try:
            document = validate_recipe_document(
                _public_json(manifest.recipe_document_url),
                source=manifest.recipe_document_url,
            )
        except HTTPError as exc:
            if exc.code == 404:
                return None
            if 500 <= exc.code < 600:
                raise CatalogUnavailable(
                    f"public recipe proof is temporarily unavailable: HTTP {exc.code}",
                    code="public_catalog_transient",
                    retryable=True,
                    source="public-models",
                ) from exc
            raise CatalogIntegrityError(
                f"public recipe proof request failed: HTTP {exc.code}",
                source="public-models",
            ) from exc
        except (TimeoutError, URLError, OSError) as exc:
            raise CatalogUnavailable(
                f"public recipe proof is temporarily unavailable: {exc}",
                code="public_catalog_transient",
                retryable=True,
                source="public-models",
            ) from exc
        except Exception as exc:
            raise CatalogIntegrityError(
                f"public recipe proof is invalid: {exc}",
                source="public-models",
            ) from exc
        observed = canonical_json_sha256(document)
        if observed != manifest.recipe_document_sha256 or observed != manifest.recipe_sha256:
            raise CatalogIntegrityError(
                "public recipe document hash mismatch",
                source="public-models",
            )
        if manifest.run_id != run_id:
            raise CatalogIntegrityError(
                "public checkpoint run identity mismatch",
                source="public-models",
            )
        return document

    def _run_recipe_document(
        self,
        run_id: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        manifest_document = (
            self.control_bucket.get_json_optional(f"runs/{run_id}/manifest.json")
            if self.control_bucket is not None
            else None
        )
        document = None
        metadata: dict[str, Any] = {"run_id": run_id}
        if manifest_document is not None:
            manifest = RunManifest.from_dict(manifest_document)
            if manifest.run_id != run_id:
                raise ValueError("run manifest identity mismatch")
            metadata.update(
                {
                    "goal_slug": manifest.goal_slug,
                    "recipe_slug": manifest.recipe_slug,
                    "recipe_sha256": manifest.recipe_sha256,
                }
            )
            document = self._control_recipe_document(manifest.recipe_sha256)
        if document is None:
            document = self._public_run_recipe_document(run_id)
        return document, metadata

    def inspect_run(
        self,
        *,
        run_id: str,
    ) -> dict[str, Any]:
        if RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise ValueError("run id must match gradlab-<32 lowercase hex>")
        document, metadata = self._run_recipe_document(run_id)
        if document is None:
            unavailable = inspection_document(
                kind="recipe",
                title=run_id,
                availability="summary-only",
                message=(
                    "The run summary is available, but no verified portable recipe "
                    "document with resolved YAML is accessible."
                ),
                metadata=metadata,
            )
            return self._inspection_envelope(
                source={"kind": "run-summary", "run_id": run_id},
                recipe=unavailable,
            )
        goal, recipe = self._recipe_document_inspections(
            document,
            title=run_id,
            metadata=metadata,
        )
        return self._inspection_envelope(
            source={"kind": "run", "run_id": run_id},
            goal=goal,
            recipe=recipe,
        )

    def inspect_goal_variant(
        self,
        *,
        environment_id: str,
        goal_id: str,
        variant_id: str,
    ) -> dict[str, Any]:
        repository_goal = self._repository_goal(
            environment_id=environment_id,
            goal_id=goal_id,
        )
        selected = None
        cursor = None
        while selected is None:
            page = self.goal_variants(
                environment_id=environment_id,
                goal_id=goal_id,
                cursor=cursor,
            )
            selected = next(
                (item for item in page.items if item.get("variant_id") == variant_id),
                None,
            )
            cursor = page.next_cursor
            if cursor is None:
                break
        if selected is None:
            raise ValueError(f"goal variant does not exist: {variant_id}")
        _current_descriptor, current_goal = self._current_goal_variant(repository_goal)
        exact_run_id = str(selected.get("exact_resolution_run_id") or "")
        if not exact_run_id:
            if selected.get("configuration_kind") == "current_default":
                goal = inspection_document(
                    kind="goal",
                    title=repository_goal.title,
                    availability="exact",
                    resolved=current_goal,
                    base=current_goal,
                    variant_id=variant_id,
                    message="This configuration matches the current checked-in goal.",
                    metadata=dict(selected),
                    allow_placeholders=True,
                )
                return {
                    **self._inspection_envelope(
                        source={"kind": "repository-goal-variant", "variant_id": variant_id},
                        goal=goal,
                    ),
                    "goal_diff": {
                        "availability": "exact",
                        "baseline": "current_checked_in_goal",
                        "change_count": 0,
                        "entries": [],
                        "message": "",
                    },
                }
            summary = inspection_document(
                kind="goal",
                title=str(selected.get("label") or repository_goal.title),
                availability="summary-only",
                variant_id=variant_id,
                message=(
                    "This historical variant has summary metadata but no verified "
                    "format-v2 resolution proof."
                ),
                metadata=dict(selected),
            )
            return {
                **self._inspection_envelope(
                    source={"kind": "goal-variant-summary", "variant_id": variant_id},
                    goal=summary,
                ),
                "goal_diff": {
                    "availability": "unavailable",
                    "baseline": "current_checked_in_goal",
                    "change_count": None,
                    "entries": [],
                    "message": (
                        "The exact historical contract is not sufficiently proven, "
                        "so no field-level comparison is shown."
                    ),
                },
            }
        metadata: dict[str, Any] = {"run_id": exact_run_id}
        recipe = None
        try:
            document, metadata = self._run_recipe_document(exact_run_id)
        except ValueError:
            document = None
        if document is not None:
            validated = validate_recipe_document(document, source=exact_run_id)
            recipe_document = dict(validated["recipe"])
            raw_descriptor = recipe_document.get("goal_variant")
            if not isinstance(raw_descriptor, Mapping):
                raise ValueError("goal variant resolution recipe has no descriptor")
            descriptor = validate_goal_variant_descriptor(raw_descriptor)
            if descriptor["variant_id"] != variant_id:
                raise ValueError("goal variant resolution run does not prove the selected variant")
            resolved_goal = recipe_document.get("goal")
            if not isinstance(resolved_goal, Mapping):
                raise ValueError("goal variant resolution recipe has no resolved goal")
            _original_goal, recipe = self._recipe_document_inspections(
                validated,
                title=exact_run_id,
                metadata=metadata,
            )
        else:
            resolved_goal = self._variant_resolved_goal_from_exact_run(
                variant_id=variant_id,
                exact_run_id=exact_run_id,
            )
            if resolved_goal is None:
                raise ValueError("goal variant resolution run has no verified goal proof")
            recipe = inspection_document(
                kind="recipe",
                title=exact_run_id,
                availability="summary-only",
                message=(
                    "The resolved goal comparison is verified, but the full historical "
                    "recipe is outside the current inspectable recipe contract."
                ),
                metadata=metadata,
            )
        kind_labels = {
            "current_default": "Current default",
            "current_modified": "Current modified",
            "previous_default": "Previous default",
            "previous_modified": "Previous modified",
        }
        kind_label = kind_labels.get(
            str(selected.get("configuration_kind") or ""),
            "Goal configuration",
        )
        goal = inspection_document(
            kind="goal",
            title=f"{kind_label} · {selected.get('display_label') or repository_goal.title}",
            availability="exact",
            resolved=dict(resolved_goal),
            base=current_goal,
            variant_id=variant_id,
            message=(
                "Goal changes compare this configuration with the current default. "
                + (
                    "Recipe changes show its launch-time recipe differences."
                    if recipe["availability"] == "exact"
                    else "The full historical recipe is shown as summary-only."
                )
            ),
            metadata={**metadata, **dict(selected)},
            allow_placeholders=True,
        )
        exact_changes = goal_contract_structural_diff(current_goal, resolved_goal)
        return {
            "source": {
                "kind": "goal-variant",
                "variant_id": variant_id,
                "exact_resolution_run_id": exact_run_id,
            },
            "schema_version": 1,
            "documents": {"goal": goal, "recipe": recipe},
            "goal_diff": {
                "availability": "exact",
                "baseline": "current_checked_in_goal",
                "change_count": len(exact_changes),
                "entries": exact_changes,
                "message": "",
            },
        }

    def run_goal(self, *, environment_id: str, run_id: str) -> str:
        goal_id, _variant_id = self.run_goal_variant(
            environment_id=environment_id,
            run_id=run_id,
        )
        return goal_id

    def public_run_route(self, *, run_id: str) -> dict[str, str]:
        """Return the hierarchical checkpoint-browser route proven by a public run."""
        if RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise ValueError("run id must match gradlab-<32 lowercase hex>")
        public_document = self._public_run_recipe_document(run_id)
        recipe = public_document.get("recipe") if isinstance(public_document, Mapping) else None
        raw_descriptor = recipe.get("goal_variant") if isinstance(recipe, Mapping) else None
        if not isinstance(raw_descriptor, Mapping):
            raise CatalogUnavailable(
                "No verified public recipe currently proves this run's goal variant.",
                code="public_proof_absent",
                retryable=False,
                source="public-models",
            )
        descriptor = validate_goal_variant_descriptor(raw_descriptor)
        goal_slug = str(descriptor["goal_slug"])
        environment_id, separator, _goal_path = goal_slug.partition("/")
        if not separator:
            environment_id = goal_slug
        if not environment_id:
            raise ValueError("public run goal slug is empty")
        for goal in self._repository_goals(environment_id=environment_id):
            if goal.goal_slug == goal_slug:
                return {
                    "level": "runs",
                    "environment_id": environment_id,
                    "goal_id": goal.goal_id,
                    "goal_variant_id": str(descriptor["variant_id"]),
                    "run_id": run_id,
                    "checkpoint_id": "",
                }
        raise ValueError(f"run goal is not declared in the repository: {goal_slug}")

    def run_goal_variant(
        self,
        *,
        environment_id: str,
        run_id: str,
    ) -> tuple[str, str]:
        if RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise ValueError("run id must match gradlab-<32 lowercase hex>")
        descriptor: Mapping[str, Any] | None = None
        if self.control_bucket is not None:
            manifest = self.control_bucket.get_json_optional(f"runs/{run_id}/manifest.json")
            if manifest is not None:
                raw_descriptor = manifest.get("goal_variant")
                if not isinstance(raw_descriptor, Mapping):
                    raise ValueError("run manifest has no goal variant descriptor")
                descriptor = raw_descriptor
        if descriptor is None:
            public_document = self._public_run_recipe_document(run_id)
            recipe = public_document.get("recipe") if isinstance(public_document, Mapping) else None
            raw_descriptor = recipe.get("goal_variant") if isinstance(recipe, Mapping) else None
            if isinstance(raw_descriptor, Mapping):
                descriptor = raw_descriptor
        if descriptor is None:
            raise CatalogUnavailable(
                "No verified public recipe currently proves this W&B run's goal variant.",
                code="public_proof_absent",
                retryable=False,
                source="public-models",
            )
        validated = validate_goal_variant_descriptor(descriptor)
        goal_slug = str(validated["goal_slug"])
        for goal in self._repository_goals(environment_id=environment_id):
            if goal.goal_slug == goal_slug:
                return goal.goal_id, str(validated["variant_id"])
        raise ValueError(f"run goal is not declared in the repository: {goal_slug}")

    def _checkpoint_evaluations(
        self,
        *,
        run_id: str,
        metric_contract: CheckpointMetricContract,
        include_wandb: bool,
    ) -> _CheckpointEvaluationData:
        def validate_wandb_config(config: Mapping[str, Any]) -> None:
            schema_version = require_current_metrics_schema(config.get("metrics_schema_version"))
            if schema_version != metric_contract.metrics_schema_version:
                raise ValueError("W&B metrics schema disagrees with the immutable recipe")
            observed_backend = str(config.get("checkpoint_eval_backend") or "").strip()
            if observed_backend != metric_contract.evaluation_backend:
                raise ValueError(
                    "W&B checkpoint evaluation backend disagrees with the immutable recipe"
                )
            observed_rank = require_objective_rank(
                config.get("selection_rank"),
                metrics_schema_version=schema_version,
            )
            if objective_rank_strings(observed_rank) != objective_rank_strings(
                metric_contract.rank
            ):
                raise ValueError("W&B checkpoint ranking disagrees with the immutable recipe")
            raw_contract = config.get("checkpoint_eval_contract")
            if metric_contract.evaluation_backend == "none":
                if raw_contract is not None:
                    raise ValueError(
                        "W&B exposes a checkpoint evaluation contract for an evaluation-disabled recipe"
                    )
                raw_acceptance = config.get("checkpoint_eval_acceptance")
                observed_acceptance = (
                    tuple(
                        normalize_metric_threshold_rules(
                            raw_acceptance,
                            label="W&B checkpoint_eval_acceptance",
                            metric_validator=lambda name: validate_evaluation_scientific_metric(
                                name,
                                schema_version=schema_version,
                            ),
                        )
                    )
                    if raw_acceptance is not None
                    else ()
                )
                if observed_acceptance != metric_contract.acceptance:
                    raise ValueError(
                        "W&B checkpoint acceptance disagrees with the immutable recipe"
                    )
            elif metric_contract.acceptance:
                if not isinstance(raw_contract, Mapping):
                    raise ValueError("W&B checkpoint evaluation contract is missing")
                observed_acceptance = tuple(
                    normalize_metric_threshold_rules(
                        raw_contract.get("acceptance"),
                        label="W&B checkpoint_eval_contract.acceptance",
                        metric_validator=lambda name: validate_evaluation_scientific_metric(
                            name,
                            schema_version=schema_version,
                        ),
                    )
                )
                if observed_acceptance != metric_contract.acceptance:
                    raise ValueError(
                        "W&B checkpoint acceptance disagrees with the immutable recipe"
                    )
            elif isinstance(raw_contract, Mapping) and raw_contract.get("acceptance"):
                raise ValueError("W&B exposes checkpoint acceptance for a training-only recipe")

        manifest = (
            self.control_bucket.get_json_optional(f"runs/{run_id}/manifest.json")
            if self.control_bucket is not None
            else None
        )
        if isinstance(manifest, Mapping):
            goal_slug = str(manifest.get("goal_slug") or "")
            generation = (
                self._control_generation_scope(
                    goal_slug=goal_slug,
                    include_archives=True,
                )
                if goal_slug
                else None
            )
            projected_run = next(
                (
                    run
                    for run in (generation or {}).get("runs", ())
                    if str(run.get("run_id") or "") == run_id
                ),
                None,
            )
            raw_evaluations = (
                projected_run.get("evaluations") if isinstance(projected_run, Mapping) else None
            )
            evaluations: dict[int, dict[str, Any]] = {}
            evaluation_seed = None
            if isinstance(raw_evaluations, Mapping):
                for checkpoint_id, raw in raw_evaluations.items():
                    if not isinstance(raw, Mapping):
                        continue
                    match = re.fullmatch(r"checkpoint-(\d+)-[0-9a-f]{16}", str(checkpoint_id))
                    if match is None:
                        continue
                    step = int(match.group(1))
                    metrics = {
                        str(name): _safe_float(value)
                        for name, value in dict(raw.get("metrics") or {}).items()
                    }
                    status = str(raw.get("status") or "")
                    candidate_seed = _safe_int(raw.get("seed"))
                    if evaluation_seed is None and candidate_seed is not None:
                        evaluation_seed = candidate_seed
                    ranked_metrics = dict(metrics)
                    ranked_metrics.setdefault(LEADER_CHECKPOINT_STEP, float(step))
                    evaluations[step] = {
                        "status": status,
                        "pass": status == "accepted",
                        "episodes_planned": _safe_int(
                            raw.get("episodes_planned") or metrics.get("episodes_planned")
                        ),
                        "episodes_completed": _safe_int(
                            raw.get("episodes_completed") or metrics.get("episodes_completed")
                        ),
                        **(
                            {"failure_count": _safe_int(raw.get("failure_count"))}
                            if raw.get("failure_count") is not None
                            else {}
                        ),
                        "criteria": [
                            dict(item)
                            for item in raw.get("criteria", ())
                            if isinstance(item, Mapping)
                        ],
                        "metrics": ranked_metrics,
                    }
            training_metric_history = {}
            warning = None
            training_seed = _safe_int(manifest.get("seed"))
            wandb = manifest.get("wandb")
            entity = str(wandb.get("entity") or "").strip() if isinstance(wandb, Mapping) else ""
            project = str(wandb.get("project") or "").strip() if isinstance(wandb, Mapping) else ""
            if include_wandb and entity and project:
                try:
                    run = self._wandb_api().run(f"{entity}/{project}/{run_id}")
                    config = dict(getattr(run, "config", {}) or {})
                    validate_wandb_config(config)
                    if training_seed is None:
                        training_seed = _safe_int(config.get("seed"))
                    training_metric_history = _checkpoint_training_metric_history(
                        run,
                        metric_contract.columns,
                    )
                except Exception as exc:
                    warning = {
                        "code": (
                            "wandb_contract_mismatch"
                            if isinstance(exc, ValueError)
                            else "wandb_enrichment_unavailable"
                        ),
                        "message": f"Optional W&B training history is unavailable: {exc}",
                        "retryable": isinstance(exc, (TimeoutError, OSError)),
                        "source": "wandb",
                    }
            return _CheckpointEvaluationData(
                evaluations=evaluations,
                training_seed=training_seed,
                evaluation_seed=evaluation_seed,
                training_metric_history=training_metric_history,
                warning=warning,
            )
        wandb_location = self._wandb_run_locations.get(run_id)
        entity = str(wandb_location.entity or "").strip() if wandb_location else ""
        project = str(wandb_location.project or "").strip() if wandb_location else ""
        if not include_wandb or not entity or not project:
            return _CheckpointEvaluationData({}, None, None, {})
        run = None
        try:
            run = self._wandb_api().run(f"{entity}/{project}/{run_id}")
            config = dict(getattr(run, "config", {}) or {})
            validate_wandb_config(config)
            require_current_metrics_schema(metric_contract.metrics_schema_version)
            training_seed = _safe_int(config.get("seed"))
            contract = config.get("checkpoint_eval_contract")
            if metric_contract.evaluation_backend == "none" or not metric_contract.acceptance:
                evaluations = {}
                evaluation_seed = None
            else:
                assert isinstance(contract, Mapping)
                evaluation_seed = _safe_int(contract.get("seed"))
                rules = metric_contract.acceptance
                result_keys = {
                    EVAL_CHECKPOINT_STEP,
                    EVAL_ACCEPTANCE_PASS,
                    EVAL_ACCEPTANCE_EPISODE_PLANNED_COUNT,
                    EVAL_ACCEPTANCE_EPISODE_COMPLETED_COUNT,
                }
                evaluations = {}
                for raw in run.scan_history(keys=sorted(result_keys), page_size=10_000):
                    if not isinstance(raw, Mapping):
                        continue
                    step = _safe_int(raw.get(EVAL_CHECKPOINT_STEP))
                    accepted = _safe_float(raw.get(EVAL_ACCEPTANCE_PASS))
                    if step is None or accepted is None:
                        continue
                    criteria: list[dict[str, Any]] = []
                    for rule in rules:
                        metric = str(rule["metric"])
                        operator = str(rule["operator"])
                        threshold = float(rule["threshold"])
                        value = _safe_float(raw.get(metric))
                        criteria.append(
                            {
                                "metric": metric,
                                "operator": operator,
                                "threshold": threshold,
                                "value": value,
                                "passed": (
                                    None
                                    if value is None
                                    else bool(EARLY_STOP_OPERATORS[operator](value, threshold))
                                ),
                            }
                        )
                    evaluations[step] = {
                        "status": "accepted" if accepted >= 0.5 else "rejected",
                        "pass": accepted >= 0.5,
                        "episodes_planned": _safe_int(
                            raw.get(EVAL_ACCEPTANCE_EPISODE_PLANNED_COUNT)
                        ),
                        "episodes_completed": _safe_int(
                            raw.get(EVAL_ACCEPTANCE_EPISODE_COMPLETED_COUNT)
                        ),
                        "criteria": criteria,
                        "metrics": {LEADER_CHECKPOINT_STEP: float(step)},
                    }
                # Fail-fast rejections intentionally omit completed eval/full metrics.
                # W&B returns no rows when scan_history requests a key that is absent
                # from some history records, so fetch each optional criterion
                # independently and merge it into the authoritative verdict rows.
                for rule_index, rule in enumerate(rules):
                    metric = str(rule["metric"])
                    for raw in run.scan_history(
                        keys=[EVAL_CHECKPOINT_STEP, metric],
                        page_size=10_000,
                    ):
                        if not isinstance(raw, Mapping):
                            continue
                        step = _safe_int(raw.get(EVAL_CHECKPOINT_STEP))
                        value = _safe_float(raw.get(metric))
                        evaluation = evaluations.get(step) if step is not None else None
                        if evaluation is None or value is None:
                            continue
                        criterion = evaluation["criteria"][rule_index]
                        criterion["value"] = value
                        criterion["passed"] = bool(
                            EARLY_STOP_OPERATORS[str(rule["operator"])](
                                value,
                                float(rule["threshold"]),
                            )
                        )
                evaluation_metric_names = dict.fromkeys(
                    str(column["metric"])
                    for column in metric_contract.columns
                    if column.get("evidence") == "evaluation"
                )
                evaluation_metric_names.pop(LEADER_CHECKPOINT_STEP, None)
                for metric in evaluation_metric_names:
                    for raw in run.scan_history(
                        keys=[EVAL_CHECKPOINT_STEP, metric],
                        page_size=10_000,
                    ):
                        if not isinstance(raw, Mapping):
                            continue
                        step = _safe_int(raw.get(EVAL_CHECKPOINT_STEP))
                        value = _safe_float(raw.get(metric))
                        evaluation = evaluations.get(step) if step is not None else None
                        if evaluation is not None and value is not None:
                            evaluation["metrics"][metric] = value
            training_metric_history = _checkpoint_training_metric_history(
                run,
                metric_contract.columns,
            )
        except Exception as exc:
            # Public checkpoints remain playable when W&B history is unavailable.
            evaluations = {}
            training_seed = None
            evaluation_seed = None
            training_metric_history = {}
            warning = {
                "code": (
                    "wandb_contract_mismatch"
                    if isinstance(exc, ValueError)
                    else "wandb_enrichment_unavailable"
                ),
                "message": f"W&B enrichment is unavailable: {exc}",
                "retryable": isinstance(exc, (TimeoutError, OSError)),
                "source": "wandb",
            }
        else:
            warning = None
        data = _CheckpointEvaluationData(
            evaluations=evaluations,
            training_seed=training_seed,
            evaluation_seed=evaluation_seed,
            training_metric_history=training_metric_history,
            warning=warning,
        )
        return data

    def checkpoints(
        self,
        *,
        run_id: str,
        query: str = "",
        goal_variant_id: str = "",
        include_wandb: bool = True,
    ) -> CheckpointPage:
        if RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise ValueError("run id must match gradlab-<32 lowercase hex>")
        url = f"{self.public_models_base_url}/runs/{run_id}/index.json"
        with ThreadPoolExecutor(max_workers=2) as executor:
            index_future = executor.submit(_public_json, url)
            recipe_future = executor.submit(self._run_recipe_document, run_id)
            index = index_future.result()
        if int(index.get("schema_version") or 0) != 1:
            raise ValueError("unsupported public run index schema")
        if str(index.get("run_id") or "") != run_id:
            raise ValueError("public run index identity mismatch")
        promotion = index.get("promotion")
        promoted_id = (
            str(promotion.get("checkpoint_id") or "") if isinstance(promotion, Mapping) else ""
        )
        manifests: list[CheckpointManifest] = []
        for raw in index.get("checkpoints") or ():
            if not isinstance(raw, Mapping):
                raise ValueError("public run index contains an invalid checkpoint")
            manifest = CheckpointManifest.from_dict(raw)
            if manifest.run_id != run_id:
                raise ValueError("checkpoint does not belong to the selected run")
            manifests.append(manifest)
        selection_fence = evaluation_selection_fence(
            run_id=run_id,
            checkpoints=[manifest.to_dict() for manifest in manifests],
        )
        warnings: list[Mapping[str, Any]] = []
        recipe_document: Mapping[str, Any] | None = None
        metric_contract: CheckpointMetricContract | None = None
        try:
            recipe_document, _metadata = recipe_future.result()
            if recipe_document is None:
                raise ValueError("no hash-verified immutable recipe is available")
            recipe_digest = canonical_json_sha256(recipe_document)
            if any(
                manifest.recipe_sha256 != recipe_digest
                or manifest.recipe_document_sha256 != recipe_digest
                for manifest in manifests
            ):
                raise ValueError("checkpoint recipes do not match the immutable run recipe")
            recipe = recipe_document.get("recipe")
            train_config = recipe.get("train_config") if isinstance(recipe, Mapping) else None
            if not isinstance(train_config, Mapping):
                raise ValueError("immutable recipe has no train_config")
            metric_contract = checkpoint_metric_contract(train_config)
        except Exception as exc:
            warnings.append(
                {
                    "code": "checkpoint_metric_contract_unavailable",
                    "message": (
                        "Checkpoint metric semantics are unavailable; showing structural "
                        f"checkpoint data only: {exc}"
                    ),
                    "retryable": isinstance(
                        exc,
                        (CatalogUnavailable, TimeoutError, URLError, OSError),
                    ),
                    "source": "recipe",
                }
            )

        expected_effective_goal_hash = ""
        selected_variant = str(goal_variant_id or "").strip()
        if selected_variant:
            raw_descriptor = None
            recipe = recipe_document.get("recipe") if isinstance(recipe_document, Mapping) else None
            if isinstance(recipe, Mapping):
                raw_descriptor = recipe.get("goal_variant")
            if not isinstance(raw_descriptor, Mapping):
                manifest_document = (
                    self.control_bucket.get_json_optional(f"runs/{run_id}/manifest.json")
                    if self.control_bucket is not None
                    else None
                )
                raw_descriptor = (
                    manifest_document.get("goal_variant")
                    if isinstance(manifest_document, Mapping)
                    else None
                )
            if not isinstance(raw_descriptor, Mapping):
                raise ValueError("run has no exact goal variant descriptor")
            descriptor = validate_goal_variant_descriptor(raw_descriptor)
            expected_effective_goal_hash = str(descriptor["effective_goal_contract_sha256"])
            if str(descriptor["variant_id"]) != selected_variant:
                raise ValueError("run does not belong to the selected goal variant")
        for manifest in manifests:
            if (
                expected_effective_goal_hash
                and manifest.goal_sha256 != expected_effective_goal_hash
            ):
                raise ValueError(
                    "checkpoint effective goal contract does not match its run variant"
                )

        evaluation_data = (
            self._checkpoint_evaluations(
                run_id=run_id,
                metric_contract=metric_contract,
                include_wandb=include_wandb,
            )
            if metric_contract is not None
            else _CheckpointEvaluationData({}, None, None, {})
        )
        if isinstance(evaluation_data.warning, Mapping):
            warnings.append(dict(evaluation_data.warning))
        columns = metric_contract.columns if metric_contract is not None else ()
        rows: list[CheckpointSummary] = []
        for manifest in manifests:
            evaluation = evaluation_data.evaluations.get(manifest.step)
            playback_seed = (
                evaluation_data.evaluation_seed
                if evaluation is not None and evaluation_data.evaluation_seed is not None
                else evaluation_data.training_seed
            )
            playback_seed_source = (
                "evaluation"
                if evaluation is not None and evaluation_data.evaluation_seed is not None
                else "training"
                if playback_seed is not None
                else None
            )
            training_metrics = _checkpoint_training_metrics(
                evaluation_data.training_metric_history,
                checkpoint_step=manifest.step,
                columns=columns,
            )
            row = CheckpointSummary(
                run_id=run_id,
                checkpoint_id=manifest.checkpoint_id,
                step=manifest.step,
                purpose=manifest.purpose,
                size_bytes=manifest.size_bytes,
                created_at=manifest.created_at,
                sha256=manifest.sha256,
                manifest_url=checkpoint_manifest_url(manifest.public_url),
                promoted=manifest.checkpoint_id == promoted_id,
                playback_seed=playback_seed,
                playback_seed_source=playback_seed_source,
                metrics=checkpoint_metric_values(
                    training_metrics,
                    evaluation,
                    columns,
                ),
                evaluation=evaluation,
            )
            rows.append(row)
        rows.sort(key=lambda row: (row.step, row.sha256), reverse=True)
        enriched_rows = checkpoint_metric_leaders(
            [row.to_dict() for row in rows],
            columns,
        )
        return CheckpointPage(
            items=filter_checkpoint_summaries(enriched_rows, query=query),
            metric_columns=columns,
            selection_fence=selection_fence,
            freshness="partial" if warnings else "fresh",
            warnings=tuple(warnings),
        )


def is_wandb_url(value: object) -> bool:
    return parse_wandb_location(value) is not None


def normalize_search_query(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:200]


__all__ = [
    "CATALOG_PAGE_SIZE",
    "CatalogPage",
    "CheckpointMetricContract",
    "CheckpointPage",
    "CheckpointSummary",
    "checkpoint_metric_contract",
    "checkpoint_metric_columns",
    "checkpoint_metric_leaders",
    "checkpoint_metric_values",
    "EnvironmentSummary",
    "GoalSummary",
    "GoalVariantSummary",
    "filter_checkpoint_summaries",
    "PlayCatalog",
    "RunSummary",
    "WandbRunLocation",
    "checkpoint_manifest_url",
    "is_wandb_url",
    "normalize_search_query",
    "parse_wandb_location",
]
