from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlparse

from gradlab.config_loader import load_mapping_document
from gradlab.contract_inspection import inspection_document
from gradlab.early_stop import EARLY_STOP_OPERATORS, normalize_metric_threshold_rules
from gradlab.goal_variants import (
    GOAL_VARIANT_INDEX_SCHEMA_VERSION,
    GOAL_VARIANT_RUN_INDEX_SCHEMA_VERSION,
    build_goal_variant_descriptor,
    goal_variant_id as compute_goal_variant_id,
    goal_variant_scope_key,
    unknown_goal_variant_id,
    validate_goal_variant_descriptor,
)
from gradlab.evaluation_projection import validate_evaluation_scientific_metric
from gradlab.metric_names import (
    EVAL_ACCEPTANCE_PASS,
    METRICS_SCHEMA_VERSION,
    TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_ROLLING_UP_TO_100_MEAN,
    TRAIN_EPISODE_RETURN_SHAPED_ACROSS_ORIGINS_ROLLING_UP_TO_100_MEAN,
    TRAIN_GLOBAL_STEP,
    TRAIN_OUTCOME_SUCCESS_ACROSS_STARTS_WINDOW_100_RATE_MIN,
    evaluation_metric_schema,
)
from gradlab.model_sources import DEFAULT_PUBLIC_MODELS_BASE_URL, _public_json
from gradlab.policy_bundle import canonical_json_sha256, validate_recipe_document
from gradlab.r2_store import BucketConfig, R2Bucket
from gradlab.ranking import RankCriterion, parse_objective_rank
from gradlab.recipe_variants import normalize_recipe_overrides, recipe_variant_id
from gradlab.recipe_documents import (
    compose_resolved_train_documents,
    load_goal_contract,
    load_recipe_source_document,
)
from gradlab.reward_programs import goal_for_contract_validation
from gradlab.run_contracts import (
    RUN_EARLY_STOP_CONDITION_SUMMARY,
    RUN_EARLY_STOP_TRIGGER_SUMMARY,
    RUN_FINAL_STEP_SUMMARY,
    RUN_ID_PATTERN,
    RUN_STOP_REASON_SUMMARY,
    RUN_TERMINAL_STATE_SUMMARY,
    CheckpointManifest,
    RunManifest,
)
from gradlab.run_authority import RunAuthority
from gradlab.wandb_utils import (
    load_wandb_env,
    resolve_wandb_project,
    wandb_entity_from_env,
)


WANDB_HOSTS = {"wandb.ai", "www.wandb.ai"}
CATALOG_PAGE_SIZE = 50
CATALOG_INDEX_SCHEMA_VERSION = 1
CATALOG_CACHE_SCHEMA_VERSION = 3
CATALOG_INDEX_FILENAME = "_catalog.yaml"
EVALUATION_CACHE_SECONDS = 10.0
WANDB_CATALOG_PAGE_SIZE = 200
RUN_CATALOG_CACHE_SECONDS = 60.0
GOAL_VARIANT_CACHE_SECONDS = 60.0
LIVE_TRAINING_METRICS = (
    (
        RankCriterion(
            direction="max",
            metric=TRAIN_OUTCOME_SUCCESS_ACROSS_STARTS_WINDOW_100_RATE_MIN,
        ),
        (TRAIN_OUTCOME_SUCCESS_ACROSS_STARTS_WINDOW_100_RATE_MIN,),
    ),
    (
        RankCriterion(
            direction="max",
            metric=TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_ROLLING_UP_TO_100_MEAN,
        ),
        (
            TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_ROLLING_UP_TO_100_MEAN,
            TRAIN_EPISODE_RETURN_SHAPED_ACROSS_ORIGINS_ROLLING_UP_TO_100_MEAN,
        ),
    ),
    (
        RankCriterion(
            direction="min",
            metric=TRAIN_GLOBAL_STEP,
        ),
        (TRAIN_GLOBAL_STEP,),
    ),
)


@dataclass(frozen=True)
class WandbLocation:
    entity: str
    project: str
    run_id: str | None = None


@dataclass(frozen=True)
class CatalogPage:
    items: tuple[dict[str, Any], ...]
    next_cursor: str | None
    metric_columns: tuple[dict[str, str], ...] = ()
    fallback_metric_columns: tuple[dict[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "items": list(self.items),
            "next_cursor": self.next_cursor,
        }
        if self.metric_columns:
            payload["metric_columns"] = list(self.metric_columns)
        if self.fallback_metric_columns:
            payload["fallback_metric_columns"] = list(self.fallback_metric_columns)
        return payload


@dataclass(frozen=True)
class ProjectSummary:
    entity: str
    name: str
    goal_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


EnvironmentSummary = ProjectSummary


@dataclass(frozen=True)
class GoalSummary:
    entity: str
    project: str
    goal_id: str
    goal_slug: str
    title: str
    recipe_count: int
    goal_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GoalVariantSummary:
    entity: str
    project: str
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
    exact_resolution_run_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunSummary:
    entity: str
    project: str
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
    project: str
    goal_id: str
    goal_slug: str
    title: str
    recipe_count: int
    goal_path: str
    rank: tuple[RankCriterion, ...] | None


@dataclass(frozen=True)
class _RepositoryNamespace:
    directory: str
    project: str
    title_template: str


@dataclass(frozen=True)
class _CheckpointEvaluationData:
    evaluations: Mapping[int, dict[str, Any]]
    training_seed: int | None
    evaluation_seed: int | None
    training_metric_history: Mapping[
        str,
        Mapping[str, tuple[tuple[int, float], ...]],
    ]


@dataclass(frozen=True)
class _WandbCatalogRun:
    run_id: str
    name: str
    state: str
    config: Mapping[str, Any]
    summary: Mapping[str, Any]
    notes: str
    created_at: str
    updated_at: str
    url: str


def parse_wandb_location(value: object) -> WandbLocation | None:
    parsed = urlparse(str(value or "").strip())
    if parsed.scheme not in {"http", "https"} or parsed.netloc.casefold() not in WANDB_HOSTS:
        return None
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return None
    entity, project = parts[:2]
    if not entity or not project:
        return None
    run_id = None
    if len(parts) >= 4 and parts[2] == "runs":
        run_id = parts[3]
    return WandbLocation(entity=entity, project=project, run_id=run_id)


def checkpoint_manifest_url(model_url: object) -> str:
    value = str(model_url or "").strip()
    if not value.endswith("/model.zip"):
        raise ValueError("public checkpoint model URL is malformed")
    return f"{value.removesuffix('/model.zip')}/manifest.json"


def _cursor_offset(value: object) -> int:
    text = str(value or "").strip()
    if not text:
        return 0
    try:
        decoded = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4)).decode("ascii")
        offset = int(decoded)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("invalid catalog cursor") from exc
    if offset < 0:
        raise ValueError("invalid catalog cursor")
    return offset


def _cursor_for(offset: int) -> str:
    return base64.urlsafe_b64encode(str(int(offset)).encode("ascii")).decode("ascii").rstrip("=")


def _search_text(*values: object) -> str:
    return " ".join(str(value or "") for value in values).casefold()


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


def _safe_summary_float(value: object) -> float | None:
    numeric = _safe_float(value)
    if numeric is not None:
        return numeric
    getter = getattr(value, "get", None)
    if not callable(getter):
        return None
    reduced = tuple(
        numeric
        for reducer in ("last", "max", "min", "mean", "best")
        if (numeric := _safe_float(getter(reducer))) is not None
    )
    return reduced[0] if len(reduced) == 1 else None


def _wandb_json_mapping(value: object, *, label: str) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str | bytes | bytearray):
        raise ValueError(f"W&B {label} must be a JSON mapping")
    try:
        decoded = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"W&B {label} is not valid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError(f"W&B {label} must be a JSON mapping")
    return dict(decoded)


def _wandb_user_config(value: object) -> dict[str, Any]:
    raw = _wandb_json_mapping(value, label="run config")
    return {
        str(key): item.get("value") if isinstance(item, Mapping) and "value" in item else item
        for key, item in raw.items()
        if key not in {"_wandb", "wandb_version"}
    }


def _first_summary_float(summary: Any, metrics: Iterable[str]) -> float | None:
    for metric in metrics:
        value = _safe_summary_float(summary.get(metric))
        if value is not None:
            return value
    return None


def _wandb_early_stop_projection(
    *,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    stop_reason: str,
) -> dict[str, Any] | None:
    condition_id = str(summary.get(RUN_EARLY_STOP_CONDITION_SUMMARY) or "").strip()
    if not condition_id and stop_reason.startswith("early_stop_") and ":" in stop_reason:
        condition_id = stop_reason.split(":", 1)[1].strip()
    trigger = str(summary.get(RUN_EARLY_STOP_TRIGGER_SUMMARY) or "").strip()
    if not condition_id and not trigger:
        return None

    projection: dict[str, Any] = {}
    if condition_id:
        projection["condition_id"] = condition_id
    if trigger:
        projection["trigger"] = trigger

    configured = config.get("early_stop")
    conditions = configured.get("conditions") if isinstance(configured, Mapping) else None
    condition = conditions.get(condition_id) if isinstance(conditions, Mapping) else None
    if isinstance(condition, Mapping):
        projected_condition = dict(condition)
        projection["condition"] = projected_condition
        metric = str(projected_condition.get("metric") or "").strip()
        configured_trigger = str(projected_condition.get("trigger") or "").strip()
        if metric:
            projection["metric"] = metric
            value = _safe_summary_float(summary.get(metric))
            if value is not None:
                projection["value"] = value
        if configured_trigger and not trigger:
            projection["trigger"] = configured_trigger
    return projection


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


def checkpoint_training_metric_columns() -> tuple[dict[str, str], ...]:
    return tuple(
        {"metric": criterion.metric, "direction": criterion.direction}
        for criterion, _sources in LIVE_TRAINING_METRICS
        if criterion.metric != TRAIN_GLOBAL_STEP
    )


def _checkpoint_training_metric_history(
    run: Any,
) -> dict[str, dict[str, tuple[tuple[int, float], ...]]]:
    history: dict[str, dict[str, tuple[tuple[int, float], ...]]] = {}
    for criterion, sources in LIVE_TRAINING_METRICS:
        if criterion.metric == TRAIN_GLOBAL_STEP:
            continue
        source_history: dict[str, tuple[tuple[int, float], ...]] = {}
        for source in sources:
            samples: dict[int, float] = {}
            try:
                rows = run.scan_history(
                    keys=[TRAIN_GLOBAL_STEP, source],
                    page_size=10_000,
                )
                for raw in rows:
                    if not isinstance(raw, Mapping):
                        continue
                    step = _safe_int(raw.get(TRAIN_GLOBAL_STEP))
                    value = _safe_float(raw.get(source))
                    if step is not None and value is not None:
                        samples[step] = value
            except Exception:
                continue
            if samples:
                source_history[source] = tuple(sorted(samples.items()))
        if source_history:
            history[criterion.metric] = source_history
    return history


def _checkpoint_training_metrics(
    history: Mapping[str, Mapping[str, tuple[tuple[int, float], ...]]],
    *,
    checkpoint_step: int,
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {TRAIN_GLOBAL_STEP: float(checkpoint_step)}
    for criterion, sources in LIVE_TRAINING_METRICS:
        if criterion.metric == TRAIN_GLOBAL_STEP:
            continue
        value = None
        source_history = history.get(criterion.metric, {})
        for source in sources:
            samples = source_history.get(source, ())
            eligible = (sample for sample in samples if sample[0] <= checkpoint_step)
            latest = max(eligible, default=None, key=lambda sample: sample[0])
            if latest is not None:
                value = latest[1]
                break
        metrics[criterion.metric] = value
    return metrics


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


def _page_items(items: list[dict[str, Any]], cursor: str | None) -> CatalogPage:
    offset = _cursor_offset(cursor)
    selected = tuple(items[offset : offset + CATALOG_PAGE_SIZE])
    next_offset = offset + CATALOG_PAGE_SIZE
    return CatalogPage(
        items=selected,
        next_cursor=_cursor_for(next_offset) if next_offset < len(items) else None,
    )


class PlayCatalog:
    """Repository catalog, W&B run metadata, and public-checkpoint discovery."""

    def __init__(
        self,
        *,
        public_models_base_url: str = DEFAULT_PUBLIC_MODELS_BASE_URL,
        repo_root: Path | str | None = None,
        cache_path: Path | str | None = None,
        control_bucket: R2Bucket | Any | None = None,
    ) -> None:
        self.public_models_base_url = str(public_models_base_url).rstrip("/")
        self.repo_root = (
            Path(repo_root).resolve()
            if repo_root is not None
            else Path(__file__).resolve().parents[2]
        )
        self.goals_root = self.repo_root / "experiments" / "goals"
        self.cache_path = (
            Path(cache_path).expanduser().resolve() if cache_path is not None else None
        )
        self.control_bucket = control_bucket
        if (
            self.control_bucket is None
            and str(os.environ.get("GRADLAB_CONTROL_R2_URI") or "").strip()
        ):
            try:
                self.control_bucket = R2Bucket(BucketConfig.from_env("GRADLAB_CONTROL_R2"))
            except ValueError:
                self.control_bucket = None
        self._api: Any | None = None
        self._lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._run_catalog_cache: dict[
            str,
            tuple[float, tuple[dict[str, Any], ...]],
        ] = {}
        self._run_catalog_refreshing: set[str] = set()
        self._goal_variant_cache: dict[
            str,
            tuple[float, tuple[dict[str, Any], ...]],
        ] = {}
        self._goal_variant_refreshing: set[str] = set()
        self._repository_project_cache: dict[
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
        self._default_entity: str | None = None
        self._evaluation_cache: dict[
            tuple[str, str, str],
            tuple[float, _CheckpointEvaluationData],
        ] = {}

    @staticmethod
    def default_cache_path(repo_root: Path | str) -> Path:
        resolved = Path(repo_root).resolve()
        identity = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16]
        cache_root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache").expanduser()
        return cache_root / "gradlab" / "play-catalog" / f"{identity}.json"

    def default_entity(self, explicit: object = None) -> str:
        text = str(explicit or "").strip()
        if text:
            return text
        with self._cache_lock:
            cached = self._default_entity
        if cached is not None:
            return cached
        load_wandb_env()
        entity = wandb_entity_from_env()
        with self._cache_lock:
            self._default_entity = entity
        return entity

    def _wandb_api(self):
        if self._api is None:
            load_wandb_env()
            import wandb

            self._api = wandb.Api(timeout=15)
        return self._api

    def _wandb_catalog_runs(
        self,
        *,
        entity: str,
        project: str,
        filters: Mapping[str, Any] | None,
    ) -> Iterator[_WandbCatalogRun]:
        api = self._wandb_api()
        client = getattr(api, "client", None)
        if client is None:
            # Test doubles and older W&B clients may not expose their GraphQL
            # client. Eager loading is important here: lazy Run objects fetch
            # config and summary individually when the catalog accesses them.
            for run in api.runs(
                f"{entity}/{project}",
                filters=dict(filters) if filters else None,
                order="-created_at",
                per_page=WANDB_CATALOG_PAGE_SIZE,
                lazy=False,
            ):
                yield _WandbCatalogRun(
                    run_id=str(getattr(run, "id", "") or ""),
                    name=str(getattr(run, "name", "") or ""),
                    state=str(getattr(run, "state", "") or ""),
                    config=dict(getattr(run, "config", {}) or {}),
                    summary=getattr(run, "summary", {}) or {},
                    notes=str(getattr(run, "notes", "") or ""),
                    created_at=str(getattr(run, "created_at", "") or ""),
                    updated_at=str(getattr(run, "updated_at", "") or ""),
                    url=str(getattr(run, "url", "") or ""),
                )
            return

        # The public Runs lazy fragment omits config and summary, causing one
        # full-data request per run when the catalog builds its cards. The full
        # fragment avoids that N+1 pattern but also downloads system metrics and
        # history metadata that the catalog never uses. Query only the fields
        # required by RunSummary while retaining server-side filtering.
        from wandb.apis.public.runs import gql

        query = gql(
            """
            query PlayCatalogRuns(
                $project: String!,
                $entity: String!,
                $cursor: String,
                $perPage: Int!,
                $order: String,
                $filters: JSONString
            ) {
                project(name: $project, entityName: $entity) {
                    runs(
                        filters: $filters,
                        after: $cursor,
                        first: $perPage,
                        order: $order
                    ) {
                        edges {
                            node {
                                name
                                displayName
                                state
                                config
                                createdAt
                                notes
                                summaryMetrics
                            }
                        }
                        pageInfo {
                            endCursor
                            hasNextPage
                        }
                    }
                }
            }
            """
        )
        cursor: str | None = None
        while True:
            response = client.execute(
                query,
                variable_values={
                    "project": project,
                    "entity": entity,
                    "cursor": cursor,
                    "perPage": WANDB_CATALOG_PAGE_SIZE,
                    "order": "-created_at",
                    "filters": json.dumps(dict(filters or {}), separators=(",", ":")),
                },
            )
            project_payload = response.get("project") if isinstance(response, Mapping) else None
            runs_payload = (
                project_payload.get("runs") if isinstance(project_payload, Mapping) else None
            )
            if not isinstance(runs_payload, Mapping):
                raise ValueError(f"could not find W&B project {entity}/{project}")
            edges = runs_payload.get("edges")
            if not isinstance(edges, list):
                raise ValueError("W&B run catalog response has no run edges")
            for edge in edges:
                node = edge.get("node") if isinstance(edge, Mapping) else None
                if not isinstance(node, Mapping):
                    raise ValueError("W&B run catalog response has an invalid run edge")
                run_id = str(node.get("name") or "")
                yield _WandbCatalogRun(
                    run_id=run_id,
                    name=str(node.get("displayName") or run_id),
                    state=str(node.get("state") or ""),
                    config=_wandb_user_config(node.get("config")),
                    summary=_wandb_json_mapping(
                        node.get("summaryMetrics"),
                        label=f"run {run_id} summary",
                    ),
                    notes=str(node.get("notes") or ""),
                    created_at=str(node.get("createdAt") or ""),
                    updated_at="",
                    url=(
                        f"{str(getattr(client, 'app_url', 'https://wandb.ai/')).rstrip('/')}"
                        f"/{entity}/{project}/runs/{run_id}"
                    ),
                )
            page_info = runs_payload.get("pageInfo")
            if not isinstance(page_info, Mapping) or not page_info.get("hasNextPage"):
                return
            next_cursor = str(page_info.get("endCursor") or "")
            if not next_cursor or next_cursor == cursor:
                raise ValueError("W&B run catalog pagination did not advance")
            cursor = next_cursor

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

    def _read_persistent_cache(self) -> dict[str, Any] | None:
        if self.cache_path is None or not self.cache_path.is_file():
            return None
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except OSError, json.JSONDecodeError:
            return None
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != CATALOG_CACHE_SCHEMA_VERSION
            or payload.get("repo_root") != str(self.repo_root)
        ):
            return None
        for section in ("projects", "run_catalogs", "goal_variants"):
            value = payload.setdefault(section, {})
            if not isinstance(value, dict):
                return None
        return payload

    def _update_persistent_cache(
        self,
        update: Callable[[dict[str, Any]], None],
    ) -> None:
        if self.cache_path is None:
            return
        with self._cache_lock:
            payload = self._read_persistent_cache() or {
                "schema_version": CATALOG_CACHE_SCHEMA_VERSION,
                "repo_root": str(self.repo_root),
                "projects": {},
                "run_catalogs": {},
                "goal_variants": {},
            }
            update(payload)
            temporary = self.cache_path.with_name(
                f".{self.cache_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
            )
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary.write_text(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")),
                    encoding="utf-8",
                )
                temporary.chmod(0o600)
                temporary.replace(self.cache_path)
            except OSError:
                temporary.unlink(missing_ok=True)

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
            extra = set(raw_metadata) - {"project", "title_template"}
            if extra:
                raise ValueError(
                    f"repository goal catalog namespace {directory!r} has unknown keys: "
                    + ", ".join(sorted(str(key) for key in extra))
                )
            project = str(raw_metadata.get("project") or "").strip()
            title_template = str(raw_metadata.get("title_template") or "").strip()
            if not project:
                raise ValueError(f"repository goal catalog namespace {directory!r} has no project")
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
                    project=project,
                    title_template=title_template,
                )
            )

        result = tuple(sorted(namespaces, key=lambda item: item.directory))
        with self._cache_lock:
            self._namespace_cache = (fingerprint, result)
        return result

    def _project_namespaces(
        self,
        project: str,
        namespaces: tuple[_RepositoryNamespace, ...],
    ) -> tuple[_RepositoryNamespace, ...]:
        return tuple(namespace for namespace in namespaces if namespace.project == project)

    def _indexed_project_fingerprint(
        self,
        namespaces: tuple[_RepositoryNamespace, ...],
    ) -> tuple[tuple[str, int, int], ...]:
        paths: list[Path] = [self.goals_root / CATALOG_INDEX_FILENAME]
        for namespace in namespaces:
            namespace_root = self.goals_root / namespace.directory
            paths.extend(namespace_root.rglob("*.yaml"))
            paths.extend(namespace_root.rglob("*.yml"))
        return self._catalog_fingerprint(paths)

    def _read_persistent_project(
        self,
        *,
        project: str,
        fingerprint: tuple[tuple[str, int, int], ...],
    ) -> tuple[_RepositoryGoal, ...] | None:
        try:
            payload = self._read_persistent_cache()
            if payload is None:
                return None
            projects = payload.get("projects")
            entry = projects.get(project) if isinstance(projects, Mapping) else None
            if not isinstance(entry, Mapping):
                return None
            cached_fingerprint = tuple(
                (str(item[0]), int(item[1]), int(item[2]))
                for item in entry.get("fingerprint", ())
                if isinstance(item, list | tuple) and len(item) == 3
            )
            if cached_fingerprint != fingerprint:
                return None
            raw_goals = entry.get("goals")
            if not isinstance(raw_goals, list):
                return None
            goals: list[_RepositoryGoal] = []
            for raw_goal in raw_goals:
                if not isinstance(raw_goal, Mapping):
                    return None
                goal = _RepositoryGoal(
                    project=str(raw_goal.get("project") or ""),
                    goal_id=str(raw_goal.get("goal_id") or ""),
                    goal_slug=str(raw_goal.get("goal_slug") or ""),
                    title=str(raw_goal.get("title") or ""),
                    recipe_count=int(raw_goal.get("recipe_count") or 0),
                    goal_path=str(raw_goal.get("goal_path") or ""),
                    rank=None,
                )
                if (
                    goal.project != project
                    or not goal.goal_id
                    or not goal.goal_slug
                    or not goal.title
                    or not goal.goal_path
                ):
                    return None
                goals.append(goal)
            return tuple(goals)
        except TypeError, ValueError:
            return None

    def _write_persistent_project(
        self,
        *,
        project: str,
        fingerprint: tuple[tuple[str, int, int], ...],
        goals: tuple[_RepositoryGoal, ...],
    ) -> None:
        def update(payload: dict[str, Any]) -> None:
            payload["projects"][project] = {
                "fingerprint": [list(item) for item in fingerprint],
                "goals": [
                    {
                        "project": goal.project,
                        "goal_id": goal.goal_id,
                        "goal_slug": goal.goal_slug,
                        "title": goal.title,
                        "recipe_count": goal.recipe_count,
                        "goal_path": goal.goal_path,
                    }
                    for goal in goals
                ],
            }

        self._update_persistent_cache(update)

    def _run_catalog_cache_key(
        self,
        *,
        entity: str,
        project: str,
        goal_slug: str,
        goal_variant_id: str = "",
        metric_specs: tuple[tuple[RankCriterion, tuple[str, ...]], ...],
        fallback_metric_specs: tuple[tuple[RankCriterion, tuple[str, ...]], ...],
    ) -> str:
        identity = {
            "entity": entity,
            "project": project,
            "goal_slug": goal_slug,
            "goal_variant_id": goal_variant_id,
            "metrics": [
                {
                    "metric": criterion.metric,
                    "direction": criterion.direction,
                    "sources": list(sources),
                }
                for criterion, sources in (*metric_specs, *fallback_metric_specs)
            ],
        }
        return hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _read_persistent_run_catalog(
        self,
        *,
        cache_key: str,
        entity: str,
        project: str,
    ) -> tuple[float, tuple[dict[str, Any], ...]] | None:
        try:
            payload = self._read_persistent_cache()
            if payload is None:
                return None
            catalogs = payload.get("run_catalogs")
            entry = catalogs.get(cache_key) if isinstance(catalogs, Mapping) else None
            if not isinstance(entry, Mapping):
                return None
            generated_at = _safe_float(entry.get("generated_at"))
            raw_items = entry.get("items")
            if generated_at is None or not isinstance(raw_items, list):
                return None
            items: list[dict[str, Any]] = []
            for raw_item in raw_items:
                if not isinstance(raw_item, Mapping):
                    return None
                run_id = str(raw_item.get("run_id") or "")
                metrics = raw_item.get("metrics")
                if (
                    RUN_ID_PATTERN.fullmatch(run_id) is None
                    or raw_item.get("entity") != entity
                    or raw_item.get("project") != project
                    or not isinstance(metrics, Mapping)
                ):
                    return None
                items.append(
                    RunSummary(
                        entity=entity,
                        project=project,
                        run_id=run_id,
                        name=str(raw_item.get("name") or run_id),
                        state=str(raw_item.get("state") or ""),
                        stop_reason=str(raw_item.get("stop_reason") or ""),
                        final_step=_safe_int(raw_item.get("final_step")),
                        early_stop=(
                            dict(raw_item["early_stop"])
                            if isinstance(raw_item.get("early_stop"), Mapping)
                            else None
                        ),
                        goal=str(raw_item.get("goal") or ""),
                        recipe=str(raw_item.get("recipe") or ""),
                        recipe_sha256=str(raw_item.get("recipe_sha256") or ""),
                        recipe_overrides=tuple(
                            str(value)
                            for value in raw_item.get("recipe_overrides", ())
                            if str(value).strip()
                        ),
                        recipe_variant_id=str(raw_item.get("recipe_variant_id") or ""),
                        goal_contract_sha256=str(raw_item.get("goal_contract_sha256") or ""),
                        effective_goal_contract_sha256=str(
                            raw_item.get("effective_goal_contract_sha256") or ""
                        ),
                        goal_variant_id=str(raw_item.get("goal_variant_id") or ""),
                        goal_variant_label=str(raw_item.get("goal_variant_label") or ""),
                        description=str(raw_item.get("description") or ""),
                        seed=_safe_int(raw_item.get("seed")),
                        created_at=str(raw_item.get("created_at") or ""),
                        updated_at=str(raw_item.get("updated_at") or ""),
                        url=str(raw_item.get("url") or ""),
                        metrics=dict(metrics),
                    ).to_dict()
                )
            return generated_at, tuple(items)
        except TypeError, ValueError:
            return None

    def _write_persistent_run_catalog(
        self,
        *,
        cache_key: str,
        generated_at: float,
        items: tuple[dict[str, Any], ...],
    ) -> None:
        def update(payload: dict[str, Any]) -> None:
            payload["run_catalogs"][cache_key] = {
                "generated_at": generated_at,
                "items": list(items),
            }

        self._update_persistent_cache(update)

    def _cached_run_catalog(
        self,
        *,
        cache_key: str,
        entity: str,
        project: str,
    ) -> tuple[float, tuple[dict[str, Any], ...]] | None:
        with self._cache_lock:
            cached = self._run_catalog_cache.get(cache_key)
        if cached is not None:
            return cached
        persisted = self._read_persistent_run_catalog(
            cache_key=cache_key,
            entity=entity,
            project=project,
        )
        if persisted is not None:
            with self._cache_lock:
                self._run_catalog_cache[cache_key] = persisted
        return persisted

    def _store_run_catalog(
        self,
        *,
        cache_key: str,
        items: tuple[dict[str, Any], ...],
    ) -> tuple[float, tuple[dict[str, Any], ...]]:
        cached = (time.time(), items)
        with self._cache_lock:
            self._run_catalog_cache[cache_key] = cached
        self._write_persistent_run_catalog(
            cache_key=cache_key,
            generated_at=cached[0],
            items=items,
        )
        return cached

    def _compose_repository_goal(self, path: Path) -> _RepositoryGoal:
        document = load_goal_contract(path, self.repo_root, validate=False)
        train = document.get("train")
        if not isinstance(train, Mapping):
            raise ValueError(f"repository goal has no train contract: {path}")
        environment = train.get("environment")
        if not isinstance(environment, Mapping):
            raise ValueError(f"repository goal has no training environment: {path}")
        env_config = environment.get("env_config", environment)
        if not isinstance(env_config, Mapping):
            raise ValueError(f"repository goal has invalid environment config: {path}")
        project = resolve_wandb_project(
            None,
            str(env_config.get("game") or ""),
            env_provider=environment.get("env_provider") or env_config.get("env_provider"),
        )
        goal_id = str(document.get("goal_id") or "").strip()
        if not project or not goal_id:
            raise ValueError(f"repository goal has no project or goal identity: {path}")
        objective = document.get("objective")
        return _RepositoryGoal(
            project=project,
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
        project: str,
        namespaces: tuple[_RepositoryNamespace, ...],
    ) -> tuple[_RepositoryGoal, ...]:
        project_namespaces = self._project_namespaces(project, namespaces)
        if not project_namespaces:
            return ()
        fingerprint = self._indexed_project_fingerprint(project_namespaces)
        with self._cache_lock:
            cached = self._repository_project_cache.get(project)
            if cached is not None and cached[0] == fingerprint:
                return cached[1]

        persisted = self._read_persistent_project(
            project=project,
            fingerprint=fingerprint,
        )
        if persisted is not None:
            with self._cache_lock:
                self._repository_project_cache[project] = (fingerprint, persisted)
            return persisted

        goals: list[_RepositoryGoal] = []
        for namespace in project_namespaces:
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
                        project=project,
                        goal_id=goal_id,
                        goal_slug=path.parent.relative_to(self.goals_root).as_posix(),
                        title=title or goal_id,
                        recipe_count=sum(1 for _ in path.parent.glob("recipes/*.yaml")),
                        goal_path=path.relative_to(self.repo_root).as_posix(),
                        rank=None,
                    )
                )
        identities = [(goal.project, goal.goal_id) for goal in goals]
        if len(identities) != len(set(identities)):
            raise ValueError("repository goals contain duplicate project/goal identities")
        result = tuple(sorted(goals, key=lambda goal: goal.goal_id))
        with self._cache_lock:
            self._repository_project_cache[project] = (fingerprint, result)
            for key in tuple(self._repository_details):
                if key[0] == project:
                    self._repository_details.pop(key, None)
        self._write_persistent_project(
            project=project,
            fingerprint=fingerprint,
            goals=result,
        )
        return result

    def _repository_goals(self, *, project: str | None = None) -> tuple[_RepositoryGoal, ...]:
        if not self.goals_root.is_dir():
            raise ValueError(f"repository goals directory does not exist: {self.goals_root}")
        namespaces = self._repository_namespaces()
        projects = sorted({namespace.project for namespace in namespaces})
        selected_projects = [project] if project is not None else projects
        return tuple(
            goal
            for selected_project in selected_projects
            for goal in self._indexed_repository_goals(
                project=selected_project,
                namespaces=namespaces,
            )
        )

    def _repository_projects(self) -> dict[str, int]:
        namespaces = self._repository_namespaces()
        counts: dict[str, int] = {}
        for namespace in namespaces:
            count = sum(1 for _ in (self.goals_root / namespace.directory).rglob("_goal.yaml"))
            counts[namespace.project] = counts.get(namespace.project, 0) + count
        return counts

    def _repository_goal(self, *, project: str, goal_id: str) -> _RepositoryGoal:
        for goal in self._repository_goals(project=project):
            if goal.project == project and goal.goal_id == goal_id:
                if goal.rank is not None:
                    return goal
                key = (project, goal_id)
                with self._cache_lock:
                    detailed = self._repository_details.get(key)
                if detailed is not None:
                    return detailed
                path = self.repo_root / goal.goal_path
                detailed = self._compose_repository_goal(path)
                if (
                    detailed.project != goal.project
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
        raise ValueError(f"repository has no goal {project}/{goal_id}")

    @staticmethod
    def _variant_cache_key(*, entity: str, project: str, goal_slug: str) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "entity": entity,
                    "project": project,
                    "goal_slug": goal_slug,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def _read_persistent_goal_variants(
        self,
        *,
        cache_key: str,
        entity: str,
        project: str,
        goal_slug: str,
    ) -> tuple[float, tuple[dict[str, Any], ...]] | None:
        payload = self._read_persistent_cache()
        entries = payload.get("goal_variants") if payload is not None else None
        entry = entries.get(cache_key) if isinstance(entries, Mapping) else None
        if not isinstance(entry, Mapping):
            return None
        generated_at = _safe_float(entry.get("generated_at"))
        raw_items = entry.get("items")
        if generated_at is None or not isinstance(raw_items, list):
            return None
        items: list[dict[str, Any]] = []
        for raw in raw_items:
            if (
                not isinstance(raw, Mapping)
                or raw.get("entity") != entity
                or raw.get("project") != project
                or raw.get("goal_slug") != goal_slug
                or not str(raw.get("variant_id") or "").strip()
            ):
                return None
            items.append(dict(raw))
        return generated_at, tuple(items)

    def _store_goal_variants(
        self,
        *,
        cache_key: str,
        items: tuple[dict[str, Any], ...],
    ) -> tuple[float, tuple[dict[str, Any], ...]]:
        cached = (time.time(), items)
        with self._cache_lock:
            self._goal_variant_cache[cache_key] = cached

        def update(payload: dict[str, Any]) -> None:
            payload["goal_variants"][cache_key] = {
                "generated_at": cached[0],
                "items": list(items),
            }

        self._update_persistent_cache(update)
        return cached

    def _cached_goal_variants(
        self,
        *,
        cache_key: str,
        entity: str,
        project: str,
        goal_slug: str,
    ) -> tuple[float, tuple[dict[str, Any], ...]] | None:
        with self._cache_lock:
            cached = self._goal_variant_cache.get(cache_key)
        if cached is None:
            cached = self._read_persistent_goal_variants(
                cache_key=cache_key,
                entity=entity,
                project=project,
                goal_slug=goal_slug,
            )
            if cached is not None:
                with self._cache_lock:
                    self._goal_variant_cache[cache_key] = cached
        return cached

    def _current_goal_variant(
        self,
        repository_goal: _RepositoryGoal,
    ) -> dict[str, Any]:
        authored = load_goal_contract(
            self.repo_root / repository_goal.goal_path,
            self.repo_root,
            validate=False,
        )
        descriptor = build_goal_variant_descriptor(
            goal_slug=repository_goal.goal_slug,
            source_sha="",
            authored_goal=authored,
            effective_goal=goal_for_contract_validation(
                authored,
                label=f"repository goal {repository_goal.goal_slug}",
            ),
        )
        return descriptor

    def _variant_summary(
        self,
        *,
        descriptor: Mapping[str, Any],
        entity: str,
        project: str,
        repository_goal: _RepositoryGoal,
        current: Mapping[str, Any],
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
        return GoalVariantSummary(
            entity=entity,
            project=project,
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
            exact_resolution_run_id=str(exact_resolution_run_id or ""),
        ).to_dict()

    def _control_goal_variants(
        self,
        *,
        entity: str,
        project: str,
        repository_goal: _RepositoryGoal,
        current: Mapping[str, Any],
    ) -> tuple[dict[str, Any], ...] | None:
        if self.control_bucket is None:
            return None
        scope_key = goal_variant_scope_key(
            entity=entity,
            project=project,
            goal_slug=repository_goal.goal_slug,
        )
        document = self.control_bucket.get_json_optional(f"{scope_key}/index.json")
        if document is None:
            return ()
        if int(document.get("schema_version") or 0) != GOAL_VARIANT_INDEX_SCHEMA_VERSION:
            raise ValueError("unsupported goal variant index schema")
        if document.get("scope") != {
            "entity": entity,
            "project": project,
            "goal_slug": repository_goal.goal_slug,
        }:
            raise ValueError("goal variant index scope mismatch")
        raw_variants = document.get("variants")
        if not isinstance(raw_variants, list):
            raise ValueError("goal variant index variants must be a list")
        items = []
        for raw in raw_variants:
            if not isinstance(raw, Mapping):
                raise ValueError("goal variant index contains an invalid entry")
            descriptor = {
                key: value
                for key, value in raw.items()
                if key
                not in {
                    "descriptor_key",
                    "first_run_id",
                    "exact_resolution_run_id",
                }
            }
            items.append(
                self._variant_summary(
                    descriptor=descriptor,
                    entity=entity,
                    project=project,
                    repository_goal=repository_goal,
                    current=current,
                    exact_resolution_run_id=str(raw.get("exact_resolution_run_id") or ""),
                )
            )
        return tuple(items)

    def _wandb_goal_variants(
        self,
        *,
        entity: str,
        project: str,
        repository_goal: _RepositoryGoal,
        current: Mapping[str, Any],
    ) -> tuple[dict[str, Any], ...]:
        variants: dict[str, dict[str, Any]] = {}
        found_unknown = False
        for run in self._wandb_catalog_runs(
            entity=entity,
            project=project,
            filters={"config.goal_slug": repository_goal.goal_slug},
        ):
            if RUN_ID_PATTERN.fullmatch(run.run_id) is None:
                continue
            config = dict(run.config)
            if str(config.get("goal_slug") or "") != repository_goal.goal_slug:
                continue
            authored_hash = str(config.get("goal_contract_sha256") or "").lower()
            effective_hash = str(config.get("effective_goal_contract_sha256") or "").lower()
            if (
                re.fullmatch(r"[0-9a-f]{64}", authored_hash) is None
                or re.fullmatch(r"[0-9a-f]{64}", effective_hash) is None
            ):
                found_unknown = True
                continue
            identifier = compute_goal_variant_id(
                goal_slug=repository_goal.goal_slug,
                goal_contract_sha256_value=authored_hash,
                effective_goal_contract_sha256=effective_hash,
            )
            configured = str(config.get("goal_variant_id") or "").strip()
            if configured and configured != identifier:
                continue
            diff: list[Mapping[str, Any]] = []
            raw_diff = config.get("goal_variant_diff_json")
            if isinstance(raw_diff, str) and raw_diff:
                try:
                    parsed = json.loads(raw_diff)
                    if isinstance(parsed, list):
                        diff = [dict(item) for item in parsed if isinstance(item, Mapping)]
                except json.JSONDecodeError:
                    pass
            authored_current = authored_hash == current["goal_contract_sha256"]
            effective_current = effective_hash == current["effective_goal_contract_sha256"]
            status = (
                "current"
                if authored_current and effective_current
                else "current changed"
                if authored_current
                else "historical"
            )
            variants[identifier] = GoalVariantSummary(
                entity=entity,
                project=project,
                goal_id=repository_goal.goal_id,
                goal_slug=repository_goal.goal_slug,
                variant_id=identifier,
                label=str(
                    config.get("goal_variant_label")
                    or f"{repository_goal.title} · historical contract {effective_hash[:12]}"
                ),
                goal_contract_sha256=authored_hash,
                effective_goal_contract_sha256=effective_hash,
                source_sha=str(config.get("source_sha") or ""),
                source_relation=str(config.get("goal_variant_source_relation") or "changed"),
                status=status,
                diff=tuple(diff),
                diff_truncated=False,
            ).to_dict()
        if found_unknown:
            identifier = unknown_goal_variant_id(goal_slug=repository_goal.goal_slug)
            variants[identifier] = GoalVariantSummary(
                entity=entity,
                project=project,
                goal_id=repository_goal.goal_id,
                goal_slug=repository_goal.goal_slug,
                variant_id=identifier,
                label=f"{repository_goal.title} · historical contract unknown",
                goal_contract_sha256="",
                effective_goal_contract_sha256="",
                source_sha="",
                source_relation="unknown",
                status="unknown",
                diff=(),
                diff_truncated=False,
            ).to_dict()
        return tuple(variants.values())

    def _load_goal_variants(
        self,
        *,
        entity: str,
        project: str,
        repository_goal: _RepositoryGoal,
    ) -> tuple[dict[str, Any], ...]:
        current = self._current_goal_variant(repository_goal)
        try:
            from_control = self._control_goal_variants(
                entity=entity,
                project=project,
                repository_goal=repository_goal,
                current=current,
            )
        except ValueError:
            raise
        except Exception:
            from_control = None
        items = (
            from_control
            if from_control is not None
            else self._wandb_goal_variants(
                entity=entity,
                project=project,
                repository_goal=repository_goal,
                current=current,
            )
        )
        return tuple(
            sorted(
                items,
                key=lambda item: (
                    item.get("status") not in {"current", "current changed"},
                    str(item.get("label") or "").casefold(),
                    str(item.get("variant_id") or ""),
                ),
            )
        )

    def _refresh_goal_variants(
        self,
        *,
        cache_key: str,
        entity: str,
        project: str,
        repository_goal: _RepositoryGoal,
    ) -> None:
        try:
            items = self._load_goal_variants(
                entity=entity,
                project=project,
                repository_goal=repository_goal,
            )
            self._store_goal_variants(cache_key=cache_key, items=items)
        except Exception:
            pass
        finally:
            with self._cache_lock:
                self._goal_variant_refreshing.discard(cache_key)

    def _schedule_goal_variant_refresh(
        self,
        *,
        cache_key: str,
        entity: str,
        project: str,
        repository_goal: _RepositoryGoal,
    ) -> None:
        with self._cache_lock:
            if cache_key in self._goal_variant_refreshing:
                return
            self._goal_variant_refreshing.add(cache_key)
        threading.Thread(
            target=self._refresh_goal_variants,
            kwargs={
                "cache_key": cache_key,
                "entity": entity,
                "project": project,
                "repository_goal": repository_goal,
            },
            name=f"gradlab-goal-variants-{cache_key[:12]}",
            daemon=True,
        ).start()

    def projects(
        self,
        *,
        entity: str,
        query: str = "",
        cursor: str | None = None,
    ) -> CatalogPage:
        normalized = str(query or "").strip().casefold()
        goal_counts = self._repository_projects()
        items = [
            ProjectSummary(
                entity=entity,
                name=project,
                goal_count=goal_count,
            ).to_dict()
            for project, goal_count in sorted(goal_counts.items())
            if not normalized or normalized in _search_text(entity, project)
        ]
        return _page_items(items, cursor)

    def environments(
        self,
        *,
        entity: str,
        query: str = "",
        cursor: str | None = None,
    ) -> CatalogPage:
        return self.projects(entity=entity, query=query, cursor=cursor)

    def initial_projects(self, explicit_entity: object = None) -> dict[str, Any]:
        entity = self.default_entity(explicit_entity)
        page = self.projects(entity=entity)
        return {"entity": entity, **page.to_dict()}

    def initial_environments(self, explicit_entity: object = None) -> dict[str, Any]:
        entity = self.default_entity(explicit_entity)
        page = self.environments(entity=entity)
        return {"entity": entity, **page.to_dict()}

    def _load_run_catalog(
        self,
        *,
        entity: str,
        project: str,
        selected_goal_slug: str,
        selected_goal_variant_id: str,
        metric_specs: tuple[tuple[RankCriterion, tuple[str, ...]], ...],
        fallback_metric_specs: tuple[tuple[RankCriterion, tuple[str, ...]], ...],
    ) -> tuple[dict[str, Any], ...]:
        control_summaries = self._control_run_catalog(
            entity=entity,
            project=project,
            selected_goal_slug=selected_goal_slug,
            selected_goal_variant_id=selected_goal_variant_id,
            metric_specs=metric_specs,
            fallback_metric_specs=fallback_metric_specs,
        )
        if control_summaries is not None:
            return control_summaries
        filters = {"config.goal_slug": selected_goal_slug} if selected_goal_slug else {}
        api_runs = self._wandb_catalog_runs(
            entity=entity,
            project=project,
            filters=filters,
        )
        summaries: list[dict[str, Any]] = []
        for run in api_runs:
            run_id = run.run_id
            if RUN_ID_PATTERN.fullmatch(run_id) is None:
                continue
            config = dict(run.config)
            goal_slug = str(config.get("goal_slug") or "")
            if selected_goal_slug and goal_slug != selected_goal_slug:
                continue
            authored_goal_hash = str(config.get("goal_contract_sha256") or "").lower()
            effective_goal_hash = str(config.get("effective_goal_contract_sha256") or "").lower()
            run_goal_variant_id = (
                compute_goal_variant_id(
                    goal_slug=goal_slug,
                    goal_contract_sha256_value=authored_goal_hash,
                    effective_goal_contract_sha256=effective_goal_hash,
                )
                if (
                    goal_slug
                    and re.fullmatch(r"[0-9a-f]{64}", authored_goal_hash)
                    and re.fullmatch(r"[0-9a-f]{64}", effective_goal_hash)
                )
                else unknown_goal_variant_id(goal_slug=goal_slug)
                if goal_slug
                else ""
            )
            configured_goal_variant_id = str(config.get("goal_variant_id") or "").strip()
            if configured_goal_variant_id and configured_goal_variant_id != run_goal_variant_id:
                continue
            if selected_goal_variant_id and run_goal_variant_id != selected_goal_variant_id:
                continue
            run_metrics = run.summary
            stop_reason = str(run_metrics.get(RUN_STOP_REASON_SUMMARY) or "")
            early_stop = _wandb_early_stop_projection(
                config=config,
                summary=run_metrics,
                stop_reason=stop_reason,
            )
            overrides = normalize_recipe_overrides(config.get("recipe_overrides"))
            configured_variant_id = str(config.get("recipe_variant_id") or "").strip()
            variant_id = configured_variant_id or (
                recipe_variant_id(
                    recipe_slug=config.get("recipe_slug"),
                    source_sha=config.get("source_sha"),
                    recipe_overrides=overrides,
                )
                if overrides
                else ""
            )
            summaries.append(
                RunSummary(
                    entity=entity,
                    project=project,
                    run_id=run_id,
                    name=run.name or run_id,
                    state=str(run_metrics.get(RUN_TERMINAL_STATE_SUMMARY) or run.state),
                    stop_reason=stop_reason,
                    final_step=_safe_int(run_metrics.get(RUN_FINAL_STEP_SUMMARY)),
                    early_stop=early_stop,
                    goal=goal_slug,
                    recipe=str(config.get("recipe_slug") or ""),
                    recipe_sha256=str(config.get("recipe_sha256") or ""),
                    recipe_overrides=overrides,
                    recipe_variant_id=variant_id,
                    goal_contract_sha256=authored_goal_hash,
                    effective_goal_contract_sha256=effective_goal_hash,
                    goal_variant_id=run_goal_variant_id,
                    goal_variant_label=str(config.get("goal_variant_label") or ""),
                    description=str(run.notes or config.get("run_description") or "").strip(),
                    seed=_safe_int(config.get("seed")),
                    created_at=run.created_at,
                    updated_at=run.updated_at,
                    url=run.url,
                    metrics={
                        criterion.metric: _first_summary_float(run_metrics, sources)
                        for criterion, sources in (*metric_specs, *fallback_metric_specs)
                    },
                ).to_dict()
            )
        _rank_run_summaries(
            summaries,
            primary=metric_specs,
            fallback=fallback_metric_specs,
        )
        return tuple(summaries)

    def _control_run_catalog(
        self,
        *,
        entity: str,
        project: str,
        selected_goal_slug: str,
        selected_goal_variant_id: str,
        metric_specs: tuple[tuple[RankCriterion, tuple[str, ...]], ...],
        fallback_metric_specs: tuple[tuple[RankCriterion, tuple[str, ...]], ...],
    ) -> tuple[dict[str, Any], ...] | None:
        if self.control_bucket is None or not selected_goal_slug:
            return None
        scope_key = goal_variant_scope_key(
            entity=entity,
            project=project,
            goal_slug=selected_goal_slug,
        )
        variant_index = self.control_bucket.get_json_optional(f"{scope_key}/index.json")
        if variant_index is None:
            return ()
        if int(variant_index.get("schema_version") or 0) != GOAL_VARIANT_INDEX_SCHEMA_VERSION:
            raise ValueError("unsupported goal variant index schema")
        if variant_index.get("scope") != {
            "entity": entity,
            "project": project,
            "goal_slug": selected_goal_slug,
        }:
            raise ValueError("goal variant index scope mismatch")
        raw_variants = variant_index.get("variants")
        if not isinstance(raw_variants, list):
            raise ValueError("goal variant index variants must be a list")
        variant_ids = [
            str(item.get("variant_id") or "")
            for item in raw_variants
            if isinstance(item, Mapping)
            and (not selected_goal_variant_id or item.get("variant_id") == selected_goal_variant_id)
        ]
        if any(not value for value in variant_ids):
            raise ValueError("goal variant index contains an invalid entry")

        summaries: list[dict[str, Any]] = []
        for variant_id in variant_ids:
            document = self.control_bucket.get_json_optional(f"{scope_key}/runs/{variant_id}.json")
            if document is None:
                continue
            if int(document.get("schema_version") or 0) != GOAL_VARIANT_RUN_INDEX_SCHEMA_VERSION:
                raise ValueError("unsupported goal variant run index schema")
            if document.get("scope") != {
                "entity": entity,
                "project": project,
                "goal_slug": selected_goal_slug,
                "variant_id": variant_id,
            }:
                raise ValueError("goal variant run index scope mismatch")
            raw_runs = document.get("runs")
            if not isinstance(raw_runs, list):
                raise ValueError("goal variant run index runs must be a list")
            for raw in raw_runs:
                if not isinstance(raw, Mapping):
                    raise ValueError("goal variant run index contains an invalid entry")
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
                    raise ValueError("goal variant run index contains an invalid run")
                summaries.append(
                    RunSummary(
                        entity=entity,
                        project=project,
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
                    ).to_dict()
                )
        _rank_run_summaries(
            summaries,
            primary=metric_specs,
            fallback=fallback_metric_specs,
        )
        return tuple(summaries)

    def _refresh_run_catalog(
        self,
        *,
        cache_key: str,
        entity: str,
        project: str,
        selected_goal_slug: str,
        selected_goal_variant_id: str,
        metric_specs: tuple[tuple[RankCriterion, tuple[str, ...]], ...],
        fallback_metric_specs: tuple[tuple[RankCriterion, tuple[str, ...]], ...],
    ) -> None:
        try:
            items = self._load_run_catalog(
                entity=entity,
                project=project,
                selected_goal_slug=selected_goal_slug,
                selected_goal_variant_id=selected_goal_variant_id,
                metric_specs=metric_specs,
                fallback_metric_specs=fallback_metric_specs,
            )
            self._store_run_catalog(cache_key=cache_key, items=items)
        except Exception:
            # A stale local catalog is preferable to turning a transient W&B
            # failure into a blocking playback-selection failure.
            pass
        finally:
            with self._cache_lock:
                self._run_catalog_refreshing.discard(cache_key)

    def _schedule_run_catalog_refresh(
        self,
        *,
        cache_key: str,
        entity: str,
        project: str,
        selected_goal_slug: str,
        selected_goal_variant_id: str,
        metric_specs: tuple[tuple[RankCriterion, tuple[str, ...]], ...],
        fallback_metric_specs: tuple[tuple[RankCriterion, tuple[str, ...]], ...],
    ) -> None:
        with self._cache_lock:
            if cache_key in self._run_catalog_refreshing:
                return
            self._run_catalog_refreshing.add(cache_key)
        threading.Thread(
            target=self._refresh_run_catalog,
            kwargs={
                "cache_key": cache_key,
                "entity": entity,
                "project": project,
                "selected_goal_slug": selected_goal_slug,
                "selected_goal_variant_id": selected_goal_variant_id,
                "metric_specs": metric_specs,
                "fallback_metric_specs": fallback_metric_specs,
            },
            name=f"gradlab-play-catalog-{cache_key[:12]}",
            daemon=True,
        ).start()

    def runs(
        self,
        *,
        entity: str,
        project: str,
        goal_id: str = "",
        goal_variant_id: str = "",
        query: str = "",
        cursor: str | None = None,
    ) -> CatalogPage:
        normalized = str(query or "").strip().casefold()
        selected_goal = str(goal_id or "").strip()
        repository_goal = (
            self._repository_goal(project=project, goal_id=selected_goal) if selected_goal else None
        )
        selected_goal_slug = repository_goal.goal_slug if repository_goal else ""
        selected_goal_variant = str(goal_variant_id or "").strip()
        if (
            selected_goal_variant
            and re.fullmatch(
                r"goal-variant-(?:[0-9a-f]{24}|unknown-[0-9a-f]{16})",
                selected_goal_variant,
            )
            is None
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
        cache_key = self._run_catalog_cache_key(
            entity=entity,
            project=project,
            goal_slug=selected_goal_slug,
            goal_variant_id=selected_goal_variant,
            metric_specs=metric_specs,
            fallback_metric_specs=fallback_metric_specs,
        )
        cached = self._cached_run_catalog(
            cache_key=cache_key,
            entity=entity,
            project=project,
        )
        if cached is None:
            summaries = self._load_run_catalog(
                entity=entity,
                project=project,
                selected_goal_slug=selected_goal_slug,
                selected_goal_variant_id=selected_goal_variant,
                metric_specs=metric_specs,
                fallback_metric_specs=fallback_metric_specs,
            )
            cached = self._store_run_catalog(cache_key=cache_key, items=summaries)
        elif time.time() - cached[0] >= RUN_CATALOG_CACHE_SECONDS:
            self._schedule_run_catalog_refresh(
                cache_key=cache_key,
                entity=entity,
                project=project,
                selected_goal_slug=selected_goal_slug,
                selected_goal_variant_id=selected_goal_variant,
                metric_specs=metric_specs,
                fallback_metric_specs=fallback_metric_specs,
            )
        filtered = [
            summary
            for summary in cached[1]
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
            )
        ]
        page = _page_items(filtered, cursor)
        return CatalogPage(
            items=page.items,
            next_cursor=page.next_cursor,
            metric_columns=metric_columns,
            fallback_metric_columns=fallback_metric_columns,
        )

    def goals(
        self,
        *,
        entity: str,
        project: str,
        query: str = "",
        cursor: str | None = None,
    ) -> CatalogPage:
        normalized = str(query or "").strip().casefold()
        items = [
            GoalSummary(
                entity=entity,
                project=project,
                goal_id=goal.goal_id,
                goal_slug=goal.goal_slug,
                title=goal.title,
                recipe_count=goal.recipe_count,
                goal_path=goal.goal_path,
            ).to_dict()
            for goal in self._repository_goals(project=project)
            if (
                not normalized
                or normalized
                in _search_text(
                    goal.goal_id,
                    goal.goal_slug,
                    goal.title,
                    goal.goal_path,
                )
            )
        ]
        return _page_items(items, cursor)

    def goal_variants(
        self,
        *,
        entity: str,
        project: str,
        goal_id: str,
        query: str = "",
        cursor: str | None = None,
    ) -> CatalogPage:
        repository_goal = self._repository_goal(project=project, goal_id=goal_id)
        cache_key = self._variant_cache_key(
            entity=entity,
            project=project,
            goal_slug=repository_goal.goal_slug,
        )
        cached = self._cached_goal_variants(
            cache_key=cache_key,
            entity=entity,
            project=project,
            goal_slug=repository_goal.goal_slug,
        )
        if cached is None:
            items = self._load_goal_variants(
                entity=entity,
                project=project,
                repository_goal=repository_goal,
            )
            cached = self._store_goal_variants(cache_key=cache_key, items=items)
        elif time.time() - cached[0] >= GOAL_VARIANT_CACHE_SECONDS:
            self._schedule_goal_variant_refresh(
                cache_key=cache_key,
                entity=entity,
                project=project,
                repository_goal=repository_goal,
            )
        normalized = str(query or "").strip().casefold()
        filtered = [
            item
            for item in cached[1]
            if not normalized
            or normalized
            in _search_text(
                item.get("label"),
                item.get("variant_id"),
                item.get("status"),
                item.get("source_relation"),
                item.get("diff"),
                item.get("goal_contract_sha256"),
                item.get("effective_goal_contract_sha256"),
            )
        ]
        return _page_items(filtered, cursor)

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
        entity: str,
        project: str,
        goal_id: str,
        query: str = "",
        cursor: str | None = None,
    ) -> CatalogPage:
        repository_goal = self._repository_goal(project=project, goal_id=goal_id)
        recipes_root = (self.repo_root / repository_goal.goal_path).parent / "recipes"
        normalized = str(query or "").strip().casefold()
        items: list[dict[str, Any]] = []
        for path in sorted(recipes_root.glob("*.yaml")):
            composed = load_recipe_source_document(path).document
            recipe_id = str(composed.get("recipe_id") or path.stem).strip()
            description = str(composed.get("description") or "").strip()
            item = {
                "entity": entity,
                "project": project,
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
        return _page_items(items, cursor)

    def inspect_goal(
        self,
        *,
        entity: str,
        project: str,
        goal_id: str,
    ) -> dict[str, Any]:
        repository_goal = self._repository_goal(project=project, goal_id=goal_id)
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
                "entity": entity,
                "project": project,
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
        entity: str,
        project: str,
        goal_id: str,
        recipe_id: str,
    ) -> dict[str, Any]:
        repository_goal = self._repository_goal(project=project, goal_id=goal_id)
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
            raise ValueError(f"repository has no recipe {project}/{goal_id}/{normalized_recipe_id}")
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
                "entity": entity,
                "project": project,
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
        resolution = validated.get("resolution")
        if isinstance(resolution, Mapping):
            goal_resolution = resolution.get("goal")
            recipe_resolution = resolution.get("recipe")
            if not isinstance(goal_resolution, Mapping) or not isinstance(
                recipe_resolution,
                Mapping,
            ):
                raise ValueError("recipe resolution proof is malformed")
            goal_base = dict(goal_resolution["base"])
            recipe_base = dict(recipe_resolution["base"])
            message = ""
        else:
            goal_base = None
            recipe_base = None
            message = (
                "This legacy recipe preserves the exact resolved contract but predates "
                "embedded base-contract proofs, so a historical diff is unavailable."
            )
        goal_variant = recipe.get("goal_variant")
        goal_variant_id = (
            str(goal_variant.get("variant_id") or "") if isinstance(goal_variant, Mapping) else ""
        )
        recipe_variant_id_value = (
            str(recipe_resolution.get("variant_id") or "")
            if isinstance(resolution, Mapping) and isinstance(recipe_resolution, Mapping)
            else recipe_variant_id(
                recipe_slug=recipe.get("recipe_id"),
                source_sha=str(validated.get("provenance", {}).get("source_commit") or ""),
                recipe_overrides=recipe.get("recipe_overrides"),
            )
        )
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
        except Exception:
            return None
        if int(index.get("schema_version") or 0) != 1 or index.get("run_id") != run_id:
            raise ValueError("public run index identity mismatch")
        manifests = []
        for raw in index.get("checkpoints") or ():
            if not isinstance(raw, Mapping):
                raise ValueError("public run index contains an invalid checkpoint")
            manifest = CheckpointManifest(**dict(raw))
            manifest.validate()
            manifests.append(manifest)
        if not manifests:
            return None
        manifest = max(manifests, key=lambda item: (item.step, item.checkpoint_id))
        document = validate_recipe_document(
            _public_json(manifest.recipe_document_url),
            source=manifest.recipe_document_url,
        )
        observed = canonical_json_sha256(document)
        if observed != manifest.recipe_document_sha256 or observed != manifest.recipe_sha256:
            raise ValueError("public recipe document hash mismatch")
        if manifest.run_id != run_id:
            raise ValueError("public checkpoint run identity mismatch")
        return document

    def inspect_run(
        self,
        *,
        entity: str,
        project: str,
        run_id: str,
    ) -> dict[str, Any]:
        if RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise ValueError("run id must match gradlab-<32 lowercase hex>")
        manifest_document = (
            self.control_bucket.get_json_optional(f"runs/{run_id}/manifest.json")
            if self.control_bucket is not None
            else None
        )
        document = None
        metadata: dict[str, Any] = {
            "entity": entity,
            "project": project,
            "run_id": run_id,
        }
        if manifest_document is not None:
            manifest = RunManifest(**dict(manifest_document))
            manifest.validate()
            if (
                manifest.run_id != run_id
                or manifest.wandb.get("entity") != entity
                or manifest.wandb.get("project") != project
            ):
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
        entity: str,
        project: str,
        goal_id: str,
        variant_id: str,
    ) -> dict[str, Any]:
        repository_goal = self._repository_goal(project=project, goal_id=goal_id)
        selected = None
        cursor = None
        while selected is None:
            page = self.goal_variants(
                entity=entity,
                project=project,
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
        exact_run_id = str(selected.get("exact_resolution_run_id") or "")
        if not exact_run_id:
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
            return self._inspection_envelope(
                source={"kind": "goal-variant-summary", "variant_id": variant_id},
                goal=summary,
            )
        envelope = self.inspect_run(
            entity=entity,
            project=project,
            run_id=exact_run_id,
        )
        goal = envelope["documents"].get("goal")
        if not isinstance(goal, Mapping) or goal.get("variant_id") != variant_id:
            raise ValueError("goal variant resolution run does not prove the selected variant")
        return {
            **envelope,
            "source": {
                "kind": "goal-variant",
                "variant_id": variant_id,
                "exact_resolution_run_id": exact_run_id,
            },
            "documents": {"goal": dict(goal)},
        }

    def run_goal(self, *, entity: str, project: str, run_id: str) -> str:
        goal_id, _variant_id = self.run_goal_variant(
            entity=entity,
            project=project,
            run_id=run_id,
        )
        return goal_id

    def run_goal_variant(
        self,
        *,
        entity: str,
        project: str,
        run_id: str,
    ) -> tuple[str, str]:
        if RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise ValueError("run id must match gradlab-<32 lowercase hex>")
        if self.control_bucket is not None:
            manifest = self.control_bucket.get_json_optional(f"runs/{run_id}/manifest.json")
            if manifest is not None:
                wandb = manifest.get("wandb")
                descriptor = manifest.get("goal_variant")
                if (
                    not isinstance(wandb, Mapping)
                    or str(wandb.get("entity") or "") != entity
                    or str(wandb.get("project") or "") != project
                ):
                    raise ValueError("run manifest W&B scope mismatch")
                if not isinstance(descriptor, Mapping):
                    raise ValueError("run manifest has no goal variant descriptor")
                validated = validate_goal_variant_descriptor(descriptor)
                goal_slug = str(validated["goal_slug"])
                for goal in self._repository_goals(project=project):
                    if goal.goal_slug == goal_slug:
                        return goal.goal_id, str(validated["variant_id"])
                raise ValueError(
                    f"run manifest goal is not declared in the repository: {goal_slug}"
                )
        run = self._wandb_api().run(f"{entity}/{project}/{run_id}")
        config = dict(getattr(run, "config", {}) or {})
        goal_slug = str(config.get("goal_slug") or "").strip()
        for goal in self._repository_goals(project=project):
            if goal.goal_slug == goal_slug:
                authored_hash = str(config.get("goal_contract_sha256") or "").lower()
                effective_hash = str(config.get("effective_goal_contract_sha256") or "").lower()
                variant_id = (
                    compute_goal_variant_id(
                        goal_slug=goal_slug,
                        goal_contract_sha256_value=authored_hash,
                        effective_goal_contract_sha256=effective_hash,
                    )
                    if (
                        re.fullmatch(r"[0-9a-f]{64}", authored_hash)
                        and re.fullmatch(r"[0-9a-f]{64}", effective_hash)
                    )
                    else unknown_goal_variant_id(goal_slug=goal_slug)
                )
                return goal.goal_id, variant_id
        if not goal_slug:
            raise ValueError("W&B run has no goal identity")
        raise ValueError(f"W&B run goal is not declared in the repository: {goal_slug}")

    def _checkpoint_evaluations(
        self,
        *,
        entity: str,
        project: str,
        run_id: str,
    ) -> _CheckpointEvaluationData:
        entity = str(entity or "").strip()
        project = str(project or "").strip()
        if not entity or not project:
            return _CheckpointEvaluationData({}, None, None, {})
        cache_key = (entity, project, run_id)
        now = time.monotonic()
        with self._lock:
            cached = self._evaluation_cache.get(cache_key)
            if cached is not None and now - cached[0] < EVALUATION_CACHE_SECONDS:
                return cached[1]
        run = None
        try:
            run = self._wandb_api().run(f"{entity}/{project}/{run_id}")
            config = dict(getattr(run, "config", {}) or {})
            metric_schema = evaluation_metric_schema(
                config.get("metrics_schema_version") or METRICS_SCHEMA_VERSION
            )
            training_seed = _safe_int(config.get("seed"))
            contract = config.get("checkpoint_eval_contract")
            if not isinstance(contract, Mapping):
                evaluations = {}
                evaluation_seed = None
            else:
                evaluation_seed = _safe_int(contract.get("seed"))
                rules = normalize_metric_threshold_rules(
                    contract.get("acceptance"),
                    label="checkpoint_eval_contract.acceptance",
                    metric_validator=lambda name: validate_evaluation_scientific_metric(
                        name,
                        schema_version=metric_schema.version,
                    ),
                )
                result_keys = {
                    metric_schema.checkpoint_step,
                    EVAL_ACCEPTANCE_PASS,
                    metric_schema.acceptance_episode_planned_count,
                    metric_schema.acceptance_episode_completed_count,
                }
                evaluations = {}
                for raw in run.scan_history(keys=sorted(result_keys), page_size=10_000):
                    if not isinstance(raw, Mapping):
                        continue
                    step = _safe_int(raw.get(metric_schema.checkpoint_step))
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
                            raw.get(metric_schema.acceptance_episode_planned_count)
                        ),
                        "episodes_completed": _safe_int(
                            raw.get(metric_schema.acceptance_episode_completed_count)
                        ),
                        "criteria": criteria,
                    }
                # Fail-fast rejections intentionally omit completed eval/full metrics.
                # W&B returns no rows when scan_history requests a key that is absent
                # from some history records, so fetch each optional criterion
                # independently and merge it into the authoritative verdict rows.
                for rule_index, rule in enumerate(rules):
                    metric = str(rule["metric"])
                    for raw in run.scan_history(
                        keys=[metric_schema.checkpoint_step, metric],
                        page_size=10_000,
                    ):
                        if not isinstance(raw, Mapping):
                            continue
                        step = _safe_int(raw.get(metric_schema.checkpoint_step))
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
        except Exception:
            # Public checkpoints remain playable when W&B history is unavailable.
            evaluations = {}
            training_seed = None
            evaluation_seed = None
        training_metric_history = (
            _checkpoint_training_metric_history(run) if run is not None else {}
        )
        data = _CheckpointEvaluationData(
            evaluations=evaluations,
            training_seed=training_seed,
            evaluation_seed=evaluation_seed,
            training_metric_history=training_metric_history,
        )
        with self._lock:
            self._evaluation_cache[cache_key] = (now, data)
        return data

    def checkpoints(
        self,
        *,
        run_id: str,
        query: str = "",
        entity: str = "",
        project: str = "",
        goal_variant_id: str = "",
    ) -> tuple[dict[str, Any], ...]:
        if RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise ValueError("run id must match gradlab-<32 lowercase hex>")
        url = f"{self.public_models_base_url}/runs/{run_id}/index.json"
        index = _public_json(url)
        if int(index.get("schema_version") or 0) != 1:
            raise ValueError("unsupported public run index schema")
        if str(index.get("run_id") or "") != run_id:
            raise ValueError("public run index identity mismatch")
        promotion = index.get("promotion")
        promoted_id = (
            str(promotion.get("checkpoint_id") or "") if isinstance(promotion, Mapping) else ""
        )
        normalized = str(query or "").strip().casefold()
        expected_effective_goal_hash = ""
        selected_variant = str(goal_variant_id or "").strip()
        if selected_variant:
            if not str(entity or "").strip() or not str(project or "").strip():
                raise ValueError("goal variant checkpoint validation requires entity and project")
            run = self._wandb_api().run(f"{entity}/{project}/{run_id}")
            config = dict(getattr(run, "config", {}) or {})
            goal_slug = str(config.get("goal_slug") or "").strip()
            authored_hash = str(config.get("goal_contract_sha256") or "").lower()
            expected_effective_goal_hash = str(
                config.get("effective_goal_contract_sha256") or ""
            ).lower()
            if (
                re.fullmatch(r"[0-9a-f]{64}", authored_hash) is None
                or re.fullmatch(r"[0-9a-f]{64}", expected_effective_goal_hash) is None
            ):
                observed_variant = unknown_goal_variant_id(goal_slug=goal_slug) if goal_slug else ""
            else:
                observed_variant = compute_goal_variant_id(
                    goal_slug=goal_slug,
                    goal_contract_sha256_value=authored_hash,
                    effective_goal_contract_sha256=expected_effective_goal_hash,
                )
            if observed_variant != selected_variant:
                raise ValueError("run does not belong to the selected goal variant")
        evaluation_data = self._checkpoint_evaluations(
            entity=entity,
            project=project,
            run_id=run_id,
        )
        rows: list[CheckpointSummary] = []
        for raw in index.get("checkpoints") or ():
            if not isinstance(raw, Mapping):
                raise ValueError("public run index contains an invalid checkpoint")
            manifest = CheckpointManifest(**dict(raw))
            manifest.validate()
            if manifest.run_id != run_id:
                raise ValueError("checkpoint does not belong to the selected run")
            if (
                expected_effective_goal_hash
                and manifest.goal_sha256 != expected_effective_goal_hash
            ):
                raise ValueError(
                    "checkpoint effective goal contract does not match its run variant"
                )
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
                metrics=_checkpoint_training_metrics(
                    evaluation_data.training_metric_history,
                    checkpoint_step=manifest.step,
                ),
                evaluation=evaluation,
            )
            if normalized and normalized not in _search_text(
                row.checkpoint_id,
                row.step,
                row.purpose,
                row.sha256,
                row.created_at,
                "promoted" if row.promoted else "",
                row.metrics,
                row.evaluation,
            ):
                continue
            rows.append(row)
        rows.sort(key=lambda row: (row.step, row.sha256), reverse=True)
        return tuple(row.to_dict() for row in rows)


def is_wandb_url(value: object) -> bool:
    return parse_wandb_location(value) is not None


def normalize_search_query(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:200]


__all__ = [
    "CATALOG_PAGE_SIZE",
    "CatalogPage",
    "CheckpointSummary",
    "checkpoint_training_metric_columns",
    "EnvironmentSummary",
    "GoalSummary",
    "GoalVariantSummary",
    "PlayCatalog",
    "ProjectSummary",
    "RunSummary",
    "WandbLocation",
    "checkpoint_manifest_url",
    "is_wandb_url",
    "normalize_search_query",
    "parse_wandb_location",
]
