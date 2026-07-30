from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
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
    goal_variant_scope_key,
    validate_goal_variant_descriptor,
)
from gradlab.json_utils import (
    canonical_json_sha256 as compact_json_sha256,
    canonical_json_text,
)
from gradlab.evaluation_projection import validate_evaluation_scientific_metric
from gradlab.metric_names import (
    EVAL_ACCEPTANCE_PASS,
    LEADER_CHECKPOINT_STEP,
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
from gradlab.recipe_documents import (
    compose_resolved_train_documents,
    load_goal_contract,
    load_recipe_source_document,
)
from gradlab.reward_programs import goal_for_contract_validation
from gradlab.run_contracts import CheckpointManifest, RUN_ID_PATTERN, RunManifest
from gradlab.run_authority import RunAuthority
from gradlab.wandb_utils import load_wandb_env


WANDB_HOSTS = {"wandb.ai", "www.wandb.ai"}
CATALOG_PAGE_SIZE = 50
CATALOG_INDEX_SCHEMA_VERSION = 2
CATALOG_CACHE_SCHEMA_VERSION = 4
CATALOG_INDEX_FILENAME = "_catalog.yaml"
EVALUATION_CACHE_SECONDS = 10.0
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
class EnvironmentSummary:
    name: str
    goal_count: int

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
    training_metric_history: Mapping[
        str,
        Mapping[str, tuple[tuple[int, float], ...]],
    ]
    evaluation_rank: tuple[RankCriterion, ...]


def parse_wandb_location(value: object) -> WandbRunLocation | None:
    parsed = urlparse(str(value or "").strip())
    if parsed.scheme not in {"http", "https"} or parsed.netloc.casefold() not in WANDB_HOSTS:
        return None
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if len(parts) != 4 or parts[2] != "runs":
        return None
    entity, project, _runs, run_id = parts
    if (
        not entity
        or not project
        or RUN_ID_PATTERN.fullmatch(run_id) is None
    ):
        return None
    return WandbRunLocation(entity=entity, project=project, run_id=run_id)


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


def _best_checkpoint_id(
    rows: Sequence[CheckpointSummary],
    rank: Sequence[RankCriterion],
    *,
    evaluation: bool = False,
) -> str:
    if not rank:
        return ""
    best_id = ""
    best_score: tuple[float, ...] | None = None
    for row in rows:
        if evaluation:
            result = row.evaluation
            metrics = result.get("metrics") if isinstance(result, Mapping) else None
        else:
            metrics = row.metrics
        if not isinstance(metrics, Mapping):
            continue
        score: list[float] = []
        for criterion in rank:
            value = _safe_float(metrics.get(criterion.metric))
            if value is None:
                break
            score.append(value if criterion.direction == "max" else -value)
        else:
            candidate = tuple(score)
            if best_score is None or candidate > best_score:
                best_id = row.checkpoint_id
                best_score = candidate
    return best_id


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
    """Repository catalog, lifecycle run metadata, and public-checkpoint discovery."""

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
        for section in ("environments", "run_catalogs", "goal_variants"):
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
                "environments": {},
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
                    canonical_json_text(payload, ensure_ascii=True),
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
            namespace
            for namespace in namespaces
            if namespace.environment_id == environment_id
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

    def _read_persistent_environment(
        self,
        *,
        environment_id: str,
        fingerprint: tuple[tuple[str, int, int], ...],
    ) -> tuple[_RepositoryGoal, ...] | None:
        try:
            payload = self._read_persistent_cache()
            if payload is None:
                return None
            environments = payload.get("environments")
            entry = (
                environments.get(environment_id)
                if isinstance(environments, Mapping)
                else None
            )
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
                    environment_id=str(raw_goal.get("environment_id") or ""),
                    goal_id=str(raw_goal.get("goal_id") or ""),
                    goal_slug=str(raw_goal.get("goal_slug") or ""),
                    title=str(raw_goal.get("title") or ""),
                    recipe_count=int(raw_goal.get("recipe_count") or 0),
                    goal_path=str(raw_goal.get("goal_path") or ""),
                    rank=None,
                )
                if (
                    goal.environment_id != environment_id
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

    def _write_persistent_environment(
        self,
        *,
        environment_id: str,
        fingerprint: tuple[tuple[str, int, int], ...],
        goals: tuple[_RepositoryGoal, ...],
    ) -> None:
        def update(payload: dict[str, Any]) -> None:
            payload["environments"][environment_id] = {
                "fingerprint": [list(item) for item in fingerprint],
                "goals": [
                    {
                        "environment_id": goal.environment_id,
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
        environment_id: str,
        goal_slug: str,
        goal_variant_id: str = "",
        metric_specs: tuple[tuple[RankCriterion, tuple[str, ...]], ...],
        fallback_metric_specs: tuple[tuple[RankCriterion, tuple[str, ...]], ...],
    ) -> str:
        identity = {
            "environment_id": environment_id,
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
        return compact_json_sha256(identity, ensure_ascii=True)

    def _read_persistent_run_catalog(
        self,
        *,
        cache_key: str,
        environment_id: str,
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
                    or raw_item.get("environment_id") != environment_id
                    or not isinstance(metrics, Mapping)
                ):
                    return None
                items.append(
                    RunSummary(
                        environment_id=environment_id,
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
        environment_id: str,
    ) -> tuple[float, tuple[dict[str, Any], ...]] | None:
        with self._cache_lock:
            cached = self._run_catalog_cache.get(cache_key)
        if cached is not None:
            return cached
        persisted = self._read_persistent_run_catalog(
            cache_key=cache_key,
            environment_id=environment_id,
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

        persisted = self._read_persistent_environment(
            environment_id=environment_id,
            fingerprint=fingerprint,
        )
        if persisted is not None:
            with self._cache_lock:
                self._repository_environment_cache[environment_id] = (
                    fingerprint,
                    persisted,
                )
            return persisted

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
        self._write_persistent_environment(
            environment_id=environment_id,
            fingerprint=fingerprint,
            goals=result,
        )
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
            counts[namespace.environment_id] = (
                counts.get(namespace.environment_id, 0) + count
            )
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

    @staticmethod
    def _variant_cache_key(*, environment_id: str, goal_slug: str) -> str:
        return compact_json_sha256(
            {
                "environment_id": environment_id,
                "goal_slug": goal_slug,
            },
            ensure_ascii=True,
        )

    def _read_persistent_goal_variants(
        self,
        *,
        cache_key: str,
        environment_id: str,
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
                or raw.get("environment_id") != environment_id
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
        environment_id: str,
        goal_slug: str,
    ) -> tuple[float, tuple[dict[str, Any], ...]] | None:
        with self._cache_lock:
            cached = self._goal_variant_cache.get(cache_key)
        if cached is None:
            cached = self._read_persistent_goal_variants(
                cache_key=cache_key,
                environment_id=environment_id,
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
            exact_resolution_run_id=str(exact_resolution_run_id or ""),
        ).to_dict()

    def _control_goal_variants(
        self,
        *,
        repository_goal: _RepositoryGoal,
        current: Mapping[str, Any],
    ) -> tuple[dict[str, Any], ...] | None:
        if self.control_bucket is None:
            return None
        scope_key = goal_variant_scope_key(goal_slug=repository_goal.goal_slug)
        document = self.control_bucket.get_json_optional(f"{scope_key}/index.json")
        if document is None:
            return ()
        if int(document.get("schema_version") or 0) != GOAL_VARIANT_INDEX_SCHEMA_VERSION:
            raise ValueError("unsupported goal variant index schema")
        if document.get("scope") != {"goal_slug": repository_goal.goal_slug}:
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
                    repository_goal=repository_goal,
                    current=current,
                    exact_resolution_run_id=str(raw.get("exact_resolution_run_id") or ""),
                )
            )
        return tuple(items)

    def _load_goal_variants(
        self,
        *,
        repository_goal: _RepositoryGoal,
    ) -> tuple[dict[str, Any], ...]:
        current = self._current_goal_variant(repository_goal)
        current_summary = self._variant_summary(
            descriptor=current,
            repository_goal=repository_goal,
            current=current,
        )
        history = self._control_goal_variants(
            repository_goal=repository_goal,
            current=current,
        )
        by_id = {
            str(item["variant_id"]): item
            for item in (history or ())
        }
        indexed_current = by_id.get(str(current_summary["variant_id"]))
        if indexed_current is not None:
            current_summary["exact_resolution_run_id"] = str(
                indexed_current.get("exact_resolution_run_id") or ""
            )
        by_id[str(current_summary["variant_id"])] = current_summary
        items = tuple(by_id.values())
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
        repository_goal: _RepositoryGoal,
    ) -> None:
        try:
            items = self._load_goal_variants(
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
                "repository_goal": repository_goal,
            },
            name=f"gradlab-goal-variants-{cache_key[:12]}",
            daemon=True,
        ).start()

    def environments(
        self,
        *,
        query: str = "",
        cursor: str | None = None,
    ) -> CatalogPage:
        normalized = str(query or "").strip().casefold()
        goal_counts = self._repository_environments()
        items = [
            EnvironmentSummary(
                name=environment_id,
                goal_count=goal_count,
            ).to_dict()
            for environment_id, goal_count in sorted(goal_counts.items())
            if not normalized or normalized in _search_text(environment_id)
        ]
        return _page_items(items, cursor)

    def initial_environments(self) -> dict[str, Any]:
        return self.environments().to_dict()

    def _load_run_catalog(
        self,
        *,
        environment_id: str,
        selected_goal_slug: str,
        selected_goal_variant_id: str,
        metric_specs: tuple[tuple[RankCriterion, tuple[str, ...]], ...],
        fallback_metric_specs: tuple[tuple[RankCriterion, tuple[str, ...]], ...],
    ) -> tuple[dict[str, Any], ...]:
        control_summaries = self._control_run_catalog(
            environment_id=environment_id,
            selected_goal_slug=selected_goal_slug,
            selected_goal_variant_id=selected_goal_variant_id,
            metric_specs=metric_specs,
            fallback_metric_specs=fallback_metric_specs,
        )
        return control_summaries or ()

    def _control_run_catalog(
        self,
        *,
        environment_id: str,
        selected_goal_slug: str,
        selected_goal_variant_id: str,
        metric_specs: tuple[tuple[RankCriterion, tuple[str, ...]], ...],
        fallback_metric_specs: tuple[tuple[RankCriterion, tuple[str, ...]], ...],
    ) -> tuple[dict[str, Any], ...] | None:
        if self.control_bucket is None or not selected_goal_slug:
            return None
        scope_key = goal_variant_scope_key(goal_slug=selected_goal_slug)
        variant_index = self.control_bucket.get_json_optional(f"{scope_key}/index.json")
        if variant_index is None:
            return ()
        if int(variant_index.get("schema_version") or 0) != GOAL_VARIANT_INDEX_SCHEMA_VERSION:
            raise ValueError("unsupported goal variant index schema")
        if variant_index.get("scope") != {"goal_slug": selected_goal_slug}:
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
        environment_id: str,
        selected_goal_slug: str,
        selected_goal_variant_id: str,
        metric_specs: tuple[tuple[RankCriterion, tuple[str, ...]], ...],
        fallback_metric_specs: tuple[tuple[RankCriterion, tuple[str, ...]], ...],
    ) -> None:
        try:
            items = self._load_run_catalog(
                environment_id=environment_id,
                selected_goal_slug=selected_goal_slug,
                selected_goal_variant_id=selected_goal_variant_id,
                metric_specs=metric_specs,
                fallback_metric_specs=fallback_metric_specs,
            )
            self._store_run_catalog(cache_key=cache_key, items=items)
        except Exception:
            # A stale local catalog is preferable to turning a transient
            # control-index failure into a blocking playback-selection failure.
            pass
        finally:
            with self._cache_lock:
                self._run_catalog_refreshing.discard(cache_key)

    def _schedule_run_catalog_refresh(
        self,
        *,
        cache_key: str,
        environment_id: str,
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
                "environment_id": environment_id,
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
        environment_id: str,
        goal_id: str = "",
        goal_variant_id: str = "",
        query: str = "",
        cursor: str | None = None,
    ) -> CatalogPage:
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
        cache_key = self._run_catalog_cache_key(
            environment_id=environment_id,
            goal_slug=selected_goal_slug,
            goal_variant_id=selected_goal_variant,
            metric_specs=metric_specs,
            fallback_metric_specs=fallback_metric_specs,
        )
        cached = self._cached_run_catalog(
            cache_key=cache_key,
            environment_id=environment_id,
        )
        if cached is None:
            summaries = self._load_run_catalog(
                environment_id=environment_id,
                selected_goal_slug=selected_goal_slug,
                selected_goal_variant_id=selected_goal_variant,
                metric_specs=metric_specs,
                fallback_metric_specs=fallback_metric_specs,
            )
            cached = self._store_run_catalog(cache_key=cache_key, items=summaries)
        elif time.time() - cached[0] >= RUN_CATALOG_CACHE_SECONDS:
            self._schedule_run_catalog_refresh(
                cache_key=cache_key,
                environment_id=environment_id,
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
        environment_id: str,
        query: str = "",
        cursor: str | None = None,
    ) -> CatalogPage:
        normalized = str(query or "").strip().casefold()
        items = [
            GoalSummary(
                environment_id=environment_id,
                goal_id=goal.goal_id,
                goal_slug=goal.goal_slug,
                title=goal.title,
                recipe_count=goal.recipe_count,
                goal_path=goal.goal_path,
            ).to_dict()
            for goal in self._repository_goals(environment_id=environment_id)
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
        environment_id: str,
        goal_id: str,
        query: str = "",
        cursor: str | None = None,
    ) -> CatalogPage:
        repository_goal = self._repository_goal(
            environment_id=environment_id,
            goal_id=goal_id,
        )
        cache_key = self._variant_cache_key(
            environment_id=environment_id,
            goal_slug=repository_goal.goal_slug,
        )
        cached = self._cached_goal_variants(
            cache_key=cache_key,
            environment_id=environment_id,
            goal_slug=repository_goal.goal_slug,
        )
        if cached is None:
            items = self._load_goal_variants(
                repository_goal=repository_goal,
            )
            cached = self._store_goal_variants(cache_key=cache_key, items=items)
        elif time.time() - cached[0] >= GOAL_VARIANT_CACHE_SECONDS:
            self._schedule_goal_variant_refresh(
                cache_key=cache_key,
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
        return _page_items(items, cursor)

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
                "repository has no recipe "
                f"{environment_id}/{goal_id}/{normalized_recipe_id}"
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
        recipe_resolution = (
            resolution.get("recipe") if isinstance(resolution, Mapping) else None
        )
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
        except Exception:
            return None
        if int(index.get("schema_version") or 0) != 1 or index.get("run_id") != run_id:
            raise ValueError("public run index identity mismatch")
        manifests = []
        for raw in index.get("checkpoints") or ():
            if not isinstance(raw, Mapping):
                raise ValueError("public run index contains an invalid checkpoint")
            manifest = CheckpointManifest.from_dict(raw)
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

    def run_goal(self, *, environment_id: str, run_id: str) -> str:
        goal_id, _variant_id = self.run_goal_variant(
            environment_id=environment_id,
            run_id=run_id,
        )
        return goal_id

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
            recipe = (
                public_document.get("recipe")
                if isinstance(public_document, Mapping)
                else None
            )
            raw_descriptor = recipe.get("goal_variant") if isinstance(recipe, Mapping) else None
            if isinstance(raw_descriptor, Mapping):
                descriptor = raw_descriptor
        if descriptor is None:
            raise ValueError("run has no current goal variant descriptor")
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
    ) -> _CheckpointEvaluationData:
        manifest = (
            self.control_bucket.get_json_optional(f"runs/{run_id}/manifest.json")
            if self.control_bucket is not None
            else None
        )
        wandb = manifest.get("wandb") if isinstance(manifest, Mapping) else None
        entity = str(wandb.get("entity") or "").strip() if isinstance(wandb, Mapping) else ""
        project = (
            str(wandb.get("project") or "").strip()
            if isinstance(wandb, Mapping)
            else ""
        )
        if not entity or not project:
            return _CheckpointEvaluationData({}, None, None, {}, ())
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
            metric_schema = evaluation_metric_schema(config.get("metrics_schema_version"))
            evaluation_rank = parse_objective_rank(
                config.get("selection_rank"),
                metrics_schema_version=metric_schema.version,
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
                        "metrics": {
                            criterion.metric: (
                                float(step)
                                if criterion.metric == LEADER_CHECKPOINT_STEP
                                else None
                            )
                            for criterion in evaluation_rank
                        },
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
                for criterion in evaluation_rank:
                    metric = criterion.metric
                    if metric == LEADER_CHECKPOINT_STEP:
                        continue
                    for raw in run.scan_history(
                        keys=[metric_schema.checkpoint_step, metric],
                        page_size=10_000,
                    ):
                        if not isinstance(raw, Mapping):
                            continue
                        step = _safe_int(raw.get(metric_schema.checkpoint_step))
                        value = _safe_float(raw.get(metric))
                        evaluation = evaluations.get(step) if step is not None else None
                        if evaluation is not None and value is not None:
                            evaluation["metrics"][metric] = value
        except Exception:
            # Public checkpoints remain playable when W&B history is unavailable.
            evaluations = {}
            training_seed = None
            evaluation_seed = None
            evaluation_rank = ()
        training_metric_history = (
            _checkpoint_training_metric_history(run) if run is not None else {}
        )
        data = _CheckpointEvaluationData(
            evaluations=evaluations,
            training_seed=training_seed,
            evaluation_seed=evaluation_seed,
            training_metric_history=training_metric_history,
            evaluation_rank=evaluation_rank,
        )
        with self._lock:
            self._evaluation_cache[cache_key] = (now, data)
        return data

    def checkpoints(
        self,
        *,
        run_id: str,
        query: str = "",
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
            manifest = (
                self.control_bucket.get_json_optional(f"runs/{run_id}/manifest.json")
                if self.control_bucket is not None
                else None
            )
            raw_descriptor = manifest.get("goal_variant") if isinstance(manifest, Mapping) else None
            if not isinstance(raw_descriptor, Mapping):
                public_document = self._public_run_recipe_document(run_id)
                recipe = (
                    public_document.get("recipe")
                    if isinstance(public_document, Mapping)
                    else None
                )
                raw_descriptor = (
                    recipe.get("goal_variant") if isinstance(recipe, Mapping) else None
                )
            if not isinstance(raw_descriptor, Mapping):
                raise ValueError("run has no current goal variant descriptor")
            descriptor = validate_goal_variant_descriptor(raw_descriptor)
            observed_variant = str(descriptor["variant_id"])
            expected_effective_goal_hash = str(
                descriptor["effective_goal_contract_sha256"]
            )
            if observed_variant != selected_variant:
                raise ValueError("run does not belong to the selected goal variant")
        evaluation_data = self._checkpoint_evaluations(
            run_id=run_id,
        )
        rows: list[CheckpointSummary] = []
        for raw in index.get("checkpoints") or ():
            if not isinstance(raw, Mapping):
                raise ValueError("public run index contains an invalid checkpoint")
            manifest = CheckpointManifest.from_dict(raw)
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
            rows.append(row)
        rows.sort(key=lambda row: (row.step, row.sha256), reverse=True)
        training_rank = tuple(
            criterion for criterion, _sources in LIVE_TRAINING_METRICS
        )
        best_training_id = _best_checkpoint_id(rows, training_rank)
        best_evaluation_id = _best_checkpoint_id(
            rows,
            evaluation_data.evaluation_rank,
            evaluation=True,
        )
        result: list[dict[str, Any]] = []
        for row in rows:
            item = {
                **row.to_dict(),
                "best_training": row.checkpoint_id == best_training_id,
                "best_evaluation": row.checkpoint_id == best_evaluation_id,
            }
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
            result.append(item)
        return tuple(result)


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
    "RunSummary",
    "WandbRunLocation",
    "checkpoint_manifest_url",
    "is_wandb_url",
    "normalize_search_query",
    "parse_wandb_location",
]
