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

from rlab.config_loader import load_mapping_document
from rlab.config_validation import load_goal_contract
from rlab.early_stop import EARLY_STOP_OPERATORS, normalize_metric_threshold_rules
from rlab.metric_names import (
    EVAL_ACCEPTANCE_EPISODES_COMPLETED,
    EVAL_ACCEPTANCE_EPISODES_PLANNED,
    EVAL_ACCEPTANCE_FAILURE_COUNT,
    EVAL_ACCEPTANCE_PASS,
    EVAL_CHECKPOINT_STEP,
    TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN,
    TRAIN_EPISODE_RETURN_SHAPED_MEAN,
    TRAIN_GLOBAL_STEP,
    TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN,
)
from rlab.model_sources import DEFAULT_PUBLIC_MODELS_BASE_URL, _public_json
from rlab.ranking import RankCriterion, parse_objective_rank
from rlab.recipe_variants import normalize_recipe_overrides, recipe_variant_id
from rlab.run_contracts import CheckpointManifest, RUN_ID_PATTERN
from rlab.wandb_utils import (
    load_wandb_env,
    resolve_wandb_project,
    wandb_entity_from_env,
)


WANDB_HOSTS = {"wandb.ai", "www.wandb.ai"}
CATALOG_PAGE_SIZE = 50
CATALOG_INDEX_SCHEMA_VERSION = 1
CATALOG_CACHE_SCHEMA_VERSION = 1
CATALOG_INDEX_FILENAME = "_catalog.yaml"
EVALUATION_CACHE_SECONDS = 10.0
LIVE_TRAINING_METRICS = (
    (
        RankCriterion(
            direction="max",
            metric=TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN,
        ),
        (TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN,),
    ),
    (
        RankCriterion(
            direction="max",
            metric=TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN,
        ),
        (
            TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN,
            TRAIN_EPISODE_RETURN_SHAPED_MEAN,
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
class RunSummary:
    entity: str
    project: str
    run_id: str
    name: str
    state: str
    goal: str
    recipe: str
    recipe_sha256: str
    recipe_overrides: tuple[str, ...]
    recipe_variant_id: str
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


def _first_summary_float(summary: Any, metrics: Iterable[str]) -> float | None:
    for metric in metrics:
        value = _safe_float(summary.get(metric))
        if value is not None:
            return value
    return None


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


class _CatalogStream:
    def __init__(self, values: Iterable[dict[str, Any]]) -> None:
        self.iterator: Iterator[dict[str, Any]] = iter(values)
        self.items: list[dict[str, Any]] = []
        self.exhausted = False

    def page(self, offset: int, limit: int) -> CatalogPage:
        target = offset + limit + 1
        while len(self.items) < target and not self.exhausted:
            try:
                self.items.append(next(self.iterator))
            except StopIteration:
                self.exhausted = True
        selected = tuple(self.items[offset : offset + limit])
        has_more = len(self.items) > offset + limit or not self.exhausted
        return CatalogPage(
            items=selected,
            next_cursor=_cursor_for(offset + limit) if has_more else None,
        )


class PlayCatalog:
    """Repository catalog, W&B run metadata, and public-checkpoint discovery."""

    def __init__(
        self,
        *,
        public_models_base_url: str = DEFAULT_PUBLIC_MODELS_BASE_URL,
        repo_root: Path | str | None = None,
        cache_path: Path | str | None = None,
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
        self._api: Any | None = None
        self._lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._streams: dict[tuple[str, ...], _CatalogStream] = {}
        self._repository_cache: (
            tuple[
                tuple[tuple[str, int, int], ...],
                tuple[_RepositoryGoal, ...],
            ]
            | None
        ) = None
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
                tuple[int, int],
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
        return cache_root / "rlab" / "play-catalog" / f"{identity}.json"

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

    def _stream(
        self,
        key: tuple[str, ...],
        values: Callable[[], Iterable[dict[str, Any]]],
    ) -> _CatalogStream:
        stream = self._streams.get(key)
        if stream is None:
            stream = _CatalogStream(values())
            self._streams[key] = stream
        return stream

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

    def _repository_namespaces(self) -> tuple[_RepositoryNamespace, ...] | None:
        index_path = self.goals_root / CATALOG_INDEX_FILENAME
        if not index_path.is_file():
            return None
        stat_result = index_path.stat()
        fingerprint = (stat_result.st_mtime_ns, stat_result.st_size)
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
        if not isinstance(raw_namespaces, Mapping) or not raw_namespaces:
            raise ValueError(f"repository goal catalog has no namespaces: {index_path}")

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
            namespace_root = self.goals_root / directory
            if not namespace_root.is_dir():
                raise ValueError(
                    f"repository goal catalog namespace does not exist: {namespace_root}"
                )
            if title_template:
                try:
                    title_template.format(goal_id="example")
                except (IndexError, KeyError, ValueError) as exc:
                    raise ValueError(
                        f"repository goal catalog namespace {directory!r} has an invalid "
                        "title_template"
                    ) from exc
            namespaces.append(
                _RepositoryNamespace(
                    directory=directory,
                    project=project,
                    title_template=title_template,
                )
            )

        declared = {namespace.directory for namespace in namespaces}
        discovered = {
            path.relative_to(self.goals_root).parts[0]
            for path in self.goals_root.rglob("_goal.yaml")
        }
        missing = discovered - declared
        stale = declared - discovered
        if missing:
            raise ValueError(
                "repository goal catalog is missing namespaces: " + ", ".join(sorted(missing))
            )
        if stale:
            raise ValueError(
                "repository goal catalog declares empty namespaces: " + ", ".join(sorted(stale))
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
        if self.cache_path is None or not self.cache_path.is_file():
            return None
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if (
                not isinstance(payload, Mapping)
                or payload.get("schema_version") != CATALOG_CACHE_SCHEMA_VERSION
                or payload.get("repo_root") != str(self.repo_root)
            ):
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
        except OSError, TypeError, ValueError, json.JSONDecodeError:
            return None

    def _write_persistent_project(
        self,
        *,
        project: str,
        fingerprint: tuple[tuple[str, int, int], ...],
        goals: tuple[_RepositoryGoal, ...],
    ) -> None:
        if self.cache_path is None:
            return
        with self._cache_lock:
            payload: dict[str, Any] = {
                "schema_version": CATALOG_CACHE_SCHEMA_VERSION,
                "repo_root": str(self.repo_root),
                "projects": {},
            }
            if self.cache_path.is_file():
                try:
                    existing = json.loads(self.cache_path.read_text(encoding="utf-8"))
                    if (
                        isinstance(existing, dict)
                        and existing.get("schema_version") == CATALOG_CACHE_SCHEMA_VERSION
                        and existing.get("repo_root") == str(self.repo_root)
                        and isinstance(existing.get("projects"), dict)
                    ):
                        payload = existing
                except OSError, json.JSONDecodeError:
                    pass
            projects = payload["projects"]
            projects[project] = {
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

    def _legacy_repository_goals(self) -> tuple[_RepositoryGoal, ...]:
        if not self.goals_root.is_dir():
            raise ValueError(f"repository goals directory does not exist: {self.goals_root}")
        paths = tuple(sorted(self.goals_root.rglob("_goal.yaml")))
        catalog_sources = tuple(
            sorted(
                (
                    *self.goals_root.rglob("*.yaml"),
                    *self.goals_root.rglob("*.yml"),
                )
            )
        )
        fingerprint = tuple(
            (
                path.relative_to(self.repo_root).as_posix(),
                path.stat().st_mtime_ns,
                path.stat().st_size,
            )
            for path in catalog_sources
        )
        with self._lock:
            cached = self._repository_cache
            if cached is not None and cached[0] == fingerprint:
                return cached[1]

        goals: list[_RepositoryGoal] = []
        for path in paths:
            goals.append(self._compose_repository_goal(path))
        identities = [(goal.project, goal.goal_id) for goal in goals]
        if len(identities) != len(set(identities)):
            raise ValueError("repository goals contain duplicate project/goal identities")
        result = tuple(sorted(goals, key=lambda goal: (goal.project, goal.goal_id)))
        with self._lock:
            self._repository_cache = (fingerprint, result)
        return result

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
        if namespaces is None:
            goals = self._legacy_repository_goals()
            return tuple(goal for goal in goals if project is None or goal.project == project)
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
        if namespaces is None:
            counts: dict[str, int] = {}
            for goal in self._legacy_repository_goals():
                counts[goal.project] = counts.get(goal.project, 0) + 1
            return counts
        counts = {}
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

    def initial_projects(self, explicit_entity: object = None) -> dict[str, Any]:
        entity = self.default_entity(explicit_entity)
        page = self.projects(entity=entity)
        return {"entity": entity, **page.to_dict()}

    def runs(
        self,
        *,
        entity: str,
        project: str,
        goal_id: str = "",
        query: str = "",
        cursor: str | None = None,
    ) -> CatalogPage:
        normalized = str(query or "").strip().casefold()
        selected_goal = str(goal_id or "").strip()
        repository_goal = (
            self._repository_goal(project=project, goal_id=selected_goal) if selected_goal else None
        )
        selected_goal_slug = repository_goal.goal_slug if repository_goal else ""
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

        def values() -> Iterator[dict[str, Any]]:
            api_runs = self._wandb_api().runs(
                f"{entity}/{project}",
                order="-created_at",
                per_page=200,
                lazy=True,
            )
            summaries: list[dict[str, Any]] = []
            for run in api_runs:
                run_id = str(getattr(run, "id", "") or "")
                if RUN_ID_PATTERN.fullmatch(run_id) is None:
                    continue
                config = dict(getattr(run, "config", {}) or {})
                goal_slug = str(config.get("goal_slug") or "")
                if selected_goal_slug and goal_slug != selected_goal_slug:
                    continue
                run_metrics = getattr(run, "summary", {}) or {}
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
                summary = RunSummary(
                    entity=entity,
                    project=project,
                    run_id=run_id,
                    name=str(getattr(run, "name", "") or run_id),
                    state=str(getattr(run, "state", "") or ""),
                    goal=goal_slug,
                    recipe=str(config.get("recipe_slug") or ""),
                    recipe_sha256=str(config.get("recipe_sha256") or ""),
                    recipe_overrides=overrides,
                    recipe_variant_id=variant_id,
                    description=str(
                        getattr(run, "notes", "")
                        or config.get("run_description")
                        or ""
                    ).strip(),
                    seed=_safe_int(config.get("seed")),
                    created_at=str(getattr(run, "created_at", "") or ""),
                    updated_at=str(getattr(run, "updated_at", "") or ""),
                    url=str(getattr(run, "url", "") or ""),
                    metrics={
                        criterion.metric: _first_summary_float(run_metrics, sources)
                        for criterion, sources in (*metric_specs, *fallback_metric_specs)
                    },
                )
                if normalized and normalized not in _search_text(
                    summary.run_id,
                    summary.name,
                    summary.state,
                    summary.goal,
                    summary.recipe,
                    summary.recipe_sha256,
                    summary.recipe_overrides,
                    summary.recipe_variant_id,
                    summary.description,
                    summary.seed,
                ):
                    continue
                summaries.append(summary.to_dict())
            _rank_run_summaries(
                summaries,
                primary=metric_specs,
                fallback=fallback_metric_specs,
            )
            yield from summaries

        with self._lock:
            stream_key = ("runs", entity, project, selected_goal, normalized)
            if cursor is None:
                stream = _CatalogStream(values())
                self._streams[stream_key] = stream
            else:
                stream = self._stream(stream_key, values)
            page = stream.page(_cursor_offset(cursor), CATALOG_PAGE_SIZE)
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

    def run_goal(self, *, entity: str, project: str, run_id: str) -> str:
        if RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise ValueError("run id must match rlab-<32 lowercase hex>")
        run = self._wandb_api().run(f"{entity}/{project}/{run_id}")
        config = dict(getattr(run, "config", {}) or {})
        goal_slug = str(config.get("goal_slug") or "").strip()
        for goal in self._repository_goals(project=project):
            if goal.goal_slug == goal_slug:
                return goal.goal_id
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
            return _CheckpointEvaluationData({}, None, None)
        cache_key = (entity, project, run_id)
        now = time.monotonic()
        with self._lock:
            cached = self._evaluation_cache.get(cache_key)
            if cached is not None and now - cached[0] < EVALUATION_CACHE_SECONDS:
                return cached[1]
        try:
            run = self._wandb_api().run(f"{entity}/{project}/{run_id}")
            config = dict(getattr(run, "config", {}) or {})
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
                )
                keys = {
                    EVAL_CHECKPOINT_STEP,
                    EVAL_ACCEPTANCE_PASS,
                    EVAL_ACCEPTANCE_EPISODES_PLANNED,
                    EVAL_ACCEPTANCE_EPISODES_COMPLETED,
                    EVAL_ACCEPTANCE_FAILURE_COUNT,
                    *(str(rule["metric"]) for rule in rules),
                }
                evaluations = {}
                for raw in run.scan_history(keys=sorted(keys), page_size=10_000):
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
                        "episodes_planned": _safe_int(raw.get(EVAL_ACCEPTANCE_EPISODES_PLANNED)),
                        "episodes_completed": _safe_int(
                            raw.get(EVAL_ACCEPTANCE_EPISODES_COMPLETED)
                        ),
                        "failure_count": _safe_int(raw.get(EVAL_ACCEPTANCE_FAILURE_COUNT)),
                        "criteria": criteria,
                    }
        except Exception:
            # Public checkpoints remain playable when W&B history is unavailable.
            evaluations = {}
            training_seed = None
            evaluation_seed = None
        data = _CheckpointEvaluationData(
            evaluations=evaluations,
            training_seed=training_seed,
            evaluation_seed=evaluation_seed,
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
    ) -> tuple[dict[str, Any], ...]:
        if RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise ValueError("run id must match rlab-<32 lowercase hex>")
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
                evaluation=evaluation,
            )
            if normalized and normalized not in _search_text(
                row.checkpoint_id,
                row.step,
                row.purpose,
                row.sha256,
                row.created_at,
                "promoted" if row.promoted else "",
                row.evaluation,
            ):
                continue
            rows.append(row)
        rows.sort(key=lambda row: (row.promoted, row.step, row.sha256), reverse=True)
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
    "GoalSummary",
    "PlayCatalog",
    "ProjectSummary",
    "RunSummary",
    "WandbLocation",
    "checkpoint_manifest_url",
    "is_wandb_url",
    "normalize_search_query",
    "parse_wandb_location",
]
