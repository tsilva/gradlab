from __future__ import annotations

import base64
import re
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal
from urllib.parse import unquote, urlparse

from rlab.early_stop import EARLY_STOP_OPERATORS, normalize_early_stop_config
from rlab.metric_names import (
    EVAL_ACCEPTANCE_EPISODES_COMPLETED,
    EVAL_ACCEPTANCE_EPISODES_PLANNED,
    EVAL_ACCEPTANCE_FAILURE_COUNT,
    EVAL_ACCEPTANCE_PASS,
    EVAL_CHECKPOINT_STEP,
)
from rlab.model_sources import DEFAULT_PUBLIC_MODELS_BASE_URL, _public_json
from rlab.run_contracts import CheckpointManifest, RUN_ID_PATTERN
from rlab.wandb_utils import load_wandb_env


WANDB_HOSTS = {"wandb.ai", "www.wandb.ai"}
CATALOG_PAGE_SIZE = 50
EVALUATION_CACHE_SECONDS = 10.0


@dataclass(frozen=True)
class WandbLocation:
    entity: str
    project: str
    run_id: str | None = None


@dataclass(frozen=True)
class CatalogPage:
    items: tuple[dict[str, Any], ...]
    next_cursor: str | None

    def to_dict(self) -> dict[str, Any]:
        return {"items": list(self.items), "next_cursor": self.next_cursor}


@dataclass(frozen=True)
class ProjectSummary:
    entity: str
    name: str
    created_at: str
    url: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GoalSummary:
    entity: str
    project: str
    goal_id: str
    goal_slug: str
    run_count: int
    updated_at: str

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
    seed: int | None
    created_at: str
    updated_at: str
    url: str

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
    evaluation: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def _goal_id(goal_slug: object) -> str:
    return str(goal_slug or "").strip().rsplit("/", 1)[-1]


def _safe_int(value: object) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric = float(value)
    return numeric if numeric == numeric and abs(numeric) != float("inf") else None


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
    """Backend-only W&B metadata and public-checkpoint discovery."""

    def __init__(self, *, public_models_base_url: str = DEFAULT_PUBLIC_MODELS_BASE_URL) -> None:
        self.public_models_base_url = str(public_models_base_url).rstrip("/")
        self._api: Any | None = None
        self._lock = threading.Lock()
        self._streams: dict[tuple[str, ...], _CatalogStream] = {}
        self._evaluation_cache: dict[
            tuple[str, str, str],
            tuple[float, dict[int, dict[str, Any]]],
        ] = {}

    def default_entity(self, explicit: object = None) -> str:
        text = str(explicit or "").strip()
        if text:
            return text
        api = self._wandb_api()
        entity = str(getattr(api, "default_entity", "") or "").strip()
        if not entity:
            raise ValueError(
                "W&B has no default entity; pass --wandb-entity or set WANDB_ENTITY"
            )
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

    def projects(
        self,
        *,
        entity: str,
        query: str = "",
        cursor: str | None = None,
    ) -> CatalogPage:
        normalized = str(query or "").strip().casefold()

        def values() -> Iterator[dict[str, Any]]:
            for project in self._wandb_api().projects(entity=entity, per_page=200):
                summary = ProjectSummary(
                    entity=str(getattr(project, "entity", entity) or entity),
                    name=str(getattr(project, "name", "") or ""),
                    created_at=str(getattr(project, "created_at", "") or ""),
                    url=str(getattr(project, "url", "") or ""),
                )
                if not summary.name:
                    continue
                if normalized and normalized not in _search_text(
                    summary.entity,
                    summary.name,
                ):
                    continue
                yield summary.to_dict()

        with self._lock:
            stream = self._stream(("projects", entity, normalized), values)
            return stream.page(_cursor_offset(cursor), CATALOG_PAGE_SIZE)

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

        def values() -> Iterator[dict[str, Any]]:
            api_runs = self._wandb_api().runs(
                f"{entity}/{project}",
                order="-created_at",
                per_page=200,
                lazy=True,
            )
            for run in api_runs:
                run_id = str(getattr(run, "id", "") or "")
                if RUN_ID_PATTERN.fullmatch(run_id) is None:
                    continue
                config = dict(getattr(run, "config", {}) or {})
                goal_slug = str(config.get("goal_slug") or "")
                if selected_goal and _goal_id(goal_slug) != selected_goal:
                    continue
                summary = RunSummary(
                    entity=entity,
                    project=project,
                    run_id=run_id,
                    name=str(getattr(run, "name", "") or run_id),
                    state=str(getattr(run, "state", "") or ""),
                    goal=goal_slug,
                    recipe=str(config.get("recipe_slug") or ""),
                    seed=_safe_int(config.get("seed")),
                    created_at=str(getattr(run, "created_at", "") or ""),
                    updated_at=str(getattr(run, "updated_at", "") or ""),
                    url=str(getattr(run, "url", "") or ""),
                )
                if normalized and normalized not in _search_text(
                    summary.run_id,
                    summary.name,
                    summary.state,
                    summary.goal,
                    summary.recipe,
                    summary.seed,
                    getattr(run, "notes", ""),
                ):
                    continue
                yield summary.to_dict()

        with self._lock:
            stream = self._stream(
                ("runs", entity, project, selected_goal, normalized),
                values,
            )
            return stream.page(_cursor_offset(cursor), CATALOG_PAGE_SIZE)

    def goals(
        self,
        *,
        entity: str,
        project: str,
        query: str = "",
        cursor: str | None = None,
    ) -> CatalogPage:
        normalized = str(query or "").strip().casefold()

        def values() -> Iterator[dict[str, Any]]:
            summaries: dict[str, GoalSummary] = {}
            api_runs = self._wandb_api().runs(
                f"{entity}/{project}",
                order="-created_at",
                per_page=200,
                lazy=True,
            )
            for run in api_runs:
                run_id = str(getattr(run, "id", "") or "")
                if RUN_ID_PATTERN.fullmatch(run_id) is None:
                    continue
                config = dict(getattr(run, "config", {}) or {})
                goal_slug = str(config.get("goal_slug") or "").strip()
                goal_id = _goal_id(goal_slug)
                if not goal_id:
                    continue
                existing = summaries.get(goal_slug)
                updated_at = str(
                    getattr(run, "updated_at", "")
                    or getattr(run, "created_at", "")
                    or ""
                )
                summaries[goal_slug] = GoalSummary(
                    entity=entity,
                    project=project,
                    goal_id=goal_id,
                    goal_slug=goal_slug,
                    run_count=(existing.run_count + 1 if existing is not None else 1),
                    updated_at=existing.updated_at if existing is not None else updated_at,
                )
            for summary in summaries.values():
                if normalized and normalized not in _search_text(
                    summary.goal_id,
                    summary.goal_slug,
                ):
                    continue
                yield summary.to_dict()

        with self._lock:
            stream = self._stream(("goals", entity, project, normalized), values)
            return stream.page(_cursor_offset(cursor), CATALOG_PAGE_SIZE)

    def run_goal(self, *, entity: str, project: str, run_id: str) -> str:
        if RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise ValueError("run id must match rlab-<32 lowercase hex>")
        run = self._wandb_api().run(f"{entity}/{project}/{run_id}")
        config = dict(getattr(run, "config", {}) or {})
        goal_id = _goal_id(config.get("goal_slug"))
        if not goal_id:
            raise ValueError("W&B run has no goal identity")
        return goal_id

    def _checkpoint_evaluations(
        self,
        *,
        entity: str,
        project: str,
        run_id: str,
    ) -> dict[int, dict[str, Any]]:
        entity = str(entity or "").strip()
        project = str(project or "").strip()
        if not entity or not project:
            return {}
        cache_key = (entity, project, run_id)
        now = time.monotonic()
        with self._lock:
            cached = self._evaluation_cache.get(cache_key)
            if cached is not None and now - cached[0] < EVALUATION_CACHE_SECONDS:
                return cached[1]
        try:
            run = self._wandb_api().run(f"{entity}/{project}/{run_id}")
            config = dict(getattr(run, "config", {}) or {})
            contract = config.get("checkpoint_eval_contract")
            if not isinstance(contract, Mapping):
                evaluations = {}
            else:
                rules = normalize_early_stop_config(
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
                                    else bool(
                                        EARLY_STOP_OPERATORS[operator](value, threshold)
                                    )
                                ),
                            }
                        )
                    evaluations[step] = {
                        "status": "accepted" if accepted >= 0.5 else "rejected",
                        "pass": accepted >= 0.5,
                        "episodes_planned": _safe_int(
                            raw.get(EVAL_ACCEPTANCE_EPISODES_PLANNED)
                        ),
                        "episodes_completed": _safe_int(
                            raw.get(EVAL_ACCEPTANCE_EPISODES_COMPLETED)
                        ),
                        "failure_count": _safe_int(
                            raw.get(EVAL_ACCEPTANCE_FAILURE_COUNT)
                        ),
                        "criteria": criteria,
                    }
        except Exception:
            # Public checkpoints remain playable when W&B history is unavailable.
            evaluations = {}
        with self._lock:
            self._evaluation_cache[cache_key] = (now, evaluations)
        return evaluations

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
            str(promotion.get("checkpoint_id") or "")
            if isinstance(promotion, Mapping)
            else ""
        )
        normalized = str(query or "").strip().casefold()
        evaluations = self._checkpoint_evaluations(
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
                evaluation=evaluations.get(manifest.step),
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
