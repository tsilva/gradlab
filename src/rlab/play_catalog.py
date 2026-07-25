from __future__ import annotations

import base64
import re
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal
from urllib.parse import unquote, urlparse

from rlab.model_sources import DEFAULT_PUBLIC_MODELS_BASE_URL, _public_json
from rlab.run_contracts import CheckpointManifest, RUN_ID_PATTERN
from rlab.wandb_utils import load_wandb_env


WANDB_HOSTS = {"wandb.ai", "www.wandb.ai"}
CATALOG_PAGE_SIZE = 50


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


def _safe_int(value: object) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
        query: str = "",
        cursor: str | None = None,
    ) -> CatalogPage:
        normalized = str(query or "").strip().casefold()

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
                summary = RunSummary(
                    entity=entity,
                    project=project,
                    run_id=run_id,
                    name=str(getattr(run, "name", "") or run_id),
                    state=str(getattr(run, "state", "") or ""),
                    goal=str(config.get("goal_slug") or ""),
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
            stream = self._stream(("runs", entity, project, normalized), values)
            return stream.page(_cursor_offset(cursor), CATALOG_PAGE_SIZE)

    def checkpoints(self, *, run_id: str, query: str = "") -> tuple[dict[str, Any], ...]:
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
            )
            if normalized and normalized not in _search_text(
                row.checkpoint_id,
                row.step,
                row.purpose,
                row.sha256,
                row.created_at,
                "promoted" if row.promoted else "",
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
    "PlayCatalog",
    "ProjectSummary",
    "RunSummary",
    "WandbLocation",
    "checkpoint_manifest_url",
    "is_wandb_url",
    "normalize_search_query",
    "parse_wandb_location",
]
