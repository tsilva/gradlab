from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from gradlab.clock import Clock, SystemClock
from gradlab.goal_catalog import (
    GOAL_CATALOG_SCHEMA_VERSION,
    goal_catalog_ack_key,
    goal_catalog_event_key,
    goal_catalog_event_prefix,
    goal_catalog_generation_digest,
    goal_catalog_generation_key,
    goal_catalog_page_digest,
    goal_catalog_page_key,
    goal_catalog_pointer_key,
    merge_goal_catalog_events,
    validate_goal_catalog_event,
    validate_goal_catalog_generation,
    validate_goal_catalog_page,
    validate_goal_catalog_pointer,
)
from gradlab.json_utils import canonical_json_sha256
from gradlab.r2_store import ConditionalWriteConflict, R2Bucket


@dataclass(frozen=True)
class CatalogReconcileResult:
    goal_slug: str
    published: bool
    generation_sha256: str
    applied_event_count: int
    acknowledged_event_count: int
    orphan_events: tuple[Mapping[str, str], ...]
    pointer: Mapping[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_slug": self.goal_slug,
            "published": self.published,
            "generation_sha256": self.generation_sha256,
            "applied_event_count": self.applied_event_count,
            "acknowledged_event_count": self.acknowledged_event_count,
            "orphan_events": [dict(item) for item in self.orphan_events],
            "pointer": None if self.pointer is None else dict(self.pointer),
        }


class GoalCatalogProjector:
    """Builds a disposable per-goal catalog from durable, source-verified events."""

    def __init__(
        self,
        *,
        control: R2Bucket,
        evaluation: R2Bucket,
        clock: Clock | None = None,
    ) -> None:
        self.control = control
        self.evaluation = evaluation
        self.clock = clock or SystemClock()

    def put_event(self, event: Mapping[str, Any]) -> str:
        validated = validate_goal_catalog_event(event)
        key = goal_catalog_event_key(validated)
        try:
            return self.control.put_json(key, validated, create_only=True)
        except ConditionalWriteConflict:
            stored = validate_goal_catalog_event(self.control.get_json(key))
            if stored != validated:
                raise ValueError("immutable goal catalog event conflicts with storage")
            return str(self.control.head(key)["etag"])

    def pointer(self, goal_slug: str) -> dict[str, Any] | None:
        document = self.control.get_json_optional(goal_catalog_pointer_key(goal_slug))
        if document is None:
            return None
        return validate_goal_catalog_pointer(document, expected_goal_slug=goal_slug)

    def generation(self, goal_slug: str) -> dict[str, Any] | None:
        pointer = self.pointer(goal_slug)
        if pointer is None:
            return None
        generation = validate_goal_catalog_generation(
            self.control.get_json(pointer["generation_key"]),
            expected_digest=pointer["generation_sha256"],
        )
        if generation["generated_at"] != pointer["generated_at"]:
            raise ValueError("goal catalog pointer timestamp disagrees with its generation")
        return generation

    def all_runs(
        self,
        goal_slug: str,
        *,
        generation: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], ...]:
        selected = (
            validate_goal_catalog_generation(generation)
            if generation is not None
            else self.generation(goal_slug)
        )
        if selected is None:
            return ()
        runs = [
            deepcopy(dict(run))
            for run in (*selected["active_runs"], *selected["terminal_runs"])
        ]
        for reference in selected["archive_pages"]:
            page = validate_goal_catalog_page(
                self.control.get_json(str(reference["page_key"])),
                expected_digest=str(reference["page_sha256"]),
            )
            if page["goal_slug"] != goal_slug:
                raise ValueError("goal catalog archive page belongs to another goal")
            runs.extend(deepcopy(dict(run)) for run in page["runs"])
        return tuple(runs)

    def _verified_events(
        self,
        goal_slug: str,
    ) -> tuple[list[dict[str, Any]], tuple[dict[str, str], ...]]:
        events: list[dict[str, Any]] = []
        orphans: list[dict[str, str]] = []
        for key in self.control.iter_keys(goal_catalog_event_prefix(goal_slug)):
            try:
                event = validate_goal_catalog_event(self.control.get_json(key))
                if event["goal_slug"] != goal_slug or key != goal_catalog_event_key(event):
                    raise ValueError("event key does not match its content")
                source = (
                    self.control
                    if event["source_bucket"] == "control"
                    else self.evaluation
                ).get_json_optional(event["source_key"])
                if source is None:
                    orphans.append(
                        {
                            "event_id": event["event_id"],
                            "event_key": key,
                            "reason": "authoritative source is missing",
                        }
                    )
                    continue
                if canonical_json_sha256(source) != event["source_sha256"]:
                    orphans.append(
                        {
                            "event_id": event["event_id"],
                            "event_key": key,
                            "reason": "authoritative source digest changed",
                        }
                    )
                    continue
                events.append(event)
            except (KeyError, TypeError, ValueError) as exc:
                orphans.append(
                    {
                        "event_id": "",
                        "event_key": key,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
        return events, tuple(orphans)

    @staticmethod
    def _projection_identity(generation: Mapping[str, Any]) -> dict[str, Any]:
        result = deepcopy(dict(generation))
        result.pop("generated_at", None)
        return result

    def _acknowledge(
        self,
        *,
        events: list[dict[str, Any]],
        generation_sha256: str,
    ) -> int:
        acknowledged = 0
        for event in events:
            key = goal_catalog_ack_key(event["event_id"])
            if self.control.get_json_optional(key) is not None:
                acknowledged += 1
                continue
            document = {
                "schema_version": GOAL_CATALOG_SCHEMA_VERSION,
                "event_id": event["event_id"],
                "goal_slug": event["goal_slug"],
                "generation_sha256": generation_sha256,
                "acknowledged_at": self.clock.utc_now(),
            }
            try:
                self.control.put_json(key, document, create_only=True)
            except ConditionalWriteConflict:
                pass
            acknowledged += 1
        return acknowledged

    def reconcile(
        self,
        goal_slug: str,
        *,
        publish: bool = True,
        maximum_attempts: int = 8,
    ) -> CatalogReconcileResult:
        normalized_goal = str(goal_slug or "").strip()
        if not normalized_goal:
            raise ValueError("goal catalog reconciliation requires goal_slug")
        last_orphans: tuple[dict[str, str], ...] = ()
        for attempt in range(max(1, int(maximum_attempts))):
            pointer_key = goal_catalog_pointer_key(normalized_goal)
            pointer_document = self.control.get_json_optional(pointer_key)
            pointer_etag = (
                str(self.control.head(pointer_key)["etag"])
                if pointer_document is not None
                else None
            )
            current_pointer = (
                validate_goal_catalog_pointer(
                    pointer_document,
                    expected_goal_slug=normalized_goal,
                )
                if pointer_document is not None
                else None
            )
            current_generation = (
                validate_goal_catalog_generation(
                    self.control.get_json(current_pointer["generation_key"]),
                    expected_digest=current_pointer["generation_sha256"],
                )
                if current_pointer is not None
                else None
            )
            events, last_orphans = self._verified_events(normalized_goal)
            generated_at = self.clock.utc_now()
            generation, pages = merge_goal_catalog_events(
                events,
                goal_slug=normalized_goal,
                generated_at=generated_at,
            )
            if (
                current_generation is not None
                and self._projection_identity(current_generation)
                == self._projection_identity(generation)
            ):
                digest = str(current_pointer["generation_sha256"])
                acknowledged = self._acknowledge(
                    events=events,
                    generation_sha256=digest,
                )
                return CatalogReconcileResult(
                    goal_slug=normalized_goal,
                    published=False,
                    generation_sha256=digest,
                    applied_event_count=len(events),
                    acknowledged_event_count=acknowledged,
                    orphan_events=last_orphans,
                    pointer=current_pointer,
                )
            digest = goal_catalog_generation_digest(generation)
            pointer = {
                "schema_version": GOAL_CATALOG_SCHEMA_VERSION,
                "goal_slug": normalized_goal,
                "generation_sha256": digest,
                "generation_key": goal_catalog_generation_key(normalized_goal, digest),
                "generated_at": generation["generated_at"],
            }
            if not publish:
                return CatalogReconcileResult(
                    goal_slug=normalized_goal,
                    published=False,
                    generation_sha256=digest,
                    applied_event_count=len(events),
                    acknowledged_event_count=sum(
                        self.control.get_json_optional(goal_catalog_ack_key(event["event_id"]))
                        is not None
                        for event in events
                    ),
                    orphan_events=last_orphans,
                    pointer=pointer,
                )
            for page in pages:
                page_digest = goal_catalog_page_digest(page)
                page_key = goal_catalog_page_key(normalized_goal, page_digest)
                try:
                    self.control.put_json(page_key, page, create_only=True)
                except ConditionalWriteConflict:
                    validate_goal_catalog_page(
                        self.control.get_json(page_key),
                        expected_digest=page_digest,
                    )
            try:
                self.control.put_json(pointer["generation_key"], generation, create_only=True)
            except ConditionalWriteConflict:
                validate_goal_catalog_generation(
                    self.control.get_json(pointer["generation_key"]),
                    expected_digest=digest,
                )
            try:
                self.control.put_json(
                    pointer_key,
                    pointer,
                    create_only=pointer_document is None,
                    if_match=pointer_etag,
                )
            except ConditionalWriteConflict:
                if attempt + 1 >= maximum_attempts:
                    break
                self.clock.sleep(min(0.01 * (2**attempt), 0.2))
                continue
            readback_pointer = self.pointer(normalized_goal)
            readback = self.generation(normalized_goal)
            if (
                readback_pointer is None
                or readback is None
                or readback_pointer["generation_sha256"] != digest
                or {row["event_id"] for row in readback["applied_events"]}
                != {event["event_id"] for event in events}
            ):
                raise RuntimeError("goal catalog publication failed read-back verification")
            acknowledged = self._acknowledge(events=events, generation_sha256=digest)
            return CatalogReconcileResult(
                goal_slug=normalized_goal,
                published=True,
                generation_sha256=digest,
                applied_event_count=len(events),
                acknowledged_event_count=acknowledged,
                orphan_events=last_orphans,
                pointer=readback_pointer,
            )
        raise ConditionalWriteConflict("goal catalog pointer changed during every CAS attempt")


__all__ = ["CatalogReconcileResult", "GoalCatalogProjector"]
