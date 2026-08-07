from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Literal

from gradlab.goal_variants import (
    goal_variant_catalog_contract,
    validate_goal_variant_descriptor,
)
from gradlab.json_utils import canonical_json_sha256
from gradlab.recipe_documents import goal_contract_sha256
from gradlab.run_contracts import ATTEMPT_ID_PATTERN, RUN_ID_PATTERN, SHA256_PATTERN


GOAL_CATALOG_SCHEMA_VERSION = 2
GOAL_CATALOG_ROOT = f"goal-catalog/v{GOAL_CATALOG_SCHEMA_VERSION}"
GOAL_CATALOG_EVENT_ROOT = f"run-index-events/v{GOAL_CATALOG_SCHEMA_VERSION}"
GOAL_CATALOG_HOT_TERMINAL_RUNS = 200
GOAL_CATALOG_ARCHIVE_PAGE_SIZE = 250
GOAL_CATALOG_PHASES = (
    "manifest",
    "attempt-terminal",
    "verified-evaluation",
    "promotion",
)
GOAL_CATALOG_SUCCESS_BADGES = (
    "train/success",
    "eval/success",
)
GOAL_CATALOG_TERMINAL_STATES = frozenset(
    {
        "succeeded",
        "failed",
        "stopped",
        "canceled",
        "cancelled",
        "interrupted",
        "resumable_failure",
        "rejected",
        "expired",
    }
)


def goal_catalog_run_success_badges(run: Mapping[str, Any]) -> tuple[str, ...]:
    """Return success facts proved by one projected lifecycle run record."""

    stop_reason = str(run.get("stop_reason") or "").strip()
    early_stop = run.get("early_stop")
    training_success = (
        isinstance(early_stop, Mapping)
        and str(early_stop.get("outcome") or "").strip().lower() == "success"
    ) or stop_reason.startswith("early_stop_success") or stop_reason in {
        "deterministic_training_acceptance",
        "first_completion",
    }

    evaluation = run.get("evaluation")
    evaluations = run.get("evaluations")
    evaluation_success = (
        isinstance(evaluation, Mapping)
        and str(evaluation.get("status") or "").strip().lower() == "accepted"
    ) or (
        isinstance(evaluations, Mapping)
        and any(
            isinstance(result, Mapping)
            and str(result.get("status") or "").strip().lower() == "accepted"
            for result in evaluations.values()
        )
    ) or isinstance(run.get("promotion"), Mapping) or stop_reason in {
        "completed_after_eval_acceptance",
        "eval_acceptance",
    }

    return tuple(
        badge
        for badge, present in (
            ("train/success", training_success),
            ("eval/success", evaluation_success),
        )
        if present
    )


def _validate_success_badges(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError("goal catalog success badges must be a list")
    badges = tuple(str(item) for item in value)
    normalized = tuple(
        badge for badge in GOAL_CATALOG_SUCCESS_BADGES if badge in set(badges)
    )
    if badges != normalized:
        raise ValueError("goal catalog success badges are invalid or out of order")
    return normalized


def goal_catalog_scope(goal_slug: object) -> str:
    normalized = str(goal_slug or "").strip()
    if not normalized:
        raise ValueError("goal catalog scope requires goal_slug")
    return canonical_json_sha256({"goal_slug": normalized})


def goal_catalog_pointer_key(goal_slug: object) -> str:
    return f"{GOAL_CATALOG_ROOT}/goals/{goal_catalog_scope(goal_slug)}/current.json"


def goal_catalog_generation_key(goal_slug: object, digest: object) -> str:
    normalized = str(digest or "").strip().lower()
    if SHA256_PATTERN.fullmatch(normalized) is None:
        raise ValueError("goal catalog generation requires a lowercase SHA-256")
    return (
        f"{GOAL_CATALOG_ROOT}/goals/{goal_catalog_scope(goal_slug)}"
        f"/generations/{normalized}.json"
    )


def goal_catalog_page_key(goal_slug: object, digest: object) -> str:
    normalized = str(digest or "").strip().lower()
    if SHA256_PATTERN.fullmatch(normalized) is None:
        raise ValueError("goal catalog page requires a lowercase SHA-256")
    return f"{GOAL_CATALOG_ROOT}/goals/{goal_catalog_scope(goal_slug)}/pages/{normalized}.json"


def goal_catalog_event_id(
    *,
    phase: str,
    run_id: str,
    attempt_id: str,
    source_bucket: str,
    source_key: str,
    source_sha256: str,
) -> str:
    if phase not in GOAL_CATALOG_PHASES:
        raise ValueError(f"unsupported goal catalog phase: {phase}")
    return canonical_json_sha256(
        {
            "phase": phase,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "source_bucket": source_bucket,
            "source_key": source_key,
            "source_sha256": source_sha256,
        }
    )


def goal_catalog_event_key(event: Mapping[str, Any]) -> str:
    validated = validate_goal_catalog_event(event)
    return (
        f"{GOAL_CATALOG_EVENT_ROOT}/goals/{goal_catalog_scope(validated['goal_slug'])}"
        f"/{validated['run_id']}/{validated['attempt_id']}"
        f"/{validated['phase']}-{validated['source_sha256']}.json"
    )


def goal_catalog_event_prefix(goal_slug: object) -> str:
    return f"{GOAL_CATALOG_EVENT_ROOT}/goals/{goal_catalog_scope(goal_slug)}/"


def goal_catalog_ack_key(event_id: object) -> str:
    normalized = str(event_id or "").strip().lower()
    if SHA256_PATTERN.fullmatch(normalized) is None:
        raise ValueError("goal catalog acknowledgement requires an event SHA-256")
    return f"{GOAL_CATALOG_EVENT_ROOT}/acks/{normalized}.json"


def _validate_run(run: Mapping[str, Any], *, goal_slug: str) -> dict[str, Any]:
    run_id = str(run.get("run_id") or "")
    attempt_id = str(run.get("attempt_id") or "")
    variant_id = str(run.get("goal_variant_id") or "")
    metrics = run.get("metrics")
    if (
        RUN_ID_PATTERN.fullmatch(run_id) is None
        or ATTEMPT_ID_PATTERN.fullmatch(attempt_id) is None
        or not re.fullmatch(r"goal-variant-[0-9a-f]{24}", variant_id)
        or str(run.get("goal_slug") or "") != goal_slug
        or not isinstance(metrics, Mapping)
        or not str(run.get("created_at") or "")
        or not str(run.get("updated_at") or "")
    ):
        raise ValueError("goal catalog contains a malformed run")
    normalized = deepcopy(dict(run))
    normalized["metrics"] = {
        str(name): float(value)
        for name, value in metrics.items()
        if not isinstance(value, bool) and isinstance(value, int | float)
    }
    if "success_badges" in run:
        badges = _validate_success_badges(run["success_badges"])
        if badges != goal_catalog_run_success_badges(run):
            raise ValueError("goal catalog run success badges disagree with lifecycle evidence")
        normalized["success_badges"] = list(badges)
    return normalized


def validate_goal_catalog_event(document: Mapping[str, Any]) -> dict[str, Any]:
    if int(document.get("schema_version") or 0) != GOAL_CATALOG_SCHEMA_VERSION:
        raise ValueError("unsupported goal catalog event schema")
    phase = str(document.get("phase") or "")
    run_id = str(document.get("run_id") or "")
    attempt_id = str(document.get("attempt_id") or "")
    goal_slug = str(document.get("goal_slug") or "").strip()
    source_bucket = str(document.get("source_bucket") or "")
    source_key = str(document.get("source_key") or "").strip("/")
    source_sha256 = str(document.get("source_sha256") or "").lower()
    event_id = goal_catalog_event_id(
        phase=phase,
        run_id=run_id,
        attempt_id=attempt_id,
        source_bucket=source_bucket,
        source_key=source_key,
        source_sha256=source_sha256,
    )
    if (
        RUN_ID_PATTERN.fullmatch(run_id) is None
        or ATTEMPT_ID_PATTERN.fullmatch(attempt_id) is None
        or not goal_slug
        or source_bucket not in {"control", "evaluation"}
        or not source_key
        or SHA256_PATTERN.fullmatch(source_sha256) is None
        or str(document.get("event_id") or "") != event_id
        or not str(document.get("created_at") or "")
    ):
        raise ValueError("goal catalog event is malformed")
    descriptor = validate_goal_variant_descriptor(document.get("variant") or {})
    run = _validate_run(document.get("run") or {}, goal_slug=goal_slug)
    if (
        descriptor["goal_slug"] != goal_slug
        or run["run_id"] != run_id
        or run["attempt_id"] != attempt_id
        or run["goal_variant_id"] != descriptor["variant_id"]
    ):
        raise ValueError("goal catalog event payload identity mismatch")
    resolved_goal = document.get("resolved_goal")
    if resolved_goal is not None:
        if not isinstance(resolved_goal, Mapping):
            raise ValueError("goal catalog resolved goal must be an object")
        if goal_contract_sha256(resolved_goal) != descriptor["effective_goal_contract_sha256"]:
            raise ValueError("goal catalog resolved goal disagrees with its variant")
    return {
        "schema_version": GOAL_CATALOG_SCHEMA_VERSION,
        "event_id": event_id,
        "phase": phase,
        "goal_slug": goal_slug,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "source_bucket": source_bucket,
        "source_key": source_key,
        "source_sha256": source_sha256,
        "created_at": str(document["created_at"]),
        "variant": descriptor,
        "run": run,
        **({"resolved_goal": deepcopy(dict(resolved_goal))} if resolved_goal is not None else {}),
    }


def build_goal_catalog_event(
    *,
    phase: str,
    goal_slug: str,
    run_id: str,
    attempt_id: str,
    source_bucket: Literal["control", "evaluation"],
    source_key: str,
    source_document: Mapping[str, Any],
    created_at: str,
    variant: Mapping[str, Any],
    run: Mapping[str, Any],
    resolved_goal: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_sha256 = canonical_json_sha256(dict(source_document))
    event_id = goal_catalog_event_id(
        phase=phase,
        run_id=run_id,
        attempt_id=attempt_id,
        source_bucket=source_bucket,
        source_key=source_key,
        source_sha256=source_sha256,
    )
    return validate_goal_catalog_event(
        {
            "schema_version": GOAL_CATALOG_SCHEMA_VERSION,
            "event_id": event_id,
            "phase": phase,
            "goal_slug": goal_slug,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "source_bucket": source_bucket,
            "source_key": source_key,
            "source_sha256": source_sha256,
            "created_at": created_at,
            "variant": dict(variant),
            "run": dict(run),
            **({"resolved_goal": dict(resolved_goal)} if resolved_goal is not None else {}),
        }
    )


def goal_catalog_page_digest(document: Mapping[str, Any]) -> str:
    return canonical_json_sha256(dict(document))


def validate_goal_catalog_page(
    document: Mapping[str, Any],
    *,
    expected_digest: str | None = None,
) -> dict[str, Any]:
    if int(document.get("schema_version") or 0) != GOAL_CATALOG_SCHEMA_VERSION:
        raise ValueError("unsupported goal catalog page schema")
    goal_slug = str(document.get("goal_slug") or "").strip()
    raw_runs = document.get("runs")
    if not goal_slug or not isinstance(raw_runs, list):
        raise ValueError("goal catalog page is malformed")
    runs = [_validate_run(run, goal_slug=goal_slug) for run in raw_runs if isinstance(run, Mapping)]
    if len(runs) != len(raw_runs) or len(runs) > GOAL_CATALOG_ARCHIVE_PAGE_SIZE:
        raise ValueError("goal catalog page contains invalid runs")
    result = {
        "schema_version": GOAL_CATALOG_SCHEMA_VERSION,
        "goal_slug": goal_slug,
        "runs": runs,
    }
    digest = goal_catalog_page_digest(result)
    if expected_digest is not None and digest != str(expected_digest).lower():
        raise ValueError("goal catalog page content does not match its reference")
    return result


def goal_catalog_generation_digest(document: Mapping[str, Any]) -> str:
    return canonical_json_sha256(dict(document))


def validate_goal_catalog_generation(
    document: Mapping[str, Any],
    *,
    expected_digest: str | None = None,
) -> dict[str, Any]:
    if int(document.get("schema_version") or 0) != GOAL_CATALOG_SCHEMA_VERSION:
        raise ValueError("unsupported goal catalog generation schema")
    goal_slug = str(document.get("goal_slug") or "").strip()
    generated_at = str(document.get("generated_at") or "")
    raw_variants = document.get("variants")
    raw_active = document.get("active_runs")
    raw_terminal = document.get("terminal_runs")
    raw_pages = document.get("archive_pages")
    raw_events = document.get("applied_events")
    if (
        not goal_slug
        or not generated_at
        or not isinstance(raw_variants, list)
        or not isinstance(raw_active, list)
        or not isinstance(raw_terminal, list)
        or not isinstance(raw_pages, list)
        or not isinstance(raw_events, list)
    ):
        raise ValueError("goal catalog generation is malformed")
    variants: list[dict[str, Any]] = []
    variant_ids: set[str] = set()
    for raw in raw_variants:
        if not isinstance(raw, Mapping):
            raise ValueError("goal catalog generation contains an invalid variant")
        descriptor = {
            key: value
            for key, value in raw.items()
            if key
            not in {
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
        validated = validate_goal_variant_descriptor(descriptor)
        variant_id = str(validated["variant_id"])
        if validated["goal_slug"] != goal_slug:
            raise ValueError("goal catalog variant belongs to another goal")
        if variant_id in variant_ids:
            raise ValueError("goal catalog generation contains a duplicate variant")
        resolved_goal = raw.get("resolved_goal")
        if resolved_goal is not None and (
            not isinstance(resolved_goal, Mapping)
            or goal_contract_sha256(resolved_goal)
            != validated["effective_goal_contract_sha256"]
        ):
            raise ValueError("goal catalog variant resolved goal is invalid")
        variant_ids.add(variant_id)
        normalized_variant = deepcopy(dict(raw))
        if "success_badges" in raw:
            normalized_variant["success_badges"] = list(
                _validate_success_badges(raw["success_badges"])
            )
        variants.append(normalized_variant)
    active_runs = [_validate_run(run, goal_slug=goal_slug) for run in raw_active if isinstance(run, Mapping)]
    terminal_runs = [_validate_run(run, goal_slug=goal_slug) for run in raw_terminal if isinstance(run, Mapping)]
    if (
        len(active_runs) != len(raw_active)
        or len(terminal_runs) != len(raw_terminal)
        or len(terminal_runs) > GOAL_CATALOG_HOT_TERMINAL_RUNS
        or any(run["goal_variant_id"] not in variant_ids for run in (*active_runs, *terminal_runs))
    ):
        raise ValueError("goal catalog generation contains invalid hot runs")
    pages: list[dict[str, Any]] = []
    for raw in raw_pages:
        if not isinstance(raw, Mapping):
            raise ValueError("goal catalog generation contains an invalid archive reference")
        digest = str(raw.get("page_sha256") or "").lower()
        if (
            SHA256_PATTERN.fullmatch(digest) is None
            or str(raw.get("page_key") or "") != goal_catalog_page_key(goal_slug, digest)
            or int(raw.get("run_count") or 0) < 1
            or int(raw.get("run_count") or 0) > GOAL_CATALOG_ARCHIVE_PAGE_SIZE
        ):
            raise ValueError("goal catalog archive reference is malformed")
        pages.append(dict(raw))
    applied: list[dict[str, str]] = []
    seen_events: set[str] = set()
    for raw in raw_events:
        if not isinstance(raw, Mapping):
            raise ValueError("goal catalog applied event is malformed")
        event_id = str(raw.get("event_id") or "").lower()
        source_sha256 = str(raw.get("source_sha256") or "").lower()
        if (
            SHA256_PATTERN.fullmatch(event_id) is None
            or SHA256_PATTERN.fullmatch(source_sha256) is None
            or event_id in seen_events
        ):
            raise ValueError("goal catalog applied event is invalid or duplicated")
        seen_events.add(event_id)
        applied.append({"event_id": event_id, "source_sha256": source_sha256})
    result = {
        "schema_version": GOAL_CATALOG_SCHEMA_VERSION,
        "goal_slug": goal_slug,
        "generated_at": generated_at,
        "variants": variants,
        "active_runs": active_runs,
        "terminal_runs": terminal_runs,
        "archive_pages": pages,
        "applied_events": applied,
    }
    digest = goal_catalog_generation_digest(result)
    if expected_digest is not None and digest != str(expected_digest).lower():
        raise ValueError("goal catalog generation content does not match its pointer")
    return result


def validate_goal_catalog_pointer(
    document: Mapping[str, Any],
    *,
    expected_goal_slug: str | None = None,
) -> dict[str, Any]:
    if int(document.get("schema_version") or 0) != GOAL_CATALOG_SCHEMA_VERSION:
        raise ValueError("unsupported goal catalog pointer schema")
    goal_slug = str(document.get("goal_slug") or "").strip()
    digest = str(document.get("generation_sha256") or "").lower()
    key = str(document.get("generation_key") or "")
    generated_at = str(document.get("generated_at") or "")
    if (
        not goal_slug
        or (expected_goal_slug is not None and goal_slug != expected_goal_slug)
        or SHA256_PATTERN.fullmatch(digest) is None
        or key != goal_catalog_generation_key(goal_slug, digest)
        or not generated_at
    ):
        raise ValueError("goal catalog pointer is malformed")
    return {
        "schema_version": GOAL_CATALOG_SCHEMA_VERSION,
        "goal_slug": goal_slug,
        "generation_sha256": digest,
        "generation_key": key,
        "generated_at": generated_at,
    }


def merge_goal_catalog_events(
    events: Sequence[Mapping[str, Any]],
    *,
    goal_slug: str,
    generated_at: str,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Rebuild one complete goal projection deterministically from verified events."""

    normalized = [validate_goal_catalog_event(event) for event in events]
    if any(event["goal_slug"] != goal_slug for event in normalized):
        raise ValueError("goal catalog rebuild received an event from another goal")
    phase_order = {phase: index for index, phase in enumerate(GOAL_CATALOG_PHASES)}
    normalized.sort(
        key=lambda event: (
            str(event["run"].get("attempt_created_at") or event["run"]["created_at"]),
            phase_order[event["phase"]],
            event["created_at"],
            event["event_id"],
        )
    )
    variants: dict[str, dict[str, Any]] = {}
    runs: dict[str, dict[str, Any]] = {}
    applied: dict[str, str] = {}
    for event in normalized:
        descriptor = event["variant"]
        variant_id = str(descriptor["variant_id"])
        existing_variant = variants.get(variant_id)
        if existing_variant is not None:
            existing_descriptor = {
                key: value
                for key, value in existing_variant.items()
                if key
                not in {"first_run_id", "exact_resolution_run_id", "resolved_goal"}
            }
            if goal_variant_catalog_contract(existing_descriptor) != goal_variant_catalog_contract(
                descriptor
            ):
                raise ValueError("goal catalog variant descriptor conflicts across events")
        variant = {**(existing_variant or {}), **deepcopy(descriptor)}
        variant.setdefault("first_run_id", event["run_id"])
        if event.get("resolved_goal") is not None:
            variant["exact_resolution_run_id"] = event["run_id"]
            variant["resolved_goal"] = deepcopy(event["resolved_goal"])
        variants[variant_id] = variant

        candidate = deepcopy(event["run"])
        candidate["event_phase"] = event["phase"]
        existing = runs.get(event["run_id"])
        candidate_order = (
            str(candidate.get("attempt_created_at") or candidate["created_at"]),
            phase_order[event["phase"]],
            event["created_at"],
            event["event_id"],
        )
        existing_order = (
            str((existing or {}).get("attempt_created_at") or (existing or {}).get("created_at") or ""),
            phase_order.get(str((existing or {}).get("event_phase") or "manifest"), 0),
            str((existing or {}).get("event_created_at") or ""),
            str((existing or {}).get("event_id") or ""),
        )
        if existing is None or candidate_order >= existing_order:
            if existing is not None and existing.get("attempt_id") == candidate.get("attempt_id"):
                if event["phase"] == "attempt-terminal" and existing.get(
                    "event_phase"
                ) == "attempt-terminal":
                    candidate["metrics"] = {
                        **dict(existing.get("metrics") or {}),
                        **dict(candidate.get("metrics") or {}),
                    }
                if event["phase"] in {"verified-evaluation", "promotion"}:
                    enrichment = candidate
                    candidate = {**enrichment, **existing}
                    candidate["metrics"] = {
                        **dict(existing.get("metrics") or {}),
                        **dict(enrichment.get("metrics") or {}),
                    }
                    candidate["updated_at"] = max(
                        str(existing.get("updated_at") or ""),
                        str(enrichment.get("updated_at") or ""),
                    )
                    for field in ("evaluation", "promotion"):
                        if field in enrichment:
                            candidate[field] = deepcopy(enrichment[field])
                    if isinstance(enrichment.get("evaluation"), Mapping):
                        evaluation = deepcopy(dict(enrichment["evaluation"]))
                        checkpoint_id = str(evaluation.get("checkpoint_id") or "")
                        evaluations = deepcopy(dict(existing.get("evaluations") or {}))
                        if checkpoint_id:
                            evaluations[checkpoint_id] = evaluation
                        candidate["evaluations"] = evaluations
            candidate["event_created_at"] = event["created_at"]
            candidate["event_id"] = event["event_id"]
            candidate["success_badges"] = list(goal_catalog_run_success_badges(candidate))
            runs[event["run_id"]] = candidate
        applied[event["event_id"]] = event["source_sha256"]

    all_runs = sorted(
        runs.values(),
        key=lambda run: (str(run.get("updated_at") or ""), str(run["run_id"])),
        reverse=True,
    )
    active = [run for run in all_runs if str(run.get("state") or "") not in GOAL_CATALOG_TERMINAL_STATES]
    terminal = [run for run in all_runs if str(run.get("state") or "") in GOAL_CATALOG_TERMINAL_STATES]
    hot_terminal = terminal[:GOAL_CATALOG_HOT_TERMINAL_RUNS]
    archived = terminal[GOAL_CATALOG_HOT_TERMINAL_RUNS:]
    pages: list[dict[str, Any]] = []
    page_refs: list[dict[str, Any]] = []
    for offset in range(0, len(archived), GOAL_CATALOG_ARCHIVE_PAGE_SIZE):
        page = validate_goal_catalog_page(
            {
                "schema_version": GOAL_CATALOG_SCHEMA_VERSION,
                "goal_slug": goal_slug,
                "runs": archived[offset : offset + GOAL_CATALOG_ARCHIVE_PAGE_SIZE],
            }
        )
        digest = goal_catalog_page_digest(page)
        pages.append(page)
        page_refs.append(
            {
                "page_sha256": digest,
                "page_key": goal_catalog_page_key(goal_slug, digest),
                "run_count": len(page["runs"]),
                "newest_updated_at": str(page["runs"][0].get("updated_at") or ""),
                "oldest_updated_at": str(page["runs"][-1].get("updated_at") or ""),
            }
        )

    projected_variants: list[dict[str, Any]] = []
    for variant_id, variant in variants.items():
        variant_runs = [run for run in all_runs if run["goal_variant_id"] == variant_id]
        active_count = sum(
            str(run.get("state") or "") not in GOAL_CATALOG_TERMINAL_STATES
            for run in variant_runs
        )
        projected_variants.append(
            {
                **variant,
                "run_count": len(variant_runs),
                "active_run_count": active_count,
                "terminal_run_count": len(variant_runs) - active_count,
                "first_used_at": min(str(run["created_at"]) for run in variant_runs),
                "last_activity_at": max(str(run["updated_at"]) for run in variant_runs),
                "success_badges": [
                    badge
                    for badge in GOAL_CATALOG_SUCCESS_BADGES
                    if any(
                        badge in goal_catalog_run_success_badges(run)
                        for run in variant_runs
                    )
                ],
            }
        )
    projected_variants.sort(
        key=lambda item: (str(item.get("last_activity_at") or ""), str(item["variant_id"])),
        reverse=True,
    )
    generation = validate_goal_catalog_generation(
        {
            "schema_version": GOAL_CATALOG_SCHEMA_VERSION,
            "goal_slug": goal_slug,
            "generated_at": generated_at,
            "variants": projected_variants,
            "active_runs": active,
            "terminal_runs": hot_terminal,
            "archive_pages": page_refs,
            "applied_events": [
                {"event_id": event_id, "source_sha256": applied[event_id]}
                for event_id in sorted(applied)
            ],
        }
    )
    return generation, tuple(pages)


__all__ = [
    "GOAL_CATALOG_ARCHIVE_PAGE_SIZE",
    "GOAL_CATALOG_EVENT_ROOT",
    "GOAL_CATALOG_HOT_TERMINAL_RUNS",
    "GOAL_CATALOG_PHASES",
    "GOAL_CATALOG_ROOT",
    "GOAL_CATALOG_SCHEMA_VERSION",
    "GOAL_CATALOG_SUCCESS_BADGES",
    "GOAL_CATALOG_TERMINAL_STATES",
    "build_goal_catalog_event",
    "goal_catalog_ack_key",
    "goal_catalog_event_key",
    "goal_catalog_event_prefix",
    "goal_catalog_generation_digest",
    "goal_catalog_generation_key",
    "goal_catalog_page_digest",
    "goal_catalog_page_key",
    "goal_catalog_pointer_key",
    "goal_catalog_run_success_badges",
    "goal_catalog_scope",
    "merge_goal_catalog_events",
    "validate_goal_catalog_event",
    "validate_goal_catalog_generation",
    "validate_goal_catalog_page",
    "validate_goal_catalog_pointer",
]
