from __future__ import annotations

import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

from gradlab.goal_catalog import (
    GOAL_CATALOG_ARCHIVE_PAGE_SIZE,
    GOAL_CATALOG_HOT_TERMINAL_RUNS,
    build_goal_catalog_event,
    goal_catalog_ack_key,
    goal_catalog_event_key,
    goal_catalog_pointer_key,
    merge_goal_catalog_events,
)
from gradlab.goal_catalog_projection import GoalCatalogProjector
from gradlab.goal_variants import build_goal_variant_descriptor
from gradlab.r2_store import BucketConfig, R2Bucket


def _timestamp(index: int) -> str:
    return (datetime(2026, 8, 1, tzinfo=UTC) + timedelta(seconds=index)).isoformat().replace(
        "+00:00", "Z"
    )


def _descriptor(goal_slug: str) -> dict:
    goal = {"goal_id": goal_slug.rsplit("/", 1)[-1]}
    return build_goal_variant_descriptor(
        goal_slug=goal_slug,
        source_sha="a" * 40,
        authored_goal=goal,
        effective_goal=goal,
    )


def _run(
    *,
    goal_slug: str,
    descriptor: dict,
    run_index: int,
    attempt_index: int = 0,
    state: str = "running",
    updated_index: int | None = None,
) -> dict:
    created_at = _timestamp(run_index * 10 + attempt_index)
    return {
        "run_id": f"gradlab-{run_index:032x}",
        "attempt_id": f"attempt-{attempt_index + 1:016x}",
        "attempt_created_at": created_at,
        "name": f"Run {run_index}",
        "state": state,
        "goal_slug": goal_slug,
        "recipe_slug": "ppo",
        "recipe_sha256": "b" * 64,
        "recipe_overrides": [],
        "recipe_variant_id": "base",
        "goal_contract_sha256": descriptor["goal_contract_sha256"],
        "effective_goal_contract_sha256": descriptor[
            "effective_goal_contract_sha256"
        ],
        "goal_variant_id": descriptor["variant_id"],
        "goal_variant_label": descriptor["label"],
        "description": "catalog projection test",
        "seed": run_index,
        "created_at": _timestamp(run_index * 10),
        "updated_at": _timestamp(updated_index if updated_index is not None else run_index * 10),
        "url": "",
        "metrics": {"train/global_step": float(run_index)},
        "stop_reason": "completed" if state != "running" else "",
        "final_step": run_index if state != "running" else None,
        "early_stop": None,
    }


def _event(
    *,
    goal_slug: str,
    descriptor: dict,
    run: dict,
    phase: str,
    source_key: str,
    source_document: dict,
) -> dict:
    return build_goal_catalog_event(
        phase=phase,
        goal_slug=goal_slug,
        run_id=run["run_id"],
        attempt_id=run["attempt_id"],
        source_bucket="control",
        source_key=source_key,
        source_document=source_document,
        created_at=run["updated_at"],
        variant=descriptor,
        run=run,
    )


def _projector(root: Path) -> GoalCatalogProjector:
    return GoalCatalogProjector(
        control=R2Bucket(BucketConfig((root / "control").resolve().as_uri())),
        evaluation=R2Bucket(BucketConfig((root / "evaluation").resolve().as_uri())),
    )


def test_event_is_idempotent_and_ack_follows_verified_readback() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        projector = _projector(Path(temporary))
        goal_slug = "Mario/Level1-1"
        descriptor = _descriptor(goal_slug)
        run = _run(goal_slug=goal_slug, descriptor=descriptor, run_index=1)
        source = {"kind": "manifest", "run_id": run["run_id"]}
        key = f"runs/{run['run_id']}/attempts/{run['attempt_id']}/manifest.json"
        event = _event(
            goal_slug=goal_slug,
            descriptor=descriptor,
            run=run,
            phase="manifest",
            source_key=key,
            source_document=source,
        )

        first = projector.put_event(event)
        second = projector.put_event(event)
        assert first == second
        orphan = projector.reconcile(goal_slug)
        assert orphan.applied_event_count == 0
        assert len(orphan.orphan_events) == 1
        assert projector.control.get_json_optional(goal_catalog_ack_key(event["event_id"])) is None

        projector.control.put_json(key, source, create_only=True)
        repaired = projector.reconcile(goal_slug)
        assert repaired.applied_event_count == 1
        assert repaired.acknowledged_event_count == 1
        generation = projector.generation(goal_slug)
        assert generation is not None
        assert generation["active_runs"][0]["run_id"] == run["run_id"]
        assert projector.control.get_json(goal_catalog_event_key(event)) == event
        assert projector.control.get_json_optional(goal_catalog_ack_key(event["event_id"]))


def test_late_old_attempt_cannot_regress_a_new_attempt() -> None:
    goal_slug = "Mario/Level1-1"
    descriptor = _descriptor(goal_slug)
    old_terminal = _run(
        goal_slug=goal_slug,
        descriptor=descriptor,
        run_index=2,
        attempt_index=0,
        state="failed",
        updated_index=500,
    )
    new_manifest = _run(
        goal_slug=goal_slug,
        descriptor=descriptor,
        run_index=2,
        attempt_index=1,
        state="running",
        updated_index=100,
    )
    old_source = {"kind": "terminal", "attempt": "old"}
    new_source = {"kind": "manifest", "attempt": "new"}
    events = [
        _event(
            goal_slug=goal_slug,
            descriptor=descriptor,
            run=old_terminal,
            phase="attempt-terminal",
            source_key="old-terminal.json",
            source_document=old_source,
        ),
        _event(
            goal_slug=goal_slug,
            descriptor=descriptor,
            run=new_manifest,
            phase="manifest",
            source_key="new-manifest.json",
            source_document=new_source,
        ),
    ]

    generation, _pages = merge_goal_catalog_events(
        list(reversed(events)),
        goal_slug=goal_slug,
        generated_at=_timestamp(999),
    )

    assert generation["terminal_runs"] == []
    assert generation["active_runs"][0]["attempt_id"] == new_manifest["attempt_id"]
    assert generation["active_runs"][0]["state"] == "running"


def test_every_supervisor_terminal_outcome_leaves_the_active_set() -> None:
    goal_slug = "Mario/Level1-1"
    descriptor = _descriptor(goal_slug)
    events = []
    terminal_states = (
        "succeeded",
        "failed",
        "stopped",
        "canceled",
        "interrupted",
        "resumable_failure",
    )
    for index, state in enumerate(terminal_states, start=1):
        run = _run(
            goal_slug=goal_slug,
            descriptor=descriptor,
            run_index=index,
            state=state,
        )
        source = {"run": run["run_id"], "state": state}
        events.append(
            _event(
                goal_slug=goal_slug,
                descriptor=descriptor,
                run=run,
                phase="attempt-terminal",
                source_key=f"runs/{run['run_id']}/terminal.json",
                source_document=source,
            )
        )

    generation, _pages = merge_goal_catalog_events(
        events,
        goal_slug=goal_slug,
        generated_at=_timestamp(999),
    )

    assert generation["active_runs"] == []
    assert {run["state"] for run in generation["terminal_runs"]} == set(terminal_states)


def test_hot_generation_and_archive_pages_are_bounded() -> None:
    goal_slug = "Mario/Level1-1"
    descriptor = _descriptor(goal_slug)
    events = []
    for index in range(1, 452):
        run = _run(
            goal_slug=goal_slug,
            descriptor=descriptor,
            run_index=index,
            state="succeeded",
        )
        source = {"run": run["run_id"]}
        events.append(
            _event(
                goal_slug=goal_slug,
                descriptor=descriptor,
                run=run,
                phase="attempt-terminal",
                source_key=f"runs/{run['run_id']}/terminal.json",
                source_document=source,
            )
        )

    generation, pages = merge_goal_catalog_events(
        events,
        goal_slug=goal_slug,
        generated_at=_timestamp(9999),
    )

    assert len(generation["terminal_runs"]) == GOAL_CATALOG_HOT_TERMINAL_RUNS
    assert [len(page["runs"]) for page in pages] == [GOAL_CATALOG_ARCHIVE_PAGE_SIZE, 1]
    assert generation["variants"][0]["run_count"] == 451


def test_concurrent_same_goal_reconciliation_converges_without_lost_events() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        first = _projector(root)
        second = _projector(root)
        goal_slug = "Mario/Level1-1"
        descriptor = _descriptor(goal_slug)
        for index, projector in ((1, first), (2, second)):
            run = _run(goal_slug=goal_slug, descriptor=descriptor, run_index=index)
            source = {"run": run["run_id"]}
            key = f"runs/{run['run_id']}/manifest.json"
            projector.control.put_json(key, source, create_only=True)
            projector.put_event(
                _event(
                    goal_slug=goal_slug,
                    descriptor=descriptor,
                    run=run,
                    phase="manifest",
                    source_key=key,
                    source_document=source,
                )
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda item: item.reconcile(goal_slug), (first, second)))

        assert all(result.applied_event_count == 2 for result in results)
        generation = first.generation(goal_slug)
        assert generation is not None
        assert {run["run_id"] for run in generation["active_runs"]} == {
            f"gradlab-{1:032x}",
            f"gradlab-{2:032x}",
        }
        assert first.control.get_json_optional(goal_catalog_pointer_key(goal_slug)) is not None


def test_concurrent_different_goal_reconciliation_uses_independent_pointers() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        projector = _projector(root)
        goals = ("Mario/Level1-1", "Mario/Level1-2")
        for index, goal_slug in enumerate(goals, start=1):
            descriptor = _descriptor(goal_slug)
            run = _run(goal_slug=goal_slug, descriptor=descriptor, run_index=index)
            source = {"run": run["run_id"], "goal_slug": goal_slug}
            key = f"runs/{run['run_id']}/manifest.json"
            projector.control.put_json(key, source, create_only=True)
            projector.put_event(
                _event(
                    goal_slug=goal_slug,
                    descriptor=descriptor,
                    run=run,
                    phase="manifest",
                    source_key=key,
                    source_document=source,
                )
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(projector.reconcile, goals))

        assert {result.goal_slug for result in results} == set(goals)
        assert goal_catalog_pointer_key(goals[0]) != goal_catalog_pointer_key(goals[1])
        for index, goal_slug in enumerate(goals, start=1):
            generation = projector.generation(goal_slug)
            assert generation is not None
            assert [run["run_id"] for run in generation["active_runs"]] == [
                f"gradlab-{index:032x}"
            ]
