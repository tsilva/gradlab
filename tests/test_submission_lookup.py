from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from gradlab.experiment_cli import _submission_status, build_parser
from gradlab.run_contracts import new_attempt_id, new_run_id
from tests import test_run_authority as fixtures


@pytest.fixture
def case():
    fixture = fixtures.RunAuthorityTests()
    fixture.setUp()
    try:
        yield fixture
    finally:
        fixture.tearDown()


def lookup(case, **kwargs):
    return _submission_status(
        case.authority,
        submission_key="intent-1",
        goal_slug="SuperMarioBros-Nes-v0/Level1-1",
        created_since="2026-01-01T00:00:00Z",
        **kwargs,
    )


def manifest(case, key="intent-1"):
    value = case.manifest(new_run_id(), new_attempt_id())
    return replace(value, compute={**value.compute, "submission_key": key})


def test_lookup_finds_committed_manifest_without_mutation(case):
    value = manifest(case)
    case.authority.create_manifest(value)
    before = sorted(case.authority.control.iter_keys(""))
    result = lookup(case)
    assert result["status"] == "found"
    assert [row["run_id"] for row in result["matches"]] == [value.run_id]
    assert result["complete_search"] and not result["mutations_performed"]
    assert sorted(case.authority.control.iter_keys("")) == before


def test_lookup_distinguishes_absence_unrelated_key_and_multiple_matches(case):
    assert lookup(case)["status"] == "not_admitted"
    case.authority.create_manifest(manifest(case, "another-intent"))
    assert lookup(case)["status"] == "not_admitted"
    for _ in range(2):
        case.authority.create_manifest(manifest(case))
    assert len(lookup(case)["matches"]) == 2


def test_partial_admission_event_never_proves_absence(case):
    value = manifest(case)
    case.authority._put_goal_catalog_event(case.authority._goal_catalog_event_for_manifest(value))
    result = lookup(case)
    assert result["status"] == "unresolved"
    assert result["unresolved_candidates"][0]["run_id"] == value.run_id


def test_lookup_ignores_older_events_but_rejects_future_or_naive_cutoff(case):
    value = replace(manifest(case), created_at="2025-12-01T00:00:00Z")
    case.authority.create_manifest(value)
    assert lookup(case)["status"] == "not_admitted"
    for cutoff in ["2026-01-01", (datetime.now(UTC) + timedelta(days=1)).isoformat()]:
        with pytest.raises(ValueError):
            _submission_status(
                case.authority,
                submission_key="intent-1",
                goal_slug=value.goal_slug,
                created_since=cutoff,
            )


def test_lookup_inventory_limit_fails_without_partial_absence(case, monkeypatch):
    monkeypatch.setattr(case.authority.control, "iter_keys", lambda _: iter(["other"] * 10001))
    with pytest.raises(RuntimeError, match="bounded"):
        lookup(case)


def test_status_parser_keeps_run_and_submission_selectors_exclusive():
    parser = build_parser()
    args = parser.parse_args(
        [
            "status",
            "--submission-key",
            "intent-1",
            "--goal-slug",
            "SuperMarioBros-Nes-v0/Level1-1",
            "--created-since",
            "2026-01-01T00:00:00Z",
            "--json",
        ]
    )
    assert args.submission_key == "intent-1" and args.run_id is None
    with pytest.raises(SystemExit):
        parser.parse_args(["status", "--run", new_run_id(), "--submission-key", "intent-1"])
