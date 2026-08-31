from __future__ import annotations

from gradlab.modal_app_lifecycle import plan_modal_app_retirements


def test_retirement_plan_frees_reserved_capacity_without_touching_live_or_foreign_apps() -> None:
    apps = {
        "main": [
            {
                "app_id": "foreign",
                "description": "unrelated-service",
                "state": "deployed",
                "tasks": "0",
                "created_at": "2026-01-01T00:00:00Z",
            },
            {
                "app_id": "legacy-main",
                "description": "rlab-eval-aaaaaaaaaaaa",
                "state": "deployed",
                "tasks": "0",
                "created_at": "2026-01-02T00:00:00Z",
            },
        ],
        "rlab-eval": [
            {
                "app_id": "legacy",
                "description": "rlab-eval-v2-bbbbbbbbbbbb",
                "state": "deployed",
                "tasks": "0",
                "created_at": "2026-01-03T00:00:00Z",
            }
        ],
        "gradlab-eval": [
            {
                "app_id": "protected",
                "description": "gradlab-eval-v3-cccccccccccc",
                "state": "deployed",
                "tasks": "0",
                "created_at": "2026-01-04T00:00:00Z",
            },
            {
                "app_id": "busy",
                "description": "gradlab-eval-v3-dddddddddddd",
                "state": "deployed",
                "tasks": "1",
                "created_at": "2026-01-05T00:00:00Z",
            },
            {
                "app_id": "idle-old",
                "description": "gradlab-eval-v3-eeeeeeeeeeee",
                "state": "deployed",
                "tasks": "0",
                "created_at": "2026-01-06T00:00:00Z",
            },
            {
                "app_id": "target",
                "description": "gradlab-eval-v3-ffffffffffff",
                "state": "deployed",
                "tasks": "0",
                "created_at": "2026-01-07T00:00:00Z",
            },
        ],
    }

    retirements = plan_modal_app_retirements(
        apps,
        protected_app_names={"gradlab-eval-v3-cccccccccccc"},
        target_app_name="gradlab-eval-v3-ffffffffffff",
        workspace_limit=8,
        reserve=3,
    )

    assert [(item.environment_name, item.app_id) for item in retirements] == [
        ("main", "legacy-main"),
        ("rlab-eval", "legacy"),
    ]


def test_retirement_plan_fails_closed_when_owned_idle_apps_cannot_make_room() -> None:
    apps = {
        "main": [
            {
                "app_id": "foreign",
                "description": "unrelated-service",
                "state": "deployed",
                "tasks": "0",
                "created_at": "2026-01-01T00:00:00Z",
            }
        ],
        "gradlab-eval": [
            {
                "app_id": "protected",
                "description": "gradlab-eval-v3-aaaaaaaaaaaa",
                "state": "deployed",
                "tasks": "0",
                "created_at": "2026-01-02T00:00:00Z",
            }
        ],
    }

    try:
        plan_modal_app_retirements(
            apps,
            protected_app_names={"gradlab-eval-v3-aaaaaaaaaaaa"},
            target_app_name="gradlab-eval-v3-bbbbbbbbbbbb",
            workspace_limit=3,
            reserve=1,
        )
    except RuntimeError as exc:
        assert "cannot free" in str(exc)
    else:
        raise AssertionError("expected retirement planning to fail closed")
