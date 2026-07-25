from __future__ import annotations

from types import SimpleNamespace

import pytest

from rlab.play import build_parser as build_play_parser
from rlab.play_catalog import PlayCatalog, parse_wandb_location
from rlab.run_contracts import checkpoint_id


RUN_ID = "rlab-" + "a" * 32


class FakeApi:
    default_entity = "research"

    def projects(self, *, entity: str, per_page: int):
        assert (entity, per_page) == ("research", 200)
        return [
            SimpleNamespace(
                entity=entity,
                name="Mario",
                created_at="2026-01-01T00:00:00Z",
                url="https://wandb.ai/research/Mario",
            ),
            SimpleNamespace(
                entity=entity,
                name="Breakout",
                created_at="2026-01-02T00:00:00Z",
                url="https://wandb.ai/research/Breakout",
            ),
        ]

    def runs(self, path: str, **kwargs):
        assert path == "research/Mario"
        assert kwargs["order"] == "-created_at"
        return [
            SimpleNamespace(
                id="legacy-run",
                name="legacy",
                state="finished",
                config={},
                notes="",
                created_at="2026-01-01T00:00:00Z",
                updated_at="2026-01-01T00:00:00Z",
                url="https://wandb.ai/research/Mario/runs/legacy-run",
            ),
            SimpleNamespace(
                id=RUN_ID,
                name="Level 1-1 seed 3",
                state="finished",
                config={
                    "goal_slug": "Mario/Level1-1",
                    "recipe_slug": "ppo",
                    "seed": 3,
                },
                notes="accepted",
                created_at="2026-01-02T00:00:00Z",
                updated_at="2026-01-03T00:00:00Z",
                url=f"https://wandb.ai/research/Mario/runs/{RUN_ID}",
            ),
        ]


def checkpoint_row(*, step: int, digest: str, purpose: str) -> dict[str, object]:
    identifier = checkpoint_id(step=step, sha256=digest)
    root = f"https://models.example/runs/{RUN_ID}/checkpoints/{identifier}"
    return {
        "run_id": RUN_ID,
        "checkpoint_id": identifier,
        "step": step,
        "purpose": purpose,
        "sha256": digest,
        "size_bytes": 1024,
        "public_url": f"{root}/model.zip",
        "model_document_url": f"{root}/model.json",
        "model_document_sha256": "b" * 64,
        "recipe_document_url": f"{root}/recipe.yaml",
        "recipe_document_sha256": "c" * 64,
        "goal_sha256": "d" * 64,
        "recipe_sha256": "e" * 64,
        "environment_sha256": "f" * 64,
        "evaluation_contract_sha256": "1" * 64,
        "recovery_sidecar_key": "recovery/key",
        "created_at": "2026-01-03T00:00:00Z",
        "schema_version": 1,
    }


def test_wandb_urls_preselect_projects_runs_or_checkpoints() -> None:
    location = parse_wandb_location("https://wandb.ai/research/Mario?nw=user")

    assert location is not None
    assert (location.entity, location.project, location.run_id) == (
        "research",
        "Mario",
        None,
    )


def test_parse_wandb_location_ignores_query_and_returns_run() -> None:
    location = parse_wandb_location(
        f"https://wandb.ai/research/Mario/runs/{RUN_ID}?nw=user"
    )

    assert location is not None
    assert (location.entity, location.project, location.run_id) == (
        "research",
        "Mario",
        RUN_ID,
    )


def test_play_parser_allows_bare_launch_and_wandb_preselection() -> None:
    parser = build_play_parser()

    assert parser.parse_args([]).artifact_ref is None
    assert (
        parser.parse_args(["https://wandb.ai/research/Mario"]).artifact_ref
        == "https://wandb.ai/research/Mario"
    )


def test_catalog_searches_projects_and_only_returns_canonical_rlab_runs() -> None:
    catalog = PlayCatalog()
    catalog._api = FakeApi()

    projects = catalog.projects(entity="research", query="mario")
    runs = catalog.runs(entity="research", project="Mario", query="seed 3")

    assert [item["name"] for item in projects.items] == ["Mario"]
    assert [item["run_id"] for item in runs.items] == [RUN_ID]
    assert runs.items[0]["recipe"] == "ppo"


def test_catalog_validates_and_orders_public_checkpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    periodic = checkpoint_row(step=250_000, digest="2" * 64, purpose="periodic")
    final = checkpoint_row(step=500_000, digest="3" * 64, purpose="final")
    monkeypatch.setattr(
        "rlab.play_catalog._public_json",
        lambda _url: {
            "schema_version": 1,
            "run_id": RUN_ID,
            "checkpoints": [final, periodic],
            "promotion": {"checkpoint_id": periodic["checkpoint_id"]},
        },
    )
    catalog = PlayCatalog(public_models_base_url="https://models.example")

    rows = catalog.checkpoints(run_id=RUN_ID)

    assert [row["checkpoint_id"] for row in rows] == [
        periodic["checkpoint_id"],
        final["checkpoint_id"],
    ]
    assert rows[0]["promoted"] is True
    assert rows[0]["manifest_url"].endswith("/manifest.json")
