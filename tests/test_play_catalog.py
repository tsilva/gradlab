from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from rlab.play import build_parser as build_play_parser
from rlab.play_catalog import PlayCatalog, parse_wandb_location
from rlab.run_contracts import checkpoint_id


RUN_ID = "rlab-" + "a" * 32


class FakeApi:
    default_entity = "research"

    def __init__(self) -> None:
        self.runs_calls = 0

    def projects(self, *, entity: str, per_page: int):
        raise AssertionError(
            f"repository-backed projects must not query W&B: {entity=}, {per_page=}"
        )

    def runs(self, path: str, **kwargs):
        self.runs_calls += 1
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
                summary={
                    "train/outcome/success/window_100/rate/min": 0.75,
                    "train/episode/return/shaped/from/target/mean": 123.5,
                },
                notes="accepted",
                created_at="2026-01-02T00:00:00Z",
                updated_at="2026-01-03T00:00:00Z",
                url=f"https://wandb.ai/research/Mario/runs/{RUN_ID}",
            ),
        ]

    def run(self, path: str):
        assert path == f"research/Mario/{RUN_ID}"
        return self.runs("research/Mario", order="-created_at", per_page=200, lazy=True)[1]


def write_goal_catalog(repo_root: Path) -> None:
    goal_root = repo_root / "experiments" / "goals" / "Mario" / "Level1-1"
    recipes = goal_root / "recipes"
    recipes.mkdir(parents=True)
    (goal_root / "_goal.yaml").write_text(
        "\n".join(
            (
                "defaults:",
                "- _self_",
                "goal_id: Level1-1",
                "title: Mario Level 1-1 completion",
                "objective:",
                "  rank:",
                "  - min(leader/checkpoint/step)",
                "  - max(eval/full/episode/return/mean)",
                "train:",
                "  environment:",
                "    env_provider: gymnasium",
                "    env_config:",
                "      game: Mario",
                "",
            )
        ),
        encoding="utf-8",
    )
    (recipes / "ppo.yaml").write_text("defaults: [_self_]\n", encoding="utf-8")


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


def test_catalog_default_entity_does_not_initialize_wandb(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    catalog = PlayCatalog(repo_root=tmp_path)
    monkeypatch.setenv("WANDB_ENTITY", "research")
    monkeypatch.setattr(
        catalog,
        "_wandb_api",
        lambda: (_ for _ in ()).throw(AssertionError("must not initialize W&B")),
    )

    assert catalog.default_entity() == "research"


def test_catalog_uses_repository_projects_and_goals_before_querying_wandb(
    tmp_path: Path,
) -> None:
    write_goal_catalog(tmp_path)
    catalog = PlayCatalog(repo_root=tmp_path)
    api = FakeApi()
    catalog._api = api

    projects = catalog.projects(entity="research", query="mario")
    goals = catalog.goals(entity="research", project="Mario")
    assert api.runs_calls == 0

    runs = catalog.runs(
        entity="research",
        project="Mario",
        goal_id="Level1-1",
        query="seed 3",
    )

    assert [item["name"] for item in projects.items] == ["Mario"]
    assert projects.items[0]["goal_count"] == 1
    assert [item["goal_id"] for item in goals.items] == ["Level1-1"]
    assert goals.items[0]["title"] == "Mario Level 1-1 completion"
    assert goals.items[0]["recipe_count"] == 1
    assert goals.items[0]["goal_path"].endswith("/Level1-1/_goal.yaml")
    assert [item["run_id"] for item in runs.items] == [RUN_ID]
    assert runs.items[0]["recipe"] == "ppo"
    assert runs.metric_columns == (
        {
            "metric": "train/outcome/success/window_100/rate/min",
            "direction": "max",
        },
        {
            "metric": "train/episode/return/shaped/from/target/mean",
            "direction": "max",
        },
    )
    assert runs.items[0]["metrics"] == {
        "train/outcome/success/window_100/rate/min": 0.75,
        "train/episode/return/shaped/from/target/mean": 123.5,
    }
    assert catalog.run_goal(entity="research", project="Mario", run_id=RUN_ID) == "Level1-1"


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


def test_catalog_attaches_goal_required_eval_results_by_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    required_metric = "eval/full/outcome/success/rate/min"

    class EvalRun:
        config = {
            "checkpoint_eval_contract": {
                "acceptance": [
                    {
                        "metric": required_metric,
                        "operator": ">=",
                        "threshold": 1.0,
                    }
                ]
            }
        }

        def scan_history(self, *, keys, page_size):
            assert required_metric in keys
            assert page_size == 10_000
            return [
                {
                    "eval/checkpoint_step": 250_000,
                    "eval/acceptance/pass": 1.0,
                    "eval/acceptance/episodes/planned": 100.0,
                    "eval/acceptance/episodes/completed": 100.0,
                    "eval/acceptance/failure/count": 0.0,
                    required_metric: 1.0,
                },
                {
                    "eval/checkpoint_step": 500_000,
                    "eval/acceptance/pass": 0.0,
                    "eval/acceptance/episodes/planned": 100.0,
                    "eval/acceptance/episodes/completed": 1.0,
                    "eval/acceptance/failure/count": 1.0,
                },
            ]

    class EvalApi:
        @staticmethod
        def run(path):
            assert path == f"research/Mario/{RUN_ID}"
            return EvalRun()

    catalog = PlayCatalog(public_models_base_url="https://models.example")
    catalog._api = EvalApi()

    rows = catalog.checkpoints(
        entity="research",
        project="Mario",
        run_id=RUN_ID,
    )

    accepted = rows[0]["evaluation"]
    assert accepted["status"] == "accepted"
    assert accepted["episodes_completed"] == 100
    assert accepted["criteria"] == [
        {
            "metric": required_metric,
            "operator": ">=",
            "threshold": 1.0,
            "value": 1.0,
            "passed": True,
        }
    ]
    rejected = rows[1]["evaluation"]
    assert rejected["status"] == "rejected"
    assert rejected["episodes_completed"] == 1
    assert rejected["failure_count"] == 1
    assert rejected["criteria"][0]["value"] is None
