from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gradlab.play import build_parser as build_play_parser
from gradlab.play_catalog import PlayCatalog, parse_wandb_location
from gradlab.goal_variants import (
    build_goal_variant_descriptor,
    goal_variant_scope_key,
)
from gradlab.recipe_documents import load_goal_contract
from gradlab.recipe_variants import recipe_variant_id
from gradlab.reward_programs import goal_for_contract_validation
from gradlab.run_contracts import checkpoint_id


RUN_ID = "gradlab-" + "a" * 32
SECOND_RUN_ID = "gradlab-" + "b" * 32


class FakeApi:
    default_entity = "research"

    def __init__(self) -> None:
        self.runs_calls = 0
        self.runs_filters: list[object] = []

    def projects(self, *, entity: str, per_page: int):
        raise AssertionError(
            f"repository-backed projects must not query W&B: {entity=}, {per_page=}"
        )

    def runs(self, path: str, **kwargs):
        self.runs_calls += 1
        self.runs_filters.append(kwargs.get("filters"))
        assert path == "research/Mario"
        assert kwargs["order"] == "-created_at"
        assert kwargs["per_page"] == 200
        assert kwargs["lazy"] is False
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
                    "recipe_sha256": "f" * 64,
                    "recipe_overrides": [
                        "train.backend.config.learning_rate=0.0002",
                    ],
                    "source_sha": "c" * 40,
                    "seed": 3,
                },
                summary={
                    "leader/checkpoint/step": 1_500_000,
                    "eval/full/episode/return/mean": 321.25,
                    "train/outcome/success/window_100/rate/min": 0.75,
                    "train/episode/return/shaped/from/target/mean": 123.5,
                    "train/global_step": 2_000_000,
                },
                notes="accepted",
                created_at="2026-01-02T00:00:00Z",
                updated_at="2026-01-03T00:00:00Z",
                url=f"https://wandb.ai/research/Mario/runs/{RUN_ID}",
            ),
        ]

    def run(self, path: str):
        assert path == f"research/Mario/{RUN_ID}"
        return self.runs("research/Mario", order="-created_at", per_page=200, lazy=False)[1]


def wandb_catalog_node(
    *,
    run_id: str,
    leader_step: int,
    created_at: str,
) -> dict[str, Any]:
    return {
        "name": run_id,
        "displayName": f"Run {run_id[-1]}",
        "state": "finished",
        "config": json.dumps(
            {
                "goal_slug": {"value": "Mario/Level1-1"},
                "recipe_slug": {"value": "ppo"},
                "recipe_sha256": {"value": "f" * 64},
                "run_description": {"value": f"description {run_id[-1]}"},
                "seed": {"value": 3},
                "_wandb": {"value": {"ignored": True}},
            }
        ),
        "createdAt": created_at,
        "notes": "",
        "summaryMetrics": json.dumps(
            {
                "leader/checkpoint/step": leader_step,
                "eval/full/episode/return/mean": 321.25,
            }
        ),
    }


class FakeCatalogGraphQLClient:
    app_url = "https://wandb.example/"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def execute(self, query: object, *, variable_values: dict[str, Any]):
        del query
        self.calls.append(variable_values)
        cursor = variable_values["cursor"]
        if cursor is None:
            edges = [
                {
                    "node": wandb_catalog_node(
                        run_id=RUN_ID,
                        leader_step=1_500_000,
                        created_at="2026-01-02T00:00:00Z",
                    )
                }
            ]
            page_info = {"endCursor": "page-2", "hasNextPage": True}
        else:
            assert cursor == "page-2"
            edges = [
                {
                    "node": wandb_catalog_node(
                        run_id=SECOND_RUN_ID,
                        leader_step=1_000_000,
                        created_at="2026-01-01T00:00:00Z",
                    )
                }
            ]
            page_info = {"endCursor": "done", "hasNextPage": False}
        return {
            "project": {
                "runs": {
                    "edges": edges,
                    "pageInfo": page_info,
                }
            }
        }


class FakeCatalogGraphQLApi:
    def __init__(self) -> None:
        self.client = FakeCatalogGraphQLClient()

    def runs(self, *args: object, **kwargs: object):
        raise AssertionError(f"catalog must use the bounded GraphQL projection: {args=} {kwargs=}")


class NoCallCatalogGraphQLClient:
    app_url = "https://wandb.example/"

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, query: object, *, variable_values: dict[str, Any]):
        self.calls += 1
        raise AssertionError(
            f"fresh persistent catalog must avoid W&B: {query=} {variable_values=}"
        )


class NoCallCatalogGraphQLApi:
    def __init__(self) -> None:
        self.client = NoCallCatalogGraphQLClient()


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
    (repo_root / "experiments" / "goals" / "_catalog.yaml").write_text(
        "\n".join(
            (
                "schema_version: 1",
                "namespaces:",
                "  Mario:",
                "    project: Mario",
                "",
            )
        ),
        encoding="utf-8",
    )


def write_indexed_goal_catalog(repo_root: Path) -> None:
    goals_root = repo_root / "experiments" / "goals"
    for namespace, goal_id, title in (
        ("Mario", "Level1-1", "Mario Level 1-1 completion"),
        ("Atari", "Breakout", "Atari Breakout completion"),
    ):
        goal_root = goals_root / namespace / goal_id
        recipes = goal_root / "recipes"
        recipes.mkdir(parents=True)
        (goal_root / "_goal.yaml").write_text(
            "\n".join(
                (
                    "defaults:",
                    "- _self_",
                    f"goal_id: {goal_id}",
                    f"title: {title}",
                    "objective:",
                    "  rank:",
                    "  - min(leader/checkpoint/step)",
                    "train:",
                    "  environment:",
                    "    env_provider: gymnasium",
                    "    env_config:",
                    f"      game: {namespace}",
                    "",
                )
            ),
            encoding="utf-8",
        )
        (recipes / "ppo.yaml").write_text("defaults: [_self_]\n", encoding="utf-8")
    (goals_root / "_catalog.yaml").write_text(
        "\n".join(
            (
                "schema_version: 1",
                "namespaces:",
                "  Atari:",
                "    project: Atari",
                "  Mario:",
                "    project: Mario",
                "",
            )
        ),
        encoding="utf-8",
    )


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
    location = parse_wandb_location(f"https://wandb.ai/research/Mario/runs/{RUN_ID}?nw=user")

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


def test_goal_variants_use_one_private_index_read_without_wandb(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    write_goal_catalog(tmp_path)
    goal_path = tmp_path / "experiments" / "goals" / "Mario" / "Level1-1" / "_goal.yaml"
    authored = load_goal_contract(goal_path, tmp_path, validate=False)
    descriptor = build_goal_variant_descriptor(
        goal_slug="Mario/Level1-1",
        source_sha="a" * 40,
        authored_goal=authored,
        effective_goal=goal_for_contract_validation(
            authored,
            label="test goal",
        ),
    )
    scope = goal_variant_scope_key(
        entity="research",
        project="Mario",
        goal_slug="Mario/Level1-1",
    )

    class OneReadControlBucket:
        calls: list[str] = []

        def get_json_optional(self, key: str):
            self.calls.append(key)
            assert key == f"{scope}/index.json"
            return {
                "schema_version": 1,
                "scope": {
                    "entity": "research",
                    "project": "Mario",
                    "goal_slug": "Mario/Level1-1",
                },
                "variants": [
                    {
                        **descriptor,
                        "descriptor_key": (f"{scope}/descriptors/{descriptor['variant_id']}.json"),
                        "first_run_id": RUN_ID,
                    }
                ],
            }

    bucket = OneReadControlBucket()
    catalog = PlayCatalog(repo_root=tmp_path, control_bucket=bucket)
    monkeypatch.setattr(
        catalog,
        "_wandb_api",
        lambda: (_ for _ in ()).throw(AssertionError("private variant index must avoid W&B")),
    )

    page = catalog.goal_variants(
        entity="research",
        project="Mario",
        goal_id="Level1-1",
    )

    assert [item["variant_id"] for item in page.items] == [descriptor["variant_id"]]
    assert bucket.calls == [f"{scope}/index.json"]


def test_run_catalog_uses_lifecycle_owned_variant_index_without_wandb(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    write_goal_catalog(tmp_path)
    goal_path = tmp_path / "experiments" / "goals" / "Mario" / "Level1-1" / "_goal.yaml"
    authored = load_goal_contract(goal_path, tmp_path, validate=False)
    descriptor = build_goal_variant_descriptor(
        goal_slug="Mario/Level1-1",
        source_sha="a" * 40,
        authored_goal=authored,
        effective_goal=goal_for_contract_validation(authored, label="test goal"),
    )
    scope = goal_variant_scope_key(
        entity="research",
        project="Mario",
        goal_slug="Mario/Level1-1",
    )

    class IndexedControlBucket:
        calls: list[str] = []

        def get_json_optional(self, key: str):
            self.calls.append(key)
            if key == f"{scope}/index.json":
                return {
                    "schema_version": 1,
                    "scope": {
                        "entity": "research",
                        "project": "Mario",
                        "goal_slug": "Mario/Level1-1",
                    },
                    "variants": [{**descriptor, "first_run_id": RUN_ID}],
                }
            assert key == f"{scope}/runs/{descriptor['variant_id']}.json"
            return {
                "schema_version": 1,
                "scope": {
                    "entity": "research",
                    "project": "Mario",
                    "goal_slug": "Mario/Level1-1",
                    "variant_id": descriptor["variant_id"],
                },
                "runs": [
                    {
                        "run_id": RUN_ID,
                        "attempt_id": "attempt-" + "b" * 16,
                        "name": "Indexed run",
                        "state": "succeeded",
                        "goal_slug": "Mario/Level1-1",
                        "recipe_slug": "ppo",
                        "recipe_sha256": "f" * 64,
                        "recipe_overrides": [],
                        "recipe_variant_id": "base",
                        "goal_contract_sha256": descriptor["goal_contract_sha256"],
                        "effective_goal_contract_sha256": descriptor[
                            "effective_goal_contract_sha256"
                        ],
                        "goal_variant_id": descriptor["variant_id"],
                        "goal_variant_label": descriptor["label"],
                        "description": "lifecycle projection",
                        "seed": 3,
                        "created_at": "2026-01-02T00:00:00Z",
                        "updated_at": "2026-01-03T00:00:00Z",
                        "url": f"https://wandb.ai/research/Mario/runs/{RUN_ID}",
                        "metrics": {
                            "leader/checkpoint/step": 1_500_000,
                            "eval/full/episode/return/mean": 321.25,
                        },
                    }
                ],
            }

    bucket = IndexedControlBucket()
    catalog = PlayCatalog(repo_root=tmp_path, control_bucket=bucket)
    monkeypatch.setattr(
        catalog,
        "_wandb_api",
        lambda: (_ for _ in ()).throw(
            AssertionError("lifecycle-owned run index must avoid W&B")
        ),
    )

    page = catalog.runs(
        entity="research",
        project="Mario",
        goal_id="Level1-1",
    )

    assert [item["run_id"] for item in page.items] == [RUN_ID]
    assert page.items[0]["description"] == "lifecycle projection"
    assert page.items[0]["metrics"]["leader/checkpoint/step"] == 1_500_000.0
    assert bucket.calls == [
        f"{scope}/index.json",
        f"{scope}/runs/{descriptor['variant_id']}.json",
    ]


def test_catalog_default_entity_loads_operator_configuration_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = 0

    def load_environment() -> None:
        nonlocal calls
        calls += 1

    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.setattr("gradlab.play_catalog.load_wandb_env", load_environment)
    monkeypatch.setattr("gradlab.play_catalog.wandb_entity_from_env", lambda: "research")
    catalog = PlayCatalog(repo_root=tmp_path)

    assert catalog.default_entity() == "research"
    assert catalog.default_entity() == "research"
    assert calls == 1


def test_repository_catalog_requires_explicit_namespace_index(tmp_path: Path) -> None:
    goal_root = tmp_path / "experiments" / "goals" / "Mario" / "Level1-1"
    goal_root.mkdir(parents=True)
    (goal_root / "_goal.yaml").write_text("goal_id: Level1-1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="repository goal catalog does not exist"):
        PlayCatalog(repo_root=tmp_path).projects(entity="research")


def test_indexed_project_listing_does_not_parse_goal_contracts_and_scopes_goal_reads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    write_indexed_goal_catalog(tmp_path)
    from gradlab import play_catalog

    loaded_paths: list[Path] = []
    real_loader = play_catalog.load_mapping_document

    def tracked_loader(path: Path, *, label: str | None = None) -> dict[str, Any]:
        loaded_paths.append(path)
        return real_loader(path, label=label)

    monkeypatch.setattr(play_catalog, "load_mapping_document", tracked_loader)
    monkeypatch.setattr(
        play_catalog,
        "load_goal_contract",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("browse-only catalog must not compose goal contracts")
        ),
    )
    catalog = PlayCatalog(repo_root=tmp_path)

    projects = catalog.projects(entity="research")
    assert [item["name"] for item in projects.items] == ["Atari", "Mario"]
    assert all(path.name != "_goal.yaml" for path in loaded_paths)

    goals = catalog.goals(entity="research", project="Mario")
    assert [item["goal_id"] for item in goals.items] == ["Level1-1"]
    parsed_goals = [path for path in loaded_paths if path.name == "_goal.yaml"]
    assert parsed_goals == [
        tmp_path / "experiments" / "goals" / "Mario" / "Level1-1" / "_goal.yaml"
    ]


def test_indexed_goal_metadata_persists_across_catalog_instances(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    write_indexed_goal_catalog(tmp_path)
    cache_path = tmp_path / "cache" / "catalog.json"
    first = PlayCatalog(repo_root=tmp_path, cache_path=cache_path)
    assert first.goals(entity="research", project="Mario").items
    assert cache_path.is_file()

    from gradlab import play_catalog

    real_loader = play_catalog.load_mapping_document

    def index_only_loader(path: Path, *, label: str | None = None) -> dict[str, Any]:
        if path.name == "_goal.yaml":
            raise AssertionError("unchanged goal metadata must come from persistent cache")
        return real_loader(path, label=label)

    monkeypatch.setattr(play_catalog, "load_mapping_document", index_only_loader)
    second = PlayCatalog(repo_root=tmp_path, cache_path=cache_path)

    goals = second.goals(entity="research", project="Mario")
    assert [item["title"] for item in goals.items] == ["Mario Level 1-1 completion"]


def test_checked_in_browse_catalog_matches_composed_goal_contracts() -> None:
    repo_root = Path(__file__).parents[1]
    catalog = PlayCatalog(repo_root=repo_root)

    for project in catalog.projects(entity="research").items:
        for goal in catalog.goals(
            entity="research",
            project=str(project["name"]),
        ).items:
            detailed = catalog._repository_goal(
                project=str(project["name"]),
                goal_id=str(goal["goal_id"]),
            )
            assert detailed.title == goal["title"]


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
    assert api.runs_filters == [{"config.goal_slug": "Mario/Level1-1"}]
    assert projects.items[0]["goal_count"] == 1
    assert [item["goal_id"] for item in goals.items] == ["Level1-1"]
    assert goals.items[0]["title"] == "Mario Level 1-1 completion"
    assert goals.items[0]["recipe_count"] == 1
    assert goals.items[0]["goal_path"].endswith("/Level1-1/_goal.yaml")
    assert [item["run_id"] for item in runs.items] == [RUN_ID]
    assert runs.items[0]["recipe"] == "ppo"
    assert runs.items[0]["description"] == "accepted"
    assert runs.items[0]["recipe_overrides"] == ("train.backend.config.learning_rate=0.0002",)
    assert runs.items[0]["recipe_variant_id"] == recipe_variant_id(
        recipe_slug="ppo",
        source_sha="c" * 40,
        recipe_overrides=["train.backend.config.learning_rate=0.0002"],
    )
    assert runs.items[0]["recipe_sha256"] == "f" * 64
    assert runs.metric_columns == (
        {
            "metric": "leader/checkpoint/step",
            "direction": "min",
        },
        {
            "metric": "eval/full/episode/return/mean",
            "direction": "max",
        },
    )
    assert runs.fallback_metric_columns == (
        {
            "metric": "train/outcome/success/window_100/rate/min",
            "direction": "max",
        },
        {
            "metric": "train/episode/return/shaped/from/target/mean",
            "direction": "max",
        },
        {
            "metric": "train/global_step",
            "direction": "min",
        },
    )
    assert runs.items[0]["metrics"] == {
        "leader/checkpoint/step": 1_500_000.0,
        "eval/full/episode/return/mean": 321.25,
        "train/outcome/success/window_100/rate/min": 0.75,
        "train/episode/return/shaped/from/target/mean": 123.5,
        "train/global_step": 2_000_000.0,
    }
    override_runs = catalog.runs(
        entity="research",
        project="Mario",
        goal_id="Level1-1",
        query="learning_rate=0.0002",
    )
    assert [item["run_id"] for item in override_runs.items] == [RUN_ID]
    assert catalog.run_goal(entity="research", project="Mario", run_id=RUN_ID) == "Level1-1"


def test_run_catalog_uses_one_bounded_projection_instead_of_lazy_run_hydration(
    tmp_path: Path,
) -> None:
    write_goal_catalog(tmp_path)
    catalog = PlayCatalog(repo_root=tmp_path)
    api = FakeCatalogGraphQLApi()
    catalog._api = api

    page = catalog.runs(
        entity="research",
        project="Mario",
        goal_id="Level1-1",
    )

    assert [item["run_id"] for item in page.items] == [SECOND_RUN_ID, RUN_ID]
    assert page.items[0]["name"] == "Run b"
    assert page.items[0]["description"] == "description b"
    assert page.items[0]["recipe"] == "ppo"
    assert page.items[0]["seed"] == 3
    assert page.items[0]["metrics"]["leader/checkpoint/step"] == 1_000_000
    assert page.items[0]["url"] == (f"https://wandb.example/research/Mario/runs/{SECOND_RUN_ID}")
    assert [call["cursor"] for call in api.client.calls] == [None, "page-2"]
    assert all(call["perPage"] == 200 for call in api.client.calls)
    assert all(call["order"] == "-created_at" for call in api.client.calls)
    assert all(
        json.loads(call["filters"]) == {"config.goal_slug": "Mario/Level1-1"}
        for call in api.client.calls
    )
    searched = catalog.runs(
        entity="research",
        project="Mario",
        goal_id="Level1-1",
        query="description b",
    )
    assert [item["run_id"] for item in searched.items] == [SECOND_RUN_ID]
    assert len(api.client.calls) == 2


def test_run_catalog_persists_ranked_summaries_across_player_processes(
    tmp_path: Path,
) -> None:
    write_goal_catalog(tmp_path)
    cache_path = tmp_path / "cache" / "catalog.json"
    first = PlayCatalog(repo_root=tmp_path, cache_path=cache_path)
    first._api = FakeCatalogGraphQLApi()

    initial = first.runs(
        entity="research",
        project="Mario",
        goal_id="Level1-1",
    )

    second = PlayCatalog(repo_root=tmp_path, cache_path=cache_path)
    no_call_api = NoCallCatalogGraphQLApi()
    second._api = no_call_api
    cached = second.runs(
        entity="research",
        project="Mario",
        goal_id="Level1-1",
    )

    assert cached == initial
    assert no_call_api.client.calls == 0


def test_stale_run_catalog_is_served_before_background_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_goal_catalog(tmp_path)
    cache_path = tmp_path / "cache" / "catalog.json"
    first = PlayCatalog(repo_root=tmp_path, cache_path=cache_path)
    first._api = FakeCatalogGraphQLApi()
    initial = first.runs(
        entity="research",
        project="Mario",
        goal_id="Level1-1",
    )
    cache_document = json.loads(cache_path.read_text(encoding="utf-8"))
    for entry in cache_document["run_catalogs"].values():
        entry["generated_at"] = 0
    cache_path.write_text(json.dumps(cache_document), encoding="utf-8")

    second = PlayCatalog(repo_root=tmp_path, cache_path=cache_path)
    no_call_api = NoCallCatalogGraphQLApi()
    second._api = no_call_api
    scheduled: list[dict[str, Any]] = []
    monkeypatch.setattr(
        second,
        "_schedule_run_catalog_refresh",
        lambda **kwargs: scheduled.append(kwargs),
    )

    stale = second.runs(
        entity="research",
        project="Mario",
        goal_id="Level1-1",
    )

    assert stale == initial
    assert no_call_api.client.calls == 0
    assert len(scheduled) == 1


def test_catalog_validates_and_orders_public_checkpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    periodic = checkpoint_row(step=250_000, digest="2" * 64, purpose="periodic")
    final = checkpoint_row(step=500_000, digest="3" * 64, purpose="final")
    monkeypatch.setattr(
        "gradlab.play_catalog._public_json",
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
        final["checkpoint_id"],
        periodic["checkpoint_id"],
    ]
    assert rows[1]["promoted"] is True
    assert rows[1]["manifest_url"].endswith("/manifest.json")


def test_catalog_attaches_goal_required_eval_results_by_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    periodic = checkpoint_row(step=250_000, digest="2" * 64, purpose="periodic")
    final = checkpoint_row(step=500_000, digest="3" * 64, purpose="final")
    monkeypatch.setattr(
        "gradlab.play_catalog._public_json",
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
            "seed": 7,
            "checkpoint_eval_contract": {
                "seed": 42_000,
                "acceptance": [
                    {
                        "metric": required_metric,
                        "operator": ">=",
                        "threshold": 1.0,
                    }
                ],
            },
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

    assert [row["step"] for row in rows] == [500_000, 250_000]
    rejected_row, accepted_row = rows
    accepted = accepted_row["evaluation"]
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
    assert accepted_row["playback_seed"] == 42_000
    assert accepted_row["playback_seed_source"] == "evaluation"
    rejected = rejected_row["evaluation"]
    assert rejected["status"] == "rejected"
    assert rejected["episodes_completed"] == 1
    assert rejected["failure_count"] == 1
    assert rejected["criteria"][0]["value"] is None
    assert rejected_row["playback_seed"] == 42_000
    assert rejected_row["playback_seed_source"] == "evaluation"


def test_catalog_uses_training_seed_when_checkpoint_has_no_eval_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    periodic = checkpoint_row(step=250_000, digest="2" * 64, purpose="periodic")
    monkeypatch.setattr(
        "gradlab.play_catalog._public_json",
        lambda _url: {
            "schema_version": 1,
            "run_id": RUN_ID,
            "checkpoints": [periodic],
            "promotion": None,
        },
    )

    class UnevaluatedRun:
        config = {
            "seed": 7,
            "checkpoint_eval_contract": {
                "seed": 42_000,
                "acceptance": [
                    {
                        "metric": "eval/full/outcome/success/rate/min",
                        "operator": ">=",
                        "threshold": 1.0,
                    }
                ],
            },
        }

        @staticmethod
        def scan_history(*, keys, page_size):
            return []

    class UnevaluatedApi:
        @staticmethod
        def run(_path):
            return UnevaluatedRun()

    catalog = PlayCatalog(public_models_base_url="https://models.example")
    catalog._api = UnevaluatedApi()

    row = catalog.checkpoints(
        entity="research",
        project="Mario",
        run_id=RUN_ID,
    )[0]

    assert row["evaluation"] is None
    assert row["playback_seed"] == 7
    assert row["playback_seed_source"] == "training"
