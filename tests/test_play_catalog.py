from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import pytest

from gradlab.catalog_generation import (
    CATALOG_GENERATION_SCHEMA_VERSION,
    CATALOG_POINTER_KEY,
    catalog_generation_digest,
    catalog_generation_key,
)
from gradlab.catalog_cache import CatalogEntryCache
from gradlab.play_catalog import (
    PlayCatalog,
    parse_wandb_location,
)
from gradlab.catalog_errors import CatalogIntegrityError, CatalogUnavailable
from gradlab.catalog_errors import CatalogSnapshotChanged
from gradlab.play_session import build_parser as build_play_parser
from gradlab.policy_bundle import build_recipe_document, canonical_json_sha256
from gradlab.r2_store import BucketConfig, RunStorageConfig
from gradlab.goal_variants import (
    GOAL_VARIANT_INDEX_SCHEMA_VERSION,
    GOAL_VARIANT_RUN_INDEX_SCHEMA_VERSION,
    build_goal_variant_descriptor,
    goal_variant_scope_key,
)
from gradlab.metric_names import METRICS_SCHEMA_VERSION
from gradlab.recipe_documents import compose_resolved_train_documents, load_goal_contract
from gradlab.reward_programs import goal_for_contract_validation
from gradlab.run_authority import RunAuthority
from gradlab.run_contracts import (
    SCHEMA_VERSION,
    RunManifest,
    checkpoint_id,
    default_liveness_policy,
    new_attempt_id,
    new_run_id,
    utc_now,
)


RUN_ID = "gradlab-" + "a" * 32


def current_goal_document(*, goal_id: str, title: str) -> str:
    return "\n".join(
        (
            "defaults:",
            "- _self_",
            f"goal_id: {goal_id}",
            "evaluation_mode: evaluated",
            f"title: {title}",
            "objective:",
            "  rank:",
            "  - min(leader/checkpoint/step)",
            "  - max(eval/full/episode/return/shaped/mean)",
            "train:",
            "  checkpoint_freq: 128",
            "  environment:",
            "    env_provider: gradlab",
            "    env_config:",
            "      game: Bandit-v0",
            "      n_envs: 8",
            "      env_args:",
            "        autoreset_mode: disabled",
            "    preprocessing: &preprocessing",
            "      frame_skip: 1",
            "      max_pool_frames: false",
            "      sticky_action_prob: 0.0",
            "      obs_resize: [0, 0]",
            "      obs_crop: [0, 0, 0, 0]",
            "      obs_crop_mode: remove",
            "      obs_crop_fill: 0",
            "      obs_resize_algorithm: area",
            "    task: &task",
            "      id: identity",
            "      action: {set: native}",
            "      signals: {}",
            "      events: {}",
            "      termination: {max_episode_steps: 1}",
            "      reward:",
            "        reward_mode: native",
            "        reward_scale: 1.0",
            "        reward_clip: false",
            "eval:",
            "  episodes: 1",
            "  environment:",
            "    env_provider: gradlab",
            "    env_config:",
            "      game: Bandit-v0",
            "      n_envs: 1",
            "      env_args:",
            "        autoreset_mode: disabled",
            "    preprocessing: *preprocessing",
            "    task: *task",
            "",
        )
    )


class WandbRunControlBucket:
    @staticmethod
    def get_json_optional(key: str) -> dict[str, object]:
        assert key == f"runs/{RUN_ID}/manifest.json"
        return {
            "wandb": {
                "entity": "research",
                "project": "Mario",
            }
        }


def write_goal_catalog(repo_root: Path) -> None:
    goal_root = repo_root / "experiments" / "goals" / "Mario" / "Level1-1"
    recipes = goal_root / "recipes"
    recipes.mkdir(parents=True)
    (goal_root / "_goal.yaml").write_text(
        current_goal_document(
            goal_id="Level1-1",
            title="Mario Level 1-1 completion",
        ),
        encoding="utf-8",
    )
    (recipes / "ppo.yaml").write_text("defaults: [_self_]\n", encoding="utf-8")
    (repo_root / "experiments" / "goals" / "_catalog.yaml").write_text(
        "\n".join(
            (
                "schema_version: 2",
                "namespaces:",
                "  Mario:",
                "    environment_id: Mario",
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
            current_goal_document(goal_id=goal_id, title=title),
            encoding="utf-8",
        )
        (recipes / "ppo.yaml").write_text("defaults: [_self_]\n", encoding="utf-8")
    (goals_root / "_catalog.yaml").write_text(
        "\n".join(
            (
                "schema_version: 2",
                "namespaces:",
                "  Atari:",
                "    environment_id: Atari",
                "  Mario:",
                "    environment_id: Mario",
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
        "schema_version": SCHEMA_VERSION,
    }


def test_wandb_project_url_is_not_a_run_reference() -> None:
    assert parse_wandb_location("https://wandb.ai/research/Mario?nw=user") is None


def test_parse_wandb_location_ignores_query_and_returns_run() -> None:
    location = parse_wandb_location(f"https://wandb.ai/research/Mario/runs/{RUN_ID}?nw=user")

    assert location is not None
    assert (location.entity, location.project, location.run_id) == (
        "research",
        "Mario",
        RUN_ID,
    )


def test_public_recipe_resolution_distinguishes_absent_transient_and_corrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = checkpoint_row(step=100, digest="a" * 64, purpose="periodic")
    index = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "checkpoints": [checkpoint],
    }
    catalog = PlayCatalog(public_models_base_url="https://models.example")

    responses: list[Any] = [
        index,
        HTTPError(str(checkpoint["recipe_document_url"]), 404, "missing", {}, None),
    ]

    def absent(_url: str) -> Any:
        value = responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr("gradlab.play_catalog._public_json", absent)
    assert catalog._public_run_recipe_document(RUN_ID) is None

    responses[:] = [
        index,
        HTTPError(str(checkpoint["recipe_document_url"]), 503, "busy", {}, None),
    ]
    with pytest.raises(CatalogUnavailable) as transient:
        catalog._public_run_recipe_document(RUN_ID)
    assert transient.value.problem.code == "public_catalog_transient"
    assert transient.value.problem.retryable is True

    responses[:] = [index, {"format_version": 999}]
    with pytest.raises(CatalogIntegrityError):
        catalog._public_run_recipe_document(RUN_ID)


def test_play_parser_allows_bare_launch_and_rejects_wandb_project_urls() -> None:
    parser = build_play_parser()

    assert parser.parse_args([]).artifact_ref is None
    assert parser.parse_args(["--rom-path", "/tmp/game.nes"]).rom_path == Path("/tmp/game.nes")
    with pytest.raises(SystemExit):
        parser.parse_args(["https://wandb.ai/research/Mario"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--rom", "/tmp/game.nes"])


def test_goal_variants_use_one_private_index_read_without_wandb(
    tmp_path: Path,
) -> None:
    write_goal_catalog(tmp_path)
    goal_path = tmp_path / "experiments" / "goals" / "Mario" / "Level1-1" / "_goal.yaml"
    authored = load_goal_contract(goal_path, tmp_path)
    descriptor = build_goal_variant_descriptor(
        goal_slug="Mario/Level1-1",
        source_sha="a" * 40,
        authored_goal=authored,
        effective_goal=goal_for_contract_validation(
            authored,
            label="test goal",
        ),
    )
    scope = goal_variant_scope_key(goal_slug="Mario/Level1-1")

    class OneReadControlBucket:
        calls: list[str] = []

        def get_json_optional(self, key: str):
            self.calls.append(key)
            if key == CATALOG_POINTER_KEY:
                return None
            assert key == f"{scope}/index.json"
            return {
                "schema_version": GOAL_VARIANT_INDEX_SCHEMA_VERSION,
                "scope": {"goal_slug": "Mario/Level1-1"},
                "variants": [
                    {
                        **descriptor,
                        "descriptor_key": (f"{scope}/descriptors/{descriptor['variant_id']}.json"),
                        "first_run_id": RUN_ID,
                        "exact_resolution_run_id": RUN_ID,
                    }
                ],
            }

    bucket = OneReadControlBucket()
    cache_path = tmp_path / "play-catalog.json"
    cache_key = PlayCatalog._variant_cache_key(
        environment_id="Mario",
        goal_slug="Mario/Level1-1",
    )
    CatalogEntryCache(cache_path.with_name(f"{cache_path.name}.entries")).write(
        "variants-v5",
        cache_key,
        {
            "authority": "explicit-control",
            "generated_at": 1.0,
            "items": [
                {
                    "environment_id": "Mario",
                    "goal_slug": "Mario/Level1-1",
                    "variant_id": "obsolete-cache-entry",
                }
            ],
        },
    )
    catalog = PlayCatalog(
        repo_root=tmp_path,
        cache_path=cache_path,
        control_bucket=bucket,
    )

    page = catalog.goal_variants(
        environment_id="Mario",
        goal_id="Level1-1",
    )

    assert [item["variant_id"] for item in page.items] == [descriptor["variant_id"]]
    assert page.items[0]["exact_resolution_run_id"] == RUN_ID
    assert page.items[0]["configuration_kind"] == "current_default"
    assert page.items[0]["display_label"] == "No behavioral changes"
    assert page.items[0]["run_count"] == 0
    assert bucket.calls == [CATALOG_POINTER_KEY, f"{scope}/index.json"]


def test_goal_variants_load_deployed_schema_one_generation_without_new_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_goal_catalog(tmp_path)
    goal_path = tmp_path / "experiments" / "goals" / "Mario" / "Level1-1" / "_goal.yaml"
    authored = load_goal_contract(goal_path, tmp_path)
    historical_authored = deepcopy(authored)
    historical_authored["train"]["checkpoint_freq"] = 64
    historical_resolved = goal_for_contract_validation(
        historical_authored,
        label="deployed historical goal",
    )
    descriptor = build_goal_variant_descriptor(
        goal_slug="Mario/Level1-1",
        source_sha="a" * 40,
        authored_goal=historical_authored,
        effective_goal=historical_resolved,
    )
    generated_at = "2026-08-01T10:00:00Z"
    generation = {
        "schema_version": 1,
        "generated_at": generated_at,
        "scopes": [
            {
                "goal_slug": "Mario/Level1-1",
                "variants": [
                    {
                        **descriptor,
                        "first_run_id": RUN_ID,
                        "exact_resolution_run_id": RUN_ID,
                    }
                ],
                "runs": [
                    {
                        "run_id": RUN_ID,
                        "goal_slug": "Mario/Level1-1",
                        "goal_variant_id": descriptor["variant_id"],
                        "created_at": "2026-07-27T09:00:00Z",
                        "updated_at": "2026-08-01T09:00:00Z",
                        "metrics": {},
                    }
                ],
            }
        ],
    }
    digest = catalog_generation_digest(generation)
    pointer = {
        "schema_version": 1,
        "generation_sha256": digest,
        "generation_key": catalog_generation_key(digest),
        "generated_at": generated_at,
    }
    recipe_document = {
        "recipe": {
            "goal_variant": descriptor,
            "goal": historical_resolved,
        }
    }
    recipe_sha256 = canonical_json_sha256(recipe_document)

    class ExactManifest:
        pass

    exact_manifest = ExactManifest()
    exact_manifest.run_id = RUN_ID
    exact_manifest.goal_slug = "Mario/Level1-1"
    exact_manifest.recipe_slug = "ppo"
    exact_manifest.recipe_sha256 = recipe_sha256

    monkeypatch.setattr(
        "gradlab.play_catalog.RunManifest.from_dict",
        lambda _document: exact_manifest,
    )

    class DeployedGenerationBucket:
        def get_json_optional(self, key: str):
            if key == CATALOG_POINTER_KEY:
                return pointer
            if key == pointer["generation_key"]:
                return generation
            if key == f"runs/{RUN_ID}/manifest.json":
                return {"run_id": RUN_ID}
            if key == RunAuthority.recipe_document_key(recipe_sha256):
                return recipe_document
            raise AssertionError(key)

    catalog = PlayCatalog(repo_root=tmp_path, control_bucket=DeployedGenerationBucket())

    page = catalog.goal_variants(environment_id="Mario", goal_id="Level1-1")

    assert len(page.items) == 2
    assert page.items[0]["configuration_kind"] == "current_default"
    historical = page.items[1]
    assert historical["configuration_kind"] == "previous_default"
    assert historical["display_label"] == "Training checkpoint frequency 128 → 64"
    assert historical["comparison_available"] is True
    assert historical["run_count"] == 1
    assert historical["first_used_at"] == "2026-07-27T09:00:00Z"
    assert historical["last_activity_at"] == "2026-08-01T09:00:00Z"
    inspection = catalog.inspect_goal_variant(
        environment_id="Mario",
        goal_id="Level1-1",
        variant_id=str(descriptor["variant_id"]),
    )
    assert inspection["documents"]["goal"]["availability"] == "exact"
    assert inspection["documents"]["recipe"]["availability"] == "summary-only"


def test_goal_variants_explain_previous_defaults_and_aggregate_run_activity(
    tmp_path: Path,
) -> None:
    write_goal_catalog(tmp_path)
    goal_path = tmp_path / "experiments" / "goals" / "Mario" / "Level1-1" / "_goal.yaml"
    current_authored = load_goal_contract(goal_path, tmp_path)
    previous_authored = deepcopy(current_authored)
    previous_authored["train"]["checkpoint_freq"] = 64
    previous_resolved = goal_for_contract_validation(
        previous_authored,
        label="previous goal",
    )
    descriptor = build_goal_variant_descriptor(
        goal_slug="Mario/Level1-1",
        source_sha="b" * 40,
        authored_goal=previous_authored,
        effective_goal=previous_resolved,
    )
    run_one = "gradlab-" + "1" * 32
    run_two = "gradlab-" + "2" * 32
    generated_at = "2026-08-01T10:00:00Z"
    generation = {
        "schema_version": CATALOG_GENERATION_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scopes": [
            {
                "goal_slug": "Mario/Level1-1",
                "variants": [
                    {
                        **descriptor,
                        "first_run_id": run_one,
                        "exact_resolution_run_id": run_one,
                        "resolved_goal": previous_resolved,
                    }
                ],
                "runs": [
                    {
                        "run_id": run_one,
                        "goal_slug": "Mario/Level1-1",
                        "goal_variant_id": descriptor["variant_id"],
                        "created_at": "2026-06-01T09:00:00Z",
                        "updated_at": "2026-06-02T09:00:00Z",
                        "metrics": {},
                    },
                    {
                        "run_id": run_two,
                        "goal_slug": "Mario/Level1-1",
                        "goal_variant_id": descriptor["variant_id"],
                        "created_at": "2026-07-01T09:00:00Z",
                        "updated_at": "2026-07-03T09:00:00Z",
                        "metrics": {},
                    },
                ],
            }
        ],
    }
    digest = catalog_generation_digest(generation)
    pointer = {
        "schema_version": 1,
        "generation_sha256": digest,
        "generation_key": catalog_generation_key(digest),
        "generated_at": generated_at,
    }

    class GenerationBucket:
        def get_json_optional(self, key: str):
            if key == CATALOG_POINTER_KEY:
                return pointer
            if key == pointer["generation_key"]:
                return generation
            raise AssertionError(key)

    catalog = PlayCatalog(repo_root=tmp_path, control_bucket=GenerationBucket())

    page = catalog.goal_variants(environment_id="Mario", goal_id="Level1-1")

    assert [item["configuration_kind"] for item in page.items] == [
        "current_default",
        "previous_default",
    ]
    previous = page.items[1]
    assert previous["comparison_available"] is True
    assert previous["display_label"] == "Training checkpoint frequency 128 → 64"
    assert previous["run_count"] == 2
    assert previous["first_used_at"] == "2026-06-01T09:00:00Z"
    assert previous["last_activity_at"] == "2026-07-03T09:00:00Z"
    search = catalog.goal_variants(
        environment_id="Mario",
        goal_id="Level1-1",
        query="checkpoint freq",
    )
    assert [item["variant_id"] for item in search.items] == [descriptor["variant_id"]]


def test_run_catalog_uses_lifecycle_owned_variant_index_without_wandb(
    tmp_path: Path,
) -> None:
    write_goal_catalog(tmp_path)
    goal_path = tmp_path / "experiments" / "goals" / "Mario" / "Level1-1" / "_goal.yaml"
    authored = load_goal_contract(goal_path, tmp_path)
    descriptor = build_goal_variant_descriptor(
        goal_slug="Mario/Level1-1",
        source_sha="a" * 40,
        authored_goal=authored,
        effective_goal=goal_for_contract_validation(authored, label="test goal"),
    )
    scope = goal_variant_scope_key(goal_slug="Mario/Level1-1")

    class IndexedControlBucket:
        calls: list[str] = []

        def get_json_optional(self, key: str):
            self.calls.append(key)
            if key == CATALOG_POINTER_KEY:
                return None
            if key == f"{scope}/index.json":
                return {
                    "schema_version": GOAL_VARIANT_INDEX_SCHEMA_VERSION,
                    "scope": {"goal_slug": "Mario/Level1-1"},
                    "variants": [{**descriptor, "first_run_id": RUN_ID}],
                }
            assert key == f"{scope}/runs/{descriptor['variant_id']}.json"
            return {
                "schema_version": GOAL_VARIANT_RUN_INDEX_SCHEMA_VERSION,
                "scope": {
                    "goal_slug": "Mario/Level1-1",
                    "variant_id": descriptor["variant_id"],
                },
                "runs": [
                    {
                        "run_id": RUN_ID,
                        "attempt_id": "attempt-" + "b" * 16,
                        "name": "Indexed run",
                        "state": "succeeded",
                        "stop_reason": "eval_acceptance",
                        "final_step": 1_750_000,
                        "early_stop": None,
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
                            "eval/full/episode/return/shaped/mean": 321.25,
                        },
                    }
                ],
            }

    bucket = IndexedControlBucket()
    catalog = PlayCatalog(repo_root=tmp_path, control_bucket=bucket)

    page = catalog.runs(
        environment_id="Mario",
        goal_id="Level1-1",
    )

    assert [item["run_id"] for item in page.items] == [RUN_ID]
    assert page.items[0]["description"] == "lifecycle projection"
    assert page.items[0]["stop_reason"] == "eval_acceptance"
    assert page.items[0]["final_step"] == 1_750_000
    assert page.items[0]["metrics"]["leader/checkpoint/step"] == 1_500_000.0
    searched = catalog.runs(
        environment_id="Mario",
        goal_id="Level1-1",
        query="eval_acceptance",
    )
    assert [item["run_id"] for item in searched.items] == [RUN_ID]
    assert bucket.calls == [
        CATALOG_POINTER_KEY,
        f"{scope}/index.json",
        f"{scope}/runs/{descriptor['variant_id']}.json",
    ]


def test_missing_control_authority_is_not_reported_as_an_empty_run_catalog(
    tmp_path: Path,
) -> None:
    write_goal_catalog(tmp_path)
    catalog = PlayCatalog(repo_root=tmp_path)

    with pytest.raises(CatalogUnavailable, match="requires control-catalog authority"):
        catalog.runs(
            environment_id="Mario",
            goal_id="Level1-1",
        )


def test_catalog_cursor_is_bound_to_one_ordered_snapshot(tmp_path: Path) -> None:
    catalog = PlayCatalog(repo_root=tmp_path)
    items = [{"run_id": f"run-{index:03d}"} for index in range(75)]

    first = catalog._page(
        items,
        None,
        identity={"route": "runs", "query": ""},
    )

    assert len(first.items) == 50
    assert first.next_cursor
    second = catalog._page(
        items,
        first.next_cursor,
        identity={"route": "runs", "query": ""},
    )
    assert [item["run_id"] for item in second.items] == [
        f"run-{index:03d}" for index in range(50, 75)
    ]
    with pytest.raises(CatalogSnapshotChanged):
        catalog._page(
            [*items, {"run_id": "run-new"}],
            first.next_cursor,
            identity={"route": "runs", "query": ""},
        )


def test_repository_catalog_requires_explicit_namespace_index(tmp_path: Path) -> None:
    goal_root = tmp_path / "experiments" / "goals" / "Mario" / "Level1-1"
    goal_root.mkdir(parents=True)
    (goal_root / "_goal.yaml").write_text("goal_id: Level1-1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="repository goal catalog does not exist"):
        PlayCatalog(repo_root=tmp_path).environments()


def test_repository_catalog_reconciles_namespace_drift(tmp_path: Path) -> None:
    write_indexed_goal_catalog(tmp_path)
    goals_root = tmp_path / "experiments" / "goals"
    catalog = PlayCatalog(repo_root=tmp_path)

    assert [item["name"] for item in catalog.environments().items] == [
        "Atari",
        "Mario",
    ]

    for path in (goals_root / "Atari").rglob("*"):
        if path.is_file():
            path.unlink()
    for path in sorted((goals_root / "Atari").rglob("*"), reverse=True):
        if path.is_dir():
            path.rmdir()
    (goals_root / "Atari").rmdir()

    assert [item["name"] for item in catalog.environments().items] == ["Mario"]

    orphan_goal = goals_root / "Undeclared" / "Hidden"
    orphan_goal.mkdir(parents=True)
    (orphan_goal / "_goal.yaml").write_text("goal_id: Hidden\n", encoding="utf-8")

    assert [item["name"] for item in catalog.environments().items] == ["Mario"]


def test_repository_catalog_allows_empty_namespace_index(tmp_path: Path) -> None:
    goals_root = tmp_path / "experiments" / "goals"
    goals_root.mkdir(parents=True)
    (goals_root / "_catalog.yaml").write_text(
        "schema_version: 2\nnamespaces: {}\n",
        encoding="utf-8",
    )

    assert PlayCatalog(repo_root=tmp_path).environments().items == ()


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

    environments = catalog.environments()
    assert [item["name"] for item in environments.items] == ["Atari", "Mario"]
    assert all(path.name != "_goal.yaml" for path in loaded_paths)

    goals = catalog.goals(environment_id="Mario")
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
    assert first.goals(environment_id="Mario").items
    assert cache_path.is_file()

    from gradlab import play_catalog

    real_loader = play_catalog.load_mapping_document

    def index_only_loader(path: Path, *, label: str | None = None) -> dict[str, Any]:
        if path.name == "_goal.yaml":
            raise AssertionError("unchanged goal metadata must come from persistent cache")
        return real_loader(path, label=label)

    monkeypatch.setattr(play_catalog, "load_mapping_document", index_only_loader)
    second = PlayCatalog(repo_root=tmp_path, cache_path=cache_path)

    goals = second.goals(environment_id="Mario")
    assert [item["title"] for item in goals.items] == ["Mario Level 1-1 completion"]


def test_checked_in_browse_catalog_matches_composed_goal_contracts() -> None:
    repo_root = Path(__file__).parents[1]
    catalog = PlayCatalog(repo_root=repo_root)

    for environment in catalog.environments().items:
        for goal in catalog.goals(
            environment_id=str(environment["name"]),
        ).items:
            detailed = catalog._repository_goal(
                environment_id=str(environment["name"]),
                goal_id=str(goal["goal_id"]),
            )
            assert detailed.title == goal["title"]


def test_checked_in_goal_and_recipe_inspection_use_resolved_repository_contracts() -> None:
    repo_root = Path(__file__).parents[1]
    catalog = PlayCatalog(repo_root=repo_root)
    project = "Bandit-v0"
    goal_id = "gradlab__bandit"

    goal = catalog.inspect_goal(
        environment_id=project,
        goal_id=goal_id,
    )
    recipes = catalog.recipes(
        environment_id=project,
        goal_id=goal_id,
    )
    recipe = catalog.inspect_recipe(
        environment_id=project,
        goal_id=goal_id,
        recipe_id=str(recipes.items[0]["recipe_id"]),
    )

    assert goal["documents"]["goal"]["availability"] == "exact"
    assert "goal_id: gradlab__bandit" in goal["documents"]["goal"]["views"]["resolved"]
    assert recipe["documents"]["recipe"]["availability"] == "static-preview"
    assert "training_backend:" in recipe["documents"]["recipe"]["views"]["resolved"]
    assert recipe["documents"]["recipe"]["views"]["changes"]["entries"] == []


def test_run_and_goal_variant_inspection_use_the_verified_v2_control_recipe(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).parents[1]
    storage = RunStorageConfig(
        control=BucketConfig((tmp_path / "control").resolve().as_uri()),
        evaluation=BucketConfig((tmp_path / "evaluation").resolve().as_uri()),
        models=BucketConfig(
            (tmp_path / "models").resolve().as_uri(),
            public_base_url="https://models.example.test",
        ),
    )
    authority = RunAuthority(storage)
    goal_path = repo_root / "experiments/goals/gradlab__bandit/_goal.yaml"
    recipe_path = goal_path.parent / "recipes/ppo.yaml"
    source_sha = "a" * 40
    resolved = compose_resolved_train_documents(
        goal_path,
        recipe_path,
        recipe_overrides=("train.backend.config.gamma=0.97",),
        source_sha=source_sha,
    )
    recipe = build_recipe_document(
        resolved.effective,
        repo_root=repo_root,
        source_commit=source_sha,
        run_description="inspect an exact queued run",
        seed=3,
        runtime_packages=("gradlab==0.1.0",),
        base_materialized_recipe=resolved.base,
        canonical_goal=resolved.canonical_goal,
    )
    recipe_sha256 = canonical_json_sha256(recipe)
    authority.put_recipe_document(recipe, expected_sha256=recipe_sha256)
    run_id = new_run_id()
    manifest = RunManifest(
        run_id=run_id,
        attempt_id=new_attempt_id(),
        created_at=utc_now(),
        source_sha=source_sha,
        image_digest="docker:example/gradlab@sha256:" + "b" * 64,
        goal_slug="gradlab__bandit",
        goal_sha256=resolved.effective["train_config"]["effective_goal_contract_sha256"],
        recipe_slug="ppo",
        recipe_sha256=recipe_sha256,
        recipe_overrides=("train.backend.config.gamma=0.97",),
        environment_sha256=str(resolved.effective["environment_hash"]).removeprefix("sha256:"),
        seed=3,
        run_description="inspect an exact queued run",
        compute={
            "request": {"kind": "local", "max_duration_seconds": 3600},
            "selected": {"kind": "local", "max_duration_seconds": 3600},
            "dstack_task": run_id,
            "runtime_workflow_run_id": "1",
            "runtime_input_sha256": "c" * 64,
            "runtime_build_source_sha": source_sha,
        },
        wandb={
            "run_id": run_id,
            "entity": "research",
            "project": "Bandit-v0",
            "url": f"https://wandb.ai/research/Bandit-v0/runs/{run_id}",
        },
        modal={"enabled": False, "rom_asset_manifest": None},
        storage=storage.manifest_locations(),
        goal_variant=resolved.effective["goal_variant"],
        liveness=default_liveness_policy(),
    )
    authority.create_manifest(manifest)
    for key in tuple(authority.control.iter_keys("goal-variants/v2/")):
        authority.control.delete(
            key,
            if_match=str(authority.control.head(key)["etag"]),
        )
    catalog = PlayCatalog(repo_root=repo_root, control_bucket=authority.control)

    run_page = catalog.runs(
        environment_id="Bandit-v0",
        goal_id="gradlab__bandit",
        goal_variant_id=str(resolved.effective["goal_variant"]["variant_id"]),
    )
    run = catalog.inspect_run(run_id=run_id)
    variant = catalog.inspect_goal_variant(
        environment_id="Bandit-v0",
        goal_id="gradlab__bandit",
        variant_id=resolved.effective["goal_variant"]["variant_id"],
    )

    assert [item["run_id"] for item in run_page.items] == [run_id]
    assert run["documents"]["goal"]["availability"] == "exact"
    assert run["documents"]["recipe"]["is_variant"] is True
    assert "/train_config/training_backend/config/gamma" in {
        entry["path"] for entry in run["documents"]["recipe"]["views"]["changes"]["entries"]
    }
    assert variant["source"]["exact_resolution_run_id"] == run_id
    assert (
        variant["documents"]["goal"]["variant_id"]
        == resolved.effective["goal_variant"]["variant_id"]
    )
    assert "compare this configuration with the current default" in str(
        variant["documents"]["goal"]["message"]
    )
    assert variant["documents"]["recipe"]["availability"] == "exact"


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


def test_catalog_attaches_latest_training_metrics_at_each_checkpoint(
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
            "promotion": None,
        },
    )

    class TrainingRun:
        config = {"seed": 7}

        @staticmethod
        def scan_history(*, keys, page_size):
            assert page_size == 10_000
            if keys == [
                "train/global_step",
                "train/outcome/success/across_starts/window_100/rate/min",
            ]:
                return [
                    {
                        "train/global_step": 200_000,
                        "train/outcome/success/across_starts/window_100/rate/min": 0.25,
                    },
                    {
                        "train/global_step": 490_000,
                        "train/outcome/success/across_starts/window_100/rate/min": 0.9,
                    },
                ]
            if keys == [
                "train/global_step",
                "train/episode/return/shaped/across_origins/rolling_up_to_100/mean",
            ]:
                return [
                    {
                        "train/global_step": 220_000,
                        "train/episode/return/shaped/across_origins/rolling_up_to_100/mean": 11.5,
                    },
                    {
                        "train/global_step": 480_000,
                        "train/episode/return/shaped/across_origins/rolling_up_to_100/mean": 22.0,
                    },
                ]
            return []

    class TrainingApi:
        @staticmethod
        def run(path):
            assert path == f"research/Mario/{RUN_ID}"
            return TrainingRun()

    catalog = PlayCatalog(
        public_models_base_url="https://models.example",
        control_bucket=WandbRunControlBucket(),
    )
    catalog._api = TrainingApi()

    final_row, periodic_row = catalog.checkpoints(run_id=RUN_ID)

    assert periodic_row["metrics"] == {
        "train/global_step": 250_000.0,
        "train/outcome/success/across_starts/window_100/rate/min": 0.25,
        "train/episode/return/shaped/from/target/rolling_up_to_100/mean": 11.5,
    }
    assert final_row["metrics"] == {
        "train/global_step": 500_000.0,
        "train/outcome/success/across_starts/window_100/rate/min": 0.9,
        "train/episode/return/shaped/from/target/rolling_up_to_100/mean": 22.0,
    }
    assert final_row["best_training"] is True
    assert periodic_row["best_training"] is False
    assert final_row["best_evaluation"] is False
    assert periodic_row["best_evaluation"] is False
    filtered = catalog.checkpoints(run_id=RUN_ID, query=periodic["checkpoint_id"])
    assert len(filtered) == 1
    assert filtered[0]["checkpoint_id"] == periodic["checkpoint_id"]
    assert filtered[0]["best_training"] is False


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
    required_metric = "eval/full/outcome/success/across_starts/rate/min"

    class EvalRun:
        config = {
            "metrics_schema_version": METRICS_SCHEMA_VERSION,
            "seed": 7,
            "selection_rank": [
                f"max({required_metric})",
                "min(leader/checkpoint/step)",
            ],
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
            assert page_size == 10_000
            if required_metric in keys:
                return [
                    {
                        "eval/checkpoint/step": 250_000,
                        required_metric: 1.0,
                    }
                ]
            return [
                {
                    "eval/checkpoint/step": 250_000,
                    "eval/acceptance/pass": 1.0,
                    "eval/acceptance/episode/planned/count": 100.0,
                    "eval/acceptance/episode/completed/count": 100.0,
                },
                {
                    "eval/checkpoint/step": 500_000,
                    "eval/acceptance/pass": 0.0,
                    "eval/acceptance/episode/planned/count": 100.0,
                    "eval/acceptance/episode/completed/count": 1.0,
                },
            ]

    class EvalApi:
        @staticmethod
        def run(path):
            assert path == f"research/Mario/{RUN_ID}"
            return EvalRun()

    catalog = PlayCatalog(
        public_models_base_url="https://models.example",
        control_bucket=WandbRunControlBucket(),
    )
    catalog._api = EvalApi()

    rows = catalog.checkpoints(run_id=RUN_ID)

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
    assert accepted["metrics"] == {
        required_metric: 1.0,
        "leader/checkpoint/step": 250_000.0,
    }
    assert accepted_row["best_evaluation"] is True
    rejected = rejected_row["evaluation"]
    assert rejected["status"] == "rejected"
    assert rejected["episodes_completed"] == 1
    assert "failure_count" not in rejected
    assert rejected["criteria"][0]["value"] is None
    assert rejected_row["playback_seed"] == 42_000
    assert rejected_row["playback_seed_source"] == "evaluation"
    assert rejected_row["best_evaluation"] is False


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
            "metrics_schema_version": METRICS_SCHEMA_VERSION,
            "seed": 7,
            "checkpoint_eval_contract": {
                "seed": 42_000,
                "acceptance": [
                    {
                        "metric": "eval/full/outcome/success/across_starts/rate/min",
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

    catalog = PlayCatalog(
        public_models_base_url="https://models.example",
        control_bucket=WandbRunControlBucket(),
    )
    catalog._api = UnevaluatedApi()

    row = catalog.checkpoints(run_id=RUN_ID)[0]

    assert row["evaluation"] is None
    assert row["playback_seed"] == 7
    assert row["playback_seed_source"] == "training"
