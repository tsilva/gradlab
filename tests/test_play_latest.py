from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError

import pytest

from gradlab.catalog_errors import CatalogUnavailable
from gradlab.model_sources import public_run_checkpoint_manifest_url
from gradlab.play import main
from gradlab.play_catalog import PlayCatalog

RUN_ID = "gradlab-" + "a" * 32


def checkpoint(step, purpose="periodic"):
    digest = "b" * 64
    base = f"https://models.example/runs/{RUN_ID}/checkpoints/{step}-{digest}"
    return dict(
        schema_version=2,
        run_id=RUN_ID,
        checkpoint_id=f"checkpoint-{step}-{digest[:16]}",
        step=step,
        purpose=purpose,
        sha256=digest,
        size_bytes=7,
        public_url=base + "/model.zip",
        model_document_url=base + "/model.json",
        model_document_sha256="c" * 64,
        recipe_document_url=base + "/recipe.json",
        recipe_document_sha256="d" * 64,
        goal_sha256="e" * 64,
        recipe_sha256="f" * 64,
        environment_sha256="1" * 64,
        evaluation_contract_sha256="2" * 64,
        recovery_sidecar_key="runs/recovery-sidecar.json",
        created_at="2026-08-06T12:00:00Z",
    )


def test_latest_checkpoint_uses_step_even_with_promotion(monkeypatch):
    older, newer = checkpoint(10, "final"), checkpoint(20)
    monkeypatch.setattr(
        "gradlab.model_sources._public_json",
        lambda url: dict(
            run_id=RUN_ID,
            checkpoints=[newer, older],
            promotion={"checkpoint_id": older["checkpoint_id"]},
        ),
    )
    assert public_run_checkpoint_manifest_url(RUN_ID, latest=True) == newer["public_url"].replace(
        "model.zip", "manifest.json"
    )
    assert public_run_checkpoint_manifest_url(RUN_ID) == older["public_url"].replace(
        "model.zip", "manifest.json"
    )


@pytest.mark.parametrize("missing_index", [False, True])
def test_latest_checkpoint_without_publication(monkeypatch, missing_index):
    def fetch(url):
        if missing_index:
            raise HTTPError(url, 404, "Not Found", {}, None)
        return dict(run_id=RUN_ID, checkpoints=[])

    monkeypatch.setattr("gradlab.model_sources._public_json", fetch)
    with pytest.raises(ValueError, match="no published checkpoint yet"):
        public_run_checkpoint_manifest_url(RUN_ID, latest=True)


def test_latest_run_uses_creation_time_across_goals(monkeypatch):
    catalog = PlayCatalog(control_bucket=Mock())
    goals = [
        SimpleNamespace(environment_id=name, goal_id=name, goal_slug=name)
        for name in ("one", "two")
    ]
    monkeypatch.setattr(catalog, "_repository_goals", lambda: goals)
    monkeypatch.setattr(catalog, "_control_generation_scopes", lambda goals: [{}, {}])

    def runs(**kwargs):
        if kwargs["environment_id"] == "one":
            return [
                dict(
                    run_id="older",
                    created_at="2026-09-10T12:00:00+02:00",
                    updated_at="2026-09-11T00:00:00Z",
                    goal_variant_id="v1",
                )
            ]
        return [
            dict(
                run_id=RUN_ID,
                created_at="2026-09-10T11:00:00Z",
                updated_at="2026-09-10T11:00:00Z",
                goal_variant_id="v2",
            )
        ]

    monkeypatch.setattr(catalog, "_control_run_catalog", runs)
    routes = catalog.latest_run_routes()
    assert [route["run_id"] for route in routes] == [RUN_ID, "older"]
    assert routes[0] == dict(
        level="runs",
        environment_id="two",
        goal_id="two",
        goal_variant_id="v2",
        run_id=RUN_ID,
        checkpoint_id="",
    )


def test_latest_requires_catalog_authority():
    with pytest.raises(CatalogUnavailable):
        PlayCatalog().latest_run_routes()


@pytest.mark.parametrize("newer_index", ["none", "missing", "empty", "unavailable"])
def test_latest_opens_exact_checkpoint(monkeypatch, newer_index):
    authority = Mock()
    monkeypatch.setattr(
        "gradlab.play_catalog_authority.start_catalog_authority_helper", lambda root: authority
    )
    monkeypatch.setattr("gradlab.play_catalog_authority.scrub_protected_environment", lambda: None)
    newer_run = "gradlab-" + "c" * 32
    routes = [dict(level="runs", run_id=RUN_ID, checkpoint_id="")]
    if newer_index != "none":
        routes.insert(0, dict(level="runs", run_id=newer_run, checkpoint_id=""))
    monkeypatch.setattr(PlayCatalog, "latest_run_routes", lambda self: routes)
    document = checkpoint(20)
    fetched = []

    def fetch(url):
        fetched.append(url)
        if newer_run in url:
            if newer_index in {"missing", "unavailable"}:
                raise HTTPError(url, 404 if newer_index == "missing" else 503, "error", {}, None)
            return dict(run_id=newer_run, checkpoints=[])
        return document if url.endswith("manifest.json") else dict(run_id=RUN_ID, checkpoints=[document])

    monkeypatch.setattr("gradlab.model_sources._public_json", fetch)
    application = Mock(return_value=23)
    monkeypatch.setattr("gradlab.play_web.run_web_player_application", application)
    if newer_index == "unavailable":
        with pytest.raises(SystemExit):
            main(["--latest", "--no-open"])
        application.assert_not_called()
        assert len(fetched) == 1
        authority.close.assert_called_once()
        return
    assert main(["--latest", "--no-open"]) == 23
    snapshot = application.call_args.args[0].snapshot()
    assert snapshot["app"]["source"]["checkpoint_id"] == document["checkpoint_id"]
    assert snapshot["app"]["source"]["value"] == document["public_url"].replace(
        "model.zip", "manifest.json"
    )
    authority.close.assert_called_once()


@pytest.mark.parametrize(
    "other",
    [
        ["--run", RUN_ID],
        ["--model", "model.zip"],
        ["--recipe", "foo"],
        ["--recording", "episode.trj"],
    ],
)
def test_latest_conflicts_with_other_sources(other):
    with pytest.raises(SystemExit) as exc:
        main(["--latest", *other])
    assert exc.value.code == 2


def test_latest_empty_catalog(monkeypatch):
    catalog = PlayCatalog(control_bucket=Mock())
    monkeypatch.setattr(catalog, "_repository_goals", lambda: [])
    monkeypatch.setattr(catalog, "_control_generation_scopes", lambda goals: [])
    with pytest.raises(CatalogUnavailable, match="No runs"):
        catalog.latest_run_routes()


def test_latest_missing_checkpoint_closes_authority_without_opening_player(monkeypatch):
    authority = Mock()
    monkeypatch.setattr(
        "gradlab.play_catalog_authority.start_catalog_authority_helper", lambda root: authority
    )
    monkeypatch.setattr("gradlab.play_catalog_authority.scrub_protected_environment", lambda: None)
    monkeypatch.setattr(
        PlayCatalog,
        "latest_run_routes",
        lambda self: [dict(level="runs", run_id=RUN_ID, checkpoint_id="")],
    )
    monkeypatch.setattr(
        "gradlab.model_sources._public_json", lambda url: dict(run_id=RUN_ID, checkpoints=[])
    )
    application = Mock()
    monkeypatch.setattr("gradlab.play_web.run_web_player_application", application)
    with pytest.raises(SystemExit) as exc:
        main(["--latest", "--no-open"])
    assert exc.value.code == 2
    application.assert_not_called()
    authority.close.assert_called_once()
