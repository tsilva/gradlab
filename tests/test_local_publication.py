import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gradlab.local_publication import LocalRunPublication
from gradlab.recipe_documents import compose_resolved_train_documents
from gradlab.policy_bundle import build_recipe_document, model_document_path, recipe_document_path
from gradlab.r2_store import BucketConfig, RunStorageConfig
from gradlab.run_authority import RunAuthority, LeaseUnavailable
from gradlab.run_contracts import new_run_id, new_attempt_id
from gradlab.supervisor_ledger import SupervisorLedger


@pytest.fixture
def local_run(tmp_path, request):
    parameters = getattr(request, "param", {})
    goal = Path("experiments/goals") / parameters.get("goal", "gradlab__bandit") / "_goal.yaml"
    resolved = compose_resolved_train_documents(
        goal,
        goal.parent / "recipes/ppo.yaml",
        source_sha="e" * 40,
        recipe_overrides=parameters.get("overrides", ()),
    )
    recipe = build_recipe_document(
        resolved.effective,
        repo_root=Path.cwd(),
        source_commit="e" * 40,
        run_description="local publication regression",
        seed=123,
        runtime_packages=("gradlab==0.1.0",),
        base_materialized_recipe=resolved.base,
        canonical_goal=resolved.canonical_goal,
    )
    directory = tmp_path / "run"
    directory.mkdir()
    config = {
        **resolved.effective["train_config"],
        "wandb_run_id": new_run_id(),
        "attempt_id": new_attempt_id(),
        "run_name": "local/test",
        "run_description": "local publication regression",
        "seed": 123,
        "timesteps": 1000,
        "wandb_entity": "test",
        "wandb_project": "Bandit-v0",
    }
    (directory / "recipe.json").write_text(json.dumps(recipe))
    (directory / "local-run.json").write_text(
        json.dumps({"started_at": "2026-09-15T12:00:00Z", "status": "completed"})
    )
    (directory / "training-result.json").write_text(
        json.dumps(
            {"status": "completed", "terminal_reason": "budget_exhausted", "final_step": 1000}
        )
    )
    store = SupervisorLedger(directory / "gradlab.sqlite")
    store.init()
    model = directory / "final_model.zip"
    model.write_bytes(b"checkpoint")
    model_document_path(model).write_text("{}")
    recipe_document_path(model).write_text(json.dumps(recipe))
    store.record_checkpoint(
        run_name="local/test", kind="final", step=1000, path=model, eval_required=False
    )
    store.append_metrics({"train/return/mean": 0.5}, step=1000, source="training")
    storage = RunStorageConfig(
        control=BucketConfig(uri=(tmp_path / "control").as_uri()),
        evaluation=BucketConfig(uri=(tmp_path / "evaluation").as_uri()),
        models=BucketConfig(
            uri=(tmp_path / "models").as_uri(), public_base_url="https://models.example.test"
        ),
    )
    return (
        directory,
        config,
        store,
        RunAuthority(storage),
        SimpleNamespace(game="Bandit-v0", env_provider="gradlab"),
    )


def test_local_run_publishes_catalog_checkpoints_and_metric_journal_idempotently(local_run):
    directory, config, store, authority, env = local_run
    for _ in range(2):
        with LocalRunPublication(directory, config, store, env, authority=authority) as publication:
            publication.publish()
            publication.finish(wandb_high_water=1)
    assert store.checkpoints()[0]["upload_status"] == "uploaded"
    manifest = authority.manifest(config["wandb_run_id"])
    assert manifest["compute"]["execution_backend"] == "local-process"
    assert manifest["modal"]["enabled"] is False
    catalog = authority.catalog_generation(manifest["goal_slug"])
    assert catalog["variants"][0]["run_count"] == 1
    terminal = authority.control.get_json(
        f"runs/{manifest['run_id']}/attempts/{manifest['attempt_id']}/terminal.json"
    )
    assert terminal["drain"]["complete"] is True
    assert terminal["drain"]["metric_segment_high_water"] == 1
    assert len(terminal["checkpoint_inventory"]) == 1
    assert authority.control.get_json_optional(f"runs/{manifest['run_id']}/terminal.json") is None
    assert authority.control.get_json_optional(f"runs/{manifest['run_id']}/promotion.json") is None


def test_local_writer_obeys_remote_lease(local_run):
    directory, config, store, authority, env = local_run
    authority.acquire_lease(
        run_id=config["wandb_run_id"], attempt_id=config["attempt_id"], holder_id="other"
    )
    with pytest.raises(LeaseUnavailable):
        with LocalRunPublication(directory, config, store, env, authority=authority):
            pytest.fail("must not acquire a second writer")


def test_missing_checkpoint_prevents_complete_receipt(local_run):
    directory, config, store, authority, env = local_run
    Path(store.checkpoints()[0]["path"]).unlink()
    with LocalRunPublication(directory, config, store, env, authority=authority) as publication:
        with pytest.raises(FileNotFoundError):
            publication.finish(wandb_high_water=1)
    assert (
        authority.control.get_json_optional(
            f"runs/{config['wandb_run_id']}/attempts/{config['attempt_id']}/terminal.json"
        )
        is None
    )


def test_periodic_and_final_checkpoints_are_both_published(local_run):
    directory, config, store, authority, env = local_run
    model = directory / "periodic.zip"
    model.write_bytes(b"earlier checkpoint")
    model_document_path(model).write_text("{}")
    recipe_document_path(model).write_text((directory / "recipe.json").read_text())
    store.record_checkpoint(
        run_name="local/test", kind="periodic", step=500, path=model, eval_required=False
    )
    with LocalRunPublication(directory, config, store, env, authority=authority) as publication:
        publication.finish(wandb_high_water=1)
    assert {row["purpose"] for row in store.checkpoint_publications()} == {"periodic", "final"}


def test_lost_lease_fences_publication(local_run):
    directory, config, store, authority, env = local_run
    with LocalRunPublication(directory, config, store, env, authority=authority) as publication:
        publication._error = LeaseUnavailable("renewal rejected")
        with pytest.raises(LeaseUnavailable, match="lost"):
            publication.publish()
    assert store.checkpoint_publications() == []


def test_remote_metric_confirmation_is_required_before_terminal(local_run, monkeypatch):
    from gradlab import local_wandb
    from unittest.mock import Mock

    directory, config, store, authority, env = local_run
    summary = {}

    class FakeRun:
        pass

    run = FakeRun()
    run.id = config["wandb_run_id"]
    run.url = "https://wandb.ai/test/run"
    run.path = "test/run"
    run.summary = summary
    run.log = lambda payload, step: summary.update(payload)
    run.define_metric = lambda *args, **kwargs: None
    projector = SimpleNamespace(run=run, close=Mock())
    monkeypatch.setattr(local_wandb.WandbProjector, "start_live", lambda *a, **k: projector)
    monkeypatch.setattr(local_wandb, "resolve_env_config", lambda *a: env)
    monkeypatch.setattr(
        local_wandb,
        "local_publication",
        lambda d, c, s, e: LocalRunPublication(d, c, s, e, authority=authority),
    )
    verify = Mock(side_effect=TimeoutError("not delivered"))
    monkeypatch.setattr(local_wandb, "_verify_remote_delivery", verify)
    with pytest.raises(TimeoutError, match="not delivered"):
        with local_wandb.local_wandb_writer(directory, config):
            pass
    assert not (directory / "publication-delivery.json").exists()
    assert (
        authority.control.get_json_optional(
            f"runs/{config['wandb_run_id']}/attempts/{config['attempt_id']}/terminal.json"
        )
        is None
    )
    verify.side_effect = None
    with local_wandb.local_wandb_writer(directory, config):
        pass
    assert (
        json.loads((directory / "publication-delivery.json").read_text())["status"] == "delivered"
    )
    assert store.pending_metric_frames() == []


def test_offline_publication_never_initializes_r2(tmp_path):
    from gradlab.local_publication import local_publication

    with local_publication(tmp_path, {"wandb_mode": "offline"}, None, None) as publication:
        assert publication is None


def test_upload_failure_is_recoverable_without_duplicate_identity(local_run, monkeypatch):
    from unittest.mock import Mock

    directory, config, store, authority, env = local_run
    publish = authority.publish_checkpoint
    monkeypatch.setattr(
        authority, "publish_checkpoint", Mock(side_effect=OSError("R2 unavailable"))
    )
    with LocalRunPublication(directory, config, store, env, authority=authority) as publication:
        with pytest.raises(OSError, match="R2 unavailable"):
            publication.publish()
    assert store.checkpoints()[0]["upload_status"] == "failed_retryable"
    assert not (directory / "publication-delivery.json").exists()
    monkeypatch.setattr(authority, "publish_checkpoint", publish)
    with LocalRunPublication(directory, config, store, env, authority=authority) as publication:
        publication.finish(wandb_high_water=1)
    assert len(store.checkpoint_publications()) == 1
    assert store.state("local_publication_identity")["run_id"] == config["wandb_run_id"]


def test_failed_learner_is_not_cataloged_as_success(local_run):
    directory, config, store, authority, env = local_run
    path = directory / "training-result.json"
    result = json.loads(path.read_text())
    path.write_text(json.dumps({**result, "status": "failed", "terminal_reason": "failed"}))
    with LocalRunPublication(directory, config, store, env, authority=authority) as publication:
        publication.finish(wandb_high_water=1)
    assert store.state("local_publication_terminal")["state"] == "failed"


@pytest.mark.parametrize(
    "local_run",
    [
        {
            "goal": "FrozenLake-v1/Default",
            "overrides": [
                "train.early_stop.conditions.return_plateau.start_after_steps=0",
                "train.early_stop.conditions.return_plateau.patience_steps=500",
            ],
        }
    ],
    indirect=True,
)
def test_neutral_early_stop_remains_neutral_in_catalog(local_run):
    from gradlab.early_stop import MetricEarlyStopStateMachine, MetricSample

    directory, config, store, authority, env = local_run
    machine = MetricEarlyStopStateMachine(config["early_stop"])
    machine.update({"train/return/mean": MetricSample(value=0.0, step=500)})
    decision = machine.update(
        {"train/return/mean": MetricSample(value=0.0, step=1000)}
    ).stop_decision
    assert decision["outcome"] == "neutral"
    (directory / f"early_stop_decision-{config['attempt_id']}.json").write_text(
        json.dumps(decision)
    )
    result_path = directory / "training-result.json"
    result = json.loads(result_path.read_text())
    result_path.write_text(json.dumps({**result, "terminal_reason": "early_stop_neutral"}))
    with LocalRunPublication(directory, config, store, env, authority=authority) as publication:
        publication.finish(wandb_high_water=1)
    terminal = store.state("local_publication_terminal")
    assert terminal["state"] == "stopped"
    assert terminal["early_stop"]["outcome"] == "neutral"
    assert terminal["stop_reason"] == "early_stop_neutral:return_plateau"
