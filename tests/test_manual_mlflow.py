from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from gradlab.manual_evaluation import ManualEvaluationSupervisor
from gradlab.metric_journal import read_control_journal
from gradlab.mlflow_delivery import MlflowDelivery
from gradlab.r2_store import BucketConfig, RunStorageConfig
from gradlab.run_authority import RunAuthority
from gradlab.supervisor_ledger import SupervisorLedger
from tests.test_experiment_cli import _manifest_only_run


def test_manual_evaluation_retains_and_projects_to_frozen_mlflow(
    tmp_path: Path, monkeypatch
) -> None:
    manifest = replace(
        _manifest_only_run(),
        wandb={},
        tracking={
            "backend": "mlflow", "delivery": "online", "operator_profile": "pilot",
            "sources": {"backend": "launch override", "delivery": "built-in default"},
        },
    )
    storage = RunStorageConfig(
        control=BucketConfig((tmp_path / "control").as_uri()),
        evaluation=BucketConfig((tmp_path / "eval").as_uri()),
        models=BucketConfig(
            (tmp_path / "models").as_uri(), public_base_url="https://models.example.test"
        ),
    )
    authority = RunAuthority(storage)
    authority.create_manifest(manifest)
    uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    service = MlflowDelivery.open(
        tracking_uri=uri, experiment_name="gradlab-Bandit-v0",
        gradlab_run_id=manifest.run_id, created_at_ms=1_800_000_000_000,
    )
    service.publish_terminal(state="succeeded", reason="training_cap_complete")
    authority.control.put_json(f"runs/{manifest.run_id}/metrics-binding.json", {
        "schema_version": 1, "run_id": manifest.run_id, "backend": "mlflow",
        "service_run_id": service.run_id, "service_url": service.service_url,
    })
    queue = ManualEvaluationSupervisor(
        authority=authority, repo_root=Path.cwd(), work_root=tmp_path / "manual",
    )
    monkeypatch.setattr(
        "gradlab.manual_evaluation.metrics_schema_version_from_recipe_document",
        lambda _document: 24,
    )
    monkeypatch.setattr(
        "gradlab.manual_evaluation.evaluation_metric_records",
        lambda *_args, **_kwargs: ({"eval/return/mean": 0.9}, None),
    )
    context = SimpleNamespace(
        manifest=manifest,
        checkpoint=SimpleNamespace(step=128, checkpoint_id="checkpoint-128-test"),
        intent=SimpleNamespace(
            idempotency_key="1" * 64,
            execution_contract={
                "environment": {"game": "Bandit-v0"}, "episodes": 1, "acceptance": [],
            },
        ),
        recipe_document={"recipe": {"train_config": {
            "env_provider": "gradlab", "game": "Bandit-v0", "game_family": "Bandit",
        }}},
    )
    ledger = SupervisorLedger(tmp_path / "manual.sqlite")
    ledger.init()
    assert queue._project_result(
        context, SimpleNamespace(status="accepted"), ledger=ledger, event_seq_offset=0
    )
    journal = read_control_journal(authority.control, manifest.run_id)
    assert len(journal) == 1
    assert journal[0]["payload"] == {"eval/return/mean": 0.9}
    history = service.client.get_metric_history(service.run_id, "eval/return/mean")
    assert [(point.value, point.step) for point in history] == [(0.9, 128)]
