from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from mlflow.tracking import MlflowClient

from gradlab.local_publication import publish_local_run
from gradlab.queued_tracking_sync import _project_queued_outcome, sync_queued_run
from gradlab.run_authority import LeaseUnavailable
from gradlab.run_contracts import RunManifest, TerminalReceipt
from gradlab.supervisor_ledger import SupervisorLedger

pytest_plugins = ("tests.test_local_publication",)


@pytest.mark.parametrize(
    "local_run",
    [{"delivery": "local_only", "overrides": ("tracking.backend=mlflow",)}],
    indirect=True,
)
def test_queued_local_only_mlflow_sync_replays_r2_journal_without_changing_receipt(
    local_run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, config, _store, authority, _environment = local_run
    run_id = config["wandb_run_id"]
    (directory / "train-config.json").write_text(json.dumps(config))
    receipt = json.loads((directory / "local-run.json").read_text())
    (directory / "local-run.json").write_text(json.dumps({**receipt, "status": "complete_local"}))
    publish_local_run(directory, authority=authority)
    manifest = authority.manifest(run_id)
    terminal_key = f"runs/{run_id}/attempts/{manifest['attempt_id']}/terminal.json"
    terminal_bytes = authority.control.get_bytes(terminal_key)

    uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(
        "gradlab.local_metrics.load_repository_operator_environment", lambda *a, **k: None
    )
    first = sync_queued_run(authority, run_id)
    second = sync_queued_run(authority, run_id)

    assert first["service_run_id"] == second["service_run_id"]
    assert first["high_water"] == second["high_water"] == 1
    assert authority.control.get_bytes(terminal_key) == terminal_bytes
    history = MlflowClient(tracking_uri=uri).get_metric_history(
        first["service_run_id"], "train/return/mean"
    )
    assert [(point.value, point.step) for point in history] == [(0.5, 1000)]
    service_run = MlflowClient(tracking_uri=uri).get_run(first["service_run_id"])
    assert service_run.data.tags["ops/state"] == "complete_local"
    public = authority.models.get_json(f"runs/{run_id}/index.json")
    assert public["telemetry"]["tracking"]["tracker_sync_status"] == "delivered"


@pytest.mark.parametrize("local_run", [{"delivery": "local_only"}], indirect=True)
def test_queued_wandb_sync_reconstructs_frozen_frames_and_keeps_receipt(
    local_run, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, config, _store, authority, _environment = local_run
    run_id = config["wandb_run_id"]
    (directory / "train-config.json").write_text(json.dumps(config))
    receipt = json.loads((directory / "local-run.json").read_text())
    (directory / "local-run.json").write_text(json.dumps({**receipt, "status": "complete_local"}))
    publish_local_run(directory, authority=authority)
    manifest = authority.manifest(run_id)
    terminal_key = f"runs/{run_id}/attempts/{manifest['attempt_id']}/terminal.json"
    terminal_bytes = authority.control.get_bytes(terminal_key)

    def observe_replay(replay_dir: Path, *, heartbeat) -> str:
        heartbeat()
        frozen = json.loads((replay_dir / "train-config.json").read_text())
        assert frozen["tracking"]["backend"] == "wandb"
        assert frozen["tracking"]["delivery"] == "local_only"
        replay = SupervisorLedger(replay_dir / "gradlab.sqlite")
        replay.init()
        with replay.connection() as connection:
            frames = connection.execute(
                "SELECT id, step, status FROM metric_frames ORDER BY id"
            ).fetchall()
        assert [tuple(row) for row in frames] == [(1, 1000, "local_only")]
        (replay_dir / "tracker-sync.json").write_text(json.dumps({
            "backend": "wandb", "run_id": run_id, "service_run_id": run_id,
            "high_water": 1, "status": "delivered",
        }))
        return f"https://wandb.example.test/runs/{run_id}"

    monkeypatch.setattr("gradlab.queued_tracking_sync.sync_local_run", observe_replay)
    monkeypatch.setattr("gradlab.queued_tracking_sync._project_queued_outcome", lambda *a, **k: None)
    result = sync_queued_run(authority, run_id)
    assert result["backend"] == "wandb"
    assert authority.control.get_bytes(terminal_key) == terminal_bytes
    assert authority.models.get_json(f"runs/{run_id}/index.json")["telemetry"]["tracking"][
        "tracker_sync_status"
    ] == "delivered"


@pytest.mark.parametrize("local_run", [{"delivery": "local_only"}], indirect=True)
def test_queued_sync_reads_journal_under_lease_and_stops_on_lease_expiry(
    local_run, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, config, _store, authority, _environment = local_run
    run_id = config["wandb_run_id"]
    (directory / "train-config.json").write_text(json.dumps(config))
    receipt = json.loads((directory / "local-run.json").read_text())
    (directory / "local-run.json").write_text(json.dumps({**receipt, "status": "complete_local"}))
    publish_local_run(directory, authority=authority)
    now = authority.clock.utc_datetime()

    def read_under_lease(_control, _run_id):
        assert authority.control.get_json(f"runs/{run_id}/writer-lease.json")["run_id"] == run_id
        return [{"event_seq": 1, "event_id": "one", "step": 1000,
                 "source": "training", "kind": "history",
                 "payload": {"train/return/mean": 0.5}}]

    def expire_after_delivery(replay_dir: Path, *, heartbeat) -> str:
        heartbeat()
        (replay_dir / "tracker-sync.json").write_text(json.dumps({
            "backend": "wandb", "run_id": run_id, "service_run_id": run_id,
            "high_water": 1, "status": "delivered",
        }))
        monkeypatch.setattr(authority.clock, "utc_datetime", lambda: now + timedelta(minutes=2))
        return f"https://wandb.example.test/runs/{run_id}"

    monkeypatch.setattr("gradlab.queued_tracking_sync.read_control_journal", read_under_lease)
    monkeypatch.setattr("gradlab.queued_tracking_sync.sync_local_run", expire_after_delivery)
    with pytest.raises(LeaseUnavailable, match="expired"):
        sync_queued_run(authority, run_id)
    assert authority.control.get_json_optional(f"runs/{run_id}/metrics-binding.json") is None
    assert authority.control.get_json_optional(f"runs/{run_id}/tracker-sync/latest.json") is None


@pytest.mark.parametrize("local_run", [{"delivery": "local_only"}], indirect=True)
def test_queued_wandb_outcome_queries_normalized_run_path(
    local_run, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, config, _store, authority, _environment = local_run
    run_id = config["wandb_run_id"]
    (directory / "train-config.json").write_text(json.dumps(config))
    receipt = json.loads((directory / "local-run.json").read_text())
    (directory / "local-run.json").write_text(json.dumps({**receipt, "status": "complete_local"}))
    publish_local_run(directory, authority=authority)
    manifest = RunManifest.from_dict(authority.manifest(run_id))
    terminal = TerminalReceipt.from_dict(authority.control.get_json(
        f"runs/{run_id}/attempts/{manifest.attempt_id}/terminal.json"
    ))
    seen: list[str] = []
    fake_run = SimpleNamespace(id=run_id, path=["entity", "project", run_id])
    monkeypatch.setattr(
        "gradlab.wandb_publisher.WandbProjector.resume",
        lambda *_args, **_kwargs: SimpleNamespace(run=fake_run, close=lambda **_kwargs: None),
    )
    monkeypatch.setattr(
        "gradlab.wandb_publisher.publish_terminal_summary", lambda *_args: None
    )

    def remote_run(path: str):
        seen.append(path)
        return SimpleNamespace(summary={"ops/state": "complete_local",
                                        "ops/reason": terminal.stop_reason})

    monkeypatch.setattr("wandb.Api", lambda **_kwargs: SimpleNamespace(run=remote_run))
    _project_queued_outcome(
        authority, manifest=manifest, recipe={}, receipt=terminal,
        config={}, service_run_id=run_id, check_lease=lambda: None,
    )
    assert seen == [f"entity/project/{run_id}"]
