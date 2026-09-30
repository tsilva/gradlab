from __future__ import annotations

from pathlib import Path
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from mlflow.tracking import MlflowClient

from gradlab.metric_store import MetricStore
from gradlab.metric_names import METRICS_SCHEMA_VERSION
from gradlab.mlflow_delivery import MlflowDelivery
from gradlab.wandb_publisher import _publish_frame
from gradlab.local_metrics import local_metrics_writer, sync_local_run
from gradlab.mlflow_rebind import replace_mlflow_binding
from gradlab.r2_store import BucketConfig, RunStorageConfig
from gradlab.r2_store import R2Bucket
from gradlab.run_authority import RunAuthority
from tests.test_experiment_cli import _manifest_only_run


@pytest.fixture(autouse=True)
def public_assets(tmp_path: Path, monkeypatch):
    uri = (tmp_path / "public-assets").as_uri()
    monkeypatch.setenv("GRADLAB_MODELS_R2_URI", uri)
    monkeypatch.setenv("GRADLAB_MODELS_R2_PUBLIC_BASE_URL", uri)


@contextmanager
def _server(tmp_path: Path, port: int):
    command = [
        sys.executable, "-m", "mlflow", "server",
        "--backend-store-uri", f"sqlite:///{tmp_path / 'tracking.db'}",
        "--default-artifact-root", str(tmp_path / "artifacts"),
        "--host", "127.0.0.1", "--port", str(port), "--workers", "1",
    ]
    with (tmp_path / "server.log").open("a") as log:
        server = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"},
        )
        try:
            client = MlflowClient(tracking_uri=f"http://127.0.0.1:{port}")
            deadline = time.monotonic() + 30
            while True:
                try:
                    client.get_experiment_by_name("gradlab-test")
                    break
                except Exception:
                    if server.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError("test MLflow server failed to start")
                    time.sleep(0.25)
            yield
        finally:
            os.killpg(server.pid, signal.SIGTERM)
            server.wait(timeout=10)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_replayed_frame_has_one_visible_scientific_point(tmp_path: Path) -> None:
    uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    client = MlflowClient(tracking_uri=uri)
    run_id = "gradlab-" + "a" * 32
    store = MetricStore(tmp_path / "journal.sqlite")
    store.init()
    store.append_metrics({"train/return/mean": 0.75}, step=128, source="train")
    frame = store.pending_metric_frames(limit=1)[0]

    first = MlflowDelivery.open(
        tracking_uri=uri,
        experiment_name="gradlab-test",
        gradlab_run_id=run_id,
        created_at_ms=1_800_000_000_000,
    )
    first.publish_frame(frame)
    # Simulate an acknowledged remote commit followed by local acknowledgement loss.
    recovered = MlflowDelivery.open(
        tracking_uri=uri,
        experiment_name="gradlab-test",
        gradlab_run_id=run_id,
        created_at_ms=1_800_000_000_000,
    )
    assert recovered.run_id == first.run_id
    recovered.publish_frame(frame)
    history = client.get_metric_history(first.run_id, "train/return/mean")
    assert [(point.value, point.step) for point in history] == [(0.75, 128)]
    assert recovered.remote_high_water() == 1


def test_wandb_and_mlflow_share_scientific_measure_and_axis(tmp_path: Path) -> None:
    run_id = "gradlab-" + "d" * 32
    store = MetricStore(tmp_path / "journal.sqlite")
    store.init()
    store.append_metrics({"train/return/mean": 0.6}, step=128, source="train")
    frame = store.pending_metric_frames(limit=1)[0]

    class WandbRun:
        def __init__(self):
            self.rows = []

        def define_metric(self, *_args, **_kwargs):
            pass

        def log(self, row, *, step):
            self.rows.append((step, row))

    wandb = WandbRun()
    _publish_frame(wandb, frame, metrics_schema_version=METRICS_SCHEMA_VERSION)
    mlflow = MlflowDelivery.open(
        tracking_uri=f"sqlite:///{tmp_path / 'tracking.db'}",
        experiment_name="gradlab-test", gradlab_run_id=run_id,
        created_at_ms=1_800_000_000_000,
    )
    mlflow.publish_frame(frame)
    client = mlflow.client
    for metric in ("train/return/mean", "train/step", "ops/sequence"):
        point = client.get_metric_history(mlflow.run_id, metric)
        assert len(point) == 1
        assert point[0].value == float(wandb.rows[0][1][metric])
    assert client.get_metric_history(mlflow.run_id, "train/return/mean")[0].step == 128


def test_mlflow_exposes_representative_video_from_canonical_r2(tmp_path: Path) -> None:
    bucket = R2Bucket(BucketConfig((tmp_path / "eval").as_uri()))
    content = b"representative-video-bytes"
    digest = hashlib.sha256(content).hexdigest()
    bucket.put_bytes("runs/video.mp4", content)
    delivery = MlflowDelivery.open(
        tracking_uri=f"sqlite:///{tmp_path / 'tracking.db'}",
        experiment_name="gradlab-test", gradlab_run_id="gradlab-" + "1" * 32,
        created_at_ms=1_800_000_000_000,
    )
    delivery.publish_frame({
        "id": 1, "kind": "evaluation_video", "step": 256, "source": "eval",
        "payload_json": json.dumps({
            "bucket_uri": bucket.config.uri, "key": "runs/video.mp4",
            "bytes": len(content), "sha256": digest,
        }),
    })
    delivery.publish_frame({
        "id": 1, "kind": "evaluation_video", "step": 256, "source": "eval",
        "payload_json": json.dumps({"bucket_uri": bucket.config.uri,
                                    "key": "runs/video.mp4", "bytes": len(content), "sha256": digest}),
    })
    tags = delivery.client.get_run(delivery.run_id).data.tags
    reference = json.loads(tags["gradlab.asset.000000000001"])
    assert Path(reference["url"].removeprefix("file://")).read_bytes() == content
    assert reference["sha256"] == digest
    assert reference["step"] == 256
    assert tags["mlflow.note.content"].count("[episode.mp4") == 1
    assert delivery.client.list_artifacts(delivery.run_id) == []
    assert delivery.remote_high_water() == 1

    with pytest.raises(ValueError, match="hash/size mismatch"):
        delivery.publish_frame({
            "id": 2, "kind": "evaluation_video", "step": 512, "source": "eval",
            "payload_json": json.dumps({"bucket_uri": bucket.config.uri,
                                        "key": "runs/video.mp4", "bytes": len(content), "sha256": "0" * 64}),
        })
    assert delivery.remote_high_water() == 1
    assert "gradlab.asset.000000000002" not in delivery.client.get_run(delivery.run_id).data.tags


def test_replay_after_tracking_server_restart_has_one_visible_point(tmp_path: Path) -> None:
    port = _free_port()
    uri = f"http://127.0.0.1:{port}"
    run_id = "gradlab-" + "b" * 32
    store = MetricStore(tmp_path / "journal.sqlite")
    store.init()
    store.append_metrics({"train/return/mean": 0.5}, step=256, source="train")
    frame = store.pending_metric_frames(limit=1)[0]
    with _server(tmp_path, port):
        first = MlflowDelivery.open(
            tracking_uri=uri, experiment_name="gradlab-test",
            gradlab_run_id=run_id, created_at_ms=1_800_000_000_000,
        )
        first.publish_frame(frame)
        first._artifact(sequence=1, name="summary.json", content=b'{"score":0.5}')
        service_run_id = first.run_id
    with _server(tmp_path, port):
        resumed = MlflowDelivery.open(
            tracking_uri=uri, experiment_name="gradlab-test",
            gradlab_run_id=run_id, created_at_ms=1_800_000_000_000,
        )
        assert resumed.run_id == service_run_id
        resumed.publish_frame(frame)
        resumed._artifact(sequence=1, name="summary.json", content=b'{"score":0.5}')
        history = MlflowClient(tracking_uri=uri).get_metric_history(
            service_run_id, "train/return/mean"
        )
        assert [(point.value, point.step) for point in history] == [(0.5, 256)]
        assert resumed.remote_high_water() == 1
        client = MlflowClient(tracking_uri=uri)
        assert client.list_artifacts(service_run_id) == []
        reference = json.loads(client.get_run(service_run_id).data.tags["gradlab.asset.000000000001"])
        assert Path(reference["url"].removeprefix("file://")).read_bytes() == b'{"score":0.5}'


def test_direct_local_online_writer_drains_to_real_mlflow_store(
    tmp_path: Path, monkeypatch
) -> None:
    uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(
        "gradlab.local_metrics.load_repository_operator_environment", lambda *a, **k: None
    )
    monkeypatch.setattr("gradlab.local_metrics.env_config_from_mapping", lambda _c: None)
    monkeypatch.setattr(
        "gradlab.local_metrics.resolve_env_config",
        lambda _c: SimpleNamespace(game="bandit"),
    )
    monkeypatch.setattr("gradlab.local_metrics.local_publication", lambda *a: nullcontext(None))
    run_id = "gradlab-" + "c" * 32
    (tmp_path / "local-run.json").write_text(
        json.dumps({"started_at": "2026-09-29T12:00:00Z"})
    )
    store = MetricStore(tmp_path / "gradlab.sqlite")
    store.init()
    with local_metrics_writer(tmp_path, {"wandb_run_id": run_id, "tracking": {"backend": "mlflow", "delivery": "online"}}) as url:
        assert url.startswith("mlflow:")
        store.append_metrics({"train/return/mean": 0.5}, step=64, source="train")
        (tmp_path / "training-result.json").write_text(
            json.dumps({"status": "completed", "terminal_reason": "resource_exhaustion"})
        )
    delivery = json.loads((tmp_path / "metrics-delivery.json").read_text())
    assert delivery["high_water"] == 1
    client = MlflowClient(tracking_uri=uri)
    history = client.get_metric_history(delivery["service_run_id"], "train/return/mean")
    assert [(point.value, point.step) for point in history] == [(0.5, 64)]


def test_later_mlflow_sync_preserves_original_local_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(
        "gradlab.local_metrics.load_repository_operator_environment", lambda *a, **k: None
    )
    monkeypatch.setattr("gradlab.local_metrics.env_config_from_mapping", lambda _c: None)
    monkeypatch.setattr(
        "gradlab.local_metrics.resolve_env_config",
        lambda _c: SimpleNamespace(game="bandit"),
    )
    run_id = "gradlab-" + "e" * 32
    original = {"run_id": run_id, "status": "complete_local", "started_at": "2026-09-29T12:00:00Z"}
    receipt = tmp_path / "local-run.json"
    receipt.write_text(json.dumps(original))
    original_bytes = receipt.read_bytes()
    store = MetricStore(tmp_path / "gradlab.sqlite")
    store.init()
    store.append_metrics({"train/return/mean": 0.75}, step=256, source="train", publish=False)
    config = {"wandb_run_id": run_id, "tracking": {"backend": "mlflow", "delivery": "local_only"}}
    (tmp_path / "train-config.json").write_text(json.dumps(config))
    first = sync_local_run(tmp_path)
    second = sync_local_run(tmp_path)
    assert first == second
    assert receipt.read_bytes() == original_bytes
    evidence = json.loads((tmp_path / "tracker-sync.json").read_text())
    history = MlflowClient(tracking_uri=uri).get_metric_history(
        evidence["service_run_id"], "train/return/mean"
    )
    assert [(point.value, point.step) for point in history] == [(0.75, 256)]


def test_mlflow_sqlite_snapshot_restores_run_identity_and_history(tmp_path: Path) -> None:
    database = tmp_path / "tracking.db"
    uri = f"sqlite:///{database}"
    run_id = "gradlab-" + "f" * 32
    store = MetricStore(tmp_path / "journal.sqlite")
    store.init()
    store.append_metrics({"train/return/mean": 0.9}, step=512, source="train")
    delivery = MlflowDelivery.open(
        tracking_uri=uri, experiment_name="gradlab-test",
        gradlab_run_id=run_id, created_at_ms=1_800_000_000_000,
    )
    delivery.publish_frame(store.pending_metric_frames(limit=1)[0])
    snapshot = tmp_path / "restored.db"
    script = Path(__file__).parents[1] / "ops" / "mlflow" / "sqlite_snapshot.py"
    subprocess.run([sys.executable, str(script), "backup", str(database), str(snapshot)], check=True)
    receipt = json.loads((tmp_path / "restored.db.json").read_text())
    assert receipt["size_bytes"] == snapshot.stat().st_size
    assert snapshot.stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "restored.db.json").stat().st_mode & 0o777 == 0o600
    restored = MlflowDelivery.open(
        tracking_uri=f"sqlite:///{snapshot}", experiment_name="gradlab-test",
        gradlab_run_id=run_id, created_at_ms=1_800_000_000_000,
    )
    assert restored.run_id == delivery.run_id
    history = restored.client.get_metric_history(restored.run_id, "train/return/mean")
    assert [(point.value, point.step) for point in history] == [(0.9, 512)]


def test_irrecoverable_mlflow_replacement_replays_journal_without_rewriting_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    storage = RunStorageConfig(
        control=BucketConfig((tmp_path / "control").as_uri()),
        evaluation=BucketConfig((tmp_path / "eval").as_uri()),
        models=BucketConfig(
            (tmp_path / "models").as_uri(), public_base_url="https://models.example.test"
        ),
    )
    authority = RunAuthority(storage)
    base = _manifest_only_run()
    manifest = replace(
        base,
        tracking={
            "backend": "mlflow", "delivery": "online", "operator_profile": "pilot",
            "sources": {"backend": "launch override", "delivery": "built-in default"},
        },
        wandb={},
    )
    authority.create_manifest(manifest)
    authority.seal_metric_segment(
        run_id=manifest.run_id,
        attempt_id=manifest.attempt_id,
        events=[{
            "event_seq": 1, "event_id": "event-1", "kind": "history", "step": 128,
            "source": "train", "payload": {"train/return/mean": 0.75},
        }],
    )
    terminal_key = f"runs/{manifest.run_id}/attempts/{manifest.attempt_id}/terminal.json"
    terminal = {
        "state": "succeeded", "stop_reason": "training_cap_complete",
        "completed_at": "2026-09-29T12:00:00Z", "drain": {"metric_segment_high_water": 1},
    }
    authority.control.put_json(terminal_key, terminal)
    monkeypatch.setattr(
        authority, "recipe_document",
        lambda _digest: {"recipe": {"train_config": {"game_family": "bandit", "selection_rank": []}}},
    )
    old = MlflowDelivery.open(
        tracking_uri=f"sqlite:///{tmp_path / 'old.db'}", experiment_name="gradlab-bandit",
        gradlab_run_id=manifest.run_id, created_at_ms=1_800_000_000_000,
    )
    binding_key = f"runs/{manifest.run_id}/metrics-binding.json"
    authority.control.put_json(binding_key, {
        "schema_version": 1, "run_id": manifest.run_id, "backend": "mlflow",
        "service_run_id": old.run_id, "service_url": old.service_url,
    })
    result = replace_mlflow_binding(
        authority, run_id=manifest.run_id,
        tracking_uri=f"sqlite:///{tmp_path / 'replacement.db'}",
        operator="test-operator", reason="irrecoverable tracking database loss",
    )
    assert result["service_run_id"] != old.run_id
    assert authority.control.get_json(terminal_key) == terminal
    assert authority.control.get_json(binding_key)["audit_key"] == result["audit_key"]
    assert authority.control.get_json(result["audit_key"])["journal_high_water"] == 1
    history = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'replacement.db'}").get_metric_history(
        result["service_run_id"], "train/return/mean"
    )
    assert [(point.value, point.step) for point in history] == [(0.75, 128)]
