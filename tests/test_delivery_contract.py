from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gradlab.metric_store import MetricStore
from gradlab.mlflow_delivery import MlflowDelivery
from gradlab.r2_store import BucketConfig, R2Bucket
from gradlab.selected_delivery import DeliveryAdapter, publish_outbox
from gradlab.wandb_publisher import WandbProjector


@pytest.fixture(autouse=True)
def public_assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    uri = (tmp_path / "public-assets").as_uri()
    monkeypatch.setenv("GRADLAB_MODELS_R2_URI", uri)
    monkeypatch.setenv("GRADLAB_MODELS_R2_PUBLIC_BASE_URL", uri)


class _WandbRun:
    def __init__(self, run_id: str, directory: Path):
        self.id = run_id
        self.dir = str(directory)
        self.path = f"test/project/{run_id}"
        self.summary: dict = {}
        self.frames: list[tuple[int, dict]] = []
        self.finished = False
        self.exit_code = None

    def log(self, payload, *, step):
        self.frames.append((step, dict(payload)))
        self.summary.update(payload)

    def define_metric(self, *_args, **_kwargs):
        pass

    def finish(self, **kwargs):
        self.finished = True
        self.exit_code = kwargs.get("exit_code")


@pytest.mark.parametrize("backend", ["wandb", "mlflow"])
def test_selected_service_adapter_contract_has_same_scientific_projection(
    backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "gradlab-" + "a" * 32
    store = MetricStore(tmp_path / "journal.sqlite")
    store.init()
    store.append_metrics({"train/return/mean": 0.75}, step=128, source="train")
    video = b"contract-video-bytes"
    bucket = R2Bucket(BucketConfig(uri=(tmp_path / "eval-r2").as_uri()))
    bucket.put_bytes("runs/contract/episode.mp4", video)
    store.enqueue_event(
        kind="evaluation_video", source="eval:video", step=128,
        payload={"bucket_uri": bucket.config.uri, "key": "runs/contract/episode.mp4",
                 "bytes": len(video), "sha256": hashlib.sha256(video).hexdigest(),
                 "scratch_headroom_bytes": 0},
    )
    if backend == "mlflow":
        adapter: DeliveryAdapter = MlflowDelivery.open(
            tracking_uri=f"sqlite:///{tmp_path / 'tracking.db'}",
            experiment_name="gradlab-test",
            gradlab_run_id=run_id,
            created_at_ms=1_800_000_000_000,
        )
    else:
        run = _WandbRun(run_id, tmp_path)
        monkeypatch.setattr(
            "wandb.Api",
            lambda **_kwargs: SimpleNamespace(run=lambda _path: SimpleNamespace(summary=run.summary)),
        )
        monkeypatch.setattr(
            "wandb.Video", lambda path, *, format: Path(path).read_bytes()
        )
        adapter = WandbProjector(run)
    assert isinstance(adapter, DeliveryAdapter)
    if backend == "wandb":
        assert adapter.run_id == run_id
    else:
        assert adapter.remote_summary()["gradlab.run_id"] == run_id
    assert publish_outbox(store, adapter, limit=10) == 2
    assert adapter.remote_high_water() == 2
    summary = adapter.remote_summary()
    assert summary["train/return/mean"] == 0.75
    if backend == "mlflow":
        assert adapter.client.list_artifacts(adapter.run_id) == []
        reference = json.loads(summary["gradlab.asset.000000000002"])
        assert reference["bytes"] == len(video)
        assert reference["sha256"] == hashlib.sha256(video).hexdigest()
        assert Path(reference["url"].removeprefix("file://")).read_bytes() == video
        assert adapter.client.get_metric_history(adapter.run_id, "train/return/mean")[0].step == 128
    else:
        assert run.frames[1][1]["eval/video"] == video
    adapter.publish_promotion(
        checkpoint_step=128,
        checkpoint_url="https://models.example.test/checkpoint.zip",
        metrics={"eval/return/mean": 0.8},
        updated_at="2026-09-29T12:00:00Z",
        selection_rank=["max(eval/return/mean)"],
        evaluation_source="verified:checkpoint",
    )
    assert int(adapter.remote_summary()["leader/step"]) == 128
    adapter.publish_terminal(state="complete_local", reason="budget_exhausted")
    assert adapter.remote_summary()["ops/state"] == "complete_local"
    if backend == "wandb":
        assert run.finished and run.exit_code == 0


def test_wandb_adapter_marks_failed_terminal_as_failed() -> None:
    run = _WandbRun("gradlab-" + "f" * 32, Path("."))
    WandbProjector(run).publish_terminal(state="resumable_failure", reason="delivery_timeout")
    assert run.summary["ops/state"] == "resumable_failure"
    assert run.exit_code == 1


def test_mlflow_media_failure_keeps_frame_unacknowledged_for_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = MetricStore(tmp_path / "journal.sqlite")
    store.init()
    payload = {"bucket_uri": (tmp_path / "eval-r2").as_uri(),
               "key": "runs/episode.mp4", "bytes": 3,
               "sha256": hashlib.sha256(b"mp4").hexdigest()}
    R2Bucket(BucketConfig(payload["bucket_uri"])).put_bytes(payload["key"], b"mp4")
    store.enqueue_event(kind="evaluation_video", source="eval:video", step=128,
                        payload=payload)
    delivery = MlflowDelivery.open(
        tracking_uri=f"sqlite:///{tmp_path / 'tracking.db'}",
        experiment_name="gradlab-test", gradlab_run_id="gradlab-" + "c" * 32,
        created_at_ms=1_800_000_000_000,
    )
    original = delivery._r2_artifact
    calls = 0

    def uncertain_artifact(**kwargs):
        nonlocal calls
        calls += 1
        original(**kwargs)
        if calls == 1:
            raise ConnectionError("artifact reply was lost")

    monkeypatch.setattr(delivery, "_r2_artifact", uncertain_artifact)
    assert publish_outbox(store, delivery, limit=10) == 0
    assert delivery.remote_high_water() == 0
    assert publish_outbox(store, delivery, limit=10) == 1
    assert delivery.remote_high_water() == 1
    assert delivery.client.get_metric_history(delivery.run_id, "ops/sequence")[0].value == 1
    assert delivery.client.list_artifacts(delivery.run_id) == []
    reference = json.loads(delivery.remote_summary()["gradlab.asset.000000000001"])
    assert reference["sha256"] == payload["sha256"]
    assert Path(reference["url"].removeprefix("file://")).read_bytes() == b"mp4"
