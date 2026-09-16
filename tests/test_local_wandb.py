from contextlib import nullcontext
import json
from types import SimpleNamespace
from unittest import mock

import pytest

from gradlab.local_wandb import local_wandb_writer, sync_local_run
from gradlab.metric_store import MetricStore


class FakeRun:
    id = "gradlab-" + "a" * 32
    url = "https://wandb.ai/test/FrozenLake8x8-v1/runs/" + id
    path = "test/FrozenLake8x8-v1/" + id

    def __init__(self):
        self.summary = {}
        self.frames = []

    def define_metric(self, *args, **kwargs):
        pass

    def log(self, payload, *, step):
        self.frames.append((step, payload))
        self.summary["ops/sequence"] = step


@pytest.fixture
def transport():
    projector = SimpleNamespace(run=FakeRun(), close=mock.Mock())
    with (
        mock.patch(
            "gradlab.local_wandb.WandbProjector.start_live", return_value=projector
        ) as start,
        mock.patch("gradlab.local_wandb.resolve_env_config"),
        mock.patch("gradlab.local_wandb.local_publication", return_value=nullcontext(None)),
        mock.patch("gradlab.local_wandb._verify_remote_delivery") as verify,
    ):
        yield projector, start, verify


def test_local_writer_publishes_real_outbox_and_confirms_terminal_delivery(tmp_path, transport):
    projector, start, verify = transport
    store = MetricStore(tmp_path / "gradlab.sqlite")
    with local_wandb_writer(tmp_path, {"wandb_mode": "online"}):
        store.append_metrics({"train/return/mean": 0.25}, step=4096, source="training")
        (tmp_path / "training-result.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "terminal_reason": "early_stop_neutral",
                }
            )
        )
    assert projector.run.frames == [
        (
            1,
            {
                "train/return/mean": 0.25,
                "train/step": 4096,
                "ops/sequence": 1,
            },
        )
    ]
    assert store.pending_metric_frames() == []
    projector.close.assert_called_once_with(timeout_seconds=60, exit_code=0)
    verify.assert_called_once_with(projector.run.path, 1)
    assert json.loads((tmp_path / "wandb-delivery.json").read_text())["status"] == "delivered"
    assert projector.run.summary["ops/reason"] == "early_stop_neutral"


def test_explicit_opt_out_never_initializes_wandb(tmp_path, transport):
    _, start, _ = transport
    with local_wandb_writer(tmp_path, {"wandb_mode": "disabled"}) as url:
        assert url is None
    start.assert_not_called()


def test_second_writer_cannot_acquire_same_run(tmp_path, transport):
    with local_wandb_writer(tmp_path, {"wandb_mode": "online"}):
        with pytest.raises(RuntimeError, match="already owns"):
            with local_wandb_writer(tmp_path, {"wandb_mode": "online"}):
                pass


def test_delivery_failure_does_not_write_success_receipt(tmp_path, transport):
    _, _, verify = transport
    verify.side_effect = TimeoutError("remote delivery incomplete")
    with pytest.raises(TimeoutError, match="remote delivery incomplete"):
        with local_wandb_writer(tmp_path, {"wandb_mode": "online"}):
            pass
    assert not (tmp_path / "wandb-delivery.json").exists()


def test_backfill_keeps_original_contract_and_reuses_identity(tmp_path, transport):
    original = json.dumps({"wandb_mode": "disabled", "run_name": "local/maze"})
    (tmp_path / "train-config.json").write_text(original)
    (tmp_path / "local-run.json").write_text('{"status":"completed"}')
    store = MetricStore(tmp_path / "gradlab.sqlite")
    store.init()
    store.append_metrics({"train/return/mean": 0.0}, step=100, source="training", publish=False)
    sync_local_run(tmp_path)
    identity = json.loads((tmp_path / "wandb-backfill.json").read_text())
    sync_local_run(tmp_path)
    assert json.loads((tmp_path / "wandb-backfill.json").read_text()) == identity
    assert (tmp_path / "train-config.json").read_text() == original
    assert len(transport[0].run.frames) == 1


def test_backfill_rejects_active_training(tmp_path):
    (tmp_path / "local-run.json").write_text('{"status":"running"}')
    with pytest.raises(ValueError, match="running"):
        sync_local_run(tmp_path)


def test_recovery_replays_sdk_acknowledged_frames_missing_remotely(tmp_path, transport):
    store = MetricStore(tmp_path / "gradlab.sqlite")
    store.init()
    frame = store.append_metrics({"train/return/mean": 0.5}, step=200, source="training")
    assert store.claim_metric_frame(frame)
    store.mark_metric_frame_published(frame, step=200)
    assert transport[0].run.summary == {}
    with local_wandb_writer(tmp_path, {"wandb_mode": "online"}):
        pass
    assert len(transport[0].run.frames) == 1
    assert transport[0].run.frames[0][0] == frame


def test_learner_exception_still_drains_metrics_and_closes_failed_run(tmp_path, transport):
    store = MetricStore(tmp_path / "gradlab.sqlite")
    with pytest.raises(ValueError, match="learner failure"):
        with local_wandb_writer(tmp_path, {"wandb_mode": "online"}):
            store.append_metrics({"train/return/mean": 0.1}, step=100, source="training")
            raise ValueError("learner failure")
    assert len(transport[0].run.frames) == 1
    transport[0].close.assert_called_once_with(timeout_seconds=60, exit_code=1)
