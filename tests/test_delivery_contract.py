from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gradlab.metric_store import MetricStore
from gradlab.mlflow_delivery import MlflowDelivery
from gradlab.selected_delivery import DeliveryAdapter, publish_outbox
from gradlab.wandb_publisher import WandbProjector


class _WandbRun:
    def __init__(self, run_id: str):
        self.id = run_id
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
    if backend == "mlflow":
        adapter: DeliveryAdapter = MlflowDelivery.open(
            tracking_uri=f"sqlite:///{tmp_path / 'tracking.db'}",
            experiment_name="gradlab-test",
            gradlab_run_id=run_id,
            created_at_ms=1_800_000_000_000,
        )
    else:
        run = _WandbRun(run_id)
        monkeypatch.setattr(
            "wandb.Api",
            lambda **_kwargs: SimpleNamespace(run=lambda _path: SimpleNamespace(summary=run.summary)),
        )
        adapter = WandbProjector(run)
    assert isinstance(adapter, DeliveryAdapter)
    assert publish_outbox(store, adapter, limit=10) == 1
    assert adapter.remote_high_water() == 1
    summary = adapter.remote_summary()
    assert summary["train/return/mean"] == 0.75
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
    run = _WandbRun("gradlab-" + "f" * 32)
    WandbProjector(run).publish_terminal(state="resumable_failure", reason="delivery_timeout")
    assert run.summary["ops/state"] == "resumable_failure"
    assert run.exit_code == 1
