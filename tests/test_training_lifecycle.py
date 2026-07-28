from __future__ import annotations

import json
from pathlib import Path

import pytest

from gradlab.batch_runtime import EpisodeRecord
from gradlab.metric_names import (
    TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN,
    TRAIN_EPISODE_RETURN_SHAPED_MEAN,
    TRAIN_OUTCOME_SUCCESS_CURRENT_RATE_MEAN,
    TRAIN_OUTCOME_SUCCESS_START_COVERAGE_RATE,
    TRAIN_OUTCOME_TERMINAL_COUNT,
    train_outcome_reason_count_metric,
    train_success_attempts_metric,
    train_success_count_metric,
)
from gradlab.task_kernels import Outcome
from gradlab.training_backend import GracefulStopFlag
from gradlab.training_lifecycle import (
    TRAINING_RESULT_FILENAME,
    CheckpointCoordinator,
    PlainProgressSink,
    TerminalReason,
    TrainingBudget,
    TrainingLifecycleOptions,
    TrainingResult,
    TrainingSession,
)
from gradlab.training_metrics import EpisodeMetricsReducer


class FakeMetricStore:
    def __init__(self) -> None:
        self.frames: list[tuple[dict, dict]] = []
        self.checkpoints: list[dict] = []

    def append_metrics(self, payload, **metadata):
        self.frames.append((dict(payload), dict(metadata)))

    def record_checkpoint(self, **metadata):
        self.checkpoints.append(dict(metadata))
        return len(self.checkpoints)


class MemoryProgressSink:
    def __init__(self) -> None:
        self.total = 0
        self.steps: list[int] = []
        self.metrics: list[dict] = []
        self.events: list[str] = []
        self.closed = False

    def start(self, *, total: int, initial: int, description: str) -> None:
        del description
        self.total = total
        self.steps.append(initial)
        self.metrics.append({})

    def update(self, *, step: int, metrics, final: bool = False) -> None:
        del final
        self.steps.append(step)
        self.metrics.append(dict(metrics))

    def event(self, message: str) -> None:
        self.events.append(message)

    def close(self) -> None:
        self.closed = True


def _episode(
    *,
    start: str,
    episode_return: float,
    outcome: Outcome,
    origin: str = "target",
    events: tuple[str, ...] = (),
) -> EpisodeRecord:
    return EpisodeRecord(
        lane=0,
        episode_index=0,
        start_id=start,
        episode_return=episode_return,
        episode_length=10,
        terminated=True,
        truncated=False,
        outcome=outcome,
        events=events,
        metrics={},
        start_origin=origin,
    )


def test_training_budget_exposes_requested_and_safe_execution_totals() -> None:
    budget = TrainingBudget.aligned(
        requested_limit=50_000_000,
        step_quantum=8_192,
    )

    assert budget.requested_limit == 50_000_000
    assert budget.execution_total == 50_003_968
    assert budget.step_quantum == 8_192


def test_plain_progress_is_bounded_and_uses_only_canonical_outcomes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    now = [0.0]
    sink = PlainProgressSink(interval_seconds=10.0, clock=lambda: now[0])
    sink.start(total=100, initial=0, description="gradlab.jerk")
    capsys.readouterr()

    metrics = {
        TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN: 5.0,
        TRAIN_OUTCOME_SUCCESS_CURRENT_RATE_MEAN: 0.25,
        "train/algorithm/go_explore/best/progress": 999.0,
    }
    sink.update(step=10, metrics=metrics)
    assert capsys.readouterr().out == ""

    now[0] = 10.0
    sink.update(step=20, metrics=metrics)
    output = capsys.readouterr().out
    assert "mean return=5" in output
    assert "completion=25.00%" in output
    assert "best" not in output


def test_episode_metrics_are_identical_across_target_and_archive_consumers() -> None:
    reducer = EpisodeMetricsReducer(
        event_names=("life_loss",),
        configured_starts=("StartA", "StartB"),
        track_success=True,
    )
    payload = reducer.consume(
        (
            _episode(start="StartA", episode_return=10.0, outcome=Outcome.SUCCESS),
            _episode(
                start="StartB",
                episode_return=2.0,
                outcome=Outcome.FAILURE,
                events=("life_loss",),
            ),
            _episode(
                start="StartA",
                episode_return=30.0,
                outcome=Outcome.SUCCESS,
                origin="archive",
            ),
        )
    )

    assert payload[TRAIN_EPISODE_RETURN_SHAPED_MEAN] == 14.0
    assert payload[TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN] == 6.0
    assert payload[TRAIN_OUTCOME_TERMINAL_COUNT] == 3
    assert payload[train_outcome_reason_count_metric("life_loss")] == 1
    assert payload[train_success_attempts_metric("StartA")] == 1
    assert payload[train_success_count_metric("StartA")] == 1
    assert payload[train_success_attempts_metric("StartB")] == 1
    assert payload[TRAIN_OUTCOME_SUCCESS_CURRENT_RATE_MEAN] == 0.5
    assert payload[TRAIN_OUTCOME_SUCCESS_START_COVERAGE_RATE] == 1.0


@pytest.mark.parametrize(
    "backend_id",
    ("sb3.ppo", "sb3.a2c", "gradlab.jerk", "gradlab.go-explore"),
)
def test_shared_session_enforces_backend_conformance(
    tmp_path: Path,
    backend_id: str,
) -> None:
    run_dir = tmp_path / backend_id
    run_dir.mkdir(parents=True)
    store = FakeMetricStore()
    progress = MemoryProgressSink()
    session = TrainingSession(
        run_dir=run_dir,
        backend_id=backend_id,
        metric_store=store,
        wandb_enabled=False,
        stop_flag=GracefulStopFlag(),
        early_stop_config=None,
        attempt_id="attempt-test",
        reducer=EpisodeMetricsReducer(
            configured_starts=("StartA",),
            track_success=True,
        ),
        options=TrainingLifecycleOptions(console_mode="silent"),
        progress_sink=progress,
    )
    session.configure_checkpoints(run_name="run-test", eval_required=False)
    budget = session.configure_budget(requested_limit=10, step_quantum=4)

    assert budget.execution_total == 12
    session.mark_ready()
    with pytest.raises(RuntimeError, match="only once"):
        session.mark_ready()

    record = _episode(start="StartA", episode_return=5.0, outcome=Outcome.SUCCESS)
    session.advance(4, (record,))
    session.report(step=4, metrics={"train/throughput/loop_fps": 100.0})
    session.report(step=4, metrics={"train/throughput/loop_fps": 101.0})

    assert len(store.frames) == 1
    assert store.frames[0][1]["source"] == "train"
    assert store.frames[0][0][TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN] == 5.0
    with pytest.raises(RuntimeError, match="regressed"):
        session.advance(3)
    with pytest.raises(RuntimeError, match="exceeded"):
        session.advance(13)

    session.finalize(
        TrainingResult(
            reason=TerminalReason.RESOURCE_LIMIT,
            step=4,
            model_kind="final",
        )
    )
    assert progress.closed is True
    result = json.loads((run_dir / TRAINING_RESULT_FILENAME).read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    assert result["terminal_reason"] == "resource_limit"


@pytest.mark.parametrize("persist_intermediate", (False, True))
def test_checkpoint_policy_is_centralized_and_deduplicated(
    tmp_path: Path,
    persist_intermediate: bool,
) -> None:
    store = FakeMetricStore()
    events: list[str] = []
    coordinator = CheckpointCoordinator(
        metric_store=store,
        run_name="run-test",
        eval_required=True,
        publish=False,
        persist_intermediate=persist_intermediate,
        event=events.append,
    )
    saves: list[tuple[Path, str, int]] = []

    def save_bundle(path: Path, kind: str, step: int) -> Path:
        saves.append((path, kind, step))
        path.write_bytes(b"model")
        return path

    checkpoint = coordinator.save(
        kind="checkpoint",
        step=4,
        model_path=tmp_path / "checkpoints" / "model.zip",
        save_bundle=save_bundle,
    )
    duplicate = coordinator.save(
        kind="checkpoint",
        step=4,
        model_path=tmp_path / "checkpoints" / "duplicate.zip",
        save_bundle=save_bundle,
    )
    terminal = coordinator.save(
        kind="final",
        step=8,
        model_path=tmp_path / "final_model.zip",
        save_bundle=save_bundle,
        terminal=True,
    )

    if persist_intermediate:
        assert checkpoint == duplicate
        assert [row["kind"] for row in store.checkpoints] == ["checkpoint", "final"]
        assert len(saves) == 2
    else:
        assert checkpoint is None
        assert duplicate is None
        assert [row["kind"] for row in store.checkpoints] == ["final"]
        assert len(saves) == 1
        assert not (tmp_path / "checkpoints").exists()
    assert terminal == tmp_path / "final_model.zip"
    with pytest.raises(RuntimeError, match="already exists"):
        coordinator.save(
            kind="final",
            step=8,
            model_path=tmp_path / "another-final.zip",
            save_bundle=save_bundle,
            terminal=True,
        )
