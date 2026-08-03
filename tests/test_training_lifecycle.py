from __future__ import annotations

import json
from pathlib import Path

import pytest

from gradlab.batch_runtime import EpisodeRecord
from gradlab.metric_names import (
    TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_ROLLING_UP_TO_100_MAX,
    TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_ROLLING_UP_TO_100_MEAN,
    TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_WINDOW_100_MEAN,
    TRAIN_EPISODE_RETURN_SHAPED_ACROSS_ORIGINS_ROLLING_UP_TO_100_MAX,
    TRAIN_EPISODE_RETURN_SHAPED_ACROSS_ORIGINS_ROLLING_UP_TO_100_MEAN,
    TRAIN_OUTCOME_SUCCESS_ACROSS_OBSERVED_STARTS_CUMULATIVE_RATE_MEAN,
    TRAIN_OUTCOME_SUCCESS_ACROSS_STARTS_COVERAGE_RATE,
    TRAIN_EPISODE_COMPLETED_COUNT,
    train_outcome_reason_count_metric,
    train_early_stop_metric,
    train_success_attempts_metric,
    train_success_count_metric,
)
from gradlab.task_kernels import Outcome
from gradlab.training_backend import GracefulStopFlag
from gradlab.training_lifecycle import (
    TRAINING_RESULT_FILENAME,
    CheckpointCoordinator,
    PlainProgressSink,
    ProgressField,
    ProgressValueFormat,
    TerminalReason,
    TrainingBudget,
    TrainingExecutionMode,
    TrainingExecutionPolicy,
    TrainingSession,
    progress_sink_for_mode,
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
        self.fields: tuple[ProgressField, ...] = ()

    def start(
        self,
        *,
        total: int,
        initial: int,
        description: str,
        fields=(),
    ) -> None:
        del description
        self.total = total
        self.fields = tuple(fields)
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


def test_progress_fields_require_a_nonempty_presentation_group() -> None:
    with pytest.raises(ValueError, match="group must be non-empty"):
        ProgressField("train/test/value", "test value", group=" ")


def test_threshold_target_progress_is_published_and_shown_as_an_outcome(
    tmp_path: Path,
) -> None:
    progress = MemoryProgressSink()
    store = FakeMetricStore()
    session = TrainingSession(
        run_dir=tmp_path,
        backend_id="sb3.ppo",
        metric_store=store,
        wandb_enabled=False,
        stop_flag=GracefulStopFlag(),
        early_stop_config={
            "conditions": {
                "target_reached": {
                    "metric": TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_WINDOW_100_MEAN,
                    "trigger": "threshold",
                    "operator": ">=",
                    "progress_baseline": 0.0,
                    "threshold": 10.0,
                    "patience_steps": 0,
                    "outcome": "success",
                    "action": "stop",
                }
            }
        },
        attempt_id="attempt-test",
        run_id="run-test",
        reducer=EpisodeMetricsReducer(track_success=False),
        execution_policy=TrainingExecutionPolicy.for_mode(TrainingExecutionMode.SUPERVISED),
        completion_signal_available=False,
        progress_sink=progress,
    )

    session.configure_budget(requested_limit=10, step_quantum=1)
    metric = train_early_stop_metric("target_reached", "target/progress")
    assert progress.fields == (
        ProgressField(
            metric,
            "target progress",
            ProgressValueFormat.PERCENT,
            group="outcomes",
        ),
    )

    session.report(
        step=1,
        metrics={
            TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_WINDOW_100_MEAN: 5.0,
        },
    )

    assert store.frames[-1][0][metric] == 0.5
    assert progress.metrics[-1][metric] == 0.5


def test_neutral_plateau_has_a_typed_learner_terminal_reason(tmp_path: Path) -> None:
    session = TrainingSession(
        run_dir=tmp_path,
        backend_id="sb3.ppo",
        metric_store=FakeMetricStore(),
        wandb_enabled=False,
        stop_flag=GracefulStopFlag(),
        early_stop_config={
            "conditions": {
                "return_plateau": {
                    "metric": TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_WINDOW_100_MEAN,
                    "trigger": "no_improvement",
                    "direction": "maximize",
                    "min_delta": 0.01,
                    "delta_mode": "relative",
                    "start_after_steps": 0,
                    "patience_steps": 1,
                    "outcome": "neutral",
                    "action": "stop",
                }
            }
        },
        attempt_id="attempt-test",
        run_id="run-test",
        reducer=EpisodeMetricsReducer(track_success=False),
        execution_policy=TrainingExecutionPolicy.for_mode(TrainingExecutionMode.SUPERVISED),
        completion_signal_available=False,
        progress_sink=MemoryProgressSink(),
    )
    session.configure_budget(requested_limit=10, step_quantum=1)

    session.report(
        step=1,
        metrics={TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_WINDOW_100_MEAN: 5.0},
    )
    session.report(
        step=2,
        metrics={TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_WINDOW_100_MEAN: 5.0},
    )

    assert session.terminal_reason() == TerminalReason.EARLY_STOP_NEUTRAL


def test_execution_modes_resolve_to_fixed_lifecycle_policies() -> None:
    local = TrainingExecutionPolicy.for_mode(TrainingExecutionMode.LOCAL_DEMO)
    supervised = TrainingExecutionPolicy.for_mode(TrainingExecutionMode.SUPERVISED)

    assert local.to_document() == {
        "mode": "local-demo",
        "console_mode": "auto",
        "persist_intermediate_checkpoints": False,
        "stop_on_first_completion": True,
        "handle_sigint": True,
    }
    assert supervised.to_document() == {
        "mode": "supervised",
        "console_mode": "plain",
        "persist_intermediate_checkpoints": True,
        "stop_on_first_completion": False,
        "handle_sigint": False,
    }


def _session(
    tmp_path: Path,
    *,
    mode: TrainingExecutionMode,
    completion_signal_available: bool = True,
) -> tuple[TrainingSession, GracefulStopFlag]:
    suffix = "with-signal" if completion_signal_available else "without-signal"
    run_dir = tmp_path / f"{mode.value}-{suffix}"
    run_dir.mkdir(parents=True)
    stop_flag = GracefulStopFlag()
    session = TrainingSession(
        run_dir=run_dir,
        backend_id="sb3.ppo",
        metric_store=FakeMetricStore(),
        wandb_enabled=False,
        stop_flag=stop_flag,
        early_stop_config=None,
        attempt_id="attempt-test",
        run_id="run-test",
        reducer=EpisodeMetricsReducer(
            configured_starts=("StartA",),
            track_success=completion_signal_available,
        ),
        execution_policy=TrainingExecutionPolicy.for_mode(mode),
        completion_signal_available=completion_signal_available,
        progress_sink=MemoryProgressSink(),
    )
    session.configure_budget(requested_limit=10, step_quantum=4)
    return session, stop_flag


def test_local_completion_records_observation_and_safe_boundary_steps(tmp_path: Path) -> None:
    session, stop_flag = _session(tmp_path, mode=TrainingExecutionMode.LOCAL_DEMO)
    success = _episode(start="StartA", episode_return=10.0, outcome=Outcome.SUCCESS)

    session.advance(2, (success,))
    assert session.observe_episode_completions(step=2, records=(success,)) is True
    assert stop_flag.reason == "first_completion:2"

    # PPO/A2C finish the current rollout and update before returning.
    session.advance(4)
    result = session.result(
        terminal_reason=session.terminal_reason(),
        final_step=4,
        model_kind="final",
    )
    assert result.terminal_reason == TerminalReason.FIRST_COMPLETION
    assert result.first_completion_step == 2
    assert result.final_step == 4


def test_archive_success_and_missing_success_signals_do_not_stop_local_training(
    tmp_path: Path,
) -> None:
    archive_session, archive_flag = _session(
        tmp_path,
        mode=TrainingExecutionMode.LOCAL_DEMO,
    )
    archive_success = _episode(
        start="StartA",
        episode_return=10.0,
        outcome=Outcome.SUCCESS,
        origin="archive",
    )
    assert archive_session.observe_episode_completions(step=2, records=(archive_success,)) is False
    assert archive_flag.requested is False

    no_signal_session, no_signal_flag = _session(
        tmp_path,
        mode=TrainingExecutionMode.LOCAL_DEMO,
        completion_signal_available=False,
    )
    assert (
        no_signal_session.observe_episode_completions(
            step=2,
            records=(
                _episode(
                    start="StartA",
                    episode_return=10.0,
                    outcome=Outcome.SUCCESS,
                ),
            ),
        )
        is False
    )
    assert no_signal_flag.requested is False
    assert no_signal_session.terminal_reason() == TerminalReason.RESOURCE_EXHAUSTION


def test_supervised_completion_continues_and_signal_reason_is_not_acceptance(
    tmp_path: Path,
) -> None:
    session, stop_flag = _session(tmp_path, mode=TrainingExecutionMode.SUPERVISED)
    success = _episode(start="StartA", episode_return=10.0, outcome=Outcome.SUCCESS)

    assert session.observe_episode_completions(step=2, records=(success,)) is False
    assert session.first_completion_step == 2
    assert stop_flag.requested is False
    stop_flag.request("SIGUSR1")
    assert session.terminal_reason() == TerminalReason.EXTERNAL_SIGNAL
    assert session.terminal_model_kind(TerminalReason.EXTERNAL_SIGNAL) == "final"
    assert session.should_persist_interrupted_checkpoint(TerminalReason.EXTERNAL_SIGNAL) is True


def test_completion_has_priority_over_early_stop_at_the_same_boundary(tmp_path: Path) -> None:
    session, _stop_flag = _session(tmp_path, mode=TrainingExecutionMode.LOCAL_DEMO)
    session.stop_controller.decision = {"outcome": "failure"}

    session.observe_completion(step=4, qualified=True)

    assert session.terminal_reason() == TerminalReason.FIRST_COMPLETION
    assert session.terminal_model_kind(TerminalReason.LOCAL_INTERRUPTION) == "interrupted"


def test_failed_session_closes_progress_and_writes_precise_result(tmp_path: Path) -> None:
    session, _stop_flag = _session(tmp_path, mode=TrainingExecutionMode.LOCAL_DEMO)
    session.advance(4)

    session.fail(RuntimeError("learner exploded"))

    assert isinstance(session.progress, MemoryProgressSink)
    assert session.progress.closed is True
    document = json.loads(
        (tmp_path / "local-demo-with-signal" / TRAINING_RESULT_FILENAME).read_text(encoding="utf-8")
    )
    assert document["status"] == "failed"
    assert document["format_version"] == 3
    assert document["run_id"] == "run-test"
    assert document["attempt_id"] == "attempt-test"
    assert document["learner_pid"] > 0
    assert document["terminal_reason"] == "failed"
    assert document["execution_mode"] == "local-demo"
    assert document["final_step"] == 4
    assert document["requested_limit"] == 10
    assert document["execution_limit"] == 12
    assert document["model"] is None
    assert document["error_type"] == "RuntimeError"
    assert document["error_message"] == "learner exploded"


def test_plain_progress_is_bounded_and_uses_only_canonical_outcomes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    now = [0.0]
    sink = PlainProgressSink(interval_seconds=10.0, clock=lambda: now[0])
    sink.start(total=100, initial=0, description="gradlab.jerk")
    capsys.readouterr()

    metrics = {
        TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_ROLLING_UP_TO_100_MEAN: 5.0,
        TRAIN_OUTCOME_SUCCESS_ACROSS_OBSERVED_STARTS_CUMULATIVE_RATE_MEAN: 0.25,
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


def test_algorithm_progress_fields_are_formatted_by_the_shared_sink(
    capsys: pytest.CaptureFixture[str],
) -> None:
    now = [0.0]
    sink = PlainProgressSink(interval_seconds=10.0, clock=lambda: now[0])
    sink.start(
        total=100,
        initial=0,
        description="gradlab.go-explore",
        fields=(
            ProgressField("algorithm/cells", "cells", ProgressValueFormat.COUNT),
            ProgressField("algorithm/new_cell_rate", "new cells", ProgressValueFormat.PERCENT),
        ),
    )
    capsys.readouterr()

    now[0] = 10.0
    sink.update(
        step=20,
        metrics={
            "algorithm/cells": 12_345,
            "algorithm/new_cell_rate": 0.125,
        },
    )

    output = capsys.readouterr().out
    assert "cells=12.3k" in output
    assert "new cells=12.50%" in output


@pytest.mark.parametrize("mode", ("auto", "interactive", "plain"))
def test_local_console_modes_without_an_injected_tui_use_plain_progress(mode: str) -> None:
    assert isinstance(progress_sink_for_mode(mode), PlainProgressSink)


def test_session_keeps_algorithm_progress_stats_between_durable_reports(
    tmp_path: Path,
) -> None:
    session, _stop_flag = _session(tmp_path, mode=TrainingExecutionMode.LOCAL_DEMO)
    assert isinstance(session.progress, MemoryProgressSink)

    session.advance(2, progress_metrics={"algorithm/cells": 7})
    session.report(step=2, metrics={"train/throughput/loop_fps": 100.0})

    assert session.progress.metrics[-1]["algorithm/cells"] == 7


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

    assert payload[TRAIN_EPISODE_RETURN_SHAPED_ACROSS_ORIGINS_ROLLING_UP_TO_100_MEAN] == 14.0
    assert payload[TRAIN_EPISODE_RETURN_SHAPED_ACROSS_ORIGINS_ROLLING_UP_TO_100_MAX] == 30.0
    assert payload[TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_ROLLING_UP_TO_100_MEAN] == 6.0
    assert payload[TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_ROLLING_UP_TO_100_MAX] == 10.0
    assert payload[TRAIN_EPISODE_COMPLETED_COUNT] == 3
    assert payload[train_outcome_reason_count_metric("life_loss")] == 1
    assert payload[train_success_attempts_metric("StartA")] == 1
    assert payload[train_success_count_metric("StartA")] == 1
    assert payload[train_success_attempts_metric("StartB")] == 1
    assert payload[TRAIN_OUTCOME_SUCCESS_ACROSS_OBSERVED_STARTS_CUMULATIVE_RATE_MEAN] == 0.5
    assert payload[TRAIN_OUTCOME_SUCCESS_ACROSS_STARTS_COVERAGE_RATE] == 1.0


def test_episode_return_max_uses_the_same_rolling_window_as_the_mean() -> None:
    reducer = EpisodeMetricsReducer(track_success=False)
    reducer.consume(
        (_episode(start="StartA", episode_return=50.0, outcome=Outcome.FAILURE),)
    )
    payload = reducer.consume(
        _episode(start="StartA", episode_return=0.0, outcome=Outcome.FAILURE)
        for _ in range(100)
    )

    assert payload[TRAIN_EPISODE_RETURN_SHAPED_ACROSS_ORIGINS_ROLLING_UP_TO_100_MEAN] == 0.0
    assert payload[TRAIN_EPISODE_RETURN_SHAPED_ACROSS_ORIGINS_ROLLING_UP_TO_100_MAX] == 0.0
    assert payload[TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_ROLLING_UP_TO_100_MEAN] == 0.0
    assert payload[TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_ROLLING_UP_TO_100_MAX] == 0.0


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
        run_id="run-test",
        reducer=EpisodeMetricsReducer(
            configured_starts=("StartA",),
            track_success=True,
        ),
        execution_policy=TrainingExecutionPolicy.for_mode(TrainingExecutionMode.SUPERVISED),
        completion_signal_available=True,
        progress_sink=progress,
    )
    session.configure_checkpoints(run_name="run-test", eval_required=False)
    budget = session.configure_budget(requested_limit=10, step_quantum=4)

    assert budget.execution_total == 12
    ready_path = session.mark_ready()
    with pytest.raises(RuntimeError, match="only once"):
        session.mark_ready()
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    assert ready["document_type"] == "gradlab.learner-ready"
    assert ready["format_version"] == 3
    assert ready["run_id"] == "run-test"
    assert ready["attempt_id"] == "attempt-test"
    assert ready["status"] == "ready"

    record = _episode(start="StartA", episode_return=5.0, outcome=Outcome.SUCCESS)
    session.advance(4, (record,))
    session.report(step=4, metrics={"train/throughput/loop_fps": 100.0})
    session.report(step=4, metrics={"train/throughput/loop_fps": 101.0})

    assert len(store.frames) == 1
    assert store.frames[0][1]["source"] == "train"
    assert store.frames[0][0][TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_ROLLING_UP_TO_100_MEAN] == 5.0
    with pytest.raises(RuntimeError, match="regressed"):
        session.advance(3)
    with pytest.raises(RuntimeError, match="exceeded"):
        session.advance(13)

    session.finalize(
        session.result(
            terminal_reason=TerminalReason.RESOURCE_EXHAUSTION,
            final_step=4,
            model_kind="final",
        )
    )
    assert progress.closed is True
    result = json.loads((run_dir / TRAINING_RESULT_FILENAME).read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    assert result["format_version"] == 3
    assert result["run_id"] == "run-test"
    assert result["attempt_id"] == "attempt-test"
    assert result["terminal_reason"] == "resource_exhaustion"
    assert result["execution_mode"] == "supervised"
    assert result["requested_limit"] == 10
    assert result["execution_limit"] == 12


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
