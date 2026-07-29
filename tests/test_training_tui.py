from __future__ import annotations

import asyncio
import multiprocessing
import os
import sys
import threading
import time
from unittest import mock

import pytest
from textual.widgets import ProgressBar, RichLog, Sparkline, Static

from gradlab.training_backend import GracefulStopFlag
from gradlab.training_lifecycle import (
    PlainProgressSink,
    ProgressField,
    ProgressValueFormat,
)
from gradlab.training_tui import (
    LearnerExecution,
    LocalTrainingApp,
    LocalTrainingIdentity,
    MetricCard,
    MetricGroup,
    TrainingProgressBridge,
    run_local_training_tui,
)


def test_progress_bridge_keeps_only_declared_fields_and_bounds_events() -> None:
    now = [1.0]
    bridge = TrainingProgressBridge(event_limit=2, clock=lambda: now[0])
    bridge.event("first")
    bridge.event("second")
    bridge.event("third")
    bridge.start(
        total=100,
        initial=10,
        description="gradlab.go-explore",
        fields=(
            ProgressField(
                "algorithm/cells",
                "cells",
                ProgressValueFormat.COUNT,
                group="exploration",
            ),
        ),
    )
    bridge.update(
        step=20,
        metrics={
            "algorithm/cells": 12,
            "not/displayed": 99,
        },
    )

    snapshot = bridge.snapshot()

    assert snapshot.step == 20
    assert snapshot.initial == 10
    assert [event.text for event in snapshot.events] == ["second", "third"]
    assert snapshot.metrics == {"algorithm/cells": 12}
    assert [field.label for field in snapshot.fields] == [
        "cells",
        "mean return",
        "completion",
    ]
    assert [field.group for field in snapshot.fields] == [
        "exploration",
        "outcomes",
        "outcomes",
    ]


def test_learner_execution_allows_exactly_one_concurrent_claim() -> None:
    execution = LearnerExecution()
    calls: list[str] = []
    barrier = threading.Barrier(3)
    claims: list[bool] = []

    def contender(name: str) -> None:
        barrier.wait()
        claims.append(execution.run(lambda: calls.append(name) or 7))

    threads = [
        threading.Thread(target=contender, args=("a",)),
        threading.Thread(target=contender, args=("b",)),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert sorted(claims) == [False, True]
    assert len(calls) == 1
    assert execution.result_or_raise() == 7


def test_textual_app_updates_algorithm_cards_inline_and_requests_safe_stop() -> None:
    bridge = TrainingProgressBridge()
    stop_flag = GracefulStopFlag()
    execution = LearnerExecution()
    started = threading.Event()
    release = threading.Event()

    def learner() -> int:
        bridge.start(
            total=1_000,
            initial=0,
            description="gradlab.go-explore",
            fields=(
                ProgressField(
                    "algorithm/cells",
                    "cells",
                    ProgressValueFormat.COUNT,
                    group="exploration",
                ),
                ProgressField(
                    "algorithm/archive-bytes",
                    "archive memory est.",
                    ProgressValueFormat.BYTES,
                    group="resources",
                ),
                ProgressField(
                    "algorithm/visits",
                    "visits",
                    ProgressValueFormat.COUNT,
                    group="traffic",
                ),
                ProgressField(
                    "algorithm/new-cell-rate",
                    "new cells",
                    ProgressValueFormat.PERCENT,
                    group="exploration",
                ),
                ProgressField(
                    "algorithm/best-progress",
                    "best progress",
                    group="exploration",
                ),
                ProgressField(
                    "algorithm/frontier-restores",
                    "frontier restores",
                    ProgressValueFormat.PERCENT,
                    group="traffic",
                ),
            ),
        )
        bridge.update(
            step=9,
            metrics={
                "algorithm/cells": 1_234,
                "algorithm/archive-bytes": 3 * 1024**3,
                "algorithm/visits": 5_678,
                "algorithm/new-cell-rate": 0.125,
                "algorithm/best-progress": 32,
                "algorithm/frontier-restores": 0.5,
            },
        )
        print("provider initialized")
        started.set()
        release.wait(timeout=5)
        bridge.close()
        return 0

    async def exercise() -> None:
        app = LocalTrainingApp(
            identity=LocalTrainingIdentity(
                "breakout/go-explore",
                123,
                "/tmp/run",
                completion_signal_available=False,
            ),
            bridge=bridge,
            stop_flag=stop_flag,
            execution=execution,
            learner=learner,
        )
        async with app.run_test(size=(120, 40)) as pilot:
            for _ in range(50):
                await pilot.pause()
                if started.is_set() and app.query(MetricCard):
                    break
            cards = list(app.query(MetricCard))
            assert len(cards) == 8
            assert len(list(app.query(MetricGroup))) == 4
            assert "1.23k" in str(app.query_one("#metric-value-0", Static).render())
            assert "3 GiB" in str(app.query_one("#metric-value-1", Static).render())
            assert "12.50%" in str(app.query_one("#metric-value-3", Static).render())
            metric_meter = app.query_one("#metric-meter-3", ProgressBar)
            for _ in range(10):
                await pilot.pause()
                if metric_meter.percentage == 0.125:
                    break
            assert metric_meter.percentage == 0.125

            bridge.update(
                step=19,
                metrics={
                    "algorithm/cells": 1_300,
                    "algorithm/new-cell-rate": 0.13,
                },
            )
            app._last_history_sample = 0.0
            app._refresh_from_bridge()
            assert "—" not in str(app.query_one("#summary-rate .summary-value", Static).render())
            assert app.query_one("#rate-sparkline", Sparkline).display is True
            assert app.query_one("#rate-sparkline", Sparkline).data

            event_log = app.query_one("#event-log", RichLog)
            assert event_log.max_lines == 256
            assert event_log.display is False
            await pilot.press("l")
            await pilot.pause()
            assert event_log.display is True

            step_bar = app.query_one("#training-progress Bar")
            progress_bar = app.query_one("#training-progress", ProgressBar)
            assert step_bar.size.width > 32
            assert step_bar.percentage is not None
            assert step_bar.percentage > 0
            assert progress_bar.gradient is not None
            assert progress_bar.gradient.get_color(0.0).hex == "#22D3EE"
            assert progress_bar.gradient.get_color(1.0).hex == "#67E8F9"
            assert app.screen.styles.background.hex == "#05090D"
            assert app.query_one("#metric-group-exploration").styles.background.hex == ("#080E14")
            assert "TRAINING-ONLY RUN" in str(app.query_one("#run-notice", Static).render())
            assert "no declared success signal" in str(
                app.query_one("#run-notice", Static).render()
            )
            assert "not declared" in str(app.query_one("#metric-value-7", Static).render())
            assert app.query_one("#metric-meter-7", ProgressBar).display is False

            await pilot.press("?")
            await pilot.pause()
            assert "ctrl+p" in str(app.query_one("#latest-event", Static).render())

            await pilot.press("q")
            await pilot.pause()
            assert stop_flag.requested is True
            assert execution.state == "running"
            assert "stop pending" in str(app.query_one("#training-status", Static).render()).lower()
            release.set()
            for _ in range(50):
                await pilot.pause()
                if execution.wait(0):
                    break

    try:
        asyncio.run(exercise())
    finally:
        release.set()

    assert execution.result_or_raise() == 0
    assert any(event.text == "provider initialized" for event in bridge.snapshot().events)


def test_textual_capture_preserves_valid_fds_for_spawned_learner_process() -> None:
    bridge = TrainingProgressBridge()
    stop_flag = GracefulStopFlag()
    execution = LearnerExecution()
    observed_fds: list[tuple[int, int]] = []

    def learner() -> int:
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()
        os.fstat(stdout_fd)
        os.fstat(stderr_fd)
        observed_fds.append((stdout_fd, stderr_fd))
        context = multiprocessing.get_context("spawn")
        process = context.Process(target=time.sleep, args=(0.01,))
        process.start()
        process.join(timeout=5)
        assert not process.is_alive()
        assert process.exitcode == 0
        return 0

    async def exercise() -> None:
        app = LocalTrainingApp(
            identity=LocalTrainingIdentity("breakout/go-explore", 123, "/tmp/run"),
            bridge=bridge,
            stop_flag=stop_flag,
            execution=execution,
            learner=learner,
        )
        async with app.run_test(size=(120, 40)) as pilot:
            for _ in range(100):
                await pilot.pause()
                if execution.wait(0):
                    break

    asyncio.run(exercise())

    assert execution.result_or_raise() == 0
    assert observed_fds
    assert all(fd >= 0 for fd in observed_fds[0])


def test_tui_failure_before_worker_start_falls_back_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    controls = []

    def learner(control) -> int:
        controls.append(control)
        return 3

    with mock.patch.object(LocalTrainingApp, "run", side_effect=RuntimeError("bad terminal")):
        result = run_local_training_tui(
            identity=LocalTrainingIdentity(
                "breakout/go-explore",
                123,
                "/tmp/run",
                notices=("local notice",),
            ),
            learner=learner,
        )

    assert result == 3
    assert len(controls) == 1
    assert isinstance(controls[0].progress_sink, PlainProgressSink)
    output = capsys.readouterr().out
    assert "using plain output" in output
    assert "local notice" in output


def test_tui_plain_fallback_preserves_learner_exception() -> None:
    def learner(_control) -> int:
        raise RuntimeError("learner exploded")

    with (
        mock.patch.object(LocalTrainingApp, "run", side_effect=RuntimeError("bad terminal")),
        pytest.raises(RuntimeError, match="learner exploded"),
    ):
        run_local_training_tui(
            identity=LocalTrainingIdentity("breakout/go-explore", 123, "/tmp/run"),
            learner=learner,
        )
