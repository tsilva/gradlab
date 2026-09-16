import pytest

from gradlab.play_diagnostics import DiagnosticRead
from tests.test_play_trajectory import live_runner


def test_chart_history_spans_episode_and_zoom_does_not_seek(tmp_path):
    runner = live_runner(tmp_path, length=150)
    try:
        for _ in range(140):
            runner._step_once()
        episode = runner.recording_status()["episode_id"]
        before = runner.snapshot()["transition"]["step"]
        full = runner.diagnostics.read(DiagnosticRead("chart", episode))
        assert (full["first"], full["last"]) == (1, 140)
        assert full["points"][0]["step"] == 1
        assert full["points"][-1]["step"] == 140
        zoom = runner.diagnostics.read(DiagnosticRead("chart", episode, 10, 20))
        assert [p["step"] for p in zoom["points"]] == list(range(10, 21))
        runner.inspect_recorded_step(episode, 2)
        assert runner.diagnostics.read(DiagnosticRead("chart", episode))["last"] == 140
        assert runner.snapshot()["transition"]["step"] == before
    finally:
        runner.stop()


def test_chart_overview_retains_peaks_and_reuses_index(tmp_path):
    from copy import deepcopy
    from types import SimpleNamespace
    from gradlab.play_chart_history import chart_history

    runner = live_runner(tmp_path, length=2)
    try:
        runner._step_once()
        transition = runner.snapshot()["transition"]
        reads = []

        def read(step):
            reads.append(step)
            value = deepcopy(transition)
            value.update(step=step, sequence=step)
            value["reward"]["provider"] = 50 if step == 2507 else 0
            value["signals"] = {"peak": -90 if step == 3127 else 0}
            return {"inspection_snapshot": {"transition": value}}

        recording = SimpleNamespace(
            root=tmp_path,
            metadata={"episode_id": "test", "first_step": 1, "episode": 1},
            status=lambda: {"last_step": 5000},
            transition=read,
        )
        source = SimpleNamespace(recording=recording, history=[])
        overview = chart_history(source, "test", None, None)
        assert len(overview["points"]) < 400
        assert {1, 2507, 3127, 5000} <= {p["step"] for p in overview["points"]}
        assert len(reads) == 5000
        zoom = chart_history(source, "test", 2500, 2510)
        assert [p["step"] for p in zoom["points"]] == list(range(2500, 2511))
        assert len(reads) == 5000
    finally:
        runner.stop()


def test_imported_episode_supports_full_chart_range(tmp_path):
    from gradlab.play_trajectory import export_trajectory
    from gradlab.play_trajectory_runner import TrajectoryPlaybackRunner

    runner = live_runner(tmp_path, length=3)
    imported = None
    try:
        for _ in range(3):
            runner._step_once()
        archive = export_trajectory(runner.freeze_trajectory(), tmp_path / "charts.trj")
        imported = TrajectoryPlaybackRunner(archive, runner.args)
        result = imported.diagnostics.read(DiagnosticRead("chart", imported.metadata["episode_id"]))
        assert [p["step"] for p in result["points"]] == [1, 2, 3]
    finally:
        if imported:
            imported.stop()
        runner.stop()


def test_worker_chart_history_is_bound_to_session_epoch(tmp_path, monkeypatch):
    from argparse import Namespace
    import pytest
    import gradlab.playback_worker as worker
    from tests.test_playback_episode_history import _episode_inspection_worker

    monkeypatch.setattr(worker, "_worker_main", _episode_inspection_worker)
    host = worker.IsolatedPlaybackHost(
        Namespace(fps=30, fixture_root=str(tmp_path)), argv=[], explicit_seed=False
    )
    try:
        host.start()
        snapshot = host.snapshot()
        epoch = snapshot["session_epoch"]
        episode = snapshot["trajectory"]["episode_id"]
        assert host.read_diagnostics(epoch, DiagnosticRead("chart", episode))["first"] == 1
        with pytest.raises(RuntimeError, match="replaced"):
            host.read_diagnostics(epoch + 1, DiagnosticRead("chart", episode))
        assert host.read_diagnostics(epoch, DiagnosticRead("chart", episode, 2, 2))["last"] == 2
    finally:
        host.stop()


def test_event_pages_preserve_every_event_beyond_inspection_buffer(tmp_path):
    from copy import deepcopy
    from types import SimpleNamespace
    from gradlab.play_event_history import event_history

    runner = live_runner(tmp_path, length=2)
    try:
        runner._step_once()
        transition = runner.snapshot()["transition"]
        reads = []

        def read(step):
            reads.append(step)
            point = deepcopy(transition)
            point.update(step=step, sequence=step, events=["brick_destroyed"], boundary=False)
            return {"inspection_snapshot": {"transition": point}}

        recording = SimpleNamespace(
            root=tmp_path,
            metadata={"episode_id": "events", "first_step": 1, "episode": 1},
            status=lambda: {"last_step": 4200},
            transition=read,
        )
        source = SimpleNamespace(recording=recording, history=[])
        steps = []
        last = None
        while True:
            page = event_history(source, "events", None, last)
            steps.extend(p["step"] for p in page["points"])
            last = page["next_last"]
            if last is None:
                break
        assert steps == list(range(4200, 0, -1))
        assert len(reads) == 4200
    finally:
        runner.stop()


def test_live_return_estimate_uses_full_suffix_and_pre_action_tail(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import patch
    from gradlab.play_chart_history import chart_history

    data = {
        1: dict(step=1, reward_shaped=2, value=99),
        2: dict(step=2, reward_shaped=3, value=99),
        3: dict(step=3, reward_shaped=100, value=8),
    }
    recording = SimpleNamespace(
        root=tmp_path,
        metadata=dict(episode_id="live", episode=1, first_step=1, discount=0.5),
        status=lambda: dict(last_step=3),
        transition=lambda step: dict(presentation=data[step]),
    )
    runner = SimpleNamespace(recording=recording, history=[])
    with patch("gradlab.play_web.history_point_payload", side_effect=lambda point: point):
        result = chart_history(runner, "live", 1, 1)
        assert result["points"][0]["estimated_return"] == 5.5
        assert result["points"][0]["return_estimate_step"] == 3
        assert "realized_return" not in result["points"][0]
        result = chart_history(runner, "live", 3, 3)
        assert result["points"][0]["estimated_return"] == 8


def test_incremental_returns_match_exact_suffixes_and_stop_at_invalid_transitions(tmp_path):
    import random
    import pytest
    from types import SimpleNamespace
    from unittest.mock import patch
    from gradlab.play_chart_history import chart_history

    rng = random.Random(7)
    for gamma in (0., .99, 1.):
        root = tmp_path / str(gamma)
        root.mkdir()
        end = 513
        data = {step: dict(step=step, sequence=step, episode=1,
                           reward_shaped=rng.uniform(-2, 3), value=rng.uniform(-2, 3))
                for step in range(1, end + 2)}
        data[100]["policy_sampled"] = False
        recording = SimpleNamespace(root=root,
            metadata=dict(episode_id="test", episode=1, first_step=1, discount=gamma),
            status=lambda: dict(last_step=end),
            transition=lambda step: dict(presentation=data[step]))
        runner = SimpleNamespace(recording=recording, history=[])
        with patch("gradlab.play_web.history_point_payload", side_effect=lambda point: point):
            for current_end in (513,514):
                end = current_end
                full = chart_history(runner, "test", None, None)
                for point in full["points"]:
                    step = point["step"]
                    if step <= 100:
                        assert "estimated_return" not in point
                    else:
                        expected = data[end]["value"]
                        for index in range(end - 1, step - 1, -1):
                            expected = data[index]["reward_shaped"] + gamma * expected
                        assert point["estimated_return"] == pytest.approx(expected, abs=1e-10)
            # A calibration amendment changes an indexed leaf and its ancestors.
            runner.history = [dict(step=300, episode=1, value_comparison_reasons=["changed contract"])]
            result = chart_history(runner, "test", 299,301)
            assert "estimated_return" not in result["points"][0]
            assert "estimated_return" in result["points"][-1]


@pytest.mark.parametrize("processing", [set(), {"game", "observation", "policy"}])
def test_stats_opened_after_episode_recovers_calibration_without_advancing(tmp_path, processing):
    runner = live_runner(tmp_path, length=3, value_contract={"discount": 0.9})
    try:
        # Only the player is connected while the episode runs.
        runner.set_processing(processing)
        for _ in range(3):
            runner._step_once()
        episode = runner.recording_status()["episode_id"]
        before = runner.session.sequence
        # Calibration must come from disk even after the bounded live history is gone.
        runner.history.clear()
        runner.set_processing({"game", "observation", "policy", "history", "critic-calibration"})
        inspected = runner.inspect_recorded_step(episode, 3)
        assert inspected["snapshot"]["session"]["critic_comparison"]["available"] is True
        result = runner.diagnostics.read(DiagnosticRead("chart", episode))
        assert [point["realized_return"] for point in result["points"]] == pytest.approx(
            [1.355, 0.95, 0.5]
        )
        assert runner.session.sequence == before
        assert runner.run_state == "paused"
    finally:
        runner.stop()


@pytest.mark.parametrize("truncated,bootstrap", [(False, None), (True, 4.0), (True, None)])
def test_late_chart_calibration_preserves_boundary_and_sampling_contract(tmp_path, truncated, bootstrap):
    from dataclasses import replace

    runner = live_runner(
        tmp_path, length=3,
        value_contract={"discount": 0.9, "truncation_bootstrap": "terminal-value"},
    )
    step = runner.session.step

    def advance(**kwargs):
        transition = step(**kwargs)
        if transition.boundary and truncated:
            transition = replace(
                transition, terminated=False, truncated=True,
                return_bootstrap_value=bootstrap,
            )
        return transition

    runner.session.step = advance
    runner.set_processing(set())
    try:
        for _ in range(3):
            runner._step_once()
        episode = runner.recording_status()["episode_id"]
        result = runner.diagnostics.read(DiagnosticRead("chart", episode, 1, 2))
        if truncated and bootstrap is None:
            assert all("realized_return" not in point for point in result["points"])
        else:
            extra = bootstrap or 0
            assert [point["realized_return"] for point in result["points"]] == pytest.approx(
                [1.355 + 0.9**3 * extra, 0.95 + 0.9**2 * extra]
            )
            assert all(point["realized_return_bootstrapped"] == truncated for point in result["points"])
    finally:
        runner.stop()


def test_late_chart_does_not_calibrate_an_episode_with_changed_sampling(tmp_path):
    runner = live_runner(tmp_path, length=3, value_contract={"discount": 0.9})
    runner.set_processing(set())
    try:
        runner.sampling_mode = "deterministic"
        runner._step_once()
        runner.sampling_mode = "stochastic"
        runner._step_once()
        runner._step_once()
        runner.history.clear()
        result = runner.diagnostics.read(DiagnosticRead("chart", runner.recording_status()["episode_id"]))
        assert all("realized_return" not in point for point in result["points"])
        assert result["points"][0]["value_comparison_reasons"]
    finally:
        runner.stop()
