from tests.test_play_trajectory import live_runner


def test_chart_history_spans_episode_and_zoom_does_not_seek(tmp_path):
    runner = live_runner(tmp_path, length=150)
    try:
        for _ in range(140):
            runner._step_once()
        episode = runner.recording_status()["episode_id"]
        before = runner.snapshot()["transition"]["step"]
        full = runner.chart_history(episode)
        assert (full["first"], full["last"]) == (1, 140)
        assert full["points"][0]["step"] == 1
        assert full["points"][-1]["step"] == 140
        zoom = runner.chart_history(episode, 10, 20)
        assert [p["step"] for p in zoom["points"]] == list(range(10, 21))
        runner.inspect_recorded_step(episode, 2)
        assert runner.chart_history(episode)["last"] == 140
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
        result = imported.chart_history(imported.metadata["episode_id"])
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
        assert host.chart_history(epoch, episode)["first"] == 1
        with pytest.raises(RuntimeError, match="replaced"):
            host.chart_history(epoch + 1, episode)
        assert host.chart_history(epoch, episode, 2, 2)["last"] == 2
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
