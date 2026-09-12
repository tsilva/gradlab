"""Compatibility through the authenticated diagnostic HTTP boundary."""

from gradlab.play_diagnostics import DiagnosticRead

import asyncio
from concurrent.futures import CancelledError
from types import SimpleNamespace

import pytest
from aiohttp import ServerDisconnectedError, web
from aiohttp.test_utils import TestClient, TestServer

from gradlab.play_web import PlaybackWebServer


@pytest.mark.parametrize("kind", ["chart", "reward", "event"])
@pytest.mark.parametrize("error,status", [
    (ValueError("too many pending diagnostic reads"), 400),
    (ValueError("the recorded episode has been replaced"), 400),
    (OSError("disk unavailable"), 400),
    (RuntimeError("CancelledError: "), 400),
    (CancelledError(), None),
])
def test_history_http_preserves_error_classification(kind, error, status):
    async def scenario():
        def fail(*args):
            raise error

        runner = SimpleNamespace(session_epoch=7, read_diagnostics=fail)
        server = PlaybackWebServer(runner, SimpleNamespace())
        app = web.Application()
        app.router.add_get("/history", getattr(server, f"{kind}_history"))
        async with TestClient(TestServer(app)) as client:
            params = dict(epoch="7", episode_id="recording")
            response = await client.get("/history", params=params)
            assert response.status == 401
            headers = {"Authorization": f"Bearer {server.token}"}
            if status is None:
                with pytest.raises(ServerDisconnectedError):
                    await client.get("/history", params=params, headers=headers)
                return
            response = await client.get("/history", params=params, headers=headers)
            assert response.status == status
            if status == 400:
                assert await response.json() == {"error": str(error)}

    asyncio.run(scenario())


def test_preparation_failure_releases_recording_and_allows_later_reads(tmp_path, monkeypatch):
    from tests.test_play_trajectory import live_runner

    runner = live_runner(tmp_path)
    try:
        runner._step_once()
        recording = runner.recording
        episode = recording.metadata["episode_id"]
        # Calibration annotations are prepared after acquiring the recording.
        with monkeypatch.context() as patch:
            patch.setattr(runner, "history", [{"realized_return": 1}])
            with pytest.raises(KeyError):
                runner.diagnostics.read(DiagnosticRead("chart", episode))
        assert runner.diagnostics.read(DiagnosticRead("chart", episode))["last"] == 1
        runner._begin_recording()
        assert not recording.root.exists(), "failed preparation must not retain retired files"
    finally:
        runner.stop()


@pytest.fixture(params=["live", "imported"])
def playback(request, tmp_path):
    from gradlab.model_sources import ResolvedModelSource
    from gradlab.play_application import PlaybackHost
    from gradlab.play_runtime import ActivePlayback, PlaybackLoader, PlaySourceSpec
    from gradlab.play_trajectory import export_trajectory
    from gradlab.play_trajectory_runner import TrajectoryPlaybackRunner
    from gradlab.policy_bundle import load_policy_bundle
    from tests.test_play_trajectory import live_runner

    live = live_runner(tmp_path, length=3)
    for _ in range(3):
        live._step_once()
    archive = export_trajectory(live.freeze_trajectory(), tmp_path / "episode.trj")
    runner = live
    if request.param == "imported":
        live.stop()
        runner = TrajectoryPlaybackRunner(archive, live.args)
        runner.start()
    host = PlaybackHost(PlaybackLoader(runner.args, argv=[], explicit_seed=False))
    host._active = ActivePlayback(
        runner, None, PlaySourceSpec("local", "fixture"),
        ResolvedModelSource(tmp_path / "model.zip", load_policy_bundle(tmp_path)),
    )
    host._phase = "active"
    try:
        yield host, runner, archive
    finally:
        host.stop()


@pytest.fixture
def calculation_threads(monkeypatch):
    """Control process scheduling only; exercise real recording and calculation code."""
    from concurrent.futures import ThreadPoolExecutor
    import gradlab.play_diagnostics as diagnostics

    monkeypatch.setattr(diagnostics, "ProcessPoolExecutor", lambda **kw: ThreadPoolExecutor(1))
    return diagnostics


@pytest.mark.parametrize("kind", ["chart", "reward", "event"])
def test_host_reads_preserve_values_and_cursor(playback, calculation_threads, kind):
    host, runner, _ = playback
    before = runner.snapshot()
    episode = runner.recording.metadata["episode_id"]
    result = host.read_diagnostics(0, DiagnosticRead(kind, episode, 1))
    assert result["episode_id"] == episode
    if kind == "chart":
        assert [point["step"] for point in result["points"]] == [1, 2, 3]
        assert [point["reward_shaped"] for point in result["points"]] == [0.5, 0.5, 0.5]
    elif kind == "reward":
        assert result["return_total"] == pytest.approx(1.355)
        assert result["reward_sum"] == 1.5
    else:
        assert [point["step"] for point in result["points"]] == [3]
        assert result["next_last"] is None
    assert runner.snapshot() == before
    with pytest.raises(ValueError, match="Session has been replaced"):
        host.read_diagnostics(1, DiagnosticRead(kind, episode))
    with pytest.raises(ValueError, match="episode has been replaced"):
        host.read_diagnostics(0, DiagnosticRead(kind, "stale"))


@pytest.mark.parametrize("kind", ["chart", "reward", "event"])
@pytest.mark.parametrize("failure", ["preparation", "submission", "calculation", "validation"])
def test_failures_release_pins_and_allow_recovery(
    playback, calculation_threads, monkeypatch, kind, failure
):
    host, runner, _ = playback
    recording = runner.recording
    query = DiagnosticRead(kind, recording.metadata["episode_id"])

    def fail(*args, **kwargs):
        raise OSError("injected read failure")

    with monkeypatch.context() as patch:
        if failure == "preparation":
            patch.setattr(calculation_threads, "calibration_annotations", fail)
        elif failure == "calculation":
            patch.setattr(calculation_threads, "query_prefix", fail)
        elif failure == "validation":
            patch.setattr(calculation_threads.RecordingRead, "validate", fail)
        else:
            from concurrent.futures import ThreadPoolExecutor
            patch.setattr(ThreadPoolExecutor, "submit", fail)
        with pytest.raises(OSError, match="injected read failure"):
            host.read_diagnostics(0, query)
    assert host.read_diagnostics(0, query)["episode_id"] == query.episode_id
    host.stop()
    assert not recording.root.exists()


@pytest.mark.parametrize("kind", ["chart", "reward", "event"])
def test_session_activation_precedes_old_diagnostic_drain(
    playback, calculation_threads, monkeypatch, kind
):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    host, runner, archive = playback
    recording = runner.recording
    entered, release, draining = threading.Event(), threading.Event(), threading.Event()
    calculate = calculation_threads.query_prefix
    close = runner.diagnostics.close

    def blocked(*args):
        entered.set()
        assert release.wait(10)
        return calculate(*args)

    def closing():
        draining.set()
        return close()

    monkeypatch.setattr(calculation_threads, "query_prefix", blocked)
    monkeypatch.setattr(runner.diagnostics, "close", closing)
    with ThreadPoolExecutor(2) as calls:
        old = calls.submit(host.read_diagnostics, 0, DiagnosticRead(kind, recording.metadata["episode_id"]))
        try:
            assert entered.wait(3)
            replacing = calls.submit(host.import_trajectory, str(archive))
            assert draining.wait(3)
            assert host.snapshot()["session_epoch"] == 1
            assert recording.root.exists()
            assert not old.done()
            assert not replacing.done()
        finally:
            release.set()
        with pytest.raises(ValueError, match="Session has been replaced"):
            old.result(timeout=3)
        replacing.result(timeout=3)
    assert not recording.root.exists()
    episode = host.snapshot()["trajectory"]["episode_id"]
    assert host.read_diagnostics(1, DiagnosticRead(kind, episode))["episode_id"] == episode


@pytest.mark.parametrize("kind", ["chart", "reward", "event"])
def test_episode_replacement_retains_read_then_rejects_result(tmp_path, calculation_threads, monkeypatch, kind):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from tests.test_play_trajectory import live_runner

    runner = live_runner(tmp_path, length=10)
    runner._step_once()
    recording = runner.recording
    entered, release = threading.Event(), threading.Event()
    calculate = calculation_threads.query_prefix

    def blocked(*args):
        entered.set()
        assert release.wait(5)
        return calculate(*args)

    monkeypatch.setattr(calculation_threads, "query_prefix", blocked)
    try:
        with ThreadPoolExecutor(2) as calls:
            old = calls.submit(runner.diagnostics.read, DiagnosticRead(kind, recording.metadata["episode_id"]))
            try:
                assert entered.wait(2)
                calls.submit(runner._begin_recording).result(timeout=1)
                assert recording.root.exists()
            finally:
                release.set()
            with pytest.raises(ValueError, match="episode has been replaced"):
                old.result(timeout=2)
        assert not recording.root.exists()
        runner._step_once()
        assert runner.diagnostics.read(DiagnosticRead(kind, runner.recording.metadata["episode_id"]))
    finally:
        runner.stop()


@pytest.mark.parametrize("kind", ["chart", "reward", "event"])
def test_admission_and_shutdown_release_running_and_cancelled_reads(
    playback, calculation_threads, monkeypatch, kind
):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    host, runner, _ = playback
    recording = runner.recording
    query = DiagnosticRead(kind, recording.metadata["episode_id"])
    entered, release, four_submitted, closing = [threading.Event() for _ in range(4)]
    calculate = calculation_threads.query_prefix

    class ControlledPool(ThreadPoolExecutor):
        def __init__(self, **kwargs):
            super().__init__(1)
            self.submissions = 0

        def submit(self, *args, **kwargs):
            future = super().submit(*args, **kwargs)
            self.submissions += 1
            if self.submissions == 4:
                four_submitted.set()
            return future

        def shutdown(self, **kwargs):
            closing.set()
            return super().shutdown(**kwargs)

    def blocked(*args):
        entered.set()
        assert release.wait(10)
        return calculate(*args)

    monkeypatch.setattr(calculation_threads, "ProcessPoolExecutor", ControlledPool)
    monkeypatch.setattr(calculation_threads, "query_prefix", blocked)
    with ThreadPoolExecutor(5) as calls:
        running = calls.submit(host.read_diagnostics, 0, query)
        try:
            assert entered.wait(3)
            queued = [calls.submit(host.read_diagnostics, 0, query) for _ in range(3)]
            assert four_submitted.wait(3)
            with pytest.raises(ValueError, match="too many pending diagnostic reads"):
                host.read_diagnostics(0, query)
            shutdown = calls.submit(runner.diagnostics.close)
            assert closing.wait(3)
            for future in queued:
                with pytest.raises(CancelledError):
                    future.result(timeout=3)
            assert not shutdown.done()
            with pytest.raises(ValueError, match="Session has been replaced"):
                host.read_diagnostics(0, query)
        finally:
            release.set()
        assert running.result(timeout=3)["episode_id"] == query.episode_id
        shutdown.result(timeout=3)
    host.stop()
    assert not recording.root.exists()


@pytest.mark.parametrize("kind", ["chart", "reward", "event"])
@pytest.mark.parametrize("diagnostic_first", [False, True])
def test_export_inspection_and_diagnostics_share_recording_retention(
    tmp_path, calculation_threads, monkeypatch, kind, diagnostic_first
):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from gradlab.play_diagnostics import RecordedPrefix
    from tests.test_play_trajectory import live_runner

    runner = live_runner(tmp_path, length=10)
    runner._step_once()
    recording = runner.recording
    archive = recording.reserve_prefix()
    inspection = recording.reserve_read()
    entered, release = threading.Event(), threading.Event()
    calculate = calculation_threads.query_prefix

    def blocked(*args):
        entered.set()
        assert release.wait(5)
        return calculate(*args)

    monkeypatch.setattr(calculation_threads, "query_prefix", blocked)
    try:
        with ThreadPoolExecutor(1) as calls:
            read = calls.submit(runner.diagnostics.read, DiagnosticRead(kind, recording.metadata["episode_id"]))
            try:
                assert entered.wait(2)
                runner._begin_recording()
                if diagnostic_first:
                    release.set()
                    with pytest.raises(ValueError, match="replaced"):
                        read.result(timeout=3)
                with recording.prefix(archive) as (metadata, written, pending):
                    assert metadata["transition_count"] == 1
                    assert written + len(pending) == 1
                assert recording.root.exists(), "inspection still owns its pin"
                assert RecordedPrefix(inspection).transition(1)["step"] == 1
                recording.release_read()
                if not diagnostic_first:
                    assert recording.root.exists(), "diagnostic still owns its pin"
            finally:
                release.set()
            with pytest.raises(ValueError, match="replaced"):
                read.result(timeout=3)
        assert not recording.root.exists()
    finally:
        runner.stop()


def _completed_reads_worker(*args):
    from pathlib import Path
    from unittest.mock import patch
    from gradlab.play_application import PlaybackHost
    from tests.test_playback_episode_history import _episode_inspection_worker

    read = PlaybackHost.read_diagnostics

    def completed(host, epoch, request):
        result = read(host, epoch, request)
        (Path(args[1].fixture_root) / f"completed-{request.first}").touch()
        return result

    with patch.object(PlaybackHost, "read_diagnostics", completed):
        _episode_inspection_worker(*args)


def test_worker_counts_completed_unpolled_reads_and_inspection_together(tmp_path, monkeypatch):
    from argparse import Namespace
    import time
    import gradlab.playback_worker as worker

    monkeypatch.setattr(worker, "_worker_main", _completed_reads_worker)
    host = worker.IsolatedPlaybackHost(
        Namespace(fps=30, fixture_root=str(tmp_path)), argv=[], explicit_seed=False
    )
    try:
        host.start()
        snapshot = host.snapshot()
        epoch, episode = snapshot["session_epoch"], snapshot["trajectory"]["episode_id"]
        jobs = [host._rpc("begin_read", kind="diagnostics", epoch=epoch,
                          query=dict(kind="chart", episode_id=episode, first=i)) for i in range(8)]
        deadline = time.monotonic() + 10
        # Marker files are cross-process completion signals, not a sleep-based race.
        while len(list(tmp_path.glob("completed-*"))) != 8 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(list(tmp_path.glob("completed-*"))) == 8
        with pytest.raises(RuntimeError, match="too many pending diagnostic reads"):
            host._rpc("begin_read", kind="inspect_recorded_step", epoch=epoch,
                      episode_id=episode, step=1)
        assert host.snapshot()["sequence"] == snapshot["sequence"]
        assert host._rpc("poll_read", job_id=jobs.pop())["done"]
        assert host.inspect_recorded_step(epoch, episode, 1)["frames"]
        for job in jobs:
            assert host._rpc("poll_read", job_id=job)["done"]
    finally:
        host.stop()
