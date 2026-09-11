"""History work must not monopolize control RPCs or rescan old chart rows."""

from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _slow_history_worker(*args):
    from gradlab.play_application import PlaybackHost
    from gradlab.play_web import WebPlaybackRunner
    from tests.test_playback_episode_history import _episode_inspection_worker

    method = getattr(args[1], "blocked_read", "chart_history")
    owner = WebPlaybackRunner if method == "inspect_recorded_step" else PlaybackHost
    original = getattr(owner, method)

    def blocked(host, *query):
        root = Path(args[1].fixture_root)
        (root / "reading").touch()
        while not (root / "release").exists():
            time.sleep(0.01)
        return original(host, *query)

    with patch.object(owner, method, blocked):
        _episode_inspection_worker(*args)


@pytest.mark.parametrize("method", ["chart_history", "inspect_recorded_step"])
def test_slow_history_does_not_hold_control_rpc(tmp_path, monkeypatch, method):
    import gradlab.playback_worker as worker

    monkeypatch.setattr(worker, "_worker_main", _slow_history_worker)
    host = worker.IsolatedPlaybackHost(
        Namespace(fps=30, fixture_root=str(tmp_path), blocked_read=method),
        argv=[],
        explicit_seed=False,
    )
    pool = ThreadPoolExecutor(2)
    try:
        host.start()
        snapshot = host.snapshot()
        query = pool.submit(getattr(host, method), snapshot["session_epoch"],
                            snapshot["trajectory"]["episode_id"],
                            *([1] if method == "inspect_recorded_step" else []))
        deadline = time.monotonic() + 5
        while not (tmp_path / "reading").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (tmp_path / "reading").exists()
        # Delayed work remains held until the finally block. This can only pass
        # if a second RPC is served while the query is still outstanding.
        assert pool.submit(host.snapshot).result(timeout=0.5)["sequence"] == snapshot["sequence"]
        assert not query.done()
    finally:
        (tmp_path / "release").touch()
        pool.shutdown(wait=True)
        host.stop()


@pytest.mark.parametrize("replace_episode", [False, True])
def test_recorded_frame_encoding_releases_lock_and_fences_replacement(tmp_path, replace_episode):
    import threading
    from PIL import Image
    from tests.test_play_trajectory import live_runner

    runner = live_runner(tmp_path, length=10)
    entered, release = threading.Event(), threading.Event()
    save = Image.Image.save

    def delayed(image, *args, **kwargs):
        if threading.current_thread().name == "fps-inspection_0":
            entered.set()
            assert release.wait(5)
        return save(image, *args, **kwargs)

    pool = ThreadPoolExecutor(2, thread_name_prefix="fps-inspection")
    try:
        runner._step_once()
        episode = runner.recording.metadata["episode_id"]
        recorded_root = runner.recording.root
        with patch.object(Image.Image, "save", delayed):
            query = pool.submit(runner.inspect_recorded_step, episode, 1)
            assert entered.wait(2)
            if replace_episode:
                pool.submit(runner._begin_recording).result(timeout=0.5)
                assert runner.recording.metadata["episode_id"] != episode
                assert recorded_root.exists(), "the outstanding read must pin its old files"
            else:
                pool.submit(runner._step_once).result(timeout=0.5)
                assert runner.session.sequence == 2
            assert not query.done()
            release.set()
            if replace_episode:
                with pytest.raises(ValueError, match="replaced"):
                    query.result(timeout=2)
                assert not recorded_root.exists()
            else:
                assert query.result(timeout=2)["snapshot"]["transition"]["step"] == 1
    finally:
        release.set()
        pool.shutdown(wait=True)
        runner.stop()


def test_recorded_step_read_does_not_wait_for_an_inflight_policy_decision(tmp_path):
    import threading
    from tests.test_play_trajectory import live_runner

    runner = live_runner(tmp_path, length=10)
    entered, release = threading.Event(), threading.Event()
    step = runner.session.step

    def delayed(**kwargs):
        entered.set()
        assert release.wait(5)
        return step(**kwargs)

    pool = ThreadPoolExecutor(2)
    try:
        runner._step_once()
        episode = runner.recording.metadata["episode_id"]
        with patch.object(runner.session, "step", delayed):
            decision = pool.submit(runner._step_once)
            assert entered.wait(2)
            result = pool.submit(runner.inspect_recorded_step, episode, 1).result(timeout=0.5)
            assert result["snapshot"]["transition"]["step"] == 1
            assert not decision.done()
    finally:
        release.set()
        pool.shutdown(wait=True)
        runner.stop()


def test_appending_one_chart_point_does_not_rescan_old_points(tmp_path, monkeypatch):
    import gradlab.play_chart_history as charts

    end = 4096

    def read(step):
        return {
            "presentation": dict(
                step=step, sequence=step, episode=1, reward_shaped=float(step % 13), value=2.0
            )
        }

    recording = SimpleNamespace(
        root=tmp_path,
        metadata=dict(episode_id="growing", episode=1, first_step=1, discount=0.99),
        status=lambda: dict(last_step=end),
        transition=read,
    )
    runner = SimpleNamespace(recording=recording, history=[])
    with patch("gradlab.play_web.history_point_payload", side_effect=lambda p: p):
        charts.chart_history(runner, "growing", None, None)
        calls = 0
        original = charts.scalars

        def counted(*args):
            nonlocal calls
            calls += 1
            yield from original(*args)

        monkeypatch.setattr(charts, "scalars", counted)
        end += 1
        result = charts.chart_history(runner, "growing", None, None)
    assert result["last"] == 4097
    assert calls < 1000, f"incremental refresh revisited {calls} scalar trees"


def test_slow_diagnostic_process_does_not_lock_inference(tmp_path):
    import threading
    from tests.test_play_trajectory import live_runner

    runner = live_runner(tmp_path, length=10)
    entered = threading.Event()
    release = threading.Event()

    def blocked(*args):
        entered.set()
        release.wait(5)
        return {}

    pool = ThreadPoolExecutor(2)
    try:
        runner._step_once()
        episode = runner.recording.metadata["episode_id"]
        with patch.object(runner._diagnostics, "query", blocked):
            query = pool.submit(runner.chart_history, episode)
            assert entered.wait(2)
            pool.submit(runner._step_once).result(timeout=0.5)
            assert runner.session.sequence == 2
            assert not query.done()
    finally:
        release.set()
        pool.shutdown(wait=True)
        runner.stop()


def test_pinned_prefix_survives_retirement_and_contains_pending_records(tmp_path):
    import threading
    from gradlab.play_diagnostics import RecordedPrefix
    from gradlab.play_trajectory import EpisodeRecording
    from tests.test_play_trajectory import live_runner

    entered = threading.Event()
    release = threading.Event()

    def delayed(root, data):
        entered.set()
        release.wait(5)
        EpisodeRecording._append(root, data)

    runner = live_runner(tmp_path, recording_options={"write_record": delayed})
    try:
        runner._step_once()
        assert entered.wait(2)
        recording = runner.recording
        descriptor = recording.reserve_read()
        assert descriptor["written"] == 0
        release.set()
        recording.close()
        assert recording.root.exists()
        assert RecordedPrefix(descriptor).transition(1)["step"] == 1
        recording.release_read()
        assert not recording.root.exists()
    finally:
        release.set()
        runner.stop()


def test_blocked_update_read_does_not_block_http(tmp_path):
    import asyncio
    import threading
    from aiohttp import ClientSession
    from gradlab.play_web import PlaybackWebServer, playback_updates
    from tests.test_play_web import HumanRecordingRunner, FakeHumanSession, human_args

    async def scenario():
        entered = threading.Event()
        release = threading.Event()
        runner = HumanRecordingRunner(FakeHumanSession(), human_args())

        def blocked():
            entered.set()
            release.wait(5)
            return playback_updates(runner)

        runner.playback_updates = blocked
        server = PlaybackWebServer(runner, human_args())
        task = asyncio.create_task(server.run())
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            async with ClientSession() as client:

                async def get():
                    async with client.get(server.origin + "/assets/app.js") as response:
                        assert response.status == 200
                        await response.read()

                await asyncio.wait_for(get(), timeout=0.5)
            assert not release.is_set()
        finally:
            release.set()
            server.stop_event.set()
            await asyncio.wait_for(task, 5)

    asyncio.run(scenario())
