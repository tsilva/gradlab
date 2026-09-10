"""Full-episode inspection must survive eviction without advancing the policy."""

import asyncio
import base64
import io
import time

import numpy as np
import pytest
from aiohttp import ClientSession
from PIL import Image

from gradlab.play_trajectory import HEADER, INDEX, MAX_RECORD_BYTES
from gradlab.play_web import HISTORY_LIMIT, PlaybackWebServer
from tests.test_play_trajectory import command, live_runner, wait_step


@pytest.mark.parametrize("filtered", [False, True])
def test_recording_reuses_transition_projection_without_changing_inspection(
    tmp_path, monkeypatch, filtered
):
    import gradlab.play_web as player

    runner = live_runner(tmp_path, length=2)
    project = player.transition_payload
    projections = []

    def counted(transition, **kwargs):
        projections.append(transition.sequence)
        return project(transition, **kwargs)

    try:
        if filtered:
            runner.processing_features = frozenset({"game", "history", "rewards"})
        monkeypatch.setattr(player, "transition_payload", counted)
        transition = runner._step_once()
        assert transition is not None
        live = runner.snapshot()
        row = runner.recording.transition(1)
        assert row["inspection_snapshot"]["transition"] == live["transition"]
        assert live["history_point"] == player.history_point_payload(live["transition"])
        assert row["presentation"]["decision"]["value"] == 2.5
        assert row["presentation"]["cnn"]["status"] == "not-recorded"
        assert live["transition"]["cnn"]["status"] == "off"
        # The full archive and a filtered viewer each need at most one conversion.
        assert projections == [1] * (2 if filtered else 1)
        runner._step_once()
        retained = runner.recording.transition(1)
        assert retained["inspection_snapshot"] == row["inspection_snapshot"]
        assert retained["presentation"] == row["presentation"]
        np.testing.assert_array_equal(retained["observation"]["image"], row["observation"]["image"])
    finally:
        runner.stop()


def test_seek_before_memory_window_preserves_live_trajectory_and_exact_frames(
    tmp_path, monkeypatch
):
    from gradlab.play_session import _PlaybackTransition

    monkeypatch.setattr(
        _PlaybackTransition,
        "events",
        property(lambda transition: ("brick",) if transition.step in (1, 100, 4000) else ()),
    )
    runner = live_runner(tmp_path, length=HISTORY_LIMIT + 10)
    try:
        runner.target_fps = 0
        for batch in range(41):
            command(runner, "step", count=100)
            wait_step(runner, (batch + 1) * 100)
        command(runner, "step", count=1)
        latest = wait_step(runner, HISTORY_LIMIT + 5)
        assert runner.history[0]["step"] > 1
        status = runner.recording_status()
        assert status["first_step"] == 1
        assert status["last_step"] == HISTORY_LIMIT + 5
        result = runner.inspect_recorded_step(status["episode_id"], 1)
        snapshot = result["snapshot"]
        assert snapshot["transition"]["step"] == 1
        assert snapshot["session"]["step"] == 1
        assert snapshot["transition"]["reward"]["return"] == 0.5
        assert snapshot["transition"]["decision"]["value"] == 2.5
        assert result["points"][0]["step"] == 1
        assert len(result["points"]) <= 129
        assert snapshot["episode_rewards"]["final"] == 0.5
        middle = runner.inspect_recorded_step(status["episode_id"], 100)["snapshot"]["episode_rewards"]
        assert middle["step"] == 100
        assert middle["final"] == 50
        runner.inspect_recorded_step(status["episode_id"], HISTORY_LIMIT)
        assert runner.inspect_recorded_step(status["episode_id"], 100)["snapshot"]["episode_rewards"] == middle
        # Return to the first page before checking sequential page reuse below.
        runner.inspect_recorded_step(status["episode_id"], 1)
        game = next(frame for frame in result["frames"] if frame["kind"] == 1)
        pixels = np.asarray(Image.open(io.BytesIO(base64.b64decode(game["png"]))))
        assert np.all(pixels == 1)
        assert runner.snapshot() == latest
        assert runner.session.sequence == HISTORY_LIMIT + 5
        overview = runner.history_payload()["timeline"]
        assert overview["through_step"] == HISTORY_LIMIT + 5
        assert [point["step"] for point in overview["points"]] == [1, 100, 4000]
        runner.inspect_recorded_step(status["episode_id"], 1)
        assert runner.history_payload()["timeline"] == overview
        # Sequential replay reuses a bounded diagnostic page instead of rereading
        # every neighboring image at every step.
        history_cache = runner._inspection_history
        runner.inspect_recorded_step(status["episode_id"], 2)
        assert runner._inspection_history is history_cache
        with pytest.raises(ValueError, match="replaced"):
            runner.inspect_recorded_step("old-episode", 1)
        for step in (0, HISTORY_LIMIT + 6):
            with pytest.raises(ValueError, match="outside"):
                runner.inspect_recorded_step(status["episode_id"], step)
    finally:
        runner.stop()


def _episode_inspection_worker(*args):
    from pathlib import Path
    from unittest.mock import patch
    from gradlab.model_sources import ResolvedModelSource
    from gradlab.play_application import PlaybackHost
    from gradlab.play_runtime import ActivePlayback, PlaySourceSpec
    from gradlab.playback_worker import _worker_main
    from gradlab.policy_bundle import load_policy_bundle

    def start(host):
        root = Path(args[1].fixture_root)
        runner = live_runner(root)
        command(runner, "step", count=2)
        wait_step(runner, 2)
        host._active = ActivePlayback(
            runner,
            None,
            PlaySourceSpec("local", "fixture"),
            ResolvedModelSource(root / "model.zip", load_policy_bundle(root)),
        )
        host._phase = "active"

    with patch.object(PlaybackHost, "start", start):
        _worker_main(*args)


def test_isolated_worker_inspects_recorded_steps_and_survives_stale_requests(tmp_path, monkeypatch):
    from argparse import Namespace
    import gradlab.playback_worker as worker

    monkeypatch.setattr(worker, "_worker_main", _episode_inspection_worker)
    host = worker.IsolatedPlaybackHost(
        Namespace(fps=30, fixture_root=str(tmp_path)), argv=[], explicit_seed=False
    )
    try:
        host.start()
        live = host.snapshot()
        result = host.inspect_recorded_step(
            live["session_epoch"], live["trajectory"]["episode_id"], 1
        )
        assert result["snapshot"]["transition"]["step"] == 1
        assert result["frames"]
        with pytest.raises(RuntimeError, match="replaced"):
            host.inspect_recorded_step(999, live["trajectory"]["episode_id"], 1)
        assert host.snapshot()["transition"]["step"] == 2
    finally:
        host.stop()


def test_writer_failure_preserves_pending_step_for_inspection(tmp_path):
    def fail_write(root, data):
        raise OSError("disk unavailable")

    runner = live_runner(tmp_path, recording_options={"write_record": fail_write})
    try:
        command(runner, "step", count=1)
        wait_step(runner, 1)
        deadline = time.monotonic() + 2
        while not runner.recording_status()["error"] and time.monotonic() < deadline:
            time.sleep(0.005)
        status = runner.recording_status()
        assert status["error"] == "disk unavailable"
        assert status["written"] == 0
        result = runner.inspect_recorded_step(status["episode_id"], 1)
        assert result["snapshot"]["transition"]["step"] == 1
    finally:
        runner.stop()


def test_storage_budget_pauses_before_advancing_and_keeps_prefix(tmp_path):
    runner = live_runner(tmp_path)
    try:
        command(runner, "step", count=1)
        wait_step(runner, 1)
        recording = runner.recording
        recording.max_bytes = (
            recording.status()["storage_bytes"] + MAX_RECORD_BYTES + HEADER.size + INDEX.size - 1
        )
        command(runner, "step", count=1)
        deadline = time.monotonic() + 2
        while runner.snapshot()["run_state"] != "paused" and time.monotonic() < deadline:
            time.sleep(0.005)
        assert runner.session.sequence == 1
        assert "storage limit reached" in runner.snapshot()["status_message"]
        assert recording.transition(1)["step"] == 1
        assert recording.status()["transitions"] == 1
    finally:
        runner.stop()


def test_recorded_step_http_is_authenticated_and_rejects_stale_sessions(tmp_path):
    runner = live_runner(tmp_path)
    command(runner, "step", count=2)
    wait_step(runner, 2)

    async def scenario():
        from gradlab.model_sources import ResolvedModelSource
        from gradlab.play_application import PlaybackHost
        from gradlab.play_runtime import PlaybackLoader, ActivePlayback, PlaySourceSpec
        from gradlab.policy_bundle import load_policy_bundle

        host = PlaybackHost(PlaybackLoader(runner.args, argv=[], explicit_seed=False))
        host._active = ActivePlayback(
            runner,
            None,
            PlaySourceSpec("local", "fixture"),
            ResolvedModelSource(tmp_path / "model.zip", load_policy_bundle(tmp_path)),
        )
        host._phase = "active"
        server = PlaybackWebServer(host, runner.args)
        task = asyncio.create_task(server.run())
        while not server.origin:
            await asyncio.sleep(0.01)
        try:
            async with ClientSession() as client:
                url = server.origin + "/api/playback/recorded-step"
                params = {
                    "epoch": 0,
                    "episode_id": runner.recording_status()["episode_id"],
                    "step": 1,
                }
                async with client.get(url, params=params) as response:
                    assert response.status == 401
                headers = {"Authorization": f"Bearer {server.token}", "Origin": server.origin}
                async with client.get(url, params=params, headers=headers) as response:
                    assert response.status == 200, await response.text()
                    result = await response.json()
                    assert result["snapshot"]["transition"]["step"] == 1
                    assert result["frames"]
                    assert response.headers["Cache-Control"] == "no-store"
                params["epoch"] = 999
                async with client.get(url, params=params, headers=headers) as response:
                    assert response.status == 400
                assert runner.session.sequence == 2
        finally:
            server.stop_event.set()
            await task

    try:
        asyncio.run(scenario())
    finally:
        runner.stop()
