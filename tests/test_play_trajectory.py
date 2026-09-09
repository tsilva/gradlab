"""Episode archive behavior at the Player runner and worker boundaries."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import replace
import time
import asyncio
import threading
import zipfile
import json

import numpy as np
import pytest
from aiohttp import ClientSession, WSMsgType

from gradlab.play_debug import PolicyDecision
from gradlab.play_session import _PlaybackTransition
from gradlab.play_web import PlaybackCommand, WebPlaybackRunner
from tests.test_policy_bundle import write_bundle


class ScriptedSession:
    def __init__(self, length=3):
        self.length = length
        self.config = {"game": "Game-v0"}
        self.sequence = self.step_index = 0
        self.episode = 1
        self.active_seed = 40_000
        self.active_task = None
        self.total_reward = 0.0
        self.max_x_pos = 0
        self.action_names = ("noop", "right")
        self.interactive = False
        self.last_transition = None
        self.current_frame = np.zeros((2, 3, 3), dtype=np.uint8)
        self.frames = ()
        self.model = Namespace(gamma=0.9)

    def step(self, *, deterministic):
        self.sequence += 1
        self.step_index += 1
        step = self.step_index
        obs = {
            "image": np.full((1, 2, 2, 3), step % 256, dtype=np.uint8),
            "context": np.array([[0.5, -2.0]], dtype="<f4"),
        }
        transition = _PlaybackTransition(
            sequence=self.sequence,
            episode=self.episode,
            step=step,
            seed=self.active_seed,
            start_id="start-a",
            model_obs=obs,
            decision=PolicyDecision(
                np.array([1]),
                np.array([1]),
                "stochastic",
                value=2.5,
                probabilities=np.array([0.25, 0.75]),
            ),
            action_source="policy",
            executed_action=np.array([1], dtype=np.int16),
            diagnostics=None,
            info={"signal": step},
            before_frame=self.current_frame.copy(),
            after_frame=np.full((2, 3, 3), step % 256, dtype=np.uint8),
            before_frames=(obs["image"][0, 0, ..., None],),
            after_frames=(),
            attribution=None,
            pre_task=None,
            next_task=None,
            reward=0.5,
            total_reward=step * 0.5,
            max_x_pos=step,
            terminated=step == self.length,
            truncated=False,
            completed=False,
            boundary=step == self.length,
            after_frame_role="terminal_observation"
            if step == self.length
            else "after_action_observation",
        )
        self.current_frame = transition.after_frame.copy()
        self.total_reward = transition.total_reward
        self.last_transition = transition
        if transition.boundary:
            self.episode += 1
            self.step_index = 0
        return replace(
            transition, next_model_obs={"image": obs["image"] + 1, "context": obs["context"].copy()}
        )


def command(runner, name, **payload):
    runner.submit(PlaybackCommand(name, "test", name, payload, None))
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if not runner.responses.empty():
            result = runner.responses.get_nowait().payload
            assert result["ok"], result
            return
        time.sleep(0.005)
    raise AssertionError("Player did not answer command")


def wait_step(runner, step):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        snapshot = runner.snapshot()
        if (snapshot.get("transition") or {}).get("step") == step:
            return snapshot
        time.sleep(0.005)
    raise AssertionError(f"Player did not reach step {step}: {runner.snapshot()}")


def live_runner(tmp_path, length=3, **options):
    from gradlab.policy_bundle import load_policy_bundle

    write_bundle(tmp_path)
    runner = WebPlaybackRunner(
        ScriptedSession(length),
        Namespace(fps=240, episodes=0, port=0, no_open=True),
        config_text="game: Game-v0",
        trajectory_bundle=load_policy_bundle(tmp_path),
        **options,
    )
    runner.start()
    return runner


def test_current_episode_download_import_preserves_exact_inputs_and_checkpoint(tmp_path):
    from gradlab.play_trajectory import export_trajectory
    from gradlab.play_trajectory_runner import TrajectoryPlaybackRunner

    runner = live_runner(tmp_path)
    imported = None
    try:
        assert runner.recording_status()["enabled"] is True
        command(runner, "step", count=3)
        wait_step(runner, 3)
        frozen = runner.freeze_trajectory()
        archive = export_trajectory(frozen, tmp_path / "episode.gradtraj")
        imported = TrajectoryPlaybackRunner(archive, runner.args)
        imported.start()
        command(imported, "seek", step=2)
        snapshot = imported.snapshot()
        assert snapshot["transition"]["reward"]["return"] == 1.0
        assert snapshot["transition"]["decision"]["value"] == 2.5
        assert snapshot["trajectory"]["complete"] is True
        assert snapshot["trajectory"]["scientific_evidence"] is False
        row = imported.recorded_transition(2)
        np.testing.assert_array_equal(
            row["observation"]["image"], np.full((1, 2, 2, 3), 2, np.uint8)
        )
        assert row["observation"]["context"].dtype == np.dtype("<f4")
        assert row["executed_action"].dtype == np.dtype("int16")
        np.testing.assert_array_equal(row["after_image"], np.full((2, 3, 3), 2, np.uint8))
        assert imported.checkpoint_path.read_bytes() == (tmp_path / "model.zip").read_bytes()
        command(imported, "seek", step=1)
        assert imported.snapshot()["transition"]["step"] == 1
    finally:
        runner.stop()
        if imported is not None:
            imported.stop()


def test_unfinished_download_survives_replacement_and_opens_in_isolated_worker(tmp_path):
    from gradlab.play_trajectory import export_trajectory
    from gradlab.playback_worker import IsolatedPlaybackHost

    runner = live_runner(tmp_path, length=4)
    host = IsolatedPlaybackHost(Namespace(fps=30), argv=[], explicit_seed=False)
    try:
        command(runner, "set_recording", enabled=True)
        command(runner, "step", count=2)
        wait_step(runner, 2)
        frozen = runner.freeze_trajectory()
        command(runner, "step", count=2)
        wait_step(runner, 4)
        command(runner, "next_episode")
        runner.stop()
        archive = export_trajectory(frozen, tmp_path / "prefix.gradtraj")
        host.start()
        host.import_trajectory(str(archive))
        host.submit(PlaybackCommand("seek", "test", "seek", {"step": 2}, None))
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            result = host.poll_response()
            if result is not None:
                assert result.payload["ok"]
                break
            time.sleep(0.01)
        snapshot = host.snapshot()
        assert snapshot["trajectory"]["last_step"] == 2
        assert snapshot["trajectory"]["complete"] is False
        assert snapshot["transition"]["boundary"] is False
        assert snapshot["transition"]["terminated"] is False
        assert snapshot["transition"]["truncated"] is False
        assert snapshot["session_epoch"] == 1
        assert host.active_publication_context() is None
    finally:
        runner.stop()
        host.stop()


def test_server_download_is_pinned_and_import_requires_control(tmp_path):
    from gradlab.play_application import PlaybackHost
    from gradlab.play_runtime import PlaybackLoader, ActivePlayback, PlaySourceSpec
    from gradlab.model_sources import ResolvedModelSource
    from gradlab.play_web import PlaybackWebServer
    from gradlab.policy_bundle import load_policy_bundle

    async def scenario():
        runner = live_runner(tmp_path)
        command(runner, "set_recording", enabled=True)
        command(runner, "step", count=2)
        wait_step(runner, 2)
        host = PlaybackHost(PlaybackLoader(runner.args, argv=[], explicit_seed=False))
        # Install the scripted provider at the runner/host boundary.
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
        headers = {"Authorization": f"Bearer {server.token}", "Origin": server.origin}
        try:
            async with ClientSession() as client:
                async with client.post(
                    server.origin + "/api/trajectory/download", headers=headers
                ) as response:
                    assert response.status == 200, await response.text()
                    url = (await response.json())["url"]
                runner.stop()
                async with client.get(server.origin + url) as response:
                    assert response.status == 200
                    archive = await response.read()
                async with client.post(
                    server.origin + "/api/trajectory/import", data=archive, headers=headers
                ) as response:
                    assert response.status == 403
                socket = await client.ws_connect(server.origin + "/ws", origin=server.origin)
                await socket.send_json(
                    {
                        "type": "hello",
                        "token": server.token,
                        "subscriptions": ["telemetry", "game", "observation"],
                    }
                )
                welcome = await socket.receive_json()
                headers["X-Gradlab-Client"] = welcome["client_id"]
                headers["X-Gradlab-Control-Epoch"] = str(server.control_epoch)
                async with client.post(
                    server.origin + "/api/trajectory/import", data=archive, headers=headers
                ) as response:
                    assert response.status == 200, await response.text()
                assert host.snapshot()["mode"] == "trajectory"
                assert host.snapshot()["trajectory"]["last_step"] == 2
                from gradlab.play_web import FRAME_HEADER, FRAME_GAME

                for step in (2, 1):
                    await socket.send_json(
                        {
                            "type": "command",
                            "id": f"seek-{step}",
                            "name": "seek",
                            "payload": {"step": step},
                        }
                    )
                    received_snapshot = received_frame = False
                    while not (received_snapshot and received_frame):
                        message = await socket.receive(timeout=5)
                        if message.type == WSMsgType.BINARY:
                            _, kind, _, _, epoch, sequence, _ = FRAME_HEADER.unpack_from(
                                message.data
                            )
                            received_frame |= kind == FRAME_GAME and epoch == 1 and sequence == step
                        elif message.type == WSMsgType.TEXT:
                            payload = json.loads(message.data)
                            received_snapshot |= (
                                payload.get("mode") == "trajectory"
                                and (payload.get("transition") or {}).get("step") == step
                            )
                    assert received_frame and received_snapshot
                await socket.close()
        finally:
            server.stop_event.set()
            await task

    asyncio.run(scenario())


@pytest.mark.parametrize("fail", [False, True])
def test_storage_backpressure_pauses_without_losing_a_transition(tmp_path, fail):
    from gradlab.play_trajectory import EpisodeRecording, export_trajectory
    from gradlab.play_trajectory_runner import TrajectoryPlaybackRunner

    release = threading.Event()
    writing = threading.Event()

    def sink(root, data):
        writing.set()
        if fail and not release.is_set():
            raise OSError("scripted storage failure")
        assert release.wait(10)
        EpisodeRecording._append(root, data)

    runner = live_runner(tmp_path, recording_options={"buffer_bytes": 1, "write_record": sink})
    imported = None
    try:
        command(runner, "set_recording", enabled=True)
        command(runner, "play")
        assert writing.wait(5)
        deadline = time.monotonic() + 5
        while runner.snapshot()["run_state"] != "paused" and time.monotonic() < deadline:
            time.sleep(0.01)
        snapshot = runner.snapshot()
        assert snapshot["run_state"] == "paused"
        assert "Recording storage" in snapshot["status_message"]
        assert snapshot["transition"]["step"] == 1
        if fail:
            archive = export_trajectory(
                runner.freeze_trajectory(), tmp_path / "failed-prefix.gradtraj"
            )
            imported = TrajectoryPlaybackRunner(archive, runner.args)
            assert imported.recorded_transition(1)["reward"] == 0.5
            assert imported.metadata["transition_count"] == 1
        release.set()
        command(runner, "set_recording", enabled=True)
        # Allow a full transition-sized slot for the remainder of this scripted episode.
        command(runner, "step", count=1)
        wait_step(runner, 2)
        command(runner, "step", count=1)
        wait_step(runner, 3)
        assert runner.snapshot()["trajectory"]["transitions"] == 3
    finally:
        release.set()
        runner.stop()
        if imported:
            imported.stop()


def test_episode_longer_than_live_history_retains_its_first_transition(tmp_path):
    from gradlab.play_trajectory import export_trajectory
    from gradlab.play_trajectory_runner import TrajectoryPlaybackRunner
    from gradlab.play_web import HISTORY_LIMIT

    runner = live_runner(tmp_path, length=HISTORY_LIMIT + 5)
    runner.target_fps = 0
    imported = None
    try:
        command(runner, "set_recording", enabled=True)
        command(runner, "play")
        wait_step(runner, HISTORY_LIMIT + 5)
        archive = export_trajectory(runner.freeze_trajectory(), tmp_path / "long.gradtraj")
        imported = TrajectoryPlaybackRunner(archive, runner.args)
        imported.start()
        command(imported, "seek", step=1)
        assert imported.snapshot()["transition"]["reward"]["return"] == 0.5
        command(imported, "seek", step=HISTORY_LIMIT + 5)
        assert imported.snapshot()["transition"]["terminated"] is True
    finally:
        runner.stop()
        if imported:
            imported.stop()


@pytest.mark.parametrize("corruption", ["hash", "path", "version"])
def test_import_rejects_unsafe_or_corrupt_archives(tmp_path, corruption):
    from gradlab.play_trajectory import export_trajectory
    from gradlab.play_trajectory_runner import TrajectoryPlaybackRunner

    runner = live_runner(tmp_path)
    try:
        command(runner, "set_recording", enabled=True)
        command(runner, "step", count=1)
        wait_step(runner, 1)
        archive = export_trajectory(runner.freeze_trajectory(), tmp_path / "episode.gradtraj")
        damaged = tmp_path / "damaged.gradtraj"
        with zipfile.ZipFile(archive) as source, zipfile.ZipFile(damaged, "w") as target:
            for member in source.infolist():
                data = source.read(member)
                if corruption == "hash" and member.filename == "checkpoint/model.zip":
                    data = data[:-1] + b"!"
                if corruption == "version" and member.filename == "manifest.json":
                    data = data.replace(b'"format_version": 1', b'"format_version": 9')
                target.writestr(member.filename, data)
            if corruption == "path":
                target.writestr("../escape.py", "raise RuntimeError('unsafe')")
        with pytest.raises(ValueError, match="hash|unsafe|version"):
            TrajectoryPlaybackRunner(damaged, runner.args)
        assert not (tmp_path / "escape.py").exists()
    finally:
        runner.stop()


@pytest.mark.parametrize("terminal_available", [False, True])
def test_capture_owns_hidden_policy_inputs_and_never_records_autoreset_as_terminal(
    tmp_path, terminal_available
):
    from gradlab.environment_fields import EnvConfig
    from gradlab.play_session import _PlaybackSession
    from gradlab.policy_runtime import POLICY_CAPABILITIES, PolicyBatchDecision
    from gradlab.policy_bundle import load_policy_bundle
    from gradlab.play_trajectory import export_trajectory
    from gradlab.play_trajectory_runner import TrajectoryPlaybackRunner

    class Provider:
        reset_infos = [{}]

        def __init__(self):
            self.obs = {
                "image": np.zeros((1, 4, 2, 2), np.uint8),
                "context": np.array([[0.5, -2]], dtype="<f4"),
            }
            self.frame = np.zeros((2, 3, 3), np.uint8)
            self.actions = []

        def seed(self, seed):
            pass

        def reset(self):
            self.obs["image"].fill(7)
            return self.obs

        def get_images(self):
            return [self.frame]

        def step(self, action):
            self.actions.append(action.copy())
            terminal = {key: value.copy() for key, value in self.obs.items()}
            terminal["image"].fill(9)
            self.obs["image"].fill(99)
            self.obs["context"].fill(88)
            self.frame.fill(99)
            info = {"terminal_observation": terminal} if terminal_available else {}
            return self.obs, np.array([1.0]), np.array([True]), [info]

        def take_step_diagnostics(self):
            return None

        def drain_records(self):
            return []

    class Runtime:
        capabilities = POLICY_CAPABILITIES["ppo"]

        def __init__(self):
            self.random = np.random.default_rng(1234)
            self.calls = 0

        def decide(self, observation, **kwargs):
            self.calls += 1
            action = np.array([self.random.integers(2)], np.int64)
            decision = PolicyDecision(action, action, "stochastic")
            return PolicyBatchDecision("stochastic", "stochastic", action, (decision,))

    write_bundle(tmp_path)
    actions = []
    for enabled in (False, True):
        provider, runtime = Provider(), Runtime()
        session = _PlaybackSession(
            model=Namespace(gamma=0.9),
            env=provider,
            config=EnvConfig(game="CartPole-v1"),
            initial_seed=40_000,
            policy_runtime=runtime,
        )
        session.restart()
        runner = WebPlaybackRunner(
            session,
            Namespace(fps=30, episodes=1),
            config_text="",
            trajectory_bundle=load_policy_bundle(tmp_path),
        )
        runner.set_processing([])
        runner.start()
        imported = None
        try:
            command(runner, "set_recording", enabled=enabled)
            command(runner, "step", count=1)
            wait_step(runner, 1)
            actions.append(provider.actions[0])
            assert runtime.calls == 1
            if enabled:
                archive = export_trajectory(
                    runner.freeze_trajectory(), tmp_path / "terminal.gradtraj"
                )
                imported = TrajectoryPlaybackRunner(archive, runner.args)
                row = imported.recorded_transition(1)
                np.testing.assert_array_equal(
                    row["observation"]["image"], np.full((1, 4, 2, 2), 7, np.uint8)
                )
                np.testing.assert_array_equal(row["observation"]["context"], [[0.5, -2.0]])
                assert row["after_image"] is None
                assert row["after_image_status"] == "terminal_missing"
                if terminal_available:
                    np.testing.assert_array_equal(
                        row["next_observation"]["image"], np.full((1, 4, 2, 2), 9, np.uint8)
                    )
                else:
                    assert row["next_observation"] is None
                    assert row["next_observation_status"] == "terminal_missing"
        finally:
            runner.stop()
            if imported:
                imported.stop()
    np.testing.assert_array_equal(actions[0], actions[1])


@pytest.mark.parametrize(
    "corruption", ["missing_metadata", "false_completion", "false_terminal", "heavy_diagnostic"]
)
def test_import_rejects_semantically_invalid_but_rehashed_archives(tmp_path, corruption):
    import hashlib
    import pyarrow as pa
    import pyarrow.parquet as pq
    from gradlab.play_trajectory import export_trajectory, parquet_schema
    from gradlab.play_trajectory_runner import TrajectoryPlaybackRunner

    runner = live_runner(tmp_path)
    try:
        command(runner, "step", count=1)
        wait_step(runner, 1)
        archive = export_trajectory(runner.freeze_trajectory(), tmp_path / "valid.gradtraj")
        with zipfile.ZipFile(archive) as source:
            files = {name: source.read(name) for name in source.namelist()}
        metadata = json.loads(files["metadata.json"])
        if corruption == "missing_metadata":
            del metadata["first_step"]
        elif corruption == "false_completion":
            metadata["complete"] = True
        else:
            rows = pq.read_table(
                pa.BufferReader(files["data/train-00000-of-00001.parquet"])
            ).to_pylist()
            if corruption == "false_terminal":
                rows[0]["next_observation_status"] = "terminal_missing"
            else:
                presentation = json.loads(rows[0]["presentation"])
                presentation["cnn"]["status"] = "ready"
                rows[0]["presentation"] = json.dumps(presentation)
            target = pa.BufferOutputStream()
            pq.write_table(pa.Table.from_pylist(rows, schema=parquet_schema()), target)
            files["data/train-00000-of-00001.parquet"] = target.getvalue().to_pybytes()
        files["metadata.json"] = json.dumps(metadata).encode()
        manifest = json.loads(files["manifest.json"])
        for name, data in files.items():
            if name != "manifest.json":
                manifest["files"][name] = {
                    "size": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
        files["manifest.json"] = json.dumps(manifest).encode()
        damaged = tmp_path / "invalid.gradtraj"
        with zipfile.ZipFile(damaged, "w") as target:
            for name, data in files.items():
                target.writestr(name, data)
        with pytest.raises(ValueError, match="Invalid trajectory archive"):
            TrajectoryPlaybackRunner(damaged, runner.args)
        # Rejection does not damage the live source.
        command(runner, "step", count=1)
        assert wait_step(runner, 2)["transition"]["step"] == 2
    finally:
        runner.stop()


def test_downloading_a_flushing_prefix_does_not_lock_live_stepping(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from gradlab.play_trajectory import EpisodeRecording, export_trajectory
    from gradlab.play_trajectory_runner import TrajectoryPlaybackRunner

    entered, release = threading.Event(), threading.Event()

    def slow_sink(root, data):
        entered.set()
        assert release.wait(10)
        EpisodeRecording._append(root, data)

    runner = live_runner(tmp_path, recording_options={"write_record": slow_sink})
    imported = None
    try:
        command(runner, "step", count=1)
        wait_step(runner, 1)
        assert entered.wait(2)
        with ThreadPoolExecutor() as executor:
            future = executor.submit(runner.freeze_trajectory)
            try:
                # The downloader is waiting for the blocked sink while the runner advances.
                command(runner, "step", count=1)
                wait_step(runner, 2)
            finally:
                release.set()
            frozen = future.result(timeout=5)
        command(runner, "step", count=1)
        wait_step(runner, 3)
        runner.stop()
        archive = export_trajectory(frozen, tmp_path / "prefix.gradtraj")
        imported = TrajectoryPlaybackRunner(archive, runner.args)
        assert imported.metadata["transition_count"] in (1, 2)
        assert imported.metadata["complete"] is False
        assert imported.recorded_transition(1)["step"] == 1
    finally:
        release.set()
        runner.stop()
        if imported:
            imported.stop()


def test_recording_preserves_training_overrides_and_later_critic_incomparability(tmp_path):
    from gradlab.play_trajectory import export_trajectory
    from gradlab.play_trajectory_runner import TrajectoryPlaybackRunner

    runner = live_runner(
        tmp_path,
        contract_details={
            "mode": "training",
            "requested_policy_override_paths": ["/train/environment/frame_skip"],
        },
        value_contract={"discount": 0.9, "action_sampling": "stochastic"},
    )
    imported = None
    try:
        command(runner, "step", count=1)
        wait_step(runner, 1)
        command(runner, "set_action_selection_mode", mode="deterministic")
        command(runner, "step", count=1)
        wait_step(runner, 2)
        archive = export_trajectory(runner.freeze_trajectory(), tmp_path / "mode.gradtraj")
        imported = TrajectoryPlaybackRunner(archive, runner.args)
        assert imported.recorded_transition(1)["classification"] == "faithful"
        imported.start()
        command(imported, "seek", step=2)
        assert imported.snapshot()["trajectory"]["classification"] == "counterfactual"
        assert any(
            "deterministic" in reason
            for reason in imported.snapshot()["session"]["critic_comparison"]["reasons"]
        )
    finally:
        runner.stop()
        if imported:
            imported.stop()


def _slow_transfer_worker(*args):
    """Replace only the disk-export boundary inside the real isolated worker."""
    from pathlib import Path
    from unittest.mock import patch
    from gradlab.play_application import PlaybackHost
    from gradlab.playback_worker import _worker_main

    def freeze(host):
        root = Path(args[1].transfer_gate)
        (root / "entered").touch()
        deadline = time.monotonic() + 10
        while not (root / "release").exists():
            if time.monotonic() > deadline:
                raise TimeoutError("test sink was not released")
            time.sleep(0.01)
        return str(root / "frozen")

    with patch.object(PlaybackHost, "freeze_trajectory", freeze):
        _worker_main(*args)


def test_isolated_worker_keeps_serving_snapshots_during_a_slow_transfer(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import gradlab.playback_worker as worker

    monkeypatch.setattr(worker, "_worker_main", _slow_transfer_worker)
    host = worker.IsolatedPlaybackHost(
        Namespace(fps=30, transfer_gate=str(tmp_path)), argv=[], explicit_seed=False
    )
    try:
        host.start()
        with ThreadPoolExecutor() as executor:
            transfer = executor.submit(host.freeze_trajectory)
            try:
                deadline = time.monotonic() + 5
                while not (tmp_path / "entered").exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert (tmp_path / "entered").exists()
                snapshot = executor.submit(host.snapshot)
                assert snapshot.result(timeout=2)["app"]["phase"] == "selecting"
                assert not transfer.done()
            finally:
                (tmp_path / "release").touch()
            assert transfer.result(timeout=5) == str(tmp_path / "frozen")
    finally:
        host.stop()


@pytest.mark.parametrize(
    "seed_source,seed,expected",
    [
        ("training", 40000, "counterfactual"),
        ("evaluation", 40000, "evaluation_reproduction"),
        ("evaluation", 40001, "counterfactual"),
    ],
)
def test_evaluation_reproduction_requires_the_recorded_evaluation_seed(
    tmp_path, seed_source, seed, expected
):
    from gradlab.play_trajectory import export_trajectory
    from gradlab.play_trajectory_runner import TrajectoryPlaybackRunner

    runner = live_runner(
        tmp_path,
        contract_details={
            "mode": "evaluation",
            "playback_seed_source": seed_source,
            "playback_seed": seed,
        },
    )
    imported = None
    try:
        command(runner, "step", count=1)
        wait_step(runner, 1)
        archive = export_trajectory(runner.freeze_trajectory(), tmp_path / "seed.gradtraj")
        imported = TrajectoryPlaybackRunner(archive, runner.args)
        assert imported.recorded_transition(1)["classification"] == expected
    finally:
        runner.stop()
        if imported:
            imported.stop()


def test_pre_episode_boundary_edits_are_recorded_and_one_transition_can_replay(tmp_path):
    from gradlab.play_trajectory import export_trajectory
    from gradlab.play_trajectory_runner import TrajectoryPlaybackRunner

    runner = live_runner(tmp_path, length=1)
    session = runner.session
    session.termination_base_config = dict(session.config)

    def change_conditions(enabled):
        session.config = {**session.config, "task": {"termination": enabled}}

    session.set_termination_conditions = change_conditions
    imported = None
    try:
        command(runner, "set_termination_conditions", enabled=["custom_boundary"])
        command(runner, "step", count=1)
        wait_step(runner, 1)
        archive = export_trajectory(runner.freeze_trajectory(), tmp_path / "boundary.gradtraj")
        imported = TrajectoryPlaybackRunner(archive, runner.args)
        assert imported.metadata["resolved_environment"]["task"]["termination"] == [
            "custom_boundary"
        ]
        assert imported.metadata["classification"] == "counterfactual"
        imported.start()
        command(imported, "seek", step=1)
        command(imported, "replay")
        assert wait_step(imported, 1)["transition"]["terminated"] is True
    finally:
        runner.stop()
        if imported:
            imported.stop()


def test_completed_evaluation_retains_its_seed_and_execution_provenance(tmp_path):
    from gradlab.play_trajectory import export_trajectory
    from gradlab.play_trajectory_runner import TrajectoryPlaybackRunner

    runner = live_runner(
        tmp_path,
        length=1,
        contract_details={
            "mode": "evaluation",
            "playback_seed_source": "evaluation",
            "playback_seed": 40000,
        },
        capture_context={
            "execution": {
                "provider_id": "fixture",
                "runtime_versions": {"fixture": "1.0"},
                "hostname": "private-host",
            }
        },
    )
    step = runner.session.step

    def advance_seed(**kwargs):
        transition = step(**kwargs)
        runner.session.active_seed = 40001
        return transition

    runner.session.step = advance_seed
    imported = None
    try:
        command(runner, "step", count=1)
        wait_step(runner, 1)
        archive = export_trajectory(runner.freeze_trajectory(), tmp_path / "complete.gradtraj")
        imported = TrajectoryPlaybackRunner(archive, runner.args)
        assert imported.metadata["classification"] == "evaluation_reproduction"
        assert imported.recorded_transition(1)["seed"] == 40000
        assert imported.metadata["execution"] == {
            "provider_id": "fixture",
            "runtime_versions": {"fixture": "1.0"},
        }
    finally:
        runner.stop()
        if imported:
            imported.stop()


def test_cancelled_download_keeps_its_files_until_freeze_finishes(tmp_path):
    from pathlib import Path
    from gradlab.play_trajectory_http import TrajectoryTransfers

    runner = live_runner(tmp_path)
    entered, release = threading.Event(), threading.Event()
    frozen_paths = []
    freeze = runner.freeze_trajectory

    def delayed_freeze():
        entered.set()
        assert release.wait(5)
        result = freeze()
        frozen_paths.append(Path(result))
        return result

    async def scenario():
        transfers = TrajectoryTransfers(
            Namespace(freeze_trajectory=delayed_freeze), lambda request: None, lambda request: None
        )
        request = asyncio.create_task(transfers.prepare(None))
        try:
            deadline = time.monotonic() + 2
            while not entered.is_set() and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            assert entered.is_set()
            request.cancel()
            await asyncio.sleep(0.01)
            assert not request.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await request
            assert frozen_paths and all(not path.exists() for path in frozen_paths)
            assert not transfers.downloads
        finally:
            release.set()
            transfers.close()

    try:
        command(runner, "step", count=1)
        wait_step(runner, 1)
        asyncio.run(scenario())
    finally:
        runner.stop()


def test_checkpoint_activation_keeps_download_available_after_candidate_cleanup(
    tmp_path, monkeypatch
):
    from gradlab.environment_fields import EnvConfig
    from gradlab.model_sources import ResolvedModelSource
    from gradlab.play_runtime import PlaybackCandidate, PlaybackLoader, PlaySourceSpec
    from gradlab.play_session import build_parser
    from gradlab.policy_bundle import load_policy_bundle
    from gradlab.play_trajectory import export_trajectory
    from gradlab.play_trajectory_runner import TrajectoryPlaybackRunner
    from gradlab.trusted_inputs import stage_model_input

    write_bundle(tmp_path)
    bundle = load_policy_bundle(tmp_path)
    checkpoint_bytes = bundle.checkpoint_path.read_bytes()
    config = EnvConfig(game="CartPole-v1", env_provider="gymnasium")
    args = build_parser().parse_args(["--model", str(bundle.checkpoint_path), "--seed", "40000"])
    source = ResolvedModelSource(bundle.checkpoint_path, bundle)
    candidate = PlaybackCandidate(
        spec=PlaySourceSpec("local", str(bundle.checkpoint_path)),
        args=args,
        source=source,
        config=config,
        display_config=config,
        rom_binding=None,
        staged=stage_model_input(bundle.checkpoint_path),
        source_identity="fixture",
        artifact_ref=None,
        termination_base_config=config,
        termination_source="training",
    )
    session = ScriptedSession(length=1)
    session.restart = lambda seed: None
    env = Namespace(action_space=None, close=lambda: None)

    def load_model(verified, **kwargs):
        assert verified.model_path.read_bytes() == checkpoint_bytes
        return Namespace(gamma=0.9)

    monkeypatch.setattr("gradlab.policy_models.load_policy_model", load_model)
    monkeypatch.setattr("gradlab.policy_runtime.PolicyRuntime", lambda *a, **k: None)
    monkeypatch.setattr("gradlab.policy_runtime.bind_policy_action_space", lambda *a, **k: None)
    monkeypatch.setattr("gradlab.play_runtime.make_eval_vec_env", lambda **k: env)
    monkeypatch.setattr("gradlab.play_runtime.assert_action_contract_compatible", lambda *a: None)
    monkeypatch.setattr("gradlab.play_runtime._PlaybackSession", lambda **k: session)
    monkeypatch.setattr("gradlab.play_runtime.resolved_play_launch_lines", lambda *a, **k: [])
    monkeypatch.setattr("gradlab.play_runtime.runtime_versions_metadata", lambda: {})
    monkeypatch.setattr("gradlab.play_runtime.player_source_provenance", lambda *a: {})
    active = None
    imported = None
    try:
        active = PlaybackLoader(args, argv=[], explicit_seed=True).activate(
            candidate,
            progress=lambda *a: None,
        )
        # This is the same ownership transfer performed by PlaybackHost.
        candidate.cleanup()
        assert not candidate.staged.root.exists()
        active.runner.start()
        assert active.runner.snapshot()["trajectory"]["available"] is True
        command(active.runner, "step", count=1)
        wait_step(active.runner, 1)
        archive = export_trajectory(active.runner.freeze_trajectory(), tmp_path / "loaded.gradtraj")
        imported = TrajectoryPlaybackRunner(archive, args)
        assert imported.checkpoint_path.read_bytes() == checkpoint_bytes
    finally:
        candidate.cleanup()
        if active:
            active.close()
        if imported:
            imported.stop()


def test_cli_opens_recording_without_a_checkpoint_or_catalog(tmp_path, monkeypatch):
    from gradlab.main import main
    from gradlab.play_trajectory import export_trajectory

    runner = live_runner(tmp_path)
    try:
        command(runner, "step", count=1)
        wait_step(runner, 1)
        archive = export_trajectory(runner.freeze_trajectory(), tmp_path / "episode.gradtraj")
    finally:
        runner.stop()

    def unavailable_catalog(*args, **kwargs):
        raise AssertionError("Opening a recording must not require the catalog")

    def inspect_player(host, args, **kwargs):
        from gradlab.play_web import PlaybackWebServer

        host.start()
        server = PlaybackWebServer(host, args, catalog=kwargs["catalog"])
        asyncio.run(server._prepare_initial_catalog())
        snapshot = host.snapshot()
        assert snapshot["mode"] == "trajectory"
        assert snapshot["app"]["route"] == {"level": "goals", "environment_id": "Game-v0"}
        assert snapshot["app"]["source"] is None
        assert snapshot["trajectory"]["last_step"] == 1
        return 0

    monkeypatch.setattr(
        "gradlab.play_catalog_authority.start_catalog_authority_helper", unavailable_catalog
    )
    monkeypatch.setattr("gradlab.play_catalog.PlayCatalog.initial_environments", unavailable_catalog)
    monkeypatch.setattr("gradlab.play_web.run_web_player_application", inspect_player)
    assert main(["play", "--recording", str(archive), "--no-open"]) == 0


@pytest.mark.parametrize("source", ["--model", "--run", "--recipe"])
def test_cli_rejects_recording_with_another_playback_source(source, capsys):
    from gradlab.main import main

    with pytest.raises(SystemExit) as error:
        main(["play", "--recording", "episode.gradtraj", source, "another-source"])
    assert error.value.code == 2
    assert "pass exactly one" in capsys.readouterr().err
