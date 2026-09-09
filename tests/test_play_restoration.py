"""Exact decision-boundary restoration, independent of the advancing sampling stream."""

from copy import deepcopy

import numpy as np
import pytest

from gradlab.batch_runtime import BatchRuntime, ProviderDescriptor, SignalSpec
from gradlab.task_kernels import IdentityTaskDefinition
from tests.test_state_archive import PortableBreakoutProvider


def runtime_fixture():
    provider = PortableBreakoutProvider(num_envs=1)
    descriptor = ProviderDescriptor(
        provider_id="env-breakoutatari2600-turbo-native",
        native_observation_space=provider.single_observation_space,
        native_action_space=provider.single_action_space,
        signal_schema={"score": SignalSpec("score", np.int64)},
        start_catalog=("Start",),
        supports_live_snapshots=True,
        live_snapshots_deterministic=True,
        snapshot_codec_id="breakout-turbo-env.state-v1",
        snapshot_compatibility_id="test-environment-v1",
    )
    kernel = IdentityTaskDefinition(signals={"score": "score"}).bind(descriptor, 1)
    runtime = BatchRuntime(provider, descriptor, kernel, run_seed=17)
    runtime.reset(seed=17)
    return runtime


def test_decision_boundary_restores_provider_task_and_episode_accounting():
    runtime = runtime_fixture()
    try:
        runtime.step(np.array([1]))
        state = runtime.capture_playback_state()
        expected = deepcopy(runtime.step(np.array([0])))
        runtime.step(np.array([1]))
        runtime.restore_playback_state(state)
        actual = runtime.step(np.array([0]))
        np.testing.assert_array_equal(actual.observations, expected.observations)
        np.testing.assert_array_equal(actual.rewards, expected.rewards)
        np.testing.assert_array_equal(actual.terminated, expected.terminated)
        np.testing.assert_array_equal(actual.truncated, expected.truncated)
        assert runtime.capture_playback_state()["episode_lengths"] == [2]
        assert runtime.capture_playback_state()["episode_returns"] == [2.0]
    finally:
        runtime.close()


def test_restoring_policy_memory_keeps_sampling_stream_and_action_mode():
    import random
    import torch
    from argparse import Namespace
    from gradlab.action_program import ActionProgramPolicy, ActionRun
    from gradlab.play_restoration import LiveRestoration
    from gradlab.policy_runtime import PolicyRuntime
    from gradlab.training.sb3_vec_env import GradLabVecEnv

    runtime = runtime_fixture()
    model = ActionProgramPolicy(
        action_names=("NOOP", "FIRE"),
        action_runs=(ActionRun(1, 3), ActionRun(0, 2)),
        fallback_action=0,
    )
    session = Namespace(
        env=GradLabVecEnv(runtime),
        model=model,
        policy_runtime=PolicyRuntime(model, algorithm_id="action-program"),
        policy_obs=np.array([[0]]),
        current_frame=None,
        frames=None,
        active_task_state=None,
        active_info_value=None,
        active_seed=17,
        episode=1,
        step_index=0,
        total_reward=0.0,
        max_x_pos=0,
        last_transition=None,
        sequence=0,
    )
    adapter = LiveRestoration(session)
    try:
        assert adapter.capability["supported"]
        model.predict(session.policy_obs)
        state = adapter.capture()
        model.predict(session.policy_obs)
        model.predict(session.policy_obs)
        torch.rand(5)
        np.random.random(3)
        random.random()
        torch_state = torch.get_rng_state().clone()
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        adapter.restore(state)
        assert model.predict(session.policy_obs)[0].tolist() == [1]
        assert torch.equal(torch_state, torch.get_rng_state())
        np.testing.assert_array_equal(numpy_state[1], np.random.get_state()[1])
        assert python_state == random.getstate()
    finally:
        runtime.close()


class ScriptedRestoration:
    """Scripted environment/Policy boundary used by the full Player workflow."""

    capability = {"supported": True, "reason": None, "activation": "explicit"}

    def __init__(self, session):
        self.session = session

    def capture(self):
        session = self.session
        return {
            name: deepcopy(getattr(session, name))
            for name in (
                "step_index",
                "episode",
                "total_reward",
                "current_frame",
                "max_x_pos",
            )
        }

    def restore(self, state):
        for name, value in state.items():
            setattr(self.session, name, deepcopy(value))
        self.session.last_transition = None


def test_bookmark_cut_and_resample_preserve_download_and_offline_annotations(tmp_path):
    from tests.test_play_trajectory import live_runner, command, wait_step
    from gradlab.play_trajectory import export_trajectory
    from gradlab.play_trajectory_runner import TrajectoryPlaybackRunner

    runner = live_runner(tmp_path, length=20, restoration_factory=ScriptedRestoration)
    imported = None
    try:
        command(runner, "set_restoration_capture", enabled=True)
        command(runner, "step", count=4)
        wait_step(runner, 4)
        trajectory = runner.snapshot()["trajectory"]
        identity = {
            "episode_id": trajectory["episode_id"],
            "trajectory_revision": trajectory["trajectory_revision"],
        }
        command(runner, "bookmark_add", step=2, name="Rare state", **identity)
        original = runner.freeze_trajectory()
        command(runner, "discard_future", step=2, **identity)
        state = runner.snapshot()
        assert state["run_state"] == "paused"
        assert state["trajectory"]["last_step"] == 2
        assert state["trajectory"]["classification"] == "counterfactual"
        assert state["trajectory"]["bookmarks"][0]["name"] == "Rare state"
        assert state["session"]["sampling_mode"] == "stochastic"
        command(runner, "step", count=1)
        wait_step(runner, 3)
        archive = export_trajectory(runner.freeze_trajectory(), tmp_path / "new.gradtraj")
        imported = TrajectoryPlaybackRunner(archive, runner.args)
        imported.start()
        command(imported, "seek", step=2)
        assert imported.snapshot()["trajectory"]["bookmarks"][0]["step"] == 2
        assert imported.recorded_transition(3)["sequence"] > 4
        assert imported.recorded_transition(2)["sequence"] == 2
        imported.stop()
        imported = TrajectoryPlaybackRunner(
            export_trajectory(original, tmp_path / "old.gradtraj"), runner.args
        )
        assert imported.metadata["transition_count"] == 4
        assert imported.metadata["classification"] == "faithful"
    finally:
        runner.stop()
        if imported is not None:
            imported.stop()


def response(runner, name, **payload):
    import time
    from gradlab.play_web import PlaybackCommand

    runner.submit(PlaybackCommand(name, "test", name, payload, None))
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if not runner.responses.empty():
            return runner.responses.get_nowait().payload
        time.sleep(0.005)
    raise AssertionError("Player did not answer")


def test_stale_confirmation_cannot_delete_new_bookmarks_and_repeated_resampling(tmp_path):
    from tests.test_play_trajectory import live_runner, command, wait_step

    runner = live_runner(tmp_path, length=30, restoration_factory=ScriptedRestoration)
    try:
        command(runner, "set_restoration_capture", enabled=True)
        command(runner, "step", count=5)
        wait_step(runner, 5)
        status = runner.snapshot()["trajectory"]
        identity = {key: status[key] for key in ("episode_id", "trajectory_revision")}
        command(runner, "bookmark_add", step=2, name="Keep", **identity)
        command(runner, "bookmark_add", step=4, name="Later", **identity)
        confirmation = runner.snapshot()["trajectory"]["bookmark_revision"]
        assert not response(runner, "discard_future", step=2, **identity)["ok"]
        command(runner, "bookmark_add", step=5, name="Newest", **identity)
        assert not response(
            runner, "discard_future", step=2, confirmed_bookmark_revision=confirmation, **identity
        )["ok"]
        assert runner.snapshot()["trajectory"]["last_step"] == 5
        command(
            runner,
            "discard_future",
            step=2,
            confirmed_bookmark_revision=confirmation + 1,
            **identity,
        )
        assert not response(runner, "discard_future", step=2, **identity)["ok"]
        assert [b["name"] for b in runner.snapshot()["trajectory"]["bookmarks"]] == ["Keep"]
        command(runner, "set_fps", fps=1)
        for _ in range(3):
            status = runner.snapshot()["trajectory"]
            command(
                runner,
                "resample_bookmark",
                bookmark_id=status["bookmarks"][0]["id"],
                **{key: status[key] for key in identity},
            )
            assert runner.snapshot()["run_state"] == "playing"
            command(runner, "pause")
        command(runner, "reset_episode")
        assert runner.snapshot()["trajectory"]["bookmarks"] == []
    finally:
        runner.stop()


def test_capture_gaps_and_failure_pause_without_losing_existing_points(tmp_path):
    from tests.test_play_trajectory import live_runner, command, wait_step

    class FailingCapture(ScriptedRestoration):
        def capture(self):
            if self.session.step_index == 6:
                raise OSError("test storage exhausted")
            return super().capture()

    runner = live_runner(tmp_path, length=20, restoration_factory=FailingCapture)
    try:
        command(runner, "set_restoration_capture", enabled=True)
        command(runner, "step", count=2)
        wait_step(runner, 2)
        command(runner, "set_restoration_capture", enabled=False)
        command(runner, "step", count=2)
        wait_step(runner, 4)
        command(runner, "set_restoration_capture", enabled=True)
        command(runner, "step", count=4)
        state = wait_step(runner, 6)
        assert state["run_state"] == "paused"
        assert state["trajectory"]["restoration"]["ranges"] == [[0, 2], [5, 5]]
        assert "storage exhausted" in state["trajectory"]["restoration"]["error"]
        command(runner, "set_restoration_capture", enabled=False)
        command(runner, "step", count=1)
        wait_step(runner, 7)
        identity = {
            key: runner.snapshot()["trajectory"][key]
            for key in ("episode_id", "trajectory_revision")
        }
        assert not response(runner, "discard_future", step=4, **identity)["ok"]
        command(runner, "discard_future", step=2, **identity)
        assert runner.snapshot()["trajectory"]["last_step"] == 2
    finally:
        runner.stop()


def test_restore_failure_preserves_timeline_bookmarks_and_exports(tmp_path):
    from tests.test_play_trajectory import live_runner, command, wait_step

    class FailingRestore(ScriptedRestoration):
        def restore(self, state):
            raise ValueError("candidate failed verification")

    runner = live_runner(tmp_path, length=20, restoration_factory=FailingRestore)
    try:
        command(runner, "set_restoration_capture", enabled=True)
        command(runner, "step", count=4)
        wait_step(runner, 4)
        before = runner.snapshot()["trajectory"]
        identity = {key: before[key] for key in ("episode_id", "trajectory_revision")}
        assert not response(runner, "discard_future", step=2, **identity)["ok"]
        after = runner.snapshot()["trajectory"]
        assert (after["episode_id"], after["trajectory_revision"], after["last_step"]) == (
            before["episode_id"],
            before["trajectory_revision"],
            4,
        )
        command(runner, "step", count=1)
        wait_step(runner, 5)
    finally:
        runner.stop()


@pytest.mark.parametrize("algorithm", ["ppo", "a2c", "action-program"])
def test_native_breakout_policy_continuation_is_exact_with_verification_sampling_state(algorithm):
    import torch
    from pathlib import Path
    from stable_baselines3 import PPO, A2C
    from gradlab.action_program import ActionProgramPolicy, ActionRun
    from gradlab.action_contract import action_contract_meanings
    from gradlab.env import make_eval_vec_env, resolve_env_config
    from gradlab.env_config import env_config_from_mapping
    from gradlab.recipe_documents import compose_train_document
    from gradlab.play_session import _PlaybackSession
    from gradlab.policy_runtime import PolicyRuntime
    from gradlab.play_restoration import LiveRestoration
    from gradlab.play_trajectory import pack_record

    root = Path("experiments/goals/Breakout-Atari2600-v0")
    config = resolve_env_config(
        env_config_from_mapping(
            compose_train_document(root / "_goal.yaml", root / "recipes/ppo.yaml")["train_config"]
        )
    )
    env = make_eval_vec_env(config, n_envs=1, seed=40000, capture_step_diagnostics=True)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        if algorithm == "ppo":
            model = PPO("MultiInputPolicy", env, n_steps=8, batch_size=8, device="cpu")
        elif algorithm == "a2c":
            model = A2C("MultiInputPolicy", env, n_steps=8, device="cpu")
        else:
            model = ActionProgramPolicy(
                action_names=action_contract_meanings(env.runtime.action_contract),
                action_runs=(ActionRun(1, 10), ActionRun(2, 10)),
                fallback_action=0,
            )
        mode = "program" if algorithm == "action-program" else "stochastic"
        session = _PlaybackSession(
            model=model,
            env=env,
            config=config,
            initial_seed=40000,
            policy_runtime=PolicyRuntime(model, algorithm_id=algorithm),
        )
        session.restart()
        session.trajectory_recording = True
        adapter = LiveRestoration(session)
        for _ in range(8):
            session.step(action_selection_mode=mode)
        state = adapter.capture()
        original_sampling = torch.get_rng_state()
        expected = [session.step(action_selection_mode=mode) for _ in range(5)]
        advancing_sampling = torch.get_rng_state()
        adapter.restore(state)
        assert torch.equal(advancing_sampling, torch.get_rng_state())
        # Only this verification rewinds sampling. User restoration never does.
        torch.set_rng_state(original_sampling)
        actual = [session.step(action_selection_mode=mode) for _ in range(5)]
        for before, after in zip(expected, actual):
            assert pack_record(before.model_obs) == pack_record(after.model_obs)
            assert pack_record(before.executed_action) == pack_record(after.executed_action)
            assert pack_record(before.after_frame) == pack_record(after.after_frame)
            assert (before.reward, before.total_reward, before.boundary) == (
                after.reward,
                after.total_reward,
                after.boundary,
            )
        # Corrupt task state fails after provider preparation; verified rollback
        # must preserve the working continuation, including its displayed decision.
        corrupted = deepcopy(state)
        corrupted["environment"]["task"]["schema_id"] = "invalid-task"
        current = adapter.capture()
        transition = session.last_transition
        import pytest

        with pytest.raises(ValueError):
            adapter.restore(corrupted)
        assert pack_record(adapter.capture()) == pack_record(current)
        assert session.last_transition is transition
    finally:
        env.close()
        torch.set_num_threads(threads)


def test_shared_viewers_receive_cut_even_when_target_frame_is_unchanged(tmp_path):
    import asyncio
    from aiohttp import ClientSession, WSMsgType
    from tests.test_play_trajectory import live_runner, command, wait_step
    from gradlab.play_web import PlaybackWebServer, FRAME_HEADER, FRAME_GAME
    from gradlab.play_application import PlaybackHost
    from gradlab.play_runtime import PlaybackLoader, ActivePlayback, PlaySourceSpec
    from gradlab.model_sources import ResolvedModelSource
    from gradlab.policy_bundle import load_policy_bundle

    async def scenario():
        runner = live_runner(tmp_path, length=20, restoration_factory=ScriptedRestoration)
        command(runner, "set_restoration_capture", enabled=True)
        command(runner, "step", count=4)
        wait_step(runner, 4)
        command(runner, "seek", step=2)
        host = PlaybackHost(PlaybackLoader(runner.args, argv=[], explicit_seed=False))
        host._active = ActivePlayback(
            runner,
            None,
            PlaySourceSpec("local", "fixture"),
            ResolvedModelSource(tmp_path / "model.zip", load_policy_bundle(tmp_path)),
        )
        host._phase = "active"
        server = PlaybackWebServer(host, runner.args)
        # This workflow must stay paused until the explicit command.
        server._auto_started_epoch = 0
        task = asyncio.create_task(server.run())
        while not server.origin:
            await asyncio.sleep(0.01)
        try:
            async with ClientSession() as client:
                sockets = []
                for workspace in ("controller", "observer"):
                    socket = await client.ws_connect(server.origin + "/ws", origin=server.origin)
                    await socket.send_json(
                        {
                            "type": "hello",
                            "protocol": 3,
                            "token": server.token,
                            "workspace_id": workspace,
                            "subscriptions": ["telemetry", "game"],
                        }
                    )
                    sockets.append(socket)
                    while True:
                        message = await socket.receive(timeout=5)
                        if (
                            message.type == WSMsgType.TEXT
                            and message.json().get("type") == "snapshot"
                        ):
                            break
                identity = {
                    key: runner.snapshot()["trajectory"][key]
                    for key in ("episode_id", "trajectory_revision")
                }
                # Non-owner requests must be rejected before reaching the runner.
                await sockets[1].send_json(
                    {
                        "type": "command",
                        "id": "unauthorized",
                        "name": "discard_future",
                        "payload": {**identity, "step": 2},
                    }
                )
                while True:
                    message = await sockets[1].receive(timeout=5)
                    if (
                        message.type == WSMsgType.TEXT
                        and message.json().get("id") == "unauthorized"
                    ):
                        assert message.json()["ok"] is False
                        break
                await sockets[0].send_json(
                    {
                        "type": "command",
                        "id": "cut",
                        "name": "discard_future",
                        "session_epoch": 0,
                        "control_epoch": server.control_epoch,
                        "payload": {**identity, "step": 2},
                    }
                )
                for socket in sockets:
                    snapshot = frame = False
                    for _ in range(30):
                        message = await socket.receive(timeout=5)
                        if message.type == WSMsgType.TEXT:
                            value = message.json()
                            if (
                                value.get("type") == "snapshot"
                                and value.get("trajectory", {}).get("trajectory_revision") == 1
                            ):
                                assert value["trajectory"]["last_step"] == 2
                                snapshot = True
                        elif message.type == WSMsgType.BINARY:
                            _, kind, _, _, _, sequence, _ = FRAME_HEADER.unpack_from(message.data)
                            frame |= snapshot and kind == FRAME_GAME and sequence == 2
                        if snapshot and frame:
                            break
                    assert snapshot and frame
                    await socket.close()
        finally:
            server.stop_event.set()
            await task
            host.stop()

    asyncio.run(scenario())


def test_queued_cut_loses_authority_when_control_changes_during_a_step(tmp_path):
    import threading
    import time
    from argparse import Namespace
    from tests.test_play_trajectory import ScriptedSession, command, wait_step
    from tests.test_policy_bundle import write_bundle
    from gradlab.policy_bundle import load_policy_bundle
    from gradlab.play_web import PlaybackCommand, WebPlaybackRunner

    entered = threading.Event()
    release = threading.Event()

    class BlockingSession(ScriptedSession):
        def step(self, **kwargs):
            if self.sequence == 2:
                entered.set()
                assert release.wait(timeout=5)
            return super().step(**kwargs)

    write_bundle(tmp_path)
    runner = WebPlaybackRunner(
        BlockingSession(20),
        Namespace(fps=60, episodes=0),
        config_text="",
        trajectory_bundle=load_policy_bundle(tmp_path),
        restoration_factory=ScriptedRestoration,
    )
    runner.start()
    try:
        runner.set_control_epoch(1)
        command(runner, "set_restoration_capture", enabled=True)
        command(runner, "step", count=2)
        wait_step(runner, 2)
        identity = {
            key: runner.snapshot()["trajectory"][key]
            for key in ("episode_id", "trajectory_revision")
        }
        command(runner, "step", count=1)
        assert entered.wait(timeout=5)
        runner.submit(
            PlaybackCommand(
                "queued-cut",
                "old-controller",
                "discard_future",
                {**identity, "step": 1},
                None,
                session_epoch=0,
                control_epoch=1,
            )
        )
        runner.set_control_epoch(2)
        release.set()
        deadline = time.monotonic() + 5
        while runner.responses.empty() and time.monotonic() < deadline:
            time.sleep(0.005)
        result = runner.responses.get_nowait().payload
        assert result["ok"] is False
        assert "ownership" in result["error"]
        assert runner.snapshot()["trajectory"]["last_step"] == 3
        assert runner.snapshot()["trajectory"]["trajectory_revision"] == 0
    finally:
        release.set()
        runner.stop()


def test_historical_contract_is_separate_from_next_action_selection(tmp_path):
    from tests.test_play_trajectory import live_runner, command, wait_step

    runner = live_runner(tmp_path, length=20, restoration_factory=ScriptedRestoration)
    try:
        command(runner, "step", count=2)
        wait_step(runner, 2)
        original = runner.snapshot()["session"]["critic_comparison"]
        command(runner, "set_action_selection_mode", mode="deterministic")
        command(runner, "seek", step=1)
        snapshot = runner.snapshot()
        assert snapshot["session"]["sampling_mode"] == "stochastic"
        assert snapshot["session"]["critic_comparison"] == original
        assert snapshot["policy"]["action_selection"]["requested_mode"] == "deterministic"
        assert snapshot["policy"]["action_selection"]["effective_mode"] == "stochastic"
    finally:
        runner.stop()
