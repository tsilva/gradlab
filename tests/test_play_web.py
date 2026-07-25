from __future__ import annotations

import argparse
import asyncio
import io
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
from aiohttp import ClientSession, WSServerHandshakeError, WSMsgType
from PIL import Image

from rlab.dataset_cli import build_parser as build_dataset_parser
from rlab.play import _PlaybackSession, _PlaybackTransition
from rlab.play_catalog import CatalogPage
from rlab.play_web import (
    FRAME_CODEC_PNG,
    FRAME_GAME,
    FRAME_HEADER,
    FRAME_MAGIC,
    FRAME_OBSERVATION,
    DatasetPlaybackRunner,
    FrameEncoder,
    HumanRecordingRunner,
    PlaybackCommand,
    PlaybackWebServer,
    WebPlaybackRunner,
    _session_environment_id,
    annotate_realized_returns,
    run_web_playback,
    source_browser_path,
    transition_payload,
)


class FakeHumanSession:
    fps = 60.0
    environment_id = "Game-v0"

    def action_from_labels(self, labels):
        return tuple(sorted(labels))


def human_args(**overrides):
    values = {
        "fps": 240.0,
        "episodes": 1,
        "port": 0,
        "no_open": True,
        "debug": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_playback_environment_title_uses_configured_env_id() -> None:
    session = argparse.Namespace(
        config={"game": "BreakoutTurbo-v0"},
        environment_id="recording-fallback",
    )

    assert _session_environment_id(session, human_args()) == "BreakoutTurbo-v0"


def test_web_playback_retains_step_zero_snapshot_and_frame() -> None:
    session = argparse.Namespace(
        config={"game": "Game-v0"},
        current_frame=np.zeros((4, 5, 3), dtype=np.uint8),
        frames=(),
        sequence=0,
        step_index=0,
        episode=1,
        active_seed=42,
        active_task=None,
        total_reward=0.0,
        max_x_pos=0,
        action_names=(),
        interactive=False,
        last_transition=None,
    )
    runner = WebPlaybackRunner(session, human_args(episodes=0), config_text="")

    runner._publish(None)

    snapshot, frames = runner.episode_start_payload()
    assert snapshot["sequence"] == 0
    assert snapshot["session"]["step"] == 0
    assert snapshot["session"]["value_discount"] is None
    assert snapshot["transition"] is None
    sequence, packet = frames[FRAME_GAME]
    assert sequence == 0
    assert FRAME_HEADER.unpack_from(packet) == (
        b"RLP2",
        FRAME_GAME,
        FRAME_CODEC_PNG,
        0,
        0,
        0,
    )


def test_completed_episode_history_gets_discounted_value_targets_and_signed_error() -> None:
    points = [
        {"episode": 1, "reward_shaped": 1.0, "value": 3.0},
        {"episode": 1, "reward_shaped": 2.0, "value": 1.0},
        {"episode": 2, "reward_shaped": 9.0, "value": 9.0},
    ]

    annotate_realized_returns(points, episode=1, discount=0.5)

    assert points[0]["realized_return"] == 2.0
    assert points[0]["value_error"] == 1.0
    assert points[1]["realized_return"] == 2.0
    assert points[1]["value_error"] == -1.0
    assert "realized_return" not in points[2]


def test_web_playback_exposes_loaded_models_value_discount() -> None:
    session = argparse.Namespace(
        model=argparse.Namespace(gamma=0.9),
        config={"game": "Game-v0"},
        current_frame=None,
        frames=(),
        sequence=0,
        step_index=0,
        episode=1,
        active_seed=42,
        active_task=None,
        total_reward=0.0,
        max_x_pos=0,
        action_names=(),
        interactive=False,
        last_transition=None,
    )
    runner = WebPlaybackRunner(session, human_args(episodes=0), config_text="")

    runner._publish(None)

    assert runner.snapshot()["session"]["value_discount"] == 0.9


def test_next_episode_dispatches_seed_sampling_and_driver_atomically() -> None:
    transition = argparse.Namespace(boundary=False, events=())
    session = argparse.Namespace(
        config={"game": "Game-v0"},
        episode=2,
        last_transition=None,
        restart=Mock(),
        step=Mock(return_value=transition),
    )
    runner = WebPlaybackRunner(session, human_args(episodes=0), config_text="")
    runner._publish = Mock()
    runner.awaiting_next_episode = True
    runner.boundaries = 1
    runner.driver = "human"

    runner._apply(
        PlaybackCommand(
            "next",
            "client",
            "next_episode",
            {
                "seed": "42",
                "sampling_mode": "deterministic",
                "driver": "policy",
            },
            None,
        )
    )

    session.restart.assert_called_once_with(42, reset_episode_index=False)
    assert runner.awaiting_next_episode is False
    assert runner.sampling_mode == "deterministic"
    assert runner.driver == "policy"
    assert runner.run_state == "playing"

    runner._step_once()

    session.step.assert_called_once_with(deterministic=True)


def test_play_does_not_mutate_active_episode_dispatch_settings() -> None:
    session = argparse.Namespace(
        config={"game": "Game-v0"},
        last_transition=None,
    )
    runner = WebPlaybackRunner(session, human_args(episodes=0), config_text="")
    runner._publish = Mock()
    runner.sampling_mode = "stochastic"
    runner.driver = "policy"

    runner._apply(
        PlaybackCommand(
            "play",
            "client",
            "play",
            {
                "sampling_mode": "deterministic",
                "driver": "human",
                "seed": "42",
            },
            None,
        )
    )

    assert runner.run_state == "playing"
    assert runner.sampling_mode == "stochastic"
    assert runner.driver == "policy"


def test_web_playback_requires_explicit_command_after_episode_boundary() -> None:
    transition = argparse.Namespace(boundary=True, events=(), episode=1)
    session = argparse.Namespace(
        config={"game": "Game-v0"},
        episode=2,
        last_transition=transition,
        step=Mock(return_value=transition),
    )
    runner = WebPlaybackRunner(session, human_args(episodes=0), config_text="")
    runner._publish = Mock()
    runner.run_state = "playing"

    runner._step_once()

    assert runner.run_state == "paused"
    assert runner.awaiting_next_episode is True
    assert runner._can_start_next_episode() is True
    assert runner.remaining_steps == 0

    runner._apply(PlaybackCommand("play", "client", "play", {}, None))
    blocked = runner.responses.get_nowait().payload
    assert blocked["ok"] is False
    assert blocked["error"] == "episode complete; choose Play next episode"
    assert runner.awaiting_next_episode is True

    runner._apply(PlaybackCommand("next", "client", "next_episode", {}, runner.revision))
    accepted = runner.responses.get_nowait().payload
    assert accepted["ok"] is True
    assert runner.awaiting_next_episode is False
    assert runner.run_state == "playing"


def test_web_playback_episode_limit_disables_next_episode() -> None:
    transition = argparse.Namespace(boundary=True, events=(), episode=1)
    session = argparse.Namespace(
        config={"game": "Game-v0"},
        episode=2,
        last_transition=transition,
        step=Mock(return_value=transition),
    )
    runner = WebPlaybackRunner(session, human_args(episodes=1), config_text="")
    runner._publish = Mock()
    runner.run_state = "playing"

    runner._step_once()

    assert runner.awaiting_next_episode is True
    assert runner._can_start_next_episode() is False
    assert runner._status_message == "episode limit reached (1)"


def test_human_dataset_recording_defaults_to_web_dashboard() -> None:
    args = build_dataset_parser().parse_args(["record", "local-session", "--env-id", "Game-v0"])

    assert args.agent == "human"
    assert not hasattr(args, "ui")
    assert not hasattr(args, "headless")
    assert args.port == 0
    assert args.no_open is False


def test_dataset_playback_uses_web_runner_and_preserves_recorded_telemetry() -> None:
    rows = [
        {
            "episode_id": "episode-1",
            "step_index": 0,
            "seed": 7,
            "actions": 1,
            "rewards": 2.5,
            "terminations": True,
            "truncations": False,
            "infos": '{"x_pos": 12, "lives": 2}',
            "collector_terminated": False,
            "env_id": "fixture-v0",
            "policy_mode": "random",
        },
        {
            "episode_id": "episode-1",
            "step_index": 1,
            "seed": 7,
            "actions": None,
            "rewards": None,
            "terminations": None,
            "truncations": None,
            "infos": None,
            "collector_terminated": False,
            "env_id": "fixture-v0",
            "policy_mode": "random",
        },
    ]
    frames = [
        np.zeros((4, 5, 3), dtype=np.uint8),
        np.full((4, 5, 3), 17, dtype=np.uint8),
    ]
    runner = DatasetPlaybackRunner(frames, rows, human_args(), fps=30.0)

    runner._publish()
    assert runner.snapshot()["mode"] == "dataset"
    assert runner.snapshot()["session"]["env_id"] == "fixture-v0"

    runner._step_once()

    snapshot = runner.snapshot()
    assert snapshot["run_state"] == "paused"
    assert snapshot["session"]["total_reward"] == 2.5
    assert snapshot["transition"]["action_source"] == "recorded"
    assert snapshot["transition"]["signals"] == {"lives": 2.0, "x_pos": 12.0}
    assert snapshot["transition"]["boundary"] is True


def test_run_web_playback_requests_paired_browser_windows() -> None:
    args = human_args()
    runner = object()
    server = AsyncMock()
    server.run.return_value = 0
    with (
        patch("rlab.play_web.WebPlaybackRunner", return_value=runner),
        patch("rlab.play_web.PlaybackWebServer", return_value=server) as server_type,
    ):
        assert run_web_playback(object(), args, config_text="config") == 0

    server_type.assert_called_once_with(runner, args, paired_windows=True)


def test_source_browser_paths_are_hierarchical_and_url_encoded() -> None:
    run_id = "rlab-" + "a" * 32
    checkpoint_id = "checkpoint-250000-" + "b" * 16

    assert source_browser_path(None) == "/"
    assert source_browser_path({"project": "Mario Bros"}) == "/projects/Mario%20Bros"
    assert source_browser_path(
        {"project": "Mario Bros", "goal_id": "Level 1-1"}
    ) == "/projects/Mario%20Bros/goals/Level%201-1"
    assert source_browser_path(
        {
            "project": "Mario Bros",
            "goal_id": "Level 1-1",
            "run_id": run_id,
        }
    ) == f"/projects/Mario%20Bros/goals/Level%201-1/runs/{run_id}"
    assert source_browser_path(
        {
            "project": "Mario Bros",
            "goal_id": "Level 1-1",
            "run_id": run_id,
            "checkpoint_id": checkpoint_id,
        }
    ) == (
        f"/projects/Mario%20Bros/goals/Level%201-1/runs/{run_id}"
        f"/checkpoints/{checkpoint_id}"
    )


def test_paired_playback_server_opens_play_and_stats_windows() -> None:
    async def scenario() -> None:
        runner = HumanRecordingRunner(FakeHumanSession(), human_args())
        server = PlaybackWebServer(
            runner,
            human_args(no_open=False),
            paired_windows=True,
        )
        with patch("rlab.play_web.webbrowser.open") as open_browser:
            task = asyncio.create_task(server.run())
            try:
                deadline = asyncio.get_running_loop().time() + 3.0
                while (
                    not server.origin or open_browser.call_count < 2
                ) and asyncio.get_running_loop().time() < deadline:
                    await asyncio.sleep(0.01)
                urls = server.dashboard_urls()
                assert urls == (
                    f"{server.origin}/?workspace=paired#token={server.token}",
                    f"{server.origin}/workspace/stats?workspace=paired#token={server.token}",
                )
                assert [call.args[0] for call in open_browser.call_args_list] == list(urls)
                assert all(
                    call.kwargs == {"new": 1, "autoraise": True}
                    for call in open_browser.call_args_list
                )
            finally:
                runner.stop()
                await asyncio.wait_for(task, timeout=3.0)

    asyncio.run(scenario())


def test_browser_button_chords_map_to_declared_discrete_actions() -> None:
    session = argparse.Namespace(action_names=("noop", "right", "right_a", "left"))

    assert _PlaybackSession.manual_action(session, []) == 0
    assert _PlaybackSession.manual_action(session, ["RIGHT", "a"]) == 2


def test_frame_encoder_emits_versioned_latest_only_png_packet() -> None:
    encoder = FrameEncoder()
    encoder.start()
    try:
        encoder.submit(FRAME_GAME, 7, np.full((3, 4, 3), 91, dtype=np.uint8))
        deadline = time.monotonic() + 2.0
        while FRAME_GAME not in encoder.latest() and time.monotonic() < deadline:
            time.sleep(0.005)
        sequence, packet = encoder.latest()[FRAME_GAME]
    finally:
        encoder.close()

    magic, kind, codec, flags, session_epoch, header_sequence = FRAME_HEADER.unpack(
        packet[: FRAME_HEADER.size]
    )
    image = Image.open(io.BytesIO(packet[FRAME_HEADER.size :]))
    assert (magic, kind, codec, flags, session_epoch, header_sequence, sequence) == (
        FRAME_MAGIC,
        FRAME_GAME,
        FRAME_CODEC_PNG,
        0,
        0,
        7,
        7,
    )
    assert image.size == (4, 3)


def test_frame_encoder_batches_game_and_observation_at_one_transition() -> None:
    encoder = FrameEncoder()
    encoder.start()
    try:
        encoder.submit_batch(
            11,
            {
                FRAME_GAME: np.full((3, 4, 3), 91, dtype=np.uint8),
                FRAME_OBSERVATION: np.full((5, 6, 3), 37, dtype=np.uint8),
            },
        )
        deadline = time.monotonic() + 2.0
        while (
            set(encoder.latest()) != {FRAME_GAME, FRAME_OBSERVATION} and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        latest = encoder.latest()
    finally:
        encoder.close()

    assert set(latest) == {FRAME_GAME, FRAME_OBSERVATION}
    for kind, (sequence, packet) in latest.items():
        (
            magic,
            header_kind,
            codec,
            flags,
            session_epoch,
            header_sequence,
        ) = FRAME_HEADER.unpack(packet[: FRAME_HEADER.size])
        assert (
            magic,
            header_kind,
            codec,
            flags,
            session_epoch,
            header_sequence,
            sequence,
        ) == (FRAME_MAGIC, kind, FRAME_CODEC_PNG, 0, 0, 11, 11)


def test_paired_auto_start_waits_for_both_workspace_windows() -> None:
    class Runner:
        session_epoch = 3
        has_active_runner = True

        def __init__(self) -> None:
            self.commands = []

        def submit(self, command) -> None:
            self.commands.append(command)

    async def scenario() -> None:
        runner = Runner()
        server = PlaybackWebServer(
            runner,
            human_args(debug=False),
            paired_windows=True,
        )
        server.control_holder = "workspace"
        server.clients["main-client"] = argparse.Namespace(
            client_id="main-client",
            workspace_id="workspace",
            window_id="main",
        )
        server._maybe_auto_start("main-client")
        assert runner.commands == []
        assert server._auto_start_task is not None

        server.clients["stats-client"] = argparse.Namespace(
            client_id="stats-client",
            workspace_id="workspace",
            window_id="stats",
        )
        server._maybe_auto_start("stats-client")
        await asyncio.sleep(0)

        assert [command.name for command in runner.commands] == ["play"]
        assert server._auto_started_epoch == 3

    asyncio.run(scenario())


def test_paired_auto_start_falls_back_when_stats_window_is_missing() -> None:
    class Runner:
        session_epoch = 4
        has_active_runner = True

        def __init__(self) -> None:
            self.commands = []

        def submit(self, command) -> None:
            self.commands.append(command)

    async def scenario() -> None:
        runner = Runner()
        server = PlaybackWebServer(
            runner,
            human_args(debug=False),
            paired_windows=True,
        )
        server.control_holder = "workspace"
        server.clients["main-client"] = argparse.Namespace(
            client_id="main-client",
            workspace_id="workspace",
            window_id="main",
        )
        with patch("rlab.play_web.PAIRED_START_GRACE_SECONDS", 0.01):
            server._maybe_auto_start("main-client")
            await asyncio.sleep(0.03)

        assert [command.name for command in runner.commands] == ["play"]
        assert server._auto_started_epoch == 4

    asyncio.run(scenario())


def test_non_paired_auto_start_is_immediate() -> None:
    class Runner:
        session_epoch = 5
        has_active_runner = True

        def __init__(self) -> None:
            self.commands = []

        def submit(self, command) -> None:
            self.commands.append(command)

    async def scenario() -> None:
        runner = Runner()
        server = PlaybackWebServer(
            runner,
            human_args(debug=False),
            paired_windows=False,
        )
        server.clients["main-client"] = argparse.Namespace(
            client_id="main-client",
            workspace_id="workspace",
            window_id="main",
        )

        server._maybe_auto_start("main-client")

        assert [command.name for command in runner.commands] == ["play"]
        assert server._auto_started_epoch == 5
        assert server._auto_start_task is None

    asyncio.run(scenario())


def test_debug_mode_never_auto_starts_paired_workspace() -> None:
    class Runner:
        session_epoch = 6
        has_active_runner = True

        def __init__(self) -> None:
            self.commands = []

        def submit(self, command) -> None:
            self.commands.append(command)

    async def scenario() -> None:
        runner = Runner()
        server = PlaybackWebServer(
            runner,
            human_args(debug=True),
            paired_windows=True,
        )
        server.control_holder = "workspace"
        for window_id in ("main", "stats"):
            client_id = f"{window_id}-client"
            server.clients[client_id] = argparse.Namespace(
                client_id=client_id,
                workspace_id="workspace",
                window_id=window_id,
            )

        server._maybe_auto_start("main-client")
        await asyncio.sleep(0)

        assert runner.commands == []
        assert server._auto_started_epoch == -1
        assert server._auto_start_task is None

    asyncio.run(scenario())


def test_transition_payload_keeps_before_decision_after_alignment() -> None:
    transition = _PlaybackTransition(
        sequence=3,
        episode=1,
        step=3,
        seed=40_000,
        start_id="Level1-1",
        model_obs=np.zeros((1, 4, 84, 84), dtype=np.uint8),
        decision=None,
        action_source="human",
        executed_action=2,
        diagnostics=None,
        info={"x_pos": 12, "credential_token": "do-not-stream"},
        before_frame=np.zeros((2, 2, 3), dtype=np.uint8),
        after_frame=np.ones((2, 2, 3), dtype=np.uint8),
        before_frames=(np.zeros((2, 2, 1), dtype=np.uint8),),
        after_frames=(np.ones((2, 2, 1), dtype=np.uint8),),
        attribution=None,
        pre_task="Level1-1",
        next_task="Level1-1",
        reward=1.5,
        total_reward=4.0,
        max_x_pos=12,
        terminated=False,
        truncated=False,
        completed=False,
        boundary=False,
    )

    payload = transition_payload(transition)

    assert payload["sequence"] == 3
    assert payload["before"]["task"] == "Level1-1"
    assert payload["decision"] is None
    assert payload["after"]["task"] == "Level1-1"
    assert payload["signals"]["x_pos"] == 12.0
    assert payload["info"]["credential_token"] == "<redacted>"


def test_human_recording_runner_requires_fresh_focus_and_streams_transition_stats() -> None:
    runner = HumanRecordingRunner(FakeHumanSession(), human_args())
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    runner.start()
    try:
        runner.submit(PlaybackCommand("play", "client", "play", {"driver": "human"}, None))
        runner.update_input(["right", "a"], focused=True)
        action, keep_recording = runner.action(frame)
        runner.observe_transition(
            reward=2.5,
            terminated=False,
            truncated=False,
            info={"x_pos": 11},
            next_frame=np.ones_like(frame),
        )
        snapshot = runner.snapshot()
    finally:
        runner.stop()

    assert keep_recording
    assert action == ("A", "RIGHT")
    assert snapshot["mode"] == "recording"
    assert snapshot["interactive"] is True
    assert snapshot["session"]["env_id"] == "Game-v0"
    assert snapshot["transition"]["reward"]["return"] == 2.5
    assert snapshot["transition"]["signals"] == {"x_pos": 11.0}
    assert runner.history_payload()["points"][0]["action_source"] == "human"


def test_loopback_server_requires_exact_origin_and_fragment_token() -> None:
    async def scenario() -> None:
        runner = HumanRecordingRunner(FakeHumanSession(), human_args())
        server = PlaybackWebServer(runner, human_args())
        task = asyncio.create_task(server.run())
        try:
            deadline = asyncio.get_running_loop().time() + 3.0
            while not server.origin and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.01)
            assert server.origin.startswith("http://127.0.0.1:")
            runner.encoder.submit(FRAME_GAME, 0, np.zeros((2, 3, 3), dtype=np.uint8))
            while FRAME_GAME not in runner.encoder.latest():
                await asyncio.sleep(0.005)
            async with ClientSession() as client:
                response = await client.get(server.origin)
                assert response.status == 200
                assert response.headers["Cache-Control"] == "no-store"
                assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
                icon_response = await client.get(f"{server.origin}/assets/tabler-icons.svg")
                assert icon_response.status == 200
                assert "image/svg+xml" in icon_response.headers["Content-Type"]
                assert 'id="ti-player-play"' in await icon_response.text()
                panel_response = await client.get(f"{server.origin}/assets/panels/catalog.js")
                assert panel_response.status == 200
                assert "javascript" in panel_response.headers["Content-Type"]
                assert "PANEL_CATALOG" in await panel_response.text()
                try:
                    await client.ws_connect(f"{server.origin}/ws", origin="http://example.test")
                except WSServerHandshakeError as exc:
                    assert exc.status == 403
                else:
                    raise AssertionError("cross-origin websocket unexpectedly connected")

                socket = await client.ws_connect(f"{server.origin}/ws", origin=server.origin)
                await socket.send_json(
                    {
                        "type": "hello",
                        "token": server.token,
                        "subscriptions": ["telemetry", "game"],
                    }
                )
                received_types = set()
                received_frame = False
                for _ in range(6):
                    message = await asyncio.wait_for(socket.receive(), timeout=2.0)
                    if message.type == WSMsgType.TEXT:
                        received_types.add(message.json()["type"])
                    elif message.type == WSMsgType.BINARY:
                        received_frame = True
                    if {"welcome", "snapshot"}.issubset(received_types) and received_frame:
                        break
                assert {"welcome", "snapshot"}.issubset(received_types)
                assert received_frame

                observer = await client.ws_connect(f"{server.origin}/ws", origin=server.origin)
                await observer.send_json(
                    {
                        "type": "hello",
                        "token": server.token,
                        "subscriptions": ["telemetry"],
                    }
                )
                observer_snapshot = None
                for _ in range(4):
                    message = await asyncio.wait_for(observer.receive(), timeout=2.0)
                    if message.type == WSMsgType.TEXT and message.json()["type"] == "snapshot":
                        observer_snapshot = message.json()
                        break
                assert observer_snapshot is not None
                assert observer_snapshot["control"]["has_control"] is False

                await observer.send_json(
                    {"type": "subscribe", "subscriptions": ["telemetry", "game"]}
                )
                observer_received_latest_frame = False
                for _ in range(4):
                    message = await asyncio.wait_for(observer.receive(), timeout=2.0)
                    if message.type == WSMsgType.BINARY:
                        observer_received_latest_frame = True
                        break
                assert observer_received_latest_frame

                await observer.send_json({"type": "acquire_control"})
                acquired_snapshot = None
                for _ in range(3):
                    message = await asyncio.wait_for(observer.receive(), timeout=2.0)
                    if message.type == WSMsgType.TEXT and message.json()["type"] == "snapshot":
                        acquired_snapshot = message.json()
                        break
                assert acquired_snapshot is not None
                assert acquired_snapshot["control"]["has_control"] is True
                assert acquired_snapshot["control_epoch"] > observer_snapshot["control_epoch"]

                sibling = await client.ws_connect(f"{server.origin}/ws", origin=server.origin)
                await sibling.send_json(
                    {
                        "type": "hello",
                        "token": server.token,
                        "subscriptions": ["telemetry"],
                        "workspace_id": acquired_snapshot["control"]["workspace_id"],
                        "window_id": "analysis-window",
                    }
                )
                sibling_snapshot = None
                for _ in range(4):
                    message = await asyncio.wait_for(sibling.receive(), timeout=2.0)
                    if message.type == WSMsgType.TEXT and message.json()["type"] == "snapshot":
                        sibling_snapshot = message.json()
                        break
                assert sibling_snapshot is not None
                assert sibling_snapshot["control"]["has_control"] is True
                assert sibling_snapshot["control"]["window_id"] == "analysis-window"
                await sibling.close()

                await observer.send_json(
                    {"type": "command", "id": "stop", "name": "stop", "payload": {}}
                )
                await observer.close()
                await socket.close()
            await asyncio.wait_for(task, timeout=3.0)
        finally:
            runner.stop()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_catalog_http_api_requires_the_fragment_session_token() -> None:
    class FakeCatalog:
        @staticmethod
        def default_entity(explicit=None):
            return explicit or "research"

        @staticmethod
        def projects(*, entity, query, cursor):
            assert (entity, query, cursor) == ("research", "mario", None)
            return CatalogPage(
                items=(
                    {
                        "entity": "research",
                        "name": "Mario",
                        "created_at": "",
                        "url": "",
                    },
                ),
                next_cursor=None,
            )

        @staticmethod
        def goals(*, entity, project, query, cursor):
            assert (entity, project, query, cursor) == (
                "research",
                "Mario",
                "",
                None,
            )
            return CatalogPage(
                items=(
                    {
                        "entity": entity,
                        "project": project,
                        "goal_id": "Level1-1",
                        "goal_slug": "Mario/Level1-1",
                        "run_count": 2,
                        "updated_at": "2026-07-25T00:00:00Z",
                    },
                ),
                next_cursor=None,
            )

        @staticmethod
        def checkpoints(*, run_id, query, entity, project):
            assert (
                run_id,
                query,
                entity,
                project,
            ) == ("rlab-" + "a" * 32, "", "research", "Mario")
            return (
                {
                    "run_id": run_id,
                    "checkpoint_id": "checkpoint-1-" + "b" * 16,
                    "evaluation": {
                        "status": "accepted",
                        "pass": True,
                        "episodes_planned": 100,
                        "episodes_completed": 100,
                        "failure_count": 0,
                        "criteria": [],
                    },
                },
            )

    async def scenario() -> None:
        runner = HumanRecordingRunner(FakeHumanSession(), human_args())
        server = PlaybackWebServer(runner, human_args(), catalog=FakeCatalog())
        task = asyncio.create_task(server.run())
        try:
            deadline = asyncio.get_running_loop().time() + 3.0
            while not server.origin and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.01)
            async with ClientSession() as client:
                denied = await client.get(f"{server.origin}/api/catalog/projects?q=mario")
                assert denied.status == 401
                accepted = await client.get(
                    f"{server.origin}/api/catalog/projects?q=mario",
                    headers={"Authorization": f"Bearer {server.token}"},
                )
                assert accepted.status == 200
                assert (await accepted.json())["items"][0]["name"] == "Mario"
                goals = await client.get(
                    f"{server.origin}/api/catalog/projects/research/Mario/goals",
                    headers={"Authorization": f"Bearer {server.token}"},
                )
                assert goals.status == 200
                assert (await goals.json())["items"][0]["goal_id"] == "Level1-1"
                checkpoints = await client.get(
                    (
                        f"{server.origin}/api/catalog/runs/rlab-{'a' * 32}/checkpoints"
                        "?entity=research&project=Mario"
                    ),
                    headers={"Authorization": f"Bearer {server.token}"},
                )
                assert checkpoints.status == 200
                assert (await checkpoints.json())["items"][0]["evaluation"]["pass"] is True
                for route in (
                    "/",
                    "/projects/Mario",
                    "/projects/Mario/goals/Level1-1",
                    f"/projects/Mario/goals/Level1-1/runs/rlab-{'a' * 32}",
                    (
                        f"/projects/Mario/goals/Level1-1/runs/rlab-{'a' * 32}"
                        f"/checkpoints/checkpoint-1-{'b' * 16}"
                    ),
                ):
                    page = await client.get(f"{server.origin}{route}")
                    assert page.status == 200
                    assert "<title>rlab player</title>" in await page.text()
        finally:
            runner.stop()
            await asyncio.wait_for(task, timeout=3.0)

    asyncio.run(scenario())


def test_web_dashboard_assets_are_packaged_beside_server() -> None:
    root = Path(__file__).parents[1] / "src" / "rlab" / "web_player"
    assert (root / "index.html").is_file()
    assert (root / "styles.css").is_file()
    assert (root / "tabler-icons.svg").is_file()
    assert (root / "tabler-chevron-down.svg").is_file()
    assert (root / "sources" / "browser.js").is_file()
    panel_root = root / "panels"
    panel_names = {
        "game",
        "controls",
        "policy",
        "reward",
        "actions",
        "observation",
        "signals",
        "events",
        "raw",
    }
    assert all((panel_root / f"{name}.js").is_file() for name in panel_names)
    markup = (root / "index.html").read_text(encoding="utf-8")
    styles = (root / "styles.css").read_text(encoding="utf-8")
    script = (root / "app.js").read_text(encoding="utf-8")
    source_browser = (root / "sources" / "browser.js").read_text(encoding="utf-8")
    icons = (root / "tabler-icons.svg").read_text(encoding="utf-8")
    catalog = (panel_root / "catalog.js").read_text(encoding="utf-8")
    runtime = (panel_root / "runtime.js").read_text(encoding="utf-8")
    shared = (panel_root / "shared.js").read_text(encoding="utf-8")
    game_markup = (panel_root / "game.js").read_text(encoding="utf-8")
    controls_markup = (panel_root / "controls.js").read_text(encoding="utf-8")
    policy_markup = (panel_root / "policy.js").read_text(encoding="utf-8")
    reward_markup = (panel_root / "reward.js").read_text(encoding="utf-8")
    actions_markup = (panel_root / "actions.js").read_text(encoding="utf-8")
    signals_markup = (panel_root / "signals.js").read_text(encoding="utf-8")
    events_markup = (panel_root / "events.js").read_text(encoding="utf-8")
    raw_markup = (panel_root / "raw.js").read_text(encoding="utf-8")
    assert '<main id="dashboard" class="dashboard"></main>' in markup
    assert '<main id="source-browser" class="source-browser" hidden></main>' in markup
    assert '<h1 id="page-title">Environment</h1>' in markup
    assert 'export function sourceRouteFromPath(' in source_browser
    assert 'export function sourceRoutePath(' in source_browser
    assert 'history.pushState(null, "", target);' in source_browser
    assert 'window.addEventListener("popstate", this.onPopState);' in source_browser
    assert 'if (this.route.level === "goals") return "Choose a goal";' in source_browser
    assert "renderGoals()" in source_browser
    assert '"Evaluation", "Size", "Created"' in source_browser
    assert 'evaluation.pass ? "Passed" : "Failed"' in source_browser
    assert "evaluationMetricLabel" in source_browser
    assert "criterion.value !== null" in source_browser
    assert 'checkpoint_id: item.checkpoint_id' in source_browser
    assert 'class="workspace-status"' not in markup
    assert markup.index('id="new-window"') < markup.index('id="connection-status"')
    assert markup.index('id="connection-status"') < markup.index('id="sampling-status"')
    assert "grid-template-columns: minmax(0, 1fr) auto;" in styles
    assert 'value="Default layout"' in markup
    assert "Mario debug" not in markup
    assert 'data-panel="' not in markup
    assert 'className: "control-panel transport"' in controls_markup
    assert "ENVIRONMENT" not in game_markup
    assert "Focus the game for human input" not in game_markup
    assert 'class="game-actions panel-actions"' in game_markup
    assert "game-overlay" not in game_markup
    assert ".game-overlay" not in styles
    assert 'id="timeline-label">STEP — · SEQ —' in markup
    assert 'id="timeline-step"' not in markup
    assert 'id="timeline-sequence"' not in markup
    assert '$("#timeline-label").textContent' in script
    assert 'state.inspectionSequence === null ? null : "INSPECTING"' in script
    assert "function episodeForSnapshot(snapshot)" in script
    assert "function clearRetainedEpisode()" in script
    assert "prepareRetainedEpisode(message);" in script
    assert "state.frameBlobs.forEach((frames) => frames.clear());" in script
    assert "state.historyLimit = Math.max(1, Number(message.history_limit) || 4096);" in script
    assert "function pruneRetainedTrace(" in script
    assert "function requiredFrameKinds(snapshot)" in script
    assert "function requiredFramesAvailable(snapshot)" in script
    assert "function exactFrameBlob(kind, sequence)" in script
    assert "nearestFrameBlob" not in script
    assert 'type: "inspection-cursor"' in script
    assert 'type: "inspection-frame-request"' in script
    assert 'type: "inspection-frame"' in script
    assert "function maybePauseForInspection()" in script
    assert 'state.inspectionPauseCommandId = command("pause");' in script
    assert "while (frames.size > 1024)" not in script
    assert "while (state.snapshots.size > 1024)" not in script
    assert "`EP ${" not in script
    assert "grid-template-columns: minmax(0, 1fr)" in styles
    assert ".timeline-labels { min-width: 0; overflow: hidden;" in styles
    assert "aspect-ratio: 256 / 240" in styles
    assert "grid-template-columns: repeat(12" in styles
    assert ".icon-only" in styles
    assert 'id="timeline-scrubber" type="range" min="0" max="0" step="1"' in markup
    assert 'title="Left/Right: move one step · Space: play or pause"' in markup
    assert 'event.key === "ArrowLeft" || event.key === "ArrowRight"' in script
    assert 'if (event.code !== "Space" || event.repeat) return;' in script
    assert "if (running) pauseCurrentPlayback();" in script
    assert "else playFromCurrentPosition();" in script
    assert 'id="return-live"' not in markup
    assert 'id="timeline-zoom-out"' not in markup
    assert 'id="timeline-zoom-in"' not in markup
    assert 'id="timeline-zoom-label"' not in markup
    assert "timelineWindow" not in script
    assert "const currentEpisode = episodeForSnapshot(state.liveSnapshot);" in script
    assert "currentEpisode === null || episodeForSnapshot(snapshot) === currentEpisode" in script
    assert "currentEpisode === null || Number(point.episode) === currentEpisode" in script
    assert "previousEpisode !== nextEpisode" in script
    assert 'scrubber.step = "1";' in script
    assert "if (index === state.timelineSequences.length - 1) returnToLive();" in script
    assert "data-return-chart" in reward_markup
    assert "data-value-chart" in reward_markup
    assert "Value estimate vs realized return-to-go" in reward_markup
    assert '["V − G", number(point?.value_error, 3)]' in reward_markup
    assert "positive overestimates, negative underestimates" in reward_markup
    assert "point.realized_return" in reward_markup
    assert "value_discount" in reward_markup
    assert "state.history[index] = { ...state.history[index], ...point };" in script
    assert controls_markup.count("data-playback-toggle data-requires-active-episode class=") == 1
    assert (
        'data-command="play" data-playback-toggle data-requires-active-episode class="primary icon-only"'
        in controls_markup
    )
    assert 'data-command="pause" class="icon-only"' not in controls_markup
    assert "services.playFromCurrentPosition()" in controls_markup
    assert "function playFromCurrentPosition()" in script
    assert 'command("play");' in script
    assert "services.pauseCurrentPlayback()" in controls_markup
    assert "services.canReplayInspection()" in controls_markup
    assert "state.replayingInspection" in controls_markup
    assert "Replay from the selected step" in controls_markup
    assert "function canReplayInspection()" in script
    assert 'state.liveSnapshot?.run_state !== "paused"' in script
    assert "function scheduleInspectionReplay()" in script
    assert "const reachedEpisodeEnd" in script
    assert "if (nextSequence === state.timelineSequences.at(-1)) returnToLive();" in script
    assert "stopInspectionReplay({ render: false });" in script
    assert (
        "if (view.inspection) snapshot = services.getState().liveSnapshot || snapshot;"
        in controls_markup
    )
    assert "playbackToggle.dataset.command = command" in controls_markup
    assert "if (playbackToggle.dataset.command === command) return;" in controls_markup
    assert (
        'playbackIcon.setAttribute("href", `/assets/tabler-icons.svg#ti-player-${command}`)'
        in controls_markup
    )
    assert "repeat(4, minmax(0, 1fr))" in styles
    assert "Continue to episode boundary" not in controls_markup
    assert 'data-command="continue-done"' not in controls_markup
    assert "End session" not in controls_markup
    assert 'data-command="stop"' not in controls_markup
    assert "ti-flag-3" not in icons
    assert "ti-power" not in icons
    assert (
        'data-command="step-ten" data-requires-active-episode class="icon-only" aria-label="Step 10 times"'
        in controls_markup
    )
    assert 'data-command="next-episode" data-next-episode' in controls_markup
    assert 'services.command("next_episode", {' in controls_markup
    assert "seed: seed.value" in controls_markup
    assert "sampling_mode: sampling.value" in controls_markup
    assert "driver: nextDriver" in controls_markup
    assert "Boolean(session.awaiting_next_episode)" in controls_markup
    assert "nextEpisode.disabled = !canPrepareNextEpisode" in controls_markup
    assert "seed.disabled = !canPrepareNextEpisode" in controls_markup
    assert "sampling.disabled = !canPrepareNextEpisode" in controls_markup
    assert "option.disabled = recording || !canPrepareNextEpisode" in controls_markup
    assert '<label for="playback-fps">Play FPS</label>' in controls_markup
    assert '<details class="playback-settings">' in controls_markup
    assert "<summary>Playback settings</summary>" in controls_markup
    assert '<details class="playback-settings" open>' not in controls_markup
    assert '<div class="playback-settings-body">' in controls_markup
    assert 'id="playback-fps" data-fps type="number" min="0"' in controls_markup
    assert 'data-command="set-fps"' in controls_markup
    assert '<select id="playback-sampling" data-sampling' in controls_markup
    assert '<option value="stochastic">Stochastic</option>' in controls_markup
    assert '<option value="deterministic">Deterministic</option>' in controls_markup
    assert '<h3 id="next-episode-heading"' in controls_markup
    assert 'class="next-episode-settings-body"' in controls_markup
    assert 'class="next-episode-seed"' in controls_markup
    assert "set_sampling_mode" not in controls_markup
    assert 'data-command="reset"' not in controls_markup
    assert 'services.command("set_driver"' not in controls_markup
    assert 'fps.addEventListener("keydown"' in controls_markup
    assert 'commands["set-fps"]();' in controls_markup
    assert "Session settings" not in controls_markup
    assert ".playback-fps" in styles
    assert ".playback-sampling" in styles
    assert ".playback-settings-body" in styles
    assert ".next-episode-settings-body" in styles
    assert 'id="ti-refresh"' in icons
    assert 'id="layouts-toggle" class="quiet icon-only"' in markup
    assert (
        'id="save-layout" class="primary button-with-icon" type="button" title="Save layout"'
        in markup
    )
    assert (
        'id="reset-layout" class="quiet button-with-icon" type="button" title="Reset default layout"'
        in markup
    )
    assert 'id="panel-hide" class="button-with-icon" type="button" title="Hide panel"' in markup
    assert "ti-device-desktop-share" not in icons
    assert "Control from this window" not in controls_markup
    assert "data-acquire" not in controls_markup
    assert "Inspect policy" not in controls_markup
    assert 'data-command="inspect"' not in controls_markup
    assert "inspect_policy" not in controls_markup
    assert "panel-inspection" not in script
    assert 'id="ti-search"' in icons
    assert 'class="driver-switch" role="group" aria-label="Driver selection"' in controls_markup
    assert 'data-driver-option="policy"' in controls_markup
    assert 'data-driver-option="human"' in controls_markup
    assert 'aria-pressed="true" aria-label="Use policy driver for next episode"' in controls_markup
    assert 'aria-pressed="false" aria-label="Use human driver for next episode"' in controls_markup
    assert "option.dataset.driverOption === nextDriver" in controls_markup
    assert '.driver-option[aria-pressed="true"]' in styles
    assert "separate scale" not in markup
    assert "shared scale" not in markup
    assert "Research workspace" not in markup
    assert "panel-kicker" not in markup
    assert 'id="workspace-sequence"' not in markup
    assert "#workspace-sequence" not in script
    assert "panel-shelf-title" not in script
    assert "scrollIntoView" not in script
    assert (
        "${snapshot.run_state.toUpperCase()} · ${snapshot.driver.toUpperCase()}" in controls_markup
    )
    assert "drawLines(returnChart" in reward_markup
    assert "cursorIndex" in reward_markup
    assert "history.slice(-1024)" not in reward_markup
    assert "history.slice(-1024)" not in signals_markup
    assert "history.slice(-1024)" not in actions_markup
    assert "highlightIndex" in actions_markup
    assert '"selected"' in events_markup
    assert "data-drag-handle" in game_markup
    assert 'aria-label="Move game panel"' in game_markup
    assert "/assets/tabler-icons.svg#ti-player-play" in controls_markup
    assert 'id="ti-grip-vertical"' in icons
    assert 'id="ti-player-play"' in icons
    assert 'id="panel-shelf" class="floating-menu panel-shelf" hidden' in markup
    assert "requestFullscreen" in game_markup
    assert 'data-transition class="json-view"' in raw_markup
    assert "function renderJson(" in shared
    assert "function niceTickStep(" in shared
    assert "function lineChartScale(" in shared
    assert "function formatAxisValue(" in shared
    assert "context.fillText(labels[index], plot.left - 6, y)" in shared
    assert "cursorIndex = null" in shared
    assert "highlightIndex = null" in shared
    assert "const scale = lineChartScale([0, max])" in shared
    assert 'class="signal-toolbar-label">Chart signal</span>' in signals_markup
    assert "grid-template-columns: max-content minmax(0, 1fr)" in styles
    assert ".signal-toolbar select" in styles
    assert 'background-image: url("/assets/tabler-chevron-down.svg")' in styles
    assert "background-position: right .85rem center" in styles
    assert "padding-right: 2.2rem" in styles
    assert "text-overflow: ellipsis" in styles
    for token_class in (
        "json-key",
        "json-string",
        "json-number",
        "json-boolean",
        "json-null",
    ):
        assert f".{token_class}" in styles
    assert "/panel/" in script
    assert "/workspace/" in script
    assert "workspace_id" in script
    assert "BroadcastChannel" in script
    assert 'type: "panel-drag-start"' in script
    assert 'type: "panel-drag-move"' in script
    assert 'type: "panel-drag-target"' in script
    assert 'type: "panel-drag-end"' in script
    assert "state.dragTarget?.window === state.windowId" in script
    assert "state.dragTarget.move >= message.move" in script
    assert "setPointerCapture" in script
    assert "clientPointFromScreen" in script
    assert "preview.style.width" in script
    assert "preview.style.height" in script
    assert ".panel-drag-overlay" in styles
    assert ".dashboard.drag-receiving" in styles
    assert "visibilitychange" in game_markup
    assert "PANEL_CATALOG" in catalog
    assert "const PAIRED_PANEL_LAYOUT" in catalog
    assert 'window: "stats"' in catalog
    assert "defaultPanelLayout({ paired = false } = {})" in catalog
    assert 'module: "./game.js"' in catalog
    assert "defaultPanelLayout" in catalog
    assert "import(definition.module)" in runtime
    assert "async ensureMounted" in runtime
    assert "this.unmount(name)" in runtime
    assert "new PanelRuntime" in script
    assert "load.title = `Load layout ${name}`" in script
    assert "remove.title = `Delete layout ${name}`" in script
    assert 'button.title = button.getAttribute("aria-label")' in script
    assert 'handle.title = handle.getAttribute("aria-label")' in script
    assert 'id="sampling-status" class="badge muted" hidden' in markup
    assert 'samplingMode === "deterministic" ? "Deterministic" : "Stochastic"' in script
    assert 'decision.sampled ? "Stochastic" : "Deterministic"' in policy_markup
    assert 'panelRuntime.invoke("controls", "render", snapshot)' in script
    assert 'name: "Default layout"' in script
    assert "state.liveSnapshot?.session?.env_id" in script
    assert "Mario debug" not in script
    assert 'new URLSearchParams(location.search).get("workspace") === "paired"' in script
    assert '"rlab.player.workspace.layout.v2"' in script
    assert "pairedWorkspace && closedWindow === STATS_WINDOW_ID" in script
    assert "body.stats-window #timeline" in styles
