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

from gradlab.dataset_cli import build_parser as build_dataset_parser
from gradlab.play import _PlaybackSession, _PlaybackTransition
from gradlab.play_catalog import CatalogPage
from gradlab.play_debug import PolicyDecision
from gradlab.play_web import (
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
    _decision_payload,
    _session_environment_id,
    annotate_realized_returns,
    history_point_payload,
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


def test_action_program_decision_payload_omits_probability_diagnostics() -> None:
    decision = PolicyDecision(
        raw_action=np.asarray([1], dtype=np.int64),
        executed_action=np.asarray([1], dtype=np.int64),
        action_selection_mode="program",
        program={"run_index": 0, "step_index": 0, "action": 1},
    )

    payload = _decision_payload(decision)

    assert payload is not None
    assert payload["selected_action"] == 1
    assert payload["probabilities"] is None
    assert payload["selected_probability"] is None
    assert payload["selected_rank"] is None
    assert payload["program"] == {"run_index": 0, "step_index": 0, "action": 1}


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


def test_incomparable_episode_suppresses_realized_value_diagnostics() -> None:
    points = [
        {"episode": 1, "reward_shaped": 1.0, "value": 3.0},
        {"episode": 1, "reward_shaped": 2.0, "value": 1.0},
    ]

    annotate_realized_returns(
        points,
        episode=1,
        discount=0.5,
        comparison_reasons=("active policy environment differs from training",),
    )

    assert all("realized_return" not in point for point in points)
    assert all("value_error" not in point for point in points)
    assert points[0]["value_comparison_reasons"] == [
        "active policy environment differs from training"
    ]


def test_realized_value_diagnostics_reject_mixed_driver_history() -> None:
    points = [
        {
            "episode": 1,
            "reward_shaped": 1.0,
            "value": 3.0,
            "action_source": "policy",
            "policy_sampled": True,
        },
        {
            "episode": 1,
            "reward_shaped": 2.0,
            "value": 1.0,
            "action_source": "human",
            "policy_sampled": None,
        },
    ]

    annotate_realized_returns(points, episode=1, discount=0.5)

    assert all("realized_return" not in point for point in points)
    assert points[0]["value_comparison_reasons"] == ["episode contains non-policy actions"]


def test_critic_comparison_requires_stochastic_policy_and_terminal_boundary() -> None:
    config = {"game": "Game-v0", "task": {"termination": {}}}
    session = argparse.Namespace(
        model=argparse.Namespace(gamma=0.9),
        config=config,
        termination_base_config=config,
    )
    runner = WebPlaybackRunner(
        session,
        human_args(),
        config_text="",
        contract_details={"comparison_reasons": []},
        value_contract={"discount": 0.9},
    )

    assert runner._critic_comparison_reasons() == []
    runner.sampling_mode = "deterministic"
    assert "deterministic trajectories" in runner._critic_comparison_reasons()[0]
    runner.sampling_mode = "stochastic"
    assert (
        "truncated episodes"
        in runner._critic_comparison_reasons(argparse.Namespace(truncated=True))[0]
    )


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


def test_next_episode_dispatches_sampling_and_driver_without_restarting() -> None:
    transition = argparse.Namespace(boundary=False, events=())
    session = argparse.Namespace(
        config={"game": "Game-v0"},
        episode=2,
        last_transition=None,
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
                "sampling_mode": "deterministic",
                "driver": "policy",
            },
            None,
        )
    )

    assert runner.awaiting_next_episode is False
    assert runner.sampling_mode == "deterministic"
    assert runner.driver == "policy"
    assert runner.run_state == "playing"

    runner._step_once()

    session.step.assert_called_once_with(deterministic=True)


def test_next_episode_applies_selected_termination_conditions() -> None:
    session = argparse.Namespace(
        config={"game": "Game-v0"},
        episode=2,
        last_transition=None,
        set_termination_conditions=Mock(),
    )
    runner = WebPlaybackRunner(session, human_args(episodes=0), config_text="")
    runner._publish = Mock()
    runner.awaiting_next_episode = True

    runner._apply(
        PlaybackCommand(
            "next",
            "client",
            "next_episode",
            {
                "enabled_termination_conditions": ["event:life_loss"],
            },
            None,
        )
    )

    session.set_termination_conditions.assert_called_once_with(["event:life_loss"])
    assert runner.awaiting_next_episode is False
    assert runner.run_state == "playing"


def test_reset_episode_uses_visible_seed_and_pauses_at_step_zero() -> None:
    session = argparse.Namespace(
        config={"game": "Game-v0"},
        active_seed=42,
        last_transition=argparse.Namespace(),
        reset_episode=Mock(),
    )
    session.reset_episode.side_effect = lambda seed: (
        setattr(session, "active_seed", seed),
        setattr(session, "last_transition", None),
    )
    runner = WebPlaybackRunner(session, human_args(episodes=0), config_text="")
    runner._publish = Mock()
    runner.run_state = "playing"

    runner._apply(
        PlaybackCommand(
            "reset",
            "client",
            "reset_episode",
            {"seed": "77"},
            None,
        )
    )

    session.reset_episode.assert_called_once_with(77)
    assert runner.awaiting_next_episode is False
    assert runner.run_state == "paused"
    assert runner._status_message == "episode reset · seed 77"


def test_reset_episode_applies_selected_termination_conditions() -> None:
    session = argparse.Namespace(
        config={"game": "Game-v0"},
        active_seed=42,
        last_transition=None,
        reset_episode=Mock(),
        set_termination_conditions=Mock(),
    )
    runner = WebPlaybackRunner(session, human_args(episodes=0), config_text="")
    runner._publish = Mock()

    runner._apply(
        PlaybackCommand(
            "reset",
            "client",
            "reset_episode",
            {
                "seed": "77",
                "enabled_termination_conditions": ["event:life_loss"],
            },
            None,
        )
    )

    session.reset_episode.assert_called_once_with(77)
    session.set_termination_conditions.assert_called_once_with(["event:life_loss"])
    assert runner.awaiting_next_episode is False
    assert runner.run_state == "paused"


def test_reset_episode_defaults_to_the_active_seed() -> None:
    session = argparse.Namespace(
        config={"game": "Game-v0"},
        active_seed=42,
        last_transition=None,
        reset_episode=Mock(),
    )
    runner = WebPlaybackRunner(session, human_args(episodes=0), config_text="")
    runner._publish = Mock()

    runner._apply(PlaybackCommand("reset", "client", "reset_episode", {"seed": ""}, None))

    session.reset_episode.assert_called_once_with(None)


def test_reset_episode_cannot_bypass_episode_limit() -> None:
    session = argparse.Namespace(
        config={"game": "Game-v0"},
        active_seed=42,
        last_transition=None,
        reset_episode=Mock(),
    )
    runner = WebPlaybackRunner(session, human_args(episodes=1), config_text="")
    runner._publish = Mock()
    runner.awaiting_next_episode = True
    runner.boundaries = 1

    runner._apply(PlaybackCommand("reset", "client", "reset_episode", {"seed": "42"}, None))

    session.reset_episode.assert_not_called()
    response = runner.responses.get_nowait().payload
    assert response["ok"] is False
    assert response["error"] == "episode limit reached (1)"


def test_session_reset_starts_a_new_attempt_only_after_steps() -> None:
    session = object.__new__(_PlaybackSession)
    session.active_seed = 42
    session.episode = 3
    session.step_index = 12
    session.restart = Mock()

    session.reset_episode()

    assert session.episode == 4
    session.restart.assert_called_once_with(42, reset_episode_index=False)


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


def test_termination_conditions_can_change_before_first_step() -> None:
    session = argparse.Namespace(
        config={"game": "Game-v0"},
        step_index=0,
        last_transition=None,
        set_termination_conditions=Mock(),
    )
    runner = WebPlaybackRunner(session, human_args(episodes=0), config_text="")
    runner._publish = Mock()

    runner._apply(
        PlaybackCommand(
            "termination",
            "client",
            "set_termination_conditions",
            {"enabled": ["event:life_loss"]},
            None,
        )
    )

    session.set_termination_conditions.assert_called_once_with(["event:life_loss"])
    assert runner.awaiting_next_episode is False
    assert runner.run_state == "paused"


def test_termination_conditions_cannot_change_mid_episode() -> None:
    session = argparse.Namespace(
        config={"game": "Game-v0"},
        step_index=12,
        last_transition=None,
        set_termination_conditions=Mock(),
    )
    runner = WebPlaybackRunner(session, human_args(episodes=0), config_text="")
    runner._publish = Mock()

    runner._apply(
        PlaybackCommand(
            "termination",
            "client",
            "set_termination_conditions",
            {"enabled": []},
            None,
        )
    )

    session.set_termination_conditions.assert_not_called()
    response = runner.responses.get_nowait().payload
    assert response["ok"] is False
    assert "before the first step or between episodes" in response["error"]


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
        patch("gradlab.play_web.WebPlaybackRunner", return_value=runner),
        patch("gradlab.play_web.PlaybackWebServer", return_value=server) as server_type,
    ):
        assert run_web_playback(object(), args, config_text="config") == 0

    server_type.assert_called_once_with(runner, args, paired_windows=True)


def test_source_browser_paths_are_hierarchical_and_url_encoded() -> None:
    run_id = "gradlab-" + "a" * 32
    checkpoint_id = "checkpoint-250000-" + "b" * 16
    variant_id = "goal-variant-" + "c" * 24

    assert source_browser_path(None) == "/"
    assert source_browser_path({"project": "Mario Bros"}) == "/environments/Mario%20Bros"
    assert (
        source_browser_path({"project": "Mario Bros", "goal_id": "Level 1-1"})
        == "/environments/Mario%20Bros/goals/Level%201-1"
    )
    assert source_browser_path(
        {
            "project": "Mario Bros",
            "goal_id": "Level 1-1",
            "goal_variant_id": variant_id,
            "run_id": run_id,
        }
    ) == (f"/environments/Mario%20Bros/goals/Level%201-1/variants/{variant_id}/runs/{run_id}")
    assert source_browser_path(
        {
            "project": "Mario Bros",
            "goal_id": "Level 1-1",
            "goal_variant_id": variant_id,
            "run_id": run_id,
            "checkpoint_id": checkpoint_id,
        }
    ) == (
        f"/environments/Mario%20Bros/goals/Level%201-1/variants/{variant_id}"
        f"/runs/{run_id}/checkpoints/{checkpoint_id}"
    )


def test_paired_playback_server_opens_play_and_stats_windows() -> None:
    async def scenario() -> None:
        runner = HumanRecordingRunner(FakeHumanSession(), human_args())
        server = PlaybackWebServer(
            runner,
            human_args(no_open=False),
            paired_windows=True,
        )
        with patch("gradlab.play_web.webbrowser.open") as open_browser:
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


def test_frame_encoder_retains_every_rapidly_submitted_observation() -> None:
    encoder = FrameEncoder()
    encoder.start()
    try:
        for sequence in range(96):
            encoder.submit_batch(
                sequence,
                {
                    FRAME_GAME: np.full((3, 4, 3), sequence, dtype=np.uint8),
                    FRAME_OBSERVATION: np.full((5, 6, 3), sequence, dtype=np.uint8),
                },
            )
        deadline = time.monotonic() + 5.0
        while not encoder.retained(95) and time.monotonic() < deadline:
            time.sleep(0.005)
        retained = [encoder.retained(sequence) for sequence in range(96)]
    finally:
        encoder.close()

    assert all(set(frames) == {FRAME_GAME, FRAME_OBSERVATION} for frames in retained)
    assert all(frames[FRAME_OBSERVATION][0] == sequence for sequence, frames in enumerate(retained))


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
        with patch("gradlab.play_web.PAIRED_START_GRACE_SECONDS", 0.01):
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
    assert payload["after"]["frame_role"] == "after_action_observation"
    assert payload["signals"]["x_pos"] == 12.0
    assert payload["info"]["credential_token"] == "<redacted>"


def test_history_point_keeps_policy_and_executed_actions_distinct() -> None:
    point = history_point_payload(
        {
            "sequence": 8,
            "episode": 2,
            "step": 4,
            "decision": {
                "selected_action": 3,
                "value": 1.25,
                "entropy": 0.4,
                "log_probability": -0.7,
            },
            "executed_action": 1,
            "action_source": "human_override",
            "reward": {
                "provider": 2.0,
                "shaped": 1.5,
                "return": 7.0,
                "components": {"progress": 1.5},
            },
            "outcome": "continuing",
            "events": ["coin"],
            "boundary": False,
            "signals": {"x_pos": 12.0},
        }
    )

    assert point["policy_action"] == 3
    assert point["executed_action"] == 1
    assert point["action_source"] == "human_override"
    assert point["log_probability"] == -0.7
    assert point["outcome"] == "continuing"
    assert "action" not in point


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
    assert snapshot["history_point"]["executed_action"] == ["A", "RIGHT"]
    assert snapshot["history_point"]["policy_action"] is None
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
                favicon_response = await client.get(f"{server.origin}/assets/favicon.svg")
                assert favicon_response.status == 200
                assert "image/svg+xml" in favicon_response.headers["Content-Type"]
                panel_response = await client.get(f"{server.origin}/assets/panels/catalog.js")
                assert panel_response.status == 200
                assert "javascript" in panel_response.headers["Content-Type"]
                assert "PANEL_TYPES" in await panel_response.text()
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

                runner.encoder.submit(
                    FRAME_GAME,
                    1,
                    np.full((2, 3, 3), 1, dtype=np.uint8),
                )
                while runner.encoder.latest()[FRAME_GAME][0] != 1:
                    await asyncio.sleep(0.005)
                await observer.send_json(
                    {
                        "type": "inspection_frames",
                        "session_epoch": 0,
                        "sequence": 0,
                        "kinds": [FRAME_GAME],
                    }
                )
                historical_sequence = None
                for _ in range(6):
                    message = await asyncio.wait_for(observer.receive(), timeout=2.0)
                    if message.type != WSMsgType.BINARY:
                        continue
                    historical_sequence = FRAME_HEADER.unpack_from(message.data)[5]
                    if historical_sequence == 0:
                        break
                assert historical_sequence == 0

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
                        "goal_count": 1,
                    },
                ),
                next_cursor=None,
            )

        environments = projects

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
                        "title": "Mario Level 1-1 completion",
                        "recipe_count": 1,
                        "goal_path": "experiments/goals/Mario/Level1-1/_goal.yaml",
                    },
                ),
                next_cursor=None,
            )

        @staticmethod
        def goal_variants(*, entity, project, goal_id, query, cursor):
            assert (entity, project, goal_id, query, cursor) == (
                "research",
                "Mario",
                "Level1-1",
                "",
                None,
            )
            return CatalogPage(
                items=(
                    {
                        "entity": entity,
                        "project": project,
                        "goal_id": goal_id,
                        "goal_slug": "Mario/Level1-1",
                        "variant_id": "goal-variant-" + "c" * 24,
                        "label": "Mario Level 1-1 completion",
                        "status": "current",
                        "source_relation": "canonical",
                        "goal_contract_sha256": "d" * 64,
                        "effective_goal_contract_sha256": "e" * 64,
                        "diff": [],
                    },
                ),
                next_cursor=None,
            )

        @staticmethod
        def recipes(*, entity, project, goal_id, query, cursor):
            assert (entity, project, goal_id, query, cursor) == (
                "research",
                "Mario",
                "Level1-1",
                "",
                None,
            )
            return CatalogPage(
                items=(
                    {
                        "recipe_id": "ppo",
                        "title": "ppo",
                        "availability": "static-preview",
                    },
                ),
                next_cursor=None,
            )

        @staticmethod
        def inspect_goal(*, entity, project, goal_id):
            assert (entity, project, goal_id) == ("research", "Mario", "Level1-1")
            return {
                "schema_version": 1,
                "source": {"kind": "repository-goal"},
                "documents": {"goal": {"availability": "exact"}},
            }

        @staticmethod
        def inspect_recipe(*, entity, project, goal_id, recipe_id):
            assert (entity, project, goal_id, recipe_id) == (
                "research",
                "Mario",
                "Level1-1",
                "ppo",
            )
            return {
                "schema_version": 1,
                "source": {"kind": "repository-recipe"},
                "documents": {"recipe": {"availability": "static-preview"}},
            }

        @staticmethod
        def inspect_goal_variant(*, entity, project, goal_id, variant_id):
            assert (entity, project, goal_id, variant_id) == (
                "research",
                "Mario",
                "Level1-1",
                "goal-variant-" + "c" * 24,
            )
            return {
                "schema_version": 1,
                "source": {"kind": "goal-variant"},
                "documents": {"goal": {"availability": "summary-only"}},
            }

        @staticmethod
        def inspect_run(*, entity, project, run_id):
            assert (entity, project, run_id) == (
                "research",
                "Mario",
                "gradlab-" + "a" * 32,
            )
            return {
                "schema_version": 1,
                "source": {"kind": "run"},
                "documents": {
                    "goal": {"availability": "exact"},
                    "recipe": {"availability": "exact"},
                },
            }

        @staticmethod
        def runs(
            *,
            entity,
            project,
            goal_id,
            goal_variant_id,
            query,
            cursor,
        ):
            assert (
                entity,
                project,
                goal_id,
                goal_variant_id,
                query,
                cursor,
            ) == (
                "research",
                "Mario",
                "Level1-1",
                "goal-variant-" + "c" * 24,
                "",
                None,
            )
            return CatalogPage(
                items=(
                    {
                        "run_id": "gradlab-" + "a" * 32,
                        "goal_variant_id": goal_variant_id,
                    },
                ),
                next_cursor=None,
            )

        @staticmethod
        def run_goal_variant(*, entity, project, run_id):
            assert (entity, project, run_id) == (
                "research",
                "Mario",
                "gradlab-" + "a" * 32,
            )
            return "Level1-1", "goal-variant-" + "c" * 24

        @staticmethod
        def checkpoints(*, run_id, query, entity, project, goal_variant_id):
            assert (
                run_id,
                query,
                entity,
                project,
                goal_variant_id,
            ) == ("gradlab-" + "a" * 32, "", "research", "Mario", "")
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

    class FakeEvaluationQueue:
        def __init__(self):
            self.requests = []

        def enqueue(self, *, run_id, checkpoint_ids):
            self.requests.append((run_id, list(checkpoint_ids)))
            return (
                {
                    "checkpoint_id": checkpoint_ids[0],
                    "state": "submitted",
                    "evaluation": None,
                    "message": None,
                },
            )

        @staticmethod
        def statuses(*, run_id, checkpoint_ids):
            del run_id, checkpoint_ids
            return {}

    async def scenario() -> None:
        runner = HumanRecordingRunner(FakeHumanSession(), human_args())
        evaluation_queue = FakeEvaluationQueue()
        server = PlaybackWebServer(
            runner,
            human_args(),
            catalog=FakeCatalog(),
            manual_evaluation_factory=lambda: evaluation_queue,
        )
        task = asyncio.create_task(server.run())
        try:
            deadline = asyncio.get_running_loop().time() + 3.0
            while not server.origin and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.01)
            async with ClientSession() as client:
                denied = await client.get(f"{server.origin}/api/catalog/projects?q=mario")
                assert denied.status == 401
                accepted = await client.get(
                    f"{server.origin}/api/catalog/environments?q=mario",
                    headers={"Authorization": f"Bearer {server.token}"},
                )
                assert accepted.status == 200
                assert (await accepted.json())["items"][0]["name"] == "Mario"
                goals = await client.get(
                    f"{server.origin}/api/catalog/environments/research/Mario/goals",
                    headers={"Authorization": f"Bearer {server.token}"},
                )
                assert goals.status == 200
                assert (await goals.json())["items"][0]["goal_id"] == "Level1-1"
                goal_inspection = await client.get(
                    (
                        f"{server.origin}/api/catalog/environments/research/Mario"
                        "/goals/Level1-1/inspection"
                    ),
                    headers={"Authorization": f"Bearer {server.token}"},
                )
                assert goal_inspection.status == 200
                assert (await goal_inspection.json())["documents"]["goal"][
                    "availability"
                ] == "exact"
                recipes = await client.get(
                    (
                        f"{server.origin}/api/catalog/environments/research/Mario"
                        "/goals/Level1-1/recipes"
                    ),
                    headers={"Authorization": f"Bearer {server.token}"},
                )
                assert recipes.status == 200
                assert (await recipes.json())["items"][0]["recipe_id"] == "ppo"
                recipe_inspection = await client.get(
                    (
                        f"{server.origin}/api/catalog/environments/research/Mario"
                        "/goals/Level1-1/recipes/ppo/inspection"
                    ),
                    headers={"Authorization": f"Bearer {server.token}"},
                )
                assert recipe_inspection.status == 200
                assert (await recipe_inspection.json())["documents"]["recipe"][
                    "availability"
                ] == "static-preview"
                variants = await client.get(
                    (
                        f"{server.origin}/api/catalog/environments/research/Mario"
                        "/goals/Level1-1/variants"
                    ),
                    headers={"Authorization": f"Bearer {server.token}"},
                )
                assert variants.status == 200
                variant_id = (await variants.json())["items"][0]["variant_id"]
                variant_inspection = await client.get(
                    (
                        f"{server.origin}/api/catalog/environments/research/Mario"
                        f"/goals/Level1-1/variants/{variant_id}/inspection"
                    ),
                    headers={"Authorization": f"Bearer {server.token}"},
                )
                assert variant_inspection.status == 200
                assert (await variant_inspection.json())["documents"]["goal"][
                    "availability"
                ] == "summary-only"
                runs = await client.get(
                    (
                        f"{server.origin}/api/catalog/environments/research/Mario"
                        f"/goals/Level1-1/variants/{variant_id}/runs"
                    ),
                    headers={"Authorization": f"Bearer {server.token}"},
                )
                assert runs.status == 200
                assert (await runs.json())["items"][0]["goal_variant_id"] == variant_id
                checkpoints = await client.get(
                    (
                        f"{server.origin}/api/catalog/runs/gradlab-{'a' * 32}/checkpoints"
                        "?entity=research&project=Mario"
                    ),
                    headers={"Authorization": f"Bearer {server.token}"},
                )
                assert checkpoints.status == 200
                assert (await checkpoints.json())["items"][0]["evaluation"]["pass"] is True
                run_inspection = await client.get(
                    (
                        f"{server.origin}/api/catalog/runs/gradlab-{'a' * 32}"
                        "/inspection?entity=research&project=Mario"
                    ),
                    headers={"Authorization": f"Bearer {server.token}"},
                )
                assert run_inspection.status == 200
                assert (await run_inspection.json())["documents"]["recipe"][
                    "availability"
                ] == "exact"
                active_inspection = await client.get(
                    f"{server.origin}/api/playback/inspection",
                    headers={"Authorization": f"Bearer {server.token}"},
                )
                assert active_inspection.status == 200
                assert (await active_inspection.json())["documents"]["recipe"][
                    "availability"
                ] == "unavailable"
                evaluation = await client.post(
                    (f"{server.origin}/api/catalog/runs/gradlab-{'a' * 32}/evaluations"),
                    headers={"Authorization": f"Bearer {server.token}"},
                    json={"checkpoint_ids": ["checkpoint-1-" + "b" * 16]},
                )
                assert evaluation.status == 202
                assert (await evaluation.json())["items"][0]["state"] == "submitted"
                assert evaluation_queue.requests == [
                    (
                        "gradlab-" + "a" * 32,
                        ["checkpoint-1-" + "b" * 16],
                    )
                ]
                legacy = await client.get(
                    (f"{server.origin}/projects/Mario/goals/Level1-1?workspace=paired"),
                    allow_redirects=False,
                )
                assert legacy.status == 308
                assert legacy.headers["Location"] == (
                    "/environments/Mario/goals/Level1-1?workspace=paired"
                )
                for route in (
                    "/",
                    "/environments/Mario",
                    "/environments/Mario/goals/Level1-1",
                    (f"/environments/Mario/goals/Level1-1/variants/goal-variant-{'c' * 24}"),
                    "/projects/Mario",
                    "/projects/Mario/goals/Level1-1",
                    f"/projects/Mario/goals/Level1-1/runs/gradlab-{'a' * 32}",
                    (
                        f"/projects/Mario/goals/Level1-1/runs/gradlab-{'a' * 32}"
                        f"/checkpoints/checkpoint-1-{'b' * 16}"
                    ),
                ):
                    page = await client.get(f"{server.origin}{route}")
                    assert page.status == 200
                    assert "<title>gradlab player</title>" in await page.text()
        finally:
            runner.stop()
            await asyncio.wait_for(task, timeout=3.0)

    asyncio.run(scenario())


def test_initial_project_catalog_is_embedded_in_selection_snapshots() -> None:
    class FakeCatalog:
        calls = 0

        @classmethod
        def initial_projects(cls, explicit_entity=None):
            cls.calls += 1
            assert explicit_entity is None
            return {
                "entity": "research",
                "items": [
                    {
                        "entity": "research",
                        "name": "Mario",
                        "goal_count": 1,
                    }
                ],
                "next_cursor": None,
            }

    runner = argparse.Namespace(session_change=0)
    server = PlaybackWebServer(runner, human_args(), catalog=FakeCatalog())
    asyncio.run(server._prepare_initial_catalog())
    client = argparse.Namespace(
        client_id="client",
        workspace_id="workspace",
        window_id="main",
    )

    snapshot = server._snapshot_for(
        client,
        {
            "type": "snapshot",
            "app": {
                "phase": "selecting",
                "route": {"level": "projects"},
            },
        },
    )

    assert FakeCatalog.calls == 1
    assert snapshot["app"]["catalog"]["entity"] == "research"
    assert snapshot["app"]["catalog"]["items"][0]["name"] == "Mario"

    nested_snapshot = server._snapshot_for(
        client,
        {
            "type": "snapshot",
            "app": {
                "phase": "selecting",
                "route": {
                    "level": "runs",
                    "project": "Mario",
                    "goal_id": "Level1-1",
                },
            },
        },
    )

    assert nested_snapshot["app"]["catalog"]["entity"] == "research"


def test_web_dashboard_assets_are_packaged_beside_server() -> None:
    root = Path(__file__).parents[1] / "src" / "gradlab" / "web_player"
    panel_root = root / "panels"
    expected_assets = (
        root / "index.html",
        root / "favicon.svg",
        root / "styles.css",
        root / "tabler-icons.svg",
        root / "tabler-chevron-down.svg",
        root / "vendor" / "gridstack" / "gridstack-all.js",
        root / "vendor" / "gridstack" / "gridstack.min.css",
        root / "sources" / "browser.js",
        root / "documents" / "viewer.js",
        panel_root / "catalog.js",
        panel_root / "layout-sizing.js",
        panel_root / "manager.js",
        panel_root / "runtime.js",
        panel_root / "shared.js",
        panel_root / "telemetry.js",
        panel_root / "telemetry-panel.js",
        panel_root / "workspace.js",
    )
    assert all(path.is_file() for path in expected_assets)
    specialized_panels = {"game", "controls", "observation", "events", "raw"}
    assert all((panel_root / f"{name}.js").is_file() for name in specialized_panels)
    removed_metric_panels = {"policy", "reward", "actions", "signals"}
    assert all(not (panel_root / f"{name}.js").exists() for name in removed_metric_panels)

    markup = (root / "index.html").read_text(encoding="utf-8")
    styles = (root / "styles.css").read_text(encoding="utf-8")
    script = (root / "app.js").read_text(encoding="utf-8")
    source_browser = (root / "sources" / "browser.js").read_text(encoding="utf-8")
    catalog = (panel_root / "catalog.js").read_text(encoding="utf-8")
    controls = (panel_root / "controls.js").read_text(encoding="utf-8")
    manager = (panel_root / "manager.js").read_text(encoding="utf-8")
    runtime = (panel_root / "runtime.js").read_text(encoding="utf-8")
    telemetry = (panel_root / "telemetry.js").read_text(encoding="utf-8")
    telemetry_panel = (panel_root / "telemetry-panel.js").read_text(encoding="utf-8")
    workspace = (panel_root / "workspace.js").read_text(encoding="utf-8")
    icons = (root / "tabler-icons.svg").read_text(encoding="utf-8")

    assert '<link rel="icon" href="/assets/favicon.svg" type="image/svg+xml">' in markup
    assert '<main id="dashboard" class="dashboard grid-stack"></main>' in markup
    assert 'href="/assets/vendor/gridstack/gridstack.min.css"' in markup
    assert 'src="/assets/vendor/gridstack/gridstack-all.js"' in markup
    assert '<main id="source-browser" class="source-browser" hidden></main>' in markup
    assert '<h1 id="page-title" hidden>Environment</h1>' in markup
    assert 'id="player-home"' in markup
    assert 'aria-label="Return to playback home"' in markup
    assert 'id="source-breadcrumbs"' in markup
    assert '$("#source-breadcrumbs")' in script
    assert '$("#player-home").addEventListener("click"' in script
    assert ".then((browser) => browser.goHome())" in script
    assert '$("#page-title").hidden = Boolean(state.sourceMode || activeCheckpointRoute);' in script
    assert '$("#page-title").textContent = "Select checkpoint"' not in script
    assert "approval_required" not in source_browser
    assert "approve_source" not in source_browser
    assert "Approve executable model" not in source_browser
    assert 'id="panel-add"' in markup
    assert 'id="panel-edit"' in markup
    assert 'id="panel-duplicate"' in markup
    assert 'id="panel-remove"' in markup
    assert 'id="panel-editor"' in markup
    for icon in ("ti-plus", "ti-edit", "ti-copy", "ti-trash"):
        assert f'id="{icon}"' in icons

    assert "PANEL_TYPES" in catalog
    assert "BUILTIN_PANEL_PRESETS" in catalog
    assert 'module: "./telemetry-panel.js"' in catalog
    assert '"policy/value"' in catalog
    assert '"reward/shaped"' in catalog
    assert '"action/executed"' in catalog
    assert '"namespace-explorer"' in catalog
    assert 'data-driver-option="human"' not in controls
    assert 'data-driver-option="policy"' not in controls
    assert 'driver: "policy"' in controls
    assert 'data-command="set-fps"' not in controls
    assert 'fps.addEventListener("input"' in controls
    assert 'services.command("set_fps", { fps: Number(fps.value) })' in controls
    assert "WORKSPACE_VERSION = 4" in workspace
    assert "createTelemetryInstance" in workspace
    assert "value.version !== WORKSPACE_VERSION" in workspace
    assert "compareWorkspaceRevisions" in workspace
    assert "class PanelManager" in manager
    assert "compatibleMetricKeys" in manager
    assert "class PanelRuntime" in runtime
    assert "this.definitionFor(workspace, id)" in runtime
    assert "import(definition.module)" in runtime
    assert "makeLineBlock" in telemetry_panel
    assert "makeHistogramBlock" in telemetry_panel
    assert "makeDistributionBlock" in telemetry_panel
    assert "makeNamespaceBlock" in telemetry_panel
    assert '"action/policy"' in telemetry
    assert '"action/executed"' in telemetry
    assert "dynamicDescriptorKey" in telemetry
    assert "function hideGoExploreValuePanel(snapshot)" in script
    assert 'search_algorithm_id !== "go-explore"' in script
    assert "hideGoExploreValuePanel(snapshot)" in script
    assert 'type: "inspection_frames"' in script
    assert "sequence < (state.receivedFrameSequence" not in script

    assert '"gradlab.player.workspace.v4.paired"' in script
    assert '"gradlab.player.workspace.v4.single"' in script
    assert "createTelemetryPanel" in script
    assert "updateTelemetryPanel" in script
    assert "snapshot.history_point" in script
    assert "historyFromTransition" not in script
    assert "window.GridStack.init" in script
    assert "column: 12" in script
    assert "viewportGridCellHeight" in script
    assert "cellHeight: DEFAULT_GRID_CELL_HEIGHT" in script
    assert "min-height: 0;" in styles
    assert "var(--grid-row)" not in styles
    assert 'gridStack.on("dragstop"' in script
    assert 'gridStack.on("resizestop"' in script
    assert "panel-drag-target" not in script
    assert ".telemetry-blocks" in styles
    assert ".panel-editor" in styles

    assert "export function sourceRouteFromPath(" in source_browser
    assert "export function sourceRoutePath(" in source_browser
    assert "export function formatDate(value, nowValue = Date.now())" in source_browser
    assert "export function recipeVariantPresentation(item)" in source_browser
    assert '{ label: "Recipe / variant" }' in source_browser
    assert "item.description || item.name || item.run_id" in source_browser
    assert 'history.pushState(null, "", target);' in source_browser
    assert 'window.addEventListener("popstate", this.onPopState);' in source_browser
    assert "goHome()" in source_browser
    assert "hydrateInitialProjects()" in source_browser
