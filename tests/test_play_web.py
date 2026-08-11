from __future__ import annotations

import argparse
import asyncio
import io
import threading
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
from aiohttp import ClientSession, WSServerHandshakeError, WSMsgType, web
from PIL import Image

from gradlab.dataset_cli import build_parser as build_dataset_parser
from gradlab.play_catalog import CatalogPage, CheckpointPage
from gradlab.play_debug import PolicyDecision
from gradlab.play_session import _PlaybackSession, _PlaybackTransition
from gradlab.play_web import (
    FRAME_ATTRIBUTION,
    FRAME_CNN_INSPECTION,
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
    _json_value,
    _session_environment_id,
    annotate_realized_returns,
    history_point_payload,
    reward_accounting_contract,
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


def test_json_projection_preserves_bounded_scalars_at_the_depth_limit() -> None:
    payload = {"a": {"b": {"c": {"d": {"label": "turn right", "items": [1]}}}}}

    projected = _json_value(payload)

    assert projected["a"]["b"]["c"]["d"]["label"] == "turn right"
    assert projected["a"]["b"]["c"]["d"]["items"] == "<list>"


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
    assert snapshot["protocol"] == 8
    assert snapshot["session"]["step"] == 0
    assert snapshot["session"]["default_seed"] == 42
    assert snapshot["session"]["value_discount"] is None
    assert snapshot["session"]["reward_accounting"] == {
        "status": "available",
        "reason": None,
        "reward_scale": 1.0,
        "clip_bounds": None,
    }
    assert snapshot["session"]["attribution"]["status"] == "off"
    assert snapshot["policy"]["attribution"]["supported_modes"] == []
    assert snapshot["session"]["cnn"]["status"] == "off"
    assert snapshot["policy"]["cnn"]["layers"] == []
    assert snapshot["transition"] is None
    sequence, packet = frames[FRAME_GAME]
    assert sequence == 0
    assert FRAME_HEADER.unpack_from(packet) == (
        b"RLP3",
        FRAME_GAME,
        FRAME_CODEC_PNG,
        0,
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


def test_invalid_attribution_command_does_not_pause_playback() -> None:
    class Session:
        config = {"game": "Game-v0"}
        model = argparse.Namespace(gamma=0.9)
        last_transition = None
        attribution_capability = {
            "target": "selected_action_log_probability",
            "supported_modes": [],
            "unavailable_reason": "not supported",
        }
        policy_capabilities = {
            "algorithm_id": "action-program",
            "action_selection": {
                "supported_modes": ["program"],
                "default_mode": "program",
            },
            "introspection": ["program"],
        }

        def configure_attribution(self, mode, interval):
            del mode, interval
            raise ValueError("not supported")

    runner = WebPlaybackRunner(Session(), human_args(), config_text="")
    runner.run_state = "playing"
    runner._publish = Mock()

    runner._apply(
        PlaybackCommand(
            "attribution",
            "client",
            "set_attribution",
            {"mode": "gradcam", "interval": 1},
            None,
        )
    )

    assert runner.run_state == "playing"
    assert runner._publish.call_count == 1
    response = runner.responses.get_nowait().payload
    assert response["ok"] is False
    assert response["error"] == "not supported"


def test_invalid_cnn_command_does_not_pause_playback() -> None:
    class Session:
        config = {"game": "Game-v0"}
        model = argparse.Namespace(gamma=0.9)
        last_transition = None
        cnn_capability = {
            "layers": [],
            "default_layer_id": None,
            "unavailable_reason": "no actor CNN",
        }

        def configure_cnn_inspection(self, **_configuration):
            raise ValueError("no actor CNN")

    runner = WebPlaybackRunner(Session(), human_args(), config_text="")
    runner.run_state = "playing"
    runner._publish = Mock()

    runner._apply(
        PlaybackCommand(
            "cnn",
            "client",
            "set_cnn_inspection",
            {"enabled": True, "layer_id": "cnn.0", "interval": 1, "top_k": 12},
            None,
        )
    )

    assert runner.run_state == "playing"
    assert runner._publish.call_count == 1
    response = runner.responses.get_nowait().payload
    assert response["ok"] is False
    assert response["error"] == "no actor CNN"


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
    assert runner.snapshot()["session"]["reward_accounting"]["status"] == "unavailable"
    assert runner.snapshot()["session"]["attribution"]["status"] == "off"
    assert "recorded datasets" in runner.snapshot()["session"]["attribution"]["unavailable_reason"]

    runner._step_once()

    snapshot = runner.snapshot()
    assert snapshot["run_state"] == "paused"
    assert snapshot["session"]["total_reward"] == 2.5
    assert snapshot["transition"]["action_source"] == "recorded"
    assert snapshot["transition"]["signals"] == {"lives": 2.0, "x_pos": 12.0}
    assert snapshot["transition"]["boundary"] is True
    assert snapshot["transition"]["reward"]["raw"] is None


def test_run_web_playback_requests_one_browser_window_by_default() -> None:
    args = human_args()
    runner = object()
    server = AsyncMock()
    server.run.return_value = 0
    with (
        patch("gradlab.play_web.WebPlaybackRunner", return_value=runner),
        patch("gradlab.play_web.PlaybackWebServer", return_value=server) as server_type,
    ):
        assert run_web_playback(object(), args, config_text="config") == 0

    server_type.assert_called_once_with(runner, args, paired_windows=False)


def test_source_browser_paths_are_hierarchical_and_url_encoded() -> None:
    run_id = "gradlab-" + "a" * 32
    checkpoint_id = "checkpoint-250000-" + "b" * 16
    variant_id = "goal-variant-" + "c" * 24

    assert source_browser_path(None) == "/"
    assert source_browser_path({"environment_id": "Mario Bros"}) == "/environments/Mario%20Bros"
    assert (
        source_browser_path({"environment_id": "Mario Bros", "goal_id": "Level 1-1"})
        == "/environments/Mario%20Bros/goals/Level%201-1"
    )
    assert source_browser_path(
        {
            "environment_id": "Mario Bros",
            "goal_id": "Level 1-1",
            "goal_variant_id": variant_id,
            "run_id": run_id,
        }
    ) == (f"/environments/Mario%20Bros/goals/Level%201-1/variants/{variant_id}/runs/{run_id}")
    assert source_browser_path(
        {
            "environment_id": "Mario Bros",
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


def test_browser_button_chords_map_to_declared_legal_tuple_actions() -> None:
    session = argparse.Namespace(
        action_contract={
            "policy": {
                "space": {"type": "multi_discrete"},
                "semantics": {
                    "status": "available",
                    "legal_entries": [
                        {
                            "value": [0, 0, 0, 0, 0, 0],
                            "semantic_id": "noop",
                            "controls": [{"player": 1, "inputs": []}],
                        },
                        {
                            "value": [0, 2, 0, 1, 0, 0],
                            "semantic_id": "attack_move_left",
                            "controls": [{"player": 1, "inputs": ["a", "left"]}],
                        },
                    ],
                },
            }
        }
    )

    assert _PlaybackSession.manual_action(session, []) == (0, 0, 0, 0, 0, 0)
    assert _PlaybackSession.manual_action(session, ["LEFT", "a"]) == (0, 2, 0, 1, 0, 0)


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

    magic, kind, codec, flags, session_epoch, header_sequence, generation = FRAME_HEADER.unpack(
        packet[: FRAME_HEADER.size]
    )
    image = Image.open(io.BytesIO(packet[FRAME_HEADER.size :]))
    assert (magic, kind, codec, flags, session_epoch, header_sequence, generation, sequence) == (
        FRAME_MAGIC,
        FRAME_GAME,
        FRAME_CODEC_PNG,
        0,
        0,
        7,
        0,
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
            generation,
        ) = FRAME_HEADER.unpack(packet[: FRAME_HEADER.size])
        assert (
            magic,
            header_kind,
            codec,
            flags,
            session_epoch,
            header_sequence,
            generation,
            sequence,
        ) == (FRAME_MAGIC, kind, FRAME_CODEC_PNG, 0, 0, 11, 0, 11)


def test_frame_encoder_merges_late_generated_frames_and_replaces_same_generation() -> None:
    encoder = FrameEncoder()
    encoder.start()
    try:
        encoder.submit_batch(
            4,
            {
                FRAME_GAME: np.full((3, 4, 3), 10, dtype=np.uint8),
                FRAME_OBSERVATION: np.full((3, 4, 3), 20, dtype=np.uint8),
            },
        )
        encoder.submit(
            FRAME_ATTRIBUTION,
            4,
            np.full((3, 4, 4), 30, dtype=np.uint8),
            generation=1,
        )
        encoder.submit(
            FRAME_ATTRIBUTION,
            4,
            np.full((3, 4, 4), 40, dtype=np.uint8),
            generation=2,
        )
        encoder.submit(
            FRAME_CNN_INSPECTION,
            4,
            np.full((6, 8, 4), 50, dtype=np.uint8),
            generation=3,
        )
        retained = encoder.retained(
            4,
            timeout=2.0,
            kinds={
                FRAME_GAME,
                FRAME_OBSERVATION,
                FRAME_ATTRIBUTION,
                FRAME_CNN_INSPECTION,
            },
        )
    finally:
        encoder.close()

    assert set(retained) == {
        FRAME_GAME,
        FRAME_OBSERVATION,
        FRAME_ATTRIBUTION,
        FRAME_CNN_INSPECTION,
    }
    attribution_packet = retained[FRAME_ATTRIBUTION][1]
    header = FRAME_HEADER.unpack(attribution_packet[: FRAME_HEADER.size])
    assert header[-2:] == (4, 2)
    image = Image.open(io.BytesIO(attribution_packet[FRAME_HEADER.size :]))
    assert image.mode == "RGBA"
    cnn_packet = retained[FRAME_CNN_INSPECTION][1]
    cnn_header = FRAME_HEADER.unpack(cnn_packet[: FRAME_HEADER.size])
    assert cnn_header[-2:] == (4, 3)
    cnn_image = Image.open(io.BytesIO(cnn_packet[FRAME_HEADER.size :]))
    assert cnn_image.mode == "RGBA"


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


def test_server_aggregates_processing_only_from_connected_windows() -> None:
    configured = []
    runner = argparse.Namespace(
        session_change=0,
        set_processing=lambda features: configured.append(frozenset(features)),
    )
    server = PlaybackWebServer(runner, human_args())
    server.clients = {
        "main": argparse.Namespace(processing=frozenset({"game"})),
        "stats": argparse.Namespace(processing=frozenset({"policy", "history"})),
    }

    asyncio.run(server._sync_player_processing())

    assert configured == [frozenset({"game", "policy", "history"})]


def test_observation_cnn_subscription_does_not_create_cnn_processing_demand() -> None:
    configured = []
    runner = argparse.Namespace(
        session_change=0,
        set_processing=lambda features: configured.append(frozenset(features)),
    )
    server = PlaybackWebServer(runner, human_args())
    server.clients = {
        "observation": argparse.Namespace(
            processing=frozenset({"observation", "attribution"}),
            subscriptions=frozenset({"observation", "attribution", "cnn-inspection"}),
        ),
        "cnn": argparse.Namespace(
            processing=frozenset({"cnn-inspection"}),
            subscriptions=frozenset({"cnn-inspection"}),
        ),
    }

    asyncio.run(server._sync_player_processing())
    del server.clients["cnn"]
    asyncio.run(server._sync_player_processing())

    assert configured == [
        frozenset({"observation", "attribution", "cnn-inspection"}),
        frozenset({"observation", "attribution"}),
    ]


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


def test_transition_payload_skips_disabled_panel_processors() -> None:
    transition = _PlaybackTransition(
        sequence=3,
        episode=1,
        step=3,
        seed=40_000,
        start_id="Level1-1",
        model_obs=np.zeros((1, 4, 84, 84), dtype=np.uint8),
        decision=PolicyDecision(
            raw_action=np.asarray(1),
            executed_action=np.asarray(1),
            action_selection_mode="stochastic",
        ),
        action_source="policy",
        executed_action=1,
        diagnostics=None,
        info={"x_pos": 12},
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

    with (
        patch("gradlab.play_web.model_input_lines", side_effect=AssertionError),
        patch("gradlab.play_web._numeric_signals", side_effect=AssertionError),
        patch("gradlab.play_web._reward_accounting_payload", side_effect=AssertionError),
        patch("gradlab.play_web._decision_payload", side_effect=AssertionError),
    ):
        payload = transition_payload(transition, processing=())

    assert payload["decision"] is None
    assert payload["executed_action"] is None
    assert payload["before"]["model_input"] == []
    assert payload["reward"]["shaped"] is None
    assert payload["signals"] == {}
    assert payload["info"] == {}


def test_reward_accounting_contract_uses_scale_then_clip_from_materialized_config() -> None:
    contract = reward_accounting_contract(
        argparse.Namespace(task={"reward": {"reward_scale": 0.4, "reward_clip": [-1, 1]}})
    )

    assert contract == {
        "status": "available",
        "reason": None,
        "reward_scale": 0.4,
        "clip_bounds": [-1.0, 1.0],
    }


def test_transition_reward_accounting_uses_only_declared_reward_components() -> None:
    base = _PlaybackTransition(
        sequence=3,
        episode=1,
        step=3,
        seed=40_000,
        start_id="Level1-1",
        model_obs=np.zeros((1, 4, 84, 84), dtype=np.uint8),
        decision=None,
        action_source="policy",
        executed_action=2,
        diagnostics=None,
        info={},
        before_frame=None,
        after_frame=None,
        before_frames=(),
        after_frames=(),
        attribution=None,
        pre_task="Level1-1",
        next_task="Level1-1",
        reward=1.0,
        total_reward=4.0,
        max_x_pos=12,
        terminated=False,
        truncated=False,
        completed=False,
        boundary=False,
    )
    diagnostics = argparse.Namespace(
        provider_reward=0.0,
        reward=1.0,
        outcome=argparse.Namespace(name="continuing"),
        task_metrics={
            "raw_reward": 4.0,
            "progress_component": 99.0,
            "native_reward_component": 1.0,
            "progress_reward_component": 4.0,
            "death_penalty_component": -1.0,
            "kill_reward_component": 2.0,
            "weapon_hold_reward_component": 0.002,
            "unknown_reward_component": 500.0,
        },
        event_transitions={},
        provider_terminated=False,
        provider_truncated=False,
        task_terminated=False,
        task_truncated=False,
        events=(),
    )

    payload = transition_payload(
        replace(base, diagnostics=diagnostics),
        reward_accounting={
            "status": "available",
            "reason": None,
            "reward_scale": 0.5,
            "clip_bounds": [-1.0, 1.0],
        },
    )

    assert payload["reward"]["raw"] == 4.0
    assert payload["reward"]["components"] == {
        "native_reward": 1.0,
        "progress_reward": 4.0,
        "death_penalty": -1.0,
        "kill_reward": 2.0,
        "weapon_hold_reward": 0.002,
    }
    assert payload["reward"]["accounting_error"] is None


def test_active_reward_transform_fails_closed_without_raw_reward() -> None:
    transition = _PlaybackTransition(
        sequence=1,
        episode=1,
        step=1,
        seed=1,
        start_id=None,
        model_obs=np.zeros((1, 1), dtype=np.uint8),
        decision=None,
        action_source="policy",
        executed_action=0,
        diagnostics=None,
        info={},
        before_frame=None,
        after_frame=None,
        before_frames=(),
        after_frames=(),
        attribution=None,
        pre_task=None,
        next_task=None,
        reward=1.0,
        total_reward=1.0,
        max_x_pos=0,
        terminated=False,
        truncated=False,
        completed=False,
        boundary=False,
    )

    payload = transition_payload(
        transition,
        reward_accounting={
            "status": "available",
            "reason": None,
            "reward_scale": 0.5,
            "clip_bounds": None,
        },
    )

    assert payload["reward"]["raw"] is None
    assert "raw_reward is missing" in payload["reward"]["accounting_error"]


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
    assert snapshot["session"]["reward_accounting"]["status"] == "unavailable"
    assert snapshot["session"]["attribution"]["status"] == "off"
    assert "human recording" in snapshot["session"]["attribution"]["unavailable_reason"]
    assert snapshot["transition"]["reward"]["return"] == 2.5
    assert snapshot["transition"]["reward"]["raw"] is None
    assert snapshot["transition"]["signals"] == {"x_pos": 11.0}
    assert snapshot["history_point"]["executed_action"] == ["A", "RIGHT"]
    assert snapshot["history_point"]["policy_action"] is None
    assert runner.history_payload()["points"][0]["action_source"] == "human"


def test_youtube_oauth_callback_returns_to_authenticated_player(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = PlaybackWebServer(argparse.Namespace(session_change=0), human_args())
    server.token = "session-token"
    server.publication_authority_client_id = "client-1"
    server.control_epoch = 3
    transaction = Mock()
    server._oauth_transactions["oauth-state"] = transaction
    paths = argparse.Namespace(
        root=Path("/private/config"),
        client=Path("/private/config/youtube_client_secret.json"),
        token=Path("/private/config/youtube_token.json"),
        lock=Path("/private/config/youtube_token.lock"),
    )
    saved: dict[str, object] = {}

    monkeypatch.setattr("gradlab.play_web.youtube_credential_paths", lambda: paths)
    monkeypatch.setattr("gradlab.play_web.credential_lock", lambda _path: nullcontext())
    monkeypatch.setattr(
        "gradlab.play_web.load_private_json",
        lambda _path, *, root: {"installed": {"client_id": "id"}},
    )
    monkeypatch.setattr(
        "gradlab.play_web.exchange_oauth_code",
        lambda _config, _transaction, *, code: {"access_token": code},
    )
    monkeypatch.setattr(
        "gradlab.play_web.save_private_json",
        lambda path, value, *, root: saved.update(path=path, value=value, root=root),
    )

    async def scenario() -> None:
        request = argparse.Namespace(query={"state": "oauth-state", "code": "code"})
        with pytest.raises(web.HTTPSeeOther) as redirect:
            await server.publication_oauth_callback(request)
        assert redirect.value.headers["Location"] == (
            "/publication/oauth/complete#token=session-token"
        )

    asyncio.run(scenario())

    transaction.validate_authority.assert_called_once_with("client-1", 3)
    assert saved == {
        "path": paths.token,
        "value": {"access_token": "code"},
        "root": paths.root,
    }


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
                assert response.headers["Cross-Origin-Opener-Policy"] == (
                    "same-origin-allow-popups"
                )
                assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
                oauth_complete = await client.get(f"{server.origin}/publication/oauth/complete")
                assert oauth_complete.status == 200
                assert oauth_complete.headers["Cross-Origin-Opener-Policy"] == (
                    "same-origin-allow-popups"
                )
                assert "Authorization complete" in await oauth_complete.text()
                icon_response = await client.get(f"{server.origin}/assets/tabler-icons.svg")
                assert icon_response.status == 200
                assert "image/svg+xml" in icon_response.headers["Content-Type"]
                assert 'id="ti-player-play"' in await icon_response.text()
                favicon_response = await client.get(f"{server.origin}/assets/favicon.svg")
                assert favicon_response.status == 200
                assert "image/svg+xml" in favicon_response.headers["Content-Type"]
                font_response = await client.get(
                    f"{server.origin}/assets/fonts/InterVariable.woff2"
                )
                assert font_response.status == 200
                assert font_response.headers["Content-Type"] == "font/woff2"
                assert "default-src 'self'" in font_response.headers["Content-Security-Policy"]
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


def test_publication_authority_is_exact_tab_private_and_rotates(tmp_path: Path) -> None:
    class FakePublicationService:
        @staticmethod
        def current():
            return {"available": True, "capture": {"capture_id": "capture-test"}, "job": None}

    async def receive_type(socket, expected):
        for _ in range(12):
            message = await asyncio.wait_for(socket.receive(), timeout=2.0)
            if message.type == WSMsgType.TEXT and message.json().get("type") == expected:
                return message.json()
        raise AssertionError(f"did not receive {expected}")

    async def scenario() -> None:
        runner = HumanRecordingRunner(FakeHumanSession(), human_args())
        server = PlaybackWebServer(
            runner,
            human_args(),
            repo_root=tmp_path,
            publication_factory=lambda **_kwargs: FakePublicationService(),
        )
        task = asyncio.create_task(server.run())
        try:
            deadline = asyncio.get_running_loop().time() + 3.0
            while not server.origin and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.01)
            async with ClientSession() as client:
                first = await client.ws_connect(f"{server.origin}/ws", origin=server.origin)
                await first.send_json(
                    {
                        "type": "hello",
                        "token": server.token,
                        "workspace_id": "shared",
                        "window_id": "main",
                    }
                )
                welcome = await receive_type(first, "welcome")
                first_authority = await receive_type(first, "publication_authority")
                assert first_authority["has_authority"] is True
                assert first_authority["capability"]

                second = await client.ws_connect(f"{server.origin}/ws", origin=server.origin)
                await second.send_json(
                    {
                        "type": "hello",
                        "token": server.token,
                        "workspace_id": "shared",
                        "window_id": "stats",
                    }
                )
                second_welcome = await receive_type(second, "welcome")
                second_snapshot = await receive_type(second, "snapshot")
                assert second_snapshot["control"]["has_control"] is True
                assert second_snapshot["publication"]["has_authority"] is False

                denied = await client.get(
                    f"{server.origin}/api/publication/current",
                    headers={
                        "Origin": server.origin,
                        "Authorization": f"Bearer {server.token}",
                        "X-Gradlab-Client": second_welcome["client_id"],
                        "X-Gradlab-Control-Epoch": str(second_snapshot["control_epoch"]),
                        "X-Gradlab-Publication-Capability": first_authority["capability"],
                    },
                )
                assert denied.status == 403

                await second.send_json({"type": "acquire_control"})
                second_authority = await receive_type(second, "publication_authority")
                assert second_authority["has_authority"] is True
                assert second_authority["capability"] != first_authority["capability"]
                allowed = await client.get(
                    f"{server.origin}/api/publication/current",
                    headers={
                        "Origin": server.origin,
                        "Authorization": f"Bearer {server.token}",
                        "X-Gradlab-Client": second_welcome["client_id"],
                        "X-Gradlab-Control-Epoch": str(second_authority["control_epoch"]),
                        "X-Gradlab-Publication-Capability": second_authority["capability"],
                    },
                )
                assert allowed.status == 200
                allowed_data = await allowed.json()
                assert "capture" in allowed_data, allowed_data
                assert allowed_data["capture"]["capture_id"] == "capture-test", allowed_data

                await second.close()
                await asyncio.sleep(0.05)
                stale = await client.get(
                    f"{server.origin}/api/publication/current",
                    headers={
                        "Origin": server.origin,
                        "Authorization": f"Bearer {server.token}",
                        "X-Gradlab-Client": welcome["client_id"],
                        "X-Gradlab-Control-Epoch": str(first_authority["control_epoch"]),
                        "X-Gradlab-Publication-Capability": first_authority["capability"],
                    },
                )
                assert stale.status == 403
                await first.send_json(
                    {"type": "command", "id": "stop", "name": "stop", "payload": {}}
                )
                await first.close()
            await asyncio.wait_for(task, timeout=3.0)
        finally:
            runner.stop()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_catalog_http_api_requires_the_fragment_session_token() -> None:
    class FakeCatalog:
        checkpoint_modes: list[bool] = []

        @staticmethod
        def environments(*, query, cursor):
            assert (query, cursor) == ("mario", None)
            return CatalogPage(
                items=(
                    {
                        "name": "Mario",
                        "goal_count": 1,
                    },
                ),
                next_cursor=None,
            )

        @staticmethod
        def goals(*, environment_id, query, cursor):
            assert (environment_id, query, cursor) == (
                "Mario",
                "",
                None,
            )
            return CatalogPage(
                items=(
                    {
                        "environment_id": environment_id,
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
        def goal_variants(*, environment_id, goal_id, query, cursor):
            assert (environment_id, goal_id, query, cursor) == (
                "Mario",
                "Level1-1",
                "",
                None,
            )
            return CatalogPage(
                items=(
                    {
                        "environment_id": environment_id,
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
        def goal_activity(*, environment_id, goal_id, query, refresh):
            assert (environment_id, goal_id, query, refresh) == (
                "Mario",
                "Level1-1",
                "",
                False,
            )
            return {
                "schema_version": 1,
                "items": [
                    {
                        "variant_id": "goal-variant-" + "c" * 24,
                        "recent_runs": [],
                    }
                ],
                "next_cursor": None,
                "revision": "f" * 64,
                "generation_sha256": "e" * 64,
                "freshness": "fresh",
                "has_active_runs": False,
            }

        @staticmethod
        def recipes(*, environment_id, goal_id, query, cursor):
            assert (environment_id, goal_id, query, cursor) == (
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
        def inspect_goal(*, environment_id, goal_id):
            assert (environment_id, goal_id) == ("Mario", "Level1-1")
            return {
                "schema_version": 1,
                "source": {"kind": "repository-goal"},
                "documents": {"goal": {"availability": "exact"}},
            }

        @staticmethod
        def inspect_recipe(*, environment_id, goal_id, recipe_id):
            assert (environment_id, goal_id, recipe_id) == (
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
        def inspect_goal_variant(*, environment_id, goal_id, variant_id):
            assert (environment_id, goal_id, variant_id) == (
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
        def inspect_run(*, run_id):
            assert run_id == "gradlab-" + "a" * 32
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
            environment_id,
            goal_id,
            goal_variant_id,
            query,
            cursor,
        ):
            assert (
                environment_id,
                goal_id,
                goal_variant_id,
                query,
                cursor,
            ) == (
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

        @classmethod
        def checkpoints(cls, *, run_id, query, goal_variant_id, include_wandb):
            assert (
                run_id,
                query,
                goal_variant_id,
            ) == ("gradlab-" + "a" * 32, "", "")
            cls.checkpoint_modes.append(include_wandb)
            return CheckpointPage(
                items=(
                    {
                        "run_id": run_id,
                        "checkpoint_id": "checkpoint-1-" + "b" * 16,
                        "sha256": "b" * 64,
                        "metrics": {
                            "train/progress/kills/origin/target/rolling/mean": 8.5,
                            "train/episode/return/shaped/origin/target/rolling/mean": 120.0,
                        },
                        "evaluation": {
                            "status": "accepted",
                            "pass": True,
                            "episodes_planned": 100,
                            "episodes_completed": 100,
                            "failure_count": 0,
                            "criteria": [],
                            "metrics": {
                                "eval/full/progress/kills/mean": 9.0,
                                "eval/full/progress/kills/max": 15.0,
                            },
                        },
                    },
                ),
                metric_columns=(
                    {
                        "metric": "eval/full/progress/kills/mean",
                        "direction": "max",
                        "label": "Eval mean kills",
                        "evidence": "evaluation",
                        "roles": ["objective", "acceptance"],
                        "rank_index": 0,
                        "acceptance": [
                            {
                                "metric": "eval/full/progress/kills/mean",
                                "operator": ">=",
                                "threshold": 10.0,
                            }
                        ],
                    },
                    {
                        "metric": "train/progress/kills/origin/target/rolling/mean",
                        "direction": "max",
                        "label": "Recent target kills mean",
                        "evidence": "training",
                        "roles": ["training_proxy"],
                        "proxy_for": "eval/full/progress/kills/mean",
                    },
                    {
                        "metric": "eval/full/progress/kills/max",
                        "direction": "max",
                        "label": "Eval max kills",
                        "evidence": "evaluation",
                        "roles": ["tie_breaker"],
                        "rank_index": 1,
                    },
                    {
                        "metric": "train/episode/return/shaped/origin/target/rolling/mean",
                        "direction": "max",
                        "label": "Recent target return mean",
                        "evidence": "training",
                        "roles": ["optimization"],
                    },
                ),
                selection_fence="f" * 64,
            )

    class FakeEvaluationQueue:
        def __init__(self):
            self.requests = []

        def enqueue(self, *, run_id, checkpoint_ids, selection_fence):
            assert len(selection_fence) == 64
            self.requests.append((run_id, list(checkpoint_ids)))
            return {
                "items": (
                    {
                        "checkpoint_id": checkpoint_ids[0],
                        "job_id": "job-" + "c" * 32,
                        "state": "queued",
                        "evaluation": None,
                        "message": None,
                    },
                ),
                "jobs": [{"job_id": "job-" + "c" * 32, "state": "queued"}],
                "worker": {"state": "started", "pid": 123, "message": None},
            }

        @staticmethod
        def statuses(*, run_id, checkpoint_ids):
            assert run_id == "gradlab-" + "a" * 32
            assert checkpoint_ids == ["checkpoint-1-" + "b" * 16]
            return {
                checkpoint_ids[0]: {
                    "state": "accepted",
                    "evaluation": {
                        "status": "accepted",
                        "pass": True,
                        "episodes_planned": 100,
                        "episodes_completed": 100,
                        "failure_count": 0,
                        "criteria": [],
                        "metrics": {
                            "eval/full/progress/kills/mean": 11.0,
                            "eval/full/progress/kills/max": 16.0,
                        },
                    },
                }
            }

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
                denied = await client.get(f"{server.origin}/api/catalog/environments?q=mario")
                assert denied.status == 401
                noncurrent_api = await client.get(
                    f"{server.origin}/api/catalog/projects?q=mario",
                    headers={"Authorization": f"Bearer {server.token}"},
                )
                assert noncurrent_api.status == 404
                noncurrent_page = await client.get(
                    f"{server.origin}/projects/Mario",
                    allow_redirects=False,
                )
                assert noncurrent_page.status == 404
                accepted = await client.get(
                    f"{server.origin}/api/catalog/environments?q=mario",
                    headers={"Authorization": f"Bearer {server.token}"},
                )
                assert accepted.status == 200
                assert (await accepted.json())["items"][0]["name"] == "Mario"
                goals = await client.get(
                    f"{server.origin}/api/catalog/environments/Mario/goals",
                    headers={"Authorization": f"Bearer {server.token}"},
                )
                assert goals.status == 200
                assert (await goals.json())["items"][0]["goal_id"] == "Level1-1"
                goal_inspection = await client.get(
                    (f"{server.origin}/api/catalog/environments/Mario/goals/Level1-1/inspection"),
                    headers={"Authorization": f"Bearer {server.token}"},
                )
                assert goal_inspection.status == 200
                assert (await goal_inspection.json())["documents"]["goal"][
                    "availability"
                ] == "exact"
                recipes = await client.get(
                    (f"{server.origin}/api/catalog/environments/Mario/goals/Level1-1/recipes"),
                    headers={"Authorization": f"Bearer {server.token}"},
                )
                assert recipes.status == 200
                assert (await recipes.json())["items"][0]["recipe_id"] == "ppo"
                recipe_inspection = await client.get(
                    (
                        f"{server.origin}/api/catalog/environments/Mario"
                        "/goals/Level1-1/recipes/ppo/inspection"
                    ),
                    headers={"Authorization": f"Bearer {server.token}"},
                )
                assert recipe_inspection.status == 200
                assert (await recipe_inspection.json())["documents"]["recipe"][
                    "availability"
                ] == "static-preview"
                variants = await client.get(
                    (f"{server.origin}/api/catalog/environments/Mario/goals/Level1-1/variants"),
                    headers={"Authorization": f"Bearer {server.token}"},
                )
                assert variants.status == 200
                variant_id = (await variants.json())["items"][0]["variant_id"]
                activity = await client.get(
                    (f"{server.origin}/api/catalog/environments/Mario/goals/Level1-1/activity"),
                    headers={"Authorization": f"Bearer {server.token}"},
                )
                assert activity.status == 200
                assert "ETag" not in activity.headers
                unchanged = await client.get(
                    (f"{server.origin}/api/catalog/environments/Mario/goals/Level1-1/activity"),
                    headers={
                        "Authorization": f"Bearer {server.token}",
                        "If-None-Match": f'"{"f" * 64}"',
                    },
                )
                assert unchanged.status == 200
                variant_inspection = await client.get(
                    (
                        f"{server.origin}/api/catalog/environments/Mario"
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
                        f"{server.origin}/api/catalog/environments/Mario"
                        f"/goals/Level1-1/variants/{variant_id}/runs"
                    ),
                    headers={"Authorization": f"Bearer {server.token}"},
                )
                assert runs.status == 200
                assert (await runs.json())["items"][0]["goal_variant_id"] == variant_id
                checkpoints = await client.get(
                    f"{server.origin}/api/catalog/runs/gradlab-{'a' * 32}/checkpoints",
                    headers={"Authorization": f"Bearer {server.token}"},
                )
                assert checkpoints.status == 200
                checkpoint_payload = await checkpoints.json()
                assert checkpoint_payload["items"][0]["evaluation"]["pass"] is True
                selection_fence = checkpoint_payload["selection_fence"]
                assert checkpoint_payload["metric_columns"] == [
                    {
                        "metric": "eval/full/progress/kills/mean",
                        "direction": "max",
                        "label": "Eval mean kills",
                        "evidence": "evaluation",
                        "roles": ["objective", "acceptance"],
                        "rank_index": 0,
                        "acceptance": [
                            {
                                "metric": "eval/full/progress/kills/mean",
                                "operator": ">=",
                                "threshold": 10.0,
                            }
                        ],
                    },
                    {
                        "metric": "train/progress/kills/origin/target/rolling/mean",
                        "direction": "max",
                        "label": "Recent target kills mean",
                        "evidence": "training",
                        "roles": ["training_proxy"],
                        "proxy_for": "eval/full/progress/kills/mean",
                    },
                    {
                        "metric": "eval/full/progress/kills/max",
                        "direction": "max",
                        "label": "Eval max kills",
                        "evidence": "evaluation",
                        "roles": ["tie_breaker"],
                        "rank_index": 1,
                    },
                    {
                        "metric": "train/episode/return/shaped/origin/target/rolling/mean",
                        "direction": "max",
                        "label": "Recent target return mean",
                        "evidence": "training",
                        "roles": ["optimization"],
                    },
                ]
                assert checkpoint_payload["items"][0]["metrics"] == {
                    "eval/full/progress/kills/mean": 11.0,
                    "train/progress/kills/origin/target/rolling/mean": 8.5,
                    "eval/full/progress/kills/max": 16.0,
                    "train/episode/return/shaped/origin/target/rolling/mean": 120.0,
                }
                assert checkpoint_payload["items"][0]["best_metrics"] == [
                    "eval/full/progress/kills/mean",
                    "train/progress/kills/origin/target/rolling/mean",
                    "eval/full/progress/kills/max",
                    "train/episode/return/shaped/origin/target/rolling/mean",
                ]
                training = await client.get(
                    (f"{server.origin}/api/catalog/runs/gradlab-{'a' * 32}/checkpoint-training"),
                    headers={"Authorization": f"Bearer {server.token}"},
                )
                assert training.status == 200
                assert (await training.json())["training_enrichment"] == "complete"
                assert FakeCatalog.checkpoint_modes == [False, True]
                run_inspection = await client.get(
                    f"{server.origin}/api/catalog/runs/gradlab-{'a' * 32}/inspection",
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
                    json={
                        "checkpoint_ids": ["checkpoint-1-" + "b" * 16],
                        "selection_fence": selection_fence,
                    },
                )
                assert evaluation.status == 202
                evaluation_payload = await evaluation.json()
                assert evaluation_payload["items"][0]["state"] == "queued"
                assert evaluation_payload["worker"]["state"] == "started"
                assert evaluation_queue.requests == [
                    (
                        "gradlab-" + "a" * 32,
                        ["checkpoint-1-" + "b" * 16],
                    )
                ]
                for route in (
                    "/",
                    "/environments/Mario",
                    "/environments/Mario/goals/Level1-1",
                    (f"/environments/Mario/goals/Level1-1/variants/goal-variant-{'c' * 24}"),
                ):
                    page = await client.get(f"{server.origin}{route}")
                    assert page.status == 200
                    assert "<title>gradlab</title>" in await page.text()
        finally:
            runner.stop()
            await asyncio.wait_for(task, timeout=3.0)

    asyncio.run(scenario())


def test_initial_environment_catalog_is_embedded_in_selection_snapshots() -> None:
    class FakeCatalog:
        calls = 0

        @classmethod
        def initial_environments(cls):
            cls.calls += 1
            return {
                "items": [
                    {
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
                "route": {"level": "environments"},
            },
        },
    )

    assert FakeCatalog.calls == 1
    assert snapshot["app"]["catalog"]["items"][0]["name"] == "Mario"

    nested_snapshot = server._snapshot_for(
        client,
        {
            "type": "snapshot",
            "app": {
                "phase": "selecting",
                "route": {
                    "level": "runs",
                    "environment_id": "Mario",
                    "goal_id": "Level1-1",
                },
            },
        },
    )
    assert nested_snapshot["app"]["catalog"]["items"][0]["name"] == "Mario"


def test_player_binds_before_initial_catalog_work() -> None:
    release = threading.Event()

    class SlowCatalog:
        @staticmethod
        def initial_environments():
            assert release.wait(timeout=3.0)
            return {"items": [], "next_cursor": None}

    async def scenario() -> None:
        runner = HumanRecordingRunner(FakeHumanSession(), human_args())
        server = PlaybackWebServer(runner, human_args(), catalog=SlowCatalog())
        started_at = time.monotonic()
        task = asyncio.create_task(server.run())
        try:
            deadline = started_at + 1.0
            while not server.origin and time.monotonic() < deadline:
                await asyncio.sleep(0.005)
            assert server.origin
            assert time.monotonic() - started_at < 0.5
            assert not task.done()
            release.set()
            await asyncio.sleep(0.05)
        finally:
            release.set()
            runner.stop()
            await asyncio.wait_for(task, timeout=3.0)

    asyncio.run(scenario())


def test_web_dashboard_assets_are_packaged_beside_server() -> None:
    root = Path(__file__).parents[1] / "src" / "gradlab" / "web_player"
    panel_root = root / "panels"
    font_root = root / "fonts"
    expected_assets = (
        root / "index.html",
        root / "oauth_complete.html",
        root / "oauth_complete.js",
        root / "favicon.svg",
        root / "styles.css",
        root / "tabler-icons.svg",
        root / "tabler-chevron-down.svg",
        root / "vendor" / "gridstack" / "gridstack-all.js",
        root / "vendor" / "gridstack" / "gridstack.min.css",
        root / "sources" / "browser.js",
        root / "documents" / "diff.js",
        root / "documents" / "viewer.js",
        root / "documents" / "syntax.js",
        font_root / "ChivoVariable.woff2",
        font_root / "InterVariable.woff2",
        font_root / "InterVariable-Italic.woff2",
        font_root / "JetBrainsMonoVariable.woff2",
        font_root / "JetBrainsMonoVariable-Italic.woff2",
        panel_root / "catalog.js",
        panel_root / "diagnostic-overlays.js",
        panel_root / "layout-sizing.js",
        panel_root / "manager.js",
        panel_root / "runtime.js",
        panel_root / "shared.js",
        panel_root / "telemetry.js",
        panel_root / "telemetry-panel.js",
        panel_root / "workspace.js",
    )
    assert all(path.is_file() for path in expected_assets)
    specialized_panels = {
        "game",
        "controls",
        "observation",
        "attribution",
        "events",
        "raw",
    }
    assert all((panel_root / f"{name}.js").is_file() for name in specialized_panels)
    removed_metric_panels = {"policy", "reward", "actions", "signals"}
    assert all(not (panel_root / f"{name}.js").exists() for name in removed_metric_panels)

    markup = (root / "index.html").read_text(encoding="utf-8")
    styles = (root / "styles.css").read_text(encoding="utf-8")
    for name in (
        "ChivoVariable.woff2",
        "InterVariable.woff2",
        "InterVariable-Italic.woff2",
        "JetBrainsMonoVariable.woff2",
        "JetBrainsMonoVariable-Italic.woff2",
    ):
        assert f'url("/assets/fonts/{name}") format("woff2")' in styles
    script = (root / "app.js").read_text(encoding="utf-8")
    oauth_script = (root / "oauth_complete.js").read_text(encoding="utf-8")
    source_browser = (root / "sources" / "browser.js").read_text(encoding="utf-8")
    contract_viewer = (root / "documents" / "viewer.js").read_text(encoding="utf-8")
    contract_diff = (root / "documents" / "diff.js").read_text(encoding="utf-8")
    contract_syntax = (root / "documents" / "syntax.js").read_text(encoding="utf-8")
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
    assert '<span class="app-wordmark eyebrow">GRADLAB</span>' in markup
    assert "GRADLAB PLAYER" not in markup
    assert 'id="source-breadcrumbs"' in markup
    assert '$("#source-breadcrumbs")' in script
    assert "snapshot?.publication_capture?.ready === true" in script
    assert "Boolean(snapshot?.publication_capture?.latest)" not in script
    assert 'await publicationApi("/api/publication/render", { method: "POST" });' in script
    assert "gradlab-youtube-oauth-complete" in script
    assert "event.source !== youtubeOAuthPopup" in script
    assert "gradlab-youtube-oauth-complete" in oauth_script
    assert "window.opener.postMessage(message, location.origin)" in oauth_script
    assert "location.replace(`/#token=${encodeURIComponent(token)}`)" in oauth_script
    assert '$("#player-home")' not in script
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
    assert 'title: "Action history"' not in catalog
    assert '"namespace-explorer"' in catalog
    assert 'data-driver-option="human"' not in controls
    assert 'data-driver-option="policy"' not in controls
    assert 'driver: "policy"' in controls
    assert 'data-command="set-fps"' not in controls
    assert 'fps.addEventListener("input"' in controls
    assert 'services.command("set_fps", { fps: Number(fps.value) })' in controls
    assert "WORKSPACE_VERSION = 7" in workspace
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
    assert "actionComparisonPresentation" in telemetry_panel
    assert "makeNamespaceBlock" in telemetry_panel
    assert '"action/policy"' in telemetry
    assert '"action/executed"' in telemetry
    assert "dynamicDescriptorKey" in telemetry
    assert "function hideGoExploreValuePanel(snapshot)" in script
    assert 'search_algorithm_id !== "go-explore"' in script
    assert "hideGoExploreValuePanel(snapshot)" in script
    assert 'features.add("rewards")' in script
    assert "state.backgroundPlaybackSnapshot = message" in script
    assert 'sourceBrowser.activeBreadcrumbRoute = ""' in script
    assert 'type: "inspection_frames"' in script
    assert "sequence < (state.receivedFrameSequence" not in script

    assert '"gradlab.player.workspace.v7.paired"' in script
    assert '"gradlab.player.workspace.v7.single"' in script
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
    assert ".syntax-key" in styles
    assert "contractSyntaxTokens(value, this.view)" in contract_viewer
    assert "buildSideBySideRows" in contract_viewer
    assert "sideBySideSearchCounts" in contract_viewer
    assert "export function buildSideBySideRows" in contract_diff
    assert 'id="contract-diff-base-scroll"' in markup
    assert 'id="contract-diff-resolved-scroll"' in markup
    assert ".contract-diff-content" in styles
    assert ".contract-diff-inline" in styles
    assert "export function contractSyntaxTokens(value, view)" in contract_syntax
    assert "export function contractSearchRanges(value, query)" in contract_syntax

    assert "export function sourceRouteFromPath(" in source_browser
    assert "export function sourceRoutePath(" in source_browser
    assert "export function formatDate(value, nowValue = Date.now())" in source_browser
    assert "export function recipeVariantPresentation(item)" in source_browser
    assert '{ label: "Recipe / variant" }' in source_browser
    assert "item.description || item.name || item.run_id" in source_browser
    assert 'history.pushState(null, "", target);' in source_browser
    assert 'window.addEventListener("popstate", this.onPopState);' in source_browser
    assert "goHome()" in source_browser
    assert "hydrateInitialEnvironments()" in source_browser
