from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock, patch
import threading

from gradlab.play_session import build_parser
from gradlab.play_web import WebPlaybackRunner


def test_default_display_fps():
    assert build_parser().get_default("fps") == 30.0


def test_policy_loop_does_not_wait_for_display_clock():
    runner = object.__new__(WebPlaybackRunner)
    runner._stop = threading.Event()
    runner._drain_commands = Mock()
    runner.recording_enabled = False
    runner.run_state = "playing"
    runner.driver = "policy"
    runner.target_fps = 1.0
    steps = []

    def step():
        steps.append(1)
        if len(steps) == 10:
            runner._stop.set()

    runner._step_once = step
    with patch("gradlab.play_web.time.perf_counter", return_value=0), patch(
        "gradlab.play_web.time.sleep", side_effect=AssertionError("inference was paced")
    ):
        runner._run()
    assert len(steps) == 10


def test_presentation_pacing_preserves_history_and_forces_final_frame():
    runner = object.__new__(WebPlaybackRunner)
    runner.target_fps = 30.0
    runner._next_presentation_at = 0.0
    runner.processing_features = {"history", "game"}
    runner.history = deque()
    runner.encoder = Mock()
    runner._snapshot_payload = Mock(return_value={})
    runner._snapshot_lock = threading.Lock()
    runner._snapshot_updates = deque()
    transition = SimpleNamespace(sequence=1, boundary=False, after_frame=object(), before_frames=())
    with patch("gradlab.play_web.time.perf_counter", return_value=1.0), patch(
        "gradlab.play_web.history_point_payload", side_effect=lambda current: current
    ):
        runner._publish(transition, current={"sequence": 1}, paced=True)
        transition.sequence = 2
        runner._publish(transition, current={"sequence": 2}, paced=True)
        assert runner.encoder.submit_batch.call_count == 1
        assert [point["sequence"] for point in runner.history] == [1, 2]
        runner._publish(transition, current={"sequence": 2})
        assert runner.encoder.submit_batch.call_count == 2
        assert runner.encoder.submit_batch.call_args.args[0] == 2
