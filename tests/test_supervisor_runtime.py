from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from gradlab.supervisor_runtime import SupervisorRuntime


def test_learner_starts_in_a_dedicated_process_group(tmp_path: Path) -> None:
    process = MagicMock(pid=4321)
    with patch("gradlab.supervisor_runtime.subprocess.Popen", return_value=process) as popen:
        learner = SupervisorRuntime().start_learner(
            ("python", "-m", "gradlab.train"),
            log_path=tmp_path / "learner.log",
            environment={"PATH": "/usr/bin"},
        )

    assert learner is process
    assert process._gradlab_process_group_id == 4321
    assert popen.call_args.kwargs["start_new_session"] is True
    process._gradlab_log.close()


def test_cooperative_stop_targets_learner_and_escalation_targets_group() -> None:
    runtime = SupervisorRuntime()
    learner = MagicMock(pid=4321)
    learner._gradlab_process_group_id = 4321

    with (
        patch("gradlab.supervisor_runtime.os.kill") as kill,
        patch("gradlab.supervisor_runtime.os.killpg") as killpg,
    ):
        assert runtime.learner_group_alive(learner) is True
        runtime.request_learner_stop(learner)
        runtime.terminate_learner_group(learner)
        runtime.kill_learner_group(learner)

    kill.assert_called_once_with(4321, signal.SIGUSR1)
    assert killpg.call_args_list == [
        call(4321, 0),
        call(4321, signal.SIGTERM),
        call(4321, signal.SIGKILL),
    ]


def test_missing_process_group_is_already_gone() -> None:
    runtime = SupervisorRuntime()
    learner = MagicMock(pid=4321)
    learner._gradlab_process_group_id = 4321

    with (
        patch(
            "gradlab.supervisor_runtime.os.kill",
            side_effect=ProcessLookupError,
        ),
        patch(
            "gradlab.supervisor_runtime.os.killpg",
            side_effect=ProcessLookupError,
        ),
    ):
        assert runtime.learner_group_alive(learner) is False
        runtime.request_learner_stop(learner)
        runtime.terminate_learner_group(learner)
        runtime.kill_learner_group(learner)


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="requires SIGUSR1")
def test_cooperative_stop_does_not_signal_learner_descendants(tmp_path: Path) -> None:
    runtime = SupervisorRuntime()
    observed_path = tmp_path / "learner-stop-observed"
    child_pid_path = tmp_path / "child-pid"
    script = """
import signal
import subprocess
import sys
import time

observed_path, child_pid_path = sys.argv[1:3]

def stop_requested(_signum, _frame):
    with open(observed_path, "w", encoding="utf-8") as target:
        target.write("observed")

signal.signal(signal.SIGUSR1, stop_requested)
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
with open(child_pid_path, "w", encoding="utf-8") as target:
    target.write(str(child.pid))
while True:
    time.sleep(0.05)
"""
    learner = runtime.start_learner(
        (sys.executable, "-c", script, str(observed_path), str(child_pid_path)),
        log_path=tmp_path / "learner.log",
        environment=os.environ,
    )

    def wait_for(path: Path, *, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.is_file():
                return
            time.sleep(0.01)
        raise AssertionError(f"timed out waiting for {path.name}")

    try:
        wait_for(child_pid_path)
        child_pid = int(child_pid_path.read_text())
        runtime.request_learner_stop(learner)
        wait_for(observed_path)

        assert learner.poll() is None
        os.kill(child_pid, 0)
    finally:
        runtime.terminate_learner_group(learner)
        try:
            learner.wait(timeout=5)
        except subprocess.TimeoutExpired:
            runtime.kill_learner_group(learner)
            learner.wait(timeout=5)
        learner._gradlab_log.close()


def test_failed_terminal_projection_closes_wandb_with_nonzero_exit() -> None:
    projector = MagicMock()
    receipt = SimpleNamespace(state="resumable_failure")

    with (
        patch(
            "gradlab.supervisor_runtime.WandbProjector.resume",
            return_value=projector,
        ) as resume,
        patch("gradlab.supervisor_runtime.publish_terminal_summary") as publish,
    ):
        SupervisorRuntime().publish_terminal(
            {"wandb_run_id": "gradlab-" + "0" * 32},
            receipt,
            timeout_seconds=12,
        )

    resume.assert_called_once_with(
        {"wandb_run_id": "gradlab-" + "0" * 32},
        update_finish_state=True,
    )
    publish.assert_called_once_with(projector.run, receipt)
    projector.close.assert_called_once_with(timeout_seconds=12, exit_code=1)


def test_stopped_terminal_projection_closes_wandb_with_zero_exit() -> None:
    projector = MagicMock()
    receipt = SimpleNamespace(state="stopped")

    with (
        patch(
            "gradlab.supervisor_runtime.WandbProjector.resume",
            return_value=projector,
        ),
        patch("gradlab.supervisor_runtime.publish_terminal_summary"),
    ):
        SupervisorRuntime().publish_terminal(
            {"wandb_run_id": "gradlab-" + "0" * 32},
            receipt,
            timeout_seconds=12,
        )

    projector.close.assert_called_once_with(timeout_seconds=12, exit_code=0)
