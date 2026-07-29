from __future__ import annotations

import signal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

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


def test_process_group_escalation_targets_the_spawned_group() -> None:
    runtime = SupervisorRuntime()
    learner = MagicMock(pid=4321)
    learner._gradlab_process_group_id = 4321

    with patch("gradlab.supervisor_runtime.os.killpg") as killpg:
        assert runtime.learner_group_alive(learner) is True
        runtime.terminate_learner_group(learner)
        runtime.kill_learner_group(learner)

    assert killpg.call_args_list == [
        call(4321, 0),
        call(4321, signal.SIGTERM),
        call(4321, signal.SIGKILL),
    ]


def test_missing_process_group_is_already_gone() -> None:
    runtime = SupervisorRuntime()
    learner = MagicMock(pid=4321)
    learner._gradlab_process_group_id = 4321

    with patch(
        "gradlab.supervisor_runtime.os.killpg",
        side_effect=ProcessLookupError,
    ):
        assert runtime.learner_group_alive(learner) is False
        runtime.terminate_learner_group(learner)
        runtime.kill_learner_group(learner)


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
