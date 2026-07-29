from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gradlab.supervision_fault_learner import run_fixture


def test_failed_fixture_writes_v3_result_before_hanging(tmp_path: Path) -> None:
    run_id = "gradlab-" + "a" * 32
    attempt_id = "attempt-" + "b" * 16
    config_path = tmp_path / "train-config.json"
    config_path.write_text(
        json.dumps(
            {
                "wandb_run_id": run_id,
                "attempt_id": attempt_id,
                "training_backend": {"id": "sb3.ppo", "config": {}},
                "runs_dir": str(tmp_path),
                "run_name": run_id,
            }
        ),
        encoding="utf-8",
    )
    child = SimpleNamespace(pid=999)
    with (
        patch(
            "gradlab.supervision_fault_learner.subprocess.Popen",
            return_value=child,
        ),
        patch(
            "gradlab.supervision_fault_learner.time.sleep",
            side_effect=RuntimeError("stop fixture"),
        ),
        pytest.raises(RuntimeError, match="stop fixture"),
    ):
        run_fixture(
            train_config_path=config_path,
            mode="failed-result-live-process",
        )

    run_dir = tmp_path / run_id
    ready = json.loads((run_dir / "learner-ready.json").read_text(encoding="utf-8"))
    result = json.loads((run_dir / "training-result.json").read_text(encoding="utf-8"))
    assert ready["format_version"] == 3
    assert ready["run_id"] == run_id
    assert ready["attempt_id"] == attempt_id
    assert result["format_version"] == 3
    assert result["status"] == "failed"
    assert result["terminal_reason"] == "failed"
    assert result["final_step"] == 0
    assert result["error_type"] == "SupervisionFaultFixture"
