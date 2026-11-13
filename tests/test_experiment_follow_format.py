import json
from pathlib import Path
from unittest.mock import patch

from gradlab.experiment_cli import _follow_run


def test_large_pretty_snapshot_preserves_all_evidence_without_oversized_lines(capsys):
    state = {
        "run_id": "gradlab-" + "a" * 32,
        "completed": True,
        "semantic": {"manifest": {"rows": [{"receipt": "x" * 150} for _ in range(1000)]}},
        "attempt_terminal": {"state": "succeeded", "final_step": 1234},
    }
    with patch("gradlab.experiment_cli._poll_status", return_value=iter([(state, False)])):
        assert (
            _follow_run(Path.cwd(), state["run_id"], timeout=10, poll_seconds=1, pretty=True) == 0
        )
    output = capsys.readouterr().out
    assert len(output) > 65536
    assert max(len(line.encode()) for line in output.splitlines()) < 65536
    assert json.loads(output) == state
