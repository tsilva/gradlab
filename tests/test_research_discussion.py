from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/update_research_discussion.py"
SPEC = importlib.util.spec_from_file_location("update_research_discussion", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_discussion_body_uses_immutable_v3_record_and_distinguishes_replay() -> None:
    body = MODULE.discussion_body(
        [
            {
                "repository": {
                    "repo_id": "tsilva/example",
                    "canonical_environment_id": "Example-v0",
                    "goal_id": "Goal1",
                    "trainer": "GradLab",
                    "algorithm": "ppo",
                },
                "release": {
                    "version": "v3",
                    "youtube_url": "https://youtu.be/example",
                    "published_at": "2026-08-07T00:00:00Z",
                },
                "evaluation": {
                    "checkpoint_step": 10,
                    "acceptance": {
                        "outcomes": [
                            {"label": "Score", "value": 12, "passed": True}
                        ]
                    },
                },
            }
        ]
    )
    assert "https://huggingface.co/tsilva/example/tree/v3" in body
    assert "Score: 12 (pass)" in body
    assert "evaluation evidence and representative replay are distinct" in body.casefold()
