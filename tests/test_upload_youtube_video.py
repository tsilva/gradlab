from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "upload_youtube_video.py"
SPEC = importlib.util.spec_from_file_location("upload_youtube_video", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_replace_description_link_block_preserves_existing_copy() -> None:
    existing = """A PPO agent completes Level 1-4.

Model: https://huggingface.co/tsilva/old
gradlab: https://github.com/tsilva/gradlab

#ReinforcementLearning #PPO #SuperMarioBros"""

    updated = MODULE.replace_description_link_block(
        existing,
        "Model: https://huggingface.co/tsilva/new\n"
        "gradlab: https://github.com/tsilva/gradlab",
    )

    assert "A PPO agent completes Level 1-4." in updated
    assert "https://huggingface.co/tsilva/new" in updated
    assert "https://huggingface.co/tsilva/old" not in updated
    assert updated.count("Model:") == 1
    assert updated.count("gradlab:") == 1
    assert updated.endswith("#ReinforcementLearning #PPO #SuperMarioBros")


def test_replace_description_link_block_rejects_unrelated_lines() -> None:
    with pytest.raises(ValueError, match="only Model: and gradlab:"):
        MODULE.replace_description_link_block("existing", "Title: not allowed")
