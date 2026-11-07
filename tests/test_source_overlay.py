from pathlib import Path

import pytest

from gradlab.file_utils import atomic_write_json, file_sha256
from gradlab.source_overlay import read_source_overlay, validate_source_overlay, verify_source_tree


def test_source_overlay_binds_recipe_and_rejects_source_drift(tmp_path: Path) -> None:
    root = tmp_path / "source"
    trainer = root / "src/gradlab/trainer.py"
    trainer.parent.mkdir(parents=True)
    trainer.write_text("VALUE = 1\n")
    name = trainer.relative_to(root).as_posix()
    base_commit = "a" * 40
    baseline = tmp_path / "baseline.json"
    atomic_write_json(baseline, {"commit": base_commit, "sha256": {name: file_sha256(trainer)}})

    trainer.write_text("VALUE = 2\n")
    digest = file_sha256(trainer)
    inputs = tmp_path / "inputs.json"
    atomic_write_json(inputs, {baseline.name: file_sha256(baseline), f"overlay/{name}": digest})
    recipe = {"provenance": {"source_files": [{"path": name, "sha256": digest}]}}
    overlay = read_source_overlay(root, baseline, inputs, recipe=recipe)
    validate_source_overlay(overlay, base_commit=base_commit)

    with pytest.raises(ValueError, match="inconsistent provenance"):
        validate_source_overlay(overlay, base_commit="b" * 40)
    unrecorded = trainer.with_name("unrecorded.py")
    unrecorded.touch()
    with pytest.raises(ValueError, match="unrecorded GradLab source"):
        verify_source_tree(root, overlay["effective_files"])
    unrecorded.unlink()
    trainer.write_text("VALUE = 3\n")
    with pytest.raises(ValueError, match="source file digest mismatch"):
        verify_source_tree(root, overlay["effective_files"])
