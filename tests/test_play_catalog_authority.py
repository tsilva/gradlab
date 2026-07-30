from __future__ import annotations

import json
from pathlib import Path

import pytest

from gradlab.catalog_errors import CatalogIntegrityError
from gradlab.play_catalog_authority import (
    start_catalog_authority_helper,
)


def test_catalog_authority_helper_reads_only_allowlisted_control_documents(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    control = tmp_path / "control"
    key = "goal-variants/v2/goals/" + "a" * 64 + "/index.json"
    target = control / key
    target.parent.mkdir(parents=True)
    target.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "scope": {"goal_slug": "Mario/Level1-1"},
                "variants": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GRADLAB_CONTROL_R2_URI", control.resolve().as_uri())
    monkeypatch.setenv("WANDB_API_KEY", "must-not-enter-control-helper")
    monkeypatch.setenv(
        "GRADLAB_OPERATOR_CONFIG",
        str(tmp_path / "missing-operator.toml"),
    )

    helper = start_catalog_authority_helper(tmp_path)
    try:
        assert helper.get_json_optional(key) == {
            "schema_version": 2,
            "scope": {"goal_slug": "Mario/Level1-1"},
            "variants": [],
        }
        with pytest.raises(CatalogIntegrityError, match="unsupported control object"):
            helper.get_json_optional("runs/not-a-run/manifest.json")
    finally:
        helper.close()
