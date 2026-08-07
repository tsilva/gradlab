from __future__ import annotations

import json
from pathlib import Path

import pytest

from gradlab.catalog_errors import CatalogIntegrityError
from gradlab.goal_catalog import GOAL_CATALOG_ROOT, GOAL_CATALOG_SCHEMA_VERSION
from gradlab.play_catalog_authority import (
    start_catalog_authority_helper,
)


def test_catalog_authority_helper_reads_only_allowlisted_control_documents(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    control = tmp_path / "control"
    key = GOAL_CATALOG_ROOT + "/goals/" + "a" * 64 + "/current.json"
    target = control / key
    target.parent.mkdir(parents=True)
    target.write_text(
        json.dumps(
            {
                "schema_version": GOAL_CATALOG_SCHEMA_VERSION,
                "goal_slug": "Mario/Level1-1",
                "generation_sha256": "b" * 64,
                "generation_key": (
                    GOAL_CATALOG_ROOT
                    + "/goals/"
                    + "a" * 64
                    + "/generations/"
                    + "b" * 64
                    + ".json"
                ),
                "generated_at": "2026-08-04T12:00:00Z",
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
            "schema_version": GOAL_CATALOG_SCHEMA_VERSION,
            "goal_slug": "Mario/Level1-1",
            "generation_sha256": "b" * 64,
            "generation_key": (
                GOAL_CATALOG_ROOT
                + "/goals/"
                + "a" * 64
                + "/generations/"
                + "b" * 64
                + ".json"
            ),
            "generated_at": "2026-08-04T12:00:00Z",
        }
        missing_key = GOAL_CATALOG_ROOT + "/goals/" + "c" * 64 + "/current.json"
        assert helper.get_json_many_optional((key, missing_key)) == {
            key: {
                "schema_version": GOAL_CATALOG_SCHEMA_VERSION,
                "goal_slug": "Mario/Level1-1",
                "generation_sha256": "b" * 64,
                "generation_key": (
                    GOAL_CATALOG_ROOT
                    + "/goals/"
                    + "a" * 64
                    + "/generations/"
                    + "b" * 64
                    + ".json"
                ),
                "generated_at": "2026-08-04T12:00:00Z",
            },
            missing_key: None,
        }
        with pytest.raises(CatalogIntegrityError, match="unsupported control object"):
            helper.get_json_optional("runs/not-a-run/manifest.json")
        with pytest.raises(CatalogIntegrityError, match="unsupported control object"):
            helper.get_json_many_optional((key, "runs/not-a-run/manifest.json"))
        with pytest.raises(CatalogIntegrityError, match="unsupported control object"):
            helper.get_json_optional("goal-variants/v3/current.json")
        with pytest.raises(CatalogIntegrityError, match="unsupported control object"):
            helper.get_json_optional(
                "goal-catalog/v1/goals/" + "a" * 64 + "/current.json"
            )
    finally:
        helper.close()
