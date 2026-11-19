from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from gradlab.catalog_errors import CatalogIntegrityError, CatalogUnavailable
from gradlab.goal_catalog import GOAL_CATALOG_ROOT, GOAL_CATALOG_SCHEMA_VERSION
from gradlab.play_catalog_authority import (
    CATALOG_EXECUTOR_ENV,
    main,
    start_catalog_authority_helper,
)


def test_catalog_probe_checks_only_manifest_head(monkeypatch, capsys):
    bucket = Mock()
    monkeypatch.setattr("gradlab.play_catalog_authority._scoped_control_bucket", lambda: bucket)
    run_id = "gradlab-" + "a" * 32

    assert main(["--probe-run", run_id]) == 0
    bucket.head.assert_called_once_with(f"runs/{run_id}/manifest.json")
    assert [call[0] for call in bucket.mock_calls] == ["head"]
    assert capsys.readouterr().out == ""  # Runbox emits the success record.


def test_external_executor_uses_private_socket_and_clean_environment(monkeypatch, tmp_path):
    control = tmp_path / "control"
    key = "runs/gradlab-" + "a" * 32 + "/manifest.json"
    target = control / key
    target.parent.mkdir(parents=True)
    target.write_text('{"synthetic": true}')
    package = str(Path(__file__).resolve().parents[1] / "src")
    command = [
        sys.executable, "-c",
        """
import os, socket, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from gradlab.play_catalog_authority import _serve
assert "GRADLAB_CONTROL_R2_SECRET_ACCESS_KEY" not in os.environ
assert "INFISICAL_AGENT_VAULT_URL" not in os.environ
os.environ["GRADLAB_CONTROL_R2_URI"] = Path(sys.argv[2]).as_uri()
os.environ["GRADLAB_OPERATOR_CONFIG"] = str(Path(sys.argv[3]) / "missing-operator.toml")
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
    path = Path(sys.argv[4])
    assert path.parent.stat().st_mode & 0o077 == 0
    connection.connect(str(path))
    assert _serve(connection, Path(sys.argv[3])) == 0
""",
        package, str(control), str(tmp_path),
    ]
    monkeypatch.setenv(CATALOG_EXECUTOR_ENV, json.dumps(command))
    monkeypatch.setenv("GRADLAB_CONTROL_R2_SECRET_ACCESS_KEY", "synthetic-key")
    monkeypatch.setenv("INFISICAL_AGENT_VAULT_URL", "https://vault.example.test")

    helper = start_catalog_authority_helper(tmp_path)
    other = None
    try:
        assert helper.get_json_optional(key) == {"synthetic": True}
        other = start_catalog_authority_helper(tmp_path)
        with pytest.raises(CatalogIntegrityError, match="unsupported control object"):
            helper.get_json_optional("secrets/stored.json")
        helper.close()
        assert other.get_json_optional(key) == {"synthetic": True}
    finally:
        helper.close()
        if other is not None:
            other.close()
    assert helper._process is not None and helper._process.returncode == 0


def test_agent_cannot_fall_back_to_internal_catalog_helper(monkeypatch, tmp_path):
    monkeypatch.delenv(CATALOG_EXECUTOR_ENV, raising=False)
    monkeypatch.setenv("INFISICAL_AGENT_VAULT_URL", "https://vault.example.test")
    with pytest.raises(CatalogUnavailable, match="external catalog executor"):
        start_catalog_authority_helper(tmp_path)
    monkeypatch.setenv(CATALOG_EXECUTOR_ENV, '["runbox"]')
    with pytest.raises(CatalogUnavailable, match="absolute command array"):
        start_catalog_authority_helper(tmp_path)


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
        monitoring_key = (
            "runs/gradlab-" + "a" * 32 + "/monitoring/" + "b" * 64
            + "/" + "c" * 64 + "/state.json"
        )
        monitoring_target = control / monitoring_key
        monitoring_target.parent.mkdir(parents=True)
        monitoring_target.write_text(json.dumps({"status": "running"}), encoding="utf-8")
        assert helper.get_json_optional(monitoring_key) == {"status": "running"}
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
