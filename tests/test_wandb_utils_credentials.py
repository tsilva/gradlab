from __future__ import annotations

import os
from pathlib import Path

import pytest

from rlab.operator_credentials import KeychainReference
from rlab.wandb_utils import load_wandb_env


def test_wandb_loader_uses_scoped_operator_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "operator.toml"
    config_path.write_text(
        """
schema_version = 1

[environment]
WANDB_ENTITY = "research"

[keychain.WANDB_API_KEY]
service = "rlab-wandb"
account = "api-key"

[modal]
path = "/does/not/exist"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    config_path.chmod(0o600)
    monkeypatch.setenv("RLAB_OPERATOR_CONFIG", str(config_path))
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.setattr(
        "rlab.operator_credentials._keychain_lookup",
        lambda reference: (
            "wandb-key"
            if reference == KeychainReference("rlab-wandb", "api-key")
            else None
        ),
    )

    load_wandb_env(tmp_path / ".env")

    assert os.environ["WANDB_API_KEY"] == "wandb-key"
    assert os.environ["WANDB_ENTITY"] == "research"
