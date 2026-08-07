from __future__ import annotations

import importlib.util
import json
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


def test_replace_description_link_block_accepts_a_direct_model_url() -> None:
    updated = MODULE.replace_description_link_block(
        "Model: https://huggingface.co/tsilva/old",
        "https://huggingface.co/tsilva/new",
    )

    assert updated == "Model: https://huggingface.co/tsilva/new"


def test_parser_defaults_youtube_credentials_to_gradlab_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)

    args = MODULE.build_parser().parse_args([])

    config = (tmp_path / ".config" / "gradlab").resolve()
    assert args.client_secret == config / "youtube_client_secret.json"
    assert args.token == config / "youtube_token.json"


def test_access_token_merges_client_config_into_player_issued_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token_path = tmp_path / "youtube_token.json"
    token_path.write_text(json.dumps({"refresh_token": "refresh"}), encoding="utf-8")
    observed: dict[str, str] = {}

    def fake_refresh(token: dict[str, str]) -> dict[str, str]:
        observed.update(token)
        return {**token, "access_token": "access"}

    monkeypatch.setattr(MODULE, "refresh_token", fake_refresh)

    value = MODULE.access_token(
        {
            "client_id": "client",
            "client_secret": "secret",
            "token_uri": "https://oauth.example/token",
        },
        token_path,
        no_browser=True,
    )

    assert value == "access"
    assert observed["client_id"] == "client"
    assert observed["client_secret"] == "secret"
    assert observed["token_uri"] == "https://oauth.example/token"
