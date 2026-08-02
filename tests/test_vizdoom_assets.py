from __future__ import annotations

from pathlib import Path

import pytest

from gradlab.env import EnvConfig
from gradlab.env_identity import environment_identity_from_train_config
from gradlab.env_providers import provider_native_vec_kwargs
from gradlab.vizdoom_assets import (
    apply_optional_local_vizdoom_iwad,
    portable_vizdoom_iwad_identity,
    validate_vizdoom_iwad_binding,
    vizdoom_iwad_binding,
)


def _iwad(path: Path, payload: bytes = b"doom") -> Path:
    path.write_bytes(b"IWAD" + payload)
    return path


def _vizdoom_document() -> dict:
    return {
        "train_config": {
            "env_provider": "vizdoom-turbo",
            "game": "VizdoomBasic-v1",
            "env_args": {"rom_path": None},
        }
    }


def test_local_vizdoom_document_uses_default_iwad_only_when_available(tmp_path: Path) -> None:
    document = _vizdoom_document()

    assert not apply_optional_local_vizdoom_iwad(
        document,
        default_path=tmp_path / "missing.wad",
    )
    assert document["train_config"]["env_args"]["rom_path"] is None

    path = _iwad(tmp_path / "doom2.wad")
    assert apply_optional_local_vizdoom_iwad(document, default_path=path)
    binding = document["train_config"]["env_args"]["rom_path"]
    assert binding["filename"] == "doom2.wad"
    assert binding["size_bytes"] == path.stat().st_size


def test_vizdoom_iwad_binding_verifies_header_size_and_digest(tmp_path: Path) -> None:
    path = _iwad(tmp_path / "doom2.wad")
    binding = vizdoom_iwad_binding(path)

    assert validate_vizdoom_iwad_binding(binding, verify_file=True) == binding
    path.write_bytes(b"IWADchanged")
    with pytest.raises(ValueError, match="size mismatch|sha256 mismatch"):
        validate_vizdoom_iwad_binding(binding, verify_file=True)

    invalid = tmp_path / "not-an-iwad.wad"
    invalid.write_bytes(b"PWADdoom")
    with pytest.raises(ValueError, match="not an IWAD"):
        vizdoom_iwad_binding(invalid)


def test_vizdoom_iwad_identity_records_bytes_but_not_local_path(tmp_path: Path) -> None:
    path = _iwad(tmp_path / "doom2.wad")
    binding = vizdoom_iwad_binding(path)
    moved = {**binding, "path": "/different/local/path/doom2.wad"}
    base = {
        "env_provider": "vizdoom-turbo",
        "game": "VizdoomBasic-v1",
        "task": {},
        "env_args": {"rom_path": binding},
    }

    identity = environment_identity_from_train_config(base)
    moved_identity = environment_identity_from_train_config(
        {**base, "env_args": {"rom_path": moved}}
    )

    assert identity == moved_identity
    assert identity["provider_args"]["rom_asset"] == portable_vizdoom_iwad_identity(binding)
    assert "path" not in identity["provider_args"]["rom_asset"]


def test_vizdoom_native_kwargs_resolve_and_verify_iwad_binding(tmp_path: Path) -> None:
    path = _iwad(tmp_path / "doom2.wad")
    binding = vizdoom_iwad_binding(path)
    config = EnvConfig(
        env_provider="vizdoom-turbo",
        game="VizdoomBasic-v1",
        state="",
        env_args={"rom_path": binding},
    )

    kwargs = provider_native_vec_kwargs(
        config,
        n_envs=2,
        native_obs_crop=lambda _config: None,
        state_weight_mapping=lambda _config: {},
    )

    assert kwargs["rom_path"] == str(path.resolve())
