from __future__ import annotations

from pathlib import Path

import pytest

import gradlab.vizdoom_assets as vizdoom_assets
from gradlab.env import EnvConfig
from gradlab.env_identity import environment_identity_from_train_config
from gradlab.env_providers import provider_native_vec_kwargs
from gradlab.vizdoom_assets import (
    bind_required_local_vizdoom_iwad,
    bind_vizdoom_iwad_to_document,
    install_vizdoom_iwad_file,
    portable_vizdoom_iwad_identity,
    required_vizdoom_iwad_binding,
    resolve_vizdoom_iwad_path,
    validate_vizdoom_iwad_binding,
    vizdoom_iwad_cache_path,
    vizdoom_iwad_binding,
)


def _iwad(path: Path, payload: bytes = b"doom") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"IWAD" + payload)
    return path


def _pin_test_iwad(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    binding = vizdoom_iwad_binding(path)
    monkeypatch.setattr(
        vizdoom_assets,
        "REQUIRED_VIZDOOM_IWAD_FILENAME",
        str(binding["filename"]),
    )
    monkeypatch.setattr(
        vizdoom_assets,
        "REQUIRED_VIZDOOM_IWAD_SIZE_BYTES",
        int(binding["size_bytes"]),
    )
    monkeypatch.setattr(
        vizdoom_assets,
        "REQUIRED_VIZDOOM_IWAD_SHA256",
        str(binding["sha256"]),
    )


def _vizdoom_document() -> dict:
    return {
        "train_config": {
            "env_provider": "vizdoom-turbo",
            "game": "VizdoomBasic-v1",
            "env_args": {"rom_path": None},
        }
    }


def test_local_vizdoom_document_requires_pinned_default_iwad(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _vizdoom_document()

    with pytest.raises(FileNotFoundError, match="IWAD path does not exist"):
        bind_required_local_vizdoom_iwad(
            document,
            default_path=tmp_path / "missing.wad",
        )
    assert document["train_config"]["env_args"]["rom_path"] is None

    path = _iwad(tmp_path / "doom2.wad")
    _pin_test_iwad(monkeypatch, path)
    assert bind_required_local_vizdoom_iwad(document, default_path=path)
    binding = document["train_config"]["env_args"]["rom_path"]
    assert binding["filename"] == "doom2.wad"
    assert binding["size_bytes"] == path.stat().st_size


def test_required_vizdoom_iwad_rejects_different_bytes(tmp_path: Path) -> None:
    path = _iwad(tmp_path / "doom2.wad")

    with pytest.raises(ValueError, match="requires the pinned Doom II IWAD"):
        required_vizdoom_iwad_binding(path)


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


def test_vizdoom_iwad_resolves_from_verified_runtime_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _iwad(tmp_path / "source" / "doom2.wad")
    binding = {**vizdoom_iwad_binding(source), "path": "~/roms/vizdoom/doom2.wad"}
    cache_root = tmp_path / "cache"

    installed = install_vizdoom_iwad_file(source, binding, cache_root=cache_root)
    source.unlink()
    monkeypatch.setenv("GRADLAB_ROM_CACHE_DIR", str(cache_root))

    assert installed == vizdoom_iwad_cache_path(cache_root, binding).resolve()
    assert resolve_vizdoom_iwad_path(binding) == str(installed)


def test_vizdoom_iwad_binding_updates_training_and_evaluation_contract(tmp_path: Path) -> None:
    binding = {
        **vizdoom_iwad_binding(_iwad(tmp_path / "doom2.wad")),
        "object_uri": "s3://private/vizdoom/doom2.wad",
    }
    document = _vizdoom_document()
    document["train_config"]["checkpoint_eval_environment"] = {"env_args": {"rom_path": None}}

    bind_vizdoom_iwad_to_document(document, binding)

    runtime_binding = {key: value for key, value in binding.items() if key != "object_uri"}
    assert document["train_config"]["env_args"]["rom_path"] == runtime_binding
    assert (
        document["train_config"]["checkpoint_eval_environment"]["env_args"]["rom_path"]
        == runtime_binding
    )
