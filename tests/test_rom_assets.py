from __future__ import annotations

import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from gradlab.env_identity import environment_identity_from_train_config
from gradlab.main import main as cli_main
from gradlab.rom_assets import (
    ROM_ASSET_IDENTITY_ALGORITHM,
    cache_path,
    direct_rom_asset_manifest,
    discover_rom_path,
    ensure_rom_cache,
    manifest_from_train_config,
    provider_rom_identity,
    rom_asset_manifest_for_game,
    sync_rom_asset,
    validate_rom_asset_manifest,
    verify_rom_file,
)
from gradlab.rom_cli import (
    ROM_IMPORT_DIR_ENV,
    RomImportPathError,
    build_parser,
    cmd_status,
    main as rom_main,
    resolve_import_path,
)


GAME = "SuperMarioBros-Nes-v0"


def _rom(path: Path, body: bytes) -> Path:
    path.write_bytes(b"NES\x1a" + bytes((1, 1)) + bytes(10) + body)
    return path


def _manifest(path: Path, *, object_uri: str | None = None) -> dict:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "schema_version": 2,
        "game": GAME,
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": digest,
        "object_uri": object_uri or path.resolve().as_uri(),
        "provider_rom_identity": provider_rom_identity(path),
        "provider_rom_identity_algorithm": ROM_ASSET_IDENTITY_ALGORITHM,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("filename", "../rom.nes", "safe basename"),
        ("sha256", "xyz", "64 lowercase"),
        ("provider_rom_identity", "xyz", "40 lowercase"),
        ("provider_rom_identity_algorithm", "sha1", "unsupported"),
        ("object_uri", "https://example.invalid/rom.nes", "s3:// or file://"),
        ("unexpected", True, "unknown ROM asset manifest field"),
    ),
)
def test_manifest_v2_validation_is_strict(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    manifest = _manifest(_rom(tmp_path / "rom.nes", b"one"))
    manifest[field] = value

    with pytest.raises(ValueError, match=message):
        validate_rom_asset_manifest(manifest)


def test_manifest_rejects_wrong_game(tmp_path: Path) -> None:
    manifest = _manifest(_rom(tmp_path / "rom.nes", b"one"))

    with pytest.raises(ValueError, match="game mismatch"):
        validate_rom_asset_manifest(manifest, expected_game="Other-Nes-v0")


def test_manifest_from_train_config_rejects_retired_schema(tmp_path: Path) -> None:
    current = _manifest(_rom(tmp_path / "rom.nes", b"one"))
    assert manifest_from_train_config(
        {"rom_asset_manifest": current},
        expected_game=GAME,
    )["sha256"] == current["sha256"]
    with pytest.raises(ValueError, match="unsupported.*schema_version"):
        manifest_from_train_config(
            {"rom_asset_manifest": {**current, "schema_version": 1}},
            expected_game=GAME,
        )


def test_discovery_ignores_duplicate_bytes_but_rejects_distinct_matches(tmp_path: Path) -> None:
    first = _rom(tmp_path / "one.nes", b"one")
    duplicate = tmp_path / "duplicate.nes"
    duplicate.write_bytes(first.read_bytes())
    second = _rom(tmp_path / "two.nes", b"two")
    identities = {
        first.resolve(): "a" * 40,
        duplicate.resolve(): "a" * 40,
        second.resolve(): "a" * 40,
    }

    with (
        patch("gradlab.rom_assets._expected_provider_identities", return_value={"a" * 40}),
        patch(
            "gradlab.rom_assets.provider_rom_identity",
            side_effect=lambda path: identities[path.resolve()],
        ),
        pytest.raises(ValueError, match="multiple distinct ROM files"),
    ):
        discover_rom_path(GAME, source_dir=tmp_path)

    second.unlink()
    with (
        patch("gradlab.rom_assets._expected_provider_identities", return_value={"a" * 40}),
        patch(
            "gradlab.rom_assets.provider_rom_identity",
            side_effect=lambda path: identities[path.resolve()],
        ),
    ):
        assert discover_rom_path(GAME, source_dir=tmp_path) == duplicate.resolve()


def test_direct_manifest_verifies_rom_without_mutating_cache_or_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rom = _rom(tmp_path / "mario.nes", b"one")
    state = tmp_path / "state.json"
    cache = tmp_path / "cache"
    monkeypatch.setenv("GRADLAB_ROM_ASSET_STATE", str(state))
    monkeypatch.setattr("gradlab.rom_assets.DEFAULT_LOCAL_ROM_CACHE", cache)

    with (
        patch(
            "gradlab.rom_assets._expected_provider_identities",
            return_value={"a" * 40},
        ),
        patch(
            "gradlab.rom_assets.provider_rom_identity",
            return_value="a" * 40,
        ),
    ):
        manifest = direct_rom_asset_manifest(GAME, rom)

    assert manifest["sha256"] == hashlib.sha256(rom.read_bytes()).hexdigest()
    assert manifest["size_bytes"] == rom.stat().st_size
    assert manifest["provider_rom_identity"] == "a" * 40
    assert manifest["object_uri"] == rom.resolve().as_uri()
    assert not state.exists()
    assert not cache.exists()


def test_direct_manifest_rejects_missing_incompatible_and_non_nes_roms(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="ROM path does not exist"):
        direct_rom_asset_manifest(GAME, tmp_path / "missing.nes")

    archive = tmp_path / "mario.zip"
    archive.write_bytes(b"not a raw ROM")
    with pytest.raises(ValueError, match=r"raw \.nes"):
        direct_rom_asset_manifest(GAME, archive)

    rom = _rom(tmp_path / "wrong.nes", b"wrong")
    with (
        patch(
            "gradlab.rom_assets._expected_provider_identities",
            return_value={"a" * 40},
        ),
        patch(
            "gradlab.rom_assets.provider_rom_identity",
            return_value="b" * 40,
        ),
        pytest.raises(ValueError, match="provider identity"),
    ):
        direct_rom_asset_manifest(GAME, rom)


def test_cache_repairs_corruption_from_local_source_and_then_reuses(
    tmp_path: Path,
) -> None:
    source = _rom(tmp_path / "rom.nes", b"one")
    manifest = _manifest(source)
    root = tmp_path / "cache"

    installed = ensure_rom_cache(manifest, cache_root=root)
    verify_rom_file(installed, manifest)
    installed.write_bytes(b"corrupt")
    repaired = ensure_rom_cache(manifest, cache_root=root)
    assert repaired.read_bytes() == source.read_bytes()
    source.unlink()
    assert ensure_rom_cache(manifest, cache_root=root) == repaired


def test_sync_pins_local_identity_and_requires_explicit_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _rom(tmp_path / "first.nes", b"one")
    second = _rom(tmp_path / "second.nes", b"two")
    monkeypatch.setenv("GRADLAB_ROM_ASSET_STATE", str(tmp_path / "state.json"))
    cache = tmp_path / "cache"

    with (
        patch("gradlab.rom_assets.discover_rom_path", return_value=first),
        patch("gradlab.rom_assets.DEFAULT_LOCAL_ROM_CACHE", cache),
    ):
        pinned = sync_rom_asset(
            GAME,
            local_cache_root=cache,
        )
        assert rom_asset_manifest_for_game(GAME) == pinned

    with (
        patch("gradlab.rom_assets.discover_rom_path", return_value=second),
        pytest.raises(ValueError, match="--replace"),
    ):
        sync_rom_asset(GAME, local_cache_root=cache)

    with (
        patch("gradlab.rom_assets.discover_rom_path", return_value=second),
        patch("gradlab.rom_assets.DEFAULT_LOCAL_ROM_CACHE", cache),
    ):
        replaced = sync_rom_asset(
            GAME,
            replace=True,
            local_cache_root=cache,
        )
        assert rom_asset_manifest_for_game(GAME) == replaced
    assert replaced["sha256"] != pinned["sha256"]


def test_rom_identity_changes_environment_hash_but_runtime_path_does_not(tmp_path: Path) -> None:
    first = _manifest(_rom(tmp_path / "one.nes", b"one"))
    second = _manifest(_rom(tmp_path / "two.nes", b"two"))
    base = {
        "env_provider": "stable-retro-turbo",
        "game": GAME,
        "state": "Level1-1",
        "task": {},
        "rom_asset_manifest": first,
    }
    first_identity = environment_identity_from_train_config(base)
    changed_path = environment_identity_from_train_config(
        {**base, "env_args": {"rom_path": "/different/cache/location.nes"}}
    )
    changed_rom = environment_identity_from_train_config(
        {**base, "rom_asset_manifest": second}
    )

    assert first_identity == changed_path
    assert first_identity != changed_rom
    assert cache_path(tmp_path / "cache", first).parts[-3:] == (
        "sha256",
        first["sha256"],
        first["filename"],
    )


def test_manifest_never_serializes_a_runtime_path(tmp_path: Path) -> None:
    normalized = validate_rom_asset_manifest(_manifest(_rom(tmp_path / "rom.nes", b"one")))
    assert "rom_path" not in json.dumps(normalized)


def test_status_exit_codes_and_default_scope(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    manifest = _manifest(_rom(tmp_path / "rom.nes", b"one"))
    args = Namespace(game=GAME, json=True)
    with (
        patch("gradlab.rom_cli.rom_asset_manifest_for_game", return_value=manifest),
        patch("gradlab.rom_cli._local_cache_status", return_value={"status": "hit"}) as local,
    ):
        assert cmd_status(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["healthy"] is True
    assert payload["games"][0]["caches"] == {"local": {"status": "hit"}}
    local.assert_called_once_with(manifest)

def test_status_rejects_removed_remote_target() -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["status", "--target", "modal"])
    assert exc.value.code == 2


def test_import_path_prefers_explicit_value_over_environment(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit"
    configured = tmp_path / "configured"

    assert resolve_import_path(
        explicit,
        environment={ROM_IMPORT_DIR_ENV: str(configured)},
    ) == explicit


def test_import_path_expands_home_for_argument_and_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    assert resolve_import_path("~/explicit", environment={}) == tmp_path / "explicit"
    assert resolve_import_path(None, environment={ROM_IMPORT_DIR_ENV: "~/roms"}) == (
        tmp_path / "roms"
    )


def test_import_path_requires_argument_or_environment() -> None:
    with pytest.raises(RomImportPathError, match=ROM_IMPORT_DIR_ENV):
        resolve_import_path(None, environment={})


def test_rom_import_uses_environment_without_loading_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(ROM_IMPORT_DIR_ENV, str(tmp_path))
    observed_argv: list[list[str]] = []

    def fake_import() -> None:
        observed_argv.append(list(sys.argv))

    with (
        patch("gradlab.rom_cli.load_env_file") as load_dotenv,
        patch("stable_retro.scripts.import_path.main", side_effect=fake_import),
    ):
        assert rom_main(["import"]) == 0

    load_dotenv.assert_not_called()
    assert observed_argv == [["stable_retro.import", str(tmp_path)]]
    assert f"ROM import finished from {tmp_path}" in capsys.readouterr().out


def test_rom_import_missing_path_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv(ROM_IMPORT_DIR_ENV, raising=False)

    with pytest.raises(SystemExit) as exc:
        rom_main(["import"])

    assert exc.value.code == 2
    assert ROM_IMPORT_DIR_ENV in capsys.readouterr().err


def test_hidden_import_roms_route_executes_shared_import_handler(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch("stable_retro.scripts.import_path.main") as stable_retro_import:
        assert cli_main(["import-roms", str(tmp_path)]) == 0

    stable_retro_import.assert_called_once_with()
    assert f"ROM import finished from {tmp_path}" in capsys.readouterr().out
