from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gradlab.cli_parser import ExactArgumentParser
from gradlab.dotenv import load_env_file
from gradlab.rom_assets import (
    DEFAULT_LOCAL_ROM_CACHE,
    cache_path,
    load_rom_asset_state,
    rom_asset_manifest_for_game,
    sync_rom_asset,
)

ROM_IMPORT_DIR_ENV = "GRADLAB_ROM_IMPORT_DIR"


class RomImportPathError(ValueError):
    pass


def add_import_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "rom_path",
        nargs="?",
        help=f"Directory containing ROMs. Defaults to ${ROM_IMPORT_DIR_ENV} when set.",
    )
    parser.add_argument(
        "--game",
        help="Optional Stable Retro game id to verify after import.",
    )


def resolve_import_path(
    value: str | Path | None,
    *,
    environment: Mapping[str, str] | None = None,
) -> Path:
    values = os.environ if environment is None else environment
    if value is not None:
        raw = str(value).strip()
    else:
        raw = str(values.get(ROM_IMPORT_DIR_ENV) or "").strip()
    if not raw:
        raise RomImportPathError(
            f"ROM import path is required; pass PATH or set {ROM_IMPORT_DIR_ENV}"
        )
    return Path(raw).expanduser()


def cmd_import(args: argparse.Namespace) -> int:
    rom_path = resolve_import_path(args.rom_path)
    from stable_retro.scripts.import_path import main as stable_retro_import

    previous_argv = sys.argv
    sys.argv = ["stable_retro.import", str(rom_path)]
    try:
        stable_retro_import()
    finally:
        sys.argv = previous_argv

    if args.game:
        import stable_retro as retro

        imported = retro.data.get_romfile_path(args.game)
        print(f"{args.game} imported at {imported}")
    else:
        print(f"ROM import finished from {rom_path}")
    return 0


def _local_cache_status(manifest: dict[str, Any]) -> dict[str, Any]:
    path = cache_path(DEFAULT_LOCAL_ROM_CACHE, manifest)
    from gradlab.rom_assets import verify_rom_file

    try:
        verify_rom_file(path, manifest)
    except FileNotFoundError:
        return {"status": "missing", "path": str(path)}
    except ValueError as exc:
        return {"status": "corrupt", "path": str(path), "detail": str(exc)}
    return {"status": "hit", "path": str(path)}


def _authority_status(game: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    try:
        manifest = rom_asset_manifest_for_game(game)
    except Exception as exc:
        return None, {"status": "unhealthy", "detail": f"{type(exc).__name__}: {exc}"}
    return manifest, {
        "status": "healthy",
        "object_uri": manifest["object_uri"],
        "sha256": manifest["sha256"],
    }


def cmd_sync(args: argparse.Namespace) -> int:
    manifest = sync_rom_asset(
        args.game,
        rom_path=args.rom_path,
        source_dir=args.source_dir,
        replace=bool(args.replace),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    games = [args.game] if args.game else sorted(load_rom_asset_state().get("games", {}))
    if not games:
        print("no ROM assets are registered locally; pass --game", file=sys.stderr)
        return 1
    rows = []
    healthy = True
    for game in games:
        manifest, authority = _authority_status(game)
        row: dict[str, Any] = {"game": game, "authority": authority, "caches": {}}
        if manifest is None:
            healthy = False
            rows.append(row)
            continue
        row["manifest"] = manifest
        status = _local_cache_status(manifest)
        row["caches"]["local"] = status
        healthy = healthy and status["status"] == "hit"
        rows.append(row)
    payload = {"schema_version": 1, "healthy": healthy, "games": rows}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for row in rows:
            cache_text = ", ".join(
                f"{target}={value['status']}" for target, value in row["caches"].items()
            )
            print(f"{row['game']}: authority={row['authority']['status']} {cache_text}".rstrip())
    return 0 if healthy else 1


def build_parser() -> argparse.ArgumentParser:
    parser = ExactArgumentParser(prog="gradlab rom")
    commands = parser.add_subparsers(dest="command", required=True)
    import_parser = commands.add_parser(
        "import",
        help="Import ROMs into the installed Stable Retro runtime.",
    )
    add_import_args(import_parser)
    import_parser.set_defaults(func=cmd_import)

    sync = commands.add_parser("sync", help="Register and cache one immutable ROM asset.")
    sync.add_argument("--game", required=True)
    source = sync.add_mutually_exclusive_group()
    source.add_argument("--rom-path", type=Path)
    source.add_argument("--source-dir", type=Path, default=Path("~/roms"))
    sync.add_argument("--replace", action="store_true")
    sync.set_defaults(func=cmd_sync)

    status = commands.add_parser("status", help="Inspect registered ROM assets and local caches.")
    status.add_argument("--game")
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "import":
        load_env_file()
    try:
        return int(args.func(args))
    except RomImportPathError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
