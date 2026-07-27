from __future__ import annotations

import argparse

from gradlab.rom_cli import RomImportPathError, build_import_parser, cmd_import


def build_parser() -> argparse.ArgumentParser:
    return build_import_parser(prog="gradlab import-roms")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return cmd_import(args)
    except RomImportPathError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
