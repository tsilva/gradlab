"""Create a verified online SQLite snapshot for the private MLflow pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from pathlib import Path


def backup(source: Path, destination: Path) -> dict[str, object]:
    if not source.is_file() or source.is_symlink():
        raise ValueError("source must be an existing regular SQLite file")
    if destination.exists() or destination.is_symlink():
        raise ValueError("snapshot destination must not exist")
    if destination.with_suffix(destination.suffix + ".json").exists():
        raise ValueError("snapshot receipt already exists")
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        raise ValueError("partial snapshot already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as input_db:
            with sqlite3.connect(temporary) as output_db:
                input_db.backup(output_db)
                if output_db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise ValueError("SQLite snapshot integrity check failed")
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        size = temporary.stat().st_size
        os.replace(temporary, destination)
        receipt: dict[str, object] = {
            "schema_version": 1,
            "source_name": source.name,
            "snapshot_name": destination.name,
            "sha256": digest,
            "size_bytes": size,
        }
        destination.with_suffix(destination.suffix + ".json").write_text(
            json.dumps(receipt, sort_keys=True, indent=2) + "\n"
        )
        return receipt
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("backup",))
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    receipt = backup(args.source, args.destination)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
