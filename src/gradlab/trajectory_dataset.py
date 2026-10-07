"""Local trajectory snapshot verification and explicit, non-destructive migrations."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

from gradlab.file_utils import file_sha256
from gradlab.json_utils import canonical_json_bytes, canonical_json_sha256
from gradlab.trajectory_format import (
    TRAJECTORY_FORMAT,
    TRAJECTORY_SCHEMA_VERSION,
    open_trajectory_parquet,
    require_current_trajectory_schema,
    table_schema,
    trajectory_schema_contract,
    trajectory_schema_identity,
)

TABLES = ("transitions", "episodes", "frames", "sessions")
LEGACY_FORMAT = "gradlab.trajectories.webp.v1-unversioned"
MIGRATIONS = {(LEGACY_FORMAT, 1): "gradlab.trajectories.adopt-webp-v1.1"}


def snapshot_files(root):
    """Only accept one self-contained snapshot, not an entire Hub history tree."""
    root = Path(root).resolve()
    files = {name: sorted((root / name).glob("*/*.parquet")) for name in TABLES}
    if any(not paths for paths in files.values()):
        raise ValueError("Snapshot requires transitions, episodes, frames and sessions tables")
    expected = {path for paths in files.values() for path in paths}
    if set(root.rglob("*.parquet")) != expected:
        raise ValueError("Snapshot contains unexpected Parquet files")
    for path in root.rglob("*"):
        # HF cache symlinks may be read, but their targets must be ordinary files.
        if path.is_symlink() and (path.is_dir() or not path.is_file()):
            raise ValueError(f"Unsupported snapshot symlink: {path}")
        if not path.is_dir() and not path.is_file():
            raise ValueError(f"Unsupported snapshot entry: {path}")
    return files


def table_inventory(root):
    """Verify physical schemas and record immutable file identities without loading rows."""
    root = Path(root)
    result = {}
    for name, paths in snapshot_files(root).items():
        for path in paths:
            with open_trajectory_parquet(path, name) as parquet:
                result[path.relative_to(root.resolve()).as_posix()] = {
                    "table": name,
                    "rows": parquet.metadata.num_rows,
                    "sha256": file_sha256(path),
                }
    return result


def verify_snapshot(root):
    root = Path(root)
    receipt = json.loads((root / "publication.json").read_bytes())
    require_current_trajectory_schema(receipt)
    if receipt.get("format") != TRAJECTORY_FORMAT:
        raise ValueError("Unsupported trajectory export format")
    schema = json.loads((root / "schema.json").read_bytes())
    if schema != trajectory_schema_contract():
        raise ValueError("Snapshot schema definition differs from the current contract")
    inventory = table_inventory(root)
    if receipt.get("tables") != inventory:
        raise ValueError("Snapshot table inventory, row counts or checksums differ")
    totals = {
        kind: sum(item["rows"] for item in inventory.values() if item["table"] == kind)
        for kind in TABLES
    }
    statistics = receipt.get("statistics", {})
    if any(
        statistics.get(key) != totals[kind]
        for key, kind in (
            ("episodes", "episodes"),
            ("transitions", "transitions"),
            ("unique_frames", "frames"),
        )
    ):
        raise ValueError("Snapshot statistics differ from table row counts")
    return {**trajectory_schema_identity(), "rows": totals, "files": len(inventory)}


def _verify_values(source, target):
    """Compare decoded values, including media bytes, in bounded batches."""
    with pq.ParquetFile(source) as original, pq.ParquetFile(target) as converted:
        if original.metadata.num_rows != converted.metadata.num_rows:
            raise ValueError(f"Migration changed row count: {source.name}")
        for before, after in zip(
            original.iter_batches(batch_size=512),
            converted.iter_batches(batch_size=512),
            strict=True,
        ):
            if not before.equals(after, check_metadata=False):
                raise ValueError(f"Migration changed values: {source.name}")
        return {"rows": original.metadata.num_rows, "values_unchanged": True}


def migrate_snapshot(source, output, *, source_revision, source_format):
    """Adopt verified legacy WebP tables; write a new snapshot, preserving all values.

    A versionless dataset is accepted only here, with an explicit source format
    and immutable Hub commit. Future migrations must register a named edge and
    implement its own validation/transformation rather than relabeling tables.
    """
    source, output = Path(source).resolve(), Path(output).absolute()
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ValueError("Migration requires a full immutable HF source revision")
    migration = MIGRATIONS.get((source_format, TRAJECTORY_SCHEMA_VERSION))
    if migration is None:
        raise ValueError(
            f"No registered migration from {source_format!r} to schema {TRAJECTORY_SCHEMA_VERSION}"
        )
    if output.exists() or output.is_symlink():
        raise ValueError("Migration output must not exist; source snapshots are never overwritten")
    if output.resolve().is_relative_to(source):
        raise ValueError("Migration output must be outside the source snapshot")
    receipt = json.loads((source / "publication.json").read_bytes())
    if (
        receipt.get("format") != TRAJECTORY_FORMAT
        or "hud_mask" not in receipt
        or receipt["hud_mask"] is not None
    ):
        raise ValueError("Migration requires an unmasked legacy WebP publication receipt")
    if any(key.startswith("trajectory_schema_") for key in receipt):
        raise ValueError(
            "Source already declares a schema; this migration accepts only unversioned data"
        )
    files = snapshot_files(source)
    # Validate every table before creating output or accepting any legacy data.
    for name, paths in files.items():
        for path in paths:
            actual = pq.read_schema(path)
            if any(key.startswith(b"gradlab.trajectory_") for key in (actual.metadata or {})):
                raise ValueError("Legacy source contains already-versioned or mixed tables")
            if not actual.equals(table_schema(name), check_metadata=False):
                raise ValueError(
                    f"{name}: legacy physical schema does not match the registered migration"
                )
            if name == "frames" and actual.metadata != {
                b"huggingface": table_schema(name).metadata[b"huggingface"]
            }:
                raise ValueError("Legacy frames have incompatible Image metadata")
    source_files = {
        p.relative_to(source).as_posix(): file_sha256(p)
        for p in sorted(source.rglob("*"))
        if p.is_file()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".trajectory-migration-", dir=output.parent))
    try:
        # Copy ancillary provenance, including the canonical episode index.
        for name in source_files:
            if not name.endswith(".parquet"):
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / name, target)
        validated = {}
        for kind, paths in files.items():
            for path in paths:
                name = path.relative_to(source).as_posix()
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with (
                    pq.ParquetFile(path) as reader,
                    pq.ParquetWriter(target, table_schema(kind), compression="zstd") as writer,
                ):
                    for batch in reader.iter_batches(batch_size=512):
                        writer.write_batch(
                            batch.replace_schema_metadata(table_schema(kind).metadata)
                        )
                validated[name] = _verify_values(path, target)
        (staging / "schema.json").write_bytes(canonical_json_bytes(trajectory_schema_contract()))
        migration_receipt = {
            "migration_id": migration,
            "source_format": source_format,
            "source_revision": source_revision,
            "source_files": source_files,
            "target": trajectory_schema_identity(),
            "validated_values": validated,
        }
        migrated = {
            **receipt,
            **trajectory_schema_identity(),
            "tables": table_inventory(staging),
            "migration": migration_receipt,
            "migration_sha256": canonical_json_sha256(migration_receipt),
        }
        (staging / "publication.json").write_bytes(canonical_json_bytes(migrated))
        result = verify_snapshot(staging)
        # Reject a source that changed while being read/copied.
        if source_files != {
            p.relative_to(source).as_posix(): file_sha256(p)
            for p in sorted(source.rglob("*"))
            if p.is_file()
        }:
            raise ValueError("Migration source changed during conversion")
        if output.exists() or output.is_symlink():
            raise ValueError("Migration output appeared during conversion")
        staging.rename(output)
        return {**result, "output": str(output), "migration_id": migration}
    finally:
        if staging.exists():
            shutil.rmtree(staging)
