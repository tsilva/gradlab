import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gradlab.dataset_cli import main
from gradlab.file_utils import file_sha256
from gradlab.json_utils import canonical_json_bytes
from gradlab.trajectory_dataset import LEGACY_FORMAT, migrate_snapshot, verify_snapshot
from gradlab.trajectory_export import Converter, FORMAT, publish_trajectories
from gradlab.trajectory_format import (
    TRAJECTORY_SCHEMA_VERSION,
    open_trajectory_parquet,
    require_current_trajectory_schema,
    table_schema,
    trajectory_schema_identity,
)
from gradlab.dataset_shards import PublicationBudget
from tests.test_trajectory_export import recording


def legacy_snapshot(root):
    _, models, episodes, _ = recording()
    converter = Converter(root / "work", {})
    converter.episode(1, episodes[0], models)
    output = root / "source"
    stats = converter.tables(output)
    converter.db.close()
    for path in output.rglob("*.parquet"):
        table = pq.read_table(path)
        metadata = {
            key: value
            for key, value in (table.schema.metadata or {}).items()
            if not key.startswith(b"gradlab.trajectory_")
        }
        pq.write_table(table.replace_schema_metadata(metadata), path)
    (output / "publication.json").write_bytes(
        canonical_json_bytes(
            {
                "format": FORMAT,
                "hud_mask": None,
                "statistics": stats,
                "source_revision": "a" * 40,
            }
        )
    )
    (output / "episodes.jsonl.gz").write_bytes(b"preserved canonical episode index")
    return output


def migrate(source, output):
    return migrate_snapshot(source, output, source_revision="b" * 40, source_format=LEGACY_FORMAT)


def test_schema_v1_fingerprint_is_frozen():
    # Changing a type, nullability, field meaning or join requires a version bump.
    assert TRAJECTORY_SCHEMA_VERSION == 1
    assert trajectory_schema_identity()["trajectory_schema_sha256"] == (
        "3af3a3bf898395756eb357517276785695898d3b6f4cc584b59e971dc4d60bd6"
    )


@pytest.mark.parametrize("version", [None, True, False, "1", 1.0, 0, 2, -1])
def test_strict_manifest_version(version):
    with pytest.raises(ValueError, match="schema version"):
        require_current_trajectory_schema(
            {**trajectory_schema_identity(), "trajectory_schema_version": version}
        )


def test_fingerprint_rejects_changed_semantics():
    with pytest.raises(ValueError, match="fingerprint"):
        require_current_trajectory_schema(
            {**trajectory_schema_identity(), "trajectory_schema_sha256": "0" * 64}
        )


@pytest.mark.parametrize("kind", ["transitions", "episodes", "frames", "sessions"])
def test_parquet_roundtrip_and_wrong_table_identity(tmp_path, kind):
    path = tmp_path / "table.parquet"
    pq.write_table(pa.Table.from_pylist([], schema=table_schema(kind)), path)
    with open_trajectory_parquet(path, kind) as parquet:
        assert parquet.metadata.num_rows == 0
    other = "frames" if kind != "frames" else "sessions"
    with pytest.raises(ValueError, match="metadata"):
        open_trajectory_parquet(path, other)


def test_physical_schema_cannot_hide_behind_correct_version(tmp_path):
    expected = table_schema("transitions")
    wrong = expected.set(0, pa.field("episode_id", pa.string()))
    path = tmp_path / "bad.parquet"
    pq.write_table(pa.Table.from_pylist([], schema=wrong), path)
    with pytest.raises(ValueError, match="physical table schema"):
        open_trajectory_parquet(path, "transitions")


def test_migration_preserves_values_sources_and_provenance(tmp_path, capsys):
    source = legacy_snapshot(tmp_path)
    original = {p.relative_to(source): file_sha256(p) for p in source.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="schema version"):
        verify_snapshot(source)
    output = tmp_path / "migrated"
    assert (
        main(
            [
                "migrate-trajectories",
                str(source),
                str(output),
                "--source-revision",
                "b" * 40,
                "--source-format",
                LEGACY_FORMAT,
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["rows"] == {"transitions": 1, "episodes": 1, "sessions": 1, "frames": 1}
    assert main(["verify-trajectories", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["trajectory_schema_version"] == 1
    assert original == {
        p.relative_to(source): file_sha256(p) for p in source.rglob("*") if p.is_file()
    }
    for relative in original:
        if relative.suffix == ".parquet":
            assert pq.read_table(source / relative).equals(
                pq.read_table(output / relative), check_metadata=False
            )
    assert (output / "episodes.jsonl.gz").read_bytes() == (
        source / "episodes.jsonl.gz"
    ).read_bytes()
    receipt = json.loads((output / "publication.json").read_bytes())
    assert receipt["migration"]["source_revision"] == "b" * 40
    assert receipt["source_revision"] == "a" * 40
    assert receipt["migration"]["source_files"] == {str(k): v for k, v in original.items()}
    assert all(
        item["values_unchanged"] for item in receipt["migration"]["validated_values"].values()
    )


def test_migration_refuses_overwrite_unknown_source_and_mixed_tables(tmp_path):
    source = legacy_snapshot(tmp_path)
    with pytest.raises(ValueError, match="must not exist"):
        migrate(source, source)
    with pytest.raises(ValueError, match="outside"):
        migrate(source, source / "new")
    with pytest.raises(ValueError, match="immutable"):
        migrate_snapshot(
            source, tmp_path / "out", source_revision="main", source_format=LEGACY_FORMAT
        )
    with pytest.raises(ValueError, match="No registered migration"):
        migrate_snapshot(source, tmp_path / "out", source_revision="b" * 40, source_format="v99")
    path = next((source / "transitions").rglob("*.parquet"))
    table = pq.read_table(path)
    pq.write_table(table.replace_schema_metadata(table_schema("transitions").metadata), path)
    with pytest.raises(ValueError, match="mixed tables"):
        migrate(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_snapshot_detects_corrupt_shard_and_migration_cleans_failed_output(tmp_path):
    source = legacy_snapshot(tmp_path)
    receipt = json.loads((source / "publication.json").read_bytes())
    receipt["statistics"]["transitions"] = 2
    (source / "publication.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="statistics"):
        migrate(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".trajectory-migration-*"))
    receipt["statistics"]["transitions"] = 1
    (source / "publication.json").write_text(json.dumps(receipt))
    migrate(source, tmp_path / "out")
    path = next((tmp_path / "out" / "episodes").rglob("*.parquet"))
    table = pq.read_table(path)
    index = table.schema.get_field_index("seed")
    pq.write_table(table.set_column(index, "seed", pa.array([42], type=pa.int64())), path)
    with pytest.raises(ValueError, match="checksums"):
        verify_snapshot(tmp_path / "out")


def test_resume_rejects_unversioned_shards(tmp_path):
    _, models, episodes, _ = recording()
    c = Converter(tmp_path, {})
    c.episode(1, episodes[0], models)
    path = tmp_path / "episode-000001.parquet"
    pq.write_table(pq.read_table(path).replace_schema_metadata(None), path)
    with pytest.raises(ValueError, match="metadata"):
        c.tables(tmp_path / "output")
    c.db.close()


@pytest.mark.parametrize("version", [None, 2])
def test_append_rejects_incompatible_remote_before_loading_images(tmp_path, version):
    api, models, episodes, _ = recording()
    contract = {"format": "test"}
    api.files["checkpoint-dataset.json"] = json.dumps(contract).encode()
    api.files["trajectory-view.json"] = json.dumps(
        {
            **trajectory_schema_identity(),
            "trajectory_schema_version": version,
            "episode_index": "old/index",
            "publication": "old/receipt",
        }
    ).encode()

    def forbidden(key):
        pytest.fail("incompatible dataset must fail before reading images")

    models.get_bytes = forbidden
    with pytest.raises(ValueError, match="schema version"):
        publish_trajectories(
            api,
            api.read,
            "test/data",
            contract,
            episodes,
            {},
            models,
            tmp_path,
            PublicationBudget(tmp_path / "budget"),
            lambda: False,
        )
    assert api.commits == []
