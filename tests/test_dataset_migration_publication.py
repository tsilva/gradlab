import json
from pathlib import Path
import shutil

import pytest
import pyarrow.parquet as pq
import yaml
from huggingface_hub.errors import EntryNotFoundError

from gradlab.dataset_migration_publication import JOB_TYPE, MigrationPublicationHandler
from gradlab.file_utils import file_sha256
from gradlab.job_queue import JobStore, JobSubject, run_flusher
from gradlab.trajectory_format import trajectory_schema_identity
from tests.test_trajectory_schema import legacy_snapshot, migrate
from tests.test_trajectory_publication import Hub


def prepare(tmp_path, monkeypatch):
    source = legacy_snapshot(tmp_path)
    root = tmp_path / "upload"
    identity = "c" * 64
    base, split = f"trajectories/{identity}", f"splits/{identity}"
    migrate(source, root / base)
    queue = JobStore(tmp_path / "queue")
    queue.init()
    tables = {}
    for kind in ("episodes", "transitions"):
        name = f"{split}/{kind}/train/00000.parquet"
        target = root / name
        target.parent.mkdir(parents=True)
        shutil.copyfile(root / base / kind / "all/00000.parquet", target)
        tables[name] = {
            "table": kind,
            "rows": pq.read_metadata(target).num_rows,
            "sha256": file_sha256(target),
        }
    (root / split / "manifest.json").write_text(
        json.dumps(
            {
                **trajectory_schema_identity(),
                "source_revision": "b" * 40,
                "source_publication": f"{base}/publication.json",
                "tables": tables,
                "statistics": {"train": {"episodes": 1, "transitions": 1}},
            }
        )
    )
    (root / "trajectory-view.json").write_text(
        json.dumps(
            {
                **trajectory_schema_identity(),
                "publication": f"{base}/publication.json",
                "episode_index": f"{base}/episodes.jsonl.gz",
            }
        )
    )
    configs = []
    for kind, prefix, partition in (
        ("transitions", split, "train"),
        ("episodes", split, "train"),
        ("frames", base, "assets"),
        ("sessions", base, "metadata"),
    ):
        configs.append(
            {
                "config_name": kind,
                "data_files": [
                    {"split": partition, "path": f"{prefix}/{kind}/{partition}/*.parquet"}
                ],
            }
        )
    (root / "README.md").write_text(
        "---\n" + yaml.safe_dump({"configs": configs}) + "---\nMigrated\n"
    )
    files = {p.relative_to(root).as_posix(): file_sha256(p) for p in root.rglob("*") if p.is_file()}
    receipt = f"migrations/{identity}/manifest.json"
    (root / receipt).parent.mkdir(parents=True)
    (root / receipt).write_text(
        json.dumps(
            {
                **trajectory_schema_identity(),
                "source_repo": "test/data",
                "source_revision": "b" * 40,
                "files": files,
            }
        )
    )
    payload = dict(
        repo="test/data",
        parent="b" * 40,
        root=str(root),
        queue_root=str(queue.root),
        repo_root=str(tmp_path),
        files={**files, receipt: file_sha256(root / receipt)},
        receipt=receipt,
    )
    hub = Hub()
    hub.head = "b" * 40
    hub.files = {
        "README.md": b"old card",
        "trajectory-view.json": b"old view",
        "old.parquet": b"old data",
    }
    hub.list_repo_files = lambda *args, **kwargs: list(hub.files)
    original_commit = hub.create_commit

    def commit(**kwargs):
        # The shared test adapter uses numeric revision counters.
        assert kwargs["parent_commit"] == hub.head
        hub.head = "0"
        return original_commit(**{**kwargs, "parent_commit": "0"})

    hub.create_commit = commit
    monkeypatch.setattr("gradlab.dataset_migration_publication.HfApi", lambda: hub)
    monkeypatch.setattr(
        "gradlab.operator_environment.load_repository_operator_environment", lambda root: None
    )

    def download(repo, name, revision, **kwargs):
        value = hub.read(name, revision)
        if value is None:
            raise EntryNotFoundError("missing")
        path = tmp_path / "downloaded-receipt"
        path.write_bytes(value)
        return str(path)

    monkeypatch.setattr("gradlab.dataset_migration_publication.hf_hub_download", download)
    return queue, payload, hub


def test_durable_atomic_migration_preserves_history_and_reconciles_lost_ack(tmp_path, monkeypatch):
    queue, payload, hub = prepare(tmp_path, monkeypatch)
    hub.lose_reply = True
    result = queue.enqueue(
        job_type=JOB_TYPE,
        handler_version=1,
        payload=payload,
        idempotency_key="migration",
        subjects=[JobSubject("dataset", "test/data")],
    )
    assert run_flusher(queue, idle_seconds=0) == 0
    assert queue.job(result.job["job_id"])["state"] == "succeeded"
    assert hub.files["old.parquet"] == b"old data"
    assert hub.files["README.md"].endswith(b"Migrated\n")
    assert MigrationPublicationHandler().publish(payload, "unused") == "1"


@pytest.mark.parametrize("failure", ["parent", "bytes", "immutable_collision"])
def test_migration_preflight_prevents_remote_writes(tmp_path, monkeypatch, failure):
    _, payload, hub = prepare(tmp_path, monkeypatch)
    if failure == "parent":
        hub.head = "d" * 40
        message = "head changed"
    elif failure == "bytes":
        (Path(payload["root"]) / "README.md").write_text("changed")
        message = "bytes changed"
    else:
        name = next(name for name in payload["files"] if name.endswith(".parquet"))
        hub.files[name] = b"existing immutable data"
        message = "overwrite immutable"
    with pytest.raises(ValueError, match=message):
        MigrationPublicationHandler().publish(payload, "unused")
    assert hub.uploaded == {}
    assert hub.files["README.md"] == b"old card"
