import hashlib
import json
from pathlib import Path

import pytest
from huggingface_hub.errors import EntryNotFoundError

from gradlab.dataset_split_publication import SplitPublicationHandler, JOB_TYPE
from gradlab.job_queue import JobStore, JobSubject, run_flusher
from tests.test_trajectory_publication import Hub
from gradlab.trajectory_format import trajectory_schema_identity, table_schema
import pyarrow as pa
import pyarrow.parquet as pq


def prepare(tmp_path, monkeypatch):
    root = tmp_path / "upload"
    (root / "splits/test").mkdir(parents=True)
    tables = {}
    for kind in ("transitions", "episodes"):
        name = f"splits/test/{kind}/train/00000.parquet"
        path = root / name
        path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([], schema=table_schema(kind)), path)
        tables[name] = {
            "table": kind,
            "rows": 0,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (root / "splits/test/manifest.json").write_text(
        json.dumps(
            {
                **trajectory_schema_identity(),
                "source_revision": "0" * 40,
                "tables": tables,
            }
        )
    )
    (root / "README.md").write_bytes(b"new split view")
    queue = JobStore(tmp_path / "queue")
    queue.init()
    payload = dict(
        repo="test/data",
        parent="0" * 40,
        root=str(root),
        queue_root=str(queue.root),
        repo_root=str(tmp_path),
        receipt="splits/test/manifest.json",
        files={
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*")
            if p.is_file()
        },
    )
    hub = Hub()
    hub.head = "0" * 40
    hub.files["README.md"] = b"old"
    hub.files["frames/original.parquet"] = b"preserved"
    hub.files["trajectory-view.json"] = json.dumps(
        {
            **trajectory_schema_identity(),
            "publication": "trajectories/base/publication.json",
        }
    ).encode()
    hub.files["trajectories/base/publication.json"] = json.dumps(
        trajectory_schema_identity()
    ).encode()
    monkeypatch.setattr("gradlab.dataset_split_publication.HfApi", lambda: hub)
    monkeypatch.setattr(
        "gradlab.operator_environment.load_repository_operator_environment", lambda root: None
    )

    def download(repo, name, revision, **kwargs):
        data = hub.read(name, revision)
        if data is None:
            raise EntryNotFoundError("missing")
        p = tmp_path / "remote-receipt"
        p.write_bytes(data)
        return str(p)

    monkeypatch.setattr("gradlab.dataset_split_publication.hf_hub_download", download)
    return queue, payload, hub


def test_durable_atomic_split_commit_reconciles_lost_ack(tmp_path, monkeypatch):
    queue, payload, hub = prepare(tmp_path, monkeypatch)
    hub.lose_reply = True
    result = queue.enqueue(
        job_type=JOB_TYPE,
        handler_version=1,
        payload=payload,
        idempotency_key="split-test",
        subjects=[JobSubject("dataset", "test/data")],
    )
    assert run_flusher(queue, idle_seconds=0) == 0
    assert queue.job(result.job["job_id"])["state"] == "succeeded"
    assert hub.files["README.md"] == b"new split view"
    assert hub.files["frames/original.parquet"] == b"preserved"
    assert hub.head == "1"


def test_reject_changed_parent_and_prepared_bytes(tmp_path, monkeypatch):
    queue, payload, hub = prepare(tmp_path, monkeypatch)
    hub.head = "1" * 40
    with pytest.raises(ValueError, match="head changed"):
        SplitPublicationHandler().publish(payload, "unused")
    (Path(payload["root"]) / "README.md").write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        SplitPublicationHandler().publish(payload, "unused")
    assert hub.files["README.md"] == b"old"


def test_reject_path_escape(tmp_path, monkeypatch):
    _, payload, _ = prepare(tmp_path, monkeypatch)
    payload["files"]["splits/test/../../escape"] = "0" * 64
    with pytest.raises(ValueError, match="file identity"):
        SplitPublicationHandler.validate_payload(payload)


@pytest.mark.parametrize("failure", ["unversioned_manifest", "future_parent", "physical_schema"])
def test_schema_rejection_precedes_any_remote_upload(tmp_path, monkeypatch, failure):
    _, payload, hub = prepare(tmp_path, monkeypatch)
    root = Path(payload["root"])
    if failure == "unversioned_manifest":
        path = root / payload["receipt"]
        manifest = json.loads(path.read_bytes())
        del manifest["trajectory_schema_version"]
        path.write_text(json.dumps(manifest))
    elif failure == "future_parent":
        view = json.loads(hub.files["trajectory-view.json"])
        view["trajectory_schema_version"] = 999
        hub.files["trajectory-view.json"] = json.dumps(view).encode()
    else:
        path = root / "splits/test/transitions/train/00000.parquet"
        schema = table_schema("transitions").set(0, pa.field("episode_id", pa.string()))
        pq.write_table(pa.Table.from_pylist([], schema=schema), path)
    payload["files"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in payload["files"]
    }
    with pytest.raises(ValueError, match="schema"):
        SplitPublicationHandler().publish(payload, "unused")
    assert hub.uploaded == {}
    assert hub.head == "0" * 40
