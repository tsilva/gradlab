import hashlib
from pathlib import Path

import pytest
from huggingface_hub.errors import EntryNotFoundError

from gradlab.dataset_split_publication import SplitPublicationHandler, JOB_TYPE
from gradlab.job_queue import JobStore, JobSubject, run_flusher
from tests.test_trajectory_publication import Hub


def prepare(tmp_path, monkeypatch):
    root = tmp_path / "upload"
    (root / "splits/test").mkdir(parents=True)
    (root / "splits/test/manifest.json").write_bytes(b'{"verified":true}')
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
