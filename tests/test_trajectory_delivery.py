import hashlib

import pytest

from gradlab.file_utils import atomic_write_json
from gradlab.r2_store import BucketConfig, R2Bucket
from gradlab.trajectory_delivery import DatasetDelivery
from gradlab.trajectory_config import FORMAT


def artifact(root):
    root.mkdir()
    data = b"representative immutable chunk"
    (root / "episode.zip").write_bytes(data)
    manifest = {
        "format": FORMAT,
        "file": "episode.zip",
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "reserved_bytes": 1024**2,
        "contract_sha256": "a" * 64,
        "episode": {"episode_id": "one", "run_id": "run", "attempt_id": "attempt"},
    }
    atomic_write_json(root / "episode.manifest.json", manifest)
    atomic_write_json(root / "producer.json", {"format": FORMAT, "contract": {}, "provenance": {}})
    atomic_write_json(root / "closed.json", {"chunks": 1, "reserved_bytes": 1024**2, "fault": None})
    return manifest


def test_delivery_verifies_manifest_before_reclaim_and_reconciles_repeat(tmp_path):
    root = tmp_path / "spool"
    expected = artifact(root)
    bucket = R2Bucket(BucketConfig(uri=(tmp_path / "r2").as_uri()))
    delivery = DatasetDelivery(root, bucket, "run", "attempt", 2 * 1024**2)
    delivery.advance(final=True)
    assert not (root / "episode.zip").exists()
    receipt = delivery.receipt()
    assert receipt["complete"]
    inventory = bucket.get_json(receipt["manifest_key"])
    assert inventory["chunks"][0]["sha256"] == expected["sha256"]
    assert bucket.get_bytes(inventory["chunks"][0]["key"]) == b"representative immutable chunk"
    delivery.advance(final=True)
    assert delivery.receipt() == receipt
    assert DatasetDelivery.reserved_bytes(bucket, "run") == 1024**2


def test_lost_ack_recovery_and_conflicting_remote_bytes(tmp_path):
    root = tmp_path / "spool"
    artifact(root)
    bucket = R2Bucket(BucketConfig(uri=(tmp_path / "r2").as_uri()))
    delivery = DatasetDelivery(root, bucket, "run", "attempt", 2 * 1024**2)
    delivery.advance(final=True)
    # Recreate the exact local pending copy after a lost local acknowledgement.
    (root / "episode.zip").write_bytes(b"representative immutable chunk")
    delivery = DatasetDelivery(root, bucket, "run", "attempt", 2 * 1024**2)
    delivery.advance(final=True)
    assert delivery.receipt()["complete"]
    inventory = bucket.get_json(delivery.receipt()["manifest_key"])
    key = inventory["chunks"][0]["key"]
    bucket.put_bytes(key, b"corrupt", create_only=False)
    delivery = DatasetDelivery(root, bucket, "run", "attempt", 2 * 1024**2)
    with pytest.raises(ValueError, match="checksum"):
        delivery.advance(final=True)
