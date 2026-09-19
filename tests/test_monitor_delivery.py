from concurrent.futures import ThreadPoolExecutor
import json
from threading import Barrier, Event

import pytest

from gradlab.monitor_worker import ContributionBudget
from gradlab.r2_store import BucketConfig, R2Bucket


def test_reservations_do_not_hold_shared_lock_during_remote_verification(tmp_path):
    bucket = R2Bucket(BucketConfig(uri=f"file://{tmp_path}/r2"))
    original = bucket.put_bytes
    overlap = Barrier(2)

    def delayed_put(key, payload, **kwargs):
        overlap.wait(timeout=2)
        return original(key, payload, **kwargs)

    bucket.put_bytes = delayed_put
    budgets = [ContributionBudget(tmp_path / "budget", bucket, "run", 10000) for _ in range(2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(b.reserve, f"chunk-{i}", 100) for i, b in enumerate(budgets)]
        for future in futures:
            future.result()
    assert len(list(bucket.iter_keys("monitoring/run/budget"))) == 2
    assert len(json.loads(budgets[0].cache.read_text())["entries"]) == 2


def test_delivery_overlaps_and_waits_for_dependencies():
    from gradlab.monitor_delivery import MonitoringDelivery

    started = Barrier(3)
    release = Event()
    committed = []

    def upload():
        started.wait(timeout=2)
        assert release.wait(timeout=2)

    with MonitoringDelivery() as delivery:
        first = delivery.submit(upload)
        second = delivery.submit(upload)
        delivery.submit(lambda: committed.append(True), after=(first, second))
        started.wait(timeout=2)
        assert not committed
        release.set()
    assert committed == [True]


def test_delivery_never_commits_after_failed_dependency():
    from gradlab.monitor_delivery import MonitoringDelivery

    committed = []
    release = Event()

    def fail():
        assert release.wait(timeout=2)
        raise OSError("remote verification failed")

    with pytest.raises(OSError, match="verification"):
        with MonitoringDelivery() as delivery:
            upload = delivery.submit(fail)
            delivery.submit(lambda: committed.append(True), after=(upload,))
            release.set()
    assert committed == []


def test_native_episodes_overlap_delivery_without_publishing_unverified_manifests(tmp_path):
    import numpy as np
    from pathlib import Path
    from gradlab.checkpoint_monitoring import monitor_episode, episode_manifest, episode_frames
    from gradlab.env import resolve_env_config
    from gradlab.env_config import env_config_from_mapping
    from gradlab.monitor_delivery import MonitoringDelivery
    from gradlab.recipe_documents import compose_train_document

    goal = Path("experiments/goals/Breakout-Atari2600-v0/FirstWall")
    train = compose_train_document(goal / "_goal.yaml", goal / "recipes/ppo.yaml")["train_config"]
    config = resolve_env_config(env_config_from_mapping(train))
    config.task["termination"]["max_episode_steps"] = 3
    bucket = R2Bucket(BucketConfig(uri=f"file://{tmp_path}/r2"))
    original = bucket.put_bytes
    both_uploading = Barrier(3)
    release = Event()

    class Policy:
        def predict(self, observations, deterministic=False):
            return np.ones(1, dtype=np.int64), None

    def blocked_put(key, payload, **kwargs):
        if key.endswith(".zip"):
            both_uploading.wait(timeout=10)
            assert release.wait(timeout=10)
        return original(key, payload, **kwargs)

    bucket.put_bytes = blocked_put

    def record():
        episodes = []
        with MonitoringDelivery() as delivery:
            for episode in episode_manifest(2):
                episodes.append(monitor_episode(
                    model=Policy(), config=config, episode=episode, bucket=bucket,
                    root=tmp_path / "spool" / episode["episode_id"],
                    prefix=f'monitor/{episode["episode_id"]}',
                    provenance={"checkpoint_id": "checkpoint", "checkpoint_step": 123},
                    delivery=delivery,
                ))
        return episodes

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(record)
        try:
            both_uploading.wait(timeout=10)
            assert len(list((tmp_path / "spool").rglob("*.zip"))) == 2
            assert not list(bucket.iter_keys("monitor"))
        finally:
            release.set()
        episodes = future.result(timeout=10)
    for episode in episodes:
        assert bucket.get_json(f'monitor/{episode["episode_id"]}/episode.json') == episode
        assert len(list(episode_frames(bucket, episode))) == 4
    assert not list((tmp_path / "spool").rglob("*.zip"))


def test_failed_remote_reservation_remains_charged_and_recovers(tmp_path):
    bucket = R2Bucket(BucketConfig(uri=f"file://{tmp_path}/r2"))
    budget = ContributionBudget(tmp_path / "budget", bucket, "run", 500)
    original = bucket.put_bytes

    def lost_response(key, payload, **kwargs):
        original(key, payload, **kwargs)
        raise OSError("lost response")

    bucket.put_bytes = lost_response
    with pytest.raises(OSError, match="lost response"):
        budget.reserve("chunk-a", 300)
    total = json.loads(budget.cache.read_text())["total"]
    with pytest.raises(ValueError, match="budget exhausted"):
        budget.reserve("chunk-b", 300)
    bucket.put_bytes = original
    budget.cache.unlink()  # A fresh attempt reconstructs the charge from durable R2.
    recovered = ContributionBudget(tmp_path / "budget", bucket, "run", 500)
    recovered.reserve("chunk-a", 300)
    assert json.loads(recovered.cache.read_text())["total"] == total
    with pytest.raises(ValueError, match="budget exhausted"):
        recovered.reserve("chunk-b", 300)


def test_object_listing_sizes_need_no_head_roundtrips(tmp_path):
    from unittest.mock import Mock

    bucket = R2Bucket(BucketConfig(uri=f"file://{tmp_path}/r2"))
    bucket.put_bytes("prefix/a", b"abc")
    bucket.put_bytes("prefix/b", b"12345")
    bucket.head = Mock(side_effect=AssertionError("listing must not HEAD"))
    assert list(bucket.iter_objects("prefix")) == [
        {"key": "prefix/a", "size": 3}, {"key": "prefix/b", "size": 5},
    ]
    remote = R2Bucket(BucketConfig(
        uri="s3://bucket/base", endpoint_url="https://r2.example", access_key_id="test",
        secret_access_key="test",
    ))
    client = Mock()
    client.get_paginator.return_value.paginate.return_value = [
        {"Contents": [{"Key": "base/prefix/a", "Size": 3}]},
        {"Contents": [{"Key": "base/prefix/b", "Size": 5}]},
    ]
    remote._client = client
    assert list(remote.iter_objects("prefix")) == [
        {"key": "prefix/a", "size": 3}, {"key": "prefix/b", "size": 5},
    ]
    client.head_object.assert_not_called()
