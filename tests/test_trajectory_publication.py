from types import SimpleNamespace
import hashlib
import json
import zipfile

import pytest

from gradlab.trajectory_publication import append_episode
from gradlab.trajectory_config import FORMAT


class Hub:
    """Controlled remote contents, including compare-and-swap and lost replies."""

    def __init__(self):
        self.files = {}
        self.head = "0"
        self.lose_reply = False
        self.conflict_once = False

    def list_repo_tree(self, *args, **kwargs):
        return [SimpleNamespace(path=key) for key in self.files]

    def repo_info(self, **kwargs):
        return SimpleNamespace(sha=self.head)

    def read(self, path, revision):
        assert revision == self.head
        return self.files.get(path)

    def create_commit(self, **kwargs):
        if self.conflict_once:
            self.conflict_once = False
            self.head = str(int(self.head) + 1)
            raise RuntimeError("remote head changed")
        assert kwargs["parent_commit"] == self.head
        for op in kwargs["operations"]:
            from pathlib import Path

            value = op.path_or_fileobj
            self.files[op.path_in_repo] = (
                value if isinstance(value, bytes) else Path(value).read_bytes()
            )
        self.head = str(int(self.head) + 1)
        if self.lose_reply:
            self.lose_reply = False
            raise RuntimeError("commit response lost")
        return SimpleNamespace(oid=self.head)


def chunk(tmp_path):
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("frames/0.png", b"initial")
        z.writestr("frames/1.png", b"terminal")
        z.writestr(
            "transitions.jsonl",
            json.dumps(
                {
                    "step": 0,
                    "policy_action": 2,
                    "executed_action": 0,
                    "native_action": 1,
                    "override_rule": "auto_serve",
                }
            )
            + "\n",
        )
    return path


def test_atomic_append_idempotent_after_lost_reply_and_concurrent_head(tmp_path):
    api = Hub()
    api.conflict_once = True
    path = chunk(tmp_path)
    manifest = {
        "format": FORMAT,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "episode": {"episode_id": "run.attempt.1"},
    }
    contract = {"format": FORMAT, "execution": "verified"}
    append_episode(api, api.read, "user/data", contract, manifest, path, tmp_path)
    first = dict(api.files)
    append_episode(api, api.read, "user/data", contract, manifest, path, tmp_path)
    assert api.files == first
    api.lose_reply = True
    manifest = {**manifest, "episode": {"episode_id": "run.attempt.2"}}
    append_episode(api, api.read, "user/data", contract, manifest, path, tmp_path)
    assert set(first) <= set(api.files)
    indexes = [
        json.loads(v)
        for k, v in api.files.items()
        if k.startswith("episodes/") and k.endswith(".json")
    ]
    assert {m["episode"]["episode_id"] for m in indexes} == {"run.attempt.1", "run.attempt.2"}


def test_conflicting_identity_and_contract_rejected(tmp_path):
    api = Hub()
    path = chunk(tmp_path)
    manifest = {
        "format": FORMAT,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "episode": {"episode_id": "same"},
    }
    append_episode(api, api.read, "user/data", {"version": 1}, manifest, path, tmp_path)
    with pytest.raises(ValueError, match="contract"):
        append_episode(api, api.read, "user/data", {"version": 2}, manifest, path, tmp_path)
    with pytest.raises(ValueError, match="identity"):
        append_episode(
            api, api.read, "user/data", {"version": 1}, {**manifest, "bytes": 5}, path, tmp_path
        )


def test_explicit_queue_publication_filters_before_assets_and_preserves_actions(
    tmp_path, monkeypatch
):
    from pathlib import Path
    from gradlab.job_queue import JobStore, WorkerStart, run_flusher
    from gradlab.r2_store import R2Bucket, BucketConfig, RunStorageConfig
    from gradlab.trajectory_delivery import DatasetDelivery
    from gradlab.trajectory_publication import enqueue_publication
    from huggingface_hub.errors import EntryNotFoundError
    import gradlab.trajectory_publication as publication
    import gradlab.operator_environment as environment
    import pyarrow.parquet as pq
    import io

    run, attempt = "gradlab-" + "a" * 32, "attempt-" + "b" * 16
    storage = RunStorageConfig(
        BucketConfig(uri=(tmp_path / "control").as_uri()),
        BucketConfig(uri=(tmp_path / "eval").as_uri()),
        BucketConfig(uri=(tmp_path / "models").as_uri()),
    )
    control, models = R2Bucket(storage.control), R2Bucket(storage.models)
    spool = tmp_path / "spool"
    spool.mkdir()
    import numpy as np
    from tests.test_training_trajectories import training_env
    from gradlab.training_trajectories import TrainingRecorder
    from gradlab.trajectory_config import CollectionConfig

    monkeypatch.setattr(
        "gradlab.training_trajectories.shutil.disk_usage",
        lambda _: SimpleNamespace(total=100 * 1024**3, free=50 * 1024**3),
    )
    train, env = training_env()
    recorder = TrainingRecorder(
        env.runtime,
        spool,
        CollectionConfig(sample_probability=1, budget_stages=1),
        {"run_id": run, "attempt_id": attempt, "train_config": train},
        position=lambda: (0, 0),
    )
    env.runtime.recording = recorder
    try:
        env.reset()
        for _ in range(3):
            env.step(np.ones(2, dtype=np.int64))
    finally:
        env.close()
    recorder.wait(10)
    delivery = DatasetDelivery(spool, models, run, attempt, 10 * 1024**3)
    for _ in range(10):
        delivery.advance(final=True)
        if delivery.receipt()["complete"]:
            break
    assert delivery.receipt()["complete"]
    control.put_json(
        f"runs/{run}/manifest.json",
        {"run_id": run, "attempt_id": attempt, "created_at": "2026-09-17"},
    )
    control.put_json(
        f"runs/{run}/attempts/{attempt}/terminal.json",
        {"drain": {"dataset_delivery": delivery.receipt()}},
    )
    api = Hub()
    api.create_repo = lambda **kwargs: None

    def download(repo, name, **kwargs):
        if name not in api.files:
            raise EntryNotFoundError(name)
        target = Path(kwargs["cache_dir"]) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(api.files[name])
        return str(target)

    monkeypatch.setattr(publication, "HfApi", lambda: api)
    monkeypatch.setattr(publication, "hf_hub_download", download)
    monkeypatch.setattr(publication, "ensure_flusher", lambda _: WorkerStart("already_running"))
    monkeypatch.setattr(environment, "load_repository_operator_environment", lambda _: None)
    monkeypatch.setattr(RunStorageConfig, "from_env", lambda: storage)
    queue = JobStore(root=tmp_path / "queue")
    job = enqueue_publication(
        runs=[run], repo="user/data", filters={"score_min": 0}, repo_root=tmp_path, store=queue
    )
    assert api.files == {}  # Admission starts no publication itself.
    run_flusher(queue, idle_seconds=0)
    row = queue.job(job["job"]["job_id"])
    assert row["state"] == "succeeded", row
    table = next(value for key, value in api.files.items() if key.endswith(".parquet"))
    actual = pq.read_table(io.BytesIO(table)).to_pylist()[0]
    assert (actual["policy_action"], actual["executed_action"], actual["native_action"]) == (
        1,
        0,
        1,
    )
    before = dict(api.files)
    repeat = enqueue_publication(
        runs=[run], repo="user/data", filters={"score_min": 0}, repo_root=tmp_path, store=queue
    )
    run_flusher(queue, idle_seconds=0)
    assert not repeat["created"]
    assert api.files == before
    assert models.get_json(delivery.receipt()["manifest_key"])["chunks"]
