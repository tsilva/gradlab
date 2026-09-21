import pytest
from types import SimpleNamespace


class Hub:
    """Controlled remote contents, including compare-and-swap and lost replies."""

    def __init__(self):
        self.files = {}
        self.head = "0"
        self.lose_reply = False
        self.conflict_once = False
        self.uploaded = {}

    def list_repo_tree(self, *args, **kwargs):
        return [SimpleNamespace(path=key) for key in self.files]

    def repo_info(self, **kwargs):
        return SimpleNamespace(sha=self.head)

    def read(self, path, revision):
        assert revision == self.head
        return self.files.get(path)

    def preupload_lfs_files(self, *args, **kwargs):
        for op in kwargs['additions']:
            if op.path_in_repo.endswith(('.zip', '.parquet', '.gz')):
                self.uploaded[op.path_in_repo] = op.path_or_fileobj
                op.path_or_fileobj = b''

    def create_commit(self, **kwargs):
        if self.conflict_once:
            self.conflict_once = False
            self.head = str(int(self.head) + 1)
            raise RuntimeError("remote head changed")
        assert kwargs["parent_commit"] == self.head
        for op in kwargs["operations"]:
            from pathlib import Path

            value = self.uploaded.get(op.path_in_repo, op.path_or_fileobj)
            self.files[op.path_in_repo] = (
                value if isinstance(value, bytes) else Path(value).read_bytes()
            )
        self.head = str(int(self.head) + 1)
        if self.lose_reply:
            self.lose_reply = False
            raise RuntimeError("commit response lost")
        return SimpleNamespace(oid=self.head)



@pytest.mark.parametrize("export_format", ["png-shards", "trajectories-webp"])
def test_durable_queue_filters_before_transfer_and_reconciles_hub_commit(tmp_path, monkeypatch, export_format):
    import json
    import gzip
    from pathlib import Path
    from gradlab.checkpoint_monitoring import monitor_episode, episode_manifest, finalize_monitoring
    from gradlab.env import resolve_env_config
    from gradlab.env_config import env_config_from_mapping
    from gradlab.recipe_documents import compose_train_document
    from gradlab.lifecycle_certification import CertificationFixture
    from gradlab.job_queue import JobStore, WorkerStart, run_flusher
    from gradlab.trajectory_publication import enqueue_publication
    from gradlab.json_utils import canonical_json_sha256
    from tests.test_checkpoint_monitoring import RequestedRight
    from huggingface_hub.errors import EntryNotFoundError

    fixture = CertificationFixture(tmp_path / "storage")
    bucket = fixture.authority.models
    prepared = fixture.prepare(run_number=90)
    manifest = prepared.supervisor.manifest
    goal = Path("experiments/goals/Breakout-Atari2600-v0/FirstWall")
    train = compose_train_document(goal / "_goal.yaml", goal / "recipes/ppo.yaml")["train_config"]
    config = resolve_env_config(env_config_from_mapping(train))
    config.task["termination"]["max_episode_steps"] = 3
    identity = "d" * 64
    prefix = f"monitoring/{manifest.run_id}/{identity}"
    episode = monitor_episode(
        model=RequestedRight(),
        config=config,
        episode=episode_manifest(2, 1)[0],
        bucket=bucket,
        root=tmp_path / "spool",
        prefix=prefix + "/monitor-000000",
        provenance=dict(
            evaluation_id=identity,
            checkpoint_id="checkpoint-test",
            checkpoint_step=100,
            planned_training_steps=200,
            run_id=manifest.run_id,
            training_seed=123,
        ),
        chunk_bytes=1024**2,
        watchdog_steps=10,
    )
    unrecorded = monitor_episode(
        model=RequestedRight(), config=config, episode=episode_manifest(2, 1)[1],
        bucket=bucket, root=tmp_path / "spool", prefix=prefix + "/monitor-000001",
        provenance=dict(evaluation_id=identity, checkpoint_id="checkpoint-test", checkpoint_step=100,
                        planned_training_steps=200, run_id=manifest.run_id, training_seed=123),
        chunk_bytes=1024**2, watchdog_steps=10,
    )
    assert unrecorded["chunks"] == []
    result = finalize_monitoring(
        [episode, unrecorded],
        episode_manifest(2, 1),
        bucket=bucket,
        root=tmp_path / "video",
        prefix=prefix,
        fps=15,
    )
    fixture.authority.control.put_json(
        f"runs/{manifest.run_id}/attempts/{manifest.attempt_id}/terminal.json",
        {
            "drain": {
                "checkpoint_monitoring": {
                    "complete": True,
                    "workers_quiescent": True,
                    "inventory": [
                        dict(
                            evaluation_id=identity,
                            status="complete",
                            result_key=prefix + "/result.json",
                            result_sha256=canonical_json_sha256(result),
                        )
                    ],
                }
            }
        },
    )
    api = Hub()
    api.create_repo = lambda **kwargs: None
    api.lose_reply = True

    def download(repo, name, *, revision, cache_dir, **kwargs):
        data = api.read(name, revision)
        if data is None:
            raise EntryNotFoundError("missing")
        path = Path(cache_dir) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)

    monkeypatch.setattr(
        "gradlab.operator_environment.load_repository_operator_environment", lambda root: None
    )
    monkeypatch.setattr("gradlab.r2_store.RunStorageConfig.from_env", lambda: fixture.storage)
    monkeypatch.setattr(
        "gradlab.trajectory_publication.ensure_flusher",
        lambda store: WorkerStart("already_running"),
    )
    monkeypatch.setattr("gradlab.trajectory_publication.HfApi", lambda: api)
    monkeypatch.setattr("gradlab.trajectory_publication.hf_hub_download", download)
    queue = JobStore(root=tmp_path / "queue")
    excluded = enqueue_publication(
        runs=[manifest.run_id],
        repo="test/data",
        filters={"stage_min": 0.75},
        repo_root=Path.cwd(),
        store=queue,
        export_format=export_format,
    )
    assert run_flusher(queue, idle_seconds=0) == 0
    assert queue.job(excluded["job"]["job_id"])["state"] == "succeeded"
    assert not any(key.startswith(("chunks/", "episodes/", "transitions/")) for key in api.files)
    selected = enqueue_publication(
        runs=[manifest.run_id],
        repo="test/data",
        filters={"stage_min": 0.25},
        repo_root=Path.cwd(),
        store=queue,
        export_format=export_format,
    )
    assert run_flusher(queue, idle_seconds=0) == 0
    assert queue.job(selected["job"]["job_id"])["state"] == "succeeded"
    index_prefix = "indexes/" if export_format == "png-shards" else "trajectories/"
    indexes = [
        json.loads(line)
        for key, data in api.files.items()
        if key.startswith(index_prefix) and key.endswith(".jsonl.gz")
        for line in gzip.decompress(data).splitlines()
    ]
    assert len(indexes) == 1 and indexes[0]["run_id"] == manifest.run_id
    assert bucket.get_bytes(episode["chunks"][0]["key"])


def test_active_snapshot_freezes_only_complete_verified_evaluations():
    from copy import deepcopy
    import pytest
    from gradlab.trajectory_publication import completed_snapshot_inventories
    from gradlab.json_utils import canonical_json_sha256

    run = "gradlab-" + "a" * 32
    identity = "b" * 64
    prefix = f"monitoring/{run}/{identity}"
    base = f"runs/{run}/monitoring/{identity}/"
    result = dict(evaluation_id=identity, checkpoint_id="checkpoint", checkpoint_step=10,
                  contract_sha256="c" * 64,
                  episodes=[dict(episode_id="episode", complete=True)])
    intent = dict(run_id=run, attempt_id="attempt-" + "d" * 16,
                  evaluation_id=identity, prefix=prefix,
                  checkpoint=dict(checkpoint_id="checkpoint", step=10),
                  contract_sha256="c" * 64, manifest=[dict(episode_id="episode")])

    class Bucket:
        def __init__(self, rows):
            self.rows = rows
        def iter_keys(self, prefix):
            return [key for key in self.rows if key.startswith(prefix)]
        def get_json(self, key):
            return deepcopy(self.rows[key])

    state = dict(status="complete", result_sha256=canonical_json_sha256(result))
    control = Bucket({base + "state.json": state, base + "intent.json": intent,
                      f"runs/{run}/monitoring/pending/state.json": dict(status="running")})
    models = Bucket({prefix + "/result.json": result})
    selected = completed_snapshot_inventories(control, models, [run])
    assert len(selected) == 1
    assert selected[0]["sha256"] == state["result_sha256"]
    result["checkpoint_step"] = 20
    with pytest.raises(ValueError, match="checksum"):
        completed_snapshot_inventories(control, models, [run])
    assert selected[0]["sha256"] == state["result_sha256"]
    state["status"] = "running"
    with pytest.raises(ValueError, match="no completed"):
        completed_snapshot_inventories(control, models, [run])


def test_publication_defers_hourly_commit_rate_limit(monkeypatch):
    import httpx
    from huggingface_hub.errors import HfHubHTTPError
    from gradlab.trajectory_publication import DatasetPublicationHandler

    error = HfHubHTTPError(
        "rate limit for repository commits (128 per hour)",
        response=httpx.Response(429, request=httpx.Request("POST", "https://huggingface.co")),
    )
    handler = DatasetPublicationHandler()
    def fail(job):
        raise error
    monkeypatch.setattr(handler, "_publish", fail)
    monkeypatch.setattr("gradlab.trajectory_publication.time.time", lambda: 100)
    result = handler.advance({"attempts": 8})
    assert result.state == "retry_wait"
    assert result.available_at == 3700
