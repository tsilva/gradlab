from types import SimpleNamespace


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


def test_durable_queue_filters_before_transfer_and_reconciles_hub_commit(tmp_path, monkeypatch):
    import json
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
        episode=episode_manifest(1)[0],
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
    result = finalize_monitoring(
        [episode],
        episode_manifest(1),
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
    api.conflict_once = True
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
    )
    assert run_flusher(queue, idle_seconds=0) == 0
    assert queue.job(selected["job"]["job_id"])["state"] == "succeeded"
    indexes = [
        json.loads(data)
        for key, data in api.files.items()
        if key.startswith("episodes/") and key.endswith(".json")
    ]
    assert len(indexes) == 1 and indexes[0]["run_id"] == manifest.run_id
    assert bucket.get_bytes(episode["chunks"][0]["key"])
