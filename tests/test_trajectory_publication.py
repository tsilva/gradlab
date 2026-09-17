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


def test_explicit_queue_publication_filters_before_assets_and_preserves_actions(tmp_path, monkeypatch):
    from pathlib import Path
    from gradlab.file_utils import atomic_write_json
    from gradlab.job_queue import JobStore, WorkerStart, run_flusher
    from gradlab.r2_store import R2Bucket, BucketConfig, RunStorageConfig
    from gradlab.trajectory_delivery import DatasetDelivery
    from gradlab.trajectory_publication import enqueue_publication
    from huggingface_hub.errors import EntryNotFoundError
    import gradlab.trajectory_publication as publication
    import gradlab.operator_environment as environment
    import pyarrow.parquet as pq
    import io

    run, attempt = 'gradlab-' + 'a'*32, 'attempt-' + 'b'*16
    storage = RunStorageConfig(
        BucketConfig(uri=(tmp_path/'control').as_uri()),
        BucketConfig(uri=(tmp_path/'eval').as_uri()),
        BucketConfig(uri=(tmp_path/'models').as_uri()),
    )
    control, models = R2Bucket(storage.control), R2Bucket(storage.models)
    spool = tmp_path/'spool'
    spool.mkdir()
    path = chunk(spool)
    document = {'format': FORMAT, 'file': path.name,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'bytes': path.stat().st_size, 'reserved_bytes': 1024**2,
                'episode': {'episode_id': f'{run}.{attempt}.0', 'run_id': run,
                            'attempt_id': attempt, 'complete': True, 'stage_start': 0.5,
                            'prefix_return': 4, 'last_transition': {'facts': {'score': 7}}}}
    atomic_write_json(spool/'episode.manifest.json', document)
    atomic_write_json(spool/'producer.json', {'contract': {'format': FORMAT}})
    atomic_write_json(spool/'closed.json', {'chunks': 1, 'fault': None})
    delivery = DatasetDelivery(spool, models, run, attempt, 2 * 1024**2)
    delivery.advance(final=True)
    control.put_json(f'runs/{run}/manifest.json', {'run_id': run, 'attempt_id': attempt, 'created_at': '2026-09-17'})
    control.put_json(f'runs/{run}/attempts/{attempt}/terminal.json', {'drain': {'dataset_delivery': delivery.receipt()}})
    api = Hub()
    api.create_repo = lambda **kwargs: None
    def download(repo, name, **kwargs):
        if name not in api.files:
            raise EntryNotFoundError(name)
        target = Path(kwargs['cache_dir'])/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(api.files[name])
        return str(target)
    monkeypatch.setattr(publication, 'HfApi', lambda: api)
    monkeypatch.setattr(publication, 'hf_hub_download', download)
    monkeypatch.setattr(publication, 'ensure_flusher', lambda _: WorkerStart('already_running'))
    monkeypatch.setattr(environment, 'load_repository_operator_environment', lambda _: None)
    monkeypatch.setattr(RunStorageConfig, 'from_env', lambda: storage)
    queue = JobStore(root=tmp_path/'queue')
    job = enqueue_publication(runs=[run], repo='user/data', filters={'score_min': 5}, repo_root=tmp_path, store=queue)
    assert api.files == {}  # Admission starts no publication itself.
    run_flusher(queue, idle_seconds=0)
    row = queue.job(job['job']['job_id'])
    assert row['state'] == 'succeeded', row
    table = next(value for key,value in api.files.items() if key.endswith('.parquet'))
    actual = pq.read_table(io.BytesIO(table)).to_pylist()[0]
    assert (actual['policy_action'], actual['executed_action'], actual['native_action']) == (2, 0, 1)
    before = dict(api.files)
    repeat = enqueue_publication(runs=[run], repo='user/data', filters={'score_min': 5}, repo_root=tmp_path, store=queue)
    run_flusher(queue, idle_seconds=0)
    assert not repeat['created']
    assert api.files == before
    assert models.get_json(delivery.receipt()['manifest_key'])['chunks']
