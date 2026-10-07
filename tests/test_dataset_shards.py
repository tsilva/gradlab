import hashlib
import gzip
import io
import json
import zipfile

import pytest
import pyarrow.parquet as pq

from gradlab.dataset_shards import PublicationBudget, PublicationDeferred, publish_sharded
from gradlab.checkpoint_monitoring import FORMAT
from gradlab.json_utils import canonical_json_bytes
from tests.test_trajectory_publication import Hub


def fixture(count):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as z:
        z.writestr("transitions.jsonl", '{"step":0,"reward":1}\n')
        z.writestr("frames/0.png", b"initial-png")
        z.writestr("frames/1.png", b"terminal-png")
    data = payload.getvalue()
    digest = hashlib.sha256(data).hexdigest()

    class Models:
        calls = 0

        def get_bytes(self, key):
            self.calls += 1
            return data

    episodes = [
        dict(
            format=FORMAT,
            episode_id=f"episode-{i}",
            evaluation_id="eval",
            run_id="run",
            training_seed=123,
            checkpoint_id="checkpoint",
            checkpoint_step=10,
            steps=1,
            native_score=1,
            shaped_return=1.0,
            normalized_brick_progress=0.1,
            success=False,
            complete=True,
            chunks=[
                dict(
                    key="chunk",
                    sha256=digest,
                    bytes=len(data),
                    first_step=0,
                    end_step=1,
                    first_frame_sha256="first",
                    last_frame_sha256="last",
                )
            ],
        )
        for i in range(count)
    ]
    api = Hub()
    api.create_repo = lambda **kw: None
    api.commits = []
    original = api.create_commit

    def commit(**kw):
        api.commits.append([op.path_in_repo for op in kw["operations"]])
        return original(**kw)

    api.create_commit = commit
    return api, Models(), episodes


def publish(tmp_path, api, models, episodes, *, work="export"):
    root = tmp_path / work
    root.mkdir(exist_ok=True)
    return publish_sharded(
        api,
        api.read,
        "test/data",
        {"format": "test"},
        episodes,
        {"episodes": [e["episode_id"] for e in episodes]},
        models,
        root,
        PublicationBudget(tmp_path / "budget"),
        lambda: False,
    )


def test_hundreds_of_episodes_in_one_atomic_commit_and_reuse(tmp_path):
    api, models, episodes = fixture(400)
    publish(tmp_path, api, models, episodes)
    assert len(api.commits) == 1
    assert len(api.commits[0]) == 7
    name = next(k for k in api.files if k.startswith("episodes/") and k.endswith(".parquet"))
    rows = pq.read_table(io.BytesIO(api.files[name])).to_pylist()
    assert len(rows) == 400
    transition = rows[0]["transition_tables"][0]
    row = pq.read_table(io.BytesIO(api.files[transition])).to_pylist()[0]
    with zipfile.ZipFile(io.BytesIO(api.files[row["archive"]])) as z:
        assert z.read(row["frame"]) == b"initial-png"
        assert z.read(row["next_frame"]) == b"terminal-png"
    count = models.calls
    publish(tmp_path, api, models, episodes)
    assert len(api.commits) == 1 and models.calls == count


def test_existing_episodes_preserved_and_overlapping_snapshot_deduplicated(tmp_path):
    api, models, episodes = fixture(3)
    e = episodes[0]
    api.files["checkpoint-dataset.json"] = canonical_json_bytes({"format": "test"})
    api.files["episodes/old.json"] = canonical_json_bytes(e)
    api.files["episodes/old.parquet"] = b"preserve-existing"
    publish(tmp_path, api, models, episodes)
    assert api.files["episodes/old.parquet"] == b"preserve-existing"
    records = [
        json.loads(row)
        for k, v in api.files.items()
        if k.startswith("indexes/")
        for row in gzip.decompress(v).splitlines()
    ]
    assert len(records) == 2
    publish(tmp_path, api, models, episodes[1:], work="overlap")
    assert len([k for k in api.files if k.startswith("indexes/")]) == 1


def test_lost_reply_reconciles_without_duplicate_commit(tmp_path):
    api, models, episodes = fixture(2)
    api.lose_reply = True
    publish(tmp_path, api, models, episodes)
    assert len(api.commits) == 1


def test_conflicting_parent_resumes_from_journal(tmp_path):
    api, models, episodes = fixture(2)
    api.conflict_once = True
    with pytest.raises(RuntimeError, match="head changed"):
        publish(tmp_path, api, models, episodes)
    publish(tmp_path, api, models, episodes)
    assert len(api.commits) == 2


def test_budget_is_shared_persistent_and_counts_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr("gradlab.dataset_shards.time.time", lambda: 1000)
    for _ in range(10):
        PublicationBudget(tmp_path).reserve("commit")
    with pytest.raises(PublicationDeferred) as error:
        PublicationBudget(tmp_path).reserve("commit")
    assert error.value.until == 4605
    monkeypatch.setattr("gradlab.dataset_shards.time.time", lambda: 4606)
    PublicationBudget(tmp_path).reserve("commit")


def test_429_honors_longest_reset_and_pauses_other_jobs(tmp_path, monkeypatch):
    import httpx
    from huggingface_hub.errors import HfHubHTTPError

    monkeypatch.setattr("gradlab.dataset_shards.time.time", lambda: 1000)
    budget = PublicationBudget(tmp_path)

    def fail():
        raise HfHubHTTPError(
            "repository commits",
            response=httpx.Response(
                429,
                headers={"retry-after": "4000", "ratelimit": '"api";r=0;t=100'},
                request=httpx.Request("POST", "https://huggingface.co"),
            ),
        )

    with pytest.raises(PublicationDeferred) as error:
        budget.call(fail)
    assert error.value.until == 5015
    with pytest.raises(PublicationDeferred):
        PublicationBudget(tmp_path).reserve()


def test_writer_lock_prevents_concurrent_publishers(tmp_path):
    with PublicationBudget(tmp_path).writer("test/data"):
        with pytest.raises(PublicationDeferred):
            with PublicationBudget(tmp_path).writer("test/data"):
                pytest.fail("second writer acquired lock")


def test_split_batches_never_expose_incomplete_episode_indexes(tmp_path, monkeypatch):
    from gradlab.dataset_shards import pack_plan

    api, models, episodes = fixture(8)
    monkeypatch.setattr("gradlab.dataset_shards.MAX_OPERATIONS", 6)
    monkeypatch.setattr("gradlab.dataset_shards.pack_plan", lambda eps: pack_plan(eps, limit=1))
    publish(tmp_path, api, models, episodes)
    assert all(len(ops) <= 6 for ops in api.commits)
    assert not any(k.startswith(("indexes/", "episodes/")) for ops in api.commits[:-1] for k in ops)
    assert any(k.startswith("indexes/") for k in api.commits[-1])
    assert any(k.startswith("contributions/") for k in api.commits[-1])


def test_partial_episode_and_corrupt_bytes_never_publish(tmp_path):
    api, models, episodes = fixture(2)
    episodes[0]["complete"] = False
    with pytest.raises(ValueError, match="incomplete"):
        publish(tmp_path, api, models, episodes)
    assert not api.commits
    episodes[0]["complete"] = True
    episodes[0]["chunks"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        publish(tmp_path, api, models, episodes)
    assert not api.commits


def test_episode_crossing_shards_keeps_boundary_and_both_tables(tmp_path, monkeypatch):
    from gradlab.dataset_shards import pack_plan

    api, models, episodes = fixture(1)
    e = episodes[0]
    first = models.get_bytes("chunk")
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w") as z:
        z.writestr("transitions.jsonl", '{"step":1,"reward":2}\n')
        z.writestr("frames/1.png", b"terminal-png")
        z.writestr("frames/2.png", b"final-png")
    second = blob.getvalue()
    e["steps"] = 2
    e["chunks"].append(
        dict(
            key="second",
            sha256=hashlib.sha256(second).hexdigest(),
            bytes=len(second),
            first_step=1,
            end_step=2,
            first_frame_sha256="last",
            last_frame_sha256="final",
        )
    )
    models.get_bytes = lambda key: first if key == "chunk" else second
    monkeypatch.setattr("gradlab.dataset_shards.pack_plan", lambda eps: pack_plan(eps, limit=1))
    publish(tmp_path, api, models, episodes)
    summary = next(v for k, v in api.files.items() if k.startswith("episodes/"))
    row = pq.read_table(io.BytesIO(summary)).to_pylist()[0]
    assert len(row["transition_tables"]) == 2
    transitions = [
        pq.read_table(io.BytesIO(api.files[p])).to_pylist()[0] for p in row["transition_tables"]
    ]
    assert [r["step"] for r in transitions] == [0, 1]
    frames = []
    for r in transitions:
        with zipfile.ZipFile(io.BytesIO(api.files[r["archive"]])) as z:
            frames.append((z.read(r["frame"]), z.read(r["next_frame"])))
    assert frames == [(b"initial-png", b"terminal-png"), (b"terminal-png", b"final-png")]
