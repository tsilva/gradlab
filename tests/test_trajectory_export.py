import hashlib
import io
import json
import zipfile

import numpy as np
import pyarrow.parquet as pq
import pytest
from PIL import Image

from gradlab.trajectory_export import Converter, publish_trajectories
from gradlab.trajectory_format import read_record, require_current_trajectory_schema, open_trajectory_parquet
from gradlab.dataset_shards import PublicationBudget
from tests.test_dataset_shards import fixture


def recording():
    api, models, episodes = fixture(1)
    e = episodes[0]
    e.update(
        environment_seed=2147483648,
        recording_contract={"native_encoding": [1, 2, 3], "environment": {"frame_skip": 2}},
    )
    rgb = np.zeros((210, 160, 3), np.uint8)
    rgb[:17] = [123, 45, 67]
    output = io.BytesIO()
    Image.fromarray(rgb).save(output, format="PNG")
    png = output.getvalue()
    row = dict(
        step=0,
        policy_action=2,
        effective_policy_action=0,
        executed_action=0,
        native_action=1,
        override_rule="auto_serve",
        reward=1.0,
        provider_reward=2.0,
        provider_terminated=False,
        provider_truncated=False,
        terminated=True,
        truncated=False,
        facts={"bricks_remaining": 108, "walls_cleared": 0, "score": 0},
        task_metrics={},
        outcome=1,
    )
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("transitions.jsonl", json.dumps(row) + "\n")
        z.writestr("frames/0.png", png)
        z.writestr("frames/1.png", png)
    data = archive.getvalue()
    models.get_bytes = lambda key: data
    e["chunks"][0].update(
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
        first_frame_sha256=hashlib.sha256(png).hexdigest(),
        last_frame_sha256=hashlib.sha256(png).hexdigest(),
    )
    return api, models, episodes, rgb


def test_unmasked_lossless_dedup_actions_annotations_and_resume(tmp_path):
    _, models, episodes, rgb = recording()
    converter = Converter(tmp_path, {"selection": 1})
    converter.episode(1, episodes[0], models)
    converter.episode(1, episodes[0], models)
    stats = converter.tables(tmp_path / "output")
    assert stats == {"episodes": 1, "transitions": 1, "unique_frames": 1}
    frame = pq.read_table(tmp_path / "output/frames/assets/00000.parquet").to_pylist()[0]
    image = Image.open(io.BytesIO(frame["image"]["bytes"]))
    assert image.format == "WEBP" and np.array_equal(np.asarray(image), rgb)
    row = pq.read_table(tmp_path / "output/transitions/all/00000.parquet").to_pylist()[0]
    assert row["source_frame_id"] == row["successor_frame_id"] == frame["frame_id"]
    assert not row["successor_frame_new"]
    assert (
        row["selected_action_json"],
        row["effective_action_json"],
        row["native_action_json"],
    ) == ("2", "0", "0")
    decoded = read_record(row["record_json"])
    assert decoded["monitoring_record"]["native_action"] == 1
    assert row["task_terminated"] is None and row["temperature"] is None
    assert row["brick_grid_quality_flags"] == 130
    episode = pq.read_table(tmp_path / "output/episodes/all/00000.parquet").to_pylist()[0]
    assert episode["split"] is None and episode["initial_frame_id"] == frame["frame_id"]
    session = pq.read_table(tmp_path / "output/sessions/metadata/00000.parquet").to_pylist()[0]
    assert read_record(session["record_json"])["monitoring_episode"] == episodes[0]
    converter.db.close()


def test_corrupt_boundary_never_marks_episode_done(tmp_path):
    _, models, episodes, _ = recording()
    episodes[0]["chunks"][0]["first_frame_sha256"] = "0" * 64
    c = Converter(tmp_path, {})
    with pytest.raises(ValueError, match="boundary image"):
        c.episode(1, episodes[0], models)
    assert c.db.execute("SELECT count(*) FROM done").fetchone()[0] == 0
    c.db.close()


def test_atomic_publication_replaces_view_preserves_assets_and_retries(tmp_path):
    api, models, episodes, _ = recording()
    contract = {"format": "test"}
    api.files.update(
        {
            "checkpoint-dataset.json": json.dumps(contract).encode(),
            "README.md": b"old",
            "shards/old.zip": b"old source",
        }
    )

    # Test adapter stages file bytes like the public Hub preupload API.
    def preupload(repo, **kwargs):
        from pathlib import Path

        for op in kwargs["additions"]:
            if isinstance(op.path_or_fileobj, str):
                op.path_or_fileobj = Path(op.path_or_fileobj).read_bytes()

    api.preupload_lfs_files = preupload
    kwargs = dict(
        api=api,
        read=api.read,
        repo="test/data",
        contract=contract,
        episodes=episodes,
        selection={"id": 1},
        models=models,
        work=tmp_path,
        budget=PublicationBudget(tmp_path / "budget"),
        canceled=lambda: False,
    )
    publish_trajectories(**kwargs)
    publish_trajectories(**kwargs)
    assert len(api.commits) == 1
    assert api.files["shards/old.zip"] == b"old source"
    assert b"split: all" in api.files["README.md"] and b"split: train" not in api.files["README.md"]
    assert "trajectory-view.json" in api.files
    from copy import deepcopy

    later = deepcopy(episodes[0])
    later["episode_id"] = "later"
    kwargs.update(episodes=[later], selection={"id": 2}, work=tmp_path / "next-job")
    publish_trajectories(**kwargs)
    view = json.loads(api.files["trajectory-view.json"])
    receipt = json.loads(api.files[view["publication"]])
    require_current_trajectory_schema(view)
    require_current_trajectory_schema(receipt)
    prefix = view["publication"].removesuffix("publication.json")
    for name, identity in receipt["tables"].items():
        data = api.files[prefix + name]
        assert hashlib.sha256(data).hexdigest() == identity["sha256"]
        with open_trajectory_parquet(io.BytesIO(data), identity["table"]) as table:
            assert table.metadata.num_rows == identity["rows"]
    assert receipt["statistics"] == {"episodes": 2, "transitions": 2, "unique_frames": 1}
    assert len(api.commits) == 2


def test_multichunk_boundary_and_failed_episode_resume(tmp_path):
    _, models, episodes, _ = recording()
    e = episodes[0]
    original = models.get_bytes("chunk")
    with zipfile.ZipFile(io.BytesIO(original)) as z:
        row = json.loads(z.read("transitions.jsonl"))
        png = z.read("frames/0.png")
    chunks, blobs = [], {}
    for step in (0, 1):
        row.update(step=step, terminated=step == 1)
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as z:
            z.writestr("transitions.jsonl", json.dumps(row) + "\n")
            z.writestr(f"frames/{step}.png", png)
            z.writestr(f"frames/{step + 1}.png", png)
        data = out.getvalue()
        key = str(step)
        blobs[key] = data
        chunks.append(
            {
                **e["chunks"][0],
                "key": key,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
                "first_step": step,
                "end_step": step + 1,
            }
        )
    e.update(chunks=chunks, steps=2)
    models.get_bytes = lambda key: blobs[key]
    c = Converter(tmp_path, {})
    second = blobs.pop("1")
    with pytest.raises(KeyError):
        c.episode(1, e, models)
    assert c.db.execute("SELECT count(*) FROM frames").fetchone()[0] == 0
    blobs["1"] = second
    c.episode(1, e, models)
    c.tables(tmp_path / "output")
    rows = pq.read_table(tmp_path / "output/transitions/all/00000.parquet").to_pylist()
    assert [r["step"] for r in rows] == [0, 1]
    assert rows[0]["successor_frame_id"] == rows[1]["source_frame_id"]
    assert rows[0]["terminated"] is False and rows[1]["terminated"] is True
    c.db.close()
