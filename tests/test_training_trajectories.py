import io
import json
import zipfile
from pathlib import Path

import pytest
import numpy as np
from PIL import Image

from gradlab.env import make_training_vec_env, resolve_env_config
from gradlab.env_config import env_config_from_mapping
from gradlab.recipe_documents import compose_train_document
from gradlab.trajectory_config import CollectionConfig
from gradlab.training_trajectories import TrainingRecorder


@pytest.fixture(autouse=True)
def available_scratch(monkeypatch):
    from collections import namedtuple

    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(
        "gradlab.training_trajectories.shutil.disk_usage",
        lambda _: usage(100 * 1024**3, 50 * 1024**3, 50 * 1024**3),
    )


def training_env(n_envs=2, max_episode_steps=3):
    root = Path("experiments/goals/Breakout-Atari2600-v0/FirstWall")
    train = compose_train_document(root / "_goal.yaml", root / "recipes/ppo.yaml")["train_config"]
    config = resolve_env_config(env_config_from_mapping(train))
    config.env_args["noop_reset_max"] = 0
    config.task["termination"]["max_episode_steps"] = max_episode_steps
    env = make_training_vec_env(config, n_envs=n_envs, seed=12)
    return train, env


def test_actual_autoserve_actions_rgb_and_reset_boundaries_round_trip(tmp_path):
    train, env = training_env()
    position = [0, 0]
    recorder = TrainingRecorder(
        env.runtime,
        tmp_path,
        CollectionConfig(sample_probability=1, budget_stages=1),
        {
            "run_id": "gradlab-" + "a" * 32,
            "attempt_id": "attempt-" + "b" * 16,
            "train_config": train,
        },
        position=lambda: tuple(position),
    )
    env.runtime.recording = recorder
    try:
        env.reset()
        initial = env.runtime.provider.render_lane(0).copy()
        frames = []
        for step in range(3):
            position[:] = [step * 2, step // 2]
            env.step(np.ones(2, dtype=np.int64))
            if step < 2:
                frames.append(env.runtime.provider.render_lane(0).copy())
    finally:
        env.close()
    recorder.wait(10)
    manifest = json.loads(next(tmp_path.glob("*.manifest.json")).read_text())
    assert manifest["episode"]["complete"]
    assert manifest["episode"]["captured_steps"] == 3
    with zipfile.ZipFile(tmp_path / manifest["file"]) as chunk:
        rows = [json.loads(line) for line in chunk.read("transitions.jsonl").splitlines()]
        np.testing.assert_array_equal(
            np.array(Image.open(io.BytesIO(chunk.read("frames/0.png")))), initial
        )
        np.testing.assert_array_equal(
            np.array(Image.open(io.BytesIO(chunk.read("frames/1.png")))), frames[0]
        )
        assert rows[0]["policy_action"] == 1
        assert rows[0]["executed_action"] == 0  # FIRE in this Run's provider table.
        assert rows[0]["native_action"] == 1
        assert rows[0]["override_rule"] == "auto_serve"
        assert rows[1]["executed_action"] == 1
        assert rows[1]["native_action"] == 2
        assert rows[1]["override_rule"] is None
        assert [row["policy_update"] for row in rows] == [0, 0, 1]
        assert rows[-1]["truncated"]
        assert not rows[-1]["provider_truncated"]
        terminal = np.array(Image.open(io.BytesIO(chunk.read("frames/3.png"))))
        assert not np.array_equal(terminal, initial)


def test_capture_leaves_seeded_training_trajectory_and_rng_unchanged(tmp_path):
    train, recorded = training_env()
    _, baseline = training_env()
    position = [0, 0]
    recorder = TrainingRecorder(
        recorded.runtime,
        tmp_path,
        CollectionConfig(sample_probability=1, budget_stages=1),
        {
            "run_id": "gradlab-" + "c" * 32,
            "attempt_id": "attempt-" + "d" * 16,
            "train_config": train,
        },
        position=lambda: tuple(position),
    )
    recorded.runtime.recording = recorder
    try:
        recorded.reset()
        baseline.reset()
        np.random.seed(983)
        rng_before = np.random.get_state()
        for step in range(20):
            position[:] = [step * 2, step // 4]
            actions = np.array([step % 3, (step + 1) % 3], dtype=np.int64)
            actual = recorded.step(actions)
            expected = baseline.step(actions)
            for key in actual[0]:
                np.testing.assert_array_equal(actual[0][key], expected[0][key])
            np.testing.assert_array_equal(actual[1], expected[1])
            np.testing.assert_array_equal(actual[2], expected[2])
            np.testing.assert_array_equal(
                recorded.runtime.provider.render_lane(1), baseline.runtime.provider.render_lane(1)
            )
        np.testing.assert_array_equal(np.random.get_state()[1], rng_before[1])
    finally:
        recorded.close()
        baseline.close()
    recorder.wait(10)


def test_recording_cutoff_is_prefix_not_environment_terminal(tmp_path):
    train, env = training_env()
    recorder = TrainingRecorder(
        env.runtime,
        tmp_path,
        CollectionConfig(sample_probability=1, budget_stages=1),
        {
            "run_id": "gradlab-" + "c" * 32,
            "attempt_id": "attempt-" + "e" * 16,
            "train_config": train,
        },
        position=lambda: (0, 0),
    )
    env.runtime.recording = recorder
    try:
        env.reset()
        env.step(np.ones(2, dtype=np.int64))
    finally:
        env.close()
    recorder.wait(10)
    document = json.loads(next(tmp_path.glob("*.manifest.json")).read_text())
    episode = document["episode"]
    assert not episode["complete"]
    assert episode["interruption_reason"] == "training_stopped"
    assert "episode_return" not in episode
    assert not episode["last_transition"]["terminated"]
    assert not episode["last_transition"]["truncated"]


def test_reset_reserves_disk_per_selected_lane_before_encoder_progress(tmp_path, monkeypatch):
    import threading
    from gradlab.trajectory_config import index_disk_bytes
    from gradlab.trajectory_delivery import spool_bytes

    train, env = training_env(n_envs=16)
    limits = CollectionConfig(
        sample_probability=1,
        budget_stages=1,
        contribution_bytes=4 * 32 * 1024**2,
        max_active_episodes=16,
    )
    from dataclasses import replace

    limits = replace(limits, disk_bytes=2 * limits.chunk_bytes + index_disk_bytes(limits) + 1024**2)
    release = threading.Event()
    original = TrainingRecorder._write

    def delayed(self, *args):
        assert release.wait(10)
        return original(self, *args)

    monkeypatch.setattr(TrainingRecorder, "_write", delayed)
    recorder = TrainingRecorder(
        env.runtime,
        tmp_path,
        limits,
        {
            "run_id": "gradlab-" + "c" * 32,
            "attempt_id": "attempt-" + "f" * 16,
            "train_config": train,
        },
        position=lambda: (0, 0),
    )
    env.runtime.recording = recorder
    try:
        env.reset()
        assert len(recorder.active) == 2
        assert recorder.reserved == 2 * limits.chunk_bytes
        assert recorder.skipped == 14
        for _ in range(10):
            env.step(np.ones(16, dtype=np.int64))
    finally:
        release.set()
        env.close()
    recorder.wait(10)
    assert spool_bytes(tmp_path) <= limits.disk_bytes


def test_handoff_memory_reserves_encoder_delivery_and_expanded_indexes(tmp_path):
    from gradlab.trajectory_config import working_memory_bytes
    from gradlab.training_trajectories import FRAME_RESERVATION

    train, env = training_env()
    limits = CollectionConfig()
    recorder = TrainingRecorder(
        env.runtime,
        tmp_path,
        limits,
        {
            "run_id": "gradlab-" + "c" * 32,
            "attempt_id": "attempt-" + "f" * 16,
            "train_config": train,
        },
        position=lambda: (0, 0),
    )
    try:
        assert working_memory_bytes(limits) >= 3 * limits.chunk_bytes
        assert (
            recorder.queue.maxsize * FRAME_RESERVATION + working_memory_bytes(limits)
            <= limits.memory_bytes
        )
    finally:
        recorder.close()
        recorder.wait(10)
        env.close()


def test_saturated_handoff_retains_contiguous_prefix_without_stopping_training(
    tmp_path, monkeypatch
):
    import threading
    from dataclasses import replace
    from gradlab.trajectory_config import working_memory_bytes

    train, env = training_env(max_episode_steps=100)
    limits = CollectionConfig(
        sample_probability=1, budget_stages=1, contribution_bytes=4 * 1024**2, chunk_bytes=1024**2
    )
    limits = replace(limits, memory_bytes=working_memory_bytes(limits) + 1024**2)
    release = threading.Event()
    original = TrainingRecorder._write

    def delayed(self, *args):
        assert release.wait(10)
        return original(self, *args)

    monkeypatch.setattr(TrainingRecorder, "_write", delayed)
    recorder = TrainingRecorder(
        env.runtime,
        tmp_path,
        limits,
        {
            "run_id": "gradlab-" + "c" * 32,
            "attempt_id": "attempt-" + "f" * 16,
            "train_config": train,
        },
        position=lambda: (0, 0),
    )
    env.runtime.recording = recorder
    try:
        env.reset()
        for _ in range(10):
            env.step(np.ones(2, dtype=np.int64))
        assert env.runtime._episode_lengths[0] == 10
        assert not recorder.active
    finally:
        release.set()
        env.close()
    recorder.wait(10)
    manifest = json.loads(next(tmp_path.glob("*.manifest.json")).read_text())
    assert not manifest["episode"]["complete"]
    assert manifest["episode"]["interruption_reason"] == "memory_pressure"
    assert 1 <= manifest["episode"]["captured_steps"] <= 4
    assert not manifest["episode"]["last_transition"]["truncated"]
    with zipfile.ZipFile(tmp_path / manifest["file"]) as chunk:
        rows = [json.loads(row) for row in chunk.read("transitions.jsonl").splitlines()]
        assert [row["step"] for row in rows] == list(range(len(rows)))
        assert len([p for p in chunk.namelist() if p.startswith("frames/")]) == len(rows) + 1


def test_changed_provider_action_contract_is_rejected_before_recording(tmp_path):
    from dataclasses import replace

    train, env = training_env()
    env.runtime.descriptor = replace(env.runtime.descriptor, action_meanings=("unverified",))
    try:
        with pytest.raises(ValueError, match="native encoding"):
            TrainingRecorder(
                env.runtime,
                tmp_path,
                CollectionConfig(),
                {
                    "run_id": "gradlab-" + "c" * 32,
                    "attempt_id": "attempt-" + "f" * 16,
                    "train_config": train,
                },
                position=lambda: (0, 0),
            )
        assert not list(tmp_path.glob("*.zip"))
    finally:
        env.close()
