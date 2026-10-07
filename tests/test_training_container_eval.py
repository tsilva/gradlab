from __future__ import annotations

import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import numpy as np
import pytest

from gradlab.training_container_eval import TrainingContainerEvalBackend
from gradlab.video import SingleEpisodeVideoRecorder


def test_training_container_backend_queues_and_recovers_expired_attempts(tmp_path: Path) -> None:
    backend = TrainingContainerEvalBackend(tmp_path / "workers", max_workers=1)
    contract = {
        "schema_version": 6,
        "checkpoint_sha256": "a" * 64,
        "runtime_image_ref": "docker:example.invalid/test@sha256:" + "b" * 64,
        "asset": None,
        "seed_protocol": "gradlab.eval-seeds.v1",
        "n_envs": 1,
        "episodes": 1,
    }
    handles = []
    for number in range(2):
        target = tmp_path / f"result-{number}.json"
        handles.append(
            backend.submit(
                {
                    "attempt_id": f"attempt-{number}",
                    "contract": contract,
                    "expires_at": 0,
                    "result_uri": target.as_uri(),
                    "result_put_url": target.as_uri(),
                }
            )
        )
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        statuses = [backend.poll(handle).status for handle in handles]
        if statuses == ["succeeded", "succeeded"]:
            break
        time.sleep(0.05)
    assert statuses == ["succeeded", "succeeded"]
    assert all(json.loads((tmp_path / f"result-{i}.json").read_text())["status"] == "expired" for i in range(2))
    recovered = TrainingContainerEvalBackend(tmp_path / "workers", max_workers=1)
    assert [recovered.poll(handle).status for handle in handles] == ["succeeded", "succeeded"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is unavailable")
def test_episode_video_contains_initial_and_terminal_native_frames(tmp_path: Path) -> None:
    frame = np.zeros((8, 10, 3), dtype=np.uint8)

    class Provider:
        def render_lane(self, lane: int) -> np.ndarray:
            assert lane == 0
            return frame

    recorder = SingleEpisodeVideoRecorder(
        tmp_path / "episode.mp4", episode_id="lane-00-episode-000"
    )
    recorder.runtime = SimpleNamespace(provider=Provider())
    recorder.reset(np.asarray([True]))
    frame[:, :, 0] = 255
    recorder.transition(
        {}, np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1),
        np.asarray([True]), np.asarray([False]), None, np.asarray([False]),
    )
    recorder.close()

    metadata = recorder.require_complete()
    assert metadata["frames"] == 2
    assert metadata["episode_id"] == "lane-00-episode-000"
    assert metadata["bytes"] > 0
