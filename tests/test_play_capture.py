from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gradlab.file_utils import file_sha256
from gradlab.play_capture import (
    CAPTURE_DOCUMENT_TYPE,
    CAPTURE_FORMAT_VERSION,
    EpisodeFrameSpool,
    EpisodeCaptureManager,
    EpisodeCaptureStore,
    capture_output_size,
    render_spooled_replay,
    validate_capture_document,
)
from gradlab.publication import verify_replay


def _execution() -> dict:
    return {
        "source": {
            "kind": "checkout",
            "distribution": "gradlab",
            "version": "0.1.1",
            "git_commit": "a" * 40,
            "source_tree_sha256": "b" * 64,
        },
        "qualified_environment_id": "env-stableretro-turbo:Game-v0",
        "provider_id": "env-stableretro-turbo",
        "provider_version": "1.0.0",
        "environment_hash": "sha256:environment",
        "runtime_versions": {"stable_retro_turbo": "1.0.0"},
        "runtime_image_digest": "",
        "asset": {"sha256": "c" * 64},
        "execution_target": "local_player",
        "device_type": "cpu",
        "contract_mode": "training",
        "overrides": [],
        "seed": 7,
    }


def _document(replay: Path) -> dict:
    return {
        "document_type": CAPTURE_DOCUMENT_TYPE,
        "format_version": CAPTURE_FORMAT_VERSION,
        "created_at": "2026-08-06T12:00:00Z",
        "checkpoint_identity": "gradlab-" + "1" * 32 + ":step-1",
        "source": {
            "kind": "public_run",
            "run_id": "gradlab-" + "1" * 32,
            "checkpoint_id": "step-1",
            "artifact_name": "https://example.invalid/manifest.json",
            "revision": "d" * 64,
        },
        "run_id": "gradlab-" + "1" * 32,
        "checkpoint_id": "step-1",
        "checkpoint_sha256": "d" * 64,
        "recipe_sha256": "e" * 64,
        "goal": {"goal_id": "Goal1"},
        "contract": {"mode": "training", "requested_policy_override_paths": []},
        "execution": _execution(),
        "episode": 1,
        "seed": 7,
        "start_id": "Start1",
        "sampling_mode": "stochastic",
        "steps": 2,
        "return": 3.0,
        "max_x_pos": 4,
        "terminated": True,
        "truncated": False,
        "success": True,
        "outcome": "success",
        "boundary_role": "terminal_observation",
        "replay": {
            "sha256": file_sha256(replay),
            "size_bytes": replay.stat().st_size,
            "frames": 3,
            "fps": 30,
            "width": 128,
            "height": 128,
            "duration_seconds": 0.1,
            "codec_name": "h264",
            "codec_tag_string": "avc1",
            "pix_fmt": "yuv420p",
        },
    }


def test_capture_output_size_uses_integer_upscale_and_bounded_downscale() -> None:
    assert capture_output_size(224, 256) == (1024, 896, "nearest")
    width, height, interpolation = capture_output_size(1200, 1600)
    assert (width, height, interpolation) == (1280, 960, "area")


def test_capture_store_commits_fenced_latest_capture(tmp_path: Path) -> None:
    replay = tmp_path / "source.mp4"
    replay.write_bytes(b"browser-safe-fixture")
    store = EpisodeCaptureStore(tmp_path / "captures", max_count=2, max_total_bytes=1024 * 1024)

    captured = store.commit(
        checkpoint_identity="run:checkpoint",
        temporary_replay=replay,
        document_without_identity=_document(replay),
    )

    assert validate_capture_document(captured) == captured
    assert store.latest_for("run:checkpoint") == captured
    assert (store.capture_dir(captured["capture_id"]) / "replay.mp4").read_bytes() == (
        b"browser-safe-fixture"
    )


def test_new_episode_makes_prior_capture_ineligible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = tmp_path / "source.mp4"
    replay.write_bytes(b"browser-safe-fixture")
    store = EpisodeCaptureStore(tmp_path / "captures")
    captured = store.commit(
        checkpoint_identity="run:checkpoint",
        temporary_replay=replay,
        document_without_identity=_document(replay),
    )
    context = {
        "source_kind": "public_run",
        "contract_mode": "training",
        "matches_contract": True,
        "checkpoint_identity": "run:checkpoint",
    }
    manager = EpisodeCaptureManager(context, store=store)

    class Writer:
        def __init__(self, _path: Path, _frame: object) -> None:
            pass

        def abort(self) -> None:
            pass

    monkeypatch.setattr("gradlab.play_capture.EpisodeFrameSpool", Writer)

    assert manager.status()["ready"] is True
    manager.begin(
        np.zeros((2, 2, 3), dtype=np.uint8),
        episode=2,
        seed=8,
        sampling_mode="stochastic",
    )

    recording = manager.status()
    assert recording["recording"] is True
    assert recording["episode_in_progress"] is True
    assert recording["ready"] is False
    assert recording["latest"] == captured

    manager.abort("current episode capture failed")
    failed = manager.status()
    assert failed["recording"] is False
    assert failed["episode_in_progress"] is True
    assert failed["ready"] is False
    assert failed["latest"] == captured


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required",
)
def test_completed_frames_render_as_browser_safe_movie_only_on_demand(tmp_path: Path) -> None:
    raw = tmp_path / "episode.rgb"
    output = tmp_path / "replay.mp4"
    frames = [np.full((32, 40, 3), index * 30, dtype=np.uint8) for index in range(3)]
    spool = EpisodeFrameSpool(raw, frames[0])
    spool.write(frames[1])
    spool.write(frames[2])

    manifest = spool.close()
    assert raw.is_file()
    assert not output.exists()

    result = render_spooled_replay(raw, output, manifest)
    probe = verify_replay(output)

    assert result["frames"] == 3
    assert probe["frames"] == 3
    assert probe["codec_name"] == "h264"
    assert probe["pix_fmt"] == "yuv420p"


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required",
)
def test_capture_manager_defers_movie_render_until_publish_action(tmp_path: Path) -> None:
    context = {
        "source_kind": "public_run",
        "contract_mode": "training",
        "matches_contract": True,
        "checkpoint_identity": "run:checkpoint",
        "run_id": "run",
        "checkpoint_id": "checkpoint",
        "checkpoint_sha256": "d" * 64,
        "recipe_sha256": "e" * 64,
        "goal": {"goal_id": "Goal1"},
        "execution": _execution(),
    }
    store = EpisodeCaptureStore(tmp_path / "captures")
    manager = EpisodeCaptureManager(context, store=store)
    frames = [np.full((32, 40, 3), index * 30, dtype=np.uint8) for index in range(3)]
    manager.begin(frames[0], episode=1, seed=7, sampling_mode="stochastic")

    for index, frame in enumerate(frames[1:], start=1):
        boundary = index == 2
        manager.record_transition(
            SimpleNamespace(
                action_source="policy",
                after_frame=frame,
                boundary=boundary,
                after_frame_role="terminal_observation" if boundary else "observation",
                start_id="Start1",
                step=index,
                episode=1,
                seed=7,
                total_reward=3.0,
                max_x_pos=4,
                terminated=boundary,
                truncated=False,
                completed=boundary,
            )
        )

    completed = manager.status()
    assert completed["ready"] is True
    assert completed["render_required"] is True
    assert completed["latest"] is None
    assert not tuple(store.root.glob("capture-*/replay.mp4"))

    capture = manager.render()

    assert capture["capture_id"].startswith("capture-")
    assert manager.status()["render_required"] is False
    assert (store.capture_dir(capture["capture_id"]) / "replay.mp4").is_file()


def test_capture_rejects_reset_frame_at_terminal_boundary(tmp_path: Path) -> None:
    context = {
        "source_kind": "public_run",
        "contract_mode": "training",
        "matches_contract": True,
        "checkpoint_identity": "run:checkpoint",
        "run_id": "run",
        "checkpoint_id": "checkpoint",
        "checkpoint_sha256": "d" * 64,
        "recipe_sha256": "e" * 64,
        "execution": _execution(),
    }
    manager = EpisodeCaptureManager(context, store=EpisodeCaptureStore(tmp_path / "captures"))

    class Writer:
        def write(self, _frame: object) -> None:
            return None

        def abort(self) -> None:
            return None

    manager.writer = Writer()  # type: ignore[assignment]
    manager.temporary_frames = tmp_path / "partial.rgb"
    manager.temporary_frames.write_bytes(b"partial")
    transition = SimpleNamespace(
        action_source="policy",
        after_frame=np.zeros((2, 2, 3), dtype=np.uint8),
        boundary=True,
        after_frame_role="next_episode_initial_observation",
    )

    assert manager.record_transition(transition) is None
    assert manager.writer is None
    assert "terminal observation" in str(manager.error)
