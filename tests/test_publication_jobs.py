from __future__ import annotations

import json
from pathlib import Path

import pytest

from gradlab.file_utils import file_sha256
from gradlab.json_utils import canonical_json_sha256
from gradlab.publication_jobs import (
    PlayerPublicationJobHandler,
    _validate_new_repository_files,
)


def _snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, str], Path]:
    repo = tmp_path / "repo"
    queue = tmp_path / "queue"
    repo.mkdir()
    queue.mkdir()
    basis = {
        "document_type": "gradlab.player_publication_request",
        "format_version": 2,
        "capture_id": "capture-" + "1" * 32,
        "capture_fence_sha256": "2" * 64,
        "repo_id": "tsilva/example",
        "release_version": "v1",
        "parent_commit": None,
        "published_at": "2026-08-06T12:00:00Z",
        "metadata": {
            "title": "Example",
            "description": "Example",
            "tags": ["gradlab"],
            "privacy": "public",
            "container_name": "GradLab — Example-v0",
            "thumbnail_time": 1.0,
            "operator_note": "",
            "feature": False,
            "thumbnail": {
                "task": "Example",
                "trainer_algorithm": "GradLab PPO",
                "step": "1M env steps",
                "metric": "Return: 1",
            },
        },
        "principals": {
            "huggingface_username": "tsilva",
            "huggingface_namespace": "tsilva",
            "youtube_channel_id": "channel",
            "youtube_channel_title": "Channel",
            "youtube_scopes": [],
        },
        "evidence_sha256": "3" * 64,
        "comparison": {"comparable": False, "reason": "no prior release selected"},
        "feature": False,
    }
    fingerprint = canonical_json_sha256(basis)
    marker = f"gradlab-publication-{fingerprint}"
    metadata = {**basis["metadata"], "tags": ["gradlab", marker]}
    request = {
        **basis,
        "fingerprint_basis": basis,
        "request_fingerprint": fingerprint,
        "metadata_sha256": canonical_json_sha256(metadata),
        "marker": marker,
        "metadata": metadata,
    }
    root = tmp_path / "requests" / fingerprint
    (root / "capture").mkdir(parents=True)
    (root / "provisional_release").mkdir()
    replay = root / "capture/replay.mp4"
    replay.write_bytes(b"episode")
    capture = {
        "capture_id": basis["capture_id"],
        "capture_fence_sha256": basis["capture_fence_sha256"],
        "replay": {"sha256": file_sha256(replay)},
    }
    (root / "capture/capture.json").write_text(json.dumps(capture), encoding="utf-8")
    (root / "request.json").write_text(json.dumps(request), encoding="utf-8")
    (root / "provisional_release/dummy").write_text("release", encoding="utf-8")
    hashes = {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in root.rglob("*")
        if path.is_file()
    }
    (root / "snapshot_hashes.json").write_text(json.dumps(hashes), encoding="utf-8")
    monkeypatch.setattr(
        "gradlab.publication_jobs.validate_capture_document",
        lambda value: value,
    )
    monkeypatch.setattr(
        "gradlab.publication_jobs.validate_release_bundle",
        lambda _root: {"publication": {"request_fingerprint": fingerprint}},
    )
    payload = {
        "repo_root": str(repo),
        "queue_root": str(queue),
        "request_root": str(root),
        "request_fingerprint": fingerprint,
    }
    return payload, replay


def test_publication_snapshot_is_hash_fenced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, replay = _snapshot(tmp_path, monkeypatch)
    handler = PlayerPublicationJobHandler()

    snapshot = handler._snapshot(handler.validate_payload(payload))
    assert snapshot["request"]["request_fingerprint"] == payload["request_fingerprint"]

    replay.write_bytes(b"different")
    with pytest.raises(ValueError, match="changed after admission"):
        handler._snapshot(handler.validate_payload(payload))


def test_publication_snapshot_rejects_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, _replay = _snapshot(tmp_path, monkeypatch)
    root = Path(payload["request_root"])
    (root / "unexpected").symlink_to(root / "request.json")

    with pytest.raises(ValueError, match="symlink"):
        PlayerPublicationJobHandler()._snapshot(payload)


def test_new_huggingface_repository_allows_only_generated_gitattributes() -> None:
    assert _validate_new_repository_files([".gitattributes"]) == {".gitattributes"}
    assert _validate_new_repository_files([]) == set()

    with pytest.raises(ValueError, match="README.md"):
        _validate_new_repository_files([".gitattributes", "README.md"])
