from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

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
        "format_version": 3,
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
        "history": [],
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
        lambda _root: {
            "publication": {"request_fingerprint": fingerprint},
            "repository": {"canonical_environment_id": "Example-v0"},
        },
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
    assert snapshot["environment_id"] == "Example-v0"

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


@pytest.mark.parametrize(
    ("title", "dashboard", "saved"),
    [
        ("Policies - Example-v0", False, False),
        ("GradLab — Example-v0", False, False),
        ("Owner's example policies", True, False),
        ("Owner's example policies", False, True),
        (None, False, False),
    ],
)
def test_release_uses_the_environment_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    title: str | None,
    dashboard: bool,
    saved: bool,
) -> None:
    payload, _ = _snapshot(tmp_path, monkeypatch)
    handler = PlayerPublicationJobHandler()
    snapshot = handler._snapshot(payload)
    collection = SimpleNamespace(
        slug="tsilva/example-collection",
        owner="tsilva",
        title=title or "Policies - Example-v0",
        private=False,
        items=[],
    )
    api = Mock()
    api.list_collections.side_effect = lambda *, owner, item=None, limit=None: (
        [collection] if title and (item is None or dashboard) else []
    )
    api.get_collection.return_value = collection
    api.create_collection.return_value = collection
    youtube = Mock()
    youtube.find_or_create_playlist.return_value = "playlist-id"
    monkeypatch.setattr(handler, "_hf_access", lambda request: (api, "fixture"))
    monkeypatch.setattr(handler, "_youtube_access", lambda request: youtube)
    monkeypatch.setattr(handler, "_checkpoint", Mock())
    state_path = tmp_path / "publication-state.json"
    state = {"phase": "huggingface_verified", "youtube_video_id": "video-id"}
    if saved:
        state["huggingface_collection"] = collection.slug

    handler._advance_containers(
        job={"job_id": "job"}, store=Mock(), snapshot=snapshot,
        state_path=state_path, state=state,
    )

    if saved:
        api.list_collections.assert_not_called()
    else:
        assert api.list_collections.call_args_list[0].kwargs == {
            "owner": "tsilva", "item": "spaces/tsilva/Example-v0_Dashboard", "limit": 2
        }
    if title is None:
        api.create_collection.assert_called_once_with(
            "Policies - Example-v0",
            namespace="tsilva",
            description="Policies that play Example-v0.",
            private=False,
            exists_ok=True,
        )
    else:
        api.get_collection.assert_called_once_with(collection.slug)
        api.create_collection.assert_not_called()
    api.add_collection_item.assert_called_once_with(
        collection.slug, "tsilva/example", "model",
        note="Immutable research release: https://huggingface.co/tsilva/example/tree/v1",
        exists_ok=False,
    )
    youtube.find_or_create_playlist.assert_called_once_with("GradLab — Example-v0", privacy="public")
    assert json.loads(state_path.read_text())["huggingface_collection"] == collection.slug
    assert state["phase"] == "containers_complete"


def test_release_stops_before_writing_when_environment_collections_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, _ = _snapshot(tmp_path, monkeypatch)
    handler = PlayerPublicationJobHandler()
    api = Mock()
    api.list_collections.side_effect = [
        [],
        [
            SimpleNamespace(title="Policies - Example-v0"),
            SimpleNamespace(title="GradLab — Example-v0"),
        ],
    ]
    youtube = Mock()
    monkeypatch.setattr(handler, "_hf_access", lambda request: (api, "fixture"))
    monkeypatch.setattr(handler, "_youtube_access", lambda request: youtube)

    with pytest.raises(ValueError, match="multiple Hugging Face collections"):
        handler._advance_containers(
            job={"job_id": "job"}, store=Mock(), snapshot=handler._snapshot(payload),
            state_path=tmp_path / "publication-state.json", state={},
        )

    api.create_collection.assert_not_called()
    api.add_collection_item.assert_not_called()
    youtube.find_or_create_playlist.assert_not_called()
