from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import httpx2
from huggingface_hub import constants
from huggingface_hub.utils import _http

from gradlab.publication import HUGGINGFACE_RELEASE_FILES


SCRIPT = Path(__file__).parents[1] / "scripts/audit_huggingface_release.py"
SPEC = importlib.util.spec_from_file_location("audit_huggingface_release", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeApi:
    def __init__(self, *, main_sha: str = "a" * 40, tag_sha: str = "a" * 40) -> None:
        self.main_sha = main_sha
        self.tag_sha = tag_sha

    def model_info(self, repo_id: str, **kwargs: object) -> SimpleNamespace:
        sha = self.tag_sha if kwargs.get("revision") else self.main_sha
        return SimpleNamespace(sha=sha, private=False)

    def list_repo_refs(self, repo_id: str, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            tags=[
                SimpleNamespace(name="v4", target_commit=self.tag_sha),
            ]
        )

    def list_repo_files(self, repo_id: str, **kwargs: object) -> list[str]:
        return sorted(HUGGINGFACE_RELEASE_FILES)

    def list_collections(self, **kwargs: object) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                title="GradLab — VizdoomDeathmatch-v1",
                slug="tsilva/gradlab-vizdoom-id",
            )
        ]

    def get_collection(self, slug: str) -> SimpleNamespace:
        return SimpleNamespace(
                title="GradLab — VizdoomDeathmatch-v1",
                slug=slug,
                private=False,
                items=[
                    SimpleNamespace(
                        item_id="tsilva/model",
                        item_type="model",
                        note="Immutable research release: https://huggingface.co/tsilva/model/tree/v4",
                    )
                ],
        )


def test_remote_release_audit_checks_commit_files_collection_and_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for filename in HUGGINGFACE_RELEASE_FILES:
        (tmp_path / filename).write_bytes(filename.encode())
    monkeypatch.setattr(
        MODULE,
        "hf_hub_download",
        lambda repo_id, filename, **kwargs: str(tmp_path / filename),
    )
    monkeypatch.setattr(
        MODULE,
        "validate_release_bundle",
        lambda root: {
            "repository": {
                "repo_id": "tsilva/model",
                "canonical_environment_id": "VizdoomDeathmatch-v1",
            },
            "release": {"version": "v4", "tier": "research"},
            "format_version": 4,
        },
    )
    monkeypatch.setattr(
        MODULE,
        "verify_replay",
        lambda path: {"codec_name": "h264", "frames": 100},
    )

    result = MODULE.audit_huggingface_release("tsilva/model", "v4", api=FakeApi())

    assert result["status"] == "passed"
    assert result["collection"] == "tsilva/gradlab-vizdoom-id"


def test_remote_release_audit_rejects_main_tag_drift() -> None:
    with pytest.raises(ValueError, match="do not point to the same commit"):
        MODULE.audit_huggingface_release(
            "tsilva/model",
            "v4",
            api=FakeApi(main_sha="a" * 40, tag_sha="b" * 40),
        )


def test_public_release_audit_ignores_ambient_hub_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = "example/model"
    commit = "a" * 40
    collection = {
        "slug": "example/gradlab-fixture",
        "title": "GradLab — Fixture-v0",
        "owner": "example",
        "items": [{
            "_id": "fixture-item", "id": repo, "type": "model", "position": 0,
            "note": {"text": f"https://huggingface.co/{repo}/tree/v4"},
        }],
        "lastUpdated": "2026-01-01T00:00:00.000Z",
        "private": False,
        "theme": "blue",
        "upvotes": 0,
    }
    model = f"/api/models/{repo}"
    requests = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        assert "authorization" not in request.headers
        path = request.url.path
        requests.append((request.method, path))
        if path in {model, f"{model}/revision/v4"}:
            return httpx2.Response(200, json={"id": repo, "sha": commit, "private": False})
        if path == f"{model}/refs":
            return httpx2.Response(200, json={"branches": [], "converts": [], "tags": [
                {"name": "v4", "ref": "refs/tags/v4", "targetCommit": commit},
            ]})
        if path in {f"{model}/tree/main", f"{model}/tree/v4"}:
            return httpx2.Response(200, json=[
                {"type": "file", "path": name, "size": len(name), "oid": commit}
                for name in HUGGINGFACE_RELEASE_FILES
            ])
        if path == "/api/collections":
            return httpx2.Response(200, json=[collection])
        if path == f"/api/collections/{collection['slug']}":
            return httpx2.Response(200, json={**collection, "position": 0})
        prefix = f"/{repo}/resolve/v4/"
        assert path.startswith(prefix), path
        name = path.removeprefix(prefix)
        assert name in HUGGINGFACE_RELEASE_FILES
        return httpx2.Response(200, content=name.encode() if request.method == "GET" else b"",
                               headers={"X-Repo-Commit": commit, "ETag": f'"{name}"',
                                        "Content-Length": str(len(name))})

    monkeypatch.setenv("HF_TOKEN", "hf_synthetic_ambient_token")
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_IMPLICIT_TOKEN", False)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(MODULE, "validate_release_bundle", lambda root: {
        "repository": {"repo_id": repo, "canonical_environment_id": "Fixture-v0"},
        "release": {"version": "v4"},
    })
    monkeypatch.setattr(MODULE, "verify_replay", lambda path: {"frames": 100})
    with httpx2.Client(transport=httpx2.MockTransport(respond)) as client:
        monkeypatch.setattr(_http, "_GLOBAL_CLIENT", client)
        assert MODULE.audit_huggingface_release(repo, "v4")["status"] == "passed"

    for name in HUGGINGFACE_RELEASE_FILES:
        path = f"/{repo}/resolve/v4/{name}"
        assert ("HEAD", path) in requests
        assert ("GET", path) in requests
