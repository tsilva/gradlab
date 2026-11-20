from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx2
import pytest
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

from gradlab.dataset_hub import HubAppendSession


class FakeFeatures(dict):
    def to_dict(self):
        return dict(self)


class FakeDataset:
    def __init__(self, episode_ids, *, session_id="53a118b3-b31e-4f83-8ad4-b47cf1b05b33"):
        self.episode_ids = list(episode_ids)
        self.features = FakeFeatures(
            {
                "episode_id": {"dtype": "string", "_type": "Value"},
                "policy_actions": {"dtype": "null", "_type": "Value"},
            }
        )
        self.session_id = session_id

    def __getitem__(self, key):
        if key == "episode_id":
            return self.episode_ids
        if key == 0:
            return {"session_id": self.session_id}
        raise KeyError(key)

    def select(self, indices):
        return FakeDataset(
            [self.episode_ids[index] for index in indices], session_id=self.session_id
        )

    def cast(self, features):
        self.features = features
        return self

    def to_parquet(self, path):
        Path(path).write_bytes(b"parquet")


@dataclass
class FakeValidation:
    path: Path
    dataset: FakeDataset
    episode_fingerprints: dict[str, str]
    summary: object


class FakeApi:
    def __init__(self):
        self.calls = []

    def create_commit(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(oid=f"head-{len(self.calls)}")


def _session(api):
    session = HubAppendSession.__new__(HubAppendSession)
    session.target = "owner/repository"
    session.revision = "main"
    session.api = api
    session.head = "head-0"
    session.features = None
    session.episode_fingerprints = {}
    session.known_artifacts = {}
    session.sessions = []
    return session


def test_append_session_advances_expected_parent_without_rescanning(tmp_path):
    episode_id = "8f48fdb2-e6b2-49f8-97e3-b58f34f93d88"
    incoming = FakeValidation(
        path=tmp_path,
        dataset=FakeDataset([episode_id]),
        episode_fingerprints={episode_id: "a" * 64},
        summary=SimpleNamespace(environment_contracts=("b" * 64,), collector_contracts=()),
    )
    api = FakeApi()
    session = _session(api)

    with (
        patch("gradlab.dataset_store.validate_tree", return_value=incoming),
        patch("gradlab.dataset_hub._secret_egress_gate"),
    ):
        first_head = session.append(tmp_path)
        no_op_head = session.append(tmp_path)

    assert first_head == "head-1"
    assert no_op_head == "head-1"
    assert len(api.calls) == 1
    assert api.calls[0]["parent_commit"] == "head-0"
    assert session.episode_fingerprints == {episode_id: "a" * 64}
    assert len(session.sessions) == 1


@pytest.mark.parametrize(
    "status, error_code, expected_create",
    [
        (404, "RepoNotFound", True),
        (404, "RevisionNotFound", False),
        (404, None, False),
        (401, "RepoNotFound", False),
        (403, None, False),
    ],
)
def test_append_session_creates_only_after_confirmed_repository_absence(
    monkeypatch, status, error_code, expected_create
):
    calls = []
    repo = "owner/repository"

    def provider(request):
        assert request.url.host == "hub.example.com"
        calls.append(request.method)
        if calls == ["GET"]:
            assert request.url.path == f"/api/datasets/{repo}/revision/main"
            return httpx2.Response(
                status,
                headers={"X-Error-Code": error_code} if error_code else {},
                json={"error": "synthetic response"},
            )
        if calls == ["GET", "POST"]:
            assert request.url.path == "/api/repos/create"
            return httpx2.Response(200, json={"url": f"https://hub.example.com/datasets/{repo}"})
        assert calls == ["GET", "POST", "GET"]
        return httpx2.Response(200, json={"id": repo, "sha": "c" * 40})

    api = HfApi(endpoint="https://hub.example.com", token=False)
    with httpx2.Client(transport=httpx2.MockTransport(provider)) as session:
        monkeypatch.setattr("huggingface_hub.hf_api.get_session", lambda: session)
        monkeypatch.setattr("huggingface_hub.HfApi", lambda: api)
        if expected_create:
            assert HubAppendSession(repo).head == "c" * 40
            assert calls == ["GET", "POST", "GET"]
        else:
            with pytest.raises(HfHubHTTPError):
                HubAppendSession(repo)
            assert calls == ["GET"]
