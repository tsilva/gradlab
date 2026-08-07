from __future__ import annotations

import base64
import hashlib
import urllib.parse
from pathlib import Path

import pytest
from PIL import Image

from gradlab.youtube_publication import (
    YouTubeClient,
    YouTubePublicationError,
    create_publication_thumbnail,
    exchange_oauth_code,
    new_oauth_transaction,
    validate_processed_video,
)


CLIENT = {"installed": {"client_id": "client-id", "client_secret": "client-secret"}}


def test_oauth_transaction_uses_s256_and_binds_authority() -> None:
    transaction = new_oauth_transaction(
        CLIENT,
        redirect_uri="http://127.0.0.1:1234/api/publication/youtube/callback",
        authority_client_id="client-1",
        control_epoch=4,
        now=100.0,
    )
    query = urllib.parse.parse_qs(urllib.parse.urlparse(transaction.authorization_url).query)
    expected = base64.urlsafe_b64encode(
        hashlib.sha256(transaction.verifier.encode()).digest()
    ).decode().rstrip("=")

    assert query["code_challenge_method"] == ["S256"]
    assert query["code_challenge"] == [expected]
    assert query["state"] == [transaction.state]
    transaction.validate_authority("client-1", 4, now=101.0)
    with pytest.raises(YouTubePublicationError, match="stale"):
        transaction.validate_authority("client-2", 4, now=101.0)


def test_processed_video_must_match_admitted_principal_metadata_and_marker() -> None:
    video = {
        "snippet": {
            "channelId": "channel-1",
            "title": "Title",
            "description": "Description",
            "tags": ["gradlab-publication-fingerprint"],
        },
        "status": {"privacyStatus": "public"},
        "processingDetails": {"processingStatus": "succeeded"},
    }

    validate_processed_video(
        video,
        channel_id="channel-1",
        privacy="public",
        marker="gradlab-publication-fingerprint",
        title="Title",
        description="Description",
    )

    video["status"]["privacyStatus"] = "private"
    with pytest.raises(YouTubePublicationError, match="privacy"):
        validate_processed_video(
            video,
            channel_id="channel-1",
            privacy="public",
            marker="gradlab-publication-fingerprint",
            title="Title",
            description="Description",
        )


def test_oauth_does_not_invent_scopes_omitted_from_partial_grant(monkeypatch) -> None:
    transaction = new_oauth_transaction(
        CLIENT,
        redirect_uri="http://127.0.0.1:1234/api/publication/oauth/callback",
        authority_client_id="client-1",
        control_epoch=1,
    )
    monkeypatch.setattr(
        "gradlab.youtube_publication.post_form",
        lambda _url, _fields: {
            "access_token": "access",
            "refresh_token": "refresh",
            "scope": "https://www.googleapis.com/auth/youtube.upload",
        },
    )

    token = exchange_oauth_code(CLIENT, transaction, code="code")

    assert token["scopes"] == ["https://www.googleapis.com/auth/youtube.upload"]


def test_find_or_create_playlist_replaces_stale_listed_match(monkeypatch) -> None:
    client = YouTubeClient("access")
    calls: list[tuple[str, str]] = []

    def request(url, *, method="GET", payload=None, timeout=60.0):
        del payload, timeout
        calls.append((method, url))
        if "mine=true" in url:
            return {
                "items": [
                    {
                        "id": "stale-playlist",
                        "snippet": {"title": "gradlab"},
                    }
                ]
            }
        if "playlistItems" in url:
            raise YouTubePublicationError("playlist not found", status=404)
        assert method == "POST"
        return {"id": "replacement-playlist"}

    monkeypatch.setattr(client, "_request_json", request)

    assert client.find_or_create_playlist("gradlab", privacy="public") == (
        "replacement-playlist"
    )
    assert any(method == "POST" for method, _url in calls)


def test_final_metadata_update_sets_requested_privacy_without_reupload(monkeypatch) -> None:
    client = YouTubeClient("access")
    monkeypatch.setattr(
        client,
        "video_metadata",
        lambda _video_id: {
            "snippet": {"categoryId": "20"},
            "status": {"selfDeclaredMadeForKids": False},
        },
    )
    observed = {}

    def request(url, *, method="GET", payload=None, timeout=60.0):
        del timeout
        observed.update(url=url, method=method, payload=payload)
        return {"id": "video-id"}

    monkeypatch.setattr(client, "_request_json", request)
    client.update_video_metadata(
        video_id="video-id",
        title="Generated",
        description="Evidence-backed",
        tags=["gradlab"],
        privacy="public",
    )
    assert observed["method"] == "PUT"
    assert observed["payload"]["status"]["privacyStatus"] == "public"


def test_playlist_metadata_update_and_removal_use_existing_ids(monkeypatch) -> None:
    client = YouTubeClient("access")
    calls: list[tuple[str, str, object]] = []

    def fake_request(url: str, *, method: str = "GET", payload=None, timeout=60.0):
        calls.append((url, method, payload))
        if method == "GET":
            return {
                "items": [
                    {
                        "id": "playlist-1",
                        "snippet": {"title": "old"},
                        "status": {"privacyStatus": "public"},
                    }
                ]
            }
        return {}

    monkeypatch.setattr(client, "_request_json", fake_request)
    assert client.playlist_metadata("playlist-1")["id"] == "playlist-1"
    client.update_playlist_metadata(
        playlist_id="playlist-1",
        title="GradLab — Example-v0",
        description="Evidence-backed releases.",
        privacy="public",
    )
    client.remove_playlist_item("item-1")
    assert calls[1][1] == "PUT"
    assert calls[1][2]["id"] == "playlist-1"
    assert calls[2][1] == "DELETE"


def test_publication_thumbnail_is_consistent_16_by_9(
    monkeypatch, tmp_path: Path
) -> None:
    def frame(_video, output, *, seconds):
        del seconds
        Image.new("RGB", (640, 480), "navy").save(output)
        return output

    monkeypatch.setattr("gradlab.youtube_publication.extract_thumbnail", frame)
    output = create_publication_thumbnail(
        tmp_path / "replay.mp4",
        tmp_path / "thumbnail.jpg",
        seconds=10,
        task="Deathmatch score attack",
        trainer_algorithm="GradLab PPO",
        step="464M env steps",
        metric="Full-eval kills mean: 34.29",
    )
    with Image.open(output) as image:
        assert image.size == (1280, 720)
