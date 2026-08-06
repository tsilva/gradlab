from __future__ import annotations

import base64
import hashlib
import urllib.parse

import pytest

from gradlab.youtube_publication import (
    YouTubePublicationError,
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
