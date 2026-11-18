"""Explicit loopback transport for a host-provisioned YouTube relay."""

from __future__ import annotations

import io
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Mapping


PLACEHOLDER = "gradlab_youtube_proxy_placeholder"
API_ROUTES = {
    ("GET", "/youtube/v3/channels"),
    ("GET", "/youtube/v3/videos"),
    ("PUT", "/youtube/v3/videos"),
    ("GET", "/youtube/v3/playlists"),
    ("POST", "/youtube/v3/playlists"),
    ("GET", "/youtube/v3/playlistItems"),
    ("POST", "/youtube/v3/playlistItems"),
    ("POST", "/upload/youtube/v3/videos"),
    ("POST", "/upload/youtube/v3/thumbnails/set"),
}


class RelayError(OSError):
    """A fixed diagnostic that can be printed without provider details."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


@dataclass(frozen=True)
class YouTubeRelay:
    origin: str
    channel_id: str

    @classmethod
    def from_environment(cls, env: Mapping[str, str] | None = None) -> YouTubeRelay | None:
        settings = os.environ if env is None else env
        origin = settings.get("GRADLAB_YOUTUBE_RELAY")
        channel = settings.get("GRADLAB_YOUTUBE_CHANNEL_ID")
        placeholder = settings.get("GRADLAB_YOUTUBE_RELAY_TOKEN")
        if origin is None and channel is None and placeholder is None:
            return None
        match = re.fullmatch(r"http://127\.0\.0\.1:([1-9][0-9]{3,4})", origin or "")
        if (match is None or not 1024 <= int(match[1]) <= 65535
                or not re.fullmatch(r"UC[A-Za-z0-9_-]{22}", channel or "")
                or placeholder != PLACEHOLDER):
            raise RelayError("Invalid external YouTube relay configuration")
        return cls(origin, channel)

    def upload_handle(self, url: str) -> str:
        if not re.fullmatch(re.escape(self.origin) + r"/gradlab-upload/[a-f0-9]{48}", url):
            raise RelayError("YouTube relay did not return a local upload handle")
        return url

    def open(self, request: urllib.request.Request, *, timeout: float):
        method = request.get_method()
        parsed = urllib.parse.urlsplit(request.full_url)
        if (parsed.scheme, parsed.netloc) == ("https", "www.googleapis.com"):
            if (method, parsed.path) not in API_ROUTES or parsed.fragment:
                raise RelayError("YouTube request is outside the external relay routes")
            url = self.origin + parsed.path + ("?" + parsed.query if parsed.query else "")
        elif method == "PUT":
            url = self.upload_handle(request.full_url)
        else:
            raise RelayError("Unexpected external YouTube relay destination")

        headers = {key.lower(): value for key, value in request.header_items()}
        if headers.get("authorization") != "Bearer " + PLACEHOLDER or set(headers) - {
            "authorization", "content-type", "content-length", "content-range",
            "x-upload-content-length", "x-upload-content-type",
        }:
            raise RelayError("Unexpected YouTube relay request headers")
        selected = urllib.request.Request(url, data=request.data, headers=headers, method=method)
        try:
            return urllib.request.build_opener(NoRedirect()).open(selected, timeout=timeout)
        except urllib.error.HTTPError as error:
            # The upload client needs 308, but only for an exact acknowledged chunk.
            status = error.code
            acknowledged = {}
            if status == 308:
                sent = re.fullmatch(r"bytes [0-9]+-([0-9]+)/[0-9]+", headers.get("content-range", ""))
                received = error.headers.get_all("Range", [])
                if sent is None or received != [f"bytes=0-{sent[1]}"]:
                    status = 502
                else:
                    acknowledged["Range"] = received[0]
            error.close()
            raise urllib.error.HTTPError(self.origin, status, "", acknowledged, io.BytesIO()) from None
        except OSError:
            raise RelayError("YouTube relay is unavailable") from None


def check_channel() -> int:
    from gradlab.youtube_publication import YouTubeClient, YouTubePublicationError

    try:
        relay = YouTubeRelay.from_environment()
        if relay is None:
            return 1
        identity = YouTubeClient(PLACEHOLDER, open_request=relay.open).channel_identity()
        return 0 if identity['channel_id'] == relay.channel_id else 1
    except (YouTubePublicationError, RelayError):
        return 1
