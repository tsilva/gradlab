from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import secrets
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from PIL import Image, ImageDraw, ImageFont, ImageOps


YOUTUBE_SCOPES = (
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",
)
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
API_URL = "https://www.googleapis.com/youtube/v3"
UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
THUMBNAIL_URL = "https://www.googleapis.com/upload/youtube/v3/thumbnails/set"
UPLOAD_CHUNK_BYTES = 8 * 1024 * 1024


class YouTubePublicationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        status: int | None = None,
        code: str = "youtube_error",
    ) -> None:
        super().__init__(message)
        self.retryable = bool(retryable)
        self.status = status
        self.code = code


class YouTubeSubmissionUncertain(YouTubePublicationError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="submission_uncertain")


@dataclass(frozen=True)
class OAuthTransaction:
    state: str
    verifier: str
    redirect_uri: str
    authorization_url: str
    expires_at: float
    authority_client_id: str
    control_epoch: int

    def validate_authority(self, client_id: str, epoch: int, *, now: float | None = None) -> None:
        if (now if now is not None else time.time()) >= self.expires_at:
            raise YouTubePublicationError("YouTube authorization expired", code="oauth_expired")
        if self.authority_client_id != client_id or self.control_epoch != int(epoch):
            raise YouTubePublicationError(
                "YouTube authorization belongs to a stale player authority",
                code="oauth_authority_changed",
            )


def _installed_client(client_config: Mapping[str, Any]) -> Mapping[str, Any]:
    installed = client_config.get("installed")
    if not isinstance(installed, Mapping):
        installed = client_config.get("web")
    if not isinstance(installed, Mapping):
        raise YouTubePublicationError("YouTube client configuration is invalid")
    if not str(installed.get("client_id") or "") or not str(installed.get("client_secret") or ""):
        raise YouTubePublicationError("YouTube client configuration is incomplete")
    return installed


def new_oauth_transaction(
    client_config: Mapping[str, Any],
    *,
    redirect_uri: str,
    authority_client_id: str,
    control_epoch: int,
    now: float | None = None,
) -> OAuthTransaction:
    installed = _installed_client(client_config)
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip(
        "="
    )
    state = secrets.token_urlsafe(32)
    params = {
        "client_id": str(installed["client_id"]),
        "redirect_uri": str(redirect_uri),
        "response_type": "code",
        "scope": " ".join(YOUTUBE_SCOPES),
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    timestamp = time.time() if now is None else float(now)
    return OAuthTransaction(
        state=state,
        verifier=verifier,
        redirect_uri=str(redirect_uri),
        authorization_url=f"{AUTH_URL}?{urllib.parse.urlencode(params)}",
        expires_at=timestamp + 600.0,
        authority_client_id=str(authority_client_id),
        control_epoch=int(control_epoch),
    )


def _http_error(exc: urllib.error.HTTPError, *, operation: str) -> YouTubePublicationError:
    body = exc.read(4096).decode("utf-8", "replace")
    retryable = exc.code == 429 or 500 <= exc.code < 600
    return YouTubePublicationError(
        f"YouTube {operation} failed with HTTP {exc.code}: {body}",
        retryable=retryable,
        status=exc.code,
    )


def post_form(url: str, fields: Mapping[str, str]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=urllib.parse.urlencode(dict(fields)).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise _http_error(exc, operation="OAuth") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise YouTubePublicationError(
            f"YouTube OAuth response failed: {exc}", retryable=True
        ) from exc
    if not isinstance(value, dict):
        raise YouTubePublicationError("YouTube OAuth response is not a JSON object")
    return value


def exchange_oauth_code(
    client_config: Mapping[str, Any],
    transaction: OAuthTransaction,
    *,
    code: str,
) -> dict[str, Any]:
    installed = _installed_client(client_config)
    token = post_form(
        TOKEN_URL,
        {
            "code": str(code),
            "client_id": str(installed["client_id"]),
            "client_secret": str(installed["client_secret"]),
            "redirect_uri": transaction.redirect_uri,
            "grant_type": "authorization_code",
            "code_verifier": transaction.verifier,
        },
    )
    token.pop("client_secret", None)
    token["client_id"] = str(installed["client_id"])
    token["token_uri"] = TOKEN_URL
    returned_scope = token.get("scope")
    token["scopes"] = sorted(
        set(str(returned_scope).split()) if returned_scope else set(YOUTUBE_SCOPES)
    )
    return token


def refresh_access_token(
    client_config: Mapping[str, Any],
    token: Mapping[str, Any],
) -> dict[str, Any]:
    installed = _installed_client(client_config)
    refresh_token = str(token.get("refresh_token") or "")
    if not refresh_token:
        raise YouTubePublicationError("YouTube reauthorization is required", code="reauthorize")
    refreshed = post_form(
        TOKEN_URL,
        {
            "client_id": str(installed["client_id"]),
            "client_secret": str(installed["client_secret"]),
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
    )
    return {
        **dict(token),
        **refreshed,
        "refresh_token": refresh_token,
        "client_id": str(installed["client_id"]),
        "token_uri": TOKEN_URL,
        "scopes": sorted(set(token.get("scopes") or ())),
    }


class YouTubeClient:
    def __init__(self, access_token: str) -> None:
        token = str(access_token).strip()
        if not token:
            raise YouTubePublicationError("YouTube access token is empty")
        self._token = token

    def _request_json(
        self,
        url: str,
        *,
        method: str = "GET",
        payload: Mapping[str, Any] | None = None,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(dict(payload)).encode()
        headers = {"Authorization": f"Bearer {self._token}"}
        if data is not None:
            headers["Content-Type"] = "application/json; charset=UTF-8"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            raise _http_error(exc, operation="API request") from exc
        except OSError as exc:
            raise YouTubePublicationError(
                f"YouTube API request failed: {exc}", retryable=True
            ) from exc
        try:
            value = json.loads(body or b"{}")
        except json.JSONDecodeError as exc:
            raise YouTubePublicationError(
                "YouTube API returned invalid JSON", retryable=True
            ) from exc
        if not isinstance(value, dict):
            raise YouTubePublicationError("YouTube API response is not a JSON object")
        return value

    def channel_identity(self) -> dict[str, Any]:
        params = urllib.parse.urlencode({"part": "snippet", "mine": "true"})
        value = self._request_json(f"{API_URL}/channels?{params}")
        items = value.get("items")
        if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], Mapping):
            raise YouTubePublicationError("YouTube login must resolve exactly one channel")
        channel = items[0]
        snippet = channel.get("snippet") if isinstance(channel.get("snippet"), Mapping) else {}
        return {
            "channel_id": str(channel.get("id") or ""),
            "channel_title": str(snippet.get("title") or ""),
            "scopes": list(YOUTUBE_SCOPES),
        }

    def start_resumable_upload(
        self,
        *,
        video_path: Path,
        title: str,
        description: str,
        tags: list[str],
        privacy: str,
        category_id: str = "20",
    ) -> str:
        metadata = {
            "snippet": {
                "title": str(title),
                "description": str(description),
                "tags": list(tags),
                "categoryId": str(category_id),
            },
            "status": {"privacyStatus": str(privacy), "selfDeclaredMadeForKids": False},
        }
        data = json.dumps(metadata).encode()
        mime = mimetypes.guess_type(video_path.name)[0] or "video/mp4"
        url = f"{UPLOAD_URL}?{urllib.parse.urlencode({'uploadType': 'resumable', 'part': 'snippet,status'})}"
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json; charset=UTF-8",
                "Content-Length": str(len(data)),
                "X-Upload-Content-Length": str(video_path.stat().st_size),
                "X-Upload-Content-Type": mime,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                location = response.headers.get("Location")
        except urllib.error.HTTPError as exc:
            raise _http_error(exc, operation="upload start") from exc
        except OSError as exc:
            raise YouTubePublicationError(
                f"YouTube upload start failed: {exc}", retryable=True
            ) from exc
        if not location:
            raise YouTubePublicationError("YouTube did not return a resumable session")
        return str(location)

    def query_resumable(self, session_uri: str, *, total_bytes: int) -> int | dict[str, Any]:
        request = urllib.request.Request(
            session_uri,
            data=b"",
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Length": "0",
                "Content-Range": f"bytes */{int(total_bytes)}",
            },
            method="PUT",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read()
                return json.loads(body or b"{}")
        except urllib.error.HTTPError as exc:
            if exc.code == 308:
                uploaded = exc.headers.get("Range", "")
                _prefix, _separator, end = uploaded.rpartition("-")
                return int(end) + 1 if end.isdigit() else 0
            if exc.code in {404, 410}:
                raise YouTubePublicationError(
                    "YouTube resumable session expired", status=exc.code, code="session_expired"
                ) from exc
            raise _http_error(exc, operation="upload reconciliation") from exc
        except OSError as exc:
            raise YouTubePublicationError(
                f"YouTube upload reconciliation failed: {exc}", retryable=True
            ) from exc

    def upload_chunks(
        self,
        session_uri: str,
        video_path: Path,
        *,
        offset: int = 0,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        total = video_path.stat().st_size
        mime = mimetypes.guess_type(video_path.name)[0] or "video/mp4"
        with video_path.open("rb") as stream:
            stream.seek(int(offset))
            current = int(offset)
            while current < total:
                chunk = stream.read(min(UPLOAD_CHUNK_BYTES, total - current))
                if not chunk:
                    raise YouTubePublicationError("video ended before its declared size")
                final = current + len(chunk) == total
                request = urllib.request.Request(
                    session_uri,
                    data=chunk,
                    headers={
                        "Authorization": f"Bearer {self._token}",
                        "Content-Type": mime,
                        "Content-Length": str(len(chunk)),
                        "Content-Range": f"bytes {current}-{current + len(chunk) - 1}/{total}",
                    },
                    method="PUT",
                )
                try:
                    with urllib.request.urlopen(request, timeout=600) as response:
                        body = response.read()
                        value = json.loads(body or b"{}")
                except urllib.error.HTTPError as exc:
                    if exc.code == 308:
                        current += len(chunk)
                        if progress is not None:
                            progress(current, total)
                        continue
                    if final and (exc.code == 429 or exc.code >= 500):
                        raise YouTubeSubmissionUncertain(
                            "final YouTube chunk outcome is uncertain; reconcile before retry"
                        ) from exc
                    raise _http_error(exc, operation="upload") from exc
                except OSError as exc:
                    if final:
                        raise YouTubeSubmissionUncertain(
                            "final YouTube chunk outcome is uncertain; reconcile before retry"
                        ) from exc
                    raise YouTubePublicationError(
                        f"YouTube upload failed: {exc}", retryable=True
                    ) from exc
                current += len(chunk)
                if progress is not None:
                    progress(current, total)
                if final:
                    if not isinstance(value, dict) or not str(value.get("id") or ""):
                        raise YouTubeSubmissionUncertain(
                            "YouTube accepted final bytes without returning a video id"
                        )
                    return value
        raise YouTubePublicationError("YouTube upload did not produce a video")

    def video_metadata(self, video_id: str) -> dict[str, Any]:
        params = urllib.parse.urlencode(
            {"part": "snippet,status,processingDetails", "id": str(video_id)}
        )
        value = self._request_json(f"{API_URL}/videos?{params}")
        items = value.get("items")
        if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], Mapping):
            raise YouTubePublicationError("YouTube video is missing or inaccessible")
        return dict(items[0])

    def update_video_metadata(
        self,
        *,
        video_id: str,
        title: str,
        description: str,
        tags: list[str],
        privacy: str,
    ) -> dict[str, Any]:
        current = self.video_metadata(video_id)
        snippet = current.get("snippet") if isinstance(current.get("snippet"), Mapping) else {}
        status = current.get("status") if isinstance(current.get("status"), Mapping) else {}
        return self._request_json(
            f"{API_URL}/videos?{urllib.parse.urlencode({'part': 'snippet,status'})}",
            method="PUT",
            payload={
                "id": video_id,
                "snippet": {
                    "title": str(title),
                    "description": str(description),
                    "tags": list(tags),
                    "categoryId": str(snippet.get("categoryId") or "20"),
                },
                "status": {
                    "privacyStatus": str(privacy),
                    "selfDeclaredMadeForKids": bool(
                        status.get("selfDeclaredMadeForKids", False)
                    ),
                },
            },
        )

    def find_playlist(self, title: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        page_token = ""
        while True:
            params = {"part": "snippet,status", "mine": "true", "maxResults": "50"}
            if page_token:
                params["pageToken"] = page_token
            value = self._request_json(f"{API_URL}/playlists?{urllib.parse.urlencode(params)}")
            for item in value.get("items") or ():
                if isinstance(item, Mapping) and str((item.get("snippet") or {}).get("title")) == title:
                    results.append(dict(item))
            page_token = str(value.get("nextPageToken") or "")
            if not page_token:
                return results

    def find_or_create_playlist(self, title: str, *, privacy: str) -> str:
        matches = self.find_playlist(title)
        accessible: list[str] = []
        for match in matches:
            playlist_id = str(match.get("id") or "")
            if not playlist_id:
                continue
            params = {
                "part": "id",
                "playlistId": playlist_id,
                "maxResults": "1",
            }
            try:
                self._request_json(
                    f"{API_URL}/playlistItems?{urllib.parse.urlencode(params)}"
                )
            except YouTubePublicationError as exc:
                if exc.status == 404:
                    continue
                raise
            accessible.append(playlist_id)
        if len(accessible) > 1:
            raise YouTubePublicationError(
                f"multiple YouTube playlists have the admitted title {title!r}"
            )
        if accessible:
            return accessible[0]
        value = self._request_json(
            f"{API_URL}/playlists?{urllib.parse.urlencode({'part': 'snippet,status'})}",
            method="POST",
            payload={
                "snippet": {
                    "title": title,
                    "description": "Reinforcement learning lab videos and model previews.",
                },
                "status": {"privacyStatus": privacy},
            },
        )
        playlist_id = str(value.get("id") or "")
        if not playlist_id:
            raise YouTubePublicationError("YouTube did not return the created playlist id")
        return playlist_id

    def playlist_metadata(self, playlist_id: str) -> dict[str, Any]:
        params = {
            "part": "snippet,status",
            "id": str(playlist_id),
            "maxResults": "1",
        }
        value = self._request_json(f"{API_URL}/playlists?{urllib.parse.urlencode(params)}")
        items = value.get("items")
        if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], Mapping):
            raise YouTubePublicationError(
                f"YouTube playlist {playlist_id!r} was not found",
                status=404,
            )
        return dict(items[0])

    def update_playlist_metadata(
        self,
        *,
        playlist_id: str,
        title: str,
        description: str,
        privacy: str,
    ) -> dict[str, Any]:
        return self._request_json(
            f"{API_URL}/playlists?{urllib.parse.urlencode({'part': 'snippet,status'})}",
            method="PUT",
            payload={
                "id": str(playlist_id),
                "snippet": {"title": str(title), "description": str(description)},
                "status": {"privacyStatus": str(privacy)},
            },
        )

    def playlist_items(self, playlist_id: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        page_token = ""
        while True:
            params = {
                "part": "snippet,contentDetails,status",
                "playlistId": str(playlist_id),
                "maxResults": "50",
            }
            if page_token:
                params["pageToken"] = page_token
            value = self._request_json(
                f"{API_URL}/playlistItems?{urllib.parse.urlencode(params)}"
            )
            results.extend(
                dict(item)
                for item in value.get("items") or ()
                if isinstance(item, Mapping)
            )
            page_token = str(value.get("nextPageToken") or "")
            if not page_token:
                return results

    def remove_playlist_item(self, playlist_item_id: str) -> None:
        self._request_json(
            f"{API_URL}/playlistItems?{urllib.parse.urlencode({'id': str(playlist_item_id)})}",
            method="DELETE",
        )

    def add_video_to_playlist(self, *, playlist_id: str, video_id: str) -> dict[str, Any]:
        page_token = ""
        while True:
            params = {
                "part": "snippet,contentDetails",
                "playlistId": playlist_id,
                "maxResults": "50",
            }
            if page_token:
                params["pageToken"] = page_token
            value = self._request_json(
                f"{API_URL}/playlistItems?{urllib.parse.urlencode(params)}"
            )
            for item in value.get("items") or ():
                if (
                    isinstance(item, Mapping)
                    and str((item.get("contentDetails") or {}).get("videoId") or "")
                    == video_id
                ):
                    return {
                        "already_present": True,
                        "playlist_item_id": str(item.get("id") or ""),
                    }
            page_token = str(value.get("nextPageToken") or "")
            if not page_token:
                break
        value = self._request_json(
            f"{API_URL}/playlistItems?{urllib.parse.urlencode({'part': 'snippet'})}",
            method="POST",
            payload={
                "snippet": {
                    "playlistId": playlist_id,
                    "resourceId": {"kind": "youtube#video", "videoId": video_id},
                }
            },
        )
        return {
            "already_present": False,
            "playlist_item_id": str(value.get("id") or ""),
        }

    def upload_thumbnail(self, *, video_id: str, thumbnail_path: Path) -> dict[str, Any]:
        data = thumbnail_path.read_bytes()
        if not data:
            raise YouTubePublicationError("generated YouTube thumbnail is empty")
        mime = mimetypes.guess_type(thumbnail_path.name)[0] or "image/jpeg"
        request = urllib.request.Request(
            f"{THUMBNAIL_URL}?{urllib.parse.urlencode({'videoId': video_id})}",
            data=data,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": mime,
                "Content-Length": str(len(data)),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                value = json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as exc:
            raise _http_error(exc, operation="thumbnail upload") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise YouTubePublicationError(
                f"YouTube thumbnail upload failed: {exc}", retryable=True
            ) from exc
        return dict(value) if isinstance(value, Mapping) else {}


def extract_thumbnail(video_path: Path, output: Path, *, seconds: float) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{max(0.0, float(seconds)):.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise YouTubePublicationError("ffmpeg is required to generate a thumbnail") from exc
    except subprocess.CalledProcessError as exc:
        raise YouTubePublicationError(
            f"could not generate YouTube thumbnail: {exc.stderr.strip()}"
        ) from exc
    if not output.is_file() or output.stat().st_size < 1:
        raise YouTubePublicationError("ffmpeg did not write a YouTube thumbnail")
    return output


def create_publication_thumbnail(
    video_path: Path,
    output: Path,
    *,
    seconds: float,
    task: str,
    trainer_algorithm: str,
    step: str,
    metric: str,
) -> Path:
    source = output.with_name(f".{output.stem}-frame.jpg")
    extract_thumbnail(video_path, source, seconds=seconds)
    try:
        with Image.open(source) as raw:
            frame = ImageOps.fit(raw.convert("RGB"), (1280, 720), method=Image.Resampling.LANCZOS)
        overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rectangle((0, 0, 1280, 150), fill=(8, 14, 24, 225))
        draw.rectangle((0, 545, 1280, 720), fill=(8, 14, 24, 235))
        font_paths = (
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        )
        font_path = next((path for path in font_paths if Path(path).is_file()), None)
        title_font = ImageFont.truetype(font_path, 52) if font_path else ImageFont.load_default()
        body_font = ImageFont.truetype(font_path, 34) if font_path else ImageFont.load_default()
        small_font = ImageFont.truetype(font_path, 27) if font_path else ImageFont.load_default()
        draw.text((48, 28), str(task), fill=(255, 255, 255, 255), font=title_font)
        draw.text(
            (48, 92),
            f"{trainer_algorithm}  •  {step}",
            fill=(113, 221, 255, 255),
            font=body_font,
        )
        draw.text((48, 570), str(metric), fill=(255, 255, 255, 255), font=small_font)
        draw.text((48, 660), "GRADLAB RESEARCH", fill=(113, 221, 255, 255), font=small_font)
        composed = Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB")
        output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        composed.save(output, format="JPEG", quality=92, optimize=True)
    finally:
        source.unlink(missing_ok=True)
    if output.stat().st_size > 2_000_000:
        with Image.open(output) as image:
            image.save(output, format="JPEG", quality=82, optimize=True)
    return output


def validate_processed_video(
    video: Mapping[str, Any],
    *,
    channel_id: str,
    privacy: str,
    marker: str,
    title: str,
    description: str,
) -> None:
    snippet = video.get("snippet") if isinstance(video.get("snippet"), Mapping) else {}
    status = video.get("status") if isinstance(video.get("status"), Mapping) else {}
    processing = (
        video.get("processingDetails")
        if isinstance(video.get("processingDetails"), Mapping)
        else {}
    )
    state = str(processing.get("processingStatus") or "")
    if state != "succeeded":
        if state == "failed":
            raise YouTubePublicationError(
                f"YouTube processing failed: {processing.get('processingFailureReason') or 'unknown'}",
                code="processing_failed",
            )
        raise YouTubePublicationError(
            f"YouTube processing is not complete: {state or 'unknown'}",
            retryable=True,
            code="processing_pending",
        )
    if str(snippet.get("channelId") or "") != str(channel_id):
        raise YouTubePublicationError("YouTube video belongs to a different channel")
    if str(status.get("privacyStatus") or "") != str(privacy):
        raise YouTubePublicationError("YouTube changed the requested privacy status")
    if str(snippet.get("title") or "") != str(title):
        raise YouTubePublicationError("YouTube video title does not match the admitted request")
    if str(snippet.get("description") or "") != str(description):
        raise YouTubePublicationError(
            "YouTube video description does not match the admitted request"
        )
    tags = {str(value) for value in snippet.get("tags") or ()}
    if marker not in tags:
        raise YouTubePublicationError("YouTube video is missing its publication marker")


__all__ = [
    "OAuthTransaction",
    "UPLOAD_CHUNK_BYTES",
    "YOUTUBE_SCOPES",
    "YouTubeClient",
    "YouTubePublicationError",
    "YouTubeSubmissionUncertain",
    "exchange_oauth_code",
    "create_publication_thumbnail",
    "extract_thumbnail",
    "new_oauth_transaction",
    "refresh_access_token",
    "validate_processed_video",
]
