from __future__ import annotations

import argparse
import asyncio
import json
import queue
import secrets
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
from aiohttp import WSMsgType, web

from gradlab.play_browser import PlaybackBrowser
from gradlab.play_dev_assets import PlayerDevAssets, development_page, source_checkout_root
from gradlab.play_engine import (
    FRAME_ATTRIBUTION,
    FRAME_CNN_INSPECTION,
    FRAME_CODEC_PNG,
    FRAME_GAME,
    FRAME_HEADER,
    FRAME_MAGIC,
    FRAME_OBSERVATION,
    HISTORY_LIMIT,
    INSPECTION_FRAME_WAIT_SECONDS,
    PROTOCOL_VERSION,
    DatasetPlaybackRunner,
    HumanRecordingRunner,
    PlaybackCommand,
    WebPlaybackRunner,
    history_point,
    history_point_payload,
    playback_updates,
    reward_accounting_contract,
    transition_payload,
)
from gradlab.play_processing import (
    PLAYER_PROCESSING_FEATURES,
    normalize_player_processing,
)
from gradlab.play_session import (
    _PlaybackSession,
)
from gradlab.publication_credentials import (
    credential_lock,
    load_private_json,
    save_private_json,
    youtube_credential_paths,
)
from gradlab.youtube_publication import (
    OAuthTransaction,
    exchange_oauth_code,
    new_oauth_transaction,
)

CLIENT_QUEUE_LIMIT = 64
LAST_CLIENT_GRACE_SECONDS = 30.0
FRAME_SUBSCRIPTIONS = {
    FRAME_GAME: "game",
    FRAME_OBSERVATION: "observation",
    FRAME_ATTRIBUTION: "attribution",
    FRAME_CNN_INSPECTION: "cnn-inspection",
}


def source_browser_path(route: Mapping[str, Any] | None) -> str:
    route = route or {}
    environment_id = str(route.get("environment_id") or "").strip()
    goal_id = str(route.get("goal_id") or "").strip()
    goal_variant_id = str(route.get("goal_variant_id") or "").strip()
    run_id = str(route.get("run_id") or "").strip()
    checkpoint_id = str(route.get("checkpoint_id") or "").strip()
    if not environment_id:
        return "/"
    path = f"/environments/{quote(environment_id, safe='')}"
    if not goal_id:
        return path
    path += f"/goals/{quote(goal_id, safe='')}"
    if not goal_variant_id or not run_id:
        return path
    path += f"/variants/{quote(goal_variant_id, safe='')}"
    path += f"/runs/{quote(run_id, safe='')}"
    if not checkpoint_id:
        return path
    return f"{path}/checkpoints/{quote(checkpoint_id, safe='')}"


class WebClient:
    def __init__(
        self,
        client_id: str,
        socket: web.WebSocketResponse,
        subscriptions: set[str],
        workspace_id: str,
        window_id: str,
        processing: frozenset[str] = PLAYER_PROCESSING_FEATURES,
    ) -> None:
        self.client_id = client_id
        self.socket = socket
        self.subscriptions = subscriptions
        self.processing = normalize_player_processing(processing)
        self.workspace_id = workspace_id
        self.window_id = window_id
        self.reliable: asyncio.Queue[str | bytes] = asyncio.Queue(CLIENT_QUEUE_LIMIT)
        self.event = asyncio.Event()
        self.pending_snapshots: deque[tuple[tuple[int, int, int, int], str]] = deque(
            maxlen=HISTORY_LIMIT
        )
        self.latest_snapshot_key: tuple[int, int, int, int] = (-1, -1, -1, -1)
        self.sent_snapshot_key: tuple[int, int, int, int] = (-1, -1, -1, -1)
        self.latest_frames: dict[int, tuple[int, bytes]] = {}
        self.sent_frames: dict[int, tuple[int, bytes]] = {}
        self.closed = False
        self.recorded_navigation = False
        self.latest_source_phase: str | None = None

    def offer_reliable(self, payload: Mapping[str, Any] | bytes) -> None:
        rendered = (
            payload
            if isinstance(payload, bytes)
            else json.dumps(payload, separators=(",", ":"), allow_nan=False)
        )
        try:
            self.reliable.put_nowait(rendered)
        except asyncio.QueueFull:
            self.closed = True
        self.event.set()

    def offer_snapshot(self, payload: Mapping[str, Any]) -> None:
        self.recorded_navigation = payload.get("mode") == "trajectory"
        source_phase = (payload.get("app") or {}).get("phase")
        phase_changed = source_phase != self.latest_source_phase
        key = (
            int(payload.get("session_epoch", 0)),
            int(payload.get("revision", 0)),
            int(payload.get("sequence", 0)),
            int(payload.get("control_epoch", 0)),
        )
        # Preparation and the active runner own independent revision counters.
        # A source-phase transition must reach viewers even with a lower revision.
        if key[0] < self.latest_snapshot_key[0] or (
            not phase_changed and key < self.latest_snapshot_key
        ):
            return
        rendered = json.dumps(payload, separators=(",", ":"), allow_nan=False)
        if self.pending_snapshots and key == self.pending_snapshots[-1][0]:
            self.pending_snapshots[-1] = (key, rendered)
        elif phase_changed or key > self.latest_snapshot_key:
            self.pending_snapshots.append((key, rendered))
        else:
            return
        self.latest_snapshot_key = key
        self.latest_source_phase = source_phase
        self.event.set()

    def offer_frame(self, kind: int, sequence: int, packet: bytes) -> None:
        if self.recorded_navigation or sequence >= self.latest_frames.get(kind, (-1, b""))[0]:
            self.latest_frames[kind] = (sequence, packet)
            self.event.set()

    def reset_session(self, epoch: int) -> None:
        self.pending_snapshots.clear()
        self.latest_snapshot_key = (int(epoch), -1, -1, -1)
        self.sent_snapshot_key = (int(epoch), -1, -1, -1)
        self.latest_frames.clear()
        self.sent_frames.clear()
        self.event.set()

    async def write(self) -> None:
        while not self.closed and not self.socket.closed:
            await self.event.wait()
            self.event.clear()
            while not self.reliable.empty():
                value = self.reliable.get_nowait()
                if isinstance(value, bytes):
                    if (
                        len(value) > FRAME_HEADER.size
                        and value[:4] == b"RLP3"
                        and value[4] == FRAME_GAME
                        and "game" not in self.subscriptions
                    ):
                        continue
                    await self.socket.send_bytes(value)
                else:
                    await self.socket.send_str(value)
            while self.pending_snapshots:
                key, snapshot = self.pending_snapshots.popleft()
                await self.socket.send_str(snapshot)
                self.sent_snapshot_key = key
            for kind, (sequence, packet) in tuple(self.latest_frames.items()):
                if kind == FRAME_GAME and "game" not in self.subscriptions:
                    continue
                if (sequence, packet) != self.sent_frames.get(kind):
                    await self.socket.send_bytes(packet)
                    self.sent_frames[kind] = (sequence, packet)


class PlaybackWebServer:
    def __init__(
        self,
        runner: Any,
        args: argparse.Namespace,
        *,
        paired_windows: bool = False,
        catalog: Any | None = None,
        manual_evaluation_factory: Any | None = None,
        repo_root: Path | None = None,
        publication_factory: Any | None = None,
    ) -> None:
        from gradlab.play_chart_transport import ChartResponses

        self._chart_responses = ChartResponses()
        self.runner = runner
        from gradlab.play_diagnostics import DiagnosticReader, DirectDiagnosticReader

        self._diagnostic_reader: DiagnosticReader = (
            DirectDiagnosticReader(getattr(runner, "diagnostics", None), self._runner_epoch)
            if isinstance(runner, (WebPlaybackRunner, DatasetPlaybackRunner))
            or not hasattr(runner, "read_diagnostics")
            else runner
        )
        self.args = args
        checkout_root = source_checkout_root()
        self.dev_assets = (
            PlayerDevAssets(checkout_root)
            if bool(getattr(args, "hot_reload", False)) and checkout_root is not None
            else None
        )
        self.paired_windows = paired_windows
        self.catalog = catalog
        self.token = secrets.token_urlsafe(32)
        self.origin = ""
        self.clients: dict[str, WebClient] = {}
        self.control_holder: str | None = None
        self.input_holder: str | None = None
        self.control_epoch = 0
        self.publication_authority_client_id: str | None = None
        self.publication_capability: str | None = None
        self.stop_event = asyncio.Event()
        self.desktop_browser: PlaybackBrowser | None = None
        self.desktop_peers: set[web.WebSocketResponse] = set()
        self.ever_connected = False
        self.last_client_at = time.monotonic()
        self._observed_session_change = int(getattr(self.runner, "session_change", 0))
        self._initial_environment_catalog: dict[str, Any] | None = None
        self._manual_evaluation_factory = manual_evaluation_factory
        self._manual_evaluation_queue: Any | None = None
        self._repo_root = None if repo_root is None else Path(repo_root).resolve()
        self._publication_factory = publication_factory
        self._publication_service: Any | None = None
        self._oauth_transactions: dict[str, OAuthTransaction] = {}
        self._media_tickets: dict[str, tuple[str, int, float]] = {}
        from gradlab.play_trajectory_http import TrajectoryTransfers

        self.trajectory_transfers = TrajectoryTransfers(
            runner,
            self._authorize_api,
            self._authorize_trajectory_import,
        )

    def _authorize_trajectory_import(self, request: web.Request) -> None:
        self._authorize_api(request)
        client = self.clients.get(request.headers.get("X-Gradlab-Client", ""))
        if (
            request.headers.get("Origin") != self.origin
            or client is None
            or client.workspace_id != self.control_holder
            or request.headers.get("X-Gradlab-Control-Epoch") != str(self.control_epoch)
        ):
            raise web.HTTPForbidden(text="current Playback control lease required")

    async def _sync_player_processing(self) -> None:
        configure = getattr(self.runner, "set_processing", None)
        if not callable(configure):
            return
        features = {feature for client in self.clients.values() for feature in client.processing}
        await asyncio.to_thread(configure, features)

    @property
    def asset_root(self) -> Path:
        root = Path(__file__).with_name("web_player")
        return root if self.dev_assets is not None else root / "dist"

    def require_player_assets(self) -> None:
        if self.dev_assets is not None:
            return
        root = self.asset_root
        required = ("index.html", "app.js", "styles.css", "vendor/gridstack/gridstack-all.js")
        missing = [name for name in required if not (root / name).is_file()]
        manifest = root / "build-manifest.json"
        if manifest.is_file():
            try:
                outputs = json.loads(manifest.read_text())["outputs"]
                missing.extend(name for name in outputs if not (root / name).is_file())
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise RuntimeError(f"Player web asset manifest is invalid: {manifest}") from exc
        if missing:
            raise RuntimeError(
                "Player web assets are missing (" + ", ".join(sorted(set(missing))) + "). "
                "From a source checkout, run pnpm install --frozen-lockfile && pnpm build:web "
                "before gradlab play; otherwise reinstall GradLab."
            )

    def dashboard_urls(self) -> tuple[str, ...]:
        main_path = "/"
        if self.catalog is not None:
            snapshot = self.runner.snapshot()
            app = snapshot.get("app") if isinstance(snapshot, Mapping) else None
            route = app.get("route") if isinstance(app, Mapping) else None
            main_path = source_browser_path(route if isinstance(route, Mapping) else None)
        if self.paired_windows:
            query = "?workspace=paired"
            return (
                f"{self.origin}{main_path}{query}#token={self.token}",
                f"{self.origin}/workspace/stats{query}#token={self.token}",
            )
        return (f"{self.origin}{main_path}#token={self.token}",)

    @web.middleware
    async def security_headers(self, request: web.Request, handler):
        response = await handler(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        vite_url = self.dev_assets.url if self.dev_assets is not None else None
        if vite_url:
            vite_ws = vite_url.replace("http://", "ws://", 1)
            response.headers["Content-Security-Policy"] = (
                f"default-src 'self'; script-src 'self' {vite_url}; "
                f"style-src 'self' {vite_url} 'unsafe-inline'; img-src 'self' blob:; "
                f"connect-src 'self' ws: wss: {vite_url} {vite_ws}; "
                "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
            )
        else:
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; "
                "connect-src 'self' ws: wss:; object-src 'none'; base-uri 'none'; "
                "frame-ancestors 'none'"
            )
        return response

    async def page(self, _request: web.Request) -> web.StreamResponse:
        if self.dev_assets is not None:
            assert self.dev_assets.url is not None
            markup = (self.asset_root / "index.html").read_text()
            response = web.Response(
                text=development_page(markup, self.dev_assets.url),
                content_type="text/html",
            )
        else:
            response = web.FileResponse(self.asset_root / "index.html")
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin-allow-popups"
        return response

    async def desktop_window(self, request: web.Request) -> web.Response:
        self._authorize_api(request)
        if self.desktop_browser is None:
            raise web.HTTPConflict(text="desktop viewer is not active")
        payload = await request.json()
        window = str(payload.get("window", ""))
        if (
            not window
            or len(window) > 64
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in window)
        ):
            raise web.HTTPBadRequest(text="invalid workspace window")
        query = "?workspace=paired" if self.paired_windows else ""
        url = (
            self.dashboard_urls()[0]
            if window == "main"
            else (f"{self.origin}/workspace/{window}{query}#token={self.token}")
        )
        await self.desktop_browser.open(url, window)
        return web.json_response({"opened": window})

    async def desktop_peer(self, request: web.Request) -> web.WebSocketResponse:
        """Relay workspace messages across native webview processes, including frames."""
        if request.headers.get("Origin") != self.origin or self.desktop_browser is None:
            raise web.HTTPForbidden()
        socket = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024)
        await socket.prepare(request)
        try:
            hello = await socket.receive_json(timeout=5)
            if not secrets.compare_digest(str(hello.get("token", "")), self.token):
                await socket.close(code=1008, message=b"invalid token")
                return socket
            self.desktop_peers.add(socket)
            await socket.send_json({"ready": True})
            async for message in socket:
                if message.type != WSMsgType.TEXT:
                    continue
                for peer in tuple(self.desktop_peers - {socket}):
                    try:
                        async with asyncio.timeout(2):
                            await peer.send_str(message.data)
                    except TimeoutError, ConnectionError:
                        await peer.close()
        finally:
            self.desktop_peers.discard(socket)
            await socket.close()
        return socket

    async def publication_oauth_complete(self, _request: web.Request) -> web.FileResponse:
        response = web.FileResponse(self.asset_root / "oauth_complete.html")
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin-allow-popups"
        return response

    async def asset(self, request: web.Request) -> web.FileResponse:
        if request.match_info["path"] in {"viewer-player.png", "viewer-stats.png"}:
            name = request.match_info["path"].removeprefix("viewer-")
            return web.FileResponse(Path(__file__).with_name("desktop") / name)
        relative = Path(request.match_info["path"])
        root = self.asset_root.resolve()
        candidate = (root / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or candidate.suffix not in {".js", ".css", ".svg", ".woff2"}
            or not candidate.is_relative_to(root)
            or not candidate.is_file()
        ):
            raise web.HTTPNotFound()
        return web.FileResponse(candidate)

    def _authorize_api(self, request: web.Request) -> None:
        origin = request.headers.get("Origin")
        if origin and origin != self.origin:
            raise web.HTTPForbidden(text="invalid request origin")
        authorization = request.headers.get("Authorization", "")
        scheme, _, supplied = authorization.partition(" ")
        if scheme.casefold() != "bearer" or not secrets.compare_digest(
            supplied,
            self.token,
        ):
            raise web.HTTPUnauthorized(text="catalog token required")

    def _authorize_publication(
        self,
        request: web.Request,
        *,
        mutation: bool = False,
    ) -> WebClient:
        origin = request.headers.get("Origin")
        if origin and origin != self.origin:
            raise web.HTTPForbidden(text="exact player origin required")
        if mutation and origin != self.origin:
            raise web.HTTPForbidden(text="exact player origin required for publication mutation")
        self._authorize_api(request)
        client_id = request.headers.get("X-Gradlab-Client", "")
        client = self.clients.get(client_id)
        try:
            epoch = int(request.headers.get("X-Gradlab-Control-Epoch", ""))
        except ValueError as exc:
            raise web.HTTPForbidden(text="publication authority epoch required") from exc
        supplied = request.headers.get("X-Gradlab-Publication-Capability", "")
        if (
            client is None
            or self.publication_authority_client_id != client_id
            or epoch != self.control_epoch
            or self.publication_capability is None
            or not secrets.compare_digest(supplied, self.publication_capability)
        ):
            raise web.HTTPForbidden(text="exact publication authority required")
        return client

    def _publications(self) -> Any:
        if self._publication_service is not None:
            return self._publication_service
        if self._repo_root is None or (
            self._publication_factory is None
            and not hasattr(self.runner, "active_publication_context")
        ):
            raise ValueError("player publication is unavailable for this playback mode")
        if self._publication_factory is None:
            from gradlab.player_publication import PlayerPublicationService

            self._publication_service = PlayerPublicationService(
                repo_root=self._repo_root,
                host=self.runner,
            )
        else:
            self._publication_service = self._publication_factory(
                repo_root=self._repo_root,
                host=self.runner,
            )
        return self._publication_service

    def _rotate_publication_authority(self, client_id: str | None) -> None:
        self.publication_authority_client_id = client_id
        self.publication_capability = secrets.token_urlsafe(32) if client_id else None
        self._oauth_transactions.clear()
        self._media_tickets.clear()
        for client in self.clients.values():
            owns = client.client_id == client_id
            client.offer_reliable(
                {
                    "type": "publication_authority",
                    "has_authority": owns,
                    "control_epoch": self.control_epoch,
                    **(
                        {"capability": self.publication_capability}
                        if owns and self.publication_capability is not None
                        else {}
                    ),
                }
            )

    async def publication_current(self, request: web.Request) -> web.Response:
        self._authorize_publication(request)
        try:
            result = await asyncio.to_thread(self._publications().current)
        except ValueError as exc:
            return web.json_response({"available": False, "message": str(exc)})
        return web.json_response(result)

    async def publication_render(self, request: web.Request) -> web.Response:
        self._authorize_publication(request, mutation=True)
        try:
            result = await asyncio.to_thread(self._publications().render)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=400)
        return web.json_response(result)

    async def publication_preflight(self, request: web.Request) -> web.Response:
        self._authorize_publication(request, mutation=True)
        try:
            result = await asyncio.to_thread(self._publications().preflight)
        except Exception as exc:
            return web.json_response({"ready": False, "message": str(exc)}, status=503)
        return web.json_response(result)

    async def publication_preview(self, request: web.Request) -> web.Response:
        self._authorize_publication(request)
        try:
            payload = await request.json()
        except json.JSONDecodeError, TypeError:
            return web.json_response({"error": "publication preview must be JSON"}, status=400)
        if not isinstance(payload, Mapping):
            return web.json_response({"error": "publication preview must be an object"}, status=400)
        try:
            result = await asyncio.to_thread(self._publications().preview, payload)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=400)
        return web.json_response(result)

    async def publication_admit(self, request: web.Request) -> web.Response:
        self._authorize_publication(request, mutation=True)
        try:
            payload = await request.json()
        except json.JSONDecodeError, TypeError:
            return web.json_response({"error": "publication request must be JSON"}, status=400)
        if not isinstance(payload, Mapping):
            return web.json_response({"error": "publication request must be an object"}, status=400)
        try:
            result = await asyncio.to_thread(self._publications().admit, payload)
        except Exception as exc:
            from gradlab.player_publication import PublicationConflict

            status = 409 if isinstance(exc, PublicationConflict) else 400
            body: dict[str, Any] = {"error": str(exc)}
            if isinstance(exc, PublicationConflict) and exc.job is not None:
                body["job"] = exc.job
            return web.json_response(body, status=status)
        return web.json_response(result, status=202 if result.get("created") else 200)

    async def publication_job(self, request: web.Request) -> web.Response:
        self._authorize_publication(request)
        try:
            result = await asyncio.to_thread(
                self._publications().job,
                request.match_info["job_id"],
            )
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=404)
        return web.json_response(result)

    async def _publication_job_action(self, request: web.Request, action: str) -> web.Response:
        self._authorize_publication(request, mutation=True)
        try:
            service = self._publications()
            if action == "resolve":
                payload = await request.json()
                if not isinstance(payload, Mapping):
                    raise ValueError("resolution must be a JSON object")
                result = await asyncio.to_thread(
                    service.resolve_youtube,
                    request.match_info["job_id"],
                    str(payload.get("video_id") or ""),
                )
            else:
                result = await asyncio.to_thread(
                    getattr(service, action),
                    request.match_info["job_id"],
                )
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=502)
        return web.json_response(result)

    async def publication_retry(self, request: web.Request) -> web.Response:
        return await self._publication_job_action(request, "retry")

    async def publication_cancel(self, request: web.Request) -> web.Response:
        return await self._publication_job_action(request, "cancel")

    async def publication_resolve(self, request: web.Request) -> web.Response:
        return await self._publication_job_action(request, "resolve")

    async def publication_cleanup(self, request: web.Request) -> web.Response:
        return await self._publication_job_action(request, "cleanup")

    async def publication_oauth_start(self, request: web.Request) -> web.Response:
        client = self._authorize_publication(request, mutation=True)
        try:
            paths = youtube_credential_paths()
            with credential_lock(paths.lock):
                config = load_private_json(paths.client, root=paths.root)
            transaction = new_oauth_transaction(
                config,
                redirect_uri=f"{self.origin}/api/publication/oauth/callback",
                authority_client_id=client.client_id,
                control_epoch=self.control_epoch,
            )
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=400)
        self._oauth_transactions[transaction.state] = transaction
        if request.headers.get("X-Gradlab-Desktop") == "1" and self.desktop_browser is not None:
            import webbrowser

            opened = await asyncio.to_thread(webbrowser.open, transaction.authorization_url)
            if not opened:
                self._oauth_transactions.pop(transaction.state, None)
                return web.json_response(
                    {"error": "Could not open the authorization browser"}, status=400
                )
            return web.json_response({"opened_external": True})
        return web.json_response({"authorization_url": transaction.authorization_url})

    async def publication_oauth_callback(self, request: web.Request) -> web.Response:
        state = str(request.query.get("state") or "")
        code = str(request.query.get("code") or "")
        transaction = self._oauth_transactions.pop(state, None)
        try:
            if transaction is None or not code:
                raise ValueError("YouTube authorization callback is incomplete or expired")
            if self.publication_authority_client_id is None:
                raise ValueError("player publication authority no longer exists")
            transaction.validate_authority(
                self.publication_authority_client_id,
                self.control_epoch,
            )
            paths = youtube_credential_paths()
            with credential_lock(paths.lock):
                config = load_private_json(paths.client, root=paths.root)
                token = exchange_oauth_code(config, transaction, code=code)
                save_private_json(paths.token, token, root=paths.root)
            authority = self.clients.get(self.publication_authority_client_id)
            if authority is not None:
                authority.offer_reliable({"type": "publication_credentials_changed"})
        except Exception as exc:
            return web.Response(
                text=f"YouTube authorization failed: {exc}",
                status=400,
                content_type="text/plain",
            )
        raise web.HTTPSeeOther(
            location=f"/publication/oauth/complete#token={quote(self.token, safe='')}"
        )

    async def publication_replay_ticket(self, request: web.Request) -> web.Response:
        client = self._authorize_publication(request, mutation=True)
        try:
            await asyncio.to_thread(self._publications().replay_path)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=404)
        ticket = secrets.token_urlsafe(32)
        self._media_tickets[ticket] = (client.client_id, self.control_epoch, time.time() + 60)
        return web.json_response({"url": f"/api/publication/replay/{ticket}"})

    async def publication_replay(self, request: web.Request) -> web.StreamResponse:
        ticket = str(request.match_info["ticket"])
        admission = self._media_tickets.get(ticket)
        if (
            admission is None
            or admission[0] != self.publication_authority_client_id
            or admission[1] != self.control_epoch
            or admission[2] <= time.time()
        ):
            raise web.HTTPForbidden(text="replay ticket expired")
        try:
            path = await asyncio.to_thread(self._publications().replay_path)
        except ValueError as exc:
            raise web.HTTPNotFound(text=str(exc)) from exc
        return web.FileResponse(path)

    @staticmethod
    def _catalog_error_response(exc: Exception) -> web.Response | None:
        from gradlab.catalog_errors import CatalogError

        if not isinstance(exc, CatalogError):
            return None
        return web.json_response(
            {
                "error": str(exc),
                "problem": exc.problem.to_dict(),
            },
            status=exc.problem.status,
        )

    async def catalog_environments(self, request: web.Request) -> web.Response:
        self._authorize_api(request)
        if self.catalog is None:
            raise web.HTTPNotFound()
        from gradlab.play_catalog import normalize_search_query

        try:
            page = await asyncio.to_thread(
                self.catalog.environments,
                query=normalize_search_query(request.query.get("q")),
                cursor=request.query.get("cursor"),
            )
        except Exception as exc:
            problem = self._catalog_error_response(exc)
            if problem is not None:
                return problem
            return web.json_response({"error": str(exc)}, status=502)
        return web.json_response(page.to_dict())

    async def _prepare_initial_catalog(self) -> None:
        if (
            self.catalog is None
            or (await asyncio.to_thread(self.runner.snapshot)).get("mode") == "trajectory"
        ):
            return
        initial_environments = getattr(self.catalog, "initial_environments", None)
        if not callable(initial_environments):
            return
        payload = await asyncio.to_thread(initial_environments)
        if not isinstance(payload, Mapping):
            raise ValueError("initial environment catalog must be a mapping")
        self._initial_environment_catalog = dict(payload)

    async def catalog_runs(self, request: web.Request) -> web.Response:
        self._authorize_api(request)
        if self.catalog is None:
            raise web.HTTPNotFound()
        from gradlab.play_catalog import normalize_search_query

        try:
            kwargs = {
                "environment_id": request.match_info["environment_id"],
                "goal_id": request.match_info.get("goal_id", ""),
                "goal_variant_id": request.match_info.get("goal_variant_id", ""),
                "query": normalize_search_query(request.query.get("q")),
                "cursor": request.query.get("cursor"),
            }
            if request.query.get("refresh") == "1":
                kwargs["refresh"] = True
            page = await asyncio.to_thread(self.catalog.runs, **kwargs)
        except Exception as exc:
            problem = self._catalog_error_response(exc)
            if problem is not None:
                return problem
            return web.json_response({"error": str(exc)}, status=502)
        return web.json_response(page.to_dict())

    async def catalog_goal_variants(self, request: web.Request) -> web.Response:
        self._authorize_api(request)
        if self.catalog is None:
            raise web.HTTPNotFound()
        from gradlab.play_catalog import normalize_search_query

        try:
            kwargs = {
                "environment_id": request.match_info["environment_id"],
                "goal_id": request.match_info["goal_id"],
                "query": normalize_search_query(request.query.get("q")),
                "cursor": request.query.get("cursor"),
            }
            if request.query.get("refresh") == "1":
                kwargs["refresh"] = True
            page = await asyncio.to_thread(self.catalog.goal_variants, **kwargs)
        except Exception as exc:
            problem = self._catalog_error_response(exc)
            if problem is not None:
                return problem
            return web.json_response({"error": str(exc)}, status=502)
        return web.json_response(page.to_dict())

    async def catalog_goal_activity(self, request: web.Request) -> web.Response:
        self._authorize_api(request)
        if self.catalog is None:
            raise web.HTTPNotFound()
        from gradlab.play_catalog import normalize_search_query

        try:
            payload = await asyncio.to_thread(
                self.catalog.goal_activity,
                environment_id=request.match_info["environment_id"],
                goal_id=request.match_info["goal_id"],
                query=normalize_search_query(request.query.get("q")),
                refresh=request.query.get("refresh") == "1",
            )
        except Exception as exc:
            problem = self._catalog_error_response(exc)
            if problem is not None:
                return problem
            return web.json_response({"error": str(exc)}, status=502)
        return web.json_response(payload)

    async def catalog_goals(self, request: web.Request) -> web.Response:
        self._authorize_api(request)
        if self.catalog is None:
            raise web.HTTPNotFound()
        from gradlab.play_catalog import normalize_search_query

        try:
            page = await asyncio.to_thread(
                self.catalog.goals,
                environment_id=request.match_info["environment_id"],
                query=normalize_search_query(request.query.get("q")),
                cursor=request.query.get("cursor"),
                include_evidence=request.query.get("evidence") != "0",
                progressive="evidence" in request.query,
            )
        except Exception as exc:
            problem = self._catalog_error_response(exc)
            if problem is not None:
                return problem
            return web.json_response({"error": str(exc)}, status=502)
        return web.json_response(page.to_dict())

    async def catalog_recipes(self, request: web.Request) -> web.Response:
        self._authorize_api(request)
        if self.catalog is None:
            raise web.HTTPNotFound()
        from gradlab.play_catalog import normalize_search_query

        try:
            page = await asyncio.to_thread(
                self.catalog.recipes,
                environment_id=request.match_info["environment_id"],
                goal_id=request.match_info["goal_id"],
                query=normalize_search_query(request.query.get("q")),
                cursor=request.query.get("cursor"),
            )
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=502)
        return web.json_response(page.to_dict())

    async def inspect_goal(self, request: web.Request) -> web.Response:
        self._authorize_api(request)
        if self.catalog is None:
            raise web.HTTPNotFound()
        try:
            document = await asyncio.to_thread(
                self.catalog.inspect_goal,
                environment_id=request.match_info["environment_id"],
                goal_id=request.match_info["goal_id"],
            )
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=502)
        return web.json_response(document)

    async def inspect_recipe(self, request: web.Request) -> web.Response:
        self._authorize_api(request)
        if self.catalog is None:
            raise web.HTTPNotFound()
        try:
            document = await asyncio.to_thread(
                self.catalog.inspect_recipe,
                environment_id=request.match_info["environment_id"],
                goal_id=request.match_info["goal_id"],
                recipe_id=request.match_info["recipe_id"],
            )
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=502)
        return web.json_response(document)

    async def inspect_goal_variant(self, request: web.Request) -> web.Response:
        self._authorize_api(request)
        if self.catalog is None:
            raise web.HTTPNotFound()
        try:
            document = await asyncio.to_thread(
                self.catalog.inspect_goal_variant,
                environment_id=request.match_info["environment_id"],
                goal_id=request.match_info["goal_id"],
                variant_id=request.match_info["goal_variant_id"],
            )
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=502)
        return web.json_response(document)

    async def inspect_run(self, request: web.Request) -> web.Response:
        self._authorize_api(request)
        if self.catalog is None:
            raise web.HTTPNotFound()
        try:
            document = await asyncio.to_thread(
                self.catalog.inspect_run,
                run_id=request.match_info["run_id"],
            )
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=502)
        return web.json_response(document)

    async def _read_diagnostics(self, request: web.Request, kind: str):
        from gradlab.play_diagnostics import DiagnosticKind, DiagnosticRead

        epoch = int(request.query["epoch"])
        query = DiagnosticRead(
            DiagnosticKind(kind),
            request.query["episode_id"],
            int(request.query["first"]) if "first" in request.query else None,
            int(request.query["last"]) if "last" in request.query else None,
        )
        if epoch != await asyncio.to_thread(self._runner_epoch):
            raise ValueError("the Playback Session has been replaced")
        result = await asyncio.to_thread(self._diagnostic_reader.read_diagnostics, epoch, query)
        # A completed worker job may wait unpolled across session replacement.
        if epoch != await asyncio.to_thread(self._runner_epoch):
            raise ValueError("the Playback Session has been replaced")
        return result

    async def chart_history(self, request: web.Request) -> web.Response:
        self._authorize_api(request)
        try:
            result = await self._read_diagnostics(request, "chart")
            if request.query.get("format") == "chart-columns-v1":
                result = await asyncio.to_thread(
                    self._chart_responses.encode, result, request.query.get("base")
                )
            body = await asyncio.to_thread(
                json.dumps, result, separators=(",", ":"), allow_nan=False
            )
            response = web.Response(
                text=body,
                content_type="application/json",
                headers={"Cache-Control": "no-store"},
                zlib_executor_size=64 * 1024,
            )
            response.enable_compression()
            return response
        except (KeyError, ValueError, OSError, RuntimeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    async def reward_history(self, request: web.Request) -> web.Response:
        self._authorize_api(request)
        try:
            result = await self._read_diagnostics(request, "reward")
            return web.json_response(result, headers={"Cache-Control": "no-store"})
        except (KeyError, ValueError, OSError, RuntimeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    async def event_history(self, request: web.Request) -> web.Response:
        self._authorize_api(request)
        try:
            result = await self._read_diagnostics(request, "event")
            return web.json_response(result, headers={"Cache-Control": "no-store"})
        except (KeyError, ValueError, OSError, RuntimeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    async def inspect_recorded_step(self, request: web.Request) -> web.Response:
        self._authorize_api(request)
        try:
            epoch = int(request.query["epoch"])
            step = int(request.query["step"])
            episode_id = request.query["episode_id"]
            inspect = getattr(self.runner, "inspect_recorded_step", None)
            if inspect is None:
                raise ValueError("recorded inspection is unavailable for this source")
            # Direct runners are used by embedded players; the application host
            # additionally binds reads to the immutable Playback Session epoch.
            if epoch != await asyncio.to_thread(self._runner_epoch):
                raise ValueError("the Playback Session has been replaced")
            args = (
                (episode_id, step)
                if isinstance(self.runner, WebPlaybackRunner)
                else (epoch, episode_id, step)
            )
            result = await asyncio.to_thread(inspect, *args)
            if epoch != await asyncio.to_thread(self._runner_epoch):
                raise ValueError("the Playback Session has been replaced")
            if request.query.get("rgb") == "off":
                result["frames"] = [
                    frame for frame in result.get("frames", []) if frame["kind"] != FRAME_GAME
                ]
            result["snapshot"]["session_epoch"] = epoch
            return web.json_response(result, headers={"Cache-Control": "no-store"})
        except (KeyError, ValueError, OSError, RuntimeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    async def inspect_active_playback(self, request: web.Request) -> web.Response:
        self._authorize_api(request)
        from gradlab.contract_inspection import inspection_document

        active_recipe = getattr(self.runner, "active_recipe_document", None)
        resolved = await asyncio.to_thread(active_recipe) if callable(active_recipe) else None
        if resolved is None or self.catalog is None:
            message = "No verified policy bundle is active in the player."
            unavailable_goal = inspection_document(
                kind="goal",
                title="Active playback",
                availability="unavailable",
                message=message,
            )
            unavailable_recipe = inspection_document(
                kind="recipe",
                title="Active playback",
                availability="unavailable",
                message=message,
            )
            return web.json_response(
                {
                    "schema_version": 1,
                    "source": {"kind": "active-playback"},
                    "documents": {
                        "goal": unavailable_goal,
                        "recipe": unavailable_recipe,
                    },
                }
            )
        recipe_document, source = resolved
        try:
            document = await asyncio.to_thread(
                self.catalog.inspect_portable_recipe,
                recipe_document,
                source=source,
            )
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=502)
        return web.json_response(document)

    async def catalog_checkpoints(self, request: web.Request) -> web.Response:
        return await self._catalog_checkpoints(request, include_wandb=False)

    async def catalog_checkpoint_training(self, request: web.Request) -> web.StreamResponse:
        if request.query.get("stream") != "1":
            return await self._catalog_checkpoints(request, include_wandb=True)
        self._authorize_api(request)
        if self.catalog is None:
            raise web.HTTPNotFound()
        response = web.StreamResponse(
            headers={
                "Content-Type": "application/x-ndjson",
                "Cache-Control": "no-store",
                "X-Accel-Buffering": "no",
            }
        )
        await response.prepare(request)
        loop = asyncio.get_running_loop()
        events: asyncio.Queue[Mapping[str, Any]] = asyncio.Queue()

        def progress(event: Mapping[str, Any]) -> None:
            if not worker.done():
                loop.call_soon_threadsafe(events.put_nowait, event)

        async def produce() -> None:
            try:
                result = await self._catalog_checkpoints(
                    request,
                    include_wandb=True,
                    on_training_progress=progress,
                )
                payload = json.loads(result.body)
                await events.put(
                    {"type": "complete" if result.status == 200 else "error", **payload}
                )
            except Exception as exc:
                await events.put({"type": "error", "error": str(exc)})

        worker = asyncio.create_task(produce())
        try:
            while True:
                event = await events.get()
                await response.write((json.dumps(event) + "\n").encode())
                if event["type"] in {"complete", "error"}:
                    break
            await response.write_eof()
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        return response

    async def _catalog_checkpoints(
        self,
        request: web.Request,
        *,
        include_wandb: bool,
        on_training_progress: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> web.Response:
        self._authorize_api(request)
        if self.catalog is None:
            raise web.HTTPNotFound()
        from gradlab.play_catalog import (
            checkpoint_metric_leaders,
            checkpoint_metric_values,
            filter_checkpoint_summaries,
            normalize_search_query,
        )

        query = normalize_search_query(request.query.get("q"))
        try:
            page = await asyncio.to_thread(
                self.catalog.checkpoints,
                run_id=request.match_info["run_id"],
                query="",
                goal_variant_id=request.query.get("goal_variant_id", ""),
                include_wandb=include_wandb,
                **({"on_training_progress": on_training_progress} if on_training_progress else {}),
            )
            items = list(page.items)
            metric_columns = list(page.metric_columns)
            warnings = list(page.warnings)
            if not include_wandb and any(
                column.get("evidence") == "training"
                for column in metric_columns
                if isinstance(column, Mapping)
            ):
                warnings.append(
                    {
                        "code": "wandb_enrichment_pending",
                        "message": "Live W&B training evidence is loading.",
                        "retryable": True,
                        "source": "wandb",
                    }
                )
        except Exception as exc:
            problem = self._catalog_error_response(exc)
            if problem is not None:
                return problem
            return web.json_response({"error": str(exc)}, status=502)
        try:
            queue_service = await asyncio.to_thread(self._manual_evaluations)
            if queue_service is not None:
                statuses = await asyncio.to_thread(
                    queue_service.statuses,
                    run_id=request.match_info["run_id"],
                    checkpoint_ids=[
                        str(item.get("checkpoint_id") or "")
                        for item in items
                        if isinstance(item, Mapping)
                    ],
                )
                enriched_items = []
                for item in items:
                    status = statuses.get(str(item["checkpoint_id"]), {})
                    evaluation = (
                        status["evaluation"]
                        if status.get("evaluation") is not None
                        else item.get("evaluation")
                    )
                    enriched_items.append(
                        {
                            **dict(item),
                            "metrics": checkpoint_metric_values(
                                dict(item.get("metrics") or {}),
                                evaluation,
                                metric_columns,
                            ),
                            "evaluation": evaluation,
                            "evaluation_queue": status or None,
                        }
                    )
                items = enriched_items
        except Exception as exc:
            warnings.append(
                {
                    "code": "manual_evaluation_status_unavailable",
                    "message": f"Manual evaluation status is unavailable: {exc}",
                    "retryable": True,
                    "source": "manual-evaluation",
                }
            )
            items = [
                {
                    **dict(item),
                    "evaluation_queue": {
                        "state": "unavailable",
                        "message": str(exc),
                    },
                }
                for item in items
            ]
        items = list(checkpoint_metric_leaders(items, metric_columns))
        items = list(filter_checkpoint_summaries(items, query=query))
        return web.json_response(
            {
                "items": items,
                "next_cursor": None,
                "metric_columns": metric_columns,
                "selection_fence": page.selection_fence,
                "run": dict(page.run) if isinstance(page.run, Mapping) else None,
                "training_enrichment": "complete" if include_wandb else "pending",
                "freshness": "partial" if warnings else page.freshness,
                "warnings": warnings,
            }
        )

    def _manual_evaluations(self) -> Any:
        if self._manual_evaluation_queue is not None:
            return self._manual_evaluation_queue
        factory = self._manual_evaluation_factory
        if factory is None:
            if self.catalog is None:
                raise RuntimeError("checkpoint evaluation is unavailable without a catalog")
            from gradlab.manual_evaluation import build_manual_evaluation_queue

            repo_root = Path(
                getattr(self.catalog, "repo_root", Path(__file__).resolve().parents[2])
            )

            def factory() -> Any:
                return build_manual_evaluation_queue(repo_root)

        self._manual_evaluation_queue = factory()
        return self._manual_evaluation_queue

    async def catalog_evaluate_checkpoints(self, request: web.Request) -> web.Response:
        self._authorize_api(request)
        if self.catalog is None:
            raise web.HTTPNotFound()
        try:
            payload = await request.json()
        except json.JSONDecodeError, TypeError:
            return web.json_response({"error": "request body must be JSON"}, status=400)
        checkpoint_ids = payload.get("checkpoint_ids") if isinstance(payload, Mapping) else None
        selection_fence = payload.get("selection_fence") if isinstance(payload, Mapping) else None
        if isinstance(checkpoint_ids, str | bytes) or not isinstance(checkpoint_ids, Sequence):
            return web.json_response(
                {"error": "checkpoint_ids must be a JSON array"},
                status=400,
            )
        if not isinstance(selection_fence, str) or not selection_fence.strip():
            return web.json_response(
                {"error": "selection_fence must be a non-empty string"},
                status=400,
            )
        try:
            queue_service = await asyncio.to_thread(self._manual_evaluations)
        except Exception as exc:
            return web.json_response(
                {"error": f"manual evaluation is unavailable: {exc}"},
                status=503,
            )
        try:
            result = await asyncio.to_thread(
                queue_service.enqueue,
                run_id=request.match_info["run_id"],
                checkpoint_ids=[str(value) for value in checkpoint_ids],
                selection_fence=selection_fence,
            )
        except Exception as exc:
            from gradlab.manual_evaluation import EvaluationSelectionChanged

            if isinstance(exc, EvaluationSelectionChanged):
                return web.json_response(
                    {
                        "error": str(exc),
                        "code": "checkpoint_catalog_changed",
                    },
                    status=409,
                )
            if isinstance(exc, ValueError):
                return web.json_response({"error": str(exc)}, status=400)
            return web.json_response({"error": str(exc)}, status=502)
        if isinstance(result, Mapping):
            response = dict(result)
            response["items"] = list(response.get("items") or ())
        else:
            response = {"items": list(result)}
        return web.json_response(response, status=202)

    def _snapshot_for(self, client: WebClient, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        payload = {
            **snapshot,
            "control_epoch": self.control_epoch,
            "control": {
                "client_id": client.client_id,
                "workspace_id": client.workspace_id,
                "window_id": client.window_id,
                "holder": self.control_holder,
                "input_holder": self.input_holder,
                "has_control": self.control_holder == client.workspace_id,
            },
            "publication": {
                "has_authority": self.publication_authority_client_id == client.client_id,
                "configured": self._repo_root is not None,
            },
        }
        app = payload.get("app")
        if (
            self._initial_environment_catalog is not None
            and isinstance(app, Mapping)
            and app.get("phase") == "selecting"
        ):
            payload["app"] = {
                **app,
                "catalog": dict(self._initial_environment_catalog),
            }
        return payload

    async def _broadcast_control(self) -> None:
        snapshot = await asyncio.to_thread(self.runner.snapshot)
        for client in self.clients.values():
            client.offer_snapshot(self._snapshot_for(client, snapshot))

    def _runner_epoch(self) -> int:
        return int(getattr(self.runner, "session_epoch", 0))

    async def _announce_session_change(self) -> None:
        self.trajectory_transfers.revoke()
        epoch = await asyncio.to_thread(self._runner_epoch)
        history = await asyncio.to_thread(self.runner.history_payload)
        for client in tuple(self.clients.values()):
            client.reset_session(epoch)
            client.offer_reliable(
                {
                    "type": "session_changed",
                    "protocol": PROTOCOL_VERSION,
                    "session_epoch": epoch,
                }
            )
            client.offer_reliable(history)

    async def websocket(self, request: web.Request) -> web.WebSocketResponse:
        if request.headers.get("Origin") != self.origin:
            raise web.HTTPForbidden(text="invalid websocket origin")
        socket = web.WebSocketResponse(
            heartbeat=10.0,
            compress=False,
            max_msg_size=256 * 1024,
            writer_limit=256 * 1024,
        )
        await socket.prepare(request)
        client: WebClient | None = None
        writer: asyncio.Task[None] | None = None
        try:
            try:
                first = await asyncio.wait_for(socket.receive(), timeout=5.0)
            except TimeoutError:
                await socket.close(code=1008, message=b"authentication timeout")
                return socket
            if first.type != WSMsgType.TEXT:
                await socket.close(code=1008, message=b"hello required")
                return socket
            try:
                hello = json.loads(first.data)
            except json.JSONDecodeError:
                await socket.close(code=1008, message=b"invalid hello")
                return socket
            if hello.get("type") != "hello" or not secrets.compare_digest(
                str(hello.get("token") or ""), self.token
            ):
                await socket.close(code=1008, message=b"authentication failed")
                return socket
            subscriptions = {
                str(value)
                for value in hello.get("subscriptions", ("telemetry",))
                if str(value) in {"telemetry", *FRAME_SUBSCRIPTIONS.values()}
            }
            processing = normalize_player_processing(
                hello.get("processing", PLAYER_PROCESSING_FEATURES)
            )
            client_id = uuid.uuid4().hex
            workspace_id = str(hello.get("workspace_id") or client_id)[:128]
            if not workspace_id or any(
                character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
                for character in workspace_id
            ):
                workspace_id = client_id
            window_id = str(hello.get("window_id") or "main")[:128]
            client = WebClient(
                client_id,
                socket,
                subscriptions,
                workspace_id,
                window_id,
                processing,
            )
            self.clients[client_id] = client
            await self._sync_player_processing()
            self.ever_connected = True
            if self.control_holder is None:
                self.control_holder = workspace_id
                self.control_epoch += 1
            client.offer_reliable(
                {
                    "type": "welcome",
                    "protocol": PROTOCOL_VERSION,
                    "client_id": client_id,
                    "workspace_id": workspace_id,
                    "window_id": window_id,
                    "history_limit": HISTORY_LIMIT,
                }
            )
            if self.publication_authority_client_id is None and self.control_holder == workspace_id:
                self._rotate_publication_authority(client_id)
            client.offer_reliable((await asyncio.to_thread(self.runner.history_payload)))
            episode_start_payload = getattr(self.runner, "episode_start_payload", None)
            if callable(episode_start_payload):
                episode_start_snapshot, episode_start_frames = await asyncio.to_thread(
                    episode_start_payload
                )
                if episode_start_snapshot:
                    client.offer_reliable(self._snapshot_for(client, episode_start_snapshot))
                for frame_kind, (_sequence, packet) in episode_start_frames.items():
                    subscription = FRAME_SUBSCRIPTIONS.get(frame_kind)
                    if subscription in client.subscriptions:
                        client.offer_reliable(packet)
            client.offer_snapshot(
                self._snapshot_for(client, (await asyncio.to_thread(self.runner.snapshot)))
            )
            for frame_kind, (sequence, packet) in (
                await asyncio.to_thread(self.runner.encoder.latest)
            ).items():
                subscription = FRAME_SUBSCRIPTIONS.get(frame_kind)
                if subscription in client.subscriptions:
                    client.offer_frame(frame_kind, sequence, packet)
            writer = asyncio.create_task(client.write())
            await self._broadcast_control()
            async for message in socket:
                if message.type == WSMsgType.ERROR:
                    break
                if message.type != WSMsgType.TEXT:
                    continue
                try:
                    payload = json.loads(message.data)
                except json.JSONDecodeError:
                    client.offer_reliable({"type": "error", "error": "invalid JSON message"})
                    continue
                kind = str(payload.get("type") or "")
                if kind == "acquire_control":
                    if (
                        self.control_holder != client.workspace_id
                        or self.publication_authority_client_id != client.client_id
                    ):
                        if self.control_holder != client.workspace_id:
                            try:
                                await asyncio.to_thread(
                                    self.runner.submit,
                                    PlaybackCommand(
                                        uuid.uuid4().hex,
                                        client.client_id,
                                        "pause",
                                        {},
                                        None,
                                    ),
                                )
                            except queue.Full:
                                client.offer_reliable(
                                    {
                                        "type": "error",
                                        "error": "cannot transfer control while the command queue is full",
                                    }
                                )
                                continue
                        self.control_holder = client.workspace_id
                        self.input_holder = None
                        self.control_epoch += 1
                        await asyncio.to_thread(self.runner.clear_input)
                        self._rotate_publication_authority(client.client_id)
                        await self._broadcast_control()
                elif kind == "subscribe":
                    previous_subscriptions = client.subscriptions
                    client.subscriptions = {
                        str(value)
                        for value in payload.get("subscriptions", ())
                        if str(value) in {"telemetry", *FRAME_SUBSCRIPTIONS.values()}
                    }
                    for frame_kind, subscription in FRAME_SUBSCRIPTIONS.items():
                        if (
                            subscription not in client.subscriptions
                            or subscription not in previous_subscriptions
                        ):
                            client.sent_frames.pop(frame_kind, None)
                        if subscription not in client.subscriptions:
                            client.latest_frames.pop(frame_kind, None)
                    if "processing" in payload:
                        client.processing = normalize_player_processing(
                            payload.get("processing") or ()
                        )
                        await self._sync_player_processing()
                    for frame_kind, (sequence, packet) in (
                        await asyncio.to_thread(self.runner.encoder.latest)
                    ).items():
                        subscription = FRAME_SUBSCRIPTIONS.get(frame_kind)
                        if subscription in client.subscriptions:
                            client.offer_frame(frame_kind, sequence, packet)
                elif kind == "history":
                    client.offer_reliable((await asyncio.to_thread(self.runner.history_payload)))
                elif kind == "inspection_frames":
                    try:
                        epoch = int(payload.get("session_epoch", -1))
                        sequence = int(payload.get("sequence", -1))
                        requested_kinds = {int(value) for value in payload.get("kinds", ())} & {
                            FRAME_GAME,
                            FRAME_OBSERVATION,
                            FRAME_ATTRIBUTION,
                            FRAME_CNN_INSPECTION,
                        }
                    except TypeError, ValueError:
                        client.offer_reliable(
                            {"type": "error", "error": "invalid inspection frame request"}
                        )
                        continue
                    if (
                        epoch != await asyncio.to_thread(self._runner_epoch)
                        or sequence < 0
                        or not requested_kinds
                    ):
                        continue
                    retained = await asyncio.to_thread(
                        self.runner.encoder.retained,
                        sequence,
                        epoch=epoch,
                        timeout=INSPECTION_FRAME_WAIT_SECONDS,
                        kinds=requested_kinds,
                    )
                    for frame_kind in requested_kinds:
                        packet = retained.get(frame_kind)
                        if packet is not None:
                            client.offer_reliable(packet[1])
                elif kind == "input":
                    if self.control_holder != client.workspace_id:
                        client.offer_reliable({"type": "error", "error": "control lease required"})
                    else:
                        labels = payload.get("pressed", ())
                        focused = bool(payload.get("focused", False))
                        if focused:
                            if self.input_holder != client_id:
                                await asyncio.to_thread(self.runner.clear_input)
                            self.input_holder = client_id
                            await asyncio.to_thread(
                                self.runner.update_input,
                                labels if isinstance(labels, list) else (),
                                focused=True,
                            )
                        elif self.input_holder == client_id:
                            self.input_holder = None
                            await asyncio.to_thread(self.runner.update_input, (), focused=False)
                elif kind == "command":
                    if self.control_holder != client.workspace_id:
                        client.offer_reliable(
                            {
                                "type": "command_result",
                                "id": str(payload.get("id") or ""),
                                "ok": False,
                                "error": "control lease required",
                            }
                        )
                        continue
                    command_name = str(payload.get("name") or "")
                    try:
                        await asyncio.to_thread(
                            self.runner.submit,
                            PlaybackCommand(
                                str(payload.get("id") or uuid.uuid4().hex),
                                client_id,
                                command_name,
                                payload.get("payload")
                                if isinstance(payload.get("payload"), Mapping)
                                else {},
                                int(payload["expected_revision"])
                                if payload.get("expected_revision") is not None
                                else None,
                            ),
                        )
                    except queue.Full:
                        client.offer_reliable(
                            {
                                "type": "command_result",
                                "id": str(payload.get("id") or ""),
                                "ok": False,
                                "error": "command queue is full",
                            }
                        )
        finally:
            if client is not None:
                client.closed = True
                client.event.set()
                self.clients.pop(client.client_id, None)
                await self._sync_player_processing()
                if self.input_holder == client.client_id:
                    self.input_holder = None
                    await asyncio.to_thread(self.runner.clear_input)
                publication_authority_closed = (
                    self.publication_authority_client_id == client.client_id
                )
                if publication_authority_closed:
                    self.control_epoch += 1
                    self._rotate_publication_authority(None)
                controlling_workspace_closed = (
                    self.control_holder == client.workspace_id
                    and not any(
                        candidate.workspace_id == client.workspace_id
                        for candidate in self.clients.values()
                    )
                )
                if controlling_workspace_closed:
                    self.control_holder = None
                    if not publication_authority_closed:
                        self.control_epoch += 1
                        self._rotate_publication_authority(None)
                    await asyncio.to_thread(self.runner.clear_input)
                    try:
                        await asyncio.to_thread(
                            self.runner.submit,
                            PlaybackCommand(
                                uuid.uuid4().hex,
                                client.client_id,
                                "pause",
                                {},
                                None,
                            ),
                        )
                    except queue.Full:
                        pass
                self.last_client_at = time.monotonic()
                await self._broadcast_control()
            if writer is not None:
                writer.cancel()
                await asyncio.gather(writer, return_exceptions=True)
        return socket

    async def pump(self) -> None:
        latest_snapshot_key = (-1, -1, -1)
        latest_frames: dict[int, tuple[int, bytes]] = {}
        while not self.stop_event.is_set():
            poll = getattr(self.runner, "playback_updates", None)
            update = await asyncio.to_thread(
                poll if callable(poll) else lambda: playback_updates(self.runner)
            )
            session_change = update["session_change"]
            if session_change != self._observed_session_change:
                self._observed_session_change = session_change
                latest_snapshot_key = (-1, -1, -1)
                latest_frames.clear()
                await self._announce_session_change()
            snapshots = update["snapshots"]
            for snapshot in snapshots:
                key = (
                    int(snapshot.get("session_epoch", 0)),
                    int(snapshot.get("revision", 0)),
                    int(snapshot.get("sequence", 0)),
                    (snapshot.get("app") or {}).get("phase"),
                )
                if key == latest_snapshot_key:
                    continue
                latest_snapshot_key = key
                for client in tuple(self.clients.values()):
                    if "telemetry" in client.subscriptions:
                        if (
                            bool((snapshot.get("transition") or {}).get("boundary"))
                            and (snapshot.get("session") or {}).get("value_discount") is not None
                        ):
                            client.offer_reliable(
                                (await asyncio.to_thread(self.runner.history_payload))
                            )
                        client.offer_snapshot(self._snapshot_for(client, snapshot))
            for kind, (sequence, packet) in update["frames"].items():
                if (sequence, packet) == latest_frames.get(kind):
                    continue
                latest_frames[kind] = (sequence, packet)
                subscription = FRAME_SUBSCRIPTIONS.get(kind)
                for client in tuple(self.clients.values()):
                    if subscription in client.subscriptions:
                        client.offer_frame(kind, sequence, packet)
            for response in update["responses"]:
                client = self.clients.get(response.client_id)
                if client is not None:
                    client.offer_reliable(response.payload)
            for client_id, client in tuple(self.clients.items()):
                if client.closed:
                    await client.socket.close(code=1013, message=b"client is too slow")
                    self.clients.pop(client_id, None)
            if update["stopped"]:
                self.stop_event.set()
                break
            if (
                self.desktop_browser is None
                and self.ever_connected
                and not self.clients
                and time.monotonic() - self.last_client_at >= LAST_CLIENT_GRACE_SECONDS
            ):
                await asyncio.to_thread(self.runner.stop)
                self.stop_event.set()
                break
            await asyncio.sleep(1.0 / 120.0)

    async def run(self) -> int:
        if self.dev_assets is None:
            return await self._run()
        try:
            url = await self.dev_assets.start()
            print(f"Player hot reload: {url}", flush=True)
            return await self._run()
        finally:
            await self.dev_assets.stop()

    async def _run(self) -> int:
        self.require_player_assets()
        app = web.Application(middlewares=[self.security_headers])
        app.add_routes(self.trajectory_transfers.routes())
        app.add_routes(
            [
                web.get("/", self.page),
                web.post("/api/desktop/window", self.desktop_window),
                web.get("/api/desktop/peer", self.desktop_peer),
                web.get("/environments/{environment_id}", self.page),
                web.get("/environments/{environment_id}/goals/{goal_id}", self.page),
                web.get(
                    (
                        "/environments/{environment_id}/goals/{goal_id}"
                        "/variants/{goal_variant_id}/runs/{run_id}"
                    ),
                    self.page,
                ),
                web.get(
                    (
                        "/environments/{environment_id}/goals/{goal_id}"
                        "/variants/{goal_variant_id}/runs/{run_id}"
                        "/checkpoints/{checkpoint_id}"
                    ),
                    self.page,
                ),
                web.get("/panel/{panel}", self.page),
                web.get("/workspace/{window}", self.page),
                web.get("/sources/{path:.*}", self.page),
                web.get("/assets/{path:.*}", self.asset),
                web.get("/api/catalog/environments", self.catalog_environments),
                web.get(
                    "/api/catalog/environments/{environment_id}/goals",
                    self.catalog_goals,
                ),
                web.get(
                    ("/api/catalog/environments/{environment_id}/goals/{goal_id}/inspection"),
                    self.inspect_goal,
                ),
                web.get(
                    ("/api/catalog/environments/{environment_id}/goals/{goal_id}/recipes"),
                    self.catalog_recipes,
                ),
                web.get(
                    (
                        "/api/catalog/environments/{environment_id}/goals/{goal_id}"
                        "/recipes/{recipe_id}/inspection"
                    ),
                    self.inspect_recipe,
                ),
                web.get(
                    ("/api/catalog/environments/{environment_id}/goals/{goal_id}/variants"),
                    self.catalog_goal_variants,
                ),
                web.get(
                    ("/api/catalog/environments/{environment_id}/goals/{goal_id}/activity"),
                    self.catalog_goal_activity,
                ),
                web.get(
                    (
                        "/api/catalog/environments/{environment_id}/goals/{goal_id}"
                        "/variants/{goal_variant_id}/inspection"
                    ),
                    self.inspect_goal_variant,
                ),
                web.get(
                    (
                        "/api/catalog/environments/{environment_id}/goals/{goal_id}"
                        "/variants/{goal_variant_id}/runs"
                    ),
                    self.catalog_runs,
                ),
                web.get(
                    "/api/catalog/runs/{run_id}/checkpoints",
                    self.catalog_checkpoints,
                ),
                web.get(
                    "/api/catalog/runs/{run_id}/checkpoint-training",
                    self.catalog_checkpoint_training,
                ),
                web.get(
                    "/api/catalog/runs/{run_id}/inspection",
                    self.inspect_run,
                ),
                web.post(
                    "/api/catalog/runs/{run_id}/evaluations",
                    self.catalog_evaluate_checkpoints,
                ),
                web.get("/api/playback/inspection", self.inspect_active_playback),
                web.get("/api/playback/recorded-step", self.inspect_recorded_step),
                web.get("/api/playback/chart-history", self.chart_history),
                web.get("/api/playback/event-history", self.event_history),
                web.get("/api/playback/reward-history", self.reward_history),
                web.get("/api/publication/current", self.publication_current),
                web.post("/api/publication/render", self.publication_render),
                web.post("/api/publication/preflight", self.publication_preflight),
                web.post("/api/publication/preview", self.publication_preview),
                web.post("/api/publication/admit", self.publication_admit),
                web.get("/api/publication/jobs/{job_id}", self.publication_job),
                web.post("/api/publication/jobs/{job_id}/retry", self.publication_retry),
                web.post("/api/publication/jobs/{job_id}/cancel", self.publication_cancel),
                web.post("/api/publication/jobs/{job_id}/resolve", self.publication_resolve),
                web.post("/api/publication/jobs/{job_id}/cleanup", self.publication_cleanup),
                web.post("/api/publication/oauth/start", self.publication_oauth_start),
                web.get("/api/publication/oauth/callback", self.publication_oauth_callback),
                web.get("/publication/oauth/complete", self.publication_oauth_complete),
                web.post("/api/publication/replay-ticket", self.publication_replay_ticket),
                web.get("/api/publication/replay/{ticket}", self.publication_replay),
                web.get("/ws", self.websocket),
            ]
        )
        app_runner = web.AppRunner(app, access_log=None)
        await app_runner.setup()
        site = web.TCPSite(app_runner, "127.0.0.1", int(self.args.port))
        await site.start()
        sockets = tuple(site._server.sockets) if site._server is not None else ()
        if not sockets:
            raise RuntimeError("player web server did not bind a socket")
        port = int(sockets[0].getsockname()[1])
        self.origin = f"http://127.0.0.1:{port}"
        urls = self.dashboard_urls()
        dashboard_label = str(getattr(self.args, "dashboard_label", "Player dashboard"))
        print(f"{dashboard_label}: {urls[0]}", flush=True)
        if self.paired_windows:
            print(f"Player stats: {urls[1]}", flush=True)
        await self._prepare_initial_catalog()
        await asyncio.to_thread(self.runner.start)
        pump = asyncio.create_task(self.pump())
        waiters = [pump, asyncio.create_task(self.stop_event.wait())]
        browser = PlaybackBrowser()
        try:
            if not bool(getattr(self.args, "no_open", False)):
                self.desktop_browser = browser
                await browser.open(urls[0])
                waiters.append(asyncio.create_task(browser.wait_closed()))
            done, _pending = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
            for completed in done:
                await completed
        finally:
            try:
                self.stop_event.set()
                for waiter in waiters:
                    waiter.cancel()
                await asyncio.gather(*waiters, return_exceptions=True)
                for peer in tuple(self.desktop_peers):
                    await peer.close(code=1001, message=b"player shutting down")
                for client in tuple(self.clients.values()):
                    await client.socket.close(code=1001, message=b"player shutting down")
                try:
                    # Socket closure does not await the request handler's finally block.
                    # Drain handlers while the worker can still clear input and processing.
                    await app_runner.cleanup()
                finally:
                    try:
                        await asyncio.to_thread(self.runner.stop)
                    finally:
                        self.trajectory_transfers.close()
            finally:
                await asyncio.to_thread(browser.close)
        return 0


def run_web_playback(
    session: _PlaybackSession,
    args: argparse.Namespace,
    *,
    config_text: str,
) -> int:
    runner = WebPlaybackRunner(session, args, config_text=config_text)
    server = PlaybackWebServer(runner, args, paired_windows=True)
    try:
        return asyncio.run(server.run())
    except KeyboardInterrupt:
        runner.stop()
        return 130


def run_web_player_application(
    host: Any,
    args: argparse.Namespace,
    *,
    catalog: Any,
    repo_root: Path,
) -> int:
    server = PlaybackWebServer(
        host,
        args,
        paired_windows=True,
        catalog=catalog,
        repo_root=repo_root,
    )
    try:
        return asyncio.run(server.run())
    except KeyboardInterrupt:
        host.stop()
        return 130


def run_web_dataset_playback(
    frames: Iterable[np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    *,
    fps: float,
    action_contract: Mapping[str, Any] | None = None,
) -> int:
    runner = DatasetPlaybackRunner(
        frames,
        rows,
        args,
        fps=fps,
        action_contract=action_contract,
    )
    args.dashboard_label = "Dataset dashboard"
    server = PlaybackWebServer(runner, args)
    try:
        return asyncio.run(server.run())
    except KeyboardInterrupt:
        runner.stop()
        return 130


class WebHumanController:
    """Synchronous human controller backed by a loopback web dashboard."""

    def __init__(self, session: Any, args: argparse.Namespace) -> None:
        self.runner = HumanRecordingRunner(session, args)
        args.dashboard_label = "Recording dashboard"
        self.server = PlaybackWebServer(self.runner, args)
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._serve,
            name="gradlab-recording-dashboard",
            daemon=True,
        )
        self._thread.start()
        deadline = time.monotonic() + 10.0
        while not self.server.origin and self._thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        if self._error is not None:
            raise RuntimeError("recording dashboard failed to start") from self._error
        if not self.server.origin:
            self.close()
            raise RuntimeError("recording dashboard did not start within 10 seconds")

    def _serve(self) -> None:
        try:
            asyncio.run(self.server.run())
        except BaseException as exc:
            self._error = exc
            self.runner.stop()

    def action(self, frame: np.ndarray) -> tuple[Any | None, bool]:
        if self._error is not None:
            raise RuntimeError("recording dashboard stopped unexpectedly") from self._error
        return self.runner.action(frame)

    def observe_transition(
        self,
        *,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
        next_frame: np.ndarray,
    ) -> None:
        self.runner.observe_transition(
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            next_frame=next_frame,
        )

    def close(self) -> None:
        self.runner.stop()
        if self._thread.is_alive():
            self._thread.join(timeout=10.0)


__all__ = [
    "FRAME_ATTRIBUTION",
    "FRAME_CNN_INSPECTION",
    "FRAME_CODEC_PNG",
    "FRAME_GAME",
    "FRAME_HEADER",
    "FRAME_MAGIC",
    "FRAME_OBSERVATION",
    "DatasetPlaybackRunner",
    "HumanRecordingRunner",
    "PlaybackCommand",
    "PlaybackWebServer",
    "PROTOCOL_VERSION",
    "WebPlaybackRunner",
    "WebHumanController",
    "history_point",
    "history_point_payload",
    "reward_accounting_contract",
    "run_web_dataset_playback",
    "run_web_player_application",
    "run_web_playback",
    "source_browser_path",
    "transition_payload",
]
