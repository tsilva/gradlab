"""Own Neutralinojs windows and their lifetimes independently of normal browsers."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import aiohttp

from gradlab.desktop_runtime import ASSETS, viewer_executable


class DesktopWindow:
    def __init__(self, executable: Path, url: str, title: str, *, role: str = "player") -> None:
        self.profile = tempfile.TemporaryDirectory(prefix="gradlab-viewer-")
        self.process: subprocess.Popen | None = None
        self._close_listener: asyncio.Task | None = None
        self._close_lock = threading.Lock()
        root = Path(self.profile.name)
        try:
            (root / "public").mkdir()
            shutil.copyfile(ASSETS / f"{role}.png", root / "icon.png")
            (root / "neutralino.config.json").write_text(
                json.dumps(
                    {
                        "applicationId": f"org.gradlab.viewer.{role}",
                        "version": "1.0.0",
                        "defaultMode": "window",
                        "url": url,
                        "port": 0,
                        "enableServer": True,
                        "documentRoot": "/public",
                        "enableNativeAPI": True,
                        "exportAuthInfo": True,
                        "nativeAllowList": ["window.show", "window.unminimize", "window.focus"],
                        "logging": {"enabled": True, "writeToLogFile": True},
                        "modes": {
                            "window": {
                                "title": title,
                                "icon": "/icon.png",
                                "width": 1440,
                                "height": 960,
                                "minWidth": 800,
                                "minHeight": 600,
                                # Neutralino 6.9.0 calls dispatch_sync(main) from the
                                # macOS close callback when this is true, which traps.
                                "exitProcessOnClose": False,
                                "useSavedState": False,
                                "injectGlobals": False,
                                "injectClientLibrary": False,
                                "extendUserAgentWith": "GradLabDesktop",
                                "newWindowPolicy": "browser",
                            }
                        },
                    }
                )
            )
            with (root / "process.log").open("wb") as log:
                self.process = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("desktop_watchdog.py")),
                        "--cleanup-directory",
                        str(root),
                        str(executable),
                        "--res-mode=directory",
                        f"--path={root}",
                    ],
                    stdin=subprocess.PIPE,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
        except BaseException:
            self.close()
            raise

    async def focus(self) -> None:
        """The native capability stays in Python; pages cannot execute native methods."""
        auth_file = Path(self.profile.name) / ".tmp/auth_info.json"
        async with asyncio.timeout(15):
            while not auth_file.is_file():
                process = self.process
                if process is None or process.poll() is not None:
                    raise RuntimeError(
                        "GradLab desktop viewer exited during startup. On Linux, install "
                        "GTK 3 and WebKitGTK 4.1, or use --no-open."
                    )
                await asyncio.sleep(0.05)
            auth = json.loads(auth_file.read_text())
            access = auth["nlToken"]
            if self._close_listener is None:
                ready = asyncio.get_running_loop().create_future()
                self._close_listener = asyncio.create_task(self._listen_for_close(auth, ready))
                await ready
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"ws://127.0.0.1:{int(auth['nlPort'])}?connectToken={auth['nlConnectToken']}"
                ) as socket:
                    for method in ("window.show", "window.unminimize", "window.focus"):
                        request_id = uuid4().hex
                        await socket.send_json(
                            {"id": request_id, "method": method, "data": {}, "accessToken": access}
                        )
                        async for message in socket:
                            if message.type != aiohttp.WSMsgType.TEXT:
                                continue
                            result = json.loads(message.data)
                            if result.get("id") == request_id:
                                if not result.get("data", {}).get("success"):
                                    raise RuntimeError(f"Desktop viewer could not {method}")
                                break
                        else:
                            raise RuntimeError("Desktop viewer disconnected")

    async def _listen_for_close(self, auth: dict, ready: asyncio.Future) -> None:
        address = (
            f"ws://127.0.0.1:{int(auth['nlPort'])}"
            f"?connectToken={auth['nlConnectToken']}"
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(address) as socket:
                    ready.set_result(None)
                    async for message in socket:
                        if message.type != aiohttp.WSMsgType.TEXT:
                            continue
                        if json.loads(message.data).get("event") == "windowClose":
                            return
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
        finally:
            if ready.done():
                # Closing the watchdog pipe sends SIGTERM to its native child,
                # avoiding Neutralino's reentrant macOS _close. A lost listener
                # also stops the viewer instead of leaving an unclosable window.
                await asyncio.to_thread(self.close)

    def close(self) -> None:
        with self._close_lock:
            if self.process is not None:
                if self.process.stdin is not None and not self.process.stdin.closed:
                    self.process.stdin.close()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    self.process.wait(timeout=5)
                self.process = None
            self.profile.cleanup()


class PlaybackBrowser:
    def __init__(self) -> None:
        self.windows: dict[str, DesktopWindow] = {}
        self.workspace_id = uuid4().hex
        self.executables: dict[str, Path] = {}
        self._lock = asyncio.Lock()
        self._closing = False

    def desktop_url(self, url: str) -> str:
        parts = urlsplit(url)
        query = dict(parse_qsl(parts.query))
        query["desktop"] = self.workspace_id
        return urlunsplit(parts._replace(query=urlencode(query)))

    @staticmethod
    def _window_closed(window: DesktopWindow) -> bool:
        process = window.process
        return process is None or process.poll() is not None

    async def open(self, url: str, window: str = "main") -> None:
        async with self._lock:
            player = self.windows.get("main")
            if self._closing or (player is not None and self._window_closed(player)):
                raise RuntimeError("Player window is closed; the desktop session is shutting down")
            existing = self.windows.get(window)
            if existing is not None and self._window_closed(existing):
                await asyncio.to_thread(existing.close)
                del self.windows[window]
                existing = None
            if existing is None:
                role = "player" if window == "main" else "stats"
                if role not in self.executables:
                    self.executables[role] = await asyncio.to_thread(viewer_executable, role)
                title = f"GradLab — {role.title()}"
                if window not in {"main", "stats"}:
                    title += f" · {window.title()}"
                existing = DesktopWindow(
                    self.executables[role],
                    self.desktop_url(url),
                    title,
                    role=role,
                )
                self.windows[window] = existing
            try:
                await existing.focus()
            except BaseException:
                if window == "main":
                    self._closing = True
                await asyncio.to_thread(existing.close)
                self.windows.pop(window, None)
                raise

    async def wait_closed(self) -> None:
        """Player owns desktop lifetime; Stats and external tabs cannot keep it alive."""
        while True:
            async with self._lock:
                player = self.windows.get("main")
                if self._closing or (player is not None and self._window_closed(player)):
                    self._closing = True
                    return
            await asyncio.sleep(0.1)

    def close(self) -> None:
        self._closing = True
        for window in self.windows.values():
            window.close()
        self.windows.clear()
