"""The desktop HTTP boundary preserves authentication and session-scoped synchronization."""

import asyncio
import json
import os
from pathlib import Path
import sys
from unittest.mock import AsyncMock, Mock

from aiohttp import ClientSession, WSServerHandshakeError
import pytest

from gradlab.play_web import HumanRecordingRunner, PlaybackWebServer
from tests.test_play_web import FakeHumanSession, human_args


def test_desktop_windows_require_auth_and_peer_frames_are_relayed():
    async def scenario():
        runner = HumanRecordingRunner(FakeHumanSession(), human_args())
        server = PlaybackWebServer(runner, human_args(), paired_windows=True)
        task = asyncio.create_task(server.run())
        try:
            async with asyncio.timeout(5):
                while not server.origin:
                    await asyncio.sleep(0.01)
            server.desktop_browser = Mock(open=AsyncMock(), windows={})
            async with ClientSession() as session:
                headers = {"Origin": server.origin, "Authorization": f"Bearer {server.token}"}
                async with session.post(
                    server.origin + "/api/desktop/window", json={"window": "stats"}
                ) as response:
                    assert response.status == 401
                async with session.post(
                    server.origin + "/api/desktop/window",
                    headers=headers,
                    json={"window": "../../external"},
                ) as response:
                    assert response.status == 400
                async with session.post(
                    server.origin + "/api/desktop/window", headers=headers, json={"window": "stats"}
                ) as response:
                    assert response.status == 200
                server.desktop_browser.open.assert_awaited_once_with(
                    server.dashboard_urls()[1], "stats"
                )
                with pytest.raises(WSServerHandshakeError):
                    await session.ws_connect(
                        server.origin + "/api/desktop/peer",
                        headers={"Origin": "https://elsewhere.invalid"},
                    )
                bad = await session.ws_connect(server.origin + "/api/desktop/peer", headers=headers)
                await bad.send_json({"token": "wrong"})
                await bad.receive()
                assert bad.close_code == 1008
                await bad.close()
                first = await session.ws_connect(
                    server.origin + "/api/desktop/peer", headers=headers
                )
                second = await session.ws_connect(
                    server.origin + "/api/desktop/peer", headers=headers
                )
                for peer in (first, second):
                    await peer.send_json({"token": server.token})
                    assert await peer.receive_json() == {"ready": True}
                frame = {
                    "type": "inspection-frame",
                    "sequence": 12,
                    "blob": {"base64": "AAEC", "type": "image/png"},
                }
                await first.send_json(frame)
                assert json.loads((await second.receive(timeout=2)).data) == frame
                await first.close()
                await second.close()
        finally:
            server.stop_event.set()
            await asyncio.wait_for(task, 5)

    asyncio.run(scenario())


@pytest.mark.parametrize("first_closed", ["main", "stats"])
def test_player_owns_native_session_lifetime_with_browser_still_connected(first_closed):
    from unittest.mock import patch
    from gradlab.play_browser import PlaybackBrowser

    async def scenario():
        browser = PlaybackBrowser()
        windows = {key: Mock() for key in ("main", "stats")}
        for window in windows.values():
            window.process.poll.return_value = None
        browser.windows = windows
        browser.open = AsyncMock()
        runner = HumanRecordingRunner(FakeHumanSession(), human_args())
        runner.stop = Mock(wraps=runner.stop)
        server = PlaybackWebServer(runner, human_args(no_open=False), paired_windows=True)
        with patch("gradlab.play_web.PlaybackBrowser", return_value=browser):
            task = asyncio.create_task(server.run())
            try:
                async with asyncio.timeout(3):
                    while not server.origin:
                        await asyncio.sleep(0.01)
                    async with ClientSession() as client:
                        socket = await client.ws_connect(
                            server.origin + "/ws", origin=server.origin
                        )
                        await socket.send_json({"type": "hello", "token": server.token})
                        while True:
                            message = await socket.receive()
                            if message.type.name == "TEXT" and message.json()["type"] == "welcome":
                                break
                        windows[first_closed].process.poll.return_value = 0
                        if first_closed == "stats":
                            await asyncio.sleep(0.25)
                            assert not task.done(), "closing Stats must keep Player running"
                            assert not runner.stopped
                            windows["main"].close.assert_not_called()
                            assert server.clients, "exercise shutdown with a live external tab"
                            windows["main"].process.poll.return_value = 0
                        # Closing Player must also close a still-running Stats window.
                        assert await asyncio.wait_for(task, 2) == 0
                        assert runner.stopped
                        runner.stop.assert_called_once()
                        assert server.clients == {}
                        assert browser.windows == {}
                        for window in windows.values():
                            window.close.assert_called_once()
                        await socket.close()
            finally:
                server.stop_event.set()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_native_window_lifetime_does_not_depend_on_a_page_connection():
    from unittest.mock import patch
    from gradlab.play_browser import PlaybackBrowser

    async def scenario():
        browser = PlaybackBrowser()
        window = Mock()
        window.process.poll.return_value = None
        browser.windows = {"main": window}
        browser.open = AsyncMock()
        runner = HumanRecordingRunner(FakeHumanSession(), human_args())
        server = PlaybackWebServer(runner, human_args(no_open=False))
        with (
            patch("gradlab.play_web.PlaybackBrowser", return_value=browser),
            patch("gradlab.play_web.LAST_CLIENT_GRACE_SECONDS", 0),
        ):
            task = asyncio.create_task(server.run())
            try:
                async with asyncio.timeout(3):
                    while not browser.open.called:
                        await asyncio.sleep(0.01)
                server.ever_connected = True
                server.last_client_at = 0
                await asyncio.sleep(0.25)
                assert not task.done(), "a reload must not close an open native window"
                window.process.poll.return_value = 0
                assert await asyncio.wait_for(task, 2) == 0
            finally:
                server.stop_event.set()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_player_close_reaps_companion_process_and_stops_app(tmp_path):
    """Use real watchdog/process ownership with a headless stand-in for the GUI."""
    from unittest.mock import patch

    executable = tmp_path / "viewer"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, sys, time\n"
        "root = pathlib.Path(next(arg[7:] for arg in sys.argv if arg.startswith('--path=')))\n"
        "(root / 'viewer.pid').write_text(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    executable.chmod(0o755)

    async def scenario():
        runner = HumanRecordingRunner(FakeHumanSession(), human_args())
        server = PlaybackWebServer(runner, human_args(no_open=False), paired_windows=True)
        with (
            patch("gradlab.play_browser.viewer_executable", return_value=executable),
            patch("gradlab.play_browser.DesktopWindow.focus", new_callable=AsyncMock),
        ):
            task = asyncio.create_task(server.run())
            try:
                async with asyncio.timeout(5):
                    while (
                        server.desktop_browser is None
                        or "main" not in server.desktop_browser.windows
                    ):
                        await asyncio.sleep(0.01)
                    browser = server.desktop_browser
                    await browser.open(server.dashboard_urls()[1], "stats")
                    windows = list(browser.windows.values())
                    pid_files = [Path(window.profile.name) / "viewer.pid" for window in windows]
                    while not all(path.exists() for path in pid_files):
                        await asyncio.sleep(0.01)
                    native_pids = [int(path.read_text()) for path in pid_files]
                    watchdogs = [window.process for window in windows]
                    browser.windows["main"].process.terminate()
                    assert await task == 0
                assert runner.stopped
                assert browser.windows == {}
                assert all(process.poll() is not None for process in watchdogs)
                for pid in native_pids:
                    with pytest.raises(ProcessLookupError):
                        os.kill(pid, 0)
            finally:
                server.stop_event.set()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
