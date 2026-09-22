"""The desktop HTTP boundary preserves authentication and session-scoped synchronization."""

import asyncio
import json
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
def test_last_native_window_stops_server_with_browser_still_connected(first_closed):
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
                        await asyncio.sleep(0.25)
                        assert not task.done(), "the other native window is still open"
                        assert server.clients, "exercise shutdown with a live external tab"
                        last = "stats" if first_closed == "main" else "main"
                        windows[last].process.poll.return_value = 0
                        assert await asyncio.wait_for(task, 2) == 0
                        assert runner.stopped
                        runner.stop.assert_called_once()
                        assert server.clients == {}
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
