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
