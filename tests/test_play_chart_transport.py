import json
from gradlab.play_chart_transport import ChartResponses


def test_chart_response_deltas_replace_constants_and_only_send_changed_rows():
    transport = ChartResponses()

    def result(last, value=2):
        return dict(
            episode_id="one",
            first=1,
            last=last,
            episode_first=1,
            episode_last=last,
            points=[
                dict(
                    step=i,
                    episode=1,
                    signals={"ball": i},
                    value=value if i == last else 1,
                    return_estimate_step=last,
                )
                for i in range(1, last + 1)
            ],
        )

    first = transport.encode(result(100))
    update = transport.encode(result(101), first["revision"])
    assert update["base"] == first["revision"]
    assert len(update["rows"]) == 2
    assert len(json.dumps(update)) < len(json.dumps(first)) / 2
    assert [["return_estimate_step"], 101] in [[list(p), v] for p, v in update["constants"]]
    unchanged = transport.encode(result(101), update["revision"])
    assert unchanged["rows"] == []
    assert transport.encode(result(101), "expired")["base"] is None


def test_changed_range_removes_old_steps_and_other_episode_resets():
    transport = ChartResponses()
    first = transport.encode(dict(episode_id="one", points=[dict(step=i) for i in range(3)]))
    next_value = dict(episode_id="one", points=[dict(step=i) for i in range(1, 3)])
    update = transport.encode(next_value, first["revision"])
    assert update["removed"] == [0]
    assert update["rows"] == []
    next_value["episode_id"] = "two"
    assert transport.encode(next_value, update["revision"])["base"] is None


def test_chart_endpoint_compresses_full_responses_and_serves_revision_deltas():
    import asyncio
    from types import SimpleNamespace
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    from gradlab.play_web import PlaybackWebServer

    async def scenario():
        result = dict(episode_id="one", points=[dict(step=i, value=i % 13) for i in range(3000)])
        runner = SimpleNamespace(session_epoch=1, read_diagnostics=lambda *args: result)
        server = PlaybackWebServer(runner, SimpleNamespace())
        app = web.Application()
        app.router.add_get("/charts", server.chart_history)
        async with TestClient(TestServer(app)) as client:
            headers = {"Authorization": f"Bearer {server.token}", "Accept-Encoding": "deflate"}
            params = dict(epoch="1", episode_id="one", format="chart-columns-v1")
            async with client.get("/charts", headers=headers, params=params) as response:
                assert response.status == 200
                assert response.headers["Content-Encoding"] == "deflate"
                first = await response.json()
                assert len(first["rows"]) == 3000
                assert int(response.headers["Content-Length"]) < len(json.dumps(first)) / 2
            params["base"] = first["revision"]
            async with client.get("/charts", headers=headers, params=params) as response:
                assert response.status == 200
                update = await response.json()
                assert update["base"] == first["revision"]
                assert update["rows"] == []
            params.pop("format")
            async with client.get("/charts", headers=headers, params=params) as response:
                assert await response.json() == result

    asyncio.run(scenario())
