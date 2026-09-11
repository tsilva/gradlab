"""Loopback player for chart lifecycle verification, without credentials or a Policy.

Run from the checkout: uv run --frozen python -m tests.web_player.fixtures.chart_player
The printed dashboard URL opens a paused, partially recorded synthetic episode.
"""

import argparse
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from aiohttp import web

from gradlab.play_web import PlaybackWebServer, WebPlaybackRunner
from gradlab.policy_bundle import load_policy_bundle
from tests.test_play_trajectory import ScriptedSession
from tests.test_policy_bundle import write_bundle


class BrowserSession(ScriptedSession):
    def reset_episode(self, seed=None):
        self.episode += 1
        self.step_index = 0
        self.total_reward = 0.0
        self.last_transition = None
        if seed is not None:
            self.active_seed = seed


class ChartPlayer(PlaybackWebServer):
    async def chart_history(self, request):
        print(f"Chart request: {request.query_string}", flush=True)
        await asyncio.sleep(self.args.chart_delay)
        if self.args.chart_failures:
            self.args.chart_failures -= 1
            return web.json_response({"error": "Temporary fixture failure"}, status=503)
        return await super().chart_history(request)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chart-delay", type=float, default=0.3)
    parser.add_argument("--chart-failures", type=int, default=1)
    args = parser.parse_args()
    args.port, args.no_open, args.episodes, args.fps = 0, True, 0, 20
    with TemporaryDirectory(prefix="gradlab-chart-player-") as temporary:
        root = Path(temporary)
        write_bundle(root)
        runner = WebPlaybackRunner(
            BrowserSession(length=1000), args, config_text="game: Game-v0",
            trajectory_bundle=load_policy_bundle(root),
        )
        async def serve():
            server = asyncio.create_task(ChartPlayer(runner, args).run())
            while not runner._thread.is_alive():
                await asyncio.sleep(0.01)
            for _ in range(140):
                await asyncio.to_thread(runner._step_once)
            print("Fixture ready at step 140, paused", flush=True)
            return await server

        try:
            asyncio.run(serve())
        finally:
            runner.stop()


if __name__ == "__main__":
    main()
