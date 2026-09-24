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
    @property
    def asset_root(self):
        return self.args.assets_root or super().asset_root

    async def page(self, request):
        html = (self.asset_root / "index.html").read_text()
        return web.Response(
            text=html.replace('<script type="module" src="/assets/app.js">', '<script src="/assets/performance-gates.js"></script><script type="module" src="/assets/app.js">').replace("</body>", '<script src="/assets/chart-player-controls.js"></script></body>'),
            content_type="text/html",
            headers={"Cross-Origin-Opener-Policy": "same-origin-allow-popups"},
        )

    async def asset(self, request):
        if request.match_info["path"] in {"chart-player-controls.js", "performance-gates.js"}:
            return web.FileResponse(Path(__file__).with_name(request.match_info["path"]))
        return await super().asset(request)

    async def chart_history(self, request):
        print(f"Chart request: {request.query_string}", flush=True)
        await asyncio.sleep(self.args.chart_delay)
        if self.args.chart_failures:
            self.args.chart_failures -= 1
            return web.json_response({"error": "Temporary fixture failure"}, status=503)
        return await super().chart_history(request)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path)
    parser.add_argument("--recorded-steps", type=int, default=140)
    parser.add_argument(
        "--episode-length",
        type=int,
        help="Override the synthetic episode length for stop-condition verification",
    )
    parser.add_argument("--chart-delay", type=float, default=0.3)
    parser.add_argument("--chart-failures", type=int, default=1)
    parser.add_argument("--imported", action="store_true", help="Inspect an exported data-only recording")
    args = parser.parse_args()
    if args.episode_length is not None and args.episode_length < 1:
        parser.error("--episode-length must be at least 1")
    args.port, args.no_open, args.episodes, args.fps = 0, True, 0, 20
    with TemporaryDirectory(prefix="gradlab-chart-player-") as temporary:
        root = Path(temporary)
        write_bundle(root)
        runner = WebPlaybackRunner(
            BrowserSession(
                length=args.episode_length or max(1000, args.recorded_steps + 1000)
            ),
            args,
            config_text="game: Game-v0",
            trajectory_bundle=load_policy_bundle(root),
        )
        if args.imported:
            from gradlab.play_trajectory import export_trajectory
            from gradlab.play_trajectory_runner import TrajectoryPlaybackRunner

            runner.start()
            for _ in range(args.recorded_steps):
                runner._step_once()
            archive = export_trajectory(runner.freeze_trajectory(), root / "browser.trj")
            runner.stop()
            runner = TrajectoryPlaybackRunner(archive, args)
            runner._load_step(args.recorded_steps)

        async def serve():
            server = asyncio.create_task(
                ChartPlayer(runner, args, paired_windows=True).run()
            )
            while not runner._thread.is_alive():
                await asyncio.sleep(0.01)
            if not args.imported:
                for _ in range(args.recorded_steps):
                    await asyncio.to_thread(runner._step_once)
            print(f"Fixture ready at step {args.recorded_steps}, paused", flush=True)
            return await server

        try:
            asyncio.run(serve())
        finally:
            runner.stop()


if __name__ == "__main__":
    main()
