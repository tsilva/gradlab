"""Complete loopback player with controlled synthetic Checkpoint preparation."""

import argparse
import asyncio
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from aiohttp import web

from gradlab.play_application import PlaybackHost
from gradlab.play_catalog import CatalogPage, CheckpointPage
from gradlab.play_runtime import ActivePlayback
from gradlab.play_web import PlaybackCommand, PlaybackWebServer, WebPlaybackRunner
from gradlab.policy_bundle import load_policy_bundle
from tests.test_play_application import FakeCandidate
from tests.test_policy_bundle import write_bundle
from tests.web_player.fixtures.chart_player import BrowserSession

RUN = "gradlab-" + "a" * 32
ROUTE = dict(
    level="runs",
    environment_id="Game-v0",
    goal_id="fixture",
    goal_variant_id="goal-variant-fixture",
    run_id=RUN,
    checkpoint_id="",
)
CHECKPOINTS = tuple(
    dict(
        run_id=RUN,
        checkpoint_id=f"checkpoint-{step}-{letter * 16}",
        step=step,
        sha256=letter * 64,
        manifest_url=f"https://fixture.invalid/runs/{RUN}/checkpoints/{step}-{letter * 64}/manifest.json",
    )
    for step, letter in ((100, "a"), (200, "b"))
)


class SelectionCatalog:
    def environments(self, **_kwargs):
        return CatalogPage((dict(name="Game-v0", goal_count=1, run_count=1),), None)

    def checkpoints(self, **_kwargs):
        return CheckpointPage(CHECKPOINTS, (), "f" * 64, run=dict(run_id=RUN, state="finished"))


class GatedEncoder:
    def __init__(self, encoder, loader):
        self.encoder, self.loader = encoder, loader

    def __getattr__(self, name):
        return getattr(self.encoder, name)

    def latest(self):
        return {} if self.loader.hold_frames else self.encoder.latest()


class SelectionRunner(WebPlaybackRunner):
    def set_processing(self, features):
        # The synthetic recording includes images even before a window mounts panels.
        super().set_processing(set(features) | {"game", "observation"})

    def start(self):
        super().start()
        for _ in range(3):
            self._step_once()
        self.drain_snapshot_updates()

    def episode_start_payload(self):
        snapshot, frames = super().episode_start_payload()
        return snapshot, {} if self.encoder.loader.hold_frames else frames


class SelectionHost(PlaybackHost):
    def submit(self, command):
        self.loader.commands.append(command.name)
        if command.name == "select_source" and self.loader.reject:
            self.loader.reject = False
            self._response(command, ok=False, error="Fixture command rejected")
            return
        super().submit(command)


class SelectionLoader:
    def __init__(self, args, root):
        self.base_args = args
        self.root = root
        self.ready = threading.Event()
        self.failure = False
        self.hold_frames = False
        self.reject = False
        self.commands = []
        self.prepared = []
        self.prefetched = []
        self.runners = []

    def prepare(self, spec, progress):
        self.prepared.append(spec.checkpoint_id)
        progress("loading", "Waiting for fixture preparation")
        if not self.ready.wait(120):
            raise RuntimeError("Fixture preparation was not released")
        self.ready.clear()
        if self.failure:
            self.failure = False
            raise RuntimeError("Fixture preparation failed")
        return FakeCandidate(spec)

    def activate(self, candidate, *, progress):
        session = BrowserSession(length=1000)
        session.current_frame[:] = (
            (220, 40, 40) if "100" in candidate.spec.checkpoint_id else (40, 80, 220)
        )
        session.step_index = 100 if "100" in candidate.spec.checkpoint_id else 200
        runner = SelectionRunner(
            session,
            self.base_args,
            config_text="game: Game-v0",
            trajectory_bundle=load_policy_bundle(self.root),
        )
        runner.encoder = GatedEncoder(runner.encoder, self)
        self.runners.append(runner)
        return ActivePlayback(
            runner=runner,
            policy_env=SimpleNamespace(close=lambda: None),
            spec=candidate.spec,
            source=SimpleNamespace(),
        )

    def prefetch(self, spec):
        self.prefetched.append(spec.checkpoint_id)


class SelectionPlayer(PlaybackWebServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.browser_ready = asyncio.Event()
        self.browser_ready.set()

    async def page(self, request):
        html = (self.asset_root / "index.html").read_text()
        return web.Response(
            text=html.replace(
                "</body>",
                '<link rel="stylesheet" href="/assets/selection-player.css">'
                '<script type="module" src="/assets/selection-player.js"></script></body>',
            ),
            content_type="text/html",
        )

    async def asset(self, request):
        path = request.match_info["path"]
        if path == "sources/browser.js":
            await self.browser_ready.wait()
        if path in {"selection-player.js", "selection-player.css", "selection-checks.js"}:
            return web.FileResponse(Path(__file__).with_name(path))
        if path == "fixture-control":
            self._authorize_api(request)
            action = request.query.get("action")
            loader = self.runner.loader
            if action == "shutdown":
                loader.ready.set()
                self.stop_event.set()
            if action == "reset":
                self.runner.submit(
                    PlaybackCommand("reset", "fixture", "browse_sources", {"route": ROUTE}, None)
                )
                loader.prepared.clear()
                loader.prefetched.clear()
                loader.commands.clear()
                loader.hold_frames = False
                loader.reject = False
            if action == "import":
                from gradlab.play_trajectory import export_trajectory

                archive = await asyncio.to_thread(
                    export_trajectory, self.runner.freeze_trajectory(), loader.root / "fixture.trj"
                )
                await asyncio.to_thread(self.runner.import_trajectory, str(archive))
            if action == "hold-browser":
                self.browser_ready.clear()
            if action == "release-browser":
                self.browser_ready.set()
            if action in {"succeed", "fail"}:
                loader.failure = action == "fail"
                loader.ready.set()
            if action in {"hold-frames", "release-frames"}:
                loader.hold_frames = action == "hold-frames"
            if action == "reject":
                loader.reject = True
            if action in {"observer", "control"}:
                self.control_holder = (
                    "fixture-observer"
                    if action == "observer"
                    else next(iter(self.clients.values())).workspace_id
                )
                self.control_epoch += 1
                await self._broadcast_control()
            snapshot = self.runner.snapshot()
            return web.json_response(
                dict(
                    prepared=loader.prepared,
                    prefetched=loader.prefetched,
                    commands=loader.commands,
                    phase=snapshot["app"]["phase"],
                    epoch=snapshot["session_epoch"],
                    paused=snapshot["run_state"],
                    mode=snapshot.get("mode"),
                    activations=len(loader.runners),
                    preparing=self.runner._worker is not None and self.runner._worker.is_alive(),
                    route=snapshot["app"]["route"],
                )
            )
        return await super().asset(request)


def main():
    args = argparse.Namespace(port=0, no_open=True, episodes=0, fps=20)
    with TemporaryDirectory(prefix="gradlab-selection-") as temporary:
        root = Path(temporary)
        write_bundle(root)
        loader = SelectionLoader(args, root)
        host = SelectionHost(loader, initial_route=ROUTE)
        try:
            asyncio.run(SelectionPlayer(host, args, catalog=SelectionCatalog()).run())
        finally:
            loader.ready.set()
            host.stop()


if __name__ == "__main__":
    main()
