from __future__ import annotations

import queue
import time
from types import SimpleNamespace

from rlab.play_application import PlaybackHost
from rlab.play_runtime import ActivePlayback, PlaySourceSpec, _implicit_playback_seed
from rlab.play_web import PlaybackCommand


class FakeEncoder:
    def __init__(self) -> None:
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def latest(self):
        return {}


class FakeRunner:
    def __init__(self) -> None:
        self.encoder = FakeEncoder()
        self.responses: queue.SimpleQueue = queue.SimpleQueue()
        self.started = False
        self.stopped = False
        self.commands = []

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def clear_input(self) -> None:
        pass

    def update_input(self, _labels, *, focused: bool) -> None:
        pass

    def submit(self, command) -> None:
        self.commands.append(command)

    def snapshot(self):
        return {
            "type": "snapshot",
            "protocol": 3,
            "revision": 4,
            "sequence": 9,
            "run_state": "paused",
            "session": {"env_id": "Game-v0", "episode": 1, "step": 9},
            "transition": None,
        }

    def history_payload(self):
        return {"type": "history", "points": [{"sequence": 9}]}

    def episode_start_payload(self):
        return {}, {}


class FakeCandidate:
    def __init__(self, spec: PlaySourceSpec, *, approval_required: bool) -> None:
        self.spec = spec
        self.approval_required = approval_required
        self.staged = SimpleNamespace(manifest_hash="a" * 64)
        self.cleaned = False

    def approval_payload(self):
        return {
            "source": self.spec.value,
            "manifest_hash": self.staged.manifest_hash,
            "files": [],
            "warning": "warning",
        }

    def cleanup(self) -> None:
        self.cleaned = True


class FakeLoader:
    def __init__(self, *, approval_required: bool = False) -> None:
        self.approval_required = approval_required
        self.activation_hashes = []
        self.runners = []
        self.prepared_specs = []

    def prepare(self, spec, progress):
        progress("verifying", "Verifying fixture")
        self.prepared_specs.append(spec)
        return FakeCandidate(spec, approval_required=self.approval_required)

    def activate(self, candidate, *, approval_hash: str, progress):
        progress("loading", "Loading fixture")
        self.activation_hashes.append(approval_hash)
        runner = FakeRunner()
        self.runners.append(runner)
        return ActivePlayback(
            runner=runner,
            policy_env=SimpleNamespace(close=lambda: None),
            spec=candidate.spec,
            source=SimpleNamespace(),
        )


def wait_for_phase(host: PlaybackHost, phase: str) -> dict:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        snapshot = host.snapshot()
        if snapshot["app"]["phase"] == phase:
            return snapshot
        time.sleep(0.005)
    raise AssertionError(f"host did not reach {phase}: {host.snapshot()}")


def source_command(name: str, payload: dict) -> PlaybackCommand:
    return PlaybackCommand("command", "client", name, payload, None)


def test_playback_host_starts_without_a_source_then_activates_selection() -> None:
    loader = FakeLoader()
    host = PlaybackHost(loader)
    host.start()
    assert host.snapshot()["app"]["phase"] == "selecting"

    host.submit(
        source_command(
            "select_source",
            {
                "source": {
                    "kind": "manifest",
                    "value": "https://models.example/manifest.json",
                    "seed": 42_000,
                },
                "route": {
                    "level": "checkpoints",
                    "entity": "research",
                    "project": "Mario",
                    "run_id": "rlab-" + "b" * 32,
                },
            },
        )
    )
    snapshot = wait_for_phase(host, "active")

    assert snapshot["session_epoch"] == 1
    assert snapshot["app"]["route"]["project"] == "Mario"
    assert loader.runners[0].encoder.epoch == 1
    assert loader.runners[0].started is True
    assert loader.prepared_specs[0].seed == 42_000
    host.stop()


def test_implicit_playback_seed_prefers_evaluation_result_then_training() -> None:
    recipe = {"train_config": {"seed": 7}}

    assert _implicit_playback_seed(recipe, evaluation_result_seed=42_000) == 42_000
    assert _implicit_playback_seed(recipe, evaluation_result_seed=None) == 7


def test_browse_sources_updates_the_shared_resource_route() -> None:
    host = PlaybackHost(FakeLoader())
    route = {
        "level": "checkpoints",
        "entity": "research",
        "project": "Mario",
        "run_id": "rlab-" + "b" * 32,
        "checkpoint_id": "",
    }

    host.submit(source_command("browse_sources", {"route": route}))

    snapshot = host.snapshot()
    assert snapshot["app"]["phase"] == "selecting"
    assert snapshot["app"]["route"] == route
    host.stop()


def test_playback_host_requires_the_exact_browser_approval_hash() -> None:
    loader = FakeLoader(approval_required=True)
    source = PlaySourceSpec("local", "/tmp/model.zip")
    host = PlaybackHost(loader, initial_source=source)
    host.start()

    approval = wait_for_phase(host, "approval_required")["app"]["approval"]
    assert approval["manifest_hash"] == "a" * 64

    host.submit(
        source_command(
            "approve_source",
            {"manifest_hash": approval["manifest_hash"]},
        )
    )
    wait_for_phase(host, "active")

    assert loader.activation_hashes == ["a" * 64]
    host.stop()
