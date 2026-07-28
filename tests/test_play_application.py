from __future__ import annotations

from contextlib import nullcontext
import queue
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gradlab.play_application import PlaybackHost
from gradlab.play_runtime import (
    ActivePlayback,
    PlaybackLoader,
    PlaySourceSpec,
    _implicit_playback_seed,
)
from gradlab.play_web import PlaybackCommand


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
    def __init__(self, spec: PlaySourceSpec) -> None:
        self.spec = spec
        self.cleaned = False

    def cleanup(self) -> None:
        self.cleaned = True


class FakeLoader:
    def __init__(self) -> None:
        self.activations = 0
        self.runners = []
        self.prepared_specs = []

    def prepare(self, spec, progress):
        progress("verifying", "Verifying fixture")
        self.prepared_specs.append(spec)
        return FakeCandidate(spec)

    def activate(self, candidate, *, progress):
        progress("loading", "Loading fixture")
        self.activations += 1
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


def wait_for_epoch(host: PlaybackHost, epoch: int) -> dict:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        snapshot = host.snapshot()
        if snapshot["app"]["phase"] == "active" and snapshot["session_epoch"] == epoch:
            return snapshot
        time.sleep(0.005)
    raise AssertionError(f"host did not reach epoch {epoch}: {host.snapshot()}")


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
                    "run_id": "gradlab-" + "b" * 32,
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


def test_playback_loader_passes_bundle_algorithm_to_model_loader(monkeypatch) -> None:
    class ActivationComplete(Exception):
        pass

    verified = object()
    model_loader = MagicMock(return_value=object())
    candidate = SimpleNamespace(
        args=SimpleNamespace(device="cpu"),
        source=SimpleNamespace(
            bundle=SimpleNamespace(
                model={
                    "policy": {
                        "training_backend_id": "sb3.ppo",
                        "algorithm_id": "ppo",
                        "model_class": "stable_baselines3.ppo.ppo.PPO",
                    }
                }
            )
        ),
        staged=object(),
    )
    monkeypatch.setattr(
        "gradlab.play_runtime.verify_staged_model",
        lambda _staged: nullcontext(verified),
    )
    monkeypatch.setattr("gradlab.play_runtime.resolve_sb3_device", lambda _device: "cpu")
    monkeypatch.setattr("gradlab.policy_models.load_policy_model", model_loader)
    monkeypatch.setattr(
        "gradlab.policy_runtime.PolicyRuntime",
        MagicMock(side_effect=ActivationComplete),
    )

    with pytest.raises(ActivationComplete):
        PlaybackLoader.__new__(PlaybackLoader).activate(
            candidate,
            progress=lambda _phase, _detail: None,
        )

    model_loader.assert_called_once_with(
        verified,
        device="cpu",
        algorithm_id="ppo",
    )


def test_browse_sources_updates_the_shared_resource_route() -> None:
    host = PlaybackHost(FakeLoader())
    route = {
        "level": "checkpoints",
        "entity": "research",
        "project": "Mario",
        "run_id": "gradlab-" + "b" * 32,
        "checkpoint_id": "",
    }

    host.submit(source_command("browse_sources", {"route": route}))

    snapshot = host.snapshot()
    assert snapshot["app"]["phase"] == "selecting"
    assert snapshot["app"]["route"] == route
    host.stop()


def test_playback_host_activates_without_model_preapproval() -> None:
    loader = FakeLoader()
    source = PlaySourceSpec("local", "/tmp/model.zip")
    host = PlaybackHost(loader, initial_source=source)
    host.start()

    snapshot = wait_for_phase(host, "active")

    assert "approval" not in snapshot["app"]
    assert loader.activations == 1
    host.stop()


def test_contract_mode_switch_atomically_replaces_the_shared_session() -> None:
    loader = FakeLoader()
    source = PlaySourceSpec("local", "/tmp/model.zip")
    host = PlaybackHost(loader, initial_source=source)
    host.start()
    wait_for_epoch(host, 1)
    first_runner = loader.runners[0]

    host.submit(source_command("set_contract_mode", {"mode": "evaluation"}))
    snapshot = wait_for_epoch(host, 2)

    assert snapshot["app"]["phase"] == "active"
    assert loader.prepared_specs[-1].contract_mode == "evaluation"
    assert loader.prepared_specs[-1].reward_clip_override is None
    assert first_runner.stopped is True
    assert loader.runners[-1].started is True

    host.submit(source_command("set_contract_mode", {"mode": "counterfactual"}))
    wait_for_epoch(host, 3)

    assert loader.prepared_specs[-1].contract_mode == "counterfactual"
    assert loader.prepared_specs[-1].reward_clip_override is False
    host.stop()
