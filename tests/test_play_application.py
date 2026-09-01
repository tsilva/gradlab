from __future__ import annotations

from contextlib import nullcontext
import queue
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gradlab.play_application import PlaybackHost
from gradlab.model_sources import NoDefaultPublicRunCheckpointError
from gradlab.play_runtime import (
    ActivePlayback,
    PlaybackLoader,
    PlaySourceSpec,
    apply_vizdoom_playback_iwad_override,
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
        self.prefetched_specs = []

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

    def prefetch(self, spec) -> None:
        self.prefetched_specs.append(spec)


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
                    "environment_id": "Mario",
                    "run_id": "gradlab-" + "b" * 32,
                },
            },
        )
    )
    snapshot = wait_for_phase(host, "active")

    assert snapshot["session_epoch"] == 1
    assert snapshot["app"]["route"]["environment_id"] == "Mario"
    assert loader.runners[0].encoder.epoch == 1
    assert loader.runners[0].started is True
    assert loader.prepared_specs[0].seed == 42_000
    host.stop()


def test_implicit_playback_seed_prefers_evaluation_then_policy_then_training() -> None:
    recipe = {"train_config": {"seed": 7}}

    assert (
        _implicit_playback_seed(
            recipe,
            evaluation_result_seed=42_000,
            policy_seed=127,
        )
        == 42_000
    )
    assert (
        _implicit_playback_seed(
            recipe,
            evaluation_result_seed=None,
            policy_seed=127,
        )
        == 127
    )
    assert (
        _implicit_playback_seed(
            recipe,
            evaluation_result_seed=None,
        )
        == 7
    )


def test_vizdoom_playback_iwad_override_is_counterfactual_only_when_bytes_change(
    tmp_path,
) -> None:
    iwad = tmp_path / "doom2.wad"
    iwad.write_bytes(b"IWADdoom")
    environment = {
        "env_provider": "env-vizdoom-turbo",
        "env_args": {"rom_path": None},
    }

    assert apply_vizdoom_playback_iwad_override(environment, rom_path=iwad) is True
    assert environment["env_args"]["rom_path"]["filename"] == "doom2.wad"
    assert apply_vizdoom_playback_iwad_override(environment, rom_path=iwad) is False


def test_playback_loader_enforces_cpu_and_passes_bundle_algorithm(monkeypatch) -> None:
    class ActivationComplete(Exception):
        pass

    verified = object()
    model_loader = MagicMock(return_value=object())
    candidate = SimpleNamespace(
        args=SimpleNamespace(device="mps"),
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


def test_playback_loader_prefetch_resolves_to_disk_without_activation(monkeypatch) -> None:
    model_path = SimpleNamespace()
    resolved = SimpleNamespace(model_path=model_path, run_config={})
    resolver = MagicMock(return_value=resolved)
    monkeypatch.setattr("gradlab.play_runtime.resolve_model_source", resolver)
    loader = PlaybackLoader(
        SimpleNamespace(
            public_model_root="/tmp/public-models",
            hf_model_root="/tmp/hf-models",
            hf_revision=None,
            public_models_base_url="https://models.example",
        ),
        argv=[],
        explicit_seed=False,
    )
    spec = PlaySourceSpec("manifest", "https://models.example/manifest.json")

    assert loader.prefetch(spec) is model_path
    resolver.assert_called_once()


def test_browse_sources_updates_the_shared_resource_route() -> None:
    host = PlaybackHost(FakeLoader())
    route = {
        "level": "checkpoints",
        "environment_id": "Mario",
        "run_id": "gradlab-" + "b" * 32,
        "checkpoint_id": "",
    }

    host.submit(source_command("browse_sources", {"route": route}))

    snapshot = host.snapshot()
    assert snapshot["app"]["phase"] == "selecting"
    assert snapshot["app"]["route"] == route
    host.stop()


def test_playback_host_prefetches_two_immutable_neighbors_without_activation() -> None:
    loader = FakeLoader()
    host = PlaybackHost(loader)
    run_id = "gradlab-" + "a" * 32

    def manifest(step: int, digest: str) -> str:
        return f"https://models.example/runs/{run_id}/checkpoints/{step}-{digest}/manifest.json"

    host.submit(
        source_command(
            "prefetch_sources",
            {
                "sources": [
                    {
                        "kind": "public_run",
                        "value": manifest(100, "b" * 64),
                        "run_id": run_id,
                        "checkpoint_id": "checkpoint-100-" + "b" * 16,
                    },
                    {
                        "kind": "public_run",
                        "value": manifest(300, "c" * 64),
                        "run_id": run_id,
                        "checkpoint_id": "checkpoint-300-" + "c" * 16,
                    },
                ],
            },
        )
    )
    deadline = time.monotonic() + 2.0
    while len(loader.prefetched_specs) < 2 and time.monotonic() < deadline:
        time.sleep(0.005)

    assert sorted(spec.checkpoint_id for spec in loader.prefetched_specs) == sorted(
        [
            "checkpoint-100-" + "b" * 16,
            "checkpoint-300-" + "c" * 16,
        ]
    )
    assert loader.activations == 0
    host.stop()


def test_browse_sources_closes_the_active_playback() -> None:
    loader = FakeLoader()
    host = PlaybackHost(
        loader,
        initial_source=PlaySourceSpec("manifest", "https://models.example/manifest.json"),
    )
    host.start()
    wait_for_phase(host, "active")
    runner = loader.runners[0]
    route = {
        "level": "goals",
        "environment_id": "Mario",
        "goal_id": "",
        "goal_variant_id": "",
        "run_id": "",
        "checkpoint_id": "",
    }

    host.submit(source_command("browse_sources", {"route": route}))

    snapshot = host.snapshot()
    assert snapshot["app"]["phase"] == "selecting"
    assert snapshot["app"]["route"] == route
    assert snapshot["app"]["has_active_runner"] is False
    assert runner.stopped is True
    assert runner.commands == []
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


def test_direct_public_run_resolves_active_checkpoint_breadcrumb_route() -> None:
    class PublicRunLoader(FakeLoader):
        def prepare(self, spec, progress):
            candidate = super().prepare(spec, progress)
            candidate.source = SimpleNamespace(
                bundle=SimpleNamespace(
                    recipe={
                        "recipe": {
                            "goal": {"goal_id": "DefendTheLine-v1"},
                            "goal_variant": {
                                "goal_id": "DefendTheLine-v1",
                                "goal_slug": "ViZDoom/DefendTheLine-v1",
                                "variant_id": "goal-variant-" + "c" * 24,
                            },
                        },
                    },
                    model={
                        "checkpoint": {
                            "step": 10_002_432,
                            "sha256": "b" * 64,
                        },
                        "provenance": {
                            "wandb_project": "ViZDoom",
                            "wandb_run_id": "gradlab-" + "a" * 32,
                        },
                    },
                ),
            )
            return candidate

    loader = PublicRunLoader()
    run_id = "gradlab-" + "a" * 32
    host = PlaybackHost(
        loader,
        initial_route={"level": "environments", "environment_id": "ViZDoom"},
        initial_source=PlaySourceSpec(
            "public_run",
            run_id,
            run_id=run_id,
        ),
    )
    host.start()

    snapshot = wait_for_phase(host, "active")

    assert snapshot["app"]["route"] == {
        "level": "runs",
        "environment_id": "ViZDoom",
        "goal_id": "DefendTheLine-v1",
        "goal_variant_id": "goal-variant-" + "c" * 24,
        "run_id": run_id,
        "checkpoint_id": "checkpoint-10002432-" + "b" * 16,
    }
    host.stop()


def test_active_public_run_falls_back_to_checkpoint_selection() -> None:
    class ActivePublicRunLoader(FakeLoader):
        def prepare(self, spec, progress):
            raise NoDefaultPublicRunCheckpointError(
                f"run {spec.run_id} has no promoted or final checkpoint"
            )

    run_id = "gradlab-" + "a" * 32
    route = {
        "level": "runs",
        "environment_id": "ViZDoom",
        "goal_id": "DefendTheLine-v1",
        "goal_variant_id": "goal-variant-" + "c" * 24,
        "run_id": run_id,
        "checkpoint_id": "",
    }
    host = PlaybackHost(
        ActivePublicRunLoader(),
        initial_route=route,
        initial_source=PlaySourceSpec("public_run", run_id, run_id=run_id),
    )
    host.start()

    snapshot = wait_for_phase(host, "selecting")

    assert snapshot["app"]["route"] == route
    assert snapshot["app"]["error"] == ""
    assert snapshot["app"]["message"] == (
        "This run has no promoted or final checkpoint yet. "
        "Choose one of its published checkpoints to play now."
    )
    host.stop()


def test_direct_public_run_resolves_top_level_goal_breadcrumb_route() -> None:
    class PublicRunLoader(FakeLoader):
        def prepare(self, spec, progress):
            candidate = super().prepare(spec, progress)
            candidate.source = SimpleNamespace(
                bundle=SimpleNamespace(
                    recipe={
                        "recipe": {
                            "goal_variant": {
                                "goal_id": "VizdoomDeathmatch-v1",
                                "goal_slug": "VizdoomDeathmatch-v1",
                                "variant_id": "goal-variant-" + "d" * 24,
                            },
                        },
                    },
                    model={
                        "checkpoint": {"step": 23_998_464, "sha256": "e" * 64},
                    },
                ),
            )
            return candidate

    loader = PublicRunLoader()
    run_id = "gradlab-" + "e" * 32
    host = PlaybackHost(
        loader,
        initial_source=PlaySourceSpec("public_run", run_id, run_id=run_id),
    )
    host.start()

    snapshot = wait_for_phase(host, "active")

    assert snapshot["app"]["route"] == {
        "level": "runs",
        "environment_id": "VizdoomDeathmatch-v1",
        "goal_id": "VizdoomDeathmatch-v1",
        "goal_variant_id": "goal-variant-" + "d" * 24,
        "run_id": run_id,
        "checkpoint_id": "checkpoint-23998464-" + "e" * 16,
    }
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
