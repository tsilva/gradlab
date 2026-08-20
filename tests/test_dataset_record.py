from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import pytest

pytest.importorskip("datasets")

from gradlab.dataset_media import iter_episode_frames  # noqa: E402
from gradlab.batch_runtime import ProviderDescriptor  # noqa: E402
from gradlab.dataset_providers import (  # noqa: E402
    EnvironmentArtifact,
    create_provider_session,
    validate_provider_request,
)
from gradlab.dataset_record import _record_one, _recover_active_episode  # noqa: E402
from gradlab.dataset_store import validate_tree  # noqa: E402
from gradlab.json_utils import canonical_json_bytes  # noqa: E402


class FakeEnv:
    def __init__(self) -> None:
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(0, 255, (4, 5, 3), dtype=np.uint8)
        self.step_index = 0
        self.closed = False

    def reset(self, *, seed):
        self.action_space.seed(seed)
        self.step_index = 0
        return np.zeros((4, 5, 3), dtype=np.uint8), {}

    def step(self, action):
        assert self.action_space.contains(int(action))
        self.step_index += 1
        return (
            np.full((4, 5, 3), self.step_index, dtype=np.uint8),
            float(self.step_index),
            self.step_index == 2,
            False,
            {"step": self.step_index},
        )

    def close(self):
        self.closed = True


class FakeSession:
    provider_id = "gradlab"
    environment_id = "fixture-v0"
    fps = 30.0

    def __init__(self) -> None:
        self.env = FakeEnv()

    def recording_observation(self, observation):
        return observation


def _environment() -> EnvironmentArtifact:
    document = {
        "document_type": "gymrec.environment",
        "format_version": 1,
        "provider_id": "gradlab",
        "provider_contract_version": 1,
        "environment_id": "fixture-v0",
        "declared_config": {},
        "effective_config": {},
        "provenance": {"distribution": "gradlab", "version": "0.1.0", "assets": {}},
        "action_space": {"type": "Discrete", "n": 2, "start": 0},
        "observation_space": {
            "type": "Box",
            "shape": [4, 5, 3],
            "dtype": "uint8",
            "low": 0,
            "high": 255,
        },
        "control_profile": "fixture",
        "fps": 30.0,
    }
    return EnvironmentArtifact(hashlib.sha256(canonical_json_bytes(document)).hexdigest(), document)


@pytest.mark.parametrize("state", [["Level1-1", "Level1-2"], {"Level1-1": 1.0}])
def test_recording_rejects_collection_valued_states_before_provider_construction(state):
    with pytest.raises(ValueError, match="scalar state"):
        validate_provider_request({"state": state})


def test_recording_constructs_the_shared_native_provider_runtime():
    vector_env = type(
        "FakeVectorEnv",
        (),
        {
            "num_envs": 1,
            "single_action_space": gym.spaces.Discrete(2),
            "single_observation_space": gym.spaces.Box(
                0, 255, (4, 84, 84), dtype=np.uint8
            ),
            "close": lambda self: None,
        },
    )()
    descriptor = ProviderDescriptor(
        provider_id="env-stableretro-turbo",
        native_observation_space=vector_env.single_observation_space,
        native_action_space=vector_env.single_action_space,
    )
    binding = object()
    with (
        patch("gradlab.dataset_providers.rom_asset_manifest_for_game", return_value={"game": "x"}),
        patch("gradlab.dataset_providers.bind_cached_rom", return_value=binding),
        patch(
            "gradlab.dataset_providers.make_native_provider",
            return_value=(vector_env, descriptor),
        ) as make_provider,
        patch("gradlab.dataset_providers.portable_rom_asset_identity", return_value={"sha256": "a"}),
        patch("gradlab.dataset_providers.provider_buttons", return_value=("A", "B")),
        patch("gradlab.dataset_providers.declared_action_contract", return_value=None),
    ):
        session = create_provider_session(
            "env-stableretro-turbo",
            "SuperMarioBros-Nes-v0",
            {"frame_skip": 2, "env_args": {"players": 1}},
        )

    resolved, n_envs = make_provider.call_args.args
    assert resolved.env_provider == "env-stableretro-turbo"
    assert resolved.game == "SuperMarioBros-Nes-v0"
    assert resolved.frame_skip == 2
    assert resolved.env_args["players"] == 1
    assert n_envs == 1
    assert make_provider.call_args.kwargs["rom_binding"] is binding
    assert session.fps == 30.0
    session.env.close()


def test_recording_rejects_provider_constructor_arguments_at_the_dataset_boundary():
    with pytest.raises(ValueError, match="unknown environment config"):
        validate_provider_request({"rom_path": "/tmp/noncurrent.nes"})
    with pytest.raises(ValueError, match="runtime owns"):
        validate_provider_request({"env_args": {"num_envs": 2}})


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_record_one_streams_n_plus_one_video_rows(tmp_path: Path):
    session = FakeSession()
    episode_directory = tmp_path / "episode"
    package = episode_directory / "package"
    episode_directory.mkdir()

    episode_id = _record_one(
        session=session,
        environment=_environment(),
        collector=None,
        model=None,
        agent="random",
        deterministic=False,
        seed=17,
        episode_directory=episode_directory,
        package=package,
        session_id="12251a8e-c032-47fa-bb24-fbc90f68f8f7",
        projected_rebuild=0,
    )

    validation = validate_tree(package)
    assert episode_id in validation.episode_fingerprints
    assert validation.summary.rows == 3
    assert validation.summary.transitions == 2
    assert session.env.closed
    assert not list((episode_directory / "active").glob("candidate-*.png"))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_interrupted_episode_recovers_only_verified_durable_prefix(tmp_path: Path):
    pytest.importorskip("PIL.Image")
    from PIL import Image

    session = FakeSession()
    episode_directory = tmp_path / "episode"
    package = episode_directory / "package"
    episode_directory.mkdir()
    episode_id = _record_one(
        session=session,
        environment=_environment(),
        collector=None,
        model=None,
        agent="random",
        deterministic=False,
        seed=19,
        episode_directory=episode_directory,
        package=package,
        session_id="12251a8e-c032-47fa-bb24-fbc90f68f8f7",
        projected_rebuild=0,
    )
    original = validate_tree(package)
    rows = [original.dataset[index] for index in range(len(original.dataset))]
    frames = list(iter_episode_frames(rows, root=package))
    video_relative = Path(rows[0]["video_path"])
    shutil.copy2(package / video_relative, episode_directory / video_relative.name)
    candidate = episode_directory / "active" / f"candidate-{len(rows) - 1:012d}.png"
    Image.fromarray(frames[-1], mode="RGB").save(candidate)
    shutil.rmtree(package)

    recovered_id = _recover_active_episode(episode_directory, package)

    assert recovered_id == episode_id
    recovered = validate_tree(package)
    assert recovered.collection_fingerprint == original.collection_fingerprint
