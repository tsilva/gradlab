from __future__ import annotations

import numpy as np
import pytest

from rlab.env_identity import environment_hash, environment_identity_from_train_config
from rlab.reward_transform import (
    LEGACY_SIGN_CLIP,
    migrate_legacy_artifact_reward_config,
    normalize_reward_mapping,
    reward_transform_from_reward,
)
from rlab.task_kernels import RewardTransformTaskKernel, TaskStep, with_reward_transform


class _StaticKernel:
    def __init__(self, rewards: list[float]):
        self.num_envs = len(rewards)
        self.rewards = np.asarray(rewards, dtype=np.float32)
        self.terminated = np.zeros(self.num_envs, dtype=np.bool_)
        self.truncated = np.zeros(self.num_envs, dtype=np.bool_)
        self.outcomes = np.zeros(self.num_envs, dtype=np.uint8)
        self.events = np.zeros(self.num_envs, dtype=np.uint64)
        self.metrics = {"component_reward_component": self.rewards.copy()}

    def process(self, *_args, **_kwargs) -> TaskStep:
        return TaskStep(
            self.rewards,
            self.terminated,
            self.truncated,
            self.outcomes,
            self.events,
            self.metrics,
        )


def test_normalizes_scale_and_clip_to_one_canonical_form() -> None:
    assert normalize_reward_mapping(
        {"reward_mode": "native", "reward_scale": 100, "reward_clip": True},
        label="reward",
    ) == {
        "reward_mode": "native",
        "reward_scale": 100.0,
        "reward_clip": [-1.0, 1.0],
    }

    with pytest.raises(ValueError, match="positive finite"):
        normalize_reward_mapping({"reward_scale": 0}, label="reward")
    with pytest.raises(ValueError, match="low <= high"):
        normalize_reward_mapping({"reward_clip": [1, -1]}, label="reward")


def test_common_transform_scales_before_clipping_and_preserves_raw_metrics() -> None:
    kernel = _StaticKernel([-9.0, -4.0, 97.0])
    transform = reward_transform_from_reward({"reward_scale": 100.0, "reward_clip": [-0.05, 0.5]})
    wrapped = RewardTransformTaskKernel(kernel, transform)

    step = wrapped.process(
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.bool_),
        np.zeros(3, dtype=np.bool_),
        {},
    )

    np.testing.assert_allclose(step.rewards, [-0.05, -0.04, 0.5])
    np.testing.assert_allclose(step.metrics["raw_reward"], [-9.0, -4.0, 97.0])
    np.testing.assert_allclose(step.metrics["shaped_reward"], [-0.05, -0.04, 0.5])
    np.testing.assert_allclose(
        step.metrics["component_reward_component"],
        [-9.0, -4.0, 97.0],
    )


def test_disabled_transform_returns_the_original_kernel() -> None:
    kernel = _StaticKernel([1.0])

    assert (
        with_reward_transform(
            kernel,
            {"reward_scale": 1.0, "reward_clip": False},
        )
        is kernel
    )


def test_legacy_provider_and_mario_reward_controls_translate_at_artifact_load() -> None:
    args, identity_task = migrate_legacy_artifact_reward_config(
        {"reward_clip": True, "num_threads": 8},
        {"id": "identity", "reward": {"reward_mode": "native"}},
    )
    assert args == {"num_threads": 8}
    assert identity_task["reward"]["reward_clip"] == [-1.0, 1.0]

    _args, mario_task = migrate_legacy_artifact_reward_config(
        {},
        {
            "id": "mario",
            "reward": {
                "reward_mode": "baseline",
                "clip_rewards": True,
                "reward_scale": 10.0,
                "time_penalty": 0.001,
            },
        },
    )
    assert mario_task["reward"]["reward_clip"] == LEGACY_SIGN_CLIP
    assert mario_task["reward"]["reward_scale"] == 10.0
    assert mario_task["reward"]["time_penalty"] == pytest.approx(0.01)

    _args, doubly_configured_mario_task = migrate_legacy_artifact_reward_config(
        {"reward_clip": False},
        {
            "id": "mario",
            "reward": {
                "reward_mode": "baseline",
                "clip_rewards": True,
                "reward_scale": 10.0,
                "time_penalty": 0.001,
            },
        },
    )
    assert doubly_configured_mario_task["reward"]["reward_clip"] == LEGACY_SIGN_CLIP


def test_reward_transform_changes_environment_hash() -> None:
    def identity(clip, *, scale=1.0):
        return environment_identity_from_train_config(
            {
                "env_provider": "rlab",
                "game": "Bandit-v0",
                "task": {
                    "id": "identity",
                    "action": {"set": "native"},
                    "signals": {},
                    "events": {},
                    "termination": {"max_episode_steps": 1},
                    "reward": {
                        "reward_mode": "native",
                        "reward_scale": scale,
                        "reward_clip": clip,
                    },
                },
            }
        )

    assert environment_hash(identity(False)) != environment_hash(identity(True))
    assert environment_hash(identity(False)) != environment_hash(identity(False, scale=100.0))
