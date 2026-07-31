from __future__ import annotations

from collections import OrderedDict

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from gradlab.routed_policy import RoutedActorCriticPolicy
from gradlab.task_advantage import GroupedAdvantagePPO, normalize_advantages_by_context


def _health_policy(*, topology: str = "shared_encoder") -> RoutedActorCriticPolicy:
    observation_space = gym.spaces.Dict(
        OrderedDict(
            [
                (
                    "observation",
                    gym.spaces.Box(0, 255, shape=(4, 84, 84), dtype=np.uint8),
                ),
                (
                    "context/health",
                    gym.spaces.Box(
                        np.asarray([0.0], dtype=np.float32),
                        np.asarray([1.0], dtype=np.float32),
                        dtype=np.float32,
                    ),
                ),
            ]
        )
    )
    encoder = {"kind": "nature_cnn", "features_dim": 64}
    topology_config = (
        {"kind": "shared_encoder", "encoder": encoder}
        if topology == "shared_encoder"
        else {
            "kind": "separate_encoders",
            "encoders": {"action": encoder, "state_value": encoder},
        }
    )
    return RoutedActorCriticPolicy(
        observation_space,
        gym.spaces.Discrete(3),
        lambda _: 1e-3,
        policy_model={
            "schema_version": 1,
            "topology": topology_config,
            "fusion": "post_encoder_concat",
            "context_encoders": {"health": {"kind": "identity"}},
            "routes": {"health": ["state_value"]},
            "heads": {
                "action": {"hidden_sizes": [], "activation": "tanh"},
                "state_value": {"hidden_sizes": [], "activation": "tanh"},
            },
            "normalize_images": True,
            "orthogonal_init": True,
        },
    )


def test_health_is_value_only_and_action_path_does_not_require_it() -> None:
    policy = _health_policy()
    image = torch.zeros((2, 4, 84, 84), dtype=torch.uint8)
    full_obs = {
        "observation": image,
        "context/health": torch.tensor([[0.1], [0.9]], dtype=torch.float32),
    }

    logits = policy.get_distribution(full_obs).distribution.logits
    partial_logits = policy.get_distribution({"observation": image}).distribution.logits
    values = policy.predict_values(full_obs)

    torch.testing.assert_close(logits[0], logits[1])
    torch.testing.assert_close(logits, partial_logits)
    assert not torch.allclose(values[0], values[1])


def test_separate_encoder_topology_supports_value_only_context() -> None:
    policy = _health_policy(topology="separate_encoders")
    obs = {
        "observation": torch.zeros((2, 4, 84, 84), dtype=torch.uint8),
        "context/health": torch.tensor([[0.2], [0.8]], dtype=torch.float32),
    }

    actions, values, log_prob = policy(obs)

    assert actions.shape == (2,)
    assert values.shape == (2, 1)
    assert log_prob.shape == (2,)
    assert policy.pi_features_extractor is not policy.vf_features_extractor


def test_named_grouped_advantages_use_categorical_context_indices() -> None:
    advantages = np.asarray([[1.0, 3.0], [10.0, 14.0]], dtype=np.float32)
    observations = {
        "context/task_id": np.asarray([[[0], [0]], [[1], [1]]], dtype=np.int64)
    }

    stats = normalize_advantages_by_context(
        advantages,
        observations,
        "task_id",
    )

    np.testing.assert_allclose(advantages, [[-1.0, 1.0], [-1.0, 1.0]])
    assert set(stats) == {0, 1}


def test_grouped_ppo_trains_saves_and_loads_named_task_id_routing(tmp_path) -> None:
    class TaskEnv(gym.Env):
        observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Box(
                    -1.0,
                    1.0,
                    shape=(4,),
                    dtype=np.float32,
                ),
                "context/task_id": gym.spaces.Discrete(2),
            }
        )
        action_space = gym.spaces.Discrete(2)

        def __init__(self) -> None:
            self.step_index = 0
            self.task_id = 0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.step_index = 0
            self.task_id = 1 - self.task_id
            return {
                "observation": np.zeros(4, dtype=np.float32),
                "context/task_id": self.task_id,
            }, {}

        def step(self, action):
            self.step_index += 1
            return (
                {
                    "observation": np.full(
                        4,
                        self.step_index / 10,
                        dtype=np.float32,
                    ),
                    "context/task_id": self.task_id,
                },
                float(action == self.task_id),
                self.step_index == 3,
                False,
                {},
            )

    policy_model = {
        "schema_version": 1,
        "topology": {
            "kind": "shared_encoder",
            "encoder": {"kind": "flatten"},
        },
        "fusion": "post_encoder_concat",
        "context_encoders": {"task_id": {"kind": "one_hot"}},
        "routes": {"task_id": ["action", "state_value"]},
        "heads": {
            "action": {"hidden_sizes": [8], "activation": "tanh"},
            "state_value": {"hidden_sizes": [8], "activation": "tanh"},
        },
        "normalize_images": False,
        "orthogonal_init": True,
    }
    env = DummyVecEnv([TaskEnv, TaskEnv])
    try:
        model = GroupedAdvantagePPO(
            RoutedActorCriticPolicy,
            env,
            advantage_context="task_id",
            n_steps=4,
            batch_size=4,
            n_epochs=1,
            policy_kwargs={"policy_model": policy_model},
            verbose=0,
        )

        model.learn(total_timesteps=8)
        checkpoint = tmp_path / "grouped.zip"
        model.save(checkpoint)
        loaded = GroupedAdvantagePPO.load(checkpoint, env=env)

        assert model.num_timesteps == 8
        assert loaded.advantage_context == "task_id"
        assert loaded.policy.policy_model["routes"] == {
            "task_id": ["action", "state_value"]
        }
    finally:
        env.close()
