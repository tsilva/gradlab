from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn

from gradlab.policy_model_config import normalize_policy_model
from gradlab.routed_policy import RoutedActorCriticPolicy
from gradlab.task_advantage import GroupedAdvantagePPO, normalize_advantages_by_context
from gradlab.training.sb3_on_policy import validate_resumed_policy_model


def _plain_policy_model() -> dict:
    return {
        "schema_version": 1,
        "topology": {"kind": "shared_encoder", "encoder": {"kind": "flatten"}},
        "fusion": "post_encoder_concat",
        "context_encoders": {},
        "routes": {},
        "heads": {
            "action": {"hidden_sizes": [8, 4], "activation": "relu"},
            "state_value": {"hidden_sizes": [6], "activation": "tanh"},
        },
        "normalize_images": False,
        "orthogonal_init": True,
    }


def _plain_policy() -> RoutedActorCriticPolicy:
    return RoutedActorCriticPolicy(
        gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
        gym.spaces.Discrete(3),
        lambda _: 1e-3,
        policy_model=_plain_policy_model(),
    )


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


def test_plain_box_observations_use_independent_configured_heads() -> None:
    policy = _plain_policy()
    observations = torch.zeros((2, 4), dtype=torch.float32)

    actions, values, log_prob = policy(observations)

    assert actions.shape == (2,)
    assert values.shape == (2, 1)
    assert log_prob.shape == (2,)
    assert len(policy.mlp_extractor.policy_net) == 4
    assert isinstance(policy.mlp_extractor.policy_net[0], nn.Linear)
    assert policy.mlp_extractor.policy_net[0].in_features == 4
    assert policy.mlp_extractor.policy_net[0].out_features == 8
    assert isinstance(policy.mlp_extractor.policy_net[1], nn.ReLU)
    assert isinstance(policy.mlp_extractor.policy_net[2], nn.Linear)
    assert policy.mlp_extractor.policy_net[2].in_features == 8
    assert policy.mlp_extractor.policy_net[2].out_features == 4
    assert isinstance(policy.mlp_extractor.value_net[0], nn.Linear)
    assert policy.mlp_extractor.value_net[0].in_features == 4
    assert policy.mlp_extractor.value_net[0].out_features == 6
    assert isinstance(policy.mlp_extractor.value_net[1], nn.Tanh)
    assert policy.action_net.in_features == 4
    assert policy.value_net.in_features == 6


def test_health_and_time_are_blended_by_256_unit_critic_head_only() -> None:
    observation_space = gym.spaces.Dict(
        OrderedDict(
            [
                (
                    "observation",
                    gym.spaces.Box(0, 255, shape=(4, 84, 84), dtype=np.uint8),
                ),
                (
                    "context/health",
                    gym.spaces.Box(-1.0, 2.0, shape=(1,), dtype=np.float32),
                ),
                (
                    "context/remaining_time",
                    gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                ),
            ]
        )
    )
    policy = RoutedActorCriticPolicy(
        observation_space,
        gym.spaces.Discrete(3),
        lambda _: 1e-3,
        policy_model={
            "schema_version": 1,
            "topology": {
                "kind": "shared_encoder",
                "encoder": {"kind": "nature_cnn", "features_dim": 512},
            },
            "fusion": "post_encoder_concat",
            "context_encoders": {
                "health": {"kind": "identity"},
                "remaining_time": {"kind": "identity"},
            },
            "routes": {
                "health": ["state_value"],
                "remaining_time": ["state_value"],
            },
            "heads": {
                "action": {"hidden_sizes": [], "activation": "tanh"},
                "state_value": {"hidden_sizes": [256], "activation": "tanh"},
            },
            "normalize_images": True,
            "orthogonal_init": True,
        },
    )

    assert len(policy.mlp_extractor.policy_net) == 0
    critic_hidden = policy.mlp_extractor.value_net[0]
    assert isinstance(critic_hidden, nn.Linear)
    assert critic_hidden.in_features == 514
    assert critic_hidden.out_features == 256
    assert isinstance(policy.mlp_extractor.value_net[1], nn.Tanh)
    assert policy.action_net.in_features == 512
    assert policy.value_net.in_features == 256


@pytest.mark.parametrize(
    "hidden_sizes",
    [[0], [-1], [True], [32, "16"]],
)
def test_configured_heads_reject_invalid_hidden_widths(hidden_sizes) -> None:
    policy_model = _plain_policy_model()
    policy_model["heads"]["state_value"]["hidden_sizes"] = hidden_sizes

    with pytest.raises(ValueError, match="must be a positive integer"):
        normalize_policy_model(policy_model)


def test_resume_requires_exact_configured_policy_model() -> None:
    policy = _plain_policy()
    model = SimpleNamespace(policy=policy)
    requested = {"policy_model": deepcopy(policy.policy_model)}

    validate_resumed_policy_model(model, requested)

    requested["policy_model"]["heads"]["state_value"]["hidden_sizes"] = [7]
    with pytest.raises(ValueError, match="does not match"):
        validate_resumed_policy_model(model, requested)
    with pytest.raises(ValueError, match="both declare"):
        validate_resumed_policy_model(SimpleNamespace(policy=object()), requested)


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
