from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn

from gradlab.actor_critic_policy import (
    SharedActorCriticFeatureExtractor,
    SharedActorCriticPolicy,
)
from gradlab.play_attribution import ActionLogProbForward, actor_image_feature_extractor
from gradlab.play_debug import inspect_policy
from gradlab.policy_model_config import (
    normalize_artifact_policy_model,
    normalize_policy_model,
)
from gradlab.routed_policy import RoutedActorCriticPolicy
from gradlab.task_advantage import GroupedAdvantagePPO, normalize_advantages_by_context
from gradlab.training.sb3_on_policy import validate_resumed_policy_model


def _policy_model(
    *,
    hidden_sizes: list[int] | None = None,
    activation: str = "relu",
) -> dict:
    return {
        "schema_version": 2,
        "encoder": {"kind": "flatten"},
        "fusion": {
            "hidden_sizes": [8, 4] if hidden_sizes is None else hidden_sizes,
            "activation": activation,
        },
        "normalize_images": False,
        "orthogonal_init": True,
    }


def _plain_policy() -> SharedActorCriticPolicy:
    return SharedActorCriticPolicy(
        gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
        gym.spaces.Discrete(3),
        lambda _: 1e-3,
        policy_model=_policy_model(),
    )


def _context_policy(*, categorical: bool = False) -> SharedActorCriticPolicy:
    context_space: gym.Space = (
        gym.spaces.Discrete(2)
        if categorical
        else gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
    )
    return SharedActorCriticPolicy(
        gym.spaces.Dict(
            OrderedDict(
                [
                    (
                        "observation",
                        gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
                    ),
                    ("context/task_id" if categorical else "context/health", context_space),
                ]
            )
        ),
        gym.spaces.Discrete(2),
        lambda _: 1e-3,
        policy_model=_policy_model(hidden_sizes=[], activation="tanh"),
    )


def _legacy_policy_model() -> dict:
    return {
        "schema_version": 1,
        "topology": {"kind": "shared_encoder", "encoder": {"kind": "flatten"}},
        "fusion": "post_encoder_concat",
        "context_encoders": {},
        "routes": {},
        "heads": {
            "action": {"hidden_sizes": [], "activation": "tanh"},
            "state_value": {"hidden_sizes": [], "activation": "tanh"},
        },
        "normalize_images": False,
        "orthogonal_init": True,
    }


def test_plain_box_observations_use_one_shared_fusion_stack() -> None:
    policy = _plain_policy()
    observations = torch.zeros((2, 4), dtype=torch.float32)

    actions, values, log_prob = policy(observations)

    assert actions.shape == (2,)
    assert values.shape == (2, 1)
    assert log_prob.shape == (2,)
    extractor = policy.features_extractor
    assert isinstance(extractor, SharedActorCriticFeatureExtractor)
    assert policy.pi_features_extractor is extractor
    assert policy.vf_features_extractor is extractor
    assert len(policy.mlp_extractor.policy_net) == 0
    assert len(policy.mlp_extractor.value_net) == 0
    assert len(extractor.fusion) == 4
    assert isinstance(extractor.fusion[0], nn.Linear)
    assert extractor.fusion[0].in_features == 4
    assert extractor.fusion[0].out_features == 8
    assert isinstance(extractor.fusion[1], nn.ReLU)
    assert isinstance(extractor.fusion[2], nn.Linear)
    assert extractor.fusion[2].in_features == 8
    assert extractor.fusion[2].out_features == 4
    assert policy.action_net.in_features == 4
    assert policy.value_net.in_features == 4


def test_continuous_context_changes_both_action_distribution_and_value() -> None:
    policy = _context_policy()
    with torch.no_grad():
        policy.action_net.weight.zero_()
        policy.action_net.bias.zero_()
        policy.action_net.weight[0, 1] = -1.0
        policy.action_net.weight[1, 1] = 1.0
        policy.value_net.weight.zero_()
        policy.value_net.bias.zero_()
        policy.value_net.weight[0, 1] = 1.0
    obs = {
        "observation": torch.zeros((2, 1), dtype=torch.float32),
        "context/health": torch.tensor([[0.0], [1.0]], dtype=torch.float32),
    }

    logits = policy.get_distribution(obs).distribution.logits
    values = policy.predict_values(obs)

    assert not torch.allclose(logits[0], logits[1])
    assert not torch.allclose(values[0], values[1])


def test_categorical_context_is_one_hot_before_shared_fusion() -> None:
    policy = _context_policy(categorical=True)
    extractor = policy.features_extractor
    assert isinstance(extractor, SharedActorCriticFeatureExtractor)
    assert extractor.features_dim == 3
    obs = {
        "observation": torch.zeros((2, 1), dtype=torch.float32),
        "context/task_id": torch.tensor([0, 1], dtype=torch.int64),
    }

    features = policy.extract_features(obs)

    torch.testing.assert_close(
        features,
        torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    )
    with torch.no_grad():
        policy.action_net.weight.zero_()
        policy.action_net.bias.zero_()
        policy.action_net.weight[0, 1] = 1.0
        policy.action_net.weight[1, 2] = 1.0
        policy.value_net.weight.zero_()
        policy.value_net.bias.zero_()
        policy.value_net.weight[0, 2] = 1.0

    logits = policy.get_distribution(obs).distribution.logits
    values = policy.predict_values(obs)

    assert not torch.allclose(logits[0], logits[1])
    assert not torch.allclose(values[0], values[1])


def test_actor_and_critic_both_require_complete_context() -> None:
    policy = _context_policy()
    incomplete = {"observation": torch.zeros((1, 1), dtype=torch.float32)}

    with pytest.raises(ValueError, match="input keys disagree"):
        policy.get_distribution(incomplete)
    with pytest.raises(ValueError, match="input keys disagree"):
        policy.predict_values(incomplete)


@pytest.mark.parametrize("invalid_key", ["health", "context/", "context/group/health"])
def test_shared_policy_rejects_invalid_context_dict_keys(invalid_key: str) -> None:
    observation_space = gym.spaces.Dict(
        {
            "observation": gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
            invalid_key: gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        }
    )

    with pytest.raises(ValueError, match="unexpected keys"):
        SharedActorCriticPolicy(
            observation_space,
            gym.spaces.Discrete(2),
            lambda _: 1e-3,
            policy_model=_policy_model(hidden_sizes=[]),
        )


def test_health_and_time_share_one_256_unit_fusion_layer() -> None:
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
    policy = SharedActorCriticPolicy(
        observation_space,
        gym.spaces.Discrete(3),
        lambda _: 1e-3,
        policy_model={
            "schema_version": 2,
            "encoder": {"kind": "nature_cnn", "features_dim": 512},
            "fusion": {"hidden_sizes": [256], "activation": "tanh"},
            "normalize_images": True,
            "orthogonal_init": True,
        },
    )

    extractor = policy.features_extractor
    assert isinstance(extractor, SharedActorCriticFeatureExtractor)
    hidden = extractor.fusion[0]
    assert isinstance(hidden, nn.Linear)
    assert hidden.in_features == 514
    assert hidden.out_features == 256
    assert policy.action_net.in_features == 256
    assert policy.value_net.in_features == 256


def test_mixed_mario_shared_fusion_removes_395264_duplicate_parameters() -> None:
    policy = SharedActorCriticPolicy(
        gym.spaces.Dict(
            {
                "observation": gym.spaces.Box(-1.0, 1.0, shape=(256,), dtype=np.float32),
                "context/task_id": gym.spaces.Discrete(2),
            }
        ),
        gym.spaces.Discrete(7),
        lambda _: 1e-3,
        policy_model={
            **_policy_model(hidden_sizes=[512, 512], activation="tanh"),
            "encoder": {"kind": "flatten"},
        },
    )
    extractor = policy.features_extractor
    assert isinstance(extractor, SharedActorCriticFeatureExtractor)
    fusion_parameters = sum(parameter.numel() for parameter in extractor.fusion.parameters())

    assert fusion_parameters == 395_264
    assert 2 * fusion_parameters - fusion_parameters == 395_264


@pytest.mark.parametrize("hidden_sizes", [[0], [-1], [True], [32, "16"]])
def test_shared_fusion_rejects_invalid_hidden_widths(hidden_sizes) -> None:
    policy_model = _policy_model()
    policy_model["fusion"]["hidden_sizes"] = hidden_sizes

    with pytest.raises(ValueError, match="must be a positive integer"):
        normalize_policy_model(policy_model)


@pytest.mark.parametrize("field", ["topology", "routes", "context_encoders", "heads"])
def test_policy_model_v2_rejects_removed_role_fields(field: str) -> None:
    policy_model = _policy_model()
    policy_model[field] = {}

    with pytest.raises(ValueError, match=f"unexpected fields.*{field}"):
        normalize_policy_model(policy_model)


def test_joint_decision_uses_one_shared_feature_pass(monkeypatch) -> None:
    policy = _context_policy()
    calls = 0
    original = policy.features_extractor.forward

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(policy.features_extractor, "forward", counted)
    policy.decision_distribution_and_value(
        {
            "observation": torch.zeros((2, 1), dtype=torch.float32),
            "context/health": torch.tensor([[0.2], [0.8]], dtype=torch.float32),
        }
    )

    assert calls == 1


def test_playback_inspection_uses_shared_context_without_sampling() -> None:
    policy = _context_policy()
    before = torch.random.get_rng_state().clone()

    decision = inspect_policy(
        SimpleNamespace(policy=policy),
        {
            "observation": np.zeros((1, 1), dtype=np.float32),
            "context/health": np.asarray([[0.5]], dtype=np.float32),
        },
    )

    assert decision.value is not None
    assert decision.probabilities is not None
    assert not decision.sampled
    assert torch.equal(before, torch.random.get_rng_state())


def test_attribution_uses_shared_image_encoder_and_fixed_context() -> None:
    observation_space = gym.spaces.Dict(
        OrderedDict(
            [
                (
                    "observation",
                    gym.spaces.Box(0, 255, shape=(4, 84, 84), dtype=np.uint8),
                ),
                (
                    "context/health",
                    gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                ),
            ]
        )
    )
    policy = SharedActorCriticPolicy(
        observation_space,
        gym.spaces.Discrete(2),
        lambda _: 1e-3,
        policy_model={
            "schema_version": 2,
            "encoder": {"kind": "nature_cnn", "features_dim": 32},
            "fusion": {"hidden_sizes": [8], "activation": "tanh"},
            "normalize_images": True,
            "orthogonal_init": True,
        },
    )
    obs = {
        "observation": np.ones((1, 4, 84, 84), dtype=np.uint8),
        "context/health": np.asarray([[0.5]], dtype=np.float32),
    }
    forward = ActionLogProbForward(policy, obs, np.asarray([1]))
    image = forward.image_tensor.detach().requires_grad_(True)

    output = forward(image)
    output.sum().backward()

    assert actor_image_feature_extractor(policy) is policy.features_extractor.observation_encoder
    assert torch.equal(
        forward.fixed_obs["context/health"],
        torch.as_tensor(obs["context/health"]),
    )
    assert image.grad is not None


def test_resume_requires_exact_shared_policy_model() -> None:
    policy = _plain_policy()
    model = SimpleNamespace(policy=policy)
    requested = {"policy_model": deepcopy(policy.policy_model)}

    validate_resumed_policy_model(model, requested)

    requested["policy_model"]["fusion"]["hidden_sizes"] = [7]
    with pytest.raises(ValueError, match="does not match"):
        validate_resumed_policy_model(model, requested)
    with pytest.raises(ValueError, match="both declare"):
        validate_resumed_policy_model(SimpleNamespace(policy=object()), requested)


def test_v1_policy_is_artifact_only_and_still_loadable(tmp_path) -> None:
    legacy = _legacy_policy_model()

    with pytest.raises(ValueError, match="schema_version must be 2"):
        normalize_policy_model(legacy)
    assert normalize_artifact_policy_model(legacy)["schema_version"] == 1
    policy = RoutedActorCriticPolicy(
        gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
        gym.spaces.Discrete(2),
        lambda _: 1e-3,
        policy_model=legacy,
    )
    assert policy.policy_model["schema_version"] == 1

    env = DummyVecEnv([_PlainEnv])
    try:
        model = PPO(
            RoutedActorCriticPolicy,
            env,
            n_steps=2,
            batch_size=2,
            policy_kwargs={"policy_model": legacy},
            verbose=0,
        )
        checkpoint = tmp_path / "legacy.zip"
        model.save(checkpoint)
        loaded = PPO.load(checkpoint, env=env)

        assert isinstance(loaded.policy, RoutedActorCriticPolicy)
        assert loaded.policy.policy_model["schema_version"] == 1
    finally:
        env.close()


def test_named_grouped_advantages_use_categorical_context_indices() -> None:
    advantages = np.asarray([[1.0, 3.0], [10.0, 14.0]], dtype=np.float32)
    observations = {"context/task_id": np.asarray([[[0], [0]], [[1], [1]]], dtype=np.int64)}

    stats = normalize_advantages_by_context(advantages, observations, "task_id")

    np.testing.assert_allclose(advantages, [[-1.0, 1.0], [-1.0, 1.0]])
    assert set(stats) == {0, 1}


class _TaskEnv(gym.Env):
    observation_space = gym.spaces.Dict(
        {
            "observation": gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
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
                "observation": np.full(4, self.step_index / 10, dtype=np.float32),
                "context/task_id": self.task_id,
            },
            float(action == self.task_id),
            self.step_index == 3,
            False,
            {},
        )


class _PlainEnv(gym.Env):
    observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(4, dtype=np.float32), float(action), True, False, {}


def test_grouped_ppo_trains_saves_and_loads_shared_task_context(tmp_path) -> None:
    env = DummyVecEnv([_TaskEnv, _TaskEnv])
    policy_model = _policy_model(hidden_sizes=[8], activation="tanh")
    try:
        model = GroupedAdvantagePPO(
            SharedActorCriticPolicy,
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
        assert loaded.policy.policy_model == normalize_policy_model(policy_model)
    finally:
        env.close()


def test_a2c_trains_saves_and_loads_shared_task_context(tmp_path) -> None:
    env = DummyVecEnv([_TaskEnv, _TaskEnv])
    policy_model = _policy_model(hidden_sizes=[8], activation="tanh")
    try:
        model = A2C(
            SharedActorCriticPolicy,
            env,
            n_steps=2,
            policy_kwargs={"policy_model": policy_model},
            verbose=0,
        )
        model.learn(total_timesteps=4)
        checkpoint = tmp_path / "a2c.zip"
        model.save(checkpoint)
        loaded = A2C.load(checkpoint, env=env)

        assert loaded.policy.policy_model == normalize_policy_model(policy_model)
    finally:
        env.close()
