from __future__ import annotations

import math
import pickle

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from gradlab.action_codecs import (
    LegalTupleMultiDiscrete,
    VIZDOOM_SHARED_MULTIDISCRETE_NVEC,
)
from gradlab.action_distributions import LegalTupleCategoricalDistribution
from gradlab.actor_critic_policy import SharedActorCriticPolicy
from gradlab.callbacks import policy_discrete_action_indices, policy_entropy_bounds
from gradlab.play_debug import actor_critic_policy_decisions


LEGAL = (
    (0, 0, 0, 0, 0, 0),
    (1, 0, 0, 0, 0, 0),
    (0, 2, 0, 0, 0, 0),
    (1, 0, 0, 1, 0, 0),
)


def _space() -> LegalTupleMultiDiscrete:
    return LegalTupleMultiDiscrete(VIZDOOM_SHARED_MULTIDISCRETE_NVEC, LEGAL, seed=7)


def test_legal_tuple_space_samples_contains_compares_and_pickles_exact_support() -> None:
    space = _space()
    assert all(space.contains(space.sample()) for _ in range(100))
    assert not space.contains(np.asarray([0, 0, 3, 0, 0, 0]))
    restored = pickle.loads(pickle.dumps(space))
    assert restored == space
    assert restored.legal_tuples == LEGAL
    assert policy_entropy_bounds(space) == (0.0, math.log(len(LEGAL)))
    actions = np.asarray([LEGAL[3], LEGAL[1], LEGAL[3], LEGAL[0]])
    np.testing.assert_array_equal(
        policy_discrete_action_indices(actions, space),
        np.asarray([3, 1, 3, 0]),
    )


def test_legal_tuple_distribution_matches_reference_categorical_math() -> None:
    distribution = LegalTupleCategoricalDistribution(_space())
    logits = torch.arange(86, dtype=torch.float32).reshape(2, 43) / 20.0
    logits.requires_grad_()
    resolved = distribution.proba_distribution(logits)

    offsets = torch.tensor([0, 3, 6, 16, 18, 20])
    indices = torch.tensor(LEGAL) + offsets
    reference_logits = logits[:, indices].sum(dim=-1)
    reference = torch.distributions.Categorical(logits=reference_logits)
    torch.testing.assert_close(resolved.distribution.probs, reference.probs)
    torch.testing.assert_close(resolved.entropy(), reference.entropy())
    expected_mode = torch.tensor([LEGAL[index] for index in reference.probs.argmax(dim=1)])
    assert torch.equal(resolved.mode(), expected_mode)

    actions = torch.tensor([LEGAL[0], LEGAL[3]])
    expected_rows = torch.tensor([0, 3])
    torch.testing.assert_close(resolved.log_prob(actions), reference.log_prob(expected_rows))
    loss = -resolved.log_prob(actions).mean()
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) > 0

    with pytest.raises(ValueError, match="outside"):
        resolved.log_prob(torch.tensor([[0, 0, 3, 0, 0, 0]]))


def test_shared_actor_critic_policy_uses_legal_tuple_distribution_automatically() -> None:
    action_space = _space()
    policy = SharedActorCriticPolicy(
        gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
        action_space,
        lambda _progress: 1e-3,
        policy_model={
            "schema_version": 2,
            "encoder": {"kind": "flatten"},
            "fusion": {"hidden_sizes": [], "activation": "tanh"},
            "normalize_images": False,
            "orthogonal_init": False,
        },
    )
    distribution = policy.get_distribution(torch.zeros((8, 4)))
    assert isinstance(distribution, LegalTupleCategoricalDistribution)
    actions = distribution.get_actions()
    assert actions.shape == (8, 6)
    assert all(action_space.contains(row.cpu().numpy()) for row in actions)

    decision = actor_critic_policy_decisions(
        type("Model", (), {"policy": policy})(),
        np.zeros((1, 4), dtype=np.float32),
        deterministic=False,
    )[0]
    assert decision.distribution_kind == "legal_tuple_categorical"
    assert decision.selected_discrete_action is not None
    assert LEGAL[decision.selected_discrete_action] == tuple(decision.raw_action)
    assert decision.selected_probability is not None
    assert decision.selected_rank is not None


def test_standalone_sb3_checkpoint_preserves_legal_distribution(tmp_path) -> None:
    class TinyEnv(gym.Env):
        observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        action_space = _space()

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(4, dtype=np.float32), {}

        def step(self, action):
            assert self.action_space.contains(action)
            return np.zeros(4, dtype=np.float32), 0.0, False, True, {}

    model = PPO(
        SharedActorCriticPolicy,
        TinyEnv(),
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        policy_kwargs={
            "policy_model": {
                "schema_version": 2,
                "encoder": {"kind": "flatten"},
                "fusion": {"hidden_sizes": [], "activation": "tanh"},
                "normalize_images": False,
                "orthogonal_init": False,
            }
        },
        verbose=0,
    )
    model.learn(total_timesteps=8)
    path = tmp_path / "legal-policy.zip"
    model.save(path)

    loaded = PPO.load(path)
    assert isinstance(loaded.action_space, LegalTupleMultiDiscrete)
    action, _state = loaded.predict(np.zeros(4, dtype=np.float32), deterministic=False)
    assert loaded.action_space.contains(action)
    assert isinstance(
        loaded.policy.get_distribution(torch.zeros((1, 4))),
        LegalTupleCategoricalDistribution,
    )
