from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gradlab.action_codecs import (
    LEGAL_TUPLE_DISTRIBUTION,
    LegalTupleMultiDiscrete,
    VIZDOOM_SHARED_MULTIDISCRETE_CODEC,
    VIZDOOM_SHARED_MULTIDISCRETE_NVEC,
)
from gradlab.action_contract import action_contract_meanings
from gradlab.action_distributions import LegalTupleCategoricalDistribution
from gradlab.action_profiles import VIZDOOM_SHARED_ACTION_PROFILE
from gradlab.actor_critic_policy import SharedActorCriticPolicy
from gradlab.env import make_training_vec_env, resolve_env_config
from gradlab.env_config import env_config_from_mapping
from gradlab.recipe_documents import compose_train_document


GOALS_ROOT = Path("experiments/goals")
GOAL_FILES = tuple(sorted(GOALS_ROOT.glob("Vizdoom*/_goal.yaml")))


def _recipe(goal: Path) -> Path:
    recipes = sorted((goal.parent / "recipes").glob("*.yaml"))
    return next((path for path in recipes if path.name == "ppo.yaml"), recipes[0])


def _profiled_document(goal: Path) -> dict:
    return compose_train_document(
        goal,
        _recipe(goal),
        recipe_overrides=(f"action_profile={VIZDOOM_SHARED_ACTION_PROFILE}",),
    )


@pytest.mark.parametrize("goal", GOAL_FILES, ids=lambda path: path.parent.name)
def test_every_vizdoom_goal_runs_one_real_step_with_the_shared_legal_head(goal: Path) -> None:
    document = _profiled_document(goal)
    config = resolve_env_config(env_config_from_mapping(document["train_config"]))
    env = make_training_vec_env(config, n_envs=1, seed=1701)

    try:
        env.reset()
        space = env.action_space
        assert isinstance(space, LegalTupleMultiDiscrete)
        assert tuple(int(value) for value in space.nvec) == VIZDOOM_SHARED_MULTIDISCRETE_NVEC
        assert space.legal_tuples[0] == (0, 0, 0, 0, 0, 0)

        _observations, rewards, dones, infos = env.step(
            np.asarray([space.legal_tuples[0]], dtype=np.int64)
        )
        assert rewards.shape == (1,)
        assert dones.shape == (1,)
        assert len(infos) == 1

        contract = env.runtime.action_contract
        assert contract["provider"]["space"]["type"] == "box"
        assert contract["policy"]["space"]["nvec"] == list(VIZDOOM_SHARED_MULTIDISCRETE_NVEC)
        assert contract["policy"]["space"]["legal_tuples"] == [
            list(row) for row in space.legal_tuples
        ]
        assert contract["policy"]["space"]["distribution"] == {
            "type": LEGAL_TUPLE_DISTRIBUTION,
            "scoring": "sum_selected_axis_logits",
        }
        assert contract["policy"]["codec"]["type"] == VIZDOOM_SHARED_MULTIDISCRETE_CODEC
        assert len(action_contract_meanings(contract)) == space.legal_tuple_count
    finally:
        env.close()


def test_shared_deathmatch_policy_samples_only_exact_legal_tuples() -> None:
    goal = GOALS_ROOT / "VizdoomDeathmatch-v1" / "_goal.yaml"
    document = _profiled_document(goal)
    config = resolve_env_config(env_config_from_mapping(document["train_config"]))
    env = make_training_vec_env(config, n_envs=2, seed=1701)

    try:
        observations = env.reset()
        policy = SharedActorCriticPolicy(
            env.observation_space,
            env.action_space,
            lambda _progress: 1e-3,
            policy_model=document["train_config"]["policy_model"],
        )
        observation_tensor, _ = policy.obs_to_tensor(observations)
        distribution = policy.get_distribution(observation_tensor)
        assert isinstance(distribution, LegalTupleCategoricalDistribution)

        actions, values, log_prob = policy(observation_tensor)
        assert tuple(actions.shape) == (2, 6)
        assert tuple(values.shape) == (2, 1)
        assert tuple(log_prob.shape) == (2,)
        assert all(env.action_space.contains(row.detach().cpu().numpy()) for row in actions)

        _next_observations, rewards, dones, infos = env.step(actions.detach().cpu().numpy())
        assert rewards.shape == (2,)
        assert dones.shape == (2,)
        assert len(infos) == 2
    finally:
        env.close()
