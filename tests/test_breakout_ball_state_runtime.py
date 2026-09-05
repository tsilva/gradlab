from pathlib import Path

import numpy as np

from gradlab.actor_critic_policy import SharedActorCriticPolicy
from gradlab.env import make_training_vec_env, resolve_env_config
from gradlab.env_config import env_config_from_mapping
from gradlab.recipe_documents import compose_train_document


BREAKOUT_ROOT = Path("experiments/goals/Breakout-Atari2600-v0")


def test_plain_ppo_recipe_exposes_brick_progress_through_the_real_vector_runtime() -> None:
    document = compose_train_document(
        BREAKOUT_ROOT / "_goal.yaml",
        BREAKOUT_ROOT / "recipes/ppo.yaml",
    )
    config = resolve_env_config(env_config_from_mapping(document["train_config"]))
    env = make_training_vec_env(config, n_envs=2, seed=1701)

    try:
        assert "bricks_destroyed" in env.signal_schema
        assert "bricks_destroyed_normalized" in env.signal_schema
        env.reset()
        _observations, rewards, dones, infos = env.step(np.zeros(2, dtype=np.int64))
        assert rewards.shape == (2,)
        assert dones.shape == (2,)
        assert len(infos) == 2
    finally:
        env.close()


def test_ball_state_recipe_runs_through_the_real_vector_runtime() -> None:
    document = compose_train_document(
        BREAKOUT_ROOT / "_goal.yaml",
        BREAKOUT_ROOT / "recipes/ppo-ball-state.yaml",
    )
    config = resolve_env_config(env_config_from_mapping(document["train_config"]))
    env = make_training_vec_env(config, n_envs=2, seed=1701)

    try:
        observations = env.reset()
        expected_keys = {
            "observation",
            "context/ball_x",
            "context/ball_y",
            "context/ball_vx",
            "context/ball_vy",
            "context/paddle_x",
        }
        assert set(observations) == expected_keys
        assert observations["observation"].shape == (2, 4, 84, 84)
        for key in expected_keys - {"observation"}:
            assert np.isneginf(env.observation_space[key].low).all()
            assert np.isposinf(env.observation_space[key].high).all()
        for key in expected_keys - {"observation"}:
            assert observations[key].shape == (2, 1)
            assert observations[key].dtype == np.float32
            assert np.all((-1.0 <= observations[key]) & (observations[key] <= 1.0))

        # A waiting-to-serve ball is an intentional boundary sentinel, not
        # missing data. Centering the provider ratio maps it exactly to -1.
        np.testing.assert_array_equal(
            observations["context/ball_y"],
            np.full((2, 1), -1.0, dtype=np.float32),
        )
        np.testing.assert_allclose(observations["context/ball_x"], 0.0)
        np.testing.assert_allclose(observations["context/ball_vx"], 0.5)
        np.testing.assert_allclose(observations["context/ball_vy"], 8 / 27)
        np.testing.assert_array_equal(
            observations["context/paddle_x"][0],
            observations["context/paddle_x"][1],
        )

        policy = SharedActorCriticPolicy(
            env.observation_space,
            env.action_space,
            lambda _: 1e-3,
            policy_model=document["train_config"]["policy_model"],
        )
        fusion = policy.features_extractor.fusion[0]
        assert fusion.in_features == 517
        assert fusion.out_features == 256
        assert policy.action_net.in_features == 256
        assert policy.value_net.in_features == 256
        actions, _state = policy.predict(observations, deterministic=True)
        assert actions.shape == (2,)

        next_observations, rewards, dones, infos = env.step(np.zeros(2, dtype=np.int64))
        assert set(next_observations) == expected_keys
        assert np.all(next_observations["context/ball_y"] > -1.0)
        for key in expected_keys - {"observation"}:
            assert np.all((-1.0 <= next_observations[key]) & (next_observations[key] <= 1.0))
        assert rewards.shape == (2,)
        assert dones.shape == (2,)
        assert len(infos) == 2
    finally:
        env.close()
