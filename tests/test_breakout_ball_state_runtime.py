from pathlib import Path

import numpy as np

from gradlab.actor_critic_policy import SharedActorCriticPolicy
from gradlab.env import make_eval_vec_env, make_training_vec_env, resolve_env_config
from gradlab.env_config import env_config_from_mapping
from gradlab.recipe_documents import compose_train_document


BREAKOUT_ROOT = Path("experiments/goals/Breakout-Atari2600-v0")


def test_brick_reward_recipe_matches_provider_deltas_without_native_score() -> None:
    document = compose_train_document(
        BREAKOUT_ROOT / "_goal.yaml", BREAKOUT_ROOT / "recipes/ppo-ball-state-brick-reward.yaml"
    )
    train = document["train_config"]
    assert train["n_envs"] == 64
    assert train["timesteps"] == 100000000
    assert len(train["task"]["model_inputs"]["context"]) == 7
    baseline = compose_train_document(
        BREAKOUT_ROOT / "_goal.yaml", BREAKOUT_ROOT / "recipes/ppo-ball-state.yaml"
    )["train_config"]
    assert train["task"]["model_inputs"] == baseline["task"]["model_inputs"]
    assert train["policy_model"] == baseline["policy_model"]
    assert train["training_backend"]["config"]["gamma"] == 0.99
    assert train["training_backend"]["config"]["gae_lambda"] == 0.95
    assert train["frame_skip"] == 2
    assert train["task"]["reward"]["event_rewards"] == {"life_loss": -0.1, "serve_stall": -5.0}
    config = resolve_env_config(env_config_from_mapping(train))
    env = make_eval_vec_env(config, n_envs=2, seed=10000)
    try:
        env.reset()
        returns = np.zeros(2, dtype=np.float64)
        completed = 0
        scored = False
        rng = np.random.default_rng(10000)
        for _ in range(6000):
            _, rewards, dones, infos = env.step(rng.integers(0, 3, size=2, dtype=np.int64))
            returns += rewards
            for lane, info in enumerate(infos):
                if dones[lane]:
                    expected = int(info["bricks_destroyed"]) - 0.1 * (5 - int(info["lives"]))
                    np.testing.assert_allclose(returns[lane], expected, atol=1e-5)
                    scored |= int(info["bricks_destroyed"]) > 0
                    completed += 1
                    returns[lane] = 0
            if completed >= 4 and scored:
                break
        assert completed >= 4
        assert scored
    finally:
        env.close()


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
            "context/paddle_width",
            "context/ball_paddle_offset",
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
        np.testing.assert_allclose(observations["context/paddle_width"], 1.0)

        policy = SharedActorCriticPolicy(
            env.observation_space,
            env.action_space,
            lambda _: 1e-3,
            policy_model=document["train_config"]["policy_model"],
        )
        fusion = policy.features_extractor.fusion[0]
        assert fusion.in_features == 519
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
