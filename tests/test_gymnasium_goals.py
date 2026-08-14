from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import PPO

from gradlab.batch_runtime import ProviderDescriptor
from gradlab.env import make_vec_envs, resolve_env_config
from gradlab.env_config import env_config_from_mapping
from gradlab.env_registry import environment_spec, evaluation_watchdog_steps
from gradlab.gymnasium_vec_env import GYMNASIUM_ENV_CONTRACTS
from gradlab.play_catalog import PlayCatalog
from gradlab.recipe_documents import compose_train_document
from gradlab.recipe_schema import validate_materialized_train_recipe
from gradlab.task_kernels import IdentityTaskDefinition, Outcome


GOALS = Path("experiments/goals")
EXPECTED = {
    "CartPole-v1": {
        "threshold": 475.0,
        "timesteps": 100_000,
        "n_envs": 8,
        "n_steps": 32,
        "batch_size": 256,
        "n_epochs": 20,
        "gamma": 0.98,
        "gae_lambda": 0.8,
        "checkpoint_freq": 10_000,
        "plateau": 25_000,
    },
    "MountainCar-v0": {
        "threshold": -110.0,
        "timesteps": 1_000_000,
        "n_envs": 16,
        "n_steps": 16,
        "batch_size": 64,
        "n_epochs": 4,
        "gamma": 0.99,
        "gae_lambda": 0.98,
        "checkpoint_freq": 100_000,
        "plateau": 250_000,
    },
    "Acrobot-v1": {
        "threshold": -100.0,
        "timesteps": 1_000_000,
        "n_envs": 16,
        "n_steps": 256,
        "batch_size": 64,
        "n_epochs": 4,
        "gamma": 0.99,
        "gae_lambda": 0.94,
        "checkpoint_freq": 100_000,
        "plateau": 250_000,
    },
}


def _document(game: str) -> dict:
    root = GOALS / game
    return compose_train_document(root / "_goal.yaml", root / "recipes/ppo.yaml")


def test_classic_control_goals_are_registered_in_player_catalog() -> None:
    catalog = PlayCatalog(repo_root=Path.cwd())
    environment_names = {item["name"] for item in catalog.environments().items}

    assert set(EXPECTED) <= environment_names


@pytest.mark.parametrize("game", tuple(EXPECTED))
def test_classic_control_goal_and_recipe_materialize_exact_contract(game: str) -> None:
    expected = EXPECTED[game]
    contract = GYMNASIUM_ENV_CONTRACTS[game]
    document = _document(game)
    validate_materialized_train_recipe(document)
    config = document["train_config"]
    goal = document["goal"]
    backend = config["training_backend"]["config"]

    assert goal["evaluation_mode"] == "evaluated"
    assert goal["objective"]["rank"] == [
        "max(eval/full/episode/return/shaped/mean)",
        "min(leader/checkpoint/step)",
    ]
    assert goal["eval"]["episodes"] == 100
    assert goal["eval"]["policy"] == {"stochastic": True}
    assert goal["eval"]["acceptance"] == [
        {
            "metric": "eval/full/episode/return/shaped/mean",
            "operator": ">=",
            "threshold": expected["threshold"],
        }
    ]
    assert goal["release"] == {"huggingface": {}}
    assert config["env_provider"] == "gymnasium"
    assert config["game"] == game
    assert config["n_envs"] == expected["n_envs"]
    assert config["checkpoint_eval_n_envs"] == 10
    assert config["post_train_eval_episodes"] == 100
    assert config["checkpoint_eval_backend"] == "modal"
    assert config["checkpoint_freq"] == expected["checkpoint_freq"]
    assert config["timesteps"] == expected["timesteps"]
    assert config["task"]["termination"]["max_episode_steps"] == contract.max_episode_steps
    assert evaluation_watchdog_steps(config) == contract.max_episode_steps
    assert document["policy_environment_hash"] == document["evaluation_environment_hash"]
    assert config["policy_model"] == {
        "schema_version": 2,
        "encoder": {"kind": "flatten"},
        "fusion": {"hidden_sizes": [64, 64], "activation": "tanh"},
        "normalize_images": False,
        "orthogonal_init": True,
    }
    for name in ("n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda"):
        assert backend[name] == expected[name]
    assert backend["device"] == "cpu"
    assert backend["ent_coef"] == 0.0
    assert backend["clip_range"] == 0.2
    plateau = config["early_stop"]["conditions"]["return_plateau"]
    assert plateau == {
        "metric": "train/episode/return/shaped/origin/target/rolling/mean",
        "trigger": "no_improvement",
        "outcome": "neutral",
        "action": "stop",
        "patience_steps": expected["plateau"],
        "start_after_steps": expected["plateau"],
        "direction": "maximize",
        "min_delta": 1.0,
        "delta_mode": "absolute",
    }

    spec = environment_spec("gymnasium", game)
    assert spec.wandb_project == game
    assert spec.game_family.startswith("Gymnasium-")


def _kernel(game: str):
    task = _document(game)["train_config"]["task"]
    descriptor = ProviderDescriptor(
        provider_id="gymnasium",
        native_observation_space=gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
        native_action_space=gym.spaces.Discrete(3),
    )
    termination = task["termination"]
    return IdentityTaskDefinition(
        max_episode_steps=termination["max_episode_steps"],
        signals=task["signals"],
        events=task["events"],
        termination=termination,
    ).bind(descriptor, 1)


def test_cartpole_failure_wins_over_simultaneous_successful_horizon() -> None:
    kernel = _kernel("CartPole-v1")
    step = kernel.process(
        np.ones(1, dtype=np.float32),
        np.ones(1, dtype=np.bool_),
        np.ones(1, dtype=np.bool_),
        {},
    )

    assert step.outcomes.tolist() == [Outcome.FAILURE]
    assert step.terminated.tolist() == [True]
    assert step.truncated.tolist() == [False]


def test_cartpole_horizon_is_successful_and_bootstrappable() -> None:
    kernel = _kernel("CartPole-v1")
    step = kernel.process(
        np.ones(1, dtype=np.float32),
        np.zeros(1, dtype=np.bool_),
        np.ones(1, dtype=np.bool_),
        {},
    )

    assert step.outcomes.tolist() == [Outcome.SUCCESS]
    assert step.terminated.tolist() == [False]
    assert step.truncated.tolist() == [True]


@pytest.mark.parametrize("game", ("MountainCar-v0", "Acrobot-v1"))
def test_control_goal_success_wins_over_simultaneous_timeout(game: str) -> None:
    kernel = _kernel(game)
    step = kernel.process(
        -np.ones(1, dtype=np.float32),
        np.ones(1, dtype=np.bool_),
        np.ones(1, dtype=np.bool_),
        {},
    )

    assert step.outcomes.tolist() == [Outcome.SUCCESS]
    assert step.terminated.tolist() == [True]
    assert step.truncated.tolist() == [False]


@pytest.mark.parametrize("game", tuple(EXPECTED))
def test_classic_control_recipe_environment_runs_short_ppo_rollout(game: str) -> None:
    config = resolve_env_config(env_config_from_mapping(_document(game)["train_config"]))
    env = make_vec_envs(config, n_envs=2, seed=31)
    try:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=8,
            batch_size=16,
            n_epochs=1,
            learning_rate=3e-4,
            seed=31,
            device="cpu",
            verbose=0,
        )
        model.learn(total_timesteps=32)
        observations = env.reset()
        actions, _state = model.predict(observations, deterministic=False)
        next_observations, rewards, dones, infos = env.step(actions)

        assert next_observations.shape == (
            2,
            *GYMNASIUM_ENV_CONTRACTS[game].observation_shape,
        )
        assert rewards.shape == (2,)
        assert dones.shape == (2,)
        assert len(infos) == 2
    finally:
        env.close()
