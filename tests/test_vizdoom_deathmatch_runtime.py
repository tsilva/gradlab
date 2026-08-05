from dataclasses import replace
from pathlib import Path

import numpy as np

from gradlab.batch_runtime import EpisodeRecord
from gradlab.env import make_training_vec_env, resolve_env_config
from gradlab.env_config import env_config_from_mapping
from gradlab.recipe_documents import compose_train_document
from gradlab.actor_critic_policy import SharedActorCriticPolicy


GOAL_ROOT = Path("experiments/goals/VizdoomDeathmatch-v1")


def test_deathmatch_recipe_defaults_to_the_sample_efficient_custom_ppo_profile() -> None:
    document = compose_train_document(
        GOAL_ROOT / "_goal.yaml",
        GOAL_ROOT / "recipes/ppo.yaml",
    )
    train_config = document["train_config"]

    assert train_config["timesteps"] == 500_000_000
    assert train_config["frame_skip"] == 2
    assert train_config["obs_crop"] == [0, 32, 0, 0]
    assert train_config["obs_crop_mode"] == "mask"
    assert train_config["obs_crop_fill"] == 0
    backend = train_config["training_backend"]
    assert backend["id"] == "gradlab.ppo"
    assert backend["config"]["execution_profile"] == "sb3-parity"
    assert backend["config"]["gamma"] == 0.995
    assert backend["config"]["gae_lambda"] == 0.95


def test_deathmatch_recipe_runs_through_the_real_single_player_vector_runtime() -> None:
    document = compose_train_document(
        GOAL_ROOT / "_goal.yaml",
        GOAL_ROOT / "recipes/ppo.yaml",
    )
    config = resolve_env_config(env_config_from_mapping(document["train_config"]))
    env = make_training_vec_env(config, n_envs=2, seed=1701)

    try:
        observations = env.reset()
        assert set(observations) == {
            "observation",
            "context/armor",
            "context/health",
            "context/selected_weapon",
            "context/selected_weapon_ammo",
            "context/weapon_ammo",
            "context/weapons_owned",
        }
        assert observations["observation"].shape == (2, 4, 84, 84)
        assert observations["context/armor"].shape == (2, 1)
        assert observations["context/health"].shape == (2, 1)
        assert observations["context/selected_weapon"].shape == (2,)
        assert observations["context/selected_weapon_ammo"].shape == (2, 1)
        assert observations["context/weapon_ammo"].shape == (2, 6)
        assert observations["context/weapons_owned"].shape == (2, 6)
        assert observations["context/armor"].dtype == np.float32
        assert observations["context/health"].dtype == np.float32
        assert observations["context/selected_weapon"].dtype == np.int64
        assert observations["context/selected_weapon_ammo"].dtype == np.float32
        assert observations["context/weapon_ammo"].dtype == np.float32
        assert observations["context/weapons_owned"].dtype == np.float32
        assert np.all(
            (0.0 <= observations["context/armor"]) & (observations["context/armor"] <= 1.0)
        )
        assert np.all(
            (0.0 <= observations["context/health"]) & (observations["context/health"] <= 2.0)
        )
        assert np.all(
            (0.0 <= observations["context/weapon_ammo"])
            & (observations["context/weapon_ammo"] <= 1.0)
        )
        assert np.all(
            (0.0 <= observations["context/weapons_owned"])
            & (observations["context/weapons_owned"] <= 1.0)
        )
        np.testing.assert_array_equal(
            observations["context/armor"],
            np.zeros((2, 1), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            observations["context/selected_weapon"],
            np.ones(2, dtype=np.int64),
        )
        np.testing.assert_allclose(
            observations["context/selected_weapon_ammo"],
            np.full((2, 1), 1.0 / 6.0, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            observations["context/weapons_owned"],
            np.asarray([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]] * 2, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            observations["context/weapon_ammo"],
            np.asarray([[0.0, 0.25, 0.0, 0.25, 0.0, 0.0]] * 2, dtype=np.float32),
        )
        assert env.action_space.n == 17
        assert {
            "killcount",
            "health",
            "armor",
            "selected_weapon",
            "selected_weapon_ammo",
            "weapon1",
            "weapon2",
            "weapon3",
            "weapon4",
            "weapon5",
            "weapon6",
            "ammo1",
            "ammo2",
            "ammo3",
            "ammo4",
            "ammo5",
            "ammo6",
            "player_dead",
        } <= set(env.reset_infos[0])

        next_observations, rewards, dones, infos = env.step(np.asarray([0, 9], dtype=np.int64))
        assert set(next_observations) == {
            "observation",
            "context/armor",
            "context/health",
            "context/selected_weapon",
            "context/selected_weapon_ammo",
            "context/weapon_ammo",
            "context/weapons_owned",
        }
        assert next_observations["context/armor"].shape == (2, 1)
        assert next_observations["context/health"].shape == (2, 1)
        assert next_observations["context/selected_weapon"].shape == (2,)
        assert next_observations["context/selected_weapon_ammo"].shape == (2, 1)
        assert next_observations["context/weapon_ammo"].shape == (2, 6)
        assert next_observations["context/weapons_owned"].shape == (2, 6)
        assert np.all(
            (0.0 <= next_observations["context/armor"])
            & (next_observations["context/armor"] <= 1.0)
        )
        assert np.all(
            (0.0 <= next_observations["context/weapon_ammo"])
            & (next_observations["context/weapon_ammo"] <= 1.0)
        )
        assert np.all(
            (0.0 <= next_observations["context/weapons_owned"])
            & (next_observations["context/weapons_owned"] <= 1.0)
        )
        assert rewards.shape == (2,)
        assert dones.shape == (2,)
        assert len(infos) == 2
        assert all(isinstance(info, dict) for info in infos)

        policy = SharedActorCriticPolicy(
            env.observation_space,
            env.action_space,
            lambda _: 1e-3,
            policy_model=document["train_config"]["policy_model"],
        )
        observation_tensor, _ = policy.obs_to_tensor(next_observations)
        actions, values, log_prob = policy(observation_tensor)
        assert tuple(actions.shape) == (2,)
        assert tuple(values.shape) == (2, 1)
        assert tuple(log_prob.shape) == (2,)

        contract = env.runtime.action_contract
        assert contract["provider"]["mode"] == "custom_discrete"
        assert contract["provider"]["space"] == {
            "type": "discrete",
            "n": 17,
            "start": 0,
            "dtype": "int64",
        }
        assert contract["requested"]["meanings"][8:10] == [
            "speed_move_forward",
            "attack_move_forward",
        ]
    finally:
        env.close()


def test_deathmatch_native_horizon_emits_the_shared_timeout_event() -> None:
    document = compose_train_document(
        GOAL_ROOT / "_goal.yaml",
        GOAL_ROOT / "recipes/ppo.yaml",
    )
    config = resolve_env_config(env_config_from_mapping(document["train_config"]))
    env_args = {
        **config.env_args,
        "vizdoom_config": {
            **config.env_args["vizdoom_config"],
            "episode_timeout": 4,
        },
    }
    config = replace(config, env_args=env_args)
    env = make_training_vec_env(config, n_envs=1, seed=1701)

    try:
        env.reset()
        records = []
        for _ in range(2):
            _observations, _rewards, _dones, _infos = env.step(np.asarray([0], dtype=np.int64))
            records.extend(
                record for record in env.drain_records() if isinstance(record, EpisodeRecord)
            )

        assert len(records) == 1
        assert records[0].episode_length == 2
        assert records[0].events == ("time_limit_reached",)
        assert records[0].outcome.name == "TIMEOUT"
        assert records[0].truncated is True
        assert records[0].terminated is False
    finally:
        env.close()
