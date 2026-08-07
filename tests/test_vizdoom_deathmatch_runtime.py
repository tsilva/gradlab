from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np

from gradlab.action_codecs import (
    VIZDOOM_DEATHMATCH_MULTIDISCRETE_BUTTONS,
    VIZDOOM_DEATHMATCH_MULTIDISCRETE_CODEC,
    VIZDOOM_DEATHMATCH_MULTIDISCRETE_NVEC,
)
from gradlab.batch_runtime import BatchMetricRecord, EpisodeRecord
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
    context = train_config["task"]["model_inputs"]["context"]
    assert context
    assert {field["history"] for field in context.values()} == {"provider_frame_stack"}
    assert train_config["env_args"]["frame_stack"] == 4


def test_gradoom_recipe_trains_on_gpu_and_keeps_reference_vizdoom_evaluation() -> None:
    document = compose_train_document(
        GOAL_ROOT / "_goal.yaml",
        GOAL_ROOT / "recipes/gradoom-ppo.yaml",
    )
    train_config = document["train_config"]

    assert train_config["env_provider"] == "gradoom"
    assert train_config["n_envs"] == 128
    assert train_config["env_args"]["compile_engine"] is True
    assert train_config["checkpoint_eval_environment"]["env_provider"] == "vizdoom-turbo"
    assert document["policy_environment_hash"] == document["evaluation_environment_hash"]
    backend = train_config["training_backend"]
    assert backend["id"] == "gradlab.ppo"
    assert backend["config"]["execution_profile"] == "max-throughput"
    assert backend["config"]["precision"] == "amp-fp16"


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
        assert observations["context/armor"].shape == (2, 4, 1)
        assert observations["context/health"].shape == (2, 4, 1)
        assert observations["context/selected_weapon"].shape == (2, 4)
        assert observations["context/selected_weapon_ammo"].shape == (2, 4, 1)
        assert observations["context/weapon_ammo"].shape == (2, 4, 6)
        assert observations["context/weapons_owned"].shape == (2, 4, 6)
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
            np.zeros((2, 4, 1), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            observations["context/selected_weapon"],
            np.ones((2, 4), dtype=np.int64),
        )
        np.testing.assert_allclose(
            observations["context/selected_weapon_ammo"],
            np.full((2, 4, 1), 1.0 / 6.0, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            observations["context/weapons_owned"],
            np.asarray(
                [[[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]] * 4] * 2,
                dtype=np.float32,
            ),
        )
        np.testing.assert_array_equal(
            observations["context/weapon_ammo"],
            np.asarray(
                [[[0.0, 0.25, 0.0, 0.25, 0.0, 0.0]] * 4] * 2,
                dtype=np.float32,
            ),
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
        assert next_observations["context/armor"].shape == (2, 4, 1)
        assert next_observations["context/health"].shape == (2, 4, 1)
        assert next_observations["context/selected_weapon"].shape == (2, 4)
        assert next_observations["context/selected_weapon_ammo"].shape == (2, 4, 1)
        assert next_observations["context/weapon_ammo"].shape == (2, 4, 6)
        assert next_observations["context/weapons_owned"].shape == (2, 4, 6)
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


def test_deathmatch_multidiscrete_codec_is_opt_in_and_runs_through_real_runtime() -> None:
    buttons_override = (
        "[" + ",".join(f'"{button}"' for button in VIZDOOM_DEATHMATCH_MULTIDISCRETE_BUTTONS) + "]"
    )
    document = compose_train_document(
        GOAL_ROOT / "_goal.yaml",
        GOAL_ROOT / "recipes/ppo.yaml",
        recipe_overrides=(
            "train.environment.env_config.env_args.use_restricted_actions=filtered",
            "train.environment.env_config.env_args.vizdoom_config.available_buttons="
            + buttons_override,
            "train.environment.task.action.set=sample-factory-v0",
            "train.environment.task.action.codec.type=" + VIZDOOM_DEATHMATCH_MULTIDISCRETE_CODEC,
        ),
    )
    train_config = document["train_config"]
    assert train_config["checkpoint_eval_environment"]["task"]["action"] == {
        "set": "sample-factory-v0",
        "codec": {"type": VIZDOOM_DEATHMATCH_MULTIDISCRETE_CODEC},
    }
    assert (
        train_config["checkpoint_eval_environment"]["env_args"]["use_restricted_actions"]
        == "filtered"
    )
    config = resolve_env_config(env_config_from_mapping(train_config))
    env = make_training_vec_env(config, n_envs=2, seed=1701)

    try:
        observations = env.reset()
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
        assert tuple(env.action_space.nvec) == VIZDOOM_DEATHMATCH_MULTIDISCRETE_NVEC

        actions = np.asarray(
            [
                [0, 0, 0, 0, 0, 10],
                [1, 2, 7, 1, 1, 20],
            ],
            dtype=np.int64,
        )
        next_observations, rewards, dones, infos = env.step(actions)
        assert set(next_observations) == set(observations)
        assert rewards.shape == (2,)
        assert dones.shape == (2,)
        assert len(infos) == 2

        policy = SharedActorCriticPolicy(
            env.observation_space,
            env.action_space,
            lambda _: 1e-3,
            policy_model=train_config["policy_model"],
        )
        observation_tensor, _ = policy.obs_to_tensor(next_observations)
        sampled_actions, values, log_prob = policy(observation_tensor)
        assert tuple(sampled_actions.shape) == (2, 6)
        assert tuple(values.shape) == (2, 1)
        assert tuple(log_prob.shape) == (2,)

        contract = env.runtime.action_contract
        assert contract["requested"]["mode"] == "filtered"
        assert contract["provider"]["space"]["type"] == "box"
        assert contract["policy"]["space"]["type"] == "multi_discrete"
        assert contract["policy"]["codec"]["type"] == (VIZDOOM_DEATHMATCH_MULTIDISCRETE_CODEC)
    finally:
        env.close()


def test_deathmatch_native_horizon_is_successful_and_keeps_bootstrap() -> None:
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
        assert records[0].outcome.name == "SUCCESS"
        assert records[0].truncated is True
        assert records[0].terminated is False
    finally:
        env.close()


def test_optional_sample_factory_reward_runs_through_real_vector_runtime() -> None:
    document = compose_train_document(
        GOAL_ROOT / "_goal.yaml",
        GOAL_ROOT / "recipes/ppo.yaml",
        recipe_overrides=("reward_shape=sample-factory-v0",),
    )
    config = resolve_env_config(env_config_from_mapping(document["train_config"]))
    env = make_training_vec_env(config, n_envs=1, seed=1701)

    try:
        env.reset()
        assert {"deathcount", "hitcount", "damagecount"} <= set(env.reset_infos[0])
        _observations, rewards, _dones, _infos = env.step(np.asarray([1], dtype=np.int64))
        metric_record = next(
            record for record in env.drain_records() if isinstance(record, BatchMetricRecord)
        )
        component_names = {
            "kill_reward_component",
            "death_penalty_component",
            "hit_reward_component",
            "damage_reward_component",
            "health_reward_component",
            "armor_reward_component",
            "weapon_reward_component",
            "ammo_reward_component",
            "weapon_hold_reward_component",
        }

        assert np.all(np.isfinite(rewards))
        assert component_names <= set(metric_record.metrics)
        component_sum = sum(
            np.asarray(metric_record.metrics[name], dtype=np.float32) for name in component_names
        )
        np.testing.assert_allclose(component_sum, metric_record.metrics["raw_reward"])
        np.testing.assert_allclose(rewards, metric_record.metrics["shaped_reward"])
        assert "native_reward_component" not in metric_record.metrics
    finally:
        env.close()
