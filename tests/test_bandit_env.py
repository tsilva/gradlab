from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import A2C, PPO

from gradlab.bandit_env import BanditVectorEnv
from gradlab.env import EnvConfig, make_vec_envs
from gradlab.env_registry import resolve_env_id, resolve_env_provider
from gradlab.metric_store import MetricStore
from gradlab.policy_bundle import (
    build_recipe_document,
    load_policy_bundle_from_checkpoint,
    write_canonical_json,
)
from gradlab.recipe_documents import (
    compose_resolved_train_documents,
    compose_train_document,
)
from gradlab.recipe_schema import validate_materialized_train_recipe
from gradlab.sb3_models import load_sb3_model
from gradlab.train import main as train_main


BANDIT_GOAL = Path("experiments/goals/gradlab__bandit/_goal.yaml")
BANDIT_RECIPE = Path("experiments/goals/gradlab__bandit/recipes/ppo.yaml")


def _bandit_recipe_document():
    return compose_train_document(BANDIT_GOAL, BANDIT_RECIPE)


def _write_versioned_recipe(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "recipe.json"
    resolved = compose_resolved_train_documents(
        BANDIT_GOAL,
        BANDIT_RECIPE,
        source_sha="a" * 40,
    )
    return write_canonical_json(
        path,
        build_recipe_document(
            document,
            repo_root=Path.cwd(),
            source_commit="a" * 40,
            run_description="ROM-free backend boundary smoke.",
            seed=7,
            runtime_image_ref="docker:ghcr.io/tsilva/gradlab-runtime@sha256:" + "b" * 64,
            base_materialized_recipe=resolved.base,
            canonical_goal=resolved.canonical_goal,
        ),
    )


def _native_env(num_envs: int = 3) -> BanditVectorEnv:
    return BanditVectorEnv(
        "Bandit-v0",
        num_envs,
        autoreset_mode=gym.vector.AutoresetMode.DISABLED,
    )


def _config() -> EnvConfig:
    return EnvConfig(
        env_provider="gradlab",
        game="Bandit-v0",
        env_args={"autoreset_mode": "disabled"},
        task={
            "id": "identity",
            "action": {"set": "native"},
            "signals": {},
            "events": {},
            "termination": {
                "success": [],
                "failure": [],
                "timeout": [],
                "max_episode_steps": 1,
            },
            "reward": {"reward_mode": "native"},
        },
        state="",
        frame_skip=1,
        max_pool_frames=False,
        obs_resize=(0, 0),
        obs_crop=(0, 0, 0, 0),
    )


def test_bandit_spaces_rewards_and_manual_reset() -> None:
    env = _native_env()
    assert env.single_observation_space == gym.spaces.Box(0.0, 0.0, shape=(1,), dtype=np.float32)
    assert env.single_action_space == gym.spaces.Discrete(2)
    assert env.metadata["autoreset_mode"] is gym.vector.AutoresetMode.DISABLED

    with pytest.raises(RuntimeError, match="require reset"):
        env.step(np.asarray([0, 1, 1], dtype=np.int64))

    observations, reset_infos = env.reset(seed=[1, 2, 3])
    assert observations.shape == (3, 1)
    assert observations.dtype == np.float32
    assert reset_infos == {}

    observations, rewards, terminated, truncated, infos = env.step(
        np.asarray([0, 1, 1], dtype=np.int64)
    )
    np.testing.assert_array_equal(observations, np.zeros((3, 1), dtype=np.float32))
    np.testing.assert_array_equal(rewards, [0.0, 1.0, 1.0])
    np.testing.assert_array_equal(terminated, [True, True, True])
    np.testing.assert_array_equal(truncated, [False, False, False])
    np.testing.assert_array_equal(infos["chosen_arm"], [0, 1, 1])
    np.testing.assert_array_equal(infos["optimal_arm"], [1, 1, 1])
    np.testing.assert_array_equal(infos["is_optimal"], [False, True, True])

    with pytest.raises(RuntimeError, match="require reset"):
        env.step(np.asarray([0, 1, 1], dtype=np.int64))


def test_bandit_masked_reset_preserves_unselected_lane_lifecycle() -> None:
    env = _native_env()
    env.reset()
    env.step(np.asarray([0, 1, 0], dtype=np.int64))

    selected = np.asarray([True, False, True], dtype=np.bool_)
    env.reset(seed=[4, None, 6], options={"reset_mask": selected})
    with pytest.raises(RuntimeError, match=r"\[1\]"):
        env.step(np.asarray([1, 1, 1], dtype=np.int64))

    env.reset(seed=[None, 5, None], options={"reset_mask": ~selected})
    _observations, rewards, terminated, truncated, _infos = env.step(
        np.asarray([1, 1, 1], dtype=np.int64)
    )
    np.testing.assert_array_equal(rewards, np.ones(3, dtype=np.float32))
    assert terminated.all()
    assert not truncated.any()


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"reset_mask": [True, False, True]}, TypeError),
        ({"reset_mask": np.ones(2, dtype=np.bool_)}, ValueError),
        ({"reset_mask": np.ones(3, dtype=np.int8)}, TypeError),
        ({"reset_mask": np.zeros(3, dtype=np.bool_)}, ValueError),
        ({"unknown": True}, ValueError),
    ],
)
def test_bandit_rejects_invalid_reset_options(options, error) -> None:
    env = _native_env()
    with pytest.raises(error):
        env.reset(options=options)


@pytest.mark.parametrize(
    ("actions", "error"),
    [
        (np.asarray([[0], [1], [0]], dtype=np.int64), ValueError),
        (np.asarray([0.0, 1.0, 0.0], dtype=np.float32), TypeError),
        (np.asarray([0, 2, 0], dtype=np.int64), ValueError),
    ],
)
def test_bandit_rejects_invalid_actions(actions, error) -> None:
    env = _native_env()
    env.reset()
    with pytest.raises(error):
        env.step(actions)


def test_gradlab_provider_is_fixed_and_rejects_unknown_environment() -> None:
    provider = resolve_env_provider("gradlab")
    assert provider.env_ids == ("Bandit-v0",)
    assert provider.supports_states is False
    assert provider.constructor_contract is not None
    assert provider.constructor_contract.required_values == {"autoreset_mode": "disabled"}
    assert resolve_env_id("gradlab:Bandit-v0").provider_env_id == "Bandit-v0"
    assert "Bandit-v0" not in gym.registry
    with pytest.raises(ValueError, match="does not register environment"):
        resolve_env_id("gradlab:Unknown-v0")

    with pytest.raises(ValueError, match="does not support state"):
        make_vec_envs(replace(_config(), state="Start"), n_envs=1, seed=1)


def test_gradlab_facade_same_step_resets_bandit_lanes() -> None:
    env = make_vec_envs(_config(), n_envs=3, seed=7)
    try:
        observations = env.reset()
        np.testing.assert_array_equal(observations, np.zeros((3, 1), dtype=np.float32))

        next_observations, rewards, dones, infos = env.step(np.asarray([0, 1, 1], dtype=np.int64))
        np.testing.assert_array_equal(next_observations, np.zeros((3, 1), dtype=np.float32))
        np.testing.assert_array_equal(rewards, [0.0, 1.0, 1.0])
        assert dones.all()
        assert all("terminal_observation" in info for info in infos)

        _next_observations, rewards, dones, _infos = env.step(np.asarray([1, 1, 1], dtype=np.int64))
        np.testing.assert_array_equal(rewards, np.ones(3, dtype=np.float32))
        assert dones.all()
    finally:
        env.close()


def test_gradlab_facade_applies_common_reward_scale_then_clip() -> None:
    config = _config()
    reward = {
        **config.task["reward"],
        "reward_scale": 2.0,
        "reward_clip": [0.0, 0.4],
    }
    env = make_vec_envs(
        replace(config, task={**config.task, "reward": reward}),
        n_envs=2,
        seed=7,
    )
    try:
        env.reset()
        _observations, rewards, _dones, _infos = env.step(np.asarray([0, 1], dtype=np.int64))
        np.testing.assert_allclose(rewards, [0.0, 0.4])
    finally:
        env.close()


def test_bandit_recipe_materializes_fixed_train_and_eval_contracts() -> None:
    document = _bandit_recipe_document()
    validate_materialized_train_recipe(document)

    train_config = document["train_config"]
    assert train_config["env_provider"] == "gradlab"
    assert train_config["game"] == "Bandit-v0"
    assert train_config["n_envs"] == 8
    assert train_config["timesteps"] == 256
    assert train_config["checkpoint_eval_backend"] == "modal"
    assert train_config["post_train_eval_episodes"] == 256
    assert train_config["checkpoint_eval_n_envs"] == 32
    assert "stop_on_acceptance" not in train_config
    assert train_config["checkpoint_eval_acceptance"] == [
        {
            "metric": "eval/full/episode/return/shaped/mean",
            "operator": ">=",
            "threshold": 0.9,
        }
    ]
    assert train_config["env_args"] == {"autoreset_mode": "disabled"}
    assert train_config["training_backend"]["id"] == "sb3.ppo"
    assert train_config["policy_model"]["topology"]["encoder"] == {"kind": "flatten"}
    assert train_config["policy_model"]["heads"] == {
        "action": {"hidden_sizes": [64, 64], "activation": "tanh"},
        "state_value": {"hidden_sizes": [64, 64], "activation": "tanh"},
    }


def test_bandit_local_demo_runs_to_cap_without_a_declared_success_signal(
    tmp_path: Path,
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    document = _bandit_recipe_document()
    recipe_path = tmp_path / "recipe.json"
    config = dict(document["train_config"])
    config.update(
        {
            "run_name": "backend-smoke",
            "run_description": "ROM-free backend boundary smoke.",
            "runs_dir": str(tmp_path),
            "timesteps": 64,
            "checkpoint_freq": 0,
            "checkpoint_eval_backend": "none",
            "early_stop": None,
            "wandb_mode": "disabled",
            "recipe_json_path": str(recipe_path),
        }
    )
    backend = dict(config["training_backend"])
    backend_config = dict(backend["config"])
    backend_config.update({"device": "cpu", "n_epochs": 1})
    backend["config"] = backend_config
    config["training_backend"] = backend
    document["train_config"] = config
    recipe_path = _write_versioned_recipe(tmp_path, document)
    path = tmp_path / "train.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    monkeypatch.setenv("GRADLAB_INTERNAL_LEARNER", "1")
    assert (
        train_main(
            [
                "--train-config-json",
                str(path),
                "--execution-mode",
                "local-demo",
            ]
        )
        == 0
    )

    run_dir = tmp_path / "backend-smoke"
    assert (run_dir / "learner-ready.json").is_file()
    assert (run_dir / "final_model.zip").is_file()
    bundle = load_policy_bundle_from_checkpoint(run_dir / "final_model.zip")
    assert bundle is not None
    assert bundle.model["policy"]["training_backend_id"] == "sb3.ppo"
    assert len(bundle.model["policy"]["training_backend_config_hash"]) == 64
    assert bundle.model["provenance"]["training_execution"]["mode"] == "local-demo"
    assert bundle.model["provenance"]["training_terminal"] == {
        "terminal_reason": "resource_exhaustion",
        "first_completion_step": None,
        "final_step": 64,
        "requested_limit": 64,
        "execution_limit": 64,
    }
    result = json.loads((run_dir / "training-result.json").read_text(encoding="utf-8"))
    assert result["execution_mode"] == "local-demo"
    assert result["requested_limit"] == 64
    assert result["execution_limit"] == 64
    assert not (run_dir / "checkpoints").exists()
    output = capsys.readouterr().out
    assert "no declared success signal" in output
    assert "| rollout/" not in output


def test_bandit_runs_through_a2c_backend_and_round_trips_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    document = _bandit_recipe_document()
    recipe_path = tmp_path / "recipe.json"
    config = dict(document["train_config"])
    config.update(
        {
            "run_name": "a2c-backend-smoke",
            "run_description": "ROM-free A2C backend boundary smoke.",
            "runs_dir": str(tmp_path),
            "timesteps": 64,
            "checkpoint_freq": 0,
            "checkpoint_eval_backend": "none",
            "early_stop": None,
            "wandb_mode": "disabled",
            "recipe_json_path": str(recipe_path),
            "training_backend": {
                "id": "sb3.a2c",
                "config": {
                    "device": "cpu",
                    "n_steps": 8,
                    "learning_rate": 0.01,
                    "gamma": 1.0,
                },
            },
        }
    )
    document["train_config"] = config
    recipe_path = _write_versioned_recipe(tmp_path, document)
    path = tmp_path / "a2c-train.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    monkeypatch.setenv("GRADLAB_INTERNAL_LEARNER", "1")
    assert (
        train_main(
            [
                "--train-config-json",
                str(path),
                "--execution-mode",
                "supervised",
            ]
        )
        == 0
    )

    run_dir = tmp_path / "a2c-backend-smoke"
    model_path = run_dir / "final_model.zip"
    bundle = load_policy_bundle_from_checkpoint(model_path)
    assert bundle is not None
    assert bundle.model["policy"]["training_backend_id"] == "sb3.a2c"
    assert bundle.model["policy"]["algorithm_id"] == "a2c"
    assert bundle.model["policy"]["model_class"] == "stable_baselines3.a2c.a2c.A2C"
    assert bundle.model["provenance"]["training_execution"]["mode"] == "supervised"
    metric_store = MetricStore(run_dir / "gradlab.sqlite")
    assert metric_store.latest_metric("train/algorithm/a2c/update/value_loss") is not None
    assert metric_store.latest_metric("train/algorithm/ppo/update/value_loss") is None
    from gradlab.trusted_inputs import approve_internal_model

    with approve_internal_model(model_path, execution_id="test-bandit") as approved:
        assert isinstance(load_sb3_model(approved, device="cpu", algorithm_id="a2c"), A2C)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_bandit_ppo_converges_under_stochastic_evaluation(seed: int) -> None:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    train_env = make_vec_envs(_config(), n_envs=8, seed=seed)
    eval_env = None
    try:
        model = PPO(
            "MlpPolicy",
            train_env,
            n_steps=8,
            batch_size=64,
            n_epochs=4,
            learning_rate=0.01,
            gamma=1.0,
            gae_lambda=1.0,
            ent_coef=0.0,
            clip_range=0.2,
            vf_coef=0.5,
            normalize_advantage=True,
            seed=seed,
            device="cpu",
            verbose=0,
        )
        model.learn(total_timesteps=256)

        eval_env = make_vec_envs(_config(), n_envs=32, seed=10_000 + seed)
        observations = eval_env.reset()
        rewards: list[float] = []
        for _ in range(32):
            actions, _state = model.predict(observations, deterministic=False)
            observations, batch_rewards, dones, _infos = eval_env.step(actions)
            assert dones.all()
            rewards.extend(float(value) for value in batch_rewards)
        assert len(rewards) == 1_024
        assert np.mean(rewards) >= 0.95
    finally:
        train_env.close()
        if eval_env is not None:
            eval_env.close()
        torch.set_num_threads(previous_threads)
