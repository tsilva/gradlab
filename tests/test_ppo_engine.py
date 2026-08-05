from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

from gradlab.ppo import GradLabPPO
from gradlab.task_advantage import normalize_advantages_by_context
from gradlab.training.ppo_engine import (
    TensorRolloutBuffer,
    _CompiledPolicyCalls,
    _EXECUTION_PROFILES,
    _normalize_grouped_advantages,
    _ppo_update,
    _Precision,
)


def test_execution_profiles_add_one_cuda_optimization_at_a_time() -> None:
    resolved = {
        name: (
            profile.compile_policy,
            profile.fused_optimizer,
            profile.torch_permutation,
        )
        for name, profile in _EXECUTION_PROFILES.items()
    }
    assert resolved == {
        "sb3-parity": (False, False, False),
        "compiled-parity": (True, False, False),
        "compiled-fused-parity": (True, True, False),
        "max-throughput": (True, True, True),
    }


@pytest.mark.parametrize(
    ("action_space", "sample_shape", "stored_width", "dtype"),
    [
        (gym.spaces.Discrete(3), (2,), 1, torch.int64),
        (gym.spaces.Box(-1.0, 1.0, shape=(2, 3), dtype=np.float32), (2, 2, 3), 6, torch.float32),
        (gym.spaces.MultiDiscrete([2, 3, 4]), (2, 3), 3, torch.int64),
        (gym.spaces.MultiBinary(4), (2, 4), 4, torch.float32),
    ],
)
def test_tensor_rollout_buffer_uses_sb3_action_storage_convention(
    action_space: gym.Space,
    sample_shape: tuple[int, ...],
    stored_width: int,
    dtype: torch.dtype,
) -> None:
    observations = np.zeros((2, 3), dtype=np.float32)
    buffer = TensorRolloutBuffer.allocate(
        observations,
        action_space=action_space,
        n_steps=2,
        n_envs=2,
        device=torch.device("cpu"),
    )
    actions = torch.zeros(sample_shape, dtype=dtype)
    buffer.add(
        observations,
        actions,
        torch.zeros(2),
        torch.ones(2, dtype=torch.bool),
        torch.zeros(2),
        torch.zeros(2),
    )

    assert buffer.actions.shape == (2, 2, stored_width)
    assert buffer.actions.dtype == dtype


def test_tensor_rollout_buffer_matches_sb3_gae_and_returns() -> None:
    observation_space = gym.spaces.Box(-10.0, 10.0, shape=(2,), dtype=np.float32)
    action_space = gym.spaces.Discrete(3)
    sb3_buffer = RolloutBuffer(
        3,
        observation_space,
        action_space,
        device="cpu",
        gamma=0.9,
        gae_lambda=0.8,
        n_envs=2,
    )
    first_observation = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    tensor_buffer = TensorRolloutBuffer.allocate(
        first_observation,
        action_space=action_space,
        n_steps=3,
        n_envs=2,
        device=torch.device("cpu"),
    )
    observations = [
        first_observation,
        np.asarray([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32),
        np.asarray([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
    ]
    actions = [
        np.asarray([[0], [1]], dtype=np.int64),
        np.asarray([[1], [2]], dtype=np.int64),
        np.asarray([[2], [0]], dtype=np.int64),
    ]
    rewards = [
        np.asarray([1.0, 0.5], dtype=np.float32),
        np.asarray([0.0, 2.0], dtype=np.float32),
        np.asarray([3.0, -1.0], dtype=np.float32),
    ]
    episode_starts = [
        np.asarray([True, True]),
        np.asarray([False, False]),
        np.asarray([True, False]),
    ]
    values = [
        torch.tensor([0.1, 0.2]),
        torch.tensor([0.3, 0.4]),
        torch.tensor([0.5, 0.6]),
    ]
    log_probs = [
        torch.tensor([-0.2, -0.3]),
        torch.tensor([-0.4, -0.5]),
        torch.tensor([-0.6, -0.7]),
    ]
    for index in range(3):
        sb3_buffer.add(
            observations[index],
            actions[index],
            rewards[index],
            episode_starts[index],
            values[index],
            log_probs[index],
        )
        tensor_buffer.add(
            observations[index],
            torch.as_tensor(actions[index]),
            torch.as_tensor(rewards[index]),
            torch.as_tensor(episode_starts[index]),
            values[index],
            log_probs[index],
        )
    last_values = torch.tensor([0.7, 0.8])
    dones = np.asarray([False, True])
    sb3_buffer.compute_returns_and_advantage(last_values=last_values, dones=dones)
    tensor_buffer.finish(
        last_values=last_values,
        dones=torch.as_tensor(dones),
        gamma=0.9,
        gae_lambda=0.8,
    )

    np.testing.assert_allclose(
        tensor_buffer.advantages.numpy(),
        sb3_buffer.advantages,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        tensor_buffer.returns.numpy(),
        sb3_buffer.returns,
        rtol=1e-6,
        atol=1e-6,
    )


def test_grouped_advantage_normalization_matches_existing_ppo_semantics() -> None:
    action_space = gym.spaces.Discrete(2)
    observations = [
        {
            "state": np.zeros((3, 2), dtype=np.float32),
            "context/task": np.asarray([[0], [1], [0]], dtype=np.int64),
        },
        {
            "state": np.ones((3, 2), dtype=np.float32),
            "context/task": np.asarray([[1], [0], [1]], dtype=np.int64),
        },
    ]
    buffer = TensorRolloutBuffer.allocate(
        observations[0],
        action_space=action_space,
        n_steps=2,
        n_envs=3,
        device=torch.device("cpu"),
    )
    for observation in observations:
        buffer.add(
            observation,
            torch.zeros(3, dtype=torch.int64),
            torch.zeros(3),
            torch.zeros(3, dtype=torch.bool),
            torch.zeros(3),
            torch.zeros(3),
        )
    advantages = np.asarray([[1.0, 5.0, 3.0], [7.0, 9.0, 11.0]], dtype=np.float32)
    buffer.advantages.copy_(torch.as_tensor(advantages))
    expected = advantages.copy()
    normalize_advantages_by_context(
        expected,
        {
            "context/task": np.stack(
                [observation["context/task"] for observation in observations]
            )
        },
        "task",
    )

    _normalize_grouped_advantages(buffer, "task")

    np.testing.assert_allclose(buffer.advantages.numpy(), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("method", ["collect_rollouts", "train", "learn"])
def test_gradlab_ppo_artifact_cannot_fall_back_to_sb3_training(method: str) -> None:
    model = object.__new__(GradLabPPO)
    with pytest.raises(RuntimeError, match="GradLabPPO"):
        getattr(model, method)()


def test_ppo_checkpoint_can_resume_across_sb3_and_gradlab_classes(tmp_path) -> None:
    env = make_vec_env("CartPole-v1", n_envs=2)
    try:
        sb3_model = PPO(
            "MlpPolicy",
            env,
            n_steps=4,
            batch_size=8,
            n_epochs=1,
            seed=7,
            device="cpu",
        )
        sb3_model.num_timesteps = 24
        observation = env.reset()
        action, _state = sb3_model.predict(observation, deterministic=True)
        sb3_model.policy.optimizer.zero_grad(set_to_none=True)
        loss = sum(parameter.square().mean() for parameter in sb3_model.policy.parameters())
        loss.backward()
        sb3_model.policy.optimizer.step()
        optimizer_state_count = len(sb3_model.policy.optimizer.state)
        sb3_path = tmp_path / "sb3.zip"
        sb3_model.save(sb3_path)

        gradlab_model = GradLabPPO.load(sb3_path, env=env, device="cpu")
        assert gradlab_model.num_timesteps == 24
        assert len(gradlab_model.policy.optimizer.state) == optimizer_state_count
        np.testing.assert_array_equal(
            gradlab_model.predict(observation, deterministic=True)[0],
            action,
        )
        gradlab_path = tmp_path / "gradlab.zip"
        gradlab_model.save(gradlab_path)

        resumed_sb3 = PPO.load(gradlab_path, env=env, device="cpu")
        assert resumed_sb3.num_timesteps == 24
        assert len(resumed_sb3.policy.optimizer.state) == optimizer_state_count
        np.testing.assert_array_equal(
            resumed_sb3.predict(observation, deterministic=True)[0],
            action,
        )
    finally:
        env.close()


def test_tensor_native_update_matches_one_sb3_ppo_update() -> None:
    env = make_vec_env("CartPole-v1", n_envs=2, seed=11)
    model_kwargs = {
        "n_steps": 2,
        "batch_size": 2,
        "n_epochs": 2,
        "learning_rate": 1e-3,
        "gamma": 0.9,
        "gae_lambda": 0.8,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "clip_range": 0.2,
        "normalize_advantage": True,
        "seed": 17,
        "device": "cpu",
    }
    try:
        sb3_model = PPO("MlpPolicy", env, **model_kwargs)
        candidate = GradLabPPO("MlpPolicy", env, **model_kwargs)
        candidate.rollout_buffer = None
        candidate.policy.load_state_dict(sb3_model.policy.state_dict())
        candidate.policy.optimizer.load_state_dict(sb3_model.policy.optimizer.state_dict())
        observations = [
            np.asarray(
                [[0.1, 0.2, 0.3, 0.4], [-0.2, 0.3, -0.4, 0.5]],
                dtype=np.float32,
            ),
            np.asarray(
                [[0.2, 0.1, 0.4, 0.3], [-0.1, 0.4, -0.3, 0.6]],
                dtype=np.float32,
            ),
        ]
        tensor_buffer = TensorRolloutBuffer.allocate(
            observations[0],
            action_space=env.action_space,
            n_steps=2,
            n_envs=2,
            device=torch.device("cpu"),
        )
        episode_starts = [np.asarray([True, True]), np.asarray([False, False])]
        rewards = [
            np.asarray([1.0, 0.5], dtype=np.float32),
            np.asarray([0.25, 1.5], dtype=np.float32),
        ]
        torch.manual_seed(23)
        for index, observation in enumerate(observations):
            with torch.no_grad():
                actions, values, log_probs = sb3_model.policy(
                    torch.as_tensor(observation)
                )
            sb3_model.rollout_buffer.add(
                observation,
                actions.numpy().reshape(2, 1),
                rewards[index],
                episode_starts[index],
                values,
                log_probs,
            )
            tensor_buffer.add(
                observation,
                actions,
                torch.as_tensor(rewards[index]),
                torch.as_tensor(episode_starts[index]),
                values,
                log_probs,
            )
        next_observation = np.asarray(
            [[0.3, 0.2, 0.5, 0.4], [0.0, 0.5, -0.2, 0.7]],
            dtype=np.float32,
        )
        with torch.no_grad():
            last_values = sb3_model.policy.predict_values(
                torch.as_tensor(next_observation)
            )
        dones = np.asarray([False, True])
        sb3_model.rollout_buffer.compute_returns_and_advantage(
            last_values=last_values,
            dones=dones,
        )
        tensor_buffer.finish(
            last_values=last_values,
            dones=torch.as_tensor(dones),
            gamma=0.9,
            gae_lambda=0.8,
        )

        sb3_model.set_logger(configure(format_strings=[]))
        sb3_model._current_progress_remaining = 0.75
        np.random.seed(29)
        torch.manual_seed(29)
        sb3_model.train()
        np.random.seed(29)
        torch.manual_seed(29)
        metrics = _ppo_update(
            candidate,
            tensor_buffer,
            calls=_CompiledPolicyCalls(candidate.policy, torch.device("cpu")),
            precision=_Precision("fp32", torch.device("cpu")),
            progress_remaining=0.75,
            normalization_mode="global",
            advantage_context=None,
            ent_coef=0.01,
            torch_permutation=False,
        )

        for name, expected in sb3_model.policy.state_dict().items():
            torch.testing.assert_close(
                candidate.policy.state_dict()[name],
                expected,
                rtol=2e-6,
                atol=2e-6,
            )
        assert metrics["train/algorithm/ppo/update/learning_rate"] == pytest.approx(1e-3)
    finally:
        env.close()
