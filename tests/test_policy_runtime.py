from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy

from gradlab.action_program import ActionProgramPolicy, ActionRun
from gradlab.policy_runtime import PolicyRuntime


def _actor_critic_runtime(action_space: gym.Space) -> tuple[PolicyRuntime, np.ndarray]:
    observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
    policy = ActorCriticPolicy(
        observation_space,
        action_space,
        lambda _progress: 1e-3,
        net_arch=[8],
    )
    return PolicyRuntime(
        SimpleNamespace(policy=policy),
        algorithm_id="ppo",
    ), np.zeros((2, 4), dtype=np.float32)


def test_actor_critic_runtime_is_batched_and_uses_one_feature_pass(monkeypatch) -> None:
    runtime, observation = _actor_critic_runtime(gym.spaces.Discrete(3))
    calls = 0
    original = runtime.model.policy.extract_features

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime.model.policy, "extract_features", counted)
    result = runtime.decide(observation, action_selection_mode="stochastic")

    assert calls == 1
    assert result.actions.shape == (2,)
    assert len(result.decisions) == 2
    assert all(decision.value is not None for decision in result.decisions)
    assert all(decision.probabilities is not None for decision in result.decisions)
    assert runtime.capabilities.payload()["introspection"] == [
        "actor_distribution",
        "entropy",
        "selected_action_log_probability",
        "state_value",
    ]


def test_action_program_runtime_exposes_cursor_without_fabricated_distribution() -> None:
    model = ActionProgramPolicy(
        action_names=("noop", "right"),
        action_runs=(ActionRun(1, 2),),
        fallback_action=0,
    )
    runtime = PolicyRuntime(model, algorithm_id="action-program")
    runtime.bind_action_space(gym.spaces.Discrete(2))
    observation = np.zeros((1, 1), dtype=np.float32)

    first = runtime.decide(observation).decisions[0]
    second = runtime.decide(observation).decisions[0]
    fallback = runtime.decide(observation).decisions[0]

    assert first.program == {
        "run_index": 0,
        "step_index": 0,
        "current_run_remaining": 2,
        "remaining_steps": 2,
        "fallback": False,
        "action": 1,
        "action_name": "right",
    }
    assert second.program["step_index"] == 1
    assert second.program["remaining_steps"] == 1
    assert fallback.program["fallback"] is True
    assert fallback.program["remaining_steps"] == 0
    assert first.distribution_kind is None
    assert first.probabilities is None
    assert first.selected_probability is None
    assert first.selected_rank is None
    assert first.value is None
    assert first.entropy is None
    assert first.log_probability is None
    assert first.sampled is None
