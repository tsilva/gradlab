from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import DQN
from stable_baselines3.common.policies import ActorCriticPolicy

from gradlab.action_program import ActionProgramPolicy, ActionRun
from gradlab.policy_runtime import PolicyRuntime, normalize_action_selection_mode


def _actor_critic_runtime(action_space: gym.Space) -> tuple[PolicyRuntime, np.ndarray]:
    observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
    policy = ActorCriticPolicy(
        observation_space,
        action_space,
        lambda _progress: 1e-3,
        net_arch=[8],
    )
    return PolicyRuntime(SimpleNamespace(policy=policy)), np.zeros((2, 4), dtype=np.float32)


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


def test_dqn_runtime_reports_q_values_without_actor_critic_placeholders(monkeypatch) -> None:
    model = DQN(
        "MlpPolicy",
        "CartPole-v1",
        seed=7,
        learning_starts=100,
        buffer_size=128,
    )
    runtime = PolicyRuntime(model)
    observation = np.zeros((1, 4), dtype=np.float32)
    calls = 0
    original = model.policy.q_net.forward

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model.policy.q_net, "forward", counted)
    result = runtime.decide(observation, action_selection_mode="greedy")
    decision = result.decisions[0]

    assert calls == 1
    assert decision.action_selection_mode == "greedy"
    assert decision.q_values is not None
    assert decision.selected_discrete_action == int(np.argmax(decision.q_values))
    assert decision.selected_q_value == pytest.approx(float(np.max(decision.q_values)))
    assert decision.value is None
    assert decision.probabilities is None
    assert decision.entropy is None
    assert decision.log_probability is None
    assert decision.sampled is None
    assert runtime.capabilities.payload()["introspection"] == ["action_value"]
    model.get_env().close()


def test_dqn_epsilon_greedy_records_requested_and_effective_mode() -> None:
    model = DQN(
        "MlpPolicy",
        "CartPole-v1",
        seed=11,
        learning_starts=100,
        buffer_size=128,
    )
    model.exploration_rate = 1.0
    result = PolicyRuntime(model).decide(
        np.zeros((1, 4), dtype=np.float32),
        action_selection_mode="stochastic",
    )
    decision = result.decisions[0]

    assert result.requested_action_selection_mode == "stochastic"
    assert result.effective_action_selection_mode == "epsilon_greedy"
    assert decision.requested_action_selection_mode == "stochastic"
    assert decision.action_selection_mode == "epsilon_greedy"
    assert decision.exploratory is True
    model.get_env().close()


def test_action_program_runtime_exposes_cursor_without_fabricated_distribution() -> None:
    model = ActionProgramPolicy(
        action_names=("noop", "right"),
        action_runs=(ActionRun(1, 2),),
        fallback_action=0,
    )
    runtime = PolicyRuntime(model)
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


def test_action_program_legacy_boolean_modes_have_explicit_program_interpretation() -> None:
    capabilities = PolicyRuntime(
        ActionProgramPolicy(
            action_names=("noop",),
            action_runs=(),
            fallback_action=0,
        )
    ).capabilities

    assert normalize_action_selection_mode(capabilities, "stochastic") == (
        "stochastic",
        "program",
    )
    assert normalize_action_selection_mode(capabilities, "deterministic") == (
        "deterministic",
        "program",
    )
    with pytest.raises(ValueError, match="supported: program"):
        normalize_action_selection_mode(capabilities, "greedy")
