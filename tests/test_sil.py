from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from gradlab.training.ppo import GradLabPPOBackend
from gradlab.training.sil import ReplayEpisode, SilConfig, SilEpisodes, SilReplay, SilUpdater


def test_sil_uses_the_existing_backend_config_boundary():
    backend = GradLabPPOBackend()
    baseline = backend.normalize_config({}, label="ppo")
    assert backend.normalize_config({"sil": {"enabled": False}}, label="ppo") == baseline
    enabled = backend.normalize_config(
        {"bootstrap_truncations": False, "sil": {"enabled": True, "updates_per_rollout": 4}},
        label="ppo",
    )
    assert enabled["sil"]["updates_per_rollout"] == 4
    backend.validate({"policy_model": {"share_features_extractor": False}}, enabled)
    with pytest.raises(ValueError, match="disjoint|share_features_extractor"):
        backend.validate({"policy_model": {}}, enabled)
    with pytest.raises(ValueError, match="unexpected fields"):
        backend.normalize_config({"sil": {"enabled": True, "udpates_per_rollout": 4}}, label="ppo")


def test_episode_capture_uses_all_rewards_before_filtering_actions():
    replay = SilReplay(SilConfig(), seed=19)
    episodes = SilEpisodes(replay, n_envs=2, gamma=0.5)
    episodes.collect(np.array([[11], [21]]), np.array([0, 1]), np.array([0, 4]), np.array([True, True]))
    episodes.complete(lane=1, length=1, success=False)
    episodes.collect(np.array([[12], [22]]), np.array([1, 0]), np.array([8, 0]), np.array([False, True]))
    episodes.complete(lane=0, length=2, success=True)
    episodes.complete(lane=1, length=1, success=False, discard=True)

    assert replay.size == 2
    assert replay.successes_seen == 1
    assert replay.successes[0].observations.tolist() == [[11]]
    assert replay.successes[0].actions.tolist() == [0]
    assert replay.successes[0].returns.tolist() == [4.0]


def test_replay_keeps_first_success_when_both_pools_evict():
    config = SilConfig(recent_capacity_transitions=4, success_capacity_transitions=4,
                       success_capacity_episodes=2, max_episode_transitions=2)
    replay = SilReplay(config, seed=19)
    for i in range(20):
        replay.add(ReplayEpisode(np.full((2, 1), i), np.zeros(2), np.ones(2), True))
    assert replay.recent[0].observations[0, 0] == 18
    assert replay.successes[0].observations[0, 0] == 0
    assert replay.size == replay.success_size == 4


class _Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Linear(1, 2, bias=False)
        self.critic = nn.Linear(1, 1, bias=False)
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.critic.weight)

    def predict_values(self, observations):
        return self.critic(observations)

    def evaluate_actions(self, observations, actions):
        distribution = torch.distributions.Categorical(logits=self.actor(observations))
        return self.predict_values(observations), distribution.log_prob(actions), None


def test_sil_skips_nonpositive_returns_and_only_updates_actor():
    policy = _Policy()
    replay = SilReplay(SilConfig(updates_per_rollout=2, batch_size=2), seed=19)
    updater = SilUpdater(policy, list(policy.actor.parameters()), replay,
                         device=torch.device("cpu"), max_grad_norm=0.5)
    observations = np.ones((1, 1), dtype=np.float32)
    actions = np.array([1])
    replay.add(ReplayEpisode(observations, actions, np.array([-1.0], dtype=np.float32), False))
    assert updater.update()["train/sil/updates"] == 0
    assert not updater.optimizer.state
    replay.add(ReplayEpisode(observations, actions, np.array([2.0], dtype=np.float32), True))
    stats = updater.update()
    assert stats["train/sil/updates"] > 0
    assert policy.actor(torch.ones((1, 1)))[0, 1] > 0
    assert policy.critic.weight.detach().tolist() == [[0.0]]
    assert policy.critic.weight.grad is None
