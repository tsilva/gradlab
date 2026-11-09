"""Actor-only self-imitation for native actor-critic learners."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
import math
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class SilConfig:
    enabled: bool = False
    learning_rate: float = 1e-4
    updates_per_rollout: int = 16
    batch_size: int = 256
    candidate_pool_size: int = 2048
    recent_capacity_transitions: int = 65536
    success_capacity_transitions: int = 65536
    success_capacity_episodes: int = 32
    success_sample_fraction: float = 0.5
    max_episode_transitions: int = 8192
    replay_seed: int | None = None
    control_signal: str | None = None
    control_value: int | None = None


def normalize_sil(value: Any, *, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping or null")
    unknown = sorted(set(value) - {field.name for field in fields(SilConfig)})
    if unknown:
        raise ValueError(f"{label} has unexpected fields: {unknown}")
    config = SilConfig(**value)
    if not isinstance(config.enabled, bool):
        raise ValueError(f"{label}.enabled must be boolean")
    for field in fields(config):
        key = field.name
        if key in {"enabled", "learning_rate", "success_sample_fraction", "replay_seed", "control_signal", "control_value"}:
            continue
        item = getattr(config, key)
        if type(item) is not int or item <= 0:
            raise ValueError(f"{label}.{key} must be a positive integer")
    rate = config.learning_rate
    if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
        raise ValueError(f"{label}.learning_rate must be finite and positive")
    fraction = config.success_sample_fraction
    if type(fraction) not in (int, float) or not 0 <= fraction <= 1:
        raise ValueError(f"{label}.success_sample_fraction must be between 0 and 1")
    if config.replay_seed is not None and (
        type(config.replay_seed) is not int or not 0 <= config.replay_seed < 2**64
    ):
        raise ValueError(f"{label}.replay_seed must be an unsigned 64-bit integer or null")
    if config.control_signal is not None and (
        not isinstance(config.control_signal, str) or not config.control_signal.strip()
        or type(config.control_value) is not int
    ):
        raise ValueError(f"{label}.control_signal requires a nonempty name and integer control_value")
    if config.control_signal is None and config.control_value is not None:
        raise ValueError(f"{label}.control_value requires control_signal")
    if config.max_episode_transitions > min(
        config.recent_capacity_transitions, config.success_capacity_transitions
    ):
        raise ValueError(f"{label}.max_episode_transitions must fit both replay pools")
    normalized = asdict(config)
    normalized.update(learning_rate=float(rate), success_sample_fraction=float(fraction))
    return normalized if config.enabled else None


def discounted_returns(rewards: Sequence[float], gamma: float) -> np.ndarray:
    returns = np.empty(len(rewards), dtype=np.float32)
    future = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        future = float(rewards[index]) + gamma * future
        returns[index] = future
    return returns


@dataclass(frozen=True)
class ReplayEpisode:
    observations: np.ndarray
    actions: np.ndarray
    returns: np.ndarray
    success: bool

    def __len__(self) -> int:
        return len(self.actions)


class SilReplay:
    def __init__(self, config: SilConfig, *, seed: int) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.replay_seed if config.replay_seed is not None else seed)
        self.recent: deque[ReplayEpisode] = deque()
        self.successes: list[ReplayEpisode] = []
        self.size = 0
        self.success_size = 0
        self.successes_seen = 0

    def add(self, episode: ReplayEpisode) -> None:
        if not len(episode):
            return
        if len(episode) > self.config.max_episode_transitions:
            raise ValueError("SIL episode exceeds its configured transition limit")
        self.recent.append(episode)
        self.size += len(episode)
        while self.size > self.config.recent_capacity_transitions:
            self.size -= len(self.recent.popleft())
        if not episode.success:
            return
        self.successes_seen += 1
        if len(self.successes) < self.config.success_capacity_episodes:
            self.successes.append(episode)
            self.success_size += len(episode)
        else:
            # Reserve index zero for the first success, as in the Mario study.
            index = int(self.rng.integers(1, self.successes_seen))
            if index < len(self.successes):
                self.success_size += len(episode) - len(self.successes[index])
                self.successes[index] = episode
        while self.success_size > self.config.success_capacity_transitions:
            self.success_size -= len(self.successes.pop(int(self.rng.integers(1, len(self.successes)))))

    def _sample(self, episodes: Sequence[ReplayEpisode], count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ends = np.cumsum([len(episode) for episode in episodes])
        locations = self.rng.integers(int(ends[-1]), size=count)
        groups = np.searchsorted(ends, locations, side="right")
        starts = np.concatenate(([0], ends[:-1]))
        offsets = locations - starts[groups]
        rows = [(episodes[group], offset) for group, offset in zip(groups, offsets, strict=True)]
        return (
            np.stack([episode.observations[offset] for episode, offset in rows]),
            np.array([episode.actions[offset] for episode, offset in rows], dtype=np.int64),
            np.array([episode.returns[offset] for episode, offset in rows], dtype=np.float32),
        )

    def sample(self, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        success_count = int(count * self.config.success_sample_fraction) if self.successes else 0
        if success_count == 0:
            return self._sample(list(self.recent), count)
        if success_count == count:
            return self._sample(self.successes, count)
        recent = self._sample(list(self.recent), count - success_count)
        success = self._sample(self.successes, success_count)
        return tuple(np.concatenate((a, b)) for a, b in zip(recent, success, strict=True))


@dataclass
class _Transition:
    observation: np.ndarray
    action: int
    reward: float
    eligible: bool


class SilEpisodes:
    """Accumulate complete episodes independently of the learner's rollout length."""

    def __init__(self, replay: SilReplay, *, n_envs: int, gamma: float) -> None:
        self.replay = replay
        self.gamma = gamma
        self.lanes: list[list[_Transition] | None] = [[] for _ in range(n_envs)]
        self.dropped_episodes = 0

    def collect(
        self, observations: np.ndarray, actions: np.ndarray, rewards: np.ndarray, eligible: np.ndarray,
    ) -> None:
        for lane, history in enumerate(self.lanes):
            if history is None:
                continue
            if len(history) == self.replay.config.max_episode_transitions:
                self.lanes[lane] = None
                self.dropped_episodes += 1
                continue
            history.append(_Transition(observations[lane], int(actions[lane]), float(rewards[lane]), bool(eligible[lane])))

    def complete(self, *, lane: int, length: int, success: bool, discard: bool = False) -> None:
        history = self.lanes[lane]
        self.lanes[lane] = []
        if history is None or discard:
            return
        if len(history) != length:
            raise RuntimeError(f"SIL episode length mismatch on lane {lane}: {len(history)} != {length}")
        eligible = np.array([row.eligible for row in history], dtype=bool)
        if eligible.any():
            returns = discounted_returns([row.reward for row in history], self.gamma)
            selected = [row for row in history if row.eligible]
            self.replay.add(ReplayEpisode(
                np.stack([row.observation for row in selected]),
                np.array([row.action for row in selected], dtype=np.int64),
                returns[eligible],
                success,
            ))


class SilPolicy(Protocol):
    def predict_values(self, observations: torch.Tensor) -> torch.Tensor: ...

    def evaluate_actions(
        self, observations: torch.Tensor, actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]: ...


class SilUpdater:
    def __init__(
        self, policy: SilPolicy, actor_parameters: Sequence[torch.nn.Parameter],
        replay: SilReplay, *, device: torch.device, max_grad_norm: float,
    ) -> None:
        import torch

        self.policy = policy
        self.parameters = tuple(actor_parameters)
        self.replay = replay
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.optimizer = torch.optim.Adam(self.parameters, lr=replay.config.learning_rate, eps=1e-5)

    def update(self) -> dict[str, int | float]:
        import torch

        replay = self.replay
        config = replay.config
        attempts = config.updates_per_rollout if replay.size else 0
        eligible = []
        losses = []
        gaps = []
        for _ in range(attempts):
            obs, actions, returns = replay.sample(min(config.candidate_pool_size, replay.size))
            with torch.no_grad():
                values = self.policy.predict_values(torch.as_tensor(obs, device=self.device)).flatten()
                gap = (torch.as_tensor(returns, device=self.device) - values).clamp_min(0)
                priorities = gap.cpu().numpy()
            if not np.isfinite(priorities).all():
                raise RuntimeError("SIL encountered a nonfinite replay priority")
            positive = priorities > 0
            eligible.append(float(positive.mean()))
            if not positive.any():
                continue
            indices = replay.rng.choice(
                len(obs), size=config.batch_size, replace=True,
                p=priorities.astype(np.float64) / priorities.sum(dtype=np.float64),
            )
            values, log_probs, _ = self.policy.evaluate_actions(
                torch.as_tensor(obs[indices], device=self.device),
                torch.as_tensor(actions[indices], device=self.device),
            )
            gap = (torch.as_tensor(returns[indices], device=self.device) - values.detach().flatten()).clamp_min(0)
            loss = -(log_probs * gap).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("SIL encountered a nonfinite policy loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm, error_if_nonfinite=True)
            self.optimizer.step()
            losses.append(float(loss.detach()))
            gaps.append(float(gap.mean()))
        metrics = {
            "train/sil/attempts": attempts,
            "train/sil/updates": len(losses),
            "train/sil/recent_transitions": replay.size,
            "train/sil/success_transitions": replay.success_size,
            "train/sil/success_episodes": len(replay.successes),
        }
        if eligible:
            metrics["train/sil/eligible_fraction"] = float(np.mean(eligible))
        if losses:
            metrics["train/sil/policy_loss"] = float(np.mean(losses))
            metrics["train/sil/positive_gap"] = float(np.mean(gaps))
        return metrics
