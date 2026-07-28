from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gradlab.action_program import ActionProgramPolicy, action_runs_from_sequence
from gradlab.task_kernels import Outcome


@dataclass
class RetainedSequence:
    actions: tuple[int, ...]
    return_sum: float = 0.0
    return_count: int = 0
    completed: bool = False
    progress: float = 0.0

    @property
    def mean_return(self) -> float:
        return self.return_sum / self.return_count if self.return_count else float("-inf")

    def observe(self, value: float, *, completed: bool, progress: float) -> None:
        self.return_sum += float(value)
        self.return_count += 1
        self.completed |= bool(completed)
        self.progress = max(self.progress, float(progress))

    @property
    def rank(self) -> tuple[float, ...]:
        return (
            float(self.completed),
            float(self.progress),
            self.mean_return,
            -float(len(self.actions)),
        )


@dataclass
class _LaneState:
    mode: str = "explore"
    actions: list[int] = field(default_factory=list)
    episode_return: float = 0.0
    best_return: float = float("-inf")
    best_length: int = 0
    archive_candidate: RetainedSequence | None = None
    replay_limit: int = 0


class JerkSearch:
    """Vectorized Just Enough Retained Knowledge action-sequence search."""

    def __init__(
        self,
        *,
        n_envs: int,
        seed: int,
        total_timesteps: int,
        action_names: Sequence[str],
        fallback_action: str,
        archive_replay_probability_initial: float,
        archive_replay_probability_max: float,
        protected_prefix_steps: int,
        max_prefix_shorten_steps: int,
        retained_limit: int,
    ) -> None:
        if n_envs < 1:
            raise ValueError("JERK requires at least one environment")
        self.n_envs = int(n_envs)
        self.total_timesteps = max(int(total_timesteps), 1)
        self.action_names = tuple(str(name) for name in action_names)
        indices = {name: index for index, name in enumerate(self.action_names)}
        if not self.action_names:
            raise ValueError("JERK requires at least one action name")
        if fallback_action not in indices:
            raise ValueError(
                f"JERK fallback action is absent from the task action set: {fallback_action}"
            )
        self.fallback_action = indices[fallback_action]
        self.archive_replay_probability_initial = float(archive_replay_probability_initial)
        self.archive_replay_probability_max = float(archive_replay_probability_max)
        if not (
            0.0
            <= self.archive_replay_probability_initial
            <= self.archive_replay_probability_max
            <= 1.0
        ):
            raise ValueError(
                "JERK probabilities must satisfy 0 <= archive_replay_probability_initial "
                "<= archive_replay_probability_max <= 1"
            )
        self.protected_prefix_steps = int(protected_prefix_steps)
        self.max_prefix_shorten_steps = int(max_prefix_shorten_steps)
        if self.protected_prefix_steps < 0:
            raise ValueError("JERK protected_prefix_steps must be non-negative")
        if self.max_prefix_shorten_steps < 1:
            raise ValueError("JERK max_prefix_shorten_steps must be positive")
        self.retained_limit = int(retained_limit)
        self.global_step = 0
        self.completed_episodes = 0
        self.archive_replay_episodes = 0
        self.archive_selected_prefix_return_sum = 0.0
        self._retained: dict[tuple[int, ...], RetainedSequence] = {}
        self._lanes = [_LaneState() for _ in range(self.n_envs)]
        self._rngs = [
            np.random.default_rng(np.random.SeedSequence([seed, lane, 0x4A45524B]))
            for lane in range(self.n_envs)
        ]

    @property
    def archive_replay_probability(self) -> float:
        return min(
            self.archive_replay_probability_max,
            self.archive_replay_probability_initial + self.global_step / self.total_timesteps,
        )

    @property
    def archive_selected_prefix_return_mean(self) -> float:
        if not self.archive_replay_episodes:
            return 0.0
        return self.archive_selected_prefix_return_sum / self.archive_replay_episodes

    @property
    def retained_count(self) -> int:
        return len(self._retained)

    def _retained_distribution(self) -> tuple[list[RetainedSequence], np.ndarray]:
        candidates = sorted(self._retained.values(), key=lambda candidate: candidate.actions)
        returns = np.asarray([candidate.mean_return for candidate in candidates], dtype=np.float64)
        weights = returns - float(np.min(returns)) + 1e-12
        probabilities = weights / float(np.sum(weights))
        return candidates, probabilities

    def _sample_retained(self, lane: int) -> RetainedSequence:
        candidates, probabilities = self._retained_distribution()
        index = int(self._rngs[lane].choice(len(candidates), p=probabilities))
        return candidates[index]

    def _start_lane(self, lane: int) -> None:
        state = _LaneState()
        if self._retained and self._rngs[lane].random() < self.archive_replay_probability:
            candidate = self._sample_retained(lane)
            state.mode = "replay"
            state.archive_candidate = candidate
            length = len(candidate.actions)
            if length > self.protected_prefix_steps:
                shorten_limit = min(
                    self.max_prefix_shorten_steps,
                    length - self.protected_prefix_steps,
                )
                shorten_steps = int(self._rngs[lane].integers(1, shorten_limit + 1))
                state.replay_limit = length - shorten_steps
            else:
                state.replay_limit = length
            self.archive_replay_episodes += 1
            self.archive_selected_prefix_return_sum += candidate.mean_return
        self._lanes[lane] = state

    def _next_exploration_action(self, lane: int) -> int:
        return int(self._rngs[lane].integers(0, len(self.action_names)))

    def next_actions(self) -> np.ndarray:
        actions = np.empty(self.n_envs, dtype=np.int64)
        for lane, state in enumerate(self._lanes):
            if state.mode == "replay":
                candidate = state.archive_candidate
                if candidate is not None and len(state.actions) < state.replay_limit:
                    action = candidate.actions[len(state.actions)]
                else:
                    state.mode = "explore"
                    state.archive_candidate = None
                    action = self._next_exploration_action(lane)
            else:
                action = self._next_exploration_action(lane)
            state.actions.append(int(action))
            actions[lane] = action
        return actions

    @staticmethod
    def _record_facts(record: Any | None) -> tuple[bool, float]:
        if record is None:
            return False, 0.0
        metrics = getattr(record, "metrics", {}) or {}
        completed = getattr(record, "outcome", Outcome.NEUTRAL) == Outcome.SUCCESS or bool(
            metrics.get("level_complete", False)
        )
        progress = float(metrics.get("max_x_pos", metrics.get("global_max_x_pos", 0.0)) or 0.0)
        return completed, progress

    def _retain_exploration(self, state: _LaneState, record: Any | None) -> None:
        completed, progress = self._record_facts(record)
        if completed:
            actions = tuple(state.actions)
            score_return = state.episode_return
        else:
            actions = tuple(state.actions[: state.best_length])
            score_return = state.best_return
        if not actions or not math.isfinite(score_return):
            return
        self._upsert_retained(
            actions,
            score_return=score_return,
            completed=completed,
            progress=progress,
        )

    def _upsert_retained(
        self,
        actions: tuple[int, ...],
        *,
        score_return: float,
        completed: bool,
        progress: float,
    ) -> None:
        candidate = self._retained.get(actions)
        if candidate is None:
            candidate = RetainedSequence(actions=actions)
            self._retained[actions] = candidate
        candidate.observe(score_return, completed=completed, progress=progress)
        if len(self._retained) > self.retained_limit:
            retained = sorted(self._retained.values(), key=lambda item: item.rank, reverse=True)
            self._retained = {item.actions: item for item in retained[: self.retained_limit]}

    def observe(
        self,
        rewards: Sequence[float],
        dones: Sequence[bool],
        records_by_lane: Mapping[int, Any] | None = None,
    ) -> None:
        rewards_array = np.asarray(rewards, dtype=np.float64)
        dones_array = np.asarray(dones, dtype=bool)
        if rewards_array.shape != (self.n_envs,) or dones_array.shape != (self.n_envs,):
            raise ValueError("JERK rewards and dones must contain one value per environment")
        records_by_lane = records_by_lane or {}
        self.global_step += self.n_envs
        for lane, state in enumerate(self._lanes):
            reward = float(rewards_array[lane])
            state.episode_return += reward
            if state.episode_return > state.best_return:
                state.best_return = state.episode_return
                state.best_length = len(state.actions)
            if dones_array[lane]:
                record = records_by_lane.get(lane)
                self._retain_exploration(state, record)
                self.completed_episodes += 1
                self._start_lane(lane)

    def best_candidate(self) -> RetainedSequence | None:
        candidates = list(self._retained.values())
        for state in self._lanes:
            if state.mode != "replay" and state.best_length > 0:
                candidates.append(
                    RetainedSequence(
                        actions=tuple(state.actions[: state.best_length]),
                        return_sum=state.best_return,
                        return_count=1,
                    )
                )
        return max(candidates, key=lambda candidate: candidate.rank, default=None)

    def policy(self) -> ActionProgramPolicy:
        candidate = self.best_candidate()
        return ActionProgramPolicy(
            action_names=self.action_names,
            action_runs=(() if candidate is None else action_runs_from_sequence(candidate.actions)),
            fallback_action=self.fallback_action,
        )
