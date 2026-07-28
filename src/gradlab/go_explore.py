"""Deterministic Go-Explore trajectory discovery over gradlab state-archive entries."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from gradlab.action_program import ActionProgramPolicy, ActionRun, canonicalize_action_runs
from gradlab.task_kernels import Outcome


RECENT_CELL_VISIT_WINDOW = 10_000
SUCCESS_GUIDED_RESTORE_PROBABILITY = 0.5
GO_EXPLORE_STATE_SEMANTIC_ID = "go-explore-state-v1"


@dataclass(frozen=True)
class CompletionEvent:
    runs: tuple[ActionRun, ...]
    episode_return: float
    progress: float
    improved: bool


@dataclass(frozen=True)
class GoExploreCandidate:
    runs: tuple[ActionRun, ...]
    episode_return: float
    progress: float
    completed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "runs", canonicalize_action_runs(self.runs))

    @property
    def step_count(self) -> int:
        return sum(run.duration for run in self.runs)


@dataclass
class GoExploreCell:
    key: Hashable
    entry_id: str
    runs: tuple[ActionRun, ...]
    episode_return: float
    progress: float
    program_steps: int = 0
    parent_key: Hashable | None = None
    best_success_return: float | None = None
    success_selections: int = 0
    visits: int = 0
    selections: int = 0
    updates: int = 0
    selection_index: int = -1
    selection_weight: float = 0.0

    @property
    def step_count(self) -> int:
        return self.program_steps


def _selection_weight(cell: GoExploreCell) -> float:
    return 1.0 / math.sqrt(1.0 + cell.selections) + 1.0 / math.sqrt(1.0 + cell.visits)


class _SelectionWeightTree:
    def __init__(self) -> None:
        self._tree = [0.0]

    def append(self, weight: float) -> None:
        position = len(self._tree)
        range_start = position - (position & -position)
        subtotal = float(weight)
        cursor = position - 1
        while cursor > range_start:
            subtotal += self._tree[cursor]
            cursor -= cursor & -cursor
        self._tree.append(subtotal)

    def add(self, index: int, delta: float) -> None:
        position = int(index) + 1
        while position < len(self._tree):
            self._tree[position] += float(delta)
            position += position & -position

    @property
    def total(self) -> float:
        subtotal = 0.0
        position = len(self._tree) - 1
        while position:
            subtotal += self._tree[position]
            position -= position & -position
        return subtotal

    def sample(self, unit: float) -> int:
        count = len(self._tree) - 1
        if count == 0:
            raise RuntimeError("cannot sample an empty Go-Explore archive")
        target = float(unit) * self.total
        index = 0
        bit = 1 << (count.bit_length() - 1)
        while bit:
            candidate = index + bit
            if candidate <= count and self._tree[candidate] <= target:
                index = candidate
                target -= self._tree[candidate]
            bit >>= 1
        return min(index, count - 1)


@dataclass(frozen=True)
class GoExploreObservation:
    archive_mask: np.ndarray
    restart_mask: np.ndarray


@dataclass
class _PendingCell:
    key: Hashable
    lane: int
    runs: tuple[ActionRun, ...]
    episode_return: float
    progress: float
    step_count: int
    parent_key: Hashable | None
    visits: int


@dataclass
class _LaneState:
    runs: list[ActionRun] = field(default_factory=list)
    episode_return: float = 0.0
    progress: float = 0.0
    program_steps: int = 0
    steps_since_restart: int = 0
    path_cell_keys: list[Hashable] = field(default_factory=list)
    exploration_action: int = 0
    exploration_remaining: int = 0


class GoExploreSearch:
    """Trajectory-finding Go-Explore; robustification is deliberately out of scope."""

    def __init__(
        self,
        *,
        n_envs: int,
        seed: int,
        action_names: Sequence[str],
        fallback_action: str,
        explore_steps: int,
        run_duration_mean: float,
        run_duration_max: int,
    ) -> None:
        if n_envs < 1 or explore_steps < 1:
            raise ValueError("Go-Explore environment and exploration counts must be positive")
        if run_duration_mean < 1.0 or run_duration_max < 1:
            raise ValueError("Go-Explore run durations must be at least one")
        self.n_envs = int(n_envs)
        self.action_names = tuple(str(name) for name in action_names)
        if not self.action_names:
            raise ValueError("Go-Explore requires at least one action")
        try:
            self.fallback_action = self.action_names.index(str(fallback_action))
        except ValueError as exc:
            raise ValueError(
                f"Go-Explore fallback action {fallback_action!r} is not configured"
            ) from exc
        self.explore_steps = int(explore_steps)
        self.run_duration_mean = float(run_duration_mean)
        self.run_duration_max = int(run_duration_max)
        self.global_step = 0
        self.completed_episodes = 0
        self.successful_episodes = 0
        self.improvement_count = 0
        self.first_success_return: float | None = None
        self._archive: dict[Hashable, GoExploreCell] = {}
        self._selection_cells: list[GoExploreCell] = []
        self._selection_weights = _SelectionWeightTree()
        self._archive_selection_count = 0
        self._archive_visit_count = 0
        self._archive_update_count = 0
        self._recent_batches: deque[tuple[int, int]] = deque()
        self._recent_visits = 0
        self._recent_new_cells = 0
        self._elite_success_keys: tuple[Hashable, ...] = ()
        self._success_guided_selection_count = 0
        self._lanes = [_LaneState() for _ in range(self.n_envs)]
        self._rngs = [
            np.random.default_rng(np.random.SeedSequence([seed, lane, 0x474F4558]))
            for lane in range(self.n_envs)
        ]
        self._pending: dict[Hashable, _PendingCell] = {}
        self._best_incomplete: GoExploreCandidate | None = None
        self._best_success: GoExploreCandidate | None = None
        self._completion_events: list[CompletionEvent] = []

    @property
    def archive_count(self) -> int:
        return len(self._archive)

    @property
    def archive_selection_count(self) -> int:
        return self._archive_selection_count

    @property
    def archive_visit_count(self) -> int:
        return self._archive_visit_count

    @property
    def archive_update_count(self) -> int:
        return self._archive_update_count

    @property
    def archive_recent_new_cell_rate(self) -> float:
        return self._recent_new_cells / self._recent_visits if self._recent_visits else 0.0

    @property
    def archive_recent_visit_window(self) -> int:
        return self._recent_visits

    @property
    def archive_visits_per_cell(self) -> float:
        return self._archive_visit_count / len(self._archive) if self._archive else 0.0

    @property
    def success_guided_cell_count(self) -> int:
        return len(self._elite_success_keys)

    @property
    def success_guided_selection_count(self) -> int:
        return self._success_guided_selection_count

    @property
    def archive(self) -> Mapping[Hashable, GoExploreCell]:
        return self._archive

    def _register(self, cell: GoExploreCell) -> None:
        cell.selection_index = len(self._selection_cells)
        cell.selection_weight = _selection_weight(cell)
        self._selection_cells.append(cell)
        self._selection_weights.append(cell.selection_weight)

    def _update_weight(self, cell: GoExploreCell) -> None:
        weight = _selection_weight(cell)
        self._selection_weights.add(
            cell.selection_index,
            weight - cell.selection_weight,
        )
        cell.selection_weight = weight

    def initialize(
        self,
        cell_keys: Sequence[Hashable],
        entry_ids: Sequence[str | None],
    ) -> None:
        if len(cell_keys) != self.n_envs or len(entry_ids) != self.n_envs:
            raise ValueError("Go-Explore initialization requires one value per lane")
        for lane, (key, entry_id) in enumerate(zip(cell_keys, entry_ids, strict=True)):
            if not entry_id:
                raise ValueError("Go-Explore initialization entries cannot be empty")
            cell = self._archive.get(key)
            if cell is None:
                cell = GoExploreCell(
                    key=key,
                    entry_id=str(entry_id),
                    runs=(),
                    episode_return=0.0,
                    progress=0.0,
                    visits=1,
                )
                self._archive[key] = cell
                self._register(cell)
            else:
                cell.visits += 1
                self._update_weight(cell)
            self._archive_visit_count += 1
            self._lanes[lane] = _LaneState(path_cell_keys=[key])

    @staticmethod
    def _append_action(state: _LaneState, action: int) -> None:
        if state.runs and state.runs[-1].action == action:
            previous = state.runs[-1]
            state.runs[-1] = ActionRun(action, previous.duration + 1)
        else:
            state.runs.append(ActionRun(action, 1))
        state.program_steps += 1
        state.steps_since_restart += 1

    def _sample_run(self, lane: int, state: _LaneState) -> None:
        rng = self._rngs[lane]
        previous = state.runs[-1].action if state.runs else None
        count = len(self.action_names)
        if previous is None or count == 1:
            action = int(rng.integers(0, count))
        else:
            sampled = int(rng.integers(0, count - 1))
            action = sampled + int(sampled >= previous)
        state.exploration_action = action
        state.exploration_remaining = min(
            int(rng.geometric(1.0 / self.run_duration_mean)),
            self.run_duration_max,
        )

    def next_actions(self) -> np.ndarray:
        if not self._archive:
            raise RuntimeError("Go-Explore must be initialized before stepping")
        actions = np.empty(self.n_envs, dtype=np.int64)
        for lane, state in enumerate(self._lanes):
            if state.exploration_remaining == 0:
                self._sample_run(lane, state)
            action = state.exploration_action
            state.exploration_remaining -= 1
            self._append_action(state, action)
            actions[lane] = action
        return actions

    @staticmethod
    def _record_facts(record: object | None) -> tuple[bool, float]:
        if record is None:
            return False, 0.0
        metrics = getattr(record, "metrics", {}) or {}
        completed = getattr(record, "outcome", Outcome.NEUTRAL) == Outcome.SUCCESS or bool(
            metrics.get("level_complete", False)
        )
        progress = float(metrics.get("max_x_pos", metrics.get("global_max_x_pos", 0.0)) or 0.0)
        return completed, progress

    @staticmethod
    def _better(
        episode_return: float,
        step_count: int,
        cell: GoExploreCell | _PendingCell | None,
    ) -> bool:
        return (
            cell is None
            or episode_return > cell.episode_return
            or (episode_return == cell.episode_return and step_count < cell.step_count)
        )

    def _consider_best(
        self,
        state: _LaneState,
        *,
        progress: float,
        completed: bool,
    ) -> None:
        if completed:
            candidate = GoExploreCandidate(
                runs=tuple(state.runs),
                episode_return=state.episode_return,
                progress=progress,
                completed=True,
            )
            previous = self._best_success
            improved = previous is None or candidate.episode_return > previous.episode_return
            if improved:
                self._best_success = candidate
                if self.first_success_return is None:
                    self.first_success_return = candidate.episode_return
                else:
                    self.improvement_count += 1
                self._record_success_lineage(state, candidate.episode_return)
            self._completion_events.append(
                CompletionEvent(
                    runs=candidate.runs,
                    episode_return=candidate.episode_return,
                    progress=candidate.progress,
                    improved=improved,
                )
            )
            return
        previous = self._best_incomplete
        if previous is None or (
            progress,
            state.episode_return,
            -state.program_steps,
        ) > (
            previous.progress,
            previous.episode_return,
            -previous.step_count,
        ):
            self._best_incomplete = GoExploreCandidate(
                runs=tuple(state.runs),
                episode_return=state.episode_return,
                progress=progress,
                completed=False,
            )

    def _lineage(self, state: _LaneState) -> tuple[Hashable, ...]:
        if not state.path_cell_keys:
            return ()
        keys: list[Hashable] = []
        seen: set[Hashable] = set()
        parent: Hashable | None = state.path_cell_keys[0]
        while parent is not None and parent not in seen:
            seen.add(parent)
            cell = self._archive.get(parent)
            if cell is None:
                break
            keys.append(parent)
            parent = cell.parent_key
        keys.reverse()
        for key in state.path_cell_keys[1:]:
            if key in self._archive and key not in seen:
                seen.add(key)
                keys.append(key)
        return tuple(keys)

    def _record_success_lineage(
        self,
        state: _LaneState,
        episode_return: float,
    ) -> None:
        lineage = self._lineage(state)
        for key in lineage:
            cell = self._archive[key]
            cell.best_success_return = max(
                episode_return,
                cell.best_success_return if cell.best_success_return is not None else -math.inf,
            )
            cell.success_selections = 0
        self._elite_success_keys = lineage

    def observe(
        self,
        rewards: Sequence[float],
        dones: Sequence[bool],
        cell_keys: Sequence[Hashable],
        records_by_lane: Mapping[int, object] | None = None,
        *,
        progresses: Sequence[float] | None = None,
    ) -> GoExploreObservation:
        rewards_array = np.asarray(rewards, dtype=np.float64)
        dones_array = np.asarray(dones, dtype=np.bool_)
        progress_array = np.asarray(
            np.zeros(self.n_envs) if progresses is None else progresses,
            dtype=np.float64,
        )
        if (
            rewards_array.shape != (self.n_envs,)
            or dones_array.shape != (self.n_envs,)
            or progress_array.shape != (self.n_envs,)
            or len(cell_keys) != self.n_envs
        ):
            raise ValueError("Go-Explore observations must contain one value per lane")
        self.global_step += self.n_envs
        self._pending = {}
        counts: dict[Hashable, int] = {}
        restart_mask = np.zeros(self.n_envs, dtype=np.bool_)
        records_by_lane = records_by_lane or {}
        for lane, state in enumerate(self._lanes):
            state.episode_return += float(rewards_array[lane])
            state.progress = max(state.progress, float(progress_array[lane]))
            completed, record_progress = self._record_facts(records_by_lane.get(lane))
            progress = max(state.progress, record_progress)
            self._consider_best(state, progress=progress, completed=completed)
            if dones_array[lane]:
                self.completed_episodes += 1
                self.successful_episodes += int(completed)
                restart_mask[lane] = True
                continue
            key = cell_keys[lane]
            counts[key] = counts.get(key, 0) + 1
            parent_key: Hashable | None = None
            if not state.path_cell_keys:
                state.path_cell_keys.append(key)
            elif state.path_cell_keys[-1] != key:
                parent_key = state.path_cell_keys[-1]
                state.path_cell_keys.append(key)
            elif len(state.path_cell_keys) >= 2:
                parent_key = state.path_cell_keys[-2]
            comparison = self._pending.get(key) or self._archive.get(key)
            if self._better(state.episode_return, state.program_steps, comparison):
                self._pending[key] = _PendingCell(
                    key=key,
                    lane=lane,
                    runs=tuple(state.runs),
                    episode_return=state.episode_return,
                    progress=progress,
                    step_count=state.program_steps,
                    parent_key=parent_key,
                    visits=0,
                )
            if state.steps_since_restart >= self.explore_steps:
                restart_mask[lane] = True
        for key, count in counts.items():
            cell = self._archive.get(key)
            if cell is not None:
                cell.visits += count
                self._update_weight(cell)
                self._archive_visit_count += count
            pending = self._pending.get(key)
            if pending is not None:
                pending.visits = count
        visits = sum(counts.values())
        new_cells = sum(key not in self._archive for key in self._pending)
        if visits:
            self._recent_batches.append((visits, new_cells))
            self._recent_visits += visits
            self._recent_new_cells += new_cells
            while (
                len(self._recent_batches) > 1
                and self._recent_visits - self._recent_batches[0][0] >= RECENT_CELL_VISIT_WINDOW
            ):
                old_visits, old_new = self._recent_batches.popleft()
                self._recent_visits -= old_visits
                self._recent_new_cells -= old_new
        archive_mask = np.zeros(self.n_envs, dtype=np.bool_)
        for pending in self._pending.values():
            archive_mask[pending.lane] = True
        return GoExploreObservation(archive_mask=archive_mask, restart_mask=restart_mask)

    def commit_archive(self, entry_ids: Sequence[str | None]) -> None:
        if len(entry_ids) != self.n_envs:
            raise ValueError("Go-Explore archive entries require one value per lane")
        for pending in self._pending.values():
            entry_id = entry_ids[pending.lane]
            if not entry_id:
                raise ValueError("Go-Explore selected archive entries cannot be empty")
            existing = self._archive.get(pending.key)
            if existing is None:
                cell = GoExploreCell(
                    key=pending.key,
                    entry_id=str(entry_id),
                    runs=pending.runs,
                    episode_return=pending.episode_return,
                    progress=pending.progress,
                    program_steps=pending.step_count,
                    parent_key=pending.parent_key,
                    visits=pending.visits,
                )
                self._archive[pending.key] = cell
                self._register(cell)
                self._archive_visit_count += pending.visits
            else:
                existing.entry_id = str(entry_id)
                existing.runs = pending.runs
                existing.episode_return = pending.episode_return
                existing.progress = pending.progress
                existing.program_steps = pending.step_count
                existing.parent_key = pending.parent_key
                existing.updates += 1
                self._archive_update_count += 1
        self._pending = {}

    def _select_cell(self, lane: int) -> GoExploreCell:
        rng = self._rngs[lane]
        if self._elite_success_keys and rng.random() < SUCCESS_GUIDED_RESTORE_PROBABILITY:
            cells = tuple(self._archive[key] for key in self._elite_success_keys)
            weights = np.asarray([1.0 / math.sqrt(1.0 + cell.success_selections) for cell in cells])
            cell = cells[int(rng.choice(len(cells), p=weights / weights.sum()))]
            cell.success_selections += 1
            self._success_guided_selection_count += 1
            return cell
        return self._selection_cells[self._selection_weights.sample(rng.random())]

    def restart(self, mask: Sequence[bool]) -> tuple[str | None, ...]:
        restart_mask = np.asarray(mask, dtype=np.bool_)
        if restart_mask.shape != (self.n_envs,):
            raise ValueError("Go-Explore restart mask must contain one value per lane")
        entry_ids: list[str | None] = [None] * self.n_envs
        for lane in np.flatnonzero(restart_mask):
            lane_index = int(lane)
            cell = self._select_cell(lane_index)
            cell.selections += 1
            self._update_weight(cell)
            self._archive_selection_count += 1
            entry_ids[lane_index] = cell.entry_id
            self._lanes[lane_index] = _LaneState(
                runs=list(cell.runs),
                episode_return=cell.episode_return,
                progress=cell.progress,
                program_steps=cell.step_count,
                path_cell_keys=[cell.key],
            )
        return tuple(entry_ids)

    def take_completion_events(self) -> tuple[CompletionEvent, ...]:
        events = tuple(self._completion_events)
        self._completion_events.clear()
        return events

    def best_candidate(self) -> GoExploreCandidate | None:
        return self._best_success or self._best_incomplete

    @staticmethod
    def _key_document(key: Hashable | None) -> str | None:
        if key is None:
            return None
        if not isinstance(key, bytes):
            raise TypeError("durable Go-Explore state requires byte cell keys")
        return key.hex()

    @staticmethod
    def _key_from_document(value: object) -> bytes | None:
        if value is None:
            return None
        encoded = str(value)
        try:
            return bytes.fromhex(encoded)
        except ValueError as exc:
            raise ValueError("durable Go-Explore state contains an invalid cell key") from exc

    @staticmethod
    def _runs_document(runs: Sequence[ActionRun]) -> list[dict[str, int]]:
        return [{"action": int(run.action), "duration": int(run.duration)} for run in runs]

    @staticmethod
    def _runs_from_document(value: object) -> tuple[ActionRun, ...]:
        if isinstance(value, str | bytes) or not isinstance(value, Sequence):
            raise ValueError("durable Go-Explore action runs must be a sequence")
        return canonicalize_action_runs(
            tuple(ActionRun(int(dict(row)["action"]), int(dict(row)["duration"])) for row in value)
        )

    @classmethod
    def _candidate_document(
        cls,
        candidate: GoExploreCandidate | None,
    ) -> dict[str, object] | None:
        if candidate is None:
            return None
        return {
            "runs": cls._runs_document(candidate.runs),
            "episode_return": candidate.episode_return,
            "progress": candidate.progress,
            "completed": candidate.completed,
        }

    @classmethod
    def _candidate_from_document(
        cls,
        value: object,
    ) -> GoExploreCandidate | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ValueError("durable Go-Explore candidate must be an object")
        return GoExploreCandidate(
            runs=cls._runs_from_document(value["runs"]),
            episode_return=float(value["episode_return"]),
            progress=float(value["progress"]),
            completed=bool(value["completed"]),
        )

    def state_document(self, lane_entry_ids: Sequence[str | None]) -> dict[str, object]:
        if len(lane_entry_ids) != self.n_envs or any(not entry_id for entry_id in lane_entry_ids):
            raise ValueError("Go-Explore recovery requires one archive entry per lane")
        if self._pending or self._completion_events:
            raise RuntimeError("Go-Explore recovery state must be written at a stable boundary")
        cells = []
        for cell in self._selection_cells:
            cells.append(
                {
                    "key": self._key_document(cell.key),
                    "entry_id": cell.entry_id,
                    "runs": self._runs_document(cell.runs),
                    "episode_return": cell.episode_return,
                    "progress": cell.progress,
                    "program_steps": cell.program_steps,
                    "parent_key": self._key_document(cell.parent_key),
                    "best_success_return": cell.best_success_return,
                    "success_selections": cell.success_selections,
                    "visits": cell.visits,
                    "selections": cell.selections,
                    "updates": cell.updates,
                }
            )
        lanes = [
            {
                "runs": self._runs_document(state.runs),
                "episode_return": state.episode_return,
                "progress": state.progress,
                "program_steps": state.program_steps,
                "steps_since_restart": state.steps_since_restart,
                "path_cell_keys": [self._key_document(key) for key in state.path_cell_keys],
                "exploration_action": state.exploration_action,
                "exploration_remaining": state.exploration_remaining,
                "entry_id": str(lane_entry_ids[lane]),
            }
            for lane, state in enumerate(self._lanes)
        ]
        return {
            "semantic_id": GO_EXPLORE_STATE_SEMANTIC_ID,
            "schema_version": 1,
            "configuration": {
                "n_envs": self.n_envs,
                "action_names": list(self.action_names),
                "fallback_action": self.fallback_action,
                "explore_steps": self.explore_steps,
                "run_duration_mean": self.run_duration_mean,
                "run_duration_max": self.run_duration_max,
            },
            "global_step": self.global_step,
            "completed_episodes": self.completed_episodes,
            "successful_episodes": self.successful_episodes,
            "improvement_count": self.improvement_count,
            "first_success_return": self.first_success_return,
            "archive_selection_count": self._archive_selection_count,
            "archive_visit_count": self._archive_visit_count,
            "archive_update_count": self._archive_update_count,
            "recent_batches": [list(item) for item in self._recent_batches],
            "recent_visits": self._recent_visits,
            "recent_new_cells": self._recent_new_cells,
            "elite_success_keys": [self._key_document(key) for key in self._elite_success_keys],
            "success_guided_selection_count": self._success_guided_selection_count,
            "cells": cells,
            "lanes": lanes,
            "rng_states": [rng.bit_generator.state for rng in self._rngs],
            "best_incomplete": self._candidate_document(self._best_incomplete),
            "best_success": self._candidate_document(self._best_success),
        }

    def restore_state(self, value: Mapping[str, object]) -> tuple[str, ...]:
        if value.get("semantic_id") != GO_EXPLORE_STATE_SEMANTIC_ID:
            raise ValueError("durable Go-Explore state has an unsupported semantic_id")
        if int(value.get("schema_version", 0)) != 1:
            raise ValueError("durable Go-Explore state has an unsupported schema_version")
        configuration = value.get("configuration")
        if not isinstance(configuration, Mapping):
            raise ValueError("durable Go-Explore state has no configuration")
        expected_configuration = {
            "n_envs": self.n_envs,
            "action_names": list(self.action_names),
            "fallback_action": self.fallback_action,
            "explore_steps": self.explore_steps,
            "run_duration_mean": self.run_duration_mean,
            "run_duration_max": self.run_duration_max,
        }
        if dict(configuration) != expected_configuration:
            raise ValueError("durable Go-Explore state configuration mismatch")
        raw_cells = value.get("cells")
        raw_lanes = value.get("lanes")
        raw_rng_states = value.get("rng_states")
        if (
            isinstance(raw_cells, str | bytes)
            or not isinstance(raw_cells, Sequence)
            or isinstance(raw_lanes, str | bytes)
            or not isinstance(raw_lanes, Sequence)
            or len(raw_lanes) != self.n_envs
            or isinstance(raw_rng_states, str | bytes)
            or not isinstance(raw_rng_states, Sequence)
            or len(raw_rng_states) != self.n_envs
        ):
            raise ValueError("durable Go-Explore state has invalid vector dimensions")
        self._archive = {}
        self._selection_cells = []
        self._selection_weights = _SelectionWeightTree()
        for raw_cell in raw_cells:
            if not isinstance(raw_cell, Mapping):
                raise ValueError("durable Go-Explore cell must be an object")
            key = self._key_from_document(raw_cell["key"])
            assert key is not None
            if key in self._archive:
                raise ValueError("durable Go-Explore state contains duplicate cells")
            cell = GoExploreCell(
                key=key,
                entry_id=str(raw_cell["entry_id"]),
                runs=self._runs_from_document(raw_cell["runs"]),
                episode_return=float(raw_cell["episode_return"]),
                progress=float(raw_cell["progress"]),
                program_steps=int(raw_cell["program_steps"]),
                parent_key=self._key_from_document(raw_cell.get("parent_key")),
                best_success_return=(
                    None
                    if raw_cell.get("best_success_return") is None
                    else float(raw_cell["best_success_return"])
                ),
                success_selections=int(raw_cell["success_selections"]),
                visits=int(raw_cell["visits"]),
                selections=int(raw_cell["selections"]),
                updates=int(raw_cell["updates"]),
            )
            self._archive[key] = cell
            self._register(cell)
        self._lanes = []
        lane_entry_ids: list[str] = []
        for raw_lane in raw_lanes:
            if not isinstance(raw_lane, Mapping):
                raise ValueError("durable Go-Explore lane must be an object")
            path_keys = [self._key_from_document(key) for key in raw_lane["path_cell_keys"]]
            if any(key is None for key in path_keys):
                raise ValueError("durable Go-Explore lane path contains a null key")
            self._lanes.append(
                _LaneState(
                    runs=list(self._runs_from_document(raw_lane["runs"])),
                    episode_return=float(raw_lane["episode_return"]),
                    progress=float(raw_lane["progress"]),
                    program_steps=int(raw_lane["program_steps"]),
                    steps_since_restart=int(raw_lane["steps_since_restart"]),
                    path_cell_keys=list(path_keys),
                    exploration_action=int(raw_lane["exploration_action"]),
                    exploration_remaining=int(raw_lane["exploration_remaining"]),
                )
            )
            lane_entry_ids.append(str(raw_lane["entry_id"]))
        for rng, raw_state in zip(self._rngs, raw_rng_states, strict=True):
            if not isinstance(raw_state, Mapping):
                raise ValueError("durable Go-Explore RNG state must be an object")
            rng.bit_generator.state = dict(raw_state)
        self.global_step = int(value["global_step"])
        self.completed_episodes = int(value["completed_episodes"])
        self.successful_episodes = int(value["successful_episodes"])
        self.improvement_count = int(value["improvement_count"])
        self.first_success_return = (
            None
            if value.get("first_success_return") is None
            else float(value["first_success_return"])
        )
        self._archive_selection_count = int(value["archive_selection_count"])
        self._archive_visit_count = int(value["archive_visit_count"])
        self._archive_update_count = int(value["archive_update_count"])
        self._recent_batches = deque((int(row[0]), int(row[1])) for row in value["recent_batches"])
        self._recent_visits = int(value["recent_visits"])
        self._recent_new_cells = int(value["recent_new_cells"])
        elite = tuple(self._key_from_document(key) for key in value["elite_success_keys"])
        if any(key is None or key not in self._archive for key in elite):
            raise ValueError("durable Go-Explore success lineage references an unknown cell")
        self._elite_success_keys = elite
        self._success_guided_selection_count = int(value["success_guided_selection_count"])
        self._pending = {}
        self._completion_events = []
        self._best_incomplete = self._candidate_from_document(value["best_incomplete"])
        self._best_success = self._candidate_from_document(value["best_success"])
        return tuple(lane_entry_ids)

    def policy(self) -> ActionProgramPolicy:
        candidate = self.best_candidate()
        return ActionProgramPolicy(
            action_names=self.action_names,
            action_runs=() if candidate is None else candidate.runs,
            fallback_action=self.fallback_action,
        )
