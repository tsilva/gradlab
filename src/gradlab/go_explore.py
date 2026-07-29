"""Deterministic Go-Explore trajectory discovery over gradlab state-archive entries."""

from __future__ import annotations

import hashlib
import math
from collections import deque
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from gradlab.action_program import ActionProgramPolicy, ActionRun, canonicalize_action_runs
from gradlab.cell_graph import (
    CellGraphEdge,
    CellGraphNode,
    CellGraphPolicy,
    route_edge_id,
    route_node_id,
    slice_action_runs,
)
from gradlab.json_utils import canonical_json_bytes
from gradlab.task_kernels import Outcome


RECENT_CELL_VISIT_WINDOW = 10_000
GO_EXPLORE_STATE_SEMANTIC_ID = "go-explore-state-v1"
GO_EXPLORE_STATE_SCHEMA_VERSION = 4


@dataclass(frozen=True)
class RoutePoint:
    key: Hashable
    step: int


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
    initial_seed: int | None = None
    route_points: tuple[RoutePoint, ...] = ()

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
    initial_seed: int | None = None
    parent_key: Hashable | None = None
    route_points: tuple[RoutePoint, ...] = ()
    best_success_return: float | None = None
    progress_selections: int = 0
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
    initial_seed: int | None
    parent_key: Hashable | None
    route_points: tuple[RoutePoint, ...]
    visits: int


@dataclass
class _LaneState:
    runs: list[ActionRun] = field(default_factory=list)
    episode_return: float = 0.0
    progress: float = 0.0
    program_steps: int = 0
    initial_seed: int | None = None
    steps_since_restart: int = 0
    path_cell_keys: list[Hashable] = field(default_factory=list)
    route_points: list[RoutePoint] = field(default_factory=list)
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
        progress_guided_restore_probability: float = 0.5,
        success_guided_restore_probability: float = 0.5,
    ) -> None:
        if n_envs < 1 or explore_steps < 1:
            raise ValueError("Go-Explore environment and exploration counts must be positive")
        if run_duration_mean < 1.0 or run_duration_max < 1:
            raise ValueError("Go-Explore run durations must be at least one")
        for label, probability in (
            ("progress-guided", progress_guided_restore_probability),
            ("success-guided", success_guided_restore_probability),
        ):
            if (
                not isinstance(probability, int | float)
                or isinstance(probability, bool)
                or not math.isfinite(float(probability))
                or not 0.0 <= float(probability) <= 1.0
            ):
                raise ValueError(f"Go-Explore {label} restore probability must be in [0, 1]")
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
        self.progress_guided_restore_probability = float(progress_guided_restore_probability)
        self.success_guided_restore_probability = float(success_guided_restore_probability)
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
        self._elite_progress_keys: tuple[Hashable, ...] = ()
        self._progress_guided_selection_count = 0
        self._pending_progress_lineage_lane: int | None = None
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
        self._best_incomplete_by_seed: dict[int | None, GoExploreCandidate] = {}
        self._best_success_by_seed: dict[int | None, GoExploreCandidate] = {}
        self._completion_events: list[CompletionEvent] = []
        self._initial_roots: dict[int, tuple[Hashable, str]] = {}
        self._legacy_state = False

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
    def progress_guided_cell_count(self) -> int:
        return len(self._elite_progress_keys)

    @property
    def progress_guided_selection_count(self) -> int:
        return self._progress_guided_selection_count

    @property
    def progress_guided_selection_rate(self) -> float:
        return (
            self._progress_guided_selection_count / self._archive_selection_count
            if self._archive_selection_count
            else 0.0
        )

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
        initial_seeds: Sequence[int | None] | None = None,
    ) -> None:
        if len(cell_keys) != self.n_envs or len(entry_ids) != self.n_envs:
            raise ValueError("Go-Explore initialization requires one value per lane")
        seeds = (
            tuple(None for _ in range(self.n_envs))
            if initial_seeds is None
            else tuple(initial_seeds)
        )
        if len(seeds) != self.n_envs:
            raise ValueError("Go-Explore initialization requires one seed per lane")
        for lane, (key, entry_id, initial_seed) in enumerate(
            zip(cell_keys, entry_ids, seeds, strict=True)
        ):
            if not entry_id:
                raise ValueError("Go-Explore initialization entries cannot be empty")
            if initial_seed is not None and int(initial_seed) < 0:
                raise ValueError("Go-Explore initial seeds must be non-negative")
            cell = self._archive.get(key)
            if cell is None:
                cell = GoExploreCell(
                    key=key,
                    entry_id=str(entry_id),
                    runs=(),
                    episode_return=0.0,
                    progress=0.0,
                    initial_seed=None if initial_seed is None else int(initial_seed),
                    route_points=(RoutePoint(key, 0),),
                    visits=1,
                )
                self._archive[key] = cell
                self._register(cell)
            else:
                cell.visits += 1
                self._update_weight(cell)
            self._archive_visit_count += 1
            self._lanes[lane] = _LaneState(
                initial_seed=None if initial_seed is None else int(initial_seed),
                path_cell_keys=[key],
                route_points=[RoutePoint(key, 0)],
            )
            if initial_seed is not None:
                self._initial_roots[int(initial_seed)] = (key, str(entry_id))

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
    def _record_completed(record: object | None) -> bool:
        if record is None:
            return False
        return getattr(record, "outcome", Outcome.NEUTRAL) == Outcome.SUCCESS

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
    ) -> bool:
        if completed:
            candidate = GoExploreCandidate(
                runs=tuple(state.runs),
                episode_return=state.episode_return,
                progress=progress,
                completed=True,
                initial_seed=state.initial_seed,
                route_points=tuple(state.route_points),
            )
            previous_for_seed = self._best_success_by_seed.get(candidate.initial_seed)
            if previous_for_seed is None or (
                candidate.episode_return,
                -candidate.step_count,
            ) > (
                previous_for_seed.episode_return,
                -previous_for_seed.step_count,
            ):
                self._best_success_by_seed[candidate.initial_seed] = candidate
            previous = self._best_success
            improved = previous is None or (
                candidate.episode_return,
                -candidate.step_count,
            ) > (
                previous.episode_return,
                -previous.step_count,
            )
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
            return False
        previous = self._best_incomplete
        previous_for_seed = self._best_incomplete_by_seed.get(state.initial_seed)
        if previous_for_seed is None or (
            progress,
            state.episode_return,
            -state.program_steps,
        ) > (
            previous_for_seed.progress,
            previous_for_seed.episode_return,
            -previous_for_seed.step_count,
        ):
            self._best_incomplete_by_seed[state.initial_seed] = GoExploreCandidate(
                runs=tuple(state.runs),
                episode_return=state.episode_return,
                progress=progress,
                completed=False,
                initial_seed=state.initial_seed,
                route_points=tuple(state.route_points),
            )
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
                initial_seed=state.initial_seed,
                route_points=tuple(state.route_points),
            )
            self._record_progress_lineage(state)
            return True
        return False

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

    def _record_progress_lineage(self, state: _LaneState) -> None:
        lineage = self._lineage(state)
        for key in lineage:
            self._archive[key].progress_selections = 0
        self._elite_progress_keys = lineage

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
        self._pending_progress_lineage_lane = None
        counts: dict[Hashable, int] = {}
        restart_mask = np.zeros(self.n_envs, dtype=np.bool_)
        records_by_lane = records_by_lane or {}
        for lane, state in enumerate(self._lanes):
            state.episode_return += float(rewards_array[lane])
            state.progress = max(state.progress, float(progress_array[lane]))
            completed = self._record_completed(records_by_lane.get(lane))
            progress = state.progress
            if dones_array[lane]:
                self._consider_best(state, progress=progress, completed=completed)
                self.completed_episodes += 1
                self.successful_episodes += int(completed)
                restart_mask[lane] = True
                continue
            key = cell_keys[lane]
            counts[key] = counts.get(key, 0) + 1
            parent_key: Hashable | None = None
            if not state.path_cell_keys:
                state.path_cell_keys.append(key)
                state.route_points.append(RoutePoint(key, state.program_steps))
            elif state.path_cell_keys[-1] != key:
                parent_key = state.path_cell_keys[-1]
                state.path_cell_keys.append(key)
                state.route_points.append(RoutePoint(key, state.program_steps))
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
                    initial_seed=state.initial_seed,
                    parent_key=parent_key,
                    route_points=tuple(state.route_points),
                    visits=0,
                )
            if self._consider_best(state, progress=progress, completed=completed):
                if key not in self._archive:
                    self._pending_progress_lineage_lane = lane
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
                    initial_seed=pending.initial_seed,
                    parent_key=pending.parent_key,
                    route_points=pending.route_points,
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
                existing.initial_seed = pending.initial_seed
                existing.parent_key = pending.parent_key
                existing.route_points = pending.route_points
                existing.updates += 1
                self._archive_update_count += 1
        if self._pending_progress_lineage_lane is not None:
            self._record_progress_lineage(self._lanes[self._pending_progress_lineage_lane])
            self._pending_progress_lineage_lane = None
        self._pending = {}

    @staticmethod
    def _select_guided_cell(
        rng: np.random.Generator,
        cells: Sequence[GoExploreCell],
        *,
        counter: str,
    ) -> GoExploreCell:
        weights = np.asarray([1.0 / math.sqrt(1.0 + int(getattr(cell, counter))) for cell in cells])
        cell = cells[int(rng.choice(len(cells), p=weights / weights.sum()))]
        setattr(cell, counter, int(getattr(cell, counter)) + 1)
        return cell

    def _select_cell(self, lane: int) -> GoExploreCell:
        rng = self._rngs[lane]
        if (
            self._best_success is None
            and self._elite_progress_keys
            and rng.random() < self.progress_guided_restore_probability
        ):
            cells = tuple(self._archive[key] for key in self._elite_progress_keys)
            cell = self._select_guided_cell(
                rng,
                cells,
                counter="progress_selections",
            )
            self._progress_guided_selection_count += 1
            return cell
        if self._elite_success_keys and rng.random() < self.success_guided_restore_probability:
            cells = tuple(self._archive[key] for key in self._elite_success_keys)
            cell = self._select_guided_cell(
                rng,
                cells,
                counter="success_selections",
            )
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
                initial_seed=cell.initial_seed,
                path_cell_keys=[cell.key],
                route_points=list(cell.route_points)
                or [RoutePoint(cell.key, cell.step_count)],
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
    def _route_document(
        cls,
        route: Sequence[RoutePoint],
    ) -> list[dict[str, object]]:
        return [
            {"key": cls._key_document(point.key), "step": int(point.step)}
            for point in route
        ]

    @classmethod
    def _route_from_document(
        cls,
        value: object,
    ) -> tuple[RoutePoint, ...]:
        if isinstance(value, str | bytes) or not isinstance(value, Sequence):
            raise ValueError("durable Go-Explore route must be a sequence")
        route: list[RoutePoint] = []
        previous_step = -1
        for raw_point in value:
            if not isinstance(raw_point, Mapping):
                raise ValueError("durable Go-Explore route point must be an object")
            key = cls._key_from_document(raw_point.get("key"))
            step = int(raw_point.get("step", -1))
            if key is None or step < 0 or step <= previous_step:
                raise ValueError("durable Go-Explore route point is invalid")
            route.append(RoutePoint(key, step))
            previous_step = step
        return tuple(route)

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
            "initial_seed": candidate.initial_seed,
            "route": cls._route_document(candidate.route_points),
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
            initial_seed=(
                None if value.get("initial_seed") is None else int(value["initial_seed"])
            ),
            route_points=cls._route_from_document(value.get("route", ())),
        )

    @classmethod
    def _candidate_map_from_document(
        cls,
        value: object,
        *,
        fallback: GoExploreCandidate | None,
    ) -> dict[int | None, GoExploreCandidate]:
        if value is None:
            return {} if fallback is None else {fallback.initial_seed: fallback}
        if isinstance(value, str | bytes) or not isinstance(value, Sequence):
            raise ValueError("durable Go-Explore per-seed candidates must be a sequence")
        candidates: dict[int | None, GoExploreCandidate] = {}
        for raw_item in value:
            if not isinstance(raw_item, Mapping):
                raise ValueError("durable Go-Explore per-seed candidate must be an object")
            seed = None if raw_item.get("seed") is None else int(raw_item["seed"])
            candidate = cls._candidate_from_document(raw_item.get("candidate"))
            if (
                candidate is None
                or candidate.initial_seed != seed
                or seed in candidates
            ):
                raise ValueError("durable Go-Explore per-seed candidate is invalid")
            candidates[seed] = candidate
        return candidates

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
                    "initial_seed": cell.initial_seed,
                    "parent_key": self._key_document(cell.parent_key),
                    "route": self._route_document(cell.route_points),
                    "best_success_return": cell.best_success_return,
                    "progress_selections": cell.progress_selections,
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
                "initial_seed": state.initial_seed,
                "steps_since_restart": state.steps_since_restart,
                "path_cell_keys": [self._key_document(key) for key in state.path_cell_keys],
                "route": self._route_document(state.route_points),
                "exploration_action": state.exploration_action,
                "exploration_remaining": state.exploration_remaining,
                "entry_id": str(lane_entry_ids[lane]),
            }
            for lane, state in enumerate(self._lanes)
        ]
        return {
            "semantic_id": GO_EXPLORE_STATE_SEMANTIC_ID,
            "schema_version": GO_EXPLORE_STATE_SCHEMA_VERSION,
            "configuration": {
                "n_envs": self.n_envs,
                "action_names": list(self.action_names),
                "fallback_action": self.fallback_action,
                "explore_steps": self.explore_steps,
                "run_duration_mean": self.run_duration_mean,
                "run_duration_max": self.run_duration_max,
                "progress_guided_restore_probability": (self.progress_guided_restore_probability),
                "success_guided_restore_probability": (self.success_guided_restore_probability),
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
            "elite_progress_keys": [self._key_document(key) for key in self._elite_progress_keys],
            "progress_guided_selection_count": self._progress_guided_selection_count,
            "elite_success_keys": [self._key_document(key) for key in self._elite_success_keys],
            "success_guided_selection_count": self._success_guided_selection_count,
            "cells": cells,
            "lanes": lanes,
            "rng_states": [rng.bit_generator.state for rng in self._rngs],
            "best_incomplete": self._candidate_document(self._best_incomplete),
            "best_success": self._candidate_document(self._best_success),
            "best_incomplete_by_seed": [
                {
                    "seed": seed,
                    "candidate": self._candidate_document(candidate),
                }
                for seed, candidate in sorted(
                    self._best_incomplete_by_seed.items(),
                    key=lambda item: (-1 if item[0] is None else item[0]),
                )
            ],
            "best_success_by_seed": [
                {
                    "seed": seed,
                    "candidate": self._candidate_document(candidate),
                }
                for seed, candidate in sorted(
                    self._best_success_by_seed.items(),
                    key=lambda item: (-1 if item[0] is None else item[0]),
                )
            ],
            "initial_roots": [
                {
                    "seed": seed,
                    "key": self._key_document(key),
                    "entry_id": entry_id,
                }
                for seed, (key, entry_id) in sorted(self._initial_roots.items())
            ],
        }

    def restore_state(self, value: Mapping[str, object]) -> tuple[str, ...]:
        if value.get("semantic_id") != GO_EXPLORE_STATE_SEMANTIC_ID:
            raise ValueError("durable Go-Explore state has an unsupported semantic_id")
        schema_version = int(value.get("schema_version", 0))
        if schema_version not in {3, GO_EXPLORE_STATE_SCHEMA_VERSION}:
            raise ValueError("durable Go-Explore state has an unsupported schema_version")
        self._legacy_state = schema_version == 3
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
            "progress_guided_restore_probability": (self.progress_guided_restore_probability),
            "success_guided_restore_probability": (self.success_guided_restore_probability),
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
                initial_seed=(
                    None if raw_cell.get("initial_seed") is None else int(raw_cell["initial_seed"])
                ),
                parent_key=self._key_from_document(raw_cell.get("parent_key")),
                route_points=(
                    self._route_from_document(raw_cell.get("route", ()))
                    if schema_version >= 4
                    else ()
                ),
                best_success_return=(
                    None
                    if raw_cell.get("best_success_return") is None
                    else float(raw_cell["best_success_return"])
                ),
                progress_selections=int(raw_cell["progress_selections"]),
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
                    initial_seed=(
                        None
                        if raw_lane.get("initial_seed") is None
                        else int(raw_lane["initial_seed"])
                    ),
                    steps_since_restart=int(raw_lane["steps_since_restart"]),
                    path_cell_keys=list(path_keys),
                    route_points=(
                        list(self._route_from_document(raw_lane.get("route", ())))
                        if schema_version >= 4
                        else []
                    ),
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
        progress_elite = tuple(self._key_from_document(key) for key in value["elite_progress_keys"])
        if any(key is None or key not in self._archive for key in progress_elite):
            raise ValueError("durable Go-Explore progress lineage references an unknown cell")
        self._elite_progress_keys = progress_elite
        self._progress_guided_selection_count = int(value["progress_guided_selection_count"])
        self._pending_progress_lineage_lane = None
        elite = tuple(self._key_from_document(key) for key in value["elite_success_keys"])
        if any(key is None or key not in self._archive for key in elite):
            raise ValueError("durable Go-Explore success lineage references an unknown cell")
        self._elite_success_keys = elite
        self._success_guided_selection_count = int(value["success_guided_selection_count"])
        self._pending = {}
        self._completion_events = []
        self._best_incomplete = self._candidate_from_document(value["best_incomplete"])
        self._best_success = self._candidate_from_document(value["best_success"])
        self._best_incomplete_by_seed = self._candidate_map_from_document(
            value.get("best_incomplete_by_seed"),
            fallback=self._best_incomplete,
        )
        self._best_success_by_seed = self._candidate_map_from_document(
            value.get("best_success_by_seed"),
            fallback=self._best_success,
        )
        self._initial_roots = {}
        if schema_version >= 4:
            raw_roots = value.get("initial_roots", ())
            if isinstance(raw_roots, str | bytes) or not isinstance(raw_roots, Sequence):
                raise ValueError("durable Go-Explore initial roots must be a sequence")
            for raw_root in raw_roots:
                if not isinstance(raw_root, Mapping):
                    raise ValueError("durable Go-Explore initial root must be an object")
                seed = int(raw_root["seed"])
                key = self._key_from_document(raw_root["key"])
                if key is None or seed in self._initial_roots:
                    raise ValueError("durable Go-Explore initial root is invalid")
                self._initial_roots[seed] = (key, str(raw_root["entry_id"]))
        return tuple(lane_entry_ids)

    def policy(self) -> ActionProgramPolicy:
        candidate = self.best_candidate()
        return ActionProgramPolicy(
            action_names=self.action_names,
            action_runs=() if candidate is None else candidate.runs,
            fallback_action=self.fallback_action,
            initial_seed=None if candidate is None else candidate.initial_seed,
        )

    @property
    def legacy_state(self) -> bool:
        return self._legacy_state

    def graph_snapshot_entry_ids(self) -> tuple[str, ...]:
        entry_ids = {
            entry_id for _key, entry_id in self._initial_roots.values()
        }
        candidate = self.best_candidate()
        if candidate is not None:
            for point in candidate.route_points:
                cell = self._archive.get(point.key)
                if cell is not None:
                    entry_ids.add(cell.entry_id)
        return tuple(sorted(entry_ids))

    def cell_graph_policy(
        self,
        *,
        detector: Mapping[str, Any],
        snapshot_mode: str = "none",
        snapshot_records: Mapping[
            str,
            tuple[Mapping[str, Any], bytes],
        ]
        | None = None,
    ) -> CellGraphPolicy:
        """Build the compact executable graph for the current best route."""

        if self._legacy_state:
            raise ValueError(
                "legacy Go-Explore recovery state cannot prove a portable cell graph"
            )
        candidate = self.best_candidate()
        records = dict(snapshot_records or {})
        snapshot_entries: dict[str, Mapping[str, Any]] = {}
        snapshot_payloads: dict[str, bytes] = {}
        nodes_by_id: dict[str, CellGraphNode] = {}
        edges_by_id: dict[str, CellGraphEdge] = {}
        roots: dict[int, str] = {}

        def snapshot_for_entry(
            entry_id: str | None,
        ) -> str | None:
            if snapshot_mode != "retained" or entry_id is None:
                return None
            record = records.get(entry_id)
            if record is None:
                return None
            entry, payload = record
            provider_snapshot = entry.get("provider_snapshot")
            if not isinstance(provider_snapshot, Mapping):
                raise ValueError("Go-Explore snapshot entry has no provider snapshot")
            ref = provider_snapshot.get("ref")
            if not isinstance(ref, Mapping):
                raise ValueError("Go-Explore snapshot entry has no snapshot ref")
            digest = str(ref.get("blob_sha256") or "")
            if not digest:
                raise ValueError("Go-Explore snapshot entry has no blob digest")
            snapshot_entries[entry_id] = dict(entry)
            snapshot_payloads[digest] = bytes(payload)
            return entry_id

        def route_entry_id(
            point: RoutePoint,
            prefix: tuple[ActionRun, ...],
            seed: int | None,
        ) -> str | None:
            if point.step == 0 and seed is not None:
                root = self._initial_roots.get(seed)
                if root is not None and root[0] == point.key:
                    return root[1]
            cell = self._archive.get(point.key)
            if (
                cell is not None
                and cell.initial_seed == seed
                and cell.step_count == point.step
                and cell.runs == prefix
            ):
                return cell.entry_id
            return None

        if candidate is None or not candidate.route_points:
            if not self._initial_roots:
                raise RuntimeError("Go-Explore has no route roots to export")
            default_seed = min(self._initial_roots)
            key, entry_id = self._initial_roots[default_seed]
            node_id = route_node_id(seed=default_seed, cell_key=key, prefix_runs=())
            nodes_by_id[node_id] = CellGraphNode(
                node_id=node_id,
                cell_key=bytes(key),
                target_distance=0,
                initial_seed=default_seed,
                snapshot_entry_id=snapshot_for_entry(entry_id),
            )
            roots[default_seed] = node_id
            target_node_id = node_id
        else:
            seed = candidate.initial_seed
            route = candidate.route_points
            if route[0].step != 0:
                raise ValueError("Go-Explore best route does not begin at step zero")
            route_node_ids: list[str] = []
            terminal_distance = 1 if candidate.completed else 0
            for index, point in enumerate(route):
                prefix = slice_action_runs(candidate.runs, 0, point.step)
                node_id = route_node_id(
                    seed=seed,
                    cell_key=bytes(point.key),
                    prefix_runs=prefix,
                )
                route_node_ids.append(node_id)
                entry_id = route_entry_id(point, prefix, seed)
                nodes_by_id[node_id] = CellGraphNode(
                    node_id=node_id,
                    cell_key=bytes(point.key),
                    target_distance=len(route) - index - 1 + terminal_distance,
                    initial_seed=seed if index == 0 else None,
                    snapshot_entry_id=snapshot_for_entry(entry_id),
                )
            for index in range(len(route) - 1):
                segment = slice_action_runs(
                    candidate.runs,
                    route[index].step,
                    route[index + 1].step,
                )
                edge_id = route_edge_id(
                    source_id=route_node_ids[index],
                    target_id=route_node_ids[index + 1],
                    action_runs=segment,
                )
                edges_by_id[edge_id] = CellGraphEdge(
                    edge_id=edge_id,
                    source_id=route_node_ids[index],
                    target_id=route_node_ids[index + 1],
                    action_runs=segment,
                    successful_suffix=candidate.completed,
                )
            if candidate.completed:
                target_node_id = hashlib.sha256(
                    canonical_json_bytes(
                        {
                            "semantic_id": "cell-graph-outcome-v1",
                            "outcome": "success",
                            "seed": seed,
                            "program_steps": candidate.step_count,
                        }
                    )
                ).hexdigest()
                nodes_by_id[target_node_id] = CellGraphNode(
                    node_id=target_node_id,
                    cell_key=None,
                    target_distance=0,
                    outcome="success",
                )
                suffix = slice_action_runs(
                    candidate.runs,
                    route[-1].step,
                    candidate.step_count,
                )
                if not suffix:
                    raise ValueError("successful Go-Explore route has no terminal action suffix")
                edge_id = route_edge_id(
                    source_id=route_node_ids[-1],
                    target_id=target_node_id,
                    action_runs=suffix,
                )
                edges_by_id[edge_id] = CellGraphEdge(
                    edge_id=edge_id,
                    source_id=route_node_ids[-1],
                    target_id=target_node_id,
                    action_runs=suffix,
                    successful_suffix=True,
                )
            else:
                target_node_id = route_node_ids[-1]
            default_seed = seed
            if seed is not None:
                roots[seed] = route_node_ids[0]

        max_distance = max(node.target_distance for node in nodes_by_id.values())
        for root_seed, (root_key, entry_id) in sorted(self._initial_roots.items()):
            if root_seed in roots:
                continue
            node_id = route_node_id(
                seed=root_seed,
                cell_key=bytes(root_key),
                prefix_runs=(),
            )
            nodes_by_id[node_id] = CellGraphNode(
                node_id=node_id,
                cell_key=bytes(root_key),
                target_distance=max_distance + 1,
                initial_seed=root_seed,
                snapshot_entry_id=snapshot_for_entry(entry_id),
            )
            roots[root_seed] = node_id

        return CellGraphPolicy(
            action_names=self.action_names,
            fallback_action=self.fallback_action,
            detector=detector,
            nodes=tuple(nodes_by_id[key] for key in sorted(nodes_by_id)),
            edges=tuple(edges_by_id[key] for key in sorted(edges_by_id)),
            roots=roots,
            target_node_id=target_node_id,
            default_seed=default_seed,
            snapshot_mode=snapshot_mode,
            snapshot_entries=snapshot_entries,
            snapshot_payloads=snapshot_payloads,
        )
