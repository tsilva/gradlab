"""Exact, source-state training measurements, independent of archive admission."""

from __future__ import annotations

import itertools
import copy
import math
import uuid
import re
import json
from types import MappingProxyType
from collections.abc import Mapping
from collections import deque
from typing import Any

import numpy as np
from numba import njit

from gradlab.cells import (
    ArchiveCellConfig,
    ArchiveCellDetector,
    bucket_number,
    normalize_archive_cell_config,
)
from gradlab.json_utils import canonical_json_sha256
from gradlab.run_contracts import new_attempt_id, new_run_id


ORIGINS = ("normal", "archive", "search")
REPORTING_CELL_LIMIT = 256
HISTORY_PAGE_WINDOWS = 8
OCCUPANCY_TABLE = "train/occupancy/table"
WINDOW_FIELDS = (
    "run_id",
    "attempt_id",
    "segment",
    "cell_space_hash",
    "sequence",
    "start_step",
    "end_step",
    "window_transitions",
    "complete",
    "uncovered_interval",
)
ROW_FIELDS = (
    "cell",
    "buckets",
    "label",
    "origin",
    "count",
    "entries",
    "denominator",
    "fraction",
    "cumulative_count",
    "cumulative_entries",
    "cumulative_denominator",
)
TRACKING_COMBINATIONS = frozenset(
    (backend, provider)
    for backend in ("sb3.ppo", "sb3.a2c", "gradlab.ppo", "gradlab.go-explore")
    for provider in ("env-breakoutatari2600-turbo-native", "env-supermariobrosnes-turbo-emu")
)


def validate_tracking_runtime(config, *, backend_id):
    if (
        config.get("occupancy") is not None
        and (backend_id, config.get("env_provider")) not in TRACKING_COMBINATIONS
    ):
        raise ValueError("occupancy is unsupported for this backend/provider combination")


class OccupancyReporter:
    """Deliver runtime aggregates at learner logging boundaries, without network I/O."""

    def __init__(self, runtime, context, *, initial_step):
        self.runtime = runtime
        self.store = context.metric_store
        self.publish = context.wandb_enabled
        collector = runtime.occupancy
        if collector is not None:
            config = context.train_config
            collector.run_id = config.get("wandb_run_id") or collector.run_id
            collector.attempt_id = config.get("attempt_id") or collector.attempt_id
            previous_end = self.store.occupancy_covered_end(
                run_id=collector.run_id, cell_space_hash=collector.contract_hash
            )
            collector.restart(
                initial_step=int(initial_step), previous_end=min(previous_end, int(initial_step))
            )

    def flush(self, *, final=False):
        windows = self.runtime.drain_occupancy(final=final)
        for index, window in enumerate(windows):
            try:
                self.store.append_occupancy(window, publish=self.publish)
            except Exception:
                self.runtime.occupancy.requeue(windows[index:])
                raise

        from gradlab.curriculum_reporting import validate_distribution

        while self.runtime.curriculum_reports:
            step, report = self.runtime.curriculum_reports[0]
            self.store.enqueue_event(
                kind="curriculum_distribution",
                payload=validate_distribution(report),
                step=step,
                source="train",
                publish=self.publish,
            )
            del self.runtime.curriculum_reports[0]


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def validate_occupancy_window(value):
    """Validate exact aggregates at the durable transport boundary."""
    if not isinstance(value, Mapping) or set(value) != {*WINDOW_FIELDS, "schema_version", "rows"}:
        raise ValueError("invalid occupancy window fields")
    if value["schema_version"] != 1 or type(value["complete"]) is not bool:
        raise ValueError("invalid occupancy window schema")
    for name in ("sequence", "start_step", "end_step", "window_transitions"):
        if type(value[name]) is not int or value[name] < 0:
            raise ValueError(f"invalid occupancy {name}")
    for name, pattern in (
        ("run_id", r"gradlab-[0-9a-f]{32}"),
        ("attempt_id", r"attempt-[0-9a-f]{16}"),
        ("segment", r"[0-9a-f]{32}"),
        ("cell_space_hash", r"[0-9a-f]{64}"),
    ):
        if not isinstance(value[name], str) or re.fullmatch(pattern, value[name]) is None:
            raise ValueError(f"invalid occupancy {name}")
    gap = value["uncovered_interval"]
    if gap is not None and (
        not isinstance(gap, (tuple, list))
        or len(gap) != 2
        or any(type(step) is not int for step in gap)
        or not 0 <= gap[0] < gap[1] <= value["start_step"]
    ):
        raise ValueError("invalid uncovered occupancy interval")
    size = value["end_step"] - value["start_step"]
    if not 0 < size <= value["window_transitions"] or (
        value["complete"] and size != value["window_transitions"]
    ):
        raise ValueError("occupancy window bounds disagree")
    rows = value["rows"]
    if (
        not isinstance(rows, (list, tuple))
        or not 4 <= len(rows) <= 4 * REPORTING_CELL_LIMIT
        or len(rows) % 4
    ):
        raise ValueError("occupancy rows exceed the bounded reporting domain")
    populations = {origin: {} for origin in (*ORIGINS, "combined")}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != set(ROW_FIELDS):
            raise ValueError("invalid occupancy row fields")
        for name in (
            "cell",
            "count",
            "entries",
            "denominator",
            "cumulative_count",
            "cumulative_entries",
            "cumulative_denominator",
        ):
            if type(row[name]) is not int or row[name] < 0:
                raise ValueError(f"invalid occupancy row {name}")
        if (
            not isinstance(row["label"], str)
            or not 0 < len(row["label"]) <= 1024
            or not isinstance(row["buckets"], (list, tuple))
            or not 1 <= len(row["buckets"]) <= 32
            or any(type(bucket) is not int for bucket in row["buckets"])
        ):
            raise ValueError("invalid occupancy cell definition")
        origin, cell = row["origin"], row["cell"]
        if origin not in populations or cell in populations[origin]:
            raise ValueError("duplicate or unknown occupancy population")
        if (
            not row["entries"] <= row["count"] <= row["denominator"]
            or not row["count"] <= row["cumulative_count"] <= row["cumulative_denominator"]
            or not row["entries"] <= row["cumulative_entries"] <= row["cumulative_count"]
        ):
            raise ValueError("occupancy counts disagree")
        expected = row["count"] / row["denominator"] if row["denominator"] else None
        if row["fraction"] != expected:
            raise ValueError("occupancy fraction disagrees with counts")
        populations[origin][cell] = row
    cells = set(range(len(rows) // 4))
    for origin, population in populations.items():
        if set(population) != cells:
            raise ValueError("occupancy must include every declared cell for every origin")
        total = sum(row["count"] for row in population.values())
        cumulative = sum(row["cumulative_count"] for row in population.values())
        if any(
            row["denominator"] != total or row["cumulative_denominator"] != cumulative
            for row in population.values()
        ):
            raise ValueError("occupancy population denominators disagree")
        if origin == "combined" and total != size:
            raise ValueError("occupancy counts must account for every transition")
    for cell in cells:
        combined = populations["combined"][cell]
        for name in ("count", "entries", "cumulative_count", "cumulative_entries"):
            if combined[name] != sum(populations[origin][cell][name] for origin in ORIGINS):
                raise ValueError("combined occupancy disagrees with origins")
        if any(
            (populations[origin][cell]["label"], populations[origin][cell]["buckets"])
            != (combined["label"], combined["buckets"])
            for origin in ORIGINS
        ):
            raise ValueError("occupancy cell labels disagree across origins")
    return _thaw(value)


def occupancy_table(window, *, page=None):
    import wandb

    windows = [validate_occupancy_window(item) for item in (page or [window])]
    if len(windows) > HISTORY_PAGE_WINDOWS:
        raise ValueError("occupancy history page exceeds its window bound")
    columns = (*WINDOW_FIELDS, *ROW_FIELDS, "dimension_0", "dimension_1")
    return wandb.Table(
        columns=list(columns),
        data=[
            [
                {
                    **item,
                    **row,
                    "uncovered_interval": (
                        json.dumps(item["uncovered_interval"])
                        if item["uncovered_interval"] is not None
                        else None
                    ),
                    "dimension_0": row["buckets"][0],
                    "dimension_1": row["buckets"][1] if len(row["buckets"]) > 1 else None,
                }[name]
                for name in columns
            ]
            for item in windows
            for row in item["rows"]
        ],
    )


def normalize_occupancy_config(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("occupancy must be an object")
    unknown = set(value) - {"cell", "domains", "units", "labels", "window_transitions"}
    if unknown:
        raise ValueError(f"unknown occupancy fields: {sorted(unknown)}")
    cell = normalize_archive_cell_config(value.get("cell"), label="occupancy.cell")
    domains = value.get("domains")
    if not isinstance(domains, list) or len(domains) != len(cell["dimensions"]):
        raise ValueError("occupancy requires one finite bucket domain per dimension")
    for domain in domains:
        if (
            not isinstance(domain, list)
            or not domain
            or any(type(item) is not int for item in domain)
            or domain != sorted(set(domain))
        ):
            raise ValueError("occupancy domains must contain sorted unique integer buckets")
    if math.prod(map(len, domains)) > REPORTING_CELL_LIMIT:
        raise ValueError(f"occupancy reporting supports at most {REPORTING_CELL_LIMIT} cells")
    units = value.get("units")
    if (
        not isinstance(units, list)
        or len(units) != len(domains)
        or any(not isinstance(unit, str) or not unit.strip() for unit in units)
    ):
        raise ValueError("occupancy requires semantic units for every dimension")
    window = value.get("window_transitions", 100_000)
    if type(window) is not int or window < 1:
        raise ValueError("occupancy.window_transitions must be a positive integer")
    labels = value.get("labels")
    if labels is not None and (
        not isinstance(labels, list)
        or len(labels) != math.prod(map(len, domains))
        or any(not isinstance(label, str) or not 0 < len(label) <= 1024 for label in labels)
    ):
        raise ValueError("occupancy.labels must name every cell in dimension order")
    if labels is not None and len(set(labels)) != len(labels):
        raise ValueError("occupancy labels must distinguish every declared cell")
    return {
        "cell": cell,
        "domains": domains,
        "units": units,
        "window_transitions": window,
        **({"labels": labels} if labels else {}),
    }


def resolve_cell_spaces(config: dict[str, Any]) -> None:
    spaces = config.get("cell_spaces") or {}
    if not isinstance(spaces, Mapping):
        raise ValueError("cell_spaces must be a named mapping")
    normalized_spaces = {}
    for name, space in spaces.items():
        if not isinstance(name, str) or not name or not isinstance(space, Mapping):
            raise ValueError("cell_spaces require nonempty names and definitions")
        # Fine search spaces may omit a reporting domain entirely.
        if set(space) - {"cell", "domains", "units", "labels"}:
            raise ValueError(f"unknown cell space fields in {name!r}")
        normalized_spaces[name] = {
            **space,
            "cell": normalize_archive_cell_config(space.get("cell"), label=f"cell_spaces.{name}"),
        }
    if spaces:
        config["cell_spaces"] = normalized_spaces
    occupancy = config.get("occupancy")
    if occupancy is not None:
        if not isinstance(occupancy, Mapping):
            raise ValueError("occupancy must be an object")
        if "space" in occupancy:
            name = occupancy["space"]
            if not isinstance(name, str) or name not in normalized_spaces:
                raise ValueError("occupancy references an unknown cell space")
            if set(occupancy) - {"space", "window_transitions"}:
                raise ValueError("named occupancy definitions cannot override their cell semantics")
            occupancy = {
                **normalized_spaces[name],
                **{key: val for key, val in occupancy.items() if key != "space"},
            }
        config["occupancy"] = normalize_occupancy_config(occupancy)
    task = config.get("task")
    reward = task.get("reward") if isinstance(task, Mapping) else None
    novelty = reward.get("cell_novelty") if isinstance(reward, Mapping) else None
    if (
        isinstance(task, Mapping)
        and isinstance(reward, Mapping)
        and isinstance(novelty, Mapping)
        and isinstance(novelty.get("cell"), str)
    ):
        name = novelty["cell"]
        if name not in normalized_spaces:
            raise ValueError("cell novelty references an unknown cell space")
        config["task"] = {
            **task,
            "reward": {
                **reward,
                "cell_novelty": {**novelty, "cell": normalized_spaces[name]["cell"]},
            },
        }
    archive = config.get("state_archive")
    if isinstance(archive, Mapping) and isinstance(archive.get("recorder"), Mapping):
        cell = archive["recorder"].get("cell")
        if isinstance(cell, str):
            if cell not in normalized_spaces:
                raise ValueError("state archive references an unknown cell space")
            config["state_archive"] = {
                **archive,
                "recorder": {**archive["recorder"], "cell": normalized_spaces[cell]["cell"]},
            }


@njit(cache=True)
def _assign_reporting(
    values, parameters, domains, domain_sizes, selected, assignments, pending, origins, origin
):
    resolved = assignments.copy()
    for lane in range(values.shape[0]):
        if not selected[lane]:
            continue
        cell = 0
        for column in range(values.shape[1]):
            bucket = bucket_number(
                values[lane, column],
                parameters[column, 0],
                parameters[column, 1],
                parameters[column, 2],
                parameters[column, 3],
            )
            size = domain_sizes[column]
            index = np.searchsorted(domains[column, :size], bucket)
            if index == size or domains[column, index] != bucket:
                raise ValueError("occupancy cell is outside its declared reporting domain")
            cell = cell * size + index
        resolved[lane] = cell
    for lane in range(len(assignments)):
        if selected[lane]:
            pending[lane] = pending[lane] or resolved[lane] != assignments[lane] or origin >= 0
            assignments[lane] = resolved[lane]
            if origin >= 0:
                origins[lane] = origin


@njit(cache=True)
def _collect_counts(assignments, origins, pending, counts, entries):
    for lane in range(len(assignments)):
        cell, origin = assignments[lane], origins[lane]
        counts[origin, cell] += 1
        if pending[lane]:
            entries[origin, cell] += 1
            pending[lane] = False


@njit(cache=True)
def _transition_counts(
    values,
    parameters,
    domains,
    domain_sizes,
    selected,
    assignments,
    pending,
    origins,
    counts,
    entries,
):
    _collect_counts(assignments, origins, pending, counts, entries)
    _assign_reporting(
        values, parameters, domains, domain_sizes, selected, assignments, pending, origins, -1
    )


class OccupancyCollector:
    def __init__(self, config: Mapping[str, Any], *, n_envs: int, environment: Any):
        self.config = copy.deepcopy(normalize_occupancy_config(config))
        self.detector = ArchiveCellDetector(
            ArchiveCellConfig.from_mapping(self.config["cell"], label="occupancy.cell")
        )
        self.n_envs = n_envs
        self.window_size = ((self.config["window_transitions"] + n_envs - 1) // n_envs) * n_envs
        self.domains = tuple(np.asarray(domain) for domain in self.config["domains"])
        self.domain_sizes = np.asarray([len(domain) for domain in self.domains], dtype=np.int64)
        self.domain_matrix = np.zeros((len(self.domains), max(self.domain_sizes)), dtype=np.int64)
        for index, domain in enumerate(self.domains):
            self.domain_matrix[index, : len(domain)] = domain
        self.all_lanes = np.ones(n_envs, dtype=np.bool_)
        self.cells = tuple(itertools.product(*self.config["domains"]))
        self.contract_hash = canonical_json_sha256(
            {
                "schema_version": 1,
                "environment": environment,
                "cell": self.config["cell"],
                "domains": self.config["domains"],
                "units": self.config["units"],
                "out_of_range": "reject",
            }
        )
        self.segment = uuid.uuid4().hex
        self.run_id = new_run_id()
        self.attempt_id = new_attempt_id()
        self.uncovered_interval = None
        self.step = 0
        self.start_step = 0
        self.sequence = 0
        self.assignments = np.zeros(n_envs, dtype=np.int64)
        self.origins = np.zeros(n_envs, dtype=np.int64)
        self.entry_pending = np.ones(n_envs, dtype=np.bool_)
        self.counts = np.zeros((3, len(self.cells)), dtype=np.int64)
        self.entries = np.zeros_like(self.counts)
        self.cumulative = np.zeros_like(self.counts)
        self.cumulative_entries = np.zeros_like(self.counts)
        self.completed: list[dict[str, Any]] = []
        self.recent: deque[np.ndarray] = deque(maxlen=0)

    def _columns(self, values, mask):
        if mask is not None:
            columns = np.empty((self.n_envs, len(self.detector.selectors)), dtype=np.float64)
            for index, selector in enumerate(self.detector.selectors):
                columns[mask, index] = np.asarray(values[selector])[mask]
            return columns
        columns = np.asarray([values[selector] for selector in self.detector.selectors]).T
        return columns.astype(np.float64) if columns.dtype.kind == "O" else columns

    def assign(self, values, *, mask=None, origin=None):
        columns = self._columns(values, mask)
        _assign_reporting(
            columns,
            self.detector.parameters,
            self.domain_matrix,
            self.domain_sizes,
            self.all_lanes if mask is None else mask,
            self.assignments,
            self.entry_pending,
            self.origins,
            -1 if origin is None else ORIGINS.index(origin),
        )

    def collect(self, values=None, *, mask=None):
        if values is None:
            _collect_counts(
                self.assignments, self.origins, self.entry_pending, self.counts, self.entries
            )
        else:
            columns = self._columns(values, mask)
            _transition_counts(
                columns,
                self.detector.parameters,
                self.domain_matrix,
                self.domain_sizes,
                self.all_lanes if mask is None else mask,
                self.assignments,
                self.entry_pending,
                self.origins,
                self.counts,
                self.entries,
            )
        self.step += self.n_envs
        if self.step - self.start_step == self.window_size:
            self._close(complete=True)

    def _close(self, *, complete):
        self.recent.append(self.counts.sum(axis=0))
        self.cumulative += self.counts
        self.cumulative_entries += self.entries
        rows = []
        labels = self.config.get("labels")
        for origin_index, origin in enumerate((*ORIGINS, "combined")):
            counts, entries, cumulative, cumulative_entries = (
                array.sum(axis=0) if origin == "combined" else array[origin_index]
                for array in (self.counts, self.entries, self.cumulative, self.cumulative_entries)
            )
            denominator = int(counts.sum())
            for index, cell in enumerate(self.cells):
                rows.append(
                    {
                        "cell": index,
                        "buckets": list(cell),
                        "label": labels[index] if labels else str(cell),
                        "origin": origin,
                        "count": int(counts[index]),
                        "entries": int(entries[index]),
                        "denominator": denominator,
                        "fraction": int(counts[index]) / denominator if denominator else None,
                        "cumulative_count": int(cumulative[index]),
                        "cumulative_entries": int(cumulative_entries[index]),
                        "cumulative_denominator": int(cumulative.sum()),
                    }
                )
        self.completed.append(
            {
                "schema_version": 1,
                "cell_space_hash": self.contract_hash,
                "run_id": self.run_id,
                "attempt_id": self.attempt_id,
                "uncovered_interval": self.uncovered_interval,
                "segment": self.segment,
                "sequence": self.sequence,
                "start_step": self.start_step,
                "end_step": self.step,
                "window_transitions": self.window_size,
                "complete": complete,
                "rows": rows,
            }
        )
        self.start_step = self.step
        self.sequence += 1
        self.counts.fill(0)
        self.entries.fill(0)

    def retain_recent_windows(self, capacity):
        self.recent = deque(maxlen=capacity)

    def recent_counts(self):
        total = self.counts.sum(axis=0)
        for counts in self.recent:
            total = total + counts
        keys = self.detector.keys_from_indices(self.cells)
        return {key.decode("ascii"): int(count) for key, count in zip(keys, total, strict=True)}

    def drain(self, *, final=False):
        if final and self.step > self.start_step:
            self._close(complete=False)
        completed, self.completed = self.completed, []
        return tuple(_freeze(window) for window in completed)

    def requeue(self, windows):
        """Keep undelivered immutable summaries after an outbox failure."""
        self.completed[:0] = [_thaw(window) for window in windows]

    def export_state(self, *, recovery_cursor):
        if not isinstance(recovery_cursor, str) or not recovery_cursor:
            raise ValueError("occupancy recovery requires a learner cursor")
        return {
            "schema_version": 1,
            "cell_space_hash": self.contract_hash,
            "window_size": self.window_size,
            "n_envs": self.n_envs,
            "recovery_cursor": recovery_cursor,
            "segment": self.segment,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "step": self.step,
            "start_step": self.start_step,
            "sequence": self.sequence,
            "uncovered_interval": self.uncovered_interval,
            "completed": copy.deepcopy(self.completed),
            "recent": [counts.tolist() for counts in self.recent],
            **{
                name: getattr(self, name).tolist()
                for name in (
                    "assignments",
                    "origins",
                    "entry_pending",
                    "counts",
                    "entries",
                    "cumulative",
                    "cumulative_entries",
                )
            },
        }

    def restart(self, *, initial_step, previous_end=0):
        if type(initial_step) is not int or not 0 <= previous_end <= initial_step:
            raise ValueError("invalid occupancy continuation interval")
        self.segment = uuid.uuid4().hex
        self.step = self.start_step = initial_step
        self.sequence = 0
        self.counts.fill(0)
        self.entries.fill(0)
        self.cumulative.fill(0)
        self.cumulative_entries.fill(0)
        self.entry_pending.fill(True)
        self.completed = []
        self.recent.clear()
        self.uncovered_interval = (
            [previous_end, initial_step] if initial_step > previous_end else None
        )

    def restore_state(self, state, *, recovery_cursor, initial_step=0):
        matches = (
            state.get("schema_version") == 1
            and state.get("cell_space_hash") == self.contract_hash
            and state.get("window_size") == self.window_size
            and state.get("n_envs") == self.n_envs
            and state.get("recovery_cursor") == recovery_cursor
        )
        if not matches:
            self.restart(initial_step=initial_step)
            return False
        completed = [validate_occupancy_window(window) for window in state["completed"]]
        recent = [np.asarray(counts) for counts in state.get("recent", [])]
        if len(recent) > (self.recent.maxlen or 0) or any(
            counts.shape != (len(self.cells),)
            or counts.dtype.kind not in "iu"
            or np.any(counts < 0)
            for counts in recent
        ):
            raise ValueError("invalid occupancy recovery recent counts")
        if type(state["sequence"]) is not int or state["sequence"] < 0:
            raise ValueError("invalid occupancy recovery sequence")
        arrays = {}
        for name in (
            "assignments",
            "origins",
            "entry_pending",
            "counts",
            "entries",
            "cumulative",
            "cumulative_entries",
        ):
            array = np.asarray(state[name])
            expected = getattr(self, name)
            if array.shape != expected.shape or array.dtype.kind not in "ibu" or np.any(array < 0):
                raise ValueError(f"invalid occupancy recovery {name}")
            arrays[name] = array.astype(expected.dtype)
        if (
            np.any(arrays["assignments"] >= len(self.cells))
            or np.any(arrays["origins"] >= len(ORIGINS))
            or np.any(arrays["entries"] > arrays["counts"])
        ):
            raise ValueError("invalid occupancy recovery lane or entry values")
        step, start = state["step"], state["start_step"]
        if (
            type(step) is not int
            or type(start) is not int
            or not 0 <= start <= step
            or step - start != int(arrays["counts"].sum())
            or step - start >= self.window_size
        ):
            raise ValueError("occupancy recovery counters disagree with learner cursor")
        for name, array in arrays.items():
            getattr(self, name)[:] = array
        self.step, self.start_step = step, start
        self.segment = state["segment"]
        self.run_id = state["run_id"]
        self.sequence = state["sequence"]
        self.uncovered_interval = state["uncovered_interval"]
        self.completed = completed
        self.recent.clear()
        self.recent.extend(counts.astype(np.int64) for counts in recent)
        return True
