"""Portable semantic cell graphs and their closed-loop policy runtime."""

from __future__ import annotations

import copy
import hashlib
import json
import zipfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from gradlab.action_contract import action_contract_meanings
from gradlab.action_program import ActionRun, canonicalize_action_runs
from gradlab.json_utils import canonical_json_bytes, canonical_json_sha256
from gradlab.state_archive import (
    ArchiveCellConfig,
    StateArchiveEntry,
    normalize_archive_cell_config,
)


CELL_GRAPH_SCHEMA_VERSION = 1
CELL_GRAPH_POLICY_TYPE = "cell-graph"
CELL_GRAPH_MODEL_CLASS = "gradlab.cell_graph.CellGraphPolicy"
CELL_GRAPH_MEMBER = "cell_graph.json"
CELL_GRAPH_ARTIFACT_IDENTITY_MEMBER = "artifact_identity.json"
CELL_GRAPH_SNAPSHOT_MANIFEST_MEMBER = "snapshot_manifest.json"
CELL_GRAPH_SNAPSHOT_BLOB_PREFIX = "snapshot_blobs/"
MAX_CELL_GRAPH_NODES = 250_000
MAX_CELL_GRAPH_EDGES = 500_000
MAX_CELL_GRAPH_ACTION_RUNS = 1_000_000
MAX_CELL_GRAPH_DOCUMENT_BYTES = 128 * 1024 * 1024
MAX_CELL_GRAPH_SNAPSHOT_BLOB_BYTES = 512 * 1024 * 1024
MAX_CELL_GRAPH_SNAPSHOT_BYTES = 4 * 1024 * 1024 * 1024


def _cell_key_document(value: bytes | None) -> str | None:
    return None if value is None else value.hex()


def _cell_key_from_document(value: object) -> bytes | None:
    if value is None:
        return None
    try:
        return bytes.fromhex(str(value))
    except ValueError as exc:
        raise ValueError("cell-graph document contains an invalid cell key") from exc


def _runs_document(runs: Sequence[ActionRun]) -> list[list[int]]:
    return [[int(run.action), int(run.duration)] for run in runs]


def _runs_from_document(value: object) -> tuple[ActionRun, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError("cell-graph action runs must be a sequence")
    try:
        runs = tuple(ActionRun(int(row[0]), int(row[1])) for row in value)
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("cell-graph action runs are invalid") from exc
    return canonicalize_action_runs(runs)


def slice_action_runs(
    runs: Sequence[ActionRun],
    start: int,
    stop: int,
) -> tuple[ActionRun, ...]:
    """Return the canonical action program covering ``[start, stop)``."""

    if start < 0 or stop < start:
        raise ValueError("cell-graph action slice is invalid")
    result: list[ActionRun] = []
    cursor = 0
    for run in canonicalize_action_runs(runs):
        run_start = cursor
        run_stop = cursor + run.duration
        cursor = run_stop
        overlap_start = max(start, run_start)
        overlap_stop = min(stop, run_stop)
        if overlap_start >= overlap_stop:
            continue
        duration = overlap_stop - overlap_start
        if result and result[-1].action == run.action:
            previous = result[-1]
            result[-1] = ActionRun(previous.action, previous.duration + duration)
        else:
            result.append(ActionRun(run.action, duration))
    if stop > cursor:
        raise ValueError("cell-graph action slice exceeds the route program")
    return tuple(result)


@dataclass(frozen=True)
class CellGraphExecutionContext:
    """Declared policy-side semantic context for one vectorized decision."""

    cell_keys: tuple[bytes, ...]
    episode_seeds: tuple[int | None, ...]
    reset_mask: tuple[bool, ...] = ()

    def __post_init__(self) -> None:
        if len(self.cell_keys) != len(self.episode_seeds):
            raise ValueError("cell-graph context requires one seed per cell key")
        if self.reset_mask and len(self.reset_mask) != len(self.cell_keys):
            raise ValueError("cell-graph context reset mask has the wrong length")

    @property
    def lane_count(self) -> int:
        return len(self.cell_keys)


@dataclass(frozen=True)
class CellGraphNode:
    node_id: str
    cell_key: bytes | None
    target_distance: int
    initial_seed: int | None = None
    outcome: str | None = None
    snapshot_entry_id: str | None = None

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("cell-graph node id must not be empty")
        if self.target_distance < 0:
            raise ValueError("cell-graph target distance must be non-negative")
        if self.initial_seed is not None and self.initial_seed < 0:
            raise ValueError("cell-graph initial seed must be non-negative")
        if (self.cell_key is None) == (self.outcome is None):
            raise ValueError("cell-graph nodes must represent exactly one of a cell or outcome")

    def document(self) -> dict[str, object]:
        return {
            "id": self.node_id,
            "cell_key": _cell_key_document(self.cell_key),
            "target_distance": self.target_distance,
            "initial_seed": self.initial_seed,
            "outcome": self.outcome,
            "snapshot_entry_id": self.snapshot_entry_id,
        }


@dataclass(frozen=True)
class CellGraphEdge:
    edge_id: str
    source_id: str
    target_id: str
    action_runs: tuple[ActionRun, ...]
    observation_count: int = 1
    seed_count: int = 1
    successful_suffix: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_runs", canonicalize_action_runs(self.action_runs))
        if not self.edge_id or not self.source_id or not self.target_id:
            raise ValueError("cell-graph edge identity fields must not be empty")
        if not self.action_runs:
            raise ValueError("cell-graph edges require at least one action")
        if self.observation_count < 1 or self.seed_count < 1:
            raise ValueError("cell-graph edge evidence counts must be positive")

    @property
    def step_count(self) -> int:
        return sum(run.duration for run in self.action_runs)

    def document(self) -> dict[str, object]:
        return {
            "id": self.edge_id,
            "source": self.source_id,
            "target": self.target_id,
            "action_runs": _runs_document(self.action_runs),
            "observation_count": self.observation_count,
            "seed_count": self.seed_count,
            "successful_suffix": self.successful_suffix,
        }


@dataclass
class _LaneRouteState:
    node_id: str | None = None
    edge_id: str | None = None
    run_index: int = 0
    run_remaining: int = 0
    tried_edges: set[str] | None = None
    replans: int = 0
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if self.tried_edges is None:
            self.tried_edges = set()


class CellGraphPolicy:
    """Portable closed-loop controller over a semantic cell graph."""

    def __init__(
        self,
        *,
        action_names: Sequence[str],
        fallback_action: int,
        detector: Mapping[str, Any],
        nodes: Sequence[CellGraphNode],
        edges: Sequence[CellGraphEdge],
        roots: Mapping[int, str],
        target_node_id: str,
        default_seed: int | None,
        snapshot_mode: str = "none",
        snapshot_entries: Mapping[str, Mapping[str, Any]] | None = None,
        snapshot_payloads: Mapping[str, bytes] | None = None,
    ) -> None:
        self.action_names = tuple(str(name) for name in action_names)
        self.fallback_action = int(fallback_action)
        self.detector = normalize_archive_cell_config(
            detector,
            label="cell_graph.detector",
        )
        detector_config = ArchiveCellConfig.from_mapping(
            self.detector,
            label="cell_graph.detector",
        )
        if detector_config.sources:
            raise ValueError("executable cell graphs require semantic signal dimensions")
        self.nodes = tuple(nodes)
        self.edges = tuple(edges)
        self.roots = {int(seed): str(node_id) for seed, node_id in roots.items()}
        self.target_node_id = str(target_node_id)
        self.default_seed = None if default_seed is None else int(default_seed)
        self.snapshot_mode = str(snapshot_mode)
        self.snapshot_entries = {
            str(entry_id): dict(value)
            for entry_id, value in (snapshot_entries or {}).items()
        }
        self.snapshot_payloads = {
            str(digest): bytes(payload)
            for digest, payload in (snapshot_payloads or {}).items()
        }
        self.action_space: gym.Space | None = None
        self.observation_space = None
        self._lane_states = [_LaneRouteState()]
        self._validate()

        self._nodes_by_id = {node.node_id: node for node in self.nodes}
        self._nodes_by_key: dict[bytes, list[CellGraphNode]] = defaultdict(list)
        for node in self.nodes:
            if node.cell_key is not None:
                self._nodes_by_key[node.cell_key].append(node)
        self._edges_by_id = {edge.edge_id: edge for edge in self.edges}
        self._edges_by_source: dict[str, list[CellGraphEdge]] = defaultdict(list)
        for edge in self.edges:
            self._edges_by_source[edge.source_id].append(edge)
        for source in self._edges_by_source:
            self._edges_by_source[source].sort(key=self._edge_rank)

    @property
    def cell_detector_config(self) -> Mapping[str, Any]:
        return self.detector

    @property
    def default_playback_seed(self) -> int | None:
        return self.default_seed

    def _validate(self) -> None:
        if not self.action_names:
            raise ValueError("cell-graph action table must not be empty")
        if not 0 <= self.fallback_action < len(self.action_names):
            raise ValueError("cell-graph fallback action is outside the action table")
        if self.default_seed is not None and self.default_seed < 0:
            raise ValueError("cell-graph default seed must be non-negative")
        if self.snapshot_mode not in {"none", "retained"}:
            raise ValueError("cell-graph snapshot mode must be 'none' or 'retained'")
        if len(self.nodes) > MAX_CELL_GRAPH_NODES:
            raise ValueError("cell-graph artifact has too many nodes")
        if len(self.edges) > MAX_CELL_GRAPH_EDGES:
            raise ValueError("cell-graph artifact has too many edges")
        node_ids = [node.node_id for node in self.nodes]
        edge_ids = [edge.edge_id for edge in self.edges]
        if len(node_ids) != len(set(node_ids)) or node_ids != sorted(node_ids):
            raise ValueError("cell-graph nodes must be sorted with unique ids")
        if len(edge_ids) != len(set(edge_ids)) or edge_ids != sorted(edge_ids):
            raise ValueError("cell-graph edges must be sorted with unique ids")
        node_by_id = {node.node_id: node for node in self.nodes}
        if self.target_node_id not in node_by_id:
            raise ValueError("cell-graph target node is unknown")
        if node_by_id[self.target_node_id].target_distance != 0:
            raise ValueError("cell-graph target node must have distance zero")
        for seed, node_id in self.roots.items():
            if seed < 0 or node_id not in node_by_id:
                raise ValueError("cell-graph root is invalid")
        action_run_count = 0
        for edge in self.edges:
            if edge.source_id not in node_by_id or edge.target_id not in node_by_id:
                raise ValueError("cell-graph edge references an unknown node")
            source = node_by_id[edge.source_id]
            target = node_by_id[edge.target_id]
            if target.target_distance >= source.target_distance:
                raise ValueError("cell-graph edges must strictly decrease target distance")
            if any(run.action < 0 or run.action >= len(self.action_names) for run in edge.action_runs):
                raise ValueError("cell-graph edge action is outside the action table")
            action_run_count += len(edge.action_runs)
        if action_run_count > MAX_CELL_GRAPH_ACTION_RUNS:
            raise ValueError("cell-graph artifact has too many action runs")
        referenced_snapshots = {
            node.snapshot_entry_id
            for node in self.nodes
            if node.snapshot_entry_id is not None
        }
        if self.snapshot_mode == "none":
            if referenced_snapshots or self.snapshot_entries or self.snapshot_payloads:
                raise ValueError("snapshot-free cell graph contains snapshot data")
        else:
            if referenced_snapshots != set(self.snapshot_entries):
                raise ValueError(
                    "cell-graph snapshot entries must exactly match node references"
                )
            referenced_blobs: set[str] = set()
            for entry_id, raw_entry in self.snapshot_entries.items():
                entry = StateArchiveEntry.from_dict(raw_entry)
                if entry.entry_id != entry_id:
                    raise ValueError("cell-graph snapshot entry id disagrees")
                ref = entry.provider_snapshot.ref
                referenced_blobs.add(ref.blob_sha256)
                payload = self.snapshot_payloads.get(ref.blob_sha256)
                if payload is None:
                    raise ValueError("cell-graph snapshot payload is missing")
                if (
                    hashlib.sha256(payload).hexdigest() != ref.blob_sha256
                    or len(payload) != ref.size_bytes
                ):
                    raise ValueError(
                        "cell-graph snapshot payload failed integrity verification"
                    )
            if referenced_blobs != set(self.snapshot_payloads):
                raise ValueError(
                    "cell-graph snapshot blobs must exactly match entry references"
                )

    def _edge_rank(self, edge: CellGraphEdge) -> tuple[object, ...]:
        target = next(node for node in self.nodes if node.node_id == edge.target_id)
        return (
            target.target_distance,
            -int(edge.successful_suffix),
            -edge.seed_count,
            -edge.observation_count,
            edge.step_count,
            edge.edge_id,
        )

    def bind_action_space(self, action_space: gym.Space) -> None:
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError("cell-graph playback requires a discrete action space")
        if int(action_space.n) != len(self.action_names):
            raise ValueError("cell-graph action table does not match the environment")
        self.action_space = action_space

    def bind_action_contract(self, action_contract: Mapping[str, Any]) -> None:
        if action_contract_meanings(action_contract) != self.action_names:
            raise ValueError("cell-graph semantic action table does not match the environment")

    def _ensure_lanes(self, count: int) -> None:
        while len(self._lane_states) < count:
            self._lane_states.append(_LaneRouteState())
        if len(self._lane_states) > count:
            self._lane_states = self._lane_states[:count]

    def reset_episode(self) -> None:
        self._lane_states = [_LaneRouteState() for _ in self._lane_states]

    def reset_lanes(self, dones: Sequence[bool]) -> None:
        mask = np.asarray(dones, dtype=bool)
        self._ensure_lanes(int(mask.size))
        for lane in np.flatnonzero(mask):
            self._lane_states[int(lane)] = _LaneRouteState()

    def resume_node(self, node_id: str, *, lane: int = 0) -> None:
        if node_id not in self._nodes_by_id:
            raise ValueError(f"unknown cell-graph representative {node_id!r}")
        self._ensure_lanes(lane + 1)
        self._lane_states[lane] = _LaneRouteState(node_id=node_id)

    def snapshot(self, node_id: str) -> tuple[Mapping[str, Any], bytes]:
        node = self._nodes_by_id.get(str(node_id))
        if node is None or node.snapshot_entry_id is None:
            raise ValueError(f"cell-graph representative {node_id!r} has no snapshot")
        entry = self.snapshot_entries[node.snapshot_entry_id]
        snapshot = entry.get("provider_snapshot")
        if not isinstance(snapshot, Mapping):
            raise ValueError("cell-graph snapshot entry is invalid")
        ref = snapshot.get("ref")
        if not isinstance(ref, Mapping):
            raise ValueError("cell-graph snapshot ref is invalid")
        digest = str(ref.get("blob_sha256") or "")
        try:
            payload = self.snapshot_payloads[digest]
        except KeyError as exc:
            raise ValueError("cell-graph snapshot payload is missing") from exc
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("cell-graph snapshot payload failed integrity verification")
        return entry, payload

    def _root_for(self, key: bytes, seed: int | None) -> str | None:
        if seed is not None:
            exact = self.roots.get(seed)
            if exact is not None and self._nodes_by_id[exact].cell_key == key:
                return exact
        return self._generalized_node_for_key(key)

    def _generalized_node_for_key(
        self,
        key: bytes,
        *,
        maximum_distance: int | None = None,
    ) -> str | None:
        candidates: list[tuple[object, ...]] = []
        for node in self._nodes_by_key.get(key, ()):
            if (
                maximum_distance is not None
                and node.target_distance >= maximum_distance
            ):
                continue
            edges = [
                edge
                for edge in self._edges_by_source.get(node.node_id, ())
                if edge.seed_count >= 2
            ]
            if not edges:
                continue
            edge = min(edges, key=self._edge_rank)
            candidates.append(
                (
                    node.target_distance,
                    self._edge_rank(edge),
                    node.node_id,
                )
            )
        if not candidates:
            return None
        return str(min(candidates)[-1])

    def _replan_for_key(self, key: bytes, *, maximum_distance: int) -> str | None:
        return self._generalized_node_for_key(
            key,
            maximum_distance=maximum_distance,
        )

    def _select_edge(self, state: _LaneRouteState) -> CellGraphEdge | None:
        if state.node_id is None:
            return None
        for edge in self._edges_by_source.get(state.node_id, ()):
            assert state.tried_edges is not None
            if edge.edge_id not in state.tried_edges:
                state.edge_id = edge.edge_id
                state.run_index = 0
                state.run_remaining = 0
                state.tried_edges.add(edge.edge_id)
                state.fallback_reason = None
                return edge
        state.edge_id = None
        state.fallback_reason = "no_target_route"
        return None

    @staticmethod
    def _edge_action(edge: CellGraphEdge, state: _LaneRouteState, *, advance: bool) -> int | None:
        if state.run_index >= len(edge.action_runs):
            return None
        run = edge.action_runs[state.run_index]
        remaining = state.run_remaining or run.duration
        action = run.action
        if advance:
            remaining -= 1
            state.run_remaining = remaining
            if remaining == 0:
                state.run_index += 1
        return action

    def _decision_for_lane(
        self,
        lane: int,
        context: CellGraphExecutionContext,
        *,
        advance: bool,
    ):
        from gradlab.play_debug import PolicyDecision

        state = self._lane_states[lane]
        key = context.cell_keys[lane]
        seed = context.episode_seeds[lane]
        reset = bool(context.reset_mask and context.reset_mask[lane])
        if reset:
            self._lane_states[lane] = state = _LaneRouteState()
        if state.node_id is None:
            state.node_id = self._root_for(key, seed)
            if state.node_id is None:
                state.fallback_reason = "unknown_or_ambiguous_root"

        edge = self._edges_by_id.get(state.edge_id or "")
        if edge is not None and state.node_id is not None:
            source = self._nodes_by_id[edge.source_id]
            target = self._nodes_by_id[edge.target_id]
            if target.cell_key is not None and key == target.cell_key:
                state.node_id = target.node_id
                state.edge_id = None
                state.run_index = 0
                state.run_remaining = 0
                state.tried_edges = set()
                edge = None
            elif source.cell_key is not None and key != source.cell_key:
                replanned = self._replan_for_key(
                    key,
                    maximum_distance=source.target_distance,
                )
                state.edge_id = None
                state.run_index = 0
                state.run_remaining = 0
                if replanned is None:
                    state.node_id = None
                    state.fallback_reason = "unknown_or_ambiguous_divergence"
                else:
                    state.node_id = replanned
                    state.tried_edges = set()
                    state.replans += 1
                edge = None
            elif state.run_index >= len(edge.action_runs):
                state.edge_id = None
                state.run_index = 0
                state.run_remaining = 0
                state.fallback_reason = "edge_exhausted"
                edge = None

        if edge is None and state.node_id is not None:
            edge = self._select_edge(state)
        action = (
            None
            if edge is None
            else self._edge_action(edge, state, advance=advance)
        )
        if action is None:
            action = self.fallback_action
        route = {
            "cell_key": key.hex(),
            "representative_id": state.node_id,
            "target_node_id": self.target_node_id,
            "edge_id": None if edge is None else edge.edge_id,
            "edge_run_index": state.run_index,
            "edge_run_remaining": state.run_remaining,
            "target_distance": (
                None
                if state.node_id is None
                else self._nodes_by_id[state.node_id].target_distance
            ),
            "replans": state.replans,
            "fallback": edge is None,
            "fallback_reason": state.fallback_reason,
            "action": action,
            "action_name": self.action_names[action],
        }
        value = np.asarray(action, dtype=np.int64)
        return PolicyDecision(
            raw_action=value,
            executed_action=value,
            action_selection_mode="route",
            distribution_kind=None,
            mode=None,
            route=route,
            sampled=None,
        )

    def policy_decisions(
        self,
        observation: Any,
        *,
        action_selection_mode: str = "route",
        execution_context: CellGraphExecutionContext | None = None,
    ):
        del observation
        if action_selection_mode != "route":
            raise ValueError("cell graphs support only route action selection")
        if execution_context is None:
            raise ValueError("cell-graph decisions require semantic execution context")
        self._ensure_lanes(execution_context.lane_count)
        return tuple(
            self._decision_for_lane(lane, execution_context, advance=True)
            for lane in range(execution_context.lane_count)
        )

    def inspect_policy_decisions(
        self,
        observation: Any,
        *,
        action_selection_mode: str = "route",
        execution_context: CellGraphExecutionContext | None = None,
    ):
        before = copy.deepcopy(self._lane_states)
        try:
            return self.policy_decisions(
                observation,
                action_selection_mode=action_selection_mode,
                execution_context=execution_context,
            )
        finally:
            self._lane_states = before

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": CELL_GRAPH_SCHEMA_VERSION,
            "purpose": "policy",
            "policy_type": CELL_GRAPH_POLICY_TYPE,
            "model_class": CELL_GRAPH_MODEL_CLASS,
            "action_names": list(self.action_names),
            "fallback_action": self.fallback_action,
            "detector": self.detector,
            "detector_sha256": canonical_json_sha256(self.detector),
            "nodes": [node.document() for node in self.nodes],
            "edges": [edge.document() for edge in self.edges],
            "roots": [
                {"seed": seed, "node_id": node_id}
                for seed, node_id in sorted(self.roots.items())
            ],
            "target_node_id": self.target_node_id,
            "default_seed": self.default_seed,
            "snapshot_mode": self.snapshot_mode,
            "summary": {
                "semantic_cell_count": len(
                    {node.cell_key for node in self.nodes if node.cell_key is not None}
                ),
                "representative_count": len(self.nodes),
                "edge_count": len(self.edges),
                "root_count": len(self.roots),
                "routable_root_count": sum(
                    bool(self._edges_by_source.get(node_id))
                    for node_id in self.roots.values()
                ),
                "snapshot_entry_count": len(self.snapshot_entries),
                "snapshot_blob_count": len(self.snapshot_payloads),
                "snapshot_blob_bytes": sum(
                    len(payload) for payload in self.snapshot_payloads.values()
                ),
            },
        }

    def save(
        self,
        path: str | Path,
        *,
        artifact_discriminator: str | None = None,
    ) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        members: dict[str, bytes] = {
            CELL_GRAPH_MEMBER: canonical_json_bytes(self.payload()) + b"\n",
        }
        if artifact_discriminator is not None:
            discriminator = str(artifact_discriminator).strip()
            if not discriminator:
                raise ValueError("cell-graph artifact discriminator must not be empty")
            members[CELL_GRAPH_ARTIFACT_IDENTITY_MEMBER] = canonical_json_bytes(
                {
                    "schema_version": 1,
                    "artifact_discriminator": discriminator,
                }
            ) + b"\n"
        if self.snapshot_mode == "retained":
            manifest = {
                "schema_version": 1,
                "entries": {
                    entry_id: self.snapshot_entries[entry_id]
                    for entry_id in sorted(self.snapshot_entries)
                },
            }
            members[CELL_GRAPH_SNAPSHOT_MANIFEST_MEMBER] = canonical_json_bytes(manifest) + b"\n"
            for digest, payload in sorted(self.snapshot_payloads.items()):
                if hashlib.sha256(payload).hexdigest() != digest:
                    raise ValueError("cell-graph snapshot payload hash is invalid")
                members[f"{CELL_GRAPH_SNAPSHOT_BLOB_PREFIX}{digest}.bin"] = payload
        with zipfile.ZipFile(destination, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(members):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(info, members[name])

    @classmethod
    def load(cls, path: str | Path) -> "CellGraphPolicy":
        with zipfile.ZipFile(Path(path)) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("cell-graph artifact contains duplicate members")
            allowed_exact = {
                CELL_GRAPH_MEMBER,
                CELL_GRAPH_ARTIFACT_IDENTITY_MEMBER,
                CELL_GRAPH_SNAPSHOT_MANIFEST_MEMBER,
            }
            unknown = [
                name
                for name in names
                if name not in allowed_exact
                and not name.startswith(CELL_GRAPH_SNAPSHOT_BLOB_PREFIX)
            ]
            if unknown or any(name.endswith("/") or ".." in Path(name).parts for name in names):
                raise ValueError("cell-graph artifact contains unsupported members")
            if CELL_GRAPH_MEMBER not in names:
                raise ValueError(f"unsupported policy artifact: missing {CELL_GRAPH_MEMBER}")
            info_by_name = {info.filename: info for info in archive.infolist()}
            for document_name in (
                CELL_GRAPH_MEMBER,
                CELL_GRAPH_ARTIFACT_IDENTITY_MEMBER,
                CELL_GRAPH_SNAPSHOT_MANIFEST_MEMBER,
            ):
                info = info_by_name.get(document_name)
                if info is not None and info.file_size > MAX_CELL_GRAPH_DOCUMENT_BYTES:
                    raise ValueError("cell-graph artifact document is too large")
            payload = json.loads(archive.read(CELL_GRAPH_MEMBER))
            snapshot_entries: dict[str, Mapping[str, Any]] = {}
            snapshot_payloads: dict[str, bytes] = {}
            if CELL_GRAPH_SNAPSHOT_MANIFEST_MEMBER in names:
                manifest = json.loads(archive.read(CELL_GRAPH_SNAPSHOT_MANIFEST_MEMBER))
                if not isinstance(manifest, Mapping) or int(manifest.get("schema_version", 0)) != 1:
                    raise ValueError("cell-graph snapshot manifest is invalid")
                entries = manifest.get("entries")
                if not isinstance(entries, Mapping):
                    raise ValueError("cell-graph snapshot entries are invalid")
                snapshot_entries = {
                    str(entry_id): dict(value)
                    for entry_id, value in entries.items()
                    if isinstance(value, Mapping)
                }
                if len(snapshot_entries) != len(entries):
                    raise ValueError("cell-graph snapshot entry is invalid")
                snapshot_bytes = 0
                for name in names:
                    if not name.startswith(CELL_GRAPH_SNAPSHOT_BLOB_PREFIX):
                        continue
                    info = info_by_name[name]
                    if info.file_size > MAX_CELL_GRAPH_SNAPSHOT_BLOB_BYTES:
                        raise ValueError("cell-graph snapshot blob is too large")
                    snapshot_bytes += info.file_size
                    if snapshot_bytes > MAX_CELL_GRAPH_SNAPSHOT_BYTES:
                        raise ValueError("cell-graph snapshot payloads are too large")
                    digest = name.removeprefix(CELL_GRAPH_SNAPSHOT_BLOB_PREFIX).removesuffix(
                        ".bin"
                    )
                    blob = archive.read(name)
                    if len(digest) != 64 or hashlib.sha256(blob).hexdigest() != digest:
                        raise ValueError("cell-graph snapshot blob failed integrity verification")
                    snapshot_payloads[digest] = blob
        if not isinstance(payload, Mapping):
            raise ValueError("cell-graph payload must be an object")
        expected_fields = {
            "schema_version",
            "purpose",
            "policy_type",
            "model_class",
            "action_names",
            "fallback_action",
            "detector",
            "detector_sha256",
            "nodes",
            "edges",
            "roots",
            "target_node_id",
            "default_seed",
            "snapshot_mode",
            "summary",
        }
        if set(payload) != expected_fields:
            raise ValueError("cell-graph payload fields disagree")
        if int(payload["schema_version"]) != CELL_GRAPH_SCHEMA_VERSION:
            raise ValueError("unsupported cell-graph schema version")
        if payload["purpose"] != "policy" or payload["policy_type"] != CELL_GRAPH_POLICY_TYPE:
            raise ValueError("cell-graph payload has the wrong purpose or policy type")
        if payload["model_class"] != CELL_GRAPH_MODEL_CLASS:
            raise ValueError("cell-graph payload has the wrong model class")
        detector = payload["detector"]
        if not isinstance(detector, Mapping):
            raise ValueError("cell-graph detector must be an object")
        normalized_detector = normalize_archive_cell_config(
            detector,
            label="cell_graph.detector",
        )
        if payload["detector_sha256"] != canonical_json_sha256(normalized_detector):
            raise ValueError("cell-graph detector hash mismatch")
        raw_nodes = payload["nodes"]
        raw_edges = payload["edges"]
        raw_roots = payload["roots"]
        if (
            isinstance(raw_nodes, str | bytes)
            or not isinstance(raw_nodes, Sequence)
            or isinstance(raw_edges, str | bytes)
            or not isinstance(raw_edges, Sequence)
            or isinstance(raw_roots, str | bytes)
            or not isinstance(raw_roots, Sequence)
        ):
            raise ValueError("cell-graph graph collections are invalid")
        nodes = []
        for value in raw_nodes:
            if not isinstance(value, Mapping) or set(value) != {
                "id",
                "cell_key",
                "target_distance",
                "initial_seed",
                "outcome",
                "snapshot_entry_id",
            }:
                raise ValueError("cell-graph node document is invalid")
            nodes.append(
                CellGraphNode(
                    node_id=str(value["id"]),
                    cell_key=_cell_key_from_document(value["cell_key"]),
                    target_distance=int(value["target_distance"]),
                    initial_seed=(
                        None if value["initial_seed"] is None else int(value["initial_seed"])
                    ),
                    outcome=None if value["outcome"] is None else str(value["outcome"]),
                    snapshot_entry_id=(
                        None
                        if value["snapshot_entry_id"] is None
                        else str(value["snapshot_entry_id"])
                    ),
                )
            )
        edges = []
        for value in raw_edges:
            if not isinstance(value, Mapping) or set(value) != {
                "id",
                "source",
                "target",
                "action_runs",
                "observation_count",
                "seed_count",
                "successful_suffix",
            }:
                raise ValueError("cell-graph edge document is invalid")
            edges.append(
                CellGraphEdge(
                    edge_id=str(value["id"]),
                    source_id=str(value["source"]),
                    target_id=str(value["target"]),
                    action_runs=_runs_from_document(value["action_runs"]),
                    observation_count=int(value["observation_count"]),
                    seed_count=int(value["seed_count"]),
                    successful_suffix=bool(value["successful_suffix"]),
                )
            )
        roots: dict[int, str] = {}
        for value in raw_roots:
            if not isinstance(value, Mapping) or set(value) != {"seed", "node_id"}:
                raise ValueError("cell-graph root document is invalid")
            seed = int(value["seed"])
            if seed in roots:
                raise ValueError("cell-graph root seeds must be unique")
            roots[seed] = str(value["node_id"])
        policy = cls(
            action_names=payload["action_names"],
            fallback_action=int(payload["fallback_action"]),
            detector=normalized_detector,
            nodes=nodes,
            edges=edges,
            roots=roots,
            target_node_id=str(payload["target_node_id"]),
            default_seed=(
                None if payload["default_seed"] is None else int(payload["default_seed"])
            ),
            snapshot_mode=str(payload["snapshot_mode"]),
            snapshot_entries=snapshot_entries,
            snapshot_payloads=snapshot_payloads,
        )
        if payload["summary"] != policy.payload()["summary"]:
            raise ValueError("cell-graph summary does not match its graph")
        return policy


def route_node_id(
    *,
    seed: int | None,
    cell_key: bytes,
    prefix_runs: Sequence[ActionRun],
) -> str:
    return canonical_json_sha256(
        {
            "semantic_id": "cell-graph-representative-v1",
            "seed": seed,
            "cell_key": cell_key.hex(),
            "prefix_runs": _runs_document(prefix_runs),
        }
    )


def route_edge_id(
    *,
    source_id: str,
    target_id: str,
    action_runs: Sequence[ActionRun],
) -> str:
    return canonical_json_sha256(
        {
            "semantic_id": "cell-graph-edge-v1",
            "source": source_id,
            "target": target_id,
            "action_runs": _runs_document(action_runs),
        }
    )


__all__ = [
    "CELL_GRAPH_MODEL_CLASS",
    "CELL_GRAPH_POLICY_TYPE",
    "CellGraphEdge",
    "CellGraphExecutionContext",
    "CellGraphNode",
    "CellGraphPolicy",
    "route_edge_id",
    "route_node_id",
    "slice_action_runs",
]
