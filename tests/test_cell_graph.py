from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pytest

from gradlab.action_program import ActionRun
from gradlab.cell_graph import (
    CELL_GRAPH_MEMBER,
    CELL_GRAPH_SNAPSHOT_BLOB_PREFIX,
    CELL_GRAPH_SNAPSHOT_MANIFEST_MEMBER,
    CellGraphEdge,
    CellGraphExecutionContext,
    CellGraphNode,
    CellGraphPolicy,
)
from gradlab.json_utils import canonical_json_bytes
from gradlab.policy_runtime import PolicyRuntime


ROOT_KEY = b"root"
MIDDLE_KEY = b"middle"


def _policy(
    *,
    snapshot_mode: str = "none",
    snapshot_entries=None,
    snapshot_payloads=None,
) -> CellGraphPolicy:
    snapshot_entry_id = (
        next(iter(snapshot_entries))
        if snapshot_mode == "retained" and snapshot_entries
        else None
    )
    nodes = (
        CellGraphNode(
            node_id="middle",
            cell_key=MIDDLE_KEY,
            target_distance=1,
        ),
        CellGraphNode(
            node_id="root",
            cell_key=ROOT_KEY,
            target_distance=2,
            initial_seed=7,
            snapshot_entry_id=snapshot_entry_id,
        ),
        CellGraphNode(
            node_id="target",
            cell_key=None,
            target_distance=0,
            outcome="success",
        ),
    )
    edges = (
        CellGraphEdge(
            edge_id="middle-target",
            source_id="middle",
            target_id="target",
            action_runs=(ActionRun(0, 1),),
            successful_suffix=True,
        ),
        CellGraphEdge(
            edge_id="root-middle",
            source_id="root",
            target_id="middle",
            action_runs=(ActionRun(1, 2),),
            observation_count=3,
            seed_count=2,
            successful_suffix=True,
        ),
    )
    return CellGraphPolicy(
        action_names=("noop", "right"),
        fallback_action=0,
        detector={"dimensions": [{"signal": "x", "bucket_size": 1}]},
        nodes=nodes,
        edges=edges,
        roots={7: "root"},
        target_node_id="target",
        default_seed=7,
        snapshot_mode=snapshot_mode,
        snapshot_entries=snapshot_entries,
        snapshot_payloads=snapshot_payloads,
    )


def _context(
    key: bytes,
    *,
    seed: int | None = 7,
    reset: bool = False,
) -> CellGraphExecutionContext:
    return CellGraphExecutionContext(
        cell_keys=(key,),
        episode_seeds=(seed,),
        reset_mask=(reset,),
    )


def test_cell_graph_runtime_routes_replans_and_reports_fallback() -> None:
    runtime = PolicyRuntime(_policy())
    observation = np.zeros((1, 1), dtype=np.float32)

    first = runtime.decide(
        observation,
        execution_context=_context(ROOT_KEY, reset=True),
    ).decisions[0]
    second = runtime.decide(
        observation,
        execution_context=_context(ROOT_KEY),
    ).decisions[0]
    replanned = runtime.decide(
        observation,
        execution_context=_context(MIDDLE_KEY),
    ).decisions[0]

    assert int(first.executed_action) == 1
    assert first.action_selection_mode == "route"
    assert first.route["representative_id"] == "root"
    assert first.route["edge_id"] == "root-middle"
    assert int(second.executed_action) == 1
    assert int(replanned.executed_action) == 0
    assert replanned.route["representative_id"] == "middle"
    assert replanned.route["edge_id"] == "middle-target"
    assert replanned.route["target_distance"] == 1

    generalized = runtime.decide(
        observation,
        execution_context=_context(ROOT_KEY, seed=99, reset=True),
    ).decisions[0]
    assert int(generalized.executed_action) == 1
    assert generalized.route["representative_id"] == "root"

    fallback = runtime.decide(
        observation,
        execution_context=_context(b"unseen", seed=99, reset=True),
    ).decisions[0]
    assert int(fallback.executed_action) == 0
    assert fallback.route["fallback"] is True
    assert fallback.route["fallback_reason"] == "unknown_or_ambiguous_root"


def test_cell_graph_inspection_does_not_advance_route() -> None:
    runtime = PolicyRuntime(_policy())
    observation = np.zeros((1, 1), dtype=np.float32)
    context = _context(ROOT_KEY, reset=True)

    inspected = runtime.inspect(
        observation,
        execution_context=context,
    ).decisions[0]
    executed = runtime.decide(
        observation,
        execution_context=context,
    ).decisions[0]

    assert inspected.route["edge_run_remaining"] == 1
    assert executed.route["edge_run_remaining"] == 1


def test_cell_graph_default_artifact_contains_no_snapshot_members(tmp_path: Path) -> None:
    path = tmp_path / "policy.zip"
    _policy().save(path)

    with zipfile.ZipFile(path) as archive:
        assert archive.namelist() == [CELL_GRAPH_MEMBER]

    loaded = CellGraphPolicy.load(path)
    assert loaded.payload() == _policy().payload()
    assert loaded.payload()["summary"] == {
        "semantic_cell_count": 2,
        "representative_count": 3,
        "edge_count": 2,
        "root_count": 1,
        "routable_root_count": 1,
        "snapshot_entry_count": 0,
        "snapshot_blob_count": 0,
        "snapshot_blob_bytes": 0,
    }

def test_cell_graph_rejects_unknown_zip_members(tmp_path: Path) -> None:
    path = tmp_path / "policy.zip"
    _policy().save(path)
    with zipfile.ZipFile(path, mode="a") as archive:
        archive.writestr("unexpected.bin", b"no")

    with pytest.raises(ValueError, match="unsupported members"):
        CellGraphPolicy.load(path)


def test_cell_graph_retained_snapshot_is_opt_in_and_integrity_checked(
    tmp_path: Path,
) -> None:
    payload = b"provider-state"
    digest = hashlib.sha256(payload).hexdigest()
    identity = {
        "semantic_id": "state-archive-v1",
        "schema_version": 1,
        "provider_snapshot": {
            "provider_id": "breakout-turbo-env",
            "compatibility_id": "test-environment-v1",
            "ref": {
                "codec_id": "breakout-turbo-env.state-v1",
                "blob_sha256": digest,
                "size_bytes": len(payload),
            }
        },
        "task_state": None,
        "runtime_state": None,
        "restore_semantics": "episode_start",
        "created_step": 0,
        "metadata": {},
    }
    entry_id = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
    entry = {"entry_id": entry_id, **identity}
    path = tmp_path / "policy.zip"
    _policy(
        snapshot_mode="retained",
        snapshot_entries={entry_id: entry},
        snapshot_payloads={digest: payload},
    ).save(path)

    with zipfile.ZipFile(path) as archive:
        assert CELL_GRAPH_SNAPSHOT_MANIFEST_MEMBER in archive.namelist()
        assert (
            f"{CELL_GRAPH_SNAPSHOT_BLOB_PREFIX}{digest}.bin"
            in archive.namelist()
        )

    loaded = CellGraphPolicy.load(path)
    assert loaded.snapshot("root") == (entry, payload)


def test_cell_graph_rejects_training_only_source_dimensions() -> None:
    with pytest.raises(ValueError, match="semantic signal"):
        CellGraphPolicy(
            action_names=("noop",),
            fallback_action=0,
            detector={"dimensions": [{"source": "ram_x", "bucket_size": 1}]},
            nodes=(
                CellGraphNode(
                    node_id="target",
                    cell_key=None,
                    target_distance=0,
                    outcome="success",
                ),
            ),
            edges=(),
            roots={},
            target_node_id="target",
            default_seed=None,
        )
