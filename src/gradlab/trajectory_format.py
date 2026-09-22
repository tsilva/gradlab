"""Trajectory table contract shared with the published Breakout collector format.

Images remain unmasked; split assignment belongs to the consumer.
"""

import base64
import json
from collections.abc import Mapping
import numpy as np
import pyarrow as pa
from gradlab.play_trajectory import encode_tree, decode_tree
from gradlab.json_utils import canonical_json_bytes as canonical
from gradlab.json_utils import canonical_json_sha256


TRAJECTORY_SCHEMA_VERSION = 1
TRAJECTORY_FORMAT = "gradlab.trajectories.webp.v1"


def schema(fields):
    return pa.schema([(name, getattr(pa, kind)()) for name, kind in fields])


transition_schema = schema(
    [
        *[
            (name, "int64")
            for name in (
                "episode_id",
                "step",
                "source_frame_id",
                "successor_frame_id",
                "policy_decision_id",
                "configured_frame_skip",
                "elapsed_native_frames",
            )
        ],
        *[
            (name, "bool_")
            for name in (
                "successor_frame_new",
                "terminated",
                "truncated",
                "native_game_over",
                "native_truncated",
                "task_terminated",
                "task_truncated",
            )
        ],
        *[
            (name, "float64")
            for name in ("policy_reward", "native_reward", "task_reward", "temperature")
        ],
        *[
            (name, "string")
            for name in (
                "session_id",
                "action_selection_mode",
                "action_override_rule_id",
                "selected_action_json",
                "effective_action_json",
                "native_action_json",
                "record_json",
            )
        ],
    ]
)
episode_schema = schema(
    [
        *[(name, "int64") for name in ("episode_id", "seed", "initial_frame_id", "length")],
        *[(name, "string") for name in ("session_id", "split", "status", "end_reason")],
        ("initial_frame_new", "bool_"),
    ]
)
session_schema = schema([("session_id", "string"), ("record_json", "string")])
image_features = {
    "frame_id": {"dtype": "int64", "_type": "Value"},
    "sha256": {"dtype": "string", "_type": "Value"},
    "image": {"_type": "Image"},
}
frame_schema = pa.schema(
    [
        ("frame_id", pa.int64()),
        ("sha256", pa.string()),
        ("image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
    ],
    metadata={b"huggingface": canonical({"info": {"features": image_features}})},
)


def record_json(value):
    tree = encode_tree(value)
    for leaf in tree["arrays"]:
        leaf["data"] = base64.b64encode(leaf["data"]).decode("ascii")
    return canonical(tree).decode()


def read_record(value):
    tree = json.loads(value)
    for leaf in tree["arrays"]:
        leaf["data"] = base64.b64decode(leaf["data"], validate=True)
    return decode_tree(tree)


def transition_record(row):
    result = {key: row.get(key) for key in transition_schema.names}
    for key in ("selected_action", "effective_action", "native_action"):
        value = row.get(key)
        if isinstance(value, np.ndarray | np.generic):
            value = value.tolist()
        result[f"{key}_json"] = json.dumps(value, allow_nan=False, separators=(",", ":"))
    result["record_json"] = record_json(row)
    return result


def annotation_fields(prefix=""):
    fields = [
        ("brick_grid", pa.list_(pa.list_(pa.int8(), 18), 6)),
        ("brick_grid_suspect", pa.bool_()),
        ("brick_grid_quality_flags", pa.uint16()),
        ("brick_count_visible", pa.uint8()),
        ("brick_grid_unknown_cells", pa.uint8()),
        ("brick_grid_min_present_support", pa.uint8()),
        ("brick_count_mismatch", pa.bool_()),
        ("is_initial_brick_layout", pa.bool_()),
    ]
    return [
        (
            prefix + ("is_brick_layout" if prefix and name == "is_initial_brick_layout" else name),
            kind,
        )
        for name, kind in fields
    ]


transition_schema = pa.schema(
    list(transition_schema)
    + [pa.field(k, v) for k, v in annotation_fields()]
    + [
        pa.field("source_is_initial_brick_layout", pa.bool_()),
        pa.field("source_brick_grid_suspect", pa.bool_()),
    ]
)
episode_schema = pa.schema(
    list(episode_schema) + [pa.field(k, v) for k, v in annotation_fields("initial_")]
)

# This contract versions meanings as well as Arrow fields. Any contract change
# requires a new version, a documented migration and an updated fingerprint test.
_TABLE_SCHEMAS = {
    "transitions": transition_schema,
    "episodes": episode_schema,
    "frames": frame_schema,
    "sessions": session_schema,
}
_SEMANTICS = {
    "identity": "IDs are snapshot-local keys, never row offsets; episode steps start at zero.",
    "joins": "Transitions join episodes on episode_id, sessions on session_id and frames on source_frame_id/successor_frame_id; episodes join initial_frame_id and session_id.",
    "frames": "Deduplicated lossless WebP, full unmasked 210x160 uint8 HWC RGB; sha256 hashes canonical RGB header v2 followed by pixels.",
    "actions": "selected_action_json is policy-requested; effective_action_json follows overrides; native_action_json is the submitted provider index, not emulator encoding.",
    "rewards": "policy_reward is shaped learner reward; native_reward is provider reward; separately recorded task_reward is nullable.",
    "boundaries": "terminated/truncated are scientific episode boundaries; native and task boundaries retain their distinct meanings; successor is the true pre-reset frame.",
    "missing": "Nullable unavailable facts remain null, never fabricated or zero-filled.",
    "records": "record_json is the GradLab tagged-tree codec with base64 array leaves; sessions retain original monitoring episode provenance.",
    "annotations": "breakout-bricks-v1; transition brick grids describe successors, source flags describe sources, initial episode annotations describe initial frames.",
    "splits": "Null means unassigned; explicit derived splits preserve complete episodes and group reused evaluation seeds across checkpoints.",
}


def trajectory_schema_contract():
    """Return a fresh, JSON-serializable definition of the complete current contract."""
    return {
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "format": TRAJECTORY_FORMAT,
        "tables": {
            name: [
                {"name": field.name, "type": str(field.type), "nullable": field.nullable}
                for field in table
            ]
            for name, table in _TABLE_SCHEMAS.items()
        },
        "semantics": dict(_SEMANTICS),
    }


def trajectory_schema_identity():
    return {
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "trajectory_schema_sha256": canonical_json_sha256(trajectory_schema_contract()),
    }


def require_current_trajectory_schema(document):
    """Fail closed on absent, malformed, historical or future declarations."""
    if not isinstance(document, Mapping):
        raise ValueError("Missing trajectory schema declaration; use an explicit migration")
    version = document.get("trajectory_schema_version")
    if type(version) is not int or version != TRAJECTORY_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported trajectory schema version: {version!r}; "
            f"expected {TRAJECTORY_SCHEMA_VERSION}. Use an explicit migration."
        )
    if (
        document.get("trajectory_schema_sha256")
        != trajectory_schema_identity()["trajectory_schema_sha256"]
    ):
        raise ValueError("Trajectory schema fingerprint mismatch")
    return version


def table_schema(name):
    """The stamped Arrow schema, including the HF Image feature metadata."""
    original = _TABLE_SCHEMAS[name]
    identity = trajectory_schema_identity()
    return original.with_metadata(
        {
            **(original.metadata or {}),
            b"gradlab.trajectory_schema_version": str(TRAJECTORY_SCHEMA_VERSION).encode(),
            b"gradlab.trajectory_schema_sha256": identity["trajectory_schema_sha256"].encode(),
            b"gradlab.trajectory_table": name.encode(),
        }
    )


def validate_table_schema(actual, name):
    expected = table_schema(name)
    metadata = actual.metadata or {}
    # Compare metadata as bytes: b"01", floats and bools are not valid versions.
    for key in (
        b"gradlab.trajectory_schema_version",
        b"gradlab.trajectory_schema_sha256",
        b"gradlab.trajectory_table",
    ):
        if metadata.get(key) != expected.metadata[key]:
            raise ValueError(
                f"{name}: missing or incompatible trajectory schema metadata ({key.decode()})"
            )
    if not actual.equals(expected, check_metadata=False):
        raise ValueError(f"{name}: physical table schema differs from declared trajectory schema")
    if name == "frames":
        try:
            features = json.loads(metadata[b"huggingface"])["info"]["features"]
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError("frames: missing Hugging Face Image metadata") from exc
        if features != image_features:
            raise ValueError("frames: incompatible Hugging Face Image metadata")


def open_trajectory_parquet(source, name):
    """Validate the footer before exposing any rows to a consumer."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(source)
    try:
        validate_table_schema(parquet.schema_arrow, name)
    except Exception:
        parquet.close()
        raise
    return parquet


transition_schema = table_schema("transitions")
episode_schema = table_schema("episodes")
frame_schema = table_schema("frames")
session_schema = table_schema("sessions")
