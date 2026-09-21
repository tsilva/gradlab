"""Trajectory table contract shared with the published Breakout collector format.

Images remain unmasked; split assignment belongs to the consumer.
"""

import base64
import json
import numpy as np
import pyarrow as pa
from gradlab.play_trajectory import encode_tree, decode_tree
from gradlab.json_utils import canonical_json_bytes as canonical


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
