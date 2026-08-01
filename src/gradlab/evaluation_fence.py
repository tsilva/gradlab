from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from gradlab.json_utils import canonical_json_sha256


EVALUATION_SELECTION_FENCE_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def evaluation_selection_fence(
    *,
    run_id: str,
    checkpoints: Sequence[Mapping[str, Any]],
) -> str:
    identities = []
    for checkpoint in checkpoints:
        checkpoint_id = str(checkpoint.get("checkpoint_id") or "")
        sha256 = str(checkpoint.get("sha256") or "").strip().lower()
        if (
            not checkpoint_id
            or EVALUATION_SELECTION_FENCE_PATTERN.fullmatch(sha256) is None
        ):
            raise ValueError("checkpoint selection contains an invalid identity")
        identities.append(
            {
                "checkpoint_id": checkpoint_id,
                "sha256": sha256,
            }
        )
    return canonical_json_sha256(
        {
            "schema_version": 1,
            "run_id": str(run_id),
            "checkpoints": sorted(
                identities,
                key=lambda item: (item["checkpoint_id"], item["sha256"]),
            ),
        }
    )


__all__ = [
    "EVALUATION_SELECTION_FENCE_PATTERN",
    "evaluation_selection_fence",
]
