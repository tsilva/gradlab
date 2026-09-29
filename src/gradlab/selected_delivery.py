"""Common behavioral boundary for a Run's one selected metrics service."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from gradlab.metric_names import METRICS_SCHEMA_VERSION


@runtime_checkable
class DeliveryAdapter(Protocol):
    """Service operations used by the lease-holding Run writer."""

    @property
    def run_id(self) -> str: ...

    def publish_frame(
        self,
        row: Mapping[str, Any],
        *,
        event_seq_offset: int = 0,
        occupancy_page: Any = None,
    ) -> None: ...

    def publish_promotion(
        self,
        *,
        checkpoint_step: int,
        checkpoint_url: str,
        metrics: Mapping[str, Any],
        updated_at: str,
        selection_rank: Sequence[str],
        evaluation_source: str,
        metrics_schema_version: int = METRICS_SCHEMA_VERSION,
    ) -> None: ...

    def publish_terminal(
        self, *, state: str, reason: str, timeout_seconds: float | None = None
    ) -> None: ...

    def remote_summary(self) -> dict[str, Any]: ...

    def remote_high_water(self) -> int: ...

    def finish_projection(self, *, timeout_seconds: float) -> None: ...


def publish_outbox(
    store: Any,
    delivery: DeliveryAdapter,
    *,
    limit: int,
    event_seq_offset: int = 0,
    heartbeat: Callable[[], None] | None = None,
    should_continue: Callable[[], bool] | None = None,
) -> int:
    """Claim each durable frame once and let the adapter handle replay semantics."""
    published = 0
    for row in store.pending_metric_frames(limit=limit):
        if should_continue is not None and not should_continue():
            break
        if heartbeat is not None:
            heartbeat()
        frame_id = int(row["id"])
        if not store.claim_metric_frame(frame_id):
            continue
        try:
            occupancy_page = (
                store.occupancy_page(json.loads(str(row["payload_json"])))
                if row["kind"] == "occupancy"
                else None
            )
            delivery.publish_frame(
                row, event_seq_offset=event_seq_offset, occupancy_page=occupancy_page
            )
        except Exception as exc:
            store.mark_metric_frame_failed(frame_id, repr(exc))
            print(f"Metrics frame publish failed id={frame_id}: {exc}", flush=True)
            break
        store.mark_metric_frame_published(
            frame_id, step=int(row["step"]) if row.get("step") is not None else None
        )
        published += 1
        if heartbeat is not None:
            heartbeat()
    return published
