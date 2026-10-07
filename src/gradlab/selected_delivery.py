"""Common behavioral boundary for a Run's one selected metrics service."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from gradlab.clock import Clock, SystemClock
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


class MetricsWriter:
    """Recover and drain one durable outbox through its selected service."""

    def __init__(
        self, store, delivery: DeliveryAdapter, *, heartbeat=None, clock: Clock | None = None
    ):
        self.store, self.delivery = store, delivery
        self.heartbeat = heartbeat or (lambda: None)
        self.clock = clock or SystemClock()

    def recover(self, remote_high_water: int, *, backfill: bool = False) -> None:
        self.store.reset_interrupted_metric_frames()
        with self.store.connection() as connection:
            if backfill:
                connection.execute(
                    "UPDATE metric_frames SET status = 'pending' WHERE status = 'local_only'"
                )
            connection.execute(
                "UPDATE metric_frames SET status = 'pending' WHERE status = 'published' AND id > ?",
                (remote_high_water,),
            )
            if backfill:
                connection.execute(
                    "UPDATE metric_frames SET status = 'published' WHERE id <= ?",
                    (remote_high_water,),
                )

    def publish(self, *, limit: int = 100, event_seq_offset: int = 0, should_continue=None) -> int:
        return publish_outbox(
            self.store,
            self.delivery,
            limit=limit,
            event_seq_offset=event_seq_offset,
            heartbeat=self.heartbeat,
            should_continue=should_continue,
        )

    @property
    def high_water(self) -> int:
        with self.store.connection() as connection:
            return int(
                connection.execute(
                    "SELECT COALESCE(MAX(id), 0) FROM metric_frames WHERE status = 'published'"
                ).fetchone()[0]
            )

    def drain(self, *, timeout_seconds: float) -> int:
        deadline = self.clock.monotonic() + timeout_seconds
        while self.store.pending_metric_frames(limit=1):
            if self.clock.monotonic() >= deadline:
                raise TimeoutError("metrics outbox did not drain within its deadline")
            if not self.publish():
                self.clock.sleep(1)
        return self.high_water

    def confirm(self, high_water: int, *, timeout_seconds: float = 60) -> None:
        deadline = self.clock.monotonic() + timeout_seconds
        while True:
            self.heartbeat()
            if self.delivery.remote_high_water() >= high_water:
                return
            if self.clock.monotonic() >= deadline:
                raise TimeoutError("metrics service has not confirmed all metric frames")
            self.clock.sleep(2)


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
