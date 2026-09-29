"""MLflow projection of GradLab's validated metric journal.

The journal sequence and the Run creation time give every metric a stable MLflow
identity. Replaying an uncertain write submits the same key, timestamp, step,
and value; the pinned SQLite-backed MLflow service retains one visible point.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mlflow.entities import Metric, RunTag
from mlflow.tracking import MlflowClient

from gradlab.evaluation_projection import validate_evaluation_metric_payload
from gradlab.metric_names import (
    EVAL_CHECKPOINT_STEP,
    LEADER_CHECKPOINT_ARTIFACT_REF,
    LEADER_CHECKPOINT_EVALUATION_SOURCE,
    LEADER_CHECKPOINT_PROJECTION_TIMESTAMP,
    LEADER_CHECKPOINT_STEP,
    METRICS_SCHEMA_VERSION,
    MONITORING_SCALAR_METRICS,
    ORCHESTRATION_EVENT_SEQUENCE,
    ORCHESTRATION_RUN_TERMINAL_REASON,
    ORCHESTRATION_RUN_TERMINAL_STATE,
    leader_metric_for_rank_metric,
    require_current_metrics_schema,
    validate_metric_payload,
)
from gradlab.ranking import require_objective_rank


RUN_ID_TAG = "gradlab.run_id"


class MlflowDelivery:
    """One lease-owned service binding for a GradLab Run."""

    def __init__(
        self,
        client: MlflowClient,
        *,
        run_id: str,
        gradlab_run_id: str,
        created_at_ms: int,
        tracking_uri: str = "",
        experiment_id: str = "",
    ) -> None:
        self.client = client
        self.run_id = run_id
        self.gradlab_run_id = gradlab_run_id
        self.created_at_ms = int(created_at_ms)
        self.tracking_uri = tracking_uri
        self.experiment_id = experiment_id

    @property
    def service_url(self) -> str:
        if not self.tracking_uri.startswith(("http://", "https://")):
            return ""
        return (
            f"{self.tracking_uri.rstrip('/')}/#/experiments/"
            f"{self.experiment_id}/runs/{self.run_id}"
        )

    @classmethod
    def open(
        cls,
        *,
        tracking_uri: str,
        experiment_name: str,
        gradlab_run_id: str,
        created_at_ms: int,
    ) -> MlflowDelivery:
        if re.fullmatch(r"gradlab-[0-9a-f]{32}", gradlab_run_id) is None:
            raise ValueError("MLflow binding requires a GradLab Run ID")
        if not tracking_uri or not experiment_name:
            raise ValueError("MLflow tracking URI and experiment name are required")
        client = MlflowClient(tracking_uri=tracking_uri)
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            try:
                experiment_id = client.create_experiment(experiment_name)
            except Exception:
                experiment = client.get_experiment_by_name(experiment_name)
                if experiment is None:
                    raise
                experiment_id = experiment.experiment_id
        else:
            experiment_id = experiment.experiment_id
        # The lease gives this Run one writer. Searching by immutable GradLab ID
        # also reconciles a create that committed before its reply was lost.
        matches = client.search_runs(
            [experiment_id],
            filter_string=f"tags.`{RUN_ID_TAG}` = '{gradlab_run_id}'",
            max_results=2,
        )
        if len(matches) > 1:
            raise RuntimeError("MLflow has multiple bindings for one GradLab Run")
        if matches:
            run_id = matches[0].info.run_id
        else:
            run_id = client.create_run(
                experiment_id,
                tags={RUN_ID_TAG: gradlab_run_id, "mlflow.runName": gradlab_run_id},
            ).info.run_id
        return cls(
            client,
            run_id=run_id,
            gradlab_run_id=gradlab_run_id,
            created_at_ms=created_at_ms,
            tracking_uri=tracking_uri,
            experiment_id=experiment_id,
        )

    def _metric_batch(self, values: Mapping[str, Any], *, sequence: int, step: int) -> None:
        payload = dict(values)
        payload[ORCHESTRATION_EVENT_SEQUENCE] = float(sequence)
        timestamp = self.created_at_ms + sequence
        metrics = [
            Metric(
                key=key,
                value=float(value),
                timestamp=timestamp,
                step=sequence if key == ORCHESTRATION_EVENT_SEQUENCE else step,
            )
            for key, value in payload.items()
        ]
        self.client.log_batch(self.run_id, metrics=metrics, synchronous=True)

    def _artifact(self, *, sequence: int, name: str, content: bytes) -> None:
        with tempfile.TemporaryDirectory(prefix="gradlab-mlflow-") as temporary:
            path = Path(temporary) / name
            path.write_bytes(content)
            artifact_path = f"journal/{sequence}"
            self.client.log_artifact(self.run_id, str(path), artifact_path=artifact_path)
            remote = Path(
                self.client.download_artifacts(
                    self.run_id, f"{artifact_path}/{name}", dst_path=temporary
                )
            )
            if hashlib.sha256(remote.read_bytes()).digest() != hashlib.sha256(content).digest():
                raise RuntimeError("MLflow media read-back disagrees with the journal")

    def publish_frame(
        self,
        row: Mapping[str, Any],
        *,
        event_seq_offset: int = 0,
        occupancy_page: Any = None,
    ) -> None:
        del occupancy_page
        sequence = int(row["id"]) + int(event_seq_offset)
        step = int(row.get("step") or 0)
        kind = str(row["kind"])
        payload = json.loads(str(row["payload_json"]))
        source = str(row.get("source") or "")
        if kind == "history":
            if source.startswith("eval"):
                validate_evaluation_metric_payload(
                    payload, schema_version=METRICS_SCHEMA_VERSION
                )
                payload[EVAL_CHECKPOINT_STEP] = float(step)
            else:
                validate_metric_payload(payload)
                if not source.startswith("orchestration"):
                    payload["train/step"] = float(step)
            self._metric_batch(payload, sequence=sequence, step=step)
            return
        if kind == "monitoring":
            metrics = dict(payload["metrics"])
            if not set(metrics).issubset(MONITORING_SCALAR_METRICS):
                raise ValueError("monitoring cannot project Acceptance metrics")
            metrics[EVAL_CHECKPOINT_STEP] = float(step)
            self._metric_batch(metrics, sequence=sequence, step=step)
            video = payload.get("video")
            if video is not None:
                self._r2_artifact(
                    sequence=sequence,
                    video={**video, "bucket_uri": payload["bucket_uri"]},
                    name="representative.mp4",
                )
            return
        if kind == "evaluation_video":
            self._metric_batch({EVAL_CHECKPOINT_STEP: float(step)}, sequence=sequence, step=step)
            self._r2_artifact(sequence=sequence, video=payload, name="episode.mp4")
            return
        if kind in {"curriculum_distribution", "occupancy", "eval_by_start"}:
            axis = EVAL_CHECKPOINT_STEP if kind == "eval_by_start" else "train/step"
            self._metric_batch({axis: float(step)}, sequence=sequence, step=step)
            self._artifact(
                sequence=sequence,
                name=f"{kind}.json",
                content=json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),
            )
            return
        raise ValueError(f"unsupported supervisor telemetry frame kind: {kind}")

    def _r2_artifact(self, *, sequence: int, video: Mapping[str, Any], name: str) -> None:
        from gradlab.r2_store import BucketConfig, R2Bucket

        bucket_uri = str(video["bucket_uri"])
        config = (
            BucketConfig(uri=bucket_uri)
            if bucket_uri.startswith("file://")
            else BucketConfig.from_env(
                "GRADLAB_EVAL_R2" if name == "episode.mp4" else "GRADLAB_MODELS_R2"
            )
        )
        if config.uri != bucket_uri:
            raise ValueError("MLflow media bucket differs from configured canonical storage")
        size = int(video["bytes"])
        if size < 1 or size > 256 * 1024**2:
            raise ValueError("MLflow video size is outside the declared bound")
        with tempfile.TemporaryDirectory(prefix="gradlab-mlflow-video-") as temporary:
            path = Path(temporary) / name
            R2Bucket(config).download_verified(
                str(video["key"]), path, size=size, sha256=str(video["sha256"])
            )
            self._artifact(sequence=sequence, name=name, content=path.read_bytes())

    def remote_high_water(self) -> int:
        run = self.client.get_run(self.run_id)
        if run.data.tags.get(RUN_ID_TAG) != self.gradlab_run_id:
            raise RuntimeError("MLflow Run binding changed")
        return int(run.data.metrics.get(ORCHESTRATION_EVENT_SEQUENCE, 0))

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
    ) -> None:
        require_current_metrics_schema(metrics_schema_version)
        criteria = require_objective_rank(
            selection_rank, metrics_schema_version=METRICS_SCHEMA_VERSION
        )
        projection = {
            LEADER_CHECKPOINT_STEP: str(checkpoint_step),
            LEADER_CHECKPOINT_ARTIFACT_REF: checkpoint_url,
            LEADER_CHECKPOINT_EVALUATION_SOURCE: evaluation_source,
            LEADER_CHECKPOINT_PROJECTION_TIMESTAMP: updated_at,
        }
        for criterion in criteria:
            name = leader_metric_for_rank_metric(
                criterion.metric, schema_version=METRICS_SCHEMA_VERSION
            )
            if name == LEADER_CHECKPOINT_STEP:
                continue
            value = metrics.get(criterion.metric)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"promoted checkpoint lacks rank metric: {criterion.metric}")
            projection[name] = str(float(value))
        self.client.log_batch(
            self.run_id,
            tags=[RunTag(key=key, value=value) for key, value in projection.items()],
            synchronous=True,
        )

    def remote_summary(self) -> dict[str, Any]:
        run = self.client.get_run(self.run_id)
        if run.data.tags.get(RUN_ID_TAG) != self.gradlab_run_id:
            raise RuntimeError("MLflow Run binding changed")
        summary: dict[str, Any] = dict(run.data.tags)
        summary.update(run.data.metrics)
        for key, value in tuple(summary.items()):
            if key.startswith("leader/") and key != LEADER_CHECKPOINT_ARTIFACT_REF:
                try:
                    summary[key] = float(value)
                except (TypeError, ValueError):
                    pass
        return summary

    def publish_terminal(
        self, *, state: str, reason: str, timeout_seconds: float | None = None
    ) -> None:
        del timeout_seconds
        self.client.set_tag(self.run_id, ORCHESTRATION_RUN_TERMINAL_STATE, state)
        self.client.set_tag(self.run_id, ORCHESTRATION_RUN_TERMINAL_REASON, reason)
        self.close(success=state in {"succeeded", "stopped", "complete_local"})

    def close(self, *, success: bool) -> None:
        self.client.set_terminated(
            self.run_id, status="FINISHED" if success else "FAILED"
        )

    def finish_projection(self, *, timeout_seconds: float) -> None:
        del timeout_seconds


def publish_pending_frames(
    store,
    delivery: MlflowDelivery,
    *,
    limit: int,
    event_seq_offset: int = 0,
    heartbeat=None,
    should_continue=None,
) -> int:
    from gradlab.selected_delivery import publish_outbox

    return publish_outbox(
        store, delivery, limit=limit, event_seq_offset=event_seq_offset,
        heartbeat=heartbeat, should_continue=should_continue,
    )
