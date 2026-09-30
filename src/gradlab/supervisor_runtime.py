from __future__ import annotations

import os
from datetime import datetime
import shutil
import signal
import subprocess
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread
from typing import Any, Protocol

from gradlab.clock import Clock, SystemClock
from gradlab.metric_names import METRICS_SCHEMA_VERSION
from gradlab.metric_store import MetricStore
from gradlab.mlflow_delivery import MlflowDelivery
from gradlab.run_contracts import TerminalReceipt
from gradlab.runtime_contract import runtime_contract
from gradlab.selected_delivery import DeliveryAdapter, MetricsWriter
from gradlab.wandb_publisher import WandbProjector


class LearnerProcess(Protocol):
    pid: int

    def poll(self) -> int | None: ...

    def wait(self) -> int: ...

    def send_signal(self, signal_number: int) -> None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


class LifecycleObserver(Protocol):
    def emit(self, kind: str, payload: Mapping[str, Any]) -> None: ...


class NullLifecycleObserver:
    def emit(self, kind: str, payload: Mapping[str, Any]) -> None:
        del kind, payload


class SupervisorRuntime:
    """Replaceable process, SDK, clock, and host boundary for one supervisor."""

    def __init__(self, *, clock: Clock | None = None) -> None:
        self.clock = clock or SystemClock()

    @contextmanager
    def maintain_lease(self, renew: Callable[[], None]) -> Iterator[None]:
        """Keep writer renewal independent of blocking storage and SDK calls."""
        stopped = Event()

        def work() -> None:
            while not stopped.wait(1.0):
                renew()

        worker = Thread(target=work, name="gradlab-writer-lease", daemon=True)
        worker.start()
        try:
            yield
        finally:
            stopped.set()
            worker.join()

    def runtime_contract(self, *, runtime_image_ref: str) -> dict[str, Any]:
        return runtime_contract(runtime_image_ref=runtime_image_ref)

    def holder_id(self) -> str:
        return f"{uuid.uuid4().hex}@{os.uname().nodename}"

    def disk_usage(self, path: Path) -> Any:
        return shutil.disk_usage(path)

    def start_wandb(
        self,
        train_config: Mapping[str, Any],
        *,
        run_dir: str,
        config: Any,
        goal_variant: Mapping[str, Any] | None = None,
    ) -> WandbProjector:
        return WandbProjector.start_live(
            train_config,
            run_dir=run_dir,
            config=config,
            goal_variant=goal_variant,
        )

    def start_mlflow(
        self,
        train_config: Mapping[str, Any],
        *,
        created_at: str,
    ) -> MlflowDelivery:
        uri = str(os.environ.get("MLFLOW_TRACKING_URI") or "").strip()
        if not uri:
            raise RuntimeError("selected MLflow service has no private tracking URI")
        created_at_ms = int(datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp() * 1000)
        return MlflowDelivery.open(
            tracking_uri=uri,
            experiment_name=f"gradlab-{train_config['game_family']}",
            gradlab_run_id=str(train_config["wandb_run_id"]),
            created_at_ms=created_at_ms,
        )

    def resume_wandb(
        self,
        train_config: Mapping[str, Any],
        *,
        allow_create: bool,
        update_finish_state: bool = True,
    ) -> WandbProjector:
        return WandbProjector.resume(
            train_config,
            allow_create=allow_create,
            update_finish_state=update_finish_state,
        )

    def publish_frames(
        self,
        store: MetricStore,
        projector: DeliveryAdapter,
        *,
        limit: int,
        event_seq_offset: int = 0,
        heartbeat: Callable[[], None] | None = None,
        should_continue: Callable[[], bool] | None = None,
    ) -> int:
        return MetricsWriter(store, projector, heartbeat=heartbeat, clock=self.clock).publish(
            limit=limit, event_seq_offset=event_seq_offset, should_continue=should_continue
        )

    def publish_promotion(
        self,
        projector: DeliveryAdapter,
        *,
        checkpoint_step: int,
        checkpoint_url: str,
        metrics: Mapping[str, Any],
        updated_at: str,
        selection_rank: Sequence[str],
        evaluation_source: str,
        metrics_schema_version: int = METRICS_SCHEMA_VERSION,
    ) -> None:
        projector.publish_promotion(
            checkpoint_step=checkpoint_step,
            checkpoint_url=checkpoint_url,
            metrics=metrics,
            updated_at=updated_at,
            selection_rank=selection_rank,
            evaluation_source=evaluation_source,
            metrics_schema_version=metrics_schema_version,
        )

    def publish_terminal(
        self,
        train_config: Mapping[str, Any],
        receipt: TerminalReceipt,
        *,
        timeout_seconds: float,
    ) -> None:
        receipt.validate()
        tracking = train_config.get("tracking") or {}
        projector: DeliveryAdapter = (
            self.start_mlflow(
                train_config, created_at=str(train_config["tracking_created_at"])
            )
            if tracking.get("backend") == "mlflow"
            else WandbProjector.resume(train_config, update_finish_state=True)
        )
        projector.publish_terminal(
            state=receipt.state,
            reason=receipt.stop_reason,
            timeout_seconds=timeout_seconds,
        )

    def remote_summary(self, run_path: str) -> dict[str, Any]:
        if run_path.startswith("mlflow:"):
            uri = str(os.environ.get("MLFLOW_TRACKING_URI") or "").strip()
            if not uri:
                raise RuntimeError("selected MLflow service has no private tracking URI")
            from mlflow.tracking import MlflowClient

            client = MlflowClient(tracking_uri=uri)
            run = client.get_run(run_path.removeprefix("mlflow:"))
            delivery = MlflowDelivery(
                client,
                run_id=run.info.run_id,
                gradlab_run_id=str(run.data.tags["gradlab.run_id"]),
                created_at_ms=0,
            )
            return delivery.remote_summary()
        import wandb

        api = wandb.Api(timeout=10)
        flush = getattr(api, "flush", None)
        if callable(flush):
            flush()
        return dict(getattr(api.run(run_path), "summary", {}) or {})

    def close_wandb(
        self,
        projector: DeliveryAdapter,
        *,
        timeout_seconds: float,
    ) -> None:
        projector.finish_projection(timeout_seconds=timeout_seconds)

    def start_learner(
        self,
        command: Sequence[str],
        *,
        log_path: Path,
        environment: Mapping[str, str],
    ) -> LearnerProcess:
        log = log_path.open("a", encoding="utf-8")
        learner = subprocess.Popen(
            list(command),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=dict(environment),
            start_new_session=True,
        )
        learner._gradlab_log = log  # type: ignore[attr-defined]
        learner._gradlab_process_group_id = learner.pid  # type: ignore[attr-defined]
        return learner

    def request_learner_stop(self, learner: LearnerProcess) -> None:
        try:
            os.kill(int(learner.pid), getattr(signal, "SIGUSR1", signal.SIGTERM))
        except ProcessLookupError:
            return

    @staticmethod
    def _learner_process_group_id(learner: LearnerProcess) -> int:
        recorded = getattr(learner, "_gradlab_process_group_id", None)
        if isinstance(recorded, int) and recorded > 0:
            return recorded
        return os.getpgid(int(learner.pid))

    def learner_group_alive(self, learner: LearnerProcess) -> bool:
        try:
            os.killpg(self._learner_process_group_id(learner), 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def terminate_learner_group(self, learner: LearnerProcess) -> None:
        try:
            os.killpg(self._learner_process_group_id(learner), signal.SIGTERM)
        except ProcessLookupError:
            return

    def kill_learner_group(self, learner: LearnerProcess) -> None:
        try:
            os.killpg(self._learner_process_group_id(learner), signal.SIGKILL)
        except ProcessLookupError:
            return

    def install_cancel_handlers(
        self,
        callback: Callable[[int, Any], None],
    ) -> tuple[Any, Any]:
        previous = (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT))
        signal.signal(signal.SIGTERM, callback)
        signal.signal(signal.SIGINT, callback)
        return previous

    def restore_cancel_handlers(self, token: tuple[Any, Any]) -> None:
        signal.signal(signal.SIGTERM, token[0])
        signal.signal(signal.SIGINT, token[1])
