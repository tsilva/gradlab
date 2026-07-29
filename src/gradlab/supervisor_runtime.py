from __future__ import annotations

import os
import shutil
import signal
import subprocess
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from gradlab.clock import Clock, SystemClock
from gradlab.metric_names import METRICS_SCHEMA_VERSION
from gradlab.metric_store import MetricStore
from gradlab.run_contracts import TerminalReceipt
from gradlab.runtime_contract import runtime_contract
from gradlab.wandb_publisher import (
    WandbProjector,
    publish_pending_frames,
    publish_promotion_summary,
    publish_terminal_summary,
)


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
        projector: WandbProjector,
        *,
        limit: int,
        event_seq_offset: int = 0,
    ) -> int:
        return publish_pending_frames(
            store,
            projector.run,
            limit=limit,
            event_seq_offset=event_seq_offset,
            metrics_schema_version=projector.metrics_schema_version,
        )

    def publish_promotion(
        self,
        projector: WandbProjector,
        *,
        checkpoint_step: int,
        checkpoint_url: str,
        metrics: Mapping[str, Any],
        updated_at: str,
        selection_rank: Sequence[str],
        evaluation_source: str,
        metrics_schema_version: int = METRICS_SCHEMA_VERSION,
    ) -> None:
        publish_promotion_summary(
            projector.run,
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
        projector = WandbProjector.resume(
            train_config,
            update_finish_state=True,
        )
        try:
            publish_terminal_summary(projector.run, receipt)
        finally:
            projector.close(
                timeout_seconds=timeout_seconds,
                exit_code=0 if receipt.state == "succeeded" else 1,
            )

    def remote_summary(self, run_path: str) -> dict[str, Any]:
        import wandb

        api = wandb.Api(timeout=10)
        flush = getattr(api, "flush", None)
        if callable(flush):
            flush()
        return dict(getattr(api.run(run_path), "summary", {}) or {})

    def close_wandb(
        self,
        projector: WandbProjector,
        *,
        timeout_seconds: float,
    ) -> None:
        projector.close(timeout_seconds=timeout_seconds)

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
        learner.send_signal(getattr(signal, "SIGUSR1", signal.SIGTERM))

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
