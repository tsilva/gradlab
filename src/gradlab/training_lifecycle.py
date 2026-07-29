from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from gradlab.early_stop import MetricEarlyStopStateMachine, MetricSample
from gradlab.file_utils import atomic_write_json
from gradlab.metric_names import (
    TRAIN_ARTIFACT_SAVE_SECONDS,
    TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN,
    TRAIN_OUTCOME_SUCCESS_CURRENT_RATE_MEAN,
    train_early_stop_metric,
    validate_metric_payload,
)
from gradlab.training_metrics import EpisodeMetricsReducer, episode_succeeded


TRAINING_RESULT_FILENAME = "training-result.json"


class TerminalReason(StrEnum):
    FIRST_COMPLETION = "first_completion"
    TRAINING_ACCEPTANCE = "deterministic_training_acceptance"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    EARLY_STOP_FAILURE = "early_stop_failure"
    EARLY_STOP_SUCCESS = "early_stop_success"
    LOCAL_INTERRUPTION = "local_interruption"
    EXTERNAL_SIGNAL = "external_signal"
    FAILED = "failed"


@dataclass(frozen=True)
class TrainingResult:
    terminal_reason: TerminalReason
    execution_mode: TrainingExecutionMode
    execution_policy: Mapping[str, Any]
    first_completion_step: int | None
    final_step: int
    requested_limit: int
    execution_limit: int
    model_kind: str
    model_path: str = "final_model.zip"

    @property
    def status(self) -> str:
        return (
            "interrupted"
            if self.terminal_reason
            in {TerminalReason.LOCAL_INTERRUPTION, TerminalReason.EXTERNAL_SIGNAL}
            else "completed"
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "document_type": "gradlab.training-result",
            "format_version": 2,
            "status": self.status,
            "terminal_reason": self.terminal_reason.value,
            "execution_mode": self.execution_mode.value,
            "execution_policy": dict(self.execution_policy),
            "first_completion_step": self.first_completion_step,
            "final_step": int(self.final_step),
            "requested_limit": int(self.requested_limit),
            "execution_limit": int(self.execution_limit),
            "model_kind": self.model_kind,
            "model": self.model_path,
        }


class TrainingExecutionMode(StrEnum):
    LOCAL_DEMO = "local-demo"
    SUPERVISED = "supervised"


@dataclass(frozen=True)
class TrainingExecutionPolicy:
    mode: TrainingExecutionMode
    console_mode: str
    persist_intermediate_checkpoints: bool
    stop_on_first_completion: bool
    handle_sigint: bool

    @classmethod
    def for_mode(
        cls,
        mode: TrainingExecutionMode | str,
    ) -> TrainingExecutionPolicy:
        resolved = TrainingExecutionMode(mode)
        if resolved == TrainingExecutionMode.LOCAL_DEMO:
            return cls(
                mode=resolved,
                console_mode="auto",
                persist_intermediate_checkpoints=False,
                stop_on_first_completion=True,
                handle_sigint=True,
            )
        return cls(
            mode=resolved,
            console_mode="plain",
            persist_intermediate_checkpoints=True,
            stop_on_first_completion=False,
            handle_sigint=False,
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "console_mode": self.console_mode,
            "persist_intermediate_checkpoints": self.persist_intermediate_checkpoints,
            "stop_on_first_completion": self.stop_on_first_completion,
            "handle_sigint": self.handle_sigint,
        }


@dataclass(frozen=True)
class TrainingBudget:
    requested_limit: int
    execution_total: int
    step_quantum: int
    initial_step: int = 0

    @classmethod
    def aligned(
        cls,
        *,
        requested_limit: int,
        step_quantum: int,
        initial_step: int = 0,
    ) -> TrainingBudget:
        requested = int(requested_limit)
        quantum = int(step_quantum)
        initial = int(initial_step)
        if requested < 0 or quantum <= 0 or initial < 0:
            raise ValueError("training budget values are invalid")
        remaining = max(requested - initial, 0)
        execution_total = initial + (math.ceil(remaining / quantum) * quantum if remaining else 0)
        return cls(
            requested_limit=requested,
            execution_total=execution_total,
            step_quantum=quantum,
            initial_step=initial,
        )


class ProgressSink(Protocol):
    def start(
        self,
        *,
        total: int,
        initial: int,
        description: str,
        fields: Sequence[ProgressField] = (),
    ) -> None: ...

    def update(
        self,
        *,
        step: int,
        metrics: Mapping[str, int | float],
        final: bool = False,
    ) -> None: ...

    def event(self, message: str) -> None: ...

    def close(self) -> None: ...


class ProgressValueFormat(StrEnum):
    NUMBER = "number"
    COUNT = "count"
    PERCENT = "percent"
    BYTES = "bytes"


@dataclass(frozen=True)
class ProgressField:
    metric: str
    label: str
    value_format: ProgressValueFormat = ProgressValueFormat.NUMBER

    def __post_init__(self) -> None:
        if not self.metric.strip():
            raise ValueError("progress field metric must be non-empty")
        if not self.label.strip():
            raise ValueError("progress field label must be non-empty")


COMMON_PROGRESS_FIELDS = (
    ProgressField(
        TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN,
        "mean return",
    ),
    ProgressField(
        TRAIN_OUTCOME_SUCCESS_CURRENT_RATE_MEAN,
        "completion",
        ProgressValueFormat.PERCENT,
    ),
)


def _compact_count(value: float) -> str:
    magnitude = abs(value)
    for scale, suffix in (
        (1_000_000_000_000.0, "T"),
        (1_000_000_000.0, "G"),
        (1_000_000.0, "M"),
        (1_000.0, "k"),
    ):
        if magnitude >= scale:
            return f"{value / scale:.3g}{suffix}"
    return f"{value:.0f}"


def _compact_bytes(value: float) -> str:
    magnitude = abs(value)
    for scale, suffix in (
        (1024.0**4, "TiB"),
        (1024.0**3, "GiB"),
        (1024.0**2, "MiB"),
        (1024.0, "KiB"),
    ):
        if magnitude >= scale:
            return f"{value / scale:.3g} {suffix}"
    return f"{value:.0f} B"


def format_progress_value(
    value: int | float | None,
    value_format: ProgressValueFormat,
) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    numeric = float(value)
    if value_format == ProgressValueFormat.COUNT:
        return _compact_count(numeric)
    if value_format == ProgressValueFormat.BYTES:
        return _compact_bytes(numeric)
    if value_format == ProgressValueFormat.PERCENT:
        return f"{100.0 * numeric:.2f}%"
    return f"{numeric:.3g}"


def resolve_progress_fields(fields: Sequence[ProgressField]) -> tuple[ProgressField, ...]:
    # Put backend-owned fields first so the most decision-relevant telemetry
    # remains visible when a terminal layout must compact.
    resolved = (*tuple(fields), *COMMON_PROGRESS_FIELDS)
    metrics = [field.metric for field in resolved]
    labels = [field.label for field in resolved]
    if len(metrics) != len(set(metrics)):
        raise ValueError("progress fields must use unique metrics")
    if len(labels) != len(set(labels)):
        raise ValueError("progress fields must use unique labels")
    return resolved


def _progress_postfix(
    metrics: Mapping[str, int | float],
    fields: Sequence[ProgressField] = (),
) -> dict[str, str]:
    return {
        field.label: format_progress_value(
            metrics.get(field.metric),
            field.value_format,
        )
        for field in resolve_progress_fields(fields)
    }


class PlainProgressSink:
    def __init__(
        self,
        *,
        interval_seconds: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.interval_seconds = float(interval_seconds)
        self.clock = clock
        self.total = 0
        self.description = "training"
        self.last_printed_at: float | None = None
        self.fields: tuple[ProgressField, ...] = ()

    def start(
        self,
        *,
        total: int,
        initial: int,
        description: str,
        fields: Sequence[ProgressField] = (),
    ) -> None:
        self.total = int(total)
        self.description = description
        self.fields = tuple(fields)
        self._print(initial, {})

    def _print(self, step: int, metrics: Mapping[str, int | float]) -> None:
        postfix = _progress_postfix(metrics, self.fields)
        fraction = min(max(step / self.total, 0.0), 1.0) if self.total else 1.0
        print(
            f"{self.description} progress: {step:,}/{self.total:,} ({fraction:.1%}) "
            + " ".join(f"{label}={value}" for label, value in postfix.items()),
            flush=True,
        )
        self.last_printed_at = self.clock()

    def update(
        self,
        *,
        step: int,
        metrics: Mapping[str, int | float],
        final: bool = False,
    ) -> None:
        now = self.clock()
        if (
            final
            or self.last_printed_at is None
            or now - self.last_printed_at >= self.interval_seconds
        ):
            self._print(step, metrics)

    def event(self, message: str) -> None:
        print(message, flush=True)

    def close(self) -> None:
        return None


class SilentProgressSink:
    def start(
        self,
        *,
        total: int,
        initial: int,
        description: str,
        fields: Sequence[ProgressField] = (),
    ) -> None:
        del total, initial, description, fields

    def update(
        self,
        *,
        step: int,
        metrics: Mapping[str, int | float],
        final: bool = False,
    ) -> None:
        del step, metrics, final

    def event(self, message: str) -> None:
        del message

    def close(self) -> None:
        return None


def progress_sink_for_mode(mode: str) -> ProgressSink:
    if mode in {"auto", "interactive", "plain"}:
        return PlainProgressSink()
    return SilentProgressSink()


class MetricFrameSink(Protocol):
    def publish(self, payload: Mapping[str, int | float], *, step: int) -> None: ...


class DirectMetricFrameSink:
    def __init__(self, metric_store: Any, *, publish: bool) -> None:
        self.metric_store = metric_store
        self.publish_enabled = bool(publish)

    def publish(self, payload: Mapping[str, int | float], *, step: int) -> None:
        document = dict(payload)
        validate_metric_payload(document)
        self.metric_store.append_metrics(
            document,
            step=int(step),
            source="train",
            publish=self.publish_enabled,
        )


class LoggerMetricFrameSink:
    def __init__(self, logger: Any) -> None:
        self.logger = logger

    def publish(self, payload: Mapping[str, int | float], *, step: int) -> None:
        del step
        for name, value in payload.items():
            self.logger.record(name, value)


class CheckpointCoordinator:
    def __init__(
        self,
        *,
        metric_store: Any,
        run_name: str,
        eval_required: bool,
        publish: bool,
        persist_intermediate: bool,
        event: Callable[[str], None],
    ) -> None:
        self.metric_store = metric_store
        self.run_name = run_name
        self.eval_required = bool(eval_required)
        self.publish = bool(publish)
        self.persist_intermediate = bool(persist_intermediate)
        self.event = event
        self._saved: dict[tuple[str, int], Path] = {}
        self._terminal: Path | None = None

    def save(
        self,
        *,
        kind: str,
        step: int,
        model_path: Path,
        save_bundle: Callable[[Path, str, int], Path],
        terminal: bool = False,
    ) -> Path | None:
        if not terminal and not self.persist_intermediate:
            return None
        if terminal and self._terminal is not None:
            raise RuntimeError(f"terminal training artifact already exists: {self._terminal}")
        key = (str(kind), int(step))
        if key in self._saved:
            return self._saved[key]
        model_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        installed = save_bundle(model_path, kind, int(step))
        checkpoint_id = self.metric_store.record_checkpoint(
            run_name=self.run_name,
            kind=kind,
            step=int(step),
            path=installed,
            sha256=None,
            eval_required=self.eval_required,
        )
        self.metric_store.append_metrics(
            {TRAIN_ARTIFACT_SAVE_SECONDS: time.perf_counter() - started},
            step=int(step),
            source=f"checkpoint-save:{kind}",
            publish=self.publish,
        )
        self._saved[key] = installed
        if terminal:
            self._terminal = installed
        self.event(f"{kind} model ready: id={checkpoint_id} step={step} path={installed}")
        return installed


class MetricStopController:
    def __init__(
        self,
        *,
        config: Any,
        decision_path: Path,
        stop_flag: Any,
        event: Callable[[str], None],
    ) -> None:
        self.machine = MetricEarlyStopStateMachine(config, label="early_stop") if config else None
        self.decision_path = decision_path
        self.stop_flag = stop_flag
        self.event = event
        self.decision: Mapping[str, Any] | None = None

    def evaluate(
        self,
        payload: Mapping[str, int | float],
        *,
        step: int,
    ) -> dict[str, int | float]:
        if (
            self.machine is None
            or self.decision is not None
            or (
                self.stop_flag.requested
                and not str(self.stop_flag.reason).startswith("early_stop:")
            )
        ):
            return {}
        watched = {str(condition["metric"]) for condition in self.machine.conditions.values()}
        samples = {
            name: MetricSample(value=float(payload[name]), step=int(step))
            for name in watched
            if name in payload and math.isfinite(float(payload[name]))
        }
        update = self.machine.update(samples)
        metrics: dict[str, int | float] = {}
        for condition_id, observation in update.observations.items():
            metrics.update(
                {
                    train_early_stop_metric(condition_id, "value"): observation.value,
                    train_early_stop_metric(condition_id, "best"): observation.best_value,
                    train_early_stop_metric(
                        condition_id, "patience/elapsed_steps"
                    ): observation.elapsed_steps,
                    train_early_stop_metric(
                        condition_id, "patience/progress"
                    ): observation.patience_progress,
                    train_early_stop_metric(condition_id, "would_trigger"): float(
                        observation.would_trigger
                    ),
                }
            )
        if update.stop_decision is not None:
            self.decision = update.stop_decision
            atomic_write_json(self.decision_path, update.stop_decision)
            self.stop_flag.request(f"early_stop:{str(update.stop_decision['condition_id'])}")
            self.event(
                "early stop: "
                f"condition={update.stop_decision['condition_id']} "
                f"outcome={update.stop_decision['outcome']} "
                f"metric={update.stop_decision['metric']} "
                f"value={float(update.stop_decision['value']):.12g} "
                f"step={int(update.stop_decision['metric_step'])}"
            )
        return metrics


class TrainingSession:
    """Enforced lifecycle facade shared by all learner backends."""

    def __init__(
        self,
        *,
        run_dir: Path,
        backend_id: str,
        metric_store: Any,
        wandb_enabled: bool,
        stop_flag: Any,
        early_stop_config: Any,
        attempt_id: str,
        reducer: EpisodeMetricsReducer,
        execution_policy: TrainingExecutionPolicy,
        completion_signal_available: bool,
        progress_sink: ProgressSink | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.backend_id = backend_id
        self.stop_flag = stop_flag
        self.reducer = reducer
        self.execution_policy = execution_policy
        self.completion_signal_available = bool(completion_signal_available)
        self.progress = progress_sink or progress_sink_for_mode(execution_policy.console_mode)
        self.metric_sink: MetricFrameSink = DirectMetricFrameSink(
            metric_store,
            publish=wandb_enabled,
        )
        self.stop_controller = MetricStopController(
            config=early_stop_config,
            decision_path=run_dir / f"early_stop_decision-{attempt_id}.json",
            stop_flag=stop_flag,
            event=self.event,
        )
        self.checkpoints = CheckpointCoordinator(
            metric_store=metric_store,
            run_name="",
            eval_required=False,
            publish=wandb_enabled,
            persist_intermediate=execution_policy.persist_intermediate_checkpoints,
            event=self.event,
        )
        self.budget: TrainingBudget | None = None
        self.current_step = 0
        self.first_completion_step: int | None = None
        self.last_report_step: int | None = None
        self.last_report_payload: dict[str, int | float] = {}
        self.progress_metrics: dict[str, int | float] = {}
        self.ready = False
        self.closed = False

    def configure_checkpoints(
        self,
        *,
        run_name: str,
        eval_required: bool,
    ) -> None:
        self.checkpoints.run_name = run_name
        self.checkpoints.eval_required = bool(eval_required)

    def configure_budget(
        self,
        *,
        requested_limit: int,
        step_quantum: int,
        initial_step: int = 0,
        progress_fields: Sequence[ProgressField] = (),
    ) -> TrainingBudget:
        if self.budget is not None:
            raise RuntimeError("training budget is already configured")
        self.budget = TrainingBudget.aligned(
            requested_limit=requested_limit,
            step_quantum=step_quantum,
            initial_step=initial_step,
        )
        self.current_step = self.budget.initial_step
        self.progress.start(
            total=self.budget.execution_total,
            initial=self.budget.initial_step,
            description=self.backend_id,
            fields=progress_fields,
        )
        return self.budget

    def set_metric_sink(self, sink: MetricFrameSink) -> None:
        self.metric_sink = sink

    def mark_ready(self) -> Path:
        if self.ready:
            raise RuntimeError("learner readiness may be emitted only once")
        path = self.run_dir / "learner_ready.json"
        atomic_write_json(
            path,
            {
                "schema_version": 1,
                "pid": os.getpid(),
                "ready_at_unix": time.time(),
                "training_backend_id": self.backend_id,
            },
        )
        self.ready = True
        return path

    def advance(
        self,
        step: int,
        records: Iterable[Any] = (),
        *,
        progress_metrics: Mapping[str, Any] | None = None,
    ) -> dict[str, int | float]:
        if self.budget is None:
            raise RuntimeError("training budget must be configured before advancing")
        current = int(step)
        if current < self.current_step:
            raise RuntimeError(f"training progress regressed from {self.current_step} to {current}")
        if current > self.budget.execution_total:
            raise RuntimeError(
                f"training progress exceeded execution total "
                f"{self.budget.execution_total}: {current}"
            )
        self.current_step = current
        metrics = self.reducer.consume(records)
        self.progress_metrics.update(self._finite_scalars(progress_metrics or {}))
        self.progress_metrics.update(metrics)
        self.progress.update(step=current, metrics=self.progress_metrics)
        return metrics

    @staticmethod
    def _finite_scalars(values: Mapping[str, Any]) -> dict[str, int | float]:
        payload: dict[str, int | float] = {}
        for name, value in values.items():
            if isinstance(value, bool):
                payload[str(name)] = int(value)
            elif isinstance(value, int | float) and math.isfinite(float(value)):
                payload[str(name)] = value
        return payload

    def report(
        self,
        *,
        step: int,
        metrics: Mapping[str, Any] | None = None,
    ) -> bool:
        canonical = self.advance(step)
        payload = self._finite_scalars(metrics or {})
        payload.update(canonical)
        if self.last_report_step == int(step):
            self.progress_metrics.update(payload)
            self.progress.update(step=int(step), metrics=self.progress_metrics)
            return self.stop_controller.decision is not None
        payload.update(self.stop_controller.evaluate(payload, step=int(step)))
        self.metric_sink.publish(payload, step=int(step))
        self.last_report_step = int(step)
        self.last_report_payload = dict(payload)
        self.progress_metrics.update(payload)
        self.progress.update(step=int(step), metrics=self.progress_metrics)
        return self.stop_controller.decision is not None

    def event(self, message: str) -> None:
        self.progress.event(message)

    def telemetry_event(self, message: str) -> None:
        if self.execution_policy.mode == TrainingExecutionMode.SUPERVISED:
            self.progress.event(message)

    def observe_completion(
        self,
        *,
        step: int,
        qualified: bool,
    ) -> bool:
        if not qualified or not self.completion_signal_available:
            return False
        if self.first_completion_step is None:
            self.first_completion_step = int(step)
            self.event(f"first target completion observed at step={int(step)}")
        if self.execution_policy.stop_on_first_completion:
            self.stop_flag.request(f"first_completion:{self.first_completion_step}")
            return True
        return False

    def observe_episode_completions(
        self,
        *,
        step: int,
        records: Iterable[Any],
    ) -> bool:
        qualified = any(
            hasattr(record, "episode_return")
            and str(getattr(record, "start_origin", "target")) == "target"
            and episode_succeeded(record)
            for record in records
        )
        return self.observe_completion(step=step, qualified=qualified)

    def terminal_reason(
        self,
        default: TerminalReason = TerminalReason.RESOURCE_EXHAUSTION,
    ) -> TerminalReason:
        if (
            self.execution_policy.stop_on_first_completion
            and self.first_completion_step is not None
        ):
            return TerminalReason.FIRST_COMPLETION
        decision = self.stop_controller.decision
        if decision is not None:
            return (
                TerminalReason.EARLY_STOP_SUCCESS
                if str(decision.get("outcome")) == "success"
                else TerminalReason.EARLY_STOP_FAILURE
            )
        if self.stop_flag.requested:
            if self.execution_policy.mode == TrainingExecutionMode.SUPERVISED:
                return TerminalReason.EXTERNAL_SIGNAL
            return TerminalReason.LOCAL_INTERRUPTION
        return default

    def terminal_model_kind(self, terminal_reason: TerminalReason) -> str:
        return "interrupted" if terminal_reason == TerminalReason.LOCAL_INTERRUPTION else "final"

    def should_persist_interrupted_checkpoint(
        self,
        terminal_reason: TerminalReason,
    ) -> bool:
        return (
            self.execution_policy.mode == TrainingExecutionMode.SUPERVISED
            and terminal_reason == TerminalReason.EXTERNAL_SIGNAL
        )

    def result(
        self,
        *,
        terminal_reason: TerminalReason,
        final_step: int,
        model_kind: str,
        model_path: str = "final_model.zip",
    ) -> TrainingResult:
        if self.budget is None:
            raise RuntimeError("training budget must be configured before producing a result")
        return TrainingResult(
            terminal_reason=terminal_reason,
            execution_mode=self.execution_policy.mode,
            execution_policy=self.execution_policy.to_document(),
            first_completion_step=self.first_completion_step,
            final_step=int(final_step),
            requested_limit=self.budget.requested_limit,
            execution_limit=self.budget.execution_total,
            model_kind=model_kind,
            model_path=model_path,
        )

    def terminal_provenance(
        self,
        *,
        terminal_reason: TerminalReason,
        final_step: int,
    ) -> dict[str, Any]:
        if self.budget is None:
            raise RuntimeError(
                "training budget must be configured before producing terminal provenance"
            )
        return {
            "terminal_reason": terminal_reason.value,
            "first_completion_step": self.first_completion_step,
            "final_step": int(final_step),
            "requested_limit": self.budget.requested_limit,
            "execution_limit": self.budget.execution_total,
        }

    def finalize(self, result: TrainingResult) -> None:
        if self.closed:
            raise RuntimeError("training session is already finalized")
        if self.budget is None:
            raise RuntimeError("training budget must be configured before finalizing")
        if result.execution_mode != self.execution_policy.mode:
            raise RuntimeError("training result execution mode disagrees with the session")
        if dict(result.execution_policy) != self.execution_policy.to_document():
            raise RuntimeError("training result execution policy disagrees with the session")
        if (
            result.requested_limit != self.budget.requested_limit
            or result.execution_limit != self.budget.execution_total
        ):
            raise RuntimeError("training result budget disagrees with the session")
        if result.first_completion_step != self.first_completion_step:
            raise RuntimeError("training result completion step disagrees with the session")
        if result.final_step != self.current_step:
            raise RuntimeError(
                f"training result final step {result.final_step} disagrees with "
                f"observed step {self.current_step}"
            )
        self.progress.update(
            step=int(result.final_step),
            metrics=self.progress_metrics or self.reducer.snapshot(),
            final=True,
        )
        self.progress.close()
        atomic_write_json(self.run_dir / TRAINING_RESULT_FILENAME, result.to_document())
        self.closed = True

    def fail(self, exc: BaseException) -> None:
        if self.closed:
            return
        self.progress.close()
        atomic_write_json(
            self.run_dir / TRAINING_RESULT_FILENAME,
            {
                "document_type": "gradlab.training-result",
                "format_version": 2,
                "status": "failed",
                "terminal_reason": TerminalReason.FAILED.value,
                "execution_mode": self.execution_policy.mode.value,
                "execution_policy": self.execution_policy.to_document(),
                "first_completion_step": self.first_completion_step,
                "final_step": int(self.current_step),
                "requested_limit": (
                    None if self.budget is None else int(self.budget.requested_limit)
                ),
                "execution_limit": (
                    None if self.budget is None else int(self.budget.execution_total)
                ),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "model_kind": None,
                "model": None,
            },
        )
        self.closed = True
