"""State owners for learner lifecycle, automatic evaluation admission, and delivery."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from gradlab.clock import parse_utc_datetime
from gradlab.training_lifecycle import (
    LEARNER_STATE_FORMAT_VERSION,
    TerminalReason,
    TrainingExecutionMode,
)


class LearnerOperationalFailure(RuntimeError):
    stop_reason = "learner_failure"


class LearnerFailure(LearnerOperationalFailure):
    """The learner emitted an authoritative failed terminal result."""


class LearnerStartupTimeout(LearnerOperationalFailure):
    """The learner did not emit readiness or a terminal result before its deadline."""

    stop_reason = "startup_timeout"


class LearnerStateContractError(LearnerOperationalFailure):
    """The learner emitted malformed, stale, or identity-mismatched state."""

    stop_reason = "invalid_result"


class LearnerExitContractMismatch(LearnerOperationalFailure):
    """The learner process exit disagreed with its terminal document."""

    stop_reason = "exit_contract_mismatch"


class LearnerTeardownTimeout(LearnerOperationalFailure):
    """The learner process group remained alive after bounded escalation."""

    stop_reason = "teardown_timeout"


class LearnerStopAcknowledgementTimeout(LearnerOperationalFailure):
    """The learner ignored bounded cooperative-stop requests."""

    stop_reason = "stop_acknowledgement_timeout"


@dataclass(frozen=True)
class LearnerState:
    kind: str
    status: str
    terminal_reason: str | None
    final_step: int | None
    document: Mapping[str, Any]


@dataclass
class LearnerLifecycle:
    started_at: float | None = None
    pid: int | None = None
    result_observed_at: float | None = None
    stop_requested_at: float | None = None
    last_stop_signal_at: float | None = None
    stop_signal_attempts: int = 0
    stop_acknowledged: bool = False
    final_step: int | None = None
    terminal_document: dict[str, Any] | None = None
    teardown_evidence: dict[str, Any] = field(default_factory=dict)

    def parse(
        self,
        document: Mapping[str, Any],
        *,
        kind: str,
        live: bool,
        run_id: str,
        attempt_id: str,
        backend_id: str,
    ) -> LearnerState:
        expected_type = "gradlab.learner-ready" if kind == "ready" else "gradlab.training-result"
        if document.get("document_type") != expected_type:
            raise LearnerStateContractError(
                f"learner {kind} document_type is not {expected_type!r}"
            )
        version = document.get("format_version")
        if version != LEARNER_STATE_FORMAT_VERSION:
            raise LearnerStateContractError(
                f"learner {kind} document has unsupported format_version {version!r}"
            )
        if document.get("run_id") != run_id:
            raise LearnerStateContractError(f"learner {kind} run_id does not match manifest")
        if document.get("attempt_id") != attempt_id:
            raise LearnerStateContractError(f"learner {kind} attempt_id does not match manifest")
        learner_pid = document.get("learner_pid")
        if isinstance(learner_pid, bool) or not isinstance(learner_pid, int) or learner_pid <= 0:
            raise LearnerStateContractError(f"learner {kind} has an invalid learner_pid")
        if live and learner_pid != self.pid:
            raise LearnerStateContractError(
                f"learner {kind} pid {learner_pid} does not match spawned pid {self.pid}"
            )
        if document.get("execution_mode") != TrainingExecutionMode.SUPERVISED.value:
            raise LearnerStateContractError(f"learner {kind} is not supervised")
        if not backend_id or document.get("training_backend_id") != backend_id:
            raise LearnerStateContractError(
                f"learner {kind} training_backend_id does not match materialized config"
            )
        timestamp_field = "ready_at" if kind == "ready" else "terminal_at"
        timestamp = document.get(timestamp_field)
        if not isinstance(timestamp, str):
            raise LearnerStateContractError(f"learner {kind} has no valid {timestamp_field}")
        try:
            parse_utc_datetime(timestamp)
        except (TypeError, ValueError) as exc:
            raise LearnerStateContractError(
                f"learner {kind} has an invalid {timestamp_field}"
            ) from exc
        if kind == "ready":
            if document.get("status") != "ready":
                raise LearnerStateContractError("learner readiness status is not 'ready'")
            return LearnerState(
                kind="ready",
                status="ready",
                terminal_reason=None,
                final_step=None,
                document=dict(document),
            )

        status = str(document.get("status") or "")
        if status not in {"completed", "interrupted", "failed"}:
            raise LearnerStateContractError("learner result has an invalid status")
        try:
            reason = TerminalReason(str(document.get("terminal_reason") or ""))
        except ValueError as exc:
            raise LearnerStateContractError(
                "learner result has an invalid terminal_reason"
            ) from exc
        if (status == "failed") != (reason == TerminalReason.FAILED):
            raise LearnerStateContractError("learner result status and terminal_reason disagree")
        interruption_reasons = {
            TerminalReason.LOCAL_INTERRUPTION,
            TerminalReason.EXTERNAL_SIGNAL,
        }
        if (status == "interrupted") != (reason in interruption_reasons):
            raise LearnerStateContractError(
                "learner result interrupted status and terminal_reason disagree"
            )
        final_step = document.get("final_step")
        if isinstance(final_step, bool) or not isinstance(final_step, int) or final_step < 0:
            raise LearnerStateContractError("learner result has an invalid final_step")
        if not isinstance(document.get("execution_policy"), Mapping):
            raise LearnerStateContractError("learner result has no execution_policy")
        if status == "failed":
            error_type = document.get("error_type")
            error_message = document.get("error_message")
            if (
                not isinstance(error_type, str)
                or not error_type
                or len(error_type) > 200
                or not isinstance(error_message, str)
                or len(error_message) > 2_000
            ):
                raise LearnerStateContractError(
                    "failed learner result has invalid bounded error evidence"
                )
        return LearnerState(
            kind="result",
            status=status,
            terminal_reason=reason.value,
            final_step=final_step,
            document=dict(document),
        )

    def observe(
        self,
        result: LearnerState | None,
        ready: LearnerState | None,
        *,
        now: float,
        startup_timeout: float,
        exit_grace: float,
        emit,
    ) -> LearnerState | None:
        if result is not None:
            self.final_step = result.final_step
            self.terminal_document = dict(result.document)
            if self.result_observed_at is None:
                self.result_observed_at = now
                emit(
                    "learner_terminal_result_observed",
                    status=result.status,
                    terminal_reason=result.terminal_reason,
                    final_step=result.final_step,
                )
            if result.status == "failed":
                error_type = str(result.document.get("error_type") or "LearnerError")
                error_message = str(result.document.get("error_message") or "")
                raise LearnerFailure(f"{error_type}: {error_message}")
            elapsed = max(0.0, now - self.result_observed_at)
            grace = exit_grace
            if elapsed >= grace:
                raise LearnerTeardownTimeout(
                    "learner wrote terminal result "
                    f"{result.terminal_reason!r} but remained alive for {elapsed:.1f}s"
                )
            return result

        if ready is not None:
            return ready
        if self.started_at is None:
            raise LearnerStateContractError("learner startup time was not recorded")
        elapsed = max(0.0, now - self.started_at)
        timeout = startup_timeout
        if elapsed >= timeout:
            raise LearnerStartupTimeout(
                f"learner emitted no readiness or terminal result within {timeout:.1f}s"
            )
        return None

    def validate_exit(self, terminal_state: LearnerState | None, returncode: int) -> LearnerState:
        if terminal_state is None:
            raise LearnerStateContractError(
                f"learner exited with code {returncode} without a terminal result"
            )
        self.final_step = terminal_state.final_step
        self.terminal_document = dict(terminal_state.document)
        if terminal_state.status == "failed":
            raise LearnerFailure(
                f"{terminal_state.document.get('error_type')}: "
                f"{terminal_state.document.get('error_message')}"
            )
        if returncode != 0:
            raise LearnerExitContractMismatch(
                f"learner emitted {terminal_state.status!r} but exited with code {returncode}"
            )
        return terminal_state


@dataclass
class AutomaticEvaluationAdmission:
    closed: bool = False
    accepted_observed_at: float | None = None

    def close(self, store, *, reason: str, closed_at: str, **identity: str) -> bool:
        if self.closed:
            return False
        self.closed = True
        store.set_state(
            "automatic_eval_admission",
            {
                "closed": True,
                "reason": reason,
                **identity,
                "closed_at": closed_at,
            },
        )
        return True


@dataclass
class DeliverySchedule:
    last_service: float = float("-inf")
    retry_after: float = 0.0
    servicing: bool = False
    remote_high_water: int = 0
    remote_visible_lag_seconds: float = 0.0

    def begin(self, now: float, *, force: bool, interval: float) -> bool:
        if (
            self.servicing
            or now < self.retry_after
            or (not force and now - self.last_service < interval)
        ):
            return False
        self.servicing = True
        self.last_service = now
        return True
