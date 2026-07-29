from __future__ import annotations

import math
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import TracebackType

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Vertical
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Footer, Header, ProgressBar, RichLog, Sparkline, Static

from gradlab.train import TrainingRuntimeControl, graceful_stop_signal_scope
from gradlab.training_backend import GracefulStopFlag
from gradlab.training_lifecycle import (
    PlainProgressSink,
    ProgressField,
    ProgressSink,
    format_progress_value,
    resolve_progress_fields,
)


TUI_REFRESH_SECONDS = 0.25
TUI_HISTORY_SAMPLE_SECONDS = 1.0
TUI_HISTORY_LENGTH = 120
TUI_EVENT_LIMIT = 256
TUI_EVENT_TEXT_LIMIT = 4_096
PLAIN_FALLBACK_INTERVAL_SECONDS = 10.0


@dataclass(frozen=True)
class LocalTrainingIdentity:
    recipe: str
    seed: int
    output: str
    notices: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrainingLogEvent:
    sequence: int
    text: str
    stderr: bool


@dataclass(frozen=True)
class TrainingProgressSnapshot:
    revision: int
    description: str
    total: int
    initial: int
    step: int
    fields: tuple[ProgressField, ...]
    metrics: Mapping[str, int | float]
    events: tuple[TrainingLogEvent, ...]
    started_at: float | None
    updated_at: float
    final: bool
    closed: bool


class TrainingProgressBridge(ProgressSink):
    """A bounded latest-value bridge from a learner thread to a renderer."""

    def __init__(
        self,
        *,
        event_limit: int = TUI_EVENT_LIMIT,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self._lock = threading.RLock()
        self._revision = 0
        self._event_sequence = 0
        self._description = ""
        self._total = 0
        self._initial = 0
        self._step = 0
        self._fields: tuple[ProgressField, ...] = ()
        self._metrics: dict[str, int | float] = {}
        self._events: deque[TrainingLogEvent] = deque(maxlen=max(1, int(event_limit)))
        self._started_at: float | None = None
        self._updated_at = self._clock()
        self._final = False
        self._closed = False

    def start(
        self,
        *,
        total: int,
        initial: int,
        description: str,
        fields: Sequence[ProgressField] = (),
    ) -> None:
        now = self._clock()
        with self._lock:
            self._description = str(description)
            self._total = int(total)
            self._initial = int(initial)
            self._step = int(initial)
            self._fields = resolve_progress_fields(fields)
            self._metrics = {}
            self._started_at = now
            self._updated_at = now
            self._final = False
            self._closed = False
            self._revision += 1

    def update(
        self,
        *,
        step: int,
        metrics: Mapping[str, int | float],
        final: bool = False,
    ) -> None:
        now = self._clock()
        with self._lock:
            selected = {
                field.metric: value
                for field in self._fields
                if (value := metrics.get(field.metric)) is not None
                and math.isfinite(float(value))
            }
            self._metrics.update(selected)
            self._step = int(step)
            self._updated_at = now
            self._final = self._final or bool(final)
            self._revision += 1

    def event(self, message: str) -> None:
        self.write_event(message)

    def write_event(self, message: str, *, stderr: bool = False) -> None:
        lines = str(message).replace("\r", "\n").splitlines()
        for line in lines:
            text = line.strip()
            if not text:
                continue
            if len(text) > TUI_EVENT_TEXT_LIMIT:
                text = text[: TUI_EVENT_TEXT_LIMIT - 1] + "…"
            with self._lock:
                self._event_sequence += 1
                self._events.append(
                    TrainingLogEvent(
                        sequence=self._event_sequence,
                        text=text,
                        stderr=bool(stderr),
                    )
                )
                self._updated_at = self._clock()
                self._revision += 1

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._updated_at = self._clock()
            self._revision += 1

    def snapshot(self, *, after_event: int = 0) -> TrainingProgressSnapshot:
        with self._lock:
            return TrainingProgressSnapshot(
                revision=self._revision,
                description=self._description,
                total=self._total,
                initial=self._initial,
                step=self._step,
                fields=self._fields,
                metrics=dict(self._metrics),
                events=tuple(
                    event for event in self._events if event.sequence > int(after_event)
                ),
                started_at=self._started_at,
                updated_at=self._updated_at,
                final=self._final,
                closed=self._closed,
            )


class _ExecutionState(StrEnum):
    NEW = "new"
    RUNNING = "running"
    DONE = "done"


class LearnerExecution:
    """Exactly-once learner ownership shared by Textual and its fallback."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state = _ExecutionState.NEW
        self._done = threading.Event()
        self._result: int | None = None
        self._error: BaseException | None = None
        self._traceback: TracebackType | None = None

    @property
    def state(self) -> _ExecutionState:
        with self._lock:
            return self._state

    def run(self, learner: Callable[[], int]) -> bool:
        with self._lock:
            if self._state != _ExecutionState.NEW:
                return False
            self._state = _ExecutionState.RUNNING
        try:
            result = learner()
        except BaseException as exc:
            with self._lock:
                self._error = exc
                self._traceback = exc.__traceback__
        else:
            with self._lock:
                self._result = int(result)
        finally:
            with self._lock:
                self._state = _ExecutionState.DONE
                self._done.set()
        return True

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)

    def result_or_raise(self) -> int:
        self._done.wait()
        with self._lock:
            error = self._error
            traceback = self._traceback
            result = self._result
        if error is not None:
            raise error.with_traceback(traceback)
        if result is None:
            raise RuntimeError("local learner completed without a result")
        return result


class MetricCard(Widget):
    def __init__(self, field: ProgressField, index: int) -> None:
        super().__init__(classes="metric-card")
        self.field = field
        self.index = index
        self._value = "—"
        self._history: tuple[float, ...] = ()

    def compose(self) -> ComposeResult:
        yield Static(self.field.label, classes="metric-label")
        yield Static(self._value, id=f"metric-value-{self.index}", classes="metric-value")
        yield Sparkline(
            self._history,
            id=f"metric-spark-{self.index}",
            classes="metric-sparkline",
        )

    def set_metric(self, value: str, history: Sequence[float]) -> None:
        self._value = value
        self._history = tuple(history)
        if not self.is_mounted:
            return
        self.query_one(f"#metric-value-{self.index}", Static).update(value)
        self.query_one(f"#metric-spark-{self.index}", Sparkline).data = self._history


class TrainingFinished(Message):
    pass


class LocalTrainingApp(App[None]):
    CSS_PATH = "training_tui.tcss"
    HORIZONTAL_BREAKPOINTS = [(0, "-narrow"), (80, "-medium"), (120, "-wide")]
    VERTICAL_BREAKPOINTS = [(0, "-short"), (24, "-tall")]
    BINDINGS = [
        Binding("q", "request_stop", "Stop safely", priority=True),
        Binding("ctrl+c", "request_stop", show=False, priority=True),
        Binding("ctrl+q", "request_stop", show=False, priority=True),
    ]

    def __init__(
        self,
        *,
        identity: LocalTrainingIdentity,
        bridge: TrainingProgressBridge,
        stop_flag: GracefulStopFlag,
        execution: LearnerExecution,
        learner: Callable[[], int],
    ) -> None:
        super().__init__()
        self._preserve_capture_file_descriptors()
        self.identity = identity
        self.bridge = bridge
        self.stop_flag = stop_flag
        self.execution = execution
        self.learner = learner
        self.title = "gradlab local training"
        self.sub_title = identity.recipe
        self._last_revision = -1
        self._last_event = 0
        self._cards: dict[str, MetricCard] = {}
        self._histories: dict[str, deque[float]] = {}
        self._last_history_sample = 0.0
        self._last_rate_step: int | None = None
        self._last_rate_time: float | None = None
        self._smoothed_rate: float | None = None

    def _preserve_capture_file_descriptors(self) -> None:
        """Keep subprocess launchers compatible with Textual print capture.

        Textual's capture streams intentionally report ``fileno() == -1``.
        Python 3.14's multiprocessing resource tracker includes
        ``sys.stderr.fileno()`` in the descriptors inherited by its child and
        rejects that sentinel before launching. Local training still needs
        Textual's write capture, but a real descriptor is safe to expose for
        subprocess inheritance.
        """

        for capture, original in (
            (self._capture_stdout, sys.__stdout__),
            (self._capture_stderr, sys.__stderr__),
        ):
            if original is None:
                continue
            try:
                descriptor = int(original.fileno())
            except (AttributeError, OSError, ValueError):
                continue
            if descriptor < 0:
                continue
            setattr(capture, "fileno", lambda descriptor=descriptor: descriptor)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="training-body"):
            yield Static(
                Text.assemble(
                    ("recipe ", "dim"),
                    self.identity.recipe,
                    ("  seed ", "dim"),
                    str(self.identity.seed),
                    ("\noutput ", "dim"),
                    self.identity.output,
                ),
                id="run-identity",
            )
            yield Static("initializing learner…", id="training-status")
            yield ProgressBar(
                total=None,
                show_percentage=False,
                show_eta=False,
                id="training-progress",
            )
            yield Static("waiting for training budget", id="progress-meta")
            yield Grid(id="metrics")
            yield RichLog(
                max_lines=TUI_EVENT_LIMIT,
                wrap=True,
                markup=False,
                auto_scroll=True,
                id="event-log",
            )
        yield Footer()

    def on_mount(self) -> None:
        self.begin_capture_print(self, stdout=True, stderr=True)
        self.set_interval(TUI_REFRESH_SECONDS, self._refresh_from_bridge)
        self._refresh_from_bridge()
        self.run_worker(
            self._run_learner,
            name="gradlab-local-learner",
            exit_on_error=False,
            thread=True,
        )

    def on_unmount(self) -> None:
        self.end_capture_print(self)

    def _run_learner(self) -> None:
        try:
            self.execution.run(self.learner)
        finally:
            self.post_message(TrainingFinished())

    def on_training_finished(self, _message: TrainingFinished) -> None:
        self._refresh_from_bridge()
        self.exit()

    def on_print(self, event: events.Print) -> None:
        self.bridge.write_event(event.text, stderr=event.stderr)

    def action_request_stop(self) -> None:
        if self.execution.state == _ExecutionState.DONE:
            return
        self.stop_flag.request("interactive_tui")
        self.bridge.write_event(
            "graceful stop requested; waiting for the learner's safe boundary"
        )
        self.query_one("#training-status", Static).update("stop pending — finishing safely")

    def _ensure_metric_cards(self, fields: Sequence[ProgressField]) -> None:
        grid = self.query_one("#metrics", Grid)
        for index, field in enumerate(fields):
            if field.metric in self._cards:
                continue
            card = MetricCard(field, index)
            self._cards[field.metric] = card
            self._histories[field.metric] = deque(maxlen=TUI_HISTORY_LENGTH)
            grid.mount(card)

    def _refresh_from_bridge(self) -> None:
        snapshot = self.bridge.snapshot(after_event=self._last_event)
        now = time.monotonic()
        if snapshot.events:
            log = self.query_one("#event-log", RichLog)
            for event in snapshot.events:
                log.write(Text(event.text, style="bold red" if event.stderr else ""))
                self._last_event = max(self._last_event, event.sequence)
        snapshot_changed = snapshot.revision != self._last_revision
        self._last_revision = snapshot.revision
        self._ensure_metric_cards(snapshot.fields)

        if snapshot.started_at is None:
            status = "stop pending — initializing safely" if self.stop_flag.requested else (
                "initializing learner…"
            )
            self.query_one("#training-status", Static).update(status)
            return

        if self.stop_flag.requested:
            status = "stop pending — waiting for a safe learner boundary"
        elif snapshot.closed:
            status = "training finished"
        elif snapshot.final:
            status = "finalizing artifacts"
        else:
            status = f"running {snapshot.description}"
        self.query_one("#training-status", Static).update(status)

        total_for_bar = snapshot.total if snapshot.total > 0 else 1
        progress_for_bar = (
            min(max(snapshot.step, 0), total_for_bar) if snapshot.total > 0 else 1
        )
        self.query_one("#training-progress", ProgressBar).update(
            total=total_for_bar,
            progress=progress_for_bar,
        )

        if self._last_rate_step is None or self._last_rate_time is None:
            self._last_rate_step = snapshot.step
            self._last_rate_time = now
        elif snapshot.step > self._last_rate_step:
            elapsed_sample = now - self._last_rate_time
            advanced = snapshot.step - self._last_rate_step
            if elapsed_sample > 0 and advanced > 0:
                observed_rate = advanced / elapsed_sample
                self._smoothed_rate = (
                    observed_rate
                    if self._smoothed_rate is None
                    else (0.25 * observed_rate) + (0.75 * self._smoothed_rate)
                )
            self._last_rate_step = snapshot.step
            self._last_rate_time = now

        elapsed = max(now - snapshot.started_at, 0.0)
        fraction = (
            min(max(snapshot.step / snapshot.total, 0.0), 1.0)
            if snapshot.total
            else 1.0
        )
        rate_text = (
            "—"
            if self._smoothed_rate is None
            else f"{self._smoothed_rate:,.0f} transitions/s"
        )
        remaining = max(snapshot.total - snapshot.step, 0)
        eta_text = (
            "—"
            if self._smoothed_rate is None or self._smoothed_rate <= 0
            else _format_duration(remaining / self._smoothed_rate)
        )
        self.query_one("#progress-meta", Static).update(
            f"{snapshot.step:,}/{snapshot.total:,}  {fraction:.1%}   "
            f"elapsed {_format_duration(elapsed)}   rate {rate_text}   ETA {eta_text}"
        )

        sample_history = now - self._last_history_sample >= TUI_HISTORY_SAMPLE_SECONDS
        if sample_history:
            self._last_history_sample = now
        for field in snapshot.fields:
            value = snapshot.metrics.get(field.metric)
            history = self._histories[field.metric]
            if value is not None and math.isfinite(float(value)) and sample_history:
                history.append(float(value))
            if snapshot_changed or sample_history:
                self._cards[field.metric].set_metric(
                    format_progress_value(value, field.value_format),
                    history,
                )


def _format_duration(seconds: float) -> str:
    total = max(int(seconds), 0)
    hours, remainder = divmod(total, 3_600)
    minutes, seconds = divmod(remainder, 60)
    return (
        f"{hours:d}:{minutes:02d}:{seconds:02d}"
        if hours
        else f"{minutes:02d}:{seconds:02d}"
    )


def _emit_plain_notices(identity: LocalTrainingIdentity) -> None:
    for notice in identity.notices:
        print(notice, flush=True)


def _plain_snapshot_line(snapshot: TrainingProgressSnapshot) -> str:
    if snapshot.started_at is None:
        return "local training progress: initializing learner"
    fraction = (
        min(max(snapshot.step / snapshot.total, 0.0), 1.0) if snapshot.total else 1.0
    )
    fields = " ".join(
        f"{field.label}={format_progress_value(snapshot.metrics.get(field.metric), field.value_format)}"
        for field in snapshot.fields
    )
    return (
        f"{snapshot.description} progress: {snapshot.step:,}/{snapshot.total:,} "
        f"({fraction:.1%}) {fields}"
    ).rstrip()


def _drain_running_learner(
    execution: LearnerExecution,
    bridge: TrainingProgressBridge,
) -> None:
    last_event = 0
    while not execution.wait(PLAIN_FALLBACK_INTERVAL_SECONDS):
        snapshot = bridge.snapshot(after_event=last_event)
        print(_plain_snapshot_line(snapshot), flush=True)
        for event in snapshot.events:
            print(event.text, flush=True)
            last_event = max(last_event, event.sequence)
    snapshot = bridge.snapshot(after_event=last_event)
    print(_plain_snapshot_line(snapshot), flush=True)
    for event in snapshot.events:
        print(event.text, flush=True)


def run_local_training_tui(
    *,
    identity: LocalTrainingIdentity,
    learner: Callable[[TrainingRuntimeControl], int],
) -> int:
    """Run one local learner with Textual, falling back without rerunning it."""

    bridge = TrainingProgressBridge()
    for notice in identity.notices:
        bridge.event(notice)
    stop_flag = GracefulStopFlag()
    execution = LearnerExecution()
    tui_control = TrainingRuntimeControl(
        progress_sink=bridge,
        stop_flag=stop_flag,
        signal_handlers_owned_by_host=True,
    )
    plain_control = TrainingRuntimeControl(
        progress_sink=PlainProgressSink(),
        stop_flag=stop_flag,
        signal_handlers_owned_by_host=True,
    )

    try:
        app = LocalTrainingApp(
            identity=identity,
            bridge=bridge,
            stop_flag=stop_flag,
            execution=execution,
            learner=lambda: learner(tui_control),
        )
    except Exception as exc:
        print(f"warning: local training TUI unavailable; using plain output: {exc}", flush=True)
        _emit_plain_notices(identity)
        with graceful_stop_signal_scope(stop_flag, include_sigint=True):
            execution.run(lambda: learner(plain_control))
        return execution.result_or_raise()

    presentation_error: Exception | None = None
    with graceful_stop_signal_scope(stop_flag, include_sigint=True):
        try:
            app.run()
        except Exception as exc:
            presentation_error = exc

        if execution.state == _ExecutionState.NEW:
            reason = (
                str(presentation_error)
                if presentation_error is not None
                else "the TUI exited before the learner started"
            )
            print(
                f"warning: local training TUI unavailable; using plain output: {reason}",
                flush=True,
            )
            _emit_plain_notices(identity)
            execution.run(lambda: learner(plain_control))
        elif execution.state == _ExecutionState.RUNNING:
            reason = (
                str(presentation_error)
                if presentation_error is not None
                else "the TUI exited while the learner was running"
            )
            print(
                f"warning: local training TUI stopped; continuing the same learner: {reason}",
                flush=True,
            )
            _drain_running_learner(execution, bridge)
        elif presentation_error is not None:
            print(
                f"warning: local training TUI stopped after the learner finished: "
                f"{presentation_error}",
                flush=True,
            )

    return execution.result_or_raise()
