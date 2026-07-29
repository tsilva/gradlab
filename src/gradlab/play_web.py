from __future__ import annotations

import argparse
import asyncio
import io
import json
import queue
import secrets
import struct
import threading
import time
import uuid
import webbrowser
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
from aiohttp import WSMsgType, web
from PIL import Image

from gradlab.action_contract import action_contract_payload
from gradlab.play_session import _PlaybackSession, _PlaybackTransition, render_obs_stack
from gradlab.play_debug import ANSI_PATTERN, PolicyDecision, model_input_lines
from gradlab.seeds import validate_playback_seed


PROTOCOL_VERSION = 4
HISTORY_LIMIT = 4096
COMMAND_QUEUE_LIMIT = 64
CLIENT_QUEUE_LIMIT = 64
FRAME_ENCODER_QUEUE_LIMIT = 64
INSPECTION_FRAME_WAIT_SECONDS = 2.0
FRAME_HEADER = struct.Struct(">4sBBHQQ")
FRAME_MAGIC = b"RLP2"
FRAME_CODEC_PNG = 1
FRAME_GAME = 1
FRAME_OBSERVATION = 2
MAX_JSON_DEPTH = 5
MAX_JSON_ITEMS = 128
MAX_JSON_TEXT = 4096
INPUT_HEARTBEAT_SECONDS = 0.25
LAST_CLIENT_GRACE_SECONDS = 30.0
PAIRED_START_GRACE_SECONDS = 2.0


def source_browser_path(route: Mapping[str, Any] | None) -> str:
    route = route or {}
    project = str(route.get("project") or "").strip()
    goal_id = str(route.get("goal_id") or "").strip()
    goal_variant_id = str(route.get("goal_variant_id") or "").strip()
    run_id = str(route.get("run_id") or "").strip()
    checkpoint_id = str(route.get("checkpoint_id") or "").strip()
    if not project:
        return "/"
    path = f"/environments/{quote(project, safe='')}"
    if not goal_id:
        return path
    path += f"/goals/{quote(goal_id, safe='')}"
    if not goal_variant_id:
        return path
    path += f"/variants/{quote(goal_variant_id, safe='')}"
    if not run_id:
        return path
    path += f"/runs/{quote(run_id, safe='')}"
    if not checkpoint_id:
        return path
    return f"{path}/checkpoints/{quote(checkpoint_id, safe='')}"


def _session_environment_id(session: Any, args: argparse.Namespace) -> str | None:
    config = getattr(session, "config", None)
    game = config.get("game") if isinstance(config, Mapping) else getattr(config, "game", None)
    for value in (game, getattr(session, "environment_id", None), getattr(args, "env_id", None)):
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _json_value(value: Any, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, bool | int | str):
        if isinstance(value, str) and len(value) > MAX_JSON_TEXT:
            return value[:MAX_JSON_TEXT] + "…"
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if depth >= MAX_JSON_DEPTH:
        return f"<{type(value).__name__}>"
    if isinstance(value, np.generic):
        return _json_value(value.item(), depth=depth + 1)
    if isinstance(value, np.ndarray):
        if value.size > MAX_JSON_ITEMS:
            finite = value[np.isfinite(value)] if np.issubdtype(value.dtype, np.number) else ()
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "min": float(np.min(finite)) if len(finite) else None,
                "max": float(np.max(finite)) if len(finite) else None,
            }
        return _json_value(value.tolist(), depth=depth + 1)
    if is_dataclass(value):
        return _json_value(asdict(value), depth=depth + 1)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= MAX_JSON_ITEMS:
                result["…"] = f"{len(value) - MAX_JSON_ITEMS} more entries"
                break
            name = str(key)
            lowered = name.casefold()
            if any(token in lowered for token in ("password", "secret", "credential", "token")):
                result[name] = "<redacted>"
            elif name in {"terminal_observation", "final_observation"}:
                result[name] = _json_value(np.asarray(item), depth=depth + 1)
            else:
                result[name] = _json_value(item, depth=depth + 1)
        return result
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray | memoryview):
        rendered = [_json_value(item, depth=depth + 1) for item in value[:MAX_JSON_ITEMS]]
        if len(value) > MAX_JSON_ITEMS:
            rendered.append(f"<{len(value) - MAX_JSON_ITEMS} more entries>")
        return rendered
    if isinstance(value, bytes | bytearray | memoryview):
        return f"<{len(value)} bytes>"
    return str(value)[:MAX_JSON_TEXT]


def _session_action_contract_payload(session: Any) -> dict[str, Any] | None:
    cached = getattr(session, "action_contract_payload", None)
    if isinstance(cached, Mapping):
        return dict(cached)
    return action_contract_payload(getattr(session, "action_contract", None))


def _decision_payload(decision: PolicyDecision | None) -> dict[str, Any] | None:
    if decision is None:
        return None
    return {
        "distribution": decision.distribution_kind,
        "requested_action_selection_mode": decision.requested_action_selection_mode,
        "action_selection_mode": decision.action_selection_mode,
        "raw_action": _json_value(decision.raw_action),
        "executed_action": _json_value(decision.executed_action),
        "value": decision.value,
        "log_probability": decision.log_probability,
        "entropy": decision.entropy,
        "mode": _json_value(decision.mode),
        "probabilities": _json_value(decision.probabilities),
        "component_probabilities": _json_value(decision.component_probabilities),
        "mean": _json_value(decision.mean),
        "stddev": _json_value(decision.stddev),
        "q_values": _json_value(decision.q_values),
        "selected_q_value": decision.selected_q_value,
        "selected_q_rank": decision.selected_q_rank,
        "exploration_rate": decision.exploration_rate,
        "exploratory": decision.exploratory,
        "program": _json_value(decision.program),
        "route": _json_value(decision.route),
        "sampled": decision.sampled,
        "selected_action": decision.selected_discrete_action,
        "selected_probability": decision.selected_probability,
        "selected_rank": decision.selected_rank,
    }


def _numeric_signals(value: Mapping[str, Any]) -> dict[str, float]:
    signals: dict[str, float] = {}
    for key, item in value.items():
        if isinstance(item, bool):
            signals[str(key)] = float(item)
        elif isinstance(item, int | float | np.number):
            numeric = float(item)
            if np.isfinite(numeric):
                signals[str(key)] = numeric
        elif isinstance(item, np.ndarray) and item.size == 1:
            numeric = float(item.reshape(-1)[0])
            if np.isfinite(numeric):
                signals[str(key)] = numeric
    return dict(sorted(signals.items())[:MAX_JSON_ITEMS])


def transition_payload(transition: _PlaybackTransition) -> dict[str, Any]:
    diagnostics = transition.diagnostics
    provider_reward = transition.reward
    task_reward = transition.reward
    outcome = "continuing"
    task_metrics: Mapping[str, Any] = {}
    event_transitions: Mapping[str, Any] = {}
    boundary_reasons: list[str] = []
    if diagnostics is not None:
        provider_reward = diagnostics.provider_reward
        task_reward = diagnostics.reward
        outcome = diagnostics.outcome.name.lower()
        task_metrics = diagnostics.task_metrics
        event_transitions = diagnostics.event_transitions
        if diagnostics.provider_terminated:
            boundary_reasons.append("provider_terminated")
        if diagnostics.provider_truncated:
            boundary_reasons.append("provider_truncated")
        if diagnostics.task_terminated:
            boundary_reasons.append("task_terminated")
        if diagnostics.task_truncated:
            boundary_reasons.append("task_truncated")
    components = {
        name.removesuffix("_component"): float(np.asarray(value).reshape(-1)[0])
        for name, value in task_metrics.items()
        if name.endswith("_component") and np.asarray(value).size == 1
    }
    return {
        "sequence": transition.sequence,
        "episode": transition.episode,
        "step": transition.step,
        "seed": transition.seed,
        "start_id": transition.start_id,
        "action_source": transition.action_source,
        "executed_action": _json_value(transition.executed_action),
        "decision": _decision_payload(transition.decision),
        "before": {
            "task": _json_value(transition.pre_task),
            "model_input": model_input_lines(transition.model_obs),
            "game_frame": transition.before_frame is not None,
            "observation_frames": len(transition.before_frames),
        },
        "after": {
            "task": _json_value(transition.next_task),
            "game_frame": transition.after_frame is not None,
            "observation_frames": len(transition.after_frames),
            "frame_role": transition.after_frame_role,
        },
        "reward": {
            "provider": provider_reward,
            "shaped": task_reward,
            "step": transition.reward,
            "return": transition.total_reward,
            "components": components,
        },
        "events": list(transition.events),
        "event_transitions": _json_value(event_transitions),
        "signals": _numeric_signals(transition.info),
        "info": _json_value(transition.info),
        "max_x_pos": transition.max_x_pos,
        "terminated": transition.terminated,
        "truncated": transition.truncated,
        "completed": transition.completed,
        "boundary": transition.boundary,
        "boundary_reasons": boundary_reasons,
        "outcome": outcome,
        "attribution": transition.attribution is not None,
    }


def history_point(transition: _PlaybackTransition) -> dict[str, Any]:
    return history_point_payload(transition_payload(transition))


def history_point_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    decision = payload["decision"] or {}
    reward = payload["reward"]
    return {
        "sequence": payload["sequence"],
        "episode": payload["episode"],
        "step": payload["step"],
        "policy_action": decision.get("selected_action"),
        "executed_action": payload.get("executed_action"),
        "action_source": payload.get("action_source"),
        "policy_sampled": decision.get("sampled"),
        "reward_provider": reward["provider"],
        "reward_shaped": reward["shaped"],
        "return": reward["return"],
        "value": decision.get("value"),
        "selected_q_value": decision.get("selected_q_value"),
        "entropy": decision.get("entropy"),
        "log_probability": decision.get("log_probability"),
        "action_selection_mode": decision.get("action_selection_mode"),
        "outcome": payload.get("outcome"),
        "events": payload["events"],
        "boundary": bool(payload.get("boundary")),
        "terminated": bool(payload.get("terminated")),
        "truncated": bool(payload.get("truncated")),
        "boundary_reasons": list(payload.get("boundary_reasons") or []),
        "signals": payload["signals"],
        "components": reward["components"],
    }


def _value_discount_factor(model: Any) -> float | None:
    value = getattr(model, "gamma", None)
    if isinstance(value, bool) or not isinstance(value, int | float | np.number):
        return None
    discount = float(value)
    return discount if np.isfinite(discount) and 0.0 <= discount <= 1.0 else None


def annotate_realized_returns(
    points: Sequence[dict[str, Any]],
    *,
    episode: int,
    discount: float | None,
    comparison_reasons: Sequence[str] = (),
) -> None:
    """Attach one on-policy Monte Carlo diagnostic when its semantics are comparable."""

    episode_points = [point for point in points if int(point.get("episode", -1)) == episode]
    reasons = list(dict.fromkeys(str(reason) for reason in comparison_reasons if str(reason)))
    if discount is None:
        reasons.append("training discount is unavailable")
    if any(
        point.get("action_source") is not None and point.get("action_source") != "policy"
        for point in episode_points
    ):
        reasons.append("episode contains non-policy actions")
    if any(
        point.get("policy_sampled") is not None and point.get("policy_sampled") is not True
        for point in episode_points
    ):
        reasons.append("episode contains non-stochastic policy actions")
    if reasons:
        for point in episode_points:
            point["value_comparison_reasons"] = reasons
        return
    realized_return = 0.0
    for point in reversed(episode_points):
        reward = point.get("reward_shaped")
        if isinstance(reward, bool) or not isinstance(reward, int | float | np.number):
            for episode_point in episode_points:
                episode_point["value_comparison_reasons"] = ["policy reward is unavailable"]
            return
        numeric_reward = float(reward)
        if not np.isfinite(numeric_reward):
            for episode_point in episode_points:
                episode_point["value_comparison_reasons"] = ["policy reward is non-finite"]
            return
        realized_return = numeric_reward + discount * realized_return
        point["realized_return"] = realized_return
        point["value_comparison_reasons"] = []
        value = point.get("value")
        if (
            not isinstance(value, bool)
            and isinstance(value, int | float | np.number)
            and np.isfinite(float(value))
        ):
            point["value_error"] = float(value) - realized_return


def _frame_packet(
    kind: int,
    sequence: int,
    frame: np.ndarray,
    *,
    session_epoch: int = 0,
) -> bytes:
    output = io.BytesIO()
    Image.fromarray(frame, mode="RGB").save(
        output,
        format="PNG",
        compress_level=1,
    )
    return (
        FRAME_HEADER.pack(
            FRAME_MAGIC,
            kind,
            FRAME_CODEC_PNG,
            0,
            session_epoch,
            sequence,
        )
        + output.getvalue()
    )


class FrameEncoder:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._pending: deque[tuple[int, int, dict[int, np.ndarray]]] = deque()
        self._latest: dict[int, tuple[int, bytes]] = {}
        self._retained: dict[tuple[int, int], dict[int, tuple[int, bytes]]] = {}
        self._epoch = 0
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="gradlab-frame-encoder")

    @property
    def epoch(self) -> int:
        with self._condition:
            return self._epoch

    def set_epoch(self, epoch: int) -> None:
        with self._condition:
            if self._thread.is_alive():
                raise RuntimeError("frame encoder epoch must be set before start")
            self._epoch = int(epoch)
            self._pending.clear()
            self._latest.clear()
            self._retained.clear()

    def start(self) -> None:
        self._thread.start()

    def submit(self, kind: int, sequence: int, frame: np.ndarray | None) -> None:
        self.submit_batch(sequence, {kind: frame})

    def submit_batch(
        self,
        sequence: int,
        frames: Mapping[int, np.ndarray | None],
    ) -> None:
        owned = {
            int(kind): np.asarray(frame, dtype=np.uint8).copy()
            for kind, frame in frames.items()
            if frame is not None
        }
        if not owned:
            return
        with self._condition:
            while len(self._pending) >= FRAME_ENCODER_QUEUE_LIMIT and not self._closed:
                self._condition.wait()
            if not self._closed:
                self._pending.append((self._epoch, int(sequence), owned))
                self._condition.notify_all()

    def latest(self) -> dict[int, tuple[int, bytes]]:
        with self._condition:
            return dict(self._latest)

    def retained(
        self,
        sequence: int,
        *,
        epoch: int | None = None,
        timeout: float = 0.0,
    ) -> dict[int, tuple[int, bytes]]:
        with self._condition:
            key = (self._epoch if epoch is None else int(epoch), int(sequence))
            deadline = time.monotonic() + max(0.0, float(timeout))
            while key not in self._retained and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            return dict(self._retained.get(key, {}))

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait()
                if self._closed and not self._pending:
                    return
                pending = self._pending.popleft()
                self._condition.notify_all()
            epoch, sequence, frames = pending
            encoded: dict[int, tuple[int, bytes]] = {}
            for kind, frame in frames.items():
                packet = _frame_packet(
                    kind,
                    sequence,
                    frame,
                    session_epoch=epoch,
                )
                encoded[kind] = (sequence, packet)
                with self._condition:
                    self._latest[kind] = (sequence, packet)
            with self._condition:
                self._retained[(epoch, sequence)] = encoded
                while len(self._retained) > HISTORY_LIMIT:
                    del self._retained[next(iter(self._retained))]
                self._condition.notify_all()


@dataclass(frozen=True)
class PlaybackCommand:
    command_id: str
    client_id: str
    name: str
    payload: Mapping[str, Any]
    expected_revision: int | None


@dataclass(frozen=True)
class PlaybackResponse:
    client_id: str
    payload: Mapping[str, Any]


class _PlaybackRunnerProtocol:
    def _init_protocol(self, *, thread_name: str | None = None) -> None:
        self.responses: queue.SimpleQueue[PlaybackResponse] = queue.SimpleQueue()
        self.encoder = FrameEncoder()
        self.history: deque[dict[str, Any]] = deque(maxlen=HISTORY_LIMIT)
        self._snapshot_lock = threading.Lock()
        self._latest_snapshot: dict[str, Any] = {}
        self._stop = threading.Event()
        self.revision = 0
        if thread_name is not None:
            self.commands: queue.Queue[PlaybackCommand] = queue.Queue(COMMAND_QUEUE_LIMIT)
            self._episode_start_snapshot: dict[str, Any] = {}
            self._episode_start_frames: dict[int, tuple[int, bytes]] = {}
            self._thread = threading.Thread(target=self._run, name=thread_name)

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def start(self) -> None:
        self.encoder.start()
        self._publish()
        thread = getattr(self, "_thread", None)
        if thread is not None:
            thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = getattr(self, "_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=10.0)
        self.encoder.close()

    def submit(self, command: PlaybackCommand) -> None:
        self.commands.put_nowait(command)

    def snapshot(self) -> dict[str, Any]:
        with self._snapshot_lock:
            return dict(self._latest_snapshot)

    def episode_start_payload(
        self,
    ) -> tuple[dict[str, Any], dict[int, tuple[int, bytes]]]:
        with self._snapshot_lock:
            return (
                dict(getattr(self, "_episode_start_snapshot", {})),
                dict(getattr(self, "_episode_start_frames", {})),
            )

    def history_payload(self) -> dict[str, Any]:
        return {"type": "history", "points": list(self.history)}

    def _response(self, command: PlaybackCommand, *, ok: bool, **extra: Any) -> None:
        self.responses.put(
            PlaybackResponse(
                command.client_id,
                {
                    "type": "command_result",
                    "id": command.command_id,
                    "ok": ok,
                    "revision": self.revision,
                    **extra,
                },
            )
        )

    def _drain_commands(self) -> None:
        for _ in range(COMMAND_QUEUE_LIMIT):
            try:
                command = self.commands.get_nowait()
            except queue.Empty:
                return
            self._apply(command)


class WebPlaybackRunner(_PlaybackRunnerProtocol):
    """The only thread allowed to call the policy or environment."""

    def __init__(
        self,
        session: _PlaybackSession,
        args: argparse.Namespace,
        *,
        config_text: str,
        contract_details: Mapping[str, Any] | None = None,
        value_contract: Mapping[str, Any] | None = None,
    ) -> None:
        self._init_protocol(thread_name="gradlab-playback-runtime")
        self.session = session
        self.args = args
        self.config_text = ANSI_PATTERN.sub("", config_text)
        self.run_state = "paused"
        self.driver = "policy"
        capabilities_value = getattr(session, "policy_capabilities", {})
        self.policy_capabilities = (
            dict(capabilities_value) if isinstance(capabilities_value, Mapping) else {}
        )
        if not self.policy_capabilities:
            self.policy_capabilities = {
                "algorithm_id": None,
                "action_selection": {
                    "supported_modes": ["stochastic", "deterministic"],
                    "default_mode": "stochastic",
                },
                "introspection": [
                    "actor_distribution",
                    "state_value",
                    "selected_action_log_probability",
                    "entropy",
                ],
            }
        action_selection = self.policy_capabilities.get("action_selection")
        action_selection = dict(action_selection) if isinstance(action_selection, Mapping) else {}
        self.supported_action_selection_modes = tuple(
            str(mode) for mode in action_selection.get("supported_modes", ()) if str(mode)
        )
        if not self.supported_action_selection_modes:
            self.supported_action_selection_modes = ("stochastic", "deterministic")
        default_mode = str(action_selection.get("default_mode") or "")
        self.sampling_mode = default_mode or (
            self.supported_action_selection_modes[0]
            if self.supported_action_selection_modes
            else "stochastic"
        )
        self.target_fps = max(0.0, float(args.fps))
        self.remaining_steps = 0
        self.continue_target: str | None = None
        self.continue_count = 0
        self.boundaries = 0
        self.awaiting_next_episode = False
        self._input_lock = threading.Lock()
        self._pressed: tuple[str, ...] = ()
        self._input_updated_at = 0.0
        self._input_focused = False
        self._status_message: str | None = None
        self.environment_id = _session_environment_id(session, args)
        self.value_discount = _value_discount_factor(getattr(session, "model", None))
        self.contract_details = dict(contract_details or {})
        self.value_contract = dict(value_contract) if isinstance(value_contract, Mapping) else None

    def _critic_comparison_reasons(
        self,
        transition: _PlaybackTransition | None = None,
    ) -> list[str]:
        reasons = [
            str(reason)
            for reason in self.contract_details.get("comparison_reasons", [])
            if str(reason)
        ]
        introspection = set(self.policy_capabilities.get("introspection") or ())
        if "state_value" not in introspection:
            reasons.append("checkpoint does not expose a state-value critic")
        elif self.value_contract is None:
            reasons.append("checkpoint has no training value contract")
        else:
            expected_discount = self.value_contract.get("discount")
            if (
                self.value_discount is None
                or isinstance(expected_discount, bool)
                or not isinstance(expected_discount, int | float)
                or not np.isclose(self.value_discount, float(expected_discount))
            ):
                reasons.append("loaded model discount differs from training")
        if self.driver != "policy":
            reasons.append("human-driven trajectories are not on-policy")
        expected_selection = (
            str(self.value_contract.get("action_sampling") or "stochastic")
            if self.value_contract is not None
            else ""
        )
        if expected_selection and self.sampling_mode != expected_selection:
            reasons.append(
                "deterministic trajectories are not sampled from the training policy"
                if (expected_selection == "stochastic" and self.sampling_mode == "deterministic")
                else "active action selection differs from the critic training contract"
            )
        active_config = getattr(self.session, "config", None)
        base_config = getattr(self.session, "termination_base_config", active_config)
        active_task = (
            active_config.get("task")
            if isinstance(active_config, Mapping)
            else getattr(active_config, "task", None)
        )
        base_task = (
            base_config.get("task")
            if isinstance(base_config, Mapping)
            else getattr(base_config, "task", None)
        )
        if active_task != base_task:
            reasons.append("episode-boundary configuration differs from the active contract")
        if transition is not None and transition.truncated:
            reasons.append("truncated episodes require the training terminal-value bootstrap")
        return list(dict.fromkeys(reasons))

    def update_input(self, labels: Sequence[str], *, focused: bool) -> None:
        with self._input_lock:
            self._pressed = tuple(sorted({str(label).casefold() for label in labels}))
            self._input_updated_at = time.monotonic()
            self._input_focused = bool(focused)

    def clear_input(self) -> None:
        with self._input_lock:
            self._pressed = ()
            self._input_focused = False
            self._input_updated_at = 0.0

    def _snapshot_payload(self, transition: _PlaybackTransition | None) -> dict[str, Any]:
        current = transition_payload(transition) if transition is not None else None
        current_history = (
            dict(self.history[-1])
            if transition is not None
            and self.history
            and int(self.history[-1]["sequence"]) == transition.sequence
            else None
        )
        try:
            event_names = list(self.session.env.runtime.kernel.event_names)
        except AttributeError:
            event_names = []
        return {
            "type": "snapshot",
            "protocol": PROTOCOL_VERSION,
            "revision": self.revision,
            "sequence": self.session.sequence,
            "run_state": self.run_state,
            "driver": self.driver,
            "interactive": self.session.interactive,
            "policy": {
                **self.policy_capabilities,
                "provenance": dict(getattr(self.session, "policy_provenance", {})),
                "action_selection": {
                    "supported_modes": list(self.supported_action_selection_modes),
                    "default_mode": (
                        (self.policy_capabilities.get("action_selection") or {}).get("default_mode")
                        if isinstance(
                            self.policy_capabilities.get("action_selection"),
                            Mapping,
                        )
                        else None
                    ),
                    "requested_mode": self.sampling_mode,
                    "effective_mode": (
                        current.get("decision", {}).get("action_selection_mode")
                        if isinstance(current, Mapping)
                        and isinstance(current.get("decision"), Mapping)
                        else self.sampling_mode
                    ),
                },
            },
            "status_message": self._status_message,
            "session": {
                "episode": self.session.episode,
                "step": self.session.step_index,
                "seed": self.session.active_seed,
                "default_seed": getattr(
                    self.session,
                    "initial_seed",
                    self.session.active_seed,
                ),
                "task": _json_value(self.session.active_task),
                "total_reward": self.session.total_reward,
                "max_x_pos": self.session.max_x_pos,
                "action_names": list(self.session.action_names),
                "action_contract": _session_action_contract_payload(self.session),
                "action_contract_comparison": action_contract_payload(
                    getattr(self.session, "policy_provenance", {}).get("action_contract"),
                ),
                "event_names": event_names,
                "env_id": self.environment_id,
                "sampling_mode": self.sampling_mode,
                "value_discount": self.value_discount,
                "target_fps": self.target_fps,
                "episodes_limit": int(self.args.episodes),
                "awaiting_next_episode": self.awaiting_next_episode,
                "can_start_next_episode": self._can_start_next_episode(),
                "history_size": len(self.history),
                "config": self.config_text,
                "termination_source": getattr(
                    self.session,
                    "termination_source",
                    "training",
                ),
                "termination_conditions": list(getattr(self.session, "termination_conditions", ())),
                "playback_contract": dict(self.contract_details),
                "critic_comparison": {
                    "available": not self._critic_comparison_reasons(transition),
                    "reasons": self._critic_comparison_reasons(transition),
                    "discount": self.value_discount,
                },
            },
            "transition": current,
            "history_point": current_history,
        }

    def _publish(self, transition: _PlaybackTransition | None = None) -> None:
        if transition is not None and (
            not self.history or int(self.history[-1]["sequence"]) != transition.sequence
        ):
            self.history.append(history_point(transition))
            if transition.boundary:
                annotate_realized_returns(
                    self.history,
                    episode=transition.episode,
                    discount=self.value_discount,
                    comparison_reasons=self._critic_comparison_reasons(transition),
                )
        if transition is not None:
            game_frame = transition.after_frame
            obs_frames = transition.before_frames
            attribution = transition.attribution
            sequence = transition.sequence
        else:
            game_frame = self.session.current_frame
            obs_frames = tuple(self.session.frames or ())
            attribution = None
            sequence = self.session.sequence
        obs_image = None
        if obs_frames:
            obs_image = render_obs_stack(
                deque(obs_frames, maxlen=len(obs_frames)),
                scale=1,
                heatmap=attribution,
                heatmap_opacity=self.session.attribution_opacity,
            )
        self.encoder.submit_batch(
            sequence,
            {
                FRAME_GAME: game_frame,
                FRAME_OBSERVATION: obs_image,
            },
        )
        payload = self._snapshot_payload(transition)
        episode_start_frames: dict[int, tuple[int, bytes]] = {}
        if transition is None and self.session.step_index == 0:
            if game_frame is not None:
                episode_start_frames[FRAME_GAME] = (
                    sequence,
                    _frame_packet(
                        FRAME_GAME,
                        sequence,
                        game_frame,
                        session_epoch=self.encoder.epoch,
                    ),
                )
            if obs_image is not None:
                episode_start_frames[FRAME_OBSERVATION] = (
                    sequence,
                    _frame_packet(
                        FRAME_OBSERVATION,
                        sequence,
                        obs_image,
                        session_epoch=self.encoder.epoch,
                    ),
                )
        with self._snapshot_lock:
            self._latest_snapshot = payload
            if transition is None and self.session.step_index == 0:
                self._episode_start_snapshot = payload
                self._episode_start_frames = episode_start_frames

    def _set_state(self, state: str, *, message: str | None = None) -> None:
        self.run_state = state
        self._status_message = message
        self.revision += 1
        self._publish(self.session.last_transition)

    def _can_start_next_episode(self) -> bool:
        limit = int(self.args.episodes)
        return self.awaiting_next_episode and (limit <= 0 or self.boundaries < limit)

    def _require_active_episode(self) -> None:
        if self.awaiting_next_episode:
            raise ValueError("episode complete; choose Play next episode")

    @staticmethod
    def _validate_enabled_termination_conditions(enabled: object) -> list[str]:
        if not isinstance(enabled, list) or any(not isinstance(value, str) for value in enabled):
            raise ValueError("enabled termination conditions must be a list of ids")
        return enabled

    def _apply(self, command: PlaybackCommand) -> None:
        if (
            bool(command.payload.get("strict_revision", False))
            and command.expected_revision is not None
            and command.name not in {"pause", "stop"}
            and command.expected_revision != self.revision
        ):
            self._response(
                command,
                ok=False,
                error=f"stale revision {command.expected_revision}; current is {self.revision}",
            )
            return
        try:
            if command.name == "pause":
                self.remaining_steps = 0
                self.continue_target = None
                self.clear_input()
                self._set_state("paused", message="paused at a completed transition")
            elif command.name == "play":
                self._require_active_episode()
                self._set_state("playing")
            elif command.name == "step":
                self._require_active_episode()
                count = int(command.payload.get("count", 1))
                if not 1 <= count <= 100:
                    raise ValueError("step count must be in [1, 100]")
                self.remaining_steps = count
                self.continue_target = None
                self._set_state("stepping")
            elif command.name == "continue":
                self._require_active_episode()
                self.continue_target = str(command.payload.get("target") or "any")
                self.continue_count = 0
                self.remaining_steps = 0
                self._set_state("continuing")
            elif command.name == "next_episode":
                if not self.awaiting_next_episode:
                    raise ValueError("the current episode is still active")
                if not self._can_start_next_episode():
                    raise ValueError(f"episode limit reached ({self.boundaries})")
                mode = str(command.payload.get("sampling_mode") or self.sampling_mode)
                if mode not in self.supported_action_selection_modes:
                    supported = ", ".join(self.supported_action_selection_modes) or "none"
                    raise ValueError(
                        f"unsupported action-selection mode {mode!r}; supported: {supported}"
                    )
                driver = str(command.payload.get("driver") or self.driver)
                if driver not in {"policy", "human"}:
                    raise ValueError(f"unsupported driver {driver!r}")
                enabled_termination_conditions = command.payload.get(
                    "enabled_termination_conditions"
                )
                if enabled_termination_conditions is not None:
                    self.session.set_termination_conditions(
                        self._validate_enabled_termination_conditions(
                            enabled_termination_conditions
                        )
                    )
                self.session.last_transition = None
                self.sampling_mode = mode
                self.driver = driver
                self.clear_input()
                self.awaiting_next_episode = False
                self.remaining_steps = 0
                self.continue_target = None
                self._set_state(
                    "playing",
                    message="playing next episode",
                )
            elif command.name == "reset_episode":
                if self.awaiting_next_episode and not self._can_start_next_episode():
                    raise ValueError(f"episode limit reached ({self.boundaries})")
                seed_value = command.payload.get("seed")
                if isinstance(seed_value, bool):
                    raise ValueError("seed must be an integer")
                seed = None if seed_value in {None, ""} else validate_playback_seed(int(seed_value))
                enabled_termination_conditions = command.payload.get(
                    "enabled_termination_conditions"
                )
                if enabled_termination_conditions is not None:
                    enabled_termination_conditions = self._validate_enabled_termination_conditions(
                        enabled_termination_conditions
                    )
                self.session.reset_episode(seed)
                if enabled_termination_conditions is not None:
                    self.session.set_termination_conditions(enabled_termination_conditions)
                self.clear_input()
                self.awaiting_next_episode = False
                self.remaining_steps = 0
                self.continue_target = None
                self._set_state(
                    "paused",
                    message=f"episode reset · seed {self.session.active_seed}",
                )
            elif command.name == "set_fps":
                fps = float(command.payload.get("fps", 0.0))
                if fps < 0 or not np.isfinite(fps):
                    raise ValueError("fps must be a finite value >= 0")
                self.target_fps = fps
                self.revision += 1
                self._publish(self.session.last_transition)
            elif command.name == "set_termination_conditions":
                if self.session.step_index != 0 and not self.awaiting_next_episode:
                    raise ValueError(
                        "termination conditions can change before the first step "
                        "or between episodes"
                    )
                enabled = self._validate_enabled_termination_conditions(
                    command.payload.get("enabled")
                )
                was_awaiting_next_episode = self.awaiting_next_episode
                self.session.set_termination_conditions(enabled)
                self.session.last_transition = None
                self.awaiting_next_episode = was_awaiting_next_episode
                self.remaining_steps = 0
                self.continue_target = None
                self.clear_input()
                self._set_state(
                    "paused",
                    message=(
                        "termination conditions applied · choose Play next episode"
                        if self.awaiting_next_episode
                        else "termination conditions applied · episode ready"
                    ),
                )
            elif command.name == "stop":
                self._response(command, ok=True)
                self._stop.set()
                return
            else:
                raise ValueError(f"unknown playback command {command.name!r}")
        except Exception as exc:
            self._set_state("paused", message=str(exc))
            self._response(command, ok=False, error=str(exc))
            return
        self._response(command, ok=True)

    def _human_labels(self) -> tuple[str, ...]:
        with self._input_lock:
            fresh = time.monotonic() - self._input_updated_at <= INPUT_HEARTBEAT_SECONDS
            if not fresh or not self._input_focused:
                raise RuntimeError("human input lease expired; playback paused")
            return self._pressed

    def _step_once(self) -> _PlaybackTransition | None:
        try:
            transition = (
                self.session.step_human(self._human_labels())
                if self.driver == "human"
                else (
                    self.session.step(action_selection_mode=self.sampling_mode)
                    if hasattr(self.session, "policy_runtime")
                    else self.session.step(deterministic=self.sampling_mode == "deterministic")
                )
            )
        except Exception as exc:
            self._set_state("paused", message=str(exc))
            return None
        self.revision += 1
        if transition.boundary:
            self.boundaries += 1
            self.awaiting_next_episode = True
            self.remaining_steps = 0
            self.continue_target = None
            self.run_state = "paused"
            if self._can_start_next_episode():
                self._status_message = "episode complete · choose Play next episode"
            else:
                self._status_message = f"episode limit reached ({self.boundaries})"
        elif self.run_state == "stepping":
            self.remaining_steps -= 1
            if self.remaining_steps <= 0:
                self.run_state = "paused"
        elif self.run_state == "continuing":
            self.continue_count += 1
            target = self.continue_target or "any"
            matched = (
                transition.boundary
                if target == "done"
                else bool(transition.events)
                if target == "any"
                else target in transition.events
            )
            if matched or transition.boundary or self.continue_count >= 10_000:
                self.run_state = "paused"
                if self.continue_count >= 10_000 and not matched:
                    self._status_message = "continue reached the 10,000-step safety limit"
        self._publish(transition)
        return transition

    def _run(self) -> None:
        next_step_at = time.perf_counter()
        while not self._stop.is_set():
            self._drain_commands()
            if self.run_state not in {"playing", "stepping", "continuing"}:
                time.sleep(0.005)
                continue
            fps = 60.0 if self.driver == "human" and self.target_fps <= 0 else self.target_fps
            if fps > 0:
                now = time.perf_counter()
                if now < next_step_at:
                    time.sleep(min(next_step_at - now, 0.005))
                    continue
                next_step_at = max(next_step_at + 1.0 / fps, now)
            self._step_once()


class DatasetPlaybackRunner(_PlaybackRunnerProtocol):
    """Provider-free playback for one verified Gymrec episode."""

    def __init__(
        self,
        frames: Iterable[np.ndarray],
        rows: Sequence[Mapping[str, Any]],
        args: argparse.Namespace,
        *,
        fps: float,
        action_contract: Mapping[str, Any] | None = None,
    ) -> None:
        if len(rows) < 2:
            raise ValueError("dataset playback requires at least one transition")
        self._init_protocol(thread_name="gradlab-dataset-playback")
        self.args = args
        self.rows = rows
        self._frames = iter(frames)
        self.current_frame = np.asarray(next(self._frames), dtype=np.uint8)
        self.sequence = 0
        self.transition_index = 0
        self.total_reward = 0.0
        self.run_state = "paused"
        self.target_fps = max(0.0, float(fps))
        self.remaining_steps = 0
        self.continue_target: str | None = None
        self.continue_count = 0
        self._status_message = "verified recorded episode ready"
        first = rows[0]
        self.environment_id = str(first.get("env_id") or "") or None
        self.episode_id = str(first.get("episode_id") or "")
        self.seed = int(first.get("seed") or 0)
        self.sampling_mode = str(first.get("policy_mode") or "recorded")
        self.action_contract = (
            dict(action_contract) if isinstance(action_contract, Mapping) else None
        )
        self.action_contract_payload = action_contract_payload(self.action_contract)
        self._transition: dict[str, Any] | None = None

    def update_input(self, _labels: Sequence[str], *, focused: bool) -> None:
        del focused

    def clear_input(self) -> None:
        return None

    def _snapshot_payload(self) -> dict[str, Any]:
        current_history = (
            dict(self.history[-1])
            if self._transition is not None
            and self.history
            and int(self.history[-1]["sequence"]) == self.sequence
            else None
        )
        return {
            "type": "snapshot",
            "protocol": PROTOCOL_VERSION,
            "mode": "dataset",
            "revision": self.revision,
            "sequence": self.sequence,
            "run_state": self.run_state,
            "driver": "recorded",
            "interactive": False,
            "policy": None,
            "status_message": self._status_message,
            "session": {
                "episode": 1,
                "step": self.transition_index,
                "seed": self.seed,
                "task": None,
                "total_reward": self.total_reward,
                "max_x_pos": _max_x_pos(self._transition),
                "action_names": [],
                "action_contract": self.action_contract_payload,
                "event_names": [],
                "env_id": self.environment_id,
                "sampling_mode": self.sampling_mode,
                "target_fps": self.target_fps,
                "episodes_limit": 1,
                "awaiting_next_episode": self.transition_index >= len(self.rows) - 1,
                "can_start_next_episode": False,
                "history_size": len(self.history),
                "config": (
                    f"Gymrec v3 episode {self.episode_id}\n"
                    "Recorded dataset playback is never checkpoint-promotion evidence."
                ),
            },
            "transition": self._transition,
            "history_point": current_history,
        }

    def _publish(self) -> None:
        self.encoder.submit(FRAME_GAME, self.sequence, self.current_frame)
        payload = self._snapshot_payload()
        with self._snapshot_lock:
            self._latest_snapshot = payload
            if self.sequence == 0:
                self._episode_start_snapshot = payload
                self._episode_start_frames = {
                    FRAME_GAME: (
                        self.sequence,
                        _frame_packet(
                            FRAME_GAME,
                            self.sequence,
                            self.current_frame,
                            session_epoch=self.encoder.epoch,
                        ),
                    )
                }

    def _set_state(self, state: str, *, message: str | None = None) -> None:
        self.run_state = state
        self._status_message = message
        self.revision += 1
        self._publish()

    def _apply(self, command: PlaybackCommand) -> None:
        try:
            complete = self.transition_index >= len(self.rows) - 1
            if command.name == "pause":
                self.remaining_steps = 0
                self.continue_target = None
                self._set_state("paused", message="paused at a recorded transition")
            elif command.name == "play":
                if complete:
                    raise ValueError("recorded episode is complete")
                self.remaining_steps = 0
                self.continue_target = None
                self._set_state("playing")
            elif command.name == "step":
                if complete:
                    raise ValueError("recorded episode is complete")
                count = int(command.payload.get("count", 1))
                if not 1 <= count <= 100:
                    raise ValueError("step count must be in [1, 100]")
                self.remaining_steps = count
                self.continue_target = None
                self._set_state("stepping")
            elif command.name == "continue":
                if complete:
                    raise ValueError("recorded episode is complete")
                self.remaining_steps = 0
                self.continue_target = str(command.payload.get("target") or "any")
                self.continue_count = 0
                self._set_state("continuing")
            elif command.name == "set_fps":
                fps = float(command.payload.get("fps", 0.0))
                if fps < 0 or not np.isfinite(fps):
                    raise ValueError("fps must be a finite value >= 0")
                self.target_fps = fps
                self.revision += 1
                self._publish()
            elif command.name == "stop":
                self._response(command, ok=True)
                self._stop.set()
                return
            elif command.name in {"next_episode", "set_driver", "restart"}:
                raise ValueError("dataset playback is read-only")
            else:
                raise ValueError(f"unknown playback command {command.name!r}")
        except Exception as exc:
            self._set_state("paused", message=str(exc))
            self._response(command, ok=False, error=str(exc))
            return
        self._response(command, ok=True)

    def _step_once(self) -> None:
        row = self.rows[self.transition_index]
        terminal = self.rows[self.transition_index + 1]
        self.current_frame = np.asarray(next(self._frames), dtype=np.uint8)
        self.transition_index += 1
        self.sequence += 1
        reward = float(row["rewards"])
        self.total_reward += reward
        self._status_message = None
        info = _dataset_info(row.get("infos"))
        terminated = bool(row.get("terminations"))
        truncated = bool(row.get("truncations"))
        collector_terminated = bool(terminal.get("collector_terminated"))
        boundary = self.transition_index >= len(self.rows) - 1
        action = _json_value(row.get("actions"))
        events_value = info.get("events", ())
        events = (
            [str(value) for value in events_value]
            if isinstance(events_value, Sequence)
            and not isinstance(events_value, str | bytes | bytearray)
            else []
        )
        boundary_reasons = [
            reason
            for active, reason in (
                (terminated, "provider_terminated"),
                (truncated, "provider_truncated"),
                (collector_terminated, "collector_terminated"),
            )
            if active
        ]
        self._transition = {
            "sequence": self.sequence,
            "episode": 1,
            "step": int(row.get("step_index") or 0),
            "seed": int(row.get("seed") or self.seed),
            "start_id": self.episode_id,
            "action_source": "recorded",
            "executed_action": action,
            "decision": None,
            "before": {
                "task": None,
                "model_input": [],
                "game_frame": True,
                "observation_frames": 0,
            },
            "after": {"task": None, "game_frame": True, "observation_frames": 0},
            "reward": {
                "provider": reward,
                "shaped": reward,
                "step": reward,
                "return": self.total_reward,
                "components": {},
            },
            "events": events,
            "event_transitions": {},
            "signals": _numeric_signals(info),
            "info": _json_value(info),
            "max_x_pos": _info_max_x_pos(info),
            "terminated": terminated,
            "truncated": truncated,
            "completed": terminated,
            "boundary": boundary,
            "boundary_reasons": boundary_reasons,
            "outcome": (
                "terminated"
                if terminated
                else "truncated"
                if truncated
                else "collector_terminated"
                if collector_terminated
                else "continuing"
            ),
            "attribution": False,
        }
        self.history.append(history_point_payload(self._transition))
        self.revision += 1
        if boundary:
            self.run_state = "paused"
            self.remaining_steps = 0
            self.continue_target = None
            self._status_message = "recorded episode complete"
        elif self.run_state == "stepping":
            self.remaining_steps -= 1
            if self.remaining_steps <= 0:
                self.run_state = "paused"
        elif self.run_state == "continuing":
            self.continue_count += 1
            target = self.continue_target or "any"
            matched = bool(events) if target == "any" else target in events
            if matched or self.continue_count >= 10_000:
                self.run_state = "paused"
                if self.continue_count >= 10_000 and not matched:
                    self._status_message = "continue reached the 10,000-step safety limit"
        self._publish()

    def _run(self) -> None:
        next_step_at = time.perf_counter()
        while not self._stop.is_set():
            self._drain_commands()
            if self.run_state not in {"playing", "stepping", "continuing"}:
                time.sleep(0.005)
                continue
            if self.target_fps > 0:
                now = time.perf_counter()
                if now < next_step_at:
                    time.sleep(min(next_step_at - now, 0.005))
                    continue
                next_step_at = max(next_step_at + 1.0 / self.target_fps, now)
            try:
                self._step_once()
            except (StopIteration, IndexError) as exc:
                self._set_state("paused", message=f"recorded media ended early: {exc}")


def _dataset_info(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str):
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, Mapping) else {}


def _info_max_x_pos(info: Mapping[str, Any]) -> float | None:
    for key in ("max_x_pos", "x_pos", "x"):
        value = info.get(key)
        if isinstance(value, int | float | np.number) and np.isfinite(value):
            return float(value)
    return None


def _max_x_pos(transition: Mapping[str, Any] | None) -> float | None:
    if transition is None:
        return None
    value = transition.get("max_x_pos")
    return float(value) if isinstance(value, int | float) else None


class HumanRecordingRunner(_PlaybackRunnerProtocol):
    """Bridge the synchronous dataset recorder to the shared web dashboard."""

    def __init__(self, session: Any, args: argparse.Namespace) -> None:
        self._init_protocol()
        self.session = session
        self.args = args
        self._condition = threading.Condition()
        self._pressed: tuple[str, ...] = ()
        self._input_updated_at = 0.0
        self._input_focused = False
        self._next_action_at = time.perf_counter()
        self._transition: dict[str, Any] | None = None
        self._last_action: Any = None
        self.total_reward = 0.0
        self.sequence = 0
        self.run_state = "paused"
        self.target_fps = max(float(getattr(args, "fps", None) or session.fps), 1.0)
        self._status_message = "Focus the game view, then press Play to begin recording"
        self.environment_id = _session_environment_id(session, args)
        self.action_contract_payload = _session_action_contract_payload(session)

    def stop(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        self.encoder.close()

    def clear_input(self) -> None:
        with self._condition:
            self._pressed = ()
            self._input_focused = False
            self._input_updated_at = 0.0
            self._condition.notify_all()

    def update_input(self, labels: Sequence[str], *, focused: bool) -> None:
        with self._condition:
            self._pressed = tuple(sorted({str(label).upper() for label in labels}))
            self._input_updated_at = time.monotonic()
            self._input_focused = bool(focused)
            self._condition.notify_all()

    def submit(self, command: PlaybackCommand) -> None:
        try:
            if command.name == "play":
                self.run_state = "playing"
                self._status_message = "Recording human controls"
            elif command.name == "pause":
                self.run_state = "paused"
                self.clear_input()
                self._status_message = "Recording paused"
            elif command.name == "set_fps":
                fps = float(command.payload.get("fps", self.target_fps))
                if not np.isfinite(fps) or fps <= 0:
                    raise ValueError("recording FPS must be a finite value > 0")
                self.target_fps = fps
                self._status_message = f"Recording at {fps:g} FPS"
            elif command.name == "set_driver":
                if command.payload.get("driver") != "human":
                    raise ValueError("dataset recording only supports human control")
                self.run_state = "paused"
                self._status_message = "Human control selected"
            elif command.name == "stop":
                self._response(command, ok=True)
                self.stop()
                return
            else:
                raise ValueError(f"{command.name or 'that command'} is unavailable while recording")
        except Exception as exc:
            self._status_message = str(exc)
            self._response(command, ok=False, error=str(exc))
        else:
            self._response(command, ok=True)
        finally:
            self.revision += 1
            self._publish()
            with self._condition:
                self._condition.notify_all()

    def _publish(self) -> None:
        with self._snapshot_lock:
            current_history = (
                dict(self.history[-1])
                if self._transition is not None
                and self.history
                and int(self.history[-1]["sequence"]) == self.sequence
                else None
            )
            self._latest_snapshot = {
                "type": "snapshot",
                "protocol": PROTOCOL_VERSION,
                "mode": "recording",
                "revision": self.revision,
                "sequence": self.sequence,
                "run_state": self.run_state,
                "driver": "human",
                "interactive": True,
                "policy": None,
                "status_message": self._status_message,
                "session": {
                    "episode": 1,
                    "step": self.sequence,
                    "seed": None,
                    "task": None,
                    "total_reward": self.total_reward,
                    "max_x_pos": 0,
                    "action_names": [],
                    "action_contract": self.action_contract_payload,
                    "event_names": [],
                    "env_id": self.environment_id,
                    "sampling_mode": None,
                    "target_fps": self.target_fps,
                    "episodes_limit": int(getattr(self.args, "episodes", None) or 0),
                    "awaiting_next_episode": False,
                    "can_start_next_episode": False,
                    "history_size": len(self.history),
                    "config": (
                        "Human dataset recording. Browser input is translated through the "
                        "provider's declared control labels. This session is never promotion evidence."
                    ),
                },
                "transition": self._transition,
                "history_point": current_history,
            }

    def action(self, frame: np.ndarray) -> tuple[Any | None, bool]:
        self.encoder.submit(FRAME_GAME, self.sequence, frame)
        self._publish()
        while not self.stopped:
            with self._condition:
                fresh = time.monotonic() - self._input_updated_at <= INPUT_HEARTBEAT_SECONDS
                if not (self.run_state == "playing" and self._input_focused and fresh):
                    self._condition.wait(timeout=0.05)
                    continue
                labels = self._pressed
            now = time.perf_counter()
            if now < self._next_action_at:
                time.sleep(self._next_action_at - now)
            self._next_action_at = max(self._next_action_at + 1.0 / self.target_fps, now)
            try:
                action = self.session.action_from_labels(labels)
            except ValueError as exc:
                self.run_state = "paused"
                self.clear_input()
                self._status_message = str(exc)
                self.revision += 1
                self._publish()
                continue
            self.sequence += 1
            self.revision += 1
            self._last_action = _json_value(action)
            return action, True
        return None, False

    def observe_transition(
        self,
        *,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
        next_frame: np.ndarray,
    ) -> None:
        self.total_reward += float(reward)
        boundary = bool(terminated or truncated)
        self._transition = {
            "sequence": self.sequence,
            "episode": 1,
            "step": self.sequence,
            "seed": None,
            "start_id": None,
            "action_source": "human",
            "executed_action": self._last_action,
            "decision": None,
            "before": {
                "task": None,
                "model_input": [],
                "game_frame": True,
                "observation_frames": 0,
            },
            "after": {"task": None, "game_frame": True, "observation_frames": 0},
            "reward": {
                "provider": float(reward),
                "shaped": float(reward),
                "step": float(reward),
                "return": self.total_reward,
                "components": {},
            },
            "events": [],
            "event_transitions": {},
            "signals": _numeric_signals(info),
            "info": _json_value(info),
            "max_x_pos": int(info.get("max_x_pos", 0)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "completed": False,
            "boundary": boundary,
            "boundary_reasons": [
                name
                for name, active in (
                    ("provider_terminated", terminated),
                    ("provider_truncated", truncated),
                )
                if active
            ],
            "outcome": "boundary" if boundary else "continuing",
            "attribution": False,
        }
        self.history.append(history_point_payload(self._transition))
        self.encoder.submit(FRAME_GAME, self.sequence, next_frame)
        self.revision += 1
        self._publish()


class WebClient:
    def __init__(
        self,
        client_id: str,
        socket: web.WebSocketResponse,
        subscriptions: set[str],
        workspace_id: str,
        window_id: str,
    ) -> None:
        self.client_id = client_id
        self.socket = socket
        self.subscriptions = subscriptions
        self.workspace_id = workspace_id
        self.window_id = window_id
        self.reliable: asyncio.Queue[str | bytes] = asyncio.Queue(CLIENT_QUEUE_LIMIT)
        self.event = asyncio.Event()
        self.latest_snapshot: str | None = None
        self.latest_snapshot_key: tuple[int, int, int, int] = (-1, -1, -1, -1)
        self.sent_snapshot_key: tuple[int, int, int, int] = (-1, -1, -1, -1)
        self.latest_frames: dict[int, tuple[int, bytes]] = {}
        self.sent_frames: dict[int, tuple[int, bytes]] = {}
        self.closed = False

    def offer_reliable(self, payload: Mapping[str, Any] | bytes) -> None:
        rendered = (
            payload
            if isinstance(payload, bytes)
            else json.dumps(payload, separators=(",", ":"), allow_nan=False)
        )
        try:
            self.reliable.put_nowait(rendered)
        except asyncio.QueueFull:
            self.closed = True
        self.event.set()

    def offer_snapshot(self, payload: Mapping[str, Any]) -> None:
        key = (
            int(payload.get("session_epoch", 0)),
            int(payload.get("revision", 0)),
            int(payload.get("sequence", 0)),
            int(payload.get("control_epoch", 0)),
        )
        if key >= self.latest_snapshot_key:
            self.latest_snapshot_key = key
            self.latest_snapshot = json.dumps(payload, separators=(",", ":"), allow_nan=False)
            self.event.set()

    def offer_frame(self, kind: int, sequence: int, packet: bytes) -> None:
        if sequence >= self.latest_frames.get(kind, (-1, b""))[0]:
            self.latest_frames[kind] = (sequence, packet)
            self.event.set()

    def reset_session(self, epoch: int) -> None:
        self.latest_snapshot = None
        self.latest_snapshot_key = (int(epoch), -1, -1, -1)
        self.sent_snapshot_key = (int(epoch), -1, -1, -1)
        self.latest_frames.clear()
        self.sent_frames.clear()
        self.event.set()

    async def write(self) -> None:
        while not self.closed and not self.socket.closed:
            await self.event.wait()
            self.event.clear()
            while not self.reliable.empty():
                value = self.reliable.get_nowait()
                if isinstance(value, bytes):
                    await self.socket.send_bytes(value)
                else:
                    await self.socket.send_str(value)
            if (
                self.latest_snapshot is not None
                and self.latest_snapshot_key > self.sent_snapshot_key
            ):
                await self.socket.send_str(self.latest_snapshot)
                self.sent_snapshot_key = self.latest_snapshot_key
            for kind, (sequence, packet) in tuple(self.latest_frames.items()):
                if (sequence, packet) != self.sent_frames.get(kind):
                    await self.socket.send_bytes(packet)
                    self.sent_frames[kind] = (sequence, packet)


class PlaybackWebServer:
    def __init__(
        self,
        runner: Any,
        args: argparse.Namespace,
        *,
        paired_windows: bool = False,
        catalog: Any | None = None,
        defer_secondary_window: bool = False,
        manual_evaluation_factory: Any | None = None,
    ) -> None:
        self.runner = runner
        self.args = args
        self.paired_windows = paired_windows
        self.catalog = catalog
        self.defer_secondary_window = bool(defer_secondary_window)
        self.token = secrets.token_urlsafe(32)
        self.origin = ""
        self.clients: dict[str, WebClient] = {}
        self.control_holder: str | None = None
        self.input_holder: str | None = None
        self.control_epoch = 0
        self.stop_event = asyncio.Event()
        self.ever_connected = False
        self.last_client_at = time.monotonic()
        self._auto_started_epoch = -1
        self._auto_start_task: asyncio.Task[None] | None = None
        self._auto_start_task_epoch = -1
        self._observed_session_change = int(getattr(self.runner, "session_change", 0))
        self._secondary_opened = False
        self._catalog_entity = ""
        self._initial_environment_catalog: dict[str, Any] | None = None
        self._manual_evaluation_factory = manual_evaluation_factory
        self._manual_evaluation_queue: Any | None = None

    @property
    def asset_root(self) -> Path:
        return Path(__file__).with_name("web_player")

    def dashboard_urls(self) -> tuple[str, ...]:
        main_path = "/"
        if self.catalog is not None:
            snapshot = self.runner.snapshot()
            app = snapshot.get("app") if isinstance(snapshot, Mapping) else None
            route = app.get("route") if isinstance(app, Mapping) else None
            main_path = source_browser_path(route if isinstance(route, Mapping) else None)
        if self.paired_windows:
            query = "?workspace=paired"
            return (
                f"{self.origin}{main_path}{query}#token={self.token}",
                f"{self.origin}/workspace/stats{query}#token={self.token}",
            )
        return (f"{self.origin}{main_path}#token={self.token}",)

    @web.middleware
    async def security_headers(self, request: web.Request, handler):
        response = await handler(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; "
            "connect-src 'self' ws: wss:; object-src 'none'; base-uri 'none'; "
            "frame-ancestors 'none'"
        )
        return response

    async def page(self, _request: web.Request) -> web.FileResponse:
        return web.FileResponse(self.asset_root / "index.html")

    async def asset(self, request: web.Request) -> web.FileResponse:
        relative = Path(request.match_info["path"])
        root = self.asset_root.resolve()
        candidate = (root / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or candidate.suffix not in {".js", ".css", ".svg"}
            or not candidate.is_relative_to(root)
            or not candidate.is_file()
        ):
            raise web.HTTPNotFound()
        return web.FileResponse(candidate)

    def _authorize_api(self, request: web.Request) -> None:
        origin = request.headers.get("Origin")
        if origin and origin != self.origin:
            raise web.HTTPForbidden(text="invalid request origin")
        authorization = request.headers.get("Authorization", "")
        scheme, _, supplied = authorization.partition(" ")
        if scheme.casefold() != "bearer" or not secrets.compare_digest(
            supplied,
            self.token,
        ):
            raise web.HTTPUnauthorized(text="catalog token required")

    async def catalog_environments(self, request: web.Request) -> web.Response:
        self._authorize_api(request)
        if self.catalog is None:
            raise web.HTTPNotFound()
        from gradlab.play_catalog import normalize_search_query

        try:
            requested_entity = str(
                request.query.get("entity") or getattr(self.args, "wandb_entity", None) or ""
            ).strip()
            entity = requested_entity or self._catalog_entity
            if not entity:
                entity = await asyncio.to_thread(self.catalog.default_entity)
                self._catalog_entity = entity
            page = await asyncio.to_thread(
                self.catalog.environments,
                entity=entity,
                query=normalize_search_query(request.query.get("q")),
                cursor=request.query.get("cursor"),
            )
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=502)
        return web.json_response({"entity": entity, **page.to_dict()})

    async def _prepare_initial_catalog(self) -> None:
        if self.catalog is None:
            return
        initial_environments = getattr(self.catalog, "initial_environments", None)
        if not callable(initial_environments):
            return
        payload = await asyncio.to_thread(
            initial_environments,
            getattr(self.args, "wandb_entity", None),
        )
        if not isinstance(payload, Mapping):
            raise ValueError("initial environment catalog must be a mapping")
        self._catalog_entity = str(payload.get("entity") or "").strip()
        self._initial_environment_catalog = dict(payload)

    async def catalog_runs(self, request: web.Request) -> web.Response:
        self._authorize_api(request)
        if self.catalog is None:
            raise web.HTTPNotFound()
        from gradlab.play_catalog import normalize_search_query

        try:
            page = await asyncio.to_thread(
                self.catalog.runs,
                entity=request.match_info["entity"],
                project=request.match_info["project"],
                goal_id=request.match_info.get("goal_id", ""),
                goal_variant_id=request.match_info.get("goal_variant_id", ""),
                query=normalize_search_query(request.query.get("q")),
                cursor=request.query.get("cursor"),
            )
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=502)
        return web.json_response(page.to_dict())

    async def catalog_goal_variants(self, request: web.Request) -> web.Response:
        self._authorize_api(request)
        if self.catalog is None:
            raise web.HTTPNotFound()
        from gradlab.play_catalog import normalize_search_query

        try:
            page = await asyncio.to_thread(
                self.catalog.goal_variants,
                entity=request.match_info["entity"],
                project=request.match_info["project"],
                goal_id=request.match_info["goal_id"],
                query=normalize_search_query(request.query.get("q")),
                cursor=request.query.get("cursor"),
            )
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=502)
        return web.json_response(page.to_dict())

    async def catalog_goals(self, request: web.Request) -> web.Response:
        self._authorize_api(request)
        if self.catalog is None:
            raise web.HTTPNotFound()
        from gradlab.play_catalog import normalize_search_query

        try:
            page = await asyncio.to_thread(
                self.catalog.goals,
                entity=request.match_info["entity"],
                project=request.match_info["project"],
                query=normalize_search_query(request.query.get("q")),
                cursor=request.query.get("cursor"),
            )
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=502)
        return web.json_response(page.to_dict())

    async def catalog_recipes(self, request: web.Request) -> web.Response:
        self._authorize_api(request)
        if self.catalog is None:
            raise web.HTTPNotFound()
        from gradlab.play_catalog import normalize_search_query

        try:
            page = await asyncio.to_thread(
                self.catalog.recipes,
                entity=request.match_info["entity"],
                project=request.match_info["project"],
                goal_id=request.match_info["goal_id"],
                query=normalize_search_query(request.query.get("q")),
                cursor=request.query.get("cursor"),
            )
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=502)
        return web.json_response(page.to_dict())

    async def inspect_goal(self, request: web.Request) -> web.Response:
        self._authorize_api(request)
        if self.catalog is None:
            raise web.HTTPNotFound()
        try:
            document = await asyncio.to_thread(
                self.catalog.inspect_goal,
                entity=request.match_info["entity"],
                project=request.match_info["project"],
                goal_id=request.match_info["goal_id"],
            )
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=502)
        return web.json_response(document)

    async def inspect_recipe(self, request: web.Request) -> web.Response:
        self._authorize_api(request)
        if self.catalog is None:
            raise web.HTTPNotFound()
        try:
            document = await asyncio.to_thread(
                self.catalog.inspect_recipe,
                entity=request.match_info["entity"],
                project=request.match_info["project"],
                goal_id=request.match_info["goal_id"],
                recipe_id=request.match_info["recipe_id"],
            )
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=502)
        return web.json_response(document)

    async def inspect_goal_variant(self, request: web.Request) -> web.Response:
        self._authorize_api(request)
        if self.catalog is None:
            raise web.HTTPNotFound()
        try:
            document = await asyncio.to_thread(
                self.catalog.inspect_goal_variant,
                entity=request.match_info["entity"],
                project=request.match_info["project"],
                goal_id=request.match_info["goal_id"],
                variant_id=request.match_info["goal_variant_id"],
            )
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=502)
        return web.json_response(document)

    async def inspect_run(self, request: web.Request) -> web.Response:
        self._authorize_api(request)
        if self.catalog is None:
            raise web.HTTPNotFound()
        try:
            document = await asyncio.to_thread(
                self.catalog.inspect_run,
                entity=str(request.query.get("entity") or ""),
                project=str(request.query.get("project") or ""),
                run_id=request.match_info["run_id"],
            )
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=502)
        return web.json_response(document)

    async def inspect_active_playback(self, request: web.Request) -> web.Response:
        self._authorize_api(request)
        from gradlab.contract_inspection import inspection_document

        active_recipe = getattr(self.runner, "active_recipe_document", None)
        resolved = await asyncio.to_thread(active_recipe) if callable(active_recipe) else None
        if resolved is None or self.catalog is None:
            message = "No verified policy bundle is active in the player."
            unavailable_goal = inspection_document(
                kind="goal",
                title="Active playback",
                availability="unavailable",
                message=message,
            )
            unavailable_recipe = inspection_document(
                kind="recipe",
                title="Active playback",
                availability="unavailable",
                message=message,
            )
            return web.json_response(
                {
                    "schema_version": 1,
                    "source": {"kind": "active-playback"},
                    "documents": {
                        "goal": unavailable_goal,
                        "recipe": unavailable_recipe,
                    },
                }
            )
        recipe_document, source = resolved
        try:
            document = await asyncio.to_thread(
                self.catalog.inspect_portable_recipe,
                recipe_document,
                source=source,
            )
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=502)
        return web.json_response(document)

    async def catalog_checkpoints(self, request: web.Request) -> web.Response:
        self._authorize_api(request)
        if self.catalog is None:
            raise web.HTTPNotFound()
        from gradlab.play_catalog import (
            checkpoint_training_metric_columns,
            normalize_search_query,
        )

        try:
            items = list(
                await asyncio.to_thread(
                    self.catalog.checkpoints,
                    run_id=request.match_info["run_id"],
                    query=normalize_search_query(request.query.get("q")),
                    entity=request.query.get("entity", ""),
                    project=request.query.get("project", ""),
                    goal_variant_id=request.query.get("goal_variant_id", ""),
                )
            )
            queue_service = await asyncio.to_thread(self._manual_evaluations)
            if queue_service is not None:
                statuses = await asyncio.to_thread(
                    queue_service.statuses,
                    run_id=request.match_info["run_id"],
                    checkpoint_ids=[
                        str(item.get("checkpoint_id") or "")
                        for item in items
                        if isinstance(item, Mapping)
                    ],
                )
                items = [
                    {
                        **dict(item),
                        "evaluation": (
                            statuses[str(item["checkpoint_id"])]["evaluation"]
                            if statuses.get(str(item["checkpoint_id"]), {}).get("evaluation")
                            is not None
                            else item.get("evaluation")
                        ),
                        "evaluation_queue": statuses.get(str(item["checkpoint_id"])),
                    }
                    for item in items
                ]
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=502)
        return web.json_response(
            {
                "items": items,
                "next_cursor": None,
                "metric_columns": list(checkpoint_training_metric_columns()),
            }
        )

    def _manual_evaluations(self) -> Any:
        if self._manual_evaluation_queue is not None:
            return self._manual_evaluation_queue
        factory = self._manual_evaluation_factory
        if factory is None:
            if self.catalog is None:
                raise RuntimeError("checkpoint evaluation is unavailable without a catalog")
            from gradlab.manual_evaluation import build_manual_evaluation_queue

            repo_root = Path(
                getattr(self.catalog, "repo_root", Path(__file__).resolve().parents[2])
            )

            def factory() -> Any:
                return build_manual_evaluation_queue(repo_root)

        self._manual_evaluation_queue = factory()
        return self._manual_evaluation_queue

    async def catalog_evaluate_checkpoints(self, request: web.Request) -> web.Response:
        self._authorize_api(request)
        if self.catalog is None:
            raise web.HTTPNotFound()
        try:
            payload = await request.json()
        except json.JSONDecodeError, TypeError:
            return web.json_response({"error": "request body must be JSON"}, status=400)
        checkpoint_ids = payload.get("checkpoint_ids") if isinstance(payload, Mapping) else None
        if isinstance(checkpoint_ids, str | bytes) or not isinstance(checkpoint_ids, Sequence):
            return web.json_response(
                {"error": "checkpoint_ids must be a JSON array"},
                status=400,
            )
        try:
            queue_service = await asyncio.to_thread(self._manual_evaluations)
        except Exception as exc:
            return web.json_response(
                {"error": f"manual evaluation is unavailable: {exc}"},
                status=503,
            )
        try:
            result = await asyncio.to_thread(
                queue_service.enqueue,
                run_id=request.match_info["run_id"],
                checkpoint_ids=[str(value) for value in checkpoint_ids],
            )
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=502)
        if isinstance(result, Mapping):
            response = dict(result)
            response["items"] = list(response.get("items") or ())
        else:
            response = {"items": list(result)}
        return web.json_response(response, status=202)

    def _snapshot_for(self, client: WebClient, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        payload = {
            **snapshot,
            "control_epoch": self.control_epoch,
            "control": {
                "client_id": client.client_id,
                "workspace_id": client.workspace_id,
                "window_id": client.window_id,
                "holder": self.control_holder,
                "input_holder": self.input_holder,
                "has_control": self.control_holder == client.workspace_id,
            },
        }
        app = payload.get("app")
        if (
            self._initial_environment_catalog is not None
            and isinstance(app, Mapping)
            and app.get("phase") == "selecting"
        ):
            payload["app"] = {
                **app,
                "catalog": dict(self._initial_environment_catalog),
            }
        return payload

    def _broadcast_control(self) -> None:
        snapshot = self.runner.snapshot()
        for client in self.clients.values():
            client.offer_snapshot(self._snapshot_for(client, snapshot))

    def _runner_epoch(self) -> int:
        return int(getattr(self.runner, "session_epoch", 0))

    def _runner_active(self) -> bool:
        return bool(getattr(self.runner, "has_active_runner", True))

    def _cancel_auto_start_task(self) -> None:
        task = self._auto_start_task
        self._auto_start_task = None
        self._auto_start_task_epoch = -1
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()

    def _auto_start_client(self, preferred_client_id: str | None = None) -> str | None:
        preferred = self.clients.get(preferred_client_id or "")
        if preferred is not None and (
            self.control_holder is None or preferred.workspace_id == self.control_holder
        ):
            return preferred.client_id
        for client in self.clients.values():
            if self.control_holder is None or client.workspace_id == self.control_holder:
                return client.client_id
        return None

    def _paired_workspace_ready(self) -> bool:
        if not self.paired_windows or self.control_holder is None:
            return not self.paired_windows
        windows = {
            client.window_id
            for client in self.clients.values()
            if client.workspace_id == self.control_holder
        }
        return {"main", "stats"}.issubset(windows)

    def _start_epoch(self, epoch: int, preferred_client_id: str | None) -> None:
        client_id = self._auto_start_client(preferred_client_id)
        if (
            epoch != self._runner_epoch()
            or not self._runner_active()
            or self._auto_started_epoch == epoch
            or bool(getattr(self.args, "debug", False))
            or client_id is None
        ):
            return
        try:
            self.runner.submit(PlaybackCommand(uuid.uuid4().hex, client_id, "play", {}, None))
        except queue.Full:
            return
        self._auto_started_epoch = epoch
        self._cancel_auto_start_task()

    async def _auto_start_after_grace(
        self,
        epoch: int,
        preferred_client_id: str | None,
    ) -> None:
        try:
            await asyncio.sleep(PAIRED_START_GRACE_SECONDS)
            self._start_epoch(epoch, preferred_client_id)
        except asyncio.CancelledError:
            return
        finally:
            if self._auto_start_task is asyncio.current_task():
                self._auto_start_task = None
                self._auto_start_task_epoch = -1

    def _maybe_auto_start(self, client_id: str | None = None) -> None:
        epoch = self._runner_epoch()
        if (
            not self._runner_active()
            or self._auto_started_epoch == epoch
            or bool(getattr(self.args, "debug", False))
        ):
            return
        if not self.paired_windows or self._paired_workspace_ready():
            self._start_epoch(epoch, client_id)
            return
        if (
            self._auto_start_task is not None
            and not self._auto_start_task.done()
            and self._auto_start_task_epoch == epoch
        ):
            return
        self._cancel_auto_start_task()
        self._auto_start_task_epoch = epoch
        self._auto_start_task = asyncio.create_task(self._auto_start_after_grace(epoch, client_id))

    def _announce_session_change(self) -> None:
        epoch = self._runner_epoch()
        self._cancel_auto_start_task()
        for client in self.clients.values():
            client.reset_session(epoch)
            client.offer_reliable(
                {
                    "type": "session_changed",
                    "protocol": PROTOCOL_VERSION,
                    "session_epoch": epoch,
                }
            )
            client.offer_reliable(self.runner.history_payload())
        if self.paired_windows and self.defer_secondary_window and not self._secondary_opened:
            self._secondary_opened = True
            stats_url = self.dashboard_urls()[1]
            print(f"Player stats: {stats_url}", flush=True)
            if not bool(getattr(self.args, "no_open", False)):
                webbrowser.open(stats_url, new=1, autoraise=True)

    async def websocket(self, request: web.Request) -> web.WebSocketResponse:
        if request.headers.get("Origin") != self.origin:
            raise web.HTTPForbidden(text="invalid websocket origin")
        socket = web.WebSocketResponse(
            heartbeat=10.0,
            compress=False,
            max_msg_size=256 * 1024,
            writer_limit=256 * 1024,
        )
        await socket.prepare(request)
        client: WebClient | None = None
        writer: asyncio.Task[None] | None = None
        try:
            try:
                first = await asyncio.wait_for(socket.receive(), timeout=5.0)
            except TimeoutError:
                await socket.close(code=1008, message=b"authentication timeout")
                return socket
            if first.type != WSMsgType.TEXT:
                await socket.close(code=1008, message=b"hello required")
                return socket
            try:
                hello = json.loads(first.data)
            except json.JSONDecodeError:
                await socket.close(code=1008, message=b"invalid hello")
                return socket
            if hello.get("type") != "hello" or not secrets.compare_digest(
                str(hello.get("token") or ""), self.token
            ):
                await socket.close(code=1008, message=b"authentication failed")
                return socket
            subscriptions = {
                str(value)
                for value in hello.get("subscriptions", ("telemetry",))
                if str(value) in {"telemetry", "game", "observation"}
            }
            client_id = uuid.uuid4().hex
            workspace_id = str(hello.get("workspace_id") or client_id)[:128]
            if not workspace_id or any(
                character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
                for character in workspace_id
            ):
                workspace_id = client_id
            window_id = str(hello.get("window_id") or "main")[:128]
            client = WebClient(
                client_id,
                socket,
                subscriptions,
                workspace_id,
                window_id,
            )
            self.clients[client_id] = client
            self.ever_connected = True
            if self.control_holder is None:
                self.control_holder = workspace_id
                self.control_epoch += 1
            client.offer_reliable(
                {
                    "type": "welcome",
                    "protocol": PROTOCOL_VERSION,
                    "client_id": client_id,
                    "workspace_id": workspace_id,
                    "window_id": window_id,
                    "history_limit": HISTORY_LIMIT,
                }
            )
            client.offer_reliable(self.runner.history_payload())
            episode_start_payload = getattr(self.runner, "episode_start_payload", None)
            if callable(episode_start_payload):
                episode_start_snapshot, episode_start_frames = episode_start_payload()
                if episode_start_snapshot:
                    client.offer_reliable(self._snapshot_for(client, episode_start_snapshot))
                for frame_kind, (_sequence, packet) in episode_start_frames.items():
                    subscription = "game" if frame_kind == FRAME_GAME else "observation"
                    if subscription in client.subscriptions:
                        client.offer_reliable(packet)
            client.offer_snapshot(self._snapshot_for(client, self.runner.snapshot()))
            for frame_kind, (sequence, packet) in self.runner.encoder.latest().items():
                subscription = "game" if frame_kind == FRAME_GAME else "observation"
                if subscription in client.subscriptions:
                    client.offer_frame(frame_kind, sequence, packet)
            writer = asyncio.create_task(client.write())
            self._broadcast_control()
            self._maybe_auto_start(client_id)
            async for message in socket:
                if message.type == WSMsgType.ERROR:
                    break
                if message.type != WSMsgType.TEXT:
                    continue
                try:
                    payload = json.loads(message.data)
                except json.JSONDecodeError:
                    client.offer_reliable({"type": "error", "error": "invalid JSON message"})
                    continue
                kind = str(payload.get("type") or "")
                if kind == "acquire_control":
                    if self.control_holder != client.workspace_id:
                        self.control_holder = client.workspace_id
                        self.input_holder = None
                        self.control_epoch += 1
                        self.runner.clear_input()
                        self._broadcast_control()
                elif kind == "subscribe":
                    client.subscriptions = {
                        str(value)
                        for value in payload.get("subscriptions", ())
                        if str(value) in {"telemetry", "game", "observation"}
                    }
                    for frame_kind, (sequence, packet) in self.runner.encoder.latest().items():
                        subscription = "game" if frame_kind == FRAME_GAME else "observation"
                        if subscription in client.subscriptions:
                            client.offer_frame(frame_kind, sequence, packet)
                elif kind == "history":
                    client.offer_reliable(self.runner.history_payload())
                elif kind == "inspection_frames":
                    try:
                        epoch = int(payload.get("session_epoch", -1))
                        sequence = int(payload.get("sequence", -1))
                        requested_kinds = {int(value) for value in payload.get("kinds", ())} & {
                            FRAME_GAME,
                            FRAME_OBSERVATION,
                        }
                    except TypeError, ValueError:
                        client.offer_reliable(
                            {"type": "error", "error": "invalid inspection frame request"}
                        )
                        continue
                    if epoch != self._runner_epoch() or sequence < 0 or not requested_kinds:
                        continue
                    retained = await asyncio.to_thread(
                        self.runner.encoder.retained,
                        sequence,
                        epoch=epoch,
                        timeout=INSPECTION_FRAME_WAIT_SECONDS,
                    )
                    for frame_kind in requested_kinds:
                        packet = retained.get(frame_kind)
                        if packet is not None:
                            client.offer_reliable(packet[1])
                elif kind == "input":
                    if self.control_holder != client.workspace_id:
                        client.offer_reliable({"type": "error", "error": "control lease required"})
                    else:
                        labels = payload.get("pressed", ())
                        focused = bool(payload.get("focused", False))
                        if focused:
                            if self.input_holder != client_id:
                                self.runner.clear_input()
                            self.input_holder = client_id
                            self.runner.update_input(
                                labels if isinstance(labels, list) else (),
                                focused=True,
                            )
                        elif self.input_holder == client_id:
                            self.input_holder = None
                            self.runner.update_input((), focused=False)
                elif kind == "command":
                    if self.control_holder != client.workspace_id:
                        client.offer_reliable(
                            {
                                "type": "command_result",
                                "id": str(payload.get("id") or ""),
                                "ok": False,
                                "error": "control lease required",
                            }
                        )
                        continue
                    command_name = str(payload.get("name") or "")
                    try:
                        self.runner.submit(
                            PlaybackCommand(
                                str(payload.get("id") or uuid.uuid4().hex),
                                client_id,
                                command_name,
                                payload.get("payload")
                                if isinstance(payload.get("payload"), Mapping)
                                else {},
                                int(payload["expected_revision"])
                                if payload.get("expected_revision") is not None
                                else None,
                            )
                        )
                        if command_name == "play":
                            self._auto_started_epoch = self._runner_epoch()
                            self._cancel_auto_start_task()
                    except queue.Full:
                        client.offer_reliable(
                            {
                                "type": "command_result",
                                "id": str(payload.get("id") or ""),
                                "ok": False,
                                "error": "command queue is full",
                            }
                        )
        finally:
            if client is not None:
                client.closed = True
                client.event.set()
                self.clients.pop(client.client_id, None)
                if self.input_holder == client.client_id:
                    self.input_holder = None
                    self.runner.clear_input()
                controlling_workspace_closed = (
                    self.control_holder == client.workspace_id
                    and not any(
                        candidate.workspace_id == client.workspace_id
                        for candidate in self.clients.values()
                    )
                )
                if controlling_workspace_closed:
                    self.control_holder = None
                    self.control_epoch += 1
                    self.runner.clear_input()
                    try:
                        self.runner.submit(
                            PlaybackCommand(
                                uuid.uuid4().hex,
                                client.client_id,
                                "pause",
                                {},
                                None,
                            )
                        )
                    except queue.Full:
                        pass
                self.last_client_at = time.monotonic()
                self._broadcast_control()
            if writer is not None:
                writer.cancel()
                await asyncio.gather(writer, return_exceptions=True)
        return socket

    async def pump(self) -> None:
        latest_snapshot_key = (-1, -1, -1)
        latest_frames: dict[int, tuple[int, bytes]] = {}
        while not self.stop_event.is_set():
            session_change = int(getattr(self.runner, "session_change", 0))
            if session_change != self._observed_session_change:
                self._observed_session_change = session_change
                latest_snapshot_key = (-1, -1, -1)
                latest_frames.clear()
                self._announce_session_change()
                if self.clients:
                    self._maybe_auto_start(next(iter(self.clients)))
            snapshot = self.runner.snapshot()
            key = (
                int(snapshot.get("session_epoch", 0)),
                int(snapshot.get("revision", 0)),
                int(snapshot.get("sequence", 0)),
            )
            if key != latest_snapshot_key:
                latest_snapshot_key = key
                for client in tuple(self.clients.values()):
                    if "telemetry" in client.subscriptions:
                        if (
                            bool((snapshot.get("transition") or {}).get("boundary"))
                            and (snapshot.get("session") or {}).get("value_discount") is not None
                        ):
                            client.offer_reliable(self.runner.history_payload())
                        client.offer_snapshot(self._snapshot_for(client, snapshot))
            for kind, (sequence, packet) in self.runner.encoder.latest().items():
                if (sequence, packet) == latest_frames.get(kind):
                    continue
                latest_frames[kind] = (sequence, packet)
                subscription = "game" if kind == FRAME_GAME else "observation"
                for client in tuple(self.clients.values()):
                    if subscription in client.subscriptions:
                        client.offer_frame(kind, sequence, packet)
            while True:
                poll_response = getattr(self.runner, "poll_response", None)
                if callable(poll_response):
                    response = poll_response()
                    if response is None:
                        break
                else:
                    try:
                        response = self.runner.responses.get_nowait()
                    except queue.Empty:
                        break
                if response is None:
                    break
                client = self.clients.get(response.client_id)
                if client is not None:
                    client.offer_reliable(response.payload)
            for client_id, client in tuple(self.clients.items()):
                if client.closed:
                    await client.socket.close(code=1013, message=b"client is too slow")
                    self.clients.pop(client_id, None)
            if self.runner.stopped:
                self.stop_event.set()
                break
            if (
                self.ever_connected
                and not self.clients
                and time.monotonic() - self.last_client_at >= LAST_CLIENT_GRACE_SECONDS
            ):
                self.runner.stop()
                self.stop_event.set()
                break
            await asyncio.sleep(1.0 / 120.0)

    async def run(self) -> int:
        await self._prepare_initial_catalog()
        app = web.Application(middlewares=[self.security_headers])
        app.add_routes(
            [
                web.get("/", self.page),
                web.get("/environments/{project_id}", self.page),
                web.get("/environments/{project_id}/goals/{goal_id}", self.page),
                web.get(
                    ("/environments/{project_id}/goals/{goal_id}/variants/{goal_variant_id}"),
                    self.page,
                ),
                web.get(
                    (
                        "/environments/{project_id}/goals/{goal_id}"
                        "/variants/{goal_variant_id}/runs/{run_id}"
                    ),
                    self.page,
                ),
                web.get(
                    (
                        "/environments/{project_id}/goals/{goal_id}"
                        "/variants/{goal_variant_id}/runs/{run_id}"
                        "/checkpoints/{checkpoint_id}"
                    ),
                    self.page,
                ),
                web.get("/panel/{panel}", self.page),
                web.get("/workspace/{window}", self.page),
                web.get("/sources/{path:.*}", self.page),
                web.get("/assets/{path:.*}", self.asset),
                web.get("/api/catalog/environments", self.catalog_environments),
                web.get(
                    "/api/catalog/environments/{entity}/{project}/goals",
                    self.catalog_goals,
                ),
                web.get(
                    ("/api/catalog/environments/{entity}/{project}/goals/{goal_id}/inspection"),
                    self.inspect_goal,
                ),
                web.get(
                    ("/api/catalog/environments/{entity}/{project}/goals/{goal_id}/recipes"),
                    self.catalog_recipes,
                ),
                web.get(
                    (
                        "/api/catalog/environments/{entity}/{project}/goals/{goal_id}"
                        "/recipes/{recipe_id}/inspection"
                    ),
                    self.inspect_recipe,
                ),
                web.get(
                    ("/api/catalog/environments/{entity}/{project}/goals/{goal_id}/variants"),
                    self.catalog_goal_variants,
                ),
                web.get(
                    (
                        "/api/catalog/environments/{entity}/{project}/goals/{goal_id}"
                        "/variants/{goal_variant_id}/inspection"
                    ),
                    self.inspect_goal_variant,
                ),
                web.get(
                    (
                        "/api/catalog/environments/{entity}/{project}/goals/{goal_id}"
                        "/variants/{goal_variant_id}/runs"
                    ),
                    self.catalog_runs,
                ),
                web.get(
                    "/api/catalog/runs/{run_id}/checkpoints",
                    self.catalog_checkpoints,
                ),
                web.get(
                    "/api/catalog/runs/{run_id}/inspection",
                    self.inspect_run,
                ),
                web.post(
                    "/api/catalog/runs/{run_id}/evaluations",
                    self.catalog_evaluate_checkpoints,
                ),
                web.get("/api/playback/inspection", self.inspect_active_playback),
                web.get("/ws", self.websocket),
            ]
        )
        app_runner = web.AppRunner(app, access_log=None)
        await app_runner.setup()
        site = web.TCPSite(app_runner, "127.0.0.1", int(self.args.port))
        await site.start()
        sockets = tuple(site._server.sockets) if site._server is not None else ()
        if not sockets:
            raise RuntimeError("player web server did not bind a socket")
        port = int(sockets[0].getsockname()[1])
        self.origin = f"http://127.0.0.1:{port}"
        urls = self.dashboard_urls()
        dashboard_label = str(getattr(self.args, "dashboard_label", "Player dashboard"))
        print(f"{dashboard_label}: {urls[0]}", flush=True)
        if self.paired_windows and not self.defer_secondary_window:
            print(f"Player stats: {urls[1]}", flush=True)
        self.runner.start()
        pump = asyncio.create_task(self.pump())
        if not bool(getattr(self.args, "no_open", False)):
            launch_urls = urls[:1] if self.defer_secondary_window else urls
            for url in launch_urls:
                webbrowser.open(url, new=1, autoraise=True)
        try:
            await self.stop_event.wait()
        finally:
            auto_start_task = self._auto_start_task
            self._cancel_auto_start_task()
            if auto_start_task is not None:
                await asyncio.gather(auto_start_task, return_exceptions=True)
            pump.cancel()
            await asyncio.gather(pump, return_exceptions=True)
            for client in tuple(self.clients.values()):
                await client.socket.close(code=1001, message=b"player shutting down")
            self.runner.stop()
            await app_runner.cleanup()
        return 0


def run_web_playback(
    session: _PlaybackSession,
    args: argparse.Namespace,
    *,
    config_text: str,
) -> int:
    runner = WebPlaybackRunner(session, args, config_text=config_text)
    server = PlaybackWebServer(runner, args, paired_windows=True)
    try:
        return asyncio.run(server.run())
    except KeyboardInterrupt:
        runner.stop()
        return 130


def run_web_player_application(
    host: Any,
    args: argparse.Namespace,
    *,
    catalog: Any,
) -> int:
    server = PlaybackWebServer(
        host,
        args,
        paired_windows=True,
        catalog=catalog,
        defer_secondary_window=True,
    )
    try:
        return asyncio.run(server.run())
    except KeyboardInterrupt:
        host.stop()
        return 130


def run_web_dataset_playback(
    frames: Iterable[np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    *,
    fps: float,
    action_contract: Mapping[str, Any] | None = None,
) -> int:
    runner = DatasetPlaybackRunner(
        frames,
        rows,
        args,
        fps=fps,
        action_contract=action_contract,
    )
    args.dashboard_label = "Dataset dashboard"
    server = PlaybackWebServer(runner, args)
    try:
        return asyncio.run(server.run())
    except KeyboardInterrupt:
        runner.stop()
        return 130


class WebHumanController:
    """Synchronous human controller backed by a loopback web dashboard."""

    def __init__(self, session: Any, args: argparse.Namespace) -> None:
        self.runner = HumanRecordingRunner(session, args)
        args.dashboard_label = "Recording dashboard"
        self.server = PlaybackWebServer(self.runner, args)
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._serve,
            name="gradlab-recording-dashboard",
            daemon=True,
        )
        self._thread.start()
        deadline = time.monotonic() + 10.0
        while not self.server.origin and self._thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        if self._error is not None:
            raise RuntimeError("recording dashboard failed to start") from self._error
        if not self.server.origin:
            self.close()
            raise RuntimeError("recording dashboard did not start within 10 seconds")

    def _serve(self) -> None:
        try:
            asyncio.run(self.server.run())
        except BaseException as exc:
            self._error = exc
            self.runner.stop()

    def action(self, frame: np.ndarray) -> tuple[Any | None, bool]:
        if self._error is not None:
            raise RuntimeError("recording dashboard stopped unexpectedly") from self._error
        return self.runner.action(frame)

    def observe_transition(
        self,
        *,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
        next_frame: np.ndarray,
    ) -> None:
        self.runner.observe_transition(
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            next_frame=next_frame,
        )

    def close(self) -> None:
        self.runner.stop()
        if self._thread.is_alive():
            self._thread.join(timeout=10.0)


__all__ = [
    "FRAME_CODEC_PNG",
    "FRAME_GAME",
    "FRAME_HEADER",
    "FRAME_MAGIC",
    "FRAME_OBSERVATION",
    "DatasetPlaybackRunner",
    "HumanRecordingRunner",
    "PlaybackCommand",
    "PlaybackWebServer",
    "PROTOCOL_VERSION",
    "WebPlaybackRunner",
    "WebHumanController",
    "history_point",
    "history_point_payload",
    "run_web_dataset_playback",
    "run_web_player_application",
    "run_web_playback",
    "source_browser_path",
    "transition_payload",
]
