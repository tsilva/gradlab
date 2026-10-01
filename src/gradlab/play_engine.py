"""Playback execution, commands, snapshots, and frame encoding without a web server.

Policy, dataset, trajectory, and human-recording adapters share one runner protocol.
The HTTP/WebSocket host and worker process consume that protocol.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import queue
import shutil
import struct
import tempfile
import threading
import time
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from gradlab.action_contract import action_contract_payload
from gradlab.play_capture import EpisodeCaptureManager
from gradlab.play_cnn import CNN_TOP_K_DEFAULT
from gradlab.play_debug import ANSI_PATTERN, PolicyDecision, model_input_lines
from gradlab.play_processing import (
    PLAYER_PROCESSING_FEATURES,
    normalize_player_processing,
)
from gradlab.play_session import (
    _PlaybackSession,
    _PlaybackTransition,
    render_attribution_stack,
    render_obs_stack,
)
from gradlab.play_stop_conditions import PlaybackStopController, default_stop_expression
from gradlab.play_trajectory import EpisodeRecording, freeze_recording, portable_metadata
from gradlab.reward_transform import reward_transform_from_reward
from gradlab.seeds import validate_playback_seed

PROTOCOL_VERSION = 9
HISTORY_LIMIT = 4096
COMMAND_QUEUE_LIMIT = 64
FRAME_ENCODER_QUEUE_LIMIT = 64
INSPECTION_FRAME_WAIT_SECONDS = 2.0
FRAME_HEADER = struct.Struct(">4sBBHQQQ")
FRAME_MAGIC = b"RLP3"
FRAME_CODEC_PNG = 1
FRAME_GAME = 1
FRAME_OBSERVATION = 2
FRAME_ATTRIBUTION = 3
FRAME_CNN_INSPECTION = 4
MAX_JSON_DEPTH = 5
MAX_JSON_ITEMS = 128
MAX_JSON_TEXT = 4096
INPUT_HEARTBEAT_SECONDS = 0.25


def unavailable_attribution(reason: str) -> dict[str, Any]:
    return {
        "mode": "none",
        "interval": 1,
        "status": "off",
        "error": None,
        "generation": 0,
        "last_computed_sequence": None,
        "unavailable_reason": str(reason),
    }


def unavailable_cnn_inspection(reason: str) -> dict[str, Any]:
    return {
        "enabled": False,
        "layer_id": None,
        "interval": 1,
        "top_k": CNN_TOP_K_DEFAULT,
        "status": "off",
        "error": None,
        "generation": 0,
        "last_computed_sequence": None,
        "unavailable_reason": str(reason),
    }


REWARD_COMPONENT_INFO_KEYS = {
    "native_reward_component": "native_reward",
    "cell_novelty_reward_component": "cell_novelty_reward",
    "event_reward_component": "event_reward",
    "progress_reward_component": "progress_reward",
    "score_reward_component": "score_reward",
    "completion_reward_component": "completion_reward",
    "death_penalty_component": "death_penalty",
    "time_penalty_component": "time_penalty",
    "kill_reward_component": "kill_reward",
    "hit_reward_component": "hit_reward",
    "damage_reward_component": "damage_reward",
    "health_reward_component": "health_reward",
    "armor_reward_component": "armor_reward",
    "weapon_reward_component": "weapon_reward",
    "ammo_reward_component": "ammo_reward",
    "weapon_hold_reward_component": "weapon_hold_reward",
}


def unavailable_reward_accounting(reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "reason": str(reason),
        "reward_scale": None,
        "clip_bounds": None,
    }


def reward_accounting_contract(config: Any) -> dict[str, Any]:
    task = config.get("task", {}) if isinstance(config, Mapping) else getattr(config, "task", {})
    reward = task.get("reward", {}) if isinstance(task, Mapping) else {}
    if not isinstance(reward, Mapping):
        return unavailable_reward_accounting("the materialized task reward contract is invalid")
    try:
        transform = reward_transform_from_reward(reward)
    except ValueError as exc:
        return unavailable_reward_accounting(str(exc))
    return {
        "status": "available",
        "reason": None,
        "reward_scale": transform.scale,
        "clip_bounds": (list(transform.clip_bounds) if transform.clip_bounds is not None else None),
    }


def _finite_scalar(value: Any) -> float | None:
    try:
        array = np.asarray(value)
        if array.size != 1:
            return None
        numeric = float(array.reshape(-1)[0])
    except TypeError, ValueError, OverflowError:
        return None
    return numeric if np.isfinite(numeric) else None


def _reward_accounting_payload(
    *,
    final_reward: Any,
    task_metrics: Mapping[str, Any],
    accounting: Mapping[str, Any],
) -> tuple[float | None, dict[str, float], str | None]:
    if accounting.get("status") != "available":
        return None, {}, None
    errors: list[str] = []
    components: dict[str, float] = {}
    for info_key, component_id in REWARD_COMPONENT_INFO_KEYS.items():
        if info_key not in task_metrics:
            continue
        value = _finite_scalar(task_metrics[info_key])
        if value is None:
            errors.append(f"{info_key} is not a finite scalar")
        else:
            components[component_id] = value

    raw_reward: float | None = None
    if "raw_reward" in task_metrics:
        raw_reward = _finite_scalar(task_metrics["raw_reward"])
        if raw_reward is None:
            errors.append("raw_reward is not a finite scalar")
    elif accounting.get("reward_scale") == 1.0 and accounting.get("clip_bounds") is None:
        raw_reward = _finite_scalar(final_reward)
        if raw_reward is None:
            errors.append("final reward is not a finite scalar")
    else:
        errors.append("raw_reward is missing while reward scaling or clipping is active")
    return raw_reward, components, "; ".join(errors) or None


def idle_playback_snapshot(
    *,
    revision: int = 0,
    status_message: str | None = None,
) -> dict[str, Any]:
    return {
        "type": "snapshot",
        "protocol": PROTOCOL_VERSION,
        "revision": int(revision),
        "sequence": 0,
        "run_state": "paused",
        "driver": "policy",
        "interactive": False,
        "policy": None,
        "status_message": status_message,
        "session": {
            "episode": 0,
            "step": 0,
            "seed": None,
            "task": None,
            "total_reward": 0.0,
            "max_x_pos": 0,
            "action_names": [],
            "event_names": [],
            "env_id": None,
            "sampling_mode": "stochastic",
            "target_fps": 0.0,
            "episodes_limit": 0,
            "awaiting_next_episode": False,
            "can_start_next_episode": False,
            "history_size": 0,
            "config": "",
            "reward_accounting": unavailable_reward_accounting("no playback environment is active"),
            "attribution": unavailable_attribution("no live policy is active"),
            "cnn": unavailable_cnn_inspection("no live policy is active"),
        },
        "transition": None,
    }


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
        "sampling_temperature": decision.sampling_temperature,
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


def transition_payload(
    transition: _PlaybackTransition,
    *,
    reward_accounting: Mapping[str, Any] | None = None,
    processing: Iterable[object] = PLAYER_PROCESSING_FEATURES,
) -> dict[str, Any]:
    features = normalize_player_processing(processing)
    detailed = "raw" in features
    diagnostics_enabled = bool(
        features & {"events", "game", "raw", "reward-accounting", "rewards", "signals"}
    )
    diagnostics = transition.diagnostics
    provider_reward = transition.reward
    task_reward = transition.reward
    outcome = "continuing"
    task_metrics: Mapping[str, Any] = {}
    event_transitions: Mapping[str, Any] = {}
    boundary_reasons: list[str] = []
    if diagnostics_enabled and diagnostics is not None:
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
    if "reward-accounting" in features or detailed:
        accounting = dict(
            reward_accounting
            or {
                "status": "available",
                "reason": None,
                "reward_scale": 1.0,
                "clip_bounds": None,
            }
        )
        raw_reward, components, accounting_error = _reward_accounting_payload(
            final_reward=task_reward,
            task_metrics=task_metrics,
            accounting=accounting,
        )
    else:
        raw_reward, components, accounting_error = None, {}, None
    if "policy" in features or detailed:
        decision = _decision_payload(transition.decision)
    elif "actions" in features and transition.decision is not None:
        decision = {
            "requested_action_selection_mode": (
                transition.decision.requested_action_selection_mode
            ),
            "action_selection_mode": transition.decision.action_selection_mode,
            "sampling_temperature": transition.decision.sampling_temperature,
            "sampled": transition.decision.sampled,
            "selected_action": transition.decision.selected_discrete_action,
        }
    else:
        decision = None
    rewards_enabled = bool(features & {"critic-calibration", "raw", "reward-accounting", "rewards"})
    events_enabled = bool(features & {"events", "game", "raw"})
    return {
        "sequence": transition.sequence,
        "episode": transition.episode,
        "step": transition.step,
        "seed": transition.seed,
        "start_id": transition.start_id,
        "action_source": transition.action_source,
        "executed_action": (
            _json_value(transition.executed_action) if features & {"actions", "raw"} else None
        ),
        "effective_action": (
            _json_value(getattr(diagnostics, "effective_policy_action", transition.executed_action))
            if diagnostics is not None and features & {"actions", "raw"}
            else None
        ),
        "native_action": (
            _json_value(getattr(diagnostics, "native_action", transition.executed_action))
            if diagnostics is not None and features & {"actions", "raw"}
            else None
        ),
        "action_override_rule_id": (
            getattr(diagnostics, "action_override_rule_id", None)
            if diagnostics is not None and features & {"actions", "raw"}
            else None
        ),
        "decision": decision,
        "before": {
            "task": _json_value(transition.pre_task) if detailed else None,
            "model_input": (
                model_input_lines(transition.model_obs)
                if features & {"observation", "raw"} and transition.model_obs is not None
                else []
            ),
            "game_frame": "game" in features and transition.before_frame is not None,
            "observation_frames": (
                len(transition.before_frames) if "observation" in features else 0
            ),
        },
        "after": {
            "task": _json_value(transition.next_task) if detailed else None,
            "game_frame": "game" in features and transition.after_frame is not None,
            "observation_frames": (
                len(transition.after_frames) if "observation" in features else 0
            ),
            "frame_role": transition.after_frame_role,
        },
        "reward": {
            "provider": provider_reward if rewards_enabled else None,
            "shaped": task_reward if rewards_enabled else None,
            "step": transition.reward if rewards_enabled else None,
            "return": transition.total_reward if rewards_enabled else None,
            "raw": raw_reward,
            "components": components,
            "accounting_error": accounting_error,
        },
        "events": list(transition.events) if events_enabled else [],
        "event_transitions": _json_value(event_transitions) if events_enabled else {},
        "signals": _numeric_signals(transition.info) if "signals" in features or detailed else {},
        "info": _json_value(transition.info) if detailed else {},
        "max_x_pos": transition.max_x_pos,
        "terminated": transition.terminated,
        "truncated": transition.truncated,
        "completed": transition.completed,
        "boundary": transition.boundary,
        "boundary_reasons": boundary_reasons if events_enabled else [],
        "outcome": outcome if events_enabled else "continuing",
        "return_bootstrap": {
            "source": (
                "terminal_state_value" if transition.return_bootstrap_value is not None else None
            ),
            "value": transition.return_bootstrap_value,
            "reason": transition.return_bootstrap_reason,
        },
        "attribution": {
            "status": transition.attribution_status,
            "mode": transition.attribution_mode,
            "generation": transition.attribution_generation,
            "reason": transition.attribution_reason,
        },
        "cnn": {
            "status": transition.cnn_status,
            "layer_id": transition.cnn_layer_id,
            "generation": transition.cnn_generation,
            "reason": transition.cnn_reason,
            "inspection": (
                None if transition.cnn_inspection is None else transition.cnn_inspection.payload()
            ),
        },
    }


def history_point(
    transition: _PlaybackTransition,
    *,
    reward_accounting: Mapping[str, Any] | None = None,
    processing: Iterable[object] = PLAYER_PROCESSING_FEATURES,
) -> dict[str, Any]:
    return history_point_payload(
        transition_payload(
            transition,
            reward_accounting=reward_accounting,
            processing=processing,
        )
    )


def history_point_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    decision = payload["decision"] or {}
    reward = payload["reward"]
    return {
        "sequence": payload["sequence"],
        "episode": payload["episode"],
        "step": payload["step"],
        "policy_action": decision.get("selected_action"),
        "executed_action": payload.get("executed_action"),
        "effective_action": payload.get("effective_action"),
        "native_action": payload.get("native_action"),
        "action_override_rule_id": payload.get("action_override_rule_id"),
        "action_source": payload.get("action_source"),
        "policy_sampled": decision.get("sampled"),
        "reward_provider": reward["provider"],
        "reward_shaped": reward["shaped"],
        "reward_raw": reward.get("raw"),
        "reward_accounting_error": reward.get("accounting_error"),
        "return": reward["return"],
        "value": decision.get("value"),
        "entropy": decision.get("entropy"),
        "log_probability": decision.get("log_probability"),
        "action_selection_mode": decision.get("action_selection_mode"),
        "outcome": payload.get("outcome"),
        "events": payload["events"],
        "boundary": bool(payload.get("boundary")),
        "terminated": bool(payload.get("terminated")),
        "truncated": bool(payload.get("truncated")),
        "boundary_reasons": list(payload.get("boundary_reasons") or []),
        "return_bootstrap": dict(payload.get("return_bootstrap") or {}),
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
    boundary_point = episode_points[-1] if episode_points else {}
    bootstrapped = bool(boundary_point.get("truncated"))
    bootstrap_value = 0.0
    if bootstrapped:
        bootstrap = boundary_point.get("return_bootstrap")
        bootstrap = bootstrap if isinstance(bootstrap, Mapping) else {}
        candidate = bootstrap.get("value")
        if (
            bootstrap.get("source") != "terminal_state_value"
            or isinstance(candidate, bool)
            or not isinstance(candidate, int | float | np.number)
            or not np.isfinite(float(candidate))
        ):
            reasons.append(
                str(bootstrap.get("reason") or "terminal-state critic bootstrap is unavailable")
            )
        else:
            bootstrap_value = float(candidate)
    if reasons:
        for point in episode_points:
            point["value_comparison_reasons"] = reasons
        return
    realized_return = bootstrap_value
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
        point["realized_return_bootstrapped"] = bootstrapped
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
    generation: int = 0,
) -> bytes:
    output = io.BytesIO()
    image_mode = "RGBA" if np.asarray(frame).shape[-1] == 4 else "RGB"
    Image.fromarray(frame, mode=image_mode).save(
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
            generation,
        )
        + output.getvalue()
    )


class FrameEncoder:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._pending: deque[tuple[int, int, dict[int, np.ndarray], dict[int, int]]] = deque()
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

    def submit(
        self,
        kind: int,
        sequence: int,
        frame: np.ndarray | None,
        *,
        generation: int = 0,
    ) -> None:
        self.submit_batch(sequence, {kind: frame}, generations={kind: generation})

    def submit_batch(
        self,
        sequence: int,
        frames: Mapping[int, np.ndarray | None],
        *,
        generations: Mapping[int, int] | None = None,
    ) -> None:
        owned = {
            int(kind): np.asarray(frame, dtype=np.uint8).copy()
            for kind, frame in frames.items()
            if frame is not None
        }
        if not owned:
            return
        owned_generations = {kind: int((generations or {}).get(kind, 0)) for kind in owned}
        with self._condition:
            while len(self._pending) >= FRAME_ENCODER_QUEUE_LIMIT and not self._closed:
                self._condition.wait()
            if not self._closed:
                self._pending.append((self._epoch, int(sequence), owned, owned_generations))
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
        kinds: Iterable[int] | None = None,
    ) -> dict[int, tuple[int, bytes]]:
        with self._condition:
            key = (self._epoch if epoch is None else int(epoch), int(sequence))
            requested = {int(kind) for kind in kinds or ()}
            deadline = time.monotonic() + max(0.0, float(timeout))
            while (
                key not in self._retained or not requested.issubset(self._retained.get(key, {}))
            ) and not self._closed:
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
            epoch, sequence, frames, generations = pending
            encoded: dict[int, tuple[int, bytes]] = {}
            for kind, frame in frames.items():
                packet = _frame_packet(
                    kind,
                    sequence,
                    frame,
                    session_epoch=epoch,
                    generation=generations[kind],
                )
                encoded[kind] = (sequence, packet)
                with self._condition:
                    self._latest[kind] = (sequence, packet)
            with self._condition:
                self._retained.setdefault((epoch, sequence), {}).update(encoded)
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
        self._snapshot_updates: deque[dict[str, Any]] = deque(maxlen=HISTORY_LIMIT)
        self._stop = threading.Event()
        self.revision = 0
        self.processing_features = PLAYER_PROCESSING_FEATURES
        if thread_name is not None:
            self.commands: queue.Queue[PlaybackCommand] = queue.Queue(COMMAND_QUEUE_LIMIT)
            self._episode_start_snapshot: dict[str, Any] = {}
            self._episode_start_frames: dict[int, tuple[int, bytes]] = {}
            self._thread = threading.Thread(target=self._run, name=thread_name)

    @property
    def effective_fps(self) -> float:
        return self.target_fps if getattr(self, "rgb_enabled", True) else 0.0

    def set_processing(self, features: Iterable[object]) -> None:
        self.processing_features = normalize_player_processing(features)
        session = getattr(self, "session", None)
        configure = getattr(session, "set_processing", None)
        if callable(configure):
            configure(self.processing_features)

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
        capture = getattr(self, "capture", None)
        if capture is not None:
            capture.abort("playback session stopped before episode completion")
        close_recording = getattr(self, "close_recording", None)
        if close_recording is not None:
            close_recording()
        self.encoder.close()

    def submit(self, command: PlaybackCommand) -> None:
        self.commands.put_nowait(command)

    def snapshot(self) -> dict[str, Any]:
        with self._snapshot_lock:
            return dict(self._latest_snapshot)

    def drain_snapshot_updates(self) -> list[dict[str, Any]]:
        with self._snapshot_lock:
            updates = list(self._snapshot_updates)
            self._snapshot_updates.clear()
            return updates

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
        capture_context: Mapping[str, Any] | None = None,
        trajectory_bundle: Any | None = None,
        recording_options: Mapping[str, Any] | None = None,
    ) -> None:
        self._init_protocol(thread_name="gradlab-playback-runtime")
        self.session = session
        self.args = args
        self.config_text = ANSI_PATTERN.sub("", config_text)
        from gradlab.play_action_summary import EpisodeActionSummary
        from gradlab.play_reward_summary import EpisodeRewardSummary

        self.episode_rewards = EpisodeRewardSummary()
        self.episode_actions = EpisodeActionSummary(_session_action_contract_payload(session))
        self.reward_accounting = reward_accounting_contract(session.config)
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
        capability_value = getattr(session, "attribution_capability", None)
        self.attribution_capability = (
            dict(capability_value)
            if isinstance(capability_value, Mapping)
            else {
                "target": "selected_action_log_probability",
                "supported_modes": [],
                "unavailable_reason": "the active playback source does not expose attribution",
            }
        )
        cnn_capability_value = getattr(session, "cnn_capability", None)
        self.cnn_capability = (
            dict(cnn_capability_value)
            if isinstance(cnn_capability_value, Mapping)
            else {
                "layers": [],
                "default_layer_id": None,
                "rank_basis": "peak_raw_positive_response",
                "unavailable_reason": ("the active playback source does not expose CNN inspection"),
            }
        )
        action_selection = self.policy_capabilities.get("action_selection")
        action_selection = dict(action_selection) if isinstance(action_selection, Mapping) else {}
        self.supported_action_selection_modes = tuple(
            str(mode) for mode in action_selection.get("supported_modes", ()) if str(mode)
        )
        if not self.supported_action_selection_modes:
            self.supported_action_selection_modes = ("stochastic", "deterministic")
        self.supports_sampling_temperature = bool(
            getattr(
                getattr(session, "policy_runtime", None),
                "supports_sampling_temperature",
                "stochastic" in self.supported_action_selection_modes,
            )
        )
        self.sampling_temperature = 1.0
        self.temperature_changed = False
        default_mode = str(action_selection.get("default_mode") or "")
        self.sampling_mode = default_mode or (
            self.supported_action_selection_modes[0]
            if self.supported_action_selection_modes
            else "stochastic"
        )
        self.target_fps = max(0.0, float(args.fps))
        self._next_presentation_at = 0.0
        self.remaining_steps = 0
        self.continue_target: str | None = None
        self.continue_count = 0
        self.boundaries = 0
        self.awaiting_next_episode = False
        termination_conditions = list(getattr(session, "termination_conditions", ()))
        event_names = {
            str(condition["event"])
            for condition in termination_conditions
            if isinstance(condition, Mapping) and condition.get("event")
        }
        try:
            event_names.update(str(name) for name in session.env.runtime.kernel.event_names)
        except AttributeError:
            pass
        signal_names = {str(name) for name in getattr(session, "info_vars", ()) if str(name)}
        self.stop_conditions = PlaybackStopController(
            event_names=event_names,
            signal_names=signal_names,
            source=default_stop_expression(termination_conditions),
        )
        self._input_lock = threading.Lock()
        self._pressed: tuple[str, ...] = ()
        self._input_updated_at = 0.0
        self._input_focused = False
        self._status_message: str | None = None
        self.environment_id = _session_environment_id(session, args)
        self.value_discount = _value_discount_factor(getattr(session, "model", None))
        self.contract_details = dict(contract_details or {})
        self.value_contract = dict(value_contract) if isinstance(value_contract, Mapping) else None
        resolved_capture_context = dict(capture_context or {})
        resolved_capture_context["expected_sampling_mode"] = self.sampling_mode
        self.capture = EpisodeCaptureManager(resolved_capture_context or None)
        self._trajectory_execution = portable_metadata(
            resolved_capture_context.get("execution", {})
        )
        self._trajectory_lock = threading.RLock()
        # Recording identity/pins need a short lock independent of Policy work.
        # Replacement acquires trajectory first, then diagnostic; reads use only
        # diagnostic and never block behind an in-flight environment decision.
        self._diagnostic_lock = threading.RLock()
        from gradlab.play_diagnostics import DiagnosticReads, LiveRecordingSource

        self.diagnostics = DiagnosticReads(
            LiveRecordingSource(
                self._diagnostic_lock, lambda: self.seek_recording, lambda: self.history
            )
        )
        self._inspection_history: tuple[str, int, int, list[dict[str, Any]]] | None = None
        self.seek_recording: EpisodeRecording | None = None
        self.recording: EpisodeRecording | None = None
        self.recording_enabled = False
        self._recording_options = dict(recording_options or {})
        self._checkpoint_root: Path | None = None
        if trajectory_bundle is not None:
            self._checkpoint_root = Path(tempfile.mkdtemp(prefix="gradlab-trajectory-checkpoint-"))
            for name, path in (
                ("model.zip", trajectory_bundle.checkpoint_path),
                ("model.json", trajectory_bundle.model_path),
                ("recipe.json", trajectory_bundle.recipe_path),
            ):
                shutil.copyfile(path, self._checkpoint_root / name)
            self._begin_recording()

    def _begin_recording(self) -> None:
        with self._trajectory_lock, self._diagnostic_lock:
            previous_seek = self.seek_recording
            previous = self.recording
            self.seek_recording = None
            self.recording = None
            if previous_seek is not None:
                previous_seek.close()
            if previous is not None:
                previous.close()
            self.session.trajectory_seeking = self._checkpoint_root is not None
            self.session.trajectory_recording = self.recording_enabled
            if self._checkpoint_root is None:
                return
            self.seek_recording = EpisodeRecording(self._recording_metadata(), compress=True)
            if self.recording_enabled:
                self._begin_full_recording()

    def _recording_metadata(self) -> dict[str, Any]:
        import yaml

        config = portable_metadata(self.session.config)
        if not isinstance(config, Mapping):
            from omegaconf import OmegaConf

            config = portable_metadata(OmegaConf.to_container(self.session.config, resolve=True))
        snapshot = self._snapshot_payload(None)
        snapshot["session"]["config"] = yaml.safe_dump(config, sort_keys=False)
        reasons = self._critic_comparison_reasons()
        snapshot["session"]["critic_comparison"] = {
            "available": not reasons,
            "reasons": reasons,
            "discount": self.value_discount,
        }
        return {
            "first_step": self.session.step_index + 1,
            "episode": self.session.episode,
            "seed": self.session.active_seed,
            "classification": self._recording_classification(),
            "resolved_environment": config,
            "contract": portable_metadata(self.contract_details),
            "execution": self._trajectory_execution,
            "discount": self.value_discount,
            "discount_semantics": "per policy decision; termination stops return, truncation requires recorded bootstrap",
            "initial_snapshot": portable_metadata(snapshot),
            "initial_observation_status": "captured with first recorded decision",
        }

    def _begin_full_recording(self) -> None:
        previous = self.recording
        self.recording = None
        if previous is not None:
            previous.close()
        self.recording = EpisodeRecording(self._recording_metadata(), **self._recording_options)

    def _recording_classification(self, seed: int | None = None) -> str:
        if (
            self.session.interactive
            or self.temperature_changed
            or self.driver == "human"
            or self.contract_details.get("mode") == "counterfactual"
            or self.sampling_mode
            != (self.policy_capabilities.get("action_selection") or {}).get(
                "default_mode", "stochastic"
            )
            or getattr(self.session, "config", None)
            != getattr(self.session, "termination_base_config", self.session.config)
        ):
            return "counterfactual"
        if self.contract_details.get("mode") == "evaluation":
            return (
                "evaluation_reproduction"
                if self.contract_details.get("playback_seed_source") == "evaluation"
                and (self.session.active_seed if seed is None else seed)
                == self.contract_details.get("playback_seed")
                else "counterfactual"
            )
        return "faithful"

    def recording_status(self) -> dict[str, Any]:
        seek = self.seek_recording.status() if self.seek_recording is not None else {}
        full = self.recording.status() if self.recording is not None else {}
        status = {
            "available": self._checkpoint_root is not None,
            "enabled": self.recording_enabled,
            "activation": "manual",
            "scientific_evidence": False,
            **seek,
            "recorded_transitions": full.get("transitions", 0),
            "recording_first_step": full.get("first_step"),
            "recording_error": full.get("error"),
        }
        if self.recording_enabled and full.get("error"):
            status["error"] = full["error"]
        initial, _ = self.episode_start_payload()
        if (
            status.get("episode_id")
            and initial.get("trajectory", {}).get("episode_id") == status["episode_id"]
        ):
            # The initial state is seekable but is not an action transition.
            status["initial_step"] = initial["session"]["step"]
        return status

    def history_payload(self) -> dict[str, Any]:
        payload = super().history_payload()
        with self._trajectory_lock:
            if self.seek_recording is not None:
                payload["timeline"] = self.seek_recording.event_overview()
        return payload

    def close_recording(self) -> None:
        self.diagnostics.close()
        with self._trajectory_lock, self._diagnostic_lock:
            if self.seek_recording is not None:
                self.seek_recording.close()
                self.seek_recording = None
            if self.recording is not None:
                self.recording.close()
                self.recording = None
            if self._checkpoint_root is not None:
                shutil.rmtree(self._checkpoint_root, ignore_errors=True)

    def inspect_recorded_step(self, episode_id: str, step: int) -> dict[str, Any]:
        """Read owned frames from a pinned prefix without holding up inference."""
        from gradlab.play_diagnostics import RecordedPrefix

        with self._diagnostic_lock:
            recording = self.seek_recording
            if recording is None or recording.metadata["episode_id"] != episode_id:
                raise ValueError("the recorded episode has been replaced")
            status = self.recording_status()
            if type(step) is int and step == status.get("initial_step"):
                snapshot, packets = self.episode_start_payload()
                snapshot["trajectory"] = status
                snapshot["initial_frames"] = list(packets)
                return {
                    "snapshot": snapshot,
                    "points": [],
                    "frames": [
                        {
                            "kind": kind,
                            "generation": 0,
                            "png": base64.b64encode(packet[FRAME_HEADER.size :]).decode("ascii"),
                        }
                        for kind, (_, packet) in packets.items()
                        if kind != FRAME_GAME or getattr(self, "rgb_enabled", True)
                    ],
                }
            if type(step) is not int or not status["first_step"] <= step <= status["last_step"]:
                raise ValueError("step is outside the recorded episode")
            cached = self._inspection_history
            reuse = (
                cached is not None
                and cached[0] == episode_id
                and cached[1] <= max(status["first_step"], step - 63)
                and step <= cached[2]
            )
            first = cached[1] if reuse else max(status["first_step"], ((step - 1) // 64) * 64 - 63)
            last = cached[2] if reuse else min(status["last_step"], first + 127)
            recent = {
                point["sequence"]: dict(point)
                for point in list(self.history)
                if first <= point["step"] <= last
            }
            descriptor = recording.reserve_read()
        try:
            prefix = RecordedPrefix(descriptor)
            row = prefix.transition(step)
            snapshot = row["inspection_snapshot"]
            snapshot["trajectory"] = status
            snapshot["history_point"] = history_point_payload(snapshot["transition"])
            if reuse:
                points = cached[3]
            else:
                previous = (
                    {point["step"]: point for point in cached[3]}
                    if cached is not None and cached[0] == episode_id
                    else {}
                )
                points = []
                for index in range(first, last + 1):
                    point = previous.get(index)
                    if point is None:
                        item = row if index == step else prefix.transition(index)
                        point = history_point_payload(item["inspection_snapshot"]["transition"])
                    points.append(point)
            cache = cached if reuse else (episode_id, first, last, points)
            # Preserve calibration amendments without inventing completed returns.
            points = [dict(recent.get(point["sequence"], point)) for point in points]
            images = {
                FRAME_GAME: row["after_image"] if getattr(self, "rgb_enabled", True) else None,
                FRAME_OBSERVATION: render_obs_stack(row["observation_frames"], 1)
                if row["observation_frames"]
                else None,
                **row.get("inspection_images", {}),
            }
            frames = []
            for kind, frame in images.items():
                if frame is None:
                    continue
                kind = int(kind)
                diagnostic = {FRAME_ATTRIBUTION: "attribution", FRAME_CNN_INSPECTION: "cnn"}.get(
                    kind
                )
                generation = snapshot["transition"][diagnostic]["generation"] if diagnostic else 0
                stream = io.BytesIO()
                Image.fromarray(frame).save(stream, format="PNG", compress_level=1)
                frames.append(
                    {
                        "kind": kind,
                        "generation": generation,
                        "png": base64.b64encode(stream.getvalue()).decode("ascii"),
                    }
                )
            with self._diagnostic_lock:
                if self.seek_recording is not recording:
                    raise ValueError("the recorded episode has been replaced")
                self._inspection_history = cache
            return {"snapshot": snapshot, "points": points, "frames": frames}
        finally:
            recording.release_read()

    def freeze_trajectory(self) -> str:
        with self._trajectory_lock:
            if self.recording is None or self._checkpoint_root is None:
                raise ValueError("start recording before downloading an episode")
            recording = self.recording
            destination = Path(tempfile.mkdtemp(prefix="gradlab-trajectory-download-"))
            try:
                shutil.copytree(
                    self._checkpoint_root, destination / "checkpoint", copy_function=os.link
                )
            except BaseException:
                shutil.rmtree(destination, ignore_errors=True)
                raise
            reservation = recording.reserve_prefix()
        # Wait for the fixed prefix outside the transition lock so live Playback continues.
        return freeze_recording(recording, destination, reservation)

    def _record_transition(
        self,
        transition: _PlaybackTransition,
        *,
        full: dict[str, Any],
    ) -> None:
        if self.seek_recording is None:
            return
        from dataclasses import fields

        inspection_snapshot = self._snapshot_payload(transition, current=full)
        inspection_snapshot["sequence"] = transition.sequence
        inspection_snapshot["session"].update(
            episode=transition.episode,
            step=transition.step,
            total_reward=transition.total_reward,
        )
        inspection_images = {}
        if transition.attribution is not None and transition.before_frames:
            inspection_images[str(FRAME_ATTRIBUTION)] = render_attribution_stack(
                transition.before_frames, transition.attribution
            )
        if transition.cnn_inspection is not None:
            inspection_images[str(FRAME_CNN_INSPECTION)] = transition.cnn_inspection.atlas
        self.seek_recording.append(
            {
                "step": transition.step,
                "boundary": transition.boundary,
                "classification": self._recording_classification(transition.seed),
                "inspection_snapshot": portable_metadata(inspection_snapshot),
                "inspection_images": inspection_images,
                "after_image": transition.after_frame
                if not transition.boundary or transition.after_frame_role == "terminal_observation"
                else None,
                "observation_frames": transition.before_frames,
                "presentation": {"events": full.get("events", [])},
            }
        )
        if not self.recording_enabled or self.recording is None:
            return
        # Archive presentation omits live diagnostics and may mark a terminal
        # image missing. Keep those edits separate from the inspection/live view.
        presentation = {**full, "after": dict(full["after"])}
        reasons = self._critic_comparison_reasons(transition)
        presentation["episode_rewards"] = self.episode_rewards.payload(full)
        presentation["episode_actions"] = self.episode_actions.payload(full)
        presentation["recorded_session"] = {
            "sampling_mode": self.sampling_mode,
            "sampling_temperature": getattr(self, "sampling_temperature", 1.0),
            "temperature_changed": getattr(self, "temperature_changed", False),
            "critic_comparison": {
                "available": not reasons,
                "reasons": reasons,
                "discount": self.value_discount,
            },
        }
        for key in ("attribution", "cnn"):
            presentation[key] = {
                "status": "not-recorded",
                "reason": "not recorded in episode archives",
                "generation": 0,
                "inspection": None,
                "mode": "none",
            }
        terminal_missing = transition.boundary and transition.next_model_obs is None
        image_missing = (
            transition.boundary and transition.after_frame_role != "terminal_observation"
        )
        if image_missing:
            presentation["after"].update(
                game_frame=False, frame_role="terminal_missing", observation_frames=0
            )
        facts = {
            field.name: getattr(transition, field.name)
            for field in fields(transition)
            if field.name
            in {
                "info",
                "pre_task",
                "next_task",
                "return_bootstrap_value",
                "return_bootstrap_reason",
            }
        }
        if transition.diagnostics is not None:
            facts["diagnostics"] = {
                field.name: getattr(transition.diagnostics, field.name)
                for field in fields(transition.diagnostics)
                if field.name not in {"terminal_frame", "outcome"}
            }
        decision = transition.decision
        # Recorded inspection must retain evidence independently of connected panels.
        self.recording.append(
            {
                "inspection_snapshot": portable_metadata(inspection_snapshot),
                "inspection_images": inspection_images,
                "sequence": transition.sequence,
                "step": transition.step,
                "seed": transition.seed,
                "start_id": transition.start_id,
                "action_source": transition.action_source,
                "reward": transition.reward,
                "return": transition.total_reward,
                "terminated": transition.terminated,
                "truncated": transition.truncated,
                "boundary": transition.boundary,
                "classification": self._recording_classification(transition.seed),
                "observation": transition.model_obs,
                "next_observation": transition.next_model_obs,
                "next_observation_status": "terminal_missing"
                if terminal_missing
                else "available"
                if transition.next_model_obs is not None
                else "unavailable",
                "selected_action": None if decision is None else decision.raw_action,
                "executed_action": transition.executed_action,
                "before_image": transition.before_frame,
                "after_image": None if image_missing else transition.after_frame,
                "after_image_status": "terminal_missing"
                if image_missing
                else "available"
                if transition.after_frame is not None
                else "unavailable",
                "observation_frames": transition.before_frames,
                "policy_outputs": None if decision is None else asdict(decision),
                "facts": portable_metadata(facts),
                "presentation": portable_metadata(presentation),
            }
        )

    def _begin_capture(self) -> None:
        if not self.capture.enabled:
            return
        if self.driver != "policy":
            self.capture.abort("human-driven episodes are not publishable")
            return
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
            self.capture.abort("episode termination differs from the faithful playback contract")
            return
        if self.temperature_changed:
            self.capture.abort("sampling temperature differs from faithful playback")
            return
        expected = str((self.capture.context or {}).get("expected_sampling_mode") or "")
        if expected and self.sampling_mode != expected:
            self.capture.abort("action selection differs from the faithful playback contract")
            return
        self.capture.begin(
            self.session.current_frame,
            episode=int(self.session.episode),
            seed=int(self.session.active_seed),
            sampling_mode=self.sampling_mode,
        )

    def start(self) -> None:
        self._begin_capture()
        super().start()

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
            if self.value_contract.get("discount_schedule") is not None:
                reasons.append(
                    "critic was trained with a changing discount; "
                    "fixed-discount calibration is unavailable"
                )
            elif (
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
        if self.temperature_changed:
            reasons.append("sampling temperature changed from the training contract")
        if expected_selection and self.sampling_mode != expected_selection:
            reasons.append(
                "V(s) was trained with stochastic action selection and may be less accurate "
                "in deterministic mode. Return comparisons are disabled because the "
                "action-selection rules differ"
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
        if (
            transition is not None
            and transition.truncated
            and "state_value" in introspection
            and self.value_contract is not None
        ):
            if self.value_contract.get("truncation_bootstrap") != "terminal-value":
                reasons.append(
                    "training value contract does not declare terminal-value truncation bootstrap"
                )
            elif getattr(transition, "return_bootstrap_value", None) is None:
                reasons.append(
                    str(
                        getattr(transition, "return_bootstrap_reason", None)
                        or "terminal-state critic bootstrap is unavailable"
                    )
                )
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

    def _snapshot_payload(
        self,
        transition: _PlaybackTransition | None,
        *,
        current: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if current is None and transition is not None:
            current = transition_payload(
                transition,
                reward_accounting=self.reward_accounting,
                processing=self.processing_features,
            )
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
        comparison_reasons = self._critic_comparison_reasons(transition)
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
                "attribution": dict(self.attribution_capability),
                "cnn": dict(self.cnn_capability),
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
                    "supports_temperature": self.supports_sampling_temperature,
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
            "publication_capture": self.capture.status(),
            "trajectory": self.recording_status(),
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
                "sampling_temperature": getattr(self, "sampling_temperature", 1.0),
                "temperature_changed": getattr(self, "temperature_changed", False),
                "value_discount": self.value_discount,
                "target_fps": self.target_fps,
                "rgb_enabled": getattr(self, "rgb_enabled", True),
                "episodes_limit": int(self.args.episodes),
                "awaiting_next_episode": self.awaiting_next_episode,
                "can_start_next_episode": self._can_start_next_episode(),
                "history_size": len(self.history),
                "config": self.config_text,
                "reward_accounting": dict(self.reward_accounting),
                "attribution": dict(
                    getattr(
                        self.session,
                        "attribution_state",
                        unavailable_attribution(
                            self.attribution_capability.get("unavailable_reason")
                            or "the active playback source does not expose attribution"
                        ),
                    )
                ),
                "cnn": dict(
                    getattr(
                        self.session,
                        "cnn_state",
                        unavailable_cnn_inspection(
                            self.cnn_capability.get("unavailable_reason")
                            or "the active playback source does not expose CNN inspection"
                        ),
                    )
                ),
                "termination_source": getattr(
                    self.session,
                    "termination_source",
                    "training",
                ),
                "termination_conditions": list(getattr(self.session, "termination_conditions", ())),
                "stop_condition": self.stop_conditions.payload(),
                "playback_contract": dict(self.contract_details),
                "critic_comparison": {
                    "available": not comparison_reasons,
                    "reasons": comparison_reasons,
                    "discount": self.value_discount,
                },
            },
            "transition": current,
            "history_point": current_history,
            "episode_rewards": self.episode_rewards.payload(current),
            "episode_actions": self.episode_actions.payload(current),
        }

    def _publish(
        self,
        transition: _PlaybackTransition | None = None,
        *,
        current: dict[str, Any] | None = None,
        paced: bool = False,
    ) -> None:
        if transition is not None and current is None:
            current = transition_payload(
                transition,
                reward_accounting=self.reward_accounting,
                processing=self.processing_features,
            )
        if (
            "history" in self.processing_features
            and transition is not None
            and (not self.history or int(self.history[-1]["sequence"]) != transition.sequence)
        ):
            self.history.append(history_point_payload(current))
            if transition.boundary and "critic-calibration" in self.processing_features:
                annotate_realized_returns(
                    self.history,
                    episode=transition.episode,
                    discount=self.value_discount,
                    comparison_reasons=self._critic_comparison_reasons(transition),
                )
        # Keep every history point, but only prepare live frames at display speed.
        # Commands and terminal transitions bypass pacing so they remain visible.
        now = time.perf_counter()
        if paced and self.effective_fps > 0 and now < self._next_presentation_at:
            return
        if self.effective_fps > 0:
            interval = 1.0 / self.effective_fps
            if paced and self._next_presentation_at > 0:
                # Advance the intended clock, not the completion time of a step.
                # Discard missed display slots without replaying them in a burst.
                missed = int((now - self._next_presentation_at) / interval)
                self._next_presentation_at += (missed + 1) * interval
                if self._next_presentation_at <= now:
                    self._next_presentation_at = now + interval
            else:
                # Commands publish immediately and rebase resume/rate changes.
                self._next_presentation_at = now + interval
        else:
            self._next_presentation_at = 0.0
        if transition is not None:
            game_frame = (
                transition.after_frame
                if "game" in self.processing_features and getattr(self, "rgb_enabled", True)
                else None
            )
            obs_frames = transition.before_frames
            sequence = transition.sequence
        else:
            game_frame = (
                self.session.current_frame
                if "game" in self.processing_features and getattr(self, "rgb_enabled", True)
                else None
            )
            obs_frames = tuple(self.session.frames or ())
            sequence = self.session.sequence
        obs_image = None
        if "observation" in self.processing_features and obs_frames:
            obs_image = render_obs_stack(
                deque(obs_frames, maxlen=len(obs_frames)),
                scale=1,
            )
        attribution_image = None
        attribution_generation = 0
        if (
            "attribution" in self.processing_features
            and transition is not None
            and transition.attribution is not None
            and obs_frames
        ):
            attribution_image = render_attribution_stack(
                obs_frames,
                transition.attribution,
            )
            attribution_generation = transition.attribution_generation
        cnn_image = None
        cnn_generation = 0
        if (
            "cnn-inspection" in self.processing_features
            and transition is not None
            and transition.cnn_inspection is not None
        ):
            cnn_image = transition.cnn_inspection.atlas
            cnn_generation = transition.cnn_generation
        self.encoder.submit_batch(
            sequence,
            {
                FRAME_GAME: game_frame,
                FRAME_OBSERVATION: obs_image,
                FRAME_ATTRIBUTION: attribution_image,
                FRAME_CNN_INSPECTION: cnn_image,
            },
            generations={
                FRAME_ATTRIBUTION: attribution_generation,
                FRAME_CNN_INSPECTION: cnn_generation,
            },
        )
        payload = self._snapshot_payload(transition, current=current)
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
            self._snapshot_updates.append(payload)
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
            raise ValueError("episode complete; press Play to start the next episode")

    def _prepare_next_episode(self) -> None:
        self.session.last_transition = None
        self.clear_input()
        self.awaiting_next_episode = False
        self.remaining_steps = 0
        self.continue_target = None
        if not self.recording_enabled:
            self._begin_recording()

    @staticmethod
    def _stop_match_message(values: Mapping[str, object]) -> str:
        facts = " · ".join(f"{name} = {value:g}" for name, value in values.items())
        return f"stop condition matched · {facts}" if facts else "stop condition matched"

    def _pause_after_transition(self, message: str) -> None:
        self.remaining_steps = 0
        self.continue_target = None
        self.run_state = "paused"
        self._status_message = message

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
        if command.name == "set_attribution":
            try:
                interval_value = command.payload.get("interval")
                interval = None if interval_value in {None, ""} else interval_value
                self.session.configure_attribution(
                    str(command.payload.get("mode") or "none"),
                    interval,
                )
            except Exception as exc:
                self.revision += 1
                self._publish(self.session.last_transition)
                self._response(command, ok=False, error=str(exc))
                return
            self.revision += 1
            self._publish(self.session.last_transition)
            self._response(command, ok=True)
            return
        if command.name == "set_cnn_inspection":
            try:
                self.session.configure_cnn_inspection(
                    enabled=command.payload.get("enabled"),
                    layer_id=command.payload.get("layer_id"),
                    interval=command.payload.get("interval"),
                    top_k=command.payload.get("top_k"),
                )
            except Exception as exc:
                self.revision += 1
                self._publish(self.session.last_transition)
                self._response(command, ok=False, error=str(exc))
                return
            self.revision += 1
            self._publish(self.session.last_transition)
            self._response(command, ok=True)
            return
        try:
            if command.name == "retry_storage":
                if self.seek_recording is not None:
                    self.seek_recording.retry()
                if self.recording is not None:
                    self.recording.retry()
                self._set_state("paused", message="storage retry requested")
            elif command.name == "set_recording":
                if self._checkpoint_root is None:
                    raise ValueError("recording requires an exact resolved Checkpoint")
                enabled = command.payload.get("enabled")
                if not isinstance(enabled, bool):
                    raise ValueError("recording enabled must be boolean")
                if enabled and not self.recording_enabled:
                    self._require_active_episode()
                    self._begin_full_recording()
                    self.recording_enabled = True
                    self.session.trajectory_recording = True
                elif not enabled:
                    self.recording_enabled = False
                    self.session.trajectory_recording = False
                elif self.recording is not None:
                    if self.seek_recording is not None:
                        self.seek_recording.retry()
                    self.recording.retry()
                self._set_state(
                    "paused",
                    message="recording enabled"
                    if enabled
                    else "recording stopped; captured prefix retained",
                )
            elif command.name == "pause":
                self.remaining_steps = 0
                self.continue_target = None
                self.clear_input()
                self._set_state("paused", message="paused at a completed transition")
            elif command.name == "set_sampling_temperature":
                temperature = float(command.payload.get("temperature"))
                if not np.isfinite(temperature) or temperature <= 0:
                    raise ValueError("sampling temperature must be finite and greater than zero")
                if not self.supports_sampling_temperature:
                    raise ValueError("this policy does not support sampling temperature")
                if temperature != self.sampling_temperature:
                    self.sampling_temperature = temperature
                    self.temperature_changed = True
                    self.capture.abort("sampling temperature changed during playback")
                self._set_state(
                    self.run_state, message=f"next sampling temperature · {temperature:g}"
                )
            elif command.name == "set_action_selection_mode":
                mode = str(command.payload.get("mode") or "")
                if mode not in self.supported_action_selection_modes:
                    supported = ", ".join(self.supported_action_selection_modes) or "none"
                    raise ValueError(
                        f"unsupported action-selection mode {mode!r}; supported: {supported}"
                    )
                if mode != self.sampling_mode:
                    self.sampling_mode = mode
                    if self.session.step_index == 0 and not self.awaiting_next_episode:
                        self._begin_capture()
                    else:
                        self.capture.abort(
                            "action selection changed during the active playback session"
                        )
                self._set_state(
                    self.run_state,
                    message=f"next action selection · {mode}",
                )
            elif command.name == "play":
                if not self.stop_conditions.valid:
                    detail = (
                        self.stop_conditions.error.message
                        if self.stop_conditions.error is not None
                        else "invalid expression"
                    )
                    raise ValueError(f"stop condition is invalid: {detail}")
                if self.awaiting_next_episode:
                    if not self._can_start_next_episode():
                        raise ValueError(f"episode limit reached ({self.boundaries})")
                    if self.recording_enabled:
                        raise ValueError("recorded episode complete; reset before recording again")
                    self._prepare_next_episode()
                self.stop_conditions.start_generation()
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
            elif command.name == "reset_episode":
                if self.awaiting_next_episode and not self._can_start_next_episode():
                    raise ValueError(f"episode limit reached ({self.boundaries})")
                seed_value = command.payload.get("seed")
                if isinstance(seed_value, bool):
                    raise ValueError("seed must be an integer")
                seed = None if seed_value in {None, ""} else validate_playback_seed(int(seed_value))
                self.temperature_changed = self.sampling_temperature != 1.0
                self.session.reset_episode(seed)
                self.clear_input()
                self.awaiting_next_episode = False
                self.remaining_steps = 0
                self.continue_target = None
                self.stop_conditions.reset()
                self._begin_capture()
                self._begin_recording()
                self._set_state(
                    "paused",
                    message=f"episode reset · seed {self.session.active_seed}",
                )
            elif command.name == "set_fps":
                fps = float(command.payload.get("fps", 0.0))
                if fps < 0 or not np.isfinite(fps):
                    raise ValueError("fps must be a finite value >= 0")
                self.target_fps = fps
                self.rgb_enabled = bool(
                    command.payload.get("rgb_enabled", getattr(self, "rgb_enabled", True))
                )
                self.revision += 1
                self._publish(self.session.last_transition)
            elif command.name == "set_stop_condition":
                if self.run_state != "paused":
                    raise ValueError("pause playback before editing the stop condition")
                source = command.payload.get("source")
                if not isinstance(source, str):
                    raise ValueError("stop condition source must be text")
                self.stop_conditions.set_source(source)
                self._set_state("paused")
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
        with self._trajectory_lock:
            return self._step_and_record()

    def _step_and_record(self) -> _PlaybackTransition | None:
        try:
            if self.seek_recording is not None:
                self.seek_recording.check_capacity()
            if self.recording_enabled and self.recording is not None:
                self.recording.check_capacity()
            transition = (
                self.session.step_human(self._human_labels())
                if self.driver == "human"
                else (
                    self.session.step(
                        action_selection_mode=self.sampling_mode,
                        **(
                            {"sampling_temperature": self.sampling_temperature}
                            if self.sampling_temperature != 1.0
                            else {}
                        ),
                    )
                    if hasattr(self.session, "policy_runtime")
                    else self.session.step(deterministic=self.sampling_mode == "deterministic")
                )
            )
            full = transition_payload(transition, reward_accounting=self.reward_accounting)
            current = (
                full
                if self.processing_features == PLAYER_PROCESSING_FEATURES
                else transition_payload(
                    transition,
                    reward_accounting=self.reward_accounting,
                    processing=self.processing_features,
                )
            )
            self.episode_rewards.append(full, self.reward_accounting)
            self.episode_actions.append(full)
            self._record_transition(transition, full=full)
        except Exception as exc:
            self._set_state("paused", message=str(exc))
            return None
        self.revision += 1
        self.capture.record_transition(transition)
        stop_match = self.stop_conditions.observe(
            events=transition.events,
            signals=transition.info,
            terminated=transition.terminated,
            truncated=transition.truncated,
            boundary=transition.boundary,
            evaluate=self.run_state == "playing",
        )
        if not self.stop_conditions.valid and self.run_state == "playing":
            detail = (
                self.stop_conditions.error.message
                if self.stop_conditions.error is not None
                else "invalid expression"
            )
            self._pause_after_transition(f"stop condition became invalid: {detail}")
        elif transition.boundary:
            self.boundaries += 1
            self.awaiting_next_episode = True
            self.remaining_steps = 0
            self.continue_target = None
            if not self._can_start_next_episode():
                self._pause_after_transition(f"episode limit reached ({self.boundaries})")
            elif stop_match is not None:
                self._pause_after_transition(self._stop_match_message(stop_match.values))
            elif self.recording_enabled:
                self._pause_after_transition("recorded episode complete")
            elif self.run_state == "playing":
                self._status_message = None
            else:
                self._pause_after_transition("episode complete")
        elif stop_match is not None:
            self._pause_after_transition(self._stop_match_message(stop_match.values))
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
        self._publish(
            transition,
            current=current,
            paced=self.run_state in {"playing", "stepping", "continuing"},
        )
        if transition.boundary and self.run_state == "playing":
            self._prepare_next_episode()
            self.revision += 1
            self._publish(None, paced=False)
        return transition

    def _run(self) -> None:
        next_step_at = time.perf_counter()
        while not self._stop.is_set():
            self._drain_commands()
            if self.seek_recording is not None:
                error = self.seek_recording.status().get("error")
                if error:
                    message = f"Playback storage failed: {error}. Reset the episode to resume."
                    if self._status_message != message:
                        self._set_state("paused", message=message)
            if self.recording_enabled and self.recording is not None:
                error = self.recording.status().get("error")
                if error:
                    message = f"Recording storage failed: {error}. Retry recording to resume."
                    if self._status_message != message:
                        self._set_state("paused", message=message)
            if self.run_state not in {"playing", "stepping", "continuing"}:
                time.sleep(0.005)
                continue
            fps = (
                (self.target_fps or 60.0)
                if self.driver == "human" and getattr(self, "rgb_enabled", True)
                else 0.0
            )
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
        self.reward_accounting = unavailable_reward_accounting(
            "recorded datasets do not preserve pre-transform reward accounting"
        )
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
                "sampling_temperature": getattr(self, "sampling_temperature", 1.0),
                "temperature_changed": getattr(self, "temperature_changed", False),
                "target_fps": self.target_fps,
                "rgb_enabled": getattr(self, "rgb_enabled", True),
                "episodes_limit": 1,
                "awaiting_next_episode": self.transition_index >= len(self.rows) - 1,
                "can_start_next_episode": False,
                "history_size": len(self.history),
                "config": (
                    f"Gymrec v3 episode {self.episode_id}\n"
                    "Recorded dataset playback is never checkpoint-promotion evidence."
                ),
                "reward_accounting": dict(self.reward_accounting),
                "attribution": unavailable_attribution(
                    "recorded datasets do not contain live-policy attribution"
                ),
                "cnn": unavailable_cnn_inspection(
                    "recorded datasets do not contain live-policy CNN activations"
                ),
            },
            "transition": self._transition,
            "history_point": current_history,
        }

    def _publish(self) -> None:
        self.encoder.submit(
            FRAME_GAME,
            self.sequence,
            self.current_frame if getattr(self, "rgb_enabled", True) else None,
        )
        payload = self._snapshot_payload()
        with self._snapshot_lock:
            self._latest_snapshot = payload
            self._snapshot_updates.append(payload)
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
        if command.name == "set_attribution":
            self._response(
                command,
                ok=False,
                error="recorded datasets do not contain live-policy attribution",
            )
            return
        if command.name == "set_cnn_inspection":
            self._response(
                command,
                ok=False,
                error="recorded datasets do not contain live-policy CNN activations",
            )
            return
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
                self.rgb_enabled = bool(
                    command.payload.get("rgb_enabled", getattr(self, "rgb_enabled", True))
                )
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
                "raw": None,
                "components": {},
                "accounting_error": None,
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
            "attribution": {
                "status": "off",
                "mode": "none",
                "generation": 0,
                "reason": "recorded_source",
            },
            "cnn": {
                "status": "off",
                "layer_id": None,
                "generation": 0,
                "reason": "recorded_source",
                "inspection": None,
            },
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
            if self.effective_fps > 0:
                now = time.perf_counter()
                if now < next_step_at:
                    time.sleep(min(next_step_at - now, 0.005))
                    continue
                next_step_at = max(next_step_at + 1.0 / self.effective_fps, now)
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
        self.reward_accounting = unavailable_reward_accounting(
            "human recordings do not preserve pre-transform reward accounting"
        )

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
        if command.name == "set_attribution":
            self._response(
                command,
                ok=False,
                error="human recording does not have a live policy to attribute",
            )
            return
        if command.name == "set_cnn_inspection":
            self._response(
                command,
                ok=False,
                error="human recording does not have a live policy to inspect",
            )
            return
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
                self.rgb_enabled = bool(
                    command.payload.get("rgb_enabled", getattr(self, "rgb_enabled", True))
                )
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
            payload = {
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
                    "rgb_enabled": getattr(self, "rgb_enabled", True),
                    "episodes_limit": int(getattr(self.args, "episodes", None) or 0),
                    "awaiting_next_episode": False,
                    "can_start_next_episode": False,
                    "history_size": len(self.history),
                    "config": (
                        "Human dataset recording. Browser input is translated through the "
                        "provider's declared control labels. This session is never promotion evidence."
                    ),
                    "reward_accounting": dict(self.reward_accounting),
                    "attribution": unavailable_attribution(
                        "human recording does not have a live policy to attribute"
                    ),
                    "cnn": unavailable_cnn_inspection(
                        "human recording does not have a live policy to inspect"
                    ),
                },
                "transition": self._transition,
                "history_point": current_history,
            }
            self._latest_snapshot = payload
            self._snapshot_updates.append(payload)

    def action(self, frame: np.ndarray) -> tuple[Any | None, bool]:
        self.encoder.submit(
            FRAME_GAME, self.sequence, frame if getattr(self, "rgb_enabled", True) else None
        )
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
            self._next_action_at = (
                max(self._next_action_at + 1.0 / self.effective_fps, now)
                if self.effective_fps > 0
                else now
            )
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
                "raw": None,
                "components": {},
                "accounting_error": None,
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
            "attribution": {
                "status": "off",
                "mode": "none",
                "generation": 0,
                "reason": "human_recording",
            },
            "cnn": {
                "status": "off",
                "layer_id": None,
                "generation": 0,
                "reason": "human_recording",
                "inspection": None,
            },
        }
        self.history.append(history_point_payload(self._transition))
        self.encoder.submit(
            FRAME_GAME, self.sequence, next_frame if getattr(self, "rgb_enabled", True) else None
        )
        self.revision += 1
        self._publish()


def playback_updates(runner):
    """One background read supplies the event loop with cached playback state."""
    drain = getattr(runner, "drain_snapshot_updates", None)
    snapshots = drain() if callable(drain) else []
    if not snapshots:
        snapshots = [runner.snapshot()]
    responses = []
    poll = getattr(runner, "poll_response", None)
    for _ in range(COMMAND_QUEUE_LIMIT):
        if callable(poll):
            response = poll()
            if response is None:
                break
        else:
            try:
                response = runner.responses.get_nowait()
            except queue.Empty:
                break
        responses.append(response)
    return dict(
        snapshots=snapshots,
        frames=runner.encoder.latest(),
        responses=responses,
        stopped=runner.stopped,
        session_change=int(getattr(runner, "session_change", 0)),
    )
