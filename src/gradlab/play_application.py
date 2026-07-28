from __future__ import annotations

import queue
import threading
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from gradlab.play_runtime import (
    ActivePlayback,
    PlaySourceSpec,
    PlaybackCandidate,
    PlaybackLoader,
)


class _EmptyEncoder:
    def latest(self) -> dict[int, tuple[int, bytes]]:
        return {}


class PlaybackHost:
    """Stable web-server facade for zero or one replaceable playback runner."""

    SOURCE_COMMANDS = {
        "approve_source",
        "browse_sources",
        "cancel_source",
        "retry_source",
        "select_source",
        "set_contract_mode",
    }

    def __init__(
        self,
        loader: PlaybackLoader,
        *,
        initial_route: Mapping[str, Any] | None = None,
        initial_source: PlaySourceSpec | None = None,
    ) -> None:
        self.loader = loader
        self._lock = threading.RLock()
        self._empty_encoder = _EmptyEncoder()
        self._responses: queue.SimpleQueue[Any] = queue.SimpleQueue()
        self._active: ActivePlayback | None = None
        self._candidate: PlaybackCandidate | None = None
        self._worker: threading.Thread | None = None
        self._generation = 0
        self._session_epoch = 0
        self._revision = 0
        self._stopped = False
        self._phase = "selecting"
        self._message = ""
        self._error = ""
        self._approval: dict[str, Any] | None = None
        self._route = dict(initial_route or {"level": "environments"})
        self._last_source = initial_source
        self._session_change = 0

    @property
    def encoder(self):
        with self._lock:
            return self._active.runner.encoder if self._active is not None else self._empty_encoder

    @property
    def stopped(self) -> bool:
        with self._lock:
            return self._stopped

    @property
    def session_epoch(self) -> int:
        with self._lock:
            return self._session_epoch

    @property
    def session_change(self) -> int:
        with self._lock:
            return self._session_change

    @property
    def has_active_runner(self) -> bool:
        with self._lock:
            return self._active is not None and self._phase == "active"

    def start(self) -> None:
        with self._lock:
            source = self._last_source
        if source is not None:
            self._begin_prepare(source)

    def stop(self) -> None:
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            self._generation += 1
            active = self._active
            candidate = self._candidate
            self._active = None
            self._candidate = None
        if active is not None:
            active.close()
        if candidate is not None:
            candidate.cleanup()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=15.0)

    def clear_input(self) -> None:
        with self._lock:
            active = self._active
        if active is not None:
            active.runner.clear_input()

    def update_input(self, labels: Sequence[str], *, focused: bool) -> None:
        with self._lock:
            active = self._active if self._phase == "active" else None
        if active is not None:
            active.runner.update_input(labels, focused=focused)

    def _app_payload(self) -> dict[str, Any]:
        return {
            "phase": self._phase,
            "message": self._message,
            "error": self._error,
            "route": dict(self._route),
            "approval": self._approval,
            "has_active_runner": self._active is not None,
            "source": self._last_source.to_dict() if self._last_source is not None else None,
        }

    def snapshot(self) -> dict[str, Any]:
        from gradlab.play_web import PROTOCOL_VERSION

        with self._lock:
            if self._active is not None and self._phase == "active":
                snapshot = self._active.runner.snapshot()
            else:
                snapshot = {
                    "type": "snapshot",
                    "protocol": PROTOCOL_VERSION,
                    "revision": self._revision,
                    "sequence": 0,
                    "run_state": "paused",
                    "driver": "policy",
                    "interactive": False,
                    "policy": None,
                    "status_message": self._message or self._error or None,
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
                    },
                    "transition": None,
                }
            return {
                **snapshot,
                "session_epoch": self._session_epoch,
                "app": self._app_payload(),
            }

    def history_payload(self) -> dict[str, Any]:
        with self._lock:
            if self._active is None or self._phase != "active":
                return {
                    "type": "history",
                    "session_epoch": self._session_epoch,
                    "points": [],
                }
            payload = dict(self._active.runner.history_payload())
            payload["session_epoch"] = self._session_epoch
            return payload

    def episode_start_payload(
        self,
    ) -> tuple[dict[str, Any], dict[int, tuple[int, bytes]]]:
        with self._lock:
            if self._active is None or self._phase != "active":
                return {}, {}
            snapshot, frames = self._active.runner.episode_start_payload()
            return (
                {
                    **snapshot,
                    "session_epoch": self._session_epoch,
                    "app": self._app_payload(),
                },
                frames,
            )

    def poll_response(self):
        try:
            return self._responses.get_nowait()
        except queue.Empty:
            pass
        with self._lock:
            active = self._active
        if active is None:
            return None
        try:
            return active.runner.responses.get_nowait()
        except queue.Empty:
            return None

    def _response(self, command, *, ok: bool, **extra: Any) -> None:
        from gradlab.play_web import PlaybackResponse

        self._responses.put(
            PlaybackResponse(
                command.client_id,
                {
                    "type": "command_result",
                    "id": command.command_id,
                    "ok": ok,
                    "revision": self._revision,
                    **extra,
                },
            )
        )

    def _set_state(
        self,
        phase: str,
        *,
        message: str = "",
        error: str = "",
        approval: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._phase = phase
            self._message = message
            self._error = error
            self._approval = approval
            self._revision += 1

    def _progress(self, generation: int, phase: str, message: str) -> None:
        with self._lock:
            if generation != self._generation or self._stopped:
                return
        self._set_state(phase, message=message)

    def _activate_candidate(
        self,
        generation: int,
        candidate: PlaybackCandidate,
        approval_hash: str,
    ) -> None:
        previous: ActivePlayback | None = None
        try:
            active = self.loader.activate(
                candidate,
                approval_hash=approval_hash,
                progress=lambda phase, message: self._progress(generation, phase, message),
            )
            with self._lock:
                if generation != self._generation or self._stopped:
                    active.close()
                    return
                previous = self._active
                self._session_epoch += 1
                set_epoch = getattr(active.runner.encoder, "set_epoch", None)
                if callable(set_epoch):
                    set_epoch(self._session_epoch)
                active.runner.start()
                self._active = active
                self._candidate = None
                self._phase = "active"
                self._message = ""
                self._error = ""
                self._approval = None
                self._revision += 1
                self._session_change += 1
            if previous is not None:
                previous.close()
        except Exception as exc:
            with self._lock:
                current = generation == self._generation and not self._stopped
                if current:
                    self._candidate = None
                    has_previous = self._active is not None
                else:
                    has_previous = False
            if current and has_previous:
                self._set_state("active", error=str(exc))
            elif current:
                self._set_state("error", error=str(exc))
        finally:
            candidate.cleanup()

    def _prepare_worker(self, generation: int, source: PlaySourceSpec) -> None:
        candidate: PlaybackCandidate | None = None
        try:
            candidate = self.loader.prepare(
                source,
                lambda phase, message: self._progress(generation, phase, message),
            )
            with self._lock:
                if generation != self._generation or self._stopped:
                    candidate.cleanup()
                    return
                self._candidate = candidate
            if candidate.approval_required:
                self._set_state(
                    "approval_required",
                    message="Review this exact executable model closure",
                    approval=candidate.approval_payload(),
                )
                return
            self._activate_candidate(generation, candidate, candidate.staged.manifest_hash)
        except Exception as exc:
            if candidate is not None:
                candidate.cleanup()
            with self._lock:
                current = generation == self._generation and not self._stopped
            if current:
                self._set_state("error", error=str(exc))

    def _begin_prepare(self, source: PlaySourceSpec) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                raise RuntimeError("another model source is still being prepared")
            self._generation += 1
            generation = self._generation
            candidate = self._candidate
            self._candidate = None
            self._last_source = source
            self._phase = "resolving"
            self._message = "Resolving model source"
            self._error = ""
            self._approval = None
            self._revision += 1
            worker = threading.Thread(
                target=self._prepare_worker,
                args=(generation, source),
                name="gradlab-playback-loader",
                daemon=True,
            )
            self._worker = worker
        if candidate is not None:
            candidate.cleanup()
        self.clear_input()
        with self._lock:
            active = self._active
        if active is not None:
            try:
                from gradlab.play_web import PlaybackCommand

                active.runner.submit(
                    PlaybackCommand(uuid.uuid4().hex, "application", "pause", {}, None)
                )
            except queue.Full:
                pass
        worker.start()

    @staticmethod
    def _source_from_payload(payload: Mapping[str, Any]) -> PlaySourceSpec:
        source = payload.get("source")
        if not isinstance(source, Mapping):
            raise ValueError("source selection is missing")
        kind = str(source.get("kind") or "")
        if kind not in {"manifest", "huggingface", "local", "public_run"}:
            raise ValueError("unsupported playback source")
        value = str(source.get("value") or "").strip()
        if not value:
            raise ValueError("playback source is empty")
        seed_value = source.get("seed")
        if seed_value is not None and (
            isinstance(seed_value, bool) or not isinstance(seed_value, int)
        ):
            raise ValueError("playback source seed must be an integer")
        contract_mode = str(source.get("contract_mode") or "training")
        if contract_mode not in {"training", "evaluation", "counterfactual"}:
            raise ValueError(f"unsupported playback contract mode {contract_mode!r}")
        reward_clip_override = source.get("reward_clip_override")
        if reward_clip_override is not None and not isinstance(
            reward_clip_override,
            bool,
        ):
            raise ValueError("reward clipping override must be a boolean or null")
        return PlaySourceSpec(
            kind=kind,  # type: ignore[arg-type]
            value=value,
            entity=str(source.get("entity") or ""),
            project=str(source.get("project") or ""),
            run_id=str(source.get("run_id") or ""),
            checkpoint_id=str(source.get("checkpoint_id") or ""),
            seed=seed_value,
            contract_mode=contract_mode,  # type: ignore[arg-type]
            reward_clip_override=reward_clip_override,
        )

    def submit(self, command) -> None:
        if command.name not in self.SOURCE_COMMANDS and command.name != "stop":
            with self._lock:
                active = self._active if self._phase == "active" else None
            if active is None:
                self._response(command, ok=False, error="no active playback session")
                return
            active.runner.submit(command)
            return
        try:
            if command.name == "select_source":
                source = self._source_from_payload(command.payload)
                route = command.payload.get("route")
                if isinstance(route, Mapping):
                    with self._lock:
                        self._route = dict(route)
                self._begin_prepare(source)
            elif command.name == "set_contract_mode":
                mode = str(command.payload.get("mode") or "")
                if mode not in {"training", "evaluation", "counterfactual"}:
                    raise ValueError(f"unsupported playback contract mode {mode!r}")
                with self._lock:
                    source = self._last_source
                if source is None:
                    raise ValueError("no playback source is active")
                reward_clip_override = False if mode == "counterfactual" else None
                self._begin_prepare(
                    replace(
                        source,
                        contract_mode=mode,  # type: ignore[arg-type]
                        reward_clip_override=reward_clip_override,
                    )
                )
            elif command.name == "approve_source":
                approval_hash = str(command.payload.get("manifest_hash") or "").strip()
                with self._lock:
                    candidate = self._candidate
                    generation = self._generation
                if candidate is None or self._phase != "approval_required":
                    raise ValueError("no model closure is awaiting approval")
                worker = threading.Thread(
                    target=self._activate_candidate,
                    args=(generation, candidate, approval_hash),
                    name="gradlab-playback-activator",
                    daemon=True,
                )
                with self._lock:
                    self._worker = worker
                    self._phase = "loading"
                    self._message = "Loading approved model"
                    self._approval = None
                    self._revision += 1
                worker.start()
            elif command.name == "browse_sources":
                with self._lock:
                    route = command.payload.get("route")
                    if isinstance(route, Mapping):
                        self._route = dict(route)
                    elif self._active is None:
                        self._route = {"level": "environments"}
                    self._phase = "selecting"
                    self._message = ""
                    self._error = ""
                    self._approval = None
                    self._revision += 1
                self.clear_input()
                with self._lock:
                    active = self._active
                if active is not None:
                    from gradlab.play_web import PlaybackCommand

                    active.runner.submit(
                        PlaybackCommand(uuid.uuid4().hex, command.client_id, "pause", {}, None)
                    )
            elif command.name == "cancel_source":
                with self._lock:
                    self._generation += 1
                    candidate = self._candidate
                    self._candidate = None
                    self._phase = "active" if self._active is not None else "selecting"
                    self._message = ""
                    self._error = ""
                    self._approval = None
                    self._revision += 1
                if candidate is not None:
                    candidate.cleanup()
            elif command.name == "retry_source":
                with self._lock:
                    source = self._last_source
                if source is None:
                    raise ValueError("no source is available to retry")
                self._begin_prepare(source)
            elif command.name == "stop":
                self._response(command, ok=True)
                self.stop()
                return
            self._response(command, ok=True)
        except Exception as exc:
            self._response(command, ok=False, error=str(exc))


__all__ = ["PlaybackHost"]
