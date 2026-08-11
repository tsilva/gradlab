from __future__ import annotations

import queue
import threading
import uuid
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from typing import Any

from gradlab.play_runtime import (
    ActivePlayback,
    PlaySourceSpec,
    PlaybackCandidate,
    PlaybackLoader,
)
from gradlab.model_sources import (
    NoDefaultPublicRunCheckpointError,
    is_public_checkpoint_manifest_ref,
)
from gradlab.play_web import idle_playback_snapshot
from gradlab.play_processing import (
    PLAYER_PROCESSING_FEATURES,
    normalize_player_processing,
)
from gradlab.run_contracts import checkpoint_id


def _resolved_public_run_route(
    candidate: PlaybackCandidate,
    current_route: Mapping[str, Any],
) -> dict[str, Any] | None:
    spec = candidate.spec
    if spec.kind != "public_run" or current_route.get("checkpoint_id"):
        return None
    bundle = candidate.source.bundle
    recipe_document = bundle.recipe.get("recipe")
    model_checkpoint = bundle.model.get("checkpoint")
    if not isinstance(recipe_document, Mapping) or not isinstance(model_checkpoint, Mapping):
        return None
    goal_variant = recipe_document.get("goal_variant")
    if not isinstance(goal_variant, Mapping):
        return None
    goal_slug = str(goal_variant.get("goal_slug") or "").strip()
    environment_id, separator, _goal_path = goal_slug.partition("/")
    if not separator:
        environment_id = goal_slug
    goal_id = str(goal_variant.get("goal_id") or "").strip()
    goal_variant_id = str(goal_variant.get("variant_id") or "").strip()
    run_id = str(spec.run_id).strip()
    if not all((environment_id, goal_id, goal_variant_id, run_id)):
        return None
    try:
        resolved_checkpoint_id = checkpoint_id(
            step=int(model_checkpoint.get("step")),
            sha256=str(model_checkpoint.get("sha256") or ""),
        )
    except TypeError, ValueError:
        return None
    return {
        "level": "runs",
        "environment_id": environment_id,
        "goal_id": goal_id,
        "goal_variant_id": goal_variant_id,
        "run_id": run_id,
        "checkpoint_id": str(spec.checkpoint_id or resolved_checkpoint_id),
    }


class _EmptyEncoder:
    def latest(self) -> dict[int, tuple[int, bytes]]:
        return {}

    def retained(
        self,
        sequence: int,
        *,
        epoch: int | None = None,
        timeout: float = 0.0,
        kinds: Iterable[int] | None = None,
    ) -> dict[int, tuple[int, bytes]]:
        del sequence, epoch, timeout, kinds
        return {}


class PlaybackHost:
    """Stable web-server facade for zero or one replaceable playback runner."""

    SOURCE_COMMANDS = {
        "browse_sources",
        "cancel_source",
        "prefetch_sources",
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
        self._prefetch_lock = threading.Lock()
        self._prefetch_pending: set[tuple[str, str]] = set()
        self._prefetched: set[tuple[str, str]] = set()
        self._generation = 0
        self._session_epoch = 0
        self._revision = 0
        self._stopped = False
        self._phase = "selecting"
        self._message = ""
        self._error = ""
        self._route = dict(initial_route or {"level": "environments"})
        self._last_source = initial_source
        self._session_change = 0
        self._processing_features = PLAYER_PROCESSING_FEATURES

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

    def active_recipe_document(self) -> tuple[dict[str, Any], dict[str, Any]] | None:
        with self._lock:
            if self._active is None or self._phase != "active":
                return None
            return (
                deepcopy(self._active.source.bundle.recipe),
                {
                    "kind": "active-playback",
                    "artifact_ref": str(self._active.source.artifact_ref or ""),
                    "artifact_name": str(self._active.source.artifact_name or ""),
                    "checkpoint_step": self._active.source.checkpoint_step,
                    "source": self._active.source.bundle.source,
                    "revision": str(self._active.source.bundle.revision or ""),
                },
            )

    def active_publication_context(self) -> dict[str, Any] | None:
        with self._lock:
            if self._active is None or self._phase != "active":
                return None
            capture = getattr(self._active.runner, "capture", None)
            status = (
                capture.status()
                if capture is not None
                else {
                    "enabled": False,
                    "recording": False,
                    "episode_in_progress": False,
                    "ready": False,
                    "error": "episode capture is unavailable",
                    "latest": None,
                }
            )
            return {
                "spec": self._active.spec,
                "source": self._active.source,
                "bundle": self._active.source.bundle,
                "capture": status,
                "session_epoch": self._session_epoch,
            }

    def render_publication_capture(self) -> dict[str, Any]:
        with self._lock:
            if self._active is None or self._phase != "active":
                raise ValueError("no active player checkpoint is available")
            capture = getattr(self._active.runner, "capture", None)
            if capture is None:
                raise ValueError("episode capture is unavailable")
            return capture.render()

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

    def set_processing(self, features: Iterable[object]) -> None:
        normalized = normalize_player_processing(features)
        with self._lock:
            self._processing_features = normalized
            active = self._active if self._phase == "active" else None
        if active is not None:
            configure = getattr(active.runner, "set_processing", None)
            if callable(configure):
                configure(normalized)

    def _app_payload(self) -> dict[str, Any]:
        return {
            "phase": self._phase,
            "message": self._message,
            "error": self._error,
            "route": dict(self._route),
            "has_active_runner": self._active is not None,
            "source": self._last_source.to_dict() if self._last_source is not None else None,
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            if self._active is not None and self._phase == "active":
                snapshot = self._active.runner.snapshot()
            else:
                snapshot = idle_playback_snapshot(
                    revision=self._revision,
                    status_message=self._message or self._error or None,
                )
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
    ) -> None:
        with self._lock:
            self._phase = phase
            self._message = message
            self._error = error
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
    ) -> None:
        previous: ActivePlayback | None = None
        try:
            active = self.loader.activate(
                candidate,
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
                configure = getattr(active.runner, "set_processing", None)
                if callable(configure):
                    configure(self._processing_features)
                active.runner.start()
                self._active = active
                self._candidate = None
                self._phase = "active"
                self._message = ""
                self._error = ""
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
                resolved_route = _resolved_public_run_route(candidate, self._route)
                if resolved_route is not None:
                    self._route = resolved_route
            self._activate_candidate(generation, candidate)
        except Exception as exc:
            if candidate is not None:
                candidate.cleanup()
            with self._lock:
                current = generation == self._generation and not self._stopped
            if current:
                if isinstance(exc, NoDefaultPublicRunCheckpointError):
                    self._set_state(
                        "selecting",
                        message=(
                            "This run has no promoted or final checkpoint yet. "
                            "Choose one of its published checkpoints to play now."
                        ),
                    )
                else:
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

    def _prefetch_worker(self, source: PlaySourceSpec, key: tuple[str, str]) -> None:
        succeeded = False
        try:
            self.loader.prefetch(source)
            succeeded = True
        except Exception:
            # Neighbor warming is opportunistic and must never disturb playback.
            pass
        finally:
            with self._prefetch_lock:
                self._prefetch_pending.discard(key)
                if succeeded:
                    self._prefetched.add(key)

    def _begin_prefetch(self, payload: Mapping[str, Any]) -> None:
        raw_sources = payload.get("sources")
        if (
            isinstance(raw_sources, str | bytes)
            or not isinstance(raw_sources, Sequence)
            or len(raw_sources) > 2
        ):
            raise ValueError("checkpoint prefetch requires at most two sources")
        for raw_source in raw_sources:
            if not isinstance(raw_source, Mapping):
                raise ValueError("checkpoint prefetch source is invalid")
            source = self._source_from_payload({"source": raw_source})
            if source.kind != "public_run" or not is_public_checkpoint_manifest_ref(source.value):
                raise ValueError(
                    "checkpoint prefetch requires an immutable public checkpoint manifest"
                )
            key = (source.kind, source.value)
            with self._prefetch_lock:
                if key in self._prefetch_pending or key in self._prefetched:
                    continue
                self._prefetch_pending.add(key)
            threading.Thread(
                target=self._prefetch_worker,
                args=(source, key),
                name="gradlab-checkpoint-prefetch",
                daemon=True,
            ).start()

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
            elif command.name == "prefetch_sources":
                self._begin_prefetch(command.payload)
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
            elif command.name == "browse_sources":
                with self._lock:
                    self._generation += 1
                    candidate = self._candidate
                    active = self._active
                    self._candidate = None
                    self._active = None
                    route = command.payload.get("route")
                    if isinstance(route, Mapping):
                        self._route = dict(route)
                    elif active is None:
                        self._route = {"level": "environments"}
                    self._phase = "selecting"
                    self._message = ""
                    self._error = ""
                    self._revision += 1
                    if active is not None:
                        self._session_change += 1
                if candidate is not None:
                    candidate.cleanup()
                if active is not None:
                    active.close()
            elif command.name == "cancel_source":
                with self._lock:
                    self._generation += 1
                    candidate = self._candidate
                    self._candidate = None
                    self._phase = "active" if self._active is not None else "selecting"
                    self._message = ""
                    self._error = ""
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
