from __future__ import annotations

import multiprocessing
import os
import signal
import threading
import traceback
from collections.abc import Mapping, Sequence
from multiprocessing.connection import Connection
from typing import Any

from gradlab.operator_credentials import PROTECTED_ENV_NAMES
from gradlab.play_runtime import PlaySourceSpec
from gradlab.play_web import idle_playback_snapshot


PLAYBACK_RPC_TIMEOUT_SECONDS = 30.0


def _worker_main(
    connection: Connection,
    args: Any,
    argv: list[str],
    explicit_seed: bool,
    initial_route: Mapping[str, Any],
    initial_source: PlaySourceSpec | None,
) -> None:
    host = None
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        leaked = sorted(name for name in PROTECTED_ENV_NAMES if os.environ.get(name))
        if leaked:
            raise RuntimeError(
                f"playback worker received protected environment names: {', '.join(leaked)}"
            )
        from gradlab.play_application import PlaybackHost
        from gradlab.play_runtime import PlaybackLoader

        host = PlaybackHost(
            PlaybackLoader(
                args,
                argv=argv,
                explicit_seed=explicit_seed,
            ),
            initial_route=initial_route,
            initial_source=initial_source,
        )
        connection.send({"ok": True, "value": {"ready": True}})
        while True:
            request = connection.recv()
            operation = str(request.get("operation") or "")
            if operation == "close":
                host.stop()
                connection.send({"ok": True, "value": None})
                return
            if operation == "start":
                value = host.start()
            elif operation == "snapshot":
                value = host.snapshot()
            elif operation == "history_payload":
                value = host.history_payload()
            elif operation == "episode_start_payload":
                value = host.episode_start_payload()
            elif operation == "poll_response":
                value = host.poll_response()
            elif operation == "encoder_latest":
                value = host.encoder.latest()
            elif operation == "encoder_retained":
                value = host.encoder.retained(
                    int(request["sequence"]),
                    epoch=int(request["epoch"]),
                    timeout=float(request["timeout"]),
                )
            elif operation == "property":
                name = str(request.get("name") or "")
                if name not in {
                    "stopped",
                    "session_epoch",
                    "session_change",
                    "has_active_runner",
                }:
                    raise ValueError("unsupported playback worker property")
                value = getattr(host, name)
            elif operation == "active_recipe_document":
                value = host.active_recipe_document()
            elif operation == "clear_input":
                value = host.clear_input()
            elif operation == "update_input":
                value = host.update_input(
                    tuple(str(item) for item in request.get("labels") or ()),
                    focused=bool(request.get("focused")),
                )
            elif operation == "submit":
                value = host.submit(request["command"])
            else:
                raise ValueError("unsupported playback worker operation")
            connection.send({"ok": True, "value": value})
    except EOFError:
        return
    except BaseException as exc:
        try:
            connection.send(
                {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=20),
                }
            )
        except Exception:
            pass
    finally:
        if host is not None:
            try:
                host.stop()
            except Exception:
                pass
        connection.close()


class _RemoteEncoder:
    def __init__(self, host: IsolatedPlaybackHost) -> None:
        self._host = host

    def latest(self) -> dict[int, tuple[int, bytes]]:
        value = self._host._rpc("encoder_latest")
        return dict(value or {})

    def retained(
        self,
        sequence: int,
        *,
        epoch: int,
        timeout: float,
    ) -> dict[int, tuple[int, bytes]]:
        value = self._host._rpc(
            "encoder_retained",
            sequence=int(sequence),
            epoch=int(epoch),
            timeout=float(timeout),
            timeout_seconds=max(
                PLAYBACK_RPC_TIMEOUT_SECONDS,
                float(timeout) + 5.0,
            ),
        )
        return dict(value or {})


class IsolatedPlaybackHost:
    """PlaybackHost facade whose model and environment live in a fresh process."""

    def __init__(
        self,
        args: Any,
        *,
        argv: Sequence[str],
        explicit_seed: bool,
        initial_route: Mapping[str, Any] | None = None,
        initial_source: PlaySourceSpec | None = None,
    ) -> None:
        self._args = args
        self._argv = list(argv)
        self._explicit_seed = bool(explicit_seed)
        self._initial_route = dict(initial_route or {"level": "environments"})
        self._initial_source = initial_source
        self._context = multiprocessing.get_context("spawn")
        self._connection: Connection | None = None
        self._process: multiprocessing.Process | None = None
        self._lock = threading.RLock()
        self._closed = False
        self.encoder = _RemoteEncoder(self)

    def _ensure_started(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("playback worker is closed")
            if self._process is not None:
                return
            parent, child = self._context.Pipe(duplex=True)
            process = self._context.Process(
                target=_worker_main,
                args=(
                    child,
                    self._args,
                    self._argv,
                    self._explicit_seed,
                    self._initial_route,
                    self._initial_source,
                ),
                name="gradlab-playback-worker",
            )
            process.start()
            child.close()
            self._connection = parent
            self._process = process
            response = self._receive(timeout_seconds=PLAYBACK_RPC_TIMEOUT_SECONDS)
            self._unwrap(response)

    def _receive(self, *, timeout_seconds: float) -> Mapping[str, Any]:
        connection = self._connection
        process = self._process
        if connection is None or process is None:
            raise RuntimeError("playback worker is not running")
        if not connection.poll(max(0.0, float(timeout_seconds))):
            if not process.is_alive():
                raise RuntimeError(
                    f"playback worker exited unexpectedly with code {process.exitcode}"
                )
            raise TimeoutError("playback worker did not respond")
        response = connection.recv()
        if not isinstance(response, Mapping):
            raise RuntimeError("playback worker returned an invalid response")
        return response

    @staticmethod
    def _unwrap(response: Mapping[str, Any]) -> Any:
        if response.get("ok") is True:
            return response.get("value")
        detail = str(response.get("error") or "unknown worker failure")
        error_type = str(response.get("error_type") or "RuntimeError")
        raise RuntimeError(f"playback worker {error_type}: {detail}")

    def _rpc(
        self,
        operation: str,
        *,
        timeout_seconds: float = PLAYBACK_RPC_TIMEOUT_SECONDS,
        **payload: Any,
    ) -> Any:
        self._ensure_started()
        with self._lock:
            connection = self._connection
            if connection is None:
                raise RuntimeError("playback worker is not running")
            try:
                connection.send({"operation": operation, **payload})
                return self._unwrap(
                    self._receive(timeout_seconds=timeout_seconds)
                )
            except (BrokenPipeError, EOFError, OSError) as exc:
                raise RuntimeError("playback worker connection failed") from exc

    def start(self) -> None:
        self._rpc("start")

    def stop(self) -> None:
        with self._lock:
            if self._closed:
                return
            process = self._process
            connection = self._connection
            self._closed = True
            if process is None or connection is None:
                return
            try:
                connection.send({"operation": "close"})
                if connection.poll(15.0):
                    connection.recv()
            except (BrokenPipeError, EOFError, OSError):
                pass
            finally:
                connection.close()
                process.join(timeout=15.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5.0)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5.0)
                self._connection = None
                self._process = None

    def snapshot(self) -> dict[str, Any]:
        if self._process is None:
            return {
                **idle_playback_snapshot(),
                "session_epoch": 0,
                "app": {
                    "phase": "selecting",
                    "message": "",
                    "error": "",
                    "route": dict(self._initial_route),
                    "has_active_runner": False,
                    "source": (
                        self._initial_source.to_dict()
                        if self._initial_source is not None
                        else None
                    ),
                },
            }
        return dict(self._rpc("snapshot"))

    def history_payload(self) -> dict[str, Any]:
        return dict(self._rpc("history_payload"))

    def episode_start_payload(
        self,
    ) -> tuple[dict[str, Any], dict[int, tuple[int, bytes]]]:
        snapshot, frames = self._rpc("episode_start_payload")
        return dict(snapshot), dict(frames)

    def poll_response(self) -> Any:
        return self._rpc("poll_response")

    def active_recipe_document(self) -> Any:
        return self._rpc("active_recipe_document")

    def clear_input(self) -> None:
        self._rpc("clear_input")

    def update_input(self, labels: Sequence[str], *, focused: bool) -> None:
        self._rpc("update_input", labels=list(labels), focused=bool(focused))

    def submit(self, command: Any) -> None:
        self._rpc("submit", command=command)

    def _property(self, name: str, default: Any) -> Any:
        if self._process is None:
            return default
        return self._rpc("property", name=name)

    @property
    def stopped(self) -> bool:
        if self._closed:
            return True
        return bool(self._property("stopped", False))

    @property
    def session_epoch(self) -> int:
        return int(self._property("session_epoch", 0))

    @property
    def session_change(self) -> int:
        return int(self._property("session_change", 0))

    @property
    def has_active_runner(self) -> bool:
        return bool(self._property("has_active_runner", False))


__all__ = ["IsolatedPlaybackHost", "PLAYBACK_RPC_TIMEOUT_SECONDS"]
