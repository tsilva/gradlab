from __future__ import annotations

import atexit
import shutil
import socket
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit

from gradlab.operator_credentials import DstackCoordinatorProfile, SshTunnelProfile


SSH_CONNECT_TIMEOUT_SECONDS = 5
TUNNEL_STARTUP_TIMEOUT_SECONDS = 10.0
TUNNEL_POLL_SECONDS = 0.05


class CoordinatorConnectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class CoordinatorConnectionReport:
    mode: Literal["direct", "existing", "managed-ssh-tunnel"]
    endpoint: str
    coordinator_id: str
    owned: bool

    def as_manifest(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "endpoint": self.endpoint,
            "coordinator_id": self.coordinator_id,
            "owned": self.owned,
        }


def _tcp_reachable(host: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class CoordinatorConnectionManager:
    def __init__(
        self,
        *,
        ssh_executable: str = "ssh",
        executable_lookup: Callable[[str], str | None] = shutil.which,
        tcp_probe: Callable[[str, int, float], bool] = _tcp_reachable,
        process_factory: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        startup_timeout: float = TUNNEL_STARTUP_TIMEOUT_SECONDS,
        poll_seconds: float = TUNNEL_POLL_SECONDS,
    ):
        self._ssh_executable = ssh_executable
        self._executable_lookup = executable_lookup
        self._tcp_probe = tcp_probe
        self._process_factory = process_factory
        self._monotonic = monotonic
        self._sleep = sleep
        self._startup_timeout = float(startup_timeout)
        self._poll_seconds = float(poll_seconds)
        self._owned: dict[str, subprocess.Popen[str]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _local_endpoint(profile: DstackCoordinatorProfile) -> tuple[str, int, str]:
        try:
            parsed = urlsplit(profile.server_url)
            host = str(parsed.hostname or "")
            port = parsed.port
        except ValueError as exc:
            raise CoordinatorConnectionError(
                f"invalid dstack coordinator endpoint: {profile.server_url}"
            ) from exc
        if parsed.scheme not in {"http", "https"} or not host or port is None:
            raise CoordinatorConnectionError(
                f"dstack coordinator endpoint must include HTTP(S), host, and port: "
                f"{profile.server_url}"
            )
        return host, port, profile.server_url.rstrip("/")

    @staticmethod
    def _tunnel_command(
        *,
        ssh_executable: str,
        destination: str,
        local_host: str,
        local_port: int,
        tunnel: SshTunnelProfile,
    ) -> list[str]:
        return [
            ssh_executable,
            "-T",
            "-N",
            "-o",
            "BatchMode=yes",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            f"ConnectTimeout={SSH_CONNECT_TIMEOUT_SECONDS}",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
            "-L",
            f"{local_host}:{local_port}:{tunnel.remote_host}:{tunnel.remote_port}",
            "--",
            destination,
        ]

    @staticmethod
    def _process_detail(process: subprocess.Popen[str]) -> str:
        try:
            _stdout, stderr = process.communicate(timeout=0.2)
        except subprocess.TimeoutExpired:
            return ""
        return " ".join(str(stderr or "").strip().split())[:500]

    @staticmethod
    def _stop_process(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)

    def _wait_until_reachable(
        self,
        process: subprocess.Popen[str],
        *,
        host: str,
        port: int,
    ) -> bool:
        deadline = self._monotonic() + self._startup_timeout
        while True:
            if self._tcp_probe(host, port, min(self._poll_seconds, 0.2)):
                return True
            if process.poll() is not None or self._monotonic() >= deadline:
                return False
            self._sleep(self._poll_seconds)

    def ensure(self, profile: DstackCoordinatorProfile) -> CoordinatorConnectionReport:
        host, port, endpoint = self._local_endpoint(profile)
        tunnel = getattr(profile, "ssh_tunnel", None)
        if tunnel is None:
            return CoordinatorConnectionReport(
                mode="direct",
                endpoint=endpoint,
                coordinator_id=profile.coordinator_id,
                owned=False,
            )
        if host not in {"127.0.0.1", "localhost"}:
            raise CoordinatorConnectionError(
                "automatic SSH tunneling requires a loopback dstack server_url"
            )
        with self._lock:
            process = self._owned.get(endpoint)
            if process is not None and process.poll() is not None:
                self._owned.pop(endpoint, None)
                process = None
            if self._tcp_probe(host, port, 0.2):
                return CoordinatorConnectionReport(
                    mode="managed-ssh-tunnel" if process is not None else "existing",
                    endpoint=endpoint,
                    coordinator_id=profile.coordinator_id,
                    owned=process is not None,
                )
            if process is not None:
                self._stop_process(process)
                self._owned.pop(endpoint, None)
            executable = self._executable_lookup(self._ssh_executable)
            if executable is None:
                raise CoordinatorConnectionError(
                    f"{self._ssh_executable} is required for coordinator {profile.coordinator_id!r}"
                )
            failures: list[str] = []
            for destination in tunnel.destinations:
                command = self._tunnel_command(
                    ssh_executable=executable,
                    destination=destination,
                    local_host=host,
                    local_port=port,
                    tunnel=tunnel,
                )
                candidate = self._process_factory(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if self._wait_until_reachable(candidate, host=host, port=port):
                    if candidate.poll() is None:
                        self._owned[endpoint] = candidate
                        return CoordinatorConnectionReport(
                            mode="managed-ssh-tunnel",
                            endpoint=endpoint,
                            coordinator_id=profile.coordinator_id,
                            owned=True,
                        )
                    return CoordinatorConnectionReport(
                        mode="existing",
                        endpoint=endpoint,
                        coordinator_id=profile.coordinator_id,
                        owned=False,
                    )
                detail = self._process_detail(candidate)
                self._stop_process(candidate)
                failures.append(f"{destination}: {detail or 'connection failed'}")
                if self._tcp_probe(host, port, 0.2):
                    return CoordinatorConnectionReport(
                        mode="existing",
                        endpoint=endpoint,
                        coordinator_id=profile.coordinator_id,
                        owned=False,
                    )
            raise CoordinatorConnectionError(
                f"failed to open SSH tunnel for coordinator {profile.coordinator_id!r}: "
                + "; ".join(failures)
            )

    def close(self) -> None:
        with self._lock:
            processes = list(self._owned.values())
            self._owned.clear()
        for process in processes:
            self._stop_process(process)


_CONNECTION_MANAGER = CoordinatorConnectionManager()


def ensure_coordinator_connection(
    profile: DstackCoordinatorProfile,
) -> CoordinatorConnectionReport:
    return _CONNECTION_MANAGER.ensure(profile)


def close_coordinator_connections() -> None:
    _CONNECTION_MANAGER.close()


atexit.register(close_coordinator_connections)
