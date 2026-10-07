from __future__ import annotations

import subprocess
from collections.abc import Sequence
from unittest import mock

import pytest

from gradlab.coordinator_connection import (
    CoordinatorConnectionError,
    CoordinatorConnectionManager,
)
from gradlab.operator_credentials import (
    DstackCoordinatorProfile,
    KeychainReference,
    SshTunnelProfile,
)


class FakeProcess:
    def __init__(self, *, returncode: int | None = None, stderr: str = ""):
        self.returncode = returncode
        self.stderr = stderr
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        del timeout
        return "", self.stderr

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        assert self.returncode is not None
        return self.returncode


def _profile(*, tunnel: bool = True) -> DstackCoordinatorProfile:
    return DstackCoordinatorProfile(
        coordinator_id="b3",
        project="main",
        server_url="http://127.0.0.1:3000",
        token=KeychainReference("dstack-b3", "operator"),
        ssh_tunnel=(
            SshTunnelProfile(
                destinations=("operator@gpu.local", "operator@gpu.fallback"),
                remote_host="127.0.0.1",
                remote_port=3000,
            )
            if tunnel
            else None
        ),
    )


def test_direct_coordinator_needs_no_ssh_transport() -> None:
    probe = mock.Mock()
    process_factory = mock.Mock()
    manager = CoordinatorConnectionManager(
        tcp_probe=probe,
        process_factory=process_factory,
    )

    report = manager.ensure(_profile(tunnel=False))

    assert report.mode == "direct"
    assert report.owned is False
    probe.assert_not_called()
    process_factory.assert_not_called()


def test_existing_tunnel_is_reused_and_never_owned() -> None:
    process_factory = mock.Mock()
    manager = CoordinatorConnectionManager(
        tcp_probe=lambda _host, _port, _timeout: True,
        process_factory=process_factory,
    )

    report = manager.ensure(_profile())
    manager.close()

    assert report.mode == "existing"
    assert report.owned is False
    process_factory.assert_not_called()


def test_managed_tunnel_uses_safe_ssh_options_and_is_closed() -> None:
    probes = iter((False, True))
    process = FakeProcess()
    calls: list[Sequence[str]] = []

    def start(command: Sequence[str], **_kwargs):
        calls.append(command)
        return process

    manager = CoordinatorConnectionManager(
        executable_lookup=lambda _name: "/usr/bin/ssh",
        tcp_probe=lambda _host, _port, _timeout: next(probes),
        process_factory=start,
    )

    report = manager.ensure(_profile())
    manager.close()

    assert report.mode == "managed-ssh-tunnel"
    assert report.owned is True
    assert calls == [
        [
            "/usr/bin/ssh",
            "-T",
            "-N",
            "-o",
            "BatchMode=yes",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ConnectTimeout=5",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
            "-L",
            "127.0.0.1:3000:127.0.0.1:3000",
            "--",
            "operator@gpu.local",
        ]
    ]
    assert process.terminated is True


def test_tunnel_falls_back_to_next_destination() -> None:
    first = FakeProcess(returncode=255, stderr="primary unavailable")
    second = FakeProcess()
    processes = iter((first, second))
    probes = iter((False, False, False, True))
    calls: list[Sequence[str]] = []

    def start(command: Sequence[str], **_kwargs):
        calls.append(command)
        return next(processes)

    manager = CoordinatorConnectionManager(
        executable_lookup=lambda _name: "/usr/bin/ssh",
        tcp_probe=lambda _host, _port, _timeout: next(probes),
        process_factory=start,
    )

    report = manager.ensure(_profile())
    manager.close()

    assert report.mode == "managed-ssh-tunnel"
    assert calls[0][-1] == "operator@gpu.local"
    assert calls[1][-1] == "operator@gpu.fallback"
    assert second.terminated is True


def test_tunnel_failure_reports_every_destination() -> None:
    processes = iter(
        (
            FakeProcess(returncode=255, stderr="primary unavailable"),
            FakeProcess(returncode=255, stderr="fallback unavailable"),
        )
    )
    manager = CoordinatorConnectionManager(
        executable_lookup=lambda _name: "/usr/bin/ssh",
        tcp_probe=lambda _host, _port, _timeout: False,
        process_factory=lambda _command, **_kwargs: next(processes),
    )

    with pytest.raises(CoordinatorConnectionError) as captured:
        manager.ensure(_profile())

    assert "operator@gpu.local: primary unavailable" in str(captured.value)
    assert "operator@gpu.fallback: fallback unavailable" in str(captured.value)


def test_auto_tunnel_rejects_non_loopback_endpoint() -> None:
    profile = _profile()
    profile = DstackCoordinatorProfile(
        coordinator_id=profile.coordinator_id,
        project=profile.project,
        server_url="https://dstack.example.com:443",
        token=profile.token,
        ssh_tunnel=profile.ssh_tunnel,
    )
    manager = CoordinatorConnectionManager()

    with pytest.raises(CoordinatorConnectionError, match="requires a loopback"):
        manager.ensure(profile)


def test_tunnel_process_timeout_is_force_killed() -> None:
    process = FakeProcess()

    def wait(*, timeout: float | None = None) -> int:
        del timeout
        if not process.killed:
            raise subprocess.TimeoutExpired(["ssh"], 3)
        return -9

    process.wait = wait  # type: ignore[method-assign]
    manager = CoordinatorConnectionManager(
        executable_lookup=lambda _name: "/usr/bin/ssh",
        tcp_probe=mock.Mock(side_effect=(False, True)),
        process_factory=lambda _command, **_kwargs: process,
    )

    manager.ensure(_profile())
    manager.close()

    assert process.terminated is True
    assert process.killed is True
