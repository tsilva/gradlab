"""Explicit authentication policy for private MLflow writer routes."""

from __future__ import annotations

import ipaddress
import os
import socket
from collections.abc import Mapping
from urllib.parse import urlparse

from gradlab.operator_credentials import OperatorConfigurationError


AUTH_MODE_ENV = "MLFLOW_AUTH_MODE"
CREDENTIAL_ENV = ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")
SERVICE_ENV = ("MLFLOW_TRACKING_URI", "MLFLOW_OPERATOR_PROFILE", "MLFLOW_ALLOWED_FLEETS")
AUTH_MODES = frozenset({"basic", "network"})
_PRIVATE_NETWORKS = tuple(
    map(
        ipaddress.ip_network,
        (
            "10.0.0.0/8",
            "172.16.0.0/12",
            "192.168.0.0/16",
            "100.64.0.0/10",
            "fc00::/7",
        ),
    )
)


def mlflow_auth_mode(tracking: Mapping | None = None) -> str:
    mode = (tracking or {}).get("auth_mode", os.environ.get(AUTH_MODE_ENV, "basic"))
    if not isinstance(mode, str) or mode not in AUTH_MODES:
        raise OperatorConfigurationError("MLFLOW_AUTH_MODE must be basic or network")
    return mode


def mlflow_service_environment(tracking: Mapping | None = None) -> tuple[str, ...]:
    if mlflow_auth_mode(tracking) == "network":
        return (*SERVICE_ENV, AUTH_MODE_ENV)
    return (*SERVICE_ENV, *CREDENTIAL_ENV)


def validate_mlflow_access(uri: str, tracking: Mapping | None = None) -> None:
    """Require application credentials or an explicitly selected private network."""
    mode = mlflow_auth_mode(tracking)
    configured = os.environ.get(AUTH_MODE_ENV)
    if configured is not None and configured != mode:
        raise OperatorConfigurationError("MLflow authentication mode differs from the frozen Run")
    if mode == "basic":
        if not all(str(os.environ.get(name) or "").strip() for name in CREDENTIAL_ENV):
            raise OperatorConfigurationError(
                "basic MLflow authentication requires username and password"
            )
        return
    if any(str(os.environ.get(name) or "").strip() for name in CREDENTIAL_ENV):
        raise OperatorConfigurationError(
            "network MLflow authentication must not supply basic credentials"
        )
    parsed = urlparse(uri)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise OperatorConfigurationError("network MLflow authentication requires private HTTPS")
    try:
        addresses = {ipaddress.ip_address(parsed.hostname)}
    except ValueError:
        try:
            addresses = {
                ipaddress.ip_address(row[4][0])
                for row in socket.getaddrinfo(
                    parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM
                )
            }
        except OSError as exc:
            raise OperatorConfigurationError("private MLflow hostname cannot be resolved") from exc
    if not addresses or any(
        address.is_loopback
        or address.is_unspecified
        or address.is_multicast
        or not any(address in network for network in _PRIVATE_NETWORKS)
        for address in addresses
    ):
        raise OperatorConfigurationError(
            "network MLflow authentication requires a non-loopback private address"
        )
