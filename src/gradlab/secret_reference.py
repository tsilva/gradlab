"""Read a single explicitly scoped Infisical secret; suppress provider output."""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from collections.abc import Mapping
from uuid import UUID
from urllib.parse import urlsplit


@dataclass(frozen=True)
class InfisicalReference:
    project_id: str
    secret_name: str
    environment: str
    path: str
    instance: str

    def __post_init__(self) -> None:
        try:
            for value in (self.project_id, self.secret_name, self.environment,
                          self.path, self.instance):
                if not isinstance(value, str) or not value or any(c.isspace() or ord(c) < 32 for c in value):
                    raise ValueError
            UUID(self.project_id)
            if not re.fullmatch(r"[A-Z][A-Z0-9_]*", self.secret_name):
                raise ValueError
            if not re.fullmatch(r"[a-z][a-z0-9-]*", self.environment):
                raise ValueError
            if not re.fullmatch(r"/(?:[a-zA-Z0-9_-]+/)*[a-zA-Z0-9_-]*", self.path):
                raise ValueError
            url = urlsplit(self.instance)
            if (url.scheme != "https" or not url.hostname or url.username is not None
                    or url.password is not None or "?" in self.instance or "#" in self.instance
                    or "\\" in self.instance or url.port == 0):
                raise ValueError
        except (ValueError, TypeError, AttributeError):
            raise ValueError("Invalid Infisical reference") from None

    @classmethod
    def parse(cls, value: object) -> InfisicalReference:
        if not isinstance(value, Mapping) or set(value) - {
            "project_id", "secret_name", "environment", "path", "instance"
        }:
            raise ValueError("Invalid Infisical reference")
        try:
            return cls(**value)
        except (ValueError, TypeError, AttributeError):
            raise ValueError("Invalid Infisical reference") from None


def read_secret(reference: InfisicalReference) -> str:
    try:
        result = subprocess.run(
            [
                "infisical", "secrets", "get", reference.secret_name, "--plain",
                "--projectId", reference.project_id, "--env", reference.environment,
                "--path", reference.path, "--domain", reference.instance,
                "--expand=false", "--include-imports=false", "--silent", "--telemetry=false",
            ],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise ValueError("Infisical unavailable or timed out; provider output suppressed") from None
    if result.returncode:
        raise ValueError("Infisical access failed; check login and binding")
    value = result.stdout.rstrip("\n")
    if not value or "\0" in value:
        raise ValueError("Infisical returned an empty or invalid credential")
    return value
