"""JSON secrets through the Infisical CLI, with redacted failures and pipe writes."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from gradlab.secret_reference import InfisicalReference


class SecretStoreError(RuntimeError):
    """Safe storage errors containing no credential values."""


class InfisicalJsonStore:
    def __init__(self, settings: dict[str, Any]):
        try:
            if not isinstance(settings, dict) or set(settings) - {
                "project_id", "environment", "path", "instance", "secrets", "executable", "login_profile"
            }:
                raise TypeError
            if not isinstance(settings.get("secrets"), dict) or not settings["secrets"]:
                raise TypeError
            self.references = {}
            for role, name in settings["secrets"].items():
                if not isinstance(role, str) or not role.strip():
                    raise ValueError
                self.references[role] = InfisicalReference(
                    project_id=settings["project_id"], secret_name=name,
                    environment=settings["environment"], path=settings["path"],
                    instance=settings["instance"],
                )
            for field in ("executable", "login_profile"):
                if field in settings and (
                    not isinstance(settings[field], str) or not settings[field].strip()
                    or any(ord(c) < 32 for c in settings[field])
                ):
                    raise ValueError
        except KeyError, TypeError, ValueError, AttributeError:
            raise SecretStoreError("Invalid Infisical credential configuration.") from None
        self.settings = {**settings, "secrets": dict(settings["secrets"])}

    def has(self, role: str) -> bool:
        return role in self.references

    def _run(self, role: str, value: str | None = None) -> str:
        if not self.has(role):
            raise SecretStoreError("This credential has no configured Infisical secret binding.")
        settings = self.settings
        reference = self.references[role]
        name = reference.secret_name
        args = [settings.get("executable", "infisical"), "secrets"]
        if value is None:
            args += ["get", name, "--plain", "--expand=false", "--include-imports=false"]
        else:
            args += ["set", f"{name}=@/dev/stdin"]
        args += [
            "--projectId",
            reference.project_id,
            "--env",
            reference.environment,
            "--path",
            reference.path,
            "--domain",
            reference.instance,
            "--silent",
            "--telemetry=false",
        ]
        if settings.get("login_profile"):
            args += ["--profile", settings["login_profile"]]
        try:
            result = subprocess.run(
                args,
                **({"input": value} if value is not None else {"stdin": subprocess.DEVNULL}),
                text=True,
                capture_output=True,
                timeout=60,
                check=False,
            )
        except OSError, subprocess.TimeoutExpired:
            raise SecretStoreError(
                "Infisical CLI unavailable or timed out; check its login and configuration."
            ) from None
        if result.returncode:
            raise SecretStoreError(
                "Infisical credential access failed; check login, secret binding and permissions."
            ) from None
        return result.stdout

    def read(self, role: str) -> dict[str, Any]:
        raw = self._run(role)
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise TypeError
        except TypeError, ValueError:
            raise SecretStoreError("Infisical credential must contain a JSON object.") from None
        return payload

    def write(self, role: str, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise SecretStoreError("Credential must contain a JSON object.")
        self._run(role, json.dumps(payload))


def load_store(path: Path) -> InfisicalJsonStore | None:
    """A present configuration selects Infisical; failures never select local files."""
    if path.is_symlink():
        raise SecretStoreError("Infisical configuration must not be a symbolic link.")
    if not path.exists():
        return None
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(settings, dict):
            raise TypeError
    except OSError, TypeError, ValueError:
        raise SecretStoreError("Infisical configuration is unreadable or invalid.") from None
    return InfisicalJsonStore(settings)
