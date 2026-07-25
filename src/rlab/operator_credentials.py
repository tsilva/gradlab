from __future__ import annotations

import os
import re
import stat
import subprocess
import sys
import tomllib
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OPERATOR_CONFIG_ENV = "RLAB_OPERATOR_CONFIG"
OPERATOR_CONFIG_SCHEMA_VERSION = 1
ENV_NAME_PATTERN = re.compile(r"[A-Z][A-Z0-9_]*")
PROTECTED_ENV_NAMES = frozenset(
    {
        "DSTACK_TOKEN",
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "WANDB_API_KEY",
        "RLAB_CONTROL_R2_ACCESS_KEY_ID",
        "RLAB_CONTROL_R2_SECRET_ACCESS_KEY",
        "RLAB_EVAL_R2_ACCESS_KEY_ID",
        "RLAB_EVAL_R2_SECRET_ACCESS_KEY",
        "RLAB_MODELS_R2_ACCESS_KEY_ID",
        "RLAB_MODELS_R2_SECRET_ACCESS_KEY",
    }
)


class OperatorConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class KeychainReference:
    service: str
    account: str | None = None


@dataclass(frozen=True)
class OperatorEnvironmentReport:
    config_path: Path
    config_present: bool
    loaded_sources: Mapping[str, str]
    unavailable_sources: Mapping[str, str]

    def source_for(self, name: str, environment: Mapping[str, str]) -> str:
        if name in self.loaded_sources:
            return str(self.loaded_sources[name])
        if str(environment.get(name) or "").strip():
            return "process-environment"
        return "missing"


def default_operator_config_path(
    environment: Mapping[str, str] | None = None,
) -> Path:
    values = os.environ if environment is None else environment
    explicit = str(values.get(OPERATOR_CONFIG_ENV) or "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    xdg_root = str(values.get("XDG_CONFIG_HOME") or "").strip()
    root = Path(xdg_root).expanduser() if xdg_root else Path.home() / ".config"
    return (root / "rlab" / "operator.toml").resolve()


def _validate_environment_name(name: Any, *, label: str) -> str:
    value = str(name or "").strip()
    if ENV_NAME_PATTERN.fullmatch(value) is None:
        raise OperatorConfigurationError(f"{label} must be an uppercase environment name")
    return value


def _read_toml(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise OperatorConfigurationError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise OperatorConfigurationError(f"{label} must contain a TOML mapping: {path}")
    return value


def _require_not_writable_by_others(path: Path, *, label: str) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o022:
        raise OperatorConfigurationError(
            f"{label} must not be group- or world-writable: {path} "
            f"(mode {mode:04o})"
        )


def _require_private_file(path: Path, *, label: str) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise OperatorConfigurationError(
            f"{label} contains credentials and must use mode 0600: {path} "
            f"(mode {mode:04o})"
        )


def _keychain_lookup(reference: KeychainReference) -> str | None:
    if sys.platform != "darwin":
        raise OperatorConfigurationError(
            "macOS Keychain references require Darwin; provide protected values "
            "through the process environment on this platform"
        )
    command = [
        "/usr/bin/security",
        "find-generic-password",
        "-s",
        reference.service,
    ]
    if reference.account is not None:
        command.extend(["-a", reference.account])
    command.append("-w")
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode:
        return None
    value = result.stdout.strip()
    return value or None


def _keychain_references(document: Mapping[str, Any]) -> dict[str, KeychainReference]:
    raw = document.get("keychain") or {}
    if not isinstance(raw, Mapping):
        raise OperatorConfigurationError("operator config [keychain] must be a mapping")
    references: dict[str, KeychainReference] = {}
    for raw_name, raw_reference in raw.items():
        name = _validate_environment_name(
            raw_name,
            label="operator config keychain entry",
        )
        if name not in PROTECTED_ENV_NAMES:
            raise OperatorConfigurationError(
                f"operator config keychain entry {name} is not a recognized protected value"
            )
        if not isinstance(raw_reference, Mapping):
            raise OperatorConfigurationError(
                f"operator config keychain.{name} must be a mapping"
            )
        service = str(raw_reference.get("service") or "").strip()
        account = str(raw_reference.get("account") or "").strip() or None
        if not service:
            raise OperatorConfigurationError(
                f"operator config keychain.{name}.service is required"
            )
        extra = set(raw_reference) - {"service", "account"}
        if extra:
            raise OperatorConfigurationError(
                f"operator config keychain.{name} has unknown keys: "
                + ", ".join(sorted(str(key) for key in extra))
            )
        references[name] = KeychainReference(service=service, account=account)
    return references


def _plain_environment(document: Mapping[str, Any]) -> dict[str, str]:
    raw = document.get("environment") or {}
    if not isinstance(raw, Mapping):
        raise OperatorConfigurationError("operator config [environment] must be a mapping")
    result: dict[str, str] = {}
    for raw_name, raw_value in raw.items():
        name = _validate_environment_name(
            raw_name,
            label="operator config environment entry",
        )
        if name in PROTECTED_ENV_NAMES:
            raise OperatorConfigurationError(
                f"protected value {name} must use [keychain], not plaintext [environment]"
            )
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise OperatorConfigurationError(
                f"operator config environment.{name} must be a non-empty string"
            )
        result[name] = raw_value.strip()
    return result


def _active_modal_profile(
    document: Mapping[str, Any],
    *,
    configured_profile: str | None,
    path: Path,
) -> Mapping[str, Any]:
    if configured_profile is not None:
        value = document.get(configured_profile)
        if not isinstance(value, Mapping):
            raise OperatorConfigurationError(
                f"Modal profile {configured_profile!r} is missing from {path}"
            )
        return value
    active = [
        (str(name), value)
        for name, value in document.items()
        if isinstance(value, Mapping) and value.get("active") is True
    ]
    if len(active) != 1:
        raise OperatorConfigurationError(
            f"Modal config must have exactly one active profile or operator.toml "
            f"must select modal.profile: {path}"
        )
    return active[0][1]


def _load_modal_environment(
    document: Mapping[str, Any],
    *,
    environment: MutableMapping[str, str],
    loaded_sources: dict[str, str],
) -> None:
    raw = document.get("modal")
    if raw is None:
        return
    if not isinstance(raw, Mapping):
        raise OperatorConfigurationError("operator config [modal] must be a mapping")
    extra = set(raw) - {"path", "profile"}
    if extra:
        raise OperatorConfigurationError(
            "operator config [modal] has unknown keys: "
            + ", ".join(sorted(str(key) for key in extra))
        )
    path = Path(str(raw.get("path") or "~/.modal.toml")).expanduser().resolve()
    if not path.is_file():
        raise OperatorConfigurationError(f"Modal credential config does not exist: {path}")
    _require_private_file(path, label="Modal credential config")
    modal_document = _read_toml(path, label="Modal credential config")
    profile_name = str(raw.get("profile") or "").strip() or None
    profile = _active_modal_profile(
        modal_document,
        configured_profile=profile_name,
        path=path,
    )
    for environment_name, field_name in (
        ("MODAL_TOKEN_ID", "token_id"),
        ("MODAL_TOKEN_SECRET", "token_secret"),
    ):
        if str(environment.get(environment_name) or "").strip():
            continue
        value = str(profile.get(field_name) or "").strip()
        if not value:
            raise OperatorConfigurationError(
                f"Modal credential profile is missing {field_name}: {path}"
            )
        environment[environment_name] = value
        loaded_sources[environment_name] = "modal-profile"


def load_operator_environment(
    *,
    environment: MutableMapping[str, str] | None = None,
    config_path: Path | None = None,
    keychain_lookup: Callable[[KeychainReference], str | None] | None = None,
) -> OperatorEnvironmentReport:
    values = os.environ if environment is None else environment
    path = (
        default_operator_config_path(values)
        if config_path is None
        else Path(config_path).expanduser().resolve()
    )
    if not path.is_file():
        return OperatorEnvironmentReport(
            config_path=path,
            config_present=False,
            loaded_sources={},
            unavailable_sources={},
        )
    _require_not_writable_by_others(path, label="operator config")
    document = _read_toml(path, label="operator config")
    allowed_top_level = {"schema_version", "environment", "keychain", "modal"}
    extra = set(document) - allowed_top_level
    if extra:
        raise OperatorConfigurationError(
            "operator config has unknown top-level keys: "
            + ", ".join(sorted(str(key) for key in extra))
        )
    if document.get("schema_version") != OPERATOR_CONFIG_SCHEMA_VERSION:
        raise OperatorConfigurationError(
            f"operator config schema_version must be {OPERATOR_CONFIG_SCHEMA_VERSION}: {path}"
        )
    loaded_sources: dict[str, str] = {}
    unavailable_sources: dict[str, str] = {}
    for name, value in _plain_environment(document).items():
        if str(values.get(name) or "").strip():
            continue
        values[name] = value
        loaded_sources[name] = "operator-config"
    lookup = _keychain_lookup if keychain_lookup is None else keychain_lookup
    for name, reference in _keychain_references(document).items():
        if str(values.get(name) or "").strip():
            continue
        value = lookup(reference)
        if value is None:
            unavailable_sources[name] = "macos-keychain"
            continue
        values[name] = value
        loaded_sources[name] = "macos-keychain"
    _load_modal_environment(
        document,
        environment=values,
        loaded_sources=loaded_sources,
    )
    return OperatorEnvironmentReport(
        config_path=path,
        config_present=True,
        loaded_sources=loaded_sources,
        unavailable_sources=unavailable_sources,
    )


def reject_protected_dotenv(path: str | Path) -> None:
    env_path = Path(path)
    if not env_path.is_file():
        return
    protected: set[str] = set()
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name = line.split("=", 1)[0].strip()
        if name in PROTECTED_ENV_NAMES:
            protected.add(name)
    if protected:
        raise OperatorConfigurationError(
            f"protected credentials must not be stored in {env_path}; move these names "
            "to process environment or operator.toml [keychain]: "
            + ", ".join(sorted(protected))
        )
