from __future__ import annotations

import os
import re
import stat
import subprocess
import sys
import tomllib
from collections.abc import Callable, Collection, Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gradlab.dstack_backend import DstackResources


OPERATOR_CONFIG_ENV = "GRADLAB_OPERATOR_CONFIG"
OPERATOR_CONFIG_SCHEMA_VERSION = 3
ENV_NAME_PATTERN = re.compile(r"[A-Z][A-Z0-9_]*")
PROTECTED_ENV_NAMES = frozenset(
    {
        "DSTACK_TOKEN",
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "WANDB_API_KEY",
        "GRADLAB_CONTROL_R2_ACCESS_KEY_ID",
        "GRADLAB_CONTROL_R2_SECRET_ACCESS_KEY",
        "GRADLAB_EVAL_R2_ACCESS_KEY_ID",
        "GRADLAB_EVAL_R2_SECRET_ACCESS_KEY",
        "GRADLAB_MODELS_R2_ACCESS_KEY_ID",
        "GRADLAB_MODELS_R2_SECRET_ACCESS_KEY",
    }
)
MODAL_ENV_NAMES = frozenset({"MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"})


class OperatorConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class KeychainReference:
    service: str
    account: str | None = None


@dataclass(frozen=True)
class DstackCoordinatorProfile:
    coordinator_id: str
    project: str
    server_url: str
    token: KeychainReference


@dataclass(frozen=True)
class DstackFleetProfile:
    name: str
    coordinator_id: str
    resources: DstackResources


@dataclass(frozen=True)
class DstackOperatorConfig:
    default_coordinator: str
    default_fleet: str
    coordinators: Mapping[str, DstackCoordinatorProfile]
    fleets: Mapping[str, DstackFleetProfile]

    def coordinator(self, coordinator_id: str | None = None) -> DstackCoordinatorProfile:
        selected = str(coordinator_id or self.default_coordinator).strip()
        try:
            return self.coordinators[selected]
        except KeyError as exc:
            raise OperatorConfigurationError(
                f"unknown dstack coordinator profile: {selected!r}"
            ) from exc

    def fleet(self, name: str | None = None) -> DstackFleetProfile:
        selected = str(name or self.default_fleet).strip()
        try:
            return self.fleets[selected]
        except KeyError as exc:
            raise OperatorConfigurationError(f"unknown dstack fleet profile: {selected!r}") from exc


@dataclass(frozen=True)
class OperatorEnvironmentReport:
    config_path: Path
    config_present: bool
    loaded_sources: Mapping[str, str]
    unavailable_sources: Mapping[str, str]
    dstack: DstackOperatorConfig | None = None

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
    return (root / "gradlab" / "operator.toml").resolve()


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
            f"{label} must not be group- or world-writable: {path} (mode {mode:04o})"
        )


def _require_private_file(path: Path, *, label: str) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise OperatorConfigurationError(
            f"{label} contains credentials and must use mode 0600: {path} (mode {mode:04o})"
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


def _keychain_references(
    document: Mapping[str, Any],
    *,
    selected_names: Collection[str] | None = None,
) -> dict[str, KeychainReference]:
    raw = document.get("keychain") or {}
    if not isinstance(raw, Mapping):
        raise OperatorConfigurationError("operator config [keychain] must be a mapping")
    references: dict[str, KeychainReference] = {}
    for raw_name, raw_reference in raw.items():
        candidate_name = str(raw_name or "").strip()
        if selected_names is not None and candidate_name not in selected_names:
            continue
        name = _validate_environment_name(
            raw_name,
            label="operator config keychain entry",
        )
        if name not in PROTECTED_ENV_NAMES:
            raise OperatorConfigurationError(
                f"operator config keychain entry {name} is not a recognized protected value"
            )
        if not isinstance(raw_reference, Mapping):
            raise OperatorConfigurationError(f"operator config keychain.{name} must be a mapping")
        service = str(raw_reference.get("service") or "").strip()
        account = str(raw_reference.get("account") or "").strip() or None
        if not service:
            raise OperatorConfigurationError(f"operator config keychain.{name}.service is required")
        extra = set(raw_reference) - {"service", "account"}
        if extra:
            raise OperatorConfigurationError(
                f"operator config keychain.{name} has unknown keys: "
                + ", ".join(sorted(str(key) for key in extra))
            )
        references[name] = KeychainReference(service=service, account=account)
    return references


def _plain_environment(
    document: Mapping[str, Any],
    *,
    selected_names: Collection[str] | None = None,
) -> dict[str, str]:
    raw = document.get("environment") or {}
    if not isinstance(raw, Mapping):
        raise OperatorConfigurationError("operator config [environment] must be a mapping")
    result: dict[str, str] = {}
    for raw_name, raw_value in raw.items():
        candidate_name = str(raw_name or "").strip()
        if selected_names is not None and candidate_name not in selected_names:
            continue
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


def _profile_name(value: object, *, label: str) -> str:
    name = str(value or "").strip()
    if not name or any(character in name for character in "\r\n\0"):
        raise OperatorConfigurationError(f"{label} must be non-empty single-line text")
    return name


def _keychain_reference(value: object, *, label: str) -> KeychainReference:
    if not isinstance(value, Mapping):
        raise OperatorConfigurationError(f"{label} must be a mapping")
    extra = set(value) - {"service", "account"}
    if extra:
        raise OperatorConfigurationError(
            f"{label} has unknown keys: " + ", ".join(sorted(str(key) for key in extra))
        )
    service = str(value.get("service") or "").strip()
    account = str(value.get("account") or "").strip() or None
    if not service:
        raise OperatorConfigurationError(f"{label}.service is required")
    return KeychainReference(service=service, account=account)


def _dstack_operator_config(document: Mapping[str, Any]) -> DstackOperatorConfig | None:
    raw_dstack = document.get("dstack") or {}
    if not isinstance(raw_dstack, Mapping):
        raise OperatorConfigurationError("operator config [dstack] must be a mapping")
    if not raw_dstack:
        return None
    extra = set(raw_dstack) - {
        "default_coordinator",
        "default_fleet",
        "coordinators",
        "fleets",
    }
    if extra:
        raise OperatorConfigurationError(
            "operator config [dstack] has unknown keys: "
            + ", ".join(sorted(str(key) for key in extra))
        )
    default_coordinator = _profile_name(
        raw_dstack.get("default_coordinator"),
        label="operator config dstack.default_coordinator",
    )
    default_fleet = _profile_name(
        raw_dstack.get("default_fleet"),
        label="operator config dstack.default_fleet",
    )
    raw_coordinators = raw_dstack.get("coordinators") or {}
    if not isinstance(raw_coordinators, Mapping):
        raise OperatorConfigurationError("operator config dstack.coordinators must be a mapping")
    coordinators: dict[str, DstackCoordinatorProfile] = {}
    for raw_id, raw_profile in raw_coordinators.items():
        coordinator_id = _profile_name(
            raw_id,
            label="operator config dstack coordinator names",
        )
        if not isinstance(raw_profile, Mapping):
            raise OperatorConfigurationError(
                f"operator config dstack.coordinators.{coordinator_id} must be a mapping"
            )
        extra_profile = set(raw_profile) - {"project", "server_url", "token"}
        if extra_profile:
            raise OperatorConfigurationError(
                f"operator config dstack.coordinators.{coordinator_id} has unknown keys: "
                + ", ".join(sorted(str(key) for key in extra_profile))
            )
        project = _profile_name(
            raw_profile.get("project"),
            label=f"operator config dstack.coordinators.{coordinator_id}.project",
        )
        server_url = _profile_name(
            raw_profile.get("server_url"),
            label=f"operator config dstack.coordinators.{coordinator_id}.server_url",
        )
        if not re.fullmatch(r"https?://[^\s]+", server_url):
            raise OperatorConfigurationError(
                f"operator config dstack.coordinators.{coordinator_id}.server_url "
                "must be an HTTP(S) URL"
            )
        coordinators[coordinator_id] = DstackCoordinatorProfile(
            coordinator_id=coordinator_id,
            project=project,
            server_url=server_url.rstrip("/"),
            token=_keychain_reference(
                raw_profile.get("token"),
                label=f"operator config dstack.coordinators.{coordinator_id}.token",
            ),
        )
    if default_coordinator not in coordinators:
        raise OperatorConfigurationError(
            "operator config dstack.default_coordinator must name a configured coordinator"
        )
    raw_fleets = raw_dstack.get("fleets") or {}
    if not isinstance(raw_fleets, Mapping):
        raise OperatorConfigurationError("operator config dstack.fleets must be a mapping")
    fleets: dict[str, DstackFleetProfile] = {}
    for raw_name, raw_profile in raw_fleets.items():
        name = _profile_name(raw_name, label="operator config dstack fleet names")
        if not isinstance(raw_profile, Mapping):
            raise OperatorConfigurationError(
                f"operator config dstack.fleets.{name} must be a mapping"
            )
        extra_profile = set(raw_profile) - {"coordinator", "cpu", "memory", "gpu", "disk"}
        if extra_profile:
            raise OperatorConfigurationError(
                f"operator config dstack.fleets.{name} has unknown keys: "
                + ", ".join(sorted(str(key) for key in extra_profile))
            )
        coordinator_id = _profile_name(
            raw_profile.get("coordinator"),
            label=f"operator config dstack.fleets.{name}.coordinator",
        )
        if coordinator_id not in coordinators:
            raise OperatorConfigurationError(
                f"operator config dstack.fleets.{name}.coordinator names an unknown profile"
            )
        try:
            resources = DstackResources.from_manifest(
                {key: value for key, value in raw_profile.items() if key != "coordinator"}
            )
        except (TypeError, ValueError) as exc:
            raise OperatorConfigurationError(
                f"operator config dstack.fleets.{name} is invalid: {exc}"
            ) from exc
        fleets[name] = DstackFleetProfile(
            name=name,
            coordinator_id=coordinator_id,
            resources=resources,
        )
    if default_fleet not in fleets:
        raise OperatorConfigurationError(
            "operator config dstack.default_fleet must name a configured fleet"
        )
    if fleets[default_fleet].coordinator_id != default_coordinator:
        raise OperatorConfigurationError(
            "operator config default_fleet must belong to default_coordinator"
        )
    return DstackOperatorConfig(
        default_coordinator=default_coordinator,
        default_fleet=default_fleet,
        coordinators=coordinators,
        fleets=fleets,
    )


def resolve_dstack_token(
    profile: DstackCoordinatorProfile,
    *,
    environment: Mapping[str, str] | None = None,
    keychain_lookup: Callable[[KeychainReference], str | None] | None = None,
) -> tuple[str, str]:
    values = os.environ if environment is None else environment
    process_value = str(values.get("DSTACK_TOKEN") or "").strip()
    if process_value:
        return process_value, "process-environment"
    lookup = _keychain_lookup if keychain_lookup is None else keychain_lookup
    value = lookup(profile.token)
    if value is None:
        raise OperatorConfigurationError(
            f"dstack token is unavailable for coordinator {profile.coordinator_id!r}"
        )
    return value, "macos-keychain"


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
    requested_names: Collection[str] | None = None,
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
        if requested_names is not None and environment_name not in requested_names:
            continue
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
    requested_names: Collection[str] | None = None,
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
            dstack=None,
        )
    _require_not_writable_by_others(path, label="operator config")
    document = _read_toml(path, label="operator config")
    allowed_top_level = {"schema_version", "environment", "keychain", "modal", "dstack"}
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
    plain_dstack_names = {
        "DSTACK_PROJECT",
        "DSTACK_SERVER_URL",
        "GRADLAB_LOCAL_FLEET",
    } & set(document.get("environment") or {})
    if plain_dstack_names:
        raise OperatorConfigurationError(
            "schema-v3 dstack routing belongs under [dstack], not [environment]: "
            + ", ".join(sorted(plain_dstack_names))
        )
    if "DSTACK_TOKEN" in set(document.get("keychain") or {}):
        raise OperatorConfigurationError(
            "schema-v3 dstack tokens belong under [dstack.coordinators.<id>.token]"
        )
    selected_names = (
        None
        if requested_names is None
        else frozenset(
            _validate_environment_name(
                name,
                label="requested operator environment name",
            )
            for name in requested_names
        )
    )
    loaded_sources: dict[str, str] = {}
    unavailable_sources: dict[str, str] = {}
    for name, value in _plain_environment(
        document,
        selected_names=selected_names,
    ).items():
        if str(values.get(name) or "").strip():
            continue
        values[name] = value
        loaded_sources[name] = "operator-config"
    lookup = _keychain_lookup if keychain_lookup is None else keychain_lookup
    for name, reference in _keychain_references(
        document,
        selected_names=selected_names,
    ).items():
        if str(values.get(name) or "").strip():
            continue
        value = lookup(reference)
        if value is None:
            unavailable_sources[name] = "macos-keychain"
            continue
        values[name] = value
        loaded_sources[name] = "macos-keychain"
    if selected_names is None or selected_names & MODAL_ENV_NAMES:
        _load_modal_environment(
            document,
            environment=values,
            loaded_sources=loaded_sources,
            requested_names=selected_names,
        )
    return OperatorEnvironmentReport(
        config_path=path,
        config_present=True,
        loaded_sources=loaded_sources,
        unavailable_sources=unavailable_sources,
        dstack=_dstack_operator_config(document),
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
            "to process environment or operator.toml [keychain]: " + ", ".join(sorted(protected))
        )
