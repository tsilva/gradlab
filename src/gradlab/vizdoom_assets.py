from __future__ import annotations

import re
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

from gradlab.file_utils import file_sha256


VIZDOOM_IWAD_BINDING_SCHEMA_VERSION = 1
DEFAULT_LOCAL_VIZDOOM_IWAD = Path("~/roms/vizdoom/doom2.wad")

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "path",
        "filename",
        "size_bytes",
        "sha256",
    }
)


def _display_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    home = Path.home().resolve()
    try:
        relative = resolved.relative_to(home)
    except ValueError:
        return str(resolved)
    return f"~/{relative.as_posix()}"


def _require_iwad(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"ViZDoom IWAD path does not exist: {resolved}")
    with resolved.open("rb") as handle:
        if handle.read(4) != b"IWAD":
            raise ValueError(f"ViZDoom game asset is not an IWAD: {resolved}")
    return resolved


def vizdoom_iwad_binding(path: Path) -> dict[str, Any]:
    resolved = _require_iwad(path)
    return {
        "schema_version": VIZDOOM_IWAD_BINDING_SCHEMA_VERSION,
        "path": _display_path(resolved),
        "filename": resolved.name,
        "size_bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def validate_vizdoom_iwad_binding(
    value: Mapping[str, Any],
    *,
    verify_file: bool = False,
) -> dict[str, Any]:
    binding = dict(value)
    unknown = sorted(set(binding) - _BINDING_FIELDS)
    if unknown:
        raise ValueError(f"unknown ViZDoom IWAD binding field(s): {', '.join(unknown)}")
    missing = sorted(_BINDING_FIELDS - set(binding))
    if missing:
        raise ValueError(f"ViZDoom IWAD binding missing field(s): {', '.join(missing)}")
    if int(binding["schema_version"]) != VIZDOOM_IWAD_BINDING_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported ViZDoom IWAD binding schema_version: {binding['schema_version']!r}"
        )
    path = str(binding["path"] or "").strip()
    filename = str(binding["filename"] or "").strip()
    if not path:
        raise ValueError("ViZDoom IWAD binding path must be non-empty")
    if not filename or Path(filename).name != filename:
        raise ValueError("ViZDoom IWAD binding filename must be a basename")
    sha256 = str(binding["sha256"] or "").strip().lower()
    if not _SHA256_RE.fullmatch(sha256):
        raise ValueError("ViZDoom IWAD binding sha256 must be lowercase hexadecimal")
    size_bytes = int(binding["size_bytes"])
    if size_bytes <= 0:
        raise ValueError("ViZDoom IWAD binding size_bytes must be positive")
    normalized = {
        "schema_version": VIZDOOM_IWAD_BINDING_SCHEMA_VERSION,
        "path": path,
        "filename": filename,
        "size_bytes": size_bytes,
        "sha256": sha256,
    }
    if verify_file:
        resolved = _require_iwad(Path(path))
        if resolved.name != filename:
            raise ValueError("ViZDoom IWAD binding filename does not match its path")
        if resolved.stat().st_size != size_bytes:
            raise ValueError(f"ViZDoom IWAD size mismatch for {resolved}")
        if file_sha256(resolved) != sha256:
            raise ValueError(f"ViZDoom IWAD sha256 mismatch for {resolved}")
    return normalized


def portable_vizdoom_iwad_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    binding = validate_vizdoom_iwad_binding(value)
    return {
        "schema_version": binding["schema_version"],
        "filename": binding["filename"],
        "size_bytes": binding["size_bytes"],
        "sha256": binding["sha256"],
    }


def resolve_vizdoom_iwad_path(value: str | Mapping[str, Any] | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        binding = validate_vizdoom_iwad_binding(value, verify_file=True)
        return str(Path(binding["path"]).expanduser().resolve())
    return str(value)


def apply_optional_local_vizdoom_iwad(
    document: MutableMapping[str, Any],
    *,
    requested_path: Path | None = None,
    default_path: Path | None = None,
) -> bool:
    train_config = document.get("train_config")
    if not isinstance(train_config, MutableMapping):
        return False
    if str(train_config.get("env_provider") or "") != "vizdoom-turbo":
        return False
    env_args = train_config.get("env_args")
    if not isinstance(env_args, Mapping):
        return False
    if env_args.get("rom_path") is not None and requested_path is None:
        return False
    candidate = requested_path or default_path or DEFAULT_LOCAL_VIZDOOM_IWAD
    if requested_path is None and not candidate.expanduser().is_file():
        return False
    updated_env_args = dict(env_args)
    updated_env_args["rom_path"] = vizdoom_iwad_binding(candidate)
    train_config["env_args"] = updated_env_args
    return True
