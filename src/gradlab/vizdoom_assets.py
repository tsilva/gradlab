from __future__ import annotations

import os
import re
import shutil
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

from gradlab.file_utils import file_sha256


VIZDOOM_IWAD_BINDING_SCHEMA_VERSION = 1
DEFAULT_LOCAL_VIZDOOM_IWAD = Path("~/roms/vizdoom/doom2.wad")
VIZDOOM_IWAD_CACHE_PREFIX = Path("vizdoom-iwad/sha256")

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


def verify_vizdoom_iwad_file(path: Path, binding: Mapping[str, Any]) -> Path:
    normalized = validate_vizdoom_iwad_binding(binding)
    resolved = _require_iwad(path)
    if resolved.name != normalized["filename"]:
        raise ValueError("ViZDoom IWAD binding filename does not match its path")
    if resolved.stat().st_size != normalized["size_bytes"]:
        raise ValueError(f"ViZDoom IWAD size mismatch for {resolved}")
    if file_sha256(resolved) != normalized["sha256"]:
        raise ValueError(f"ViZDoom IWAD sha256 mismatch for {resolved}")
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
        verify_vizdoom_iwad_file(Path(path), normalized)
    return normalized


def portable_vizdoom_iwad_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    binding = validate_vizdoom_iwad_binding(value)
    return {
        "schema_version": binding["schema_version"],
        "filename": binding["filename"],
        "size_bytes": binding["size_bytes"],
        "sha256": binding["sha256"],
    }


def vizdoom_iwad_cache_path(cache_root: Path, value: Mapping[str, Any]) -> Path:
    binding = validate_vizdoom_iwad_binding(value)
    return (
        cache_root.expanduser()
        / VIZDOOM_IWAD_CACHE_PREFIX
        / binding["sha256"]
        / binding["filename"]
    )


def install_vizdoom_iwad_file(
    source: Path,
    binding: Mapping[str, Any],
    *,
    cache_root: Path,
) -> Path:
    normalized = validate_vizdoom_iwad_binding(binding)
    verified = verify_vizdoom_iwad_file(source, normalized)
    destination = vizdoom_iwad_cache_path(cache_root, normalized)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(verified, destination)
    return verify_vizdoom_iwad_file(destination, normalized)


def resolve_vizdoom_iwad_path(value: str | Mapping[str, Any] | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        binding = validate_vizdoom_iwad_binding(value)
        configured = Path(binding["path"]).expanduser()
        cache_root = Path(
            os.environ.get("GRADLAB_ROM_CACHE_DIR") or "~/.cache/gradlab/roms"
        ).expanduser()
        cached = vizdoom_iwad_cache_path(cache_root, binding)
        verification_errors: list[str] = []
        for candidate in (configured, cached):
            if candidate.is_file():
                try:
                    return str(verify_vizdoom_iwad_file(candidate, binding))
                except ValueError as exc:
                    verification_errors.append(str(exc))
        raise FileNotFoundError(
            "verified ViZDoom IWAD is unavailable at either "
            f"{configured} or {cached}"
            + (f" ({'; '.join(verification_errors)})" if verification_errors else "")
        )
    return str(value)


def bind_vizdoom_iwad_to_document(
    document: MutableMapping[str, Any],
    binding: Mapping[str, Any],
) -> None:
    normalized = validate_vizdoom_iwad_binding(binding)
    train_config = document.get("train_config")
    if not isinstance(train_config, MutableMapping):
        raise ValueError("ViZDoom IWAD binding requires a materialized train_config")
    if str(train_config.get("env_provider") or "") != "vizdoom-turbo":
        raise ValueError("ViZDoom IWAD binding requires env_provider='vizdoom-turbo'")
    env_args = train_config.get("env_args")
    if not isinstance(env_args, Mapping):
        raise ValueError("ViZDoom IWAD binding requires train_config.env_args")
    train_config["env_args"] = {**env_args, "rom_path": normalized}
    evaluation = train_config.get("checkpoint_eval_environment")
    if isinstance(evaluation, MutableMapping):
        evaluation_args = evaluation.get("env_args")
        if not isinstance(evaluation_args, Mapping):
            raise ValueError("ViZDoom checkpoint evaluation requires env_args")
        evaluation["env_args"] = {**evaluation_args, "rom_path": normalized}


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
    bind_vizdoom_iwad_to_document(document, vizdoom_iwad_binding(candidate))
    return True
