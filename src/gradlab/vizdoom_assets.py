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
REQUIRED_VIZDOOM_IWAD_FILENAME = "doom2.wad"
REQUIRED_VIZDOOM_IWAD_SIZE_BYTES = 14_604_584
REQUIRED_VIZDOOM_IWAD_SHA256 = "10d67824b11025ddd9198e8cfc87ca335ee6e2d3e63af4180fa9b8a471893255"
VIZDOOM_IWAD_CACHE_PREFIX = Path("vizdoom-iwad/sha256")
VIZDOOM_IWAD_OBJECT_PREFIX = "vizdoom-iwad/v1"

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
_OPTIONAL_BINDING_FIELDS = frozenset({"object_uri"})


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
    unknown = sorted(set(binding) - _BINDING_FIELDS - _OPTIONAL_BINDING_FIELDS)
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
    if "object_uri" in binding:
        object_uri = str(binding["object_uri"] or "").strip()
        if not object_uri:
            raise ValueError("ViZDoom IWAD binding object_uri must be non-empty")
        normalized["object_uri"] = object_uri
    if verify_file:
        verify_vizdoom_iwad_file(Path(path), normalized)
    return normalized


def required_vizdoom_iwad_binding(
    path: Path | None = None,
) -> dict[str, Any]:
    binding = vizdoom_iwad_binding(path or DEFAULT_LOCAL_VIZDOOM_IWAD)
    expected = {
        "filename": REQUIRED_VIZDOOM_IWAD_FILENAME,
        "size_bytes": REQUIRED_VIZDOOM_IWAD_SIZE_BYTES,
        "sha256": REQUIRED_VIZDOOM_IWAD_SHA256,
    }
    actual = {key: binding[key] for key in expected}
    if actual != expected:
        raise ValueError(
            "ViZDoom requires the pinned Doom II IWAD "
            f"{REQUIRED_VIZDOOM_IWAD_FILENAME} "
            f"(size={REQUIRED_VIZDOOM_IWAD_SIZE_BYTES}, "
            f"sha256={REQUIRED_VIZDOOM_IWAD_SHA256}); got "
            f"filename={actual['filename']}, size={actual['size_bytes']}, "
            f"sha256={actual['sha256']}"
        )
    return binding


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
    runtime_binding = {
        "schema_version": normalized["schema_version"],
        "path": normalized["path"],
        "filename": normalized["filename"],
        "size_bytes": normalized["size_bytes"],
        "sha256": normalized["sha256"],
    }
    train_config = document.get("train_config")
    if not isinstance(train_config, MutableMapping):
        raise ValueError("ViZDoom IWAD binding requires a materialized train_config")
    provider_id = str(train_config.get("env_provider") or "")
    if provider_id not in {"gradoom", "vizdoom-turbo"}:
        raise ValueError("Doom IWAD binding requires env_provider='gradoom' or 'vizdoom-turbo'")
    env_args = train_config.get("env_args")
    if not isinstance(env_args, Mapping):
        raise ValueError("ViZDoom IWAD binding requires train_config.env_args")
    train_config["env_args"] = {**env_args, "rom_path": runtime_binding}
    evaluation = train_config.get("checkpoint_eval_environment")
    if isinstance(evaluation, MutableMapping):
        evaluation_args = evaluation.get("env_args")
        if not isinstance(evaluation_args, Mapping):
            raise ValueError("ViZDoom checkpoint evaluation requires env_args")
        evaluation["env_args"] = {**evaluation_args, "rom_path": runtime_binding}


def bind_required_local_vizdoom_iwad(
    document: MutableMapping[str, Any],
    *,
    requested_path: Path | None = None,
    default_path: Path | None = None,
) -> bool:
    train_config = document.get("train_config")
    if not isinstance(train_config, MutableMapping):
        return False
    if str(train_config.get("env_provider") or "") not in {"gradoom", "vizdoom-turbo"}:
        return False
    env_args = train_config.get("env_args")
    if not isinstance(env_args, Mapping):
        return False
    configured = env_args.get("rom_path")
    if isinstance(configured, Mapping) and requested_path is None:
        normalized = validate_vizdoom_iwad_binding(configured)
        resolved = resolve_vizdoom_iwad_path(normalized)
        assert resolved is not None
        required = required_vizdoom_iwad_binding(Path(resolved))
        bind_vizdoom_iwad_to_document(document, required)
        return True
    candidate = requested_path or default_path or DEFAULT_LOCAL_VIZDOOM_IWAD
    bind_vizdoom_iwad_to_document(document, required_vizdoom_iwad_binding(candidate))
    return True
