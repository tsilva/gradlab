from __future__ import annotations

import fcntl
import json
import os
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from gradlab.file_utils import atomic_write_json
from gradlab.local_paths import default_gradlab_config_dir


class CredentialSecurityError(RuntimeError):
    pass


def _lstat(path: Path) -> os.stat_result:
    try:
        return path.lstat()
    except FileNotFoundError as exc:
        raise CredentialSecurityError(f"credential path does not exist: {path}") from exc


def _require_owned_directory(path: Path, *, create: bool = False) -> Path:
    if create:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    metadata = _lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise CredentialSecurityError(f"credential directory must be a real directory: {path}")
    if metadata.st_uid != os.getuid():
        raise CredentialSecurityError(f"credential directory is not owned by this user: {path}")
    os.chmod(path, 0o700)
    verified = _lstat(path)
    if stat.S_IMODE(verified.st_mode) != 0o700 or verified.st_uid != os.getuid():
        raise CredentialSecurityError(f"credential directory could not be secured: {path}")
    return path.resolve(strict=True)


def _require_owned_file(
    path: Path,
    *,
    root: Path,
    create: bool = False,
) -> Path:
    if create and not path.exists():
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        os.close(descriptor)
    metadata = _lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise CredentialSecurityError(f"credential file must be a regular non-symlink: {path}")
    if metadata.st_uid != os.getuid():
        raise CredentialSecurityError(f"credential file is not owned by this user: {path}")
    resolved = path.resolve(strict=True)
    if resolved.parent != root:
        raise CredentialSecurityError(f"credential file escapes its private directory: {path}")
    os.chmod(path, 0o600)
    verified = _lstat(path)
    if (
        not stat.S_ISREG(verified.st_mode)
        or verified.st_uid != os.getuid()
        or stat.S_IMODE(verified.st_mode) != 0o600
    ):
        raise CredentialSecurityError(f"credential file could not be secured: {path}")
    return resolved


@dataclass(frozen=True)
class YouTubeCredentialPaths:
    root: Path
    client: Path
    token: Path
    lock: Path


def youtube_credential_paths(
    *,
    environment: Mapping[str, str] | None = None,
) -> YouTubeCredentialPaths:
    config_root = default_gradlab_config_dir(environment)
    resolved_root = _require_owned_directory(config_root, create=True)
    client = _require_owned_file(
        config_root / "youtube_client_secret.json",
        root=resolved_root,
    )
    token_path = config_root / "youtube_token.json"
    token = (
        _require_owned_file(token_path, root=resolved_root)
        if token_path.exists()
        else token_path
    )
    lock = _require_owned_file(
        config_root / "youtube_token.lock",
        root=resolved_root,
        create=True,
    )
    return YouTubeCredentialPaths(resolved_root, client, token, lock)


@contextmanager
def credential_lock(path: Path) -> Iterator[None]:
    descriptor = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
            raise CredentialSecurityError("credential lock changed identity")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def load_private_json(path: Path, *, root: Path) -> dict[str, Any]:
    secured = _require_owned_file(path, root=root)
    descriptor = os.open(secured, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
            raise CredentialSecurityError("credential file changed identity")
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            descriptor = -1
            value = json.load(stream)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(value, Mapping):
        raise CredentialSecurityError(f"credential file must contain a JSON object: {path}")
    return dict(value)


def save_private_json(path: Path, value: Mapping[str, Any], *, root: Path) -> None:
    if path.parent.resolve() != root:
        raise CredentialSecurityError("credential destination escapes its private directory")
    atomic_write_json(path, dict(value))
    _require_owned_file(path, root=root)


@dataclass(frozen=True)
class HuggingFaceCredential:
    token: str
    source: str


def _secure_hf_file(path: Path, *, root: Path) -> None:
    if path.exists() or path.is_symlink():
        _require_owned_file(path, root=root)


def _read_private_text(path: Path, *, root: Path) -> str:
    secured = _require_owned_file(path, root=root)
    descriptor = os.open(secured, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
            raise CredentialSecurityError("credential file changed identity")
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            descriptor = -1
            return stream.read().strip()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def resolve_huggingface_credential(
    environment: Mapping[str, str] | None = None,
) -> HuggingFaceCredential:
    values = os.environ if environment is None else environment
    environment_token = str(values.get("HF_TOKEN") or "").strip()
    if environment_token:
        return HuggingFaceCredential(environment_token, "environment")

    configured_home = str(values.get("HF_HOME") or "").strip()
    if configured_home:
        hf_home = Path(configured_home).expanduser()
    else:
        xdg_cache = str(values.get("XDG_CACHE_HOME") or "").strip()
        cache_root = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
        hf_home = cache_root / "huggingface"
    if not hf_home.exists():
        raise CredentialSecurityError("Hugging Face login is missing; run `hf auth login`")
    try:
        resolved_home = _require_owned_directory(hf_home)
    except CredentialSecurityError as exc:
        raise CredentialSecurityError(
            f"Hugging Face auth home is not private; configure a private HF_HOME: {exc}"
        ) from exc
    configured_token_path = str(values.get("HF_TOKEN_PATH") or "").strip()
    token_path = (
        Path(configured_token_path).expanduser()
        if configured_token_path
        else hf_home / "token"
    )
    configured_stored_path = str(values.get("HF_STORED_TOKENS_PATH") or "").strip()
    stored_tokens_path = (
        Path(configured_stored_path).expanduser()
        if configured_stored_path
        else hf_home / "stored_tokens"
    )
    if token_path.parent.resolve() != resolved_home or stored_tokens_path.parent.resolve() != resolved_home:
        raise CredentialSecurityError("Hugging Face token paths must stay inside private HF_HOME")
    _secure_hf_file(token_path, root=resolved_home)
    _secure_hf_file(stored_tokens_path, root=resolved_home)
    token = _read_private_text(token_path, root=resolved_home) if token_path.exists() else ""
    if not token:
        raise CredentialSecurityError("Hugging Face login is missing; run `hf auth login`")
    return HuggingFaceCredential(token, "file")


__all__ = [
    "CredentialSecurityError",
    "HuggingFaceCredential",
    "YouTubeCredentialPaths",
    "credential_lock",
    "load_private_json",
    "resolve_huggingface_credential",
    "save_private_json",
    "youtube_credential_paths",
]
