from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import yaml


YAML_EXTENSIONS = {".yaml", ".yml"}


@dataclass(frozen=True)
class ComposedDocument:
    document: dict[str, Any]
    sources: tuple[Path, ...]


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    cfg = OmegaConf.merge(OmegaConf.create(dict(base)), OmegaConf.create(dict(override)))
    return _plain_dict(cfg)


def load_config_document(path: Path, *, default: Any = None) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in YAML_EXTENSIONS:
        loaded = yaml.safe_load(text)
    else:
        loaded = json.loads(text)
    return default if loaded is None else loaded


def load_mapping_document(path: Path, *, label: str | None = None) -> dict[str, Any]:
    payload = load_config_document(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label or path} must contain a JSON/YAML object")
    return dict(payload)


def _plain_dict(value: Any) -> dict[str, Any]:
    payload = OmegaConf.to_container(value, resolve=False)
    if not isinstance(payload, Mapping):
        raise ValueError("composed config must contain a JSON/YAML object")
    return dict(payload)


def _default_entry_to_path(entry: Any) -> str | None:
    if isinstance(entry, str):
        if entry == "_self_" or entry.startswith("override ") or entry.startswith("optional "):
            return None
        return entry.split("@", 1)[0]
    if isinstance(entry, Mapping) and len(entry) == 1:
        key, value = next(iter(entry.items()))
        if key is None or key == "_self_":
            return None
        key = str(key)
        if key.startswith("override ") or key.startswith("optional "):
            return None
        if value in (None, "null"):
            return None
        if isinstance(value, str):
            return f"{key.split('@', 1)[0]}/{value.split('@', 1)[0]}"
    return None


def _resolve_default_path(default_path: str, *, base_dir: Path, config_root: Path) -> Path:
    is_absolute_default = default_path.startswith("/")
    path = default_path.lstrip("/")
    if path.endswith(".yaml") or path.endswith(".yml") or path.endswith(".json"):
        candidate = Path(path)
    else:
        candidate = Path(f"{path}.yaml")
    if not candidate.is_absolute():
        candidate = (config_root if is_absolute_default else base_dir) / candidate
    return candidate.resolve()


def _collect_hydra_sources(
    path: Path,
    *,
    config_root: Path,
    stack: tuple[Path, ...] = (),
) -> tuple[Path, ...]:
    resolved_path = path.resolve()
    if resolved_path in stack:
        chain = " -> ".join(str(item) for item in (*stack, resolved_path))
        raise ValueError(f"cyclic Hydra defaults chain: {chain}")

    document = load_mapping_document(resolved_path, label=str(path))
    sources: list[Path] = []
    for entry in document.get("defaults", []) or []:
        default_path = _default_entry_to_path(entry)
        if default_path is None:
            continue
        source = _resolve_default_path(
            default_path,
            base_dir=resolved_path.parent,
            config_root=config_root,
        )
        if source.is_file():
            sources.extend(
                _collect_hydra_sources(
                    source,
                    config_root=config_root,
                    stack=(*stack, resolved_path),
                )
            )

    sources.append(resolved_path)
    return tuple(sources)


def load_composed_mapping(
    path: Path,
    *,
    stack: tuple[Path, ...] = (),
    cycle_label: str = "config",
) -> ComposedDocument:
    resolved_path = path.resolve()
    if stack:
        raise ValueError("load_composed_mapping no longer accepts recursive stack callers")
    if resolved_path.suffix.lower() not in YAML_EXTENSIONS:
        return ComposedDocument(
            document=load_mapping_document(resolved_path, label=str(path)),
            sources=(resolved_path,),
        )
    try:
        sources = _collect_hydra_sources(resolved_path, config_root=resolved_path.parent)
        with initialize_config_dir(version_base=None, config_dir=str(resolved_path.parent)):
            cfg = compose(config_name=resolved_path.stem)
    except Exception as exc:
        raise ValueError(f"failed to compose {cycle_label} config {path}: {exc}") from exc
    return ComposedDocument(document=_plain_dict(cfg), sources=sources)
