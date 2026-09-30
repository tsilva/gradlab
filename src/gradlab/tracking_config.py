"""Resolve the non-scientific metrics destination of a Run."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from gradlab.config_loader import load_mapping_document


DEFAULT_TRACKING = {"backend": "mlflow", "delivery": "online"}
BACKENDS = frozenset({"wandb", "mlflow"})
DELIVERY_MODES = frozenset({"online", "local_only"})


def validate_tracking(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    unknown = sorted(set(value) - set(DEFAULT_TRACKING))
    if unknown:
        raise ValueError(f"{label} has unknown field(s): {', '.join(map(str, unknown))}")
    result: dict[str, str] = {}
    for key, choices in (("backend", BACKENDS), ("delivery", DELIVERY_MODES)):
        if key not in value:
            continue
        selected = value[key]
        if not isinstance(selected, str) or selected not in choices:
            raise ValueError(f"{label}.{key} must be one of {', '.join(sorted(choices))}")
        result[key] = selected
    return result


def _layer(path: Path) -> dict[str, str]:
    document = load_mapping_document(path, label=f"tracking source {path}")
    if "tracking" not in document:
        return {}
    return validate_tracking(document["tracking"], label=f"{path}.tracking")


def resolve_tracking(
    *,
    experiments_root: Path,
    goal_sources: Sequence[Path],
    recipe_sources: Sequence[Path],
    launch_tracking: object = None,
    overridden_keys: Sequence[str] = (),
) -> dict[str, Any]:
    values = dict(DEFAULT_TRACKING)
    sources = {key: "built-in default" for key in values}
    layers: list[tuple[str, dict[str, str]]] = []
    def source_name(path: Path) -> str:
        return path.resolve().relative_to(experiments_root.resolve().parent).as_posix()

    project = experiments_root / "tracking.yaml"
    if project.is_file():
        document = load_mapping_document(project, label=f"tracking source {project}")
        layers.append((source_name(project), validate_tracking(document, label=str(project))))
    for path in goal_sources:
        layers.append((source_name(path), _layer(path)))
    for path in recipe_sources:
        layers.append((source_name(path), _layer(path)))
    for label, layer in layers:
        for key, value in layer.items():
            values[key] = value
            sources[key] = label
    if launch_tracking is not None:
        layer = validate_tracking(launch_tracking, label="launch tracking")
        for key in overridden_keys:
            if key in layer:
                values[key] = layer[key]
                sources[key] = "launch override"
    return {**values, "sources": sources}
