from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


LOCAL_RUN_RECEIPT = "local-run.json"


@dataclass(frozen=True)
class RecipeSource:
    reference: str
    goal_path: Path
    recipe_path: Path
    repository_root: Path


def experiments_root() -> Path:
    """Return the source-checkout or wheel-bundled experiments tree."""

    source_root = Path(__file__).resolve().parents[2] / "experiments"
    if source_root.is_dir():
        return source_root
    packaged_root = Path(__file__).resolve().parent / "experiments"
    if packaged_root.is_dir():
        return packaged_root
    raise FileNotFoundError(
        "gradlab's built-in experiments are unavailable; reinstall gradlab from a complete wheel"
    )


def _repository_root_for(path: Path) -> Path:
    for parent in path.resolve().parents:
        if parent.name == "experiments":
            return parent.parent
    raise ValueError(f"recipe {path} is not under an experiments tree")


def _local_recipe_source(path: Path) -> RecipeSource:
    recipe_path = path.expanduser().resolve()
    if recipe_path.parent.name != "recipes":
        raise ValueError(
            f"recipe file {recipe_path} must live directly under a goal's recipes directory"
        )
    goal_path = recipe_path.parent.parent / "_goal.yaml"
    if not goal_path.is_file():
        raise FileNotFoundError(f"recipe goal file not found: {goal_path}")
    repository_root = _repository_root_for(recipe_path)
    try:
        reference = recipe_path.relative_to(repository_root).as_posix()
    except ValueError:
        reference = str(recipe_path)
    return RecipeSource(reference, goal_path, recipe_path, repository_root)


def resolve_recipe_source(reference: str | Path) -> RecipeSource:
    """Resolve a local recipe YAML or a bundled ``goal/path/recipe`` reference."""

    text = str(reference).strip()
    if not text:
        raise ValueError("recipe reference must be non-empty")
    local_path = Path(text).expanduser()
    if local_path.is_file():
        return _local_recipe_source(local_path)
    if local_path.suffix.lower() in {".yaml", ".yml"} or local_path.is_absolute():
        raise FileNotFoundError(f"recipe file not found: {local_path}")

    normalized = text.removeprefix("builtin:").strip("/")
    pure = PurePosixPath(normalized)
    if (
        len(pure.parts) < 2
        or any(part in {"", ".", ".."} for part in pure.parts)
        or "\\" in normalized
    ):
        raise ValueError(
            "built-in recipe references use <goal-path>/<recipe>, for example "
            "VizdoomBasic-v1/ppo"
        )
    goal_parts = pure.parts[:-1]
    recipe_name = pure.parts[-1]
    root = experiments_root()
    goals_root = (root / "goals").resolve()
    goal_path = (goals_root.joinpath(*goal_parts) / "_goal.yaml").resolve()
    recipe_dir = (goal_path.parent / "recipes").resolve()
    if not goal_path.is_relative_to(goals_root) or not recipe_dir.is_relative_to(goals_root):
        raise ValueError(f"unsafe built-in recipe reference: {reference}")
    recipe_path = recipe_dir / f"{recipe_name}.yaml"
    if not goal_path.is_file() or not recipe_path.is_file():
        raise FileNotFoundError(
            f"built-in recipe {normalized!r} was not found; expected {recipe_path}"
        )
    return RecipeSource(
        normalized,
        goal_path,
        recipe_path,
        root.parent,
    )


def recipe_identity(materialized_recipe: dict[str, Any]) -> tuple[str, str]:
    goal = materialized_recipe.get("goal")
    if not isinstance(goal, dict):
        raise ValueError("materialized recipe has no goal")
    goal_id = str(goal.get("goal_id") or "").strip()
    recipe_id = str(materialized_recipe.get("recipe_id") or "").strip()
    if not goal_id or not recipe_id:
        raise ValueError("materialized recipe has no goal_id or recipe_id")
    return goal_id, recipe_id


def latest_local_recipe_model(
    runs_dir: Path,
    *,
    goal_id: str,
    recipe_id: str,
) -> Path:
    """Find the newest completed local run for one materialized recipe identity."""

    root = runs_dir.expanduser()
    candidates: list[tuple[str, int, Path]] = []
    if root.is_dir():
        for receipt_path in root.rglob(LOCAL_RUN_RECEIPT):
            try:
                if receipt_path.stat().st_size > 1024 * 1024:
                    continue
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if not isinstance(receipt, dict) or receipt.get("status") != "completed":
                continue
            if (
                str(receipt.get("goal_id") or "") != goal_id
                or str(receipt.get("recipe_id") or "") != recipe_id
                or receipt.get("model") != "final_model.zip"
            ):
                continue
            model_path = receipt_path.parent / "final_model.zip"
            if not model_path.is_file():
                continue
            candidates.append(
                (
                    str(receipt.get("completed_at") or ""),
                    receipt_path.stat().st_mtime_ns,
                    model_path,
                )
            )
    if not candidates:
        raise FileNotFoundError(
            f"no completed local model for {goal_id}/{recipe_id} under {root}; "
            "train it first with `gradlab train <recipe>`"
        )
    return max(candidates)[2]
