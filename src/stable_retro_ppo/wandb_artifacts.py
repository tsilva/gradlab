from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from stable_retro_ppo.artifacts import model_metadata_path
from stable_retro_ppo.wandb_utils import load_wandb_env


def safe_artifact_stem(value: str, fallback: str = "artifact") -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or fallback


def model_artifact_ref(
    *,
    project: str,
    run_name: str,
    kind: str,
    version: str = "latest",
) -> str:
    if not run_name:
        raise ValueError("run_name is required")
    return f"{project}/{run_name}-{kind}:{version}"


def artifact_download_dir(root: Path, ref: str) -> Path:
    return root / safe_artifact_stem(ref.replace("/", "_").replace(":", "_"))


def download_model_artifact(ref: str, root: Path) -> Path:
    load_wandb_env()

    import wandb

    root.mkdir(parents=True, exist_ok=True)
    artifact = wandb.Api().artifact(ref, type="model")
    path = Path(artifact.download(root=str(root)))
    model_path = model_zip_from_download(path)
    write_downloaded_artifact_metadata(model_path, artifact)
    return model_path


def metadata_from_wandb_artifact(artifact, model_path: Path) -> dict:
    metadata = getattr(artifact, "metadata", None)
    if isinstance(metadata, dict) and metadata.get("training_metadata"):
        return dict(metadata)
    if isinstance(metadata, dict):
        print(
            f"warning: W&B artifact for {model_path.name} has no training_metadata",
            file=sys.stderr,
        )
        return dict(metadata)
    return {}


def write_downloaded_artifact_metadata(model_path: Path, artifact) -> Path | None:
    metadata = metadata_from_wandb_artifact(artifact, model_path)
    if not metadata:
        return None
    path = model_metadata_path(model_path)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def model_zip_from_download(path: Path) -> Path:
    zip_files = sorted(path.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No .zip model file found in downloaded artifact: {path}")
    if len(zip_files) > 1:
        print(f"Multiple model files found; using {zip_files[0]}", file=sys.stderr)
    return zip_files[0]
