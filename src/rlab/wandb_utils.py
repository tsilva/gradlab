from __future__ import annotations

from pathlib import Path

from rlab.dotenv import load_env_file

DEFAULT_WANDB_ENTITY = "tsilva"
DEFAULT_WANDB_PROJECT = "SuperMarioBros-NES"
DEFAULT_WANDB_PROJECT_PATH = f"{DEFAULT_WANDB_ENTITY}/{DEFAULT_WANDB_PROJECT}"

WANDB_ENV_PREFIXES = ("WANDB_", "AWS_")
WANDB_ARTIFACT_ENV_KEYS = {
    "CHECKPOINT_BUCKET_URI",
}


def load_wandb_env(dotenv_path: str | Path = ".env") -> None:
    """Load W&B and artifact storage env vars without adding a dotenv dependency."""
    load_env_file(
        dotenv_path,
        key_filter=lambda key: key.startswith(WANDB_ENV_PREFIXES)
        or key in WANDB_ARTIFACT_ENV_KEYS,
    )
