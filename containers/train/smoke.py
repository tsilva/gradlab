#!/usr/bin/env python
from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
from pathlib import Path


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def smoke_lunar_lander() -> None:
    import gymnasium as gym

    env = gym.make("LunarLander-v3")
    try:
        observation, _info = env.reset(seed=0)
        if getattr(observation, "shape", None) != (8,):
            raise RuntimeError(
                f"LunarLander-v3 reset returned unexpected observation shape "
                f"{getattr(observation, 'shape', None)!r}"
            )
    finally:
        env.close()
    print("lunar_lander_reset=ok")


def main() -> None:
    root = Path(os.environ.get("GRADLAB_PROJECT_ROOT", "/root/gradlab"))
    print("gradlab_container_smoke=ok")
    print(f"python={platform.python_version()}")
    print(f"platform={platform.platform()}")
    for package in (
        "gradlab",
        "stable-retro-turbo",
        "vizdoom-turbo",
        "stable-baselines3",
        "torch",
        "wandb",
    ):
        print(f"package/{package}={package_version(package)}")

    lock_path = root / "uv.lock"
    if lock_path.is_file():
        print(f"uv_lock_sha256={file_sha256(lock_path)}")

    try:
        import torch

        print(f"torch_cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"torch_cuda_device={torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"torch_probe_error={type(exc).__name__}: {exc}")

    smoke_lunar_lander()

    game = os.environ.get("RETRO_GAME")
    if game:
        import stable_retro as retro

        states = list(retro.data.list_states(game))[:12]
        print(f"retro_game={game}")
        print(f"retro_states_preview={states}")


if __name__ == "__main__":
    main()
