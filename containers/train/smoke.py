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


def smoke_breakout_render() -> None:
    import numpy as np

    from env_breakoutatari2600_turbo_native import BreakoutVecEnv

    env = BreakoutVecEnv(
        "Breakout-Atari2600-v0",
        num_envs=1,
        render_mode="rgb_array",
    )
    try:
        observation, _infos = env.reset(seed=0)
        frame = env.render()
    finally:
        env.close()

    if observation.shape != (1, 4, 84, 84) or observation.dtype != np.uint8:
        raise RuntimeError("Breakout policy observation contract mismatch")
    if frame.shape != (210, 160, 3) or frame.dtype != np.uint8:
        raise RuntimeError("Breakout RGB render contract mismatch")
    colors = {tuple(color) for color in np.unique(frame.reshape(-1, 3), axis=0)}
    expected = {
        (0, 0, 0),
        (136, 136, 136),
        (200, 72, 72),
        (192, 104, 56),
        (176, 120, 48),
        (160, 160, 40),
        (72, 160, 72),
        (64, 72, 200),
        (64, 152, 128),
    }
    if colors != expected:
        raise RuntimeError(f"Breakout canonical Stella palette mismatch: {sorted(colors)!r}")
    print("breakout_render=canonical-stella-rgb")


def main() -> None:
    root = Path(os.environ.get("GRADLAB_PROJECT_ROOT", "/root/gradlab"))
    print("gradlab_container_smoke=ok")
    print(f"python={platform.python_version()}")
    print(f"platform={platform.platform()}")
    for package in (
        "gradlab",
        "env-breakoutatari2600-turbo-native",
        "env-stableretro-turbo",
        "env-vizdoom-turbo",
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
    smoke_breakout_render()

    game = os.environ.get("RETRO_GAME")
    if game:
        import stable_retro as retro

        states = list(retro.data.list_states(game))[:12]
        print(f"retro_game={game}")
        print(f"retro_states_preview={states}")


if __name__ == "__main__":
    main()
