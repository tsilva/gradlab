from __future__ import annotations

import random

import numpy as np
import torch
from stable_baselines3.common.utils import set_random_seed


def snapshot(label: str) -> None:
    print(label)
    print("python", [round(random.random(), 12) for _ in range(5)])
    print("numpy", [round(float(x), 12) for x in np.random.random(5)])
    print("torch", [round(float(x), 12) for x in torch.rand(5)])


def main() -> None:
    set_random_seed(23)
    snapshot("baseline_after_seed")

    set_random_seed(23)
    import wandb

    run = wandb.init(
        project="rlab",
        name="rng-probe-offline",
        mode="offline",
        sync_tensorboard=True,
    )
    snapshot("after_wandb_offline_init")
    run.finish()

    set_random_seed(23)
    run = wandb.init(
        project="rlab",
        name="rng-probe-disabled",
        mode="disabled",
        sync_tensorboard=True,
    )
    snapshot("after_wandb_disabled_init")
    run.finish()


if __name__ == "__main__":
    main()
