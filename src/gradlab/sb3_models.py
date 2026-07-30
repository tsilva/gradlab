from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from gradlab.policy_registry import (
    SB3_ALGORITHMS,
    Sb3AlgorithmId,
    resolve_policy_algorithm,
)
from gradlab.trusted_inputs import ApprovedModelInput


def resolve_sb3_algorithm(metadata: Mapping[str, Any] | None) -> Sb3AlgorithmId:
    return cast(
        Sb3AlgorithmId,
        resolve_policy_algorithm(metadata, allowed=SB3_ALGORITHMS),
    )


def load_sb3_model(
    model_input: ApprovedModelInput,
    *,
    device: str,
    env: Any | None = None,
    tensorboard_log: str | None = None,
    algorithm_id: Sb3AlgorithmId,
):
    if not isinstance(model_input, ApprovedModelInput):
        raise TypeError("load_sb3_model requires an ApprovedModelInput")
    model_input.verify()
    path = model_input.model_path
    if algorithm_id == "a2c":
        from stable_baselines3 import A2C

        model_class = A2C
    else:
        from stable_baselines3 import PPO

        model_class = PPO
    kwargs: dict[str, Any] = {"device": device}
    if env is not None:
        kwargs["env"] = env
    if tensorboard_log is not None:
        kwargs["tensorboard_log"] = tensorboard_log
    return model_class.load(str(path), **kwargs)
