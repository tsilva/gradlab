from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast


PolicyAlgorithmId: TypeAlias = Literal[
    "ppo",
    "a2c",
    "jerk",
    "dqn",
    "recurrent-ppo",
]
RuntimePolicyAlgorithmId: TypeAlias = Literal["ppo", "a2c", "jerk"]
Sb3AlgorithmId: TypeAlias = Literal["ppo", "a2c"]


@dataclass(frozen=True)
class TrainingBackendSpec:
    module_name: str
    algorithm_id: RuntimePolicyAlgorithmId


TRAINING_BACKEND_SPECS: dict[str, TrainingBackendSpec] = {
    "gradlab.go-explore": TrainingBackendSpec("gradlab.training.go_explore", "jerk"),
    "gradlab.jerk": TrainingBackendSpec("gradlab.training.jerk", "jerk"),
    "sb3.a2c": TrainingBackendSpec("gradlab.training.sb3", "a2c"),
    "sb3.ppo": TrainingBackendSpec("gradlab.training.sb3", "ppo"),
}
MODEL_CLASS_ALGORITHMS: dict[str, PolicyAlgorithmId] = {
    "gradlab.jerk.JerkPolicy": "jerk",
    "gradlab.task_advantage.PerTaskAdvantagePPO": "ppo",
    "sb3_contrib.ppo_recurrent.ppo_recurrent.RecurrentPPO": "recurrent-ppo",
    "stable_baselines3.a2c.a2c.A2C": "a2c",
    "stable_baselines3.dqn.dqn.DQN": "dqn",
    "stable_baselines3.ppo.ppo.PPO": "ppo",
}
ALGORITHM_MODEL_CLASSES: dict[str, frozenset[str]] = {
    algorithm_id: frozenset(
        model_class
        for model_class, registered_algorithm in MODEL_CLASS_ALGORITHMS.items()
        if registered_algorithm == algorithm_id
    )
    for algorithm_id in frozenset(MODEL_CLASS_ALGORITHMS.values())
}

RUNTIME_POLICY_ALGORITHMS = frozenset[PolicyAlgorithmId]({"ppo", "a2c", "jerk"})
SB3_ALGORITHMS = frozenset[PolicyAlgorithmId]({"ppo", "a2c"})


def resolve_policy_algorithm(
    metadata: Mapping[str, Any] | None,
    *,
    allowed: frozenset[PolicyAlgorithmId] = RUNTIME_POLICY_ALGORITHMS,
) -> PolicyAlgorithmId:
    metadata = metadata or {}
    resolved: set[PolicyAlgorithmId] = set()

    backend_id = str(metadata.get("training_backend_id") or "").strip()
    if backend_id:
        backend = TRAINING_BACKEND_SPECS.get(backend_id)
        if backend is None:
            raise ValueError(f"unsupported checkpoint training backend: {backend_id}")
        resolved.add(backend.algorithm_id)

    algorithm_id = str(metadata.get("algorithm_id") or "").strip()
    if algorithm_id:
        if algorithm_id not in ALGORITHM_MODEL_CLASSES:
            raise ValueError(f"unsupported checkpoint algorithm: {algorithm_id}")
        resolved.add(cast(PolicyAlgorithmId, algorithm_id))

    model_class = str(metadata.get("model_class") or "").strip()
    if model_class:
        algorithm = MODEL_CLASS_ALGORITHMS.get(model_class)
        if algorithm is None:
            raise ValueError(f"unsupported checkpoint model class: {model_class}")
        resolved.add(algorithm)

    if not resolved:
        raise ValueError("checkpoint metadata must identify its policy algorithm")
    if len(resolved) > 1:
        raise ValueError("checkpoint backend, algorithm, and model class metadata disagree")

    algorithm = next(iter(resolved))
    if algorithm not in allowed:
        raise ValueError(f"unsupported checkpoint algorithm: {algorithm}")
    return algorithm
