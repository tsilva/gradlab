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
RuntimePolicyAlgorithmId: TypeAlias = Literal["ppo", "a2c", "dqn", "jerk"]
Sb3AlgorithmId: TypeAlias = Literal["ppo", "a2c", "dqn"]


@dataclass(frozen=True)
class TrainingBackendSpec:
    module_name: str
    algorithm_id: RuntimePolicyAlgorithmId


@dataclass(frozen=True)
class BackendProvenanceSpec:
    """Portable checkpoint provenance, independent of local launch support."""

    algorithm_id: PolicyAlgorithmId


TRAINING_BACKEND_SPECS: dict[str, TrainingBackendSpec] = {
    "gradlab.go-explore": TrainingBackendSpec("gradlab.training.go_explore", "jerk"),
    "gradlab.jerk": TrainingBackendSpec("gradlab.training.jerk", "jerk"),
    "sb3.a2c": TrainingBackendSpec("gradlab.training.sb3", "a2c"),
    "sb3.ppo": TrainingBackendSpec("gradlab.training.sb3", "ppo"),
}
BACKEND_PROVENANCE_SPECS: dict[str, BackendProvenanceSpec] = {
    **{
        backend_id: BackendProvenanceSpec(spec.algorithm_id)
        for backend_id, spec in TRAINING_BACKEND_SPECS.items()
    },
    # GradLab can validate, load, evaluate, and play archived SB3 DQN
    # checkpoints without claiming that DQN is a launchable training backend.
    "sb3.dqn": BackendProvenanceSpec("dqn"),
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

RUNTIME_POLICY_ALGORITHMS = frozenset[PolicyAlgorithmId]({"ppo", "a2c", "dqn", "jerk"})
SB3_ALGORITHMS = frozenset[PolicyAlgorithmId]({"ppo", "a2c", "dqn"})


def backend_provenance_algorithm(backend_id: str) -> PolicyAlgorithmId:
    backend = BACKEND_PROVENANCE_SPECS.get(str(backend_id).strip())
    if backend is None:
        raise ValueError(f"unsupported checkpoint training backend: {backend_id}")
    return backend.algorithm_id


def default_action_selection_mode(algorithm_id: PolicyAlgorithmId) -> str:
    if algorithm_id in {"ppo", "a2c", "recurrent-ppo"}:
        return "stochastic"
    if algorithm_id == "dqn":
        return "epsilon_greedy"
    if algorithm_id == "jerk":
        return "program"
    raise ValueError(f"unsupported checkpoint algorithm: {algorithm_id}")


def resolve_policy_algorithm(
    metadata: Mapping[str, Any] | None,
    *,
    allowed: frozenset[PolicyAlgorithmId] = RUNTIME_POLICY_ALGORITHMS,
) -> PolicyAlgorithmId:
    metadata = metadata or {}
    resolved: set[PolicyAlgorithmId] = set()

    backend_id = str(metadata.get("training_backend_id") or "").strip()
    if backend_id:
        resolved.add(backend_provenance_algorithm(backend_id))

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
