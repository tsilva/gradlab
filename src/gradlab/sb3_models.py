from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from gradlab.policy_registry import (
    SB3_ALGORITHMS,
    Sb3AlgorithmId,
    resolve_policy_algorithm,
)
from gradlab.trusted_inputs import ApprovedModelInput


def _approved_model_document(model_input: ApprovedModelInput) -> Mapping[str, Any]:
    from gradlab.policy_bundle import load_model_document, model_document_path

    path = model_input.model_path
    document_path = (
        path.parent / "model.json"
        if path.name == "model.zip"
        else model_document_path(path)
    )
    return load_model_document(document_path)


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

        document = _approved_model_document(model_input)
        artifact_model_class = str(document["policy"]["model_class"])
        if artifact_model_class == "gradlab.task_advantage.GroupedAdvantagePPO":
            from gradlab.task_advantage import GroupedAdvantagePPO

            model_class = GroupedAdvantagePPO
        elif artifact_model_class == "gradlab.task_advantage.PerTaskAdvantagePPO":
            from gradlab.task_advantage import PerTaskAdvantagePPO

            model_class = PerTaskAdvantagePPO
        else:
            model_class = PPO
    kwargs: dict[str, Any] = {"device": device}
    if env is not None:
        kwargs["env"] = env
    if tensorboard_log is not None:
        kwargs["tensorboard_log"] = tensorboard_log
    model = model_class.load(str(path), **kwargs)
    if algorithm_id == "a2c":
        document = _approved_model_document(model_input)
    provenance = document.get("provenance")
    training_metadata = (
        provenance.get("training_metadata")
        if isinstance(provenance, Mapping)
        else None
    )
    execution_contract = (
        training_metadata.get("policy_execution_contract")
        if isinstance(training_metadata, Mapping)
        else None
    )
    if isinstance(execution_contract, Mapping):
        from gradlab.policy_execution import (
            normalize_policy_execution_contract,
            verify_policy_execution_contract,
        )

        model.gradlab_policy_execution_contract = normalize_policy_execution_contract(
            execution_contract
        )
        if env is not None:
            verify_policy_execution_contract(model, env)
    elif isinstance(getattr(model.policy, "policy_model", None), Mapping):
        raise ValueError("routed policy artifact is missing its execution contract")
    return model
