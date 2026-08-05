from __future__ import annotations

from pathlib import Path
from typing import Any

from gradlab.policy_registry import (
    PolicyAlgorithmId,
    resolve_policy_algorithm,
)
from gradlab.trusted_inputs import (
    ApprovedModelInput,
    approve_internal_model,
    stage_and_approve_model,
)

def load_policy_model(
    model_input: ApprovedModelInput,
    *,
    device: str,
    env: Any | None = None,
    tensorboard_log: str | None = None,
    algorithm_id: PolicyAlgorithmId,
    ppo_model_class: type | None = None,
):
    if not isinstance(model_input, ApprovedModelInput):
        raise TypeError("load_policy_model requires an ApprovedModelInput")
    model_input.verify()
    path = model_input.model_path
    if algorithm_id == "action-program":
        from gradlab.action_program import ActionProgramPolicy

        return ActionProgramPolicy.load(path)
    if algorithm_id == "cell-graph":
        from gradlab.cell_graph import CellGraphPolicy

        return CellGraphPolicy.load(path)
    from gradlab.sb3_models import load_sb3_model

    return load_sb3_model(
        model_input,
        device=device,
        env=env,
        tensorboard_log=tensorboard_log,
        algorithm_id=algorithm_id,
        ppo_model_class=ppo_model_class,
    )


def load_external_policy_model(
    model_path: str | Path,
    *,
    device: str,
    env: Any | None = None,
    tensorboard_log: str | None = None,
    algorithm_id: PolicyAlgorithmId,
    source_identity: str | None = None,
    approval_hash: str | None = None,
):
    with stage_and_approve_model(
        model_path,
        source_identity=source_identity,
        expected_hash=approval_hash,
    ) as approved:
        return load_policy_model(
            approved,
            device=device,
            env=env,
            tensorboard_log=tensorboard_log,
            algorithm_id=algorithm_id,
        )


def load_internal_policy_model(
    model_path: str | Path,
    *,
    execution_id: str,
    device: str,
    env: Any | None = None,
    tensorboard_log: str | None = None,
    algorithm_id: PolicyAlgorithmId,
):
    with approve_internal_model(model_path, execution_id=execution_id) as approved:
        return load_policy_model(
            approved,
            device=device,
            env=env,
            tensorboard_log=tensorboard_log,
            algorithm_id=algorithm_id,
        )


def load_pinned_remote_policy_model(
    source: str,
    *,
    download_root: Path,
    approval_hash: str,
    manifest: Any,
    device: str,
    env: Any | None = None,
    tensorboard_log: str | None = None,
    expected_algorithm_id: PolicyAlgorithmId,
    ppo_model_class: type | None = None,
):
    from gradlab.model_sources import download_remote_model_source
    from gradlab.trusted_inputs import approve_staged_model, stage_model_input

    resolved = download_remote_model_source(source, root=download_root, require_pinned=True)
    algorithm_id = resolve_policy_algorithm(resolved.bundle.model["policy"])
    if algorithm_id != expected_algorithm_id:
        raise ValueError(
            "pinned remote model algorithm does not match the queued job: "
            f"expected {expected_algorithm_id}, got {algorithm_id}"
        )
    staged = stage_model_input(resolved.model_path, source_identity=source)
    try:
        expected_manifest = list(manifest) if isinstance(manifest, list | tuple) else None
        actual_manifest = [entry.as_dict() for entry in staged.manifest]
        if expected_manifest is None or actual_manifest != expected_manifest:
            raise ValueError("pinned remote model byte manifest does not match the queued job")
        with approve_staged_model(
            staged,
            expected_hash=approval_hash,
            interactive=False,
        ) as approved:
            return load_policy_model(
                approved,
                device=device,
                env=env,
                tensorboard_log=tensorboard_log,
                algorithm_id=algorithm_id,
                ppo_model_class=ppo_model_class,
            )
    except Exception:
        staged.cleanup()
        raise
