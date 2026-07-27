from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import modal

from gradlab.modal_eval_config import load_modal_eval_config, modal_app_name


repo_root = Path(__file__).resolve().parents[2]
config = load_modal_eval_config(repo_root / "experiments" / "modal_eval.yaml")
runtime_image_ref = os.environ.get("GRADLAB_MODAL_EVAL_RUNTIME_IMAGE", "").strip()
if not runtime_image_ref:
    raise RuntimeError("GRADLAB_MODAL_EVAL_RUNTIME_IMAGE must be an immutable docker image ref")
source_sha = os.environ.get("GRADLAB_EXPECTED_SOURCE_SHA", "").strip()
registry_ref = runtime_image_ref.removeprefix("docker:")
app_name = modal_app_name(config.deployment.app_name_prefix, source_sha)
registry_secret_name = os.environ.get("GRADLAB_MODAL_REGISTRY_SECRET", "").strip()
registry_secret = modal.Secret.from_name(registry_secret_name) if registry_secret_name else None
image = modal.Image.from_registry(registry_ref, secret=registry_secret)
image = image.env({"GRADLAB_MODAL_EVAL_RUNTIME_IMAGE": runtime_image_ref})
app = modal.App(app_name)


@app.function(
    name=config.deployment.function_name,
    image=image,
    cpu=config.resources.cpu,
    memory=config.resources.memory_mib,
    min_containers=config.resources.min_containers,
    buffer_containers=config.resources.buffer_containers,
    max_containers=config.resources.max_containers,
    scaledown_window=config.resources.scaledown_window_seconds,
    retries=0,
    timeout=config.timeouts.worker_seconds,
    startup_timeout=config.resources.startup_timeout_seconds,
    single_use_containers=config.resources.single_use_containers,
    include_source=False,
    serialized=True,
)
def evaluate_checkpoint(payload: dict) -> dict:
    from gradlab.modal_eval_worker import execute_attempt

    return execute_attempt(payload, cache_root=Path("/tmp/gradlab-rom-cache"))


@app.function(
    name="startup_probe",
    image=image,
    cpu=0.125,
    memory=128,
    min_containers=0,
    buffer_containers=0,
    max_containers=1,
    retries=0,
    timeout=30,
    startup_timeout=config.resources.startup_timeout_seconds,
    single_use_containers=True,
    include_source=False,
    serialized=True,
)
def startup_probe() -> dict[str, Any]:
    """Prove the deployed image can import its packaged evaluator contract."""
    from gradlab.runtime_contract import runtime_contract

    return {
        **runtime_contract(runtime_image_ref=runtime_image_ref),
        "app_name": app_name,
        "source_deployment": source_sha,
        "presigned_object_transport": True,
    }
