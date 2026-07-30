from __future__ import annotations

import importlib.metadata
import io
import json
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import StringConstraints

from gradlab.boundary_schema import BoundaryModel, validate_boundary
from gradlab.runtime_contract import (
    RUNTIME_DESCRIPTOR_SCHEMA_VERSION,
    train_config_contract_sha256,
)


DEFAULT_IMAGE_WORKFLOW = "gradlab train image"
DEFAULT_IMAGE_ARTIFACT = "gradlab-train-image"
DEFAULT_IMAGE_ARTIFACT_FILE = "gradlab-train-image.json"
DEFAULT_MODAL_WORKFLOW = "gradlab Modal eval deployment"
DEFAULT_MODAL_ARTIFACT = "gradlab-modal-eval-readiness"
DEFAULT_MODAL_ARTIFACT_FILE = "gradlab-modal-eval-readiness.json"
MODAL_READINESS_SCHEMA_VERSION = 3
VIZDOOM_SMOKE_CONTRACT_VERSION = 2
DEFAULT_RUNTIME_READINESS_TIMEOUT_SECONDS = 20 * 60

DIGEST_IMAGE_REF_RE = re.compile(r"^docker:[^\s@]+@sha256:(?P<digest>[0-9a-fA-F]{64})$")
ACTIVE_WORKFLOW_STATUSES = frozenset({"queued", "in_progress", "pending", "requested", "waiting"})

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[
    str,
    StringConstraints(strip_whitespace=True, to_lower=True, pattern=r"^[0-9a-f]{64}$"),
]
BuildSourceSha = Annotated[
    str,
    StringConstraints(strip_whitespace=True, pattern=r"^[0-9a-fA-F]{40,64}$"),
]


class _BaseImages(BoundaryModel):
    gpu: str = ""
    dependencies: str = ""
    python: NonEmptyText | None = None
    uv: NonEmptyText | None = None


class _VizdoomSmoke(BoundaryModel):
    contract_version: Literal[VIZDOOM_SMOKE_CONTRACT_VERSION]
    image_digest: NonEmptyText
    provider_distribution: Literal["vizdoom-turbo"]
    provider_version: NonEmptyText
    evidence_sha256: Sha256


class _RuntimeReleasePayload(BoundaryModel):
    schema_version: Literal[RUNTIME_DESCRIPTOR_SCHEMA_VERSION]
    runtime_image_ref: NonEmptyText
    source_sha: NonEmptyText
    runtime_input_sha256: Sha256
    runtime_build_source_sha: BuildSourceSha
    overlay_key: Sha256
    dependency_key: Sha256
    gpu_key: Sha256
    train_plan_sha256: Sha256
    gpu_plan_sha256: Sha256
    tags: list[NonEmptyText]
    uv_lock_sha256: Sha256
    base_images: _BaseImages
    workflow_run_id: NonEmptyText
    vizdoom_smoke: _VizdoomSmoke
    digest: str = ""
    image: str = ""
    workflow_run_attempt: str = ""


class _ModalReadinessPayload(BoundaryModel):
    schema_version: Literal[MODAL_READINESS_SCHEMA_VERSION]
    runtime_image_ref: NonEmptyText
    source_sha: NonEmptyText
    runtime_input_sha256: Sha256
    runtime_build_source_sha: BuildSourceSha
    modal_app_name: NonEmptyText
    startup_probe: dict[str, Any]
    workflow_run_id: str = ""
    workflow_run_attempt: str = ""


@dataclass(frozen=True)
class RuntimeImageInfo:
    runtime_image_ref: str
    source_sha: str
    commit_message: str
    published_at: str
    workflow_run_id: str
    schema_version: int = 0
    runtime_input_sha256: str = ""
    runtime_build_source_sha: str = ""
    overlay_key: str = ""
    dependency_key: str = ""
    gpu_key: str = ""
    train_plan_sha256: str = ""
    gpu_plan_sha256: str = ""
    train_config_contract_sha256: str = ""
    modal_app_name: str = ""
    startup_probe: dict[str, Any] | None = None
    vizdoom_smoke_contract_version: int = 0
    vizdoom_provider_version: str = ""
    vizdoom_smoke_evidence_sha256: str = ""


@dataclass(frozen=True)
class ModalReadinessInfo:
    runtime_image_ref: str
    source_sha: str
    modal_app_name: str
    startup_probe: dict[str, Any]
    workflow_run_id: str
    schema_version: int = MODAL_READINESS_SCHEMA_VERSION
    runtime_input_sha256: str = ""
    runtime_build_source_sha: str = ""


def normalize_runtime_image_ref(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("runtime image ref is required")
    if not DIGEST_IMAGE_REF_RE.fullmatch(text):
        raise ValueError(
            "runtime image ref must be an immutable docker digest ref like "
            "docker:ghcr.io/owner/image@sha256:<64-hex-digest>"
        )
    return text


def runtime_image_payload_from_file(path: Path) -> dict[str, Any]:
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"runtime image descriptor must contain a JSON object: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"runtime image descriptor must contain a JSON object: {path}")
    return dict(payload)


def runtime_release_from_payload(
    payload: Mapping[str, Any],
    *,
    label: str,
    expected_source_sha: str,
) -> RuntimeImageInfo:
    schema_version = int(payload.get("schema_version") or 0)
    if schema_version == RUNTIME_DESCRIPTOR_SCHEMA_VERSION:
        receipt = validate_boundary(_RuntimeReleasePayload, payload, label=label)
    else:
        raise ValueError(
            f"{label} schema_version must be {RUNTIME_DESCRIPTOR_SCHEMA_VERSION}"
        )
    runtime_image_ref = normalize_runtime_image_ref(receipt.runtime_image_ref)
    source_sha = receipt.source_sha
    if source_sha != expected_source_sha:
        raise ValueError(
            f"{label} source_sha mismatch: expected {expected_source_sha}, "
            f"got {source_sha or 'missing'}"
        )
    digest = receipt.digest.removeprefix("sha256:")
    if digest and digest.lower() != runtime_image_digest(runtime_image_ref):
        raise ValueError(f"{label} digest does not match runtime_image_ref")
    runtime_input_sha256 = receipt.runtime_input_sha256
    runtime_build_source_sha = receipt.runtime_build_source_sha
    expected_tag = f"runtime-{runtime_input_sha256}"
    if expected_tag not in receipt.tags:
        raise ValueError(f"{label} must include its content-addressed runtime tag")
    try:
        normalize_runtime_image_ref(receipt.base_images.dependencies)
    except ValueError as exc:
        raise ValueError(
            f"{label} must include an immutable dependency image identity"
        ) from exc
    try:
        normalize_runtime_image_ref(receipt.base_images.gpu)
    except ValueError as exc:
        raise ValueError(f"{label} must include an immutable GPU image identity") from exc
    smoke = getattr(receipt, "vizdoom_smoke", None)
    if smoke is not None:
        expected_digest = f"sha256:{runtime_image_digest(runtime_image_ref)}"
        if smoke.image_digest != expected_digest:
            raise ValueError(f"{label} ViZDoom smoke digest does not match runtime image")
        expected_provider_version = importlib.metadata.version(smoke.provider_distribution)
        if smoke.provider_version != expected_provider_version:
            raise ValueError(
                f"{label} ViZDoom smoke provider version mismatch: expected "
                f"{expected_provider_version}, got {smoke.provider_version}"
            )
    return RuntimeImageInfo(
        runtime_image_ref=runtime_image_ref,
        source_sha=source_sha,
        commit_message="",
        published_at="",
        workflow_run_id=receipt.workflow_run_id,
        schema_version=schema_version,
        runtime_input_sha256=runtime_input_sha256,
        runtime_build_source_sha=runtime_build_source_sha,
        overlay_key=receipt.overlay_key,
        dependency_key=receipt.dependency_key,
        gpu_key=receipt.gpu_key,
        train_plan_sha256=receipt.train_plan_sha256,
        gpu_plan_sha256=receipt.gpu_plan_sha256,
        train_config_contract_sha256=train_config_contract_sha256(),
        vizdoom_smoke_contract_version=(
            int(smoke.contract_version) if smoke is not None else 0
        ),
        vizdoom_provider_version=(
            str(smoke.provider_version) if smoke is not None else ""
        ),
        vizdoom_smoke_evidence_sha256=(
            str(smoke.evidence_sha256) if smoke is not None else ""
        ),
    )


def modal_readiness_from_payload(
    payload: Mapping[str, Any],
    *,
    label: str,
    expected_source_sha: str,
    expected_runtime_image_ref: str,
    expected_runtime_input_sha256: str = "",
    expected_runtime_build_source_sha: str = "",
) -> ModalReadinessInfo:
    schema_version = int(payload.get("schema_version") or 0)
    if schema_version != MODAL_READINESS_SCHEMA_VERSION:
        raise ValueError(
            f"{label} schema_version must be {MODAL_READINESS_SCHEMA_VERSION}"
        )
    receipt = validate_boundary(_ModalReadinessPayload, payload, label=label)
    source_sha = receipt.source_sha
    if source_sha != expected_source_sha:
        raise ValueError(
            f"{label} source_sha mismatch: expected {expected_source_sha}, "
            f"got {source_sha or 'missing'}"
        )
    runtime_image_ref = normalize_runtime_image_ref(receipt.runtime_image_ref)
    expected_runtime_image_ref = normalize_runtime_image_ref(expected_runtime_image_ref)
    if runtime_image_ref != expected_runtime_image_ref:
        raise ValueError(f"{label} runtime image does not match the image receipt")
    modal_app_name = receipt.modal_app_name
    startup_probe = receipt.startup_probe
    runtime_input_sha256 = receipt.runtime_input_sha256
    runtime_build_source_sha = receipt.runtime_build_source_sha
    expected_runtime_input_sha256 = str(expected_runtime_input_sha256).strip().lower()
    expected_runtime_build_source_sha = str(expected_runtime_build_source_sha).strip()
    if (
        not expected_runtime_input_sha256
        or runtime_input_sha256 != expected_runtime_input_sha256
    ):
        raise ValueError(f"{label} runtime_input_sha256 does not match image receipt")
    if (
        not expected_runtime_build_source_sha
        or runtime_build_source_sha != expected_runtime_build_source_sha
    ):
        raise ValueError(f"{label} runtime_build_source_sha does not match image receipt")
    expected_probe = {
        "runtime_image_ref": runtime_image_ref,
        "app_name": modal_app_name,
    }
    expected_probe.update(
        {
            "runtime_build_source_sha": runtime_build_source_sha,
            "runtime_input_sha256": runtime_input_sha256,
        }
    )
    for key, expected in expected_probe.items():
        if startup_probe.get(key) != expected:
            raise ValueError(f"{label} startup_probe.{key} does not match readiness")
    probe_contract_sha256 = str(
        startup_probe.get("train_config_contract_sha256") or ""
    ).strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", probe_contract_sha256) is None:
        raise ValueError(
            f"{label} startup_probe.train_config_contract_sha256 is invalid"
        )
    if probe_contract_sha256 != train_config_contract_sha256():
        raise ValueError(
            f"{label} startup_probe.train_config_contract_sha256 does not match readiness"
        )
    return ModalReadinessInfo(
        runtime_image_ref=runtime_image_ref,
        source_sha=source_sha,
        modal_app_name=modal_app_name,
        startup_probe=startup_probe,
        workflow_run_id=receipt.workflow_run_id,
        schema_version=schema_version,
        runtime_input_sha256=runtime_input_sha256,
        runtime_build_source_sha=runtime_build_source_sha,
    )


def clean_git_source_sha(repo_root: Path | str = ".") -> str:
    root = Path(repo_root)
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if status.returncode != 0:
        raise RuntimeError(status.stderr.strip() or "failed to inspect git worktree")
    if status.stdout.strip():
        raise RuntimeError(
            "dstack training requires a clean worktree so source and runtime are exact; "
            "commit or isolate local changes first"
        )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if revision.returncode != 0 or not revision.stdout.strip():
        raise RuntimeError(revision.stderr.strip() or "failed to resolve git HEAD")
    return revision.stdout.strip()


def current_git_branch(repo_root: Path | str = ".") -> str:
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=Path(repo_root),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    branch = result.stdout.strip() if result.returncode == 0 else ""
    if not branch:
        raise RuntimeError("automatic runtime builds require a named current Git branch")
    return branch


def require_remote_source(source_sha: str, *, branch: str, repo_root: Path | str = ".") -> None:
    result = subprocess.run(
        ["git", "ls-remote", "--heads", "origin", f"refs/heads/{branch}"],
        cwd=Path(repo_root),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    remote_sha = result.stdout.split(maxsplit=1)[0] if result.stdout.strip() else ""
    if result.returncode != 0 or remote_sha != source_sha:
        detail = result.stderr.strip()
        raise RuntimeError(
            f"exact source {source_sha} is not the pushed head of origin/{branch}; "
            f"push the commit before running dstack training"
            + (f" ({detail})" if detail else "")
        )


@lru_cache(maxsize=1)
def _gh_executable() -> str:
    discovered = shutil.which("gh")
    if discovered:
        return discovered
    for candidate in (Path("/opt/homebrew/bin/gh"), Path("/usr/local/bin/gh")):
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError("gh CLI is required to resolve runtime artifacts")


def _resolved_gh_command(command: Sequence[str]) -> list[str]:
    parts = [str(part) for part in command]
    if not parts or parts[0] != "gh":
        raise ValueError("GitHub command must begin with gh")
    parts[0] = _gh_executable()
    return parts


def _run_gh_json(command: Sequence[str]) -> Any:
    try:
        result = subprocess.run(
            _resolved_gh_command(command),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("gh CLI is required to resolve the latest runtime image") from exc
    if result.returncode != 0:
        output = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
        raise RuntimeError(f"gh command failed: {' '.join(command)}\n{output}")
    return json.loads(result.stdout or "null")


def _run_gh_bytes(command: Sequence[str]) -> bytes:
    try:
        result = subprocess.run(
            _resolved_gh_command(command),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("gh CLI is required to download runtime artifacts") from exc
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"gh command failed: {' '.join(command)}\n{error}")
    return bytes(result.stdout)


def _run_gh(command: Sequence[str]) -> None:
    try:
        result = subprocess.run(
            _resolved_gh_command(command),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("gh CLI is required to dispatch the runtime workflow") from exc
    if result.returncode != 0:
        output = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
        raise RuntimeError(f"gh command failed: {' '.join(command)}\n{output}")


@lru_cache(maxsize=1)
def _repository_name() -> str:
    payload = _run_gh_json(["gh", "repo", "view", "--json", "nameWithOwner"])
    name = str(payload.get("nameWithOwner") or "") if isinstance(payload, Mapping) else ""
    if not name:
        raise RuntimeError("gh repo view did not return nameWithOwner")
    return name


def _artifact_payload_for_run(
    run_id: str,
    artifact_name: str,
    artifact_file: str,
) -> dict[str, Any] | None:
    repository = _repository_name()
    listing = _run_gh_json(
        [
            "gh",
            "api",
            f"repos/{repository}/actions/runs/{run_id}/artifacts?per_page=100",
        ]
    )
    artifacts = listing.get("artifacts", []) if isinstance(listing, Mapping) else []
    artifact = next(
        (
            item
            for item in artifacts
            if isinstance(item, Mapping)
            and str(item.get("name") or "") == artifact_name
            and not bool(item.get("expired"))
        ),
        None,
    )
    if artifact is None:
        return None
    artifact_id = str(artifact.get("id") or "").strip()
    if not artifact_id:
        raise RuntimeError(f"artifact {artifact_name!r} from run {run_id} has no id")
    archive = _run_gh_bytes(
        ["gh", "api", f"repos/{repository}/actions/artifacts/{artifact_id}/zip"]
    )
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        try:
            text = bundle.read(artifact_file).decode("utf-8")
        except KeyError as exc:
            raise RuntimeError(
                f"artifact {artifact_name!r} from run {run_id} lacks {artifact_file}"
            ) from exc
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"artifact {artifact_name!r} from run {run_id} must contain an object")
    return dict(payload)


def _workflow_runs(
    *, workflow: str, branch: str | None = None, limit: int = 50
) -> list[dict[str, Any]]:
    command = [
        "gh",
        "run",
        "list",
        "--workflow",
        workflow,
        "--limit",
        str(limit),
        "--json",
        "databaseId,headSha,displayTitle,createdAt,updatedAt,status,conclusion,url",
    ]
    if branch:
        command[5:5] = ["--branch", branch]
    payload = _run_gh_json(command)
    return (
        [dict(row) for row in payload if isinstance(row, Mapping)]
        if isinstance(payload, list)
        else []
    )


def _matching_runs(
    *, source_sha: str, workflow: str, branch: str | None = None
) -> list[dict[str, Any]]:
    return [
        row
        for row in _workflow_runs(workflow=workflow, branch=branch)
        if str(row.get("headSha") or "") == source_sha
    ]


def runtime_release_for_source(
    *,
    source_sha: str,
    workflow: str = DEFAULT_IMAGE_WORKFLOW,
    branch: str | None = None,
    artifact_name: str = DEFAULT_IMAGE_ARTIFACT,
) -> RuntimeImageInfo:
    errors: list[str] = []
    runs = _matching_runs(source_sha=source_sha, workflow=workflow, branch=branch)
    for run in runs:
        run_id = str(run.get("databaseId") or "").strip()
        if not run_id:
            continue
        try:
            payload = _artifact_payload_for_run(run_id, artifact_name, DEFAULT_IMAGE_ARTIFACT_FILE)
            if payload is None:
                continue
            info = runtime_release_from_payload(
                payload,
                label=f"runtime image receipt {run_id}",
                expected_source_sha=source_sha,
            )
            return replace(
                info,
                commit_message=str(run.get("displayTitle") or info.commit_message).strip(),
                published_at=str(
                    run.get("updatedAt") or run.get("createdAt") or info.published_at
                ).strip(),
                workflow_run_id=info.workflow_run_id or run_id,
            )
        except Exception as exc:
            errors.append(f"run {run_id}: {exc}")
    detail = f"no {workflow!r} workflow run exists for {source_sha}"
    if runs:
        latest = runs[0]
        detail = (
            f"latest workflow status={latest.get('conclusion') or latest.get('status')} "
            f"url={latest.get('url')}"
        )
    if errors:
        detail += "; invalid receipts: " + " | ".join(errors)
    raise RuntimeError(f"no exact-source runtime image receipt exists for {source_sha}; {detail}")


def _workflow_status_detail(runs: Sequence[Mapping[str, Any]]) -> str:
    if not runs:
        return "no workflow run is visible yet"
    latest = runs[0]
    return f"status={latest.get('conclusion') or latest.get('status')} url={latest.get('url')}"


def wait_for_runtime_release(
    *,
    source_sha: str,
    workflow: str,
    branch: str,
    artifact_name: str,
    timeout: float,
    repo_root: Path | str = ".",
    poll_seconds: float = 5.0,
) -> RuntimeImageInfo:
    deadline = time.monotonic() + max(timeout, 0.0)
    dispatched = False
    run_ids_before_dispatch: set[str] = set()
    watched_run_ids: set[str] = set()
    while True:
        try:
            return runtime_release_for_source(
                source_sha=source_sha,
                workflow=workflow,
                branch=None,
                artifact_name=artifact_name,
            )
        except RuntimeError as receipt_error:
            runs = _matching_runs(source_sha=source_sha, workflow=workflow)
            active_runs = [
                row for row in runs if str(row.get("status") or "") in ACTIVE_WORKFLOW_STATUSES
            ]
            active = bool(active_runs)
            watched_run_ids.update(str(row.get("databaseId") or "") for row in active_runs)
            current_run_ids = {str(row.get("databaseId") or "") for row in runs}
            watched_terminal = bool(watched_run_ids & current_run_ids) and not active
            if watched_terminal:
                raise RuntimeError(
                    f"exact-source runtime workflow completed without a usable image receipt; "
                    f"{_workflow_status_detail(runs)}"
                ) from receipt_error
            if not active and not dispatched:
                run_ids_before_dispatch = {str(row.get("databaseId") or "") for row in runs}
                require_remote_source(source_sha, branch=branch, repo_root=repo_root)
                _run_gh(
                    [
                        "gh",
                        "workflow",
                        "run",
                        workflow,
                        "--ref",
                        branch,
                        "-f",
                        f"source_sha={source_sha}",
                    ]
                )
                dispatched = True
                print(
                    f"runtime image missing; dispatched {workflow!r} for {source_sha}",
                    file=sys.stderr,
                    flush=True,
                )
            dispatched_terminal = bool(current_run_ids - run_ids_before_dispatch)
            if not active and dispatched and dispatched_terminal:
                raise RuntimeError(
                    f"exact-source runtime workflow completed without a usable image receipt; "
                    f"{_workflow_status_detail(runs)}"
                ) from receipt_error
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for exact-source runtime image after {timeout:g}s; "
                    f"{_workflow_status_detail(runs)}"
                ) from receipt_error
            time.sleep(max(poll_seconds, 0.1))


def modal_readiness_for_release(
    release: RuntimeImageInfo,
    *,
    artifact_name: str = DEFAULT_MODAL_ARTIFACT,
    image_workflow: str = DEFAULT_IMAGE_WORKFLOW,
) -> ModalReadinessInfo:
    run_ids = [release.workflow_run_id] if release.workflow_run_id else []
    for workflow in (image_workflow, DEFAULT_MODAL_WORKFLOW):
        for row in _matching_runs(source_sha=release.source_sha, workflow=workflow):
            run_id = str(row.get("databaseId") or "").strip()
            if run_id and run_id not in run_ids:
                run_ids.append(run_id)
    errors: list[str] = []
    for run_id in run_ids:
        try:
            payload = _artifact_payload_for_run(run_id, artifact_name, DEFAULT_MODAL_ARTIFACT_FILE)
            if payload is None:
                continue
            return modal_readiness_from_payload(
                payload,
                label=f"Modal readiness receipt {run_id}",
                expected_source_sha=release.source_sha,
                expected_runtime_image_ref=release.runtime_image_ref,
                expected_runtime_input_sha256=release.runtime_input_sha256,
                expected_runtime_build_source_sha=release.runtime_build_source_sha,
            )
        except Exception as exc:
            errors.append(f"run {run_id}: {exc}")
    detail = "; ".join(errors) if errors else "readiness artifact is not available"
    raise RuntimeError(f"Modal is not ready for {release.runtime_image_ref}: {detail}")


def wait_for_modal_readiness(
    release: RuntimeImageInfo,
    *,
    timeout: float,
    image_workflow: str = DEFAULT_IMAGE_WORKFLOW,
    poll_seconds: float = 5.0,
) -> RuntimeImageInfo:
    deadline = time.monotonic() + max(timeout, 0.0)
    while True:
        try:
            readiness = modal_readiness_for_release(
                release,
                image_workflow=image_workflow,
            )
            return replace(
                release,
                modal_app_name=readiness.modal_app_name,
                startup_probe=readiness.startup_probe,
            )
        except RuntimeError as readiness_error:
            image_runs = _matching_runs(source_sha=release.source_sha, workflow=image_workflow)
            modal_runs = _matching_runs(
                source_sha=release.source_sha, workflow=DEFAULT_MODAL_WORKFLOW
            )
            runs = [*image_runs, *modal_runs]
            active = any(str(row.get("status") or "") in ACTIVE_WORKFLOW_STATUSES for row in runs)
            if runs and not active:
                raise RuntimeError(
                    f"Modal deployment completed without valid readiness for "
                    f"{release.runtime_image_ref}; {_workflow_status_detail(runs)}"
                ) from readiness_error
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for Modal readiness after {timeout:g}s; "
                    f"{_workflow_status_detail(runs)}"
                ) from readiness_error
            time.sleep(max(poll_seconds, 0.1))


def runtime_release_from_args(
    args: Any,
    *,
    repo_root: Path | str = ".",
    wait_for_modal: bool = True,
) -> RuntimeImageInfo:
    source_sha = clean_git_source_sha(repo_root)
    workflow = getattr(args, "image_workflow", DEFAULT_IMAGE_WORKFLOW)
    artifact_name = getattr(args, "image_artifact", DEFAULT_IMAGE_ARTIFACT)
    timeout = float(
        getattr(
            args,
            "runtime_readiness_timeout",
            DEFAULT_RUNTIME_READINESS_TIMEOUT_SECONDS,
        )
    )
    ref_file = getattr(args, "runtime_image_ref_file", None)
    existing_only = bool(getattr(args, "existing_runtime_only", False))
    readiness_started = time.monotonic()
    if ref_file:
        payload = runtime_image_payload_from_file(Path(ref_file))
        release = runtime_release_from_payload(
            payload,
            label=f"runtime image descriptor {ref_file}",
            expected_source_sha=source_sha,
        )
    elif existing_only:
        release = runtime_release_for_source(
            source_sha=source_sha,
            workflow=workflow,
            branch=None,
            artifact_name=artifact_name,
        )
    else:
        branch = getattr(args, "image_branch", None) or current_git_branch(repo_root)
        release = wait_for_runtime_release(
            source_sha=source_sha,
            workflow=workflow,
            branch=branch,
            artifact_name=artifact_name,
            timeout=timeout,
            repo_root=repo_root,
        )
    expected = {
        "runtime_image_ref": str(getattr(args, "expected_runtime_image_ref", "") or "").strip(),
        "runtime_input_sha256": str(
            getattr(args, "expected_runtime_input_sha256", "") or ""
        ).strip(),
        "runtime_build_source_sha": str(
            getattr(args, "expected_runtime_build_source_sha", "") or ""
        ).strip(),
    }
    supplied = [key for key, value in expected.items() if value]
    if supplied and len(supplied) != len(expected):
        missing = sorted(set(expected) - set(supplied))
        raise ValueError(
            "expected runtime guards must be supplied together; missing " + ", ".join(missing)
        )
    mismatches = {
        key: {"expected": value, "actual": str(getattr(release, key) or "")}
        for key, value in expected.items()
        if value and str(getattr(release, key) or "") != value
    }
    if mismatches:
        detail = "; ".join(
            f"{key}: expected {values['expected']!r}, got {values['actual']!r}"
            for key, values in sorted(mismatches.items())
        )
        raise RuntimeError(f"resolved runtime does not match the pinned research runtime; {detail}")
    if wait_for_modal:
        remaining = max(timeout - (time.monotonic() - readiness_started), 0.0)
        release = wait_for_modal_readiness(
            release,
            timeout=remaining,
            image_workflow=workflow,
        )
    return release


def runtime_image_digest(value: str) -> str:
    match = DIGEST_IMAGE_REF_RE.fullmatch(normalize_runtime_image_ref(value))
    assert match is not None
    return match.group("digest").lower()
