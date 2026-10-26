"""Observe remote study evidence and persist verified local bookkeeping.

This module never submits, retries, cancels, or changes remote work. The study
controller remains responsible for reservations, ranking, and application.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from botocore.exceptions import BotoCoreError, ClientError

from gradlab import autoresearch as study
from gradlab.run_authority import RunAuthority
from gradlab.training_backend import load_training_backend


class ScanLimitReached(RuntimeError):
    """A bounded discovery read could not establish the submission inventory."""


def discover_submissions(
    authority: RunAuthority, submission_keys: set[str], *, max_scan_keys: int
) -> dict[str, list[dict[str, Any]]]:
    matches: dict[str, list[dict[str, Any]]] = {key: [] for key in submission_keys}
    for count, key in enumerate(authority.control.iter_keys("runs/"), start=1):
        if count > max_scan_keys:
            raise ScanLimitReached("submission discovery exceeded --max-scan-keys")
        parts = key.split("/")
        if len(parts) != 3 or parts[2] != "manifest.json":
            continue
        manifest = authority.control.get_json(key)
        submission = str((manifest.get("compute") or {}).get("submission_key") or "")
        if submission in matches:
            if manifest.get("run_id") != parts[1]:
                raise ValueError("submission manifest Run identity disagrees with its storage key")
            matches[submission].append(manifest)
    return matches


def manifest_launch_row(
    state: Mapping[str, Any],
    wave: Mapping[str, Any],
    manifest: Mapping[str, Any],
    authority: RunAuthority,
) -> dict[str, Any]:
    """Recover launch fields from a manifest and its hash-validated recipe."""
    document = authority.recipe_document(str(manifest["recipe_sha256"]))
    provenance = document["provenance"]
    sources = {item["path"]: item["sha256"] for item in provenance["source_files"]}
    expected = {item["path"]: item["sha256"] for item in state["source_files"]}
    if sources != expected or provenance.get("source_commit") != state["source_sha"]:
        raise ValueError("recovered recipe differs from the pinned composition sources")
    recipe = document["recipe"]
    config = recipe["train_config"]
    compute = manifest["compute"]
    selected = compute["selected"]
    if (
        selected.get("kind") != "local"
        or selected.get("target") != state["policy"]["compute_target"]
    ):
        raise ValueError("recovered Run changed the pinned local fleet")
    if state["policy"].get("dstack_coordinator_id") != compute.get("dstack_coordinator_id"):
        raise ValueError("recovered Run changed the pinned coordinator")
    seed = int(manifest["seed"])
    candidate = state["candidates"][wave["candidate_id"]]
    backend = config["training_backend"]
    expected_backend = load_training_backend(state["backend"]).normalize_config(
        {**state["baseline"]["backend_config"], **candidate["delta"]},
        label="autoresearch recovered backend",
    )
    runtime = {
        "image_ref": manifest["image_digest"],
        "input_sha256": compute["runtime_input_sha256"],
        "build_source_sha": compute["runtime_build_source_sha"],
    }
    if (
        study.RUN_ID_PATTERN.fullmatch(str(manifest["run_id"])) is None
        or manifest["source_sha"] != state["source_sha"]
        or seed not in wave["seeds"]
        or compute["submission_key"] != wave["submission_key"]
        or compute["checkpoint_eval_backend"] != "none"
        or not study.tracking_matches(manifest["tracking"], state["policy"]["tracking"])
        or list(manifest["recipe_overrides"]) != study.expected_recipe_overrides(state, wave)
        or manifest["run_description"] != study.run_description(state, wave, seed)
        or not all(runtime.values())
        or (state.get("runtime") is not None and runtime != state["runtime"])
        or (manifest["run_id"] not in wave["run_ids"] and wave["status"] != "reserved")
        or int(config["seed"]) != seed
        or int(config["timesteps"]) != int(wave["timesteps"])
        or backend["id"] != state["backend"]
        or backend["config"] != expected_backend
        or list(document["resolution"]["recipe"]["overrides"]) != list(manifest["recipe_overrides"])
        or provenance["runtime"].get("image_ref") != manifest["image_digest"]
    ):
        raise ValueError("recovered recipe disagrees with the reserved Run contract")
    return {
        "run_id": manifest["run_id"],
        "seed": seed,
        "submission_key": compute["submission_key"],
        "source_sha": manifest["source_sha"],
        "goal_file": state["goal_path"],
        "recipe_file": state["recipe_path"],
        "recipe_overrides": manifest["recipe_overrides"],
        "run_description": manifest["run_description"],
        "checkpoint_eval_backend": compute["checkpoint_eval_backend"],
        "tracking": manifest["tracking"],
        "image_digest": manifest["image_digest"],
        "runtime_input_sha256": compute["runtime_input_sha256"],
        "runtime_build_source_sha": compute["runtime_build_source_sha"],
    }


def _status(root: Path, run_id: str) -> dict[str, Any]:
    from gradlab.experiment_cli import _status as experiment_status

    return experiment_status(root, run_id)


def _authority(root: Path) -> RunAuthority:
    study.load_repository_operator_environment(root)
    return RunAuthority(study.RunStorageConfig.from_env())


def _guard(path: Path) -> dict[str, Any]:
    with study.edit_state(path) as state:
        if state["status"] == "active":
            try:
                study.source_guard(state)
            except RuntimeError as exc:
                study.record_source_pause(state, exc)
    return state


def _pause(path: Path, event: str, error: Exception) -> None:
    with study.edit_state(path) as state:
        if state["status"] == "active":
            state["status"] = "paused"
            state["pause_reason"] = {"event": event, "detail": str(error)}


def _report(path: Path, observations: list[dict[str, Any]]) -> dict[str, Any]:
    state = study.load_state(path)
    action = study.next_action(state)
    blocked = state["status"] == "paused"
    retryable_read = any(row["action"] == "retry_reconcile" for row in observations)
    waiting = retryable_read or action["action"] == "await_runs"
    if blocked or retryable_read:
        observations = [row for row in observations if row["action"] != "launch_reserved_wave"]
    return {
        "study": str(path),
        "state": "blocked"
        if blocked
        else "done"
        if action["action"] == "done"
        else "waiting"
        if waiting
        else "ready",
        "next": action,
        "observations": observations,
    }


def reconcile_study(
    path: Path,
    *,
    max_scan_keys: int = 10000,
    authority: RunAuthority | None = None,
    status_reader: Callable[[Path, str], dict[str, Any]] = _status,
) -> dict[str, Any]:
    if max_scan_keys < 1:
        raise ValueError("--max-scan-keys must be positive")
    observations: list[dict[str, Any]] = []
    # Serialize reconciliation processes without holding the state lock over network I/O.
    with study.lock(path.parent / ".reconcile.lock"):
        state = _guard(path)
        if state["status"] != "active" or not state["waves"]:
            return _report(path, observations)
        try:
            authority = authority or _authority(Path(state["repo_root"]))
            reserved = [wave for wave in state["waves"] if wave["status"] == "reserved"]
            matches = (
                discover_submissions(
                    authority,
                    {wave["submission_key"] for wave in reserved},
                    max_scan_keys=max_scan_keys,
                )
                if reserved
                else {}
            )
            for wave in reserved:
                state = _guard(path)
                if state["status"] != "active":
                    return _report(path, observations)
                manifests = matches[wave["submission_key"]]
                if not manifests:
                    observations.append(
                        {
                            "submission_key": wave["submission_key"],
                            "state": "ready",
                            "action": "launch_reserved_wave",
                            "commands": [
                                study.launch_command(state, wave, seed) for seed in wave["seeds"]
                            ],
                        }
                    )
                    continue
                rows = [
                    manifest_launch_row(state, wave, manifest, authority) for manifest in manifests
                ]
                with study.edit_state(path) as current:
                    study.source_guard(current)
                    if current["status"] != "active":
                        return _report(path, observations)
                    study.record_launch(current, wave["submission_key"], {"runs": rows})
                if study.load_state(path)["status"] != "active":
                    return _report(path, observations)

            state = _guard(path)
            for wave in state["waves"]:
                if wave.get("closed") or wave["status"] == "reserved":
                    continue
                recorded = {record["run_id"] for record in wave["terminal_runs"]}
                for run_id in wave["run_ids"]:
                    if run_id in recorded:
                        continue
                    current = _guard(path)
                    if current["status"] != "active":
                        return _report(path, observations)
                    snapshot = status_reader(Path(current["repo_root"]), run_id)
                    if snapshot.get("run_id") != run_id:
                        raise ValueError("status observation changed the recorded Run identity")
                    manifest = snapshot["semantic"]["manifest"]
                    row = manifest_launch_row(current, wave, manifest, authority)
                    with study.edit_state(path) as checked:
                        study.source_guard(checked)
                        if checked["status"] != "active":
                            return _report(path, observations)
                        # Validate the row without reinterpreting a partial live wave as a submission.
                        if (
                            row["seed"] not in wave["seeds"]
                            or row["submission_key"] != wave["submission_key"]
                        ):
                            raise ValueError(
                                "status observation changed the reserved seed or submission"
                            )
                        if not snapshot.get("completed"):
                            observations.append(
                                {"run_id": run_id, "state": "waiting", "action": "monitor"}
                            )
                            continue
                        study.record_terminal(checked, snapshot)
                    if study.load_state(path)["status"] != "active":
                        return _report(path, observations)

            state = _guard(path)
            if state["status"] != "active":
                return _report(path, observations)
            for wave in state["waves"]:
                if wave.get("closed"):
                    continue
                for record in wave["terminal_runs"]:
                    if record.get("training_evidence") is None:
                        study.collect_training_evidence(path, record["run_id"], authority=authority)
                        if study.load_state(path)["status"] != "active":
                            return _report(path, observations)
        except (OSError, BotoCoreError, ClientError, ScanLimitReached) as exc:
            # Report only the error type: service errors can contain private URLs.
            observations.append(
                {"state": "waiting", "action": "retry_reconcile", "error_type": type(exc).__name__}
            )
        except (ValueError, KeyError, RuntimeError) as exc:
            _pause(path, "reconciliation_evidence_mismatch", exc)
        return _report(path, observations)
