"""Bounded additive HF publication of finalized training datasets via the local queue."""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
import time
import zipfile
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from gradlab.job_queue import (
    HandlerResult,
    JobStore,
    JobSubject,
    SubjectUpdate,
    ensure_flusher,
    register_handler,
)
from gradlab.json_utils import canonical_json_bytes, canonical_json_sha256
from gradlab.trajectory_config import FORMAT
from gradlab.trajectory_delivery import verify_bytes

JOB_TYPE = "training-dataset-publication"
DATASET_README = b"""---
configs:
- config_name: transitions
  data_files: data/*.parquet
---
# Training trajectories

Each Parquet row identifies an episode, an action-cadence transition, and two
lossless PNG members of its ZIP archive. `transition_json` preserves rewards,
boundaries, Policy-update attribution, requested/executed actions and override
facts. `episodes/` contains queryable episode summaries; `contributions/` records
source provenance and selection predicates. A prefix is not a complete episode.

Use whole-Run or seed-group holdouts as a starting point. No scientific train or
validation split is assigned. Training trajectories are not Acceptance evidence.
Stages refer to the original planned training budget, including after early stop.
Sampling is reward-independent but availability- and budget-limited; gaps are
not evidence of unbiased coverage. R2 sources are retained after publication.
"""


def finalized_inventories(control, models, runs):
    references = []
    for run in sorted(set(runs)):
        manifests = [control.get_json(key) for key in control.iter_keys(f"runs/{run}/attempts")
                     if key.endswith("/manifest.json")]
        if not manifests:
            manifests = [control.get_json(f"runs/{run}/manifest.json")]
        manifests.sort(key=lambda value: (value["created_at"], value["attempt_id"]))
        latest = manifests[-1]["attempt_id"]
        for manifest in manifests:
            attempt = manifest["attempt_id"]
            terminal = control.get_json_optional(f"runs/{run}/attempts/{attempt}/terminal.json")
            delivery = (terminal or {}).get("drain", {}).get("dataset_delivery") or {}
            if attempt == latest and not delivery.get("complete"):
                raise ValueError(f"Run has no finalized verified dataset inventory: {run}")
            if not delivery.get("complete"):
                continue
            expected = f"datasets/runs/{run}/attempts/{attempt}/final.json"
            if delivery.get("manifest_key") != expected:
                raise ValueError("dataset inventory is outside its Run/Attempt namespace")
            document = models.get_json(expected)
            if canonical_json_sha256(document) != delivery["manifest_sha256"]:
                raise ValueError("Run dataset inventory checksum mismatch")
            references.append({"run_id": run, "attempt_id": attempt,
                               "key": expected, "sha256": delivery["manifest_sha256"]})
    return references


def _episode_paths(identity):
    digest = hashlib.sha256(identity.encode()).hexdigest()
    return f"episodes/{digest}.json", f"assets/{digest}.zip", f"data/{digest}.parquet"


def _commit(api, read, repo, files, *, compatible=None):
    """Re-read at each immutable head, including after an ambiguous commit response."""
    last_error = None
    for _ in range(5):
        head = api.repo_info(repo_id=repo, repo_type="dataset").sha
        if compatible is not None:
            compatible(head)
        missing = []
        for name, data in files.items():
            current = read(name, head)
            if current is None:
                missing.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=data))
            elif current != (data if isinstance(data, bytes) else Path(data).read_bytes()):
                raise ValueError(f"dataset immutable identity conflict: {name}")
        if not missing:
            return head
        try:
            commit = api.create_commit(
                repo_id=repo,
                repo_type="dataset",
                parent_commit=head,
                operations=missing,
                commit_message="Append immutable training dataset contribution",
            )
            return commit.oid
        except Exception as exc:
            last_error = exc
    raise RuntimeError("HF append did not converge after five head reconciliations") from last_error


def append_episode(api, read, repo, contract, manifest, archive: Path, work: Path):
    index_path, asset_path, table_path = _episode_paths(manifest["episode"]["episode_id"])
    metadata = canonical_json_bytes(manifest)
    contract_data = canonical_json_bytes(contract)

    def compatible(head):
        current = read("training-dataset.json", head)
        if current is None and any(
            entry.path != ".gitattributes"
            for entry in api.list_repo_tree(repo, repo_type="dataset", revision=head)
        ):
            raise ValueError("HF target is not an empty or compatible training dataset")
        if current is not None and current != contract_data:
            raise ValueError("incompatible training dataset execution contract")
        existing = read(index_path, head)
        if existing is not None and existing != metadata:
            raise ValueError("conflicting episode identity")

    head = api.repo_info(repo_id=repo, repo_type="dataset").sha
    compatible(head)
    if read(index_path, head) == metadata:
        # Immutable index is only committed atomically with its complete asset set.
        return head
    verify_bytes(archive.read_bytes(), manifest)
    import pyarrow as pa
    import pyarrow.parquet as pq

    with zipfile.ZipFile(archive) as chunk:
        rows = []
        for index, raw in enumerate(chunk.read("transitions.jsonl").splitlines()):
            row = json.loads(raw)
            if row["step"] != index or "policy_action" not in row or "executed_action" not in row:
                raise ValueError("unaligned or incomplete transition action evidence")
            for frame in (f"frames/{index}.png", f"frames/{index + 1}.png"):
                if frame not in chunk.namelist():
                    raise ValueError("missing referenced episode image")
            rows.append(
                {
                    "episode_id": manifest["episode"]["episode_id"],
                    "step": index,
                    "archive": asset_path,
                    "frame": f"frames/{index}.png",
                    "next_frame": f"frames/{index + 1}.png",
                    "policy_action": int(row["policy_action"]),
                    "executed_action": int(row["executed_action"]),
                    "native_action": int(row["native_action"]),
                    "transition_json": canonical_json_bytes(row).decode(),
                }
            )
        schema = pa.schema(
            [
                (name, dtype)
                for name, dtype in (
                    ("episode_id", pa.string()),
                    ("step", pa.int64()),
                    ("archive", pa.string()),
                    ("frame", pa.string()),
                    ("next_frame", pa.string()),
                    ("policy_action", pa.int64()),
                    ("executed_action", pa.int64()),
                    ("native_action", pa.int64()),
                    ("transition_json", pa.string()),
                )
            ]
        )
        table = work / "transitions.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), table, compression="zstd")
    # Assets are only considered published once the episode index is visible in this commit.
    return _commit(
        api,
        read,
        repo,
        {
            "training-dataset.json": contract_data,
            index_path: metadata,
            asset_path: archive,
            table_path: table,
        },
        compatible=compatible,
    )


def selected_episode(episode, filters):
    if not episode["complete"] and not filters.get("include_prefixes", False):
        return False
    last = episode.get("last_transition") or {}
    facts = last.get("facts") or {}
    values = {
        "stage": episode["stage_start"],
        "return": episode["prefix_return"],
        "score": facts.get("score"),
        "bricks": facts.get("bricks_destroyed"),
    }
    for name, value in values.items():
        for bound in ("min", "max"):
            threshold = filters.get(f"{name}_{bound}")
            if threshold is not None and (
                value is None or (value < threshold if bound == "min" else value > threshold)
            ):
                return False
    return True


class DatasetPublicationHandler:
    job_type = JOB_TYPE
    version = 1

    @classmethod
    def validate_payload(cls, payload):
        if set(payload) != {"runs", "repo", "filters", "queue_root", "repo_root", "inventories"}:
            raise ValueError("malformed dataset publication request")
        runs = payload["runs"]
        if (
            not isinstance(runs, list)
            or not 1 <= len(runs) <= 100
            or any(not re.fullmatch(r"gradlab-[0-9a-f]{32}", run) for run in runs)
        ):
            raise ValueError("dataset publication requires one to 100 Run identities")
        if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", payload["repo"]):
            raise ValueError("dataset target must be owner/repository")
        filters = payload["filters"]
        allowed = {"include_prefixes"} | {
            f"{name}_{bound}"
            for name in ("stage", "return", "score", "bricks")
            for bound in ("min", "max")
        }
        if not isinstance(filters, dict) or set(filters) - allowed:
            raise ValueError("unknown dataset selection predicate")
        for key, value in filters.items():
            if key == "include_prefixes":
                if not isinstance(value, bool):
                    raise ValueError("include_prefixes must be boolean")
            elif (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise ValueError("dataset predicates must be finite numbers")
        for key in ("stage", "return", "score", "bricks"):
            if filters.get(f"{key}_min", -math.inf) > filters.get(f"{key}_max", math.inf):
                raise ValueError("dataset selection bounds are reversed")
        for reference in payload["inventories"]:
            if (reference["run_id"] not in runs
                    or not re.fullmatch(r"attempt-[0-9a-f]{16}", reference["attempt_id"])
                    or reference["key"] != f"datasets/runs/{reference['run_id']}/attempts/{reference['attempt_id']}/final.json"
                    or not re.fullmatch(r"[0-9a-f]{64}", reference["sha256"])):
                raise ValueError("invalid immutable dataset inventory reference")
        return {**payload, "runs": sorted(set(runs))}

    def advance(self, job):
        try:
            return self._publish(job)
        except (ValueError, KeyError, TypeError):
            raise
        except Exception as exc:
            if int(job.get("attempts", 1)) >= 8:
                return HandlerResult(state="blocked", message=f"Publication needs retry: {type(exc).__name__}")
            return HandlerResult(state="retry_wait", available_at=time.time() + 30,
                                 message=f"Publication retry: {type(exc).__name__}")

    def _publish(self, job):
        from gradlab.operator_environment import load_repository_operator_environment
        from gradlab.r2_store import RunStorageConfig, R2Bucket

        payload = self.validate_payload(job["payload"])
        load_repository_operator_environment(Path(payload["repo_root"]))
        storage = RunStorageConfig.from_env()
        models = R2Bucket(storage.models)
        store = JobStore(root=Path(payload["queue_root"]))
        api = HfApi()
        repo = payload["repo"]
        inventory = []
        contract = None
        sources = {}
        # Validate every selected Run before any remote HF mutation or image transfer.
        for reference in payload["inventories"]:
            run = reference["run_id"]
            document = models.get_json(reference["key"])
            if canonical_json_sha256(document) != reference["sha256"]:
                raise ValueError("Run dataset inventory checksum mismatch")
            current = {"format": FORMAT, "execution": document["producer"]["contract"]}
            if contract is not None and current != contract:
                raise ValueError("selected Runs have incompatible dataset execution contracts")
            contract = current
            sources[f"{run}/{reference['attempt_id']}"] = {
                "manifest_sha256": reference["sha256"],
                "producer": document["producer"],
            }
            for chunk in document["chunks"]:
                if selected_episode(chunk["episode"], payload["filters"]):
                    inventory.append(chunk)
        selection = {
            "format": FORMAT,
            "runs": payload["runs"],
            "sources": sources,
            "filters": payload["filters"],
            "episodes": [
                {"episode_id": c["episode"]["episode_id"], "sha256": c["sha256"]} for c in inventory
            ],
        }
        selection_id = canonical_json_sha256(selection)
        api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="dataset-export-", dir=store.root) as temporary:
            work = Path(temporary)

            def read(name, revision):
                try:
                    return Path(
                        hf_hub_download(
                            repo,
                            name,
                            repo_type="dataset",
                            revision=revision,
                            cache_dir=work / "cache",
                        )
                    ).read_bytes()
                except EntryNotFoundError:
                    return None

            head = api.repo_info(repo_id=repo, repo_type="dataset").sha
            if read("training-dataset.json", head) is None:
                if any(
                    entry.path != ".gitattributes"
                    for entry in api.list_repo_tree(repo, repo_type="dataset", revision=head)
                ):
                    raise ValueError("HF target is not an empty or compatible training dataset")
            for chunk in inventory:
                if store.job(str(job["job_id"]))["cancel_requested"]:
                    return HandlerResult(state="canceled", message="Dataset publication canceled")
                if chunk["bytes"] > 128 * 1024**2:
                    raise ValueError("dataset source exceeds bounded export chunk size")
                archive = work / "episode.zip"
                data = models.get_bytes(chunk["key"])
                verify_bytes(data, chunk)
                archive.write_bytes(data)
                del data
                append_episode(api, read, repo, contract, chunk, archive, work)
                archive.unlink()
                import shutil

                shutil.rmtree(work / "cache", ignore_errors=True)
            key = f"contributions/{selection_id}.json"
            revision = _commit(api, read, repo, {key: canonical_json_bytes(selection),
                               "training-dataset.json": canonical_json_bytes(contract),
                               "README.md": DATASET_README})
        return HandlerResult(
            state="succeeded",
            message=f"Published dataset revision {revision}",
            subjects=(
                SubjectUpdate(
                    "dataset",
                    repo,
                    "succeeded",
                    {
                        "revision": revision,
                        "selection_id": selection_id,
                        "episode_count": len(inventory),
                        "url": f"https://huggingface.co/datasets/{repo}/tree/{revision}",
                    },
                ),
            ),
        )


def register_job_handler():
    register_handler(JOB_TYPE, 1, DatasetPublicationHandler, replace=True)


def enqueue_publication(*, runs, repo, filters, repo_root, store=None):
    queue = store or JobStore()
    queue.init()
    from gradlab.operator_environment import load_repository_operator_environment
    from gradlab.r2_store import RunStorageConfig, R2Bucket

    load_repository_operator_environment(Path(repo_root))
    storage = RunStorageConfig.from_env()
    inventories = finalized_inventories(R2Bucket(storage.control), R2Bucket(storage.models), runs)
    payload = DatasetPublicationHandler.validate_payload(
        {
            "runs": list(runs),
            "inventories": inventories,
            "repo": repo,
            "filters": filters,
            "repo_root": str(Path(repo_root).resolve()),
            "queue_root": str(queue.root),
        }
    )
    result = queue.enqueue(
        job_type=JOB_TYPE,
        handler_version=1,
        payload=payload,
        idempotency_key=canonical_json_sha256(payload),
        subjects=[JobSubject("dataset", repo)],
    )
    worker = ensure_flusher(queue)
    return {"job": result.job, "created": result.created, "worker": worker.to_dict()}
