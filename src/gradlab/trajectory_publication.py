"""Bounded additive HF publication of finalized training datasets via the local queue."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
import zipfile
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError

from gradlab.job_queue import (
    HandlerResult,
    JobStore,
    JobSubject,
    SubjectUpdate,
    ensure_flusher,
    register_handler,
)
from gradlab.json_utils import canonical_json_bytes, canonical_json_sha256
from gradlab.checkpoint_monitoring import FORMAT

JOB_TYPE = "training-dataset-publication"
DATASET_README = b"""---
configs:
- config_name: episodes
  data_files: episodes/*.parquet
---
# Checkpoint monitoring trajectories

Each complete episode row links to immutable transition tables and lossless PNG
chunks. The episode index retains Run, training seed, Checkpoint, evaluation,
episode conditions, actions, rewards, boundaries and source/runtime provenance.
Frames are full unmasked native RGB at the contracted action cadence. Chunk joins
preserve the initial and true terminal image without duplicated transitions.
Staged chunks without a complete episode index are not dataset episodes.

Monitoring is observational and never Acceptance evidence. Use whole-Run or
training-seed-group holdouts to avoid leakage across correlated Checkpoints;
split assignment is left to consumers. R2 sources remain canonical.
"""


def finalized_inventories(control, models, runs):
    references = []
    for run in sorted(set(runs)):
        manifests = [
            control.get_json(key)
            for key in control.iter_keys(f"runs/{run}/attempts")
            if key.endswith("/manifest.json")
        ]
        if not manifests:
            manifests = [control.get_json(f"runs/{run}/manifest.json")]
        latest = max(manifests, key=lambda m: (m["created_at"], m["attempt_id"]))
        attempt = latest["attempt_id"]
        terminal = control.get_json_optional(f"runs/{run}/attempts/{attempt}/terminal.json")
        delivery = (terminal or {}).get("drain", {}).get("checkpoint_monitoring") or {}
        if not delivery.get("complete") or not delivery.get("workers_quiescent"):
            raise ValueError(f"Run has no finalized verified monitoring inventory: {run}")
        for evaluation in delivery["inventory"]:
            identity = evaluation["evaluation_id"]
            key = f"monitoring/{run}/{identity}/result.json"
            if evaluation["status"] != "complete" or evaluation["result_key"] != key:
                raise ValueError("monitoring inventory is incomplete or outside its Run namespace")
            document = models.get_json(key)
            digest = canonical_json_sha256(document)
            if digest != evaluation["result_sha256"]:
                raise ValueError("monitoring inventory checksum mismatch")
            references.append(dict(run_id=run, attempt_id=attempt, key=key, sha256=digest))
    return references



def completed_snapshot_inventories(control, models, runs):
    """Freeze completed, supervisor-verified evaluations without waiting for run drain."""
    references = []
    for run in sorted(set(runs)):
        for state_key in sorted(control.iter_keys(f"runs/{run}/monitoring/")):
            if not state_key.endswith("/state.json"):
                continue
            state = control.get_json(state_key)
            if state.get("status") != "complete":
                continue
            intent = control.get_json(state_key.removesuffix("state.json") + "intent.json")
            identity = intent["evaluation_id"]
            prefix = f"monitoring/{run}/{identity}"
            if intent["run_id"] != run or intent["prefix"] != prefix:
                raise ValueError("snapshot inventory is outside its Run namespace")
            key = prefix + "/result.json"
            document = models.get_json(key)
            digest = canonical_json_sha256(document)
            if digest != state["result_sha256"]:
                raise ValueError("snapshot inventory checksum mismatch")
            if (
                document["evaluation_id"] != identity
                or document["checkpoint_id"] != intent["checkpoint"]["checkpoint_id"]
                or document["checkpoint_step"] != intent["checkpoint"]["step"]
                or document["contract_sha256"] != intent["contract_sha256"]
                or len(document["episodes"]) != len(intent["manifest"])
                or {e["episode_id"] for e in document["episodes"]}
                != {e["episode_id"] for e in intent["manifest"]}
                or not all(e.get("complete") for e in document["episodes"])
            ):
                raise ValueError("snapshot requires a complete bound evaluation manifest")
            references.append(dict(run_id=run, attempt_id=intent["attempt_id"],
                                   key=key, sha256=digest))
    if not references:
        raise ValueError("selected Runs have no completed monitoring evaluations")
    return references


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
            if isinstance(exc, HfHubHTTPError) and exc.response.status_code == 429:
                raise
            last_error = exc
    raise RuntimeError("HF append did not converge after five head reconciliations") from last_error


def selected_episode(episode, filters):
    if not episode["complete"] and not filters.get("include_prefixes", False):
        return False
    values = {
        "stage": episode["checkpoint_step"] / episode["planned_training_steps"],
        "return": episode["shaped_return"],
        "score": episode.get("native_score"),
        "bricks": episode.get("bricks_destroyed"),
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
    version = 2

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
        if not payload["inventories"]:
            raise ValueError("selected Runs have no recorded dataset inventories")
        for reference in payload["inventories"]:
            if (
                reference["run_id"] not in runs
                or not re.fullmatch(r"attempt-[0-9a-f]{16}", reference["attempt_id"])
                or not re.fullmatch(
                    r"monitoring/" + re.escape(reference["run_id"]) + r"/[0-9a-f]{64}/result\.json",
                    reference["key"],
                )
                or not re.fullmatch(r"[0-9a-f]{64}", reference["sha256"])
            ):
                raise ValueError("invalid immutable dataset inventory reference")
        return {**payload, "runs": sorted(set(runs))}

    def advance(self, job):
        from gradlab.dataset_shards import PublicationDeferred
        try:
            return self._publish(job)
        except PublicationDeferred as exc:
            return HandlerResult(state="retry_wait", available_at=exc.until, message=str(exc))
        except ValueError, KeyError, TypeError:
            raise
        except Exception as exc:
            logging.getLogger(__name__).exception("Dataset publication failed")
            if isinstance(exc, HfHubHTTPError) and exc.response.status_code == 429:
                # Commit quota is hourly and distinct from the generic API window.
                delay = 3600 if "repository commits" in str(exc) else 300
                return HandlerResult(
                    state="retry_wait", available_at=time.time() + delay,
                    message=f"HF rate limit; retry in {delay} seconds",
                )
            if int(job.get("attempts", 1)) >= 8:
                return HandlerResult(
                    state="blocked", message=f"Publication needs retry: {type(exc).__name__}"
                )
            return HandlerResult(
                state="retry_wait",
                available_at=time.time() + 30,
                message=f"Publication retry: {type(exc).__name__}",
            )

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
            if document.get("format") != FORMAT or not document.get("episodes"):
                raise ValueError("incompatible historical or empty dataset schema")
            current = {"format": FORMAT, "execution": document["episodes"][0]["recording_contract"]}
            if contract is not None and current != contract:
                raise ValueError("selected Runs have incompatible dataset execution contracts")
            contract = current
            sources[f"{run}/{reference['attempt_id']}/{document['evaluation_id']}"] = {
                "manifest_sha256": reference["sha256"],
                "evaluation_id": document["evaluation_id"],
            }
            for episode in document["episodes"]:
                if selected_episode(episode, payload["filters"]):
                    inventory.append(episode)
        selection = {
            "format": FORMAT,
            "runs": payload["runs"],
            "sources": sources,
            "filters": payload["filters"],
            "episodes": [
                {
                    "episode_id": c["episode_id"],
                    "evaluation_id": c["evaluation_id"],
                    "sha256": canonical_json_sha256(c),
                }
                for c in inventory
            ],
        }
        selection_id = canonical_json_sha256(selection)
        from gradlab.dataset_shards import PublicationBudget, publish_sharded
        budget = PublicationBudget(store.root)
        work = store.root / "work" / str(job["job_id"]) / "dataset"
        work.mkdir(parents=True, exist_ok=True)

        def read(name, revision):
            try:
                return Path(hf_hub_download(
                    repo, name, repo_type="dataset", revision=revision,
                    cache_dir=work / "cache",
                )).read_bytes()
            except EntryNotFoundError:
                return None

        revision = publish_sharded(
            api, read, repo, contract, inventory, selection, models, work, budget,
            lambda: store.job(str(job["job_id"]))["cancel_requested"],
        )
        import shutil
        shutil.rmtree(work / "cache", ignore_errors=True)
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
    register_handler(JOB_TYPE, 2, DatasetPublicationHandler, replace=True)


def enqueue_publication(*, runs, repo, filters, repo_root, store=None, completed_snapshot=False):
    queue = store or JobStore()
    queue.init()
    from gradlab.operator_environment import load_repository_operator_environment
    from gradlab.r2_store import RunStorageConfig, R2Bucket

    load_repository_operator_environment(Path(repo_root))
    storage = RunStorageConfig.from_env()
    select = completed_snapshot_inventories if completed_snapshot else finalized_inventories
    inventories = select(R2Bucket(storage.control), R2Bucket(storage.models), runs)
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
        handler_version=2,
        payload=payload,
        idempotency_key=canonical_json_sha256(payload),
        subjects=[JobSubject("dataset", repo)],
    )
    worker = ensure_flusher(queue)
    return {"job": result.job, "created": result.created, "worker": worker.to_dict()}


def append_monitoring_episode(api, read, repo, contract, episode, models, work):
    """Stage bounded immutable chunks, then atomically expose a complete episode.

    The dataset's query surface is the episode table. Staged chunks are never
    listed there until all assets and transition tables have committed.
    """
    from gradlab.checkpoint_monitoring import FORMAT as MONITOR_FORMAT, verified_get
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not episode.get("complete") or episode.get("format") != MONITOR_FORMAT:
        raise ValueError("Publication requires a current complete monitoring episode")
    identity = f"{episode['evaluation_id']}/{episode['episode_id']}"
    digest = hashlib.sha256(identity.encode()).hexdigest()
    index = f"episodes/{digest}.json"
    encoded = canonical_json_bytes(episode)
    contract_data = canonical_json_bytes(contract)

    def compatible(head):
        existing = read("checkpoint-dataset.json", head)
        if existing is None and any(
            e.path != ".gitattributes"
            for e in api.list_repo_tree(repo, repo_type="dataset", revision=head)
        ):
            raise ValueError("HF target is not an empty checkpoint-monitoring dataset")
        if existing is not None and existing != contract_data:
            raise ValueError("incompatible checkpoint dataset execution contract")
        previous = read(index, head)
        if previous is not None and previous != encoded:
            raise ValueError("conflicting monitoring episode identity")

    head = api.repo_info(repo_id=repo, repo_type="dataset").sha
    compatible(head)
    if read(index, head) == encoded:
        return head
    _commit(api, read, repo, {"checkpoint-dataset.json": contract_data}, compatible=compatible)
    step, previous = 0, None
    references = []
    for chunk in episode["chunks"]:
        if (
            not 0 < chunk["bytes"] <= 128 * 1024**2
            or chunk["first_step"] != step
            or chunk["end_step"] <= step
            or previous is not None
            and previous != chunk["first_frame_sha256"]
        ):
            raise ValueError("invalid monitoring chunk size or episode join")
        payload = verified_get(models, chunk)
        archive_path = Path(work) / "chunk.zip"
        archive_path.write_bytes(payload)
        asset = f"chunks/{chunk['sha256']}.zip"
        table_name = f"transitions/{digest}/{chunk['sha256']}.parquet"
        with zipfile.ZipFile(archive_path) as archive:
            rows = []
            for raw in archive.read("transitions.jsonl").splitlines():
                row = json.loads(raw)
                if row["step"] != step:
                    raise ValueError("monitoring transition gap or duplicate")
                for frame in (f"frames/{step}.png", f"frames/{step + 1}.png"):
                    if frame not in archive.namelist():
                        raise ValueError("missing monitoring frame")
                rows.append(
                    dict(
                        trajectory_id=identity,
                        episode_id=episode["episode_id"],
                        evaluation_id=episode["evaluation_id"],
                        run_id=episode.get("run_id"),
                        training_seed=episode.get("training_seed"),
                        checkpoint_id=episode["checkpoint_id"],
                        checkpoint_step=episode["checkpoint_step"],
                        step=step,
                        archive=asset,
                        frame=f"frames/{step}.png",
                        next_frame=f"frames/{step + 1}.png",
                        transition_json=canonical_json_bytes(row).decode(),
                    )
                )
                step += 1
        if step != chunk["end_step"]:
            raise ValueError("monitoring chunk transition count mismatch")
        table = Path(work) / "chunk.parquet"
        pq.write_table(pa.Table.from_pylist(rows), table, compression="zstd")
        _commit(api, read, repo, {asset: archive_path, table_name: table}, compatible=compatible)
        references.append(table_name)
        previous = chunk["last_frame_sha256"]
        archive_path.unlink()
        table.unlink()
        import shutil

        shutil.rmtree(Path(work) / "cache", ignore_errors=True)
    if step != episode["steps"] or not references:
        raise ValueError("Publication cannot expose an incomplete monitoring episode")
    summary = Path(work) / "episode.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                dict(
                    trajectory_id=identity,
                    episode_id=episode["episode_id"],
                    evaluation_id=episode["evaluation_id"],
                    run_id=episode.get("run_id"),
                    training_seed=episode.get("training_seed"),
                    checkpoint_id=episode["checkpoint_id"],
                    checkpoint_step=episode["checkpoint_step"],
                    steps=episode["steps"],
                    native_score=episode["native_score"],
                    shaped_return=episode["shaped_return"],
                    normalized_brick_progress=episode["normalized_brick_progress"],
                    success=episode["success"],
                    transition_tables=references,
                )
            ]
        ),
        summary,
        compression="zstd",
    )
    head = _commit(
        api,
        read,
        repo,
        {index: encoded, f"episodes/{digest}.parquet": summary},
        compatible=compatible,
    )
    summary.unlink()
    return head
