"""Durable atomic publication of explicitly prepared trajectory split views."""

from pathlib import Path
import hashlib
import json
import re

from huggingface_hub import CommitOperationAdd, hf_hub_download

from gradlab.dataset_publication import AtomicDatasetPublication
from gradlab.job_queue import JobStore, register_handler
from gradlab.trajectory_format import open_trajectory_parquet, require_current_trajectory_schema

JOB_TYPE = "dataset-split-publication"


class SplitPublicationHandler(AtomicDatasetPublication):
    job_type = JOB_TYPE
    publication_kind = "split"
    commit_message = "Add balanced grouped 80/10/10 trajectory splits"
    parent_changed_message = "Dataset head changed; review split publication before retry"

    @classmethod
    def validate_payload(cls, payload):
        if set(payload) != {
            "repo",
            "parent",
            "root",
            "queue_root",
            "repo_root",
            "files",
            "receipt",
        }:
            raise ValueError("Malformed split publication request")
        if not re.fullmatch(r"[\w-]+/[\w.-]+", payload["repo"]) or not re.fullmatch(
            r"[0-9a-f]{40}", payload["parent"]
        ):
            raise ValueError("Split publication requires a repository and immutable parent")
        if not 1 <= len(payload["files"]) <= 100 or payload["receipt"] not in payload["files"]:
            raise ValueError("Split snapshot must have a receipt and at most 100 files")
        if not payload["receipt"].startswith("splits/") or not payload["receipt"].endswith(
            "/manifest.json"
        ):
            raise ValueError("Invalid split receipt")
        prefix = payload["receipt"].removesuffix("manifest.json")
        for name, digest in payload["files"].items():
            if (
                ".." in Path(name).parts
                or not (name == "README.md" or name.startswith(prefix))
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
            ):
                raise ValueError("Invalid split file identity")
        for key in ("root", "queue_root", "repo_root"):
            if not Path(payload[key]).is_absolute():
                raise ValueError("Split publication paths must be absolute")
        return dict(payload)

    def publish(self, p, job_id):
        from gradlab.operator_environment import load_repository_operator_environment

        load_repository_operator_environment(Path(p["repo_root"]))
        root = Path(p["root"])
        # Validate all prepared bytes before any remote write.
        for name, digest in p["files"].items():
            with (root / name).open("rb") as f:
                if hashlib.file_digest(f, "sha256").hexdigest() != digest:
                    raise ValueError(f"Prepared split bytes changed: {name}")
        receipt = (root / p["receipt"]).read_bytes()
        manifest = json.loads(receipt)
        require_current_trajectory_schema(manifest)
        if manifest.get("source_revision") != p["parent"]:
            raise ValueError("Split manifest must pin the requested source revision")
        prefix = p["receipt"].removesuffix("manifest.json")
        inventory = {}
        for name, digest in p["files"].items():
            if not name.endswith(".parquet"):
                continue
            parts = Path(name.removeprefix(prefix)).parts
            if len(parts) != 3 or parts[0] not in {"transitions", "episodes"}:
                raise ValueError("Split Parquet must be a transitions or episodes table")
            with open_trajectory_parquet(root / name, parts[0]) as parquet:
                inventory[name] = {"table": parts[0], "rows": parquet.metadata.num_rows,
                                   "sha256": digest}
        if {item["table"] for item in inventory.values()} != {"transitions", "episodes"}:
            raise ValueError("Split snapshot requires transitions and episodes tables")
        if manifest.get("tables") != inventory:
            raise ValueError("Split manifest table inventory differs from prepared shards")

        return self.commit(p, job_id, receipt)

    def prepare_commit(self, api, budget, p, job_id, head):
        root = Path(p["root"])

        def read_json(name, revision):
            return json.loads(Path(budget.call(
                hf_hub_download, p["repo"], name, repo_type="dataset", revision=revision,
            )).read_bytes())

        view = read_json("trajectory-view.json", head)
        require_current_trajectory_schema(view)
        require_current_trajectory_schema(read_json(view["publication"], head))
        operations = []
        for name in sorted(p["files"]):
            if JobStore(p["queue_root"]).job(job_id)["cancel_requested"]:
                raise ValueError("Split publication canceled")
            op = CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(root / name))
            budget.call(
                api.preupload_lfs_files,
                p["repo"],
                repo_type="dataset",
                additions=[op],
                num_threads=2,
            )
            operations.append(op)
        return operations


def register_job_handler():
    register_handler(JOB_TYPE, 1, SplitPublicationHandler, replace=True)
