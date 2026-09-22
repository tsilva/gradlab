"""Durable atomic publication of explicitly prepared trajectory split views."""

from pathlib import Path
import hashlib
import re
import time

from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from gradlab.dataset_shards import PublicationBudget, PublicationDeferred
from gradlab.job_queue import HandlerResult, JobStore, SubjectUpdate, register_handler

JOB_TYPE = "dataset-split-publication"


class SplitPublicationHandler:
    job_type = JOB_TYPE
    version = 1

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

    def advance(self, job):
        p = self.validate_payload(job["payload"])
        try:
            revision = self.publish(p, job["job_id"])
            return HandlerResult(
                state="succeeded",
                message=f"Published split revision {revision}",
                subjects=(
                    SubjectUpdate("dataset", p["repo"], "succeeded", {"revision": revision}),
                ),
            )
        except PublicationDeferred as exc:
            return HandlerResult(state="retry_wait", available_at=exc.until, message=str(exc))
        except ValueError, KeyError, TypeError:
            raise
        except Exception as exc:
            return HandlerResult(
                state="blocked" if job.get("attempts", 1) >= 8 else "retry_wait",
                available_at=time.time() + 30,
                message=f"{type(exc).__name__}: {exc}",
            )

    def publish(self, p, job_id):
        from gradlab.operator_environment import load_repository_operator_environment

        load_repository_operator_environment(Path(p["repo_root"]))
        root = Path(p["root"])
        api = HfApi()
        budget = PublicationBudget(p["queue_root"])
        # Validate all prepared bytes before any remote write.
        for name, digest in p["files"].items():
            with (root / name).open("rb") as f:
                if hashlib.file_digest(f, "sha256").hexdigest() != digest:
                    raise ValueError(f"Prepared split bytes changed: {name}")
        receipt = (root / p["receipt"]).read_bytes()

        def read_receipt(revision):
            try:
                return Path(
                    budget.call(
                        hf_hub_download,
                        p["repo"],
                        p["receipt"],
                        repo_type="dataset",
                        revision=revision,
                    )
                ).read_bytes()
            except EntryNotFoundError:
                return None

        with budget.writer(p["repo"]):
            head = budget.call(api.repo_info, repo_id=p["repo"], repo_type="dataset").sha
            previous = read_receipt(head)
            if previous is not None:
                if previous != receipt:
                    raise ValueError("Conflicting immutable split receipt")
                return head
            if head != p["parent"]:
                raise ValueError("Dataset head changed; review split publication before retry")
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
            budget.reserve("commit")
            try:
                return budget.call(
                    api.create_commit,
                    repo_id=p["repo"],
                    repo_type="dataset",
                    parent_commit=head,
                    operations=operations,
                    commit_message="Add balanced grouped 80/10/10 trajectory splits",
                ).oid
            except PublicationDeferred:
                raise
            except Exception:
                current = budget.call(api.repo_info, repo_id=p["repo"], repo_type="dataset").sha
                if read_receipt(current) == receipt:
                    return current
                raise


def register_job_handler():
    register_handler(JOB_TYPE, 1, SplitPublicationHandler, replace=True)
