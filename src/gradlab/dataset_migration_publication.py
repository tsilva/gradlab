"""Publish verified schema migrations without replacing historical dataset paths."""

import json
from pathlib import Path
import re
import time

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from gradlab.dataset_shards import PublicationBudget, PublicationDeferred
from gradlab.file_utils import file_sha256
from gradlab.job_queue import HandlerResult, JobStore, SubjectUpdate, register_handler
from gradlab.trajectory_dataset import verify_snapshot
from gradlab.trajectory_format import open_trajectory_parquet, require_current_trajectory_schema

JOB_TYPE = "dataset-migration-publication"


class MigrationPublicationHandler:
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
            raise ValueError("Malformed migration publication request")
        if not re.fullmatch(r"[\w-]+/[\w.-]+", payload["repo"]) or not re.fullmatch(
            r"[0-9a-f]{40}", payload["parent"]
        ):
            raise ValueError("Migration requires a repository and immutable parent")
        match = re.fullmatch(r"migrations/([0-9a-f]{64})/manifest.json", payload["receipt"])
        if match is None or payload["receipt"] not in payload["files"]:
            raise ValueError("Missing migration receipt")
        identity = match[1]
        if not 4 <= len(payload["files"]) <= 100:
            raise ValueError("Migration requires at most 100 atomic file operations")
        prefixes = (f"trajectories/{identity}/", f"splits/{identity}/", f"migrations/{identity}/")
        for name, digest in payload["files"].items():
            if (
                ".." in Path(name).parts
                or not (name in {"README.md", "trajectory-view.json"} or name.startswith(prefixes))
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
            ):
                raise ValueError("Invalid migration file identity")
        if not {"README.md", "trajectory-view.json"} <= payload["files"].keys():
            raise ValueError("Migration must atomically switch both public views")
        for key in ("root", "queue_root", "repo_root"):
            if not Path(payload[key]).is_absolute():
                raise ValueError("Migration paths must be absolute")
        return dict(payload)

    def advance(self, job):
        p = self.validate_payload(job["payload"])
        try:
            revision = self.publish(p, job["job_id"])
            return HandlerResult(
                state="succeeded",
                message=f"Published migration revision {revision}",
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
        import yaml

        p = self.validate_payload(p)
        root = Path(p["root"])
        if sum((root / name).stat().st_size for name in p["files"]) > 8 * 1024**3:
            raise ValueError("Prepared migration exceeds the 8 GiB publication bound")
        for name, digest in p["files"].items():
            if file_sha256(root / name) != digest:
                raise ValueError(f"Prepared migration bytes changed: {name}")
        receipt_bytes = (root / p["receipt"]).read_bytes()
        receipt = json.loads(receipt_bytes)
        require_current_trajectory_schema(receipt)
        if receipt["source_revision"] != p["parent"] or receipt["source_repo"] != p["repo"]:
            raise ValueError("Migration provenance differs from requested source")
        if receipt["files"] != {k: v for k, v in p["files"].items() if k != p["receipt"]}:
            raise ValueError("Migration receipt file inventory differs")
        identity = p["receipt"].split("/")[1]
        base, split = f"trajectories/{identity}", f"splits/{identity}"
        verify_snapshot(root / base)
        publication = json.loads((root / base / "publication.json").read_bytes())
        if publication["migration"]["source_revision"] != p["parent"]:
            raise ValueError("Base migration source revision differs")
        view = json.loads((root / "trajectory-view.json").read_bytes())
        require_current_trajectory_schema(view)
        if (
            view["publication"] != f"{base}/publication.json"
            or view["episode_index"] != f"{base}/episodes.jsonl.gz"
        ):
            raise ValueError("Migration view must reference its new immutable snapshot")
        manifest = json.loads((root / split / "manifest.json").read_bytes())
        require_current_trajectory_schema(manifest)
        if (
            manifest["source_revision"] != p["parent"]
            or manifest["source_publication"] != view["publication"]
        ):
            raise ValueError("Split migration source differs")
        inventory = {}
        counts = {}
        for name, digest in p["files"].items():
            if name.startswith(split + "/") and name.endswith(".parquet"):
                parts = name.split("/")
                if len(parts) != 5 or parts[2] not in {"episodes", "transitions"}:
                    raise ValueError("Invalid split table path")
                with open_trajectory_parquet(root / name, parts[2]) as parquet:
                    rows = parquet.metadata.num_rows
                inventory[name] = {"table": parts[2], "rows": rows, "sha256": digest}
                key = (parts[3], parts[2])
                counts[key] = counts.get(key, 0) + rows
        if manifest["tables"] != inventory or not inventory:
            raise ValueError("Split migration inventory differs")
        for partition, stats in manifest["statistics"].items():
            for kind in ("episodes", "transitions"):
                if counts.get((partition, kind)) != stats[kind]:
                    raise ValueError("Split migration row counts differ")
        card = (root / "README.md").read_text()
        frontmatter = yaml.safe_load(card.split("---", 2)[1])
        expected = {
            "transitions": {f"{split}/transitions/{s}/*.parquet" for s in manifest["statistics"]},
            "episodes": {f"{split}/episodes/{s}/*.parquet" for s in manifest["statistics"]},
            "frames": {f"{base}/frames/assets/*.parquet"},
            "sessions": {f"{base}/sessions/metadata/*.parquet"},
        }
        actual = {
            c["config_name"]: {f["path"] for f in c["data_files"]} for c in frontmatter["configs"]
        }
        if actual != expected:
            raise ValueError("Dataset card must reference exactly the migrated tables")
        load_repository_operator_environment(Path(p["repo_root"]))
        api, budget = HfApi(), PublicationBudget(p["queue_root"])

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
                if previous != receipt_bytes:
                    raise ValueError("Conflicting immutable migration receipt")
                return head
            if head != p["parent"]:
                raise ValueError("Dataset head changed; revalidate migration before publication")
            existing = set(
                budget.call(api.list_repo_files, p["repo"], repo_type="dataset", revision=head)
            )
            if existing & (p["files"].keys() - {"README.md", "trajectory-view.json"}):
                raise ValueError("Migration would overwrite immutable dataset files")
            if JobStore(p["queue_root"]).job(job_id)["cancel_requested"]:
                raise ValueError("Migration publication canceled")
            operations = [
                CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(root / name))
                for name in sorted(p["files"])
            ]
            budget.call(
                api.preupload_lfs_files,
                p["repo"],
                repo_type="dataset",
                additions=operations,
                num_threads=2,
            )
            if JobStore(p["queue_root"]).job(job_id)["cancel_requested"]:
                raise ValueError("Migration publication canceled")
            budget.reserve("commit")
            try:
                return budget.call(
                    api.create_commit,
                    repo_id=p["repo"],
                    repo_type="dataset",
                    parent_commit=head,
                    operations=operations,
                    commit_message="Migrate trajectory tables to schema v1; preserve grouped splits",
                ).oid
            except PublicationDeferred:
                raise
            except Exception:
                current = budget.call(api.repo_info, repo_id=p["repo"], repo_type="dataset").sha
                if read_receipt(current) == receipt_bytes:
                    return current
                raise


def register_job_handler():
    register_handler(JOB_TYPE, 1, MigrationPublicationHandler, replace=True)
