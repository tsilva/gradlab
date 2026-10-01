"""Queue retries and expected-parent commits for prepared dataset publications."""

import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from gradlab.dataset_shards import PublicationBudget, PublicationDeferred
from gradlab.job_queue import HandlerResult, SubjectUpdate


class AtomicDatasetPublication:
    version = 1

    def advance(self, job):
        payload = self.validate_payload(job["payload"])
        try:
            revision = self.publish(payload, job["job_id"])
            return HandlerResult(
                state="succeeded",
                message=f"Published {self.publication_kind} revision {revision}",
                subjects=(
                    SubjectUpdate("dataset", payload["repo"], "succeeded", {"revision": revision}),
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

    def commit(self, payload, job_id, receipt):
        """Reconcile immutable receipts before and after an atomic remote commit."""
        api = HfApi()
        budget = PublicationBudget(payload["queue_root"])

        def read_receipt(revision):
            try:
                return Path(
                    budget.call(
                        hf_hub_download,
                        payload["repo"],
                        payload["receipt"],
                        repo_type="dataset",
                        revision=revision,
                    )
                ).read_bytes()
            except EntryNotFoundError:
                return None

        with budget.writer(payload["repo"]):
            head = budget.call(api.repo_info, repo_id=payload["repo"], repo_type="dataset").sha
            previous = read_receipt(head)
            if previous is not None:
                if previous != receipt:
                    raise ValueError(f"Conflicting immutable {self.publication_kind} receipt")
                return head
            if head != payload["parent"]:
                raise ValueError(self.parent_changed_message)
            operations = self.prepare_commit(api, budget, payload, job_id, head)
            budget.reserve("commit")
            try:
                return budget.call(
                    api.create_commit,
                    repo_id=payload["repo"],
                    repo_type="dataset",
                    parent_commit=head,
                    operations=operations,
                    commit_message=self.commit_message,
                ).oid
            except PublicationDeferred:
                raise
            except Exception:
                current = budget.call(
                    api.repo_info, repo_id=payload["repo"], repo_type="dataset"
                ).sha
                if read_receipt(current) == receipt:
                    return current
                raise
