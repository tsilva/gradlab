"""Bounded HF exports: stage bytes separately from atomic dataset publication."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import gzip
import io
import json
import sqlite3
import time
import zipfile
from pathlib import Path

from huggingface_hub import CommitOperationAdd
from huggingface_hub.errors import HfHubHTTPError

from gradlab.checkpoint_monitoring import FORMAT, verified_get
from gradlab.json_utils import canonical_json_bytes, canonical_json_sha256

SHARD_BYTES = 512 * 1024**2
MAX_OPERATIONS = 50


class PublicationDeferred(Exception):
    def __init__(self, until, message):
        super().__init__(message)
        self.until = until


class PublicationBudget:
    """Persistent shared dataset budget; reservations survive lost acknowledgements."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "hf-publication.sqlite3"
        with self.connection() as db:
            db.execute("CREATE TABLE IF NOT EXISTS requests (kind TEXT, at REAL)")
            db.execute("CREATE TABLE IF NOT EXISTS cooldown (id INTEGER PRIMARY KEY, until REAL)")

    @contextlib.contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    def defer(self, until):
        with self.connection() as db:
            db.execute(
                "INSERT INTO cooldown VALUES (1, ?) ON CONFLICT(id) DO UPDATE SET until=max(until, excluded.until)",
                (until,),
            )

    def reserve(self, kind="request"):
        now = time.time()
        window, limit = (3600, 10) if kind == "commit" else (300, 60)
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT until FROM cooldown WHERE id=1").fetchone()
            if row and row[0] > now:
                raise PublicationDeferred(row[0], "Shared HF cooldown")
            db.execute("DELETE FROM requests WHERE at < ?", (now - 3600,))
            recent = [
                r[0]
                for r in db.execute(
                    "SELECT at FROM requests WHERE kind=? AND at>? ORDER BY at",
                    (kind, now - window),
                )
            ]
            if len(recent) >= limit:
                raise PublicationDeferred(recent[0] + window + 5, f"HF {kind} budget")
            db.execute("INSERT INTO requests VALUES (?, ?)", (kind, now))

    def call(self, fn, *args, **kwargs):
        self.reserve()
        try:
            return fn(*args, **kwargs)
        except HfHubHTTPError as exc:
            if exc.response is None or exc.response.status_code != 429:
                raise
            # The action quota can be longer than the generic API reset header.
            delay = 3600 if "repository commits" in str(exc) else 300
            headers = exc.response.headers
            try:
                delay = max(delay, float(headers.get("retry-after", 0)))
            except ValueError:
                from email.utils import parsedate_to_datetime

                try:
                    delay = max(
                        delay,
                        parsedate_to_datetime(headers["retry-after"]).timestamp() - time.time(),
                    )
                except ValueError, KeyError, TypeError:
                    pass
            import re

            for value in re.findall(r"(?:^|[;,])\s*t=(\d+)", headers.get("ratelimit", "")):
                delay = max(delay, int(value))
            until = time.time() + delay + 15
            self.defer(until)
            raise PublicationDeferred(until, "HF rate limit; shared publisher paused") from exc

    @contextlib.contextmanager
    def writer(self, repo):
        name = hashlib.sha256(repo.encode()).hexdigest()
        with (self.root / f"hf-writer-{name}.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise PublicationDeferred(
                    time.time() + 30, "Another dataset writer owns this repository"
                )
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


def episode_identity(episode):
    return f"{episode['evaluation_id']}/{episode['episode_id']}"


def pack_plan(episodes, limit=SHARD_BYTES):
    """Stable groups of canonical chunks, including episodes longer than a shard."""
    groups, group, size = [], [], 0
    for episode in sorted(episodes, key=episode_identity):
        step, previous = 0, None
        if not episode.get("complete") or episode.get("format") != FORMAT:
            raise ValueError("Cannot publish an incomplete episode")
        for chunk in episode["chunks"]:
            if (
                not 0 < chunk["bytes"] <= 128 * 1024**2
                or chunk["first_step"] != step
                or chunk["end_step"] <= step
                or previous is not None
                and previous != chunk["first_frame_sha256"]
            ):
                raise ValueError("Invalid chunk size or episode continuity")
            if group and size + chunk["bytes"] > limit:
                groups.append(group)
                group, size = [], 0
            group.append((episode, chunk))
            size += chunk["bytes"]
            step, previous = chunk["end_step"], chunk["last_frame_sha256"]
        if not episode["chunks"] or step != episode["steps"]:
            raise ValueError("Cannot publish an incomplete episode")
    if group:
        groups.append(group)
    return groups


def build_shard(group, models, work):
    """Copy original PNG bytes; stream transition row groups one source chunk at a time."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    identity = canonical_json_sha256([[episode_identity(e), c["sha256"]] for e, c in group])
    asset = f"shards/{identity}.zip"
    table_name = f"transitions/shard-{identity}.parquet"
    archive_path, table_path = Path(work) / "shard.zip", Path(work) / "shard.parquet"
    writer = None
    written = set()
    try:
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as output:
            for episode, chunk in group:
                payload = verified_get(models, chunk)
                with zipfile.ZipFile(io.BytesIO(payload)) as source:
                    prefix = chunk["sha256"]
                    rows = []
                    step = chunk["first_step"]
                    names = set(source.namelist())
                    for raw in source.read("transitions.jsonl").splitlines():
                        row = json.loads(raw)
                        if row["step"] != step:
                            raise ValueError("Transition gap or duplicate")
                        for n in (step, step + 1):
                            if f"frames/{n}.png" not in names:
                                raise ValueError("Missing monitoring frame")
                        rows.append(
                            dict(
                                trajectory_id=episode_identity(episode),
                                episode_id=episode["episode_id"],
                                evaluation_id=episode["evaluation_id"],
                                run_id=episode.get("run_id"),
                                training_seed=episode.get("training_seed"),
                                checkpoint_id=episode["checkpoint_id"],
                                checkpoint_step=episode["checkpoint_step"],
                                step=step,
                                archive=asset,
                                frame=f"{prefix}/frames/{step}.png",
                                next_frame=f"{prefix}/frames/{step + 1}.png",
                                transition_json=canonical_json_bytes(row).decode(),
                            )
                        )
                        step += 1
                    if step != chunk["end_step"]:
                        raise ValueError("Transition count mismatch")
                    if prefix not in written:
                        for n in range(chunk["first_step"], chunk["end_step"] + 1):
                            info = zipfile.ZipInfo(
                                f"{prefix}/frames/{n}.png", date_time=(1980, 1, 1, 0, 0, 0)
                            )
                            output.writestr(info, source.read(f"frames/{n}.png"))
                        written.add(prefix)
                    table = pa.Table.from_pylist(rows)
                    if writer is None:
                        writer = pq.ParquetWriter(table_path, table.schema, compression="zstd")
                    writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    return {asset: archive_path, table_name: table_path}


def publish_sharded(api, read, repo, contract, episodes, selection, models, work, budget, canceled):
    """Append compact shards alongside existing complete episode contributions."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    work = Path(work)
    selection_id = canonical_json_sha256(selection)
    contribution = f"contributions/{selection_id}.json"
    contract_bytes = canonical_json_bytes(contract)
    from gradlab.trajectory_publication import DATASET_README

    with budget.writer(repo):
        budget.call(api.create_repo, repo_id=repo, repo_type="dataset", exist_ok=True)
        head = budget.call(api.repo_info, repo_id=repo, repo_type="dataset").sha

        # One pinned directory inventory, then only small metadata reads.
        def tree(revision):
            return {
                e.path: e
                for e in budget.call(
                    lambda: list(
                        api.list_repo_tree(
                            repo, repo_type="dataset", revision=revision, recursive=True
                        )
                    )
                )
            }

        files = tree(head)

        def get(name, revision):
            cache = work / "metadata" / hashlib.sha256((revision + "/" + name).encode()).hexdigest()
            if cache.exists():
                return cache.read_bytes()
            data = budget.call(read, name, revision)
            if data is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_bytes(data)
            return data

        existing = (
            get("checkpoint-dataset.json", head) if "checkpoint-dataset.json" in files else None
        )
        if existing is not None and existing != contract_bytes:
            raise ValueError("Incompatible checkpoint dataset execution contract")
        if existing is None and set(files) - {".gitattributes"}:
            raise ValueError("HF target is not empty")
        if contribution in files:
            if get(contribution, head) != canonical_json_bytes(selection):
                raise ValueError("Conflicting immutable contribution")
            return head
        published = {}
        # Both forms are explicit supported indexes in the additive dataset layout.
        for name in files:
            if name.startswith("episodes/") and name.endswith(".json"):
                e = json.loads(get(name, head))
                published[episode_identity(e)] = canonical_json_sha256(e)
            elif name.startswith("indexes/") and name.endswith(".jsonl.gz"):
                for line in gzip.decompress(get(name, head)).splitlines():
                    e = json.loads(line)
                    published[episode_identity(e)] = canonical_json_sha256(e)
        pending = []
        for e in episodes:
            previous = published.get(episode_identity(e))
            if previous is not None and previous != canonical_json_sha256(e):
                raise ValueError("Conflicting episode identity")
            if previous is None:
                pending.append(e)
        # Persist the chosen groups so an interrupted export cannot repack its own
        # already-published episodes into different shards on the next attempt.
        journal = work / "plan.json"
        if journal.exists():
            plan = json.loads(journal.read_bytes())
            if plan["selection"] != selection_id:
                raise ValueError("Publication journal selection changed")
            by_id = {episode_identity(e): e for e in episodes}
            pending = [by_id[i] for i in plan["episodes"] if i not in published]
        else:
            temp = journal.with_suffix(".tmp")
            temp.write_bytes(
                canonical_json_bytes(
                    {"selection": selection_id, "episodes": [episode_identity(e) for e in pending]}
                )
            )
            temp.replace(journal)
        additions, expected, summaries = [], {}, {}

        def commit():
            nonlocal head, files, additions, expected
            if not additions:
                return
            budget.reserve("commit")
            try:
                response = budget.call(
                    api.create_commit,
                    repo_id=repo,
                    repo_type="dataset",
                    parent_commit=head,
                    operations=additions,
                    commit_message="Append verified checkpoint dataset snapshot",
                )
                head = response.oid
                additions, expected = [], {}
                files = tree(head)
                return
            except PublicationDeferred:
                raise
            except Exception:
                # A timed-out response can already have committed. Check the
                # immutable paths before reconstructing SDK operations on retry.
                head = budget.call(api.repo_info, repo_id=repo, repo_type="dataset").sha
                files = tree(head)
                if all(
                    name in files and remote_matches(name, digest)
                    for name, digest in expected.items()
                ):
                    additions, expected = [], {}
                    return
                # Mutated preupload objects must not be reused for another commit.
                raise

        def remote_matches(name, digest):
            entry = files[name]
            lfs = getattr(entry, "lfs", None)
            if lfs:
                return getattr(lfs, "sha256", None) == digest or (
                    isinstance(lfs, dict) and lfs.get("sha256") == digest
                )
            return hashlib.sha256(get(name, head)).hexdigest() == digest

        def stage(mapping):
            for name, value in mapping.items():
                if canceled():
                    raise PublicationDeferred(time.time(), "Dataset publication canceled")
                data = value if isinstance(value, bytes) else Path(value).read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                if name in files:
                    if not remote_matches(name, digest):
                        raise ValueError(f"Immutable dataset conflict: {name}")
                    continue
                op = CommitOperationAdd(path_in_repo=name, path_or_fileobj=data)
                # Supported preupload API frees binary payload while preserving the
                # operation for this one commit. No private SDK state is persisted.
                budget.call(
                    api.preupload_lfs_files,
                    repo,
                    repo_type="dataset",
                    additions=[op],
                    num_threads=2,
                )
                if len(data) > 1024**2 and op.path_or_fileobj != b"":
                    raise ValueError(
                        "HF must track large dataset files as LFS/Xet to bound upload memory"
                    )
                additions.append(op)
                expected[name] = digest
                if len(additions) == MAX_OPERATIONS:
                    commit()

        stage({"checkpoint-dataset.json": contract_bytes})
        for group in pack_plan(pending):
            if canceled():
                raise PublicationDeferred(time.time(), "Dataset publication canceled")
            mapping = build_shard(group, models, work)
            table_name = next(k for k in mapping if k.endswith(".parquet"))
            for e, _ in group:
                refs = summaries.setdefault(episode_identity(e), [])
                if table_name not in refs:
                    refs.append(table_name)
            stage(mapping)
            for path in mapping.values():
                path.unlink()
        # Publish all indexes and the contribution together; never split the
        # visibility markers across separate commits at a batch boundary.
        if len(additions) + (4 if pending else 2) > MAX_OPERATIONS:
            commit()
        # No index is staged until every asset has been uploaded successfully.
        if pending:
            rows = []
            for e in pending:
                rows.append(
                    {
                        **{
                            k: e.get(k)
                            for k in (
                                "episode_id",
                                "evaluation_id",
                                "run_id",
                                "training_seed",
                                "checkpoint_id",
                                "checkpoint_step",
                                "steps",
                                "native_score",
                                "shaped_return",
                                "normalized_brick_progress",
                                "success",
                            )
                        },
                        "trajectory_id": episode_identity(e),
                        "transition_tables": summaries[episode_identity(e)],
                    }
                )
            table_path = work / "episodes.parquet"
            pq.write_table(pa.Table.from_pylist(rows), table_path, compression="zstd")
            stage(
                {
                    f"episodes/snapshot-{selection_id}.parquet": table_path,
                    f"indexes/{selection_id}.jsonl.gz": gzip.compress(
                        b"".join(canonical_json_bytes(e) + b"\n" for e in pending), mtime=0
                    ),
                }
            )
            table_path.unlink()
        stage({contribution: canonical_json_bytes(selection), "README.md": DATASET_README})
        commit()
        return head
