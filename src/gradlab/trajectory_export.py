"""Resumable, disk-bounded conversion of canonical monitoring to trajectory tables."""

from __future__ import annotations

import hashlib
import gzip
import io
import json
import shutil
import sqlite3
import zipfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from gradlab.checkpoint_monitoring import verified_get
from gradlab.dataset_brick_labels import episode_issues, extract_wall_batch
from gradlab.dataset_shards import episode_identity, pack_plan
from gradlab.json_utils import canonical_json_bytes, canonical_json_sha256
from gradlab.trajectory_format import (
    episode_schema,
    frame_schema,
    session_schema,
    transition_schema,
    record_json,
    transition_record,
    TRAJECTORY_FORMAT,
    open_trajectory_parquet,
    require_current_trajectory_schema,
    trajectory_schema_contract,
    trajectory_schema_identity,
)
from gradlab.trajectory_dataset import table_inventory, verify_snapshot

FORMAT = TRAJECTORY_FORMAT
REFERENCE = "https://huggingface.co/datasets/tsilva/gradlab-breakout-trajectories/tree/9a22e4c0b6b9796a1358f36a6854f2569cbee0af"
MAX_DISK = 8 * 1024**3


def guard_space(root):
    used = sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())
    if used > MAX_DISK or shutil.disk_usage(root).free < 1024**3:
        raise ValueError("Trajectory export exceeds 8 GiB spool or 1 GiB free-space reserve")


def annotations(grid, support, flags, initial, native=None, prefix=""):
    values = dict(
        brick_grid=grid.tolist(),
        brick_grid_suspect=bool(flags),
        brick_grid_quality_flags=int(flags),
        brick_count_visible=int((grid == 1).sum()),
        brick_grid_unknown_cells=int((grid == -1).sum()),
        brick_grid_min_present_support=int(np.where(grid == 1, support, 48).min()),
        brick_count_mismatch=None if native is None else bool((grid == 1).sum() != native),
        is_initial_brick_layout=bool(initial),
    )
    return {
        prefix + ("is_brick_layout" if prefix and k == "is_initial_brick_layout" else k): v
        for k, v in values.items()
    }


class Converter:
    def __init__(self, root, selection):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "frames.sqlite3")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS frames(id INTEGER PRIMARY KEY, sha TEXT UNIQUE, image BLOB, grid BLOB, support BLOB, flags INTEGER)"
        )
        self.db.execute("CREATE TABLE IF NOT EXISTS png(sha TEXT PRIMARY KEY, frame INTEGER)")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS done(id INTEGER PRIMARY KEY, identity TEXT UNIQUE, summary TEXT, session TEXT)"
        )
        marker = self.root / "selection.json"
        value = canonical_json_bytes({**selection, **trajectory_schema_identity()})
        if marker.exists() and marker.read_bytes() != value:
            raise ValueError("Export selection changed")
        marker.write_bytes(value)

    def frame(self, png):
        digest = hashlib.sha256(png).hexdigest()
        row = self.db.execute(
            "SELECT f.id,f.grid,f.support,f.flags FROM png p JOIN frames f ON f.id=p.frame WHERE p.sha=?",
            (digest,),
        ).fetchone()
        if row:
            return (
                row[0],
                False,
                np.frombuffer(row[1], np.int8).reshape(6, 18),
                np.frombuffer(row[2], np.uint8).reshape(6, 18),
                row[3],
            )
        image = Image.open(io.BytesIO(png))
        if image.mode != "RGB" or image.size != (160, 210):
            raise ValueError("Expected full native unmasked RGB")
        rgb = np.asarray(image)
        header = canonical_json_bytes(
            dict(version=2, shape=[210, 160, 3], dtype="uint8", order="HWC", channels="RGB")
        )
        sha = hashlib.sha256(header + rgb.tobytes()).hexdigest()
        row = self.db.execute(
            "SELECT id,grid,support,flags FROM frames WHERE sha=?", (sha,)
        ).fetchone()
        new = row is None
        if new:
            out = io.BytesIO()
            image.save(out, format="WEBP", lossless=True, quality=100, method=4, exact=True)
            if not np.array_equal(np.asarray(Image.open(io.BytesIO(out.getvalue()))), rgb):
                raise ValueError("Lossless WebP roundtrip mismatch")
            grid, support, flags = extract_wall_batch(rgb[None, 57:93, 8:152])
            fid = self.db.execute(
                "INSERT INTO frames(sha,image,grid,support,flags) VALUES(?,?,?,?,?)",
                (sha, out.getvalue(), grid[0].tobytes(), support[0].tobytes(), int(flags[0])),
            ).lastrowid
            row = (fid, grid[0].tobytes(), support[0].tobytes(), int(flags[0]))
        self.db.execute("INSERT INTO png VALUES(?,?)", (digest, row[0]))
        return (
            row[0],
            new,
            np.frombuffer(row[1], np.int8).reshape(6, 18),
            np.frombuffer(row[2], np.uint8).reshape(6, 18),
            row[3],
        )

    def episode(self, index, episode, models):
        identity = episode_identity(episode)
        done = self.db.execute("SELECT identity FROM done WHERE id=?", (index,)).fetchone()
        if done:
            if done[0] != identity:
                raise ValueError("Export episode order changed")
            return
        guard_space(self.root)
        if not 0 < episode["steps"] <= 100000:
            raise ValueError("Episode exceeds bounded 100,000-transition conversion limit")
        pack_plan([episode])  # Validate complete chunk inventory before reading bytes.
        rows, frames = [], []
        contract = episode["recording_contract"]
        with self.db:
            for chunk in episode["chunks"]:
                with zipfile.ZipFile(io.BytesIO(verified_get(models, chunk))) as source:
                    first = source.read(f"frames/{chunk['first_step']}.png")
                    last = source.read(f"frames/{chunk['end_step']}.png")
                    if (
                        hashlib.sha256(first).hexdigest() != chunk["first_frame_sha256"]
                        or hashlib.sha256(last).hexdigest() != chunk["last_frame_sha256"]
                    ):
                        raise ValueError("Chunk boundary image checksum mismatch")
                    if not frames:
                        frames.append(self.frame(first))
                    elif self.frame(first)[0] != frames[-1][0]:
                        raise ValueError("Chunk frame join mismatch")
                    for raw in source.read("transitions.jsonl").splitlines():
                        source_row = json.loads(raw)
                        step = source_row["step"]
                        if step != len(rows) or step >= chunk["end_step"]:
                            raise ValueError("Transition gap or duplicate")
                        frames.append(self.frame(source.read(f"frames/{step + 1}.png")))
                        action = source_row["executed_action"]
                        if contract["native_encoding"][action] != source_row["native_action"]:
                            raise ValueError("Action encoding mismatch")
                        facts = dict(
                            episode_id=index,
                            session_id=identity,
                            step=step,
                            source_frame_id=frames[-2][0],
                            successor_frame_id=frames[-1][0],
                            successor_frame_new=frames[-1][1],
                            policy_decision_id=step,
                            configured_frame_skip=contract["environment"]["frame_skip"],
                            elapsed_native_frames=None,
                            temperature=None,
                            action_selection_mode="stochastic",
                            selected_action=source_row["policy_action"],
                            effective_action=source_row["effective_policy_action"],
                            native_action=action,
                            action_override_rule_id=source_row["override_rule"],
                            policy_reward=source_row["reward"],
                            native_reward=source_row["provider_reward"],
                            task_reward=None,
                            native_game_over=source_row["provider_terminated"],
                            native_truncated=source_row["provider_truncated"],
                            task_terminated=None,
                            task_truncated=None,
                            terminated=source_row["terminated"],
                            truncated=source_row["truncated"],
                            labels=source_row["facts"],
                            task_metrics=source_row["task_metrics"],
                            monitoring_record=source_row,
                        )
                        rows.append(facts)
                    if len(rows) != chunk["end_step"]:
                        raise ValueError("Chunk transition count mismatch")
            if len(rows) != episode["steps"] or not (
                rows[-1]["terminated"] or rows[-1]["truncated"]
            ):
                raise ValueError("Incomplete episode boundary")
            grids = np.stack([f[2] for f in frames])
            native = np.array([r["labels"]["bricks_remaining"] for r in rows])
            walls = np.array([r["labels"]["walls_cleared"] for r in rows])
            scores = np.array([r["labels"]["score"] for r in rows])
            flags, initial = episode_issues(grids, [f[4] for f in frames], native, walls, scores)
            # One episode is bounded by the recorded scientific episode cap; RGB stays on disk.
            table_rows = []
            for i, row in enumerate(rows):
                table_rows.append(
                    {
                        **transition_record(row),
                        **annotations(
                            grids[i + 1], frames[i + 1][3], flags[i + 1], initial[i + 1], native[i]
                        ),
                        "source_is_initial_brick_layout": bool(initial[i]),
                        "source_brick_grid_suspect": bool(flags[i]),
                    }
                )
            path = self.root / f"episode-{index:06d}.parquet"
            pq.write_table(
                pa.Table.from_pylist(table_rows, schema=transition_schema), path, compression="zstd"
            )
            summary = dict(
                episode_id=index,
                session_id=identity,
                seed=episode["environment_seed"],
                initial_frame_id=frames[0][0],
                initial_frame_new=frames[0][1],
                length=len(rows),
                split=None,
                status="complete",
                end_reason="environment_boundary",
                **annotations(grids[0], frames[0][3], flags[0], initial[0], prefix="initial_"),
            )
            session = record_json(
                dict(
                    session_id=identity,
                    monitoring_episode=episode,
                    export_format=FORMAT,
                    **trajectory_schema_identity(),
                    hud_mask=None,
                    split_assignment=None,
                )
            )
            self.db.execute(
                "INSERT INTO done VALUES(?,?,?,?)", (index, identity, json.dumps(summary), session)
            )

    def tables(self, output):
        """Stream compact shards, at most 100k transitions/frames in a file."""
        output = Path(output)
        for kind in ("frames/assets", "transitions/all", "episodes/all", "sessions/metadata"):
            (output / kind).mkdir(parents=True, exist_ok=True)
        cursor = self.db.execute("SELECT id,sha,image FROM frames ORDER BY id")
        number, count, writer = 0, 0, None
        try:
            while batch := cursor.fetchmany(512):
                if writer is None:
                    writer = pq.ParquetWriter(
                        output / f"frames/assets/{number:05d}.parquet",
                        frame_schema,
                        compression="zstd",
                    )
                writer.write_table(
                    pa.Table.from_pylist(
                        [
                            dict(frame_id=i, sha256=s, image=dict(bytes=b, path=None))
                            for i, s, b in batch
                        ],
                        schema=frame_schema,
                    )
                )
                count += len(batch)
                if count >= 100000:
                    writer.close()
                    writer = None
                    count = 0
                    number += 1
        finally:
            if writer:
                writer.close()
        number, count, writer = 0, 0, None
        try:
            for (index,) in self.db.execute("SELECT id FROM done ORDER BY id"):
                with open_trajectory_parquet(
                    self.root / f"episode-{index:06d}.parquet", "transitions"
                ) as parquet:
                    for batch in parquet.iter_batches(batch_size=2048):
                        if writer is None:
                            writer = pq.ParquetWriter(
                                output / f"transitions/all/{number:05d}.parquet",
                                transition_schema,
                                compression="zstd",
                            )
                        writer.write_batch(batch)
                        count += len(batch)
                        if count >= 100000:
                            writer.close()
                            writer = None
                            count = 0
                            number += 1
                guard_space(self.root)
        finally:
            if writer:
                writer.close()
        summaries, sessions = [], []
        for identity, summary, session in self.db.execute(
            "SELECT identity,summary,session FROM done ORDER BY id"
        ):
            summaries.append(json.loads(summary))
            sessions.append(dict(session_id=identity, record_json=session))
        pq.write_table(
            pa.Table.from_pylist(summaries, schema=episode_schema),
            output / "episodes/all/00000.parquet",
            compression="zstd",
        )
        pq.write_table(
            pa.Table.from_pylist(sessions, schema=session_schema),
            output / "sessions/metadata/00000.parquet",
            compression="zstd",
        )
        for name, schema in (("frames/assets", frame_schema), ("transitions/all", transition_schema)):
            if not any((output / name).glob("*.parquet")):
                pq.write_table(pa.Table.from_pylist([], schema=schema), output / name / "00000.parquet")
        return dict(
            episodes=len(summaries),
            transitions=sum(e["length"] for e in summaries),
            unique_frames=self.db.execute("SELECT count(*) FROM frames").fetchone()[0],
        )


def publish_trajectories(
    api, read, repo, contract, episodes, selection, models, work, budget, canceled
):
    """Publish a frozen converted snapshot atomically; retain previous revisions/assets."""
    from huggingface_hub import CommitOperationAdd

    selection = {**selection, "export_format": FORMAT, **trajectory_schema_identity()}
    sid = canonical_json_sha256(selection)
    prefix = f"trajectories/{sid}"
    marker = f"{prefix}/publication.json"
    root = Path(work) / "trajectory-export"
    root.mkdir(parents=True, exist_ok=True)
    with budget.writer(repo):
        budget.call(api.create_repo, repo_id=repo, repo_type="dataset", exist_ok=True)
        head = budget.call(api.repo_info, repo_id=repo, repo_type="dataset").sha
        previous = budget.call(read, marker, head)
        if previous is not None:
            receipt = json.loads(previous)
            require_current_trajectory_schema(receipt)
            if receipt["selection"] != selection:
                raise ValueError("Immutable trajectory contribution conflict")
            return head
        existing = budget.call(read, "checkpoint-dataset.json", head)
        if existing is not None and json.loads(existing) != contract:
            raise ValueError("Incompatible checkpoint dataset execution contract")
        tree = budget.call(
            lambda: list(
                api.list_repo_tree(repo, repo_type="dataset", revision=head, recursive=True)
            )
        )
        view = budget.call(read, "trajectory-view.json", head)
        previous_episodes = []
        if view is not None:
            view = json.loads(view)
            require_current_trajectory_schema(view)
            previous_receipt = json.loads(budget.call(read, view["publication"], head))
            require_current_trajectory_schema(previous_receipt)
            if previous_receipt.get("format") != FORMAT:
                raise ValueError("Incompatible trajectory export format")
            previous_index = view["episode_index"]
            previous_episodes = [
                json.loads(line)
                for line in gzip.decompress(budget.call(read, previous_index, head)).splitlines()
            ]
        else:
            # Explicit format migration must include earlier complete contributions,
            # even when this request selects only a later checkpoint or filtered subset.
            for entry in tree:
                if entry.path.startswith("indexes/") and entry.path.endswith(".jsonl.gz"):
                    previous_episodes.extend(
                        json.loads(line)
                        for line in gzip.decompress(
                            budget.call(read, entry.path, head)
                        ).splitlines()
                    )
                elif entry.path.startswith("episodes/") and entry.path.endswith(".json"):
                    previous_episodes.append(json.loads(budget.call(read, entry.path, head)))
        merged = {episode_identity(e): e for e in previous_episodes}
        for episode in episodes:
            identity = episode_identity(episode)
            if identity in merged and merged[identity] != episode:
                raise ValueError("Conflicting immutable episode")
            merged[identity] = episode
        episodes = list(merged.values())
        if existing is None and any(e.path != ".gitattributes" for e in tree):
            raise ValueError("HF target is not an empty or compatible monitoring dataset")
        converter = Converter(root, selection)
        try:
            for index, episode in enumerate(sorted(episodes, key=episode_identity), 1):
                if canceled():
                    raise ValueError("Trajectory publication canceled")
                converter.episode(index, episode, models)
            output = root / "output"
            stats = converter.tables(output)
        finally:
            converter.db.close()
        receipt = dict(
            **trajectory_schema_identity(),
            format=FORMAT,
            selection=selection,
            statistics=stats,
            reference=REFERENCE,
            hud_mask=None,
            split_assignment=None,
            source_revision=head,
            tables=table_inventory(output),
        )
        (output / "schema.json").write_bytes(canonical_json_bytes(trajectory_schema_contract()))
        (output / "publication.json").write_bytes(canonical_json_bytes(receipt))
        verify_snapshot(output)
        (output / "episodes.jsonl.gz").write_bytes(
            gzip.compress(
                b"".join(
                    canonical_json_bytes(e) + b"\n" for e in sorted(episodes, key=episode_identity)
                ),
                mtime=0,
            )
        )
        readme = ["---", "configs:"]
        for name, split in [
            ("transitions", "all"),
            ("frames", "assets"),
            ("episodes", "all"),
            ("sessions", "metadata"),
        ]:
            readme += [f"- config_name: {name}"]
            if name == "transitions":
                readme += ["  default: true"]
            readme += [
                "  data_files:",
                f"  - split: {split}",
                f"    path: {prefix}/{name}/{split}/*.parquet",
            ]
        readme += [
            "---",
            "# Breakout checkpoint trajectories",
            "",
            f"{stats['episodes']} complete episodes, {stats['transitions']} transitions, {stats['unique_frames']} unique lossless WebP RGB images.",
            "",
            f"Trajectory schema version {receipt['trajectory_schema_version']}; [machine-readable contract]({prefix}/schema.json).",
            "The publication receipt and every Parquet shard declare the version and contract fingerprint.",
            "",
            f"Table schemas and brick annotations match [the existing trajectories dataset]({REFERENCE}).",
            "The full 210×160 RGB image includes the HUD. There is no train/validation/test assignment;",
            "`all` is a single container split and the episode `split` column is null.",
            "Frame IDs are identifiers, never row offsets. `source_frame_id` and `successor_frame_id` join the frames table.",
            'Use `load_dataset(repo, "transitions", split="all")` and `load_dataset(repo, "frames", split="assets")`.',
            "Session `record_json` uses the collector tagged-tree codec and retains the complete original monitoring episode,",
            "including Run, training seed, Checkpoint, evaluation, episode seeds, recording contract, R2 hashes and start facts.",
            "Transition `record_json` retains original facts and the entire original `monitoring_record`.",
            "`native_action_json` is the executed provider action index, as in the reference collector; the internal",
            "emulator encoding is retained in `monitoring_record.native_action` and the session action contract.",
            "Unavailable separately recorded task reward/boundaries, temperature and elapsed native frames are null.",
            "Brick annotations use the reference breakout-bricks-v1 detector, including quality/initial-layout flags.",
            "No suspect frames are filtered or corrected. FirstWall episode boundaries are unchanged.",
            "R2 remains canonical. Earlier PNG export files and immutable HF revisions remain available for provenance.",
            "This publication replaces the current dataset view, not the original source recordings.",
            "",
        ]
        operations = []
        files = sorted(output.rglob("*.parquet")) + [
            output / "publication.json",
            output / "schema.json",
            output / "episodes.jsonl.gz",
        ]
        if len(files) + 2 + (existing is None) > 100:
            raise ValueError(
                "Snapshot exceeds bounded 100-file atomic publication; use larger shards"
            )
        for path in files:
            guard_space(root)
            op = CommitOperationAdd(
                path_in_repo=f"{prefix}/{path.relative_to(output).as_posix()}",
                path_or_fileobj=str(path),
            )
            budget.call(
                api.preupload_lfs_files, repo, repo_type="dataset", additions=[op], num_threads=2
            )
            operations.append(op)
        operations.append(
            CommitOperationAdd(
                path_in_repo="trajectory-view.json",
                path_or_fileobj=canonical_json_bytes(
                    {**trajectory_schema_identity(), "episode_index": f"{prefix}/episodes.jsonl.gz", "publication": marker}
                ),
            )
        )
        operations.append(
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj="\n".join(readme).encode())
        )
        if existing is None:
            operations.append(
                CommitOperationAdd(
                    path_in_repo="checkpoint-dataset.json",
                    path_or_fileobj=canonical_json_bytes(contract),
                )
            )
        if canceled():
            raise ValueError("Trajectory publication canceled")
        budget.reserve("commit")
        try:
            result = budget.call(
                api.create_commit,
                repo_id=repo,
                repo_type="dataset",
                parent_commit=head,
                operations=operations,
                commit_message="Publish unsplit lossless WebP trajectory tables with full HUD",
            )
            return result.oid
        except Exception:
            from gradlab.dataset_shards import PublicationDeferred
            import sys

            if isinstance(sys.exception(), PublicationDeferred):
                raise
            current = budget.call(api.repo_info, repo_id=repo, repo_type="dataset").sha
            committed = budget.call(read, marker, current)
            if committed == canonical_json_bytes(receipt):
                return current
            raise
