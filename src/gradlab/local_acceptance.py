"""Import a completed overlay Run and queue its final checkpoint privately."""

import json
import os
import shutil
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from urllib.parse import unquote, urlparse

from gradlab.cli_parser import ExactArgumentParser
from gradlab.evaluation_attempts import promotion_receipt
from gradlab.evaluation_fence import evaluation_selection_fence
from gradlab.file_utils import file_sha256
from gradlab.job_queue import JobStore
from gradlab.local_publication import publication_storage_authority, publish_local_run
from gradlab.manual_evaluation import ManualEvaluationQueue, private_evaluation_authority
from gradlab.modal_eval_protocol import normalize_attempt_result
from gradlab.policy_bundle import load_policy_bundle_from_checkpoint
from gradlab.publication_evidence import validate_publication_evidence
from gradlab.rom_assets import verify_rom_file
from gradlab.run_authority import RunAuthority
from gradlab.run_contracts import (
    RUN_ID_PATTERN,
    CheckpointManifest,
    EvalIntent,
    EvalResult,
    RunManifest,
    document_sha256,
)
from gradlab.source_overlay import read_source_overlay


def publish_accepted_local_run(
    run_dir: Path, queue_dir: Path, *, authority: RunAuthority | None = None
) -> str:
    """Publish a local Run and transfer its exact verified Acceptance evidence."""
    origin = private_evaluation_authority(queue_dir)
    receipt = json.loads((run_dir / "local-run.json").read_bytes())
    run_id = receipt["run_id"]
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("local Run ID is invalid")
    manifest = RunManifest.from_dict(origin.manifest(run_id))
    overlay = manifest.compute["source_overlay"]
    prefix = f"runs/{run_id}"
    promotion = origin.control.get_json(f"{prefix}/promotion.json")
    key = promotion["eval_idempotency_key"]
    intent = EvalIntent.from_dict(origin.eval_intent(run_id=run_id, idempotency_key=key))
    raw = origin.eval_result(run_id=run_id, idempotency_key=key)
    result = EvalResult.from_dict(origin.evaluation.get_json(f"{prefix}/evals/{key}/verified-result.json"))
    if promotion["eval_result_sha256"] != document_sha256(result.to_dict()):
        raise ValueError("private Promotion does not bind the verified evaluation")
    model = run_dir / "final_model.zip"
    bundle = load_policy_bundle_from_checkpoint(model)
    index = origin.models.get_json(f"{prefix}/index.json")
    [checkpoint] = [
        row for row in index["checkpoints"]
        if row["purpose"] == "final" and row["checkpoint_id"] == intent.checkpoint_id
    ]
    if (
        bundle is None
        or bundle.checkpoint_sha256 != checkpoint["sha256"]
        or file_sha256(bundle.model_path) != checkpoint["model_document_sha256"]
        or file_sha256(bundle.recipe_path) != checkpoint["recipe_document_sha256"]
    ):
        raise ValueError("original final checkpoint or its sidecars differ from Acceptance")
    evidence = {
        "recipe": bundle.recipe, "intent": intent.to_dict(), "raw_result": raw,
        "verified_result": result.to_dict(), "checkpoint_manifest": checkpoint,
    }
    validate_publication_evidence(evidence)
    video = raw.get("video")
    if result.video != video:
        raise ValueError("private raw and verified Acceptance video bindings differ")
    video_path = None
    if video:
        video_key = origin.evaluation.key_from_uri(video["object_uri"])
        uri = urlparse(origin.evaluation.uri(video_key))
        video_path = Path(unquote(uri.path))
        if file_sha256(video_path) != video["sha256"] or video_path.stat().st_size != video["bytes"]:
            raise ValueError("private Acceptance video differs from its receipt")
    if authority is None:
        authority = publication_storage_authority()
    staging = queue_dir / "publication" / document_sha256(authority.storage.manifest_locations())
    imported = import_run_copy(run_dir, staging, run_id)
    index_url = publish_local_run(imported, authority=authority, source_overlay=overlay)
    published = authority.models.get_json(f"{prefix}/index.json")
    [target] = [
        CheckpointManifest.from_dict(row) for row in published["checkpoints"]
        if row["checkpoint_id"] == intent.checkpoint_id
    ]
    if target.model_document_sha256 != checkpoint["model_document_sha256"]:
        raise ValueError("published model document differs from Acceptance")
    evidence["checkpoint_manifest"] = target.to_dict()
    validate_publication_evidence(evidence)
    published_manifest = RunManifest.from_dict(authority.manifest(run_id))
    lease = authority.acquire_lease(
        run_id=run_id, attempt_id=published_manifest.attempt_id,
        holder_id=f"local-acceptance-import-{key[:20]}",
    )
    try:
        if video_path is not None:
            authority.evaluation.put_file(
                video_key, video_path, sha256=video["sha256"], content_type="video/mp4"
            )
            raw = {**raw, "video": {**video, "object_uri": authority.evaluation.uri(video_key)}}
            normalized = normalize_attempt_result(
                raw, contract=intent.execution_contract, attempt_id=raw["attempt_id"]
            )
            result = replace(
                result, video=normalized["video"], evidence_sha256=normalized["evidence_sha256"]
            )
        validate_publication_evidence({
            **evidence, "raw_result": raw, "verified_result": result.to_dict()
        })
        authority.evaluation.put_json(
            f"{prefix}/evals/{key}/local-import.json",
            {
                "schema_version": 1,
                "origin_raw_sha256": document_sha256(evidence["raw_result"]),
                "origin_verified_sha256": document_sha256(evidence["verified_result"]),
                "published_raw_sha256": document_sha256(raw),
                "published_verified_sha256": document_sha256(result.to_dict()),
            },
            create_only=True,
        )
        authority.put_eval_intent(intent)
        authority.evaluation.put_json(intent.result_key, raw, create_only=True)
        authority.put_verified_eval_result(result)
        authority.create_promotion(promotion_receipt(
            run_id=run_id, checkpoint_id=intent.checkpoint_id, checkpoint_step=target.step,
            eval_idempotency_key=key, result=result.to_dict(),
            promoted_at=promotion["promoted_at"],
        ))
    finally:
        authority.release_lease(lease)
    return index_url


def import_run_copy(run_dir: Path, queue_dir: Path, run_id: str) -> Path:
    destination = queue_dir / "imported-runs" / run_id
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        for name in ("local-run.json", "recipe.json", "train-config.json", "training-result.json"):
            if file_sha256(destination / name) != file_sha256(run_dir / name):
                raise ValueError(f"previous import differs from original {name}")
        return destination
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        staging = Path(temporary) / "run"
        staging.mkdir()
        for name in ("local-run.json", "recipe.json", "train-config.json", "training-result.json"):
            shutil.copyfile(run_dir / name, staging / name)
        with (
            closing(sqlite3.connect(
                (run_dir / "gradlab.sqlite").as_uri() + "?mode=ro", uri=True
            )) as original,
            closing(sqlite3.connect(staging / "gradlab.sqlite")) as copy,
        ):
            original.backup(copy)
        os.rename(staging, destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = ExactArgumentParser(
        prog="gradlab evaluate-local",
        description="Queue formal evaluation of a completed source-overlay Run on this host.",
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--baseline-receipt", required=True, type=Path)
    parser.add_argument("--input-receipt", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--queue-dir", required=True, type=Path)
    parser.add_argument("--rom-path", type=Path)
    args = parser.parse_args(argv)

    run_dir = args.run_dir.expanduser().resolve()
    source = args.source.expanduser().resolve()
    receipt = json.loads((run_dir / "local-run.json").read_bytes())
    if receipt["status"] != "complete_local" or receipt["model_kind"] != "final":
        raise ValueError("local Acceptance requires a completed final checkpoint")
    run_id = receipt["run_id"]
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("local Run ID is invalid")
    model = run_dir / "final_model.zip"
    if file_sha256(model) != args.checkpoint_sha256:
        raise ValueError("final checkpoint differs from the requested SHA-256")
    bundle = load_policy_bundle_from_checkpoint(model)
    recipe = json.loads((run_dir / "recipe.json").read_bytes())
    if bundle is None or bundle.recipe != recipe or bundle.recipe_sha256 != receipt["recipe_sha256"]:
        raise ValueError("checkpoint recipe differs from the completed Run")
    if bundle.model["checkpoint"]["step"] != receipt["final_step"]:
        raise ValueError("checkpoint step differs from the completed Run")
    asset = recipe["provenance"].get("asset")
    if asset:
        if not args.rom_path:
            raise ValueError("provide the recorded ROM through --rom-path")
        verify_rom_file(args.rom_path.expanduser(), asset)
    overlay = read_source_overlay(
        source, args.baseline_receipt.expanduser(), args.input_receipt.expanduser(), recipe=recipe
    )

    queue_dir = args.queue_dir.expanduser().resolve()
    queue_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    if queue_dir.stat().st_mode & 0o077:
        raise ValueError("local evaluation queue directory must be private (mode 0700)")
    authority = private_evaluation_authority(queue_dir)
    imported = import_run_copy(run_dir, queue_dir, run_id)
    publish_local_run(imported, authority=authority, source_overlay=overlay)
    index = authority.models.get_json(f"runs/{run_id}/index.json")
    [checkpoint] = [
        row for row in index["checkpoints"]
        if row["purpose"] == "final" and row["sha256"] == args.checkpoint_sha256
    ]
    queue = ManualEvaluationQueue(
        repo_root=source, store=JobStore(queue_dir), local_source=source,
        rom_path=args.rom_path.expanduser() if args.rom_path else None,
    )
    result = queue.enqueue(
        run_id=run_id,
        checkpoint_ids=[checkpoint["checkpoint_id"]],
        selection_fence=evaluation_selection_fence(run_id=run_id, checkpoints=index["checkpoints"]),
    )
    print(json.dumps({"run_id": run_id, "items": result["items"], "worker": result["worker"]}))
    return int(result["worker"]["state"] == "start_failed")
