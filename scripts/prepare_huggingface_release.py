from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from gradlab.cli_parser import ExactArgumentParser
from gradlab.config_validation import load_goal_contract
from gradlab.publication import (
    GITATTRIBUTES_TEXT,
    MIT_LICENSE_TEXT,
    build_model_repo_id,
    build_release_manifest,
    normalize_publication_evaluation,
    publication_identity_from_policy_bundle,
    publication_source_from_policy_bundle,
    release_replay_from_capture,
    release_artifact_records,
    render_model_card,
    validate_release_bundle,
    verify_replay,
)
from gradlab.policy_bundle import (
    PolicyBundle,
    build_model_document,
    evaluation_contract_sha256,
    load_model_document,
    load_policy_bundle,
    load_recipe_document,
    sha256_file,
    write_canonical_json,
)


def _load_object(path: Path, *, label: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a JSON object")
    return dict(value)


def _parser() -> argparse.ArgumentParser:
    parser = ExactArgumentParser(
        description="Build and validate one deterministic gradlab Hugging Face release bundle."
    )
    parser.add_argument("--goal-file", type=Path, required=True)
    parser.add_argument("--model-metadata", type=Path, required=True)
    parser.add_argument("--identity-only", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--evaluation-json", type=Path)
    parser.add_argument("--capture-json", type=Path)
    parser.add_argument("--publication-json", type=Path)
    parser.add_argument("--release-version")
    parser.add_argument("--published-at")
    parser.add_argument("--youtube-url")
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    goal = load_goal_contract(args.goal_file)
    source_model = load_model_document(args.model_metadata)
    if args.identity_only:
        if args.recipe is None:
            raise ValueError("--identity-only requires --recipe with the current model.json")
        recipe_document = load_recipe_document(args.recipe)
        bundle = PolicyBundle(
            checkpoint_path=(
                args.model
                if args.model is not None
                else args.model_metadata.parent / source_model["checkpoint"]["filename"]
            ),
            model_path=args.model_metadata,
            recipe_path=args.recipe,
            model=source_model,
            recipe=recipe_document,
            source=str(args.model_metadata),
        )
        identity = publication_identity_from_policy_bundle(goal.get("goal_id"), bundle)
        summary = {"repo_id": build_model_repo_id(identity), **identity.__dict__}
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    required = {
        "--model": args.model,
        "--recipe": args.recipe,
        "--replay": args.replay,
        "--evaluation-json": args.evaluation_json,
        "--capture-json": args.capture_json,
        "--publication-json": args.publication_json,
        "--release-version": args.release_version,
        "--youtube-url": args.youtube_url,
        "--output-dir": args.output_dir,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError("full bundle preparation requires " + ", ".join(missing))
    assert args.model is not None
    assert args.recipe is not None
    assert args.replay is not None
    assert args.evaluation_json is not None
    assert args.capture_json is not None
    assert args.publication_json is not None
    assert args.release_version is not None
    assert args.youtube_url is not None
    assert args.output_dir is not None
    for path in (
        args.model,
        args.recipe,
        args.replay,
        args.evaluation_json,
        args.capture_json,
        args.publication_json,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"release output directory is not empty: {args.output_dir}")

    verify_replay(args.replay)
    evaluation_document = _load_object(args.evaluation_json, label="evaluation")
    capture_document = _load_object(args.capture_json, label="capture")
    publication_document = _load_object(args.publication_json, label="publication")
    replay_value = release_replay_from_capture(capture_document)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.model, args.output_dir / "model.zip")
    shutil.copy2(args.recipe, args.output_dir / "recipe.json")
    recipe_document = load_recipe_document(args.output_dir / "recipe.json")
    shutil.copy2(args.replay, args.output_dir / "replay.mp4")
    (args.output_dir / ".gitattributes").write_text(GITATTRIBUTES_TEXT, encoding="utf-8")
    (args.output_dir / "LICENSE").write_text(MIT_LICENSE_TEXT, encoding="utf-8")
    metadata = deepcopy(dict(source_model["provenance"]))
    metadata.update(source_model["policy"])
    metadata["checkpoint_step"] = source_model["checkpoint"].get("step")
    metadata["kind"] = source_model["checkpoint"].get("kind")
    write_canonical_json(
        args.output_dir / "model.json",
        build_model_document(
            args.output_dir / "model.zip",
            args.output_dir / "recipe.json",
            metadata,
        ),
    )
    bundle = load_policy_bundle(args.output_dir, source=str(args.output_dir))
    identity = publication_identity_from_policy_bundle(goal.get("goal_id"), bundle)
    repo_id = build_model_repo_id(identity)
    summary = {"repo_id": repo_id, **identity.__dict__}
    evaluation = normalize_publication_evaluation(
        evaluation_document,
        algorithm_id=str(bundle.model["policy"].get("algorithm_id") or ""),
    )
    source = publication_source_from_policy_bundle(bundle, evaluation)
    evidence = evaluation_document.get("evaluation_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("release evaluation is missing evaluation_evidence")
    expected_evidence = {
        "checkpoint_sha256": sha256_file(args.output_dir / "model.zip"),
        "recipe_sha256": sha256_file(args.output_dir / "recipe.json"),
        "recipe_format_version": int(recipe_document["format_version"]),
        "evaluation_contract_sha256": evaluation_contract_sha256(recipe_document),
    }
    for key, expected in expected_evidence.items():
        if evidence.get(key) != expected:
            raise ValueError(f"release evaluation {key} does not match the policy bundle")
    if evidence.get("exact_contract") is not True:
        raise ValueError("release evaluation must be exact-contract evidence")
    evaluation_value = evaluation.as_manifest_value()
    evaluation_value.update(expected_evidence, exact_contract=True)
    published_at = args.published_at or datetime.now(UTC).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )
    provisional_manifest = build_release_manifest(
        identity,
        bundle,
        release_version=args.release_version,
        published_at=published_at,
        source=source,
        evaluation=evaluation_value,
        artifacts={},
        youtube_url=args.youtube_url,
        replay=replay_value,
        publication=publication_document,
    )
    (args.output_dir / "README.md").write_text(
        render_model_card(provisional_manifest, bundle), encoding="utf-8"
    )
    artifact_records = release_artifact_records(args.output_dir)
    manifest = build_release_manifest(
        identity,
        bundle,
        release_version=args.release_version,
        published_at=published_at,
        source=source,
        evaluation=evaluation_value,
        artifacts=artifact_records,
        youtube_url=args.youtube_url,
        replay=replay_value,
        publication=publication_document,
    )
    (args.output_dir / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_release_bundle(args.output_dir)
    print(json.dumps({**summary, "bundle": str(args.output_dir)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
