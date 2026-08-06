from __future__ import annotations

import sys
from pathlib import Path

from gradlab.cli_args import explicit_arg_dests
from gradlab.play_session import build_parser


def main(argv: list[str] | None = None) -> int:
    from gradlab.model_sources import is_huggingface_model_ref
    from gradlab.play_catalog import PlayCatalog, parse_wandb_location
    from gradlab.play_runtime import PlaySourceSpec
    from gradlab.playback_worker import IsolatedPlaybackHost
    from gradlab.play_web import run_web_player_application
    from gradlab.recipe_catalog import (
        experiments_root,
        latest_local_recipe_model,
        recipe_identity,
        resolve_recipe_source,
    )
    from gradlab.recipe_documents import compose_train_document

    parser = build_parser()
    argv_list = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv_list)
    selected_sources = sum(
        bool(value)
        for value in (
            args.artifact_ref,
            args.model,
            args.recipe,
            args.run,
        )
    )
    if selected_sources > 1:
        parser.error("pass exactly one of --run, --recipe, a positional remote source, or --model")
    wandb_location = parse_wandb_location(args.artifact_ref)
    if args.recipe:
        recipe_source = resolve_recipe_source(args.recipe)
        materialized_recipe = compose_train_document(
            recipe_source.goal_path,
            recipe_source.recipe_path,
        )
        goal_id, recipe_id = recipe_identity(materialized_recipe)
        args.model = str(
            latest_local_recipe_model(
                args.runs_dir,
                goal_id=goal_id,
                recipe_id=recipe_id,
            )
        )
    args.respect_task_termination = not args.continuous_play
    explicit_dests = explicit_arg_dests(parser, argv_list)

    repo_root = experiments_root().parent
    catalog_authority = None
    catalog_control_error = ""
    from gradlab.catalog_errors import CatalogError, CatalogIntegrityError, CatalogUnavailable
    from gradlab.play_catalog_authority import (
        scrub_protected_environment,
        start_catalog_authority_helper,
    )

    def start_private_catalog() -> None:
        nonlocal catalog_authority, catalog_control_error
        try:
            catalog_authority = start_catalog_authority_helper(repo_root)
        except CatalogError as exc:
            catalog_control_error = str(exc)

    if selected_sources == 0:
        start_private_catalog()
    catalog = PlayCatalog(
        public_models_base_url=args.public_models_base_url,
        repo_root=repo_root,
        cache_path=PlayCatalog.default_cache_path(repo_root),
        control_bucket=catalog_authority,
        control_error=catalog_control_error,
        wandb_run_location=wandb_location,
    )
    initial_route: dict[str, object] = {"level": "environments"}
    initial_source: PlaySourceSpec | None = None
    if args.run:
        initial_route = {
            "level": "runs",
            "run_id": str(args.run),
        }
        try:
            initial_route = catalog.public_run_route(run_id=str(args.run))
        except CatalogError:
            pass
        except ValueError:
            pass
        initial_source = PlaySourceSpec("public_run", str(args.run), run_id=str(args.run))
    elif args.model:
        initial_source = PlaySourceSpec("local", str(Path(args.model).expanduser()))
    elif args.artifact_ref:
        if wandb_location is not None:
            try:
                goal_id, goal_variant_id = catalog.run_goal_variant(
                    environment_id=wandb_location.project,
                    run_id=wandb_location.run_id,
                )
            except CatalogIntegrityError:
                raise
            except CatalogUnavailable as exc:
                if exc.problem.code not in {
                    "public_proof_absent",
                    "public_catalog_transient",
                }:
                    raise
                start_private_catalog()
                if catalog_authority is None:
                    raise CatalogUnavailable(
                        catalog_control_error
                        or "private catalog authority could not be started",
                        code="catalog_configuration",
                        retryable=False,
                        source="control-catalog",
                    ) from exc
                catalog = PlayCatalog(
                    public_models_base_url=args.public_models_base_url,
                    repo_root=repo_root,
                    cache_path=PlayCatalog.default_cache_path(repo_root),
                    control_bucket=catalog_authority,
                    control_error=catalog_control_error,
                    wandb_run_location=wandb_location,
                )
                goal_id, goal_variant_id = catalog.run_goal_variant(
                    environment_id=wandb_location.project,
                    run_id=wandb_location.run_id,
                )
            initial_route = {
                "level": "runs",
                "environment_id": wandb_location.project,
                "goal_id": goal_id,
                "goal_variant_id": goal_variant_id,
                "run_id": wandb_location.run_id,
            }
        elif is_huggingface_model_ref(args.artifact_ref):
            initial_source = PlaySourceSpec("huggingface", str(args.artifact_ref))
        else:
            initial_source = PlaySourceSpec("manifest", str(args.artifact_ref))

    scrub_protected_environment()

    host = IsolatedPlaybackHost(
        args,
        argv=argv_list,
        explicit_seed="seed" in explicit_dests,
        initial_route=initial_route,
        initial_source=initial_source,
    )
    try:
        return run_web_player_application(host, args, catalog=catalog)
    finally:
        if catalog_authority is not None:
            catalog_authority.close()


if __name__ == "__main__":
    raise SystemExit(main())
