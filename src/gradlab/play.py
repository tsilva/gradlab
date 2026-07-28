from __future__ import annotations

import sys
from pathlib import Path

from gradlab.cli_args import explicit_arg_dests
from gradlab.play_session import (
    _PlaybackSession,
    _PlaybackTransition,
    build_parser,
    optional_fast_env_frames,
    playback_model_observation,
    playback_should_end_episode,
    render_obs_stack,
    task_conditioning_change_message,
    vector_env_frame,
)

__all__ = (
    "_PlaybackSession",
    "_PlaybackTransition",
    "build_parser",
    "main",
    "optional_fast_env_frames",
    "playback_model_observation",
    "playback_should_end_episode",
    "render_obs_stack",
    "task_conditioning_change_message",
    "vector_env_frame",
)


def main(argv: list[str] | None = None) -> int:
    from gradlab.model_sources import is_huggingface_model_ref
    from gradlab.play_application import PlaybackHost
    from gradlab.play_catalog import PlayCatalog, parse_wandb_location
    from gradlab.play_runtime import PlaySourceSpec, PlaybackLoader
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
    if args.attribution_interval is None:
        args.attribution_interval = 8 if args.attribution == "occlusion" else 1

    repo_root = experiments_root().parent
    catalog = PlayCatalog(
        public_models_base_url=args.public_models_base_url,
        repo_root=repo_root,
        cache_path=PlayCatalog.default_cache_path(repo_root),
    )
    initial_route: dict[str, object] = {
        "level": "environments",
        "entity": str(args.wandb_entity or ""),
    }
    initial_source: PlaySourceSpec | None = None
    if args.run:
        initial_source = PlaySourceSpec("public_run", str(args.run), run_id=str(args.run))
    elif args.model:
        initial_source = PlaySourceSpec("local", str(Path(args.model).expanduser()))
    elif args.artifact_ref:
        wandb_location = parse_wandb_location(args.artifact_ref)
        if wandb_location is not None:
            goal_id, goal_variant_id = (
                catalog.run_goal_variant(
                    entity=wandb_location.entity,
                    project=wandb_location.project,
                    run_id=wandb_location.run_id,
                )
                if wandb_location.run_id
                else ("", "")
            )
            initial_route = {
                "level": "runs" if wandb_location.run_id else "goals",
                "entity": wandb_location.entity,
                "project": wandb_location.project,
                "goal_id": goal_id,
                "goal_variant_id": goal_variant_id,
                "run_id": wandb_location.run_id or "",
            }
            args.wandb_entity = wandb_location.entity
        elif is_huggingface_model_ref(args.artifact_ref):
            initial_source = PlaySourceSpec("huggingface", str(args.artifact_ref))
        else:
            initial_source = PlaySourceSpec("manifest", str(args.artifact_ref))

    loader = PlaybackLoader(
        args,
        argv=argv_list,
        explicit_seed="seed" in explicit_dests,
    )
    host = PlaybackHost(
        loader,
        initial_route=initial_route,
        initial_source=initial_source,
    )
    return run_web_player_application(host, args, catalog=catalog)


if __name__ == "__main__":
    raise SystemExit(main())
