from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from rlab.artifacts import load_playback_env_config
from rlab.device import resolve_sb3_device
from rlab.env import assert_provider_runtime_available, make_eval_vec_env, resolve_env_config
from rlab.env_metadata import env_config_from_config_dict
from rlab.env_registry import resolve_env_provider
from rlab.model_sources import (
    ResolvedModelSource,
    download_huggingface_model_source,
    download_public_checkpoint_manifest_source,
    download_public_run_source,
)
from rlab.policy_bundle import load_policy_bundle_from_checkpoint, playback_contract
from rlab.play_termination import (
    configured_termination_ids,
    with_enabled_termination_conditions,
)
from rlab.rom_assets import rom_asset_manifest_for_game
from rlab.rom_runtime import ensure_local_rom_binding
from rlab.seeds import validate_eval_seed, validate_playback_seed
from rlab.trusted_inputs import (
    ModelApprovalError,
    StagedModelInput,
    approve_staged_model,
    stage_model_input,
)


ProgressCallback = Callable[[str, str], None]
SourceKind = Literal["manifest", "huggingface", "local", "public_run"]


@dataclass(frozen=True)
class PlaySourceSpec:
    kind: SourceKind
    value: str
    entity: str = ""
    project: str = ""
    run_id: str = ""
    checkpoint_id: str = ""
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "value": self.value,
            "entity": self.entity,
            "project": self.project,
            "run_id": self.run_id,
            "checkpoint_id": self.checkpoint_id,
            "seed": self.seed,
        }


def _implicit_playback_seed(
    recipe: Mapping[str, Any],
    *,
    evaluation_result_seed: int | None,
) -> int:
    if evaluation_result_seed is not None:
        return validate_playback_seed(
            evaluation_result_seed,
            label="evaluation result seed",
        )
    train_config = recipe.get("train_config")
    if not isinstance(train_config, Mapping):
        raise ValueError("policy bundle recipe has no training seed")
    return validate_playback_seed(
        train_config.get("seed"),
        label="training seed",
    )


@dataclass
class PlaybackCandidate:
    spec: PlaySourceSpec
    args: argparse.Namespace
    source: ResolvedModelSource
    config: Any
    display_config: Any
    rom_binding: Any
    staged: StagedModelInput
    source_identity: str
    artifact_ref: str | None
    approval_required: bool
    termination_base_config: Any
    termination_source: str

    def approval_payload(self) -> dict[str, Any]:
        return {
            "source": self.source_identity,
            "manifest_hash": self.staged.manifest_hash,
            "files": [entry.as_dict() for entry in self.staged.manifest],
            "warning": (
                "External Python model content can execute arbitrary code with your current "
                "operating-system authority, including access to ambient credentials."
            ),
        }

    def cleanup(self) -> None:
        self.staged.cleanup()


@dataclass
class ActivePlayback:
    runner: Any
    policy_env: Any
    spec: PlaySourceSpec
    source: ResolvedModelSource

    def close(self) -> None:
        try:
            self.runner.stop()
        finally:
            try:
                session = getattr(self.runner, "session", None)
                active_env = getattr(session, "env", self.policy_env)
                active_env.close()
            except Exception:
                pass


class PlaybackLoader:
    """Resolve and build one playback runner without owning application state."""

    def __init__(
        self,
        args: argparse.Namespace,
        *,
        argv: list[str],
        explicit_seed: bool,
    ) -> None:
        self.base_args = deepcopy(args)
        self.argv = list(argv)
        self.explicit_seed = bool(explicit_seed)

    def _resolve_source(
        self,
        spec: PlaySourceSpec,
        args: argparse.Namespace,
    ) -> tuple[ResolvedModelSource, str | None]:
        if spec.kind == "public_run":
            return (
                download_public_run_source(
                    spec.value,
                    root=Path(args.public_model_root),
                    public_base_url=str(args.public_models_base_url),
                ),
                None,
            )
        if spec.kind == "manifest":
            return (
                download_public_checkpoint_manifest_source(
                    spec.value,
                    root=Path(args.public_model_root),
                ),
                spec.value,
            )
        if spec.kind == "huggingface":
            return (
                download_huggingface_model_source(
                    spec.value,
                    root=Path(args.hf_model_root),
                    revision=getattr(args, "hf_revision", None),
                ),
                spec.value,
            )
        model_path = Path(spec.value).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(f"local model checkpoint not found: {model_path}")
        return (
            ResolvedModelSource(
                model_path=model_path,
                bundle=load_policy_bundle_from_checkpoint(model_path),
            ),
            None,
        )

    def prepare(
        self,
        spec: PlaySourceSpec,
        progress: ProgressCallback,
    ) -> PlaybackCandidate:
        args = deepcopy(self.base_args)
        progress("resolving", "Resolving model source")
        source, artifact_ref = self._resolve_source(spec, args)
        args.model = str(source.model_path)

        progress("verifying", "Validating playback contract")
        contract: Mapping[str, Any] | None = None
        termination_source = "training"
        if source.bundle is not None:
            contract = playback_contract(source.bundle.recipe)
            recipe = source.bundle.recipe.get("recipe", {})
            termination_source = (
                "evaluation"
                if isinstance(recipe, Mapping) and isinstance(recipe.get("eval"), Mapping)
                else "training"
            )
            artifact_config = env_config_from_config_dict(contract["environment"])
            if artifact_config is None:
                raise ValueError("policy bundle recipe has no playback environment")
            artifact_config = resolve_env_config(artifact_config)
            if not self.explicit_seed:
                if not isinstance(recipe, Mapping):
                    raise ValueError("policy bundle recipe is invalid")
                args.seed = _implicit_playback_seed(
                    recipe,
                    evaluation_result_seed=spec.seed,
                )
        else:
            artifact_config = load_playback_env_config(
                source.model_path,
                respect_task_termination=True,
            )
        if args.env_provider:
            artifact_config = resolve_env_config(
                replace(artifact_config, env_provider=str(args.env_provider))
            )
        termination_base_config = artifact_config
        enabled_termination_ids = (
            () if args.continuous_play else configured_termination_ids(termination_base_config)
        )
        artifact_config = with_enabled_termination_conditions(
            termination_base_config,
            enabled_termination_ids,
        )
        args.seed = (
            validate_eval_seed(args.seed)
            if self.explicit_seed
            else validate_playback_seed(args.seed)
        )
        display_config = artifact_config

        progress("verifying", "Checking environment provider")
        rom_binding = None
        if resolve_env_provider(artifact_config.env_provider).requires_external_rom_asset:
            asset = contract.get("asset") if contract is not None else None
            rom_binding = (
                ensure_local_rom_binding(asset, game=artifact_config.game)
                if isinstance(asset, Mapping)
                else ensure_local_rom_binding(
                    rom_asset_manifest_for_game(artifact_config.game),
                    game=artifact_config.game,
                )
            )
        assert_provider_runtime_available(artifact_config, rom_binding=rom_binding)

        progress("verifying", "Hashing executable model closure")
        source_identity = str(source.artifact_name or artifact_ref or source.model_path)
        staged = stage_model_input(source.model_path, source_identity=source_identity)
        approval_required = False
        try:
            approve_staged_model(staged, interactive=False)
        except ModelApprovalError as exc:
            if "external model approval is required" not in str(exc):
                staged.cleanup()
                raise
            approval_required = True
        return PlaybackCandidate(
            spec=spec,
            args=args,
            source=source,
            config=artifact_config,
            display_config=display_config,
            rom_binding=rom_binding,
            staged=staged,
            source_identity=source_identity,
            artifact_ref=artifact_ref,
            approval_required=approval_required,
            termination_base_config=termination_base_config,
            termination_source=termination_source,
        )

    def activate(
        self,
        candidate: PlaybackCandidate,
        *,
        approval_hash: str,
        progress: ProgressCallback,
    ) -> ActivePlayback:
        from rlab.play import (
            PolicyActionAttributor,
            _PlaybackSession,
            resolved_play_launch_lines,
        )
        from rlab.play_web import WebPlaybackRunner
        from rlab.policy_models import load_policy_model

        args = candidate.args
        progress("loading", "Loading policy runtime")
        with approve_staged_model(
            candidate.staged,
            expected_hash=approval_hash,
            interactive=False,
        ) as approved:
            model = load_policy_model(
                approved,
                device=resolve_sb3_device(args.device),
            )

        if args.attribution != "none":
            if not hasattr(model, "policy"):
                raise ValueError("policy attribution is unavailable for non-neural policies")
            progress("loading", "Preparing policy attribution")
            attributor = PolicyActionAttributor(model)
        else:
            attributor = None

        progress("loading", "Creating policy environment")

        def make_policy_env(config, seed):
            return make_eval_vec_env(
                config=config,
                n_envs=1,
                seed=seed,
                capture_step_diagnostics=True,
                rom_binding=candidate.rom_binding,
            )

        policy_env = make_policy_env(candidate.config, args.seed)
        try:
            from rlab.policy_runtime import bind_policy_action_space

            bind_policy_action_space(model, policy_env.action_space)
            session = _PlaybackSession(
                model=model,
                env=policy_env,
                config=candidate.config,
                initial_seed=args.seed,
                attributor=attributor,
                attribution_mode=args.attribution,
                attribution_interval=args.attribution_interval,
                attribution_opacity=args.attribution_opacity,
                env_factory=make_policy_env,
                termination_base_config=candidate.termination_base_config,
                termination_source=candidate.termination_source,
            )
            progress("loading", "Resetting policy environment")
            session.restart(args.seed)
            config_text = "\n".join(
                resolved_play_launch_lines(
                    args,
                    argv=self.argv,
                    artifact_ref=candidate.artifact_ref,
                    policy_config=candidate.config,
                    display_config=candidate.display_config,
                )
            )
            config_text += (
                "\n"
                f"checkpoint_step={candidate.source.checkpoint_step or '-'} "
                f"environment_hash={candidate.source.run_config.get('environment_hash', '-')}"
            )
            runner = WebPlaybackRunner(session, args, config_text=config_text)
            return ActivePlayback(
                runner=runner,
                policy_env=policy_env,
                spec=candidate.spec,
                source=candidate.source,
            )
        except Exception:
            policy_env.close()
            raise


__all__ = [
    "ActivePlayback",
    "PlaySourceSpec",
    "PlaybackCandidate",
    "PlaybackLoader",
]
