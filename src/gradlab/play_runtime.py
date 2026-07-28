from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from gradlab.artifacts import load_playback_env_config
from gradlab.device import resolve_sb3_device
from gradlab.env import assert_provider_runtime_available, make_eval_vec_env, resolve_env_config
from gradlab.env_metadata import env_config_from_config_dict, env_config_metadata
from gradlab.env_registry import resolve_env_provider
from gradlab.model_sources import (
    ResolvedModelSource,
    download_huggingface_model_source,
    download_public_checkpoint_manifest_source,
    download_public_run_source,
)
from gradlab.policy_bundle import (
    critic_value_contract,
    load_policy_bundle_from_checkpoint,
    playback_contract,
    playback_contract_audit,
)
from gradlab.play_attribution import PolicyActionAttributor
from gradlab.play_session import (
    _PlaybackSession,
    resolved_play_launch_lines,
)
from gradlab.play_termination import (
    configured_termination_ids,
    with_enabled_termination_conditions,
)
from gradlab.play_web import WebPlaybackRunner
from gradlab.rom_assets import rom_asset_manifest_for_game
from gradlab.rom_runtime import ensure_local_rom_binding
from gradlab.seeds import validate_eval_seed, validate_playback_seed
from gradlab.trusted_inputs import (
    ModelApprovalError,
    StagedModelInput,
    approve_staged_model,
    stage_model_input,
)


ProgressCallback = Callable[[str, str], None]
SourceKind = Literal["manifest", "huggingface", "local", "public_run"]
PlaybackContractMode = Literal["training", "evaluation", "counterfactual"]


@dataclass(frozen=True)
class PlaySourceSpec:
    kind: SourceKind
    value: str
    entity: str = ""
    project: str = ""
    run_id: str = ""
    checkpoint_id: str = ""
    seed: int | None = None
    contract_mode: PlaybackContractMode = "training"
    reward_clip_override: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "value": self.value,
            "entity": self.entity,
            "project": self.project,
            "run_id": self.run_id,
            "checkpoint_id": self.checkpoint_id,
            "seed": self.seed,
            "contract_mode": self.contract_mode,
            "reward_clip_override": self.reward_clip_override,
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
    contract_details: dict[str, Any] = field(default_factory=dict)
    value_contract: dict[str, Any] | None = None

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
        contract_details: dict[str, Any] = {
            "mode": "training",
            "available_modes": ["training"],
            "matches_training": False,
            "comparison_reasons": ["checkpoint has no portable training contract"],
        }
        value_contract: dict[str, Any] | None = None
        if source.bundle is not None:
            requested_mode = str(spec.contract_mode or "training")
            if requested_mode not in {"training", "evaluation", "counterfactual"}:
                raise ValueError(f"unsupported playback contract mode {requested_mode!r}")
            base_mode = "evaluation" if requested_mode == "evaluation" else "training"
            contract = playback_contract(source.bundle.recipe, mode=base_mode)
            recipe = source.bundle.recipe.get("recipe", {})
            value_contract = critic_value_contract(source.bundle.recipe)
            contract_audit = playback_contract_audit(source.bundle.recipe)
            active_environment = deepcopy(dict(contract["environment"]))
            if requested_mode == "counterfactual":
                if spec.reward_clip_override is None:
                    raise ValueError(
                        "counterfactual playback requires an explicit reward clipping override"
                    )
                task = active_environment.get("task")
                if not isinstance(task, dict):
                    raise ValueError("playback environment has no configurable task reward")
                reward = task.get("reward")
                if not isinstance(reward, dict):
                    raise ValueError("playback environment has no configurable reward contract")
                reward["reward_clip"] = bool(spec.reward_clip_override)
            artifact_config = env_config_from_config_dict(active_environment)
            if artifact_config is None:
                raise ValueError("policy bundle recipe has no playback environment")
            artifact_config = resolve_env_config(artifact_config)
            from gradlab.env_identity import policy_environment_hash

            active_hash = policy_environment_hash(env_config_metadata(artifact_config))
            training_hash = str(contract["training_policy_environment_hash"])
            evaluation_available = isinstance(recipe, Mapping) and isinstance(
                recipe.get("eval"), Mapping
            )
            evaluation_matches_training: bool | None = None
            if evaluation_available:
                evaluation_matches_training = bool(
                    playback_contract(source.bundle.recipe, mode="evaluation")["matches_training"]
                )
            available_modes = ["training"]
            if evaluation_available:
                available_modes.append("evaluation")
            available_modes.append("counterfactual")
            comparison_reasons = (
                []
                if active_hash == training_hash
                else ["active policy environment differs from training"]
            )
            contract_details = {
                "mode": requested_mode,
                "available_modes": available_modes,
                "reward_clip_override": spec.reward_clip_override,
                "policy_environment_hash": active_hash,
                "training_policy_environment_hash": training_hash,
                "matches_training": not comparison_reasons,
                "comparison_reasons": comparison_reasons,
                "evaluation_matches_training": evaluation_matches_training,
                "legacy_mismatch": bool(
                    evaluation_available and evaluation_matches_training is False
                ),
                "mismatch_paths": list(contract_audit["mismatch_paths"]),
                "legacy_override_provenance": bool(contract_audit["legacy_override_provenance"]),
                "requested_policy_override_paths": list(
                    contract_audit["requested_policy_override_paths"]
                ),
            }
            termination_source = requested_mode
            if not self.explicit_seed:
                if not isinstance(recipe, Mapping):
                    raise ValueError("policy bundle recipe is invalid")
                args.seed = _implicit_playback_seed(
                    recipe,
                    evaluation_result_seed=spec.seed,
                )
        else:
            if spec.contract_mode != "training" or spec.reward_clip_override is not None:
                raise ValueError(
                    "contract selection requires a versioned policy bundle with recipe.json"
                )
            artifact_config = load_playback_env_config(
                source.model_path,
                respect_task_termination=True,
            )
        if args.env_provider:
            artifact_config = resolve_env_config(
                replace(artifact_config, env_provider=str(args.env_provider))
            )
            if source.bundle is not None:
                contract_details["mode"] = "counterfactual"
                contract_details["matches_training"] = False
                contract_details["comparison_reasons"] = [
                    "environment provider override differs from training"
                ]
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
            contract_details=contract_details,
            value_contract=value_contract,
        )

    def activate(
        self,
        candidate: PlaybackCandidate,
        *,
        approval_hash: str,
        progress: ProgressCallback,
    ) -> ActivePlayback:
        from gradlab.policy_models import load_policy_model
        from gradlab.policy_runtime import PolicyRuntime

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
        # Validate the executable policy contract before creating or stepping
        # an environment. Optional telemetry can degrade later, but action
        # execution and its declared selection modes cannot.
        policy_runtime = PolicyRuntime(model)

        if args.attribution != "none":
            if policy_runtime.capabilities.algorithm_id not in {"ppo", "a2c"}:
                raise ValueError(
                    "selected-action log-probability attribution is unavailable for "
                    f"{policy_runtime.capabilities.algorithm_id} policies"
                )
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
            from gradlab.policy_runtime import bind_policy_action_space

            bind_policy_action_space(model, policy_env.action_space)
            policy_provenance: dict[str, Any] = {}
            if candidate.source.bundle is not None:
                model_document = candidate.source.bundle.model
                policy_value = model_document.get("policy")
                policy = policy_value if isinstance(policy_value, Mapping) else {}
                provenance_value = model_document.get("provenance")
                provenance = (
                    provenance_value if isinstance(provenance_value, Mapping) else {}
                )
                policy_provenance = {
                    "training_backend_id": str(policy.get("training_backend_id") or ""),
                }
                search_algorithm_id = str(
                    provenance.get("search_algorithm_id") or ""
                ).strip()
                if search_algorithm_id:
                    policy_provenance["search_algorithm_id"] = search_algorithm_id
                summary_value = provenance.get("state_archive_summary")
                if isinstance(summary_value, Mapping):
                    safe_summary_fields = {
                        "semantic_id",
                        "schema_version",
                        "persistence",
                        "provider_id",
                        "codec_id",
                        "compatibility_id",
                        "entry_count",
                        "blob_count",
                        "blob_bytes",
                        "view_ids",
                    }
                    policy_provenance["state_archive_summary"] = {
                        str(key): deepcopy(value)
                        for key, value in summary_value.items()
                        if key in safe_summary_fields
                    }
            session = _PlaybackSession(
                model=model,
                env=policy_env,
                config=candidate.config,
                initial_seed=args.seed,
                attributor=attributor,
                attribution_mode=args.attribution,
                attribution_interval=args.attribution_interval,
                attribution_opacity=args.attribution_opacity,
                policy_runtime=policy_runtime,
                policy_provenance=policy_provenance,
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
            runner = WebPlaybackRunner(
                session,
                args,
                config_text=config_text,
                contract_details=candidate.contract_details,
                value_contract=candidate.value_contract,
            )
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
