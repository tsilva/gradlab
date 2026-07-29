from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from gradlab.reward_transform import PROVIDER_REWARD_TRANSFORM_KEYS


EXTERNAL_ROM_ASSET_NONE = "none"
STABLE_RETRO_DIRECT_PATH_V1 = "stable_retro_direct_path_v1"
EXTERNAL_ROM_ASSET_STRATEGIES = frozenset({EXTERNAL_ROM_ASSET_NONE, STABLE_RETRO_DIRECT_PATH_V1})


@dataclass(frozen=True)
class ProviderConstructorContract:
    canonical_args: frozenset[str]
    explicit_env_args: frozenset[str]
    required_values: Mapping[str, object]
    optional_env_args: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        overlap = (
            (self.canonical_args & self.explicit_env_args)
            | (self.canonical_args & self.optional_env_args)
            | (self.explicit_env_args & self.optional_env_args)
        )
        if overlap:
            raise ValueError(f"provider constructor argument ownership overlaps: {sorted(overlap)}")
        unknown_required = set(self.required_values) - set(self.explicit_env_args)
        if unknown_required:
            raise ValueError(
                f"provider required values are not explicit env args: {sorted(unknown_required)}"
            )
        object.__setattr__(self, "required_values", MappingProxyType(dict(self.required_values)))


@dataclass(frozen=True)
class EvalProgressField:
    info_key: str
    result_key: str
    rank: bool = False


@dataclass(frozen=True)
class EvalSemantics:
    completion_reason: str | None = None
    completion_info_keys: tuple[str, ...] = ()
    completion_blocking_info_keys: tuple[str, ...] = ()
    progress_fields: tuple[EvalProgressField, ...] = ()
    death_flag_key: str | None = None
    death_position_key: str | None = None
    best_episode_rank: tuple[str, ...] = ("reward",)


@dataclass(frozen=True)
class EnvironmentSpec:
    spec_id: str
    game_family: str
    wandb_project: str
    task_id: str = "identity"
    default_state: str = ""
    default_obs_crop: tuple[int, int, int, int] | None = None
    eval_semantics: EvalSemantics = EvalSemantics()


@dataclass(frozen=True)
class EnvRegistration:
    spec_id: str
    env_id_game_family_fallback: bool = True
    env_id_wandb_project_fallback: bool = True


MARIO_EVAL_SEMANTICS = EvalSemantics(
    completion_reason="level_change",
    completion_info_keys=("completion_event", "level_complete"),
    completion_blocking_info_keys=("died", "life_loss"),
    progress_fields=(
        EvalProgressField("max_x_pos", "max_x_pos", rank=True),
        EvalProgressField("level_max_x_pos", "max_level_x_pos"),
    ),
    death_flag_key="died",
    death_position_key="death_x_pos",
    best_episode_rank=("completion", "progress", "reward"),
)

ENVIRONMENT_SPECS: Mapping[str, EnvironmentSpec] = MappingProxyType(
    {
        "Bandit-v0": EnvironmentSpec("Bandit-v0", "Bandit", "Bandit-v0"),
        "SuperMarioBros-Nes-v0": EnvironmentSpec(
            "SuperMarioBros-Nes-v0",
            "NES-SuperMarioBros",
            "SuperMarioBros-Nes-v0",
            task_id="mario",
            default_state="Level1-1",
            default_obs_crop=(32, 0, 0, 0),
            eval_semantics=MARIO_EVAL_SEMANTICS,
        ),
        "SuperMarioBros3-Nes-v0": EnvironmentSpec(
            "SuperMarioBros3-Nes-v0",
            "NES-SuperMarioBros3",
            "SuperMarioBros3-Nes-v0",
            default_state="1Player.World1.Level1",
        ),
        "Breakout-Atari2600-v0": EnvironmentSpec(
            "Breakout-Atari2600-v0",
            "Atari2600-Breakout",
            "Breakout-Atari2600-v0",
        ),
        "MsPacman-Atari2600-v0": EnvironmentSpec(
            "MsPacman-Atari2600-v0",
            "Atari2600-MsPacman",
            "MsPacman-Atari2600-v0",
        ),
        "VizdoomBasic-v1": EnvironmentSpec(
            "VizdoomBasic-v1", "Doom-ViZDoom-Basic", "VizdoomBasic-v1"
        ),
        "VizdoomBasic-Plus-v1": EnvironmentSpec(
            "VizdoomBasic-Plus-v1",
            "Doom-ViZDoom-Basic-Plus",
            "VizdoomBasic-Plus-v1",
        ),
        "VizdoomDeadlyCorridor-v1": EnvironmentSpec(
            "VizdoomDeadlyCorridor-v1",
            "Doom-ViZDoom-DeadlyCorridor",
            "VizdoomDeadlyCorridor-v1",
        ),
        "VizdoomDeathmatch-v1": EnvironmentSpec(
            "VizdoomDeathmatch-v1",
            "Doom-ViZDoom-Deathmatch",
            "VizdoomDeathmatch-v1",
        ),
        "VizdoomDefendCenter-v1": EnvironmentSpec(
            "VizdoomDefendCenter-v1",
            "Doom-ViZDoom-DefendCenter",
            "VizdoomDefendCenter-v1",
        ),
        "VizdoomDefendLine-v1": EnvironmentSpec(
            "VizdoomDefendLine-v1",
            "Doom-ViZDoom-DefendLine",
            "VizdoomDefendLine-v1",
        ),
        "VizdoomDefendLine-Plus-v1": EnvironmentSpec(
            "VizdoomDefendLine-Plus-v1",
            "Doom-ViZDoom-DefendLine-Plus",
            "VizdoomDefendLine-Plus-v1",
        ),
        "VizdoomHealthGathering-v1": EnvironmentSpec(
            "VizdoomHealthGathering-v1",
            "Doom-ViZDoom-HealthGathering",
            "VizdoomHealthGathering-v1",
        ),
        "VizdoomHealthGatheringSupreme-v1": EnvironmentSpec(
            "VizdoomHealthGatheringSupreme-v1",
            "Doom-ViZDoom-HealthGatheringSupreme",
            "VizdoomHealthGatheringSupreme-v1",
        ),
        "VizdoomMyWayHome-v1": EnvironmentSpec(
            "VizdoomMyWayHome-v1",
            "Doom-ViZDoom-MyWayHome",
            "VizdoomMyWayHome-v1",
        ),
        "VizdoomPredictPosition-v1": EnvironmentSpec(
            "VizdoomPredictPosition-v1",
            "Doom-ViZDoom-PredictPosition",
            "VizdoomPredictPosition-v1",
        ),
        "VizdoomTakeCover-v1": EnvironmentSpec(
            "VizdoomTakeCover-v1",
            "Doom-ViZDoom-TakeCover",
            "VizdoomTakeCover-v1",
        ),
    }
)


@dataclass(frozen=True)
class EnvProvider:
    provider_id: str
    import_name: str
    distribution_name: str
    environments: Mapping[str, EnvRegistration]
    supports_states: bool = True
    external_rom_asset_strategy: str = EXTERNAL_ROM_ASSET_NONE
    allows_unregistered_env_ids: bool = False
    constructor_contract: ProviderConstructorContract | None = None
    turbo_api_version: int | None = None

    def __post_init__(self) -> None:
        if self.external_rom_asset_strategy not in EXTERNAL_ROM_ASSET_STRATEGIES:
            raise ValueError(
                f"unsupported external ROM asset strategy: {self.external_rom_asset_strategy}"
            )
        unknown_specs = {
            registration.spec_id
            for registration in self.environments.values()
            if registration.spec_id not in ENVIRONMENT_SPECS
        }
        if unknown_specs:
            raise ValueError(
                f"provider references unknown environment specs: {sorted(unknown_specs)}"
            )
        object.__setattr__(self, "environments", MappingProxyType(dict(self.environments)))

    @property
    def env_ids(self) -> tuple[str, ...]:
        return tuple(self.environments)

    @property
    def requires_external_rom_asset(self) -> bool:
        return self.external_rom_asset_strategy != EXTERNAL_ROM_ASSET_NONE


@dataclass(frozen=True)
class ResolvedEnvId:
    qualified_id: str
    provider_id: str
    provider_env_id: str
    import_name: str


_TURBO_CANONICAL_ARGS = frozenset(
    {
        "frame_skip",
        "game",
        "maxpool_last_two",
        "num_envs",
        "obs_crop",
        "obs_crop_fill",
        "obs_crop_mode",
        "obs_resize",
        "obs_resize_algorithm",
        "state",
        "state_catalog",
        "sticky_action_prob",
    }
)
_TURBO_EXPLICIT_ENV_ARGS = frozenset(
    {
        "frame_stack",
        "info",
        "info_filter",
        "inttype",
        "noop_reset_max",
        "num_threads",
        "obs_copy",
        "obs_grayscale",
        "obs_layout",
        "obs_type",
        "players",
        "record",
        "render_mode",
        "rom_path",
        "scenario",
        "use_fire_reset",
        "use_restricted_actions",
    }
)


STABLE_RETRO_TURBO_PROVIDER = EnvProvider(
    provider_id="stable-retro-turbo",
    import_name="stable_retro",
    distribution_name="stable-retro-turbo",
    environments={
        spec_id: EnvRegistration(spec_id)
        for spec_id in (
            "SuperMarioBros-Nes-v0",
            "SuperMarioBros3-Nes-v0",
            "Breakout-Atari2600-v0",
            "MsPacman-Atari2600-v0",
        )
    },
    external_rom_asset_strategy=STABLE_RETRO_DIRECT_PATH_V1,
    turbo_api_version=1,
    constructor_contract=ProviderConstructorContract(
        canonical_args=_TURBO_CANONICAL_ARGS,
        explicit_env_args=_TURBO_EXPLICIT_ENV_ARGS,
        required_values={},
    ),
)

SUPERMARIOBROS_NES_TURBO_PROVIDER = EnvProvider(
    provider_id="supermariobrosnes-turbo",
    import_name="supermariobrosnes_turbo",
    distribution_name="supermariobrosnes-turbo",
    environments={
        "SuperMarioBros-Nes-v0": EnvRegistration("SuperMarioBros-Nes-v0"),
    },
    external_rom_asset_strategy=STABLE_RETRO_DIRECT_PATH_V1,
    turbo_api_version=1,
    constructor_contract=ProviderConstructorContract(
        canonical_args=_TURBO_CANONICAL_ARGS,
        explicit_env_args=_TURBO_EXPLICIT_ENV_ARGS,
        required_values={},
    ),
)

VIZDOOM_TURBO_PROVIDER = EnvProvider(
    provider_id="vizdoom-turbo",
    import_name="vizdoom_turbo",
    distribution_name="vizdoom-turbo",
    environments={
        spec_id: EnvRegistration(spec_id)
        for spec_id in (
            "VizdoomBasic-v1",
            "VizdoomBasic-Plus-v1",
            "VizdoomDeadlyCorridor-v1",
            "VizdoomDeathmatch-v1",
            "VizdoomDefendCenter-v1",
            "VizdoomDefendLine-v1",
            "VizdoomDefendLine-Plus-v1",
            "VizdoomHealthGathering-v1",
            "VizdoomHealthGatheringSupreme-v1",
            "VizdoomMyWayHome-v1",
            "VizdoomPredictPosition-v1",
            "VizdoomTakeCover-v1",
        )
    },
    allows_unregistered_env_ids=True,
    turbo_api_version=1,
    constructor_contract=ProviderConstructorContract(
        canonical_args=_TURBO_CANONICAL_ARGS,
        explicit_env_args=_TURBO_EXPLICIT_ENV_ARGS
        | {
            "doom_map",
            "doom_skill",
            "game_args",
            "game_variables",
            "treat_episode_timeout_as_truncation",
            "vizdoom_config",
        },
        required_values={},
        optional_env_args=frozenset({"enemy_variants", "surface_variants"}),
    ),
)

BREAKOUT_TURBO_ENV_PROVIDER = EnvProvider(
    provider_id="breakout-turbo-env",
    import_name="breakout_turbo_env",
    distribution_name="breakout-turbo-env",
    environments={
        "Breakout-Atari2600-v0": EnvRegistration("Breakout-Atari2600-v0"),
    },
    turbo_api_version=1,
    constructor_contract=ProviderConstructorContract(
        canonical_args=_TURBO_CANONICAL_ARGS,
        explicit_env_args=_TURBO_EXPLICIT_ENV_ARGS,
        required_values={},
        optional_env_args=frozenset(),
    ),
)

ALE_PY_PROVIDER = EnvProvider(
    provider_id="ale-py",
    import_name="ale_py",
    distribution_name="ale-py",
    environments={
        "breakout": EnvRegistration(
            "Breakout-Atari2600-v0",
            env_id_wandb_project_fallback=False,
        ),
        "ms_pacman": EnvRegistration(
            "MsPacman-Atari2600-v0",
            env_id_wandb_project_fallback=False,
        ),
    },
    supports_states=False,
    constructor_contract=ProviderConstructorContract(
        canonical_args=frozenset(
            {
                "frameskip",
                "game",
                "img_height",
                "img_width",
                "maxpool",
                "num_envs",
                "repeat_action_probability",
            }
        ),
        explicit_env_args=frozenset(
            {
                "autoreset_mode",
                "batch_size",
                "continuous",
                "continuous_action_threshold",
                "episodic_life",
                "full_action_space",
                "grayscale",
                "life_loss_info",
                "max_num_frames_per_episode",
                "noop_max",
                "num_threads",
                "stack_num",
                "thread_affinity_offset",
                "use_fire_reset",
            }
        ),
        required_values={"autoreset_mode": "next_step"},
    ),
)

GYMNASIUM_PROVIDER = EnvProvider(
    provider_id="gymnasium",
    import_name="gymnasium",
    distribution_name="gymnasium",
    environments={},
    supports_states=False,
    allows_unregistered_env_ids=True,
)

GRADLAB_PROVIDER = EnvProvider(
    provider_id="gradlab",
    import_name="gradlab",
    distribution_name="gradlab",
    environments={"Bandit-v0": EnvRegistration("Bandit-v0")},
    supports_states=False,
    constructor_contract=ProviderConstructorContract(
        canonical_args=frozenset({"game", "num_envs"}),
        explicit_env_args=frozenset({"autoreset_mode"}),
        required_values={"autoreset_mode": "disabled"},
    ),
)

ENV_PROVIDERS: dict[str, EnvProvider] = {
    GRADLAB_PROVIDER.provider_id: GRADLAB_PROVIDER,
    BREAKOUT_TURBO_ENV_PROVIDER.provider_id: BREAKOUT_TURBO_ENV_PROVIDER,
    STABLE_RETRO_TURBO_PROVIDER.provider_id: STABLE_RETRO_TURBO_PROVIDER,
    SUPERMARIOBROS_NES_TURBO_PROVIDER.provider_id: SUPERMARIOBROS_NES_TURBO_PROVIDER,
    VIZDOOM_TURBO_PROVIDER.provider_id: VIZDOOM_TURBO_PROVIDER,
    ALE_PY_PROVIDER.provider_id: ALE_PY_PROVIDER,
    GYMNASIUM_PROVIDER.provider_id: GYMNASIUM_PROVIDER,
}

STABLE_RETRO_ATARI_ENV_IDS = frozenset({"Breakout-Atari2600-v0", "MsPacman-Atari2600-v0"})


def _environment_identity(provider_id: object, env_id: object) -> tuple[str, str]:
    provider = str(provider_id or "").strip()
    environment = str(env_id or "").strip()
    if not provider and ":" in environment:
        provider, environment = environment.split(":", 1)
    return provider, environment


def _canonical_identity_by_env_id(
    env_id: str,
    *,
    fallback_field: str,
) -> EnvironmentSpec | None:
    matches = {
        ENVIRONMENT_SPECS[registration.spec_id]
        for provider in ENV_PROVIDERS.values()
        if (registration := provider.environments.get(env_id)) is not None
        and getattr(registration, fallback_field)
    }
    if len(matches) == 1:
        return matches.pop()
    return None


def _canonical_environment_identity(
    provider_id: object,
    env_id: object,
    *,
    fallback_field: str,
    allow_env_id_fallback: bool = True,
) -> tuple[str, EnvironmentSpec | None]:
    """Resolve a registered public identity while preserving historical reads."""

    provider, environment = _environment_identity(provider_id, env_id)
    registration = (
        ENV_PROVIDERS[provider].environments.get(environment) if provider in ENV_PROVIDERS else None
    )
    identity = ENVIRONMENT_SPECS[registration.spec_id] if registration is not None else None
    if identity is None and allow_env_id_fallback:
        identity = _canonical_identity_by_env_id(
            environment,
            fallback_field=fallback_field,
        )
    return environment, identity


def environment_spec(provider_id: object, env_id: object) -> EnvironmentSpec:
    """Return the one registered environment contract or a generic runtime spec."""

    environment, spec = _canonical_environment_identity(
        provider_id,
        env_id,
        fallback_field="env_id_game_family_fallback",
    )
    if spec is not None:
        return spec
    return EnvironmentSpec(
        spec_id=environment or "environment",
        game_family=_fallback_game_family(environment, fallback="environment"),
        wandb_project=environment or "environment",
    )


def _fallback_game_family(env_id: str, *, fallback: str) -> str:
    value = env_id or fallback
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", value)
    return re.sub(r"[^a-z0-9]+", "-", words.lower()).strip("-") or "environment"


def game_family_for_environment(
    provider_id: object,
    env_id: object,
    *,
    strict: bool = False,
    fallback: str = "environment",
) -> str:
    """Return the provider-neutral public family for an environment.

    Training metadata retains the historical fallback for arbitrary Gymnasium
    environments. Publication passes ``strict=True`` and therefore requires an
    explicit registered family rather than guessing a public model identity.
    """

    environment, identity = _canonical_environment_identity(
        provider_id,
        env_id,
        fallback_field="env_id_game_family_fallback",
        allow_env_id_fallback=not strict,
    )
    if identity is not None:
        return identity.game_family
    if strict:
        provider, _environment = _environment_identity(provider_id, env_id)
        qualified = f"{provider}:{environment}" if provider else environment
        raise ValueError(f"environment {qualified!r} has no registered canonical game family")
    return _fallback_game_family(environment, fallback=fallback)


def wandb_project_for_environment(
    provider_id: object,
    env_id: object,
    *,
    fallback: str,
) -> str:
    """Return the registered W&B project with historical providerless fallback."""

    environment, identity = _canonical_environment_identity(
        provider_id,
        env_id,
        fallback_field="env_id_wandb_project_fallback",
    )
    if identity is not None:
        return identity.wandb_project
    return environment or fallback


def is_stable_retro_atari_env(provider_id: str, game: str) -> bool:
    return (
        str(provider_id) == STABLE_RETRO_TURBO_PROVIDER.provider_id
        and str(game) in STABLE_RETRO_ATARI_ENV_IDS
    )


def env_supports_states(provider_id: str, game: str) -> bool:
    provider = resolve_env_provider(provider_id)
    return provider.supports_states


def registered_env_ids() -> tuple[str, ...]:
    return tuple(
        f"{provider.provider_id}:{env_id}"
        for provider in ENV_PROVIDERS.values()
        for env_id in provider.env_ids
    )


def resolve_env_provider(provider_id: str) -> EnvProvider:
    provider_id = str(provider_id).strip()
    if not provider_id:
        raise ValueError("environment provider id is required")
    provider = ENV_PROVIDERS.get(provider_id)
    if provider is None:
        known = ", ".join(sorted(ENV_PROVIDERS))
        raise ValueError(f"unknown environment provider {provider_id!r}; known providers: {known}")
    return provider


def validate_provider_constructor_args(
    provider_id: str,
    env_args: Any,
    *,
    label: str,
) -> None:
    provider = resolve_env_provider(provider_id)
    if not isinstance(env_args, Mapping):
        raise ValueError(f"{label} must explicitly define provider arguments")
    provider_reward_args = sorted(set(env_args) & PROVIDER_REWARD_TRANSFORM_KEYS)
    if provider_reward_args:
        raise ValueError(
            f"{label} configures provider-side reward transform(s) "
            f"{provider_reward_args}; use task.reward.reward_scale and "
            "task.reward.reward_clip instead"
        )
    contract = provider.constructor_contract
    if contract is None:
        return
    actual_args = set(env_args)
    missing_args = sorted(contract.explicit_env_args - actual_args)
    if missing_args:
        raise ValueError(
            f"{label} missing explicit {provider.provider_id} constructor argument(s): "
            + ", ".join(missing_args)
        )
    unexpected_args = sorted(actual_args - contract.explicit_env_args - contract.optional_env_args)
    if unexpected_args:
        raise ValueError(
            f"{label} has unexpected or canonically-owned {provider.provider_id} "
            f"constructor argument(s): {', '.join(unexpected_args)}"
        )
    for key, expected in contract.required_values.items():
        actual = env_args.get(key)
        if actual == expected:
            continue
        expected_text = "null" if expected is None else repr(expected)
        raise ValueError(f"{label}.{key} must be {expected_text}; got {actual!r}")


def qualify_env_id(provider_id: str, provider_env_id: str) -> str:
    provider_env_id = str(provider_env_id).strip()
    if not provider_env_id:
        raise ValueError("provider environment id is required")
    provider = resolve_env_provider(provider_id)
    if not provider.allows_unregistered_env_ids and provider_env_id not in provider.env_ids:
        known = ", ".join(provider.env_ids)
        raise ValueError(
            f"provider {provider.provider_id!r} does not register environment {provider_env_id!r}; "
            f"known envs: {known}"
        )
    return f"{provider.provider_id}:{provider_env_id}"


def resolve_env_id(env_id: str) -> ResolvedEnvId:
    value = str(env_id).strip()
    if ":" not in value:
        raise ValueError(
            f"environment env_id must be fully qualified as <provider>:<env>, got {value!r}"
        )
    provider_id, provider_env_id = value.split(":", 1)
    provider_id = provider_id.strip()
    provider_env_id = provider_env_id.strip()
    if not provider_id or not provider_env_id:
        raise ValueError(
            f"environment env_id must be fully qualified as <provider>:<env>, got {value!r}"
        )
    provider = resolve_env_provider(provider_id)
    if not provider.allows_unregistered_env_ids and provider_env_id not in provider.env_ids:
        known = ", ".join(provider.env_ids)
        raise ValueError(
            f"provider {provider_id!r} does not register environment {provider_env_id!r}; "
            f"known envs: {known}"
        )
    return ResolvedEnvId(
        qualified_id=f"{provider.provider_id}:{provider_env_id}",
        provider_id=provider.provider_id,
        provider_env_id=provider_env_id,
        import_name=provider.import_name,
    )
