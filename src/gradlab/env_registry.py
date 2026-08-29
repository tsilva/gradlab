from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from gradlab.reward_transform import PROVIDER_REWARD_TRANSFORM_KEYS
from gradlab.gymnasium_vec_env import GYMNASIUM_ENV_IDS


EXTERNAL_ROM_ASSET_NONE = "none"
STABLE_RETRO_DIRECT_PATH_V1 = "stable_retro_direct_path_v1"
EXTERNAL_ROM_ASSET_STRATEGIES = frozenset({EXTERNAL_ROM_ASSET_NONE, STABLE_RETRO_DIRECT_PATH_V1})


@dataclass(frozen=True)
class ProviderConstructorContract:
    canonical_args: frozenset[str]
    explicit_env_args: frozenset[str]
    required_values: Mapping[str, object]
    optional_env_args: frozenset[str] = frozenset()
    required_config_values: Mapping[str, object] = field(default_factory=dict)
    required_nested_values: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    allowed_nested_args: Mapping[str, frozenset[str]] = field(default_factory=dict)

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
        unknown_config = set(self.required_config_values) - set(self.canonical_args)
        if unknown_config:
            raise ValueError(
                f"provider required config values are not canonical args: {sorted(unknown_config)}"
            )
        nested_parents = set(self.required_nested_values) | set(self.allowed_nested_args)
        unknown_nested = nested_parents - set(self.explicit_env_args)
        if unknown_nested:
            raise ValueError(
                f"provider nested contracts are not explicit env args: {sorted(unknown_nested)}"
            )
        required_nested_values = {
            str(parent): MappingProxyType(dict(values))
            for parent, values in self.required_nested_values.items()
        }
        allowed_nested_args = {
            str(parent): frozenset(values) for parent, values in self.allowed_nested_args.items()
        }
        for parent, values in required_nested_values.items():
            unknown_children = set(values) - set(allowed_nested_args.get(parent, ()))
            if parent in allowed_nested_args and unknown_children:
                raise ValueError(
                    f"provider required nested values are not allowed for {parent}: "
                    f"{sorted(unknown_children)}"
                )
        object.__setattr__(self, "required_values", MappingProxyType(dict(self.required_values)))
        object.__setattr__(
            self,
            "required_config_values",
            MappingProxyType(dict(self.required_config_values)),
        )
        object.__setattr__(
            self,
            "required_nested_values",
            MappingProxyType(required_nested_values),
        )
        object.__setattr__(self, "allowed_nested_args", MappingProxyType(allowed_nested_args))


@dataclass(frozen=True)
class NativeEpisodeHorizonContract:
    """Describe a provider-owned episode horizon in its resolved environment config."""

    env_args_path: tuple[str, ...]
    unit: str
    action_repeat_key: str = "frame_skip"
    truncation_env_arg: str | None = None

    def __post_init__(self) -> None:
        if not self.env_args_path or any(not str(part).strip() for part in self.env_args_path):
            raise ValueError("native episode horizon path must not be empty")
        if not self.unit.strip():
            raise ValueError("native episode horizon unit must not be empty")
        if not self.action_repeat_key.strip():
            raise ValueError("native episode horizon action repeat key must not be empty")


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
    policy_compatibility_id: str | None = None


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

VIZDOOM_DEATHMATCH_EVAL_SEMANTICS = EvalSemantics(
    progress_fields=(EvalProgressField("killcount", "kills", rank=True),),
    best_episode_rank=("progress", "reward"),
)

ENVIRONMENT_SPECS: Mapping[str, EnvironmentSpec] = MappingProxyType(
    {
        "Bandit-v0": EnvironmentSpec("Bandit-v0", "Bandit", "Bandit-v0"),
        "CartPole-v1": EnvironmentSpec(
            "CartPole-v1", "Gymnasium-CartPole", "CartPole-v1"
        ),
        "MountainCar-v0": EnvironmentSpec(
            "MountainCar-v0", "Gymnasium-MountainCar", "MountainCar-v0"
        ),
        "Acrobot-v1": EnvironmentSpec(
            "Acrobot-v1", "Gymnasium-Acrobot", "Acrobot-v1"
        ),
        "LunarLander-v3": EnvironmentSpec(
            "LunarLander-v3", "Gymnasium-LunarLander", "LunarLander-v3"
        ),
        "FrozenLake-v1": EnvironmentSpec(
            "FrozenLake-v1", "Gymnasium-FrozenLake", "FrozenLake-v1"
        ),
        "FrozenLake8x8-v1": EnvironmentSpec(
            "FrozenLake8x8-v1", "Gymnasium-FrozenLake8x8", "FrozenLake8x8-v1"
        ),
        "CliffWalking-v1": EnvironmentSpec(
            "CliffWalking-v1", "Gymnasium-CliffWalking", "CliffWalking-v1"
        ),
        "CliffWalkingSlippery-v1": EnvironmentSpec(
            "CliffWalkingSlippery-v1",
            "Gymnasium-CliffWalkingSlippery",
            "CliffWalkingSlippery-v1",
        ),
        "Taxi-v3": EnvironmentSpec("Taxi-v3", "Gymnasium-Taxi", "Taxi-v3"),
        "Blackjack-v1": EnvironmentSpec(
            "Blackjack-v1", "Gymnasium-Blackjack", "Blackjack-v1"
        ),
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
            eval_semantics=VIZDOOM_DEATHMATCH_EVAL_SEMANTICS,
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
    native_episode_horizon: NativeEpisodeHorizonContract | None = None

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


@dataclass(frozen=True)
class ResolvedNativeEpisodeHorizon:
    value: int
    unit: str
    action_repeat: int

    @property
    def watchdog_steps(self) -> int:
        return math.ceil(self.value / self.action_repeat)


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
        "transport",
        "info_frame_stack_keys",
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
    provider_id="env-stableretro-turbo",
    import_name="stable_retro",
    distribution_name="env-stableretro-turbo",
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
    turbo_api_version=2,
    constructor_contract=ProviderConstructorContract(
        canonical_args=_TURBO_CANONICAL_ARGS,
        explicit_env_args=_TURBO_EXPLICIT_ENV_ARGS,
        required_values={},
    ),
)

SUPERMARIOBROS_NES_TURBO_PROVIDER = EnvProvider(
    provider_id="env-supermariobrosnes-turbo-emu",
    import_name="supermariobrosnes_turbo",
    distribution_name="env-supermariobrosnes-turbo-emu",
    environments={
        "SuperMarioBros-Nes-v0": EnvRegistration("SuperMarioBros-Nes-v0"),
    },
    external_rom_asset_strategy=STABLE_RETRO_DIRECT_PATH_V1,
    turbo_api_version=2,
    constructor_contract=ProviderConstructorContract(
        canonical_args=_TURBO_CANONICAL_ARGS,
        explicit_env_args=_TURBO_EXPLICIT_ENV_ARGS,
        required_values={},
    ),
)

VIZDOOM_TURBO_PROVIDER = EnvProvider(
    provider_id="env-vizdoom-turbo",
    import_name="vizdoom_turbo",
    distribution_name="env-vizdoom-turbo",
    environments={
        spec_id: EnvRegistration(
            spec_id,
            policy_compatibility_id=(
                "doom-deathmatch-p1-v1" if spec_id == "VizdoomDeathmatch-v1" else None
            ),
        )
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
    turbo_api_version=2,
    native_episode_horizon=NativeEpisodeHorizonContract(
        env_args_path=("vizdoom_config", "episode_timeout"),
        unit="tics",
        truncation_env_arg="treat_episode_timeout_as_truncation",
    ),
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

GRADOOM_PROVIDER = EnvProvider(
    provider_id="env-gradoom-turbo-torch",
    import_name="gradoom",
    distribution_name="env-gradoom-turbo-torch",
    environments={
        "VizdoomDeathmatch-v1": EnvRegistration(
            "VizdoomDeathmatch-v1",
            policy_compatibility_id="doom-deathmatch-p1-v1",
        ),
    },
    turbo_api_version=2,
    native_episode_horizon=NativeEpisodeHorizonContract(
        env_args_path=("vizdoom_config", "episode_timeout"),
        unit="tics",
        truncation_env_arg="treat_episode_timeout_as_truncation",
    ),
    constructor_contract=ProviderConstructorContract(
        canonical_args=_TURBO_CANONICAL_ARGS | {"device"},
        explicit_env_args=_TURBO_EXPLICIT_ENV_ARGS
        | {
            "doom_map",
            "doom_skill",
            "game_args",
            "game_variables",
            "treat_episode_timeout_as_truncation",
            "vizdoom_config",
        },
        required_values={"doom_skill": 3},
        optional_env_args=frozenset({"compile_engine"}),
        required_config_values={
            "obs_crop": (0, 32, 0, 0),
            "obs_crop_mode": "mask",
            "obs_crop_fill": 0,
        },
        required_nested_values={"vizdoom_config": {"render_hud": False}},
        allowed_nested_args={
            "vizdoom_config": frozenset(
                {"episode_timeout", "render_hud", "render_screen_flashes"}
            )
        },
    ),
)

BREAKOUT_TURBO_ENV_PROVIDER = EnvProvider(
    provider_id="env-breakoutatari2600-turbo-native",
    import_name="env_breakoutatari2600_turbo_native",
    distribution_name="env-breakoutatari2600-turbo-native",
    environments={
        "Breakout-Atari2600-v0": EnvRegistration("Breakout-Atari2600-v0"),
    },
    turbo_api_version=2,
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
        "breakout": EnvRegistration("Breakout-Atari2600-v0"),
        "ms_pacman": EnvRegistration("MsPacman-Atari2600-v0"),
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
    environments={env_id: EnvRegistration(env_id) for env_id in GYMNASIUM_ENV_IDS},
    supports_states=False,
    turbo_api_version=2,
    constructor_contract=ProviderConstructorContract(
        canonical_args=frozenset({"game", "num_envs"}),
        explicit_env_args=frozenset(
            {
                "autoreset_mode",
                "copy",
                "daemon",
                "multiprocessing_context",
                "observation_mode",
                "render_mode",
                "shared_memory",
                "vectorization_mode",
            }
        ),
        required_values={
            "autoreset_mode": "disabled",
            "copy": True,
            "daemon": True,
            "multiprocessing_context": "spawn",
            "observation_mode": "same",
            "render_mode": "rgb_array",
            "shared_memory": True,
            "vectorization_mode": "async",
        },
    ),
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
    GRADOOM_PROVIDER.provider_id: GRADOOM_PROVIDER,
    ALE_PY_PROVIDER.provider_id: ALE_PY_PROVIDER,
    GYMNASIUM_PROVIDER.provider_id: GYMNASIUM_PROVIDER,
}

STABLE_RETRO_ATARI_ENV_IDS = frozenset({"Breakout-Atari2600-v0", "MsPacman-Atari2600-v0"})


def _environment_identity(provider_id: object, env_id: object) -> tuple[str, str]:
    provider = str(provider_id or "").strip()
    environment = str(env_id or "").strip()
    if not provider:
        raise ValueError("environment provider identity is required")
    if not environment:
        raise ValueError("environment id is required")
    return provider, environment


def _canonical_environment_identity(
    provider_id: object,
    env_id: object,
) -> tuple[str, EnvironmentSpec | None]:
    """Resolve an exact provider-owned public environment identity."""

    provider, environment = _environment_identity(provider_id, env_id)
    provider_spec = ENV_PROVIDERS.get(provider)
    if provider_spec is None:
        raise ValueError(f"unknown environment provider: {provider}")
    registration = provider_spec.environments.get(environment)
    if registration is None and not provider_spec.allows_unregistered_env_ids:
        raise ValueError(f"environment {environment!r} is not registered for provider {provider!r}")
    identity = ENVIRONMENT_SPECS[registration.spec_id] if registration is not None else None
    return environment, identity


def environment_spec(provider_id: object, env_id: object) -> EnvironmentSpec:
    """Return the one registered environment contract or a generic runtime spec."""

    environment, spec = _canonical_environment_identity(provider_id, env_id)
    if spec is not None:
        return spec
    return EnvironmentSpec(
        spec_id=environment or "environment",
        game_family=_fallback_game_family(environment, fallback="environment"),
        wandb_project=environment or "environment",
    )


def policy_environment_compatibility_id(provider_id: object, env_id: object) -> str | None:
    """Return an explicit cross-provider policy-transfer contract, when declared."""

    provider, environment = _environment_identity(provider_id, env_id)
    resolved = resolve_env_provider(provider)
    registration = resolved.environments.get(environment)
    return registration.policy_compatibility_id if registration is not None else None


def _fallback_game_family(env_id: str, *, fallback: str) -> str:
    value = env_id or fallback
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", value)
    return re.sub(r"[^a-z0-9]+", "-", words.lower()).strip("-") or "environment"


def game_family_for_environment(
    provider_id: object,
    env_id: object,
    *,
    require_registered: bool = False,
) -> str:
    """Return the provider-neutral public family for an environment."""

    environment, identity = _canonical_environment_identity(provider_id, env_id)
    if identity is not None:
        return identity.game_family
    if require_registered:
        provider, _environment = _environment_identity(provider_id, env_id)
        qualified = f"{provider}:{environment}"
        raise ValueError(f"environment {qualified!r} has no registered canonical game family")
    return _fallback_game_family(environment, fallback="environment")


def wandb_project_for_environment(
    provider_id: object,
    env_id: object,
) -> str:
    """Return the exact registered W&B project or current dynamic environment id."""

    environment, identity = _canonical_environment_identity(provider_id, env_id)
    if identity is not None:
        return identity.wandb_project
    return environment


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


def resolve_native_episode_horizon(
    environment: Mapping[str, Any],
) -> ResolvedNativeEpisodeHorizon | None:
    """Resolve one provider-native horizon from a flattened environment config."""

    provider_id = str(environment.get("env_provider") or "").strip()
    provider = resolve_env_provider(provider_id)
    contract = provider.native_episode_horizon
    if contract is None:
        return None
    env_args = environment.get("env_args")
    if not isinstance(env_args, Mapping):
        raise ValueError("native episode horizon requires materialized env_args")
    value: Any = env_args
    for part in contract.env_args_path:
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    if not isinstance(value, int) or isinstance(value, bool):
        path = ".".join(("env_args", *contract.env_args_path))
        raise ValueError(f"{path} must be an integer")
    if value < 0:
        path = ".".join(("env_args", *contract.env_args_path))
        raise ValueError(f"{path} must be non-negative")
    if value == 0:
        return None
    if (
        contract.truncation_env_arg is not None
        and env_args.get(contract.truncation_env_arg) is not True
    ):
        raise ValueError(
            f"env_args.{contract.truncation_env_arg} must be true when a native "
            "episode horizon is configured"
        )
    action_repeat = environment.get(contract.action_repeat_key)
    if not isinstance(action_repeat, int) or isinstance(action_repeat, bool) or action_repeat < 1:
        raise ValueError(f"{contract.action_repeat_key} must be a positive integer")
    return ResolvedNativeEpisodeHorizon(
        value=int(value),
        unit=contract.unit,
        action_repeat=int(action_repeat),
    )


def evaluation_watchdog_steps(environment: Mapping[str, Any]) -> int:
    """Derive an operational evaluation watchdog from scientific episode boundaries."""

    candidates: list[int] = []
    native = resolve_native_episode_horizon(environment)
    if native is not None:
        candidates.append(native.watchdog_steps)
    task = environment.get("task")
    termination = task.get("termination") if isinstance(task, Mapping) else None
    task_limit = termination.get("max_episode_steps") if isinstance(termination, Mapping) else None
    if task_limit is not None:
        if not isinstance(task_limit, int) or isinstance(task_limit, bool) or task_limit < 0:
            raise ValueError("task.termination.max_episode_steps must be non-negative")
        if task_limit > 0:
            candidates.append(int(task_limit))
    if not candidates:
        raise ValueError(
            "evaluated environment must materialize a positive native or task episode boundary"
        )
    return min(candidates)


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
        if _provider_contract_values_equal(actual, expected):
            continue
        expected_text = "null" if expected is None else repr(expected)
        raise ValueError(f"{label}.{key} must be {expected_text}; got {actual!r}")
    for parent, allowed in contract.allowed_nested_args.items():
        nested = env_args.get(parent)
        if not isinstance(nested, Mapping):
            raise ValueError(f"{label}.{parent} must be an object")
        unexpected_nested = sorted(set(nested) - set(allowed))
        if unexpected_nested:
            raise ValueError(
                f"{label}.{parent} has unsupported {provider.provider_id} option(s): "
                + ", ".join(unexpected_nested)
            )
    for parent, required in contract.required_nested_values.items():
        nested = env_args.get(parent)
        if not isinstance(nested, Mapping):
            raise ValueError(f"{label}.{parent} must be an object")
        for key, expected in required.items():
            actual = nested.get(key)
            if _provider_contract_values_equal(actual, expected):
                continue
            expected_text = "null" if expected is None else repr(expected)
            raise ValueError(
                f"{label}.{parent}.{key} must be {expected_text}; got {actual!r}"
            )


def _provider_contract_values_equal(actual: object, expected: object) -> bool:
    if isinstance(expected, Sequence) and not isinstance(expected, str | bytes):
        if not isinstance(actual, Sequence) or isinstance(actual, str | bytes):
            return False
        if len(actual) != len(expected):
            return False
        return all(
            _provider_contract_values_equal(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected)
        )
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    return actual == expected


def validate_provider_resolved_config(
    provider_id: str,
    config: Mapping[str, object] | object,
    *,
    label: str,
) -> None:
    provider = resolve_env_provider(provider_id)
    contract = provider.constructor_contract
    if contract is None:
        return
    for key, expected in contract.required_config_values.items():
        actual = config.get(key) if isinstance(config, Mapping) else getattr(config, key, None)
        if _provider_contract_values_equal(actual, expected):
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
