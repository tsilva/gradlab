from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, StringConstraints, field_validator, model_validator

from gradlab.boundary_schema import BoundaryModel, validate_boundary
from gradlab.config_loader import load_mapping_document
from gradlab.metric_names import validate_metric_name
from gradlab.recipe_documents import compose_train_document
from gradlab.train_config import validate_and_normalize_train_config
from gradlab.validation import int_list, require_mapping, string_list


BENCHMARK_PROFILE_SCHEMA_VERSION = 1
DEFAULT_PROFILE_DIR = Path("experiments/benchmarks/profiles")
DEFAULT_RESULT_DIR = Path("logs/benchmarks")
STATE_NONE_VALUES = {"", "none", "state.none"}
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
NonNegativeNumber = Annotated[int | float, Field(ge=0)]


class _Profile(BoundaryModel):
    schema_version: Literal[BENCHMARK_PROFILE_SCHEMA_VERSION]
    name: NonEmptyText
    description: str = ""


class _EnvironmentThroughputProfile(_Profile):
    kind: Literal["env_throughput"]
    env_provider: NonEmptyText
    game: NonEmptyText
    state: NonEmptyText
    modes: list[NonEmptyText] = ["compare"]
    envs: list[PositiveInt] = [1]
    steps: PositiveInt
    warmup: NonNegativeInt
    max_runtime_overhead: NonNegativeNumber | None = None
    allow_state_none: bool = False
    repeats: PositiveInt | None = None
    script: NonEmptyText | None = None
    seed: NonNegativeInt | None = None

    @model_validator(mode="after")
    def validate_gate(self) -> "_EnvironmentThroughputProfile":
        if self.state.strip().lower() in STATE_NONE_VALUES and not self.allow_state_none:
            raise ValueError(
                f"state must be an actual saved state for {self.game}; "
                "set allow_state_none=true only for emulator hot-path diagnostics"
            )
        if self.modes != ["compare"]:
            raise ValueError("modes must be [compare] for an executable overhead gate")
        return self


class _LocalSmokeProfile(_Profile):
    kind: Literal["local_smoke"]
    goal_file: NonEmptyText
    recipe_file: NonEmptyText
    target: NonEmptyText
    seed: NonNegativeInt
    max_duration: NonEmptyText | None = None
    runtime_image_ref_file: NonEmptyText | None = None


class _MetricProfile(_Profile):
    recipe_overrides: list[NonEmptyText] = []
    required_metrics: list[NonEmptyText]
    run_description: str = ""
    seed: NonNegativeInt | None = None

    @field_validator("required_metrics")
    @classmethod
    def validate_required_metrics(cls, values: list[str]) -> list[str]:
        if not values:
            raise ValueError("must not be empty")
        for metric_name in values:
            validate_metric_name(metric_name)
        return values


class _TrainLoopThroughputProfile(_MetricProfile):
    kind: Literal["train_loop_throughput"]
    goal_file: NonEmptyText
    recipe_file: NonEmptyText
    run_name: str = ""


class _TrainLoopComparisonProfile(_MetricProfile):
    kind: Literal["train_loop_comparison"]
    goal_file: NonEmptyText
    baseline_recipe_file: NonEmptyText
    candidate_recipe_file: NonEmptyText
    candidate_required_metrics: list[NonEmptyText] = []
    repeats: PositiveInt | None = None
    max_candidate_slowdown: Annotated[int | float, Field(ge=0, lt=1)]

    @field_validator("candidate_required_metrics")
    @classmethod
    def validate_candidate_metrics(cls, values: list[str]) -> list[str]:
        for metric_name in values:
            validate_metric_name(metric_name)
        return values


_PROFILE_MODELS = {
    "env_throughput": _EnvironmentThroughputProfile,
    "local_smoke": _LocalSmokeProfile,
    "train_loop_comparison": _TrainLoopComparisonProfile,
    "train_loop_throughput": _TrainLoopThroughputProfile,
}


@dataclass(frozen=True)
class BenchmarkCommand:
    label: str
    argv: tuple[str, ...]
    cwd: Path | None = None
    env: Mapping[str, str] | None = None
    stdin: str | None = None

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "label": self.label,
            "argv": list(self.argv),
        }
        if self.cwd is not None:
            payload["cwd"] = str(self.cwd)
        if self.env:
            payload["env"] = dict(self.env)
        if self.stdin is not None:
            payload["stdin_json"] = json.loads(self.stdin)
        return payload


@dataclass(frozen=True)
class BenchmarkProfile:
    path: Path
    payload: dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.payload["name"])

    @property
    def kind(self) -> str:
        return str(self.payload["kind"])

    @property
    def description(self) -> str:
        return str(self.payload.get("description") or "")


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip().lower()).strip("-")
    return text or "benchmark"


def _profile_payload(path: Path) -> dict[str, Any]:
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"benchmark profile must be YAML: {path}")
    return load_mapping_document(path, label=f"benchmark profile {path}")


def validate_benchmark_profile(payload: Mapping[str, Any], *, label: str = "profile") -> None:
    kind = payload.get("kind")
    model = _PROFILE_MODELS.get(kind) if isinstance(kind, str) else None
    if model is None:
        known = ", ".join(sorted(_PROFILE_MODELS))
        raise ValueError(f"{label}.kind must be one of {known}")
    if "expectations" in payload or "gates" in payload:
        raise ValueError(
            f"{label} must use executable kind-specific gate fields, not expectations or gates"
        )
    if "environment_contract" in payload:
        raise ValueError(
            f"{label}.environment_contract is unsupported; derive it from executed inputs"
        )
    validated = validate_boundary(model, payload, label=label)

    if (
        kind == "env_throughput"
        and isinstance(validated, _EnvironmentThroughputProfile)
        and validated.max_runtime_overhead is None
    ):
        raise ValueError(f"{label}.max_runtime_overhead must be a non-negative number")

    if kind == "train_loop_throughput":
        config = _train_loop_config(payload)
        if not str(config.get("game") or "").strip():
            raise ValueError(f"{label}.train_config.game must be a non-empty string")
        if not isinstance(config.get("timesteps"), int) or isinstance(
            config.get("timesteps"), bool
        ):
            raise ValueError(f"{label}.train_config.timesteps must be an integer")
        if isinstance(validated, _TrainLoopThroughputProfile) and validated.seed is None:
            raise ValueError(f"{label}.seed must be an integer")

    if kind == "train_loop_comparison":
        for recipe_field in ("baseline_recipe_file", "candidate_recipe_file"):
            _train_loop_config(payload, recipe_file=str(payload[recipe_field]))
        if isinstance(validated, _TrainLoopComparisonProfile) and validated.seed is None:
            raise ValueError(f"{label}.seed must be an integer")


def load_benchmark_profile(path: Path) -> BenchmarkProfile:
    payload = _profile_payload(path)
    validate_benchmark_profile(payload, label=f"profile file {path}")
    return BenchmarkProfile(path=path, payload=dict(payload))


def load_benchmark_profiles(profile_dir: Path = DEFAULT_PROFILE_DIR) -> list[BenchmarkProfile]:
    if not profile_dir.is_dir():
        raise ValueError(f"benchmark profile directory does not exist: {profile_dir}")
    paths = sorted([*profile_dir.glob("*.yaml"), *profile_dir.glob("*.yml")])
    return [load_benchmark_profile(path) for path in paths]


def find_benchmark_profile(
    name_or_path: str, *, profile_dir: Path = DEFAULT_PROFILE_DIR
) -> BenchmarkProfile:
    candidate = Path(name_or_path)
    if candidate.is_file():
        return load_benchmark_profile(candidate)
    for profile in load_benchmark_profiles(profile_dir):
        if profile.name == name_or_path or profile.path.stem == name_or_path:
            return profile
    raise ValueError(f"unknown benchmark profile {name_or_path!r}")


def _command(
    label: str,
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    stdin: str | None = None,
) -> BenchmarkCommand:
    return BenchmarkCommand(
        label=label,
        argv=tuple(str(part) for part in argv),
        cwd=cwd,
        env=env,
        stdin=stdin,
    )


def _local_smoke_commands(profile: Mapping[str, Any]) -> list[BenchmarkCommand]:
    goal_file = str(profile["goal_file"])
    recipe_file = str(profile["recipe_file"])
    target = str(profile.get("target") or "b3")
    enqueue = [
        "gradlab",
        "experiment",
        "launch",
        "--goal-file",
        goal_file,
        "--recipe-file",
        recipe_file,
        "--seed",
        str(profile.get("seed", 123)),
        "--run-description",
        str(profile.get("description") or "dstack local integration smoke"),
        "--compute",
        "local",
        "--target",
        target,
        "--max-duration",
        str(profile.get("max_duration") or "30m"),
        "--json",
    ]
    if profile.get("runtime_image_ref_file"):
        enqueue.extend(["--runtime-image-ref-file", str(profile["runtime_image_ref_file"])])
    return [_command("train-local-smoke", enqueue)]


def _env_throughput_commands(profile: Mapping[str, Any]) -> list[BenchmarkCommand]:
    script = str(profile.get("script") or "experiments/scripts/benchmarks/benchmark_env_sps.py")
    commands: list[BenchmarkCommand] = []
    for mode in string_list(profile.get("modes", ["compare"]), label="modes"):
        for envs in int_list(profile.get("envs", [1]), label="envs"):
            commands.append(
                _command(
                    f"{mode}-{envs}env",
                    [
                        sys.executable,
                        script,
                        "--env-provider",
                        str(profile["env_provider"]),
                        "--game",
                        str(profile["game"]),
                        "--state",
                        str(profile["state"]),
                        "--mode",
                        mode,
                        "--envs",
                        str(envs),
                        "--steps",
                        str(profile["steps"]),
                        "--warmup",
                        str(profile["warmup"]),
                        "--seed",
                        str(profile.get("seed", 123)),
                        "--repeats",
                        str(profile.get("repeats", 3)),
                        "--max-overhead",
                        str(profile.get("max_runtime_overhead", 0.05)),
                    ],
                    env={"STABLE_RETRO_DISABLE_AUDIO": "1"},
                )
            )
    return commands


def _train_loop_config(
    profile: Mapping[str, Any],
    *,
    recipe_file: str | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    document = compose_train_document(
        Path(str(profile["goal_file"])),
        Path(str(recipe_file or profile["recipe_file"])),
        recipe_overrides=profile.get("recipe_overrides", ()),
    )
    config = dict(require_mapping(document["train_config"], label="train_config"))
    config["checkpoint_eval_backend"] = "none"
    config["early_stop"] = None
    config["post_train_eval_episodes"] = 0
    config.pop("rom_asset_manifest", None)
    config["wandb_mode"] = "disabled"
    config["seed"] = int(profile.get("seed", 123))
    config["run_name"] = str(
        run_name or profile.get("run_name") or f"benchmark_{_slug(str(profile['name']))}"
    )
    config["run_description"] = str(
        profile.get("run_description")
        or f"Benchmark profile {profile['name']} training-loop probe."
    )
    return validate_and_normalize_train_config(config, label="train_loop_benchmark.train_config")


def _train_loop_commands(profile: Mapping[str, Any]) -> list[BenchmarkCommand]:
    config = _train_loop_config(profile)
    return [
        _command(
            "train",
            [
                sys.executable,
                "-m",
                "gradlab.train",
                "--train-config-json",
                "/dev/stdin",
                "--execution-mode",
                "supervised",
            ],
            env={"GRADLAB_INTERNAL_LEARNER": "1"},
            stdin=json.dumps(config, sort_keys=True),
        )
    ]


def _train_loop_comparison_commands(profile: Mapping[str, Any]) -> list[BenchmarkCommand]:
    commands: list[BenchmarkCommand] = []
    variants = {
        "baseline": str(profile["baseline_recipe_file"]),
        "candidate": str(profile["candidate_recipe_file"]),
    }
    repeats = int(profile.get("repeats", 2))
    for repeat in range(repeats):
        order = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
        for variant in order:
            label = f"{variant}-{repeat + 1}"
            config = _train_loop_config(
                profile,
                recipe_file=variants[variant],
                run_name=f"benchmark_{_slug(str(profile['name']))}_{label}",
            )
            commands.append(
                _command(
                    label,
                    [
                        sys.executable,
                        "-m",
                        "gradlab.train",
                        "--train-config-json",
                        "/dev/stdin",
                        "--execution-mode",
                        "supervised",
                    ],
                    env={"GRADLAB_INTERNAL_LEARNER": "1"},
                    stdin=json.dumps(config, sort_keys=True),
                )
            )
    return commands


def build_benchmark_commands(profile: BenchmarkProfile) -> list[BenchmarkCommand]:
    payload = profile.payload
    kind = profile.kind
    if kind == "local_smoke":
        return _local_smoke_commands(payload)
    if kind == "env_throughput":
        return _env_throughput_commands(payload)
    if kind == "train_loop_throughput":
        return _train_loop_commands(payload)
    if kind == "train_loop_comparison":
        return _train_loop_comparison_commands(payload)
    raise ValueError(f"unsupported benchmark profile kind {kind!r}")
