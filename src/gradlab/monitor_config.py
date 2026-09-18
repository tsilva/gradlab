"""Frozen monitoring limits and calibration admission; never changes a Goal."""

from dataclasses import asdict, dataclass
from collections.abc import Mapping
import math

from gradlab.json_utils import canonical_json_sha256


@dataclass(frozen=True)
class MonitoringConfig:
    enabled: bool = False
    episodes: int = 400
    active_workers: int = 1
    task_cpus: int = 1
    worker_memory_bytes: int = 2 * 1024**3
    worker_spool_bytes: int = 512 * 1024**2
    memory_bytes: int = 2 * 1024**3
    spool_bytes: int = 512 * 1024**2
    contribution_bytes: int = 10 * 1024**3
    chunk_bytes: int = 32 * 1024**2
    scratch_headroom_bytes: int = 1024**3
    whole_run_seconds: int = 3600
    watchdog_steps: int = 1_000_000
    calibration: dict | None = None


def calibration_binding(train, settings):
    # Scientific, producer and runtime inputs are retained; launch identities and
    # the evidence report itself are excluded from this reusable configuration key.
    fields = (
        "game",
        "env_provider",
        "env_args",
        "task",
        "frame_skip",
        "max_pool_frames",
        "sticky_action_prob",
        "obs_resize",
        "obs_crop",
        "obs_crop_mode",
        "obs_crop_fill",
        "obs_resize_algorithm",
        "training_backend",
        "policy_model",
        "n_envs",
        "frame_stack",
        "model_inputs",
        "reward_scale",
        "clip_reward",
        "action_profile_sha256",
        "torch_threads",
        "timesteps",
        "checkpoint_freq",
        "effective_goal_contract_sha256",
        "state",
        "states",
        "state_probs",
    )
    return canonical_json_sha256(
        {
            "train": {k: train.get(k) for k in fields},
            "monitoring": {
                k: v for k, v in settings.items() if k not in {"calibration", "enabled"}
            },
            "schema": "gradlab.checkpoint-monitoring.v1",
        }
    )


def resolve_monitoring(value, train):
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) - set(MonitoringConfig.__dataclass_fields__):
        raise ValueError("checkpoint_monitoring contains unknown settings")
    settings = asdict(MonitoringConfig(**value))
    if type(settings["enabled"]) is not bool:
        raise ValueError("checkpoint_monitoring.enabled must be boolean")
    for key, number in settings.items():
        if key in {"enabled", "calibration"}:
            continue
        if type(number) is not int or number <= 0:
            raise ValueError(f"checkpoint_monitoring.{key} must be a positive finite integer")
    if not 1024**2 <= settings["chunk_bytes"] <= 128 * 1024**2:
        raise ValueError("monitoring chunks must be between 1 and 128 MiB")
    if settings["episodes"] > 100_000 or settings["task_cpus"] > 1024:
        raise ValueError("monitoring manifest or worker count exceeds supported bounds")
    if settings["active_workers"] > settings["task_cpus"]:
        raise ValueError("active monitoring workers exceed the task CPU allocation")
    if (
        settings["memory_bytes"] < settings["task_cpus"] * settings["worker_memory_bytes"]
        or settings["spool_bytes"] < settings["task_cpus"] * settings["worker_spool_bytes"]
        or settings["worker_memory_bytes"] < 8 * settings["chunk_bytes"]
        or settings["worker_spool_bytes"] < 4 * settings["chunk_bytes"]
    ):
        raise ValueError("monitoring budgets cannot support bounded full-allocation finalization")
    if settings["enabled"]:
        if (
            train.get("game") != "Breakout-Atari2600-v0"
            or train.get("env_provider") != "env-breakoutatari2600-turbo-native"
            or (train.get("training_backend") or {}).get("id")
            not in {"gradlab.ppo", "sb3.ppo", "sb3.a2c"}
            or train.get("sticky_action_prob", 0) != 0
        ):
            raise ValueError("monitoring requires verified native Breakout PPO/A2C")
        report = settings["calibration"]
        if not isinstance(report, dict) or report.get("status") != "supported":
            raise ValueError("monitoring requires complete supported calibration before launch")
        if report.get("binding") != calibration_binding(train, settings):
            raise ValueError("monitoring calibration is stale for the resolved Run Configuration")
        if report.get("episodes") != settings["episodes"]:
            raise ValueError("monitoring calibration cannot change the episode count")
        for name in (
            "maximum_throughput_loss_upper95",
            "estimated_retained_bytes",
            "estimated_total_seconds",
        ):
            number = report.get(name)
            if (
                isinstance(number, bool)
                or not isinstance(number, int | float)
                or not math.isfinite(number)
            ):
                raise ValueError(f"monitoring calibration requires finite {name}")
        if report.get("maximum_throughput_loss_upper95", 1) > 0.02:
            raise ValueError("monitoring calibration does not establish the 2% throughput target")
        if (
            report.get("estimated_retained_bytes", float("inf")) > settings["contribution_bytes"]
            or report.get("estimated_total_seconds", float("inf")) > settings["whole_run_seconds"]
        ):
            raise ValueError("monitoring calibration is infeasible within declared budgets")
    return settings


def validate_monitoring_allocation(settings, *, source_sha, image_digest, resources, duration):
    """Check the real selected allocation before any Run is admitted."""
    if not settings or not settings.get("enabled"):
        return
    report = settings["calibration"]
    if report.get("source_sha") != source_sha or report.get("runtime_image") != image_digest:
        raise ValueError("monitoring calibration source/runtime differs from the launch")
    if report.get("hardware_allocation") != resources:
        raise ValueError("monitoring calibration hardware allocation differs from the launch")
    if settings["task_cpus"] != resources["cpu"]:
        raise ValueError("monitoring must declare the full task CPU allocation")
    if settings["whole_run_seconds"] > duration:
        raise ValueError("monitoring whole-run deadline exceeds the task duration")
