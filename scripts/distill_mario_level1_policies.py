from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import torch
from stable_baselines3 import PPO

from stable_retro_ppo.artifacts import (
    env_config_from_model_metadata,
    load_model_metadata,
    write_model_metadata,
)
from stable_retro_ppo.env import (
    EnvConfig,
    assert_rom_imported,
    make_training_vec_env,
    resolve_env_config,
    resolve_mixed_state_config,
)


DEFAULT_LEVEL1_1_TEACHER = Path(
    "runs/wandb_artifacts/"
    "tsilva_SuperMarioBros-NES_b44_b40_winning_recipe_repro_5m_stop100ep100_seed53_20260620_111820-final_latest/"
    "final_model.zip"
)
DEFAULT_LEVEL1_2_TEACHER = Path(
    "runs/wandb_artifacts/"
    "tsilva_SuperMarioBros-NES_b46_b44_level1_2_same_hparams_5m_stop100ep100_seed60_20260620_124408-checkpoint_latest/"
    "ppo_supermariobros-nes-v0_5000000_steps.zip"
)
DEFAULT_OUTPUT_DIR = Path("runs/distilled_level1_1_level1_2")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Distill the successful SuperMarioBros-NES Level1-1 and Level1-2 PPO "
            "policies into one student policy by matching both teachers' action distributions."
        )
    )
    parser.add_argument("--level1-1-teacher", type=Path, default=DEFAULT_LEVEL1_1_TEACHER)
    parser.add_argument("--level1-2-teacher", type=Path, default=DEFAULT_LEVEL1_2_TEACHER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default="student_model")
    parser.add_argument("--updates", type=positive_int, default=10_000)
    parser.add_argument("--n-envs-per-teacher", type=positive_int, default=8)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--init-from-teacher",
        choices=("level1-1", "level1-2", "none"),
        default="level1-1",
        help="Initial student weights. Starting from Level1-1 preserves the known first policy while absorbing Level1-2.",
    )
    parser.add_argument(
        "--deterministic-teachers",
        action="store_true",
        help="Step collection envs with teacher argmax actions instead of sampled teacher actions.",
    )
    parser.add_argument("--log-interval", type=positive_int, default=100)
    parser.add_argument("--save-interval", type=nonnegative_int, default=1_000)
    parser.add_argument(
        "--run-description",
        default=(
            "Behavior distillation student trained by KL matching the successful "
            "B44 Level1-1 and B46 Level1-2 PPO teachers."
        ),
    )
    return parser.parse_args()


def require_model(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"teacher model not found: {path}")
    return path


def teacher_config(model_path: Path, state: str) -> EnvConfig:
    config = env_config_from_model_metadata(model_path, fallback=EnvConfig(state=state))
    if config is None:
        raise ValueError(f"{model_path} has no usable metadata sidecar")
    return resolve_env_config(replace(config, state=state, states=(), state_probs=()))


def student_config_from_teachers(configs: list[EnvConfig]) -> EnvConfig:
    first = configs[0]
    for config in configs[1:]:
        comparable_a = asdict(replace(first, state="", states=(), state_probs=()))
        comparable_b = asdict(replace(config, state="", states=(), state_probs=()))
        if comparable_a != comparable_b:
            raise ValueError(
                "teacher env configs differ beyond state; distillation needs matching "
                "preprocessing/action spaces"
            )
    states = tuple(config.state for config in configs)
    probs = tuple(1.0 / len(configs) for _ in configs)
    return resolve_mixed_state_config(
        replace(first, state=states[0], states=states, state_probs=probs),
        n_envs=len(configs),
    )


def base_distribution(sb3_distribution: Any) -> torch.distributions.Distribution:
    distribution = getattr(sb3_distribution, "distribution", None)
    if distribution is None:
        raise TypeError(f"unsupported SB3 distribution: {type(sb3_distribution).__name__}")
    return distribution


def distribution_entropy(distribution: torch.distributions.Distribution) -> torch.Tensor:
    return distribution.entropy().mean()


def categorical_mode(distribution: torch.distributions.Distribution) -> torch.Tensor:
    if not hasattr(distribution, "probs"):
        raise TypeError(f"expected categorical distribution, got {type(distribution).__name__}")
    return distribution.probs.argmax(dim=-1)


def save_student(
    student: PPO,
    output_path: Path,
    args: argparse.Namespace,
    config: EnvConfig,
    kind: str,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    student.save(output_path)
    if output_path.suffix != ".zip":
        output_path = output_path.with_suffix(".zip")
    metadata_args = argparse.Namespace(
        run_name=args.output_dir.name,
        run_description=args.run_description,
    )
    write_model_metadata(output_path, metadata_args, config, kind=kind)
    return output_path


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    teacher_specs = [
        ("level1-1", require_model(args.level1_1_teacher), "Level1-1"),
        ("level1-2", require_model(args.level1_2_teacher), "Level1-2"),
    ]
    configs = [teacher_config(path, state) for _, path, state in teacher_specs]
    config = student_config_from_teachers(configs)
    assert_rom_imported(config.game)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name
    metrics_path = output_dir / "distillation_metrics.jsonl"
    summary_path = output_dir / "distillation_summary.json"

    teachers = {
        name: PPO.load(path, device=args.device)
        for name, path, _ in teacher_specs
    }
    collectors = []
    try:
        for index, (name, path, _state) in enumerate(teacher_specs):
            env = make_training_vec_env(
                config=configs[index],
                n_envs=args.n_envs_per_teacher,
                seed=args.seed + index * 10_000,
            )
            obs = env.reset()
            collectors.append({"name": name, "path": path, "env": env, "obs": obs})

        student_env = make_training_vec_env(config=configs[0], n_envs=args.n_envs_per_teacher, seed=args.seed)
        student = PPO(
            "CnnPolicy",
            student_env,
            learning_rate=args.learning_rate,
            n_steps=512,
            batch_size=512,
            n_epochs=1,
            gamma=0.9,
            gae_lambda=1.0,
            ent_coef=0.0,
            vf_coef=1.0,
            clip_range=0.15,
            normalize_advantage=False,
            policy_kwargs={"optimizer_kwargs": {"eps": 1e-8}},
            device=args.device,
            verbose=0,
        )
        if args.init_from_teacher != "none":
            student.policy.load_state_dict(teachers[args.init_from_teacher].policy.state_dict())

        print(
            "distilling teachers into one student: "
            + ", ".join(f"{name}={path}" for name, path, _ in teacher_specs),
            flush=True,
        )
        print(f"writing outputs under {output_dir}", flush=True)

        for update in range(1, args.updates + 1):
            student.policy.optimizer.zero_grad(set_to_none=True)
            weighted_losses: list[torch.Tensor] = []
            metrics: dict[str, float] = {}

            for collector in collectors:
                name = collector["name"]
                teacher = teachers[name]
                obs = collector["obs"]
                obs_tensor, _ = student.policy.obs_to_tensor(obs)
                with torch.no_grad():
                    teacher_dist = teacher.policy.get_distribution(obs_tensor)
                    teacher_base = base_distribution(teacher_dist)
                    actions_tensor = teacher_dist.get_actions(
                        deterministic=args.deterministic_teachers
                    )
                    teacher_mode = categorical_mode(teacher_base)

                student_dist = student.policy.get_distribution(obs_tensor)
                student_base = base_distribution(student_dist)
                loss = torch.distributions.kl_divergence(teacher_base, student_base).mean()
                weighted_losses.append(loss / len(collectors))

                student_mode = categorical_mode(student_base)
                metrics[f"{name}/kl"] = float(loss.detach().cpu())
                metrics[f"{name}/teacher_entropy"] = float(
                    distribution_entropy(teacher_base).detach().cpu()
                )
                metrics[f"{name}/student_entropy"] = float(
                    distribution_entropy(student_base).detach().cpu()
                )
                metrics[f"{name}/argmax_agreement"] = float(
                    (student_mode == teacher_mode).float().mean().detach().cpu()
                )

                actions = actions_tensor.detach().cpu().numpy()
                next_obs, _rewards, _dones, _infos = collector["env"].step(actions)
                collector["obs"] = next_obs

            total_loss = torch.stack(weighted_losses).sum()
            total_loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(student.policy.parameters(), args.max_grad_norm)
            student.policy.optimizer.step()

            if update % args.log_interval == 0 or update == 1 or update == args.updates:
                row = {"update": update, "loss": float(total_loss.detach().cpu()), **metrics}
                with metrics_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
                print(json.dumps(row, sort_keys=True), flush=True)

            if args.save_interval and update % args.save_interval == 0:
                checkpoint_path = output_dir / f"{args.output_name}_{update}_updates"
                save_student(student, checkpoint_path, args, config, kind="distilled-checkpoint")

        final_path = save_student(student, output_path, args, config, kind="distilled")
        summary = {
            "student_model": str(final_path),
            "student_metadata": str(final_path.with_suffix(".metadata.json")),
            "metrics": str(metrics_path),
            "updates": args.updates,
            "n_envs_per_teacher": args.n_envs_per_teacher,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "init_from_teacher": args.init_from_teacher,
            "teachers": {
                name: {
                    "path": str(path),
                    "state": state,
                    "metadata": load_model_metadata(path),
                }
                for name, path, state in teacher_specs
            },
            "student_env_config": asdict(config),
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"saved distilled student: {final_path}", flush=True)
        print(
            "playback example: "
            f"UV_CACHE_DIR=.uv-cache uv run python -m stable_retro_ppo.play --model {final_path} "
            "--episodes 0 --stochastic",
            flush=True,
        )
    finally:
        for collector in collectors:
            collector["env"].close()
        if "student_env" in locals():
            student_env.close()


if __name__ == "__main__":
    main()
