#!/usr/bin/env python3
"""Benchmark steady-state GraDOOM rollout plus PPO update throughput.

This operator-run benchmark excludes environment, policy, and optimizer graph
compilation through complete untimed warmup rollouts. Beast-3 runs require an
explicitly confirmed quiet window before launching this tool.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from gradlab.env import resolve_env_config
from gradlab.env_config import env_config_from_mapping
from gradlab.gradoom_device_runtime import make_gradoom_device_vec_env
from gradlab.ppo import GradLabPPO
from gradlab.recipe_documents import compose_train_document
from gradlab.training.ppo_engine import (
    TensorRolloutBuffer,
    _bootstrap_device_time_limits,
    _CompiledPolicyCalls,
    _configure_optimizer_for_device,
    _observation_tensor,
    _ppo_update,
    _Precision,
)
from gradlab.training.sb3_on_policy import policy_kwargs_from_config, policy_type_for_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iwad", type=Path, required=True)
    parser.add_argument("--goal-root", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--warmup-rollouts", type=int, default=1)
    parser.add_argument("--measured-rollouts", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if min(args.num_envs, args.warmup_rollouts, args.measured_rollouts) <= 0:
        raise ValueError("environment and rollout counts must be positive")

    document = compose_train_document(
        args.goal_root / "_goal.yaml",
        args.goal_root / "recipes/gradoom-ppo.yaml",
    )
    common = document["train_config"]
    common["env_args"] = {**common["env_args"], "rom_path": str(args.iwad)}
    common["n_envs"] = args.num_envs
    config = resolve_env_config(env_config_from_mapping(common))
    backend = common["training_backend"]["config"]
    n_steps = int(args.n_steps or backend["n_steps"])
    batch_size = int(args.batch_size or backend["batch_size"])
    n_epochs = int(args.n_epochs or backend["n_epochs"])
    if min(n_steps, batch_size, n_epochs) <= 0:
        raise ValueError("PPO shape arguments must be positive")
    rollout_transitions = args.num_envs * n_steps
    if batch_size > rollout_transitions:
        raise ValueError("batch size cannot exceed the rollout size")

    seed = int(common.get("seed", document.get("seeds", [0])[0]))
    device = torch.device("cuda")
    env = make_gradoom_device_vec_env(config, n_envs=args.num_envs, seed=seed)
    try:
        runtime = env.runtime
        observations = runtime.reset(seed=seed)
        model = GradLabPPO(
            policy_type_for_config(env.observation_space, common),
            env,
            learning_rate=float(backend["learning_rate"]),
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=float(backend["gamma"]),
            gae_lambda=float(backend["gae_lambda"]),
            ent_coef=float(backend["ent_coef"]),
            vf_coef=float(backend["vf_coef"]),
            clip_range=float(backend["clip_range"]),
            normalize_advantage=True,
            policy_kwargs=policy_kwargs_from_config(
                backend,
                common_config=common,
                optimizer_eps=float(backend["adam_eps"]),
            ),
            device="cuda",
            verbose=0,
        )
        model.rollout_buffer = None
        model.batch_size = batch_size
        model.n_epochs = n_epochs
        _configure_optimizer_for_device(model.policy.optimizer, device, fused=True)
        calls = _CompiledPolicyCalls(model.policy, device, compile_policy=True)
        precision = _Precision(str(backend["precision"]), device)
        buffer = TensorRolloutBuffer.allocate(
            observations,
            action_space=env.action_space,
            n_steps=n_steps,
            n_envs=args.num_envs,
            device=device,
            store_final_observations=True,
        )
        episode_starts = torch.ones(args.num_envs, dtype=torch.bool, device=device)
        dones = torch.zeros_like(episode_starts)

        def rollout_and_update() -> tuple[float, float]:
            nonlocal dones, episode_starts, observations
            buffer.position = 0
            model.policy.set_training_mode(False)
            torch.cuda.synchronize(device)
            rollout_started = time.perf_counter()
            for _step in range(n_steps):
                obs_tensor = _observation_tensor(observations, device)
                with torch.no_grad(), precision.autocast():
                    actions, values, log_probs = calls.forward(obs_tensor)
                transition = runtime.step(actions)
                buffer.add(
                    obs_tensor,
                    actions,
                    transition.rewards,
                    episode_starts,
                    values,
                    log_probs,
                    final_observations=transition.final_observations,
                    truncated=transition.truncated,
                )
                observations = transition.observations
                dones = (transition.terminated | transition.truncated).clone()
                episode_starts = dones
            with torch.no_grad(), precision.autocast():
                last_values = calls.predict_values(_observation_tensor(observations, device))
            _bootstrap_device_time_limits(
                buffer,
                calls=calls,
                precision=precision,
                gamma=float(model.gamma),
            )
            buffer.finish(
                last_values=last_values,
                dones=dones,
                gamma=float(model.gamma),
                gae_lambda=float(model.gae_lambda),
            )
            runtime.drain_records()
            torch.cuda.synchronize(device)
            rollout_seconds = time.perf_counter() - rollout_started

            update_started = time.perf_counter()
            _ppo_update(
                model,
                buffer,
                calls=calls,
                precision=precision,
                progress_remaining=1.0,
                normalization_mode="global",
                advantage_context=None,
                ent_coef=float(backend["ent_coef"]),
                torch_permutation=True,
            )
            torch.cuda.synchronize(device)
            return rollout_seconds, time.perf_counter() - update_started

        for _repeat in range(args.warmup_rollouts):
            rollout_and_update()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        rollout_samples: list[float] = []
        update_samples: list[float] = []
        loop_sps_samples: list[float] = []
        for _repeat in range(args.measured_rollouts):
            rollout_seconds, update_seconds = rollout_and_update()
            rollout_samples.append(rollout_seconds)
            update_samples.append(update_seconds)
            loop_sps_samples.append(rollout_transitions / (rollout_seconds + update_seconds))

        result = {
            "batch_size": batch_size,
            "cuda": torch.version.cuda,
            "cuda_memory_allocated_bytes": torch.cuda.memory_allocated(device),
            "cuda_memory_reserved_bytes": torch.cuda.memory_reserved(device),
            "cuda_peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "cuda_peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "device": torch.cuda.get_device_name(device),
            "environment_backend": runtime.provider.engine_backend,
            "iwad_sha256": runtime.provider.iwad_sha256,
            "loop_transitions_per_second_median": statistics.median(loop_sps_samples),
            "loop_transitions_per_second_samples": loop_sps_samples,
            "measured_rollouts": args.measured_rollouts,
            "n_envs": args.num_envs,
            "n_epochs": n_epochs,
            "n_steps": n_steps,
            "policy_compiled": True,
            "precision": str(backend["precision"]),
            "rollout_seconds_samples": rollout_samples,
            "rollout_transitions": rollout_transitions,
            "scenario_sha256": runtime.provider.scenario_sha256,
            "torch": torch.__version__,
            "update_seconds_samples": update_samples,
            "warmup_rollouts_excluded": args.warmup_rollouts,
        }
        print(json.dumps(result, sort_keys=True))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
