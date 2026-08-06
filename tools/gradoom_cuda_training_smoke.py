#!/usr/bin/env python3
"""Exercise one GraDOOM rollout and PPO update without collecting timings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gradlab.action_contract import assert_action_contract_compatible
from gradlab.env import make_training_vec_env, resolve_env_config
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iwad", type=Path, required=True)
    parser.add_argument("--goal-root", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=128)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    document = compose_train_document(
        args.goal_root / "_goal.yaml",
        args.goal_root / "recipes/gradoom-ppo.yaml",
    )
    common = document["train_config"]
    common["env_args"] = {**common["env_args"], "rom_path": str(args.iwad)}
    common["n_envs"] = args.num_envs
    config = resolve_env_config(env_config_from_mapping(common))
    backend = common["training_backend"]["config"]
    seed = int(common.get("seed", document.get("seeds", [0])[0]))
    device = torch.device("cuda")
    env = make_gradoom_device_vec_env(
        config,
        n_envs=args.num_envs,
        seed=seed,
    )
    try:
        runtime = env.runtime
        observations = runtime.reset(seed=seed)
        assert all(value.device.type == device.type for value in observations.values())
        model = GradLabPPO(
            policy_type_for_config(env.observation_space, common),
            env,
            learning_rate=float(backend["learning_rate"]),
            n_steps=2,
            batch_size=args.num_envs * 2,
            n_epochs=1,
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
        model.batch_size = args.num_envs * 2
        model.n_epochs = 1
        _configure_optimizer_for_device(model.policy.optimizer, device, fused=True)
        calls = _CompiledPolicyCalls(model.policy, device, compile_policy=False)
        precision = _Precision("amp-fp16", device)
        buffer = TensorRolloutBuffer.allocate(
            observations,
            action_space=env.action_space,
            n_steps=2,
            n_envs=args.num_envs,
            device=device,
            store_final_observations=True,
        )
        episode_starts = torch.ones(args.num_envs, dtype=torch.bool, device=device)
        dones = torch.zeros_like(episode_starts)
        for _step in range(2):
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
            dones = transition.terminated | transition.truncated
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
        update = _ppo_update(
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
        reference_mapping = dict(common["checkpoint_eval_environment"])
        reference_mapping["env_args"] = {
            **reference_mapping["env_args"],
            "rom_path": str(args.iwad),
        }
        reference_config = resolve_env_config(env_config_from_mapping(reference_mapping))
        reference = make_training_vec_env(reference_config, n_envs=1, seed=seed)
        try:
            if (
                runtime.action_contract["semantic_hash"]
                != reference.runtime.action_contract["semantic_hash"]
            ):
                print(
                    json.dumps(
                        {
                            "gradoom": runtime.action_contract["provider"]["semantics"],
                            "vizdoom": reference.runtime.action_contract["provider"]["semantics"],
                        },
                        sort_keys=True,
                    )
                )
            action_compatibility = assert_action_contract_compatible(
                runtime.action_contract,
                reference.runtime.action_contract,
            )
        finally:
            reference.close()
        torch.cuda.synchronize()
        print(
            json.dumps(
                {
                    "action_contract_hash": runtime.action_contract["contract_hash"],
                    "cuda": torch.version.cuda,
                    "device": torch.cuda.get_device_name(device),
                    "episode_records": len(runtime.drain_records()),
                    "num_envs": args.num_envs,
                    "observation_keys": sorted(observations),
                    "ppo_update_metrics": sorted(update),
                    "reference_action_contract": action_compatibility["status"],
                    "torch": torch.__version__,
                },
                sort_keys=True,
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
