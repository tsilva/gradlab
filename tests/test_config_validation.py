from __future__ import annotations

import io
import inspect
import json
import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from gradlab import experiment_contracts
from gradlab.config_validation import (
    main as validate_main,
    validate_experiment_tree,
)
from gradlab.experiment_contracts import validate_goal_contract_document
from gradlab.env_registry import resolve_env_provider, validate_provider_constructor_args
from gradlab.env_providers import _stable_retro_packaged_data_path
from gradlab.main import COMMANDS
from gradlab.recipe_documents import compose_train_document, load_goal_contract
from gradlab.recipe_schema import validate_materialized_train_recipe


class ConfigValidationTests(unittest.TestCase):
    BREAKOUT_GOAL = Path("experiments/goals/Breakout-Atari2600-v0/_goal.yaml")
    BREAKOUT_RECIPE = BREAKOUT_GOAL.parent / "recipes/ppo.yaml"
    BREAKOUT_NO_NOOP_GOAL = BREAKOUT_GOAL.parent / "no-noop/_goal.yaml"
    BREAKOUT_NO_NOOP_RECIPE = BREAKOUT_NO_NOOP_GOAL.parent / "recipes/ppo.yaml"
    MARIO_L11_GOAL = Path("experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml")
    MARIO_L12_GOAL = Path("experiments/goals/SuperMarioBros-Nes-v0/Level1-2/_goal.yaml")
    MARIO_L13_GOAL = Path("experiments/goals/SuperMarioBros-Nes-v0/Level1-3/_goal.yaml")
    MARIO_END_TO_END_GOAL = Path("experiments/goals/SuperMarioBros-Nes-v0/EndToEnd/_goal.yaml")
    MARIO_SINGLE_RECIPES = MARIO_L11_GOAL.parent / "recipes"
    VIZDOOM_BASIC_GOAL = Path("experiments/goals/VizdoomBasic-v1/_goal.yaml")
    VIZDOOM_BASIC_RECIPE = VIZDOOM_BASIC_GOAL.parent / "recipes/ppo.yaml"

    def test_provider_reward_transforms_are_rejected_in_favor_of_task_reward(self) -> None:
        for key in (
            "reward_clip",
            "reward_clipping",
            "normalize_reward",
            "norm_reward",
            "reward_normalization",
        ):
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(
                    ValueError,
                    "task.reward.reward_scale",
                ),
            ):
                validate_provider_constructor_args(
                    "gymnasium",
                    {key: True},
                    label="train.environment.env_config.env_args",
                )

    def test_common_reward_override_is_mirrored_into_vizdoom_evaluation(self) -> None:
        document = compose_train_document(
            self.VIZDOOM_BASIC_GOAL,
            self.VIZDOOM_BASIC_RECIPE,
            recipe_overrides=(
                "train.environment.task.reward.reward_scale=100",
                "train.environment.task.reward.reward_clip=[-0.05,0.5]",
            ),
        )
        expected = {
            "reward_mode": "native",
            "reward_scale": 100.0,
            "reward_clip": [-0.05, 0.5],
        }

        self.assertEqual(document["train_config"]["task"]["reward"], expected)
        self.assertEqual(
            document["train_config"]["checkpoint_eval_environment"]["task"]["reward"],
            expected,
        )
        self.assertNotIn("reward_clip", document["train_config"]["env_args"])

    def test_recipe_preset_allowlist_is_independent_of_current_directory(self) -> None:
        goal = self.MARIO_L11_GOAL.resolve()
        recipe = (self.MARIO_SINGLE_RECIPES / "ppo.yaml").resolve()
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            try:
                os.chdir(temporary)
                document = compose_train_document(goal, recipe)
            finally:
                os.chdir(previous)
        self.assertEqual(document["recipe_id"], "ppo")

    def test_explicit_goal_arg_contract_covers_provider_signatures(self) -> None:
        from ale_py.vector_env import AtariVectorEnv
        from breakout_turbo_env import BreakoutVecEnv
        from gradlab.bandit_env import BanditVectorEnv
        from stable_retro import RetroVecEnv
        from supermariobrosnes_turbo import SuperMarioBrosNesTurboVecEnv
        from vizdoom_turbo import VizdoomTurboVecEnv
        from gradlab.reward_transform import PROVIDER_REWARD_TRANSFORM_KEYS

        constructors = {
            "ale-py": AtariVectorEnv,
            "breakout-turbo-env": BreakoutVecEnv,
            "gradlab": BanditVectorEnv,
            "stable-retro-turbo": RetroVecEnv,
            "supermariobrosnes-turbo": SuperMarioBrosNesTurboVecEnv,
            "vizdoom-turbo": VizdoomTurboVecEnv,
        }
        for provider_id, constructor in constructors.items():
            with self.subTest(provider_id=provider_id):
                signature_args = {
                    name
                    for name, parameter in inspect.signature(constructor).parameters.items()
                    if parameter.kind
                    not in {
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    }
                }
                contract = resolve_env_provider(provider_id).constructor_contract
                self.assertIsNotNone(contract)
                covered_args = set(contract.canonical_args) | set(contract.explicit_env_args)
                public_signature_args = signature_args - PROVIDER_REWARD_TRANSFORM_KEYS
                if provider_id == "breakout-turbo-env":
                    # GradLab's compatibility adapter accepts the shared Stable Retro
                    # contract even when an older installed Turbo release ignores
                    # adapter-only fields through **unsupported.
                    self.assertLessEqual(public_signature_args, covered_args)
                else:
                    self.assertEqual(covered_args, public_signature_args)

    def test_breakout_goal_hotswaps_provider_without_changing_semantics(self) -> None:
        document = compose_train_document(
            self.BREAKOUT_GOAL,
            self.BREAKOUT_RECIPE,
        )

        train_config = document["train_config"]
        self.assertEqual(document["recipe_id"], "ppo")
        self.assertEqual(train_config["timesteps"], 500_000_000)
        self.assertEqual(train_config["training_backend"]["id"], "sb3.ppo")
        backend_config = train_config["training_backend"]["config"]
        self.assertEqual(backend_config["n_steps"], 64)
        self.assertEqual(backend_config["batch_size"], 256)
        self.assertEqual(backend_config["n_epochs"], 4)
        self.assertEqual(backend_config["learning_rate"], 2.5e-4)
        self.assertEqual(backend_config["learning_rate_final"], 5.0e-5)
        self.assertEqual(backend_config["learning_rate_schedule_timesteps"], 40_001_536)
        self.assertEqual(backend_config["ent_coef"], 0.01)
        self.assertEqual(backend_config["ent_coef_final"], 0.002)
        self.assertEqual(backend_config["ent_coef_schedule_timesteps"], 40_001_536)
        self.assertEqual(backend_config["gamma"], 0.99)
        self.assertEqual(backend_config["gae_lambda"], 0.95)
        self.assertEqual(backend_config["clip_range"], 0.1)
        self.assertEqual(backend_config["vf_coef"], 0.5)
        self.assertIsNone(backend_config.get("target_kl"))
        self.assertEqual(train_config["env_provider"], "breakout-turbo-env")
        self.assertEqual(train_config["game"], "Breakout-Atari2600-v0")
        self.assertEqual(train_config["state"], "Start")
        self.assertEqual(train_config["task"]["action"]["set"], "native")
        self.assertFalse(train_config["max_pool_frames"])

        self.assertEqual(train_config["sticky_action_prob"], 0.0)
        self.assertEqual(train_config["env_args"]["noop_reset_max"], 30)
        self.assertEqual(train_config["obs_crop"], [17, 0, 0, 0])
        self.assertEqual(train_config["obs_crop_mode"], "mask")
        self.assertEqual(
            train_config["task"]["signals"],
            {"ball_y": "ball_y", "score": "score"},
        )
        self.assertEqual(
            train_config["task"]["events"],
            {
                "serve_stall": {
                    "signal": "ball_y",
                    "operation": "equals_for",
                    "value": 0,
                    "steps": 256,
                }
            },
        )
        self.assertEqual(
            train_config["task"]["termination"],
            {"failure": ["serve_stall"], "max_episode_steps": 54000},
        )
        self.assertEqual(train_config["checkpoint_eval_backend"], "none")
        self.assertFalse(train_config["stop_on_acceptance"])
        self.assertIsNone(train_config.get("early_stop"))
        self.assertNotIn("eval", document["goal"])
        self.assertNotIn("release", document["goal"])
        self.assertEqual(
            train_config["selection_rank"],
            [
                "max(train/episode/return/shaped/from/target/mean)",
                "min(train/global_step)",
            ],
        )

        stable_retro = compose_train_document(
            self.BREAKOUT_GOAL,
            self.BREAKOUT_RECIPE,
            env_provider="stable-retro-turbo",
        )
        stable_train = stable_retro["train_config"]
        self.assertEqual(stable_train["env_provider"], "stable-retro-turbo")
        self.assertNotIn("checkpoint_eval_environment", stable_train)
        for key in ("game", "state", "env_args", "task", "selection_rank"):
            self.assertEqual(train_config[key], stable_train[key])
        self.assertEqual(document["goal"]["objective"], stable_retro["goal"]["objective"])
        self.assertNotIn("eval", stable_retro["goal"])

    def test_vizdoom_basic_ppo_recipe_has_evaluated_kill_contract(self) -> None:
        document = compose_train_document(
            self.VIZDOOM_BASIC_GOAL,
            self.VIZDOOM_BASIC_RECIPE,
        )

        train_config = document["train_config"]
        self.assertEqual(document["goal"]["goal_id"], "VizdoomBasic-v1")
        self.assertEqual(document["recipe_id"], "ppo")
        self.assertEqual(train_config["timesteps"], 2_000_000)
        self.assertEqual(train_config["training_backend"]["id"], "sb3.ppo")
        self.assertEqual(train_config["env_provider"], "vizdoom-turbo")
        self.assertEqual(train_config["game"], "VizdoomBasic-v1")
        self.assertEqual(train_config["state"], "default")
        self.assertEqual(train_config["n_envs"], 32)
        self.assertEqual(train_config["env_args"]["num_threads"], 32)
        self.assertEqual(train_config["env_args"]["use_restricted_actions"], "discrete")
        self.assertEqual(train_config["env_args"]["game_variables"], ["KILLCOUNT"])
        self.assertEqual(
            train_config["env_args"]["info_filter"],
            {"mode": "all", "keys": ["killcount"]},
        )
        self.assertEqual(
            train_config["task"]["events"]["monster_killed"],
            {"signal": "kills", "operation": "increase"},
        )
        self.assertEqual(
            train_config["task"]["termination"],
            {"success": ["monster_killed"], "max_episode_steps": 72},
        )
        expected_reward = {
            "reward_mode": "native",
            "reward_scale": 100.0,
            "reward_clip": False,
        }
        self.assertEqual(train_config["task"]["reward"], expected_reward)
        self.assertEqual(
            train_config["checkpoint_eval_environment"]["task"]["reward"],
            expected_reward,
        )
        self.assertEqual(document["goal"]["evaluation_mode"], "evaluated")
        eval_config = document["goal"]["eval"]["environment"]["env_config"]
        self.assertEqual(eval_config["n_envs"], 16)
        self.assertEqual(eval_config["env_args"]["num_threads"], 16)
        self.assertEqual(
            eval_config["env_args"]["info_filter"],
            {"mode": "all", "keys": ["killcount"]},
        )
        self.assertEqual(
            document["goal"]["eval"]["acceptance"],
            [
                {
                    "metric": "eval/full/outcome/success/rate/min",
                    "operator": ">=",
                    "threshold": 0.95,
                }
            ],
        )
        self.assertEqual(document["goal"]["release"], {"huggingface": {}})

    def test_breakout_no_noop_ablation_changes_only_goal_identity_and_action_table(self) -> None:
        baseline = compose_train_document(
            self.BREAKOUT_GOAL,
            self.BREAKOUT_RECIPE,
        )
        ablation = compose_train_document(
            self.BREAKOUT_NO_NOOP_GOAL,
            self.BREAKOUT_NO_NOOP_RECIPE,
        )
        stable_ablation = compose_train_document(
            self.BREAKOUT_NO_NOOP_GOAL,
            self.BREAKOUT_NO_NOOP_RECIPE,
            env_provider="stable-retro-turbo",
        )

        expected_actions = [["BUTTON"], ["RIGHT"], ["LEFT"]]
        self.assertEqual(ablation["goal"]["goal_id"], "no-noop")
        self.assertEqual(
            ablation["goal"]["title"],
            "Atari 2600 Breakout no-NOOP action ablation",
        )
        self.assertIn("three-action no-NOOP", ablation["description"])
        self.assertEqual(
            ablation["train_config"]["env_args"]["use_restricted_actions"],
            expected_actions,
        )
        self.assertEqual(
            stable_ablation["train_config"]["env_args"]["use_restricted_actions"],
            expected_actions,
        )
        self.assertEqual(
            stable_ablation["train_config"]["env_provider"],
            "stable-retro-turbo",
        )
        self.assertNotEqual(baseline["environment_hash"], ablation["environment_hash"])

        baseline_config = deepcopy(baseline["train_config"])
        ablation_config = deepcopy(ablation["train_config"])
        for config in (baseline_config, ablation_config):
            config.pop("goal_contract_sha256")
            config.pop("effective_goal_contract_sha256")
            config["env_args"].pop("use_restricted_actions")
        self.assertEqual(baseline_config, ablation_config)

    def test_goal_environment_rejects_implicit_provider_defaults(self) -> None:
        environment = {
            "env_provider": "supermariobrosnes-turbo",
            "env_config": {
                "game": "SuperMarioBros-Nes-v0",
                "state": "Level1-1",
                "n_envs": 1,
                "frame_skip": 4,
                "max_pool_frames": False,
                "sticky_action_prob": 0.0,
                "obs_resize": [84, 84],
                "obs_crop": [32, 0, 0, 0],
                "obs_crop_mode": "mask",
                "obs_crop_fill": 0,
                "obs_resize_algorithm": "area",
                "env_args": {},
            },
        }

        with self.assertRaisesRegex(ValueError, "missing explicit.*constructor argument"):
            experiment_contracts._validate_explicit_goal_environment_args(
                environment,
                environment["env_config"],
                label="goal.train.environment",
            )

    def test_materialized_recipe_rejects_missing_provider_constructor_arg(self) -> None:
        document = compose_train_document(
            self.MARIO_L11_GOAL,
            self.MARIO_SINGLE_RECIPES / "ppo.yaml",
        )
        document["train_config"]["env_args"].pop("frame_stack")

        with self.assertRaisesRegex(ValueError, "missing explicit.*frame_stack"):
            validate_materialized_train_recipe(document, label="recipe")

    def test_level1_1_a2c_recipe_is_native_and_preserves_goal_contract(self) -> None:
        document = compose_train_document(
            self.MARIO_L11_GOAL,
            self.MARIO_SINGLE_RECIPES / "a2c.yaml",
        )

        train_config = document["train_config"]
        backend = train_config["training_backend"]
        self.assertEqual(backend["id"], "sb3.a2c")
        self.assertEqual(backend["config"]["n_steps"], 64)
        self.assertTrue(backend["config"]["use_rms_prop"])
        self.assertFalse(backend["config"]["normalize_advantage"])
        self.assertTrue(
            {
                "batch_size",
                "n_epochs",
                "clip_range",
                "clip_range_vf",
                "target_kl",
                "adam_eps",
            }.isdisjoint(backend["config"])
        )
        self.assertEqual(
            train_config["task"]["termination"]["failure"],
            ["life_loss", "stalled"],
        )
        self.assertEqual(train_config["checkpoint_eval_backend"], "modal")

    def test_level1_1_jerk_recipe_is_playable_native_policy_search(self) -> None:
        document = compose_train_document(
            self.MARIO_L11_GOAL,
            self.MARIO_SINGLE_RECIPES / "jerk.yaml",
        )

        train_config = document["train_config"]
        backend = train_config["training_backend"]
        self.assertEqual(backend["id"], "gradlab.jerk")
        self.assertEqual(backend["config"]["archive_replay_probability_initial"], 0.25)
        self.assertEqual(backend["config"]["archive_replay_probability_max"], 0.9)
        self.assertEqual(backend["config"]["protected_prefix_steps"], 128)
        self.assertEqual(backend["config"]["max_prefix_shorten_steps"], 128)
        self.assertEqual(backend["config"]["acceptance_mode"], "first_training_success")
        self.assertEqual(train_config["timesteps"], 10000000)
        self.assertEqual(train_config["checkpoint_eval_backend"], "none")
        self.assertEqual(
            train_config["early_stop"]["conditions"]["return_plateau"]["action"],
            "stop",
        )
        self.assertEqual(
            train_config["early_stop"]["conditions"]["return_plateau"]["outcome"],
            "failure",
        )
        self.assertNotIn("checkpoint_eval_stages", train_config)

    def test_level1_3_training_clear_recipe_disables_eval_and_stops_at_100_of_100(
        self,
    ) -> None:
        document = compose_train_document(
            self.MARIO_L13_GOAL,
            self.MARIO_L13_GOAL.parent / "recipes/ppo-train-clear-100.yaml",
        )

        train_config = document["train_config"]
        self.assertEqual(document["recipe_id"], "ppo-train-clear-100")
        self.assertEqual(train_config["checkpoint_eval_backend"], "none")
        self.assertFalse(train_config["stop_on_acceptance"])
        self.assertEqual(
            set(train_config["early_stop"]["conditions"]),
            {"clear_100", "return_plateau"},
        )
        self.assertEqual(
            train_config["early_stop"]["conditions"]["clear_100"],
            {
                "metric": "train/outcome/success/window_100/rate/min",
                "trigger": "threshold",
                "outcome": "success",
                "action": "stop",
                "patience_steps": 0,
                "operator": ">=",
                "threshold": 1.0,
            },
        )

    def test_level1_1_on_policy_recipes_share_common_config(self) -> None:
        ppo = compose_train_document(self.MARIO_L11_GOAL, self.MARIO_SINGLE_RECIPES / "ppo.yaml")
        a2c = compose_train_document(self.MARIO_L11_GOAL, self.MARIO_SINGLE_RECIPES / "a2c.yaml")

        self.assertEqual(ppo["recipe_id"], "ppo")
        self.assertEqual(ppo["train_config"]["training_backend"]["id"], "sb3.ppo")
        for field in (
            "learning_rate_final",
            "learning_rate_schedule_timesteps",
            "ent_coef",
            "ent_coef_final",
            "ent_coef_schedule_timesteps",
            "gamma",
            "gae_lambda",
            "vf_coef",
            "normalize_advantage",
        ):
            with self.subTest(field=field):
                self.assertEqual(
                    ppo["train_config"]["training_backend"]["config"][field],
                    a2c["train_config"]["training_backend"]["config"][field],
                )
        self.assertEqual(ppo["train_config"]["timesteps"], 50000000)
        self.assertEqual(a2c["train_config"]["timesteps"], 50000000)
        self.assertTrue(ppo["train_config"]["wandb"])
        self.assertTrue(a2c["train_config"]["wandb"])

    def test_every_mario_recipe_composes_the_shared_plateau_condition(self) -> None:
        mario_root = Path("experiments/goals/SuperMarioBros-Nes-v0")
        recipes = sorted(mario_root.glob("*/recipes/*.yaml"))

        self.assertEqual(len(recipes), 25)
        for recipe_path in recipes:
            with self.subTest(recipe=recipe_path):
                goal_path = recipe_path.parent.parent / "_goal.yaml"
                document = compose_train_document(goal_path, recipe_path)
                conditions = document["train_config"]["early_stop"]["conditions"]
                self.assertIn("return_plateau", conditions)
                self.assertEqual(conditions["return_plateau"]["outcome"], "failure")
                self.assertEqual(conditions["return_plateau"]["action"], "stop")
                if recipe_path.name == "ppo-train-clear-100.yaml":
                    self.assertIn("clear_100", conditions)
                if recipe_path.name == "dstack-smoke.yaml":
                    self.assertLess(
                        document["train_config"]["timesteps"],
                        conditions["return_plateau"]["start_after_steps"],
                    )

    def test_removed_provider_lifecycle_args_are_rejected(self) -> None:
        for provider_id in ("stable-retro-turbo", "supermariobrosnes-turbo"):
            contract = resolve_env_provider(provider_id).constructor_contract
            self.assertIsNotNone(contract)
            self.assertNotIn("done_on", contract.explicit_env_args)
            self.assertNotIn("autoreset_mode", contract.explicit_env_args)

    def test_goal_validator_rejects_deterministic_policy_eval(self) -> None:
        with self.assertRaisesRegex(ValueError, "eval.policy.stochastic must be true"):
            experiment_contracts._validate_goal_eval(
                {
                    "evaluation_mode": "evaluated",
                    "eval": {"episodes": 1, "policy": {"stochastic": False}},
                },
                label="goal",
            )

    def test_checked_in_experiment_tree_validates(self) -> None:
        report = validate_experiment_tree(Path("."))

        self.assertEqual(report.issues, ())
        self.assertEqual(report.counts["json_files"], 0)
        self.assertGreaterEqual(report.counts["yaml_files"], 15)
        self.assertGreaterEqual(report.counts["goals"], 1)
        self.assertEqual(report.counts["train_recipes"], 41)
        self.assertGreaterEqual(report.counts["env_configs"], 0)
        self.assertEqual(report.counts["benchmark_profiles"], 4)

    def test_recipe_cannot_be_launched_for_a_different_goal(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not belong to goal"):
            compose_train_document(self.MARIO_L11_GOAL, self.BREAKOUT_RECIPE)

    def test_breakout_recipe_loads_with_stable_retro_start_state(self) -> None:
        document = compose_train_document(
            self.BREAKOUT_GOAL,
            self.BREAKOUT_RECIPE,
            env_provider="stable-retro-turbo",
        )

        train_config = document["train_config"]
        self.assertEqual(train_config["env_provider"], "stable-retro-turbo")
        self.assertEqual(train_config["game"], "Breakout-Atari2600-v0")
        self.assertEqual(train_config["checkpoint_eval_backend"], "none")
        self.assertFalse(train_config["stop_on_acceptance"])
        self.assertEqual(
            train_config["selection_rank"],
            [
                "max(train/episode/return/shaped/from/target/mean)",
                "min(train/global_step)",
            ],
        )
        self.assertEqual(
            train_config["env_args"],
            {
                "scenario": "scenario",
                "info": "data",
                "use_restricted_actions": "simple",
                "record": False,
                "players": 1,
                "inttype": "stable",
                "obs_type": "image",
                "render_mode": "rgb_array",
                "info_filter": "all",
                "num_threads": 6,
                "rom_path": None,
                "obs_copy": "safe_view",
                "obs_grayscale": True,
                "obs_layout": "chw",
                "frame_stack": 4,
                "noop_reset_max": 30,
                "use_fire_reset": False,
            },
        )
        self.assertEqual(train_config["state"], "Start")
        self.assertNotIn("states", train_config)
        self.assertEqual(train_config["n_envs"], 128)
        self.assertNotIn("checkpoint_eval_n_envs", train_config)
        self.assertNotIn("checkpoint_eval_environment", train_config)
        self.assertEqual(
            train_config["task"]["reward"],
            {
                "reward_mode": "native",
                "reward_scale": 1.0,
                "reward_clip": False,
            },
        )
        self.assertNotIn("env_threads", train_config)
        self.assertEqual(train_config["frame_skip"], 4)
        self.assertFalse(train_config["max_pool_frames"])
        self.assertEqual(train_config["sticky_action_prob"], 0.0)
        self.assertEqual(train_config["obs_resize"], [84, 84])
        self.assertNotIn("max_episode_steps", train_config)
        self.assertEqual(train_config["task"]["termination"]["max_episode_steps"], 54000)
        self.assertEqual(
            train_config["task"]["action"],
            {"set": "native"},
        )
        self.assertEqual(
            train_config["task"]["signals"],
            {"ball_y": "ball_y", "score": "score"},
        )

        info_path = _stable_retro_packaged_data_path(
            train_config["game"],
            "data.json",
        )
        self.assertEqual(
            json.loads(info_path.read_text(encoding="utf-8"))["info"]["ball_y"],
            {"address": 229, "type": "|u1"},
        )
        self.assertEqual(
            train_config["task"]["events"],
            {
                "serve_stall": {
                    "signal": "ball_y",
                    "operation": "equals_for",
                    "value": 0,
                    "steps": 256,
                }
            },
        )
        self.assertEqual(
            train_config["task"]["termination"],
            {"failure": ["serve_stall"], "max_episode_steps": 54000},
        )
        self.assertNotIn("reward_clip", train_config["env_args"])
        self.assertEqual(train_config["obs_crop"], [17, 0, 0, 0])
        self.assertEqual(train_config["obs_crop_mode"], "mask")
        self.assertEqual(train_config["obs_crop_fill"], 0)
        self.assertEqual(document["environment"]["preprocessing"]["obs_crop"], [17, 0, 0, 0])
        self.assertNotIn("eval", document["goal"])
        self.assertEqual(train_config["obs_resize_algorithm"], "area")
        self.assertEqual(
            document["environment"]["env_id"],
            "stable-retro-turbo:Breakout-Atari2600-v0",
        )
        self.assertEqual(document["environment"]["preprocessing"]["frame_skip"], 4)

    def test_all_breakout_recipes_use_30_reset_noops(self) -> None:
        recipe_root = self.BREAKOUT_GOAL.parent
        recipes = sorted(recipe_root.glob("**/recipes/*.yaml"))
        self.assertTrue(recipes)
        for recipe in recipes:
            goal = recipe.parent.parent / "_goal.yaml"
            document = compose_train_document(goal, recipe)
            self.assertEqual(
                document["train_config"]["env_args"]["noop_reset_max"],
                30,
                str(recipe),
            )

    def test_breakout_archive_curriculum_is_fully_opt_in(self) -> None:
        recipe = self.BREAKOUT_GOAL.parent / "recipes" / "ppo-archive-curriculum.yaml"
        document = compose_train_document(self.BREAKOUT_GOAL, recipe)

        self.assertEqual(document["recipe_id"], "ppo-archive-curriculum")
        archive = document["train_config"]["state_archive"]
        self.assertEqual(archive["semantic_id"], "state-archive-v1")
        self.assertEqual(archive["persistence"], "durable")
        self.assertEqual(
            archive["recorder"]["cell"],
            {"signal": "score", "bucket_size": 50.0},
        )
        self.assertEqual(archive["curriculum"]["archive_share"], 0.2)
        self.assertEqual(archive["curriculum"]["priority_metric"], "value_error")
        self.assertTrue(archive["curriculum"]["restore_entries"])
        self.assertNotIn(
            "state_archive",
            compose_train_document(self.BREAKOUT_GOAL, self.BREAKOUT_RECIPE)["train_config"],
        )

    def test_go_explore_persists_only_the_best_action_program(self) -> None:
        recipe = self.MARIO_L11_GOAL.parent / "recipes/go-explore-20m.yaml"
        document = compose_train_document(self.MARIO_L11_GOAL, recipe)
        train_config = document["train_config"]

        self.assertEqual(train_config["state_archive"]["persistence"], "ephemeral")
        self.assertEqual(
            train_config["training_backend"]["config"]["compaction_interval_steps"],
            250_000,
        )

    def test_breakout_stable_updates_recipe_adds_late_update_guards(self) -> None:
        document = compose_train_document(
            self.BREAKOUT_GOAL,
            self.BREAKOUT_GOAL.parent / "recipes/ppo-stable-updates.yaml",
            env_provider="stable-retro-turbo",
        )

        train_config = document["train_config"]
        self.assertEqual(document["recipe_id"], "ppo-stable-updates")
        backend_config = train_config["training_backend"]["config"]
        self.assertEqual(backend_config["learning_rate"], 2.5e-4)
        self.assertEqual(backend_config["learning_rate_final"], 2.5e-5)
        self.assertEqual(backend_config["learning_rate_schedule_timesteps"], 100_000_000)
        self.assertEqual(backend_config["target_kl"], 0.03)
        self.assertIs(train_config["task"]["reward"]["reward_clip"], False)
        self.assertNotIn("checkpoint_eval_environment", train_config)

    def test_mspacman_recipe_loads_with_breakout_base_config_and_hud_mask(self) -> None:
        breakout = compose_train_document(
            self.BREAKOUT_GOAL,
            self.BREAKOUT_RECIPE,
        )
        document = compose_train_document(
            Path("experiments/goals/alepy__mspacman/_goal.yaml"),
            Path("experiments/goals/alepy__mspacman/recipes/ppo.yaml"),
        )

        train_config = document["train_config"]
        self.assertEqual(train_config["env_provider"], "stable-retro-turbo")
        self.assertEqual(train_config["game"], "MsPacman-Atari2600-v0")
        self.assertEqual(train_config["n_envs"], 16)
        self.assertEqual(train_config["env_args"]["num_threads"], 4)
        self.assertEqual(train_config["task"]["reward"]["reward_clip"], [-1.0, 1.0])
        self.assertIs(train_config["env_args"]["use_fire_reset"], False)
        self.assertIs(
            train_config["checkpoint_eval_environment"]["env_args"]["use_fire_reset"],
            False,
        )
        self.assertEqual(train_config["state"], "Start")
        self.assertNotIn("states", train_config)
        self.assertEqual(train_config["obs_crop"], [0, 0, 37, 0])
        self.assertEqual(train_config["obs_crop_mode"], "mask")
        self.assertEqual(train_config["obs_crop_fill"], 0)
        self.assertEqual(train_config["task"]["action"], {"set": "native"})
        self.assertEqual(train_config["task"]["action"], breakout["train_config"]["task"]["action"])
        self.assertEqual(
            {key: train_config["task"]["reward"][key] for key in ("reward_mode", "reward_scale")},
            {
                key: breakout["train_config"]["task"]["reward"][key]
                for key in ("reward_mode", "reward_scale")
            },
        )
        self.assertIs(breakout["train_config"]["task"]["reward"]["reward_clip"], False)
        for key in ("env_threads",):
            self.assertNotIn(key, train_config)
        self.assertEqual(train_config["frame_skip"], 4)
        self.assertTrue(train_config["max_pool_frames"])
        self.assertEqual(train_config["sticky_action_prob"], 0.25)
        self.assertEqual(train_config["obs_resize"], [84, 84])
        self.assertEqual(train_config["task"]["termination"]["max_episode_steps"], 54000)
        self.assertEqual(train_config["obs_resize_algorithm"], "area")
        self.assertEqual(
            document["environment"]["env_id"],
            "stable-retro-turbo:MsPacman-Atari2600-v0",
        )
        self.assertEqual(document["environment"]["preprocessing"]["frame_skip"], 4)

    def test_goal_validator_rejects_legacy_eval_driven_early_stop(self) -> None:
        path = self.MARIO_L11_GOAL.resolve()
        document = load_goal_contract(path)
        document["train"]["early_stop"] = [
            {
                "metric": "train/outcome/success/window_100/rate/min",
                "operator": ">",
                "threshold": 0.99,
            }
        ]
        with self.assertRaisesRegex(ValueError, "must be an object"):
            validate_goal_contract_document(document, path, Path(".").resolve())

    def test_goal_validator_rejects_rank_forms_the_runtime_cannot_parse(self) -> None:
        with self.assertRaisesRegex(ValueError, "max\\(metric\\) or min\\(metric\\)"):
            experiment_contracts._validate_rank_order(
                [{"metric": "eval/full/episode/return/mean", "direction": "maximize"}],
                label="objective.rank",
            )

    def test_goal_validator_rejects_unknown_top_level_field(self) -> None:
        path = Path("experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml").resolve()
        document = load_goal_contract(path)
        document["hypotesis"] = "typo"

        with self.assertRaisesRegex(ValueError, "unknown field.*hypotesis"):
            validate_goal_contract_document(document, path, Path(".").resolve())

    def test_goal_validator_requires_explicit_evaluation_mode(self) -> None:
        path = self.MARIO_L11_GOAL.resolve()
        document = load_goal_contract(path)
        document.pop("evaluation_mode")

        with self.assertRaisesRegex(ValueError, "evaluation_mode must be one of"):
            validate_goal_contract_document(document, path, Path(".").resolve())

    def test_training_only_goal_rejects_eval_owned_contract_fields(self) -> None:
        path = self.BREAKOUT_GOAL.resolve()
        base = load_goal_contract(path)
        mario_eval = load_goal_contract(self.MARIO_L11_GOAL.resolve())["eval"]
        invalid_documents = []

        eval_rank = deepcopy(base)
        eval_rank["objective"]["rank"] = ["max(eval/full/episode/return/mean)"]
        invalid_documents.append((eval_rank, "may use only training metrics"))

        eval_config = deepcopy(base)
        eval_config["eval"] = deepcopy(mario_eval)
        invalid_documents.append((eval_config, "eval must be omitted"))

        release = deepcopy(base)
        release["release"] = {"huggingface": {}}
        invalid_documents.append((release, "release is unsupported"))

        for document, message in invalid_documents:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                validate_goal_contract_document(document, path, Path(".").resolve())

    def test_evaluated_goal_still_requires_eval_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "eval is required"):
            experiment_contracts._validate_goal_eval(
                {"evaluation_mode": "evaluated"},
                label="goal",
            )

    def test_goal_validator_requires_slug_to_match_goal_directory(self) -> None:
        path = self.MARIO_L11_GOAL.resolve()
        document = load_goal_contract(path)
        document["goal_id"] = "stale-short-name"

        with self.assertRaisesRegex(
            ValueError, "goal_id.*must match goal directory name: Level1-1"
        ):
            validate_goal_contract_document(document, path, Path(".").resolve())

    def test_goal_validator_rejects_objective_success(self) -> None:
        path = self.MARIO_L11_GOAL.resolve()
        document = load_goal_contract(path)
        document["objective"]["success"] = {
            "metric": "train/outcome/success/window_100/rate/min",
            "operator": ">",
            "threshold": 0.99,
        }

        with self.assertRaisesRegex(ValueError, "objective\\.success moved to train\\.early_stop"):
            validate_goal_contract_document(document, path, Path(".").resolve())

    def test_goal_validator_rejects_environment_hash(self) -> None:
        path = self.MARIO_L11_GOAL.resolve()
        document = load_goal_contract(path)
        document["environment_hash"] = "sha256:deadbeef"

        with self.assertRaisesRegex(ValueError, "environment_hash"):
            validate_goal_contract_document(document, path, Path(".").resolve())

    def test_load_goal_contract_returns_composed_document(self) -> None:
        document = load_goal_contract(
            Path("experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml")
        )

        self.assertEqual(document["goal_id"], "Level1-1")
        absent_fields = (
            (
                document,
                {
                    "extends",
                    "schema_version",
                    "status",
                    "seed_protocol",
                    "historical_context",
                    "updated_at",
                    "notes",
                    "runtime",
                    "search_protocol",
                    "batch_record_fields",
                    "capacity_policy_file",
                    "cap_policy",
                    "constraints",
                    "default_eval_profile",
                    "default_train_profile",
                    "environment_hash",
                    "execution",
                    "selection_policy",
                },
            ),
            (
                document["objective"],
                {
                    "game",
                    "algorithm",
                    "states",
                    "forbidden_stop_rules",
                    "max_train_timesteps",
                    "success",
                },
            ),
            (
                document["train"],
                {"checkpoint_eval_stages", "max_train_timesteps"},
            ),
        )
        for mapping, fields in absent_fields:
            for field in fields:
                with self.subTest(field=field):
                    self.assertNotIn(field, mapping)
        self.assertEqual(document["evaluation_mode"], "evaluated")
        self.assertEqual(
            document["train"]["early_stop"]["conditions"]["return_plateau"]["action"],
            "stop",
        )
        self.assertEqual(
            document["eval"]["acceptance"],
            [
                {
                    "metric": "eval/full/outcome/success/rate/min",
                    "operator": ">=",
                    "threshold": 1.0,
                }
            ],
        )
        self.assertEqual(
            document["objective"]["rank"],
            [
                "min(leader/checkpoint/step)",
                "max(eval/full/episode/return/mean)",
            ],
        )
        stalled_event = {"signal": "x", "operation": "unchanged_for", "steps": 300}
        for phase in ("train", "eval"):
            environment = document[phase]["environment"]
            config = environment["env_config"]
            task = environment["task"]
            with self.subTest(phase=phase):
                self.assertEqual(environment["env_provider"], "supermariobrosnes-turbo")
                self.assertEqual(config["game"], "SuperMarioBros-Nes-v0")
                self.assertEqual(config["n_envs"], 16)
                self.assertEqual(config["obs_crop"], [32, 0, 0, 0])
                self.assertEqual(config["obs_resize"], [84, 84])
                self.assertTrue({"env_provider", "env_threads", "reward_mode"}.isdisjoint(config))
                self.assertEqual(task["events"]["stalled"], stalled_event)
        train_config = document["train"]["environment"]["env_config"]
        self.assertEqual(train_config["state"], "Level1-1")
        self.assertFalse(train_config["max_pool_frames"])
        self.assertEqual(train_config["sticky_action_prob"], 0.0)
        self.assertEqual(
            document["train"]["environment"]["task"]["termination"]["failure"],
            ["life_loss", "stalled"],
        )
        self.assertTrue({"policy", "schema_version"}.isdisjoint(document["eval"]))
        self.assertEqual(
            document["eval"]["environment"]["task"]["termination"]["failure"],
            ["life_loss", "stalled"],
        )
        self.assertEqual(document["eval"]["episodes"], 100)
        self.assertTrue(
            {"max_episodes", "seed", "max_steps"}.isdisjoint(
                document["eval"]["environment"]["env_config"]
            )
        )
        self.assertEqual(
            document["eval"]["environment"]["task"]["termination"]["success"],
            ["level_change"],
        )
        self.assertEqual(document["train"]["environment"]["task"]["id"], "mario")

    def test_load_goal_contract_can_skip_semantic_validation_for_catalogues(self) -> None:
        with patch(
            "gradlab.recipe_documents.validate_goal_contract_document",
            side_effect=AssertionError("semantic validation should be skipped"),
        ):
            document = load_goal_contract(self.MARIO_L11_GOAL, validate=False)

        self.assertEqual(document["goal_id"], "Level1-1")

    def test_level_1_2_resets_no_progress_attractor_and_caps_ppo_at_30m(self) -> None:
        recipe = self.MARIO_L12_GOAL.parent / "recipes/ppo.yaml"
        document = compose_train_document(self.MARIO_L12_GOAL, recipe)
        stalled_event = {"signal": "x", "operation": "unchanged_for", "steps": 300}

        self.assertEqual(document["train_config"]["timesteps"], 30_000_000)
        self.assertEqual(
            document["train_config"]["training_backend"]["config"]["target_kl"],
            0.03,
        )
        self.assertEqual(document["train_config"]["state"], "Level1-2")
        self.assertEqual(document["train_config"]["task"]["events"]["stalled"], stalled_event)
        self.assertEqual(
            document["train_config"]["task"]["termination"]["failure"],
            ["life_loss", "stalled"],
        )
        eval_task = document["goal"]["eval"]["environment"]["task"]
        self.assertEqual(eval_task["events"]["stalled"], stalled_event)
        self.assertEqual(eval_task["termination"]["failure"], ["life_loss", "stalled"])
        self.assertEqual(eval_task["termination"]["success"], ["level_change"])

    def test_end_to_end_mario_goal_only_terminates_successfully_after_level_8_4(self) -> None:
        document = load_goal_contract(self.MARIO_END_TO_END_GOAL)

        self.assertEqual(
            document["train"]["environment"]["env_config"]["state"],
            "Level1-1",
        )
        for phase in ("train", "eval"):
            task = document[phase]["environment"]["task"]
            self.assertEqual(
                task["events"]["game_complete"],
                {
                    "signal": "game_mode",
                    "operation": "equals",
                    "value": 2,
                    "when": {"signal": "level", "value": [7, 3]},
                },
            )
            self.assertEqual(task["termination"]["failure"], [])
            self.assertEqual(task["termination"]["success"], ["game_complete"])
            self.assertEqual(task["termination"]["max_episode_steps"], 144000)

    def test_all_mario_level_goals_share_attempt_termination_contract(self) -> None:
        root = self.MARIO_L11_GOAL.parent.parent
        expected_stalled = {"signal": "x", "operation": "unchanged_for", "steps": 300}
        goal_paths = sorted(
            path for path in root.glob("*/_goal.yaml") if path.parent.name != "EndToEnd"
        )

        self.assertEqual(len(goal_paths), 17)
        for goal_path in goal_paths:
            document = load_goal_contract(goal_path)
            for phase in ("train", "eval"):
                task = document[phase]["environment"]["task"]
                with self.subTest(goal=goal_path.parent.name, phase=phase):
                    self.assertEqual(task["events"]["stalled"], expected_stalled)
                    self.assertEqual(
                        task["termination"]["failure"],
                        ["life_loss", "stalled"],
                    )
                    self.assertEqual(
                        task["termination"]["success"],
                        ["level_change"],
                    )

    def test_validate_is_registered_on_unified_cli(self) -> None:
        self.assertIn("validate", COMMANDS)
        self.assertIn("env", COMMANDS)

    def test_retired_commands_are_not_registered_on_unified_cli(self) -> None:
        self.assertTrue({"monitor", "promote", "release"}.isdisjoint(COMMANDS))

    def test_goal_validator_accepts_huggingface_release_target(self) -> None:
        document = load_goal_contract(
            Path("experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml")
        )

        self.assertEqual(document["release"]["huggingface"], {})

    def test_goal_validator_rejects_manual_huggingface_release_identity(self) -> None:
        path = self.MARIO_L11_GOAL.resolve()
        document = load_goal_contract(path)
        document["release"]["huggingface"]["repo"] = "manual-name"

        with self.assertRaisesRegex(ValueError, "marker-only"):
            validate_goal_contract_document(document, path, Path(".").resolve())

    def test_validate_cli_success(self) -> None:
        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            exit_code = validate_main([])

        self.assertEqual(exit_code, 0)
        self.assertIn("YAML config validation passed", stdout.getvalue())

    def test_validate_cli_load_goal_emits_composed_json(self) -> None:
        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            exit_code = validate_main(
                [
                    "--load-goal",
                    "experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml",
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0)
        document = json.loads(stdout.getvalue())
        self.assertNotIn("extends", document)
        self.assertNotIn("schema_version", document)
        self.assertNotIn("status", document)
        self.assertEqual(document["goal_id"], "Level1-1")
        self.assertNotIn("goal_dir", document)
        self.assertNotIn("seed_protocol", document)
        self.assertNotIn("historical_context", document)
        self.assertNotIn("updated_at", document)
        self.assertEqual(
            document["train"]["environment"]["env_config"]["game"], "SuperMarioBros-Nes-v0"
        )
        self.assertNotIn("execution", document)


if __name__ == "__main__":
    unittest.main()
