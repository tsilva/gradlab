from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from gradlab.env import EnvConfig
from gradlab.train_config import (
    add_env_config_args,
    checkpoint_eval_requires_acceptance,
    load_materialized_train_config,
    train_config_field_for_key,
    validate_and_normalize_train_config,
    validate_train_config_fields,
    validate_train_config_value,
    wandb_publication_enabled,
)
from gradlab.train import build_parser as build_train_parser, parse_train_invocation
from gradlab.training_lifecycle import TrainingExecutionMode


POLICY_MODEL = {
    "schema_version": 2,
    "encoder": {"kind": "flatten"},
    "fusion": {"hidden_sizes": [8], "activation": "tanh"},
    "normalize_images": False,
    "orthogonal_init": True,
}


class TrainConfigFieldSchemaTests(unittest.TestCase):
    def test_internal_train_cli_requires_one_materialized_json_input(self) -> None:
        parser = build_train_parser()
        options = {option for action in parser._actions for option in action.option_strings}

        self.assertEqual(
            options,
            {"-h", "--help", "--train-config-json", "--execution-mode"},
        )
        with self.assertRaises(SystemExit):
            parser.parse_args(["--train-config-json", "train.json"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--training-backend", '{"id":"sb3.ppo"}'])

    def test_train_config_json_rejects_invalid_field_types_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.json"
            path.write_text(
                json.dumps(
                    {
                        "timesteps": "many",
                        "training_backend": {"id": "sb3.ppo", "config": {}},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "timesteps must be an integer"):
                parse_train_invocation(
                    [
                        "--train-config-json",
                        str(path),
                        "--execution-mode",
                        "supervised",
                    ]
                )

    def test_materialized_config_loader_matches_internal_invocation_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.json"
            path.write_text(
                json.dumps(
                    {
                        "game": "SuperMarioBros-Nes-v0",
                        "policy_model": POLICY_MODEL,
                        "seed": 7,
                        "wandb_tags": "one",
                        "training_backend": {"id": "sb3.ppo", "config": {}},
                    }
                ),
                encoding="utf-8",
            )

            invocation_config, execution_mode = parse_train_invocation(
                [
                    "--train-config-json",
                    str(path),
                    "--execution-mode",
                    "supervised",
                ]
            )
            worker_config = load_materialized_train_config(path)

        for key in ("game", "seed", "wandb_tags", "frame_skip", "checkpoint_eval_n_envs"):
            self.assertEqual(worker_config[key], invocation_config[key])
        self.assertEqual(execution_mode, TrainingExecutionMode.SUPERVISED)

    def test_obs_resize_is_the_only_cli_flag(self) -> None:
        parser = argparse.ArgumentParser()
        add_env_config_args(
            parser,
            watchdog_steps_default=987,
            defaults=EnvConfig(),
            parse_json_value=json.loads,
            parse_obs_crop=lambda value: tuple(int(item) for item in value.split(",")),
        )

        self.assertEqual(parser.parse_args(["--obs-resize", "72,96"]).obs_resize, (72, 96))
        self.assertEqual(parser.parse_args([]).watchdog_steps, 987)
        with self.assertRaises(SystemExit):
            parser.parse_args(["--observation-size", "72"])
        normalized = validate_and_normalize_train_config({"obs_resize": [72, 96]})
        self.assertEqual(normalized["obs_resize"], (72, 96))
        with self.assertRaisesRegex(ValueError, "both be zero or both be positive"):
            validate_and_normalize_train_config({"obs_resize": [0, 96]})

    def test_task_resolves_to_train_config_field(self) -> None:
        field = train_config_field_for_key("task")

        self.assertIsNotNone(field)
        self.assertEqual(field.dest, "task")
        self.assertTrue(field.environment)

    def test_checkpoint_eval_backend_supports_only_modal_and_none(self) -> None:
        normalized = validate_and_normalize_train_config({"checkpoint_eval_backend": "none"})
        self.assertEqual(
            normalized["checkpoint_eval_backend"],
            "none",
        )
        with self.assertRaisesRegex(ValueError, "must be one of modal, none"):
            validate_and_normalize_train_config({"checkpoint_eval_backend": "local"})

    def test_metrics_schema_version_accepts_only_active_v17(self) -> None:
        self.assertEqual(
            validate_and_normalize_train_config({"metrics_schema_version": 17})[
                "metrics_schema_version"
            ],
            17,
        )
        with self.assertRaisesRegex(ValueError, "must be >= 17"):
            validate_and_normalize_train_config({"metrics_schema_version": 16})
        with self.assertRaisesRegex(ValueError, "must be <= 17"):
            validate_and_normalize_train_config({"metrics_schema_version": 18})

    def test_no_eval_config_rejects_eval_metric_stop_behavior(self) -> None:
        with self.assertRaisesRegex(ValueError, "must use a train/\\* metric"):
            validate_and_normalize_train_config(
                {
                    "checkpoint_eval_backend": "none",
                    "early_stop": {
                        "conditions": {
                            "invalid": {
                                "metric": "eval/full/outcome/success/across_starts/rate/min",
                                "trigger": "threshold",
                                "operator": ">=",
                                "threshold": 1.0,
                                "patience_steps": 0,
                                "outcome": "failure",
                                "action": "stop",
                            }
                        }
                    },
                }
            )

    def test_no_eval_config_allows_training_metric_stop_behavior(self) -> None:
        config = validate_and_normalize_train_config(
            {
                "checkpoint_eval_backend": "none",
                "early_stop": {
                    "conditions": {
                        "clear": {
                            "metric": "train/outcome/success/across_starts/window_100/rate/min",
                            "trigger": "threshold",
                            "operator": ">=",
                            "threshold": 1.0,
                            "patience_steps": 0,
                            "outcome": "success",
                            "action": "stop",
                        }
                    }
                },
            }
        )

        self.assertEqual(
            config["early_stop"],
            {
                "conditions": {
                    "clear": {
                        "metric": "train/outcome/success/across_starts/window_100/rate/min",
                        "trigger": "threshold",
                        "outcome": "success",
                        "action": "stop",
                        "start_after_steps": 0,
                        "patience_steps": 0,
                        "operator": ">=",
                        "threshold": 1.0,
                    }
                }
            },
        )

    def test_eval_acceptance_allows_training_success_as_a_learner_stop(self) -> None:
        failure_condition = {
            "metric": "train/episode/return/shaped/from/target/rolling_up_to_100/mean",
            "trigger": "no_improvement",
            "direction": "maximize",
            "min_delta": 0.01,
            "delta_mode": "relative",
            "start_after_steps": 1_000_000,
            "patience_steps": 1_000_000,
            "outcome": "failure",
            "action": "observe",
        }
        acceptance = [
            {
                "metric": "eval/full/outcome/success/across_starts/rate/min",
                "operator": ">=",
                "threshold": 1.0,
            }
        ]
        config = validate_and_normalize_train_config(
            {
                "checkpoint_eval_acceptance": acceptance,
                "early_stop": {"conditions": {"plateau": failure_condition}},
            }
        )
        self.assertEqual(
            config["early_stop"]["conditions"]["plateau"]["outcome"],
            "failure",
        )

        success_condition = {
            "metric": "train/outcome/success/across_starts/window_100/rate/min",
            "trigger": "threshold",
            "operator": ">=",
            "threshold": 1.0,
            "patience_steps": 0,
            "outcome": "success",
            "action": "stop",
        }
        dual_mode = validate_and_normalize_train_config(
            {
                "checkpoint_eval_backend": "modal",
                "checkpoint_eval_acceptance": acceptance,
                "early_stop": {"conditions": {"clear": success_condition}},
            }
        )
        self.assertTrue(checkpoint_eval_requires_acceptance(dual_mode))
        self.assertEqual(dual_mode["checkpoint_eval_backend"], "modal")
        self.assertEqual(
            dual_mode["early_stop"]["conditions"]["clear"]["outcome"],
            "success",
        )
        with self.assertRaisesRegex(ValueError, "is not a known train config field"):
            validate_and_normalize_train_config({"stop_on_acceptance": True})

    def test_train_config_rejects_noncurrent_stochastic_eval_flag(self) -> None:
        with self.assertRaisesRegex(ValueError, "is not a known train config field"):
            validate_and_normalize_train_config({"post_train_eval_stochastic": False})
        with self.assertRaisesRegex(ValueError, "is not a known train config field"):
            validate_and_normalize_train_config({"supervision_liveness": {}})

    def test_field_validation_uses_choices_and_numeric_bounds(self) -> None:
        validate_train_config_fields(
            {"wandb_mode": "online", "frame_skip": 1},
            keys=("wandb_mode", "frame_skip"),
        )

        with self.assertRaisesRegex(ValueError, "wandb_mode.*one of online, offline, disabled"):
            validate_train_config_value("wandb_mode", "maybe")

    def test_wandb_publication_is_derived_from_mode(self) -> None:
        self.assertTrue(wandb_publication_enabled({}))
        self.assertTrue(wandb_publication_enabled({"wandb_mode": "online"}))
        self.assertTrue(wandb_publication_enabled({"wandb_mode": "offline"}))
        self.assertFalse(wandb_publication_enabled({"wandb_mode": "disabled"}))
        with self.assertRaisesRegex(ValueError, "is not a known train config field"):
            validate_and_normalize_train_config({"wandb": True})
        with self.assertRaisesRegex(ValueError, "frame_skip.*>= 1"):
            validate_train_config_value("frame_skip", 0)

    def test_field_validation_uses_sequence_and_crop_shapes(self) -> None:
        validate_train_config_fields(
            {
                "states": ["Level1-1", "Level1-2"],
                "obs_crop": [32, 0, 0, 0],
                "task": {"id": "identity"},
            },
            keys=("states", "obs_crop", "task"),
        )

        with self.assertRaisesRegex(ValueError, "states must be a list"):
            validate_train_config_value("states", "Level1-1")
        with self.assertRaisesRegex(ValueError, r"obs_crop\[0\].*non-negative integer"):
            validate_train_config_value("obs_crop", [-1, 0, 0, 0])
        with self.assertRaisesRegex(ValueError, "task must be an object"):
            validate_train_config_value("task", ["identity"])


if __name__ == "__main__":
    unittest.main()
