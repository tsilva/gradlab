from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rlab.config_validation import (
    main as validate_main,
    validate_experiment_tree,
    validate_goal_contract,
)
from rlab.main import COMMANDS


class ConfigValidationTests(unittest.TestCase):
    def test_checked_in_experiment_tree_validates(self) -> None:
        report = validate_experiment_tree(Path("."))

        self.assertEqual(report.issues, ())
        self.assertEqual(report.counts["json_files"], 0)
        self.assertGreaterEqual(report.counts["yaml_files"], 196)
        self.assertGreaterEqual(report.counts["train_specs"], 179)
        self.assertGreaterEqual(report.counts["goals"], 5)
        self.assertGreaterEqual(report.counts["benchmark_profiles"], 7)

    def test_goal_validator_reports_missing_default_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            goal_dir = root / "experiments" / "goals" / "bad"
            goal_dir.mkdir(parents=True)
            (root / "experiments" / "policies").mkdir(parents=True)
            (root / "experiments" / "instances.yaml").write_text("instances: {}\n", encoding="utf-8")
            (root / "experiments" / "fleet.yaml").write_text("hosts: {}\n", encoding="utf-8")
            (root / "experiments" / "policies" / "capacity_policy.yaml").write_text(
                "schema_version: 1\nlanes: []\n",
                encoding="utf-8",
            )
            goal_path = goal_dir / "goal.yaml"
            goal_path.write_text(
                """
schema_version: 1
goal_slug: bad
title: Bad Goal
status: draft
goal_dir: experiments/goals/bad
objective:
  game: SuperMarioBros-Nes-v0
  states: [Level1-1]
  algorithm: PPO
  primary_metric: train/info/level_complete/rate/min/last
  success_threshold: 1.0
  success_window_attempts: 100
  max_train_timesteps: 5000000
selection_policy:
  rank_order: [train/info/level_complete/rate/min/last]
seed_protocol:
  screen: [23]
  confirm: [23]
default_train_spec_file: experiments/goals/bad/specs/missing.yaml
capacity_policy_file: experiments/policies/capacity_policy.yaml
execution:
  hardware_config_file: experiments/instances.yaml
  fleet_config_file: experiments/fleet.yaml
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "default_train_spec_file.*does not exist"):
                validate_goal_contract(goal_path, root)

    def test_validate_is_registered_on_unified_cli(self) -> None:
        self.assertIn("validate", COMMANDS)

    def test_validate_cli_success(self) -> None:
        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            exit_code = validate_main([])

        self.assertEqual(exit_code, 0)
        self.assertIn("YAML config validation passed", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
