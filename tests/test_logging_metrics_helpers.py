from __future__ import annotations

import io
import itertools
import re
import tomllib
import unittest
from pathlib import Path


import gradlab.metric_names as metric_names
from gradlab.training.sb3_helpers import (
    CompactTrainingOutputFormat,
    Sb3HumanOutputFormatHelper,
    disable_sb3_human_output_truncation,
)


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


class Sb3LoggerTests(unittest.TestCase):
    def test_human_output_truncation_is_disabled_for_long_level_complete_metrics(self) -> None:
        from stable_baselines3.common.logger import HumanOutputFormat

        key_values = {
            "train/outcome/success/from/Level1-2_bonus_room_checkpoint/episode/count": 1,
            "train/outcome/success/from/Level1-2_bonus_room_checkpoint/rate/current": 0.0,
        }
        key_excluded = {key: () for key in key_values}

        with self.assertRaisesRegex(ValueError, "truncated"):
            HumanOutputFormat(io.StringIO()).write(key_values, key_excluded)

        output_format = HumanOutputFormat(io.StringIO())

        class FakeLogger:
            output_formats = [output_format]

        class FakeModel:
            logger = FakeLogger()

        disable_sb3_human_output_truncation(FakeModel())

        output_format.write(key_values, key_excluded)
        self.assertEqual(output_format.max_length, 512)

    def test_uninitialized_sb3_logger_is_ignored(self) -> None:
        class FakeSb3Model:
            @property
            def logger(self):
                raise AttributeError("'FakeSb3Model' object has no attribute '_logger'")

        disable_sb3_human_output_truncation(FakeSb3Model())

    def test_callback_updates_logger_after_training_starts(self) -> None:
        from stable_baselines3.common.logger import HumanOutputFormat

        output_format = HumanOutputFormat(io.StringIO())

        class FakeLogger:
            output_formats = [output_format]

        class FakeModel:
            _logger = FakeLogger()

        callback = Sb3HumanOutputFormatHelper(max_length=256)
        callback.model = FakeModel()
        callback._on_training_start()

        self.assertEqual(output_format.max_length, 256)

    def test_compact_local_output_shows_only_mean_return_and_completion_rate(
        self,
    ) -> None:
        output = io.StringIO()
        output_format = CompactTrainingOutputFormat(output)

        output_format.write(
            {
                "rollout/ep_rew_mean": 99.0,
                "train/episode/return/shaped/origin/target/rolling/mean": 357.25,
                "train/outcome/success/starts/observed/cumulative/rate/mean": 0.125,
                "train/algorithm/ppo/update/value_loss": 42.0,
                "time/fps": 1_344,
            },
            {},
        )

        rendered = strip_ansi(output.getvalue())
        self.assertIn("mean return", rendered)
        self.assertIn("357", rendered)
        self.assertIn("completion rate", rendered)
        self.assertIn("12.50%", rendered)
        self.assertNotIn("value_loss", rendered)
        self.assertNotIn("fps", rendered)
        self.assertNotIn("99", rendered)

    def test_compact_callback_replaces_only_the_human_writer(self) -> None:
        from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger

        human_output = io.StringIO()
        human_format = HumanOutputFormat(human_output)

        class CompleteMetricWriter(KVWriter):
            def __init__(self) -> None:
                self.received: dict[str, object] = {}

            def write(self, key_values, key_excluded, step=0) -> None:
                del key_excluded, step
                self.received = dict(key_values)

            def close(self) -> None:
                pass

        complete_format = CompleteMetricWriter()
        logger = Logger(folder=None, output_formats=[human_format, complete_format])

        class FakeModel:
            _logger = logger

        callback = Sb3HumanOutputFormatHelper(compact=True)
        callback.model = FakeModel()
        callback._on_training_start()

        self.assertIsInstance(
            logger.output_formats[0],
            CompactTrainingOutputFormat,
        )
        self.assertIs(logger.output_formats[1], complete_format)

        logger.record("train/episode/return/shaped/origin/target/rolling/mean", 10.0)
        logger.record("train/outcome/success/starts/observed/cumulative/rate/mean", 0.5)
        logger.record("train/algorithm/ppo/update/value_loss", 42.0)
        logger.dump(step=8_192)

        self.assertEqual(
            complete_format.received["train/algorithm/ppo/update/value_loss"],
            42.0,
        )
        rendered = strip_ansi(human_output.getvalue())
        self.assertIn("mean return", rendered)
        self.assertIn("completion rate", rendered)
        self.assertNotIn("value_loss", rendered)


class MetricsDocumentationTests(unittest.TestCase):
    def test_registry_rejects_unknown_metrics_and_unsafe_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown metric"):
            metric_names.validate_metric_name("train/mystery/value")
        with self.assertRaisesRegex(ValueError, "unknown metric"):
            metric_names.validate_metric_name("eval/full/episode/return/typo")
        with self.assertRaisesRegex(ValueError, "unknown metric"):
            metric_names.validate_metric_name("leader/checkpoint/typo")
        with self.assertRaisesRegex(ValueError, "metric dimension"):
            metric_names.train_success_count_metric("unsafe start")

    def test_registry_rejects_noncurrent_metric_names(self) -> None:
        removed = (
            "global_step",
            "train/episode/count",
            "train/episode/return/shaped/mean",
            "train/throughput/rollout_fps",
            "train/early_stop/clear_100/would_trigger",
            "train/outcome/success/from/Start/rate/current",
            "eval/checkpoint_step",
            "eval/full/episode/return/mean",
            "eval/full/episode/count",
            "eval/full/outcome/success/from/Start/rate",
            "eval/full/outcome/reason/stalled/count",
            "eval/full/checkpoint/step",
            "train/throughput/loop_seconds",
            "leader/checkpoint/steps_to_goal",
            "leader/checkpoint/local_path",
            "leader/checkpoint/rank",
            "leader/checkpoint/objective_name",
            "leader/checkpoint/objective",
            "leader/checkpoint/rank_values",
            "leader/checkpoint/acceptance_pass",
            "eval/screen/candidate/pass",
            "eval/acceptance/failure/count",
        )
        for name in removed:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "unknown metric"):
                    metric_names.validate_metric_name(name)

    def test_logger_boundary_rejects_misspelled_gradlab_metrics(self) -> None:
        with self.assertRaisesRegex(ValueError, "logger boundary"):
            metric_names.canonical_training_scalars({"train/outcome/succes/current/rate/min": 0.5})

    def test_a2c_training_scalars_use_the_a2c_metric_namespace(self) -> None:
        payload = metric_names.canonical_training_scalars(
            {
                "train/policy_loss": -0.25,
                "train/value_loss": 1.5,
                "train/entropy_loss": -0.75,
                "train/learning_rate": 0.0007,
            },
            algorithm_id="a2c",
        )

        self.assertEqual(
            payload,
            {
                "train/algorithm/a2c/update/policy_gradient_loss": -0.25,
                "train/algorithm/a2c/update/value_loss": 1.5,
                "train/algorithm/a2c/policy/entropy": 0.75,
                "train/algorithm/a2c/update/learning_rate": 0.0007,
            },
        )
        self.assertFalse(any("/ppo/" in name for name in payload))

    def test_cardinality_has_no_start_by_reason_scalar_product(self) -> None:
        starts = [f"Start-{index}" for index in range(32)]
        reasons = [f"reason-{index}" for index in range(5)]
        names = {metric_names.train_success_count_metric(start) for start in starts} | {
            metric_names.train_outcome_reason_rolling_rate_metric(reason)
            for reason in reasons
        }
        self.assertEqual(len(names), len(starts) + len(reasons))
        self.assertFalse(any("/reason/" in name and "/from/" in name for name in names))

    def test_eval_outcome_cardinality_stays_bounded(self) -> None:
        names = {metric_names.EVAL_FULL_START_TABLE}
        names.update(
            {
                metric_names.EVAL_FULL_OUTCOME_SUCCESS_STARTS_RATE_MIN,
                metric_names.EVAL_FULL_OUTCOME_SUCCESS_STARTS_RATE_MEAN,
            }
        )

        self.assertEqual(len(names), 3)
        self.assertLessEqual(len(names), 50)

    def test_cardinality_margins_and_single_start_lifecycle(self) -> None:
        starts = ["Start"]
        reasons = [f"reason-{index}" for index in range(5)]
        values = {
            "algorithm": list(metric_names.TRAIN_ACTOR_CRITIC_ALGORITHMS),
            "reason": reasons,
            "start": starts,
            "component": ["progress"],
            "progress": ["x"],
            "condition": ["return_plateau"],
        }
        scalar_names: set[str] = set()
        for definition in metric_names.METRIC_DEFINITIONS:
            if definition.unit == "table" or definition.placement == "summary":
                continue
            placeholders = re.findall(r"\{([^}]+)\}", definition.name)
            for replacements in itertools.product(*(values[name] for name in placeholders)):
                name = definition.name
                for placeholder, replacement in zip(placeholders, replacements, strict=True):
                    name = name.replace(f"{{{placeholder}}}", replacement, 1)
                scalar_names.add(name)

        self.assertEqual(len(metric_names.METRIC_DEFINITIONS), 98)
        self.assertEqual(len(scalar_names), 109)
        self.assertEqual(
            len(
                {
                    metric_names.train_success_count_metric("A"),
                    metric_names.train_success_rolling_rate_metric("A"),
                }
            ),
            2,
        )

    def test_registry_grammar_units_templates_and_package_data_are_bounded(self) -> None:
        allowed_units = {
            "boolean",
            "boundaries",
            "bytes",
            "calls",
            "cells",
            "checkpoints",
            "entries",
            "episodes",
            "evaluations",
            "events",
            "fraction",
            "metadata",
            "nats",
            "return",
            "scalar",
            "seconds",
            "sequences",
            "steps",
            "table",
            "text",
            "timestamp",
            "trajectories",
            "transitions",
            "value",
            "visits",
            "transitions/second",
        }
        for definition in metric_names.METRIC_DEFINITIONS:
            with self.subTest(metric=definition.name):
                self.assertRegex(
                    definition.name,
                    r"^(train|eval|leader|orchestration)/",
                )
                self.assertNotIn("//", definition.name)
                self.assertIn(definition.unit, allowed_units)
                placeholders = re.findall(r"\{([^}]+)\}", definition.name)
                self.assertTrue(
                    set(placeholders) <= set(metric_names._PLACEHOLDER_PATTERNS)
                )
                sample_values = {
                    "algorithm": "ppo",
                    "protocol": "full",
                }
                sample = definition.name
                for placeholder in placeholders:
                    sample = sample.replace(
                        f"{{{placeholder}}}",
                        sample_values.get(placeholder, "sample"),
                        1,
                    )
                matches = [
                    candidate.name
                    for candidate, pattern in metric_names._DEFINITION_PATTERNS
                    if pattern.fullmatch(sample)
                ]
                self.assertEqual(matches, [definition.name])

        project_root = Path(__file__).resolve().parents[1]
        pyproject = tomllib.loads(
            (project_root / "pyproject.toml").read_text(encoding="utf-8")
        )
        force_include = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"][
            "force-include"
        ]
        self.assertEqual(force_include["METRICS.md"], "gradlab/METRICS.md")
        self.assertEqual(
            len(
                {
                    metric_names.train_reward_component_metric("active", stat)
                    for stat in ("mean", "nonzero_rate", "share")
                }
            ),
            3,
        )
