from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from gradlab.batch_runtime import EpisodeRecord
from gradlab.env import EnvConfig
from gradlab.eval import ScriptedPolicy
from gradlab.eval_metrics import (
    eval_by_start_records,
    episode_rank,
    episode_result_from_record,
    episode_reasons,
    is_level_complete,
    run_eval_episode,
    summarize_episode_results,
)
from gradlab.eval_runner import (
    _acceptance_runtime_config,
    evaluate_model_episodes,
)
from gradlab.metric_names import metric_path_segment
from gradlab.modal_eval_protocol import SEED_PROTOCOL
from gradlab.env_registry import environment_spec
from gradlab.task_kernels import Outcome
from gradlab.training_metrics import EpisodeMetricsReducer
from gradlab.ranking import rank_score, require_objective_rank
from gradlab.task_kernels import default_task_document
from gradlab.video import PolicyObservationPreview
from gradlab.checkpoint_acceptance import build_checkpoint_eval_contract


MARIO_RANK = [
    "min(leader/checkpoint/step)",
    "max(eval/full/episode/return/shaped/mean)",
]


def eval_checkpoint_score(
    metrics: dict[str, object],
    selection_rank: object,
) -> tuple[float, ...]:
    return rank_score(metrics, require_objective_rank(selection_rank))


class EvalPreviewEquivalenceTests(unittest.TestCase):
    def test_policy_observation_capture_does_not_change_actions_or_results(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.actions: list[list[int]] = []

            def predict(self, obs, deterministic):
                actions = [int(obs[lane, -1, 0, 0]) % 2 for lane in range(obs.shape[0])]
                self.actions.append(actions)
                return np.asarray(actions, dtype=np.int64), None

        class FakeVecEnv:
            def __init__(self) -> None:
                self.step_count = 0
                self.records = []

            def reset(self):
                return np.zeros((2, 4, 84, 84), dtype=np.uint8)

            def step(self, _actions):
                self.step_count += 1
                obs = np.full((2, 4, 84, 84), self.step_count, dtype=np.uint8)
                if self.step_count == 2:
                    self.records = [
                        EpisodeRecord(
                            lane=lane,
                            episode_index=0,
                            start_id="Level1-1",
                            episode_return=float(lane + 1),
                            episode_length=2,
                            terminated=False,
                            truncated=True,
                            outcome=Outcome.TIMEOUT,
                            events=(),
                            metrics={"max_x_pos": 10 + lane},
                        )
                        for lane in range(2)
                    ]
                dones = np.asarray([self.step_count == 2] * 2, dtype=bool)
                return obs, np.zeros(2), dones, [{}, {}]

            def drain_records(self):
                records, self.records = self.records, []
                return records

            def close(self) -> None:
                pass

        config = EnvConfig(
            game="SuperMarioBros-Nes-v0",
            task=default_task_document("mario"),
        )
        models = [FakeModel(), FakeModel()]
        with patch(
            "gradlab.eval_runner.make_eval_vec_env", side_effect=[FakeVecEnv(), FakeVecEnv()]
        ):
            baseline, _ = evaluate_model_episodes(
                model=models[0],
                config=config,
                episodes=2,
                seed=7,
                watchdog_steps=10,
                deterministic=False,
                n_envs=2,
            )
            capture = PolicyObservationPreview(max_frames=10, max_lanes=2)
            recorded, _ = evaluate_model_episodes(
                model=models[1],
                config=config,
                episodes=2,
                seed=7,
                watchdog_steps=10,
                deterministic=False,
                n_envs=2,
                preview_capture=capture,
            )

        self.assertEqual(models[0].actions, models[1].actions)
        self.assertEqual(baseline, recorded)
        self.assertEqual(len(capture.frames), 2)


class EvalByStartTableTests(unittest.TestCase):
    def test_records_preserve_skewed_start_returns_and_overlapping_reasons(self) -> None:
        records = eval_by_start_records(
            [
                {
                    "start_state": "A",
                    "return": 1.0,
                    "level_complete": True,
                    "events": ["level_change"],
                    "terminated": True,
                    "truncated": False,
                },
                {
                    "start_state": "A",
                    "return": 9.0,
                    "level_complete": False,
                    "events": ["life_loss", "level_change"],
                    "terminated": True,
                    "truncated": False,
                },
                {
                    "start_state": "B",
                    "return": 100.0,
                    "level_complete": False,
                    "events": [],
                    "terminated": False,
                    "truncated": True,
                },
            ]
        )

        self.assertEqual(
            records,
            [
                {
                    "start_id": "A",
                    "episode_count": 2,
                    "success_count": 1,
                    "success_rate": 0.5,
                    "shaped_return_mean": 5.0,
                    "failure_reasons": {"level_change": 1, "life_loss": 1},
                },
                {
                    "start_id": "B",
                    "episode_count": 1,
                    "success_count": 0,
                    "success_rate": 0.0,
                    "shaped_return_mean": 100.0,
                    "failure_reasons": {"timeout": 1},
                },
            ],
        )


class EvalMetricTests(unittest.TestCase):
    def test_episode_result_normalizes_numpy_terminal_info_for_strict_json(self) -> None:
        record = EpisodeRecord(
            lane=0,
            episode_index=0,
            start_id="default",
            episode_return=-1.0,
            episode_length=10,
            terminated=False,
            truncated=True,
            outcome=Outcome.TIMEOUT,
            events=(),
            metrics={"nested_count": np.int64(3)},
        )

        result = episode_result_from_record(
            record,
            semantics=environment_spec("gymnasium", "Taxi-v3").eval_semantics,
            terminal_info={
                "action_mask": np.asarray([1, 0, 1, 0, 0, 1], dtype=np.int8),
                "terminal_observation": np.asarray(42, dtype=np.int64),
            },
        )

        self.assertEqual(result["final_info"]["action_mask"], [1, 0, 1, 0, 0, 1])
        self.assertEqual(result["final_info"]["nested_count"], 3)
        self.assertNotIn("terminal_observation", result["final_info"])
        json.dumps(result, allow_nan=False)

    def test_training_and_eval_share_terminal_reason_suffixes(self) -> None:
        record = EpisodeRecord(
            lane=0,
            episode_index=0,
            start_id="Start",
            episode_return=-1.0,
            episode_length=10,
            terminated=True,
            truncated=False,
            outcome=Outcome.FAILURE,
            events=(),
            metrics={},
        )
        training = EpisodeMetricsReducer(track_success=False).consume((record,))
        result = episode_result_from_record(
            record,
            semantics=environment_spec("gymnasium", "CartPole-v1").eval_semantics,
        )

        self.assertEqual(
            training["train/outcome/failure/reason/terminated/rolling/rate"],
            1.0,
        )
        self.assertEqual(episode_reasons(result), {"terminated"})

    def test_deathmatch_projects_and_summarizes_raw_killcount(self) -> None:
        semantics = environment_spec(
            "env-vizdoom-turbo",
            "VizdoomDeathmatch-v1",
        ).eval_semantics
        record = EpisodeRecord(
            lane=0,
            episode_index=0,
            start_id="default",
            episode_return=99.0,
            episode_length=10,
            terminated=True,
            truncated=False,
            outcome=Outcome.FAILURE,
            events=("player_died",),
            metrics={"kills": 12},
        )
        first = episode_result_from_record(record, semantics=semantics)
        second = {**first, "kills": 8, "return": -100.0}

        self.assertEqual(first["kills"], 12)
        self.assertEqual(episode_rank(first, semantics), (12.0, 99.0))
        summary = summarize_episode_results(
            [first, second],
            deterministic=False,
            semantics=semantics,
        )
        self.assertEqual(summary["eval/full/progress/kills/mean"], 10.0)
        self.assertEqual(summary["eval/full/progress/kills/max"], 12)

    def test_model_eval_rejects_deterministic_sampling(self) -> None:
        with self.assertRaisesRegex(ValueError, "deterministic policy evaluation is unsupported"):
            evaluate_model_episodes(
                model=object(),
                config=EnvConfig(game="SuperMarioBros-Nes-v0"),
                episodes=1,
                seed=10_000,
                watchdog_steps=10,
                deterministic=True,
            )

    def test_scripted_policy_resets_per_episode_and_uses_bound_action_space(self) -> None:
        class ActionSpace:
            def sample(self) -> int:
                return 7

        right = ScriptedPolicy("right", ("noop", "right_b", "right_a_b"))
        first_action, _ = right.predict(None, deterministic=False)
        right.predict(None, deterministic=False)
        right.reset_episode()
        reset_action, _ = right.predict(None, deterministic=False)
        self.assertEqual(first_action.tolist(), reset_action.tolist())

        random = ScriptedPolicy("random", ())
        random.bind_action_space(ActionSpace())
        action, _ = random.predict(None, deterministic=False)
        self.assertEqual(action.tolist(), [7])

    def test_scripted_noop_resolves_a_legal_tuple_action(self) -> None:
        contract = {
            "policy": {
                "space": {"type": "multi_discrete"},
                "semantics": {
                    "status": "available",
                    "legal_entries": [
                        {"value": [0, 0, 0, 0, 0, 0], "semantic_id": "noop"},
                        {"value": [1, 0, 0, 0, 0, 0], "semantic_id": "move_forward"},
                    ],
                },
            }
        }
        policy = ScriptedPolicy("noop", ())
        policy.bind_action_contract(contract)

        action, _state = policy.predict(None, deterministic=False)

        self.assertEqual(action.tolist(), [[0, 0, 0, 0, 0, 0]])

    def test_episode_record_clean_completion_precedence(self) -> None:
        success = EpisodeRecord(
            lane=0,
            episode_index=2,
            start_id="Level1-1",
            episode_return=12.5,
            episode_length=40,
            terminated=True,
            truncated=False,
            outcome=Outcome.SUCCESS,
            events=("level_change",),
            metrics={"max_x_pos": 3200, "completion_event": True, "died": False},
        )
        simultaneous_failure = EpisodeRecord(
            lane=1,
            episode_index=3,
            start_id="Level1-2",
            episode_return=-2.0,
            episode_length=12,
            terminated=True,
            truncated=False,
            outcome=Outcome.FAILURE,
            events=("life_loss", "level_change"),
            metrics={"max_x_pos": 900, "completion_event": False, "died": True},
        )

        success_result = episode_result_from_record(success)
        failure_result = episode_result_from_record(simultaneous_failure)

        self.assertTrue(success_result["level_complete"])
        self.assertEqual(success_result["outcome"], "success")
        self.assertEqual(success_result["start_state"], "Level1-1")
        self.assertFalse(failure_result["level_complete"])
        self.assertTrue(failure_result["died"])
        self.assertGreater(episode_rank(success_result), episode_rank(failure_result))

    def test_checkpoint_score_uses_explicit_v2_rank(self) -> None:
        metrics = {
            "eval/full/outcome/success/starts/rate/min": 0.80,
            "eval/full/outcome/success/starts/rate/mean": 0.90,
            "checkpoint_step": 5000000,
            "eval/full/episode/return/shaped/mean": 1200.0,
        }

        self.assertEqual(
            eval_checkpoint_score(metrics, MARIO_RANK),
            (-5_000_000.0, 1200.0),
        )

    def test_checkpoint_score_rejects_missing_or_noncurrent_rank(self) -> None:
        with self.assertRaisesRegex(ValueError, "objective.rank"):
            eval_checkpoint_score({}, ())
        with self.assertRaisesRegex(ValueError, "objective.rank"):
            eval_checkpoint_score({}, ["max(eval/full/reward/mean)"])

    def test_eval_separates_terminal_level_change_from_clean_completion(self) -> None:
        success = episode_result_from_record(
            EpisodeRecord(
                lane=0,
                episode_index=0,
                start_id="Level1-1",
                episode_return=1.0,
                episode_length=10,
                terminated=True,
                truncated=False,
                outcome=Outcome.SUCCESS,
                events=("level_change",),
                metrics={"completion_event": True, "died": False},
            )
        )
        simultaneous_failure = episode_result_from_record(
            EpisodeRecord(
                lane=1,
                episode_index=1,
                start_id="Level1-1",
                episode_return=-1.0,
                episode_length=10,
                terminated=True,
                truncated=False,
                outcome=Outcome.FAILURE,
                events=("level_change", "life_loss"),
                metrics={"completion_event": False, "died": True},
            )
        )

        metrics = summarize_episode_results(
            [success, simultaneous_failure],
            deterministic=False,
            event_names=("level_change", "life_loss"),
            track_success=True,
        )

        self.assertFalse(any("/outcome/reason/" in key for key in metrics))
        self.assertNotIn("eval/full/outcome/success/from/Level1-1/rate", metrics)
        self.assertEqual(metrics["eval/full/outcome/success/starts/rate/min"], 0.5)

    def test_checkpoint_score_uses_reward_when_completion_is_absent(self) -> None:
        metrics = {
            "eval/full/episode/return/shaped/mean": 34.0,
            "eval/full/episode/return/shaped/max": 55.0,
            "checkpoint_step": 5000000,
        }

        rank = [
            "max(eval/full/episode/return/shaped/mean)",
            "max(eval/full/episode/return/shaped/max)",
            "min(leader/checkpoint/step)",
        ]
        self.assertEqual(eval_checkpoint_score(metrics, rank), (34.0, 55.0, -5000000.0))

    def test_checkpoint_score_executes_explicit_goal_rank(self) -> None:
        metrics = {
            "eval/full/episode/return/shaped/mean": 34.0,
            "checkpoint_step": 5000000,
        }

        self.assertEqual(
            eval_checkpoint_score(
                metrics,
                [
                    "min(leader/checkpoint/step)",
                    "max(eval/full/episode/return/shaped/mean)",
                ],
            ),
            (-5000000.0, 34.0),
        )

    def test_generic_eval_summary_does_not_emit_mario_completion_metrics(self) -> None:
        summary = summarize_episode_results(
            [
                {
                    "start_state": "default",
                    "return": 10.0,
                    "steps": 100,
                    "terminated": True,
                    "truncated": False,
                    "final_info": {"ale.lives": 4},
                },
                {
                    "start_state": "default",
                    "return": 4.0,
                    "steps": 200,
                    "terminated": False,
                    "truncated": True,
                    "final_info": {"ale.lives": 3},
                },
            ],
            deterministic=False,
            semantics=environment_spec("gymnasium", "CartPole-v1").eval_semantics,
        )

        self.assertEqual(summary["return_mean"], 7.0)
        self.assertNotIn("eval/full/episode/completed/count", summary)
        self.assertNotIn("eval/full/outcome/reason/terminated/count", summary)
        self.assertFalse(any("/outcome/reason/" in key for key in summary))
        self.assertNotIn("eval/full/outcome/reason/max_steps/count", summary)
        self.assertNotIn("success_count", summary)
        self.assertNotIn("eval/full/outcome/reason/level_change/count", summary)
        self.assertNotIn("max_x_mean", summary)
        self.assertNotIn("death_count", summary)

    def test_generic_success_contract_emits_zero_rates_when_every_episode_fails(self) -> None:
        summary = summarize_episode_results(
            [
                {
                    "start_state": "Start",
                    "return": -1.0,
                    "steps": 10,
                    "terminated": True,
                    "truncated": False,
                    "outcome": "failure",
                    "events": ["goal_reached"],
                }
            ],
            deterministic=False,
            event_names=("goal_reached",),
            track_success=True,
            semantics=environment_spec("gymnasium", "CartPole-v1").eval_semantics,
        )

        self.assertNotIn("eval/full/outcome/success/from/Start/rate", summary)
        self.assertEqual(summary["eval/full/outcome/success/starts/rate/min"], 0.0)
        self.assertEqual(summary["eval/full/outcome/success/starts/rate/mean"], 0.0)

    def test_non_mario_goal_reached_uses_generic_success_outcome(self) -> None:
        summary = summarize_episode_results(
            [
                {
                    "start_state": "Start",
                    "return": 1.0,
                    "steps": 10,
                    "terminated": True,
                    "truncated": False,
                    "outcome": "success",
                    "events": ["goal_reached"],
                }
            ],
            deterministic=False,
            event_names=("goal_reached",),
            track_success=True,
            semantics=environment_spec("gymnasium", "CartPole-v1").eval_semantics,
        )

        self.assertNotIn("eval/full/outcome/success/from/Start/rate", summary)
        self.assertNotIn("eval/full/outcome/reason/goal_reached/count", summary)

    def test_canonical_summary_contains_best_return_before_ranking(self) -> None:
        summary = summarize_episode_results(
            [
                {
                    "start_state": "Start",
                    "return": 5.0,
                    "steps": 10,
                    "terminated": True,
                    "truncated": False,
                    "events": [],
                },
                {
                    "start_state": "Start",
                    "return": 9.0,
                    "steps": 10,
                    "terminated": True,
                    "truncated": False,
                    "events": [],
                },
            ],
            deterministic=False,
            semantics=environment_spec("gymnasium", "CartPole-v1").eval_semantics,
        )

        rank = [
            "max(eval/full/episode/return/shaped/mean)",
            "max(eval/full/episode/return/shaped/max)",
            "min(leader/checkpoint/step)",
        ]
        summary["checkpoint_step"] = 123
        self.assertEqual(eval_checkpoint_score(summary, rank), (7.0, 9.0, -123.0))

    def test_generic_eval_summary_counts_configured_terminal_events(self) -> None:
        summary = summarize_episode_results(
            [
                {
                    "start_state": "Start",
                    "return": 10.0,
                    "steps": 856,
                    "terminated": True,
                    "truncated": False,
                    "events": ["serve_stall"],
                    "final_info": {"ball_y": 0},
                },
                {
                    "start_state": "Start",
                    "return": 4.0,
                    "steps": 54000,
                    "terminated": False,
                    "truncated": True,
                    "events": [],
                    "final_info": {"ball_y": 32},
                },
            ],
            deterministic=False,
            semantics=environment_spec("gymnasium", "CartPole-v1").eval_semantics,
        )

        self.assertNotIn("eval/full/outcome/reason/serve_stall/count", summary)
        self.assertFalse(any("/outcome/reason/" in key for key in summary))
        self.assertNotIn("eval/full/outcome/reason/max_steps/count", summary)
        self.assertFalse(any("/from/Start" in name and "/reason/" in name for name in summary))

    def test_evaluation_passes_the_exact_task_contract_to_the_runtime(self) -> None:
        config = EnvConfig(
            game="Breakout-Atari2600-v0",
            task={
                "termination": {
                    "failure": ["life_loss", "serve_stall"],
                    "max_episode_steps": 54000,
                }
            },
        )

        class FakeEnv:
            def close(self) -> None:
                pass

        fake_env = FakeEnv()
        result = {
            "actions": [],
            "start_state": "default",
            "return": 0.0,
            "steps": 1,
            "outcome": "failure",
            "terminated": True,
            "truncated": False,
            "final_info": {},
        }

        with (
            patch("gradlab.eval_runner.make_eval_vec_env", return_value=fake_env) as make_env,
            patch("gradlab.eval_runner.run_eval_episode", return_value=result),
        ):
            evaluate_model_episodes(
                model=object(),
                config=config,
                episodes=1,
                seed=10_000,
                watchdog_steps=54000,
                deterministic=False,
            )

        self.assertIs(make_env.call_args.kwargs["config"], config)
        self.assertEqual(
            make_env.call_args.kwargs["config"].task["termination"],
            config.task["termination"],
        )

    def test_checkpoint_score_prefers_fewer_timesteps_after_completion_goal(self) -> None:
        slower_higher_reward = {
            "eval/full/outcome/success/starts/rate/min": 1.0,
            "eval/full/outcome/success/starts/rate/mean": 1.0,
            "checkpoint_step": 5000000,
            "eval/full/episode/return/shaped/mean": 1200.0,
        }
        faster_lower_reward = {
            "eval/full/outcome/success/starts/rate/min": 1.0,
            "eval/full/outcome/success/starts/rate/mean": 1.0,
            "checkpoint_step": 3500000,
            "eval/full/episode/return/shaped/mean": 900.0,
        }

        self.assertGreater(
            eval_checkpoint_score(faster_lower_reward, MARIO_RANK),
            eval_checkpoint_score(slower_higher_reward, MARIO_RANK),
        )

    def test_metric_path_segment_preserves_retro_state_names(self) -> None:
        self.assertEqual(metric_path_segment("Level1-2"), "Level1-2")
        with self.assertRaisesRegex(ValueError, "metric dimension"):
            metric_path_segment("Level 1/2")

    def test_episode_rank_prefers_completion_then_progress_then_reward(self) -> None:
        incomplete = {"level_complete": False, "max_x_pos": 4000, "return": 1000.0}
        complete = {"level_complete": True, "max_x_pos": 100, "return": -10.0}
        better_progress = {"level_complete": False, "max_x_pos": 4500, "return": 0.0}
        self.assertGreater(episode_rank(complete), episode_rank(incomplete))
        self.assertGreater(episode_rank(better_progress), episode_rank(incomplete))

    def test_level_complete_uses_explicit_completion_flag(self) -> None:
        self.assertFalse(
            is_level_complete(
                {"level_complete": False, "level_changed": False, "level_max_x_pos": 5000},
            )
        )
        self.assertFalse(
            is_level_complete(
                {"level_complete": False, "level_changed": True},
            )
        )
        self.assertTrue(
            is_level_complete(
                {"level_complete": True, "level_changed": True},
            )
        )

    def test_run_eval_episode_does_not_stop_on_completion(self) -> None:
        class FakeModel:
            def predict(self, obs, deterministic):
                return np.array([0], dtype=np.int64), None

        class FakeEnv:
            def __init__(self) -> None:
                self.step_count = 0
                self.records = []

            def seed(self, seed: int) -> None:
                self.seed_value = seed

            def reset(self):
                self.step_count = 0
                self.records = []
                return np.zeros((1, 4, 84, 84), dtype=np.uint8)

            def step(self, action):
                self.step_count += 1
                obs = np.zeros((1, 4, 84, 84), dtype=np.uint8)
                if self.step_count == 1:
                    return (
                        obs,
                        np.array([1.0], dtype=np.float32),
                        np.array([False]),
                        [
                            {
                                "start_state": "Level1-1",
                                "state": "Level1-1",
                                "max_x_pos": 100,
                                "level_max_x_pos": 100,
                                "level_changed": True,
                                "score": 10,
                                "lives": 3,
                                "time": 300,
                            }
                        ],
                    )
                self.records = [
                    EpisodeRecord(
                        lane=0,
                        episode_index=0,
                        start_id="Level1-1",
                        episode_return=3.0,
                        episode_length=2,
                        terminated=False,
                        truncated=True,
                        outcome=Outcome.TIMEOUT,
                        events=("level_change",),
                        metrics={
                            "max_x_pos": 250,
                            "level_max_x_pos": 150,
                            "completion_event": True,
                        },
                    )
                ]
                return (
                    obs,
                    np.array([2.0], dtype=np.float32),
                    np.array([True]),
                    [
                        {
                            "state": "Level1-2",
                            "max_x_pos": 250,
                            "level_max_x_pos": 150,
                            "score": 20,
                            "lives": 3,
                            "time": 299,
                        }
                    ],
                )

            def drain_records(self):
                records, self.records = self.records, []
                return records

        result = run_eval_episode(
            FakeEnv(),
            FakeModel(),
            watchdog_steps=2,
            deterministic=False,
            seed=7,
            default_start_state="Level1-1",
        )

        self.assertEqual(result["steps"], 2)
        self.assertEqual(result["return"], 3.0)
        self.assertEqual(result["max_x_pos"], 250)
        self.assertEqual(result["start_state"], "Level1-1")
        self.assertTrue(result["level_complete"])
        self.assertFalse(result["terminated"])
        self.assertTrue(result["truncated"])

    def test_acceptance_runtime_pins_manifest_starts_instead_of_sampling(self) -> None:
        starts = ("post400-000", "post400-001")
        contract = build_checkpoint_eval_contract(
            environment={
                "env_provider": "env-breakoutatari2600-turbo-native",
                "env_config": {"states": list(starts), "state_probs": [1, 1]},
            },
            episodes=4,
            n_envs=2,
            watchdog_steps=10,
            seed=10_000,
            seed_protocol=SEED_PROTOCOL,
            acceptance=[
                {
                    "metric": "eval/full/outcome/success/starts/rate/min",
                    "operator": ">=",
                    "threshold": 1.0,
                }
            ],
        )

        runtime_config = _acceptance_runtime_config(
            EnvConfig(
                game="Breakout-Atari2600-v0",
                states=starts,
                state_probs=(1.0, 1.0),
            ),
            acceptance_contract=contract,
            n_envs=2,
        )

        self.assertEqual(runtime_config.state, "")
        self.assertEqual(runtime_config.states, starts)
        self.assertEqual(runtime_config.state_probs, ())

        shared_contract = build_checkpoint_eval_contract(
            environment={"game": "Breakout-Atari2600-v0", "state": "full"},
            episodes=2,
            n_envs=2,
            watchdog_steps=10,
            seed=10_000,
            seed_protocol=SEED_PROTOCOL,
            acceptance=contract["acceptance"],
        )
        shared_runtime_config = _acceptance_runtime_config(
            EnvConfig(game="Breakout-Atari2600-v0", state="full"),
            acceptance_contract=shared_contract,
            n_envs=2,
        )

        self.assertEqual(shared_runtime_config.state, "full")
        self.assertEqual(shared_runtime_config.states, ())
        self.assertEqual(shared_runtime_config.state_probs, ())

    def test_mean_return_acceptance_runs_every_episode_after_failure(self) -> None:
        class FakeEnv:
            action_space = None

            def close(self) -> None:
                pass

        contract = build_checkpoint_eval_contract(
            environment={"game": "SuperMarioBros-Nes-v0", "state": "Level1-1"},
            episodes=3,
            n_envs=1,
            watchdog_steps=10,
            seed=10_000,
            seed_protocol=SEED_PROTOCOL,
            acceptance=[
                {
                    "metric": "eval/full/episode/return/shaped/mean",
                    "operator": ">=",
                    "threshold": 2.0,
                }
            ],
        )
        results = [
            {
                "actions": [],
                "start_state": "Level1-1",
                "return": episode_return,
                "steps": 1,
                "outcome": "failure" if index == 0 else "success",
                "level_complete": index != 0,
                "terminated": True,
                "truncated": False,
                "final_info": {},
            }
            for index, episode_return in enumerate((0.0, 2.0, 4.0))
        ]

        with (
            patch("gradlab.eval_runner.make_eval_vec_env", return_value=FakeEnv()),
            patch("gradlab.eval_runner.run_eval_episode", side_effect=results) as run_episode,
        ):
            metrics, video_path = evaluate_model_episodes(
                model=object(),
                config=EnvConfig(game="SuperMarioBros-Nes-v0", state="Level1-1"),
                episodes=3,
                seed=10_000,
                watchdog_steps=10,
                deterministic=False,
                n_envs=1,
                acceptance_contract=contract,
            )

        self.assertIsNone(video_path)
        self.assertEqual(run_episode.call_count, 3)
        self.assertEqual(len(metrics["episode_results"]), 3)
        self.assertEqual(metrics["eval/full/episode/return/shaped/mean"], 2.0)
        self.assertEqual(metrics["acceptance_verdict"], "accepted")
        self.assertEqual(metrics["acceptance_aggregates"]["failure_count"], 1)

    def test_vector_eval_accumulates_completed_slots_independently(self) -> None:
        class FakeModel:
            def predict(self, obs, deterministic):
                return np.zeros(obs.shape[0], dtype=np.int64), None

        class FakeVecEnv:
            num_envs = 2

            def __init__(self) -> None:
                self.step_count = 0
                self.records = []

            def reset(self):
                self.records = []
                return np.zeros((2, 4, 84, 84), dtype=np.uint8)

            def step(self, action):
                self.step_count += 1
                obs = np.zeros((2, 4, 84, 84), dtype=np.uint8)
                if self.step_count == 1:
                    self.records = [
                        EpisodeRecord(
                            lane=1,
                            episode_index=0,
                            start_id="Level1-2",
                            episode_return=2.0,
                            episode_length=1,
                            terminated=False,
                            truncated=True,
                            outcome=Outcome.TIMEOUT,
                            events=(),
                            metrics={
                                "max_x_pos": 20,
                                "level_max_x_pos": 20,
                                "died": True,
                            },
                        )
                    ]
                    return (
                        obs,
                        np.array([1.0, 2.0], dtype=np.float32),
                        np.array([False, True]),
                        [
                            {"max_x_pos": 10, "level_max_x_pos": 10},
                            {
                                "start_state": "Level1-2",
                                "max_x_pos": 20,
                                "level_max_x_pos": 20,
                                "died": True,
                                "death_x_pos": 20,
                                "TimeLimit.truncated": True,
                                "score": 100,
                                "lives": 2,
                            },
                        ],
                    )
                self.records = [
                    EpisodeRecord(
                        lane=0,
                        episode_index=0,
                        start_id="Level1-1",
                        episode_return=4.0,
                        episode_length=2,
                        terminated=True,
                        truncated=False,
                        outcome=Outcome.SUCCESS,
                        events=("level_change",),
                        metrics={
                            "max_x_pos": 30,
                            "level_max_x_pos": 30,
                            "completion_event": True,
                        },
                    )
                ]
                return (
                    obs,
                    np.array([3.0, 4.0], dtype=np.float32),
                    np.array([True, False]),
                    [
                        {
                            "start_state": "Level1-1",
                            "max_x_pos": 30,
                            "level_max_x_pos": 30,
                            "level_changed": True,
                            "score": 200,
                            "lives": 3,
                        },
                        {"max_x_pos": 40, "level_max_x_pos": 40},
                    ],
                )

            def drain_records(self):
                records, self.records = self.records, []
                return records

            def close(self) -> None:
                pass

        config = EnvConfig(
            game="SuperMarioBros-Nes-v0",
            task=default_task_document("mario"),
        )
        with patch("gradlab.eval_runner.make_eval_vec_env", return_value=FakeVecEnv()):
            metrics, video_path = evaluate_model_episodes(
                model=FakeModel(),
                config=config,
                episodes=2,
                seed=7,
                watchdog_steps=10,
                deterministic=False,
                n_envs=2,
            )

        self.assertIsNone(video_path)
        self.assertEqual(metrics["eval_n_envs"], 2)
        self.assertEqual(metrics["episodes"], 2)
        self.assertEqual(metrics["return_mean"], 3.0)
        self.assertEqual(metrics["success_count"], 1)
        self.assertEqual(metrics["death_count"], 1)
        self.assertNotIn("eval/full/outcome/reason/level_change/count", metrics)
        self.assertNotIn("eval/full/outcome/reason/max_steps/count", metrics)
        self.assertFalse(any("/outcome/reason/" in key for key in metrics))
        self.assertFalse(any("/outcome/success/from/" in key for key in metrics))
        self.assertEqual(metrics["eval/full/outcome/success/starts/rate/min"], 0.0)
        self.assertEqual(metrics["eval/full/outcome/success/starts/rate/mean"], 0.5)
        self.assertEqual(metrics["episode_results"][0]["env_index"], 1)
        self.assertEqual(metrics["episode_results"][0]["seed"], 7)
        self.assertEqual(metrics["episode_results"][0]["seed_protocol"], SEED_PROTOCOL)
        self.assertEqual(metrics["episode_results"][0]["seed_lane"], 1)
        self.assertEqual(metrics["episode_results"][0]["seed_episode_ordinal"], 0)
        self.assertEqual(metrics["episode_results"][0]["start_state"], "Level1-2")
        self.assertEqual(metrics["episode_results"][0]["return"], 2.0)
        self.assertEqual(metrics["episode_results"][1]["env_index"], 0)
        self.assertEqual(metrics["episode_results"][1]["seed_lane"], 0)
        self.assertEqual(metrics["episode_results"][1]["seed_episode_ordinal"], 0)
        self.assertEqual(metrics["episode_results"][1]["start_state"], "Level1-1")
        self.assertEqual(metrics["episode_results"][1]["return"], 4.0)

    def test_vector_eval_uses_canonical_episode_records(self) -> None:
        class FakeModel:
            def predict(self, obs, deterministic):
                return np.zeros(obs.shape[0], dtype=np.int64), None

        class FakeRecordVecEnv:
            num_envs = 2

            def __init__(self) -> None:
                self.records = []

            def reset(self):
                return np.zeros((2, 4, 84, 84), dtype=np.uint8)

            def step(self, action):
                self.records = [
                    EpisodeRecord(
                        lane=0,
                        episode_index=0,
                        start_id="Level1-1",
                        episode_return=4.0,
                        episode_length=2,
                        terminated=True,
                        truncated=False,
                        outcome=Outcome.SUCCESS,
                        events=("level_change",),
                        metrics={"max_x_pos": 300, "completion_event": True},
                    ),
                    EpisodeRecord(
                        lane=1,
                        episode_index=0,
                        start_id="Level1-2",
                        episode_return=-1.0,
                        episode_length=2,
                        terminated=True,
                        truncated=False,
                        outcome=Outcome.FAILURE,
                        events=("life_loss", "level_change"),
                        metrics={"max_x_pos": 100, "completion_event": False, "died": True},
                    ),
                ]
                return (
                    np.zeros((2, 4, 84, 84), dtype=np.uint8),
                    np.zeros(2, dtype=np.float32),
                    np.ones(2, dtype=bool),
                    [{"score": 10, "lives": 3}, {"score": 5, "lives": 2}],
                )

            def drain_records(self):
                records, self.records = self.records, []
                return records

            def close(self) -> None:
                pass

        with patch("gradlab.eval_runner.make_eval_vec_env", return_value=FakeRecordVecEnv()):
            metrics, video_path = evaluate_model_episodes(
                model=FakeModel(),
                config=EnvConfig(
                    game="SuperMarioBros-Nes-v0",
                    task=default_task_document("mario"),
                ),
                episodes=2,
                seed=7,
                watchdog_steps=10,
                deterministic=False,
                n_envs=2,
            )

        self.assertIsNone(video_path)
        self.assertEqual(metrics["success_count"], 1)
        self.assertEqual(metrics["death_count"], 1)
        self.assertEqual(metrics["best_episode"]["outcome"], "success")
        self.assertTrue(metrics["episode_results"][0]["level_complete"])
        self.assertFalse(metrics["episode_results"][1]["level_complete"])

    def test_vector_eval_watchdog_aborts_a_lane_without_episode_records(self) -> None:
        class FakeModel:
            def predict(self, obs, deterministic):
                return np.zeros(obs.shape[0], dtype=np.int64), None

        class FakeVecEnv:
            num_envs = 2

            def __init__(self) -> None:
                self.step_count = 0
                self.closed = False

            def reset(self):
                return np.zeros((2, 4, 84, 84), dtype=np.uint8)

            def step(self, action):
                self.step_count += 1
                return (
                    np.zeros((2, 4, 84, 84), dtype=np.uint8),
                    np.zeros(2, dtype=np.float32),
                    np.zeros(2, dtype=bool),
                    [{}, {}],
                )

            def drain_records(self):
                return []

            def close(self) -> None:
                self.closed = True

        fake_env = FakeVecEnv()
        with (
            patch("gradlab.eval_runner.make_eval_vec_env", return_value=fake_env),
            self.assertRaisesRegex(
                RuntimeError,
                r"watchdog expired.*lanes \[0, 1\]",
            ),
        ):
            evaluate_model_episodes(
                model=FakeModel(),
                config=EnvConfig(
                    game="SuperMarioBros-Nes-v0",
                    task=default_task_document("mario"),
                ),
                episodes=2,
                seed=7,
                watchdog_steps=2,
                deterministic=False,
                n_envs=2,
            )

        self.assertEqual(fake_env.step_count, 2)
        self.assertTrue(fake_env.closed)

    def test_vector_eval_does_not_stop_on_completion(self) -> None:
        class FakeModel:
            def predict(self, obs, deterministic):
                return np.zeros(obs.shape[0], dtype=np.int64), None

        class FakeVecEnv:
            num_envs = 2

            def __init__(self) -> None:
                self.step_count = 0
                self.records = []

            def reset(self):
                self.step_count = 0
                self.records = []
                return np.zeros((2, 4, 84, 84), dtype=np.uint8)

            def step(self, action):
                self.step_count += 1
                obs = np.zeros((2, 4, 84, 84), dtype=np.uint8)
                if self.step_count == 1:
                    return (
                        obs,
                        np.array([1.0, 10.0], dtype=np.float32),
                        np.array([False, False]),
                        [
                            {
                                "start_state": "Level1-1",
                                "state": "Level1-1",
                                "max_x_pos": 100,
                                "level_max_x_pos": 100,
                                "level_changed": True,
                            },
                            {
                                "start_state": "Level1-2",
                                "state": "Level1-2",
                                "max_x_pos": 110,
                                "level_max_x_pos": 110,
                                "level_changed": True,
                            },
                        ],
                    )
                self.records = [
                    EpisodeRecord(
                        lane=0,
                        episode_index=0,
                        start_id="Level1-1",
                        episode_return=3.0,
                        episode_length=2,
                        terminated=False,
                        truncated=True,
                        outcome=Outcome.TIMEOUT,
                        events=("level_change",),
                        metrics={
                            "max_x_pos": 250,
                            "level_max_x_pos": 150,
                            "completion_event": True,
                        },
                    )
                ]
                return (
                    obs,
                    np.array([2.0, 20.0], dtype=np.float32),
                    np.array([True, False]),
                    [
                        {
                            "state": "Level1-2",
                            "max_x_pos": 250,
                            "level_max_x_pos": 150,
                        },
                        {
                            "state": "Level1-3",
                            "max_x_pos": 260,
                            "level_max_x_pos": 160,
                        },
                    ],
                )

            def drain_records(self):
                records, self.records = self.records, []
                return records

            def close(self) -> None:
                pass

        fake_env = FakeVecEnv()
        config = EnvConfig(
            game="SuperMarioBros-Nes-v0",
            task=default_task_document("mario"),
        )
        with patch("gradlab.eval_runner.make_eval_vec_env", return_value=fake_env):
            metrics, video_path = evaluate_model_episodes(
                model=FakeModel(),
                config=config,
                episodes=1,
                seed=7,
                watchdog_steps=2,
                deterministic=False,
                n_envs=2,
            )

        self.assertIsNone(video_path)
        self.assertEqual(fake_env.step_count, 2)
        self.assertEqual(metrics["episodes"], 1)
        self.assertEqual(metrics["success_count"], 1)
        self.assertNotIn("eval/full/outcome/reason/level_change/count", metrics)
        self.assertNotIn("eval/full/outcome/reason/max_steps/count", metrics)
        self.assertEqual(metrics["episode_results"][0]["steps"], 2)
        self.assertEqual(metrics["episode_results"][0]["return"], 3.0)
        self.assertEqual(metrics["episode_results"][0]["max_x_pos"], 250)
        self.assertEqual(metrics["episode_results"][0]["start_state"], "Level1-1")
        self.assertTrue(metrics["episode_results"][0]["level_complete"])
        self.assertFalse(metrics["episode_results"][0]["terminated"])
        self.assertTrue(metrics["episode_results"][0]["truncated"])

    def test_evaluate_model_episodes_updates_progress_bar(self) -> None:
        class FakeEnv:
            def close(self) -> None:
                pass

        class FakeProgressBar:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs
                self.updates: list[int] = []

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                pass

            def update(self, count: int) -> None:
                self.updates.append(count)

        progress_bars: list[FakeProgressBar] = []

        def fake_tqdm(**kwargs) -> FakeProgressBar:
            progress_bar = FakeProgressBar(**kwargs)
            progress_bars.append(progress_bar)
            return progress_bar

        def fake_run_eval_episode(*args, **kwargs) -> dict:
            return {
                "actions": [],
                "start_state": "Level1-1",
                "return": 1.0,
                "max_x_pos": 10,
                "max_level_x_pos": 10,
                "score": 0,
                "lives": 3,
                "time": 399,
                "steps": 1,
                "terminated": True,
                "truncated": False,
                "level_complete": True,
                "died": False,
                "death_x_pos": None,
                "final_info": {"start_state": "Level1-1"},
            }

        with (
            patch("gradlab.eval_runner.make_eval_vec_env", return_value=FakeEnv()),
            patch("gradlab.eval_runner.run_eval_episode", side_effect=fake_run_eval_episode),
            patch("gradlab.eval_runner.tqdm", side_effect=fake_tqdm),
        ):
            metrics, video_path = evaluate_model_episodes(
                model=object(),
                config=EnvConfig(game="SuperMarioBros-Nes-v0"),
                episodes=3,
                seed=7,
                watchdog_steps=10,
                deterministic=False,
                progress=True,
                progress_description="eval checkpoint 4100000",
            )

        self.assertIsNone(video_path)
        self.assertEqual(metrics["episodes"], 3)
        self.assertEqual(len(progress_bars), 1)
        self.assertEqual(progress_bars[0].kwargs["total"], 3)
        self.assertEqual(progress_bars[0].kwargs["desc"], "eval checkpoint 4100000")
        self.assertEqual(progress_bars[0].kwargs["disable"], False)
        self.assertEqual(progress_bars[0].updates, [1, 1, 1])

    def test_best_episode_video_replays_through_eval_vec_env(self) -> None:
        class FakePolicyEnv:
            def close(self) -> None:
                pass

        class FakeVideoEnv:
            def __init__(self) -> None:
                self.actions = []
                self.frame = 0

            def seed(self, seed: int) -> None:
                self.seed_value = seed

            def reset(self):
                self.frame = 0
                return np.zeros((1, 4, 84, 84), dtype=np.uint8)

            def step(self, action):
                self.actions.append(np.asarray(action).copy())
                self.frame += 1
                return (
                    np.zeros((1, 4, 84, 84), dtype=np.uint8),
                    np.zeros(1, dtype=np.float32),
                    np.zeros(1, dtype=bool),
                    [{}],
                )

            def get_images(self):
                return [np.full((4, 4, 3), self.frame, dtype=np.uint8)]

            def close(self) -> None:
                pass

        result = {
            "actions": [1, 2],
            "start_state": "Level1-1",
            "return": 3.0,
            "max_x_pos": 20,
            "max_level_x_pos": 20,
            "score": 0,
            "lives": 3,
            "time": 399,
            "steps": 2,
            "terminated": True,
            "truncated": False,
            "level_complete": True,
            "died": False,
            "death_x_pos": None,
            "final_info": {},
        }
        video_env = FakeVideoEnv()
        output = Path("/tmp/gradlab-eval-video.mp4")
        config = EnvConfig(
            game="SuperMarioBros-Nes-v0",
            task={"termination": {"success": ["goal_reached"]}},
        )
        with (
            patch(
                "gradlab.eval_runner.make_eval_vec_env",
                side_effect=[FakePolicyEnv(), video_env],
            ) as make_env,
            patch("gradlab.eval_runner.run_eval_episode", return_value=result),
            patch("gradlab.eval_runner.write_video") as write_video,
        ):
            metrics, video_path = evaluate_model_episodes(
                model=object(),
                config=config,
                episodes=1,
                seed=10_007,
                watchdog_steps=10,
                deterministic=False,
                capture_best_video=True,
                video_path=output,
            )

        self.assertEqual(video_path, output)
        self.assertEqual(metrics["best_episode_video"], str(output))
        self.assertEqual(make_env.call_count, 2)
        self.assertIs(make_env.call_args_list[1].kwargs["config"], config)
        self.assertEqual(config.task["termination"]["success"], ["goal_reached"])
        self.assertEqual(len(video_env.actions), 2)
        written_frames = write_video.call_args.args[0]
        self.assertEqual(len(written_frames), 3)
