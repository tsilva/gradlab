from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from gymnasium import spaces

from gradlab.action_program import ActionProgramPolicy
from gradlab.batch_runtime import EpisodeRecord
from gradlab.env import EnvConfig
from gradlab.go_explore import GoExploreSearch
from gradlab.metric_names import (
    TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN,
    TRAIN_OUTCOME_SUCCESS_CURRENT_RATE_MEAN,
)
from gradlab.task_kernels import Outcome
from gradlab.training.go_explore import (
    GO_EXPLORE_PROVIDER_INFO_KEYS,
    _runtime_environment_config,
    normalize_config,
    run_go_explore,
)
from gradlab.training_backend import BackendContext, GracefulStopFlag
from gradlab.training_lifecycle import (
    TerminalReason,
    TrainingExecutionMode,
    TrainingExecutionPolicy,
    TrainingSession,
)
from gradlab.training_metrics import EpisodeMetricsReducer


class GoExploreSearchTests(unittest.TestCase):
    @staticmethod
    def search() -> GoExploreSearch:
        return GoExploreSearch(
            n_envs=2,
            seed=17,
            action_names=("noop", "right", "jump"),
            fallback_action="noop",
            explore_steps=1,
            run_duration_mean=2.0,
            run_duration_max=4,
        )

    def test_durable_state_round_trip_preserves_exact_next_actions(self) -> None:
        search = self.search()
        search.initialize((b"initial-a", b"initial-b"), ("entry-a", "entry-b"))
        search.next_actions()
        observation = search.observe(
            rewards=np.asarray([1.0, 2.0]),
            dones=np.asarray([False, False]),
            cell_keys=(b"cell-a", b"cell-b"),
            progresses=np.asarray([8.0, 16.0]),
        )
        self.assertTrue(np.all(observation.archive_mask))
        self.assertTrue(np.all(observation.restart_mask))
        search.commit_archive(("entry-cell-a", "entry-cell-b"))
        search.take_completion_events()
        search.restart(observation.restart_mask)

        document = search.state_document(("lane-a", "lane-b"))
        restored = self.search()
        self.assertEqual(restored.restore_state(document), ("lane-a", "lane-b"))
        self.assertEqual(restored.state_document(("lane-a", "lane-b")), document)
        np.testing.assert_array_equal(restored.next_actions(), search.next_actions())

    def test_runtime_environment_selects_route_and_task_provider_info(self) -> None:
        config = EnvConfig(
            env_provider="supermariobrosnes-turbo",
            game="SuperMarioBros-Nes-v0",
            env_args={"info_filter": "all"},
            task={
                "id": "mario",
                "signals": {
                    "x": ["xscrollHi", "xscrollLo"],
                    "custom": "game_mode",
                },
            },
        )

        runtime_config = _runtime_environment_config(config)
        info_filter = runtime_config.env_args["info_filter"]

        self.assertEqual(info_filter["mode"], "all")
        self.assertEqual(
            set(info_filter["keys"]),
            set(GO_EXPLORE_PROVIDER_INFO_KEYS) | {"game_mode"},
        )
        self.assertEqual(config.env_args["info_filter"], "all")

    def test_backend_uses_compaction_without_legacy_archive_recovery(self) -> None:
        config = normalize_config(
            "gradlab.go-explore",
            {"compaction_interval_steps": 500_000},
            label="backend",
        )

        self.assertEqual(config["compaction_interval_steps"], 500_000)
        with self.assertRaisesRegex(ValueError, "recovery_interval_steps"):
            normalize_config(
                "gradlab.go-explore",
                {"recovery_interval_steps": 500_000},
                label="backend",
            )

    def test_search_exports_a_neutral_action_program(self) -> None:
        self.assertIsInstance(self.search().policy(), ActionProgramPolicy)

    def test_local_run_saves_only_one_final_or_interrupted_model(self) -> None:
        for interrupted in (False, True):
            with self.subTest(interrupted=interrupted), tempfile.TemporaryDirectory() as tmp:
                checkpoints, progress, result = self._run_backend(
                    Path(tmp),
                    execution_mode=TrainingExecutionMode.LOCAL_DEMO,
                    interrupted=interrupted,
                    completion_event=False,
                )

                self.assertEqual(len(checkpoints), 1)
                self.assertEqual(checkpoints[0]["path"], Path(tmp) / "final_model.zip")
                self.assertFalse((Path(tmp) / "checkpoints").exists())
                self.assertEqual(
                    checkpoints[0]["kind"],
                    "interrupted" if interrupted else "final",
                )
                self.assertEqual(
                    result.terminal_reason,
                    (
                        TerminalReason.LOCAL_INTERRUPTION
                        if interrupted
                        else TerminalReason.RESOURCE_EXHAUSTION
                    ),
                )
                self.assertEqual(progress.n, 1 if interrupted else 2)
                self.assertNotIn(
                    TRAIN_OUTCOME_SUCCESS_CURRENT_RATE_MEAN,
                    progress.metrics,
                )

    def test_queued_run_preserves_intermediate_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoints, _progress, result = self._run_backend(
                Path(tmp),
                execution_mode=TrainingExecutionMode.SUPERVISED,
                interrupted=False,
                completion_event=True,
            )

        self.assertEqual(
            [checkpoint["kind"] for checkpoint in checkpoints],
            ["checkpoint", "final"],
        )
        self.assertEqual(result.first_completion_step, 1)
        self.assertEqual(result.final_step, 2)
        self.assertEqual(result.terminal_reason, TerminalReason.RESOURCE_EXHAUSTION)

    def test_supervised_interruption_preserves_interrupted_and_final_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoints, _progress, result = self._run_backend(
                Path(tmp),
                execution_mode=TrainingExecutionMode.SUPERVISED,
                interrupted=True,
                completion_event=False,
            )

        self.assertEqual(
            [checkpoint["kind"] for checkpoint in checkpoints],
            ["interrupted", "final"],
        )
        self.assertEqual(result.terminal_reason, TerminalReason.EXTERNAL_SIGNAL)
        self.assertEqual(result.model_kind, "final")

    def test_local_run_stops_on_first_completion_without_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoints, progress, result = self._run_backend(
                Path(tmp),
                execution_mode=TrainingExecutionMode.LOCAL_DEMO,
                interrupted=False,
                completion_event=True,
            )

        self.assertEqual([checkpoint["kind"] for checkpoint in checkpoints], ["final"])
        self.assertEqual(checkpoints[0]["step"], 1)
        self.assertEqual(result.terminal_reason, TerminalReason.FIRST_COMPLETION)
        self.assertEqual(result.first_completion_step, 1)
        self.assertEqual(result.final_step, 1)
        self.assertEqual(progress.n, 1)
        self.assertEqual(
            progress.metrics[TRAIN_EPISODE_RETURN_SHAPED_FROM_TARGET_MEAN],
            10.0,
        )
        self.assertEqual(
            progress.metrics[TRAIN_OUTCOME_SUCCESS_CURRENT_RATE_MEAN],
            1.0,
        )

    @staticmethod
    def _run_backend(
        root: Path,
        *,
        execution_mode: TrainingExecutionMode,
        interrupted: bool,
        completion_event: bool,
    ) -> tuple[list[dict], SimpleNamespace, object]:
        stop_flag = GracefulStopFlag()

        class FakeRuntime:
            action_space = spaces.Discrete(2)

            def __init__(self) -> None:
                self.steps = 0

            def reset(self, *, seed: int) -> None:
                del seed

            def capture_archive_entries(self, mask, *, metadata_by_lane):
                del mask, metadata_by_lane
                return ("initial",)

            def step(self, actions):
                del actions
                self.steps += 1
                return SimpleNamespace(
                    rewards=np.asarray([0.0]),
                    terminated=np.asarray([False]),
                    truncated=np.asarray([False]),
                    transition_info={
                        "xscrollHi": np.asarray([0]),
                        "xscrollLo": np.asarray([0]),
                    },
                )

            def drain_records(self):
                if completion_event and self.steps == 1:
                    return (
                        EpisodeRecord(
                            lane=0,
                            episode_index=0,
                            start_id="Level1-1",
                            episode_return=10.0,
                            episode_length=1,
                            terminated=True,
                            truncated=False,
                            outcome=Outcome.SUCCESS,
                            events=("level_change",),
                            metrics={"level_complete": True},
                        ),
                    )
                return ()

            def state_archive_summary(self):
                return {}

            def close(self) -> None:
                pass

        class FakeSearch:
            def __init__(self) -> None:
                self.global_step = 0
                self.archive = {}
                self.archive_count = 1
                self.archive_selection_count = 0
                self.archive_visit_count = 0
                self.archive_update_count = 0
                self.archive_recent_new_cell_rate = 0.0
                self.archive_recent_visit_window = 0
                self.archive_visits_per_cell = 0.0
                self.success_guided_cell_count = 0
                self.success_guided_selection_count = 0
                self.improvement_count = 0

            def initialize(self, cell_keys, initial_entries) -> None:
                del cell_keys, initial_entries

            def next_actions(self):
                return np.asarray([0])

            def observe(self, rewards, dones, cell_keys, records, *, progresses):
                del rewards, dones, cell_keys, records, progresses
                self.global_step += 1
                if interrupted:
                    stop_flag.request("SIGINT")
                return SimpleNamespace(
                    archive_mask=np.asarray([False]),
                    restart_mask=np.asarray([False]),
                )

            def take_completion_events(self):
                return (SimpleNamespace(improved=True),) if completion_event else ()

            def best_candidate(self):
                return SimpleNamespace(
                    episode_return=10.0,
                    progress=20.0,
                    completed=completion_event,
                    step_count=1,
                    runs=(),
                )

            def policy(self):
                return SimpleNamespace(
                    save=lambda path, **_kwargs: Path(path).write_bytes(b"policy")
                )

        class FakeProgress:
            def __init__(self) -> None:
                self.n = 0
                self.metrics = {}
                self.closed = False

            def start(self, *, total: int, initial: int, description: str) -> None:
                del total, description
                self.n = initial

            def update(self, *, step: int, metrics, final: bool = False) -> None:
                del final
                self.n = step
                self.metrics = dict(metrics)

            def event(self, message: str) -> None:
                del message

            def close(self) -> None:
                self.closed = True

        class FakeMetricStore:
            def __init__(self) -> None:
                self.checkpoints: list[dict] = []
                self.payloads: list[tuple[dict, dict]] = []

            def record_checkpoint(self, **kwargs):
                self.checkpoints.append(dict(kwargs))
                return len(self.checkpoints)

            def append_metrics(self, payload, **kwargs):
                self.payloads.append((dict(payload), dict(kwargs)))

        runtime = FakeRuntime()
        search = FakeSearch()
        progress = FakeProgress()
        metric_store = FakeMetricStore()
        checkpoint_dir = root / "checkpoints"
        policy = TrainingExecutionPolicy.for_mode(execution_mode)
        session = TrainingSession(
            run_dir=root,
            backend_id="gradlab.go-explore",
            metric_store=metric_store,
            wandb_enabled=False,
            stop_flag=stop_flag,
            early_stop_config=None,
            attempt_id="attempt-test",
            reducer=EpisodeMetricsReducer(
                configured_starts=("Level1-1",),
                track_success=True,
            ),
            execution_policy=policy,
            completion_signal_available=True,
            progress_sink=progress,
        )
        session.configure_checkpoints(run_name="run-test", eval_required=False)
        context = BackendContext(
            train_config={
                "attempt_id": "attempt-test",
                "checkpoint_eval_backend": "none",
                "checkpoint_freq": 1,
                "early_stop": None,
                "resolved_n_envs": 1,
                "run_name": "run-test",
                "seed": 123,
                "state_archive": {"persistence": "ephemeral"},
                "timesteps": 2,
                "training_backend": {
                    "id": "gradlab.go-explore",
                    "config": {
                        "compaction_interval_steps": 100,
                        "explore_steps": 1,
                        "fallback_action": "noop",
                        "log_interval_steps": 100,
                        "run_duration_max": 1,
                        "run_duration_mean": 1.0,
                    },
                },
            },
            environment=SimpleNamespace(game="SuperMarioBros-Nes-v0"),
            run_dir=root,
            checkpoint_dir=checkpoint_dir,
            metric_store=mock.Mock(),
            wandb_enabled=False,
            stop_flag=stop_flag,
            rom_binding=None,
            session=session,
        )

        def install_test_bundle(path, *, save_checkpoint, **_kwargs):
            save_checkpoint(path)
            return path

        with (
            mock.patch(
                "gradlab.training.go_explore._runtime_environment_config",
                return_value=context.environment,
            ),
            mock.patch(
                "gradlab.training.go_explore.preflight_state_archive_provider",
                return_value={"status": "passed"},
            ),
            mock.patch(
                "gradlab.training.go_explore.make_training_batch_runtime",
                return_value=runtime,
            ),
            mock.patch(
                "gradlab.training.go_explore.GoExploreSearch",
                return_value=search,
            ),
            mock.patch(
                "gradlab.training.go_explore.configured_action_meanings",
                return_value=("noop", "right"),
            ),
            mock.patch(
                "gradlab.training.go_explore._reset_info_columns",
                return_value={},
            ),
            mock.patch(
                "gradlab.training.go_explore._cell_keys",
                return_value=(b"cell",),
            ),
            mock.patch(
                "gradlab.training.go_explore.install_model_bundle",
                side_effect=install_test_bundle,
            ),
        ):
            result = run_go_explore(context)
        session.finalize(result)

        return (
            metric_store.checkpoints,
            SimpleNamespace(
                n=progress.n,
                metrics=progress.metrics,
                closed=progress.closed,
            ),
            result,
        )


if __name__ == "__main__":
    unittest.main()
