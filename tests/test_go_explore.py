from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from gymnasium import spaces

from gradlab.action_program import ActionProgramPolicy
from gradlab.env import EnvConfig
from gradlab.go_explore import GoExploreSearch
from gradlab.training.go_explore import (
    GO_EXPLORE_PROVIDER_INFO_KEYS,
    _runtime_environment_config,
    normalize_config,
    run_go_explore,
)
from gradlab.training_backend import BackendContext, GracefulStopFlag


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
                save_policy, progress = self._run_backend(
                    Path(tmp),
                    persist_intermediate_checkpoints=False,
                    interrupted=interrupted,
                )

                save_policy.assert_called_once()
                call = save_policy.call_args
                self.assertEqual(call.kwargs["model_path"], Path(tmp) / "final_model.zip")
                self.assertEqual(
                    call.kwargs["kind"],
                    "interrupted" if interrupted else "final",
                )
                self.assertEqual(progress.n, 1 if interrupted else 2)
                self.assertEqual(progress.postfix["completed"], "no")

    def test_queued_run_preserves_intermediate_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            save_policy, _progress = self._run_backend(
                Path(tmp),
                persist_intermediate_checkpoints=True,
                interrupted=False,
            )

        self.assertEqual(
            [call.kwargs["kind"] for call in save_policy.call_args_list],
            ["checkpoint", "checkpoint", "final"],
        )

    @staticmethod
    def _run_backend(
        root: Path,
        *,
        persist_intermediate_checkpoints: bool,
        interrupted: bool,
    ) -> tuple[mock.Mock, SimpleNamespace]:
        stop_flag = GracefulStopFlag()

        class FakeRuntime:
            action_space = spaces.Discrete(2)

            def reset(self, *, seed: int) -> None:
                del seed

            def capture_archive_entries(self, mask, *, metadata_by_lane):
                del mask, metadata_by_lane
                return ("initial",)

            def step(self, actions):
                del actions
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
                return ()

            def close(self) -> None:
                pass

        class FakeSearch:
            def __init__(self) -> None:
                self.global_step = 0
                self.archive = {}

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
                return (SimpleNamespace(improved=True),)

            def best_candidate(self):
                return SimpleNamespace(
                    episode_return=10.0,
                    progress=20.0,
                    completed=False,
                )

        class FakeProgress:
            def __init__(self) -> None:
                self.n = 0
                self.postfix = {}
                self.closed = False

            def update(self, value: int) -> None:
                self.n += value

            def set_postfix(self, values, *, refresh: bool) -> None:
                del refresh
                self.postfix = dict(values)

            def close(self) -> None:
                self.closed = True

        runtime = FakeRuntime()
        search = FakeSearch()
        progress = FakeProgress()
        checkpoint_dir = root / "checkpoints"
        checkpoint_dir.mkdir()
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
            compact_console=True,
            persist_intermediate_checkpoints=persist_intermediate_checkpoints,
        )
        save_policy = mock.Mock(return_value=root / "model.zip")

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
                "gradlab.training.go_explore._publish_metrics",
                return_value=False,
            ),
            mock.patch(
                "gradlab.training.go_explore._save_policy",
                save_policy,
            ),
            mock.patch(
                "gradlab.training.go_explore.tqdm",
                return_value=progress,
            ),
        ):
            run_go_explore(context)

        return save_policy, SimpleNamespace(
            n=progress.n,
            postfix=progress.postfix,
            closed=progress.closed,
        )


if __name__ == "__main__":
    unittest.main()
