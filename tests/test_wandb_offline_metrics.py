from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import wandb

from gradlab.metric_names import LEADER_CHECKPOINT_ARTIFACT_REF
from gradlab.metric_store import MetricStore
from gradlab.wandb_publisher import (
    _publish_frame,
    publish_pending_frames,
    publish_promotion_summary,
)
from gradlab.wandb_utils import configure_wandb_metrics


class WandbOfflineMetricIntegrationTests(unittest.TestCase):
    def test_configures_scientific_axes_without_overlapping_catchall(self) -> None:
        class FakeRun:
            def __init__(self) -> None:
                self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

            def define_metric(self, *args, **kwargs) -> None:
                self.calls.append((args, kwargs))

        run = FakeRun()

        self.assertIs(configure_wandb_metrics(run), run)
        self.assertIn(
            (("train/*",), {"step_metric": "train/global_step"}),
            run.calls,
        )
        self.assertIn(
            (("eval/*",), {"step_metric": "eval/checkpoint_step"}),
            run.calls,
        )
        self.assertIn(
            (("orchestration/*",), {"step_metric": "orchestration/event_seq"}),
            run.calls,
        )
        self.assertNotIn("*", {str(args[0]) for args, _kwargs in run.calls})

    def test_replayed_outbox_event_reuses_the_same_wandb_step(self) -> None:
        class FakeRun:
            def __init__(self) -> None:
                self.calls: list[tuple[dict, int]] = []
                self.metric_calls: list[
                    tuple[tuple[object, ...], dict[str, object]]
                ] = []

            def log(self, payload, *, step):
                self.calls.append((dict(payload), int(step)))

            def define_metric(self, *args, **kwargs) -> None:
                self.metric_calls.append((args, kwargs))

        with tempfile.TemporaryDirectory() as tmp:
            store = MetricStore(Path(tmp) / "gradlab.sqlite")
            store.init()
            store.append_metrics(
                {
                    "train/episode/return/shaped/from/target/mean": 5.0,
                    "train/early_stop/return_plateau/patience/progress": 0.25,
                    "train/early_stop/return_plateau/would_trigger": 0.0,
                },
                step=400_000,
                source="train:rollout",
            )
            row = store.pending_metric_frames(limit=1)[0]
            run = FakeRun()

            _publish_frame(run, row)
            _publish_frame(run, row)

        self.assertEqual([step for _payload, step in run.calls], [row["id"], row["id"]])
        self.assertEqual(
            {payload["orchestration/event_id"] for payload, _step in run.calls},
            {row["event_id"]},
        )
        self.assertEqual(
            {payload["orchestration/event_seq"] for payload, _step in run.calls},
            {row["id"]},
        )
        self.assertEqual(
            {payload["train/global_step"] for payload, _step in run.calls},
            {400_000},
        )
        for metric_name in (
            "train/episode/return/shaped/from/target/mean",
            "train/early_stop/return_plateau/patience/progress",
            "train/early_stop/return_plateau/would_trigger",
        ):
            definition = (
                (metric_name,),
                {"step_metric": "train/global_step"},
            )
            self.assertIn(definition, run.metric_calls)
            self.assertEqual(
                run.metric_calls.count(definition),
                1,
            )

    def test_supervisor_publishes_eval_metrics_table_and_promotion_without_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MetricStore(Path(tmp) / "gradlab.sqlite")
            store.init()
            store.append_metrics(
                {
                    "eval/full/episode/return/mean": 5.0,
                    "eval/full/episode/return/best": 5.0,
                    "eval/full/outcome/success/rate/min": 1.0,
                    "eval/full/outcome/success/rate/mean": 1.0,
                },
                step=100,
                source="eval:modal",
            )
            store.enqueue_event(
                kind="eval_by_start",
                payload={
                    "rows": [
                        ["Start", 1, 1, 1.0, 5.0, 0.0, 5.0, "", 0, 0.0]
                    ]
                },
                step=100,
                source="eval:modal",
            )
            run = configure_wandb_metrics(
                wandb.init(
                    project="gradlab-metrics-schema-test",
                    dir=tmp,
                    mode="offline",
                    reinit="finish_previous",
                    settings=wandb.Settings(silent=True, disable_git=True),
                )
            )
            assert run is not None
            published = publish_pending_frames(
                store,
                run,
                limit=10,
            )
            publish_promotion_summary(
                run,
                checkpoint_step=100,
                checkpoint_url="https://models.example/model.zip",
                metrics={
                    "eval/full/episode/return/mean": 5.0,
                    "eval/full/episode/return/best": 5.0,
                    "eval/full/outcome/success/rate/min": 1.0,
                    "eval/full/outcome/success/rate/mean": 1.0,
                },
                updated_at="2026-07-24T00:00:00Z",
            )

            self.assertEqual(published, 2)
            self.assertEqual(
                run.summary[LEADER_CHECKPOINT_ARTIFACT_REF],
                "https://models.example/model.zip",
            )
            self.assertEqual(store.metric_outbox_stats()["frames"], 0)
            run.finish()


if __name__ == "__main__":
    unittest.main()
