from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import wandb
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore

from gradlab.metric_names import (
    EVAL_ACCEPTANCE_PASS,
    EVAL_CHECKPOINT_STEP,
    LEADER_CHECKPOINT_ARTIFACT_REF,
    ORCHESTRATION_EVENT_SEQ,
    ORCHESTRATION_QUEUE_DEPTH,
    TRAIN_EPISODE_RETURN_SHAPED_ACROSS_ORIGINS_ROLLING_UP_TO_100_MEAN,
    TRAIN_GLOBAL_STEP,
    TRAIN_OUTCOME_SUCCESS_ACROSS_STARTS_WINDOW_100_RATE_MEAN,
)
from gradlab.metric_store import MetricStore
from gradlab.run_contracts import TerminalReceipt, new_attempt_id, new_run_id, utc_now
from gradlab.wandb_publisher import (
    _publish_frame,
    publish_pending_frames,
    publish_promotion_summary,
    publish_terminal_summary,
)
from gradlab.wandb_utils import configure_wandb_metrics


def _offline_wandb_records(root: Path):
    transactions = list(root.rglob("*.wandb"))
    if len(transactions) != 1:
        raise AssertionError(f"expected one W&B transaction, found {transactions!r}")
    store = DataStore()
    store.open_for_scan(str(transactions[0]))
    while data := store.scan_data():
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        yield record


def _history_payload(record) -> dict[str, object]:
    return {
        "/".join(item.nested_key) or item.key: item.value_json
        for item in record.history.item
    }


class WandbOfflineMetricIntegrationTests(unittest.TestCase):
    def test_terminal_receipt_is_projected_into_summary_metadata(self) -> None:
        class FakeRun:
            summary: dict[str, object] = {}

        receipt = TerminalReceipt(
            run_id=new_run_id(),
            attempt_id=new_attempt_id(),
            state="failed",
            acceptance_required=False,
            stop_reason="early_stop_failure:return_plateau",
            final_step=500_000,
            checkpoint_inventory=(),
            eval_inventory=(),
            wandb_high_water_mark=10,
            drain={"complete": True},
            completed_at=utc_now(),
        )
        run = FakeRun()

        publish_terminal_summary(run, receipt)

        self.assertEqual(run.summary["gradlab/run/terminal_state"], "failed")
        self.assertEqual(
            run.summary["gradlab/run/stop_reason"],
            "early_stop_failure:return_plateau",
        )
        self.assertEqual(run.summary["gradlab/run/final_step"], 500_000)

    def test_configures_scientific_axes_without_overlapping_catchall(self) -> None:
        class FakeRun:
            def __init__(self) -> None:
                self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

            def define_metric(self, *args, **kwargs) -> None:
                self.calls.append((args, kwargs))

        run = FakeRun()

        self.assertIs(configure_wandb_metrics(run), run)
        self.assertIn(
            ((TRAIN_GLOBAL_STEP,), {"summary": "max"}),
            run.calls,
        )
        self.assertIn(
            ((EVAL_CHECKPOINT_STEP,), {"summary": "max"}),
            run.calls,
        )
        self.assertIn(
            ((ORCHESTRATION_EVENT_SEQ,), {"summary": "max"}),
            run.calls,
        )
        self.assertIn(
            (
                (EVAL_ACCEPTANCE_PASS,),
                {"step_metric": EVAL_CHECKPOINT_STEP, "summary": "max"},
            ),
            run.calls,
        )
        self.assertIn(
            (
                (
                    TRAIN_EPISODE_RETURN_SHAPED_ACROSS_ORIGINS_ROLLING_UP_TO_100_MEAN,
                ),
                {"step_metric": TRAIN_GLOBAL_STEP, "summary": "last"},
            ),
            run.calls,
        )
        self.assertFalse(
            any("*" in str(args[0]) for args, _kwargs in run.calls),
            run.calls,
        )

    def test_real_sdk_serializes_only_concrete_scientific_axis_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = MetricStore(root / "gradlab.sqlite")
            store.init()
            store.append_metrics(
                {
                    TRAIN_OUTCOME_SUCCESS_ACROSS_STARTS_WINDOW_100_RATE_MEAN: 0.55,
                },
                step=8192,
                source="train:rollout",
            )
            store.append_metrics(
                {EVAL_ACCEPTANCE_PASS: 1.0},
                step=4096,
                source="eval:modal",
            )
            store.append_metrics(
                {ORCHESTRATION_QUEUE_DEPTH: 2.0},
                step=0,
                source="orchestration:supervisor",
            )
            run = configure_wandb_metrics(
                wandb.init(
                    project="gradlab-metrics-axis-test",
                    dir=tmp,
                    mode="offline",
                    reinit="finish_previous",
                    settings=wandb.Settings(
                        silent=True,
                        disable_git=True,
                        x_server_side_expand_glob_metrics=False,
                    ),
                )
            )
            assert run is not None

            published = publish_pending_frames(store, run, limit=10)
            run.finish()

            self.assertEqual(published, 3)
            self.assertEqual(store.metric_outbox_stats()["frames"], 0)
            records = list(_offline_wandb_records(root))

        metric_records = [
            record.metric for record in records if record.WhichOneof("record_type") == "metric"
        ]
        self.assertFalse(
            any(metric.glob_name or "*" in metric.name for metric in metric_records),
            metric_records,
        )
        bindings = {
            (metric.name, metric.step_metric)
            for metric in metric_records
            if metric.name and metric.step_metric
        }
        self.assertIn(
            (
                TRAIN_OUTCOME_SUCCESS_ACROSS_STARTS_WINDOW_100_RATE_MEAN,
                TRAIN_GLOBAL_STEP,
            ),
            bindings,
        )
        self.assertIn((EVAL_ACCEPTANCE_PASS, EVAL_CHECKPOINT_STEP), bindings)
        self.assertIn((ORCHESTRATION_QUEUE_DEPTH, ORCHESTRATION_EVENT_SEQ), bindings)

        history = [
            _history_payload(record)
            for record in records
            if record.WhichOneof("record_type") == "history"
        ]
        train_row = next(
            row
            for row in history
            if TRAIN_OUTCOME_SUCCESS_ACROSS_STARTS_WINDOW_100_RATE_MEAN in row
        )
        eval_row = next(row for row in history if EVAL_ACCEPTANCE_PASS in row)
        orchestration_row = next(row for row in history if ORCHESTRATION_QUEUE_DEPTH in row)
        self.assertEqual(train_row[TRAIN_GLOBAL_STEP], "8192")
        self.assertEqual(eval_row[EVAL_CHECKPOINT_STEP], "4096")
        self.assertNotEqual(train_row["_step"], train_row[TRAIN_GLOBAL_STEP])
        self.assertNotEqual(eval_row["_step"], eval_row[EVAL_CHECKPOINT_STEP])
        self.assertEqual(
            orchestration_row["_step"],
            orchestration_row[ORCHESTRATION_EVENT_SEQ],
        )

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
                    "train/episode/return/shaped/from/target/rolling_up_to_100/mean": 5.0,
                    "train/early_stop/return_plateau/patience/progress": 0.25,
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
            "train/episode/return/shaped/from/target/rolling_up_to_100/mean",
            "train/early_stop/return_plateau/patience/progress",
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
                    "eval/full/episode/return/shaped/mean": 5.0,
                    "eval/full/episode/return/shaped/max": 5.0,
                    "eval/full/outcome/success/across_starts/rate/min": 1.0,
                    "eval/full/outcome/success/across_starts/rate/mean": 1.0,
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
                    "eval/full/episode/return/shaped/mean": 5.0,
                    "eval/full/episode/return/shaped/max": 5.0,
                    "eval/full/outcome/success/across_starts/rate/min": 1.0,
                    "eval/full/outcome/success/across_starts/rate/mean": 1.0,
                },
                updated_at="2026-07-24T00:00:00Z",
                selection_rank=[
                    "max(eval/full/outcome/success/across_starts/rate/min)",
                    "max(eval/full/episode/return/shaped/mean)",
                    "min(leader/checkpoint/step)",
                ],
                evaluation_source="modal:test",
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
