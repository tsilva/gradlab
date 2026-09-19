"""Collect calibration evidence from durable journals and terminal inventories."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gradlab.json_utils import canonical_json_sha256
from gradlab.monitor_campaign import campaign_plan, collect_campaign
from gradlab.monitor_calibration import PHASES
from gradlab.r2_store import BucketConfig, R2Bucket
from gradlab.recipe_documents import compose_train_document
from gradlab.run_contracts import TerminalReceipt


@pytest.mark.parametrize(
    "slow_interval,inference_at,expected",
    [(False, 11.5, "supported"), (True, 11.5, "unproven"), (False, 21, "incomplete")],
)
def test_campaign_uses_elapsed_training_time_and_requires_concurrent_capture(
    tmp_path, slow_interval, inference_at, expected
):
    campaign = dict(
        launch_args=["--recipe-file", "recipe.yaml", "--max-duration", "1h"],
        seeds=[123, 234, 345],
        settings={},
        warmup_updates=1,
        representatives={
            role: dict(seed=123, checkpoint_step=step)
            for role, step in [
                ("early", 100),
                ("intermediate", 200),
                ("stronger", 300),
                ("long-episode", 300),
            ]
        },
    )
    identity, settings, commands = campaign_plan(campaign)
    goal = Path("experiments/goals/Breakout-Atari2600-v0/FirstWall")
    train = compose_train_document(goal / "_goal.yaml", goal / "recipes/ppo.yaml")["train_config"]
    train.update(timesteps=400, checkpoint_freq=100)
    control = R2Bucket(BucketConfig(uri=f"file://{tmp_path}/control"))
    models = R2Bucket(BucketConfig(uri=f"file://{tmp_path}/models"))
    manifests, recipes, runs = {}, {}, {}
    for index, command in enumerate(commands):
        run_id = f"gradlab-{index:032x}"
        attempt = "attempt-" + "a" * 16
        configured = deepcopy(train)
        configured["checkpoint_monitoring"] = {
            **settings,
            "enabled": command["enabled"],
            "calibration": dict(status="measuring", campaign_id=identity),
        }
        recipe_hash = canonical_json_sha256(configured)
        recipes[recipe_hash] = dict(recipe=dict(train_config=configured))
        manifests[run_id] = dict(
            run_id=run_id,
            attempt_id=attempt,
            seed=command["seed"],
            recipe_sha256=recipe_hash,
            source_sha="b" * 40,
            image_digest="sha256:" + "c" * 64,
            created_at="1970-01-01T00:00:00Z",
            compute=dict(
                submission_key=command["key"], resources=dict(cpu=1), selected=dict(cpu=1)
            ),
        )
        runs[command["key"]] = dict(run_id=run_id)
        events = [
            dict(
                event_seq=i + 1,
                event_id=f"event-{i}",
                kind="history",
                step=(i + 1) * 100,
                created_at=[10, 11, 12, 14 if slow_interval and command["enabled"] else 13][i],
                payload={
                    "train/throughput/rate": 50
                    if i == 3 and slow_interval and command["enabled"]
                    else 100
                },
            )
            for i in range(4)
        ]
        encoded = b"\n".join(json.dumps(e).encode() for e in events)
        control.put_bytes(
            f"expiring-metric-journals/{run_id}/1-4-{hashlib.sha256(encoded).hexdigest()}.jsonl",
            encoded,
        )
        inventory = []
        if command["enabled"]:
            for step in (100, 200, 300):
                result = dict(
                    episodes=[
                        dict(checkpoint_sha256=str(step), first_inference_at=inference_at)
                        for _ in range(400)
                    ],
                    metrics={"eval/monitor/progress/mean": step / 400},
                    measurements=dict(
                        uninterrupted=True,
                        seconds=0.1,
                        retained_bytes=1000,
                        peak_memory_bytes=1024**2,
                        peak_spool_bytes=1024**2,
                        longest_episode_steps=9000,
                        phase_seconds={phase: 0.01 for phase in PHASES},
                    ),
                )
                key = f"{run_id}/{step}/result.json"
                models.put_json(key, result)
                inventory.append(
                    dict(
                        checkpoint_step=step,
                        status="complete",
                        result_key=key,
                        result_sha256=canonical_json_sha256(result),
                        wandb_media_delivery_seconds=0.01,
                    )
                )
        terminal = TerminalReceipt(
            run_id=run_id,
            attempt_id=attempt,
            state="succeeded",
            acceptance_required=False,
            stop_reason="training_completed",
            final_step=400,
            checkpoint_inventory=[],
            eval_inventory=[],
            wandb_high_water_mark=4,
            completed_at="1970-01-01T00:00:30Z",
            drain=dict(
                complete=True,
                wandb_remote_high_water_mark=4,
                wandb_final_drain_seconds=0.01,
                calibration_host_load=dict(count=3, sum=3),
                checkpoint_monitoring=dict(inventory=inventory),
            ),
        )
        control.put_json(f"runs/{run_id}/attempts/{attempt}/terminal.json", asdict(terminal))
    authority = SimpleNamespace(
        control=control,
        models=models,
        manifest=manifests.__getitem__,
        recipe_document=recipes.__getitem__,
    )
    report = collect_campaign(campaign, runs, authority)
    assert report["status"] == expected
    if slow_interval:
        # Median(100, 100, 50) is 100; actual 300 / (1 + 1 + 2) is 75.
        assert report["maximum_throughput_loss_upper95"] == pytest.approx(0.25)
    if inference_at == 21:
        assert "capture" in report["reason"]
