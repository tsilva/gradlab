"""Explicit, resumable matched calibration campaigns through normal supervised Runs.

This operation launches training only when explicitly invoked with a campaign.
All metrics, media, terminal receipts and credentials keep their ordinary owners.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import math
import subprocess
import sys
import time

from gradlab.file_utils import atomic_write_json
from gradlab.json_utils import canonical_json_sha256
from gradlab.monitor_config import MonitoringConfig, calibration_binding
from gradlab.monitor_calibration import assess_calibration
from gradlab.r2_store import RunStorageConfig
from gradlab.run_authority import RunAuthority
from gradlab.run_contracts import TerminalReceipt
from gradlab.seeds import validate_training_seed


def campaign_plan(campaign):
    if set(campaign) != {"launch_args", "seeds", "settings", "warmup_updates", "representatives"}:
        raise ValueError(
            "campaign requires launch_args, seeds, settings, warmup_updates and representatives"
        )
    seeds = campaign["seeds"]
    if not isinstance(seeds, list) or not 3 <= len(seeds) <= 20 or len(set(seeds)) != len(seeds):
        raise ValueError("calibration requires 3 to 20 distinct matched training seeds")
    for seed in seeds:
        validate_training_seed(seed)
    if type(campaign["warmup_updates"]) is not int or campaign["warmup_updates"] < 1:
        raise ValueError("calibration requires a positive warm-up exclusion")
    arguments = campaign["launch_args"]
    if not isinstance(arguments, list) or not all(isinstance(a, str) for a in arguments):
        raise ValueError("launch_args must be an argument list, never a shell command")
    prohibited = (
        "--seed",
        "--follow",
        "--json",
        "--submission-key",
        "--run-description",
        "--monitor-calibration-id",
    )
    if any(a.split("=", 1)[0] in prohibited for a in arguments) or any(
        "checkpoint_monitoring" in a for a in arguments
    ):
        raise ValueError(
            "campaign owns seeds, monitoring settings, submission identity and output mode"
        )
    if "--recipe-file" not in arguments or "--max-duration" not in arguments:
        raise ValueError("campaign must explicitly select a recipe and finite per-Run duration")
    settings = {**asdict(MonitoringConfig()), **campaign["settings"]}
    if set(settings) != set(MonitoringConfig.__dataclass_fields__):
        raise ValueError("unknown campaign monitoring settings")
    if set(campaign["representatives"]) != {"early", "intermediate", "stronger", "long-episode"}:
        raise ValueError("declare representative role selectors before measuring")
    for selector in campaign["representatives"].values():
        if (
            set(selector) != {"seed", "checkpoint_step"}
            or selector["seed"] not in seeds
            or type(selector["checkpoint_step"]) is not int
            or selector["checkpoint_step"] < 0
        ):
            raise ValueError("representatives require a campaign seed and checkpoint_step")
    identity = canonical_json_sha256(campaign)
    commands = []
    for index, seed in enumerate(seeds):
        # Counterbalance order so background host drift does not always favor off.
        for enabled in (False, True) if index % 2 == 0 else (True, False):
            mode = "on" if enabled else "off"
            setting = {
                **settings,
                "enabled": enabled,
                "calibration": {"status": "measuring", "campaign_id": identity},
            }
            key = f"{identity}:{seed}:{mode}"
            argv = [
                sys.executable,
                "-m",
                "gradlab.main",
                "experiment",
                "launch",
                *arguments,
                "--seed",
                str(seed),
                "--submission-key",
                key,
                "--run-description",
                f"Checkpoint monitoring calibration {identity[:12]} seed {seed} {mode}",
                "--monitor-calibration-id",
                identity,
                "--json",
                "--set",
                "train.checkpoint_monitoring=" + json.dumps(setting, separators=(",", ":")),
            ]
            commands.append(dict(key=key, seed=seed, enabled=enabled, argv=argv))
    return identity, settings, commands


def _events(authority, run_id):
    keys = set(authority.control.iter_keys(f"expiring-metric-journals/{run_id}/"))
    keys.update(
        k
        for k in authority.control.iter_keys(f"runs/{run_id}/attempts/")
        if "/metric-segments/" in k
    )
    events = {}
    for key in sorted(keys):
        if not key.endswith(".jsonl"):
            continue
        data = authority.control.get_bytes(key)
        if hashlib.sha256(data).hexdigest() != key.removesuffix(".jsonl").rsplit("-", 1)[-1]:
            raise ValueError("calibration metric journal checksum mismatch")
        for line in data.splitlines():
            event = json.loads(line)
            old = events.setdefault(event["event_id"], event)
            if old != event:
                raise ValueError("conflicting calibration metric event")
    return sorted(events.values(), key=lambda e: e["event_seq"])


def _training_measurement(events, warmup):
    updates = [
        e for e in events if e["kind"] == "history" and "train/throughput/rate" in e["payload"]
    ]
    intervals = []
    previous_step = 0
    for index, event in enumerate(updates):
        step = event["step"]
        rate = event["payload"]["train/throughput/rate"]
        if step is None or step <= previous_step or not math.isfinite(rate) or rate <= 0:
            raise ValueError(
                "calibration needs finite rates and strictly increasing training steps"
            )
        delta = step - previous_step
        previous_step = step
        if index >= warmup:
            duration = delta / rate
            end = event["created_at"]
            intervals.append(dict(steps=delta, seconds=duration, start=end - duration, end=end))
    if len(intervals) < 3:
        return None
    return dict(
        rate=sum(i["steps"] for i in intervals) / sum(i["seconds"] for i in intervals),
        intervals=intervals,
    )


def _concurrent_capture(authority, inventory, intervals):
    for reference in inventory:
        if reference["status"] != "complete":
            continue
        result = authority.models.get_json(reference["result_key"])
        if canonical_json_sha256(result) != reference["result_sha256"]:
            raise ValueError("calibration result checksum mismatch")
        if not (result.get("measurements") or {}).get("uninterrupted"):
            continue
        for episode in result["episodes"]:
            timestamp = episode.get("first_inference_at")
            if timestamp is not None and any(
                i["start"] <= timestamp <= i["end"] for i in intervals
            ):
                return True
    return False


def collect_campaign(campaign, runs, authority):
    identity, settings, commands = campaign_plan(campaign)
    measurements = dict(samples=[], pairs=[])
    resolved = {}
    binding = None
    for command in commands:
        record = runs.get(command["key"])
        if not record or not record.get("run_id"):
            return assess_calibration(measurements)
        run_id = record["run_id"]
        manifest = authority.manifest(run_id)
        if (
            manifest["compute"]["submission_key"] != command["key"]
            or manifest["seed"] != command["seed"]
        ):
            raise ValueError("calibration Run differs from its campaign assignment")
        terminal_doc = authority.control.get_json_optional(
            f"runs/{run_id}/attempts/{manifest['attempt_id']}/terminal.json"
        )
        if terminal_doc is None:
            return assess_calibration(measurements)
        terminal = TerminalReceipt.from_dict(terminal_doc)
        if (
            not terminal.drain.get("complete")
            or terminal.state in {"failed", "canceled"}
            or terminal.wandb_high_water_mark <= 0
            or terminal.drain.get("wandb_remote_high_water_mark", 0)
            < terminal.wandb_high_water_mark
        ):
            return dict(
                status="incomplete", reason=f"calibration Run {run_id} lacks complete delivery"
            )
        train = authority.recipe_document(manifest["recipe_sha256"])["recipe"]["train_config"]
        configured = train.get("checkpoint_monitoring") or {}
        if (
            configured.get("calibration") != {"status": "measuring", "campaign_id": identity}
            or configured.get("enabled") != command["enabled"]
        ):
            raise ValueError("calibration Run monitoring contract differs from its campaign")
        actual = (
            calibration_binding(train, settings),
            manifest["source_sha"],
            manifest["image_digest"],
            {
                "resources": manifest["compute"]["resources"],
                "selected": manifest["compute"]["selected"],
                "offer": manifest["compute"].get("selected_offer"),
            },
        )
        if binding is not None and actual != binding:
            return dict(
                status="incomplete",
                reason="paired Runs have different workload, source, runtime or selected allocation",
            )
        binding = actual
        measured = _training_measurement(_events(authority, run_id), campaign["warmup_updates"])
        if measured is None:
            return dict(
                status="incomplete", reason="insufficient throughput observations after warm-up"
            )
        resolved[(command["seed"], command["enabled"])] = dict(
            manifest=manifest,
            terminal=terminal_doc,
            **measured,
            train=train,
        )
    for seed in campaign["seeds"]:
        off, on = resolved[(seed, False)], resolved[(seed, True)]
        loads = [r["terminal"]["drain"].get("calibration_host_load") or {} for r in (off, on)]
        comparable = all(load.get("count", 0) >= 3 for load in loads)
        if comparable:
            difference = abs(
                loads[0]["sum"] / loads[0]["count"] - loads[1]["sum"] / loads[1]["count"]
            )
            comparable = difference <= settings["active_workers"] + max(
                1, settings["task_cpus"] * 0.1
            )
        inventory = on["terminal"]["drain"]["checkpoint_monitoring"]["inventory"]
        measurements["pairs"].append(
            dict(
                seed=seed,
                off_rate=off["rate"],
                on_rate=on["rate"],
                warmup_excluded=True,
                comparable_host_load=comparable,
                checkpoint_freq=on["train"]["checkpoint_freq"],
                equivalent_workload=off["terminal"]["final_step"] == on["terminal"]["final_step"],
                nonzero_capture=_concurrent_capture(authority, inventory, on["intervals"]),
                off_run=off["manifest"]["run_id"],
                on_run=on["manifest"]["run_id"],
            )
        )
    for role, selector in campaign["representatives"].items():
        run = resolved[(selector["seed"], True)]
        inventory = run["terminal"]["drain"]["checkpoint_monitoring"]["inventory"]
        selected = [
            row for row in inventory if row["checkpoint_step"] == selector["checkpoint_step"]
        ]
        if len(selected) != 1 or selected[0]["status"] != "complete":
            return dict(
                status="incomplete", reason=f"{role} representative Checkpoint was not completed"
            )
        reference = selected[0]
        result = authority.models.get_json(reference["result_key"])
        if canonical_json_sha256(result) != reference["result_sha256"]:
            raise ValueError("calibration result checksum mismatch")
        sample = result.get("measurements")
        if not sample or not sample.get("uninterrupted"):
            return dict(
                status="incomplete", reason=f"{role} has no uninterrupted pipeline measurements"
            )
        sample = {**sample, "phase_seconds": dict(sample["phase_seconds"])}
        sample["phase_seconds"]["wandb_media_delivery"] = (
            reference["wandb_media_delivery_seconds"]
            + run["terminal"]["drain"]["wandb_final_drain_seconds"]
        )
        sample["seconds"] += (
            reference["wandb_media_delivery_seconds"]
            + run["terminal"]["drain"]["wandb_final_drain_seconds"]
        )
        sample.update(
            role=role,
            episodes_completed=len(result["episodes"]),
            verified_terminal_receipt=True,
            checkpoint_sha256=result["episodes"][0]["checkpoint_sha256"],
            normalized_progress=result["metrics"]["eval/progress/bricks_destroyed_normalized/mean"],
        )
        measurements["samples"].append(sample)
    by_role = {s["role"]: s for s in measurements["samples"]}
    if (
        by_role["long-episode"]["longest_episode_steps"] <= 8192
        or by_role["stronger"]["normalized_progress"] <= by_role["early"]["normalized_progress"]
    ):
        return dict(
            status="incomplete",
            reason="representatives do not establish stronger and genuinely long-episode behavior",
        )
    measurements.update(
        train_config=next(iter(resolved.values()))["train"],
        settings=settings,
        observed_total_seconds=max(
            (
                datetime.fromisoformat(r["terminal"]["completed_at"].replace("Z", "+00:00"))
                - datetime.fromisoformat(r["manifest"]["created_at"].replace("Z", "+00:00"))
            ).total_seconds()
            for r in resolved.values()
        ),
        source_sha=binding[1],
        runtime_image=binding[2],
        hardware_allocation=binding[3],
    )
    return assess_calibration(measurements)


def run_campaign(campaign, *, root):
    identity, _settings, commands = campaign_plan(campaign)
    root = Path(root).expanduser().resolve() / identity
    root.mkdir(parents=True, exist_ok=True)
    with (root / "campaign.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("calibration campaign is already running") from None
        return _execute_campaign(campaign, root, commands)


def _execute_campaign(campaign, root, commands):
    from gradlab.operator_environment import load_repository_operator_environment

    load_repository_operator_environment(Path.cwd())
    authority = RunAuthority(RunStorageConfig.from_env())
    ledger = root / "campaign.json"
    state = json.loads(ledger.read_text()) if ledger.exists() else dict(campaign=campaign, runs={})
    if state["campaign"] != campaign:
        raise ValueError("calibration campaign identity conflict")
    for command in commands:
        record = state["runs"].get(command["key"])
        if record and not record.get("run_id"):
            # Submission may have succeeded before its response reached this process.
            matches = []
            for key in authority.control.iter_keys("runs/"):
                if key.endswith("/manifest.json") and "/attempts/" not in key:
                    manifest = authority.control.get_json(key)
                    if manifest.get("compute", {}).get("submission_key") == command["key"]:
                        matches.append(manifest)
            if len(matches) != 1:
                raise RuntimeError(
                    "ambiguous calibration launch: inspect campaign.json before any resubmission"
                )
            record.update(run_id=matches[0]["run_id"])
        if not record:
            state["runs"][command["key"]] = dict(status="submitting")
            atomic_write_json(ledger, state)
            completed = subprocess.run(
                command["argv"], capture_output=True, text=True, timeout=3600
            )
            (root / f"{command['seed']}-{'on' if command['enabled'] else 'off'}.log").write_text(
                completed.stdout + completed.stderr
            )
            if completed.returncode:
                raise RuntimeError(
                    "calibration launch did not confirm submission; inspect the saved launch log"
                )
            output = json.loads(completed.stdout)
            record = state["runs"][command["key"]] = dict(run_id=output["run_id"], status="running")
            print(
                json.dumps(dict(run_id=record["run_id"], wandb=output.get("wandb_url"))), flush=True
            )
        atomic_write_json(ledger, state)
        manifest = authority.manifest(record["run_id"])
        deadline = (
            datetime.fromisoformat(manifest["created_at"].replace("Z", "+00:00")).timestamp()
            + manifest["compute"]["selected"]["max_duration_seconds"]
            + 120
        )
        while True:
            terminal = authority.control.get_json_optional(
                f"runs/{record['run_id']}/attempts/{manifest['attempt_id']}/terminal.json"
            )
            if terminal:
                record["status"] = terminal["state"]
                atomic_write_json(ledger, state)
                if not terminal["drain"].get("complete") or terminal["state"] in {
                    "failed",
                    "canceled",
                }:
                    return dict(
                        status="incomplete",
                        reason=f"calibration Run {record['run_id']} failed or did not drain",
                    )
                break
            if time.time() >= deadline:
                return dict(
                    status="incomplete",
                    reason=f"Run {record['run_id']} has no terminal receipt within its finite deadline",
                )
            time.sleep(10)
    report = collect_campaign(campaign, state["runs"], authority)
    atomic_write_json(root / "report.json", report)
    return report
