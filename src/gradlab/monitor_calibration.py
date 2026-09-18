"""Conservative calibration from complete, attributable measurement campaigns.

The operation reports missing evidence explicitly. It never synthesizes throughput
measurements from process separation or short untrained-policy rollouts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

from gradlab.json_utils import canonical_json_sha256
from gradlab.monitor_config import calibration_binding

ROLES = {"early", "intermediate", "stronger", "long-episode"}
PHASES = {
    "inference",
    "capture",
    "encoding",
    "r2_delivery",
    "video_generation",
    "wandb_media_delivery",
}


def assess_calibration(measurements):
    samples = measurements.get("samples", [])
    pairs = measurements.get("pairs", [])
    base = dict(status="incomplete", measurements_sha256=canonical_json_sha256(measurements))
    if not ROLES <= {s.get("role") for s in samples}:
        return {
            **base,
            "reason": "representative early, intermediate, stronger and long-episode evidence is required",
        }
    if len(pairs) < 3 or len({p.get("seed") for p in pairs}) != len(pairs):
        return {
            **base,
            "reason": "at least three independent matched training seed pairs are required",
        }
    train = measurements["train_config"]
    settings = measurements["settings"]
    count = settings["episodes"]
    for sample in samples:
        workers = sample.get("execution_workers", 1)
        if type(workers) is not int or not 1 <= workers <= settings["task_cpus"]:
            return {**base, "reason": "invalid episode process count"}
        if (
            sample.get("episodes_completed") != count
            or not sample.get("verified_terminal_receipt")
            or not sample.get("checkpoint_sha256")
            or not PHASES <= set(sample.get("phase_seconds", {}))
        ):
            return {
                **base,
                "reason": "complete episode sets, terminal receipts and every delivery phase are required",
            }
        for field in (
            "seconds",
            "retained_bytes",
            "peak_memory_bytes",
            "peak_spool_bytes",
            "longest_episode_steps",
        ):
            value = sample.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value <= 0
            ):
                return {
                    **base,
                    "reason": f"representative measurement {field} is missing or invalid",
                }
        if any(
            not isinstance(v, int | float) or not math.isfinite(v) or v < 0
            for v in sample["phase_seconds"].values()
        ):
            return {**base, "reason": "invalid phase duration"}
    if len({s["checkpoint_sha256"] for s in samples}) < 3:
        return {**base, "reason": "at least three distinct representative checkpoints are required"}
    logs = []
    for pair in pairs:
        if (
            not pair.get("warmup_excluded")
            or not pair.get("comparable_host_load")
            or pair.get("checkpoint_freq") != train["checkpoint_freq"]
            or not pair.get("equivalent_workload")
            or not pair.get("nonzero_capture")
        ):
            return {
                **base,
                "reason": "matched cadence, workload, capture, host load and warm-up evidence is required",
            }
        off, on = pair["off_rate"], pair["on_rate"]
        if any(
            isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v) or v <= 0
            for v in (off, on)
        ):
            return {**base, "reason": "invalid measured training throughput"}
        logs.append(math.log(on / off))
    # Two-sided 95% Student-t lower bound on the paired log throughput ratio.
    # For larger samples the df=9 critical value is deliberately conservative.
    critical = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}.get(
        len(logs) - 1, 2.262
    )
    lower = statistics.mean(logs) - critical * statistics.stdev(logs) / math.sqrt(len(logs))
    loss_upper = 1 - math.exp(lower)
    # Parallel wall time cannot establish single-worker active cadence. Multiplying
    # by the process count is a conservative serialized-work upper bound, including
    # loading, waiting and finalization; never divide this speedup a second time.
    duration = max(s["seconds"] * s.get("execution_workers", 1) for s in samples) * 1.25
    retained = max(s["retained_bytes"] for s in samples) * 1.25
    checkpoints = math.ceil(train["timesteps"] / train["checkpoint_freq"]) + 1
    training_seconds = train["timesteps"] / min(p["on_rate"] for p in pairs)
    final_tail = math.ceil(checkpoints / settings["task_cpus"]) * duration
    total_seconds = max(
        training_seconds + final_tail, measurements.get("observed_total_seconds", 0) * 1.25
    )
    recommended_spacing = math.ceil(
        duration * max(p["on_rate"] for p in pairs) / settings["active_workers"]
    )
    report = dict(
        **base,
        binding=calibration_binding(train, settings),
        episodes=count,
        source_sha=measurements["source_sha"],
        runtime_image=measurements["runtime_image"],
        hardware_allocation=measurements["hardware_allocation"],
        maximum_throughput_loss_upper95=loss_upper,
        paired_geometric_mean_throughput_loss=1 - math.exp(statistics.mean(logs)),
        recommended_checkpoint_freq=max(train["checkpoint_freq"], recommended_spacing),
        estimated_checkpoint_count=checkpoints,
        estimated_retained_bytes=math.ceil(checkpoints * retained),
        estimated_total_seconds=math.ceil(total_seconds),
        estimated_final_tail_seconds=math.ceil(final_tail),
        peak_worker_memory_bytes=max(s["peak_memory_bytes"] for s in samples),
        peak_worker_spool_bytes=max(s["peak_spool_bytes"] for s in samples),
    )
    feasible = (
        report["estimated_retained_bytes"] <= settings["contribution_bytes"]
        and total_seconds <= settings["whole_run_seconds"]
        and report["peak_worker_memory_bytes"] * 1.25 <= settings["worker_memory_bytes"]
        and report["peak_worker_spool_bytes"] * 1.25 <= settings["worker_spool_bytes"]
        and train["checkpoint_freq"] >= recommended_spacing
    )
    report["status"] = (
        "infeasible" if not feasible else "unproven" if loss_upper > 0.02 else "supported"
    )
    report["reason"] = (
        "measured configuration fits declared budgets and throughput target"
        if report["status"] == "supported"
        else "declared budgets/cadence or throughput uncertainty do not support enablement"
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="gradlab monitor")
    commands = parser.add_subparsers(dest="command", required=True)
    calibrate = commands.add_parser(
        "calibrate", help="assess complete representative and matched-throughput measurements"
    )
    source = calibrate.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--measurements", type=Path, help="Assess existing measurements without launching work"
    )
    source.add_argument(
        "--campaign",
        type=Path,
        help="Execute/resume an explicit matched supervised training campaign",
    )
    calibrate.add_argument(
        "--campaign-root", type=Path, default=Path.home() / ".config/gradlab/runs/calibration"
    )
    calibrate.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.campaign:
        from gradlab.monitor_campaign import run_campaign

        report = run_campaign(json.loads(args.campaign.read_text()), root=args.campaign_root)
    else:
        report = assess_calibration(json.loads(args.measurements.read_text()))
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0 if report["status"] == "supported" else 2
