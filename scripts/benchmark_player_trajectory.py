"""Compare the same scripted Player episode with recording off and on.

Run from a checkout: uv run --frozen python scripts/benchmark_player_trajectory.py
Each case runs in a fresh process so peak RSS measurements are independent.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.test_play_trajectory import ScriptedSession, command  # noqa: E402
from tests.test_policy_bundle import write_bundle  # noqa: E402
from gradlab.play_trajectory import export_trajectory  # noqa: E402
from gradlab.play_web import WebPlaybackRunner  # noqa: E402
from gradlab.policy_bundle import load_policy_bundle  # noqa: E402


class MeasuredSession(ScriptedSession):
    def __init__(self, steps, profile):
        super().__init__(steps)
        shape = (1, 4, 84, 84) if profile == "standard" else (1, 32, 160, 160)
        dtype = np.uint8 if profile == "standard" else np.float32
        self.input = np.zeros(shape, dtype=dtype)
        self.current_frame = np.zeros(
            (240, 256, 3) if profile == "standard" else (480, 640, 3), np.uint8
        )
        self.observations = tuple(np.zeros((84, 84, 1), np.uint8) for _ in range(4))

    def step(self, **kwargs):
        frame = self.current_frame
        transition = super().step(**kwargs)
        self.current_frame = frame
        return replace(
            transition,
            model_obs=self.input,
            next_model_obs=self.input,
            before_frame=frame,
            after_frame=frame,
            before_frames=self.observations,
        )


class MeasuredRunner(WebPlaybackRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.starts = []
        self.durations = []
        self.backlog = []
        self.memory = []

    def _step_once(self):
        start = time.perf_counter()
        self.starts.append(start)
        result = super()._step_once()
        self.durations.append((time.perf_counter() - start) * 1000)
        self.backlog.append(self.recording_status().get("backlog_bytes", 0))
        if len(self.starts) % 30 == 0:
            self.memory.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return result


def run_case(args):
    with tempfile.TemporaryDirectory(prefix="gradlab-trajectory-benchmark-") as temp:
        root = Path(temp)
        write_bundle(root)
        session = MeasuredSession(args.steps, args.profile)
        runner = MeasuredRunner(
            session,
            argparse.Namespace(fps=args.fps, episodes=1),
            config_text="",
            trajectory_bundle=load_policy_bundle(root),
        )
        runner.start()
        command(runner, "set_recording", enabled=args.record)
        baseline_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        try:
            command(runner, "play")
            deadline = time.monotonic() + 45
            while runner.run_state != "paused" and time.monotonic() < deadline:
                time.sleep(0.01)
            # State changes precede publication; wait for the entire last measured step.
            while len(runner.durations) != len(runner.starts) and time.monotonic() < deadline:
                time.sleep(0.01)
            if session.sequence != args.steps or len(runner.durations) != args.steps:
                raise RuntimeError(f"benchmark did not finish: {runner.snapshot()}")
            durations = np.asarray(runner.durations)
            intervals = np.diff(runner.starts) * 1000
            rss_scale = 1 if sys.platform == "darwin" else 1024
            result = {
                "profile": args.profile,
                "recording": args.record,
                "target_fps": args.fps,
                "requested_steps": args.steps,
                "steps": session.sequence,
                "step_p50_ms": float(np.percentile(durations, 50)),
                "step_p99_ms": float(np.percentile(durations, 99)),
                "pacing_p50_ms": float(np.percentile(intervals, 50)),
                "pacing_p99_ms": float(np.percentile(intervals, 99)),
                "late_intervals": int(np.count_nonzero(intervals > 1500 / args.fps)),
                "max_backlog_mib": max(runner.backlog) / 1024**2,
                "peak_rss_growth_mib": (
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - baseline_rss
                )
                * rss_scale
                / 1024**2,
                "rss_samples_mib": [value * rss_scale / 1024**2 for value in runner.memory],
                "pause_reason": runner.snapshot().get("status_message"),
            }
            if args.record:
                start = time.perf_counter()
                frozen = runner.freeze_trajectory()
                result["freeze_ms"] = (time.perf_counter() - start) * 1000
                start = time.perf_counter()
                archive = export_trajectory(frozen, root / "episode.gradtraj")
                result["export_seconds"] = time.perf_counter() - start
                result["archive_mib"] = archive.stat().st_size / 1024**2
            print(json.dumps(result))
        finally:
            runner.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="store_true")
    parser.add_argument("--profile", choices=("standard", "large"), default="standard")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--fps", type=float, default=60)
    args = parser.parse_args()
    if args.case:
        run_case(args)
    else:
        for profile in ("standard", "large"):
            for record in (False, True):
                subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        "--case",
                        "--profile",
                        profile,
                        "--steps",
                        str(args.steps),
                        "--fps",
                        str(args.fps),
                        *(["--record"] if record else []),
                    ],
                    check=True,
                )
