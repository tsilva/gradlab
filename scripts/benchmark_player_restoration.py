"""Measure exact native Breakout capture at the real Player decision boundary.

Run each case in a fresh process with --capture or without it. No training,
publication, credentials or acceptance evaluation is involved.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import resource
import tempfile
import time
import json

import numpy as np
import torch
from stable_baselines3 import PPO
from gradlab.env import make_eval_vec_env, resolve_env_config
from gradlab.env_config import env_config_from_mapping
from gradlab.recipe_documents import compose_train_document
from gradlab.play_session import _PlaybackSession
from gradlab.policy_runtime import PolicyRuntime
from gradlab.play_restoration import LiveRestoration
from gradlab.play_trajectory import pack_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--fps", type=float, default=60)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path("experiments/goals/Breakout-Atari2600-v0")
    config = resolve_env_config(
        env_config_from_mapping(
            compose_train_document(root / "_goal.yaml", root / "recipes/ppo.yaml")["train_config"]
        )
    )
    env = make_eval_vec_env(config, n_envs=1, seed=40000, capture_step_diagnostics=True)
    try:
        model = PPO("MultiInputPolicy", env, n_steps=8, batch_size=8, device="cpu", seed=17)
        session = _PlaybackSession(
            model=model,
            env=env,
            config=config,
            initial_seed=40000,
            policy_runtime=PolicyRuntime(model, algorithm_id="ppo"),
        )
        session.restart()
        adapter = LiveRestoration(session)
        # Warm the actual tensor/runtime paths before the measured workload.
        session.step(action_selection_mode="stochastic")
        adapter.capture()
        durations, starts, capture_times, restore_times = [], [], [], []
        baseline_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        with tempfile.TemporaryDirectory(prefix="gradlab-restore-measure-") as temp:
            path = Path(temp) / "restore-points.bin"
            next_at = time.perf_counter()
            with path.open("wb") as stream:
                for _ in range(args.steps):
                    time.sleep(max(0, next_at - time.perf_counter()))
                    start = time.perf_counter()
                    starts.append(start)
                    transition = session.step(action_selection_mode="stochastic")
                    if args.capture and not transition.boundary:
                        before_capture = time.perf_counter()
                        state = adapter.capture()
                        data = pack_record(state)
                        stream.write(data)
                        stream.flush()
                        capture_times.append((time.perf_counter() - before_capture) * 1000)
                    durations.append((time.perf_counter() - start) * 1000)
                    next_at += 1 / args.fps
            state = adapter.capture()
            for _ in range(10):
                start = time.perf_counter()
                adapter.restore(state)
                restore_times.append((time.perf_counter() - start) * 1000)
            intervals = np.diff(starts) * 1000
            print(
                json.dumps(
                    {
                        "capture": args.capture,
                        "steps": args.steps,
                        "fps": args.fps,
                        "step_ms_p50_p99": np.percentile(durations, [50, 99]).tolist(),
                        "interval_ms_p50_p99": np.percentile(intervals, [50, 99]).tolist(),
                        "late_intervals": int(np.count_nonzero(intervals > 1500 / args.fps)),
                        "capture_ms_p50_p99": np.percentile(capture_times, [50, 99]).tolist()
                        if capture_times
                        else [],
                        "restore_ms_p50_p99": np.percentile(restore_times, [50, 99]).tolist(),
                        "restore_disk_mib": path.stat().st_size / 1024**2,
                        "peak_rss_growth_mib": (
                            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - baseline_rss
                        )
                        / 1024**2,
                    }
                )
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
