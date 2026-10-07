"""Bounded process workers for the collector's existing single-lane adapter.

Only the parent owns a policy or dataset writer. Each worker owns disjoint
environments and returns one complete, ordered batch before admitting another.
"""

from __future__ import annotations

import multiprocessing as mp
from types import SimpleNamespace
import traceback

import numpy as np

from collector import VectorPolicyExecution


def environment_worker(connection, config, lane_ids, observation_space, action_mode):
    from collector import PolicyExecution, capture_rgb, encode_rgb
    from gradlab.env import make_eval_vec_env
    import torch

    torch.set_num_threads(1)
    lanes = {}
    mask_hud = False
    try:
        runtime = SimpleNamespace(
            capabilities=SimpleNamespace(default_action_selection_mode=action_mode),
            supports_sampling_temperature=True,
            model=SimpleNamespace(observation_space=observation_space),
        )
        for lane in lane_ids:
            env = make_eval_vec_env(config, 1, 0, capture_step_diagnostics=True)
            lanes[lane] = PolicyExecution(runtime, env, config, contract={}, provenance={})
        connection.send((True, None))
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            if command == "capture":
                mask_hud = payload
                result = None
            elif command == "reset":
                lane, seed = payload
                image = lanes[lane].reset(seed, reset_policy=False)
                result = (image, lanes[lane].obs)
            elif command == "step":
                result = {}
                for lane, (actions, raw_action) in payload.items():
                    image, facts = lanes[lane].apply(actions, raw_action)
                    encoded = encode_rgb(capture_rgb(image, mask_hud=mask_hud))
                    result[lane] = (image, facts, lanes[lane].obs, encoded)
            else:
                raise ValueError(f"unknown environment worker command: {command}")
            connection.send((True, result))
    except EOFError, BrokenPipeError:
        pass
    except BaseException:
        connection.send((False, traceback.format_exc()))
    finally:
        for lane in lanes.values():
            lane.close()
        connection.close()


class ParallelPolicyExecution(VectorPolicyExecution):
    def __init__(self, runtime, config, contract, provenance, n_envs, workers):
        self.runtime, self.contract, self.n_envs = runtime, contract, n_envs
        self.action_selection_mode = runtime.capabilities.default_action_selection_mode
        self.supports_temperature = runtime.supports_sampling_temperature
        self.provenance = {
            **provenance,
            "vectorization": {
                "n_envs": n_envs,
                "environment_workers": workers,
                "environment_execution": "independent_process_groups",
                "image_encoding": "worker_lossless_webp",
                "inference": "batched_by_temperature",
                "policy_rng": "session_stream_seeded_by_first_episode_seed",
            },
        }
        self.seeded = False
        self.lanes = [SimpleNamespace(obs=None) for _ in range(n_envs)]
        self.encoded_frames = {}
        self.workers = []
        self.owner = {}
        context = mp.get_context("spawn")  # Never fork an initialized CUDA runtime.
        try:
            for index in range(workers):
                lane_ids = list(range(index, n_envs, workers))
                parent, child = context.Pipe()
                process = context.Process(
                    target=environment_worker,
                    args=(
                        child,
                        config,
                        lane_ids,
                        runtime.model.observation_space,
                        self.action_selection_mode,
                    ),
                )
                process.start()
                child.close()
                self.workers.append((parent, process))
                self.owner.update({lane: index for lane in lane_ids})
            for index in range(workers):
                self.receive(index)
        except BaseException:
            self.close()
            raise

    def receive(self, worker):
        connection, process = self.workers[worker]
        if not connection.poll(120):
            raise RuntimeError(f"environment worker {worker} timed out (exit={process.exitcode})")
        try:
            success, result = connection.recv()
        except EOFError as error:
            raise RuntimeError(f"environment worker {worker} exited unexpectedly") from error
        if not success:
            raise RuntimeError(f"environment worker {worker} failed:\n{result}")
        return result

    def configure_capture(self, mask_hud):
        for connection, _ in self.workers:
            connection.send(("capture", mask_hud))
        for index in range(len(self.workers)):
            self.receive(index)

    def reset_lane(self, lane, seed):
        if not self.seeded:
            import torch

            torch.manual_seed(seed)
            np.random.seed(seed)
            self.runtime.reset()
            self.seeded = True
        worker = self.owner[lane]
        self.workers[worker][0].send(("reset", (lane, seed)))
        image, self.lanes[lane].obs = self.receive(worker)
        return image

    def step_batch(self, temperatures):
        decisions = self.decide_batch(temperatures)
        groups = {}
        for lane in temperatures:
            groups.setdefault(self.owner[lane], {})[lane] = decisions[lane]
        for worker, actions in groups.items():
            self.workers[worker][0].send(("step", actions))
        results = {}
        self.encoded_frames = {}
        for worker in groups:
            for lane, (image, facts, obs, encoded) in self.receive(worker).items():
                results[lane] = (image, facts)
                self.lanes[lane].obs = obs
                self.encoded_frames[lane] = encoded
        # Preserve capture/discovery ordering independently of worker completion.
        return {lane: results[lane] for lane in temperatures}

    def close(self):
        for connection, process in self.workers:
            if process.is_alive():
                try:
                    connection.send(("close", None))
                except BrokenPipeError, EOFError, OSError:
                    pass
        for connection, process in self.workers:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            connection.close()
        self.workers.clear()
