"""Selected-lane training capture with bounded nonblocking handoff and lossless chunks.

The learner copies only selected frames/facts. One encoder owns all disk work.
A reserved chunk allowance is charged at admission and is never recycled.
"""

from __future__ import annotations

import importlib.metadata
import io
import queue
import random
import shutil
import threading
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from gradlab.file_utils import atomic_write_json, file_sha256, fsync_path
from gradlab.json_utils import canonical_json_bytes, canonical_json_sha256, json_value
from gradlab.trajectory_config import (
    CollectionConfig,
    FORMAT,
    PROVIDER,
    PROVIDER_VERSION,
    MAX_MANIFEST_BYTES,
    MAX_PRODUCER_BYTES,
    working_memory_bytes,
    index_disk_bytes,
)
from gradlab.trajectory_delivery import spool_bytes

# A full RGB frame plus bounded scalar metadata and queue/Python overhead.
FRAME_RESERVATION = 256 * 1024
METADATA_RESERVATION = 64 * 1024


@dataclass
class _Episode:
    identity: str
    lane: int
    metadata: dict
    submitted: int = 0
    processed: int = 0
    finish_reason: str | None = None
    complete: bool = False
    writer: Any = None
    rows: Any = None
    last: dict = field(default_factory=dict)
    prefix_return: float = 0.0
    byte_count: int = 0


def attach_training_recorder(context, runtime, model):
    """Attach before the first reset, using the supervisor's durable admission."""
    from gradlab.trajectory_delivery import read_document

    train = context.train_config
    collection = train.get("trajectory_collection") or {}
    if not collection.get("enabled"):
        return None
    root = context.run_dir / "trajectories" / str(train["attempt_id"])
    admission = read_document(root / "admission.json")
    if not admission["budget_available"]:
        return None
    recorder = TrainingRecorder(
        runtime, root, CollectionConfig(**collection),
        {"run_id": train["wandb_run_id"], "attempt_id": train["attempt_id"],
         "train_config": train},
        position=lambda: (int(model.num_timesteps), int(model._n_updates)),
        previous_reserved_bytes=int(admission["previous_reserved_bytes"]),
    )
    runtime.recording = recorder
    return recorder


def verify_recording_provider(runtime) -> dict:
    from env_breakoutatari2600_turbo_native import BreakoutVecEnv
    from gradlab.env_providers import _StartInfoAdapter

    provider = runtime.provider
    native = provider.env if type(provider) is _StartInfoAdapter else provider
    if (
        runtime.descriptor.provider_id != PROVIDER
        or type(native) is not BreakoutVecEnv
        or importlib.metadata.version(PROVIDER) != PROVIDER_VERSION
    ):
        raise ValueError("training capture requires the exact verified native Breakout provider")
    table = runtime.descriptor.action_meanings
    expected = ("noop", "button", "right", "left")
    if (
        not table
        or any(name not in expected for name in table)
        or tuple(runtime.descriptor.action_table or ())
        != tuple(() if name == "noop" else (name.upper(),) for name in table)
    ):
        raise ValueError("training capture requires the verified four-action native encoding")
    # Rendering reads current unmasked native pixels; it does not change transition behavior.
    native.render_mode = "rgb_array"
    return {
        "provider": PROVIDER,
        "version": PROVIDER_VERSION,
        "execution_evidence": "verified-gradlab-submission-v1",
        "native_action_meanings": list(expected),
        "native_encoding": [expected.index(name) for name in table],
        "rgb_shape": [210, 160, 3],
        "source": "https://github.com/tsilva/env-BreakoutAtari2600-turbo-native/blob/v0.5.13/src/lib.rs",
        "action_contract": json_value(runtime.action_contract),
        "signal_metadata": json_value(native.signal_metadata),
    }


class TrainingRecorder:
    def __init__(
        self,
        runtime,
        root: Path,
        config: CollectionConfig,
        provenance: dict,
        *,
        position: Callable[[], tuple[int, int]],
        previous_reserved_bytes: int = 0,
    ):
        self.contract = verify_recording_provider(runtime)
        self.runtime = runtime
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.position = position
        self.provenance = json_value(provenance)
        self.planned_steps = int(provenance["train_config"]["timesteps"])
        self.prefix = f"{provenance['run_id']}.{provenance['attempt_id']}"
        self.random = random.Random(f"{self.prefix}.collection")
        self.sessions: dict[int, _Episode] = {}
        self.active: dict[int, _Episode] = {}
        self.serial = 0
        self.reserved = previous_reserved_bytes
        self.queue = queue.Queue(
            maxsize=max(
                1, (config.memory_bytes - working_memory_bytes(config)) // FRAME_RESERVATION
            )
        )
        self.closed = False
        self.fault: str | None = None
        self.can_admit = False
        self.available_slots = 0
        self.skipped = 0
        self.incomplete = 0
        self.captured = 0
        self.encoded_bytes = 0
        self.pause_reason = "starting"
        self.pause_seconds = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._actions: dict[int, dict] = {}
        self._thread = threading.Thread(target=self._encode, name="trajectory-encoder", daemon=True)
        # All expensive provider validation and provenance hashing precede training.
        train = self.provenance["train_config"]
        self.contract["execution"] = {
            key: train.get(key)
            for key in (
                "game",
                "frame_skip",
                "task",
                "sticky_action_prob",
                "goal_contract_sha256",
                "effective_goal_contract_sha256",
            )
        }
        self.contract["execution"]["discount"] = {
            key: (train.get("training_backend", {}).get("config") or {}).get(key)
            for key in ("gamma", "gamma_final", "gamma_schedule_timesteps")
        }
        self.contract_hash = canonical_json_sha256(self.contract)
        producer = {
            "format": FORMAT,
            "provenance": self.provenance,
            "contract": self.contract,
            "config": asdict(config),
        }
        if len(canonical_json_bytes(producer, ensure_ascii=True)) > MAX_PRODUCER_BYTES:
            raise ValueError("recording provenance exceeds bounded metadata allowance")
        atomic_write_json(self.root / "producer.json", producer)
        self._refresh_capacity()
        self._thread.start()

    def _fault(self, reason: str):
        self.fault = reason
        self.pause_reason = "recording_fault"
        for episode in self.active.values():
            episode.finish_reason = reason
        self.active.clear()

    def reset(self, mask, *, after_step=False):
        if self.closed or self.fault:
            return
        step, update = self.position()
        if after_step:
            step += self.runtime.num_envs
        for lane in np.flatnonzero(mask):
            lane = int(lane)
            if self.random.random() > self.config.sample_probability:
                self.skipped += 1
                continue
            stage = min(
                self.config.budget_stages,
                1 + step * self.config.budget_stages // self.planned_steps,
            )
            allowance = max(
                self.config.chunk_bytes,
                self.config.contribution_bytes * stage // self.config.budget_stages,
            )
            if (
                not self.can_admit
                or len(self.sessions) >= self.config.max_active_episodes
                or self.reserved + self.config.chunk_bytes > allowance
            ):
                self.skipped += 1
                if self.reserved + self.config.chunk_bytes > allowance:
                    self.pause_reason = (
                        "budget" if stage == self.config.budget_stages else "stage_budget"
                    )
                continue
            if not self._lock.acquire(blocking=False):
                self.skipped += 1
                continue
            try:
                if (
                    self.available_slots <= 0
                    or len(self.sessions) >= self.config.max_active_episodes
                ):
                    self.skipped += 1
                    continue
                self.serial += 1
                identity = f"{self.prefix}.{self.runtime.global_lane_ids[lane]}.{self.serial}"
                metadata = {
                    "episode_id": identity,
                    "run_id": self.provenance["run_id"],
                    "attempt_id": self.provenance["attempt_id"],
                    "lane": self.runtime.global_lane_ids[lane],
                    "episode_index": int(self.runtime._episode_indices[lane]),
                    "seed": self.runtime._episode_seeds[lane],
                    "start_id": self.runtime._start_ids[lane],
                    "start_origin": self.runtime._start_origins[lane],
                    "curriculum_cell_id": self.runtime._curriculum_cell_ids[lane],
                    "initial_episode_steps": int(self.runtime._episode_lengths[lane]),
                    "training_start_step": step,
                    "policy_start_update": update,
                    "planned_steps": self.planned_steps,
                    "stage_start": step / self.planned_steps,
                    "start_facts": self.runtime.reset_infos[lane],
                }
                episode = _Episode(identity, lane, json_value(metadata))
                frame = self._frame(lane)
                self.queue.put_nowait((episode, None, frame))
                self.sessions[self.serial] = episode
                self.active[lane] = episode
                self.reserved += self.config.chunk_bytes
                self.available_slots -= 1
                self.can_admit = self.available_slots > 0
                self.pause_reason = ""
            except queue.Full:
                self.serial -= 1
                self.skipped += 1
                self.pause_reason = "memory"
            except Exception as exc:
                self._fault(type(exc).__name__)
            finally:
                self._lock.release()

    def _frame(self, lane):
        frame = self.runtime.provider.render_lane(lane)
        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.shape != (210, 160, 3)
        ):
            raise ValueError("missing full native RGB")
        return frame.copy()

    def before_step(self, requested, submitted):
        self._actions.clear()
        try:
            for lane in tuple(self.active):
                action = int(np.asarray(submitted)[lane])
                if action not in range(len(self.contract["native_encoding"])):
                    raise ValueError("missing verified executed action")
                effective = getattr(self.runtime.kernel, "effective_action", None)
                reason = getattr(self.runtime.kernel, "action_override_rule_id", None)
                self._actions[lane] = {
                    "policy_action": json_value(np.asarray(requested)[lane].copy()),
                    "effective_policy_action": json_value(
                        effective(lane) if effective else np.asarray(requested)[lane]
                    ),
                    "executed_action": action,
                    "native_action": self.contract["native_encoding"][action],
                    "override_rule": reason(lane) if reason else None,
                }
        except Exception as exc:
            self._fault(type(exc).__name__)

    def transition(
        self,
        infos,
        provider_rewards,
        provider_terminated,
        provider_truncated,
        rewards,
        terminated,
        truncated,
        task_step,
        forced,
    ):
        step, update = self.position()
        try:
            for lane, episode in tuple(self.active.items()):
                if episode.finish_reason is not None:
                    self.active.pop(lane, None)
                    continue
                if self.queue.full():
                    episode.finish_reason = "memory_pressure"
                    self.active.pop(lane, None)
                    self.pause_reason = "memory"
                    continue
                facts = {}
                for key, values in infos.items():
                    if key.startswith("_"):
                        continue
                    mask = infos.get("_" + key)
                    if mask is not None and not mask[lane]:
                        continue
                    value = np.asarray(values)[lane]
                    if np.asarray(value).size <= 128:
                        facts[key] = json_value(value)
                row = {
                    **self._actions[lane],
                    "step": episode.submitted,
                    "training_step": step + self.runtime.num_envs,
                    "policy_update": update,
                    "provider_reward": float(provider_rewards[lane]),
                    "reward": float(rewards[lane]),
                    "provider_terminated": bool(provider_terminated[lane]),
                    "provider_truncated": bool(provider_truncated[lane]),
                    "terminated": bool(terminated[lane]),
                    "truncated": bool(truncated[lane]),
                    "boundary_reason": "forced_reset"
                    if forced[lane]
                    else "terminated"
                    if terminated[lane]
                    else "truncated"
                    if truncated[lane]
                    else None,
                    "outcome": int(task_step.outcomes[lane]),
                    "facts": facts,
                    "task_metrics": {
                        key: json_value(np.asarray(value)[lane])
                        for key, value in task_step.metrics.items()
                    },
                }
                frame = self._frame(lane)
                episode.submitted += 1
                try:
                    self.queue.put_nowait((episode, row, frame))
                except queue.Full:
                    episode.submitted -= 1
                    episode.finish_reason = "memory_pressure"
                    self.active.pop(lane, None)
                    continue
                self.captured += 1
                if terminated[lane] or truncated[lane]:
                    episode.complete = episode.metadata["initial_episode_steps"] == 0
                    episode.finish_reason = row["boundary_reason"]
                    self.active.pop(lane, None)
        except Exception as exc:
            self._fault(type(exc).__name__)

    def close(self, reason="training_stopped"):
        self.closed = True
        for episode in self.active.values():
            episode.finish_reason = reason
        self.active.clear()
        self._stop.set()

    def wait(self, timeout):
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("trajectory encoding did not finish before deadline")
        if self.fault:
            raise RuntimeError(f"trajectory encoding failed: {self.fault}")

    def _refresh_capacity(self):
        used = spool_bytes(self.root)
        usage = shutil.disk_usage(self.root)
        free = usage.free
        # Conservatively retain a whole reservation for every writing session,
        # in addition to its existing bytes and all future bounded index growth.
        reserve = len(self.sessions) * self.config.chunk_bytes + index_disk_bytes(self.config)
        available = (
            min(
                self.config.disk_bytes - used,
                free - max(self.config.scratch_headroom_bytes, int(usage.total * 0.06)),
            )
            - reserve
        )
        self.available_slots = max(0, available // self.config.chunk_bytes)
        self.can_admit = self.available_slots > 0
        if not self.can_admit:
            self.pause_reason = "disk"
        elif self.pause_reason in {"disk", "memory"} and not self.queue.full():
            self.pause_reason = ""
        return used

    def _write(self, episode, row, frame):
        if episode.writer is None:
            episode.writer = zipfile.ZipFile(
                self.root / f"{episode.identity}.partial", "x", compression=zipfile.ZIP_STORED
            )
            episode.rows = (self.root / f"{episode.identity}.rows").open("xb")
        if row is not None and (
            episode.processed >= 8192
            or episode.byte_count + FRAME_RESERVATION + METADATA_RESERVATION
            >= self.config.chunk_bytes
        ):
            episode.finish_reason = "chunk_limit"
            episode.complete = False
            episode.processed += 1
            return
        stream = io.BytesIO()
        Image.fromarray(frame).save(stream, format="PNG", compress_level=1)
        payload = stream.getvalue()
        index = episode.processed + 1 if row is not None else 0
        episode.writer.writestr(f"frames/{index}.png", payload)
        episode.byte_count += len(payload) + 128
        if row is not None:
            data = canonical_json_bytes(row) + b"\n"
            if len(data) > 16 * 1024:
                raise ValueError("transition metadata exceeds recording contract")
            episode.rows.write(data)
            episode.byte_count += 2 * len(data)  # Sidecar and final archive coexist while sealing.
            episode.processed += 1
            episode.prefix_return += row["reward"]
            episode.last = row

    def _seal(self, episode):
        episode.rows.close()
        rows_path = self.root / f"{episode.identity}.rows"
        episode.writer.write(rows_path, "transitions.jsonl")
        captured_steps = episode.last.get("step", -1) + 1
        summary = {
            **episode.metadata,
            "complete": episode.complete,
            "interruption_reason": None if episode.complete else episode.finish_reason,
            "captured_start": episode.metadata["initial_episode_steps"],
            "captured_steps": captured_steps,
            "prefix_return": episode.prefix_return,
            "last_transition": episode.last,
            "training_end_step": episode.last.get(
                "training_step", episode.metadata["training_start_step"]
            ),
            "stage_end": episode.last.get("training_step", episode.metadata["training_start_step"])
            / self.planned_steps,
        }
        if episode.complete:
            summary.update(
                episode_return=episode.prefix_return,
                episode_length=captured_steps,
                outcome=episode.last.get("outcome"),
            )
        episode.writer.writestr("episode.json", canonical_json_bytes(summary))
        episode.writer.close()
        rows_path.unlink()
        partial = self.root / f"{episode.identity}.partial"
        fsync_path(partial)
        path = partial.with_suffix(".zip")
        partial.replace(path)
        size = path.stat().st_size
        if size + METADATA_RESERVATION > self.config.chunk_bytes:
            raise RuntimeError("trajectory chunk exceeds its reserved allowance")
        manifest = {
            "format": FORMAT,
            "file": path.name,
            "sha256": file_sha256(path),
            "bytes": size,
            "reserved_bytes": self.config.chunk_bytes,
            "contract_sha256": self.contract_hash,
            "episode": summary,
        }
        if len(canonical_json_bytes(manifest, ensure_ascii=True)) > MAX_MANIFEST_BYTES:
            raise ValueError("episode index exceeds bounded metadata allowance")
        atomic_write_json(self.root / f"{episode.identity}.manifest.json", manifest)
        self.encoded_bytes += size
        self.incomplete += int(not episode.complete)

    def _encode(self):
        last = time.monotonic()
        try:
            while not self._stop.is_set() or not self.queue.empty() or self.sessions:
                try:
                    episode, row, frame = self.queue.get(timeout=0.05)
                except queue.Empty:
                    pass
                else:
                    self._write(episode, row, frame)
                    self.queue.task_done()
                with self._lock:
                    for key, episode in tuple(self.sessions.items()):
                        if (
                            episode.finish_reason
                            and episode.processed == episode.submitted
                            and episode.writer
                        ):
                            self._seal(episode)
                            self.sessions.pop(key)
                    now = time.monotonic()
                    if now - last >= 0.5 or self._stop.is_set():
                        used = self._refresh_capacity()
                        if self.pause_reason:
                            self.pause_seconds += now - last
                        atomic_write_json(
                            self.root / "status.json",
                            {
                                "captured": self.captured,
                                "encoded_bytes": self.encoded_bytes,
                                "reserved_bytes": self.reserved,
                                "skipped": self.skipped,
                                "incomplete": self.incomplete,
                                "pause_seconds": self.pause_seconds,
                                "pause_reason": self.pause_reason,
                                "spool_bytes": used,
                                "queue_bytes": self.queue.qsize() * FRAME_RESERVATION,
                                "fault": self.fault,
                            },
                        )
                        last = now
            atomic_write_json(
                self.root / "closed.json",
                {
                    "format": FORMAT,
                    "reserved_bytes": self.reserved,
                    "chunks": self.serial,
                    "fault": self.fault,
                },
            )
        except Exception as exc:
            self._fault(type(exc).__name__)
            atomic_write_json(self.root / "fault.json", {"error": type(exc).__name__})
