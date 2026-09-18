"""Complete observational episodes on immutable checkpoint contracts.

One environment and RNG stream per episode makes placement independent of the
number of concurrent checkpoint workers. This runs only outside the learner.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import statistics
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from gradlab.json_utils import canonical_json_bytes, json_value
from gradlab.seeds import MONITOR_SEED_START, MONITOR_POLICY_SEED_START

FORMAT = "gradlab.checkpoint-monitoring.v1"


def episode_manifest(count=400):
    if type(count) is not int or not 1 <= count <= 100_000:
        raise ValueError("monitoring episode count must be between 1 and 100000")
    return [
        dict(
            episode_id=f"monitor-{i:06d}",
            ordinal=i,
            environment_seed=MONITOR_SEED_START + i,
            policy_seed=MONITOR_POLICY_SEED_START + i,
        )
        for i in range(count)
    ]


def verified_put(bucket, key, payload):
    """A lost upload response is safe: create-only writes compare existing bytes."""
    bucket.put_bytes(key, payload, metadata={"sha256": hashlib.sha256(payload).hexdigest()})
    if bucket.get_bytes(key) != payload:
        raise ValueError("monitoring object verification failed")
    return dict(key=key, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())


def verified_get(bucket, reference):
    payload = bucket.get_bytes(reference["key"])
    if (
        len(payload) != reference["bytes"]
        or hashlib.sha256(payload).hexdigest() != reference["sha256"]
    ):
        raise ValueError("monitoring object size/hash mismatch")
    return payload


def _png(frame):
    stream = io.BytesIO()
    Image.fromarray(frame).save(stream, format="PNG", compress_level=1)
    return stream.getvalue()


class EpisodeRecording:
    """The existing runtime recording boundary, with complete multi-chunk episodes."""

    def __init__(self, runtime, *, bucket, root, prefix, chunk_bytes, reserve=None, guard=None):
        from gradlab.recording_contract import verify_recording_provider

        self.contract = verify_recording_provider(runtime)
        self.runtime, self.bucket = runtime, bucket
        self.root, self.prefix = Path(root), prefix
        self.root.mkdir(parents=True, exist_ok=True)
        self.limit, self.reserve = chunk_bytes, reserve
        self.guard = guard
        self.chunks, self.rows = [], []
        self.entries = {}
        self.size = 0
        self.step = 0
        self.start = 0
        self.complete = False
        self.started = False
        self.last = None
        self.initial = None
        self.phase_seconds = {"capture": 0.0, "encoding": 0.0}

    def _frame(self):
        started = time.perf_counter()
        value = self.runtime.provider.render_lane(0)
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.uint8
            or value.shape != (210, 160, 3)
        ):
            raise ValueError("missing full unmasked native RGB")
        copied = value.copy()
        encoded_at = time.perf_counter()
        data = _png(copied)
        self.phase_seconds["capture"] += encoded_at - started
        self.phase_seconds["encoding"] += time.perf_counter() - encoded_at
        return data

    def reset(self, mask, after_step=False):
        if self.started or after_step:
            return
        self.started = True
        self.initial = json_value(self.runtime.reset_infos[0])
        self._add("frames/0.png", self._frame())

    def _add(self, name, data):
        self.entries[name] = data
        self.size += len(data) + 2 * len(name) + 128

    def before_step(self, requested, submitted):
        if self.guard is not None:
            self.guard()
        action = int(np.asarray(submitted)[0])
        if action not in range(len(self.contract["native_encoding"])):
            raise ValueError("missing verified executed action")
        effective = getattr(self.runtime.kernel, "effective_action", None)
        override = getattr(self.runtime.kernel, "action_override_rule_id", None)
        self.actions = dict(
            policy_action=json_value(np.asarray(requested)[0]),
            effective_policy_action=json_value(
                effective(0) if effective else np.asarray(requested)[0]
            ),
            executed_action=action,
            native_action=self.contract["native_encoding"][action],
            override_rule=override(0) if override else None,
        )

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
        if self.complete:
            raise ValueError("monitoring episode already complete")
        facts = {}
        for key, values in infos.items():
            if key.startswith("_") or ("_" + key in infos and not infos["_" + key][0]):
                continue
            value = np.asarray(values)[0]
            if np.asarray(value).size <= 128:
                facts[key] = json_value(value)
        if forced[0]:
            raise ValueError("monitoring encountered an operational forced reset")
        row = dict(
            **self.actions,
            step=self.step,
            provider_reward=float(provider_rewards[0]),
            reward=float(rewards[0]),
            provider_terminated=bool(provider_terminated[0]),
            provider_truncated=bool(provider_truncated[0]),
            terminated=bool(terminated[0]),
            truncated=bool(truncated[0]),
            outcome=int(task_step.outcomes[0]),
            facts=facts,
            task_metrics={k: json_value(np.asarray(v)[0]) for k, v in task_step.metrics.items()},
        )
        data = canonical_json_bytes(row) + b"\n"
        frame = self._frame()  # before the runtime resets a terminal lane
        if self.size + len(frame) + len(data) + 1024 > self.limit:
            previous = self.entries[f"frames/{self.step}.png"]
            self.seal()
            self._add(f"frames/{self.step}.png", previous)
        if self.size + len(frame) + len(data) + 1024 > self.limit:
            raise ValueError("monitoring chunk cannot hold a native transition")
        self.rows.append(data)
        self.size += len(data)
        self.step += 1
        self._add(f"frames/{self.step}.png", frame)
        self.last = row
        self.complete = bool(terminated[0] or truncated[0])
        if self.complete:
            self.seal()

    def seal(self):
        if not self.rows:
            return
        local = self.root / "chunk.zip"
        started = time.perf_counter()
        with zipfile.ZipFile(local, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, value in self.entries.items():
                archive.writestr(zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0)), value)
            archive.writestr(
                zipfile.ZipInfo("transitions.jsonl", (1980, 1, 1, 0, 0, 0)), b"".join(self.rows)
            )
        payload = local.read_bytes()
        self.phase_seconds["encoding"] += time.perf_counter() - started
        if len(payload) > self.limit:
            raise ValueError("monitoring chunk exceeded byte limit")
        # An interrupted partial chunk must never poison the resumed episode's
        # immutable chunk ordinal. Identical complete chunks still deduplicate.
        digest = hashlib.sha256(payload).hexdigest()
        key = f"{self.prefix}/chunks/{self.start:08d}-{self.step:08d}-{digest}.zip"
        if self.reserve:
            self.reserve(key, len(payload))
        reference = verified_put(self.bucket, key, payload)
        reference.update(
            first_step=self.start,
            end_step=self.step,
            first_frame_sha256=hashlib.sha256(self.entries[f"frames/{self.start}.png"]).hexdigest(),
            last_frame_sha256=hashlib.sha256(self.entries[f"frames/{self.step}.png"]).hexdigest(),
        )
        manifest_data = canonical_json_bytes(reference)
        if self.reserve:
            self.reserve(key + ".json", len(manifest_data))
        verified_put(self.bucket, key + ".json", manifest_data)
        # Both the chunk and its reconstruction reference are recoverable remotely.
        local.unlink()
        self.chunks.append(reference)
        self.entries, self.rows, self.size = {}, [], 0
        self.start = self.step

    def close(self):
        if not self.complete:
            self.seal()  # retained diagnostic prefix; never promoted to complete evidence


def monitor_episode(
    *,
    model,
    config,
    episode,
    bucket,
    root,
    prefix,
    provenance,
    chunk_bytes=32 * 1024**2,
    watchdog_steps=1_000_000,
    reserve=None,
    policy_runtime=None,
    guard=None,
    measure=False,
):
    import torch
    from gradlab.env import make_eval_vec_env
    from gradlab.env_registry import environment_spec
    from gradlab.eval_metrics import run_eval_episode
    from gradlab.action_contract import assert_action_contract_compatible
    from gradlab.policy_runtime import bind_policy_action_space
    from gradlab.policy_execution import verify_policy_execution_contract

    expected = episode_manifest(episode["ordinal"] + 1)[-1]
    if episode != expected:
        raise ValueError("monitoring episode identity/seeds differ from the frozen manifest")
    env = make_eval_vec_env(config, n_envs=1, seed=episode["environment_seed"])
    recorder = None
    inference_seconds = 0.0

    class TimedPolicy:
        def __init__(self, policy):
            self.policy = policy

        def __getattr__(self, name):
            return getattr(self.policy, name)

        def decide(self, *args, **kwargs):
            nonlocal inference_seconds
            started = time.perf_counter()
            try:
                return self.policy.decide(*args, **kwargs)
            finally:
                inference_seconds += time.perf_counter() - started

    if measure and policy_runtime is not None:
        policy_runtime = TimedPolicy(policy_runtime)
    try:
        recorder = EpisodeRecording(
            env.runtime,
            bucket=bucket,
            root=root,
            prefix=prefix,
            chunk_bytes=chunk_bytes,
            reserve=reserve,
            guard=guard,
        )
        env.runtime.recording = recorder
        expected_actions = provenance.get("action_contract")
        if expected_actions is not None:
            assert_action_contract_compatible(expected_actions, env.runtime.action_contract)
        bind_policy_action_space(model, env.action_space, env.runtime.action_contract)
        verify_policy_execution_contract(model, env)
        # Each invocation owns an isolated process RNG. No batched sampling can
        # consume another episode's random stream.
        with torch.random.fork_rng(devices=[]):
            state = np.random.get_state()
            try:
                torch.manual_seed(episode["policy_seed"])
                np.random.seed(episode["policy_seed"])
                result = run_eval_episode(
                    env,
                    model,
                    watchdog_steps,
                    False,
                    episode["environment_seed"],
                    policy_runtime=policy_runtime,
                    semantics=environment_spec(config.env_provider, config.game).eval_semantics,
                )
            finally:
                np.random.set_state(state)
        if not recorder.complete:
            raise ValueError("monitoring episode lacks a scientific boundary")
        facts = {**recorder.last["facts"], **recorder.last["task_metrics"]}
        if "score" not in facts or "bricks_destroyed" not in facts:
            raise ValueError("monitoring requires native score and brick evidence")
        score, bricks = float(facts["score"]), float(facts["bricks_destroyed"])
        normalized = float(facts["bricks_destroyed_normalized"])
        if not math.isclose(normalized, bricks / 216, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError("native normalized brick denominator differs from 216")
        if not math.isfinite(score) or not math.isfinite(bricks) or bricks < 0:
            raise ValueError("invalid native score or brick progress")
        from dataclasses import asdict

        recorder.contract["environment"] = json_value(asdict(config))
        document = dict(
            format=FORMAT,
            **provenance,
            **episode,
            complete=True,
            steps=result["steps"],
            shaped_return=result["return"],
            native_score=score,
            bricks_destroyed=bricks,
            brick_denominator=216,
            normalized_brick_progress=bricks / 216,
            success=result["outcome"] == "success",
            outcome=result["outcome"],
            terminated=result["terminated"],
            truncated=result["truncated"],
            start_facts=recorder.initial,
            recording_contract=recorder.contract,
            chunks=recorder.chunks,
        )
        if measure:
            document["phase_seconds"] = {**recorder.phase_seconds, "inference": inference_seconds}
        data = canonical_json_bytes(document)
        if reserve:
            reserve(f"{prefix}/episode.json", len(data))
        verified_put(bucket, f"{prefix}/episode.json", data)
        return document
    finally:
        env.close()


def episode_frames(bucket, episode, *, guard=None):
    """Reconstruct one complete trajectory with exact frame/transition joins."""
    step, previous = 0, None
    for reference in episode["chunks"]:
        if reference["first_step"] != step or reference["end_step"] <= step:
            raise ValueError("noncontiguous monitoring chunks")
        if previous is not None and reference["first_frame_sha256"] != previous:
            raise ValueError("monitoring frame join mismatch")
        with zipfile.ZipFile(io.BytesIO(verified_get(bucket, reference))) as archive:
            rows = [json.loads(row) for row in archive.read("transitions.jsonl").splitlines()]
            if [row["step"] for row in rows] != list(range(step, reference["end_step"])):
                raise ValueError("monitoring transition indices are not contiguous")
            for index in range(step, reference["end_step"] + 1):
                if guard is not None:
                    guard()
                payload = archive.read(f"frames/{index}.png")
                if index in (step, reference["end_step"]):
                    expected = (
                        reference["first_frame_sha256"]
                        if index == step
                        else reference["last_frame_sha256"]
                    )
                    if hashlib.sha256(payload).hexdigest() != expected:
                        raise ValueError("monitoring frame hash mismatch")
                if index == step and previous is not None:
                    continue
                with Image.open(io.BytesIO(payload)) as image:
                    if image.mode != "RGB" or image.size != (160, 210):
                        raise ValueError("monitoring frame is not native RGB")
                    yield np.array(image)
            step = reference["end_step"]
            previous = reference["last_frame_sha256"]
    if step != episode["steps"]:
        raise ValueError("monitoring episode length differs from chunks")


def verify_monitoring_inventory(bucket, result, manifest, *, prefix):
    """Shared worker/supervisor/terminal validation of every complete episode."""
    metrics, selection, _ = monitoring_aggregates(result["episodes"], manifest)
    if metrics != result["metrics"] or selection != result["selection"]:
        raise ValueError("monitoring aggregate or representative selection mismatch")
    for episode in result["episodes"]:
        if any(
            episode.get(key) != result.get(key)
            for key in ("evaluation_id", "contract_sha256", "checkpoint_id", "checkpoint_step")
        ):
            raise ValueError("monitoring episode belongs to another evaluation")
        step, previous = 0, None
        for chunk in episode["chunks"]:
            if (
                chunk["first_step"] != step
                or chunk["end_step"] <= step
                or not 0 < chunk["bytes"] <= 128 * 1024**2
                or previous is not None
                and chunk["first_frame_sha256"] != previous
            ):
                raise ValueError("monitoring episode has noncontiguous chunks or frame joins")
            if not chunk["key"].startswith(prefix + "/"):
                raise ValueError("monitoring chunk outside evaluation namespace")
            head = bucket.head(chunk["key"])
            digest = head["etag"] if bucket.scheme == "file" else head["metadata"].get("sha256")
            if head["size"] != chunk["bytes"] or digest != chunk["sha256"]:
                raise ValueError("monitoring durable chunk identity mismatch")
            step, previous = chunk["end_step"], chunk["last_frame_sha256"]
        if not episode["chunks"] or step != episode["steps"]:
            raise ValueError("monitoring episode has incomplete chunks")
    video = result["video"]
    if video["key"] != prefix + "/representative.mp4":
        raise ValueError("monitoring video outside evaluation namespace")
    head = bucket.head(video["key"])
    digest = head["etag"] if bucket.scheme == "file" else head["metadata"].get("sha256")
    if head["size"] != video["bytes"] or digest != video["sha256"]:
        raise ValueError("monitoring durable video identity mismatch")


def monitoring_aggregates(episodes, manifest):
    by_id = {row["episode_id"]: row for row in episodes}
    if len(by_id) != len(episodes) or set(by_id) != {e["episode_id"] for e in manifest}:
        raise ValueError("monitoring requires the complete manifest, without duplicates")
    ordered = []
    for entry in manifest:
        row = by_id[entry["episode_id"]]
        if not row.get("complete") or any(row.get(k) != v for k, v in entry.items()):
            raise ValueError("monitoring result differs from the complete manifest")
        if type(row["success"]) is not bool or row["brick_denominator"] != 216:
            raise ValueError("monitoring result has invalid outcome/progress evidence")
        for name in ("normalized_brick_progress", "native_score", "shaped_return", "steps"):
            if isinstance(row[name], bool) or not math.isfinite(row[name]):
                raise ValueError("monitoring requires finite complete episode facts")
        if not math.isclose(
            row["normalized_brick_progress"],
            row["bricks_destroyed"] / 216,
            rel_tol=1e-6,
            abs_tol=1e-8,
        ):
            raise ValueError("monitoring normalized progress denominator mismatch")
        ordered.append(row)
    if not ordered:
        raise ValueError("monitoring requires a nonempty complete manifest")
    if len({(r["checkpoint_id"], r["checkpoint_step"]) for r in ordered}) != 1:
        raise ValueError("monitoring cannot mix checkpoints")
    n = len(ordered)
    rate = sum(r["success"] for r in ordered) / n
    z = 1.959963984540054
    divisor = 1 + z * z / n
    center = (rate + z * z / (2 * n)) / divisor
    radius = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / divisor
    progress = [r["normalized_brick_progress"] for r in ordered]
    median = statistics.median(progress)
    selected = min(
        ordered, key=lambda r: (abs(r["normalized_brick_progress"] - median), r["ordinal"])
    )
    metrics = {
        "success/rate": rate,
        "success/ci95/lower": max(0.0, center - radius),
        "success/ci95/upper": min(1.0, center + radius),
        "progress/mean": statistics.mean(progress),
        "progress/median": median,
        "score/mean": statistics.mean(r["native_score"] for r in ordered),
        "return/mean": statistics.mean(r["shaped_return"] for r in ordered),
        "episode_steps/mean": statistics.mean(r["steps"] for r in ordered),
        "episodes/count": n,
    }
    selection = dict(
        rule="nearest-median-normalized-bricks-then-ordinal-v1",
        metric="normalized_brick_progress",
        median=median,
        selected_value=selected["normalized_brick_progress"],
        episode_id=selected["episode_id"],
        ordinal=selected["ordinal"],
    )
    return {f"eval/monitor/{k}": v for k, v in metrics.items()}, selection, selected


def finalize_monitoring(
    episodes,
    manifest,
    *,
    bucket,
    root,
    prefix,
    fps,
    reserve=None,
    guard=None,
    measurements=None,
    max_video_bytes=128 * 1024**2,
):
    from gradlab.video import write_video

    metrics, selection, selected = monitoring_aggregates(episodes, manifest)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    output = root / "representative.mp4"
    key = f"{prefix}/representative.mp4"
    saved = bucket.get_json_optional(key + ".json")
    video_started = time.perf_counter()
    if saved is not None:
        if saved["selection"] != selection:
            raise ValueError("committed monitoring video selection changed")
        video = saved["video"]
        verified_get(bucket, video)
    else:

        def video_guard():
            if guard is not None:
                guard()
            if output.exists() and output.stat().st_size > max_video_bytes:
                raise OSError("complete monitoring video exceeds the declared working allocation")

        write_video(
            episode_frames(bucket, selected, guard=video_guard), output, fps=fps, scale=1, threads=1
        )
        video_guard()
        data = output.read_bytes()
        if reserve:
            reserve(key, len(data))
        video = verified_put(bucket, key, data)
        pointer = canonical_json_bytes(dict(selection=selection, video=video))
        if reserve:
            reserve(key + ".json", len(pointer))
        verified_put(bucket, key + ".json", pointer)
        output.unlink()
    video_seconds = time.perf_counter() - video_started
    result = dict(
        format=FORMAT,
        **{key: selected[key] for key in ("evaluation_id", "contract_sha256") if key in selected},
        checkpoint_id=selected["checkpoint_id"],
        checkpoint_step=selected["checkpoint_step"],
        metrics=metrics,
        selection=selection,
        video=video,
        episodes=episodes,
    )
    if measurements is not None:
        result["measurements"] = measurements()
        result["measurements"]["phase_seconds"]["video_generation"] = video_seconds
    data = canonical_json_bytes(result)
    if reserve:
        reserve(f"{prefix}/result.json", len(data))
    verified_put(bucket, f"{prefix}/result.json", data)
    return result
