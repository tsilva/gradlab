"""Monitoring evidence through the real native episode executor."""

import io
import json
import zipfile
from pathlib import Path

import pytest
import numpy as np
from PIL import Image

from gradlab.env import resolve_env_config
from gradlab.env_config import env_config_from_mapping
from gradlab.recipe_documents import compose_train_document
from gradlab.r2_store import BucketConfig, R2Bucket
from gradlab.checkpoint_monitoring import monitor_episode, episode_manifest


class RequestedRight:
    def predict(self, observations, deterministic=False):
        return np.ones(1, dtype=np.int64), None


def test_monitor_records_complete_native_episode_without_acceptance(tmp_path):
    goal = Path("experiments/goals/Breakout-Atari2600-v0/FirstWall")
    train = compose_train_document(goal / "_goal.yaml", goal / "recipes/ppo.yaml")["train_config"]
    config = resolve_env_config(env_config_from_mapping(train))
    config.env_args["noop_reset_max"] = 0
    config.task["termination"]["max_episode_steps"] = 3
    bucket = R2Bucket(BucketConfig(uri=f"file://{tmp_path}/r2"))
    result = monitor_episode(
        model=RequestedRight(),
        config=config,
        episode=episode_manifest(1)[0],
        bucket=bucket,
        root=tmp_path / "spool",
        prefix="monitor/test",
        provenance={"checkpoint_id": "checkpoint-test", "checkpoint_step": 123},
        chunk_bytes=1024**2,
        watchdog_steps=10,
    )
    assert result["complete"] and result["steps"] == 3
    assert result["brick_denominator"] == 216
    assert result["checkpoint_step"] == 123
    assert "accepted" not in result
    assert not list((tmp_path / "spool").glob("*.zip"))
    chunk = result["chunks"][0]
    with zipfile.ZipFile(io.BytesIO(bucket.get_bytes(chunk["key"]))) as archive:
        rows = [json.loads(line) for line in archive.read("transitions.jsonl").splitlines()]
        assert [r["step"] for r in rows] == [0, 1, 2]
        assert rows[0]["policy_action"] == 1
        assert rows[0]["native_action"] == 1  # actual auto-serve FIRE
        assert rows[0]["override_rule"] == "auto_serve"
        assert rows[1]["native_action"] == 2  # requested right
        assert rows[-1]["truncated"] and not rows[-1]["provider_truncated"]
        initial = np.array(Image.open(io.BytesIO(archive.read("frames/0.png"))))
        terminal = np.array(Image.open(io.BytesIO(archive.read("frames/3.png"))))
        assert initial.shape == terminal.shape == (210, 160, 3)
        assert not np.array_equal(initial, terminal)


def test_complete_manifest_metrics_and_video_reuse_committed_frames(tmp_path):
    import pytest
    from gradlab.checkpoint_monitoring import finalize_monitoring

    goal = Path("experiments/goals/Breakout-Atari2600-v0/FirstWall")
    train = compose_train_document(goal / "_goal.yaml", goal / "recipes/ppo.yaml")["train_config"]
    config = resolve_env_config(env_config_from_mapping(train))
    config.env_args["noop_reset_max"] = 0
    config.task["termination"]["max_episode_steps"] = 12
    bucket = R2Bucket(BucketConfig(uri=f"file://{tmp_path}/r2"))
    planned = episode_manifest(2)
    episodes = [
        monitor_episode(
            model=RequestedRight(),
            config=config,
            episode=episode,
            bucket=bucket,
            root=tmp_path / "spool",
            prefix=f"monitor/test/{episode['episode_id']}",
            provenance={"checkpoint_id": "checkpoint-test", "checkpoint_step": 123},
            chunk_bytes=16 * 1024,
            watchdog_steps=20,
        )
        for episode in reversed(planned)
    ]
    with pytest.raises(ValueError, match="complete manifest"):
        finalize_monitoring(
            episodes[:1],
            planned,
            bucket=bucket,
            root=tmp_path / "video",
            prefix="monitor/test",
            fps=15,
        )
    result = finalize_monitoring(
        episodes, planned, bucket=bucket, root=tmp_path / "video", prefix="monitor/test", fps=15
    )
    from gradlab.metric_names import MONITORING_SCALAR_METRICS

    assert set(result["metrics"]) == MONITORING_SCALAR_METRICS
    assert result["metrics"]["eval/episodes/count"] == 2
    assert result["metrics"]["eval/success/mean"] == 0
    assert result["metrics"]["eval/monitor/success/ci95/upper"] == pytest.approx(0.65761977)
    assert result["selection"]["episode_id"] == planned[0]["episode_id"]
    assert result["metrics"]["eval/monitor/progress/median"] == 0
    assert result["video"]["bytes"] > 0
    assert bucket.get_bytes(result["video"]["key"])
    assert not list((tmp_path / "video").glob("*.mp4"))

    from copy import deepcopy
    from gradlab.checkpoint_monitoring import verify_monitoring_inventory

    verify_monitoring_inventory(bucket, result, planned, prefix="monitor/test")
    broken = deepcopy(result)
    nonselected = next(
        e for e in broken["episodes"] if e["episode_id"] != result["selection"]["episode_id"]
    )
    assert len(nonselected["chunks"]) >= 3
    nonselected["chunks"].pop(1)
    with pytest.raises(ValueError, match="noncontiguous"):
        verify_monitoring_inventory(bucket, broken, planned, prefix="monitor/test")


def test_supervisor_retains_all_checkpoints_and_expands_monitoring_after_learner_exit(tmp_path):
    from gradlab.lifecycle_certification import CertificationFixture
    from gradlab.eval_backend import EvalHandle, EvalPoll

    class CPU:
        def __init__(self):
            self.submissions = []

        def submit(self, intent):
            self.submissions.append(intent)
            return EvalHandle("cpu", str(len(self.submissions)))

        def poll(self, handle):
            return EvalPoll("running")

        def cancel(self, handle):
            pass

    fixture = CertificationFixture(tmp_path)
    prepared = fixture.prepare(run_number=85)
    supervisor = prepared.supervisor
    supervisor.evaluation_required = False
    supervisor.eval_admission_closed = True
    supervisor.monitor_backend = CPU()
    supervisor.train_config["checkpoint_monitoring"] = {
        "enabled": True,
        "episodes": 2,
        "active_workers": 1,
        "task_cpus": 3,
        "whole_run_seconds": 3600,
        "contribution_bytes": 1024**3,
        "memory_bytes": 3 * 1024**3,
        "spool_bytes": 3 * 1024**3,
        "worker_memory_bytes": 1024**3,
        "worker_spool_bytes": 1024**3,
        "chunk_bytes": 1024**2,
        "watchdog_steps": 100,
    }
    for step, kind in [(100, "periodic"), (200, "periodic"), (300, "final")]:
        fixture.record_checkpoint(prepared, step=step, kind=kind)
    supervisor.active_iteration()
    assert len(supervisor.monitor_backend.submissions) == 1
    supervisor.drain_iteration()
    assert len(supervisor.monitor_backend.submissions) == 3
    assert [i["checkpoint"]["step"] for i in supervisor.monitor_backend.submissions] == [
        100,
        200,
        300,
    ]
    assert not supervisor.store.evals()
    assert not supervisor.stop_reason


@pytest.mark.parametrize("record_count", [0, 1, 2])
def test_supervisor_drains_monitoring_only_after_metrics_and_media_delivery(tmp_path, record_count):
    from gradlab.lifecycle_certification import CertificationFixture
    from gradlab.eval_backend import EvalHandle, EvalPoll
    from gradlab.checkpoint_monitoring import finalize_monitoring

    fixture = CertificationFixture(tmp_path)
    prepared = fixture.prepare(run_number=86, publish_failures=1)
    supervisor = prepared.supervisor
    supervisor.evaluation_required = False
    supervisor.eval_admission_closed = True
    settings = dict(
        enabled=True,
        episodes=2,
        record_episodes=record_count,
        active_workers=1,
        task_cpus=1,
        whole_run_seconds=3600,
        contribution_bytes=1024**3,
        memory_bytes=1024**3,
        spool_bytes=1024**3,
        worker_memory_bytes=1024**3,
        worker_spool_bytes=1024**3,
        chunk_bytes=1024**2,
        watchdog_steps=100,
    )
    supervisor.train_config["checkpoint_monitoring"] = settings
    goal = Path("experiments/goals/Breakout-Atari2600-v0/FirstWall")
    train = compose_train_document(goal / "_goal.yaml", goal / "recipes/ppo.yaml")["train_config"]
    config = resolve_env_config(env_config_from_mapping(train))
    config.env_args["noop_reset_max"] = 0
    config.task["termination"]["max_episode_steps"] = 3

    class CPU:
        def submit(self, intent):
            episodes = [
                monitor_episode(
                    model=RequestedRight(),
                    config=config,
                    episode=episode,
                    bucket=supervisor.authority.models,
                    root=tmp_path / "worker",
                    prefix=f"{intent['prefix']}/{episode['episode_id']}",
                    provenance=dict(
                        checkpoint_id=intent["checkpoint"]["checkpoint_id"],
                        checkpoint_step=intent["checkpoint"]["step"],
                        evaluation_id=intent["evaluation_id"],
                        contract_sha256=intent["contract_sha256"],
                    ),
                    chunk_bytes=1024**2,
                    watchdog_steps=10,
                )
                for episode in intent["manifest"]
            ]
            self.result = finalize_monitoring(
                episodes,
                intent["manifest"],
                bucket=supervisor.authority.models,
                root=tmp_path / "video",
                prefix=intent["prefix"],
                fps=15,
            )
            return EvalHandle("cpu", intent["evaluation_id"])

        def poll(self, handle):
            return EvalPoll("succeeded", provider_result=self.result)

        def cancel(self, handle):
            pass

    supervisor.monitor_backend = CPU()
    fixture.record_checkpoint(prepared, step=100, kind="final")
    supervisor.active_iteration()
    supervisor.store.append_metrics({"train/return/mean": 1}, step=900, source="train")
    complete = False
    for _ in range(5):
        _, complete = supervisor.drain_iteration()
        if complete:
            break
    assert complete
    events = [e for e in prepared.runtime.wandb_events if e["kind"] == "monitoring"]
    assert len(events) == 1 and events[0]["step"] == 100
    assert bool(events[0]["payload"]["video"]) == bool(record_count)
    assert sum(bool(e["chunks"]) for e in supervisor.monitor_backend.result["episodes"]) == record_count
    assert events[0]["payload"]["metrics"]["eval/episodes/count"] == 2
    assert "eval/pass" not in events[0]["payload"]["metrics"]
    assert supervisor.monitoring.receipt["workers_quiescent"]
    assert supervisor.monitoring.receipt["complete"]
    assert not supervisor.store.evals()


def test_same_host_backend_rejects_unverified_checkpoint_and_quiesces(tmp_path):
    import time
    from gradlab.monitor_backend import SameHostEvalBackend

    backend = SameHostEvalBackend(tmp_path / "workers", BucketConfig(uri=f"file://{tmp_path}/r2"))
    handle = backend.submit(
        {
            "evaluation_id": "a" * 64,
            "execution_attempt": 1,
            "deadline": time.time() + 30,
            "settings": {"worker_memory_bytes": 2 * 1024**3},
            "checkpoint": {},
        }
    )
    for _ in range(300):
        result = backend.poll(handle)
        if result.status != "running":
            break
        time.sleep(0.1)
    else:
        backend.cancel(handle)
        raise AssertionError("CPU worker failed to quiesce within its deadline")
    assert result.status == "failed"


def test_monitoring_launch_configuration_requires_calibration_and_rejects_old_capture():
    import pytest
    from gradlab.recipe_documents import compose_resolved_train_documents

    goal = Path("experiments/goals/Breakout-Atari2600-v0/FirstWall")
    for setting, message in [
        ("train.checkpoint_monitoring.enabled=true", "calibration"),
        ("train.trajectory_collection.enabled=true", "trajectory_collection"),
    ]:
        with pytest.raises(ValueError, match=message):
            compose_resolved_train_documents(
                goal / "_goal.yaml", goal / "recipes/ppo.yaml", recipe_overrides=[setting]
            )


def test_publication_appends_complete_multichunk_episode_idempotently(tmp_path):
    from tests.test_trajectory_publication import Hub
    from gradlab.trajectory_publication import append_monitoring_episode
    from gradlab.checkpoint_monitoring import FORMAT

    goal = Path("experiments/goals/Breakout-Atari2600-v0/FirstWall")
    train = compose_train_document(goal / "_goal.yaml", goal / "recipes/ppo.yaml")["train_config"]
    config = resolve_env_config(env_config_from_mapping(train))
    config.env_args["noop_reset_max"] = 0
    config.task["termination"]["max_episode_steps"] = 12
    bucket = R2Bucket(BucketConfig(uri=f"file://{tmp_path}/r2"))
    episode = monitor_episode(
        model=RequestedRight(),
        config=config,
        episode=episode_manifest(1)[0],
        bucket=bucket,
        root=tmp_path / "spool",
        prefix="monitor/test",
        provenance={
            "checkpoint_id": "checkpoint-test",
            "checkpoint_step": 123,
            "evaluation_id": "eval-one",
            "run_id": "run-one",
            "training_seed": 123,
        },
        chunk_bytes=16 * 1024,
        watchdog_steps=20,
    )
    assert len(episode["chunks"]) > 1
    api = Hub()
    api.conflict_once = True
    contract = {"format": FORMAT, "execution": episode["recording_contract"]}
    append_monitoring_episode(api, api.read, "user/data", contract, episode, bucket, tmp_path)
    before = dict(api.files)
    api.lose_reply = True
    append_monitoring_episode(api, api.read, "user/data", contract, episode, bucket, tmp_path)
    assert api.files == before
    indexes = [
        json.loads(v)
        for k, v in api.files.items()
        if k.startswith("episodes/") and k.endswith(".json")
    ]
    assert len(indexes) == 1
    assert indexes[0]["steps"] == 12
    assert indexes[0]["checkpoint_step"] == 123
    assert indexes[0]["training_seed"] == 123
    assert all(bucket.get_bytes(c["key"]) for c in episode["chunks"])
    # Identical image bytes from a different checkpoint have distinct table provenance.
    other = {**episode, "evaluation_id": "eval-two", "checkpoint_id": "checkpoint-two"}
    append_monitoring_episode(api, api.read, "user/data", contract, other, bucket, tmp_path)
    assert len([p for p in api.files if p.startswith("episodes/") and p.endswith(".json")]) == 2
    assert len([p for p in api.files if p.startswith("transitions/")]) == 2 * len(episode["chunks"])


def test_calibration_cli_reports_missing_representative_evidence_without_enabling(tmp_path, capsys):
    from gradlab.main import main

    path = tmp_path / "measurements.json"
    path.write_text(json.dumps({"samples": [], "pairs": []}))
    assert main(["monitor", "calibrate", "--measurements", str(path)]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "incomplete"
    assert "representative" in report["reason"]


@pytest.mark.parametrize("backend_id", ["gradlab.ppo", "sb3.ppo", "sb3.a2c"])
@pytest.mark.parametrize("workers", [1, 3])
def test_worker_loads_real_immutable_policy_bundle(backend_id, workers, tmp_path, monkeypatch):
    import time
    import torch
    import gymnasium as gym
    from stable_baselines3 import PPO, A2C
    from gradlab.ppo import GradLabPPO
    from gradlab.env import make_eval_vec_env
    from gradlab.policy_bundle import build_recipe_document, write_canonical_json
    from gradlab.recipe_documents import compose_resolved_train_documents
    from gradlab.artifacts import install_model_bundle
    from gradlab.training_backend import training_backend_config_hash
    from gradlab.lifecycle_certification import CertificationFixture
    from gradlab.monitor_config import MonitoringConfig
    from gradlab.monitor_backend import SameHostEvalBackend
    from dataclasses import asdict

    torch.set_num_threads(1)
    goal = Path("experiments/goals/Breakout-Atari2600-v0/FirstWall")

    def prepare(value):
        from gradlab.training_backend import normalize_training_backend

        common = value["train_config"]
        common["task"]["termination"]["max_episode_steps"] = 20
        common["checkpoint_monitoring"].update(
            enabled=True, episodes=6, record_episodes=3,
            calibration={"status": "measuring", "campaign_id": "e" * 64},
        )
        common["training_backend"] = normalize_training_backend(
            {"id": backend_id, "config": {}}, common_config=common, label="backend"
        )

    resolved = compose_resolved_train_documents(
        goal / "_goal.yaml",
        goal / "recipes/ppo.yaml",
        source_sha="a" * 40,
        prepare_materialized=prepare,
    )
    document = build_recipe_document(
        resolved.effective,
        repo_root=Path.cwd(),
        source_commit="a" * 40,
        seed=7,
        runtime_image_ref="docker:registry.example/train@sha256:" + "b" * 64,
        base_materialized_recipe=resolved.base,
        canonical_goal=resolved.canonical_goal,
    )
    assert document["recipe"]["monitoring"]["manifest"] == episode_manifest(6, 3)
    train = document["recipe"]["train_config"]
    config = resolve_env_config(env_config_from_mapping(train))
    env = make_eval_vec_env(config, n_envs=1, seed=7)
    cls = {"gradlab.ppo": GradLabPPO, "sb3.ppo": PPO, "sb3.a2c": A2C}[backend_id]
    try:
        kind = (
            "MultiInputPolicy"
            if isinstance(env.observation_space, gym.spaces.Dict)
            else "MlpPolicy"
        )
        model = cls(
            kind,
            env,
            n_steps=2,
            device="cpu",
            policy_kwargs={"net_arch": [8]},
            **({"batch_size": 2} if backend_id != "sb3.a2c" else {}),
        )
        path = tmp_path / "checkpoints" / "model_100_steps.zip"
        recipe_path = write_canonical_json(tmp_path / "recipe.json", document)
        install_model_bundle(
            path,
            save_checkpoint=model.save,
            train_config={
                **train,
                "recipe_json_path": str(recipe_path),
                "algorithm_id": "a2c" if backend_id == "sb3.a2c" else "ppo",
                "model_class": f"{cls.__module__}.{cls.__name__}",
                "training_backend_id": backend_id,
                "training_backend_config_hash": training_backend_config_hash(train),
            },
            config=config,
            kind="checkpoint",
            checkpoint_step_value=100,
        )
    finally:
        env.close()
    fixture = CertificationFixture(tmp_path / "storage")
    hashes = {
        k: "a" * 64
        for k in (
            "goal_sha256",
            "recipe_sha256",
            "environment_sha256",
            "evaluation_contract_sha256",
        )
    }
    run_id = "gradlab-" + "a" * 32
    checkpoint = fixture.authority.publish_checkpoint(
        run_id=run_id,
        model_path=path,
        step=100,
        purpose="final",
        contract_hashes=hashes,
        recovery_sidecar={},
    )
    settings = {
        **asdict(MonitoringConfig()),
        "episodes": 6,
        "record_episodes": 3,
        "task_cpus": workers,
        "watchdog_steps": 20000,
        "calibration": {"status": "measuring", "campaign_id": "e" * 64},
    }
    intent = dict(
        settings=settings,
        checkpoint=checkpoint.to_dict(),
        run_id=run_id,
        recipe_sha256="a" * 64,
        evaluation_id="b" * 64,
        prefix=f"monitoring/{run_id}/" + "b" * 64,
        manifest=episode_manifest(6, 3),
        deadline=time.time() + 90,
        attempt_id="attempt-" + "a" * 16,
        contract_sha256="c" * 64,
        source_sha="a" * 40,
        runtime="test",
        goal_sha256="a" * 64,
        environment_sha256="a" * 64,
        goal_variant=None,
        training_seed=7,
        execution_attempt=1,
    )
    backend = SameHostEvalBackend(tmp_path / "workers", fixture.storage.models)
    handle = backend.submit(intent)
    if workers > 1:
        backend.expand([handle], workers)
        assert len(backend._helpers(handle)) == workers - 1
    try:
        for _ in range(900):
            result = backend.poll(handle)
            if result.status != "running":
                break
            time.sleep(0.1)
        assert result.status == "succeeded", result.error
        assert result.provider_result["metrics"]["eval/episodes/count"] == 6
        assert result.provider_result["episodes"][0]["steps"] > 0
        assert sum(bool(e["chunks"]) for e in result.provider_result["episodes"]) == 3
        assert result.provider_result["measurements"]["phase_seconds"]["inference"] > 0
        assert backend.quiescent()
        if workers > 1:
            parallel = result.provider_result
            serial = backend.submit({**intent, "prefix": intent["prefix"] + "-serial"})
            try:
                for _ in range(900):
                    comparison = backend.poll(serial)
                    if comparison.status != "running":
                        break
                    time.sleep(0.1)
                assert comparison.status == "succeeded", comparison.error
                # Exact original RGB, stochastic actions, facts, rewards and
                # boundaries survive different process counts and dispatch order.
                assert [[c["sha256"] for c in e["chunks"]] for e in parallel["episodes"]] == [
                    [c["sha256"] for c in e["chunks"]] for e in comparison.provider_result["episodes"]
                ]
                assert parallel["metrics"] == comparison.provider_result["metrics"]
                assert parallel["selection"] == comparison.provider_result["selection"]
            finally:
                backend.cancel(serial)
    finally:
        backend.cancel(handle)


@pytest.mark.parametrize("record_video", [False, True])
def test_real_wandb_outbox_stages_video_with_checkpoint_axis(tmp_path, record_video):
    import wandb
    from gradlab.video import write_video
    from gradlab.checkpoint_monitoring import verified_put
    from gradlab.metric_store import MetricStore
    from gradlab.wandb_publisher import publish_pending_frames
    from gradlab.wandb_utils import configure_wandb_metrics
    from tests.test_wandb_offline_metrics import _offline_wandb_records, _history_payload

    bucket = R2Bucket(BucketConfig(uri=f"file://{tmp_path}/r2"))
    movie = tmp_path / "video.mp4"
    write_video([np.zeros((210, 160, 3), dtype=np.uint8)] * 3, movie, fps=15, scale=1, threads=1)
    from gradlab.metric_names import MONITORING_SCALAR_METRICS

    reference = verified_put(bucket, "monitoring/test/video.mp4", movie.read_bytes())
    store = MetricStore(tmp_path / "metrics.sqlite")
    store.init()
    store.append_metrics({"train/return/mean": 1}, step=900, source="train")
    result = dict(
        evaluation_id="eval-video",
        checkpoint_step=100,
        metrics={name: (400 if name == "eval/episodes/count" else 0.5)
                 for name in MONITORING_SCALAR_METRICS},
        video=reference if record_video else None,
    )
    store.append_monitoring(result, bucket_uri=bucket.config.uri)
    store.append_monitoring(result, bucket_uri=bucket.config.uri)
    run = configure_wandb_metrics(
        wandb.init(
            project="gradlab-monitor-tests",
            dir=str(tmp_path),
            mode="offline",
            settings=wandb.Settings(silent=True, disable_git=True),
        )
    )
    try:
        assert publish_pending_frames(store, run, limit=10) == 2
    finally:
        run.finish()
    assert store.monitoring_delivered("eval-video")
    history = [
        _history_payload(r)
        for r in _offline_wandb_records(tmp_path)
        if r.WhichOneof("record_type") == "history"
    ]
    rows = [r for r in history if "eval/episodes/count" in r]
    assert len(rows) == 1, history
    if record_video:
        assert json.loads(rows[0]["eval/monitor/video/_type"]) == "video-file"
    else:
        assert "eval/monitor/video/_type" not in rows[0]
    assert MONITORING_SCALAR_METRICS.issubset(rows[0])
    assert rows[0]["eval/step"] == "100"
    assert "eval/pass" not in rows[0]
    assert bool(list(tmp_path.glob("wandb/*/files/media/videos/eval/monitor/*.mp4"))) == record_video


def test_long_episode_crosses_8192_and_reconstructs_true_boundary(tmp_path, monkeypatch):
    from gradlab.checkpoint_monitoring import episode_frames

    goal = Path("experiments/goals/Breakout-Atari2600-v0/FirstWall")
    train = compose_train_document(goal / "_goal.yaml", goal / "recipes/ppo.yaml")["train_config"]
    config = resolve_env_config(env_config_from_mapping(train))
    config.env_args["noop_reset_max"] = 0
    config.task["action"]["conditional_overrides"] = []
    config.task["termination"]["timeout"] = []
    config.task["termination"]["max_episode_steps"] = 8193
    config.task["termination"]["success"] = []
    from dataclasses import replace

    config = replace(config, frame_skip=1)
    import gradlab.env as env_module

    make_native = env_module.make_eval_vec_env
    holder = {}

    def make_env(*args, **kwargs):
        env = make_native(*args, **kwargs)
        holder["env"] = env
        return env

    monkeypatch.setattr(env_module, "make_eval_vec_env", make_env)

    class Noop:
        def predict(self, observations, deterministic=False):
            runtime = holder["env"].runtime
            last = runtime.recording.last
            facts = last["facts"] if last else runtime.reset_infos[0]
            dx = facts["ball_x"] - facts["paddle_x"]
            action = 1 if facts["ball_y"] == 0 else 2 if dx > 1 else 3 if dx < -1 else 0
            return np.array([action], dtype=np.int64), None

    bucket = R2Bucket(BucketConfig(uri=f"file://{tmp_path}/r2"))
    episode = monitor_episode(
        model=Noop(),
        config=config,
        episode=episode_manifest(1)[0],
        bucket=bucket,
        root=tmp_path / "spool",
        prefix="long",
        provenance={"checkpoint_id": "long", "checkpoint_step": 1},
        chunk_bytes=1024**2,
        watchdog_steps=9000,
    )
    assert episode["steps"] == 8193 and episode["complete"] and episode["truncated"]
    assert len(episode["chunks"]) > 1
    assert sum(1 for _ in episode_frames(bucket, episode)) == 8194
    assert not list((tmp_path / "spool").glob("*.zip"))


def test_interrupted_episode_prefix_does_not_poison_retry(tmp_path):
    from gradlab.monitor_worker import ContributionBudget

    goal = Path("experiments/goals/Breakout-Atari2600-v0/FirstWall")
    train = compose_train_document(goal / "_goal.yaml", goal / "recipes/ppo.yaml")["train_config"]
    config = resolve_env_config(env_config_from_mapping(train))
    config.task["termination"]["max_episode_steps"] = 12
    bucket = R2Bucket(BucketConfig(uri=f"file://{tmp_path}/r2"))
    budget = ContributionBudget(tmp_path / "budget", bucket, "test", 10 * 1024**2)

    class Interrupted(RequestedRight):
        calls = 0

        def predict(self, observations, deterministic=False):
            self.calls += 1
            if self.calls == 4:
                raise OSError("transient worker interruption")
            return super().predict(observations, deterministic)

    args = dict(
        config=config,
        episode=episode_manifest(1)[0],
        bucket=bucket,
        root=tmp_path / "spool",
        prefix="retry/episode",
        provenance={"checkpoint_id": "retry", "checkpoint_step": 123},
        chunk_bytes=1024**2,
        watchdog_steps=20,
        reserve=budget.reserve,
    )
    with pytest.raises(OSError, match="transient"):
        monitor_episode(model=Interrupted(), **args)
    assert bucket.get_json_optional("retry/episode/episode.json") is None
    before = list(bucket.iter_keys("retry/episode/chunks"))
    assert before
    result = monitor_episode(model=RequestedRight(), **args)
    assert result["complete"] and result["steps"] == 12
    assert set(before) <= set(bucket.iter_keys("retry/episode/chunks"))


def test_campaign_plan_counterbalances_pairs_and_requires_explicit_measurement_admission():
    from gradlab.monitor_campaign import campaign_plan
    from gradlab.monitor_config import resolve_monitoring, validate_monitoring_allocation

    campaign = dict(
        launch_args=["--recipe-file", "recipe.yaml", "--max-duration", "1h"],
        seeds=[123, 234, 345],
        settings={},
        warmup_updates=2,
        representatives={
            role: {"seed": 123, "checkpoint_step": step}
            for role, step in [
                ("early", 100),
                ("intermediate", 200),
                ("stronger", 300),
                ("long-episode", 300),
            ]
        },
    )
    identity, settings, commands = campaign_plan(campaign)
    assert [(c["seed"], c["enabled"]) for c in commands] == [
        (123, False),
        (123, True),
        (234, True),
        (234, False),
        (345, False),
        (345, True),
    ]
    settings.update(enabled=True, calibration={"status": "measuring", "campaign_id": identity})
    train = dict(
        game="Breakout-Atari2600-v0",
        env_provider="env-breakoutatari2600-turbo-native",
        training_backend={"id": "gradlab.ppo"},
    )
    frozen = resolve_monitoring(settings, train)
    allocation = dict(
        source_sha="a" * 40, image_digest="image", resources={"cpu": 1}, duration=3600
    )
    with pytest.raises(ValueError, match="explicit calibration campaign"):
        validate_monitoring_allocation(frozen, **allocation)
    validate_monitoring_allocation(frozen, **allocation, calibration_campaign=identity)


def test_monitoring_deadline_does_not_extend_on_new_attempt(tmp_path):
    from dataclasses import replace, asdict
    from datetime import datetime, timedelta
    from gradlab.lifecycle_certification import CertificationFixture
    from gradlab.monitor_config import MonitoringConfig
    from gradlab.monitor_supervisor import MonitoringQueue

    fixture = CertificationFixture(tmp_path)
    supervisor = fixture.prepare(run_number=89).supervisor
    supervisor.train_config["checkpoint_monitoring"] = asdict(MonitoringConfig())
    original = MonitoringQueue(supervisor, object()).deadline
    later = datetime.fromisoformat(
        supervisor.manifest.created_at.replace("Z", "+00:00")
    ) + timedelta(hours=1)
    supervisor.manifest = replace(
        supervisor.manifest, created_at=later.isoformat(), attempt_id="attempt-" + "b" * 16
    )
    assert MonitoringQueue(supervisor, object()).deadline == original

@pytest.mark.parametrize("metric", ["eval/pass", "eval/return/max", "leader/return/mean"])
def test_monitoring_rejects_metrics_outside_observational_allowlist(tmp_path, metric):
    from gradlab.metric_store import MetricStore

    store = MetricStore(tmp_path / "metrics.sqlite")
    store.init()
    with pytest.raises(ValueError):
        store.append_monitoring(
            {"metrics": {metric: 1}, "evaluation_id": "invalid",
             "checkpoint_step": 100, "video": {}},
            bucket_uri=f"file://{tmp_path}/r2",
        )


@pytest.mark.parametrize("record_count", [0, 1, 3])
def test_recording_subset_preserves_full_metrics_and_verified_inventory(tmp_path, monkeypatch, record_count):
    from copy import deepcopy
    from gradlab.checkpoint_monitoring import (
        EpisodeRecording, finalize_monitoring, monitoring_aggregates, verify_monitoring_inventory,
    )
    from gradlab.trajectory_publication import selected_episode

    goal = Path("experiments/goals/Breakout-Atari2600-v0/FirstWall")
    train = compose_train_document(goal / "_goal.yaml", goal / "recipes/ppo.yaml")["train_config"]
    config = resolve_env_config(env_config_from_mapping(train))
    config.env_args["noop_reset_max"] = 0
    config.task["termination"]["max_episode_steps"] = 3
    bucket = R2Bucket(BucketConfig(uri=f"file://{tmp_path}/r2"))
    planned = episode_manifest(3, record_count)
    original_frame = EpisodeRecording._frame

    def frame(recorder):
        assert recorder.record, "Unrecorded episodes must not render or encode frames"
        return original_frame(recorder)

    monkeypatch.setattr(EpisodeRecording, "_frame", frame)
    episodes = [monitor_episode(
        model=RequestedRight(), config=config, episode=entry, bucket=bucket,
        root=tmp_path / "spool", prefix=f"monitor/test/{entry['episode_id']}",
        provenance=dict(checkpoint_id="checkpoint-test", checkpoint_step=100,
                        planned_training_steps=200),
        chunk_bytes=1024**2, watchdog_steps=10,
    ) for entry in reversed(planned)]
    assert sum(bool(e["chunks"]) for e in episodes) == record_count
    assert len(list((tmp_path / "r2").rglob("episode.json"))) == 3
    assert sum(selected_episode(e, {}) for e in episodes) == record_count
    # Different scientific outcomes ensure the unrecorded rows affect aggregates
    # and selection still uses the full-set median, not the recorded-set median.
    for episode in episodes:
        episode["bricks_destroyed"] = episode["ordinal"] * 54
        episode["normalized_brick_progress"] = episode["bricks_destroyed"] / 216
    result = finalize_monitoring(episodes, planned, bucket=bucket, root=tmp_path / "video",
                                 prefix="monitor/test", fps=15)
    assert result["metrics"]["eval/episodes/count"] == 3
    assert result["metrics"]["eval/progress/bricks_destroyed_normalized/mean"] == 0.25
    assert result["metrics"]["eval/progress/bricks_destroyed_normalized/max"] == 0.5
    assert result["metrics"]["eval/monitor/progress/median"] == 0.25
    if record_count:
        assert result["selection"]["ordinal"] == (0 if record_count == 1 else 1)
        assert result["video"]["bytes"] > 0
    else:
        assert result["selection"] is result["video"] is None
    assert monitoring_aggregates(episodes[::-1], planned)[0:2] == (result["metrics"], result["selection"])
    verify_monitoring_inventory(bucket, result, planned, prefix="monitor/test")
    repeated = finalize_monitoring(episodes, planned, bucket=bucket, root=tmp_path / "video",
                                   prefix="monitor/test", fps=15)
    assert repeated == result
    if record_count:
        broken = deepcopy(result)
        next(e for e in broken["episodes"] if e["record"])["chunks"] = []
        with pytest.raises(ValueError, match="incomplete chunks"):
            verify_monitoring_inventory(bucket, broken, planned, prefix="monitor/test")
    if record_count == 1:
        broken = deepcopy(result)
        recorded = next(e for e in broken["episodes"] if e["record"])
        next(e for e in broken["episodes"] if not e["record"])["chunks"] = recorded["chunks"]
        with pytest.raises(ValueError, match="unexpected chunks"):
            verify_monitoring_inventory(bucket, broken, planned, prefix="monitor/test")


def test_recording_does_not_change_episode_scientific_results(tmp_path):
    from gradlab.checkpoint_monitoring import EpisodeRecording
    from unittest.mock import patch

    goal = Path("experiments/goals/Breakout-Atari2600-v0/FirstWall")
    train = compose_train_document(goal / "_goal.yaml", goal / "recipes/ppo.yaml")["train_config"]
    config = resolve_env_config(env_config_from_mapping(train))
    config.env_args["noop_reset_max"] = 0
    config.task["termination"]["max_episode_steps"] = 12
    bucket = R2Bucket(BucketConfig(uri=f"file://{tmp_path}/r2"))
    arguments = dict(model=RequestedRight(), config=config, bucket=bucket, root=tmp_path / "spool",
                     provenance=dict(checkpoint_id="test", checkpoint_step=100), watchdog_steps=20)
    recorded = monitor_episode(**arguments, episode=episode_manifest(1, 1)[0], prefix="recorded")
    with patch.object(EpisodeRecording, "_frame", side_effect=AssertionError("must not capture")):
        unrecorded = monitor_episode(**arguments, episode=episode_manifest(1, 0)[0], prefix="unrecorded")
    assert recorded.pop("record") is True
    assert unrecorded.pop("record") is False
    assert recorded.pop("chunks")
    assert unrecorded.pop("chunks") == []
    assert recorded == unrecorded


@pytest.mark.parametrize("value", [-1, 101, True, 0.5, "50"])
def test_reject_invalid_recording_counts(value):
    from gradlab.monitor_config import resolve_monitoring
    with pytest.raises(ValueError, match="record_episodes"):
        resolve_monitoring(dict(episodes=100, record_episodes=value), {})
    with pytest.raises(ValueError, match="record_episodes"):
        episode_manifest(100, value)


def test_breakout_recipe_freezes_disabled_100_50_and_30gb():
    from gradlab.recipe_documents import compose_resolved_train_documents
    from gradlab.monitor_config import resolve_monitoring, calibration_binding
    goal = Path("experiments/goals/Breakout-Atari2600-v0/FirstWall")
    documents = compose_resolved_train_documents(goal / "_goal.yaml", goal / "recipes/ppo.yaml")
    train = documents.effective["train_config"]
    settings = train["checkpoint_monitoring"]
    assert settings["enabled"] is False
    assert settings["episodes"] == 100 and settings["record_episodes"] == 50
    assert settings["contribution_bytes"] == 30_000_000_000
    assert train["checkpoint_eval_backend"] == "none"
    assert train["checkpoint_freq"] == 10_000_000
    assert calibration_binding(train, settings) != calibration_binding(train, {**settings, "record_episodes": 49})
    assert resolve_monitoring(dict(episodes=100), {})["record_episodes"] is None
    assert all(e.get("record", True) for e in episode_manifest(100))


def test_uncalibrated_override_preserves_contract_and_resource_gates():
    from gradlab.recipe_documents import compose_resolved_train_documents
    from gradlab.monitor_config import resolve_monitoring, validate_monitoring_allocation

    goal = Path("experiments/goals/Breakout-Atari2600-v0/FirstWall")
    overrides = [
        "train.environment.preprocessing.frame_skip=1",
        "train.checkpoint_monitoring.enabled=true",
        "train.checkpoint_monitoring.allow_uncalibrated=true",
    ]
    documents = compose_resolved_train_documents(
        goal / "_goal.yaml", goal / "recipes/ppo.yaml", recipe_overrides=overrides
    )
    train = documents.effective["train_config"]
    settings = resolve_monitoring(train["checkpoint_monitoring"], train)
    assert settings["allow_uncalibrated"] is True
    assert settings["calibration"] is None
    assert train["frame_skip"] == 1
    assert train["checkpoint_eval_backend"] == "none"
    allocation = dict(source_sha="a" * 40, image_digest="image", resources={"cpu": 1}, duration=3600)
    validate_monitoring_allocation(settings, **allocation)
    with pytest.raises(ValueError, match="full task CPU"):
        validate_monitoring_allocation(settings, **(allocation | {"resources": {"cpu": 12}}))
    with pytest.raises(ValueError, match="deadline"):
        validate_monitoring_allocation(settings, **(allocation | {"duration": 1}))
    with pytest.raises(ValueError, match="budgets"):
        resolve_monitoring(settings | {"memory_bytes": 1}, train)
    with pytest.raises(ValueError, match="verified native"):
        resolve_monitoring(settings, train | {"env_provider": "unsupported"})
    with pytest.raises(ValueError, match="must be boolean"):
        resolve_monitoring(settings | {"allow_uncalibrated": "true"}, train)
    with pytest.raises(ValueError, match="calibration"):
        resolve_monitoring(settings | {"allow_uncalibrated": False}, train)
