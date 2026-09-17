"""Exercise recorder ownership through the actual custom trainer entrypoint."""

import json
import zipfile
from pathlib import Path

import pytest
import torch

from gradlab.env import resolve_env_config
from gradlab.env_config import env_config_from_mapping
from gradlab.file_utils import atomic_write_json
from gradlab.metric_store import MetricStore
from gradlab.recipe_documents import compose_resolved_train_documents
from gradlab.train_config import validate_and_normalize_train_config
from gradlab.training_backend import BackendContext, GracefulStopFlag, load_training_backend
from gradlab.training_lifecycle import TrainingExecutionMode, TrainingExecutionPolicy, TrainingSession
from gradlab.training_metrics import EpisodeMetricsReducer


def capture_context(root, overrides=()):
    goal = Path("experiments/goals/Breakout-Atari2600-v0/FirstWall")
    documents = compose_resolved_train_documents(
        goal / "_goal.yaml", goal / "recipes/ppo.yaml",
        recipe_overrides=[
            "train.timesteps=32", "train.environment.env_config.n_envs=2",
            "train.backend.config.device=cpu", "train.backend.config.execution_profile=sb3-parity",
            "train.backend.config.n_steps=8", "train.backend.config.batch_size=16",
            "train.backend.config.n_epochs=1", "train.trajectory_collection.enabled=true",
            "train.trajectory_collection.sample_probability=1", "train.trajectory_collection.budget_stages=1",
            *overrides,
        ],
    )
    train = validate_and_normalize_train_config(documents.effective["train_config"])
    train.update(seed=123, resolved_n_envs=2, checkpoint_freq=0, checkpoint_eval_backend="none",
                 attempt_id="attempt-" + "b" * 16, wandb_run_id="gradlab-" + "a" * 32)
    train["occupancy"] = None
    env = resolve_env_config(env_config_from_mapping(train))
    env.task["termination"]["max_episode_steps"] = 12
    root.mkdir(parents=True, exist_ok=True)
    store = MetricStore(root / "metrics.sqlite")
    store.init()
    stop = GracefulStopFlag()
    session = TrainingSession(
        run_dir=root, backend_id="gradlab.ppo", metric_store=store, wandb_enabled=False,
        stop_flag=stop, early_stop_config=None, attempt_id=train["attempt_id"],
        run_id=train["wandb_run_id"], reducer=EpisodeMetricsReducer(),
        execution_policy=TrainingExecutionPolicy.for_mode(TrainingExecutionMode.SUPERVISED),
        completion_signal_available=True,
    )
    return BackendContext(train, env, root, root / "checkpoints", store, False, stop, None, session)


@pytest.mark.parametrize("ending", ["complete", "cancel", "fault", "disabled", "exhausted"])
def test_custom_ppo_owns_capture_lifecycle(tmp_path, monkeypatch, ending):
    import gradlab.training.ppo_engine as engine

    torch.set_num_threads(1)
    context = capture_context(tmp_path)
    spool = tmp_path / "trajectories" / context.train_config["attempt_id"]
    if ending == "disabled":
        context.train_config["trajectory_collection"] = None
    else:
        atomic_write_json(spool / "admission.json", {
            "budget_available": ending != "exhausted", "previous_reserved_bytes": 0,
        })
    # Checkpoint packaging is independent of capture; retain real training and session behavior.
    monkeypatch.setattr(engine, "save_model_bundle", lambda **kwargs: None)
    advance = context.session.advance

    def after_transition(step, *args, **kwargs):
        result = advance(step, *args, **kwargs)
        if step == 2 and ending == "cancel":
            context.stop_flag.request("operator_cancel")
        if step == 2 and ending == "fault":
            raise RuntimeError("injected learner fault")
        return result

    monkeypatch.setattr(context.session, "advance", after_transition)
    backend = load_training_backend("gradlab.ppo")
    if ending == "fault":
        with pytest.raises(RuntimeError, match="injected learner fault"):
            backend.run(context)
    else:
        backend.run(context)
    if ending in {"disabled", "exhausted"}:
        assert not (spool / "producer.json").exists()
        assert not list(spool.glob("*.zip"))
        return
    assert (spool / "closed.json").exists(), "trainer must drain its encoder before returning"
    assert not (spool / "fault.json").exists()
    manifests = [json.loads(path.read_text()) for path in spool.glob("*.manifest.json")]
    assert manifests
    rows = []
    for manifest in manifests:
        with zipfile.ZipFile(spool / manifest["file"]) as chunk:
            rows.extend(json.loads(row) for row in chunk.read("transitions.jsonl").splitlines())
    assert any(row["policy_update"] == 0 for row in rows)
    if ending == "complete":
        assert any(row["policy_update"] > 0 for row in rows)
        assert any(m["episode"]["complete"] for m in manifests)
    else:
        assert any(not m["episode"]["complete"] for m in manifests)
        assert all(m["episode"]["interruption_reason"] for m in manifests if not m["episode"]["complete"])
