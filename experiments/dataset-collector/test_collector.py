from types import SimpleNamespace

import numpy as np
import pytest

from collector import Collection, DatasetReader, Limits, validate_dataset


class ScriptedExecution:
    """The Policy/environment boundary, including an intentionally reused RGB buffer."""

    contract = {"provider": "scripted", "frame_skip": 2}
    provenance = {"checkpoint": "test-checkpoint"}
    action_selection_mode = "stochastic"
    supports_temperature = True

    def __init__(self):
        self.image = np.zeros((210, 160, 3), dtype=np.uint8)
        self.steps = 0

    def reset(self, seed):
        self.steps = 0
        self.image.fill(0)
        self.image[0, 0] = [1, 2, 3]  # HUD pixels must survive.
        return self.image

    def step(self, temperature):
        self.steps += 1
        if self.steps == 2:
            self.image[100, 50] = [4, 5, 6]
        return self.image, {
            "selected_action": np.array([self.steps], dtype=np.int16),
            "effective_action": 0,
            "native_action": 0,
            "action_override_rule_id": "serve",
            "policy_reward": 0.5,
            "native_reward": 1.0,
            "terminated": self.steps == 2,
            "truncated": False,
            "native_game_over": self.steps == 2,
            "task_terminated": False,
            "labels": {"ball": np.array([12, 13], dtype=np.float32)},
            "elapsed_native_frames": None,
        }

    def close(self):
        pass


def test_collection_reuses_exact_rgb_without_losing_transition_occurrences(tmp_path):
    with Collection(tmp_path / "data", ScriptedExecution(), limits=Limits(max_steps=4)) as run:
        run.run()
    with DatasetReader(tmp_path / "data") as reader:
        episodes = list(reader.episodes())
        assert len(episodes) == 2
        rows = list(reader.transitions(episodes[0]["episode_id"]))
        assert len(rows) == 2
        assert rows[0]["source_frame_id"] == rows[0]["successor_frame_id"]
        assert rows[1]["source_frame_id"] == rows[0]["successor_frame_id"]
        np.testing.assert_array_equal(reader.frame(rows[0]["source_frame_id"])[0, 0], [1, 2, 3])
        np.testing.assert_array_equal(
            reader.frame(rows[1]["successor_frame_id"])[100, 50], [4, 5, 6]
        )
        assert rows[0]["selected_action"].dtype == np.int16
        assert rows[0]["labels"]["ball"].dtype == np.float32
        assert rows[0]["elapsed_native_frames"] is None
        progress = reader.progress()
        assert progress["unique_frames"] == 2
        assert progress["captured_occurrences"] == 6
        assert progress["transitions"] == 4
        assert progress["reuse_fraction"] == 1 - 2 / 6
    assert validate_dataset(tmp_path / "data")["valid"]


def test_temperature_blocks_are_seeded_and_reject_ineffective_exploration(tmp_path):
    import pytest
    from collector import TemperatureSchedule

    class LongExecution(ScriptedExecution):
        def step(self, temperature):
            image, facts = super().step(temperature)
            facts["terminated"] = facts["native_game_over"] = False
            return image, facts

    settings = TemperatureSchedule(
        enabled=True, values=(0.75, 1.25), probabilities=(0.5, 0.5), block_decisions=2, seed=0
    )
    for name in ("a", "b"):
        with Collection(
            tmp_path / name, LongExecution(), limits=Limits(max_steps=16), schedule=settings
        ) as run:
            run.run()
    with DatasetReader(tmp_path / "a") as a, DatasetReader(tmp_path / "b") as b:
        temperatures = [r["temperature"] for r in a.transitions(1)]
        assert temperatures == [r["temperature"] for r in b.transitions(1)]
        assert set(temperatures) == {0.75, 1.25}
        assert all(temperatures[i] == temperatures[i + 1] for i in (0, 2, 4, 6))
    execution = LongExecution()
    execution.action_selection_mode = "deterministic"
    with pytest.raises(ValueError, match="stochastic"):
        Collection(tmp_path / "invalid", execution, schedule=settings)
    assert execution.steps == 0


@pytest.mark.parametrize("missing_terminal", [False, True])
def test_policy_execution_captures_terminal_rgb_before_autoreset(tmp_path, missing_terminal):
    from collector import PolicyExecution

    class AutoResetEnv:
        reset_infos = [{}]

        def seed(self, seed):
            pass

        def reset(self):
            self.rgb = np.zeros((210, 160, 3), np.uint8)
            return np.zeros((1, 4, 84, 84), np.uint8)

        def get_images(self):
            return [self.rgb]

        def step(self, actions):
            self.rgb.fill(99)  # The *next* episode must never become the terminal frame.
            return self.reset_obs(), [0.5], [True], [{}]

        def reset_obs(self):
            return np.zeros((1, 4, 84, 84), np.uint8)

        def take_step_diagnostics(self):
            return SimpleNamespace(
                terminal_frame=None if missing_terminal else np.full((210, 160, 3), 7, np.uint8),
                provider_info={"ball_x": np.int16(11)},
                task_metrics={},
                policy_action=1,
                effective_policy_action=0,
                native_action=0,
                action_override_rule_id="serve",
                provider_reward=1.0,
                provider_terminated=True,
                provider_truncated=False,
                task_reward=0.5,
                task_terminated=False,
                task_truncated=False,
                reward=0.5,
                terminated=True,
                truncated=False,
                events=(),
                outcome=0,
                event_transitions={},
                episode_seed=0,
                start_id="default",
            )

        def drain_records(self):
            return []

        def close(self):
            pass

    class Runtime:
        model = SimpleNamespace(observation_space=None, use_sde=False)
        capabilities = SimpleNamespace(default_action_selection_mode="stochastic")
        supports_sampling_temperature = True

        def reset(self):
            pass

        def decide(self, observation, **kwargs):
            return SimpleNamespace(
                actions=np.array([1]), decisions=(SimpleNamespace(raw_action=np.array(1)),)
            )

    execution = PolicyExecution(
        Runtime(),
        AutoResetEnv(),
        SimpleNamespace(frame_skip=2),
        contract={"frame_skip": 2},
        provenance={},
    )
    if missing_terminal:
        with pytest.raises(ValueError, match="terminal RGB"):
            with Collection(tmp_path / "data", execution, limits=Limits(max_steps=1)) as run:
                run.run()
        with DatasetReader(tmp_path / "data") as reader:
            assert reader.episode(1)["status"] == "incomplete"
            assert reader.episode(1)["length"] == 0
        assert validate_dataset(tmp_path / "data")["valid"]
        return
    with Collection(tmp_path / "data", execution, limits=Limits(max_steps=1)) as run:
        run.run()
    with DatasetReader(tmp_path / "data") as reader:
        row = reader.transition(1, 0)
        assert np.all(reader.frame(row["successor_frame_id"]) == 7)
        assert row["native_game_over"] is True
        assert row["native_reward"] == 1.0
        assert row["policy_reward"] == 0.5
        assert row["effective_action"] == 0
        assert row["labels"]["ball_x"].dtype == np.int16


def test_debug_pause_and_readback_use_committed_disk_images(tmp_path):
    import pytest
    from collector import DebugController

    execution = ScriptedExecution()
    with Collection(tmp_path / "data", execution, limits=Limits(max_steps=3)) as run:
        debug = DebugController(run)
        for _ in range(20):
            assert debug.tick() is None
        assert execution.steps == 0
        debug.command("step")
        first = debug.tick()
        assert first["pixel_equal"] is True
        for _ in range(20):
            assert debug.tick()["row"]["step"] == 0
        assert execution.steps == 1
        shard = next((tmp_path / "data").glob("rgb-*.bin"))
        shard.write_bytes(b"corrupt")
        with pytest.raises((ValueError, __import__("zlib").error)):
            debug.tick()


def test_crash_recovery_keeps_committed_prefix_and_never_reuses_seed(tmp_path):
    from collector import Inspector

    root = tmp_path / "data"
    # Simulate process loss without Collection's graceful exit.
    run = Collection(root, ScriptedExecution(), limits=Limits(max_steps=10, batch_steps=1))
    run.step()
    run.writer.close()
    run.execution.close()
    with Collection(
        root, ScriptedExecution(), limits=Limits(max_steps=2, batch_steps=1)
    ) as resumed:
        resumed.run()
    with DatasetReader(root) as reader:
        episodes = list(reader.episodes())
        assert [(e["seed"], e["status"], e["length"]) for e in episodes] == [
            (0, "incomplete", 1),
            (1, "complete", 2),
        ]
        assert episodes[0]["end_reason"] == "interrupted"
        assert reader.progress()["unique_frames"] == 2
    with Inspector(root) as inspector:
        assert inspector.snapshot()["row"] is None
        inspector.move(1)
        assert inspector.snapshot()["row"]["step"] == 0
        inspector.move_episode(True)
        assert inspector.snapshot()["episode"]["seed"] == 1
        inspector.move(1)
        inspector.move(-1)
        assert inspector.snapshot()["row"] is None
    assert validate_dataset(root)["valid"]


def test_exclusive_writer_and_incompatible_append_are_rejected(tmp_path):
    import pytest

    root = tmp_path / "data"
    with Collection(root, ScriptedExecution(), limits=Limits(max_steps=1)) as run:
        with pytest.raises(ValueError, match="active writer"):
            Collection(root, ScriptedExecution())
        run.run()
    different = ScriptedExecution()
    different.contract = {**different.contract, "frame_skip": 4}
    with pytest.raises(ValueError, match="incompatible"):
        Collection(root, different)
    assert validate_dataset(root)["transitions"] == 1


def test_uncommitted_batch_is_invisible_and_allocated_seed_survives(tmp_path, monkeypatch):
    import collector
    import pytest

    root = tmp_path / "data"
    run = Collection(root, ScriptedExecution(), limits=Limits(max_steps=3, batch_steps=1))
    run.step()
    original = collector.os.fsync
    monkeypatch.setattr(
        collector.os, "fsync", lambda root: (_ for _ in ()).throw(OSError("power loss"))
    )
    with pytest.raises(OSError, match="power loss"):
        run.step()
    run.writer.close()
    run.execution.close()
    monkeypatch.setattr(collector.os, "fsync", original)
    assert validate_dataset(root)["transitions"] == 1
    with Collection(root, ScriptedExecution(), limits=Limits(max_steps=1)) as resumed:
        resumed.run()
    with DatasetReader(root) as reader:
        assert [e["seed"] for e in reader.episodes()] == [0, 1]
        assert reader.progress()["transitions"] == 2


def test_disk_limit_includes_files_outside_committed_index(tmp_path):
    import pytest

    root = tmp_path / "data"
    with Collection(root, ScriptedExecution(), limits=Limits(max_steps=1)) as run:
        run.run()
    (root / "uncommitted.tmp").write_bytes(b"x" * 1024**2)
    with pytest.raises(ValueError, match="disk limit"):
        Collection(root, ScriptedExecution(), limits=Limits(max_bytes=2 * 1024**2))
    # Constructor failure must release the writer lock too.
    with Collection(root, ScriptedExecution(), limits=Limits(max_steps=1)) as run:
        run.run()
    assert validate_dataset(root)["transitions"] == 2


def test_full_game_override_only_changes_episode_boundaries():
    from collector import collection_environment

    original = {
        "env_provider": "env-breakoutatari2600-turbo-native",
        "game": "Breakout-Atari2600-v0",
        "frame_skip": 2,
        "task": {
            "termination": {
                "success": ["wall"],
                "failure": ["life"],
                "timeout": ["stall"],
                "max_episode_steps": 50,
            },
            "reward": {"reward_clip": True},
            "observation": {"context": ["ball"]},
        },
    }
    faithful = collection_environment(original, full_game=False, episode_steps=None)
    assert faithful == original
    effective = collection_environment(original, full_game=True, episode_steps=1000)
    assert effective["task"]["termination"] == {"max_episode_steps": 1000}
    assert effective["frame_skip"] == 2
    assert effective["task"]["reward"] == original["task"]["reward"]
    assert effective["task"]["observation"] == original["task"]["observation"]
    assert original["task"]["termination"]["success"] == ["wall"]


def test_progress_separates_novelty_compression_growth_and_pending(tmp_path):
    root = tmp_path / "data"
    with Collection(root, ScriptedExecution(), limits=Limits(max_steps=3)) as run:
        before = run.progress()
        assert before["reuse_fraction"] is None
        run.step()
        pending = run.progress()
        assert pending["transitions"] == 0
        assert pending["pending_transitions"] == 1
        assert pending["pending_captured_occurrences"] == 1
        run.step()
        progress = run.progress()
        assert progress["transitions"] == 2
        assert progress["captured_occurrences"] == 3
        assert progress["unique_frames"] == 2
        assert progress["compression_fraction_saved"] > 0.9
        assert (
            progress["actual_bytes"] >= progress["compressed_rgb_bytes"] + progress["index_bytes"]
        )
        assert progress["recent_new_images_per_second"] >= 0
        assert progress["projected_bytes_per_hour"] >= 0
    assert (root / "progress.json").is_file()


def test_native_breakout_recording_and_visible_controls_preserve_seeded_policy_prefix(
    tmp_path, monkeypatch
):
    from copy import deepcopy
    from dataclasses import replace
    from pathlib import Path
    import torch
    from gradlab.actor_critic_policy import SharedActorCriticPolicy
    from gradlab.env import make_eval_vec_env, resolve_env_config
    from gradlab.env_config import env_config_from_mapping
    from gradlab.policy_runtime import PolicyRuntime
    from gradlab.recipe_documents import compose_train_document
    from collector import DebugController, PolicyExecution

    goal = Path("experiments/goals/Breakout-Atari2600-v0")
    train = compose_train_document(goal / "_goal.yaml", goal / "recipes/ppo-ball-state.yaml")[
        "train_config"
    ]
    config = resolve_env_config(env_config_from_mapping(train))
    config = replace(config, task={**config.task, "termination": {"max_episode_steps": 8}})
    plain_env = make_eval_vec_env(config, 1, 0, capture_step_diagnostics=True)
    torch.manual_seed(0)
    policy = SharedActorCriticPolicy(
        plain_env.observation_space,
        plain_env.action_space,
        lambda _: 1e-3,
        policy_model={
            "schema_version": 2,
            "encoder": {"kind": "flatten"},
            "fusion": {"hidden_sizes": [8], "activation": "relu"},
            "normalize_images": False,
            "orthogonal_init": True,
        },
    )
    model = SimpleNamespace(policy=policy, observation_space=plain_env.observation_space)
    runtime = PolicyRuntime(model, algorithm_id="ppo")
    baseline = []
    torch.manual_seed(0)
    np.random.seed(0)
    plain_env.seed(0)
    obs = plain_env.reset()
    runtime.reset()
    try:
        for _ in range(8):
            inputs = deepcopy(obs)
            decision = runtime.decide(obs, include_diagnostics=False)
            obs, reward, done, _ = plain_env.step(decision.actions)
            diagnostic = plain_env.take_step_diagnostics()
            rgb = diagnostic.terminal_frame if done[0] else plain_env.get_images()[0]
            baseline.append(
                (inputs, decision.actions.copy(), float(reward[0]), bool(done[0]), rgb.copy())
            )
            plain_env.drain_records()
    finally:
        plain_env.close()
    ordinary_decide = runtime.decide
    observed_inputs = []

    def observe_inputs(observation, **kwargs):
        observed_inputs.append(deepcopy(observation))
        return ordinary_decide(observation, **kwargs)

    monkeypatch.setattr(runtime, "decide", observe_inputs)
    for mode in ("headless", "visible"):
        observed_inputs.clear()
        env = make_eval_vec_env(config, 1, 0, capture_step_diagnostics=True)
        execution = PolicyExecution(
            runtime, env, config, contract={"frame_skip": config.frame_skip}, provenance={}
        )
        with Collection(tmp_path / mode, execution, limits=Limits(max_steps=8)) as run:
            if mode == "headless":
                run.run()
            else:
                debug = DebugController(run)
                for _ in range(8):
                    before = torch.random.get_rng_state().clone()
                    for _ in range(3):
                        debug.tick()
                    assert torch.equal(before, torch.random.get_rng_state())
                    debug.command("step")
                    assert debug.tick()["pixel_equal"] is True
        with DatasetReader(tmp_path / mode) as reader:
            rows = list(reader.transitions(1))
            assert len(rows) == 8
            for row, (_, actions, reward, done, rgb) in zip(rows, baseline):
                assert row["policy_action"] == actions[0]
                assert row["policy_reward"] == reward
                assert (
                    row["terminated"] or row["truncated"]
                    if done
                    else not (row["terminated"] or row["truncated"])
                )
                np.testing.assert_array_equal(reader.frame(row["successor_frame_id"]), rgb)
            assert rows[-1]["task_truncated"] is True
            assert rows[-1]["native_game_over"] is False
        for actual, (expected, *_) in zip(observed_inputs, baseline):
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key])
        assert validate_dataset(tmp_path / mode)["valid"]


def test_repeated_frame_workload_has_bounded_memory_and_retains_all_steps(tmp_path):
    import tracemalloc

    class Repeating(ScriptedExecution):
        def step(self, temperature):
            self.steps += 1
            return self.image, {
                "selected_action": self.steps % 3,
                "terminated": False,
                "truncated": False,
                "labels": {"history": self.steps},
                "policy_reward": self.steps % 2,
            }

    tracemalloc.start()
    try:
        with Collection(
            tmp_path / "data", Repeating(), limits=Limits(max_steps=1500, batch_steps=32)
        ) as run:
            run.run()
        _, peak = tracemalloc.get_traced_memory()
        assert peak < 32 * 1024**2
    finally:
        tracemalloc.stop()
    with DatasetReader(tmp_path / "data") as reader:
        assert reader.progress()["unique_frames"] == 1
        assert reader.progress()["captured_occurrences"] == 1501
        assert reader.transition(1, 1499)["labels"]["history"] == 1500
        assert reader.episode(1)["status"] == "incomplete"
    assert validate_dataset(tmp_path / "data")["transitions"] == 1500


def test_split_seeds_and_dataset_only_cli_survive_resume(tmp_path, monkeypatch, capsys):
    from collector import main
    import gradlab.policy_models

    root = tmp_path / "data"
    for _ in range(2):
        with Collection(root, ScriptedExecution(), limits=Limits(max_steps=6)) as run:
            run.run()
    monkeypatch.setattr(
        gradlab.policy_models, "load_policy_model", lambda *a, **kw: pytest.fail("offline loading")
    )
    with DatasetReader(root) as reader:
        assert [(e["split"], e["seed"]) for e in reader.episodes()] == [
            ("train", 0),
            ("train", 1),
            ("train", 2),
            ("train", 3),
            ("heldout", 10000),
            ("train", 4),
        ]
    assert main(["validate", str(root)]) == 0
    assert '"valid": true' in capsys.readouterr().out
    step_file = next(root.glob("steps-*.parquet"))
    step_file.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        main(["validate", str(root)])


def test_write_failure_preserves_error_and_last_committed_prefix(tmp_path, monkeypatch):
    import collector

    root = tmp_path / "data"
    run = Collection(root, ScriptedExecution(), limits=Limits(max_steps=3, batch_steps=1))
    with pytest.raises(OSError, match="storage offline"):
        with run:
            run.step()
            monkeypatch.setattr(
                collector.os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("storage offline"))
            )
            run.step()
    monkeypatch.undo()
    with Collection(root, ScriptedExecution(), limits=Limits(max_steps=1)) as resumed:
        resumed.run()
    assert validate_dataset(root)["transitions"] == 2


def test_external_ppo_loader_fails_before_pickle_execution(tmp_path, monkeypatch):
    from collector import load_execution
    from pathlib import Path
    from gradlab.recipe_documents import compose_resolved_train_documents
    from gradlab.policy_bundle import (
        build_recipe_document,
        build_model_document,
        write_canonical_json,
    )
    from gradlab.training_backend import training_backend_config_hash
    import gradlab.policy_models

    goal = Path("experiments/goals/Breakout-Atari2600-v0")
    resolved = compose_resolved_train_documents(
        goal / "_goal.yaml", goal / "recipes/ppo.yaml", source_sha="a" * 40
    )
    recipe = build_recipe_document(
        resolved.effective,
        repo_root=Path.cwd(),
        source_commit="a" * 40,
        run_description="Collector rejection regression",
        seed=0,
        runtime_packages=("gradlab==0.2.2",),
        base_materialized_recipe=resolved.base,
        canonical_goal=resolved.canonical_goal,
    )
    recipe_path = write_canonical_json(tmp_path / "recipe.json", recipe)
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"never deserialize this test Checkpoint")
    train = recipe["recipe"]["train_config"]
    metadata = {
        "kind": "checkpoint",
        "checkpoint_step": 1,
        "algorithm_id": "ppo",
        "model_class": "gradlab.ppo.GradLabPPO",
        "training_backend_id": train["training_backend"]["id"],
        "training_backend_config_hash": training_backend_config_hash(train),
    }
    write_canonical_json(
        tmp_path / "model.json", build_model_document(checkpoint, recipe_path, metadata)
    )
    monkeypatch.setattr(
        gradlab.policy_models,
        "load_policy_model",
        lambda *a, **kw: pytest.fail("unsafe loader called"),
    )
    with pytest.raises(ValueError, match="Data-Only Policy loader"):
        load_execution(tmp_path)


@pytest.mark.parametrize(
    "settings",
    [
        {"values": (float("nan"),)},
        {"values": (0,)},
        {"probabilities": (-0.2, 0.6, 0.6)},
        {"probabilities": (0.2, 0.6)},
        {"probabilities": (0.2, 0.6, 0.6)},
        {"block_decisions": 0},
        {"seed": -1},
    ],
)
def test_invalid_temperature_configuration_fails_before_collection(settings):
    from collector import TemperatureSchedule

    with pytest.raises(ValueError):
        TemperatureSchedule(**settings)
