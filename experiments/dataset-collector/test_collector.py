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


class ScriptedVectorExecution(ScriptedExecution):
    n_envs = 3

    def __init__(self):
        self.lanes = [ScriptedExecution() for _ in range(self.n_envs)]
        self.seeds = []
        self.batch_sizes = []

    def reset_lane(self, lane, seed):
        self.seeds.append(seed)
        return self.lanes[lane].reset(seed)

    def step_batch(self, temperatures):
        self.batch_sizes.append(len(temperatures))
        return {lane: self.lanes[lane].step(temp) for lane, temp in temperatures.items()}


def test_hud_mask_reuses_hud_only_changes_and_preserves_provider_and_facts(tmp_path):
    from collector import HUD_MASK

    class HudChanges(ScriptedExecution):
        def reset(self, seed):
            image = super().reset(seed)
            image[:17] = 50
            image[17] = 123
            return image

        def step(self, temperature):
            image, facts = super().step(temperature)
            image[:17] = 50 + self.steps
            image[100, 50] = 0
            return image, facts

    for masked in (False, True):
        execution = HudChanges()
        with Collection(
            tmp_path / str(masked), execution, limits=Limits(max_steps=2), mask_hud=masked
        ) as run:
            run.run()
        assert np.all(execution.image[:17] == 52)
        result = validate_dataset(tmp_path / str(masked))
        assert result["transitions"] == 2
        assert result["captured_occurrences"] == 3
        assert result["unique_frames"] == (1 if masked else 3)
    with DatasetReader(tmp_path / "True") as masked, DatasetReader(tmp_path / "False") as raw:
        assert masked.manifest["contract"]["capture_transform"] == HUD_MASK
        for masked_row, raw_row in zip(masked.transitions(1), raw.transitions(1), strict=True):
            image = masked.frame(masked_row["successor_frame_id"])
            original = raw.frame(raw_row["successor_frame_id"])
            assert image.shape == (210, 160, 3)
            assert not image[:17].any()
            np.testing.assert_array_equal(image[17:], original[17:])
            np.testing.assert_array_equal(masked_row["selected_action"], raw_row["selected_action"])
            assert masked_row["policy_reward"] == raw_row["policy_reward"]
            assert masked_row["terminated"] == raw_row["terminated"]


@pytest.mark.parametrize("first_mask", [False, True])
def test_hud_mask_is_an_immutable_append_contract(tmp_path, first_mask):
    with Collection(
        tmp_path, ScriptedExecution(), limits=Limits(max_steps=2), mask_hud=first_mask
    ) as run:
        run.run()
    with pytest.raises(ValueError, match="incompatible append"):
        Collection(tmp_path, ScriptedExecution(), mask_hud=not first_mask)
    with Collection(
        tmp_path, ScriptedExecution(), limits=Limits(max_steps=1), mask_hud=first_mask
    ) as run:
        run.run()
    assert validate_dataset(tmp_path)["transitions"] == 3


def test_masked_vector_debug_reads_back_masked_terminal_and_initial_frames(tmp_path):
    from collector import DebugController, Inspector, prepare_huggingface

    with Collection(
        tmp_path, ScriptedVectorExecution(), limits=Limits(max_steps=6), mask_hud=True
    ) as run:
        controller = DebugController(run)
        for _ in range(3):
            controller.command("step")
            snapshot = controller.tick()
            assert snapshot["mask_hud"] and snapshot["pixel_equal"]
            assert not snapshot["source"][:17].any()
        assert snapshot["row"]["terminated"]
        assert run.writer.reader.session(run.writer.session_id)["settings"]["mask_hud"]
    with Inspector(tmp_path) as inspector:
        assert inspector.snapshot()["mask_hud"]
        assert not inspector.snapshot()["decoded"][:17].any()
    assert prepare_huggingface(tmp_path)["transitions"] == 6
    card = (tmp_path / "README.md").read_text()
    assert "top 17 rows" in card
    assert "{{CAPTURE_DESCRIPTION}}" not in card
    assert "Full RGB, including HUD pixels, is preserved" not in card


def test_hud_mask_rejects_unrecognized_capture_dimensions():
    from collector import capture_rgb

    with pytest.raises(ValueError, match="210x160"):
        capture_rgb(np.zeros((84, 84, 3), np.uint8), mask_hud=True)


def test_validation_rejects_nonzero_hud_in_a_masked_dataset(tmp_path):
    from collector import DatasetWriter, HUD_MASK

    with DatasetWriter(tmp_path, {"capture_transform": HUD_MASK}, Limits()) as writer:
        writer.session({}, {})
        writer.reserve_episode()
        writer.initial(np.ones((210, 160, 3), np.uint8))
        writer.end_episode("collection_cutoff", complete=False)
    with pytest.raises(ValueError, match="HUD mask contract"):
        validate_dataset(tmp_path)


def test_vector_collection_preserves_lanes_dedup_seeds_and_exact_cutoff(tmp_path):
    execution = ScriptedVectorExecution()
    with Collection(tmp_path, execution, limits=Limits(max_steps=11, batch_steps=4)) as run:
        run.run()
    assert execution.batch_sizes == [3, 3, 3, 2]
    assert execution.seeds == [0, 1, 2, 3, 10000, 4]
    result = validate_dataset(tmp_path)
    assert result["transitions"] == 11
    assert result["unique_frames"] == 2
    assert result["captured_occurrences"] == 17
    assert result["complete_episodes"] == 5
    assert result["incomplete_episodes"] == 1
    with DatasetReader(tmp_path) as reader:
        assert [e["length"] for e in reader.episodes()] == [2, 2, 2, 2, 2, 1]
        for episode in reader.episodes():
            assert [
                r["selected_action"].item() for r in reader.transitions(episode["episode_id"])
            ] == list(range(1, episode["length"] + 1))


def test_vector_restart_closes_every_lane_and_reserves_fresh_seeds(tmp_path):
    with Collection(tmp_path, ScriptedVectorExecution(), limits=Limits(max_steps=3)) as run:
        run.step()
        run.writer.flush()
        run.finished = True  # Simulate exit without graceful episode closure.
    execution = ScriptedVectorExecution()
    with Collection(tmp_path, execution, limits=Limits(max_steps=1)) as run:
        run.run()
    assert execution.seeds == [3]
    with DatasetReader(tmp_path) as reader:
        assert [e["end_reason"] for e in reader.episodes()] == ["interrupted"] * 3 + [
            "collection_limit"
        ]
    assert validate_dataset(tmp_path)["transitions"] == 4


def test_vector_temperature_groups_keep_actions_attached_to_the_correct_lane():
    from collector import VectorPolicyExecution

    calls, resets = [], []

    class Runtime:
        capabilities = SimpleNamespace(algorithm_id="ppo")
        model = SimpleNamespace(use_sde=False)

        def decide(self, obs, **kwargs):
            temperature = kwargs["sampling_temperature"]
            calls.append((obs["image"].shape[0], temperature))
            actions = obs["task"][:, 0] + temperature
            return SimpleNamespace(
                actions=actions, decisions=[SimpleNamespace(raw_action=a) for a in actions]
            )

    runtime = Runtime()

    class Lane(ScriptedExecution):
        def __init__(self, lane):
            self.runtime = runtime
            self.obs = {"image": np.zeros((1, 4, 84, 84), np.uint8), "task": np.array([[lane]])}

        def reset(self, seed, *, reset_policy):
            resets.append((seed, reset_policy))

        def apply(self, actions, raw):
            assert actions.shape == (1,)
            assert actions[0] == raw
            return raw

    execution = VectorPolicyExecution([Lane(i) for i in range(3)])
    execution.reset_lane(0, 0)
    execution.reset_lane(1, 1)
    execution.reset_lane(0, 3)
    assert resets == [(0, True), (1, False), (3, False)]
    assert execution.step_batch({0: 0.75, 1: 1.25, 2: 0.75}) == {0: 0.75, 1: 2.25, 2: 2.75}
    assert calls == [(2, 0.75), (1, 1.25)]


def test_vector_discovery_order_can_differ_from_episode_order(tmp_path):
    class DifferentLanes(ScriptedVectorExecution):
        def step_batch(self, temperatures):
            results = super().step_batch(temperatures)
            # Lane 1 discovers this image on tick 1; lane 0 reaches it on tick 2.
            if self.lanes[1].steps == 1:
                self.lanes[1].image[100, 50] = [4, 5, 6]
            return results

    with Collection(tmp_path, DifferentLanes(), limits=Limits(max_steps=6)) as run:
        run.run()
    assert validate_dataset(tmp_path)["unique_frames"] == 2


def test_many_lanes_fit_a_small_dataset_budget(tmp_path):
    class ManyLanes(ScriptedVectorExecution):
        n_envs = 64

    with Collection(
        tmp_path, ManyLanes(), limits=Limits(max_steps=64, max_bytes=8 * 1024**2)
    ) as run:
        run.run()
    result = validate_dataset(tmp_path)
    assert result["transitions"] == 64
    assert result["incomplete_episodes"] == 64


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
        shard = next((tmp_path / "data").glob("frames-*.parquet"))
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
    train = compose_train_document(goal / "_goal.yaml", goal / "recipes/ppo.yaml")[
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


@pytest.mark.parametrize("explore", [False, True])
@pytest.mark.parametrize("n_envs", [1, 3])
def test_ppo_checkpoint_uses_existing_loader_and_collects_native_rgb(
    tmp_path, explore, n_envs, monkeypatch
):
    from collector import load_execution
    from pathlib import Path
    from gradlab.recipe_documents import compose_resolved_train_documents
    from gradlab.policy_bundle import (
        build_recipe_document,
        build_model_document,
        write_canonical_json,
    )
    from gradlab.training_backend import training_backend_config_hash
    from stable_baselines3 import PPO
    from gradlab.actor_critic_policy import SharedActorCriticPolicy
    from gradlab.env import make_eval_vec_env, resolve_env_config
    from gradlab.env_config import env_config_from_mapping
    from gradlab.policy_execution import compile_policy_execution_contract
    from collector import TemperatureSchedule

    goal = Path("experiments/goals/Breakout-Atari2600-v0")
    resolved = compose_resolved_train_documents(
        goal / "_goal.yaml", goal / "recipes/ppo.yaml", source_sha="a" * 40
    )
    recipe = build_recipe_document(
        resolved.effective,
        repo_root=Path.cwd(),
        source_commit="a" * 40,
        run_description="Collector PPO loading regression",
        seed=0,
        runtime_packages=("gradlab==0.2.2",),
        base_materialized_recipe=resolved.base,
        canonical_goal=resolved.canonical_goal,
    )
    recipe_path = write_canonical_json(tmp_path / "recipe.json", recipe)
    checkpoint = tmp_path / "model.zip"
    train = recipe["recipe"]["train_config"]
    config = resolve_env_config(env_config_from_mapping(train))
    env = make_eval_vec_env(config, 1, 0, capture_step_diagnostics=True)
    try:
        model = PPO(
            SharedActorCriticPolicy,
            env,
            n_steps=2,
            batch_size=2,
            seed=0,
            device="cpu",
            policy_kwargs={
                "policy_model": {
                    "schema_version": 2,
                    "encoder": {"kind": "flatten"},
                    "fusion": {"hidden_sizes": [8], "activation": "relu"},
                    "normalize_images": False,
                    "orthogonal_init": True,
                }
            },
        )
        execution_contract = compile_policy_execution_contract(model, env)
        action_contract = dict(env.runtime.action_contract)
        model.save(checkpoint)
    finally:
        env.close()
    metadata = {
        "kind": "checkpoint",
        "checkpoint_step": 1,
        "algorithm_id": "ppo",
        "model_class": "stable_baselines3.ppo.ppo.PPO",
        "training_metadata": {
            "policy_execution_contract": execution_contract,
            "action_contract": action_contract,
        },
        "training_backend_id": train["training_backend"]["id"],
        "training_backend_config_hash": training_backend_config_hash(train),
    }
    write_canonical_json(
        tmp_path / "model.json", build_model_document(checkpoint, recipe_path, metadata)
    )
    original_bytes = checkpoint.read_bytes()
    schedule = TemperatureSchedule(
        enabled=explore, values=(0.75,), probabilities=(1.0,), block_decisions=2
    )
    execution = load_execution(
        tmp_path, full_game=True, episode_steps=3, schedule=schedule, n_envs=n_envs
    )
    assert execution.action_selection_mode == "stochastic"
    assert execution.contract["frame_skip"] == config.frame_skip
    with Collection(
        tmp_path / "dataset", execution, limits=Limits(max_steps=5 * n_envs), schedule=schedule
    ) as run:
        run.run()
    with DatasetReader(tmp_path / "dataset") as reader:
        rows = list(reader.transitions(1))
        assert len(rows) == 3
        assert {r["temperature"] for r in rows} == ({0.75} if explore else {1.0})
        assert rows[-1]["task_truncated"] is True
        assert rows[-1]["native_game_over"] is False
        assert reader.frame(rows[-1]["successor_frame_id"]).shape == (210, 160, 3)
        session = reader.session(reader.episode(1)["session_id"])
        assert session["provenance"]["checkpoint"]["sha256"]
        assert session["provenance"]["model_sha256"]
        assert session["provenance"]["recipe_sha256"]
    assert checkpoint.read_bytes() == original_bytes
    assert validate_dataset(tmp_path / "dataset")["valid"]
    if n_envs > 1:
        from collector import DebugController, image_hash, record_json

        execution = load_execution(
            tmp_path,
            full_game=True,
            episode_steps=3,
            schedule=schedule,
            n_envs=n_envs,
            environment_workers=2,
        )
        with Collection(
            tmp_path / "debug", execution, limits=Limits(max_steps=5 * n_envs), schedule=schedule
        ) as run:
            debugger = DebugController(run)
            while not run.finished:
                debugger.command("step")
                snapshot = debugger.tick()
                assert snapshot["pixel_equal"]
                debugger.command("pause")
                debugger.tick()

        def trajectory(root):
            with DatasetReader(root) as reader:
                return [
                    (
                        episode["seed"],
                        record_json({**row, "session_id": None}),
                        image_hash(reader.frame(row["successor_frame_id"])),
                    )
                    for episode in reader.episodes()
                    for row in reader.transitions(episode["episode_id"])
                ]

        assert trajectory(tmp_path / "dataset") == trajectory(tmp_path / "debug")
        assert validate_dataset(tmp_path / "debug")["transitions"] == 5 * n_envs
        if not explore:
            from gradlab.policy_runtime import PolicyRuntime

            def sde_runtime(*args, **kwargs):
                runtime = PolicyRuntime(*args, **kwargs)
                runtime.model.use_sde = True
                return runtime

            def unexpected_environment(*args, **kwargs):
                pytest.fail("unsupported vector execution allocated an environment")

            with monkeypatch.context() as patch:
                patch.setattr("gradlab.policy_runtime.PolicyRuntime", sde_runtime)
                patch.setattr("gradlab.env.make_eval_vec_env", unexpected_environment)
                with pytest.raises(ValueError, match="state-dependent exploration"):
                    load_execution(tmp_path, n_envs=64)


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


def test_rejected_transition_does_not_commit_orphan_rgb(tmp_path):
    class BadMetadata(ScriptedExecution):
        def step(self, temperature):
            image, facts = super().step(temperature)
            image[10, 10] = 23
            facts["labels"] = object()
            return image, facts

    root = tmp_path / "data"
    with pytest.raises(ValueError, match="unsupported trajectory"):
        with Collection(root, BadMetadata()) as run:
            run.run()
    result = validate_dataset(root)
    assert result["unique_frames"] == result["captured_occurrences"] == 1
    assert result["transitions"] == 0


def test_counterfactual_classification_and_overrides_remain_visible_offline(tmp_path):
    from collector import DebugController, Inspector

    execution = ScriptedExecution()
    execution.provenance = {"overrides": {"full_game": True, "episode_steps": 100}}
    root = tmp_path / "data"
    with Collection(root, execution, limits=Limits(max_steps=1)) as run:
        debug = DebugController(run)
        debug.command("step")
        snapshot = debug.tick()
        assert snapshot["classification"] == "Counterfactual Playback"
        assert snapshot["overrides"]["full_game"] is True
    with Inspector(root) as inspector:
        assert inspector.snapshot()["classification"] == "Counterfactual Playback"
        assert inspector.snapshot()["overrides"]["episode_steps"] == 100


def test_concurrent_progress_reports_one_committed_prefix(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    ready, read_ready, done = threading.Event(), threading.Event(), threading.Event()
    root = tmp_path / "data"

    class UniqueFrames(ScriptedExecution):
        def step(self, temperature):
            self.steps += 1
            self.image[0, 0] = [self.steps, 0, 0]
            return self.image, {"terminated": False, "truncated": False}

    def write():
        try:
            with Collection(
                root, UniqueFrames(), limits=Limits(max_steps=120, batch_steps=1)
            ) as run:
                run.step()
                ready.set()
                assert read_ready.wait(10)
                run.run()
        finally:
            ready.set()
            done.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(write)
        assert ready.wait(10)
        with DatasetReader(root) as reader:
            read_ready.set()
            samples = 0
            while not done.is_set():
                progress = reader.progress()
                assert progress["captured_occurrences"] == progress["unique_frames"]
                assert progress["transitions"] + 1 == progress["captured_occurrences"]
                assert progress["reuse_fraction"] == 0
                samples += 1
            assert samples > 0
        future.result()


@pytest.mark.parametrize("execution_type", [ScriptedExecution, ScriptedVectorExecution])
def test_native_huggingface_tables_preserve_frames_splits_and_records(tmp_path, execution_type):
    import io
    import json
    import base64
    import pyarrow.parquet as pq
    import yaml
    from PIL import Image
    from gradlab.play_trajectory import decode_tree
    from collector import prepare_huggingface

    root = tmp_path / "data"
    with Collection(
        root, execution_type(), limits=Limits(max_steps=9, batch_steps=2), heldout_every=2
    ) as run:
        run.run()
    before = {p.name: p.read_bytes() for p in root.glob("*.parquet")}
    assert not list(root.glob("*.bin"))
    report = prepare_huggingface(root)
    assert report["transitions"] == 9
    assert before == {p.name: p.read_bytes() for p in root.glob("*.parquet")}
    card = yaml.safe_load((root / "README.md").read_text().split("---")[1])
    configs = {c["config_name"]: c for c in card["configs"]}
    assert set(configs) == {"frames", "transitions", "episodes", "sessions"}

    def rows(config, split):
        paths = next(d["path"] for d in configs[config]["data_files"] if d["split"] == split)
        return pq.read_table([root / name for name in paths]).to_pylist()

    frames = rows("frames", "assets")
    assert len(frames) == 2
    with DatasetReader(root) as reader:
        frame_files = configs["frames"]["data_files"][0]["path"]
        assert reader.progress()["compressed_rgb_bytes"] == sum(
            (root / name).stat().st_size for name in frame_files
        )
        for row in frames:
            rgb = np.asarray(Image.open(io.BytesIO(row["image"]["bytes"])))
            np.testing.assert_array_equal(rgb, reader.frame(row["frame_id"]))
        exported = []
        for split in ("train", "heldout"):
            episodes = rows("episodes", split)
            transitions = rows("transitions", split)
            assert all(reader.episode(e["episode_id"])["split"] == split for e in episodes)
            assert all(reader.episode(r["episode_id"])["split"] == split for r in transitions)
            exported.extend(transitions)
        assert len(exported) == 9
        for row in exported:
            original = reader.transition(row["episode_id"], row["step"])
            assert row["source_frame_id"] == original["source_frame_id"]
            assert row["successor_frame_id"] == original["successor_frame_id"]
            assert row["policy_reward"] == original["policy_reward"]
            assert json.loads(row["selected_action_json"]) == original["selected_action"].tolist()
            tree = json.loads(row["record_json"])
            for leaf in tree["arrays"]:
                leaf["data"] = base64.b64decode(leaf["data"])
            restored = decode_tree(tree)
            assert restored["selected_action"].dtype == np.int16
            assert restored["labels"]["ball"].dtype == np.float32
            np.testing.assert_array_equal(restored["labels"]["ball"], [12, 13])
        assert rows("episodes", "train")[-1]["status"] == "incomplete"
    paths = {f["path"] for f in report["files"]}
    assert "index.sqlite" not in paths
    episode_count = 5 if execution_type is ScriptedExecution else 6
    assert len([p for p in paths if p.startswith("episode-")]) == episode_count
    assert len(list(root.glob("episode-*.parquet"))) > episode_count
    assert json.loads((root / "upload.json").read_text())["source_manifest_sha256"]


def test_huggingface_preparation_refuses_writer_and_excludes_orphan_files(tmp_path):
    from collector import prepare_huggingface

    root = tmp_path / "data"
    with Collection(root, ScriptedExecution(), limits=Limits(max_steps=2)) as run:
        run.run()
        with pytest.raises(ValueError, match="writer"):
            prepare_huggingface(root)
    assert not (root / "README.md").exists()
    (root / "frames-orphan.parquet").write_bytes(b"unfinished")
    report = prepare_huggingface(root)
    assert "frames-orphan.parquet" not in {f["path"] for f in report["files"]}
    with DatasetReader(root) as reader:
        shard = reader.db.execute("SELECT shard FROM frames LIMIT 1").fetchone()[0]
    (root / shard).write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        prepare_huggingface(root)


def test_older_dataset_format_is_preserved_and_rejected(tmp_path):
    root = tmp_path / "old"
    root.mkdir()
    original = b'{"format_version":1}'
    (root / "manifest.json").write_bytes(original)
    with pytest.raises(ValueError, match="version"):
        DatasetReader(root)
    with pytest.raises(ValueError, match="version"):
        Collection(root, ScriptedExecution())
    assert (root / "manifest.json").read_bytes() == original


@pytest.mark.parametrize("failure", ["receipt_write", "card_sync"])
def test_interrupted_preparation_cannot_leave_a_stale_upload_receipt(
    tmp_path, monkeypatch, failure
):
    import collector

    root = tmp_path / "data"
    with Collection(root, ScriptedExecution(), limits=Limits(max_steps=2)) as run:
        run.run()
    collector.prepare_huggingface(root)
    with Collection(root, ScriptedExecution(), limits=Limits(max_steps=2)) as run:
        run.run()
    before = {p.name: p.read_bytes() for p in root.glob("*.parquet")}
    original = collector.write_synced

    def fail_receipt(path, data):
        if failure == "receipt_write" and path.name.startswith(".upload.json-"):
            raise OSError("injected receipt write failure")
        return original(path, data)

    original_sync = collector.sync_directory
    sync_calls = 0

    def fail_card_sync(path):
        nonlocal sync_calls
        sync_calls += 1
        if failure == "card_sync" and sync_calls == 2:
            raise OSError("injected receipt publication sync failure")
        original_sync(path)

    with monkeypatch.context() as patch:
        patch.setattr(collector, "sync_directory", fail_card_sync)
        patch.setattr(collector, "write_synced", fail_receipt)
        with pytest.raises(OSError, match="receipt"):
            collector.prepare_huggingface(root)
    assert not (root / "upload.json").exists()
    assert before == {p.name: p.read_bytes() for p in root.glob("*.parquet")}
    assert validate_dataset(root)["transitions"] == 4
    assert collector.prepare_huggingface(root)["transitions"] == 4


def test_lossless_webp_preserves_arbitrary_rgb_and_rejects_lossy_bytes():
    import io
    from PIL import Image
    from collector import IMAGE_ENCODING, decode_rgb, encode_rgb

    image = np.random.default_rng(47).integers(0, 256, (37, 53, 3), dtype=np.uint8)
    encoded = encode_rgb(image)
    assert encoded[8:16] == b"WEBPVP8L"
    np.testing.assert_array_equal(decode_rgb(encoded, image.shape), image)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="WEBP", lossless=False, quality=100)
    with pytest.raises(ValueError, match="lossless"):
        decode_rgb(buffer.getvalue(), image.shape)
    with pytest.raises(ValueError, match="encoding"):
        decode_rgb(encoded, image.shape, encoding={**IMAGE_ENCODING, "lossless": False})


def test_existing_png_dataset_remains_readable_but_requires_explicit_conversion(tmp_path):
    import hashlib
    import io
    from PIL import Image
    from collector import canonical, parquet_bytes, prepare_huggingface

    with Collection(tmp_path, ScriptedExecution(), limits=Limits(max_steps=2)) as run:
        run.run()
        writer = run.writer
        with DatasetReader(tmp_path) as reader:
            expected = {i: reader.frame(i) for i in (1, 2)}
        for (name,) in list(writer.db.execute("SELECT DISTINCT shard FROM frames")):
            rows = writer.reader._table(name).to_pylist()
            for row in rows:
                buffer = io.BytesIO()
                Image.fromarray(expected[row["frame_id"]]).save(buffer, format="PNG")
                row["image"]["bytes"] = buffer.getvalue()
                writer.db.execute(
                    "UPDATE frames SET size=? WHERE frame_id=?",
                    (len(buffer.getvalue()), row["frame_id"]),
                )
            data = parquet_bytes("frames", rows)
            (tmp_path / name).write_bytes(data)
            writer.db.execute(
                "UPDATE files SET sha256=?,size=? WHERE name=?",
                (hashlib.sha256(data).hexdigest(), len(data), name),
            )
        manifest = dict(writer.manifest)
        del manifest["image_encoding"]
        data = canonical(manifest)
        (tmp_path / "manifest.json").write_bytes(data)
        writer.db.execute("UPDATE identity SET sha256=?", (hashlib.sha256(data).hexdigest(),))
        writer.db.commit()
    assert validate_dataset(tmp_path)["unique_frames"] == 2
    with DatasetReader(tmp_path) as reader:
        for i, image in expected.items():
            np.testing.assert_array_equal(reader.frame(i), image)
    prepare_huggingface(tmp_path)
    assert "lossless PNG" in (tmp_path / "README.md").read_text()
    with pytest.raises(ValueError, match="incompatible append"):
        Collection(tmp_path, ScriptedExecution())


def test_reuse_target_includes_pending_captures_and_records_full_vector_batch(tmp_path):
    execution = ScriptedVectorExecution()
    limits = Limits(max_steps=30, batch_steps=128, target_reuse=0.8, min_reuse_captures=6)
    with Collection(tmp_path, execution, limits=limits) as run:
        run.run()
        assert run.stop_reason == "reuse_target"
        assert execution.batch_sizes == [3]
        assert run.progress()["reuse_fraction"] == pytest.approx(5 / 6)
        assert run.progress()["stop_reason"] == "reuse_target"
    result = validate_dataset(tmp_path)
    assert result["transitions"] == 3
    assert result["captured_occurrences"] == 6
    assert result["incomplete_episodes"] == 3
    resumed = ScriptedVectorExecution()
    with Collection(tmp_path, resumed, limits=limits) as run:
        run.run()
        assert run.steps == 0 and not resumed.seeds
        assert run.stop_reason == "reuse_target"
    assert validate_dataset(tmp_path)["transitions"] == 3


def test_reuse_target_waits_for_minimum_sample(tmp_path):
    with Collection(
        tmp_path,
        ScriptedVectorExecution(),
        limits=Limits(max_steps=30, target_reuse=0.2, min_reuse_captures=10),
    ) as run:
        run.run()
        assert run.steps == 9
        assert run.progress()["captured_occurrences"] == 15
        assert run.stop_reason == "reuse_target"


@pytest.mark.parametrize("target", [0, -0.1, 1.01, float("nan"), float("inf"), True])
def test_invalid_reuse_target_is_rejected(target):
    with pytest.raises(ValueError, match="target_reuse"):
        Limits(target_reuse=target)


def test_reuse_window_counts_completed_episodes_and_forgets_old_discoveries(tmp_path):
    limits = Limits(max_steps=100, target_reuse=0.9, min_reuse_captures=6, reuse_window_episodes=2)
    with Collection(tmp_path, ScriptedExecution(), limits=limits) as run:
        run.run()
        assert run.steps == 6
        assert run.stop_reason == "reuse_target"
        assert run.window_progress()["reuse_window_fraction"] == 1
        assert run.window_progress()["reuse_window_captures"] == 6
        assert run.progress()["reuse_fraction"] < 0.9
    with Collection(tmp_path, ScriptedExecution(), limits=limits) as resumed:
        resumed.run()
        assert resumed.steps == 4  # Two newly completed episodes warm up the window.
    assert validate_dataset(tmp_path)["transitions"] == 10


def test_signal_during_append_stops_after_a_consistent_vector_batch(tmp_path, monkeypatch):
    import os
    import signal

    original_handler = signal.getsignal(signal.SIGINT)
    with Collection(tmp_path, ScriptedVectorExecution(), limits=Limits(max_steps=100)) as run:
        original_frame = run.writer._frame
        sent = False

        def interrupted_frame(image, encoded_rgb=None):
            nonlocal sent
            value = original_frame(image, encoded_rgb)
            if run.writer.current_frame is not None and not sent:
                sent = True
                os.kill(os.getpid(), signal.SIGINT)
            return value

        monkeypatch.setattr(run.writer, "_frame", interrupted_frame)
        run.run()
        assert sent
        assert run.stop_reason == "interrupted"
        assert run.steps == 3
    assert signal.getsignal(signal.SIGINT) == original_handler
    result = validate_dataset(tmp_path)
    assert result["transitions"] == 3
    assert result["captured_occurrences"] == 6


def test_parallel_validation_checks_images_and_detects_corruption(tmp_path):
    import sqlite3

    with Collection(tmp_path, ScriptedExecution(), limits=Limits(max_steps=6)) as run:
        run.run()
    assert validate_dataset(tmp_path, image_workers=2)["transitions"] == 6
    with sqlite3.connect(tmp_path / "index.sqlite") as db:
        name = db.execute("SELECT shard FROM frames LIMIT 1").fetchone()[0]
    path = tmp_path / name
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="integrity failure"):
        validate_dataset(tmp_path, image_workers=2)


def test_scalar_record_byte_bound_covers_arrow_storage():
    import pyarrow as pa
    from collector import transition_record, transition_schema, record_byte_bound

    row = transition_record(
        {
            "episode_id": 1,
            "step": 0,
            "labels": {"unicode": "café 🔥"},
            "native_action": np.array([0], dtype=np.int64),
        }
    )
    assert record_byte_bound(row) >= pa.Table.from_pylist([row], schema=transition_schema).nbytes
