import numpy as np
import pytest

from gradlab.batch_runtime import BatchRuntime
from gradlab.task_kernels import IdentityTaskDefinition
from tests.test_batch_runtime import DeterministicNativeVectorProvider, descriptor_for


def runtime_with_tracking(window=5):
    provider = DeterministicNativeVectorProvider()
    descriptor = descriptor_for(provider)
    runtime = BatchRuntime(
        provider,
        descriptor,
        IdentityTaskDefinition().bind(descriptor, 2),
        occupancy={
            "cell": {"dimensions": [{"source": "x", "bucket_size": 10}]},
            "domains": [[0, 1, 2]],
            "units": ["pixels"],
            "window_transitions": window,
        },
    )
    return provider, runtime


def test_counts_source_states_across_reused_buffers_and_terminal_resets():
    provider, runtime = runtime_with_tracking()
    runtime.reset(seed=7)
    actions = np.zeros((2, 3), dtype=np.int8)
    provider.queue_step(x=[10, 20])
    runtime.step(actions)
    provider.queue_step(x=[20, 20], terminated=[True, False])
    runtime.step(actions)
    provider.queue_step(x=[10, 20])
    runtime.step(actions)
    windows = runtime.drain_occupancy()
    assert len(windows) == 1
    window = windows[0]
    assert window["window_transitions"] == 6
    assert (window["start_step"], window["end_step"]) == (0, 6)
    combined = [row for row in window["rows"] if row["origin"] == "combined"]
    assert [row["count"] for row in combined] == [3, 1, 2]
    assert [row["entries"] for row in combined] == [3, 1, 1]
    assert [row["fraction"] for row in combined] == [0.5, 1 / 6, 1 / 3]
    assert all(row["fraction"] is None for row in window["rows"] if row["origin"] == "archive")
    runtime.step(actions)
    final = runtime.drain_occupancy(final=True)[0]
    assert final["complete"] is False
    assert final["end_step"] == 8
    combined = [row for row in final["rows"] if row["origin"] == "combined"]
    assert [row["count"] for row in combined] == [0, 1, 1]
    assert [row["entries"] for row in combined] == [0, 1, 0]
    assert [row["cumulative_count"] for row in combined] == [3, 2, 3]
    runtime.close()


def test_zero_separated_six_brick_bands_share_archive_assignment():
    provider = DeterministicNativeVectorProvider(num_envs=6)
    descriptor = descriptor_for(provider)
    cell = {"dimensions": [{"source": "x", "bucket_size": 6, "zero_separate": True}]}
    runtime = BatchRuntime(
        provider,
        descriptor,
        IdentityTaskDefinition().bind(descriptor, 6),
        occupancy={
            "cell": cell,
            "domains": [[0, 1, 2]],
            "units": ["bricks"],
            "window_transitions": 6,
        },
    )
    runtime.reset()
    provider.queue_step(x=[0, 1, 5, 6, 7, 12])
    runtime.step(np.zeros((6, 3), dtype=np.int8))
    runtime.drain_occupancy()
    runtime.step(np.zeros((6, 3), dtype=np.int8))
    rows = runtime.drain_occupancy()[0]["rows"]
    assert [row["count"] for row in rows if row["origin"] == "combined"] == [1, 3, 2]
    from gradlab.cells import ArchiveCellConfig, ArchiveCellDetector

    detector = ArchiveCellDetector(ArchiveCellConfig.from_mapping(cell, label="test"))
    assert detector.keys({("source", "x"): np.array([0, 1, 5, 6, 7, 12])}, n_envs=6) == (
        b"[0]",
        b"[1]",
        b"[1]",
        b"[1]",
        b"[2]",
        b"[2]",
    )
    runtime.close()


def test_recovery_requires_matching_learner_cursor_and_retains_pending_entries():
    provider, runtime = runtime_with_tracking(window=8)
    runtime.reset()
    provider.queue_step(x=[10, 20])
    runtime.step(np.zeros((2, 3), dtype=np.int8))
    state = runtime.occupancy_state(recovery_cursor="consistent-learner-state")
    _, resumed = runtime_with_tracking(window=8)
    resumed.reset()
    assert resumed.restore_occupancy(state, recovery_cursor="consistent-learner-state")
    resumed.step(np.zeros((2, 3), dtype=np.int8))
    window = resumed.drain_occupancy(final=True)[0]
    assert window["segment"] == state["segment"]
    assert [row["count"] for row in window["rows"] if row["origin"] == "combined"] == [2, 1, 1]
    _, interrupted = runtime_with_tracking(window=8)
    interrupted.reset()
    assert not interrupted.restore_occupancy(state, recovery_cursor="model-only", initial_step=10)
    interrupted.step(np.zeros((2, 3), dtype=np.int8))
    window = interrupted.drain_occupancy(final=True)[0]
    assert window["segment"] != state["segment"]
    assert (window["start_step"], window["end_step"]) == (10, 12)
    assert window["uncovered_interval"] == (0, 10)
    assert (
        sum(row["cumulative_count"] for row in window["rows"] if row["origin"] == "combined") == 2
    )
    for item in (runtime, resumed, interrupted):
        item.close()


def test_invalid_recovery_is_rejected_before_mutating_live_totals():
    _, runtime = runtime_with_tracking()
    runtime.reset()
    runtime.step(np.zeros((2, 3), dtype=np.int8))
    state = runtime.occupancy_state(recovery_cursor="cursor")
    state["counts"] = [[-1]]
    with pytest.raises(ValueError):
        runtime.restore_occupancy(state, recovery_cursor="cursor")
    assert runtime.drain_occupancy(final=True)[0]["end_step"] == 2


def test_outbox_replay_keeps_one_validated_window_and_publisher_table(tmp_path):
    from gradlab.metric_store import MetricStore
    from gradlab.wandb_publisher import publish_pending_frames

    _, runtime = runtime_with_tracking()
    runtime.reset()
    runtime.step(np.zeros((2, 3), dtype=np.int8))
    window = runtime.drain_occupancy(final=True)[0]
    store = MetricStore(tmp_path / "metrics.sqlite")
    store.init()
    store.append_occupancy(window)
    store.append_occupancy(window)
    assert len(store.pending_metric_frames()) == 1

    class Run:
        def __init__(self):
            self.logged = []

        def define_metric(self, *args, **kwargs):
            pass

        def log(self, payload, **kwargs):
            self.logged.append(payload)

    run = Run()
    assert publish_pending_frames(store, run, limit=10) == 1
    assert publish_pending_frames(store, run, limit=10) == 0
    table = run.logged[0]["train/occupancy/table"]
    assert len(table.data) == 12
    assert "cell_space_hash" in table.columns
    assert "uncovered_interval" in table.columns
    assert run.logged[0]["train/step"] == 2
    invalid = {**window, "rows": [{**row, "count": 99} for row in window["rows"]]}
    with pytest.raises(ValueError):
        store.append_occupancy(invalid)


def test_breakout_tracking_uses_actual_width_and_preserves_policy_trajectory():
    from pathlib import Path
    from gradlab.env import make_training_batch_runtime
    from gradlab.recipe_documents import compose_train_document
    from gradlab.env_config import env_config_from_mapping
    from gradlab.env import resolve_env_config

    goal = Path("experiments/goals/Breakout-Atari2600-v0")
    train = compose_train_document(goal / "_goal.yaml", goal / "recipes/ppo-ball-state.yaml")[
        "train_config"
    ]
    config = resolve_env_config(env_config_from_mapping(train))
    tracking = {
        "cell": {
            "dimensions": [
                {"source": "walls_cleared", "bucket_size": 1, "clamp": [0, 3]},
                {"source": "bricks_remaining", "bucket_size": 6, "zero_separate": True},
                {"source": "paddle_width", "bucket_size": 1},
            ]
        },
        "domains": [list(range(4)), list(range(19)), [8, 16]],
        "units": ["walls", "bricks", "pixels"],
        "window_transitions": 64,
    }
    plain = make_training_batch_runtime(config, 2, 13)
    tracked = make_training_batch_runtime(config, 2, 13, occupancy=tracking)
    try:
        a, b = plain.reset(seed=13), tracked.reset(seed=13)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        for step in range(32):
            actions = np.array([step % 3, (step + 1) % 3], dtype=np.int64)
            a, b = plain.step(actions), tracked.step(actions)
            np.testing.assert_array_equal(a.rewards, b.rewards)
            np.testing.assert_array_equal(a.terminated, b.terminated)
            for key in a.observations:
                np.testing.assert_array_equal(a.observations[key], b.observations[key])
        window = tracked.drain_occupancy()[0]
        assert sum(row["count"] for row in window["rows"] if row["origin"] == "combined") == 64
        assert tracked.state_archive is None
    finally:
        plain.close()
        tracked.close()


def test_named_cell_spaces_resolve_once_for_tracking_and_archive():
    from gradlab.train_config import validate_and_normalize_train_config

    cell = {"dimensions": [{"source": "x", "bucket_size": 10}]}
    config = validate_and_normalize_train_config(
        {
            "cell_spaces": {"stage": {"cell": cell, "domains": [[0, 1, 2]], "units": ["pixels"]}},
            "occupancy": {"space": "stage", "window_transitions": 1000},
            "state_archive": {"recorder": {"mode": "cell_transition", "cell": "stage"}},
        }
    )
    assert config["occupancy"]["cell"] == config["state_archive"]["recorder"]["cell"]
    assert config["occupancy"]["window_transitions"] == 1000


def test_masked_restore_counts_new_archive_origin_without_changing_other_lanes(tmp_path):
    from gradlab.batch_runtime import ProviderDescriptor, SignalSpec
    from tests.test_state_archive import PortableBreakoutProvider, archive_config

    provider = PortableBreakoutProvider(num_envs=2)
    descriptor = ProviderDescriptor(
        provider_id="env-breakoutatari2600-turbo-native",
        native_observation_space=provider.single_observation_space,
        native_action_space=provider.single_action_space,
        signal_schema={"score": SignalSpec("score", np.int64)},
        start_catalog=("Start",),
        supports_live_snapshots=True,
        live_snapshots_deterministic=True,
        snapshot_codec_id="breakout-turbo-env.state-v1",
        snapshot_compatibility_id="test-environment-v1",
    )
    runtime = BatchRuntime(
        provider,
        descriptor,
        IdentityTaskDefinition(signals={"score": "score"}).bind(descriptor, 2),
        state_archive=archive_config(curriculum=False),
        state_archive_root=tmp_path,
        occupancy={
            "cell": {"dimensions": [{"signal": "score", "bucket_size": 50}]},
            "domains": [list(range(5))],
            "units": ["score"],
            "window_transitions": 6,
        },
    )
    runtime.reset()
    runtime.step(np.zeros(2, dtype=np.int64))
    selected = np.array([True, False])
    entries = runtime.capture_archive_entries(selected)
    runtime.step(np.zeros(2, dtype=np.int64))
    runtime.restore_archive_entries(selected, entries)
    runtime.step(np.zeros(2, dtype=np.int64))
    rows = runtime.drain_occupancy()[0]["rows"]
    assert [row["count"] for row in rows if row["origin"] == "combined"] == [2, 3, 1, 0, 0]
    assert [row["count"] for row in rows if row["origin"] == "archive"] == [0, 1, 0, 0, 0]
    assert sum(row["count"] for row in rows if row["origin"] == "normal") == 5
    runtime.close()


def test_inactive_reset_fields_are_not_read_by_tracking():
    class MaskedProvider(DeterministicNativeVectorProvider):
        def reset(self, **kwargs):
            observations, infos = super().reset(**kwargs)
            mask = kwargs["options"]["reset_mask"]
            infos["x"] = infos["x"].astype(object)
            infos["x"][~mask] = None
            infos["_x"] = mask
            return observations, infos

    provider = MaskedProvider()
    descriptor = descriptor_for(provider)
    runtime = BatchRuntime(
        provider,
        descriptor,
        IdentityTaskDefinition().bind(descriptor, 2),
        occupancy={
            "cell": {"dimensions": [{"source": "x", "bucket_size": 10}]},
            "domains": [[0, 1, 2]],
            "units": ["pixels"],
            "window_transitions": 6,
        },
    )
    runtime.reset()
    provider.queue_step(x=[10, 20], terminated=[True, False])
    runtime.step(np.zeros((2, 3), dtype=np.int8))
    runtime.step(np.zeros((2, 3), dtype=np.int8))
    rows = runtime.drain_occupancy(final=True)[0]["rows"]
    assert [row["count"] for row in rows if row["origin"] == "combined"] == [3, 0, 1]
    runtime.close()


def test_native_mario_level_and_position_tracking_is_passive():
    from gradlab.env import make_training_batch_runtime
    from pathlib import Path
    from gradlab.recipe_documents import compose_train_document
    from gradlab.env_config import env_config_from_mapping

    root = Path("experiments/goals/SuperMarioBros-Nes-v0/Level1-1")
    train = compose_train_document(root / "_goal.yaml", root / "recipes/ppo.yaml")["train_config"]
    config = env_config_from_mapping(train)
    tracking = {
        "cell": {
            "dimensions": [
                {"source": "levelHi", "bucket_size": 1},
                {"source": "levelLo", "bucket_size": 1},
                {"signal": "x", "bucket_size": 256},
            ]
        },
        "domains": [[0], [0], list(range(32))],
        "units": ["world index", "level index", "pixels"],
        "window_transitions": 64,
    }
    from gradlab.rom_assets import discover_rom_path, direct_rom_asset_manifest
    from gradlab.rom_runtime import bind_rom_path

    try:
        path = discover_rom_path(config.game)
    except FileNotFoundError:
        pytest.skip("requires the operator-owned Mario ROM")
    binding = bind_rom_path(direct_rom_asset_manifest(config.game, path), path)
    plain = make_training_batch_runtime(config, 2, 17, rom_binding=binding)
    tracked = make_training_batch_runtime(config, 2, 17, occupancy=tracking, rom_binding=binding)
    try:
        a, b = plain.reset(seed=17), tracked.reset(seed=17)
        np.testing.assert_array_equal(a, b)
        for _ in range(32):
            action = np.array([1, 2], dtype=np.int64)
            a, b = plain.step(action), tracked.step(action)
            np.testing.assert_array_equal(a.observations, b.observations)
            np.testing.assert_array_equal(a.rewards, b.rewards)
            np.testing.assert_array_equal(a.terminated, b.terminated)
        rows = tracked.drain_occupancy()[0]["rows"]
        assert sum(row["count"] for row in rows if row["origin"] == "combined") == 64
    finally:
        plain.close()
        tracked.close()


def test_immutable_windows_and_conflicting_outbox_ranges(tmp_path):
    from gradlab.metric_store import MetricStore

    _, runtime = runtime_with_tracking()
    runtime.reset()
    runtime.step(np.zeros((2, 3), dtype=np.int8))
    window = runtime.drain_occupancy(final=True)[0]
    with pytest.raises(TypeError):
        window["rows"][0]["count"] = 100
    store = MetricStore(tmp_path / "metrics.sqlite")
    store.init()
    store.append_occupancy(window)
    with pytest.raises(ValueError, match="conflicting"):
        store.append_occupancy({**window, "attempt_id": "attempt-" + "f" * 16})
    with pytest.raises(ValueError, match="overlapping"):
        store.append_occupancy({**window, "sequence": 1})
    runtime.close()


def test_reporter_recovery_marks_only_uncovered_tail(tmp_path):
    from types import SimpleNamespace
    from gradlab.metric_store import MetricStore
    from gradlab.occupancy import OccupancyReporter

    store = MetricStore(tmp_path / "metrics.sqlite")
    store.init()
    _, runtime = runtime_with_tracking(window=2)
    runtime.reset()
    runtime.step(np.zeros((2, 3), dtype=np.int8))
    prior = runtime.drain_occupancy()[0]
    store.append_occupancy(prior)
    _, resumed = runtime_with_tracking(window=2)
    resumed.reset()
    OccupancyReporter(
        resumed,
        SimpleNamespace(
            metric_store=store,
            wandb_enabled=False,
            train_config={"wandb_run_id": prior["run_id"]},
        ),
        initial_step=4,
    )
    resumed.step(np.zeros((2, 3), dtype=np.int8))
    window = resumed.drain_occupancy()[0]
    assert tuple(window["uncovered_interval"]) == (2, 4)
    assert window["segment"] != prior["segment"]
    assert window["rows"][0]["cumulative_denominator"] == 2
    from gradlab.occupancy import occupancy_table

    table = occupancy_table(window)
    assert table.data[0][table.columns.index("uncovered_interval")] == "[2, 4]"
    runtime.close()
    resumed.close()


def test_history_pages_preserve_late_windows_without_growing_tables(tmp_path):
    from gradlab.metric_store import MetricStore
    from gradlab.occupancy import occupancy_table

    store = MetricStore(tmp_path / "metrics.sqlite")
    store.init()
    _, runtime = runtime_with_tracking(window=2)
    runtime.reset()
    for _ in range(128):
        runtime.step(np.zeros((2, 3), dtype=np.int8))
        window = runtime.drain_occupancy()[0]
        store.append_occupancy(window)
    page = store.occupancy_page(window)
    assert [item["sequence"] for item in page] == list(range(120, 128))
    table = occupancy_table(window, page=page)
    assert len(table.data) == 8 * 3 * 4
    assert sum(item["window_transitions"] for item in page) == 16
    runtime.close()


def test_curriculum_table_preserves_publishing_run_identity(tmp_path):
    from gradlab.metric_store import MetricStore
    from gradlab.wandb_publisher import publish_pending_frames
    from gradlab.curriculum_reporting import TABLE

    class Run:
        id = "gradlab-" + "a" * 32

        def define_metric(self, *args, **kwargs):
            pass

        def log(self, payload, **kwargs):
            self.payload = payload

    store = MetricStore(tmp_path / "metrics.sqlite")
    store.init()
    store.enqueue_event(
        kind="curriculum_distribution",
        step=16,
        source="train",
        payload={"rows": [["cell-a", 2, 1.0, False, "coverage", 1, 1, 4, 16]]},
    )
    run = Run()
    assert publish_pending_frames(store, run, limit=1) == 1
    table = run.payload[TABLE]
    assert table.data[0][table.columns.index("run_id")] == run.id
    assert table.data[0][table.columns.index("probability")] == 1.0


def archive_runtime(root, *, max_entries=4, max_bytes=65536, restore=False, strategy="value_error"):
    from gradlab.batch_runtime import ProviderDescriptor, SignalSpec
    from tests.test_state_archive import PortableBreakoutProvider, archive_config

    provider = PortableBreakoutProvider(num_envs=5)
    descriptor = ProviderDescriptor(
        provider_id="env-breakoutatari2600-turbo-native",
        native_observation_space=provider.single_observation_space,
        native_action_space=provider.single_action_space,
        signal_schema={"score": SignalSpec("score", np.int64)},
        start_catalog=("Start",),
        supports_live_snapshots=True,
        live_snapshots_deterministic=True,
        snapshot_codec_id="breakout-turbo-env.state-v1",
        snapshot_compatibility_id="test-v1",
    )
    config = archive_config(n_envs=5)
    config["curriculum"].update(
        restore_entries=restore,
        entries_per_cell=2,
        max_entries=max_entries,
        max_bytes=max_bytes,
        strategy=strategy,
    )
    if strategy == "coverage":
        config["curriculum"]["archive_share"] = 0.2
    runtime = BatchRuntime(
        provider,
        descriptor,
        IdentityTaskDefinition(signals={"score": "score"}).bind(descriptor, 5),
        state_archive=config,
        state_archive_root=root,
        run_seed=17,
        occupancy={
            "cell": config["recorder"]["cell"],
            "domains": [list(range(100))],
            "units": ["score"],
            "window_transitions": 25,
        }
        if strategy == "coverage"
        else None,
    )
    return runtime


def test_capture_has_physical_budgets_and_recovers_inventory(tmp_path):
    runtime = archive_runtime(tmp_path)
    runtime.reset()
    for _ in range(12):
        runtime.step(np.zeros(5, dtype=np.int64))
    metrics = runtime.curriculum_complete_rollout()
    assert 0 < runtime.state_archive_summary()["entry_count"] <= 4
    assert metrics["admission_candidate_count"] > metrics["admission_accepted_count"]
    assert runtime.state_archive_summary()["physical_bytes"] <= 65536
    inventory = runtime.state_archive_view("curriculum")["cells"]
    runtime.close()
    resumed = archive_runtime(tmp_path)
    assert resumed.archive_curriculum.entry_count > 0
    assert resumed.state_archive_view("curriculum")["cells"] == inventory
    resumed.close()


def test_archive_publication_protects_generation_while_pruning(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from gradlab.r2_store import BucketConfig, RunStorageConfig
    from gradlab.run_authority import RunAuthority
    from gradlab.run_contracts import new_run_id, new_attempt_id

    runtime = archive_runtime(tmp_path / "archive")
    runtime.reset()
    for _ in range(4):
        runtime.step(np.zeros(5, dtype=np.int64))
    runtime.curriculum_complete_rollout()
    authority = RunAuthority(
        RunStorageConfig(
            control=BucketConfig(uri=f"file://{tmp_path}/control"),
            evaluation=BucketConfig(uri=f"file://{tmp_path}/eval"),
            models=BucketConfig(
                uri=f"file://{tmp_path}/models", public_base_url="https://example.test"
            ),
        )
    )
    uploading, release, pruning = Event(), Event(), Event()
    original = authority.control.put_bytes

    def gated_upload(*args, **kwargs):
        uploading.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(authority.control, "put_bytes", gated_upload)

    def prune():
        pruning.set()
        runtime.state_archive.write_view("curriculum", {}, referenced_entry_ids=[])
        runtime.state_archive.retain_entries([])

    run_id = new_run_id()
    with ThreadPoolExecutor(max_workers=2) as pool:
        published = pool.submit(
            authority.publish_state_archive,
            run_id=run_id,
            attempt_id=new_attempt_id(),
            archive_root=tmp_path / "archive",
        )
        try:
            assert uploading.wait(5)
            mutation = pool.submit(prune)
            assert pruning.wait(5)
            assert not mutation.done()
        finally:
            release.set()
        receipt = published.result(timeout=5)
        mutation.result(timeout=5)
    assert not (tmp_path / "archive/closure.json").exists()
    assert runtime.state_archive.summary()["entry_count"] == 0
    restored = authority.restore_state_archive(run_id=run_id, destination=tmp_path / "restored")
    assert restored == receipt
    assert restored["archive"]["entry_count"] > 0
    # Do not persist the now deliberately removed curriculum a second time.
    runtime.state_archive.close()
    runtime.state_archive = None
    runtime.archive_curriculum = None
    runtime.close()


def test_capture_reclaims_entries_that_exceed_the_physical_budget(tmp_path):
    runtime = archive_runtime(tmp_path, max_bytes=1)
    runtime.reset()
    runtime.step(np.zeros(5, dtype=np.int64))
    assert runtime.state_archive_summary()["entry_count"] == 0
    assert runtime.archive_curriculum.entry_count == 0
    runtime.close()


def test_coverage_sampler_freezes_prior_combined_exposure_and_preserves_caps():
    from gradlab.state_archive import ArchiveCurriculum, ArchiveCurriculumConfig
    from tests.test_state_archive import archive_config

    value = archive_config(n_envs=10)
    value["curriculum"].update(strategy="coverage", archive_share=0.2)
    config = ArchiveCurriculumConfig.from_mapping(value, n_envs=10)
    sampler = ArchiveCurriculum(config, n_envs=10, run_seed=17, global_lane_ids=range(10))
    for cell in ("a", "b", "c", "d", "e"):
        sampler.admit(cell, cell + "-entry")
    sampler.update_coverage({"a": 100, "b": 10, "c": 10, "d": 10, "e": 0})
    sampler.begin_rollout()
    prior = sampler.distribution()
    assert prior["e"] > prior["a"]
    assert sum(prior.values()) == pytest.approx(1)
    assert max(prior.values()) <= 0.25
    sampler.update_coverage({"a": 100, "b": 10, "c": 10, "d": 10, "e": 1000})
    assert sampler.distribution() == prior
    sampler.begin_rollout()
    assert sampler.distribution()["e"] < prior["e"]
    assert np.count_nonzero(sampler.archive_lane_mask) == 2
    twin = ArchiveCurriculum(config, n_envs=10, run_seed=17, global_lane_ids=range(10))
    for cell in ("a", "b", "c", "d", "e"):
        twin.admit(cell, cell + "-entry")
    twin.update_coverage({"a": 100, "b": 10, "c": 10, "d": 10, "e": 1000})
    twin.begin_rollout()
    assert [sampler.sample(lane=0, episode_index=i).cell_id for i in range(15)] == [
        twin.sample(lane=0, episode_index=i).cell_id for i in range(15)
    ]


def test_coverage_runtime_counts_assisted_practice_and_protects_active_entries(tmp_path):
    runtime = archive_runtime(tmp_path, restore=True, strategy="coverage", max_entries=20)
    runtime.reset()
    runtime.curriculum_begin_rollout()
    runtime.step(np.zeros(5, dtype=np.int64))
    runtime.curriculum_complete_rollout()
    runtime.curriculum_begin_rollout()
    frozen = runtime.archive_curriculum.distribution()
    runtime.step(np.zeros(5, dtype=np.int64))
    active = runtime.archive_curriculum.retained_entry_ids()
    for _ in range(3):
        runtime.step(np.zeros(5, dtype=np.int64))
    windows = runtime.drain_occupancy()
    rows = windows[0]["rows"]
    assert sum(row["count"] for row in rows if row["origin"] == "archive") == 3
    assert sum(row["count"] for row in rows if row["origin"] == "combined") == 25
    assert runtime.archive_curriculum.distribution() == frozen
    assert all(runtime.state_archive.entry(entry_id) for entry_id in active)
    runtime.curriculum_complete_rollout()
    runtime.curriculum_begin_rollout()
    assert runtime.archive_curriculum.distribution() != frozen
    runtime.close()


def test_coverage_full_archive_cannot_change_the_frozen_distribution(tmp_path):
    runtime = archive_runtime(tmp_path, restore=True, strategy="coverage", max_entries=2)
    runtime.reset()
    runtime.curriculum_begin_rollout()
    runtime.step(np.zeros(5, dtype=np.int64))
    runtime.curriculum_complete_rollout()
    runtime.curriculum_begin_rollout()
    frozen = runtime.archive_curriculum.distribution()
    representatives = runtime.archive_curriculum.retained_entry_ids()
    for _ in range(10):
        runtime.step(np.zeros(5, dtype=np.int64))
        assert runtime.archive_curriculum.distribution() == frozen
        assert sum(runtime.archive_curriculum.distribution().values()) == pytest.approx(1)
        assert all(runtime.state_archive.entry(entry_id) for entry_id in representatives)
    runtime.close()


def test_coverage_inventory_turns_over_only_between_rollouts():
    from gradlab.state_archive import ArchiveCurriculum, ArchiveCurriculumConfig
    from tests.test_state_archive import archive_config

    value = archive_config(n_envs=5)
    value["curriculum"].update(
        strategy="coverage", archive_share=0.2, max_entries=2, entries_per_cell=1
    )
    sampler = ArchiveCurriculum(
        ArchiveCurriculumConfig.from_mapping(value, n_envs=5),
        n_envs=5,
        run_seed=17,
        global_lane_ids=range(5),
    )
    sampler.admit("early-a", "a")
    sampler.admit("early-b", "b")
    sampler.begin_rollout()
    frozen = sampler.distribution()
    assert not sampler.admit("later", "later-state")
    assert sampler.distribution() == frozen
    sampler.complete_rollout()
    sampler.begin_rollout()
    next_frozen = sampler.distribution()
    assert sampler.admit("later", "later-state")
    assert sampler.distribution() == next_frozen
    sampler.complete_rollout()
    sampler.begin_rollout()
    assert "later" in sampler.distribution()
    assert sampler.entry_count <= 2


def test_coverage_byte_limit_allows_later_inventory_with_spare_entry_slots(tmp_path):
    runtime = archive_runtime(tmp_path, strategy="coverage", max_entries=20, max_bytes=2000)
    runtime.reset()
    runtime.curriculum_begin_rollout()
    runtime.step(np.zeros(5, dtype=np.int64))
    runtime.curriculum_complete_rollout()
    runtime.curriculum_begin_rollout()
    first = runtime.archive_curriculum.distribution()
    assert first and runtime.archive_curriculum.entry_count < 20
    runtime.step(np.zeros(5, dtype=np.int64))
    assert runtime.archive_curriculum.distribution() == first
    assert runtime.state_archive_summary()["physical_bytes"] <= 2000
    runtime.curriculum_complete_rollout()
    runtime.curriculum_begin_rollout()
    runtime.step(np.zeros(5, dtype=np.int64))
    runtime.curriculum_complete_rollout()
    assert set(runtime.archive_curriculum.distribution()) - set(first)
    assert runtime.state_archive_summary()["physical_bytes"] <= 2000
    runtime.close()
