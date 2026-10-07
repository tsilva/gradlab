"""Process-level episode claims and bounded final-drain admission."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from gradlab.monitor_work import EpisodeWork


def consume(root, count):
    work = EpisodeWork(root, count)
    claimed = []
    while (ordinal := work.claim()) is not None:
        work.complete(ordinal, {"ordinal": ordinal})
        claimed.append(ordinal)
    return claimed


def test_processes_claim_every_episode_once_and_collect_in_manifest_order(tmp_path):
    with ProcessPoolExecutor(4, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(consume, tmp_path, 400) for _ in range(4)]
        claimed = [i for future in futures for i in future.result()]
    assert sorted(claimed) == list(range(400))
    work = EpisodeWork(tmp_path, 400)
    assert work.claim() is None
    assert work.results() == [{"ordinal": i} for i in range(400)]


def test_claim_is_not_completion_and_new_execution_reuses_no_stale_claim(tmp_path):
    work = EpisodeWork(tmp_path / "attempt1", 2)
    assert work.claim() == 0
    assert work.results() is None
    work.complete(0, {"ordinal": 0})
    assert work.results() is None
    retry = EpisodeWork(tmp_path / "attempt2", 2)
    assert retry.claim() == 0


def test_final_workers_are_bounded_recoverable_and_all_canceled(tmp_path, monkeypatch):
    from dataclasses import asdict
    from pathlib import Path
    import signal
    import pytest
    import subprocess
    import sys
    import time
    from gradlab.monitor_backend import SameHostEvalBackend
    from gradlab.monitor_config import MonitoringConfig
    from gradlab.r2_store import BucketConfig

    settings = asdict(
        MonitoringConfig(task_cpus=12, memory_bytes=25 * 1024**3, spool_bytes=7 * 1024**3)
    )
    popen = subprocess.Popen
    spawned = []

    def spawn(argv, **kwargs):
        if argv[0] == "/bin/ps":
            return popen(argv, **kwargs)
        request = Path(argv[-1])
        code = """import os,time,json,pathlib
from gradlab.monitor_bootstrap import process_start
pathlib.Path(os.environ["TEST_PROCESS_FILE"]).write_text(json.dumps(dict(pid=os.getpid(),created=process_start(os.getpid()))))
time.sleep(60)
"""
        kwargs["env"]["TEST_PROCESS_FILE"] = str(request.parent / "process.json")
        assert "WANDB_API_KEY" not in kwargs["env"]
        assert "GRADLAB_CONTROL_R2_SECRET_ACCESS_KEY" not in kwargs["env"]
        p = popen([sys.executable, "-c", code], **kwargs)
        spawned.append(p)
        return p

    monkeypatch.setattr(subprocess, "Popen", spawn)
    config = BucketConfig(uri=f"file://{tmp_path}/r2")
    backend = SameHostEvalBackend(tmp_path / "workers", config)
    try:
        handles = [
            backend.submit(
                dict(
                    settings=settings,
                    manifest=list(range(400)),
                    deadline=time.time() + 60,
                    checkpoint=i,
                )
            )
            for i in range(3)
        ]
        assert len(spawned) == 3
        backend.expand(handles, 12)
        assert len(spawned) == 12
        assert [len(backend._helpers(h)) for h in handles] == [3, 3, 3]
        backend.expand(handles, 12)
        assert len(spawned) == 12
        assert backend.available_slots(12) == 0
        assert not backend.quiescent()
        with pytest.raises(RuntimeError, match="cannot reclaim"):
            backend.forget(handles[0])
        assert len(backend._helpers(handles[0])) == 3
        child = backend._helpers(handles[0])[0]
        backend.processes[child.call_id].send_signal(signal.SIGKILL)
        backend.processes[child.call_id].wait(timeout=5)
        assert backend.poll(handles[0]).status == "failed"
        recovered = SameHostEvalBackend(backend.root, config)
        recovered.close()
        assert recovered.quiescent()
        for p in spawned:
            p.wait(timeout=5)
        for h in handles:
            recovered.forget(h)
        assert not list(backend.root.rglob("request.json"))
    finally:
        backend.close()
        for p in spawned:
            p.wait(timeout=5)


def test_helper_spawn_failure_is_reported_by_poll_for_checkpoint_retry(tmp_path, monkeypatch):
    import json
    import time
    from gradlab.monitor_backend import SameHostEvalBackend
    from gradlab.eval_backend import EvalHandle
    from gradlab.r2_store import BucketConfig

    backend = SameHostEvalBackend(tmp_path, BucketConfig(uri=f"file://{tmp_path}/r2"))
    handle = EvalHandle("cpu", "a" * 64)
    directory = tmp_path / handle.call_id
    directory.mkdir()
    intent = dict(manifest=[{}], deadline=time.time() + 60)
    (directory / "request.json").write_text(json.dumps(intent))
    monkeypatch.setattr(backend, "_locked", lambda key: str(key) == handle.call_id)
    monkeypatch.setattr(backend, "available_slots", lambda limit: 1)

    def fail_spawn(root, intent, **kwargs):
        root.mkdir(parents=True)
        (root / "request.json").write_text(json.dumps(intent))
        raise OSError("process launch unavailable")

    monkeypatch.setattr(backend, "_spawn", fail_spawn)
    backend.expand([handle], 2)
    result = backend.poll(handle)
    assert result.status == "failed"
    assert "process launch unavailable" in result.error


def test_calibration_does_not_reuse_parallel_speedup_for_active_cadence():
    from dataclasses import asdict
    from gradlab.monitor_calibration import assess_calibration, PHASES, ROLES
    from gradlab.monitor_config import MonitoringConfig

    settings = asdict(MonitoringConfig(task_cpus=4))
    sample = dict(
        episodes_completed=400,
        verified_terminal_receipt=True,
        seconds=8,
        retained_bytes=1000,
        peak_memory_bytes=1024**2,
        peak_spool_bytes=1024**2,
        longest_episode_steps=100,
        phase_seconds=dict.fromkeys(PHASES, 0.1),
    )
    data = dict(
        settings=settings,
        train_config=dict(timesteps=10000, checkpoint_freq=1000),
        source_sha="a" * 40,
        runtime_image="test",
        hardware_allocation={},
        samples=[
            dict(sample, role=role, checkpoint_sha256=str(i))
            for i, role in enumerate(sorted(ROLES))
        ],
        pairs=[
            dict(
                seed=i,
                warmup_excluded=True,
                comparable_host_load=True,
                checkpoint_freq=1000,
                equivalent_workload=True,
                nonzero_capture=True,
                off_rate=100,
                on_rate=100,
            )
            for i in range(3)
        ],
    )
    serial = assess_calibration(data)
    for row in data["samples"]:
        row["execution_workers"] = 4
    parallel = assess_calibration(data)
    assert parallel["recommended_checkpoint_freq"] == 4 * serial["recommended_checkpoint_freq"]
    assert parallel["estimated_final_tail_seconds"] == 4 * serial["estimated_final_tail_seconds"]
    assert parallel["peak_worker_memory_bytes"] == serial["peak_worker_memory_bytes"]
