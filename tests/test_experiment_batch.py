import json
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime

import pytest

from gradlab import experiment_batch as batch
from gradlab.job_queue import JobStore, JobSubject, SubjectUpdate


RUN = "gradlab-" + "a" * 32
SHA = "b" * 40


def setup_job(tmp_path, *, count=2, after_run=None):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "gradlab"\n')
    (tmp_path / "src/gradlab").mkdir(parents=True)
    recipe = "experiments/goals/Mario/Level1-1/recipes/ppo.yaml"
    path = tmp_path / recipe
    path.parent.mkdir(parents=True)
    path.write_text("{}\n")
    payload = {
        "repo_root": str(tmp_path),
        "queue_root": str(tmp_path / "queue"),
        "source_sha": SHA,
        "batch_id": "bounded-test",
        "target": "local-test",
        "deadline": time.time() + 10000,
        "after_run": after_run,
        "intents": [
            {
                "key": f"intent-{i}",
                "recipe": recipe,
                "seed": i,
                "max_duration_seconds": 14400,
                "overrides": [],
                "description": "fresh test",
            }
            for i in range(count)
        ],
    }
    store = JobStore(payload["queue_root"])
    admitted = store.enqueue(
        job_type=batch.TYPE,
        handler_version=batch.VERSION,
        payload=payload,
        idempotency_key=payload["batch_id"],
        subjects=[JobSubject("experiment-intent", x["key"]) for x in payload["intents"]],
    )
    job = store.claim_next()
    assert job["job_id"] == admitted.job["job_id"]
    return store, job


def git_ready(monkeypatch):
    monkeypatch.setattr(
        batch.subprocess, "check_output", lambda argv, **kw: SHA if argv[1] == "rev-parse" else ""
    )


def state(store, job, value, detail=None):
    store.checkpoint(
        job["job_id"],
        kind="test-state",
        subjects=(SubjectUpdate("experiment-intent", "intent-0", value, detail),),
    )


def startup_proof(store, job):
    work = store.work_root / job["job_id"]
    work.mkdir(parents=True, exist_ok=True)
    (work / "startup-verified.json").write_text(
        json.dumps(
            {
                "run_id": RUN,
                "image_digest": "image-sha",
                "gpu_assigned": True,
                "cuda_tensor_verified": True,
                "learner_cuda_verified": True,
            }
        )
    )


def test_cli_records_accept_multiline_json_and_admission_events():
    rows = batch._rows(
        'provider notice\n{"event":"bound"}\n' + json.dumps({"status": "ready"}, indent=2)
    )
    assert rows == [{"event": "bound"}, {"status": "ready"}]


def test_follow_lock_allows_only_one_real_observer_process(tmp_path, monkeypatch):
    started = threading.Event()
    original = subprocess.Popen
    children = []
    status = json.dumps({"run_id": RUN, "dstack": {"status": "done"}, "completed": True}, indent=2)
    monkeypatch.setattr(
        batch,
        "_command",
        lambda *a, **kw: [
            sys.executable,
            "-u",
            "-c",
            "import time; print(" + repr(status) + ",flush=True); time.sleep(0.3)",
        ],
    )

    def popen(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        started.set()
        return child

    monkeypatch.setattr(batch.subprocess, "Popen", popen)
    first = threading.Thread(target=batch._follow, args=(str(tmp_path), RUN, tmp_path))
    first.start()
    assert started.wait(5)
    assert batch._follow(str(tmp_path), RUN, tmp_path) == 0
    first.join(5)
    assert not first.is_alive()
    assert len(children) == 1
    assert json.loads((tmp_path / (RUN + ".json")).read_text()) == json.loads(status)


def test_successful_process_without_terminal_snapshot_stops_instead_of_auth_loop(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        batch,
        "_command",
        lambda *a, **kw: [sys.executable, "-c", "print('[oversized command output withheld]')"],
    )
    assert batch._follow(str(tmp_path), RUN, tmp_path) == 1
    with pytest.raises(RuntimeError, match="recovery review"):
        batch._observer({"repo_root": str(tmp_path)}, RUN, tmp_path)


def test_failed_real_observer_requires_review_before_reconnect(tmp_path, monkeypatch):
    monkeypatch.setattr(
        batch, "_command", lambda *a, **kw: [sys.executable, "-c", "raise SystemExit(3)"]
    )
    assert batch._follow(str(tmp_path), RUN, tmp_path) == 3
    with pytest.raises(RuntimeError, match="recovery review"):
        batch._observer({"repo_root": str(tmp_path)}, RUN, tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("deadline", float("inf")),
        ("deadline", float("nan")),
        ("deadline", True),
        ("source_sha", "latest"),
    ],
)
def test_rejects_nonfinite_or_unfrozen_batches(tmp_path, field, value):
    _, job = setup_job(tmp_path)
    with pytest.raises(ValueError):
        batch.ExperimentBatch.validate_payload({**job["payload"], field: value})


def test_waiting_for_predecessor_creates_no_admission_session(tmp_path, monkeypatch):
    _, job = setup_job(tmp_path, after_run=RUN)
    monkeypatch.setattr(batch, "_observer", lambda *a: {"completed": False})
    monkeypatch.setattr(batch, "_call", lambda *a: pytest.fail("No service command while waiting"))
    assert batch.ExperimentBatch().advance(job).state == "retry_wait"


def test_one_dispatch_persists_uncertainty_before_child_start(tmp_path, monkeypatch):
    store, job = setup_job(tmp_path)
    git_ready(monkeypatch)
    commands = []

    def call(payload, arguments, output):
        commands.append(arguments)
        if arguments[1] == "operator-preflight":
            return {"status": "ready"}
        assert store.subjects(job["job_id"])[0]["state"] == "submission_uncertain"
        assert store.subjects(job["job_id"])[1]["state"] == "running"
        assert not (store.subjects(job["job_id"])[1].get("detail") or {}).get("run_id")
        return {"run_id": RUN, "image_digest": "image-sha"}

    monkeypatch.setattr(batch, "_call", call)
    result = batch.ExperimentBatch().advance(job)
    assert result.subjects[0].state == "submitted"
    assert [argv[1] for argv in commands] == ["operator-preflight", "launch"]
    assert commands[1][commands[1].index("--max-duration") + 1] == "14400"
    assert "--existing-runtime-only" in commands[1]


def test_lost_acknowledgement_never_resubmits(tmp_path, monkeypatch):
    store, job = setup_job(tmp_path)
    state(store, job, "submission_uncertain", {"submission_key": "intent-0"})
    monkeypatch.setattr(
        batch, "_call", lambda *a: pytest.fail("Must reconcile instead of resubmitting")
    )
    assert batch.ExperimentBatch().advance(job).state == "blocked"


@pytest.mark.parametrize("failure_state", ["blocked", "failed"])
def test_generic_job_retry_cannot_erase_uncertain_admission(tmp_path, monkeypatch, failure_state):
    store, job = setup_job(tmp_path)
    state(
        store,
        job,
        "submission_uncertain",
        {"submission_key": "intent-0", "admission_started": True},
    )
    store.finish(job["job_id"], batch.HandlerResult(state=failure_state))
    store.retry(job["job_id"])
    recovered = store.claim_next()
    monkeypatch.setattr(
        batch, "_call", lambda *a: pytest.fail("Retry must not duplicate uncertain admission")
    )
    assert batch.ExperimentBatch().advance(recovered).state == "blocked"


def test_generic_job_retry_retains_an_admitted_run(tmp_path, monkeypatch):
    store, job = setup_job(tmp_path)
    state(store, job, "submitted", {"run_id": RUN, "image_digest": "image-sha"})
    store.finish(job["job_id"], batch.HandlerResult(state="blocked"))
    store.retry(job["job_id"])
    recovered = store.claim_next()
    monkeypatch.setattr(batch, "_observer", lambda *a: {"completed": False})
    monkeypatch.setattr(
        batch, "_call", lambda *a: pytest.fail("Observe the admitted Run instead of new launch")
    )
    assert batch.ExperimentBatch().advance(recovered).state == "retry_wait"


@pytest.mark.parametrize("invalid", [None, "goal", "cutoff", "stale", "matches", "partial"])
def test_reviewed_absence_recovers_only_matching_complete_post_failure_evidence(
    tmp_path, monkeypatch, invalid
):
    store, job = setup_job(tmp_path)
    state(store, job, "submission_uncertain", {"admission_started": True})
    store.finish(job["job_id"], batch.HandlerResult(state="failed"))
    store.retry(job["job_id"])
    recovered = store.claim_next()
    work = store.work_root / job["job_id"]
    work.mkdir(parents=True, exist_ok=True)
    failed = work / "intent-0-admission.json"
    failed.write_text('{"returncode":1,"records":[]}')
    proof = {
        "reviewed": True,
        "status": "not_admitted",
        "complete_search": True,
        "submission_key": "intent-0",
        "goal_slug": "Mario/Level1-1",
        "matches": [],
        "unresolved_candidates": [],
        "observed_at": time.time() + 1,
        "created_since": datetime.fromtimestamp(job["created_at"] - 60, UTC).isoformat(),
    }
    if invalid == "goal":
        proof["goal_slug"] = "Mario/Level1-2"
    elif invalid == "cutoff":
        proof["created_since"] = datetime.fromtimestamp(job["created_at"] + 60, UTC).isoformat()
    elif invalid == "stale":
        proof["observed_at"] = failed.stat().st_mtime - 1
    elif invalid == "matches":
        proof["matches"] = [{"run_id": RUN}]
    elif invalid == "partial":
        proof["complete_search"] = False
    (work / "intent-0-reviewed-absence.json").write_text(json.dumps(proof))
    monkeypatch.setattr(batch, "_call", lambda *a: pytest.fail("No admission during review"))
    result = batch.ExperimentBatch().advance(recovered)
    assert result.state == ("blocked" if invalid else "retry_wait")
    detail = store.subjects(job["job_id"])[0]["detail"]
    assert detail["admission_started"] is bool(invalid)
    if not invalid:
        assert len(list(work.glob("intent-0-failed-before-reconciliation-*.json"))) == 1
        assert not failed.exists()


def test_source_drift_never_approves_or_admits(tmp_path, monkeypatch):
    _, job = setup_job(tmp_path)
    monkeypatch.setattr(batch.subprocess, "check_output", lambda *a, **kw: "changed")
    monkeypatch.setattr(batch, "_call", lambda *a: pytest.fail("No access under source drift"))
    assert batch.ExperimentBatch().advance(job).state == "blocked"


@pytest.mark.parametrize("finished", [False, True])
def test_no_next_admission_without_terminal_and_first_gpu_verification(
    tmp_path, monkeypatch, finished
):
    store, job = setup_job(tmp_path)
    state(store, job, "submitted", {"run_id": RUN, "image_digest": "image-sha"})
    monkeypatch.setattr(
        batch,
        "_observer",
        lambda *a: {
            "completed": finished,
            "attempt_terminal": {"state": "succeeded", "drain": {"complete": True}},
        },
    )
    monkeypatch.setattr(batch, "_call", lambda *a: pytest.fail("First worker not verified"))
    assert batch.ExperimentBatch().advance(job).state == ("blocked" if finished else "retry_wait")


def test_complete_gpu_verified_run_still_requires_delivery_drain(tmp_path, monkeypatch):
    store, job = setup_job(tmp_path)
    state(store, job, "submitted", {"run_id": RUN, "image_digest": "image-sha"})
    startup_proof(store, job)
    monkeypatch.setattr(
        batch,
        "_observer",
        lambda *a: {
            "completed": True,
            "attempt_terminal": {"state": "succeeded", "drain": {"complete": False}},
        },
    )
    assert batch.ExperimentBatch().advance(job).state == "blocked"


def test_operational_failure_stops_batch_instead_of_spending_more_slots(tmp_path, monkeypatch):
    store, job = setup_job(tmp_path)
    state(store, job, "submitted", {"run_id": RUN, "image_digest": "image-sha"})
    startup_proof(store, job)
    monkeypatch.setattr(
        batch,
        "_observer",
        lambda *a: {"completed": True, "attempt_terminal": {"state": "resumable_failure"}},
    )
    result = batch.ExperimentBatch().advance(job)
    assert result.state == "blocked"
    assert result.subjects[0].state == "failed"


@pytest.mark.parametrize("terminal_state", ["succeeded", "stopped"])
def test_first_terminal_run_records_evidence_before_advancing(
    tmp_path, monkeypatch, terminal_state
):
    store, job = setup_job(tmp_path)
    state(store, job, "submitted", {"run_id": RUN, "image_digest": "image-sha"})
    startup_proof(store, job)
    monkeypatch.setattr(
        batch,
        "_observer",
        lambda *a: {
            "completed": True,
            "attempt_terminal": {
                "state": terminal_state,
                "final_step": 1000,
                "drain": {"complete": True},
            },
        },
    )
    monkeypatch.setattr(
        batch, "_call", lambda *a: pytest.fail("Do not admit next Run in same advance")
    )
    result = batch.ExperimentBatch().advance(job)
    assert result.state == "retry_wait"
    assert result.subjects[0].state == "succeeded"
    assert result.subjects[0].detail["attempt_terminal"]["final_step"] == 1000


def test_dispatch_uses_the_normal_cli_and_needs_no_personal_runner(tmp_path):
    assert batch._command({"repo_root": str(tmp_path)}, ["experiment", "status"]) == [
        sys.executable, "-m", "gradlab.main", "experiment", "status"
    ]
    assert batch._command({"repo_root": str(tmp_path)}, ["--run", RUN], follow=True) == [
        sys.executable, "-m", "gradlab.main", "experiment", "follow", "--run", RUN
    ]


def test_checkout_validation_does_not_require_launcher_metadata(tmp_path):
    _, job = setup_job(tmp_path)
    assert not (tmp_path / ".runbox.yaml").exists()
    assert batch.ExperimentBatch.validate_payload(job["payload"])["repo_root"] == str(tmp_path)


def test_supplied_command_executors_are_explicit_argv_and_do_not_fall_back(tmp_path, monkeypatch):
    monkeypatch.setenv("GRADLAB_EXPERIMENT_COMMAND", json.dumps([sys.executable, "-B"]))
    monkeypatch.setenv("GRADLAB_FOLLOW_COMMAND", json.dumps([sys.executable, "-u"]))
    assert batch._command({}, ["experiment", "status"]) == [sys.executable, "-B", "experiment", "status"]
    assert batch._command({}, ["--run", RUN], follow=True) == [sys.executable, "-u", "--run", RUN]
    for value in ["not json", "[]", '["relative"]', json.dumps([sys.executable, "bad\nargument"]),
                  json.dumps([sys.executable, "x" * 4097]), json.dumps([sys.executable] * 33)]:
        monkeypatch.setenv("GRADLAB_EXPERIMENT_COMMAND", value)
        with pytest.raises(ValueError):
            batch._command({}, ["experiment", "status"])


def test_real_command_receives_supplied_environment_without_secret_bootstrap(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPPLIED_OPERATOR_FIXTURE", "present")
    script = "import os,json; print(json.dumps({'supplied':os.environ.get('SUPPLIED_OPERATOR_FIXTURE')=='present'}))"
    monkeypatch.setenv("GRADLAB_EXPERIMENT_COMMAND", json.dumps([sys.executable, "-c", script]))
    assert batch._call({"repo_root": str(tmp_path)}, [], tmp_path / "receipt.json") == {"supplied": True}
