"""Domain checks and dispatch for the explicitly selected training transport."""

import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class LaunchError(RuntimeError):
    pass


def check_storage():
    import uuid
    from gradlab.r2_store import ConditionalWriteConflict, ObjectNotFound, R2Bucket, RunStorageConfig

    from gradlab.operator_environment import load_repository_operator_environment
    load_repository_operator_environment(ROOT)
    storage = RunStorageConfig.from_env()
    key = "vault-verification/" + uuid.uuid4().hex
    payload = b"GradLab scoped Agent Vault R2 verification\n" * 32
    checked = []
    for role in ("control", "evaluation", "models"):
        bucket = R2Bucket(getattr(storage, role))
        created = False
        copy_key = key + "/copy"
        try:
            etag = bucket.put_bytes(key, payload, metadata={"sha256": hashlib.sha256(payload).hexdigest()})
            created = True
            assert bucket.get_bytes(key) == payload
            assert bucket.head(key)["metadata"]["sha256"] == hashlib.sha256(payload).hexdigest()
            try:
                bucket.put_bytes(key, b"conflicting synthetic bytes")
            except ConditionalWriteConflict:
                pass
            else:
                raise LaunchError("R2 conditional create did not reject conflicting bytes")
            changed = payload + b"conditional replacement\n"
            etag = bucket.put_bytes(key, changed, create_only=False, if_match=etag)
            assert bucket.get_bytes(key) == changed
            bucket.copy_within(key, copy_key)
            assert bucket.get_bytes(copy_key) == changed
            bucket.delete(copy_key)
            assert key in list(bucket.iter_keys("vault-verification/"))
            bucket.delete(key, if_match=etag)
            created = False
            try:
                bucket.head(key)
            except ObjectNotFound:
                pass
            else:
                raise LaunchError("R2 verification cleanup is incomplete")
            checked.append(role)
        finally:
            bucket.delete(copy_key)
            if created:
                bucket.delete(key)
    print(json.dumps({"ok": True, "r2_roles": checked, "conditional_writes": True,
                      "hashes_verified": True, "probe_objects_removed": True}))


def inventory():
    from gradlab.dstack_backend import DstackBackend
    from gradlab.operator_credentials import resolve_dstack_token
    from gradlab.operator_environment import load_repository_operator_environment

    report = load_repository_operator_environment(ROOT)
    fleet = report.dstack.fleet()
    coordinator = report.dstack.coordinator(fleet.coordinator_id)
    token, _ = resolve_dstack_token(coordinator)
    backend = DstackBackend(project=coordinator.project, server_url=coordinator.server_url, token=token)
    result = backend._command(["offer", "--project", coordinator.project, "--json", "--fleet", fleet.name,
                               "--reuse", "--cpu", "12..", "--memory", "40GB..", "--gpu", "1", "--disk", "50GB.."])
    offers = json.loads(result.stdout)["offers"]
    print(json.dumps({"fleet": fleet.name, "offers": offers}))


def main(arguments=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    if arguments not in (["check"], ["inventory"]) and (len(arguments) < 2
            or arguments[0] != "experiment" or arguments[1] not in {
                "operator-preflight", "launch", "status", "follow", "logs", "cancel", "retry", "resume-submit"}):
        raise LaunchError("Use check, inventory or a reviewed experiment lifecycle operation")
    if arguments[:2] == ["experiment", "follow"] and os.environ.get("GRADLAB_TRAINING_FOLLOW") != "1":
        raise LaunchError("Use the reviewed long-lived follow profile")
    from gradlab.training_vault import validate_session
    validate_session()
    # Preserve the existing command-scoped HTTPS/gh transport for domain Git work.
    os.environ.update({
        "GIT_CONFIG_COUNT": "3",
        "GIT_CONFIG_KEY_0": "url.https://github.com/.insteadOf", "GIT_CONFIG_VALUE_0": "git@github.com:",
        "GIT_CONFIG_KEY_1": "credential.helper", "GIT_CONFIG_VALUE_1": "",
        "GIT_CONFIG_KEY_2": "credential.https://github.com.helper", "GIT_CONFIG_VALUE_2": "!gh auth git-credential",
    })
    if arguments == ["check"]:
        check_storage()
    elif arguments == ["inventory"]:
        from gradlab.training_vault import TrainingVaultCommandError

        try:
            inventory()
        except TrainingVaultCommandError as error:
            print(json.dumps({"probe_error": error.category}))
            raise SystemExit(1) from None
    else:
        if arguments[:2] == ["experiment", "launch"] and "--json" not in arguments:
            arguments.append("--json")
        sys.argv = ["gradlab", *arguments]
        runpy.run_module("gradlab.main", run_name="__main__")


if __name__ == "__main__":
    try:
        main()
    except (LaunchError, OSError, ValueError, KeyError):
        print("Training access unavailable; private details withheld. No fallback.", file=sys.stderr)
        raise SystemExit(1) from None
