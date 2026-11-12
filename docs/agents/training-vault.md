# Training through Agent Vault

The reviewed local training route uses Infisical Agent Vault 0.45.6 with a pinned
HTTPS instance, proxy CA and private metadata profile. Operator commands and
queued supervisors receive public placeholders and finite session capabilities.
Only the trusted proxy and signing gateway handle the original service keys.

```sh
python3 -I -S -B scripts/training_vault.py check
python3 -I -S -B scripts/training_vault.py inventory
python3 -I -S -B scripts/training_vault.py experiment operator-preflight --recipe-file RECIPE --json
python3 -I -S -B scripts/training_vault.py experiment launch --recipe-file RECIPE --seed SEED --compute local --max-duration DURATION --existing-runtime-only --json
python3 -I -S -B scripts/training_vault.py experiment follow --run RUN_ID
```

The launcher also supports `status`, `logs`, `cancel`, `retry` and
`resume-submit`. These commands retain their normal scientific and authorization
rules. The check writes synthetic objects to each original bucket, verifies
hashes, metadata, list visibility and conditional mutations, and deletes its
objects. It does not admit training. Do not print proxy URLs or presigned URLs.
`inventory` reads coordinator offers for the configured fleet at the reviewed
12-CPU, 40-GB, one-GPU, 50-GB resource floor; it never allocates an instance.

## Bindings

The Gradlab Secret Manager source remains the seven exact `prod`, `/` references
in the reviewed private operator configuration: the access ID and signing secret
for each of control, evaluation and models storage, and the selected coordinator
token. Its native development Hugging Face binding remains separate. Training's
Vault project and bundle IDs are explicitly pinned; native source metadata does
not configure service permissions.
Vault mode requires the original S3 buckets. Local paths, swapped roles and
unreviewed prefixes fail before the storage client starts.

| Service | Operator | Supervisor | Credential handling |
| --- | --- | --- | --- |
| Three original R2 buckets | Yes | Yes | Native Basic substitution into one fixed-account, fixed-bucket SigV4 gateway per role |
| Selected local dstack API | Yes | No | Native bearer substitution |
| Existing private MLflow API | Yes | Yes | Network authentication, verified private TLS CA |
| Public model objects | Read | Read | No source credential |
| Runtime receipt APIs and signed artifact downloads | Read | No | Existing local GitHub CLI authentication remains separate |
| Session self-revocation | Yes | Yes | The caller's own finite session token |

The native proxy binds to the selected host's Docker bridge. The operator uses
a command-owned forward over its existing managed SSH master. Gateways bind
only to host loopback. The reviewed backend token permissions are unchanged;
neither Cloudflare REST permissions nor new account tokens are required.
Gateways accept GET, HEAD, PUT and DELETE and bodies up to 256 MiB. Multipart
transfers and temporary AWS credentials require separate compatibility work.

## Durable sessions and cleanup

Each launch or retry creates a separate seven-hour worker session. A unique
per-Attempt reference transfers only that capability through dstack's encrypted
secret store before immutable manifest admission. Manifests contain its ID,
expiry, proxy address and reference, never the token. `resume-submit` reuses
the admitted reference; retry creates a fresh session. Once child startup is
attempted, a failed or interrupted submission retains its finite worker
capability for recovery: a lost acknowledgement must not revoke an admitted
worker's access. Failures before child startup revoke the unused session.

The supervisor revokes its session after finalization, including failures.
Native proxy refresh can briefly retain a revoked session; the dedicated proxy
polls every ten seconds. Expiry bounds hard-kill failures. The offline learner
receives neither the capability nor source credentials. Operator sessions are
revoked when their native launcher exits.

## Trusted administration

Provisioning and rotation are separate from ordinary runs. The maintained
`scripts/training_vault_admin.py` validates the exact source bindings and a
successful native fake-credential proof before provisioning. It copies service
keys only in administrative process memory and HTTPS request bodies, creates
dedicated bundles/services, and installs checksum-verified native Linux tooling
and pinned broker code under a root-owned directory. No credential files are
written. The proxy bootstrap is kept encrypted in Gradlab development at
`/agents/training-proxy`; it reaches the protected broker over SSH stdin.

The broker survives the submitting SSH client and restarts its native child
from memory. Reboot or broker termination loses that memory. After verifying
the existing broker is stopped and resolving any stale PID record, trusted
administration runs `training_vault_admin.py bootstrap`; it never recreates a
local token cache. Changes to code/runtime/CA/operator metadata require review,
fake-first verification and `refresh-profile`, rather than automatic repinning
by the ordinary launcher. An occupied port or uncertain process must be
inspected without replacing unrelated work.

Run the credential-free unit coverage with the locked development environment:

```sh
uv run --frozen pytest tests/test_training_vault.py tests/test_r2_signing_gateway.py tests/test_r2_store_config.py tests/test_dstack_backend.py tests/test_operator_credentials.py
```

The native integration fixture `tests/fixtures/training_vault_native.py` creates
disposable fake-key services and exercises the actual boto, dstack and MLflow
clients, host/path denials, error redaction, worker separation, session survival
after creator exit, revocation and cleanup. It needs approved native Vault
administration, but never reads original service keys.

This route does not migrate Modal, W&B, YouTube or durable Publication. It does
not isolate a full-access same-user agent from credential administrators. Keep
existing global and standalone Hugging Face launchers unchanged.
