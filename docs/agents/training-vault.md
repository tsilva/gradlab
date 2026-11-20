# Worker access and trusted administration

Operators and supervisors use separate finite capabilities. Never pass them to
offline learners. Launch and retry create fresh seven-hour worker sessions with
unique admission contexts. Once child startup is attempted, retain its capability
even when acknowledgement is lost: admission may have succeeded. The supervisor
or expiry owns cleanup. Failures before child startup revoke an unused session.
Resume uses the admitted reference; retry receives a fresh capability.

Keep the fixed coordinator, original storage buckets, role separation,
conditional-object semantics and publication guards. Missing access fails closed.
`inventory` is read-only; `check` performs synthetic conditional storage writes and
cleanup and needs explicit authorization. Neither should start training merely
to verify access.

The operator lifecycle adapter includes `experiment reconcile` for a terminal
compute task missing its Attempt receipt. It validates the existing operator
session and preserves the CLI's evidence hashes and failure reason. It creates
no worker capability or training task. Adopting changed protected adapter source
requires a fresh explicit Runbox consumer review.

The same operator adapter accepts `autoresearch reconcile` and
`autoresearch collect-training-evidence`. These commands read existing Run
evidence through the selected training transport and update only the local study.
The agent command starts with
`runbox --profile training run -- gradlab-training autoresearch`.
A missing or incompatible agent session fails before reading operator
credential sources.

Before reserving another study wave, query the study's pinned fleet with
`runbox --profile training run -- gradlab-training reservations --study <study>`.
The command returns the fleet and its active GradLab reservation count. An
incomplete listing or an unresolved fleet target stops the count. Resolve
effective capacity separately from the configured fleet inventory.

For an agent-operated player that needs the private catalog, set
`GRADLAB_CATALOG_EXECUTOR` to a JSON argument array containing an absolute
Runbox executable, `--repo` and the canonical checkout,
`--profile playback-catalog`, and `run -- gradlab-catalog --socket`. The
player appends a private socket path and the helper performs only bounded
catalog reads. Its reviewed runtime enables Runbox's shared forwarding on a
dedicated local port. Each player closes its own connection on exit. Runbox
keeps the forward available until the last catalog helper exits.

GradLab accepts `GRADLAB_ACTIONS_READ_EXECUTOR` for GitHub Actions run listings
and runtime artifact downloads, and `GRADLAB_ACTIONS_DISPATCH_EXECUTOR` for
workflow dispatch. Each takes a JSON array starting with an absolute executable.
GradLab appends the GitHub CLI arguments and expects the same JSON or ZIP output.
Set the read executor alone for read-only access. With either executor configured,
an operation whose executor is missing or fails stops. When both are unset,
GradLab uses the operator's GitHub CLI as usual.

If an external dispatch needs to protect nonterminal Modal evaluators, select
`GRADLAB_MODAL_PROTECTION_EXECUTOR` with an absolute command array. Use the
reviewed read-only control-storage route for
`python -m gradlab.modal_protection`. It requires a scoped
Training Vault session and returns `complete: true` with the protected app
names only after reading the full inventory. Missing or incomplete inventory
blocks dispatch before the operator's storage credentials are read. Ordinary
operator dispatch can still use its configured storage directly.

Installation, bootstrap and rotation are trusted administrative operations,
separate from application startup. The administrative helper validates source
scope and current native fake-credential proof. Runtime refresh prepares a
metadata-only candidate for explicit review; it never overwrites an active run's
profile or approves drift automatically. Preserve existing broker processes and
immutable published-source snapshots during maintenance.

`GRADLAB_DSTACK_EXECUTABLE` selects a reviewed absolute coordinator CLI. Explicit
backend constructor arguments take precedence; otherwise the default is `dstack`.
