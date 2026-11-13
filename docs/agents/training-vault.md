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

Installation, bootstrap and rotation are trusted administrative operations,
separate from application startup. The administrative helper validates source
scope and current native fake-credential proof. Runtime refresh prepares a
metadata-only candidate for explicit review; it never overwrites an active run's
profile or approves drift automatically. Preserve existing broker processes and
immutable published-source snapshots during maintenance.

`GRADLAB_DSTACK_EXECUTABLE` selects a reviewed absolute coordinator CLI. Explicit
backend constructor arguments take precedence; otherwise the default is `dstack`.
