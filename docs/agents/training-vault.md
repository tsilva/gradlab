# Optional training transport

Normal GradLab training interfaces are independent of personal secret tooling.
An approved local Runbox profile can supply scoped placeholder-based operator
access. The shared runner owns configuration/runtime integrity, public CA trust,
managed forwarding and bounded provider sessions. Domain code owns storage,
coordinator and experiment behavior; failed access never becomes raw injection.

Use the reviewed `gradlab-training` command for inventory and experiment lifecycle
operations through Runbox. Inventory is read-only. The separate `check` operation
performs synthetic conditional R2 writes and cleanup and requires that operation's
authorization; an ordinary online access check uses inventory. `gradlab-follow`
selects the existing longer-lived monitoring session. Do not start training,
retry, cancellation or storage writes merely to verify a migration.

## Durable ownership

Launch and retry create separate finite seven-hour worker capabilities. A unique
context accompanies each admission. Once child startup is attempted, retain that
capability even when the acknowledgement is lost: the workload may have been
admitted. The supervisor or bounded expiry owns cleanup. Failures before child
startup revoke an unused session. Retry receives a fresh capability; resume uses
the admitted reference. Operator sessions remain separate and revoke on exit.

The supervisor revokes its capability during finalization. The offline learner
receives neither capabilities nor source credentials. Keep the fixed coordinator,
original R2 buckets, role separation, conditional object semantics and existing
publication/experiment guards. A credential proxy does not isolate full-access
same-user processes, and this route adds no OS sandbox.

The operator may select its reviewed absolute CLI with
`GRADLAB_DSTACK_EXECUTABLE`. An explicit backend constructor command takes
precedence; otherwise the normal default remains `dstack`.

## Trusted administration

Broker installation, bootstrap and source rotation remain separate trusted
operations. Ordinary startup never retrieves source keys. The administrative
helper retains source validation and requires a current native fake-credential
proof; old proof does not approve changed Runbox or domain code. Runtime refresh
prepares a metadata-only candidate for explicit review and never overwrites an
active run's profile or automatically approves drift. Preserve current broker
processes and immutable published-source snapshots during unrelated maintenance.

Verify domain guards with the locked unit suite and shared orchestration with
synthetic native CLI, admission and managed-SSH fixtures before bounded live
reads. Live GPU jobs and real storage writes are separate qualification work.
