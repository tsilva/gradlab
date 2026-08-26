# Compute operations

This file defines GradLab's portable compute policy. Machine names, fleet
names, hostnames, SSH identities, operator endpoints, enrollment state, and
host-specific capacity belong in the operator-local inventory at
`~/.config/gradlab/instances.md`.

## Operator configuration

Copy `ops/operator.example.toml` to
`~/.config/gradlab/operator.toml`. Its schema-v3 `[dstack]` section names the
default coordinator and fleet. Each coordinator records an immutable ID,
project, private endpoint, Keychain token reference, and optional SSH tunnel
metadata; each fleet records its owning coordinator and task resources.

The file may contain non-sensitive metadata and Keychain references only.
Secrets remain in the referenced credential store or an explicit process
environment. Verify the resolved project, fleet, source, and storage access
before launching:

```bash
gradlab experiment operator-preflight --json
```

An explicit `--target` overrides `dstack.default_fleet`. Local or automatic
compute fails closed when the selected fleet is not configured.

## Selection policy

| Need | Request |
| --- | --- |
| Use the configured local fleet, otherwise wait | `--compute auto` |
| Require the configured local fleet | `--compute local` |
| Override the local fleet for one launch | `--compute local --target <fleet>` |
| Prefer local, then permit bounded spot | `--compute auto --max-price … --max-cost-usd …` |
| Permit bounded spot compute | `--compute spot --max-price … --max-cost-usd …` |
| Permit on-demand compute | `--compute on-demand --allow-on-demand --max-cost-usd …` |
| Short CPU evaluation | Modal, dispatched by the training supervisor |

`auto` never enters paid cloud compute without finite hourly-price and
total-cost bounds. On-demand remains opt-in even with a budget. Every task has a
finite `max_duration`.

GradLab v1 schedules one training container per single-GPU host. The generic
workload floor remains 12 CPU, 40 GiB memory, one GPU, and 50 GiB disk; local
inventory records whether a particular host satisfies it. An explicitly
authorized host-specific exception may declare a different task request under
`[dstack.fleets.<fleet>]` in private `operator.toml`; the resolved CPU, memory,
GPU, and disk request is frozen into the run manifest and applies only to that
fleet.

## dstack control plane

GradLab pins dstack CLI and server `0.20.28`. A deployment must keep its API
private, its state host-owned, and its secrets outside source control. Operators
normally reach the endpoint through SSH destinations recorded under the
coordinator's `ssh_tunnel` entry in private `operator.toml`.

Every dstack-backed lifecycle command reuses a reachable loopback endpoint when
one already exists. Otherwise, GradLab opens a process-scoped SSH tunnel,
trying that coordinator's destinations in order, and keeps it alive through
submission and `--follow`. Command exit or interruption closes only a tunnel
started by that GradLab process; an existing operator-owned tunnel is left
alone. Destination fallback is transport fallback to the same coordinator,
never coordinator or fleet failover. GradLab does not upgrade dstack or install
a persistent tunnel service.

`gradlab experiment launch` refreshes encrypted project-scoped dstack secrets
from the operator's private environment. Submitted tasks contain secret
references, never credential values. Use `gradlab experiment status`, `follow`,
`logs`, and `cancel` instead of treating raw dstack inventory as the run
authority.

Each attempt has a create-only `coordinator.json` beside its immutable manifest.
It binds the coordinator ID, project, target, manifest hash, basis, and binding
time before dstack submission. Every lifecycle command resolves only that
coordinator and never probes another server. Retries retain the coordinator,
fleet, and resources without failover. Read-only status can report an
unreachable coordinator alongside authoritative R2 state; dstack mutations fail
closed.

Automatic retries are limited to dstack `no-capacity` and genuine
`interruption`. Generic errors require an evidence-backed manual retry. The CLI
requires the previous task to be terminal, the R2 writer lease to expire, and a
quiescence interval. A finished learner retries in drain-only mode.

## Local fleet templates

Portable examples live under `ops/dstack/`. Copy and specialize them outside
the repository, for example beneath `~/.config/gradlab/dstack/`. Never check in
real hostnames, SSH users, identity paths, or fleet names.

## Modal evaluation

Modal is the v1 evaluation backend for short CPU acceptance jobs. It receives
no W&B or control-private credentials and writes only evaluation-private
results/evidence. The lease-holding training supervisor projects accepted
evaluation metrics into W&B.

A native Modal hard budget must be configured before a final acceptance launch.
Per-run forecasts or alerts are not distributed reservations.

## Host image cleanup

dstack does not safely prune all unused runtime images. Enrolled local hosts
may install the root-owned timer and script under `ops/dstack/`.

`/etc/gradlab/dstack/server.env` must explicitly provide
`DSTACK_SERVER_ADMIN_TOKEN`, `DSTACK_PROJECT`, and
`GRADLAB_IMAGE_REPOSITORY`. Cleanup fails closed when any is absent or dstack
inventory is invalid. It preserves:

- images used by running containers;
- immutable images demanded by pending, submitted, provisioning, running, or
  terminating dstack tasks.

It never removes unrelated images, containers, volumes, or build cache. Use
`GRADLAB_IMAGE_CLEANUP_DRY_RUN=1` to audit a host before enabling deletion.

## Runtime-image pipeline

Training starts only after the exact source SHA has a verified immutable image
receipt. The runtime combines a CUDA/PyTorch foundation, a disjoint non-GPU
dependency environment, and an exact-source application overlay. Generic
runtime history belongs in release or experiment reports; machine-specific
benchmark and access facts belong in the operator-local inventory.

## Operational checklist

Before launch:

1. Read this file and, for local compute, `~/.config/gradlab/instances.md`.
2. Confirm the selected goal and recipe.
3. Confirm the Git revision is clean and pushed.
4. Run `gradlab experiment operator-preflight --json`.
5. Confirm exact-source runtime and evaluation deployment receipts.
6. Confirm budget bounds and a finite maximum duration.

After terminal state, verify the private R2 terminal receipt, dstack resource
release, public checkpoint/index access, and credential-free playback. A
successful dstack task without the terminal receipt is an operational failure.
