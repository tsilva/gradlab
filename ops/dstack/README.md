# dstack control plane

GradLab pins dstack CLI and server `0.20.28`. The server image is pinned to the
multi-platform digest
`sha256:86b820cf5f6e0cfc54dd387527493168a4045b362ca9459265ea9828eef0b4af`.

Concrete deployment state, endpoint details, fleet membership, hostnames, SSH
identities, and capacity belong outside source control. Record them in
`~/.config/gradlab/instances.md` and keep specialized fleet YAML beneath
`~/.config/gradlab/dstack/`. The tracked fleet and task files are placeholders
to copy, not shared infrastructure declarations.

The dstack API must remain private. The checked-in systemd unit runs the pinned
image directly, so a host does not need Docker Compose.

Host-owned configuration and secrets include:

- `/etc/gradlab/dstack/server.env`, which must provide the server admin token,
  dstack project, and GradLab image repository;
- the dstack state directory and AES-256-GCM project encryption key;
- local `DSTACK_TOKEN` and endpoint metadata from private operator
  configuration.

`gradlab experiment launch` synchronizes workload credentials into encrypted,
project-scoped dstack secrets and submits only `${{ secrets.NAME }}`
references. It never embeds credentials in a task configuration.

Do not use raw `dstack ps --json` as the operator-facing status interface. Use
`gradlab experiment status` or `follow`, which expose a small allowlist of
dstack fields plus authoritative R2 semantic state.

## Read-only local ROM cache

An enrolled local host may keep source ROM bytes in a host-owned directory.
Install `gradlab-rom-cache-mount` and
`gradlab-rom-cache-readonly.service` to expose them through a
kernel-enforced read-only bind mount. Configure the concrete source and mount
paths on the host and record them only in the private inventory.

## NVIDIA detection workaround

dstack 0.20.28 checks `/dev/kfd` before `/dev/nvidiactl`. On a host where an
unused AMD integrated GPU masks the NVIDIA training GPU, install
`dstack-shim-override.conf` as a dstack shim drop-in. It removes only the
unused AMD compute node and regenerates host inventory on shim start; it does
not remove `/dev/dri`. Do not install it on hosts that schedule AMD compute.

## Container DNS

If a host's Docker containers would otherwise retain stale router DNS, configure
stable resolvers in the Docker daemon and dstack server unit. Preserve the
host's system resolver and split-DNS configuration; this setting applies only
to containers.

## Host image cleanup

The pinned runner removes terminated task containers but does not prune all
unused images. Install the isolated pinned dstack CLI at
`/opt/gradlab/dstack-cli/bin/dstack`, install
`gradlab-dstack-image-cleanup` under `/usr/local/libexec`, and enable
`gradlab-dstack-image-cleanup.timer` only after an audit-only run.

The host-owned `/etc/gradlab/dstack/server.env` must explicitly set
`DSTACK_SERVER_ADMIN_TOKEN`, `DSTACK_PROJECT`, and
`GRADLAB_IMAGE_REPOSITORY`; the script has no project or repository fallback.
It fails closed unless it obtains valid run inventory, preserves images demanded
by pending or active tasks and images used by running containers, and considers
only immutable images in the configured repository. It does not prune other
images, containers, volumes, or build cache. Set
`GRADLAB_IMAGE_CLEANUP_DRY_RUN=1` for an audit-only invocation.

## R2 metric-journal expiry

The control-private bucket must have an object lifecycle rule named
`expire-delivered-metric-journals` for prefix
`expiring-metric-journals/`, expiring objects after seven days. Active journals
remain beneath the attempt until W&B remote visibility and terminal drain gates
pass; only then does the supervisor atomically relocate them.

## Learner-supervision fault fixture

`gradlab experiment fault-test --mode failed-result-live-process --json` creates
a fresh training-only logical run on the configured local fleet with an
exact-source runtime image, a short task cap, and manifest-bound teardown
deadlines. An explicit `--target` overrides the private fleet setting.

The learner writes an identity-bound failed result, leaves a child process
alive, and waits. The real supervisor must detect it, close evaluation
admission, reap the process group, write an R2 resumable-failure receipt,
project it to W&B, and exit nonzero. dstack must terminalize without retry
unless the failure is `no-capacity` or `interruption`.

The alternate `completed-result-hung-process` mode ignores graceful stop and
SIGTERM so teardown must reach SIGKILL and terminalize with
`teardown_timeout`. Neither fixture constructs an environment or trains a
policy.
