# MLflow pilot

This template prepares a single-host, version-pinned MLflow 3.16.1 service. It does not enroll a compute fleet or authorize a live training campaign. Hostnames, credentials, endpoints, capacity, and enrollment evidence belong in the operator-local inventory.

## Installation and route

Install the checked-in `mlflow-server` dependency group from a pinned source revision with `uv sync --frozen --only-group mlflow-server --no-install-project`. The group includes the S3 client needed for proxied R2 artifact storage. Place that environment under `/opt/gradlab/mlflow/.venv` and install the sample systemd unit after adapting its paths. Give the service user exclusive access to `/var/lib/gradlab/mlflow`. The service binds only to host loopback; expose its writer API to approved operators and training containers through a private authenticated TLS route. Keep the writer route inaccessible from the public Internet. Do not put its credentials in a recipe, Run manifest, image, or Modal worker.

Use MLflow's `basic-auth` app with a private server environment file (`0600`). It needs `MLFLOW_FLASK_SERVER_SECRET_KEY`, a one-time `MLFLOW_AUTH_ADMIN_PASSWORD` of at least 12 characters for initial setup, `GRADLAB_MLFLOW_ARTIFACTS_DESTINATION=s3://<separate-private-bucket>/gradlab`, `MLFLOW_S3_ENDPOINT_URL`, and server-side `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`. Remove the bootstrap admin password after the first start and issue a dedicated writer account with only the necessary experiment permissions. Keep the auth SQLite database and tracking SQLite database on persistent host storage and back up both. MLflow's proxied artifact mode keeps R2 credentials on the server; clients need only `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, and `MLFLOW_TRACKING_PASSWORD`.

Before enabling a compute fleet, verify its private route and exact credentials from inside that fleet's training image, and perform an artifact upload/download round trip through the MLflow server. Set `MLFLOW_OPERATOR_PROFILE` to one logical destination name and `MLFLOW_ALLOWED_FLEETS` to the comma-separated exact dstack fleet IDs verified for that route. Launch preflight rejects other compute targets and records the profile in the immutable Run manifest. An unreachable or unauthenticated selected service also fails preflight. Keep the MLflow artifact bucket distinct from GradLab's control, evaluation, and public-model buckets. The GradLab journal remains the scientific authority if MLflow disagrees.

For a private CA, place its single PEM certificate in operator configuration as base64 `GRADLAB_MLFLOW_TLS_CA_B64`, and trust that CA on the operator machine for launch preflight. The queued task receives the CA through dstack secrets, appends it to the system trust bundle, and passes that bundle to MLflow's HTTP client. The Run manifest records that a private CA is required so retries cannot silently omit it. The CA certificate is public material; keep its private signing key only on the operator-controlled host.

## Optional public inspection

Anonymous inspection is opt-in. Keep MLflow basic authentication enabled on the
writer service. Create a separate reader account with `NO_PERMISSIONS` by
default and explicit `READ` grants only for experiments approved for public
inspection. All tags, parameters, histories, and artifacts in those experiments
become public, so keep credentials and operator-only data out of them.

Install `public_gateway.py` under `/opt/gradlab/mlflow/public` and adapt
`gradlab-mlflow-public.service.example`. Run it with Gunicorn from the pinned
server environment. Put `GRADLAB_MLFLOW_READER_UPSTREAM=http://127.0.0.1:5001`,
`GRADLAB_MLFLOW_READER_USERNAME`, and `GRADLAB_MLFLOW_READER_PASSWORD` in
`/etc/gradlab/mlflow/reader.env`, readable only by root and the service group.
The gateway accepts only loopback upstreams, replaces visitor credentials with
the reader identity, and permits explicit UI, history, artifact-download, and
search routes. GraphQL is limited to approved read query roots; mutations and
unrecognized routes are rejected before reaching MLflow. The upstream reader
ACL also restricts which experiments are visible. MLflow's edit controls may
remain visible but their requests are denied.

Expose only gateway port 5003 through the public HTTPS route. A Cloudflare Quick
Tunnel can provide a temporary pilot URL without an account-owned hostname;
it has no availability guarantee and obtains a new random URL after restart.
Use a managed tunnel with a stable hostname for ongoing service. Never expose
port 5001 or the private writer TLS route through either tunnel. Verify anonymous
charts and video in a browser, public write rejection, and rejection of a Run
outside the reader's grants before publishing the URL.

Published Checkpoint videos also have a durable, hash-verified projection in the
public model bucket. GradLab's catalog links those recordings from the journal
telemetry, so their access does not depend on this optional gateway or tunnel.

## Backup and recovery

Back up both SQLite files using `python ops/mlflow/sqlite_snapshot.py backup <database> <snapshot>` while the service is running. The script uses SQLite's online backup API, verifies integrity, and writes a SHA-256 receipt beside the snapshot. Copy snapshots and receipts to an operator-controlled durable location, along with a versioned inventory of the artifact bucket. Test restoration into an isolated private service and verify Run IDs, metric histories, media, and permissions before replacing a failed service. Stop the service before replacing a live database; preserve the failed bytes and receipts. Do not rewrite GradLab terminal receipts. If restoration is impossible, point private operator configuration at the replacement service and run `gradlab rebind-mlflow <run-id> --operator <name> --reason <incident> --confirm-irrecoverable`. It replays the verified journal and media, confirms remote visibility, and appends an audit record before atomically replacing the mutable service binding. Preserve incident evidence alongside backup receipts.

## Pilot gate

The bounded live pilot requires separate compute authorization. Compare matched W&B and MLflow Runs, measure metric ingest rate, remote-visibility latency, database and artifact growth, representative-video access, server CPU/memory, and learner throughput. Verify GradLab public views, private-R2 journal integrity, restore, and resource release. Keep MLflow unavailable as a selectable production target on other fleets until private route and capacity checks pass there.
