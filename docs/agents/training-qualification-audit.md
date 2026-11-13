# Completed Mario training audit

Use the normal status command to audit a completed fresh campaign Run without
changing its state or admitting evaluation:

```sh
gradlab experiment status --run RUN_ID --audit-training --json
```

The audit requires task release and complete delivery. It reads the immutable
recipe, verifies every retained journal segment and its contiguous event
sequence, and checks the real MLflow Run identity and delivery watermark. The
supported contract is native PPO with one canonical Mario level start, fresh
initialization, no archive curriculum and no automatic evaluation or Monitoring.
The learner's own SIL replay is allowed; it cannot add target-origin episodes.

Delivery-only recovery does not invalidate fresh training. Every later Attempt
must explicitly be drain-only, preserve the frozen scientific contract, and
retain the original learning boundary without new training-history events. The
audit reports the learning Attempt and stop reason separately from the delivery
Attempt and stop reason. Training resumes do not satisfy this fresh-run audit.

The Run's declared environment identity and the artifact's resolved environment
identity are checked separately. The audit reconstructs the former from the
stored training configuration and the latter from its effective environment
defaults and frozen ROM binding. It requires the stored artifact metadata and
ROM identity to match that reconstruction; the two hashes need not be equal.

For each saved checkpoint, the audit matches the training observation at that
exact transition step. It selects the highest retained qualifying window, with
earlier checkpoints breaking ties. An unretained history peak is reported
separately and is never assigned to another model. The selected window, cumulative
episode count and cumulative clear count must also exist at the same MLflow step.
The model is then downloaded through the configured models store and its byte
length and SHA-256 are checked against the publication receipt. No model file
or credentials are persisted by the audit.

The result contains the selected checkpoint, genuine 100-episode window,
cumulative counts, source/runtime and configuration identities, remote model
verification and MLflow evidence. Runs without a qualifying retained checkpoint
return `qualified: false`. Corrupt, incomplete or unsupported evidence fails
closed. This training criterion does not establish frozen-checkpoint quality,
Acceptance, Promotion or reliability across training seeds.

Reads are bounded to 50,000 inventory objects, 20,000 journal segments, 128 MiB
of journal content and a 256 MiB model. Journal downloads are sequential to leave shared gateway
capacity for admitted training and existing observers. A Run exceeding these limits needs a separately reviewed
verification procedure; do not bypass the bounds or substitute credentials.

Select any personal access tooling externally, according to the operator's
approved setup. The package command consumes ordinary environment settings.
