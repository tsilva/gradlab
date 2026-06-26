# Structured Experiment Specs

Experiment specs are checked-in JSON documents that describe the hypothesis,
training delta, seed set, and selection gate for a campaign candidate.

Use them when the queue should own the experiment payload instead of relying on
ad hoc shell history.

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-campaign add-spec-file \
  experiments/specs/mario-level1/b55-lowkl-lrdecay-post21-revalidate.json

UV_CACHE_DIR=.uv-cache uv run rlab-campaign enqueue-train-from-spec \
  experiments/specs/mario-level1/b55-lowkl-lrdecay-post21-revalidate.json
```

The spec file is allowed to contain `run_name_template` and
`run_description_template` values with `{seed}` and `{utc}` placeholders.
`train_config` should omit secrets and should not include seed-specific fields
unless the spec intentionally runs a single seed.

Training enqueue defaults are profileless and use the latest successful
train-image digest. Pass `--runtime-image-ref-file` or `--runtime-image-ref` only
to pin a non-latest digest, and pass `--profile` only for an intentionally
profile-locked lane.
