# Training datasets

This specification governs opt-in collection of training trajectories and explicit
dataset Publication. It records the requirements approved in issue #50, including
the clarified native Breakout action-evidence contract.

## Capture and scientific meaning

- Collection must be explicitly enabled in the resolved Run Configuration and initially support only the verified native Breakout training contract, with both GradLab PPO and SB3 PPO/A2C.
- Collection must reuse training transitions without changing Policy sampling, random streams, rewards, action overrides, resets, curricula, episode boundaries, evaluation admission, Acceptance, or Promotion.
- Select episodes at reset independently of their eventual reward or outcome, spread admission across the planned training duration, and expose availability-driven sampling gaps.
- Preserve full unmasked lossless provider RGB at the contracted action cadence, including initial and true pre-reset terminal frames, without intermediate native frames or repeated processed Policy stacks.
- Every transition must preserve the original Policy action, the action executed by the environment, applicable intermediate action and native encoding, the action contract, and any override reason.
- For the exact verified native Breakout runtime, the post-override action at GradLab's provider execution boundary and its native encoding may establish execution evidence because the provider applies no further behavioral override. Verify this contract against provider source and integration tests; no provider-returned action field is required.
- Providers with internal action replacement, sticky actions, or unverified execution contracts must supply authoritative execution evidence or be rejected. Missing evidence must cause an explicit recording fault without fabricating actions or changing training behavior.
- Preserve aligned provider and training rewards, termination, truncation, and capture cutoffs separately; distinguish nominal action repeat from measured elapsed frames and leave unavailable facts absent.
- Index each episode by Run, Attempt, lane, episode, seed, Research Goal, Goal Revision, Goal Variant, environment, provider, runtime, source, start origin, training steps, progress against the original planned budget, and Policy-update attribution, including episodes spanning updates.
- Preserve individual shaped return, native score, absolute and normalized brick progress with its denominator, length, outcome, boundary reason, and goal-declared progress facts without substituting rolling aggregates.
- Distinguish complete episodes from contiguous recorded prefixes, retain interruption reasons and captured ranges, and keep prefix statistics separate from any independently known full-episode result.

## Resources and durability

- Bound memory, local disk, worker concurrency, and each Run's total contribution, with a configurable finite contribution default of 10 GiB that survives Attempt retries and is not replenished by local reclamation.
- Encoding, hashing, validation, writing, and network delivery must occur outside the learner's critical path; selected-data handoff must be bounded and nonblocking.
- Account for buffers, temporary files, indexes, journals, orphaned writes, and upload working space; preserve Checkpoint and supervisor headroom and pause before global scratch pressure can block training.
- Store immutable size-limited chunks with versioned data-only tabular metadata, lossless images, globally scoped identities, hashes, and unambiguous transition joins; keep metadata growth bounded.
- Upload sealed chunks continuously to scoped R2 storage, verify identity, size, checksum, and recoverable manifests before local reclamation, and reconcile lost acknowledgements without duplicate contributions.
- R2 remains canonical until explicit cleanup, including after HF Publication. Training needs neither an HF target nor HF credentials and must never start HF Publication automatically.
- The supervisor must own worker lifecycle and dataset delivery, retain its writer lease, preserve existing lifecycle guarantees, and require a complete verified dataset inventory in the terminal receipt before successful completion.
- A dedicated finite configurable dataset-drain deadline must produce honest failure with recoverable evidence when delivery cannot finish; cancellation preserves explicit prefixes, and host loss must not be represented as lossless recovery of unuploaded bytes.
- Only the lease-holding supervisor may publish W&B diagnostics, including capture, encoding and upload rates, pending bytes and age, actual spool bytes, failures, pauses and reasons, skipped episodes, and incomplete recordings through terminal drain.
- Supported capture must have repeated matched collection-on/off throughput evidence with meaningful nonzero capture and a target of at most 2% throughput loss; report sampling, captured volume, resource use, and backlog. This requirement does not authorize training or a compute purchase.

## Explicit dataset Publication

- An explicit local command must select finalized verified Run inventories and an HF dataset target, with optional stage/performance predicates and explicit prefix inclusion; publish complete episodes by default.
- Filter lightweight indexes before transferring images, use bounded working sets, and preserve reconstructable frame/action sequences and immutable source identities.
- Use the existing durable local queue so publication survives requesting-client exit, supports safe resume, and reports status plus immutable HF revision evidence without a separately operated service or database.
- Append immutable contributions without losing previous work; retries, overlapping selections, concurrent publishers, and lost commit responses must not duplicate episodes or overwrite conflicting identities.
- Reject incompatible schemas, execution contracts, or equal identities with different content; expose complete asset/metadata sets atomically using expected-parent commits or equivalent concurrency control.
- Record selected Runs, predicates, and the resulting episode inventory; retain grouping identities and leave split assignment to consumers, with whole-Run or seed-group holdout recommended.
- Dataset Publication must not change training state, allocate evaluation seeds, create Acceptance evidence, delete R2 sources, or reinterpret existing isolated-collector datasets.

## Verification

- Use the existing shared runtime/provider, deterministic lifecycle certification, local queue, and Hub adapter boundaries to test observable trajectories, resource bounds, durable artifacts, receipts, and publication contents.
- Cover actual auto-serve overrides, ordinary and lane-specific actions, missing evidence, mutable-buffer reuse, distinct terminal/reset RGB, seeded trajectory equivalence, update-spanning episodes, incomplete prefixes, resource saturation, delivery faults, recovery, cancellation, and concurrent/idempotent Publication.
