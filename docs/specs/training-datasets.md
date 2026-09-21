# Checkpoint Monitoring and datasets

This specification governs issue #52's opt-in Checkpoint Monitoring and explicit
dataset Publication, replacing issue #50's active training-transition capture.
Historical immutable datasets retain their meaning; the isolated collector is
unchanged. No silent legacy execution fallback is permitted.

## Scientific contract

- Initially support verified native Breakout Checkpoints produced by GradLab PPO and SB3 PPO/A2C, including Training-Only Goals without inventing Acceptance criteria.
- Monitoring must never feed the learner, grant Acceptance, trigger Promotion or Release, or stop training because of its scores; existing Modal Acceptance authority remains unchanged.
- Freeze the Run/Attempt/Checkpoint hash and training step, Goal/Revision/Variant, environment/provider/source/runtime, starts, actions, rewards, termination, limits, episode manifest, metric schema, resources and representative-video rule.
- Target 400 complete episodes per unique saved Checkpoint, including final; keep the full manifest and cadence fixed during training, retain older queued Checkpoints, and deduplicate immutable identities.
- Give every episode immutable identity and distinct environment and stochastic Policy-sampling seeds, independent of lane, worker, vector width, retries or active/finalization scheduling; keep monitoring seeds separate from training and untouched Acceptance seeds under centralized ownership.
- Preserve the Goal's starts, Policy inputs, action cadence/overrides, rewards and scientific episode boundaries; FirstWall must remain FirstWall. Watchdog expiry and worker errors are operational failures, not fabricated failed episodes.

## Complete evidence and metrics

- Record every declared episode, including successes and failures, with full unmasked lossless native RGB at contracted action cadence, initial image and true terminal image before reset.
- Record Policy-requested, applicable effective and executed actions, native encoding and override reasons; the exact verified native Breakout submission boundary is sufficient only because that provider performs no further replacement. Reject unsupported or missing evidence.
- Preserve provider/shaped rewards, termination, truncation, interruptions, available timing, start facts, native score, shaped return, length, outcomes, absolute/normalized brick progress and its denominator.
- Preserve normalized brick progress as a float with denominator 216; FirstWall success at 108 bricks has progress 0.5, distinct from success fraction. Missing required score/progress is invalid, never zero-filled.
- Use checkpoint/evaluation provenance rather than invented live-trainer counters. Complete episodes may span immutable byte-bounded chunks with ordered hashes, contiguous global transition indices and verifiable frame joins; old 8,192-transition and one-chunk lifetime limits do not apply.
- Diagnostic interrupted prefixes remain incomplete and cannot count toward the fixed manifest or final metrics.
- Emit final metrics only after every planned episode is complete and verified: success rate and two-sided 95% Wilson interval, mean/median normalized brick progress, mean native score, mean shaped return, mean episode length and completed count.
- Name monitoring metrics with training counterparts identically except for replacing train/ with eval/; retain eval/monitor names for metrics/media without training counterparts, attribute them to the Checkpoint's eval/step, and keep supervisor delivery order on ops/sequence. Monitoring must not emit eval/pass or influence leader authority.
- Select the complete episode nearest the full-set median normalized brick progress, breaking ties by manifest ordinal; record the rule, median, selected value and identity.
- Generate one full-episode video from the selected episode's original committed RGB frames at contracted cadence, without environment replay; store canonical bytes/hashes in R2 and deliver actual playable video plus full-set metrics through the sole W&B supervisor.

## Resources, durability and recovery

- Use a separate bounded same-host CPU process during training; constrain inference/provider/encoding/delivery pools together and preserve learner memory/scratch headroom. Delivery may backpressure evaluation but never the learner.
- After learner exit use all task-allocated CPU capacity and multiple concurrent Checkpoint evaluations within shared finite memory, local-spool, cumulative R2 contribution and whole-task deadlines.
- Account for actual committed durable bytes plus bounded in-flight reservations, metadata and temporary work; deduplicate retries and preserve contribution usage across Attempts. Reclaiming local data must not replenish retained-data allowance.
- Upload sealed chunks continuously; verify identity, size, hash and recoverable remote manifests before deleting local bytes. R2 remains canonical after local cleanup and HF Publication.
- Retain lightweight episode results; after selection download only the representative episode's chunks and stream video through bounded working space. Remove temporary video inputs/output after confirmed required delivery, retaining recovery journals until every acknowledgement exists.
- Maintain a durable lightweight queue of immutable Checkpoint references without dropping older work; monitoring failures do not stop training. Retry missing operational work once, reusing verified complete episodes and delivered artifacts with unchanged identities.
- Track episode execution, dataset verification, video generation, R2 media and W&B delivery separately so late failures do not restart scientific work unnecessarily.
- Successful terminal receipt must prove required Checkpoint/evaluation/dataset/video inventories, W&B acknowledgements and worker quiescence. Cancellation/deadline/budget exhaustion must preserve honest partial evidence and release resources; training and monitoring outcomes remain separate.

## Calibration and enablement

- Provide explicit reusable calibration bound to recipe/Policy architecture, provider/source/runtime, episode contract, schema/encoder, hardware allocation and worker settings.
- Measure representative early/intermediate/stronger compatible Checkpoints and long episodes, including inference, capture, encoding, R2, video/W&B delivery, memory/spool/retained bytes and concurrent learner throughput; unavailable representative inputs mean incomplete calibration.
- Keep episode count and scientific conditions fixed while choosing feasible worker limits and conservative checkpoint spacing with headroom, total checkpoint count and finite final drain. Reject infeasibility without lowering counts, weakening conditions, raising budgets or using implicit remote compute.
- Invalidate calibration when bound inputs change. Repeated matched off/on training seeds at identical resolved cadence, comparable host load and warm-up exclusion must support at most 2% throughput loss; report uncertainty and total completion/GPU idle tail separately. Otherwise support remains unproven.
- Enable FirstWall PPO only after its resolved configuration is supported by calibration. Tests may explicitly disable monitoring; this specification does not authorize live training or compute purchases.

## Explicit Publication and verification

- Use finalized verified Run inventories by default; explicit completed-snapshot publication may freeze only completed supervisor-verified evaluations from an active Run, excluding unfinished evaluations and without changing training. Use optional stage/performance predicates, filtering lightweight indexes before images and preserving complete multi-chunk episodes and grouping identities. Leave split assignment to consumers.
- Publish snapshots in a few bounded commits using compact immutable shards, keeping complete-episode visibility atomic; coordinate publication budgets and cooldowns across jobs to avoid HF throttling, and preserve already published contributions.
- Use the existing durable local Publication queue and Hub adapter with bounded temporary space, immutable additive contributions, expected-parent protection, idempotent retries/overlaps/lost acknowledgements and conflict/schema rejection. Keep R2 sources; training requires no HF target/credentials and never publishes automatically.
- Test through the real supervisor certification harness, real Policy/episode executor and durable Publication entry point with Hub adapter, using real temporary persistence and observable artifacts/metrics/receipts.
- Cover schedule-independent randomness, all supported trainers, action/RGB fidelity, episodes longer than 8,192 transitions, missing evidence, resource pressure, one retry, acknowledgement loss, median ties/even counts, reconstructed video after local reclamation, late checkpoint-step delivery, cancellation and finite drain.
- Run the complete deterministic lifecycle gate twice unchanged and preserve/replay failures; passing simulation does not establish live visibility or hardware overhead.
- A separately authorized bounded live campaign must cover at least three unique Checkpoints including final, production 400-episode sets, active contention and parallel final drain; inspect actual W&B video/charts in the in-app browser, remote integrity, cleanup and resource release. Live HF Publication requires an explicitly authorized target.
