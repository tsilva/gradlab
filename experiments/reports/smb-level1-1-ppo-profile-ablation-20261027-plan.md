# SMB Level1-1 PPO execution-profile ablation — preregistered 2026-10-27

Preregistered before admission. The user stopped execution on 2026-10-28 after thirteen learners completed, with twelve fully assessed and one partially assessed. See the [retained partial results](smb-level1-1-ppo-profile-ablation-20261027-stopped.md). The protocol below records the original plan; the five-seed study was not completed.

The study tests whether the earlier transition-to-training-stop advantage persists when measuring a common frozen-policy quality threshold, and which execution-profile change is associated with it. The earlier six Runs remain historical context and are excluded from the new cohort.

## Frozen protocol

- Five new training seeds: 100, 200, 300, 400 and 500, distinct from the previous 12–14. Five arms per seed: `sb3.ppo`, then `gradlab.ppo` profiles `sb3-parity`, `compiled-parity`, `compiled-fused-parity`, and `max-throughput`. Twenty-five Runs total.
- Sequential balanced cyclic ordering: each arm occupies every ordinal position once across the five seed blocks. One training/evaluation container at a time, preserving unrelated services and the same RTX 4090 host.
- Same PPO mathematics and hyperparameters as the previous comparison, FP32 throughout. Absolute learning-rate and entropy schedules still end their decay at 4M transitions. No hyperparameter search or tuning.
- Every Run requests 8,000,000 transitions, rounded to 8,003,584 for complete 8192-transition rollouts, with a 45-minute whole-training-task limit. The existing training streak condition is retained as observe-only and cannot stop this study's learners.
- Scheduled snapshots after completed updates at requested 1M increments. Actual transition counts: 1,007,616; 2,007,040; 3,006,464; 4,005,888; 5,005,312; 6,004,736; 7,004,160; 8,003,584 (final). Candidate-based checkpoint selection is disabled.
- Each of the eight frozen policies is assessed after its Run releases training resources. All policies use the same new 400-episode manifest, base seed 20000, `vector-lane-v1`, 16 lanes with 25 episodes each, stochastic actions, the saved environment/ROM/reward/action/observation contracts, and the same 4500-transition bounds. Fail-fast is disabled.
- Primary empirical quality threshold: at least 380/400 clears (95%). Every episode contributes; failures remain in full aggregates. The immutable Goal's Acceptance protocol remains unchanged and these assessments confer no Acceptance, Promotion or Release.
- CPU assessments use 8 CPU, 40 GiB RAM, one Torch intra-op/inter-op thread, and a five-minute container deadline per snapshot. Each records the first manifest episode as original RGB video. No cloud compute is admitted. The study has a 36-hour maximum and at most 200 unique assessments / 80,000 assessment episodes.

## Analysis fixed before outcomes

The primary sample-efficiency result is the earliest scheduled frozen checkpoint meeting the empirical threshold. A crossing lies between scheduled samples; no finer transition estimate is invented. Runs without a passing checkpoint remain right-censored at the shared cap. Cap wall time is never substituted for time to success. Plot the full common-grid success and return curves, with per-policy binomial uncertainty, and compare quality-curve area as a secondary summary.

Report learner wall time from actual learner-process creation, including initialization and compilation, to saved checkpoint readiness. This excludes provisioning and post-training assessment latency; separately report that latency and full task/campaign cost. Weighted throughput uses complete matched transition windows, aligned according to `METRICS.md`.

Primary adjacent contrasts: SB3 → custom parity (implementation path); parity → compiled (compilation bundle); compiled → fused (fused Adam); fused → max (Torch/GPU minibatch permutations, associated flattening order and RNG consumption). This is a conditional sequence, not a full factorial interaction experiment, and the final contrast does not separate its bundled mechanisms.

Five trained policies per arm are the independent experiment units. Assessment episodes do not substitute for training-seed replication. Reusing one assessment manifest over the checkpoint grid makes it a comparison/selection panel, not a future untouched test. Empirical 95% attainment does not prove a true success probability of at least 95%. More precise mechanism claims require appropriate follow-up evidence.

Operational failures pause further admissions. Preserve Run/Attempt identities and evidence; no automatic training retries, replacement policies, extra seeds, threshold changes or tuning are authorized by this plan.

## Provenance

- Source: `2069964e7ce38de8f87392de084145e6af8f87d3`.
- Immutable runtime: `docker:ghcr.io/tsilva/gradlab/gradlab-train@sha256:e5b0b2978a608984c6d499d272de11081c6b8b07a7c9354a9bdcc865071cc1a3`.
- Private frozen-plan SHA-256: `dd0d9f84f3466544312e693793b394bbe49c1006af5a34e6512cdbc43e65c74c`.

## Execution tracking

First admitted Run: [SB3 seed 100 — gradlab-2dee34efd065108ed76d446ba5ada337](https://pub-fc35c0b186ce4aad8eea5e93d38c99db.r2.dev/runs/gradlab-2dee34efd065108ed76d446ba5ada337/index.json). Thirteen Runs were admitted; the remaining twelve were canceled before admission. No automatic continuation remains.

## Retained quality view

The separate MLflow experiment **SMB PPO profile ablation — frozen-policy quality** retains companion views with arm and training-seed identifiers. Plot `assessment/success/mean` against **Step** for sample efficiency, with `assessment/threshold` at 0.95; `assessment/return/mean` shows the corresponding shaped-return curve. Each point comes from a complete hash-verified 400-episode result and video. The x-axis is the actual saved checkpoint transition count; timestamps retain assessment completion time. The projector was stopped at cancellation; the final post-stop collected point is included in the portable data linked above. The companion view is rebuildable and grants no Acceptance, Promotion or Release. Conditional Wilson bounds describe episode uncertainty for one frozen policy; training seeds remain the independent units, and the planned five-seed replication was not completed.
