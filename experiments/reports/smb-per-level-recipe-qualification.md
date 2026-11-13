# Mario per-level recipe qualification

This training-only campaign covers ordinary Levels 1-1 through 8-3. Level8-4
is deferred. Qualification requires at least 80 genuine canonical-start clears
in the latest full window of 100 completed episodes, preferring 100/100 within
the declared bounds. These observations do not establish frozen-checkpoint
performance, across-seed reliability, Acceptance or Promotion.

## Qualified recipes

| Level | Canonical recipe | Retained checkpoint window | Metric/checkpoint step | Seed |
| --- | --- | --- | --- | --- |
| Level1-1 | [PPO](../goals/SuperMarioBros-Nes-v0/Level1-1/recipes/ppo.yaml) | 95/100 | 2,629,632 / 2,629,632 | 12 |
| Level1-2 | [PPO](../goals/SuperMarioBros-Nes-v0/Level1-2/recipes/ppo.yaml) | 84/100 | 3,301,376 / 3,301,376 | 12 |

The other 29 included levels remain unqualified by this campaign.

## Level1-1 — 2026-11-13

Fresh plain PPO Run `gradlab-34847b00808fecbe3f765c80d381a31a`, Attempt
`attempt-a5683cbba2b1ae90`, first reached 80/100 at 1,589,248 transitions.
Its best observed window was 98/100 at 3,481,600 transitions. The declared
1,048,576-transition no-improvement condition stopped training at 4,530,176,
with a latest window of 90/100. The Run did not reach 100/100.

The selected retained checkpoint corresponds to a 95/100 window. At that exact
metric step, 7,695 genuine target-origin episodes had completed and 2,608 had
cleared cumulatively. The learner used one canonical Level1-1 start, fresh
weights, no checkpoint initialization, no archive curriculum and no SIL.
The checkpoint is
[`checkpoint-2629632-4ff49f51e5bbd21e`](https://pub-fc35c0b186ce4aad8eea5e93d38c99db.r2.dev/runs/gradlab-34847b00808fecbe3f765c80d381a31a/checkpoints/2629632-4ff49f51e5bbd21e6f0ceab38df60722a6466868e068db32b50b6744142a7c80/model.zip):

- SHA-256: `4ff49f51e5bbd21e6f0ceab38df60722a6466868e068db32b50b6744142a7c80`
- Size: 20,445,801 bytes. An independent download through the configured models
  store verified the complete remote model SHA-256 and size against its receipt.
- Source: `7e643f4f820ecc0a82b6c7f90a787810c64c5ee8`.
- Runtime digest: `sha256:5d316181cc88dce83c9613543e2439214544379adf5682459f1120590b322991`.
- Provider: `env-supermariobrosnes-turbo-emu` 0.7.4.
- Executed recipe document SHA-256:
  `bb813babcf7c5103460a5e7e7985e18ed4af0cb5e94acc770d155a136ea01ef5`.
- Canonical JSON SHA-256 of the resolved training configuration:
  `914da334e124bddae5b9a5b8875b28ac03d50eb38f2efa71b8de5f1433ecc9a0`.

The finite bound was 16,777,216 transitions and four hours. Online MLflow
acknowledged all 711 emitted events; the durable R2 journal reached the same
high-water mark. A subsequent read-only audit verified all 544 journal segments
and matched the selected window and cumulative counts to actual MLflow history.
Delivery completed and the task terminated normally after the
neutral plateau stop. Automatic/post-training evaluation and Monitoring were
disabled; no Acceptance result is claimed.

The [executed recipe](https://github.com/tsilva/gradlab/blob/7e643f4f820ecc0a82b6c7f90a787810c64c5ee8/experiments/goals/SuperMarioBros-Nes-v0/Level1-1/recipes/ppo-train-clear-80.yaml)
used `train.timesteps=16777216`. Consolidation preserves its resolved training
configuration and environment in `ppo.yaml`; the recipe identifier and path
change. Earlier recipes and resolved historical configurations remain available
at their immutable source revisions. Shared presets, assets and checkpoints
are preserved.


## Level1-2 — 2026-11-13

Fresh plain PPO Run `gradlab-1c675c917e2cc2eca7b2a58cc71c9081`, Attempt
`attempt-a09b5e15a675fe68`, reached a best observed mature window of 89/100.
The retained qualifying checkpoint has 84 clears in its latest 100 genuine
canonical-start episodes at 3,301,376 transitions. At that exact step, 8,990
target/all episodes had completed and 3,752 had cleared cumulatively. The
unretained 89/100 history peak is not assigned to this model.

The selected checkpoint is
[`checkpoint-3301376-4e71fe59fa4827de`](https://pub-fc35c0b186ce4aad8eea5e93d38c99db.r2.dev/runs/gradlab-1c675c917e2cc2eca7b2a58cc71c9081/checkpoints/3301376-4e71fe59fa4827def356d50731b2979ffe920edb5362aeb88a3baa53992c6620/model.zip):

- SHA-256: `4e71fe59fa4827def356d50731b2979ffe920edb5362aeb88a3baa53992c6620`.
- Size: 20,445,798 bytes, independently downloaded through the configured models
  store and verified against the immutable publication receipt.
- Source: `7e643f4f820ecc0a82b6c7f90a787810c64c5ee8`.
- Runtime digest: `sha256:5d316181cc88dce83c9613543e2439214544379adf5682459f1120590b322991`.
- Provider: `env-supermariobrosnes-turbo-emu` 0.7.4.
- Executed recipe document SHA-256:
  `0ff7126a4548c414789be5766d26492809a28c4703a96ce4fa0821000aea45ac`.
- Canonical JSON SHA-256 of the normalized portable training configuration:
  `c1b32cd8e9dce40df56fdfb07c3aa695c48ef0b12d496b056983e279288c47a7`.

The learner used seed 12, fresh weights, one canonical Level1-2 start, no archive
curriculum and no SIL. Its finite bound was nominally 50,000,000 transitions
(50,003,968 at a safe rollout boundary) and six hours. The declared neutral
plateau condition stopped the Run at 3,301,376 transitions. All 520 journal
events were hash- and sequence-verified across 409 retained segments; actual
MLflow history independently matched the selected window and cumulative counts.
Delivery/drain completed and the task released its resources. Automatic and
post-training evaluation and Monitoring were disabled. No Acceptance or
across-seed reliability claim is made.

The [executed recipe](https://github.com/tsilva/gradlab/blob/7e643f4f820ecc0a82b6c7f90a787810c64c5ee8/experiments/goals/SuperMarioBros-Nes-v0/Level1-2/recipes/ppo-train-clear-80.yaml)
is consolidated into `ppo.yaml`. Its normalized training configuration and full
resolved environment exactly match the immutable executed recipe. The competing
launchable YAML is removed; historical source, checkpoints and shared presets
remain available.
