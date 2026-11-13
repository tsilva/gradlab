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

The other 30 included levels remain unqualified by this campaign.

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
- Size: 20,445,801 bytes. The local model hash and size match its R2 publication
  receipt; an independent remote byte download was not repeated for this Run.
- Source: `7e643f4f820ecc0a82b6c7f90a787810c64c5ee8`.
- Runtime digest: `sha256:5d316181cc88dce83c9613543e2439214544379adf5682459f1120590b322991`.
- Provider: `env-supermariobrosnes-turbo-emu` 0.7.4.
- Executed recipe document SHA-256:
  `bb813babcf7c5103460a5e7e7985e18ed4af0cb5e94acc770d155a136ea01ef5`.
- Canonical JSON SHA-256 of the resolved training configuration:
  `914da334e124bddae5b9a5b8875b28ac03d50eb38f2efa71b8de5f1433ecc9a0`.

The finite bound was 16,777,216 transitions and four hours. Online MLflow
acknowledged all 711 emitted events; the durable R2 journal reached the same
high-water mark. Delivery completed and the task terminated normally after the
neutral plateau stop. Automatic/post-training evaluation and Monitoring were
disabled; no Acceptance result is claimed.

The [executed recipe](https://github.com/tsilva/gradlab/blob/7e643f4f820ecc0a82b6c7f90a787810c64c5ee8/experiments/goals/SuperMarioBros-Nes-v0/Level1-1/recipes/ppo-train-clear-80.yaml)
used `train.timesteps=16777216`. Consolidation preserves its resolved training
configuration and environment in `ppo.yaml`; the recipe identifier and path
change. Earlier recipes and resolved historical configurations remain available
at their immutable source revisions. Shared presets, assets and checkpoints
are preserved.
