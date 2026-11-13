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
| Level1-3 | [PPO + SIL](../goals/SuperMarioBros-Nes-v0/Level1-3/recipes/ppo.yaml) | 95/100 | 4,210,688 / 4,210,688 | 1801 |
| Level1-4 | [PPO](../goals/SuperMarioBros-Nes-v0/Level1-4/recipes/ppo.yaml) | 80/100 | 2,121,728 / 2,121,728 | 12 |
| Level2-1 | [PPO](../goals/SuperMarioBros-Nes-v0/Level2-1/recipes/ppo.yaml) | 90/100 | 7,012,352 / 7,012,352 | 12 |
| Level2-2 | [PPO + SIL](../goals/SuperMarioBros-Nes-v0/Level2-2/recipes/ppo.yaml) | 84/100 | 7,155,712 / 7,155,712 | 1801 |
| Level2-3 | [PPO](../goals/SuperMarioBros-Nes-v0/Level2-3/recipes/ppo.yaml) | 90/100 | 3,424,256 / 3,424,256 | 12 |
| Level3-1 | [PPO](../goals/SuperMarioBros-Nes-v0/Level3-1/recipes/ppo.yaml) | 96/100 | 5,505,024 / 5,505,024 | 12 |

The other 23 included levels remain unqualified by this campaign.

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
- Canonical JSON SHA-256 of the normalized portable training configuration:
  `52f394e169de24c506adb92d2583e34bb49a14c42d07384a96d05e2e0f4b227f`.

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

<!-- additional-qualifications:start -->

## Level1-3 — 2026-11-13

Fresh PPO with SIL Run `gradlab-a2683a03dd3e0495f486730dbaeb54d0`, Attempt
`attempt-32b1819dffd484e3`, used seed 1801 and one canonical Level1-3 start.
The retained checkpoint has **95/100** genuine target-origin clears
at 4,210,688 transitions. At that exact step, 14,333
episodes had completed and 4,684 had cleared cumulatively.
Its separate history peak was 99/100; the final window was
93/100 at 5,472,256 transitions. History peaks are not assigned to
another saved model.

The selected model is [`checkpoint-4210688-5a4006950b60c529`](https://pub-fc35c0b186ce4aad8eea5e93d38c99db.r2.dev/runs/gradlab-a2683a03dd3e0495f486730dbaeb54d0/checkpoints/4210688-5a4006950b60c52967481e42ed51c9a41703f7777013e65042542eede0aa2541/model.zip):

- SHA-256: `5a4006950b60c52967481e42ed51c9a41703f7777013e65042542eede0aa2541`; 40,663,534 bytes, independently downloaded and verified.
- Source: `7e643f4f820ecc0a82b6c7f90a787810c64c5ee8`.
- Runtime digest: `sha256:5d316181cc88dce83c9613543e2439214544379adf5682459f1120590b322991`.
- Executed recipe document SHA-256: `8d076d7fa0ef1f8bdc1357806ca3b9e9d2d855019c236fb3ce7c0b633b99c412`.
- Normalized portable training configuration SHA-256: `98a4c0095155460eb0cc661a47efbac77da3ba8bc043a5e0abb1626ac54e5d0b`.

The finite bound was 16,777,216 transitions and 4 hours. The declared neutral plateau stopped learning; complete delivery and task release were verified.
The read-only audit checked all 1,305 retained journal segments
and delivery through 1,642 events, with the selected window and
cumulative counts independently matched to actual MLflow history. No imported
initialization, archive curriculum, automatic evaluation or Monitoring was used.
This is training qualification, not Acceptance or across-seed reliability.

The [executed recipe](https://github.com/tsilva/gradlab/blob/7e643f4f820ecc0a82b6c7f90a787810c64c5ee8/experiments/goals/SuperMarioBros-Nes-v0/Level1-3/recipes/ppo-sil-train-clear-80.yaml)
is consolidated into `ppo.yaml`. Its normalized training configuration and full
resolved environment are unchanged. Competing launchable YAMLs were removed only
after qualification; historical source, shared presets and checkpoints are preserved.


## Level1-4 — 2026-11-13

Fresh plain PPO Run `gradlab-85c9b1922b2545d8631b6d4415ef21dd`, Attempt
`attempt-287e38645f328f8e`, used seed 12 and one canonical Level1-4 start.
The retained checkpoint has **80/100** genuine target-origin clears
at 2,121,728 transitions. At that exact step, 10,021
episodes had completed and 1,317 had cleared cumulatively.
Its separate history peak was 82/100; the final window was
59/100 at 3,252,224 transitions. History peaks are not assigned to
another saved model.

The selected model is [`checkpoint-2121728-cb8578c931c69b1d`](https://pub-fc35c0b186ce4aad8eea5e93d38c99db.r2.dev/runs/gradlab-85c9b1922b2545d8631b6d4415ef21dd/checkpoints/2121728-cb8578c931c69b1dcb53bc13d061b990b969bb7ae565653d0e1b43a7612126e6/model.zip):

- SHA-256: `cb8578c931c69b1dcb53bc13d061b990b969bb7ae565653d0e1b43a7612126e6`; 20,445,797 bytes, independently downloaded and verified.
- Source: `7e643f4f820ecc0a82b6c7f90a787810c64c5ee8`.
- Runtime digest: `sha256:5d316181cc88dce83c9613543e2439214544379adf5682459f1120590b322991`.
- Executed recipe document SHA-256: `97e0f4f802d8ee1e13f470706f740125090316cbe241603514afbc0ae7677a2b`.
- Normalized portable training configuration SHA-256: `229ba121652cd7ef76747a1a37e62778e1513a59e4593727b5062faf0e20840e`.

The finite bound was nominally 50,000,000 transitions (50,003,968 at a safe rollout boundary) and 6 hours. The declared neutral plateau stopped learning; complete delivery and task release were verified.
The read-only audit checked all 395 retained journal segments
and delivery through 512 events, with the selected window and
cumulative counts independently matched to actual MLflow history. No imported
initialization, archive curriculum, automatic evaluation or Monitoring was used.
This is training qualification, not Acceptance or across-seed reliability.

The [executed recipe](https://github.com/tsilva/gradlab/blob/7e643f4f820ecc0a82b6c7f90a787810c64c5ee8/experiments/goals/SuperMarioBros-Nes-v0/Level1-4/recipes/ppo-train-clear-80.yaml)
is consolidated into `ppo.yaml`. Its normalized training configuration and full
resolved environment are unchanged. Competing launchable YAMLs were removed only
after qualification; historical source, shared presets and checkpoints are preserved.


## Level2-1 — 2026-11-13

Fresh plain PPO Run `gradlab-f4352ce3ca1bc537393d49074020b040`, Attempt
`attempt-5e5d357e088e3a65`, used seed 12 and one canonical Level2-1 start.
The retained checkpoint has **90/100** genuine target-origin clears
at 7,012,352 transitions. At that exact step, 21,201
episodes had completed and 4,191 had cleared cumulatively.
Its separate history peak was 90/100; the final window was
67/100 at 8,060,928 transitions. History peaks are not assigned to
another saved model.

The selected model is [`checkpoint-7012352-fa509e1b0a933c83`](https://pub-fc35c0b186ce4aad8eea5e93d38c99db.r2.dev/runs/gradlab-f4352ce3ca1bc537393d49074020b040/checkpoints/7012352-fa509e1b0a933c8377861e373aeaf3e31a77195f79009108297a52e9bf6f072e/model.zip):

- SHA-256: `fa509e1b0a933c8377861e373aeaf3e31a77195f79009108297a52e9bf6f072e`; 20,445,806 bytes, independently downloaded and verified.
- Source: `7e643f4f820ecc0a82b6c7f90a787810c64c5ee8`.
- Runtime digest: `sha256:5d316181cc88dce83c9613543e2439214544379adf5682459f1120590b322991`.
- Executed recipe document SHA-256: `cb096d7aa2e7f89fdc84030213989061e6b659e712c4f110227462d0388ca283`.
- Normalized portable training configuration SHA-256: `3d39ca13238d3fc8ffe1b4ab7482e4fe151c8672e841b0c1f5bb8c0302d80fce`.

The finite bound was nominally 50,000,000 transitions (50,003,968 at a safe rollout boundary) and 6 hours. The declared neutral plateau stopped learning; complete delivery and task release were verified.
The read-only audit checked all 981 retained journal segments
and delivery through 1,262 events, with the selected window and
cumulative counts independently matched to actual MLflow history. No imported
initialization, archive curriculum, automatic evaluation or Monitoring was used.
This is training qualification, not Acceptance or across-seed reliability.

The [executed recipe](https://github.com/tsilva/gradlab/blob/7e643f4f820ecc0a82b6c7f90a787810c64c5ee8/experiments/goals/SuperMarioBros-Nes-v0/Level2-1/recipes/ppo-train-clear-80.yaml)
is consolidated into `ppo.yaml`. Its normalized training configuration and full
resolved environment are unchanged. Competing launchable YAMLs were removed only
after qualification; historical source, shared presets and checkpoints are preserved.


## Level2-2 — 2026-11-13

Fresh PPO with SIL Run `gradlab-13618bff8573053e27fc095d41c4ce13`, Attempt
`attempt-f2a7acaa16ba008c`, used seed 1801 and one canonical Level2-2 start.
The retained checkpoint has **84/100** genuine target-origin clears
at 7,155,712 transitions. At that exact step, 11,853
episodes had completed and 4,275 had cleared cumulatively.
Its separate history peak was 87/100; the final window was
84/100 at 7,155,712 transitions. History peaks are not assigned to
another saved model.

The selected model is [`checkpoint-7155712-aba3cc4688a4fd9a`](https://pub-fc35c0b186ce4aad8eea5e93d38c99db.r2.dev/runs/gradlab-13618bff8573053e27fc095d41c4ce13/checkpoints/7155712-aba3cc4688a4fd9a426bfa7238b384a1fc6273076f0ac9c632edf23cebaeebe7/model.zip):

- SHA-256: `aba3cc4688a4fd9a426bfa7238b384a1fc6273076f0ac9c632edf23cebaeebe7`; 40,663,536 bytes, independently downloaded and verified.
- Source: `7e643f4f820ecc0a82b6c7f90a787810c64c5ee8`.
- Runtime digest: `sha256:5d316181cc88dce83c9613543e2439214544379adf5682459f1120590b322991`.
- Executed recipe document SHA-256: `78fdf86fa49eec1cd9cd4f9e59b71c89eb797ae7d48a7a6b4a18593eff9f06a5`.
- Normalized portable training configuration SHA-256: `6f9f4b222e7dd43ba7458f1556d2eda67dbcb91eb9b602952bc29b1908f10a21`.

The finite bound was 16,777,216 transitions and 4 hours. The declared neutral plateau stopped learning; complete delivery and task release were verified.
The read-only audit checked all 1,670 retained journal segments
and delivery through 2,096 events, with the selected window and
cumulative counts independently matched to actual MLflow history. No imported
initialization, archive curriculum, automatic evaluation or Monitoring was used.
This is training qualification, not Acceptance or across-seed reliability.

The [executed recipe](https://github.com/tsilva/gradlab/blob/7e643f4f820ecc0a82b6c7f90a787810c64c5ee8/experiments/goals/SuperMarioBros-Nes-v0/Level2-2/recipes/ppo-sil-train-clear-80.yaml)
is consolidated into `ppo.yaml`. Its normalized training configuration and full
resolved environment are unchanged. Competing launchable YAMLs were removed only
after qualification; historical source, shared presets and checkpoints are preserved.


## Level2-3 — 2026-11-13

Fresh plain PPO Run `gradlab-9a24c8fd278534de9d72ebe8cb563601`, Attempt
`attempt-e2348f7b7b33096c`, used seed 12 and one canonical Level2-3 start.
The retained checkpoint has **90/100** genuine target-origin clears
at 3,424,256 transitions. At that exact step, 10,074
episodes had completed and 3,011 had cleared cumulatively.
Its separate history peak was 91/100; the final window was
84/100 at 5,177,344 transitions. History peaks are not assigned to
another saved model.

The selected model is [`checkpoint-3424256-4269b1261b7ac9e8`](https://pub-fc35c0b186ce4aad8eea5e93d38c99db.r2.dev/runs/gradlab-9a24c8fd278534de9d72ebe8cb563601/checkpoints/3424256-4269b1261b7ac9e85b7878581b911fb78101757019943667c34b4eea5e71e58c/model.zip):

- SHA-256: `4269b1261b7ac9e85b7878581b911fb78101757019943667c34b4eea5e71e58c`; 20,445,811 bytes, independently downloaded and verified.
- Source: `7e643f4f820ecc0a82b6c7f90a787810c64c5ee8`.
- Runtime digest: `sha256:5d316181cc88dce83c9613543e2439214544379adf5682459f1120590b322991`.
- Executed recipe document SHA-256: `8e5142d5fba999368f8ffc4bc40019a9400889404df1de8381bc4b10a92d3d8c`.
- Normalized portable training configuration SHA-256: `5dfb0eada53b823eadf5321c638c538fe608660fb5e8d1b235c612158ec2107e`.

The finite bound was nominally 50,000,000 transitions (50,003,968 at a safe rollout boundary) and 6 hours. The declared neutral plateau stopped learning; complete delivery and task release were verified.
The read-only audit checked all 634 retained journal segments
and delivery through 813 events, with the selected window and
cumulative counts independently matched to actual MLflow history. No imported
initialization, archive curriculum, automatic evaluation or Monitoring was used.
This is training qualification, not Acceptance or across-seed reliability.

The [executed recipe](https://github.com/tsilva/gradlab/blob/7e643f4f820ecc0a82b6c7f90a787810c64c5ee8/experiments/goals/SuperMarioBros-Nes-v0/Level2-3/recipes/ppo-train-clear-80.yaml)
is consolidated into `ppo.yaml`. Its normalized training configuration and full
resolved environment are unchanged. Competing launchable YAMLs were removed only
after qualification; historical source, shared presets and checkpoints are preserved.


## Level3-1 — 2026-11-13

Fresh plain PPO Run `gradlab-1314875913c13bf83c8bebf78d4ee17d`, Attempt
`attempt-d99d8f6950eef3e2`, used seed 12 and one canonical Level3-1 start.
The retained checkpoint has **96/100** genuine target-origin clears
at 5,505,024 transitions. At that exact step, 14,619
episodes had completed and 5,769 had cleared cumulatively.
Its separate history peak was 96/100; the final window was
67/100 at 6,553,600 transitions. History peaks are not assigned to
another saved model.

The selected model is [`checkpoint-5505024-ed89c28b554142b2`](https://pub-fc35c0b186ce4aad8eea5e93d38c99db.r2.dev/runs/gradlab-1314875913c13bf83c8bebf78d4ee17d/checkpoints/5505024-ed89c28b554142b24d6310da3d1e4d8570ba34a026ebfdef1f28c13cdb5c80a3/model.zip):

- SHA-256: `ed89c28b554142b24d6310da3d1e4d8570ba34a026ebfdef1f28c13cdb5c80a3`; 20,445,814 bytes, independently downloaded and verified.
- Source: `7e643f4f820ecc0a82b6c7f90a787810c64c5ee8`.
- Runtime digest: `sha256:5d316181cc88dce83c9613543e2439214544379adf5682459f1120590b322991`.
- Executed recipe document SHA-256: `7a21514200405bebcbece64c27fe2b4f8400f917a7a85eb6dec397e3b28477ea`.
- Normalized portable training configuration SHA-256: `3f1c466be2804dfa26c15bbb4761816b7ceb2c24a75586877f88ff089029edad`.

The finite bound was nominally 50,000,000 transitions (50,003,968 at a safe rollout boundary) and 6 hours. The declared neutral plateau stopped learning; complete delivery and task release were verified.
The read-only audit checked all 765 retained journal segments
and delivery through 1,026 events, with the selected window and
cumulative counts independently matched to actual MLflow history. No imported
initialization, archive curriculum, automatic evaluation or Monitoring was used.
This is training qualification, not Acceptance or across-seed reliability.

The [executed recipe](https://github.com/tsilva/gradlab/blob/7e643f4f820ecc0a82b6c7f90a787810c64c5ee8/experiments/goals/SuperMarioBros-Nes-v0/Level3-1/recipes/ppo-train-clear-80.yaml)
is consolidated into `ppo.yaml`. Its normalized training configuration and full
resolved environment are unchanged. Competing launchable YAMLs were removed only
after qualification; historical source, shared presets and checkpoints are preserved.
