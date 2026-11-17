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
| Level2-4 | [PPO + SIL](../goals/SuperMarioBros-Nes-v0/Level2-4/recipes/ppo.yaml) | 95/100 | 15,142,912 / 15,142,912 | 2801 |
| Level3-1 | [PPO](../goals/SuperMarioBros-Nes-v0/Level3-1/recipes/ppo.yaml) | 96/100 | 5,505,024 / 5,505,024 | 12 |
| Level3-2 | [PPO](../goals/SuperMarioBros-Nes-v0/Level3-2/recipes/ppo.yaml) | 96/100 | 2,007,040 / 2,007,040 | 12 |
| Level3-4 | [PPO](../goals/SuperMarioBros-Nes-v0/Level3-4/recipes/ppo.yaml) | 95/100 | 2,580,480 / 2,580,480 | 12 |
| Level4-1 | [PPO](../goals/SuperMarioBros-Nes-v0/Level4-1/recipes/ppo.yaml) | 98/100 | 5,496,832 / 5,496,832 | 12 |
| Level5-1 | [PPO](../goals/SuperMarioBros-Nes-v0/Level5-1/recipes/ppo.yaml) | 80/100 | 6,758,400 / 6,758,400 | 12 |
| Level5-2 | [PPO](../goals/SuperMarioBros-Nes-v0/Level5-2/recipes/ppo.yaml) | 95/100 | 5,005,312 / 5,005,312 | 12 |
| Level5-3 | [PPO](../goals/SuperMarioBros-Nes-v0/Level5-3/recipes/ppo.yaml) | 95/100 | 28,688,384 / 28,688,384 | 12 |
| Level5-4 | [PPO](../goals/SuperMarioBros-Nes-v0/Level5-4/recipes/ppo.yaml) | 95/100 | 4,120,576 / 4,120,576 | 12 |
| Level6-2 | [PPO](../goals/SuperMarioBros-Nes-v0/Level6-2/recipes/ppo.yaml) | 80/100 | 7,430,144 / 7,430,144 | 12 |
| Level6-4 | [PPO](../goals/SuperMarioBros-Nes-v0/Level6-4/recipes/ppo.yaml) | 100/100 | 4,562,944 / 4,562,944 | 12 |
| Level7-1 | [PPO](../goals/SuperMarioBros-Nes-v0/Level7-1/recipes/ppo.yaml) | 80/100 | 3,252,224 / 3,252,224 | 12 |
| Level8-2 | [PPO](../goals/SuperMarioBros-Nes-v0/Level8-2/recipes/ppo.yaml) | 83/100 | 10,780,672 / 10,780,672 | 12 |
| Level8-3 | [PPO](../goals/SuperMarioBros-Nes-v0/Level8-3/recipes/ppo.yaml) | 81/100 | 4,612,096 / 4,612,096 | 12 |

The other 10 included levels remain unqualified by this campaign.

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


## Level2-4 — 2026-11-15

Fresh PPO with SIL Run `gradlab-2c62561b9b478b9bd3600b822a42675c`, Attempt
`attempt-58c08af6f1d99503`, used seed 2801 and one canonical Level2-4 start.
The retained checkpoint has **95/100** genuine target-origin clears
at 15,142,912 transitions. At that exact step, 36,353
episodes had completed and 5,181 had cleared cumulatively.
Its separate history peak was 97/100; the final window was
86/100 at 16,199,680 transitions. History peaks are not assigned to
another saved model.

The selected model is [`checkpoint-15142912-a095703629b52c63`](https://pub-fc35c0b186ce4aad8eea5e93d38c99db.r2.dev/runs/gradlab-2c62561b9b478b9bd3600b822a42675c/checkpoints/15142912-a095703629b52c634b3339a8df5f347965c5c0e91b9be4b2fe3cce4554cb8ba4/model.zip):

- SHA-256: `a095703629b52c634b3339a8df5f347965c5c0e91b9be4b2fe3cce4554cb8ba4`; 40,663,537 bytes, independently downloaded and verified.
- Source: `7e643f4f820ecc0a82b6c7f90a787810c64c5ee8`.
- Runtime digest: `sha256:5d316181cc88dce83c9613543e2439214544379adf5682459f1120590b322991`.
- Executed recipe document SHA-256: `b363b56b0f408567d6241c825f1dabd71a37712079b9f2a8ab15ff3349db6463`.
- Normalized portable training configuration SHA-256: `4687ab219c983e922aac537aa50c71d8c8d788a34da678819c2a890c9884e436`.

The finite bound was 16,777,216 transitions and 4 hours. The declared neutral plateau stopped learning; complete delivery and task release were verified.
The read-only audit checked all 3,673 retained journal segments
and delivery through 4,724 events, with the selected window and
cumulative counts independently matched to actual MLflow history. No imported
initialization, archive curriculum, automatic evaluation or Monitoring was used.
This is training qualification, not Acceptance or across-seed reliability.

The [executed recipe](https://github.com/tsilva/gradlab/blob/7e643f4f820ecc0a82b6c7f90a787810c64c5ee8/experiments/goals/SuperMarioBros-Nes-v0/Level2-4/recipes/ppo-train-clear-80.yaml)
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


## Level3-2 — 2026-11-15

Fresh plain PPO Run `gradlab-0946a7afd6d15e6613f57963360229f5`, Attempt
`attempt-f36aa5ece01a4ee8`, used seed 12 and one canonical Level3-2 start.
The retained checkpoint has **96/100** genuine target-origin clears
at 2,007,040 transitions. At that exact step, 6,694
episodes had completed and 2,003 had cleared cumulatively.
Its separate history peak was 99/100; the final window was
90/100 at 3,088,384 transitions. History peaks are not assigned to
another saved model.

The selected model is [`checkpoint-2007040-706aed0b953ab904`](https://pub-fc35c0b186ce4aad8eea5e93d38c99db.r2.dev/runs/gradlab-0946a7afd6d15e6613f57963360229f5/checkpoints/2007040-706aed0b953ab90416e7aedaa0bf38dda38dba5355e114f2f8290571eaa256b4/model.zip):

- SHA-256: `706aed0b953ab90416e7aedaa0bf38dda38dba5355e114f2f8290571eaa256b4`; 20,445,806 bytes, independently downloaded and verified.
- Source: `7e643f4f820ecc0a82b6c7f90a787810c64c5ee8`.
- Runtime digest: `sha256:5d316181cc88dce83c9613543e2439214544379adf5682459f1120590b322991`.
- Executed recipe document SHA-256: `a138cdc99773a4edeb2158c07774ad6ad170d278ea3fb72c9e6b95d377255da7`.
- Normalized portable training configuration SHA-256: `9b7bac348e5c1f8cf3afd03755758452d9304d8e1235d151e28d7ea2f50137a1`.

The finite bound was nominally 50,000,000 transitions (50,003,968 at a safe rollout boundary) and 6 hours. The declared neutral plateau stopped learning; complete delivery and task release were verified.
The read-only audit checked all 358 retained journal segments
and delivery through 489 events, with the selected window and
cumulative counts independently matched to actual MLflow history. No imported
initialization, archive curriculum, automatic evaluation or Monitoring was used.
This is training qualification, not Acceptance or across-seed reliability.

The [executed recipe](https://github.com/tsilva/gradlab/blob/7e643f4f820ecc0a82b6c7f90a787810c64c5ee8/experiments/goals/SuperMarioBros-Nes-v0/Level3-2/recipes/ppo-train-clear-80.yaml)
is consolidated into `ppo.yaml`. Its normalized training configuration and full
resolved environment are unchanged. Competing launchable YAMLs were removed only
after qualification; historical source, shared presets and checkpoints are preserved.


## Level3-4 — 2026-11-16

Fresh plain PPO Run `gradlab-0e4aa3be8c4ec268bf212c2b852d25be`, Attempt
`attempt-b03e1da4a8a6f7d8`, used seed 12 and one canonical Level3-4 start.
The retained checkpoint has **95/100** genuine target-origin clears
at 2,580,480 transitions. At that exact step, 10,729
episodes had completed and 2,503 had cleared cumulatively.
Its separate history peak was 98/100; the final window was
74/100 at 3,792,896 transitions. History peaks are not assigned to
another saved model.

The selected model is [`checkpoint-2580480-cec220f5264c8f17`](https://pub-fc35c0b186ce4aad8eea5e93d38c99db.r2.dev/runs/gradlab-0e4aa3be8c4ec268bf212c2b852d25be/checkpoints/2580480-cec220f5264c8f175febbb2dcfbc0cf1b6a221a05bb04e83d549e8f9caffaffc/model.zip):

- SHA-256: `cec220f5264c8f175febbb2dcfbc0cf1b6a221a05bb04e83d549e8f9caffaffc`; 20,445,806 bytes, independently downloaded and verified.
- Source: `7e643f4f820ecc0a82b6c7f90a787810c64c5ee8`.
- Runtime digest: `sha256:5d316181cc88dce83c9613543e2439214544379adf5682459f1120590b322991`.
- Executed recipe document SHA-256: `eef0ad49b9d1a35f25914d63f42e42edd620fe0ac5faecb011d18695e3e8b573`.
- Normalized portable training configuration SHA-256: `27cffbc59ba165aac5d1e2845829e520e719fc323cd5b4d9db9776adbcfb1ecf`.

The finite bound was nominally 50,000,000 transitions (50,003,968 at a safe rollout boundary) and 6 hours. The declared neutral plateau stopped learning; complete delivery and task release were verified.
The read-only audit checked all 454 retained journal segments
and delivery through 598 events, with the selected window and
cumulative counts independently matched to actual MLflow history. No imported
initialization, archive curriculum, automatic evaluation or Monitoring was used.
This is training qualification, not Acceptance or across-seed reliability.

The [executed recipe](https://github.com/tsilva/gradlab/blob/7e643f4f820ecc0a82b6c7f90a787810c64c5ee8/experiments/goals/SuperMarioBros-Nes-v0/Level3-4/recipes/ppo-train-clear-80.yaml)
is consolidated into `ppo.yaml`. Its normalized training configuration and full
resolved environment are unchanged. Competing launchable YAMLs were removed only
after qualification; historical source, shared presets and checkpoints are preserved.


## Level4-1 — 2026-11-16

Fresh plain PPO Run `gradlab-cd5a1316a963f0e784e9cc5fc97b9799`, Attempt
`attempt-eb60ffa0f2ad7be1`, used seed 12 and one canonical Level4-1 start.
The retained checkpoint has **98/100** genuine target-origin clears
at 5,496,832 transitions. At that exact step, 14,382
episodes had completed and 6,984 had cleared cumulatively.
Its separate history peak was 99/100; the final window was
98/100 at 5,496,832 transitions. History peaks are not assigned to
another saved model.

The selected model is [`checkpoint-5496832-86597f65f2f145fe`](https://pub-fc35c0b186ce4aad8eea5e93d38c99db.r2.dev/runs/gradlab-cd5a1316a963f0e784e9cc5fc97b9799/checkpoints/5496832-86597f65f2f145fe05336d18aa30784becc4a4c4a65f14951210c4f7d901f082/model.zip):

- SHA-256: `86597f65f2f145fe05336d18aa30784becc4a4c4a65f14951210c4f7d901f082`; 20,445,806 bytes, independently downloaded and verified.
- Source: `7e643f4f820ecc0a82b6c7f90a787810c64c5ee8`.
- Runtime digest: `sha256:5d316181cc88dce83c9613543e2439214544379adf5682459f1120590b322991`.
- Executed recipe document SHA-256: `4eab63a8e9c3c441926f322b83907048088afef8bca22ee57a3100ba761fd130`.
- Normalized portable training configuration SHA-256: `beeb7a1c85768fd7a67d0e8cafa65011c04e115f5cc8491e21e2878ae096bc68`.

The finite bound was nominally 50,000,000 transitions (50,003,968 at a safe rollout boundary) and 6 hours. The declared neutral plateau stopped learning; complete delivery and task release were verified.
The read-only audit checked all 660 retained journal segments
and delivery through 864 events, with the selected window and
cumulative counts independently matched to actual MLflow history. No imported
initialization, archive curriculum, automatic evaluation or Monitoring was used.
This is training qualification, not Acceptance or across-seed reliability.

The [executed recipe](https://github.com/tsilva/gradlab/blob/7e643f4f820ecc0a82b6c7f90a787810c64c5ee8/experiments/goals/SuperMarioBros-Nes-v0/Level4-1/recipes/ppo-train-clear-80.yaml)
is consolidated into `ppo.yaml`. Its normalized training configuration and full
resolved environment are unchanged. Competing launchable YAMLs were removed only
after qualification; historical source, shared presets and checkpoints are preserved.


## Level5-1 — 2026-11-16

Fresh plain PPO Run `gradlab-f52c27c6655dcb4ef033b4fafca3c44a`, Attempt
`attempt-f6333b41b28078cd`, used seed 12 and one canonical Level5-1 start.
The retained checkpoint has **80/100** genuine target-origin clears
at 6,758,400 transitions. At that exact step, 23,859
episodes had completed and 5,296 had cleared cumulatively.
Its separate history peak was 88/100; the final window was
70/100 at 8,806,400 transitions. History peaks are not assigned to
another saved model.

The selected model is [`checkpoint-6758400-fd3e845f26dc2e0b`](https://pub-fc35c0b186ce4aad8eea5e93d38c99db.r2.dev/runs/gradlab-f52c27c6655dcb4ef033b4fafca3c44a/checkpoints/6758400-fd3e845f26dc2e0b7d03ede9e1fbc50c021e58e092455acf8b94e3a39b5098d9/model.zip):

- SHA-256: `fd3e845f26dc2e0b7d03ede9e1fbc50c021e58e092455acf8b94e3a39b5098d9`; 20,445,804 bytes, independently downloaded and verified.
- Source: `7e643f4f820ecc0a82b6c7f90a787810c64c5ee8`.
- Runtime digest: `sha256:5d316181cc88dce83c9613543e2439214544379adf5682459f1120590b322991`.
- Executed recipe document SHA-256: `c36e6172e6e8dbc945ccce18d6186431825908ea395ff3c563a568438c472805`.
- Normalized portable training configuration SHA-256: `86f21290395f36f805467687980f32c5853796e5746e460bf1dbb5550c893755`.

The finite bound was nominally 50,000,000 transitions (50,003,968 at a safe rollout boundary) and 6 hours. The declared neutral plateau stopped learning; complete delivery and task release were verified.
The read-only audit checked all 1,062 retained journal segments
and delivery through 1,376 events, with the selected window and
cumulative counts independently matched to actual MLflow history. No imported
initialization, archive curriculum, automatic evaluation or Monitoring was used.
This is training qualification, not Acceptance or across-seed reliability.

The [executed recipe](https://github.com/tsilva/gradlab/blob/7e643f4f820ecc0a82b6c7f90a787810c64c5ee8/experiments/goals/SuperMarioBros-Nes-v0/Level5-1/recipes/ppo-train-clear-80.yaml)
is consolidated into `ppo.yaml`. Its normalized training configuration and full
resolved environment are unchanged. Competing launchable YAMLs were removed only
after qualification; historical source, shared presets and checkpoints are preserved.


## Level5-2 — 2026-11-16

Fresh plain PPO Run `gradlab-117d89d33ef2315e87dc5b4c33c83f1b`, Attempt
`attempt-c0617e6fe372b123`, used seed 12 and one canonical Level5-2 start.
The retained checkpoint has **95/100** genuine target-origin clears
at 5,005,312 transitions. At that exact step, 16,895
episodes had completed and 3,663 had cleared cumulatively.
Its separate history peak was 97/100; the final window was
92/100 at 7,094,272 transitions. History peaks are not assigned to
another saved model.

The selected model is [`checkpoint-5005312-18d56433bf5222a6`](https://pub-fc35c0b186ce4aad8eea5e93d38c99db.r2.dev/runs/gradlab-117d89d33ef2315e87dc5b4c33c83f1b/checkpoints/5005312-18d56433bf5222a6c8d30a4ae9cf162a2a85da5fc6bad001d01de83a69c1db50/model.zip):

- SHA-256: `18d56433bf5222a6c8d30a4ae9cf162a2a85da5fc6bad001d01de83a69c1db50`; 20,445,806 bytes, independently downloaded and verified.
- Source: `7e643f4f820ecc0a82b6c7f90a787810c64c5ee8`.
- Runtime digest: `sha256:5d316181cc88dce83c9613543e2439214544379adf5682459f1120590b322991`.
- Executed recipe document SHA-256: `20576c302378fa4ade8ba69fa8d0b03d897affaa0ec144781b2def91b0b6e8c0`.
- Normalized portable training configuration SHA-256: `41b8668918692501f06539578ae2cbb324d07d69c0cfcbf27997b3f9a6c9ff44`.

The finite bound was nominally 50,000,000 transitions (50,003,968 at a safe rollout boundary) and 6 hours. The declared neutral plateau stopped learning; complete delivery and task release were verified.
The read-only audit checked all 867 retained journal segments
and delivery through 1,112 events, with the selected window and
cumulative counts independently matched to actual MLflow history. No imported
initialization, archive curriculum, automatic evaluation or Monitoring was used.
This is training qualification, not Acceptance or across-seed reliability.

The [executed recipe](https://github.com/tsilva/gradlab/blob/7e643f4f820ecc0a82b6c7f90a787810c64c5ee8/experiments/goals/SuperMarioBros-Nes-v0/Level5-2/recipes/ppo-train-clear-80.yaml)
is consolidated into `ppo.yaml`. Its normalized training configuration and full
resolved environment are unchanged. Competing launchable YAMLs were removed only
after qualification; historical source, shared presets and checkpoints are preserved.


## Level5-3 — 2026-11-16

Fresh plain PPO Run `gradlab-fb5ba3813922285fb9796afa0cc4ecc4`, Attempt
`attempt-6f53e3eb88ec2ed1`, used seed 12 and one canonical Level5-3 start.
The retained checkpoint has **95/100** genuine target-origin clears
at 28,688,384 transitions. At that exact step, 182,396
episodes had completed and 612 had cleared cumulatively.
Its separate history peak was 95/100; the final window was
60/100 at 29,736,960 transitions. History peaks are not assigned to
another saved model.

The selected model is [`checkpoint-28688384-29200c0e3f06b081`](https://pub-fc35c0b186ce4aad8eea5e93d38c99db.r2.dev/runs/gradlab-fb5ba3813922285fb9796afa0cc4ecc4/checkpoints/28688384-29200c0e3f06b08150a8d2e4d0db10532297c0fcfb38bdf4a5496763d914770d/model.zip):

- SHA-256: `29200c0e3f06b08150a8d2e4d0db10532297c0fcfb38bdf4a5496763d914770d`; 20,445,808 bytes, independently downloaded and verified.
- Source: `7e643f4f820ecc0a82b6c7f90a787810c64c5ee8`.
- Runtime digest: `sha256:5d316181cc88dce83c9613543e2439214544379adf5682459f1120590b322991`.
- Executed recipe document SHA-256: `318701ec2e58f2863b16142e5bfdb98ee4bd9d12556de83cf3c747a856cee8b6`.
- Normalized portable training configuration SHA-256: `1bfa789d8e56ca0168e4e7ba6d8a65ef49767ef8de3fa166eb7377163ec23627`.

The finite bound was nominally 50,000,000 transitions (50,003,968 at a safe rollout boundary) and 6 hours. The declared neutral plateau stopped learning; complete delivery and task release were verified.
The read-only audit checked all 3,685 retained journal segments
and delivery through 4,634 events, with the selected window and
cumulative counts independently matched to actual MLflow history. No imported
initialization, archive curriculum, automatic evaluation or Monitoring was used.
This is training qualification, not Acceptance or across-seed reliability.

The [executed recipe](https://github.com/tsilva/gradlab/blob/7e643f4f820ecc0a82b6c7f90a787810c64c5ee8/experiments/goals/SuperMarioBros-Nes-v0/Level5-3/recipes/ppo-train-clear-80.yaml)
is consolidated into `ppo.yaml`. Its normalized training configuration and full
resolved environment are unchanged. Competing launchable YAMLs were removed only
after qualification; historical source, shared presets and checkpoints are preserved.


## Level5-4 — 2026-11-16

Fresh plain PPO Run `gradlab-054e7c52878b67c632a71cc39d687a4c`, Attempt
`attempt-e53d00b709e3f05c`, used seed 12 and one canonical Level5-4 start.
The retained checkpoint has **95/100** genuine target-origin clears
at 4,120,576 transitions. At that exact step, 16,109
episodes had completed and 4,274 had cleared cumulatively.
Its separate history peak was 98/100; the final window was
86/100 at 5,447,680 transitions. History peaks are not assigned to
another saved model.

The selected model is [`checkpoint-4120576-8c347d1fe7357f32`](https://pub-fc35c0b186ce4aad8eea5e93d38c99db.r2.dev/runs/gradlab-054e7c52878b67c632a71cc39d687a4c/checkpoints/4120576-8c347d1fe7357f325df882dd78b637856e47ffcf121dede63585ce3198acf14e/model.zip):

- SHA-256: `8c347d1fe7357f325df882dd78b637856e47ffcf121dede63585ce3198acf14e`; 20,445,806 bytes, independently downloaded and verified.
- Source: `7e643f4f820ecc0a82b6c7f90a787810c64c5ee8`.
- Runtime digest: `sha256:5d316181cc88dce83c9613543e2439214544379adf5682459f1120590b322991`.
- Executed recipe document SHA-256: `0b633520340d790a328cab7e7f5bff989d51d99ae835031afaac8b9cfdc84e80`.
- Normalized portable training configuration SHA-256: `d0255638fd48769943c5070cafbc619c85f81e2d5074b02b0921a7b7ef7dc7f4`.

The finite bound was nominally 50,000,000 transitions (50,003,968 at a safe rollout boundary) and 6 hours. The declared neutral plateau stopped learning; complete delivery and task release were verified.
The read-only audit checked all 657 retained journal segments
and delivery through 856 events, with the selected window and
cumulative counts independently matched to actual MLflow history. No imported
initialization, archive curriculum, automatic evaluation or Monitoring was used.
This is training qualification, not Acceptance or across-seed reliability.

The [executed recipe](https://github.com/tsilva/gradlab/blob/7e643f4f820ecc0a82b6c7f90a787810c64c5ee8/experiments/goals/SuperMarioBros-Nes-v0/Level5-4/recipes/ppo-train-clear-80.yaml)
is consolidated into `ppo.yaml`. Its normalized training configuration and full
resolved environment are unchanged. Competing launchable YAMLs were removed only
after qualification; historical source, shared presets and checkpoints are preserved.


## Level6-2 — 2026-11-17

Fresh plain PPO Run `gradlab-337bb4878a21bf033d101dbfbe586b52`, Attempt
`attempt-236635117a6ed888`, used seed 12 and one canonical Level6-2 start.
The retained checkpoint has **80/100** genuine target-origin clears
at 7,430,144 transitions. At that exact step, 21,186
episodes had completed and 5,342 had cleared cumulatively.
Its separate history peak was 80/100; the final window was
70/100 at 8,478,720 transitions. History peaks are not assigned to
another saved model.

The selected model is [`checkpoint-7430144-f21dade919d4f45f`](https://pub-fc35c0b186ce4aad8eea5e93d38c99db.r2.dev/runs/gradlab-337bb4878a21bf033d101dbfbe586b52/checkpoints/7430144-f21dade919d4f45f17a1e3cb072e8e889bd95f293163775c15acb6334520162d/model.zip):

- SHA-256: `f21dade919d4f45f17a1e3cb072e8e889bd95f293163775c15acb6334520162d`; 20,445,810 bytes, independently downloaded and verified.
- Source: `7e643f4f820ecc0a82b6c7f90a787810c64c5ee8`.
- Runtime digest: `sha256:5d316181cc88dce83c9613543e2439214544379adf5682459f1120590b322991`.
- Executed recipe document SHA-256: `71700ef772c39fafb365b858d8abeb3211c8b90a14850706dd427ca97c1e1c0a`.
- Normalized portable training configuration SHA-256: `20d0f1f5a6a52266c856d7722fce2063809ac94e2e6430eebb3c2e94d389d894`.

The finite bound was nominally 50,000,000 transitions (50,003,968 at a safe rollout boundary) and 6 hours. The declared neutral plateau stopped learning; complete delivery and task release were verified.
The read-only audit checked all 1,040 retained journal segments
and delivery through 1,325 events, with the selected window and
cumulative counts independently matched to actual MLflow history. No imported
initialization, archive curriculum, automatic evaluation or Monitoring was used.
This is training qualification, not Acceptance or across-seed reliability.

The [executed recipe](https://github.com/tsilva/gradlab/blob/7e643f4f820ecc0a82b6c7f90a787810c64c5ee8/experiments/goals/SuperMarioBros-Nes-v0/Level6-2/recipes/ppo-train-clear-80.yaml)
is consolidated into `ppo.yaml`. Its normalized training configuration and full
resolved environment are unchanged. Competing launchable YAMLs were removed only
after qualification; historical source, shared presets and checkpoints are preserved.


## Level6-4 — 2026-11-17

Fresh plain PPO Run `gradlab-46310e7b3fc6c69484249b5bf5e8c9bb`, Attempt
`attempt-1d4e3d4fa3253660`, used seed 12 and one canonical Level6-4 start.
The retained checkpoint has **100/100** genuine target-origin clears
at 4,562,944 transitions. At that exact step, 16,908
episodes had completed and 3,565 had cleared cumulatively.
Its separate history peak was 100/100; the final window was
100/100 at 4,562,944 transitions. History peaks are not assigned to
another saved model.

The selected model is [`checkpoint-4562944-460e5cdbc350db3b`](https://pub-fc35c0b186ce4aad8eea5e93d38c99db.r2.dev/runs/gradlab-46310e7b3fc6c69484249b5bf5e8c9bb/checkpoints/4562944-460e5cdbc350db3b1e23b4d2d8af519aaecbe81b5553650e120d9a3453dedf82/model.zip):

- SHA-256: `460e5cdbc350db3b1e23b4d2d8af519aaecbe81b5553650e120d9a3453dedf82`; 20,445,806 bytes, independently downloaded and verified.
- Source: `7e643f4f820ecc0a82b6c7f90a787810c64c5ee8`.
- Runtime digest: `sha256:5d316181cc88dce83c9613543e2439214544379adf5682459f1120590b322991`.
- Executed recipe document SHA-256: `33dcd94c4f22eff0c0523b02e761082ac52112383e325b4856e77370f4af3c4c`.
- Normalized portable training configuration SHA-256: `b62c3d9982bd0cff3a17e4896b14786b3c9294865f434e6970d7b1302cfd6378`.

The finite bound was nominally 50,000,000 transitions (50,003,968 at a safe rollout boundary) and 6 hours. The 100/100 clear target stopped learning; complete delivery and task release were verified.
The read-only audit checked all 543 retained journal segments
and delivery through 719 events, with the selected window and
cumulative counts independently matched to actual MLflow history. No imported
initialization, archive curriculum, automatic evaluation or Monitoring was used.
This is training qualification, not Acceptance or across-seed reliability.

The [executed recipe](https://github.com/tsilva/gradlab/blob/7e643f4f820ecc0a82b6c7f90a787810c64c5ee8/experiments/goals/SuperMarioBros-Nes-v0/Level6-4/recipes/ppo-train-clear-80.yaml)
is consolidated into `ppo.yaml`. Its normalized training configuration and full
resolved environment are unchanged. Competing launchable YAMLs were removed only
after qualification; historical source, shared presets and checkpoints are preserved.


## Level7-1 — 2026-11-17

Fresh plain PPO Run `gradlab-c41628637d85caf5bf0ab3313ee27ff8`, Attempt
`attempt-971ad00916decb56`, used seed 12 and one canonical Level7-1 start.
The retained checkpoint has **80/100** genuine target-origin clears
at 3,252,224 transitions. At that exact step, 12,692
episodes had completed and 1,409 had cleared cumulatively.
Its separate history peak was 87/100; the final window was
78/100 at 4,694,016 transitions. History peaks are not assigned to
another saved model.

The selected model is [`checkpoint-3252224-f769be6850bd89d8`](https://pub-fc35c0b186ce4aad8eea5e93d38c99db.r2.dev/runs/gradlab-c41628637d85caf5bf0ab3313ee27ff8/checkpoints/3252224-f769be6850bd89d8778e2b7045023e5efc4d95e4f40ff6b0347758a38ed06c03/model.zip):

- SHA-256: `f769be6850bd89d8778e2b7045023e5efc4d95e4f40ff6b0347758a38ed06c03`; 20,445,807 bytes, independently downloaded and verified.
- Source: `7e643f4f820ecc0a82b6c7f90a787810c64c5ee8`.
- Runtime digest: `sha256:5d316181cc88dce83c9613543e2439214544379adf5682459f1120590b322991`.
- Executed recipe document SHA-256: `1e55a6f1e6e91f301bbf76d24d2087d220e3890d3306ff5338777722e55cb2fd`.
- Normalized portable training configuration SHA-256: `b3ecb18d2c3452df187e58b2e161cc8beb3d55fa74c81731341f96baf745939f`.

The finite bound was nominally 50,000,000 transitions (50,003,968 at a safe rollout boundary) and 6 hours. The declared neutral plateau stopped learning; complete delivery and task release were verified.
The read-only audit checked all 571 retained journal segments
and delivery through 737 events, with the selected window and
cumulative counts independently matched to actual MLflow history. No imported
initialization, archive curriculum, automatic evaluation or Monitoring was used.
This is training qualification, not Acceptance or across-seed reliability.

The [executed recipe](https://github.com/tsilva/gradlab/blob/7e643f4f820ecc0a82b6c7f90a787810c64c5ee8/experiments/goals/SuperMarioBros-Nes-v0/Level7-1/recipes/ppo-train-clear-80.yaml)
is consolidated into `ppo.yaml`. Its normalized training configuration and full
resolved environment are unchanged. Competing launchable YAMLs were removed only
after qualification; historical source, shared presets and checkpoints are preserved.


## Level8-2 — 2026-11-17

Fresh plain PPO Run `gradlab-b94f25d7f2960b060093f46aeca74ac7`, Attempt
`attempt-e11815089f0478bd`, used seed 12 and one canonical Level8-2 start.
The retained checkpoint has **83/100** genuine target-origin clears
at 10,780,672 transitions. At that exact step, 33,266
episodes had completed and 5,253 had cleared cumulatively.
Its separate history peak was 86/100; the final window was
30/100 at 12,066,816 transitions. History peaks are not assigned to
another saved model.

The selected model is [`checkpoint-10780672-70ede32bc9d7c909`](https://pub-fc35c0b186ce4aad8eea5e93d38c99db.r2.dev/runs/gradlab-b94f25d7f2960b060093f46aeca74ac7/checkpoints/10780672-70ede32bc9d7c909d68f9312e3eb3b81ba14ad4b6955d8e59d95d6ac396fe3c1/model.zip):

- SHA-256: `70ede32bc9d7c909d68f9312e3eb3b81ba14ad4b6955d8e59d95d6ac396fe3c1`; 20,445,808 bytes, independently downloaded and verified.
- Source: `7e643f4f820ecc0a82b6c7f90a787810c64c5ee8`.
- Runtime digest: `sha256:5d316181cc88dce83c9613543e2439214544379adf5682459f1120590b322991`.
- Executed recipe document SHA-256: `959346438653186324105eeae1ce3f627c12dbf261dc1fb29a0aa00bcaab7b03`.
- Normalized portable training configuration SHA-256: `1840154dacb38ff47a4190a597b327053f81043a4823df7ce6057d4895ef6179`.

The finite bound was nominally 50,000,000 transitions (50,003,968 at a safe rollout boundary) and 6 hours. The declared neutral plateau stopped learning; complete delivery and task release were verified.
The read-only audit checked all 1,476 retained journal segments
and delivery through 1,884 events, with the selected window and
cumulative counts independently matched to actual MLflow history. No imported
initialization, archive curriculum, automatic evaluation or Monitoring was used.
This is training qualification, not Acceptance or across-seed reliability.

The [executed recipe](https://github.com/tsilva/gradlab/blob/7e643f4f820ecc0a82b6c7f90a787810c64c5ee8/experiments/goals/SuperMarioBros-Nes-v0/Level8-2/recipes/ppo-train-clear-80.yaml)
is consolidated into `ppo.yaml`. Its normalized training configuration and full
resolved environment are unchanged. Competing launchable YAMLs were removed only
after qualification; historical source, shared presets and checkpoints are preserved.


## Level8-3 — 2026-11-17

Fresh plain PPO Run `gradlab-f789eeefefa8936d790da4590c4b2b37`, Attempt
`attempt-88510340ae25e289`, used seed 12 and one canonical Level8-3 start.
The retained checkpoint has **81/100** genuine target-origin clears
at 4,612,096 transitions. At that exact step, 16,717
episodes had completed and 1,965 had cleared cumulatively.
Its separate history peak was 88/100; the final window was
66/100 at 5,791,744 transitions. History peaks are not assigned to
another saved model.

The selected model is [`checkpoint-4612096-9628934ee3823db9`](https://pub-fc35c0b186ce4aad8eea5e93d38c99db.r2.dev/runs/gradlab-f789eeefefa8936d790da4590c4b2b37/checkpoints/4612096-9628934ee3823db9b11fb1fee892efbd8aee285d5c55bf258fc62c2b11e77743/model.zip):

- SHA-256: `9628934ee3823db9b11fb1fee892efbd8aee285d5c55bf258fc62c2b11e77743`; 20,445,806 bytes, independently downloaded and verified.
- Source: `7e643f4f820ecc0a82b6c7f90a787810c64c5ee8`.
- Runtime digest: `sha256:5d316181cc88dce83c9613543e2439214544379adf5682459f1120590b322991`.
- Executed recipe document SHA-256: `5282d5aaa4e12b0c25a001cad125e270ff83b4387b42a2fbf6b944f3b7ed8640`.
- Normalized portable training configuration SHA-256: `06133dcab70c1e17a1f292c31ebc3032872439d80a1a06e5c2abce55091c7070`.

The finite bound was nominally 50,000,000 transitions (50,003,968 at a safe rollout boundary) and 6 hours. The declared neutral plateau stopped learning; complete delivery and task release were verified.
The read-only audit checked all 672 retained journal segments
and delivery through 906 events, with the selected window and
cumulative counts independently matched to actual MLflow history. No imported
initialization, archive curriculum, automatic evaluation or Monitoring was used.
This is training qualification, not Acceptance or across-seed reliability.

The [executed recipe](https://github.com/tsilva/gradlab/blob/7e643f4f820ecc0a82b6c7f90a787810c64c5ee8/experiments/goals/SuperMarioBros-Nes-v0/Level8-3/recipes/ppo-train-clear-80.yaml)
is consolidated into `ppo.yaml`. Its normalized training configuration and full
resolved environment are unchanged. Competing launchable YAMLs were removed only
after qualification; historical source, shared presets and checkpoints are preserved.
