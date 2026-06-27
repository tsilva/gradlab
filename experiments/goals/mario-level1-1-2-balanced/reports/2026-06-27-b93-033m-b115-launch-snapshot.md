# 2026-06-27 B93 3.33M Monitor And B115 Launch Snapshot

Goal metric: peak `train/info/level_complete/rate/min/last`, requiring both
seeds in the same two-seed batch to exceed strict `0.80` within the 5M training
cap. This snapshot uses full-row W&B history scans and judges by high-watermark,
not final value.

Scan time: `2026-06-27T14:22:22Z`.

## Live Queue And Fleet

beast-3 remains healthy and already reconciled.

- Train jobs after B115 enqueue: `6` failed, `10` succeeded, `5` running,
  `42` pending.
- Eval jobs: none.
- Fleet plan: no action; keep the existing beast-3 RTX4090 managed runner.
- Runner digest: `c672be38cd0f`.
- Disk check before enqueue: `/` has about `40G` free and is at `82%` use.

Active beast-3 workers:

- `job=77`: `b110_l11l12_gentlerpress_complete25_l12soft_s206_20260627T133104Z`
- `job=78`: `b110_l11l12_gentlerpress_complete25_l12soft_s207_20260627T133104Z`
- `job=44`: `b92_l11l12_slowent_s199_20260627T112706Z`
- `job=41`: `b93_l11l12_slowent_l12bias_s200_20260627T112704Z`
- `job=42`: `b93_l11l12_slowent_l12bias_s201_20260627T112704Z`

New next-in-queue arm:

- `job=87`: `b115_l11l12_slowent_l12bias_complete25_s200_20260627T142436Z`
- `job=88`: `b115_l11l12_slowent_l12bias_complete25_s201_20260627T142436Z`

B115 is queued ahead of B111-B114 because it incorporates the strongest live
two-seed evidence from B93.

## W&B High-Watermarks

No same-batch candidate has solved the goal yet.

| Run | Step | Peak min-rate | Peak step | Latest min-rate | Latest L1-1 rate | Latest L1-2 rate | L1-1 clears | L1-2 clears | FPS | Reward | Approx KL | Clip frac | EV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `b110_l11l12_gentlerpress_complete25_l12soft_s206_20260627T133104Z` | `3,364,784` | `0.17` | `2,007,472` | `0.03` | `0.61` | `0.03` | `231` | `88` | `1,213` | `2389.735` | `0.0112` | `0.0891` | `0.9691` |
| `b110_l11l12_gentlerpress_complete25_l12soft_s207_20260627T133104Z` | `3,333,392` | `0.17` | `1,786,272` | `0.03` | `0.73` | `0.03` | `467` | `101` | `1,201` | `2285.512` | `0.0091` | `0.0773` | `0.9825` |
| `b92_l11l12_slowent_s199_20260627T112706Z` | `3,317,760` | `0.11` | `3,297,600` | `0.11` | `0.90` | `0.11` | `1457` | `33` | `1,192` | `2352.919` | `0.0146` | `0.1275` | `0.9864` |
| `b93_l11l12_slowent_l12bias_s200_20260627T112704Z` | `3,317,760` | `0.20` | `2,022,256` | `0.15` | `0.82` | `0.15` | `540` | `141` | `1,190` | `2427.715` | `0.0179` | `0.1526` | `0.9866` |
| `b93_l11l12_slowent_l12bias_s201_20260627T112704Z` | `3,334,144` | `0.30` | `3,275,968` | `0.27` | `0.64` | `0.27` | `271` | `204` | `1,191` | `2189.495` | `0.0131` | `0.1371` | `0.9871` |

B91 near-miss reference from the same scan:

| Run | State | Step | Peak min-rate | Peak step | Latest min-rate | Peak L1-1 | Peak L1-2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `b91_l11l12_lowpress_complete25_s206_20260627T112013Z` | finished | `4,599,488` | `0.88` | `4,089,456` | `0.00` | `0.89` | `0.88` |
| `b91_l11l12_lowpress_complete25_s207_20260627T112013Z` | crashed | `4,600,464` | `0.67` | `3,001,392` | `0.03` | `0.76` | `0.67` |

## Interpretation

- B110 remains below the threshold and is mostly Level1-2-limited despite
  strong Level1-1 rates. Its peak remains `0.17` on both seeds.
- B92 is improving but still not balanced: latest Level1-1 is `0.90`, while
  Level1-2 is only `0.11`.
- B93 is now the strongest active two-seed direction. Seed 200 is at peak
  `0.20`, and seed 201 reached peak `0.30` while currently holding `0.27`.
- B93's improvement is not enough for completion, but it is a better live clue
  than B110: slow entropy plus a stronger `40/60` Level1-2 sampling bias is
  producing both-level competence on both seeds.
- B91 remains the only strict crossing clue, and its differentiating lever was
  `completion_reward=25`.

## B115 Launch Decision

Queued B115 from spec:
`experiments/goals/mario-level1-1-2-balanced/specs/b115-slowent-l12bias-complete25-l11l12-two-seed.json`.

B115 preserves B93 exactly on the currently useful recipe shape:

- `state_probs=[0.4, 0.6]`
- `target_kl=0.2`
- `clip_range=0.15`
- `learning_rate=0.00015`
- `learning_rate_final=0.00015`
- `ent_coef_final=0.001`
- `ent_coef_schedule_timesteps=4000000`

The single intended delta from B93 is:

- add `completion_reward=25`

Reason:

- B93 is the best live two-seed trend but lacks the B91 completion lift.
- B91's completion reward produced the only strict `>0.80` crossing, but B91's
  low-pressure shape was not reliable across both seeds.
- B115 tests whether B93's better retention plus B91's clean-completion bonus
  can raise the paired high-watermark without sacrificing Level1-1.

Next monitor condition:

- Continue scanning active B93 through cap because it is still improving.
- When the current slots finish, B115 should start before B111-B114. Compare
  B115 against B93 at roughly `3.0M`: paired peak min-rate, seed 201 Level1-2
  retention, and whether seed 200 can move beyond the current `0.20` ceiling.
