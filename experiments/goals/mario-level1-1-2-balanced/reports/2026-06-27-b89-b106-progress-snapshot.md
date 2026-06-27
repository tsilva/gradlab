# 2026-06-27 B89-B106 Progress Snapshot

Goal metric: peak `train/info/level_complete/rate/min/last`, requiring both
seeds in the same two-seed batch to exceed strict `0.80` within the 5M training
cap. This snapshot uses full-row W&B history scans for peak detection.

## Live Status

Queue after adding B105 and B106:

- Succeeded train jobs: `10`
- Running train jobs: `5`
- Pending train jobs: `30`
- Eval jobs: none

Active beast-3 workers:

- `job=36`: `b89_l11l12_complete25_s203_20260627T110832Z`
- `job=37`: `b90_l11l12_lowpress_l12bias_s204_20260627T112001Z`
- `job=38`: `b90_l11l12_lowpress_l12bias_s205_20260627T112001Z`
- `job=39`: `b91_l11l12_lowpress_complete25_s206_20260627T112013Z`
- `job=40`: `b91_l11l12_lowpress_complete25_s207_20260627T112013Z`

Fleet plan: keep the existing beast-3 RTX4090 container with five workers on
digest `c672be38cd0f`; no reconcile action required.

## Current W&B High-Watermarks

No two-seed batch has solved the goal.

| Group | Peaks | Decision |
| --- | ---: | --- |
| B86 | `0.16`, `0.33`, `0.80`, `0.16`, `0.11` | near-miss only; no strict `>0.80` and no two-seed success |
| B87 | `0.09`, `0.12` | failed; low update pressure alone did not fix Level1-2 |
| B88 | `0.07`, `0.09` | failed; Level1-2 sampling bias alone did not fix Level1-2 |
| B89 | `0.41`, `0.02` so far | seed 202 best active signal but below target; seed 203 running |
| B90 | `0.01`, `0.01` so far | low pressure plus Level1-2 bias still weak |
| B91 | `0.03`, `0.00` so far | low pressure plus completion reward has the best new active hint |

Active run detail at scan time:

| Run | Step | Peak min-rate | Last min-rate | Last L1-1 | Last L1-2 | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `b89_l11l12_complete25_s203_20260627T110832Z` | `1,302,528` | `0.02` | `0.02` | `0.08` | `0.02` | paired B89 seed remains weak |
| `b90_l11l12_lowpress_l12bias_s204_20260627T112001Z` | `1,220,608` | `0.01` | `0.00` | `0.00` | `0.00` | no useful signal yet |
| `b90_l11l12_lowpress_l12bias_s205_20260627T112001Z` | `1,232,592` | `0.01` | `0.00` | `0.00` | `0.00` | no useful signal yet |
| `b91_l11l12_lowpress_complete25_s206_20260627T112013Z` | `1,222,320` | `0.03` | `0.03` | `0.03` | `0.06` | best new active hint |
| `b91_l11l12_lowpress_complete25_s207_20260627T112013Z` | `1,217,824` | `0.00` | `0.00` | `0.02` | `0.00` | paired seed not lifting yet |

Interpretation:

- Completion reward remains the only lever with a material signal.
- Low update pressure plus completion reward is weak so far, but it is the only
  active branch besides B89 seed 202 with any balanced lift.
- B103/B104 already cover stronger completion reward with soft Level1-2
  sampling, but not lower update pressure.

## Added Backfills

B105 and B106 cover the missing stronger-completion plus lower-update-pressure
interaction while keeping the scope to legal reward/hyperparameter/state
sampling knobs.

### B105

- Spec: `specs/b105-lowpress-complete50-l11l12-two-seed.json`
- W&B group: `b105-l11l12-lowpress-complete50-two-seed`
- Seeds: `198,199`
- Delta from B91: `completion_reward=50` instead of `25`
- Jobs:
  - `67`: `b105_l11l12_lowpress_complete50_s198_20260627T123820Z`
  - `68`: `b105_l11l12_lowpress_complete50_s199_20260627T123820Z`

B105 tests whether the B91 recipe needs a stronger true-clear reward before
adding sampling bias or slower entropy.

### B106

- Spec: `specs/b106-lowpress-complete50-slowent-l12soft-l11l12-two-seed.json`
- W&B group: `b106-l11l12-lowpress-complete50-slowent-l12soft-two-seed`
- Seeds: `202,203`
- Delta from B104: add lower update pressure,
  `target_kl=0.16`, `learning_rate_final=0.0001`, and
  `learning_rate_schedule_timesteps=4000000`
- Jobs:
  - `69`: `b106_l11l12_lowpress_complete50_slowent_l12soft_s202_20260627T123823Z`
  - `70`: `b106_l11l12_lowpress_complete50_slowent_l12soft_s203_20260627T123823Z`

B106 tests the strongest compact anti-collapse bundle currently queued:
`completion_reward=50`, low update pressure, slower entropy decay, higher
entropy floor, and `state_probs=[0.45,0.55]`.

Both specs validated with `rlab.job_queue.load_spec_document` and were enqueued
profileless for `rtx4090` with runtime image digest `c672be38cd0f`.
