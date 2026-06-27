# 2026-06-27 B87-B100 Progress Snapshot

Goal metric: peak `train/info/level_complete/rate/min/last`, requiring both
seeds in the same two-seed batch to exceed `0.80` within the 5M training cap.
The metric is selected by W&B history high-watermark, not the final summary
value.

## Live Status

Queue snapshot after adding B100:

- Running: 5 train jobs on beast-3.
- Pending: 23 train jobs.
- Succeeded: 5 B86 incumbent runs.
- Fleet plan: keep existing beast-3 RTX4090 container with 5 workers on digest
  `c672be38cd0f`; no reconcile action required.

Running jobs:

- `b87_l11l12_lowpress_s198_20260627T110807Z`
- `b87_l11l12_lowpress_s199_20260627T110807Z`
- `b88_l11l12_l12bias_s200_20260627T110819Z`
- `b88_l11l12_l12bias_s201_20260627T110819Z`
- `b89_l11l12_complete25_s202_20260627T110832Z`

Next pending jobs start with B89 seed 203, then B90/B91. B92-B100 are queued
behind those jobs.

## Current W&B High-Watermarks

Full-row W&B scans over B86-B99 groups currently find W&B runs only for B86 and
the active B87/B88/B89 tranche. B90-B100 are still pending and have no W&B run
history yet.

| Group | Run | State | Peak min-rate | Peak step | Last min-rate | Last L1-1 | Last L1-2 | Last step | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B86 | `b86_l11l12_b74current_s193_20260627T091726Z` | finished | `0.33` | `1,992,720` | `0.08` | `0.78` | `0.08` | `5,005,312` | incumbent support seed |
| B86 | `b86_l11l12_b74current_s194_20260627T091726Z` | finished | `0.16` | `2,911,312` | `0.01` | `0.75` | `0.01` | `5,005,312` | L1-2 collapse |
| B86 | `b86_l11l12_b74current_s195_20260627T091726Z` | finished | `0.80` | `4,070,256` | `0.13` | `0.87` | `0.13` | `5,005,312` | near-miss; not strict `>0.80` |
| B86 | `b86_l11l12_b74current_s196_20260627T091726Z` | finished | `0.11` | `2,894,560` | `0.00` | `0.50` | `0.00` | `5,005,312` | L1-2 collapse |
| B86 | `b86_l11l12_b74current_s197_20260627T091726Z` | finished | `0.16` | `3,372,064` | `0.06` | `0.76` | `0.06` | `5,005,312` | L1-2 collapse |
| B87 | `b87_l11l12_lowpress_s198_20260627T110807Z` | running | `0.09` | `2,121,920` | `0.04` | `0.83` | `0.04` | `3,940,352` | low pressure alone not lifting L1-2 |
| B87 | `b87_l11l12_lowpress_s199_20260627T110807Z` | running | `0.12` | `2,512,032` | `0.01` | `0.56` | `0.01` | `3,891,200` | low pressure alone not lifting L1-2 |
| B88 | `b88_l11l12_l12bias_s200_20260627T110819Z` | running | `0.06` | `1,925,408` | `0.05` | `0.80` | `0.05` | `3,922,768` | L1-2 bias alone not enough |
| B88 | `b88_l11l12_l12bias_s201_20260627T110819Z` | running | `0.09` | `3,084,848` | `0.00` | `0.19` | `0.00` | `3,908,896` | weak balanced lift |
| B89 | `b89_l11l12_complete25_s202_20260627T110832Z` | running | `0.41` | `2,512,208` | `0.02` | `0.77` | `0.02` | `3,923,968` | best active lift; L1-2 still collapses |

Decision: no batch is solved. B86 remains a near miss because its best seed
peaked at exactly `0.80`, not strictly greater than `0.80`, and no same-batch
pair crossed the threshold. Among active runs, B89 is the only meaningful lift;
B87 and B88 mostly teach that lower update pressure or Level1-2 sampling bias
without a completion reward does not solve the bottleneck.

## Added Backfill

B100 stays on the completion-reward branch and combines two anti-collapse
adjustments that are already queued separately:

- Keep `completion_reward=25`, because B89 is the only active arm with a
  material balanced high-watermark.
- Use B91-style lower update pressure: `target_kl=0.16`,
  `learning_rate_final=0.0001`, and
  `learning_rate_schedule_timesteps=4000000`.
- Use B99-style soft Level1-2 sampling bias: `state_probs=[0.45,0.55]`.

Added spec:

- `specs/b100-lowpress-complete25-l12soft-l11l12-two-seed.json`
  - Seeds: `206,207`
  - W&B group: `b100-l11l12-lowpress-complete25-l12soft-two-seed`
  - Run names:
    - `b100_l11l12_lowpress_complete25_l12soft_s206_20260627T120559Z`
    - `b100_l11l12_lowpress_complete25_l12soft_s207_20260627T120559Z`
  - Jobs: `57`, `58`

The spec validated with `rlab.job_queue.load_spec_document` and was enqueued
profileless for `rtx4090` with runtime image digest `c672be38cd0f`.
