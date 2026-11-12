# Mario per-level training recipe qualification

This campaign qualifies current launchable recipes independently for ordinary
Super Mario Bros. Levels 1-1 through 8-3. Level8-4 is explicitly deferred; it
remains unfinished scope of the original all-level objective. Mixed-level and
end-to-end goals are outside this campaign.

- A recipe qualifies only after fresh training reaches at least 80 genuine
  level clears in a full latest rolling window of 100 completed target-origin
  episodes from that level's canonical start. Historical checkpoints and results
  guide ordering but do not qualify the current recipe.
- Warm-up windows, archive or curriculum starts, imported replay successes,
  shaped returns and progress measures cannot satisfy this criterion. Actor-only
  SIL may reuse this learner's own episodes for learning; replay updates do not
  add completed episodes to the success window.
- Stop at the first qualifying reported window and the next safe learner
  boundary. Do not continue to pursue a perfect window. Preserve the metric
  observation step separately from the saved checkpoint step.
- Record the window's episode and clear counts, cumulative episode count,
  qualifying training step, seed, initialization, recipe and resolved-config
  hashes, exact source/runtime and hash-verifiable checkpoint identity.
- Use the selected local GPU sequentially, one learner and one target level at
  a time. Freeze finite transition, time and attempt bounds before admission.
  When a level exhausts those bounds, retain its best evidence and unresolved
  status and proceed to the next level; campaign completion requires every
  included level to qualify.
- PPO is the baseline. Existing actor-only SIL evidence may guide fresh
  attempts on levels where PPO has struggled; prior Level1-3 success does not
  establish that SIL transfers to another level.
- Log every admitted Run online to MLflow and preserve checkpoints and receipts
  in the configured Cloudflare R2 stores through approved access routes. Require
  exact-source execution and verified CUDA use and resource release. Missing
  prerequisites block admission; they do not authorize credential bypasses or
  alternative paid compute.
- Disable automatic and post-training checkpoint evaluation for campaign Runs.
  This online training criterion is separate from frozen-checkpoint evaluation,
  across-seed reliability, formal Acceptance, Promotion and Release. Preserve
  those contracts for later explicitly requested work. Do not publish to
  Hugging Face or YouTube as part of this campaign.
- After a level qualifies, keep one canonical working launchable recipe YAML
  for it and remove its competing launchable recipe YAMLs only after repairing
  references and verifying the retained recipe loads. Preserve shared presets,
  resolved historical configurations, initialization assets, checkpoints,
  datasets and research evidence. Keep competitors for unresolved levels.

## Rolling metric interpretation

`train/success/min` is the minimum of the mature per-start success fractions.
For one configured canonical start, it is the fraction of clears among that
start's latest 100 genuine completed target-origin episodes. Its minimum is
across starts, not across the binary episode outcomes. Threshold `0.8` therefore
means at least 80/100; threshold `1.0` requires 100 consecutive clears. The
campaign recipes retain the inherited perfect-window condition as observation
only and use `clear_80` as the stopping condition. Existing goal defaults and
the separate 100/100 Acceptance rule remain available.
