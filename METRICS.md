# Metrics schema v21

This file is the source of truth for gradlab telemetry. The Python registry loads the table below
and requires every emitted metric to match an exact registry entry or a bounded template.

`Training Success` is a provisional Run classification produced by a Research Goal's declared
threshold over training-time metrics. It is a cheap proxy used to classify and compare Runs; it
does not establish that a Research Goal is solved. `Acceptance` is the separate determination from
an Evaluated Goal's stricter checkpoint evaluation on seeds different from the training seeds.
Only Acceptance can authorize Promotion, and neither metric delivery nor its W&B projection is the
authority for that decision.

## Surfaces and dimensions

- W&B is the authoritative scientific metric surface. One supervisor process inside the training
  container is the only process allowed to open and write the logical W&B run.
- The learner writes structured events only to its embedded SQLite WAL outbox. It performs no
  network I/O for metrics, checkpoint publication, or evaluation dispatch.
- W&B-disabled runs retain history frames in SQLite with `local_only` delivery status so bounded
  benchmarks can evaluate every rollout rather than only the latest scalar; those frames never
  enter the publisher retry queue.
- Modal never receives W&B credentials. The supervisor validates Modal results and appends their
  metrics to the same W&B run.
- SQLite and private-R2 JSONL metric segments are delivery and recovery transports, not competing
  scientific metric stores. Verified terminal journals expire after seven days.
- Public model R2 contains immutable checkpoint closures and a mutable no-cache run index. Private
  eval R2 contains intents, results, and episode evidence. Private control R2 contains leases,
  journals, promotions, and terminal receipts.
- Player checkpoint tables populate full-evaluation columns only from verified checkpoint-evaluation
  evidence. Training-proxy columns sample W&B history at the latest `train/global_step` no greater
  than the checkpoint step, but only after W&B's metrics schema, selection rank, and checkpoint
  acceptance contract match the immutable recipe. Any contract mismatch suppresses all optional
  W&B enrichment and surfaces a warning rather than displaying potentially misbound proxy values.
  When `checkpoint_eval_backend` is `none`, the supervisor intentionally omits
  `checkpoint_eval_contract`; the catalog must validate that expected absence against the immutable
  recipe and W&B run dimensions without suppressing otherwise compatible training-proxy history.
  Full-evaluation columns remain unavailable until verified checkpoint-evaluation evidence exists.
- W&B config contains run-defining dimensions: `metrics_schema_version: 21`,
  `metrics_episode_window_size: 100`, `training_backend_id`,
  `training_backend_config_hash`, `algorithm_id`, goal,
  environment, starts, seed, frame skip, environment count, hyperparameters, eval protocol, and
  runtime versions. Operational attribution includes `compute_target`,
  `dstack_coordinator_id`, `dstack_project`, `dstack_task`, and `attempt_id`; these are immutable
  run/attempt dimensions, not scientific metrics.
- `experiments/goals/_workspaces.yaml` is the presentation source for managed W&B project
  workspace views. It selects registered metrics and derives each panel title from its exact metric
  selectors without emitting aliases or changing their scientific axes or semantics; every project
  resolved from an active checked-in goal inherits its default profile unless the declaration
  assigns a complete project-specific profile. Project overrides omit panels or individual metric
  selectors that do not apply to that project's active goals and recipes rather than rendering
  empty placeholders. A managed line panel contains one distinct measure; only mutually exclusive
  algorithm-specific variants of that same measure share a panel.
- `goal_contract_sha256` is the semantic SHA-256 of the fully composed, rendered, validated goal
  contract. Generated goal reports use it with `goal_slug` to keep current-contract leaderboards
  comparable; noncurrent contracts are not queried or rendered.
- Catalog-backed runs also record `effective_goal_contract_sha256`, `reward_program_kind`,
  `reward_program_revision`, `reward_shape`, `reward_shape_sha256`, and
  `reward_shape_is_default`. Reward-derived returns are comparable only when the selected reward
  semantic identity and effective goal contract match; the readable key alone is not sufficient.
- `goal_variant_id` is the goal-scoped digest of `goal_slug`, `goal_contract_sha256`, and
  `effective_goal_contract_sha256`; it groups only runs with the same authored and fully
  materialized effective goal contracts. `goal_variant_label`, `goal_variant_source_relation`,
  `goal_variant_descriptor_sha256`, and the bounded `goal_variant_diff_json` are catalog/search
  projections, not scientific authority. The immutable run manifest and `recipe.json` retain the
  complete descriptor, while the rebuildable private-control-R2 per-goal index serves the player
  without scanning runs, objects, or artifacts.
- `leader/*` contains diagnostic projections of the selected checkpoint. The
  create-only private-R2 `PromotionReceipt` is the authoritative selection.
- `ops/terminal/state` and `ops/terminal/reason` are W&B summary-only
  catalog projections, not history metrics; the private-R2 `TerminalReceipt` remains authoritative.
- Heavy model bytes, videos, replays, episode rows, diagnostics, and recovery payloads never go to
  W&B.
- Interactive playback uses local descriptor keys such as `reward/shaped`, `policy/value`, and
  `action/executed` to configure live panels. They are typed projections of one streamed transition
  or its bounded in-browser history, are not emitted metrics, and must not be interpreted as aliases
  for similarly named W&B registry entries. `reward/provider` is the provider output before the
  task program and gradlab-owned reward transform; `reward/shaped` is the final policy-facing reward
  after task shaping, scaling, and clipping. Playback computes return-to-go and `V(s) - G(s)` only
  for a completed, stochastic, policy-driven trajectory whose policy environment, reward stream,
  discount, action sampling, and boundary/bootstrap semantics match training. True terminations use
  a zero terminal value; truncations use the checkpoint critic's value of the exact final policy
  input and display that bootstrap explicitly. Otherwise those critic diagnostics are explicitly
  unavailable rather than numerically compared.

The active checkpoint protocol is `acceptance`; complete evaluation evidence additionally emits
the registered evaluation aggregates. Dimension IDs must be unique and match `[A-Za-z0-9_.-]+`; unsafe IDs are
rejected rather than silently rewritten. Starts use the same readable ID in training and evaluation.
Provider `info` fields never become metrics automatically.

Configuration-selected internal learner feedback, such as an archive curriculum's per-start
priority statistic, is not telemetry merely because it has a readable name. Internal feedback
identifiers do not use metric paths and are not published to W&B unless a separately registered
metric explicitly projects them. If projected, the emitted name and semantics must appear in the
registry below. For SB3 PPO and A2C, archive-curriculum `priority_metric: value_error` specifically
means the arithmetic mean of `abs(A_t)` over one completed archive-origin trajectory, where `A_t`
is raw GAE before PPO minibatch normalization. That scalar updates the archive's cell-level EMA and
is intentionally not emitted; `train/curriculum/archive/feedback/trajectory/count` reports only how
many such trajectory updates were committed.

An episode metric is a **return**. `reward` is reserved for per-step shaping and component
attribution. Frame skip remains run config. W&B uses three explicit axes:

- `train/global_step`: policy environment transitions consumed by training.
- `eval/checkpoint/step`: step of the checkpoint represented by an evaluation row.
- `ops/event_sequence`: durable supervisor delivery order.

Across frame-skip ablations, equal `train/global_step` means equal policy-transition counts,
not equal simulated game time. Nominal native-frame exposure is `train/global_step * frame_skip`;
this estimate excludes reset work and may differ from actual frames when action repeats end early.
Similar learning curves on this axis establish similar observed policy-transition efficiency,
not native-frame efficiency or wall-clock efficiency.

Each axis is configured with a W&B `max` summary reducer. W&B's public API may therefore expose
its summary value as a reducer mapping such as `{"max": 5046272}` rather than as a bare number.
Catalog and report consumers must unwrap the configured reducer value; a recipe's requested
`timesteps` cap is not a substitute for the observed `train/global_step`.
Before its first history row is logged, every concrete metric is explicitly bound to its applicable
scientific axis. W&B's internal `Step` is delivery order only and must not become the default X-axis
for scientific charts.

Asynchronous evaluations may arrive after later training rows without changing their scientific
X-axis. Each producer writes only its applicable scientific axis; durable delivery order uses
`ops/event_sequence`.

Current runs declare schema v21, and the supervisor validates and emits only v21 names. GradLab
does not read, project, or preserve noncurrent W&B or R2 schemas.

Recent training return, progress, episode-length, and success statistics (including compact
names without `rolling`) use the run-configured
`metrics_episode_window_size`, currently 100. During warm-up they reduce all eligible observations
seen so far; metrics whose meaning requires complete start coverage are withheld until every start
has filled the window. The window size lives once in config rather than being duplicated in every
metric path. `lifetime` and `total` explicitly mean all eligible observations seen so far. Boundary-event
paths retain `rolling` to distinguish window counts from lifetime counts.

A checkpoint continuation preserves the learner's cumulative step counter but creates fresh
episode metric windows and fresh environment episodes. Its initial mean progress and length
therefore describe only newly completed episodes, not the preceding run's last window. Short
episodes can finish first after the reset, transiently biasing the initial window toward shorter,
lower-progress outcomes. A boundary dip alone does not establish lost policy weights; compare
mature windows and the actual optimizer diagnostics. A later plateau also does not identify
resuming as its cause without a matched uninterrupted continuation.

## Research interpretation

- Player event labels show `ep` for the episode number and `step` for the transition
  number within that episode. The internal `sequence` identifies playback transitions
  across episodes and supports inspection navigation. Each executed environment step
  advances `sequence`; an episode boundary resets the next episode's `step` to 1
  while `sequence` continues. These are playback coordinates,
  not event counts or the learner's training step counter.

- Player disk-backed inspection reads original step rewards, cumulative returns,
  and recorded Policy decisions. Seeking does not add samples or recompute Policy
  diagnostics. History charts show a bounded window around the selected step;
  a partial window does not establish a realized full-episode critic return.
  Timeline event dots use a separate bounded episode-wide overview and may group
  nearby events for display; they are not individual scientific metric samples.

- Playback `V(s)` is the critic's expectation of discounted future policy-facing return under the
  checkpoint policy, while realized `G(s)` is one completed trajectory sample from that
  distribution. Exact pointwise agreement on one episode is not expected; assess calibration and
  residual bias across many contract-comparable trajectories without conditioning only on
  successful outcomes. At a selected step, `G(s)` includes only discounted rewards from that step
  onward; it is neither the cumulative whole-episode return nor a success/survival flag. Near a
  finite time limit it can therefore be small even in a successful episode. If remaining time is
  absent from the policy observation, visually similar early and late states are aliased for the
  critic, which can amplify the terminal-horizon discrepancy. Learner explained variance uses its
  rollout value targets and is not the same statistic as a single playback trajectory's
  `V(s) - G(s)`. A true termination closes the return with zero future value. A truncation does not:
  the unobserved continuation prevents a fully realized Monte Carlo `G(s)`, but a
  truncation-compatible N-step diagnostic can close the trace with the exact final policy input's
  critic value: `Ĝ_t = r_t + γ r_{t+1} + … + γ^{T-t} V(s_T)`. That quantity is partly bootstrapped,
  not a fully realized return or generally the learner's GAE lambda-return, and is valid for critic
  comparison only when the final observation, preprocessing and conditioning, checkpoint critic,
  discount, and truncation semantics match the training value contract. Playback labels this
  bootstrap when it computes the diagnostic and withholds the comparison if the exact final-state
  critic value is unavailable; treating truncation as termination by substituting a zero bootstrap
  would fabricate a misleading residual.
- Mario recipes disable automatic checkpoint evaluation and stop when
  `train/target/success/start_rate_min` first reaches one. For a single start,
  that means 100 consecutive genuine target-origin clears; for multiple starts, every configured
  start's latest 100 attempts must all clear. This training stop is not acceptance or promotion;
  explicitly evaluated Mario checkpoints rank by earliest `leader/step`, then highest
  `eval/return_mean`. Breakout is training-only and ranks individual current-contract runs using `train/target/progress/bricks_destroyed/mean`, which
  excludes archive-curriculum origins and non-episode control boundaries; ties prefer higher
  rolling maximum target-origin bricks, then lower rolling mean episode length across all origins.
- Aggregate training `train/target/success/observed_start_rate_lifetime_*` is cumulative. Aggregate
  `train/target/success/start_rate_*` uses the configured recent-episode window and appears only after
  every configured start has filled it. Observed-start aggregates intentionally describe only
  starts attempted so far; the path makes that scope explicit without a duplicate coverage metric.
- A bounded training-only search may use per-start success counts and the history peak and first
  threshold crossing of `train/target/success/start_rate_min` to screen and rank
  recipes.
  That evidence is not checkpoint evaluation and cannot establish checkpoint promotion, goal
  acceptance, or release evidence.
- Failure-reason metrics inspect events on the terminal episode record, not an event history
  accumulated across the episode. They treat each reason as a per-episode presence flag, not an
  occurrence count or necessarily the event that caused termination. The runtime records events
  firing on the final transition, and the reducer does not filter them to declared failure events.
  For each reason, the window rate is the number
  of unsuccessful episodes whose terminal record contains that reason divided by all completed episodes in the latest
  configured recent-episode window. Multiple reasons may belong to one episode, so rates need not
  sum to one; successful episodes contribute zero to every
  failure-reason numerator while remaining in the window denominator.
- ViZDoom's `time_limit_reached` reason is the classified provider-native tic horizon, independent
  of policy frame skip. An unclassified provider truncation uses the fallback reason `timeout`.
  An evaluation watchdog expiry is an execution error and emits no episode or outcome metrics.
- `VizdoomBasic-v1` and `VizdoomBasic-Plus-v1` end on the first physical pistol shot, identified by
  decreasing `ammo2`. A simultaneous `hitcount` increase classifies that episode as success;
  otherwise `shot_fired` classifies it as failure. A requested attack that consumes no ammunition
  is not a shot, and an episode that never fires reaches the ordinary `time_limit_reached` timeout.
- `VizdoomDefendLine-v1` and `VizdoomDefendLine-Plus-v1` classify reaching their 2,100-native-tic
  horizon as success and stop training when
  `train/target/success/start_rate_min` reaches one. Each has one configured
  start, so this requires 100 consecutive horizon-reaching training episodes; it is training
  success evidence, not checkpoint acceptance or promotion.
- Positive actor-critic policy entropy and dominant-action rate diagnose discrete policy collapse.
  Dominant-action rate is the largest empirical frequency among sampled rollout actions, not the
  mean highest action probability at each state. It measures action-use imbalance and does not
  alone establish policy collapse; a deterministic policy can use different actions in different states.
  For a categorical `Discrete(n)` policy, entropy has infimum zero and
  maximum `ln(n)` nats at the uniform distribution; a `MultiDiscrete(nvec)` maximum is
  `sum(ln(nvec))`, while a constrained legal-tuple categorical policy has maximum
  `ln(legal_tuple_count)` regardless of the enclosing `nvec`; a `MultiBinary(d)` maximum is
  `d * ln(2)`. Continuous `Box` policies report differential entropy, which has no finite
  action-space-only minimum or maximum. The bounds remain a pure policy-space calculation used by
  diagnostics and are not duplicated as W&B metrics.
- For custom PPO, `train/ppo/clip_fraction` averages minibatch fractions across attempted epochs and counts sampled action-probability ratios outside the clipping interval. It does not measure how far those ratios moved or the fraction of gradients disabled; similar fractions can accompany different policy changes. `train/ppo/approx_kl` averages minibatch estimates from the last attempted epoch, not a fresh full-rollout evaluation of the final policy. Interpret both alongside learning rate and progress at matched timesteps; neither has a universally desirable target.
- Changing GAE lambda changes both the policy advantage estimator and the critic return targets. Value loss and explained variance across different lambda settings therefore describe different target distributions; larger loss alone does not prove critic divergence or greater advantage variance. Advantage normalization rescales the estimator but does not remove noise in its action ranking. Rollout advantage standard deviation measures dispersion across sampled transitions, not conditional estimator noise at fixed states; differences across runs can reflect changed state occupancy and true action-value variation.
- Actor-critic explained variance is `1 - Var(value_target - value_prediction) /
  Var(value_target)`: one means the residual variance is zero (perfect up to a constant prediction
  offset), zero means the critic explains no more target variance than a constant baseline, and
  negative values are worse than that baseline. Because variance centers the residuals, explained
  variance alone does not detect constant value bias; pair it with residual bias or calibration
  measures when absolute accuracy matters. A near-zero value is not by
  itself evidence of exploding value loss. Pair it with value-prediction and advantage dispersion:
  predictions with much less dispersion than the residual advantages indicate a mostly
  state-insensitive baseline, while tiny target variance can make the ratio ill-conditioned.
  Common causes include partially observed return-relevant state, rapidly changing policy state
  occupancy, shared actor-critic feature drift, and bootstrap-heavy rollouts.
- GradLab does not currently emit separate actor, critic, shared-trunk, pre-clipping, or
  post-clipping gradient norms for SB3 actor-critic updates. SB3 backpropagates one combined policy,
  entropy, and value objective and clips the norm over the complete policy parameter set, so
  `update/policy_gradient_loss` and `update/value_loss` are not proxies for their respective
  gradient magnitudes.
- For reward-transform ablations, first compare `train/reward/pre_transform/*` with
  `train/reward/shaped/*`. `task.reward.reward_scale` is a finite multiplier from zero through one,
  so values below one attenuate the policy-facing reward. If raw rewards match but shaped
  magnitudes diverge, inspect
  value loss and explained variance before policy entropy, dominant-action rate, KL, and clip
  fraction: squared-error value loss can grow roughly with the square of the target scale, while
  advantage normalization does not protect the critic from poorly conditioned targets.
- Do not compare shaped episode-return or value magnitudes as policy quality across different reward
  transforms. Use task success and acceptance-evaluation metrics for the outcome comparison; use
  reward, critic, and policy metrics to locate the causal chain.
- `train/target/return_mean` begins with the first genuine
  target-origin episode and rolls over the configured window. It is an online behavior-policy
  proxy whose episodes may span learner updates, not an estimate of one frozen checkpoint's
  evaluation performance. A threshold condition with `progress_baseline` additionally emits
  `train/early_stop/{condition}/target/progress` as the current metric's clamped fraction from that
  baseline to its threshold. Only goal-owned checkpoint evaluation may establish acceptance.
- `train/all/episode_steps_mean` counts policy transitions; nominal native frames
  per step follow the run's recorded `frame_skip` (two in the ball-state recipe). It combines target- and archive-origin
  episodes and is therefore not a clean survival comparison when archive curricula differ. Within
  identical start, frame-skip, reset, and termination contracts, increasing length can indicate
  better ball defense when target return or progress also improves and failure-reason rates do not;
  it can instead reflect delayed serving, low-progress play, or the episode cap, while faster wall
  completion can shorten a successful episode. Treat it as a diagnostic, not a monotonic success or
  acceptance measure.
- Breakout declares `score` as reward-independent episode progress.
  `train/target/progress/score/mean` is the mean terminal provider-native Atari
  score over the most recent target-origin episodes, including warm-up before the 100-episode
  window is full. GradLab-owned life-loss and serve-stall penalties do not change it. Atari score
  weights brick rows differently, so this metric is not a brick count; it remains an online
  behavior-policy training proxy rather than frozen-checkpoint evaluation evidence.
- Breakout also declares `bricks_destroyed` and `bricks_destroyed_normalized` as
  reward-independent episode progress. The Breakout workspace primary panels show normalized
  bricks destroyed mean, maximum, and minimum as fractions; diagnostics show the absolute mean.
  Their `train/target/progress/{progress}/mean`
  metrics average the terminal cumulative values over the most recent 100 target-origin episodes,
  while `train/target/progress/{progress}/max` and `/min` report the maximum and minimum
  over that same window. Minimum bricks describes the worst episode in the current window,
  including failures; it is not an all-time minimum or a fourth ranking criterion. The normalized value is the `0.0..1.0` two-wall completion fraction. All three statistics
  include warm-up before the window is full and are online behavior-policy training proxies rather
  than frozen-checkpoint evaluation evidence. Breakout ranks runs first by the rolling mean
  terminal `bricks_destroyed` count, then by higher rolling maximum target-origin bricks,
  then by lower rolling mean episode length across all origins. A low rolling minimum can persist
  while the mean improves because a single low-progress episode determines the minimum until it
  leaves the window. For an illustrative fixed policy with independent episodes and probability
  `p` of finishing below a chosen brick threshold, a 100-episode window contains such an episode
  with probability `1 - (1 - p)^100`; even `p = 0.01` gives about 63.4%. Online windows overlap
  and policies change, so this calculation is intuition, not a failure-rate estimate from history.
  The median of logged rolling minima is not an episode percentile, and the fraction of logged
  windows below a threshold is not the fraction of episodes below it. Mean/minimum/maximum
  aggregates alone do not identify episode variance or lower-tail quantiles; use individual
  episode evidence for those measures. Displaying or ranking by a metric does not add that
  metric to the learner's reward objective. Criteria are lexicographic:
  later criteria apply only when earlier values tie, without a tolerance band.
  `leaders runs` applies the complete configured training ranking to individual runs and excludes
  runs missing any criterion. Recipe cohorts retain their separate cross-seed aggregation order;
  the individual-run tie-breaks do not redefine cohort statistics.
  The default `ppo` recipe terminates successfully when the second wall clears. For that
  contract, episode length measures completion time on successful episodes; its rolling mean
  also includes failures, so decreasing length indicates faster completion only when full
  completion remains consistent. Recipes that continue after wall completion would need a
  separate first-clear timing field to measure completion speed. The length tie-break prefers
  shorter episodes but does not independently establish better performance below full completion;
  archive curricula and differing frame skips also limit its comparability.
- Time to a training-progress threshold is elapsed training wall time until the declared rolling target-progress mean reaches that threshold; it is distinct from `train/all/episode_steps_mean`, which measures episode duration in policy steps. Compare both elapsed time and global steps, declare any persistence requirement before comparison, and treat unreached thresholds as unreached rather than estimating completion by extrapolation. W&B `_runtime` is logger runtime and excludes pre-run provisioning; end-to-end comparisons must include launch/setup separately.
- Training episode reduction aggregates return, length, outcome, success, the explicitly supported
  target-origin cell-novelty statistic, and goal-declared numeric episode progress fields.
  Progress field names refer to task-semantic signals and must be populated independently of the
  selected reward shape; `VizdoomDeathmatch-v1` maps task signal `kills` to provider field
  `killcount`, so
  `train/target/progress/kills/mean` reports recent mean native monster frags for
  genuine target-origin training episodes. Native shaped return is not an exact
  substitute because different monster kills can contribute different score values.
  `eval/progress/kills/{mean|max}` remains frozen-checkpoint evaluation evidence.
- Snapshot-curriculum `sampling/probability/max` and `sampling/effective/cell/count` summarize the
  current cell-probability distribution. They do not report realized per-cell selection frequency
  or identify which resident cells were selected.
- Derived throughput phase timing satisfies `loop wall time = provider step time +
  train/throughput/rollout/overhead/seconds + train/throughput/between/rollouts/seconds`. Compare
  those phases on matching workloads to identify a training-loop bottleneck. Rollout overhead includes
  policy inference plus wrapper, buffer, reset, task, and callback work outside the native provider.
  between-rollout time includes optimizer updates, callbacks, and logging, so it is deliberately
  not named optimization time. The corresponding rates are
  `train/throughput/loop/rate` and `train/throughput/provider/step/rate`.
- Reward components are emitted only when configured. Each component mean includes every policy
  transition in the rollout, including zero-valued transitions; nonzero rate separately reports
  the fraction on which the component contributed. Each component also has an absolute share;
  raw reward appears only when it differs from shaped reward. Mario's `progress` component includes
  both its base new-progress reward and any configured additional new-progress reward above
  `progress_reward_boost_start_x`. An identity task's `event` component is the sum of its declared
  signed `event_rewards` for events firing on that transition. Its per-event reward metrics split
  that sum by declared event. An identity `equals` event contributes on every policy transition
  whose post-transition signal matches; `previous_equals` contributes when the signal at the start
  of the transition matches, using the reset signal on the first transition when available;
  `equals_for` contributes only when the consecutive-match counter first reaches its threshold.
  ViZDoom Deathmatch's optional `sample-factory-v0` shape
  exposes `kill`, `death`, `hit`, `damage`, `health`, `armor`, `weapon`, `ammo`, and `weapon_hold`
  components; their sum is the pre-transform task reward and excludes the replaced provider reward.
- The player's protocol-v8 Reward analysis ledger is local playback telemetry, not a W&B metric.
  It shows raw component values, multiplies each impact by the unit-interval reward scale,
  accounts for unattributed raw reward and per-transition clipping, and reports both signed
  contribution (`impact / abs(final reward)`) and absolute transformed activity share. Signed
  contributions preserve penalties, can exceed 100%, and are unavailable at zero final reward;
  this is intentionally different from `train/reward/component/{component}/share` below.
- Under the current `VizdoomDefendCenter-v1` identity-reward contract, every spawned target has one
  health point, its death adds `+1`, the player starts with 52 pistol rounds, and the scenario has no
  ammo replenishment. A normal episode return is therefore `player kills - 1` when the player dies
  and `player kills` when it reaches the native time limit; 52 is the perfect-accuracy ammunition
  ceiling, not a score guaranteed by possessing the ammunition.
- `VizdoomDefendCenter-v1` classifies reaching 52 kills as success and stops training when
  `train/target/success/start_rate_min` reaches one. With its single configured
  start, this requires 100 consecutive perfect-score training episodes; it is training success
  evidence, not checkpoint acceptance or promotion.
- `VizdoomHealthGathering-Plus-v1` is a surface-variant identity over the regular
  `VizdoomHealthGathering-v1` task. Both classify the 2,100-native-tic horizon as success, stop at a
  mature rolling success rate of one, use the same neutral return-plateau fallback stop, and
  require an evaluation success rate of at least 0.95 for acceptance.
- A ViZDoom success-rate target is success-based early stopping only when its `target_reached`
  condition has `action: stop`. Every ViZDoom goal with a binary success event now stops when
  `train/target/success/start_rate_min` reaches one. With the current configured
  window this requires 100
  consecutive successful training episodes for each configured start. For
  `VizdoomDeathmatch-v1`, reaching the 4,200-native-tic horizon is a successful outcome while the
  episode boundary remains truncated so training bootstraps from the final observation. Its single
  configured start therefore stops after 100 consecutive horizon-reaching training episodes. This
  training success evidence does not replace its goal-owned mean-kills checkpoint acceptance rule.
- Episode-return means are neither a best-episode metric nor the score of a currently visible lane:
  they reduce the configured recent episode window across all applicable vector lanes. W&B chart
  smoothing, when enabled, is applied on top of that already-rolling value. Under the root Breakout
  goal contract (`reward_mode: native`, unclipped), shaped episode return is the sum of Atari
  row-score deltas unless a recipe declares additional shaping. The checked-in Breakout `ppo`
  recipe uses event-only reward: one per destroyed brick, minus 0.1 per life-loss event and five
  per serve-stall event, with scale one and clipping disabled. Its shaped episode return is
  therefore distinct from native Atari score. An individual high-return episode can coexist
  with a much lower mean when other lanes finish with lower returns.
- Episode returns, success rates, failure reasons, policy entropy, and optimizer diagnostics
  describe performance or mechanism; no one of them should be treated as a generic stall-stop
  signal. A configured plateau condition may watch any registered numeric training metric, with
  direction, minimum meaningful improvement, warmup, and patience owned by the goal or recipe.
  `train/early_stop/{condition}/*` projects that condition's local state for diagnosis and shadow
  calibration. It means only that the selected metric has not improved under the declared
  condition, not that the task is impossible or that a checkpoint is accepted. Private control-R2
  receipts, never W&B diagnostics, are authoritative for an active early-stop outcome. For an
  evaluated run, the receipt is provisional until already-submitted evaluations settle: acceptance
  overrides the plateau, complete valid rejections establish a neutral stopped attempt, and
  incomplete evaluation evidence remains resumable. New plateau receipts use `outcome: neutral`,
  terminal state `stopped`, and `early_stop_neutral:<condition_id>`; historical immutable
  `failed`/`early_stop_failure:<condition_id>` plateau receipts retain the same neutral diagnostic
  interpretation without being rewritten. Threshold-based failure conditions remain failures.

## Full-evaluation table

`eval/start/table` has one row per start with these columns:

`start_id`, `episode_count`, `success_count`, `success_rate`,
`shaped_return_mean`, and `failure_reasons`.

`failure_reasons` is a structured mapping from reason to episode count. The checkpoint step is
already the row's `eval/checkpoint/step` axis and is therefore not duplicated in the table.

Episode-level evidence stays in R2. Confidence intervals and start-by-reason scalar products
are intentionally computed offline rather than added to W&B history.

An acceptance contract may reject fail-fast only when the first failed outcome proves its rule
cannot pass. That rejection is complete evidence of failure, but not a complete 100-episode
evaluation, so it emits no partial evaluation aggregates. Aggregate contracts such as mean return
disable outcome-based fail-fast, run every planned episode, and emit complete evaluation aggregates
for either verdict. W&B history always receives `eval/checkpoint/step`, pass, and
planned/completed episodes. Complete full-evaluation projections additionally include
`eval/start/table`. Per-start success and failure-reason summaries are derived from immutable
private-R2 episode rows. Duration, artifact, source, and raw failure details stay in typed result,
evidence, or checkpoint metadata rather than being duplicated in the W&B-shaped metric map.

`eval/acceptance/pass` is per-checkpoint history. W&B summarizes that history with `max`, so the
summary means that some checkpoint passed; it is not the run verdict. The authoritative verdict is
the create-only private-R2 `PromotionReceipt`, whose selected result is hash-bound to the complete
acceptance evidence. At terminal publication, that receipt projects
`ops/terminal/*`, the diagnostic `leader/*` fields, and the accepted
W&B projection. Leader fields mirror only finite
selected metrics required by the configured rank plus available diagnostics; no generic objective,
serialized rank tuple, constant acceptance alias, or fabricated default is emitted. Later rejected
checkpoint projections remain in history and never modify the active projection. Raw acceptance
aggregates and episode evidence remain authoritative in private eval R2.

## Delivery, backpressure, and recovery

Every event has a stable content-derived internal event ID. Delivery to W&B is at least once; the
durable `ops/event_sequence` is also W&B's internal step, so replay after an interrupted
local acknowledgement cannot append a second scientific point. The event ID remains a transport
invariant and is not duplicated as a W&B metric. Promotion, terminal state, and early-stop
authority are exactly once through conditional private-R2 receipts.

Attempt-receipt fields `drain.metric_segment_high_water` and
`drain.wandb_remote_high_water_mark` both contain highest event-sequence IDs despite the former's
historical name. Their difference counts events not yet remotely visible at receipt creation, not
metric-segment objects. Because an attempt receipt is immutable, W&B may later catch up after a
failure receipt without changing the recorded drain completeness or terminal state.

Historical publication imports may display retired, source-bound metric names from their immutable
evaluation contracts. Those names are evidence labels only: they are not current emitted metrics,
registry aliases, or permission to translate a historical acceptance rule to a similar current
metric.

The supervisor seals immutable metric-journal segments to private R2 every five seconds or 1,000
events and batches pending frames to W&B. A retry reconstructs its local SQLite state from those
segments before producing new events. It resumes the same W&B run with `resume="must"`.

Backpressure is sampled every 15 seconds. W&B receives pending outbox count, oldest unpublished
age, remote-visible lag, pending checkpoints, pending evaluations, scratch utilization, and
post-learner idle-GPU time. Ingress, publication capacity, durable high-water marks, and
accepted-result-to-stop timing remain transport invariants or receipt evidence rather than
duplicated public metrics.

Unpublished W&B age warns at 45 seconds and is unhealthy at 60 seconds. Evaluation drain is governed
by the declared per-attempt expiry windows. The 300-second terminal delivery deadline begins only
after evaluations settle and covers checkpoint and local W&B delivery. If neither W&B nor private
R2 can preserve pending metrics, or task scratch usage reaches 80%, the supervisor requests a safe
learner stop and emits a resumable failure rather than discarding evidence.

A logical run succeeds only when its private-R2 `TerminalReceipt` proves the complete checkpoint
inventory, the terminal inventory of automatically submitted evaluations, a promotion, the W&B
high-water mark, and a complete drain. Checkpoints published after acceptance may remain
unevaluated for future explicit user action. dstack process exit alone is never scientific success.

## Registry

<!-- METRIC_REGISTRY_START -->
| Metric or template | Display label | Meaning | Unit | Cadence | Placement | Summary | Axis | Evidence | Leader | Training proxy |
|---|---|---|---|---|---|---|---|---|---|---|
| `train/target/return_mean` | Recent target return mean | Mean shaped return over the most recent genuine target-origin episodes, including warm-up before the configured window is full. | return | rollout | history | last | train/global_step | training | - | - |
| `train/target/return_max` | Recent target return max | Maximum shaped return over the same recent target-origin episodes as the rolling mean. | return | rollout | history | last | train/global_step | training | - | - |
| `train/all/episode_steps_mean` | Recent episode length mean | Mean policy-transition count over the most recent genuine completed episodes across target and archive origins. | steps | rollout | history | last | train/global_step | training | - | - |
| `train/target/unique_cells_mean` | Recent target unique cells mean | Mean episodic unique-cell count over recent target-origin episodes when cell-novelty shaping is active; the reset cell is included. | cells | rollout | history | last | train/global_step | training | - | - |
| `train/target/progress/{progress}/mean` | Recent target {progress} mean | Mean of a goal-declared finite numeric progress field over recent genuine target-origin episodes. | value | rollout | history | last | train/global_step | training | - | - |
| `train/target/progress/{progress}/max` | Recent target {progress} max | Maximum of a goal-declared finite numeric progress field over recent genuine target-origin episodes. | value | rollout | history | last | train/global_step | training | - | - |
| `train/target/progress/{progress}/min` | Recent target {progress} min | Minimum of a goal-declared finite numeric progress field over recent genuine target-origin episodes. | value | rollout | history | last | train/global_step | training | - | - |
| `train/all/episodes_total` | Completed episodes | Cumulative genuine completed training episodes across origins. | episodes | rollout | history | last | train/global_step | training | - | - |
| `train/all/boundary_event/{reason}/count` | Failure {reason} count | Cumulative unsuccessful completed episodes whose terminal record contains the reason; reason presence is counted at most once per episode. | episodes | rollout | history | last | train/global_step | training | - | - |
| `train/all/boundary_event/{reason}/rolling/count` | Recent failure {reason} count | Unsuccessful completed episodes whose terminal record contains the reason among the most recent 100 genuine completed episodes, including warm-up before the window is full; reason presence is counted at most once per episode. | episodes | rollout | history | last | train/global_step | training | - | - |
| `train/all/boundary_event/{reason}/rolling/rate` | Recent failure {reason} rate | Recent unsuccessful completed episodes whose terminal record contains the reason divided by all recent completed episodes; reason presence is boolean per episode. | fraction | rollout | history | last | train/global_step | training | - | - |
| `train/target/success/by_start/{start}/episodes_total` | Successful target episodes from {start} | Cumulative successful genuine target-origin episodes from one start. | episodes | rollout | history | last | train/global_step | training | - | - |
| `train/target/success/by_start/{start}/rate` | Recent success rate from {start} | Success fraction over recent genuine target-origin attempts from one start. | fraction | rollout | history | last | train/global_step | training | - | - |
| `train/target/success/observed_start_rate_lifetime_min` | Observed-start success rate min | Minimum cumulative target-origin success rate across starts with at least one genuine attempt. | fraction | rollout | history | last | train/global_step | training | - | - |
| `train/target/success/observed_start_rate_lifetime_mean` | Observed-start success rate mean | Mean cumulative target-origin success rate across starts with at least one genuine attempt. | fraction | rollout | history | last | train/global_step | training | - | - |
| `train/target/success/start_rate_min` | Recent all-start success rate min | Minimum recent target-origin success rate, emitted after every configured start fills the episode window. | fraction | rollout | history | last | train/global_step | training | - | - |
| `train/target/success/start_rate_mean` | Recent all-start success rate mean | Mean recent target-origin success rate, emitted after every configured start fills the episode window. | fraction | rollout | history | last | train/global_step | training | - | - |
| `train/early_stop/{condition}/patience/progress` | Early-stop {condition} patience | Policy-step patience progress capped at one; one means the condition would trigger. For `no_improvement` conditions, progress measures steps since the later of eligibility and the last qualifying improvement, so each qualifying improvement resets the patience clock. | fraction | watched metric sample | history | last | train/global_step | training | - | - |
| `train/early_stop/{condition}/target/progress` | Early-stop {condition} target | Threshold progress from the declared baseline to the target in the improving direction, clamped to zero through one. | fraction | watched metric sample | history | last | train/global_step | training | - | - |
| `train/reward/shaped/mean` | Shaped reward mean | Mean learner-facing per-step reward after gradlab scaling and clipping. | scalar | rollout | history | last | train/global_step | training | - | - |
| `train/reward/shaped/std` | Shaped reward std | Standard deviation of learner-facing per-step reward after gradlab scaling and clipping. | scalar | rollout | history | last | train/global_step | training | - | - |
| `train/reward/shaped/nonzero/rate` | Shaped nonzero reward rate | Fraction of learner-facing per-step rewards that are nonzero after scaling and clipping. | fraction | rollout | history | last | train/global_step | training | - | - |
| `train/reward/pre_transform/mean` | Raw reward mean | Mean completed task reward immediately before gradlab-owned scaling and clipping, emitted when distinct from shaped reward. | scalar | rollout | history | last | train/global_step | training | - | - |
| `train/reward/pre_transform/std` | Raw reward std | Standard deviation of completed task reward immediately before gradlab-owned scaling and clipping, emitted when distinct from shaped reward. | scalar | rollout | history | last | train/global_step | training | - | - |
| `train/reward/component/{component}/mean` | Reward {component} mean | Mean configured reward-component contribution over every policy transition in the rollout, including zero-valued transitions, in pre-transform task-reward units. In identity `reward_mode: events`, native score contributes zero and the active `event` component sums fixed and delta rewards; provider-native score remains available as episode progress. | scalar | rollout | history | last | train/global_step | training | - | - |
| `train/reward/component/{component}/nonzero/rate` | Reward {component} activity rate | Fraction of active reward-component values that are nonzero. | fraction | rollout | history | last | train/global_step | training | - | - |
| `train/reward/component/{component}/share` | Reward {component} share | Absolute contribution share computed from components in pre-transform task-reward units. | fraction | rollout | history | last | train/global_step | training | - | - |
| `train/reward/event/{event}/mean` | Reward event {event} mean | Mean contribution from one declared event reward in pre-transform task-reward units, including zero-valued transitions. Fixed event rewards pay the coefficient once per firing; `event_delta_rewards` pay it times the absolute signal change on a declared increase/decrease event. | scalar | rollout | history | last | train/global_step | training | - | - |
| `train/reward/event/{event}/nonzero/rate` | Reward event {event} activity rate | Fraction of policy transitions on which one declared event reward contributes a nonzero value; multiple units in one delta-reward transition count as one active transition. | fraction | rollout | history | last | train/global_step | training | - | - |
| `train/ppo/approx_kl` | PPO approximate KL | Approximate KL divergence for the PPO update. | scalar | rollout | history | last | train/global_step | training | - | - |
| `train/ppo/clip_fraction` | PPO clip fraction | Fraction of sampled policy ratios outside PPO's clipping interval. | fraction | rollout | history | last | train/global_step | training | - | - |
| `train/jerk/retained/count` | JERK retained sequences | Distinct action sequences retained by JERK search. | sequences | rollout | history | last | train/global_step | training | - | - |
| `train/jerk/best/return/mean` | JERK best return mean | Mean observed return of JERK's highest-ranked retained sequence. | return | rollout | history | last | train/global_step | training | - | - |
| `train/jerk/best/program/steps` | JERK best program steps | Action length of JERK's highest-ranked retained sequence. | steps | rollout | history | last | train/global_step | training | - | - |
| `train/go-explore/archive/cell/count` | Go-Explore archive cells | Semantic cells currently retained by Go-Explore. | cells | interval | history | last | train/global_step | training | - | - |
| `train/go-explore/archive/blob/bytes` | Go-Explore archive bytes | Uncompressed bytes in distinct retained provider-state blobs. | bytes | interval | history | last | train/global_step | training | - | - |
| `train/go-explore/archive/visit/count` | Go-Explore archive visits | Cumulative semantic-cell visits. | visits | interval | history | last | train/global_step | training | - | - |
| `train/go-explore/archive/cell/discovery/rate` | Go-Explore cell discovery rate | New semantic cells divided by visits in the bounded recent visit window. | fraction | interval | history | last | train/global_step | training | - | - |
| `train/go-explore/best/progress` | Go-Explore best progress | Greatest task progress reached by the best retained trajectory. | value | interval | history | last | train/global_step | training | - | - |
| `train/go-explore/best/return` | Go-Explore best return | Shaped return of the best retained trajectory. | return | interval | history | last | train/global_step | training | - | - |
| `train/go-explore/best/program/steps` | Go-Explore best program steps | Environment steps in the best retained action program. | steps | interval | history | last | train/global_step | training | - | - |
| `train/{algorithm}/explained_variance` | {algorithm} explained variance | Actor-critic value-function explained variance. | scalar | rollout | history | last | train/global_step | training | - | - |
| `train/{algorithm}/policy_loss` | {algorithm} policy-gradient loss | Actor-critic policy-gradient loss. | scalar | rollout | history | last | train/global_step | training | - | - |
| `train/{algorithm}/value_loss` | {algorithm} value loss | Actor-critic value loss. | scalar | rollout | history | last | train/global_step | training | - | - |
| `train/{algorithm}/learning_rate` | {algorithm} learning rate | Current actor-critic learning rate. | scalar | rollout | history | last | train/global_step | training | - | - |
| `train/{algorithm}/entropy` | {algorithm} policy entropy | Positive actor-critic policy entropy. | nats | rollout | history | last | train/global_step | training | - | - |
| `train/{algorithm}/action_std` | {algorithm} policy distribution std | Continuous-action distribution standard deviation. | scalar | rollout | history | last | train/global_step | training | - | - |
| `train/{algorithm}/dominant_action_rate` | {algorithm} dominant action rate | Largest empirical frequency among sampled discrete rollout actions; not a per-state maximum policy probability. | fraction | rollout | history | last | train/global_step | training | - | - |
| `train/{algorithm}/rollout_value/mean` | {algorithm} value prediction mean | Mean rollout value prediction. | scalar | rollout | history | last | train/global_step | training | - | - |
| `train/{algorithm}/rollout_value/std` | {algorithm} value prediction std | Standard deviation of rollout value predictions. | scalar | rollout | history | last | train/global_step | training | - | - |
| `train/{algorithm}/rollout_advantage/mean` | {algorithm} advantage mean | Mean rollout advantage. | scalar | rollout | history | last | train/global_step | training | - | - |
| `train/{algorithm}/rollout_advantage/std` | {algorithm} advantage std | Standard deviation of rollout advantages. | scalar | rollout | history | last | train/global_step | training | - | - |
| `train/throughput/loop/rate` | Training loop throughput | Policy transitions divided by rollout-start-to-next-rollout-start wall time. | transitions/second | rollout | history | last | train/global_step | training | - | - |
| `train/throughput/provider/step/rate` | Provider step throughput | Policy transitions divided by native-provider step wall time, when native timing is available. | transitions/second | rollout | history | last | train/global_step | training | - | - |
| `train/throughput/rollout/overhead/seconds` | Rollout overhead | Rollout wall time outside native-provider step calls. | seconds | rollout | history | last | train/global_step | training | - | - |
| `train/throughput/between/rollouts/seconds` | Between-rollout time | Wall time after rollout collection and before the next rollout, including updates, callbacks, and logging. | seconds | rollout | history | last | train/global_step | training | - | - |
| `train/artifact/save/seconds` | Model save time | Local model save duration. | seconds | artifact | history | last | train/global_step | training | - | - |
| `eval/return_mean` | Full-eval return mean | Mean shaped return across completed full-evaluation episodes. | return | evaluation | history | last | eval/checkpoint/step | evaluation | leader/return_mean | train/target/return_mean |
| `eval/return_max` | Full-eval return max | Maximum shaped return across completed full-evaluation episodes. | return | evaluation | history | last | eval/checkpoint/step | evaluation | leader/return_max | train/target/return_max |
| `eval/success/start_rate_min` | Full-eval start success rate min | Minimum success rate across represented evaluation starts. | fraction | evaluation | history | last | eval/checkpoint/step | evaluation | leader/success/start_rate_min | train/target/success/start_rate_min |
| `eval/success/start_rate_mean` | Full-eval start success rate mean | Mean success rate across represented evaluation starts. | fraction | evaluation | history | last | eval/checkpoint/step | evaluation | - | train/target/success/start_rate_mean |
| `eval/progress/{progress}/mean` | Full-eval {progress} mean | Mean goal-declared progress value across completed full-evaluation episodes. | value | evaluation | history | last | eval/checkpoint/step | evaluation | leader/progress/{progress}/mean | train/target/progress/{progress}/mean |
| `eval/progress/{progress}/max` | Full-eval {progress} max | Maximum goal-declared progress value across completed full-evaluation episodes. | value | evaluation | history | last | eval/checkpoint/step | evaluation | leader/progress/{progress}/max | - |
| `eval/acceptance/pass` | Acceptance pass | Per-checkpoint acceptance result; its W&B history summary uses max and is not the terminal run verdict. | boolean | acceptance evaluation | history | max | eval/checkpoint/step | acceptance | - | - |
| `eval/acceptance/episode/planned/count` | Acceptance episodes planned | Exact episode identities required by the acceptance manifest. | episodes | acceptance evaluation | history | last | eval/checkpoint/step | acceptance | - | - |
| `eval/acceptance/episode/completed/count` | Acceptance episodes completed | Valid planned episode rows completed before acceptance or fail-fast rejection. | episodes | acceptance evaluation | history | last | eval/checkpoint/step | acceptance | - | - |
| `eval/start/table` | Full-eval evidence by start | Structured full-evaluation evidence by start, including success, return, and failure-reason aggregates. | table | evaluation | history | none | eval/checkpoint/step | evaluation_table | - | - |
| `leader/success/start_rate_min` | Leader start success rate min | Selected-checkpoint projection of minimum success rate across starts. | fraction | selection | summary | none | - | selection | - | - |
| `leader/return_mean` | Leader return mean | Selected-checkpoint mean shaped episode return. | return | selection | summary | none | - | selection | - | - |
| `leader/return_max` | Leader return max | Selected-checkpoint maximum shaped episode return. | return | selection | summary | none | - | selection | - | - |
| `leader/progress/{progress}/mean` | Leader {progress} mean | Selected-checkpoint mean for one named progress dimension. | value | selection | summary | none | - | selection | - | - |
| `leader/progress/{progress}/max` | Leader {progress} max | Selected-checkpoint maximum for one named progress dimension. | value | selection | summary | none | - | selection | - | - |
| `leader/step` | Leader checkpoint step | Selected checkpoint policy step. | steps | selection | summary | none | - | selection | leader/step | - |
| `leader/artifact/ref` | Leader artifact | Selected checkpoint immutable artifact reference. | metadata | selection | summary | none | - | selection | - | - |
| `leader/evaluation/source` | Leader evaluation source | Selected checkpoint evaluation source. | text | selection | summary | none | - | selection | - | - |
| `leader/projection/timestamp` | Leader projection time | Selected checkpoint projection update timestamp. | timestamp | selection | summary | none | - | selection | - | - |
| `train/global_step` | Training global step | Scientific training X-axis: policy environment transitions consumed. | steps | frame | history | max | train/global_step | training | - | - |
| `eval/checkpoint/step` | Evaluation checkpoint step | Scientific evaluation X-axis: step of the evaluated checkpoint. | steps | evaluation | history | max | eval/checkpoint/step | acceptance | - | - |
| `ops/event_sequence` | Orchestration event sequence | Monotonic local outbox event sequence used as W&B delivery order. | events | frame | history | max | ops/event_sequence | operational | - | - |
| `ops/outbox/pending` | Pending outbox frames | Metric outbox frames not yet acknowledged by the W&B SDK. | events | supervisor sample | history | last | ops/event_sequence | operational | - | - |
| `ops/outbox/oldest_age_seconds` | Oldest unpublished age | Age of the oldest metric frame not yet acknowledged by the W&B SDK. | seconds | supervisor sample | history | last | ops/event_sequence | operational | - | - |
| `ops/outbox/visibility_lag_seconds` | Remote visibility lag | Age of the newest local metric event not yet observed through the W&B API. | seconds | remote visibility probe | history | last | ops/event_sequence | operational | - | - |
| `ops/checkpoints_pending` | Pending checkpoints | Ready local checkpoints not yet verified in public model R2. | checkpoints | supervisor sample | history | last | ops/event_sequence | operational | - | - |
| `ops/evals_pending` | Pending evaluations | Persisted evaluation intents pending submission or a verified result; intents deferred after acceptance are excluded. | evaluations | supervisor sample | history | last | ops/event_sequence | operational | - | - |
| `ops/drain/gpu_idle_seconds` | GPU idle drain time | Time the training container retained its GPU after the learner exited. | seconds | terminal drain | history | last | ops/event_sequence | operational | - | - |
| `ops/scratch/used_fraction` | Scratch used | Fraction of the task scratch filesystem currently used. | fraction | supervisor sample | history | last | ops/event_sequence | operational | - | - |
| `ops/terminal/state` | Terminal run state | Receipt-backed terminal run state. | text | terminal receipt | summary | none | ops/event_sequence | operational | - | - |
| `ops/terminal/reason` | Terminal run reason | Receipt-backed terminal reason when the run did not succeed. | text | terminal receipt | summary | none | ops/event_sequence | operational | - | - |
| `train/curriculum/archive/cell/count` | Curriculum archive cells | Current archive-curriculum cell count. | cells | rollout | history | last | train/global_step | training | - | - |
| `train/curriculum/archive/entry/count` | Curriculum archive entries | Current immutable entry count retained by the curriculum view. | entries | rollout | history | last | train/global_step | training | - | - |
| `train/curriculum/archive/admission/candidate/count` | Curriculum admission candidates | Non-terminal cell-crossing candidates observed during the rollout. | transitions | rollout | history | last | train/global_step | training | - | - |
| `train/curriculum/archive/admission/accepted/count` | Curriculum admissions accepted | Candidate entries accepted into cell reservoirs during the rollout. | entries | rollout | history | last | train/global_step | training | - | - |
| `train/curriculum/archive/evicted/count` | Curriculum cells evicted | Curriculum cells evicted during the rollout. | cells | rollout | history | last | train/global_step | training | - | - |
| `train/curriculum/archive/capture/call/count` | Curriculum capture calls | Batched portable provider-state capture calls during the rollout. | calls | rollout | history | last | train/global_step | training | - | - |
| `train/curriculum/archive/restore/episode/count` | Curriculum restore episodes | Archive-origin episodes started during the rollout. | episodes | rollout | history | last | train/global_step | training | - | - |
| `train/curriculum/archive/restore/forced_boundary/count` | Curriculum forced boundaries | Non-episode control truncations used to activate archive lanes. | boundaries | rollout | history | last | train/global_step | training | - | - |
| `train/curriculum/archive/feedback/trajectory/count` | Curriculum feedback trajectories | Completed archive-origin trajectories committed to the priority sampler. | trajectories | rollout | history | last | train/global_step | training | - | - |
| `train/curriculum/archive/transition/share` | Curriculum transition share | Fraction of policy transitions whose origin is the archive curriculum. | fraction | rollout | history | last | train/global_step | training | - | - |
| `train/curriculum/archive/sampling/probability/max` | Curriculum max sampling probability | Largest final cell probability in the archive sampler. | fraction | rollout | history | last | train/global_step | training | - | - |
| `train/curriculum/archive/sampling/effective/cell/count` | Curriculum effective cell count | Inverse-Simpson effective cell count of the archive sampling distribution. | cells | rollout | history | last | train/global_step | training | - | - |
| `train/curriculum/archive/capture/seconds` | Curriculum capture time | Portable state capture wall time accumulated during the rollout. | seconds | rollout | history | last | train/global_step | training | - | - |
| `train/curriculum/archive/restore/seconds` | Curriculum restore time | Provider restore wall time for reset calls containing archive lanes. | seconds | rollout | history | last | train/global_step | training | - | - |
<!-- METRIC_REGISTRY_END -->

## Registry relationships and dashboard applicability

The registry owns each metric's axis, evidence category, leader projection, and optional training
proxy. A training proxy remains training evidence. Progress proxy mappings require that progress
field to be declared by the run; names never infer scientific equivalence.

Workspace profiles with `primary_metrics: goal_rank` generate their primary history charts in goal
ranking order, omitting summary-only criteria and axis fields. Goals sharing a project must agree on
that order; conflicting orders require an explicit profile. Secondary charts are filtered against
the union of resolved recipe capabilities and omit duplicate primary series. This removes inactive
algorithm, success, reward-event, progress, and curriculum charts without hiding required goal
metrics. Applicability does not guarantee a value: window maturity and runtime-dependent diagnostic
availability still determine emission.

Schema v21 shortens names without changing populations or reduction semantics. No old-name aliases
are emitted. Managed workspaces filter to the current schema; historical runs keep their original
names and immutable evidence. Reward statistics use streaming moments, and episode snapshots are
reused until a new episode arrives; both retain their existing emission cadence.

Breakout explicitly pins mean bricks, maximum bricks, minimum bricks, and mean episode steps
in that display order. This four-chart display is independent of its three-criterion goal ranking;
minimum bricks remains diagnostic and does not participate in ranking.
