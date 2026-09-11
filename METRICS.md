# Metrics schema v22

This file is the source of truth for gradlab telemetry. The Python registry loads the table below
and requires every emitted metric to match an exact registry entry or a bounded template.

`Training Success` is a provisional Run classification produced by a Research Goal's declared
threshold over training-time metrics. It is a cheap proxy used to classify and compare Runs; it
does not establish that a Research Goal is solved. `Acceptance` is the separate determination from
an Evaluated Goal's stricter checkpoint evaluation on seeds different from the training seeds.
Only Acceptance can authorize Promotion, and neither metric delivery nor its W&B projection is the
authority for that decision.

## Surfaces and dimensions

- The game panel's Render FPS, Decode ms, and Draw ms are browser-local diagnostics,
  never emitted to W&B. Render FPS counts changed canvases at most once per animation
  refresh divided by the actual elapsed sampling time, updated about once per second.
  It includes playback pacing and frame delivery delays and falls to zero when frames
  stop. The chart and min–max range retain the latest 60 samples, with a vertical
  scale from zero to at least 60 FPS. Decode and Draw appear in the tooltip and report mean image-decode and synchronous canvas-update time
  per drawn frame in that interval, including draws coalesced before a refresh.
  They exclude server work, transport, other panels, and GPU presentation. Unavailable
  frames, session resets, and tab visibility changes clear the measurements.
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
  evidence. Training-proxy columns sample W&B history at the latest `train/step` no greater
  than the checkpoint step, but only after W&B's metrics schema, selection rank, and checkpoint
  acceptance contract match the immutable recipe. Any contract mismatch suppresses all optional
  W&B enrichment and surfaces a warning rather than displaying potentially misbound proxy values.
  When `checkpoint_eval_backend` is `none`, the supervisor intentionally omits
  `checkpoint_eval_contract`; the catalog must validate that expected absence against the immutable
  recipe and W&B run dimensions without suppressing otherwise compatible training-proxy history.
  Full-evaluation columns remain unavailable until verified checkpoint-evaluation evidence exists.
- W&B config contains run-defining dimensions: `metrics_schema_version: 22`,
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
- `ops/state` and `ops/reason` are W&B summary-only
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
is intentionally not emitted; `train/curriculum/feedback/count` reports only how
many such trajectory updates were committed.

An episode metric is a **return**. `reward` is reserved for per-step shaping and component
attribution. Frame skip remains run config. W&B uses three explicit axes:

- `train/step`: policy environment transitions consumed by training.
- `eval/step`: step of the checkpoint represented by an evaluation row.
- `ops/sequence`: durable supervisor delivery order.

Across frame-skip ablations, equal `train/step` means equal policy-transition counts,
not equal simulated game time. Nominal native-frame exposure is `train/step * frame_skip`;
this estimate excludes reset work and may differ from actual frames when action repeats end early.
Similar learning curves on this axis establish similar observed policy-transition efficiency,
not native-frame efficiency or wall-clock efficiency.

Each axis is configured with a W&B `max` summary reducer. W&B's public API may therefore expose
its summary value as a reducer mapping such as `{"max": 5046272}` rather than as a bare number.
Catalog and report consumers must unwrap the configured reducer value; a recipe's requested
`timesteps` cap is not a substitute for the observed `train/step`.
Before its first history row is logged, every concrete metric is explicitly bound to its applicable
scientific axis. W&B's internal `Step` is delivery order only and must not become the default X-axis
for scientific charts.

Asynchronous evaluations may arrive after later training rows without changing their scientific
X-axis. Each producer writes only its applicable scientific axis; durable delivery order uses
`ops/sequence`.

Current runs declare schema v22, and the supervisor validates and emits only v22 names. GradLab
does not read, project, or preserve noncurrent W&B or R2 schemas.

Recent training return, progress, episode-length, and success statistics (including compact
names without `rolling`) use the run-configured
`metrics_episode_window_size`, currently 100. Return, progress, episode-length, and boundary-event
statistics reduce eligible observations during warm-up. A per-start recent success rate is withheld
until that start fills its window; all-start recent success aggregates are withheld until every
configured start fills its window. The window size lives once in config rather than being duplicated in every
metric path. `train/episodes/count` and per-start successful episode counts accumulate since reducer initialization.
Recent unsuccessful-event fractions retain one representation; counts can be reconstructed from the
same sample using `round(fraction * min(episodes_count, window_size))`.

A checkpoint continuation preserves the learner's cumulative step counter but creates fresh
episode metric windows and fresh environment episodes. The episode reducer's cumulative episode
and success counters also start fresh; these counts do not imply that
loading a model checkpoint reconstructs counts from the preceding learner execution. Its initial mean progress and length
therefore describe only newly completed episodes, not the preceding run's last window. Short
episodes can finish first after the reset, transiently biasing the initial window toward shorter,
lower-progress outcomes. A boundary dip alone does not establish lost policy weights; compare
mature windows and the actual optimizer diagnostics. A later plateau also does not identify
resuming as its cause without a matched uninterrupted continuation.

## Research interpretation

- A first rolling maximum of normalized bricks equal to 1 means at least one episode in
  that window completed both walls; it is not maximum shaped return or reliable completion.
  Its proximity to a learning-rate or entropy-coefficient schedule endpoint does not establish
  causation: training exposure and both coefficients change together. Compare matched-seed
  runs changing one schedule at a time, using mean progress and episode completion frequency
  alongside first-hit timing. An entropy coefficient controls regularization strength, not
  the policy's measured entropy directly.

- Player event labels show `ep` for the episode number and `step` for the transition
  number within that episode. The internal `sequence` identifies playback transitions
  across episodes and supports inspection navigation. Each executed environment step
  advances `sequence`; an episode boundary resets the next episode's `step` to 1
  while `sequence` continues. These are playback coordinates,
  not event counts or the learner's training step counter.

- Player disk-backed inspection reads original step rewards, cumulative returns,
  and recorded Policy decisions. Seeking does not add samples or recompute Policy
  diagnostics. History charts default to the full recorded episode with bounded,
  extrema-preserving display samples. Drag zoom selects a shared episode-step range
  across history charts, shown on the playbar; seeking does not change that range.
  Exact inspection and action frequencies still use their own recorded windows;
  chart downsampling must not supply samples for scientific calculations.
  The full-episode recorded chart refresh is throttled to roughly once per second.
  Between refreshes, charts append bounded streamed transitions through the presented
  playback step, respecting the selected episode and chart window without extra requests.
  Changing the session, episode, or chart window clears obsolete plots until the new
  history loads. Refreshing the same selection preserves valid recorded samples.
  Loading, recovery, and error status are shared by affected panels in each window;
  they are display state, not scientific measures. Recovery retries transient failures
  after 1, 2, and 4 seconds even while paused, then requires explicit Retry, a new
  selection, or restored visibility. Hidden or suspended chart panels do not fetch.
  A line chart legend reports the hovered point while the pointer is over the chart;
  otherwise it follows the selected playback transition. In inspection, that transition
  can precede the end of the recorded curve, so its legend need not equal the last value.
  The independent Reward table follows the shared cursor and chart window. Its bounded rows
  start at the explicitly selected return reference; seeking changes the row highlight without
  moving that reference. Delay is the row step minus the reference step, weight is `gamma^delay`,
  and contribution is that weight times the recorded shaped reward. The table's `G(s)` and `V(s)`
  retain their per-row state semantics; the contribution is not a causal attribution.
  A partial window does not establish a realized full-episode critic return.
  Timeline event dots use a separate bounded episode-wide overview and may group
  nearby events for display; they are not individual scientific metric samples.

- Playback `V(s)` is the critic's expectation of discounted future policy-facing return under the
  checkpoint policy, while realized `G(s)` is one completed trajectory sample from that
  distribution. Its units follow the training reward, not raw score or undiscounted remaining
  bricks. For example, with +1 per brick and gamma 0.99 per policy transition, a brick reward
  discounted by 100 transitions contributes about 0.366, and by 300 about 0.049; many distant
  bricks can therefore coexist with a small value estimate even for a strong policy.
  Exact pointwise agreement on one episode is not expected; assess calibration and
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
- Playback's Action decision panel separates the Policy's selected action and its probability
  from `transition.effective_action`, the action after conditional overrides expressed in the
  Policy action space. It labels that effective action as “Environment received” using the recorded
  Policy action semantics and shows `action_override_rule_id` when present. The provider-native
  encoding is recorded separately as `native_action`; it must not be decoded as a Policy index.
  Missing effective-action evidence remains unavailable rather than falling back to the selected
  action or pre-override `executed_action`. These are local transition diagnostics, not W&B metrics.
- Playback reward analysis “Episode to cursor” sums steps 1 through the selected step using
  recorded cumulative accounting, independent of the bounded inspection window. Positive and
  negative activity and per-component absolute activity accumulate before cancellation; scaling,
  unattributed task reward, and clip adjustments reconcile to the recorded shaped episode return.
  A missing prefix, accounting error, or cursor mismatch remains explicitly unavailable.
- Playback action frequencies use the selected episode's trailing 64 transitions through the
  cursor (or steps 1 through the cursor for shorter prefixes), with step range and sample counts
  displayed. “STEP” is the selected decision's probability distribution. “POLICY” counts recorded
  Policy choices on Policy-driven transitions, excluding human input. “ENV” counts recorded
  effective actions after overrides in Policy action space, including human input. Missing actions
  or an incomplete window withhold the affected frequencies; future steps and other episodes are
  excluded. These inspection statistics are local diagnostics, not W&B metrics.
- Mario recipes disable automatic checkpoint evaluation and stop when
  `train/success/min` first reaches one. For a single start,
  that means 100 consecutive genuine target-origin clears; for multiple starts, every configured
  start's latest 100 attempts must all clear. This training stop is not acceptance or promotion;
  explicitly evaluated Mario checkpoints rank by earliest `leader/step`, then highest
  `eval/return/mean`. Breakout is training-only and ranks individual current-contract runs using `train/progress/bricks_destroyed/mean`, which
  excludes archive-curriculum origins and non-episode control boundaries; ties prefer higher
  rolling maximum target-origin bricks, then lower rolling mean episode length across all origins.
- Recent training `train/success/min` and `train/success/mean` reduce success fractions across
  configured starts only after every start fills its window. For one start, publish the minimum
  and cumulative successful episode count; suppress duplicate mean and per-start recent fraction
  unless a scientific selector explicitly requires them. Multi-start runs retain all three views.
  Cumulative success mean across attempted starts remains a local completion display during warm-up,
  without a W&B history series. The same single-start mean suppression applies to evaluation history;
  authoritative evaluation aggregates still retain the exact values needed by acceptance rules.
- A bounded training-only search may use per-start success counts and the history peak and first
  threshold crossing of `train/success/min` to screen and rank
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
  `train/success/min` reaches one. Each has one configured
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
- For custom PPO, `train/clip/fraction` averages minibatch fractions across attempted epochs and counts sampled action-probability ratios outside the clipping interval. It does not measure how far those ratios moved or the fraction of gradients disabled; similar fractions can accompany different policy changes. `train/kl/mean` averages minibatch estimates from the last attempted epoch, not a fresh full-rollout evaluation of the final policy. Interpret both alongside learning rate and progress at matched timesteps; neither has a universally desirable target.
- Custom PPO checks each minibatch's approximate KL against `1.5 * target_kl` before its optimizer step. Exceeding the threshold stops remaining optimization on the collected rollout, retaining earlier optimizer steps; it neither interrupts rollout collection nor rolls back the policy. The reference is the policy that collected that rollout, and PPO ratio clipping remains enabled. The logged epoch mean is not the individual triggering estimate, and an average KL cannot establish behavior preservation in rare states.
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
  `train/policy_loss/mean` and `train/value_loss/mean`, with `algorithm` equal to
  `ppo` or `a2c`, are not proxies for their respective
  gradient magnitudes.
- For reward-transform ablations, first compare `train/reward/task/{mean,std}` with
  `train/reward/{mean,std}`. `task.reward.reward_scale` is a finite multiplier from zero through one,
  so values below one attenuate the policy-facing reward. If raw rewards match but shaped
  magnitudes diverge, inspect
  value loss and explained variance before policy entropy, dominant-action rate, KL, and clip
  fraction: squared-error value loss can grow roughly with the square of the target scale, while
  advantage normalization does not protect the critic from poorly conditioned targets.
- Do not compare shaped episode-return or value magnitudes as policy quality across different reward
  transforms. Use task success and acceptance-evaluation metrics for the outcome comparison; use
  reward, critic, and policy metrics to locate the causal chain.
- `train/return/mean` begins with the first genuine
  target-origin episode and rolls over the configured window. It is an online behavior-policy
  proxy whose episodes may span learner updates, not an estimate of one frozen checkpoint's
  evaluation performance. A threshold condition with `progress_baseline` additionally shows
  local target progress as the current metric's clamped fraction from that baseline to its threshold.
  This display is derived locally and is not another published series. Only goal-owned checkpoint evaluation may establish acceptance.
- `train/episode_steps/mean` is the only emitted episode-length reduction; episode-length
  minimum and maximum are not currently emitted. It averages the configured recent completed-episode
  window (currently 100), including warm-up; still-running episodes do not enter the window.
  It counts policy transitions; nominal native frames
  per step follow the run's recorded `frame_skip`. It combines target- and archive-origin
  episodes and is therefore not a clean survival comparison when archive curricula differ. Within
  identical start, frame-skip, reset, and termination contracts, increasing length can indicate
  better ball defense when target return or progress also improves and failure-reason rates do not;
  it can instead reflect delayed serving, low-progress play, or the episode cap, while faster wall
  completion can shorten a successful episode. Treat it as a diagnostic, not a monotonic success or
  acceptance measure.
- Breakout declares `score` as reward-independent episode progress.
  `train/progress/score/mean` is the mean terminal provider-native Atari
  score over the most recent target-origin episodes, including warm-up before the 100-episode
  window is full. GradLab-owned life-loss and serve-stall penalties do not change it. Atari score
  weights brick rows differently, so this metric is not a brick count; it remains an online
  behavior-policy training proxy rather than frozen-checkpoint evaluation evidence.
- Breakout also declares `bricks_destroyed` and `bricks_destroyed_normalized` as
  reward-independent episode progress. The Breakout workspace primary panels show normalized
  bricks destroyed mean, maximum, and minimum as fractions; diagnostics show the absolute mean.
  Their `train/progress/{progress}/mean`
  metrics average the terminal cumulative values over the most recent 100 target-origin episodes,
  while `train/progress/{progress}/max` and `/min` report the maximum and minimum
  over that same window. Minimum bricks describes the worst episode in the current window,
  including failures; it is not an all-time minimum or a fourth ranking criterion. The normalized value is the `0.0..1.0` two-wall completion fraction.
  The default `ppo` recipe terminates successfully after the first cleared wall,
  which corresponds to `0.5` on this provider-normalized metric. It does not
  rescale the metric to the recipe's episode boundary; use the run's recorded
  success rule when interpreting completion and progress-based occupancy buckets.
  All three statistics
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
- Time to a training-progress threshold is elapsed training wall time until the declared rolling target-progress mean reaches that threshold; it is distinct from `train/episode_steps/mean`, which measures episode duration in policy steps. Compare both elapsed time and global steps, declare any persistence requirement before comparison, and treat unreached thresholds as unreached rather than estimating completion by extrapolation. W&B `_runtime` is logger runtime and excludes pre-run provisioning; end-to-end comparisons must include launch/setup separately.
- Training episode reduction aggregates return, length, outcome, success, the explicitly supported
  target-origin cell-novelty statistic, and goal-declared numeric episode progress fields.
  Progress field names refer to task-semantic signals and must be populated independently of the
  selected reward shape; `VizdoomDeathmatch-v1` maps task signal `kills` to provider field
  `killcount`, so
  `train/progress/kills/mean` reports recent mean native monster frags for
  genuine target-origin training episodes. Native shaped return is not an exact
  substitute because different monster kills can contribute different score values.
  `eval/progress/kills/{mean|max}` remains frozen-checkpoint evaluation evidence.
- Snapshot-curriculum `train/curriculum/probability/max` and `train/curriculum/effective_cells/count` summarize the
  current cell-probability distribution. They do not report realized per-cell selection frequency
  or identify which resident cells were selected.
- Go-Explore `train/go-explore/visits/count` includes initialization visits and subsequent
  nonterminal destination-cell visits. It is not a count of every collected pre-action state.
  Curriculum admission counts refer to cell-crossing candidates, while archive cell and entry
  counts describe retained inventory. The separate `train/occupancy/table` counts each collected transition once by its
  pre-action cell, including terminal and truncated transitions. Resets, restores, and optimizer
  reuse add no counts. Entries count the first collected transition after a reset, restore, or
  cell crossing. They do not count distinct episodes or prove reachability.
- Occupancy windows default to 100,000 transitions, rounded up to a full vector batch.
  Combined denominators include normal, archive, and search origins. An empty origin has null
  fractions. Declared cells with no exposure have zero counts. Gaps are uncovered intervals,
  never zero-filled windows. Partial windows retain their actual denominators and completeness.
  Cumulative counts belong to one collection segment. Model-only continuation starts a new
  segment; a matching collector and learner recovery cursor is required for continuous totals.
- Occupancy tables identify the Run, Attempt, segment, cell contract, sequence, and step bounds.
  Bounded eight-window pages preserve exact historical inspection. The recent panel reads the
  latest page; the historical panel uses an explicit table index because W&B's unindexed
  history query samples versions. All versions remain addressable. The selected-window chart and exact
  coarser grouping sum raw counts and retain the population denominator. They never average
  cell fractions. Display labels do not change the cell contract hash.
- `train/curriculum/distribution` lists retained representatives separately from exposure.
  Its table records the publishing Run identity; charts separate Runs rather than adding their
  inventories or probabilities together.
  Its probabilities describe the regular sampler frozen at rollout admission. Cold-cell
  dispatch is explicit and precedes that sampler. The cap is at least `1 / eligible_cells`
  when fewer cells make the configured cap infeasible. Coverage feedback uses recent combined
  counts, including assisted practice. Existing `value_error` feedback remains mean absolute
  raw GAE for completed archive-origin trajectories.
- Occupancy and curriculum tables have `train/step` as their scientific axis and require their
  respective configuration. They confer no Training Success, Acceptance, or Promotion authority.
- Derived throughput phase timing satisfies `loop wall time = provider step time +
  train/rollout_overhead/seconds + train/between_rollouts/seconds`. Compare
  those phases on matching workloads to identify a training-loop bottleneck. Rollout overhead includes
  policy inference plus wrapper, buffer, reset, task, and callback work outside the native provider.
  between-rollout time includes optimizer updates, callbacks, and logging, so it is deliberately
  not named optimization time. The corresponding rates are
  `train/throughput/rate` and `train/provider/rate`.
- Reward components are emitted only when configured. Each component mean includes every policy
  transition in the report window, including zero-valued transitions; nonzero rate separately reports
  the fraction on which the component contributed. Each component also has an absolute share;
  pre-transform task statistics appear when the immutable transform contract is non-identity or a scientific selector requires them. Mario's `progress` component includes
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
- Reward reports cover one actor-critic rollout or one JERK/Go-Explore reporting interval.
  Every backend consumes the shared task reward records, including the final incomplete interval.
- Base identity tasks expose final and pre-transform reward streams to the shared training
  accumulator even without reward shaping. Availability follows the immutable contract rather
  than per-rollout sample equality. A non-identity transform publishes both streams even when a
  rollout happens to be unchanged. With an identity transform, suppress duplicate task statistics.
  For an identity task with one native or event component and no active transform, suppress that
  component's duplicate mean/activity/share. Its share is one on an active window and zero on an
  all-zero window. Fixed event-reward means can be derived from the effective float32 coefficient
  times event activity fraction; delta rewards retain both because a firing can change several
  units. Required scientific selectors retain their exact series despite these suppression rules.
  Current fixed and delta event coefficients must be finite and nonzero.
- The player's protocol-v8 Reward analysis ledger is local playback telemetry, not a W&B metric.
  It shows raw component values, multiplies each impact by the unit-interval reward scale,
  accounts for unattributed raw reward and per-transition clipping, and reports both signed
  contribution (`impact / abs(final reward)`) and absolute transformed activity share. Signed
  contributions preserve penalties, can exceed 100%, and are unavailable at zero final reward;
  this is intentionally different from `train/reward/part/{component}/share` below.
- Under the current `VizdoomDefendCenter-v1` identity-reward contract, every spawned target has one
  health point, its death adds `+1`, the player starts with 52 pistol rounds, and the scenario has no
  ammo replenishment. A normal episode return is therefore `player kills - 1` when the player dies
  and `player kills` when it reaches the native time limit; 52 is the perfect-accuracy ammunition
  ceiling, not a score guaranteed by possessing the ammunition.
- `VizdoomDefendCenter-v1` classifies reaching 52 kills as success and stops training when
  `train/success/min` reaches one. With its single configured
  start, this requires 100 consecutive perfect-score training episodes; it is training success
  evidence, not checkpoint acceptance or promotion.
- `VizdoomHealthGathering-Plus-v1` is a surface-variant identity over the regular
  `VizdoomHealthGathering-v1` task. Both classify the 2,100-native-tic horizon as success, stop at a
  mature rolling success rate of one, use the same neutral return-plateau fallback stop, and
  require an evaluation success rate of at least 0.95 for acceptance.
- A ViZDoom success-rate target is success-based early stopping only when its `target_reached`
  condition has `action: stop`. Every ViZDoom goal with a binary success event now stops when
  `train/success/min` reaches one. With the current configured
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
  `train/patience/{condition}/fraction` projects that condition's patience state for diagnosis and shadow
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

`eval/starts/table` has one row per start with these columns:

`start_id`, `episode_count`, `success_count`, `success_rate`,
`shaped_return_mean`, and `failure_reasons`.

`failure_reasons` is a structured mapping from reason to episode count. The checkpoint step is
already the row's `eval/step` axis and is therefore not duplicated in the table.

Episode-level evidence stays in R2. Confidence intervals and start-by-reason scalar products
are intentionally computed offline rather than added to W&B history.

Complete authoritative acceptance aggregates include both `eval/return/mean` and `eval/return/max`.
The supervisor recomputes these from the complete episode evidence before evaluation and leader
projections; registration alone never supplies a missing result.

An acceptance contract may reject fail-fast only when the first failed outcome proves its rule
cannot pass. That rejection is complete evidence of failure, but not a complete 100-episode
evaluation, so it emits no partial evaluation aggregates. Aggregate contracts such as mean return
disable outcome-based fail-fast, run every planned episode, and emit complete evaluation aggregates
for either verdict. W&B history always receives `eval/step`, pass, and
completed episode count. Planned episode count belongs to the exact evaluation contract metadata,
not a constant history series. Complete full-evaluation projections additionally include
`eval/starts/table`. Per-start success and failure-reason summaries are derived from immutable
private-R2 episode rows. Duration, artifact, source, and raw failure details stay in typed result,
evidence, or checkpoint metadata rather than being duplicated in the W&B-shaped metric map.

`eval/pass` is per-checkpoint history. W&B summarizes that history with `max`, so the
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
durable `ops/sequence` is also W&B's internal step, so replay after an interrupted
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

## Naming and publication rules

Use short paths rooted in `train`, `eval`, `leader`, or `ops`. Keep a subject and its statistic or
unit, adding a group or dimension only when it distinguishes meaning. Statistics use `/min`, `/max`,
`/mean`, `/std`, `/count`, and `/fraction`; timings use `/seconds`, sizes `/bytes`, and throughput
`/rate`. Genuine compound subjects use snake_case, such as `value_loss` or `episode_steps`.
Operation counts and durations share a subject, such as `capture/count` and `capture/seconds`.
Keys contain no spaces; labels use consistent readable spacing.

Common PPO/A2C metrics share names, while immutable algorithm/backend configuration retains their
formula and comparability distinctions. The registry summary reducer is independent of the metric's
statistic: recent `train/return/max` still uses `last`, not an all-time `max`. Applicability follows
explicit family and reward-contract predicates, not removed namespace segments. Local completion
and target-progress fields are not registry metrics and cannot enter the published payload.

## Registry

<!-- METRIC_REGISTRY_START -->
| Metric or template | Display label | Meaning | Unit | Cadence | Placement | Summary | Axis | Evidence | Leader | Training proxy |
|---|---|---|---|---|---|---|---|---|---|---|
| `train/return/mean` | Recent target return mean | Mean shaped return over the most recent genuine target-origin episodes, including warm-up before the configured window is full. | return | rollout | history | last | train/step | training | - | - |
| `train/return/max` | Recent target return max | Maximum shaped return over the same recent target-origin episodes as the rolling mean. | return | rollout | history | last | train/step | training | - | - |
| `train/episode_steps/mean` | Recent episode length mean | Mean policy-transition count over the most recent genuine completed episodes across target and archive origins. | steps | rollout | history | last | train/step | training | - | - |
| `train/unique_cells/mean` | Recent target unique cells mean | Mean episodic unique-cell count over recent target-origin episodes when cell-novelty shaping is active; the reset cell is included. | cells | rollout | history | last | train/step | training | - | - |
| `train/progress/{progress}/mean` | Recent target {progress} mean | Mean of a goal-declared finite numeric progress field over recent genuine target-origin episodes. | value | rollout | history | last | train/step | training | - | - |
| `train/progress/{progress}/max` | Recent target {progress} max | Maximum of a goal-declared finite numeric progress field over recent genuine target-origin episodes. | value | rollout | history | last | train/step | training | - | - |
| `train/progress/{progress}/min` | Recent target {progress} min | Minimum of a goal-declared finite numeric progress field over recent genuine target-origin episodes. | value | rollout | history | last | train/step | training | - | - |
| `train/episodes/count` | Completed episodes | Cumulative genuine completed training episodes across origins. | episodes | rollout | history | last | train/step | training | - | - |
| `train/unsuccessful/{reason}/fraction` | Recent unsuccessful {reason} fraction | Recent unsuccessful completed episodes whose terminal record contains the reason divided by all recent completed episodes; reason presence is boolean per episode. | fraction | rollout | history | last | train/step | training | - | - |
| `train/success/{start}/count` | Successful target episodes from {start} | Cumulative successful genuine target-origin episodes from one start. | episodes | rollout | history | last | train/step | training | - | - |
| `train/success/{start}/fraction` | Recent success rate from {start} | Success fraction over recent genuine target-origin attempts from one start, emitted only after that start fills the configured episode window. | fraction | rollout | history | last | train/step | training | - | - |
| `train/success/min` | Recent all-start success rate min | Minimum recent target-origin success rate, emitted after every configured start fills the episode window. | fraction | rollout | history | last | train/step | training | - | - |
| `train/success/mean` | Recent all-start success rate mean | Mean recent target-origin success rate, emitted after every configured start fills the episode window. | fraction | rollout | history | last | train/step | training | - | - |
| `train/patience/{condition}/fraction` | Early-stop {condition} patience | Policy-step patience progress capped at one; one means the condition would trigger. For `no_improvement` conditions, progress measures steps since the later of eligibility and the last qualifying improvement, so each qualifying improvement resets the patience clock. | fraction | watched metric sample | history | last | train/step | training | - | - |
| `train/reward/mean` | Shaped reward mean | Mean learner-facing per-step reward after gradlab scaling and clipping. | scalar | report | history | last | train/step | training | - | - |
| `train/reward/std` | Shaped reward std | Standard deviation of learner-facing per-step reward after gradlab scaling and clipping. | scalar | report | history | last | train/step | training | - | - |
| `train/reward/nonzero/fraction` | Shaped nonzero reward rate | Fraction of learner-facing per-step rewards that are nonzero after scaling and clipping. | fraction | report | history | last | train/step | training | - | - |
| `train/reward/task/mean` | Task reward mean | Mean task reward before scaling and clipping; emitted for a non-identity transform or when explicitly required by a scientific selector. | scalar | report | history | last | train/step | training | - | - |
| `train/reward/task/std` | Task reward std | Standard deviation of task reward before scaling and clipping; emitted for a non-identity transform or when explicitly required by a scientific selector. | scalar | report | history | last | train/step | training | - | - |
| `train/reward/part/{component}/mean` | Reward {component} mean | Mean configured reward-component contribution over every policy transition in the report window, including zero-valued transitions, in pre-transform task-reward units. In identity `reward_mode: events`, native score contributes zero and the active `event` component sums fixed and delta rewards; provider-native score remains available as episode progress. | scalar | report | history | last | train/step | training | - | - |
| `train/reward/part/{component}/fraction` | Reward {component} activity rate | Fraction of active reward-component values that are nonzero. | fraction | report | history | last | train/step | training | - | - |
| `train/reward/part/{component}/share` | Reward {component} share | Absolute contribution share computed from components in pre-transform task-reward units. | fraction | report | history | last | train/step | training | - | - |
| `train/reward/event/{event}/mean` | Reward event {event} mean | Mean contribution from one declared event reward in pre-transform task-reward units, including zero-valued transitions. Fixed event rewards pay the coefficient once per firing; `event_delta_rewards` pay it times the absolute signal change on a declared increase/decrease event. | scalar | report | history | last | train/step | training | - | - |
| `train/reward/event/{event}/fraction` | Reward event {event} activity rate | Fraction of policy transitions on which one declared event reward contributes a nonzero value; multiple units in one delta-reward transition count as one active transition. | fraction | report | history | last | train/step | training | - | - |
| `train/kl/mean` | PPO approximate KL | Approximate KL divergence for the PPO update. | scalar | rollout | history | last | train/step | training | - | - |
| `train/clip/fraction` | PPO clip fraction | Fraction of sampled policy ratios outside PPO's clipping interval. | fraction | rollout | history | last | train/step | training | - | - |
| `train/jerk/programs/count` | JERK retained sequences | Distinct action sequences retained by JERK search. | sequences | rollout | history | last | train/step | training | - | - |
| `train/jerk/return/mean` | JERK best return mean | Mean observed return of JERK's highest-ranked candidate, selected from retained sequences and live exploration prefixes; a live prefix contributes one return observation. | return | rollout | history | last | train/step | training | - | - |
| `train/program/steps` | Selected program steps | Policy-transition instructions in the program selected by the run algorithm; JERK may select a live prefix and Go-Explore selects a retained trajectory. | steps | report | history | last | train/step | training | - | - |
| `train/go-explore/cells/count` | Go-Explore archive cells | Semantic cells currently retained by Go-Explore. | cells | interval | history | last | train/step | training | - | - |
| `train/go-explore/bytes` | Go-Explore archive bytes | Uncompressed bytes in distinct retained provider-state blobs. | bytes | interval | history | last | train/step | training | - | - |
| `train/go-explore/visits/count` | Go-Explore archive visits | Cumulative semantic-cell visits. | visits | interval | history | last | train/step | training | - | - |
| `train/go-explore/discovery/fraction` | Go-Explore cell discovery rate | New semantic cells divided by visits in the bounded recent visit window. | fraction | interval | history | last | train/step | training | - | - |
| `train/go-explore/progress` | Go-Explore best progress | Greatest task progress reached by the best retained trajectory. | value | interval | history | last | train/step | training | - | - |
| `train/go-explore/return` | Go-Explore best return | Shaped return of the best retained trajectory. | return | interval | history | last | train/step | training | - | - |
| `train/explained_variance` | Explained variance | Actor-critic value-function explained variance. | scalar | rollout | history | last | train/step | training | - | - |
| `train/policy_loss/mean` | Policy-gradient loss | Actor-critic policy-gradient loss. | scalar | rollout | history | last | train/step | training | - | - |
| `train/value_loss/mean` | Value loss | Actor-critic value loss. | scalar | rollout | history | last | train/step | training | - | - |
| `train/gamma` | Discount factor | Discount frozen at rollout start, used for timeout bootstrapping and GAE; scheduled by absolute training transitions and restored on resume. | scalar | rollout | history | last | train/step | training | - | - |
| `train/learning_rate` | Learning rate | Current actor-critic learning rate. | scalar | rollout | history | last | train/step | training | - | - |
| `train/entropy/mean` | Policy entropy | Actor-critic policy entropy with the entropy-loss sign reversed, averaged over the update samples. Discrete entropy is nonnegative; continuous differential entropy can be negative. | nats | rollout | history | last | train/step | training | - | - |
| `train/noise/std/mean` | Policy distribution std | Arithmetic mean of exp(policy.log_std) over its parameter entries; this summarizes continuous-policy scale parameters, not the empirical standard deviation of sampled or executed actions. | scalar | rollout | history | last | train/step | training | - | - |
| `train/action/fraction/max` | Dominant action rate | Largest empirical frequency among sampled discrete rollout actions; not a per-state maximum policy probability. | fraction | rollout | history | last | train/step | training | - | - |
| `train/value/mean` | Value prediction mean | Mean rollout value prediction. | scalar | rollout | history | last | train/step | training | - | - |
| `train/value/std` | Value prediction std | Standard deviation of rollout value predictions. | scalar | rollout | history | last | train/step | training | - | - |
| `train/advantage/mean` | Advantage mean | Mean rollout advantage. | scalar | rollout | history | last | train/step | training | - | - |
| `train/advantage/std` | Advantage std | Standard deviation of rollout advantages. | scalar | rollout | history | last | train/step | training | - | - |
| `train/throughput/rate` | Training loop throughput | Policy transitions divided by rollout-start-to-next-rollout-start wall time. | transitions/second | rollout | history | last | train/step | training | - | - |
| `train/provider/rate` | Provider step throughput | Policy transitions divided by native-provider step wall time, when native timing is available. | transitions/second | rollout | history | last | train/step | training | - | - |
| `train/rollout_overhead/seconds` | Rollout overhead | Rollout wall time outside native-provider step calls. | seconds | rollout | history | last | train/step | training | - | - |
| `train/between_rollouts/seconds` | Between-rollout time | Wall time after rollout collection and before the next rollout, including updates, callbacks, and logging. | seconds | rollout | history | last | train/step | training | - | - |
| `train/save/seconds` | Model save time | Local model save duration. | seconds | artifact | history | last | train/step | training | - | - |
| `eval/return/mean` | Full-eval return mean | Mean shaped return across completed full-evaluation episodes. | return | evaluation | history | last | eval/step | evaluation | leader/return/mean | train/return/mean |
| `eval/return/max` | Full-eval return max | Maximum shaped return across completed full-evaluation episodes. | return | evaluation | history | last | eval/step | evaluation | leader/return/max | train/return/max |
| `eval/success/min` | Full-eval start success rate min | Minimum success rate across represented evaluation starts. | fraction | evaluation | history | last | eval/step | evaluation | leader/success/min | train/success/min |
| `eval/success/mean` | Full-eval start success rate mean | Mean success rate across represented evaluation starts. | fraction | evaluation | history | last | eval/step | evaluation | - | train/success/mean |
| `eval/progress/{progress}/mean` | Full-eval {progress} mean | Mean goal-declared progress value across completed full-evaluation episodes. | value | evaluation | history | last | eval/step | evaluation | leader/progress/{progress}/mean | train/progress/{progress}/mean |
| `eval/progress/{progress}/max` | Full-eval {progress} max | Maximum goal-declared progress value across completed full-evaluation episodes. | value | evaluation | history | last | eval/step | evaluation | leader/progress/{progress}/max | - |
| `eval/pass` | Acceptance pass | Per-checkpoint acceptance result; its W&B history summary uses max and is not the terminal run verdict. | boolean | acceptance evaluation | history | max | eval/step | acceptance | - | - |
| `eval/episodes/count` | Acceptance episodes completed | Valid planned episode rows completed before acceptance or fail-fast rejection. | episodes | acceptance evaluation | history | last | eval/step | acceptance | - | - |
| `eval/starts/table` | Full-eval evidence by start | Structured full-evaluation evidence by start, including success, return, and failure-reason aggregates. | table | evaluation | history | none | eval/step | evaluation_table | - | - |
| `leader/success/min` | Leader start success rate min | Selected-checkpoint projection of minimum success rate across starts. | fraction | selection | summary | none | - | selection | - | - |
| `leader/return/mean` | Leader return mean | Selected-checkpoint mean shaped episode return. | return | selection | summary | none | - | selection | - | - |
| `leader/return/max` | Leader return max | Selected-checkpoint maximum shaped episode return. | return | selection | summary | none | - | selection | - | - |
| `leader/progress/{progress}/mean` | Leader {progress} mean | Selected-checkpoint mean for one named progress dimension. | value | selection | summary | none | - | selection | - | - |
| `leader/progress/{progress}/max` | Leader {progress} max | Selected-checkpoint maximum for one named progress dimension. | value | selection | summary | none | - | selection | - | - |
| `leader/step` | Leader checkpoint step | Selected checkpoint policy step. | steps | selection | summary | none | - | selection | leader/step | - |
| `leader/artifact` | Leader artifact | Selected checkpoint immutable artifact reference. | metadata | selection | summary | none | - | selection | - | - |
| `leader/source` | Leader evaluation source | Selected checkpoint evaluation source. | text | selection | summary | none | - | selection | - | - |
| `leader/updated_at` | Leader projection time | Selected checkpoint projection update timestamp. | timestamp | selection | summary | none | - | selection | - | - |
| `train/step` | Training global step | Scientific training X-axis: policy environment transitions consumed. | steps | frame | history | max | train/step | training | - | - |
| `eval/step` | Evaluation checkpoint step | Scientific evaluation X-axis: step of the evaluated checkpoint. | steps | evaluation | history | max | eval/step | acceptance | - | - |
| `ops/sequence` | Orchestration event sequence | Monotonic local outbox event sequence used as W&B delivery order. | events | frame | history | max | ops/sequence | operational | - | - |
| `ops/outbox/count` | Pending outbox frames | Metric outbox frames not yet acknowledged by the W&B SDK. | events | supervisor sample | history | last | ops/sequence | operational | - | - |
| `ops/outbox/age/seconds` | Oldest unpublished age | Age of the oldest metric frame not yet acknowledged by the W&B SDK. | seconds | supervisor sample | history | last | ops/sequence | operational | - | - |
| `ops/visibility/seconds` | Remote visibility lag | Age of the oldest local metric frame beyond the W&B API's observed event-sequence high-water mark, sampled at the last successful remote probe. | seconds | remote visibility probe | history | last | ops/sequence | operational | - | - |
| `ops/checkpoints/count` | Pending checkpoints | Ready local checkpoints not yet verified in public model R2. | checkpoints | supervisor sample | history | last | ops/sequence | operational | - | - |
| `ops/evals/count` | Pending evaluations | Persisted evaluation intents pending submission or a verified result; intents deferred after acceptance are excluded. | evaluations | supervisor sample | history | last | ops/sequence | operational | - | - |
| `ops/drain/seconds` | GPU idle drain time | Elapsed wall time since learner exit, sampled after the first terminal drain. This is not measured GPU utilization and excludes subsequent publication and terminal work. | seconds | terminal drain | history | last | ops/sequence | operational | - | - |
| `ops/scratch/fraction` | Scratch used | Fraction of the task scratch filesystem currently used. | fraction | supervisor sample | history | last | ops/sequence | operational | - | - |
| `ops/state` | Terminal run state | Receipt-backed terminal run state. | text | terminal receipt | summary | none | ops/sequence | operational | - | - |
| `ops/reason` | Terminal run reason | Receipt-backed terminal reason when the run did not succeed. | text | terminal receipt | summary | none | ops/sequence | operational | - | - |
| `train/curriculum/cells/count` | Curriculum archive cells | Current archive-curriculum cell count. | cells | rollout | history | last | train/step | training | - | - |
| `train/curriculum/entries/count` | Curriculum archive entries | Current immutable entry count retained by the curriculum view. | entries | rollout | history | last | train/step | training | - | - |
| `train/curriculum/candidates/count` | Curriculum admission candidates | Non-terminal cell-crossing candidates observed during the rollout. | transitions | rollout | history | last | train/step | training | - | - |
| `train/curriculum/admitted/count` | Curriculum admissions accepted | Candidate entries accepted into cell reservoirs during the rollout. | entries | rollout | history | last | train/step | training | - | - |
| `train/curriculum/evicted/count` | Curriculum cells evicted | Curriculum cells evicted during the rollout. | cells | rollout | history | last | train/step | training | - | - |
| `train/curriculum/capture/count` | Curriculum capture calls | Batched portable provider-state capture calls during the rollout. | calls | rollout | history | last | train/step | training | - | - |
| `train/curriculum/restore/count` | Curriculum restore episodes | Archive-origin episodes started during the rollout. | episodes | rollout | history | last | train/step | training | - | - |
| `train/curriculum/forced_boundaries/count` | Curriculum forced boundaries | Non-episode control truncations used to activate archive lanes. | boundaries | rollout | history | last | train/step | training | - | - |
| `train/curriculum/feedback/count` | Curriculum feedback trajectories | Completed archive-origin trajectories committed to the priority sampler. | trajectories | rollout | history | last | train/step | training | - | - |
| `train/curriculum/transitions/fraction` | Curriculum transition share | Fraction of policy transitions whose origin is the archive curriculum. | fraction | rollout | history | last | train/step | training | - | - |
| `train/curriculum/probability/max` | Curriculum max sampling probability | Largest final cell probability in the archive sampler. | fraction | rollout | history | last | train/step | training | - | - |
| `train/curriculum/effective_cells/count` | Curriculum effective cell count | Inverse-Simpson effective cell count of the archive sampling distribution. | cells | rollout | history | last | train/step | training | - | - |
| `train/curriculum/capture/seconds` | Curriculum capture time | Portable state capture wall time accumulated during the rollout. | seconds | rollout | history | last | train/step | training | - | - |
| `train/curriculum/restore/seconds` | Curriculum restore time | Provider restore wall time for reset calls containing archive lanes. | seconds | rollout | history | last | train/step | training | - | - |
| `train/occupancy/table` | Collected cell occupancy | Exact pre-action cell counts, entries and origin denominators in a bounded page of up to eight collection windows; each row retains its own bounds, sequence and denominator. Includes declared zero cells and unavailable fractions for empty origins. Use exact indexed history or latest-page queries, never sampled table history. | table | transition window | history | last | train/step | training | - | - |
| `train/curriculum/distribution` | Curriculum start distribution | Compatible retained representatives, cold-cell status and intended per-cell start probabilities at rollout admission; recent combined counts are coverage feedback, distinct from value-error feedback. | table | rollout start | history | last | train/step | training | - | - |
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
availability still determine emission. Declared discrete action encodings omit policy-scale
series, and plain MultiDiscrete encodings omit sampled categorical action-frequency series.
Unknown action capabilities remain conditional until the runtime inspects the actual policy
and action space; the inventory does not infer them from a game name.

Schema v22 uses short subject/statistic paths, removes redundant history series, and preserves retained populations and reductions. No old-name aliases
are emitted. Managed workspaces filter to the current schema; historical runs keep their original
names and immutable evidence. Reward statistics use streaming moments, and episode snapshots are
reused until a new episode arrives; both retain their existing emission cadence.

Breakout explicitly pins normalized bricks destroyed (mean, maximum, minimum), then mean episode
steps in that display order. Raw mean bricks remains in Breakout diagnostics. This four-chart
display is independent of its three-criterion goal ranking; minimum normalized bricks remains
diagnostic and does not participate in ranking.

## Reward discount overlay in Playback

The standard Step reward panel shows a dashed amber overlay of shaped reward contributions
using a separately pinned reference step and recorded discount: `gamma**(reward_step - reference_step) *
reward_shaped`. Rewards before the reference are excluded and dimmed; the reference has delay zero.
The reference initializes at the first displayed transition of each episode and changes only
when the user presses “Set return reference to cursor”.
Hovering does not change the reference state, table rows, values, or highlighted row.
The compact table stays anchored to the selected Playback step and shows up to five current and future
recorded samples with native (`reward_provider`) and shaped (`reward_shaped`) rewards,
step delay, discount weight, discounted shaped contribution, G(s), and V(s).
G(s) uses the authoritative full-episode `realized_return` for that row’s pre-action
state, not a sum of sampled chart points or rewards from the pinned reference. V(s)
is the recorded pre-action `value` for the same state. For an unfinished episode, G(s) shows a provisional
`estimated_return` from all recorded shaped rewards before the latest recorded
pre-action state plus its discounted V(s), independently of chart sampling and zoom.
A ⚠ icon and tooltip identify the estimate and bootstrap step. It is never calibration
evidence. Missing critic or discount data leaves G(s) Pending; comparison failures show Incomparable with their reasons.
Truncated returns that include terminal-state value are marked with ⚠ and a bootstrap tooltip.
Rows before either the cursor or the return reference are excluded. It prioritizes nonzero
reward samples and includes the exact selected transition when inside the chart window.
The table refreshes when Playback selection or recorded chart history changes, including zoom. Clicking a table step seeks the shared Playback
cursor without changing the discount reference. Missing values and past contributions
are shown as unavailable, not zero; tiny nonzero values use scientific notation.
The overlay uses the chart's recorded sample
points and does not compute an episode return. These contributions describe discount
accounting, not causal action credit. Episode return remains undiscounted reward accumulated through each step.
