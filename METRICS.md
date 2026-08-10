# Metrics schema v19

This file is the source of truth for gradlab telemetry. The Python registry loads the table below
and requires every emitted metric to match an exact registry entry or a bounded template.

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
- W&B config contains run-defining dimensions: `metrics_schema_version: 19`,
  `metrics_episode_window_size: 100`, `training_backend_id`,
  `training_backend_config_hash`, `algorithm_id`, goal,
  environment, starts, seed, frame skip, environment count, hyperparameters, eval protocol, and
  runtime versions. Operational attribution includes `compute_target`,
  `dstack_coordinator_id`, `dstack_project`, `dstack_task`, and `attempt_id`; these are immutable
  run/attempt dimensions, not scientific metrics.
- `experiments/goals/_workspaces.yaml` is the presentation source for managed W&B project
  workspace views. It selects registered metrics without emitting aliases or changing their
  scientific axes or semantics; every project resolved from an active checked-in goal inherits its
  default profile unless the declaration assigns a complete project-specific profile.
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
- `leader/checkpoint/*` contains diagnostic projections of the selected checkpoint. The
  create-only private-R2 `PromotionReceipt` is the authoritative selection.
- `orchestration/run/terminal/state` and `orchestration/run/terminal/reason` are W&B summary-only
  catalog projections, not history metrics; the private-R2 `TerminalReceipt` remains authoritative.
- Heavy model bytes, videos, replays, episode rows, diagnostics, and recovery payloads never go to
  W&B.
- Interactive playback uses local descriptor keys such as `reward/shaped`, `policy/value`, and
  `action/executed` to configure live panels. They are typed projections of one streamed transition
  or its bounded in-browser history, are not emitted metrics, and must not be interpreted as aliases
  for similarly named W&B registry entries. `reward/provider` is the provider output before the
  task program and gradlab-owned reward transform; `reward/shaped` is the final policy-facing reward
  after task shaping, scaling, and clipping. Playback computes realized return-to-go and `V(s) -
  G(s)` only for a terminal, stochastic, policy-driven trajectory whose policy environment,
  reward stream, discount, action sampling, and boundary/bootstrap semantics match training.
  Otherwise those critic diagnostics are explicitly unavailable rather than numerically compared.

The active checkpoint protocol is `acceptance`; complete accepted evidence additionally emits the
`full` metric family. Dimension IDs must be unique and match `[A-Za-z0-9_.-]+`; unsafe IDs are
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
- `orchestration/event/sequence`: durable supervisor delivery order.

Each axis is configured with a W&B `max` summary reducer. W&B's public API may therefore expose
its summary value as a reducer mapping such as `{"max": 5046272}` rather than as a bare number.
Catalog and report consumers must unwrap the configured reducer value; a recipe's requested
`timesteps` cap is not a substitute for the observed `train/global_step`.
Before its first history row is logged, every concrete metric is explicitly bound to its applicable
scientific axis. W&B's internal `Step` is delivery order only and must not become the default X-axis
for scientific charts.

Asynchronous evaluations may arrive after later training rows without changing their scientific
X-axis. Each producer writes only its applicable scientific axis; durable delivery order uses
`orchestration/event/sequence`.

Current runs declare schema v19, and the supervisor validates and emits only v19 names. GradLab
does not read, project, or preserve noncurrent W&B or R2 schemas.

All metrics whose path contains `rolling` use the run-configured
`metrics_episode_window_size`, currently 100. During warm-up they reduce all eligible observations
seen so far; metrics whose meaning requires complete start coverage are withheld until every start
has filled the window. The window size lives once in config rather than being duplicated in every
metric path. `cumulative` explicitly means all eligible observations seen so far.

## Research interpretation

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
  `V(s) - G(s)`.
- Mario recipes disable automatic checkpoint evaluation and stop when
  `train/outcome/success/starts/all/rolling/rate/min` first reaches one. For a single start,
  that means 100 consecutive genuine target-origin clears; for multiple starts, every configured
  start's latest 100 attempts must all clear. This training stop is not acceptance or promotion;
  explicitly evaluated Mario checkpoints rank by earliest `leader/checkpoint/step`, then highest
  `eval/full/episode/return/shaped/mean`. Breakout is training-only and ranks current-contract seeded
  recipe cohorts using `train/episode/return/shaped/origin/target/rolling/mean`, which
  excludes archive-curriculum origins and non-episode control boundaries; tied cohorts prefer fewer
  policy transitions.
- Aggregate training `starts/observed/cumulative/rate/*` is cumulative. Aggregate
  `starts/all/rolling/rate/*` uses the configured recent-episode window and appears only after
  every configured start has filled it. Observed-start aggregates intentionally describe only
  starts attempted so far; the path makes that scope explicit without a duplicate coverage metric.
- A bounded training-only search may use per-start success counts and the history peak and first
  threshold crossing of `train/outcome/success/starts/all/rolling/rate/min` to screen and rank
  recipes.
  That evidence is not checkpoint evaluation and cannot establish checkpoint promotion, goal
  acceptance, or release evidence.
- Failure-reason metrics treat each reason as a per-episode presence flag, not an occurrence count
  or necessarily the event that ended the episode. For each reason, the window rate is the number
  of unsuccessful episodes containing that reason divided by all completed episodes in the latest
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
  `train/outcome/success/starts/all/rolling/rate/min` reaches one. Each has one configured
  start, so this requires 100 consecutive horizon-reaching training episodes; it is training
  success evidence, not checkpoint acceptance or promotion.
- Positive actor-critic policy entropy and dominant-action rate diagnose discrete policy collapse.
  For a categorical `Discrete(n)` policy, entropy has infimum zero and
  maximum `ln(n)` nats at the uniform distribution; a `MultiDiscrete(nvec)` maximum is
  `sum(ln(nvec))`, while a constrained legal-tuple categorical policy has maximum
  `ln(legal_tuple_count)` regardless of the enclosing `nvec`; a `MultiBinary(d)` maximum is
  `d * ln(2)`. Continuous `Box` policies report differential entropy, which has no finite
  action-space-only minimum or maximum. The bounds remain a pure policy-space calculation used by
  diagnostics and are not duplicated as W&B metrics.
- Actor-critic explained variance is `1 - Var(value_target - value_prediction) /
  Var(value_target)`: one is perfect, zero means the critic explains no more target variance than a
  constant baseline, and negative values are worse than that baseline. A near-zero value is not by
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
- For reward-transform ablations, first compare `train/reward/raw/*` with
  `train/reward/shaped/*`. `task.reward.reward_scale` is a finite multiplier from zero through one,
  so values below one attenuate the policy-facing reward. If raw rewards match but shaped
  magnitudes diverge, inspect
  value loss and explained variance before policy entropy, dominant-action rate, KL, and clip
  fraction: squared-error value loss can grow roughly with the square of the target scale, while
  advantage normalization does not protect the critic from poorly conditioned targets.
- Do not compare shaped episode-return or value magnitudes as policy quality across different reward
  transforms. Use task success and acceptance-evaluation metrics for the outcome comparison; use
  reward, critic, and policy metrics to locate the causal chain.
- `train/episode/return/shaped/origin/target/rolling/mean` begins with the first genuine
  target-origin episode and rolls over the configured window. It is an online behavior-policy
  proxy whose episodes may span learner updates, not an estimate of one frozen checkpoint's
  evaluation performance. A threshold condition with `progress_baseline` additionally emits
  `train/early_stop/{condition}/target/progress` as the current metric's clamped fraction from that
  baseline to its threshold. Only goal-owned checkpoint evaluation may establish acceptance.
- Training episode reduction aggregates return, length, outcome, success, the explicitly supported
  target-origin cell-novelty statistic, and goal-declared numeric episode progress fields.
  Progress field names refer to task-semantic signals and must be populated independently of the
  selected reward shape; `VizdoomDeathmatch-v1` maps task signal `kills` to provider field
  `killcount`, so
  `train/progress/kills/origin/target/rolling/mean` reports recent mean native monster frags for
  genuine target-origin training episodes. Native shaped return is not an exact
  substitute because different monster kills can contribute different score values.
  `eval/full/progress/kills/{mean|max}` remains frozen-checkpoint evaluation evidence.
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
- Reward components are emitted only when active. Each component has mean, nonzero rate, and share;
  raw reward appears only when it differs from shaped reward. Mario's `progress` component includes
  both its base new-progress reward and any configured additional new-progress reward above
  `progress_reward_boost_start_x`. ViZDoom Deathmatch's optional `sample-factory-v0` shape exposes
  `kill`, `death`, `hit`, `damage`, `health`, `armor`, `weapon`, `ammo`, and `weapon_hold`
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
  `train/outcome/success/starts/all/rolling/rate/min` reaches one. With its single configured
  start, this requires 100 consecutive perfect-score training episodes; it is training success
  evidence, not checkpoint acceptance or promotion.
- `VizdoomHealthGathering-Plus-v1` is a surface-variant identity over the regular
  `VizdoomHealthGathering-v1` task. Both classify the 2,100-native-tic horizon as success, stop at a
  mature rolling success rate of one, use the same neutral return-plateau fallback stop, and
  require an evaluation success rate of at least 0.95 for acceptance.
- A ViZDoom success-rate target is success-based early stopping only when its `target_reached`
  condition has `action: stop`. Every ViZDoom goal with a binary success event now stops when
  `train/outcome/success/starts/all/rolling/rate/min` reaches one. With the current configured
  window this requires 100
  consecutive successful training episodes for each configured start. For
  `VizdoomDeathmatch-v1`, reaching the 4,200-native-tic horizon is a successful outcome while the
  episode boundary remains truncated so training bootstraps from the final observation. Its single
  configured start therefore stops after 100 consecutive horizon-reaching training episodes. This
  training success evidence does not replace its goal-owned mean-kills checkpoint acceptance rule.
- Episode-return means are neither a best-episode metric nor the score of a currently visible lane:
  they reduce the configured recent episode window across all applicable vector lanes. W&B chart
  smoothing, when enabled, is applied on top of that already-rolling value. Under the root Breakout
  contract (`reward_mode: native`, unclipped), shaped episode return is the sum of Atari row-score
  deltas, so individual 400-plus games can coexist with a much lower mean when other lanes finish
  with lower scores.
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

`eval/full/start/table` has one row per start with these columns:

`start_id`, `episode_count`, `success_count`, `success_rate`,
`shaped_return_mean`, and `failure_reasons`.

`failure_reasons` is a structured mapping from reason to episode count. The checkpoint step is
already the row's `eval/checkpoint/step` axis and is therefore not duplicated in the table.

Episode-level evidence stays in R2. Confidence intervals and start-by-reason scalar products
are intentionally computed offline rather than added to W&B history.

An acceptance contract may reject fail-fast only when the first failed outcome proves its rule
cannot pass. That rejection is complete evidence of failure, but not a complete 100-episode
evaluation, so it emits no partial `eval/full/*` result. Aggregate contracts such as mean return
disable outcome-based fail-fast, run every planned episode, and emit complete `eval/full/*` metrics
for either verdict. W&B history always receives `eval/checkpoint/step`, pass, and
planned/completed episodes. Complete full-evaluation projections additionally include
`eval/full/start/table`. Per-start success and failure-reason summaries are derived from immutable
private-R2 episode rows. Duration, artifact, source, and raw failure details stay in typed result,
evidence, or checkpoint metadata rather than being duplicated in the W&B-shaped metric map.

`eval/acceptance/pass` is per-checkpoint history. W&B summarizes that history with `max`, so the
summary means that some checkpoint passed; it is not the run verdict. The authoritative verdict is
the create-only private-R2 `PromotionReceipt`, whose selected result is hash-bound to the complete
acceptance evidence. At terminal publication, that receipt projects
`orchestration/run/terminal/*`, the diagnostic `leader/checkpoint/*` fields, and the accepted
W&B projection. Leader fields mirror only finite
selected metrics required by the configured rank plus available diagnostics; no generic objective,
serialized rank tuple, constant acceptance alias, or fabricated default is emitted. Later rejected
checkpoint projections remain in history and never modify the active projection. Raw acceptance
aggregates and episode evidence remain authoritative in private eval R2.

## Delivery, backpressure, and recovery

Every event has a stable content-derived internal event ID. Delivery to W&B is at least once; the
durable `orchestration/event/sequence` is also W&B's internal step, so replay after an interrupted
local acknowledgement cannot append a second scientific point. The event ID remains a transport
invariant and is not duplicated as a W&B metric. Promotion, terminal state, and early-stop
authority are exactly once through conditional private-R2 receipts.

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
| Metric or template | Display label | Meaning | Unit | Cadence | Placement | Summary |
|---|---|---|---|---|---|---|
| `train/episode/return/shaped/origin/target/rolling/mean` | Recent target return mean | Mean shaped return over the most recent genuine target-origin episodes, including warm-up before the configured window is full. | return | rollout | history | last |
| `train/episode/return/shaped/origin/target/rolling/max` | Recent target return max | Maximum shaped return over the same recent target-origin episodes as the rolling mean. | return | rollout | history | last |
| `train/episode/length/origin/all/rolling/mean` | Recent episode length mean | Mean length over the most recent genuine completed episodes across target and archive origins. | steps | rollout | history | last |
| `train/exploration/cell/unique/origin/target/rolling/mean` | Recent target unique cells mean | Mean episodic unique-cell count over recent target-origin episodes when cell-novelty shaping is active; the reset cell is included. | cells | rollout | history | last |
| `train/progress/{progress}/origin/target/rolling/mean` | Recent target {progress} mean | Mean of a goal-declared finite numeric progress field over recent genuine target-origin episodes. | value | rollout | history | last |
| `train/episode/completed/count` | Completed episodes | Cumulative genuine completed training episodes across origins. | episodes | rollout | history | last |
| `train/outcome/failure/reason/{reason}/rolling/rate` | Recent failure {reason} rate | Recent unsuccessful completed episodes containing the reason divided by all recent completed episodes; reason presence is boolean per episode. | fraction | rollout | history | last |
| `train/outcome/success/start/{start}/episode/count` | Successful target episodes from {start} | Cumulative successful genuine target-origin episodes from one start. | episodes | rollout | history | last |
| `train/outcome/success/start/{start}/rolling/rate` | Recent success rate from {start} | Success fraction over recent genuine target-origin attempts from one start. | fraction | rollout | history | last |
| `train/outcome/success/starts/observed/cumulative/rate/min` | Observed-start success rate min | Minimum cumulative target-origin success rate across starts with at least one genuine attempt. | fraction | rollout | history | last |
| `train/outcome/success/starts/observed/cumulative/rate/mean` | Observed-start success rate mean | Mean cumulative target-origin success rate across starts with at least one genuine attempt. | fraction | rollout | history | last |
| `train/outcome/success/starts/all/rolling/rate/min` | Recent all-start success rate min | Minimum recent target-origin success rate, emitted after every configured start fills the episode window. | fraction | rollout | history | last |
| `train/outcome/success/starts/all/rolling/rate/mean` | Recent all-start success rate mean | Mean recent target-origin success rate, emitted after every configured start fills the episode window. | fraction | rollout | history | last |
| `train/early_stop/{condition}/patience/progress` | Early-stop {condition} patience | Policy-step patience progress capped at one; one means the condition would trigger. For `no_improvement` conditions, progress measures steps since the later of eligibility and the last qualifying improvement, so each qualifying improvement resets the patience clock. | fraction | watched metric sample | history | last |
| `train/early_stop/{condition}/target/progress` | Early-stop {condition} target | Threshold progress from the declared baseline to the target in the improving direction, clamped to zero through one. | fraction | watched metric sample | history | last |
| `train/reward/shaped/mean` | Shaped reward mean | Mean learner-facing per-step reward after gradlab scaling and clipping. | scalar | rollout | history | last |
| `train/reward/shaped/std` | Shaped reward std | Standard deviation of learner-facing per-step reward after gradlab scaling and clipping. | scalar | rollout | history | last |
| `train/reward/shaped/nonzero/rate` | Shaped nonzero reward rate | Fraction of learner-facing per-step rewards that are nonzero after scaling and clipping. | fraction | rollout | history | last |
| `train/reward/raw/mean` | Raw reward mean | Mean completed task reward immediately before gradlab-owned scaling and clipping, emitted when distinct from shaped reward. | scalar | rollout | history | last |
| `train/reward/raw/std` | Raw reward std | Standard deviation of completed task reward immediately before gradlab-owned scaling and clipping, emitted when distinct from shaped reward. | scalar | rollout | history | last |
| `train/reward/component/{component}/mean` | Reward {component} mean | Mean active reward-component contribution in pre-transform task-reward units. | scalar | rollout | history | last |
| `train/reward/component/{component}/nonzero/rate` | Reward {component} activity rate | Fraction of active reward-component values that are nonzero. | fraction | rollout | history | last |
| `train/reward/component/{component}/share` | Reward {component} share | Absolute contribution share computed from components in pre-transform task-reward units. | fraction | rollout | history | last |
| `train/algorithm/ppo/update/approx_kl` | PPO approximate KL | Approximate KL divergence for the PPO update. | scalar | rollout | history | last |
| `train/algorithm/ppo/update/clip_fraction` | PPO clip fraction | Fraction of sampled policy ratios outside PPO's clipping interval. | fraction | rollout | history | last |
| `train/algorithm/jerk/retained/count` | JERK retained sequences | Distinct action sequences retained by JERK search. | sequences | rollout | history | last |
| `train/algorithm/jerk/best/return/mean` | JERK best return mean | Mean observed return of JERK's highest-ranked retained sequence. | return | rollout | history | last |
| `train/algorithm/jerk/best/program/steps` | JERK best program steps | Action length of JERK's highest-ranked retained sequence. | steps | rollout | history | last |
| `train/algorithm/go-explore/archive/cell/count` | Go-Explore archive cells | Semantic cells currently retained by Go-Explore. | cells | interval | history | last |
| `train/algorithm/go-explore/archive/blob/bytes` | Go-Explore archive bytes | Uncompressed bytes in distinct retained provider-state blobs. | bytes | interval | history | last |
| `train/algorithm/go-explore/archive/visit/count` | Go-Explore archive visits | Cumulative semantic-cell visits. | visits | interval | history | last |
| `train/algorithm/go-explore/archive/cell/discovery/rate` | Go-Explore cell discovery rate | New semantic cells divided by visits in the bounded recent visit window. | fraction | interval | history | last |
| `train/algorithm/go-explore/best/progress` | Go-Explore best progress | Greatest task progress reached by the best retained trajectory. | value | interval | history | last |
| `train/algorithm/go-explore/best/return` | Go-Explore best return | Shaped return of the best retained trajectory. | return | interval | history | last |
| `train/algorithm/go-explore/best/program/steps` | Go-Explore best program steps | Environment steps in the best retained action program. | steps | interval | history | last |
| `train/algorithm/{algorithm}/value/explained_variance` | {algorithm} explained variance | Actor-critic value-function explained variance. | scalar | rollout | history | last |
| `train/algorithm/{algorithm}/update/policy_gradient_loss` | {algorithm} policy-gradient loss | Actor-critic policy-gradient loss. | scalar | rollout | history | last |
| `train/algorithm/{algorithm}/update/value_loss` | {algorithm} value loss | Actor-critic value loss. | scalar | rollout | history | last |
| `train/algorithm/{algorithm}/update/learning_rate` | {algorithm} learning rate | Current actor-critic learning rate. | scalar | rollout | history | last |
| `train/algorithm/{algorithm}/policy/entropy` | {algorithm} policy entropy | Positive actor-critic policy entropy. | nats | rollout | history | last |
| `train/algorithm/{algorithm}/policy/distribution/std` | {algorithm} policy distribution std | Continuous-action distribution standard deviation. | scalar | rollout | history | last |
| `train/algorithm/{algorithm}/policy/dominant/action/rate` | {algorithm} dominant action rate | Fraction assigned to the most frequent sampled discrete action. | fraction | rollout | history | last |
| `train/algorithm/{algorithm}/rollout/value/prediction/mean` | {algorithm} value prediction mean | Mean rollout value prediction. | scalar | rollout | history | last |
| `train/algorithm/{algorithm}/rollout/value/prediction/std` | {algorithm} value prediction std | Standard deviation of rollout value predictions. | scalar | rollout | history | last |
| `train/algorithm/{algorithm}/rollout/advantage/mean` | {algorithm} advantage mean | Mean rollout advantage. | scalar | rollout | history | last |
| `train/algorithm/{algorithm}/rollout/advantage/std` | {algorithm} advantage std | Standard deviation of rollout advantages. | scalar | rollout | history | last |
| `train/throughput/loop/rate` | Training loop throughput | Policy transitions divided by rollout-start-to-next-rollout-start wall time. | transitions/second | rollout | history | last |
| `train/throughput/provider/step/rate` | Provider step throughput | Policy transitions divided by native-provider step wall time, when native timing is available. | transitions/second | rollout | history | last |
| `train/throughput/rollout/overhead/seconds` | Rollout overhead | Rollout wall time outside native-provider step calls. | seconds | rollout | history | last |
| `train/throughput/between/rollouts/seconds` | Between-rollout time | Wall time after rollout collection and before the next rollout, including updates, callbacks, and logging. | seconds | rollout | history | last |
| `train/artifact/save/seconds` | Model save time | Local model save duration. | seconds | artifact | history | last |
| `eval/full/episode/return/shaped/mean` | Full-eval return mean | Mean shaped return across completed full-evaluation episodes. | return | evaluation | history | last |
| `eval/full/episode/return/shaped/max` | Full-eval return max | Maximum shaped return across completed full-evaluation episodes. | return | evaluation | history | last |
| `eval/full/outcome/success/starts/rate/min` | Full-eval start success rate min | Minimum success rate across represented evaluation starts. | fraction | evaluation | history | last |
| `eval/full/outcome/success/starts/rate/mean` | Full-eval start success rate mean | Mean success rate across represented evaluation starts. | fraction | evaluation | history | last |
| `eval/full/progress/{progress}/mean` | Full-eval {progress} mean | Mean goal-declared progress value across completed full-evaluation episodes. | value | evaluation | history | last |
| `eval/full/progress/{progress}/max` | Full-eval {progress} max | Maximum goal-declared progress value across completed full-evaluation episodes. | value | evaluation | history | last |
| `eval/acceptance/pass` | Acceptance pass | Per-checkpoint acceptance result; its W&B history summary uses max and is not the terminal run verdict. | boolean | acceptance evaluation | history | max |
| `eval/acceptance/episode/planned/count` | Acceptance episodes planned | Exact episode identities required by the acceptance manifest. | episodes | acceptance evaluation | history | last |
| `eval/acceptance/episode/completed/count` | Acceptance episodes completed | Valid planned episode rows completed before acceptance or fail-fast rejection. | episodes | acceptance evaluation | history | last |
| `eval/full/start/table` | Full-eval evidence by start | Structured full-evaluation evidence by start, including success, return, and failure-reason aggregates. | table | evaluation | history | none |
| `leader/checkpoint/outcome/success/starts/rate/min` | Leader start success rate min | Selected-checkpoint projection of minimum success rate across starts. | fraction | selection | summary | none |
| `leader/checkpoint/episode/return/shaped/mean` | Leader return mean | Selected-checkpoint mean shaped episode return. | return | selection | summary | none |
| `leader/checkpoint/episode/return/shaped/max` | Leader return max | Selected-checkpoint maximum shaped episode return. | return | selection | summary | none |
| `leader/checkpoint/progress/{progress}/mean` | Leader {progress} mean | Selected-checkpoint mean for one named progress dimension. | value | selection | summary | none |
| `leader/checkpoint/progress/{progress}/max` | Leader {progress} max | Selected-checkpoint maximum for one named progress dimension. | value | selection | summary | none |
| `leader/checkpoint/step` | Leader checkpoint step | Selected checkpoint policy step. | steps | selection | summary | none |
| `leader/checkpoint/artifact/ref` | Leader artifact | Selected checkpoint immutable artifact reference. | metadata | selection | summary | none |
| `leader/checkpoint/evaluation/source` | Leader evaluation source | Selected checkpoint evaluation source. | text | selection | summary | none |
| `leader/checkpoint/projection/timestamp` | Leader projection time | Selected checkpoint projection update timestamp. | timestamp | selection | summary | none |
| `train/global_step` | Training global step | Scientific training X-axis: policy environment transitions consumed. | steps | frame | history | max |
| `eval/checkpoint/step` | Evaluation checkpoint step | Scientific evaluation X-axis: step of the evaluated checkpoint. | steps | evaluation | history | max |
| `orchestration/event/sequence` | Orchestration event sequence | Monotonic local outbox event sequence used as W&B delivery order. | events | frame | history | max |
| `orchestration/outbox/pending/count` | Pending outbox frames | Metric outbox frames not yet acknowledged by the W&B SDK. | events | supervisor sample | history | last |
| `orchestration/outbox/oldest/age/seconds` | Oldest unpublished age | Age of the oldest metric frame not yet acknowledged by the W&B SDK. | seconds | supervisor sample | history | last |
| `orchestration/outbox/remote/visibility/lag/seconds` | Remote visibility lag | Age of the newest local metric event not yet observed through the W&B API. | seconds | remote visibility probe | history | last |
| `orchestration/checkpoint/pending/count` | Pending checkpoints | Ready local checkpoints not yet verified in public model R2. | checkpoints | supervisor sample | history | last |
| `orchestration/eval/pending/count` | Pending evaluations | Persisted evaluation intents pending submission or a verified result; intents deferred after acceptance are excluded. | evaluations | supervisor sample | history | last |
| `orchestration/drain/gpu/idle/seconds` | GPU idle drain time | Time the training container retained its GPU after the learner exited. | seconds | terminal drain | history | last |
| `orchestration/scratch/used/fraction` | Scratch used | Fraction of the task scratch filesystem currently used. | fraction | supervisor sample | history | last |
| `orchestration/run/terminal/state` | Terminal run state | Receipt-backed terminal run state. | text | terminal receipt | summary | none |
| `orchestration/run/terminal/reason` | Terminal run reason | Receipt-backed terminal reason when the run did not succeed. | text | terminal receipt | summary | none |
| `train/curriculum/archive/cell/count` | Curriculum archive cells | Current archive-curriculum cell count. | cells | rollout | history | last |
| `train/curriculum/archive/entry/count` | Curriculum archive entries | Current immutable entry count retained by the curriculum view. | entries | rollout | history | last |
| `train/curriculum/archive/admission/candidate/count` | Curriculum admission candidates | Non-terminal cell-crossing candidates observed during the rollout. | transitions | rollout | history | last |
| `train/curriculum/archive/admission/accepted/count` | Curriculum admissions accepted | Candidate entries accepted into cell reservoirs during the rollout. | entries | rollout | history | last |
| `train/curriculum/archive/evicted/count` | Curriculum cells evicted | Curriculum cells evicted during the rollout. | cells | rollout | history | last |
| `train/curriculum/archive/capture/call/count` | Curriculum capture calls | Batched portable provider-state capture calls during the rollout. | calls | rollout | history | last |
| `train/curriculum/archive/restore/episode/count` | Curriculum restore episodes | Archive-origin episodes started during the rollout. | episodes | rollout | history | last |
| `train/curriculum/archive/restore/forced_boundary/count` | Curriculum forced boundaries | Non-episode control truncations used to activate archive lanes. | boundaries | rollout | history | last |
| `train/curriculum/archive/feedback/trajectory/count` | Curriculum feedback trajectories | Completed archive-origin trajectories committed to the priority sampler. | trajectories | rollout | history | last |
| `train/curriculum/archive/transition/share` | Curriculum transition share | Fraction of policy transitions whose origin is the archive curriculum. | fraction | rollout | history | last |
| `train/curriculum/archive/sampling/probability/max` | Curriculum max sampling probability | Largest final cell probability in the archive sampler. | fraction | rollout | history | last |
| `train/curriculum/archive/sampling/effective/cell/count` | Curriculum effective cell count | Inverse-Simpson effective cell count of the archive sampling distribution. | cells | rollout | history | last |
| `train/curriculum/archive/capture/seconds` | Curriculum capture time | Portable state capture wall time accumulated during the rollout. | seconds | rollout | history | last |
| `train/curriculum/archive/restore/seconds` | Curriculum restore time | Provider restore wall time for reset calls containing archive lanes. | seconds | rollout | history | last |
<!-- METRIC_REGISTRY_END -->
