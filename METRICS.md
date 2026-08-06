# Metrics schema v18

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
- W&B config contains run-defining dimensions: `metrics_schema_version: 18`, `training_backend_id`,
  `training_backend_config_hash`, `algorithm_id`, goal,
  environment, starts, seed, frame skip, environment count, hyperparameters, eval protocol, and
  runtime versions.
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
- `gradlab/run/terminal_state`, `gradlab/run/stop_reason`, `gradlab/run/final_step`,
  `gradlab/run/early_stop_trigger`, and `gradlab/run/early_stop_condition` are W&B summary-only
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
- `orchestration/event_seq`: durable supervisor delivery order.

Each axis is configured with a W&B `max` summary reducer. W&B's public API may therefore expose
its summary value as a reducer mapping such as `{"max": 5046272}` rather than as a bare number.
Catalog and report consumers must unwrap the configured reducer value; a recipe's requested
`timesteps` cap is not a substitute for the observed `train/global_step`.
Before its first history row is logged, every concrete metric is explicitly bound to its applicable
scientific axis. W&B's internal `Step` is delivery order only and must not become the default X-axis
for scientific charts.

Asynchronous evaluations may arrive after later training rows without changing their scientific
X-axis. Each producer writes only its applicable scientific axis; durable delivery order uses
`orchestration/event_seq`.

Current runs declare schema v18, and the supervisor validates and emits only v18 names. GradLab
does not read, project, or preserve noncurrent W&B or R2 schemas.

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
  `train/outcome/success/across_starts/window_100/rate/min` first reaches one. For a single start,
  that means 100 consecutive genuine target-origin clears; for multiple starts, every configured
  start's latest 100 attempts must all clear. This training stop is not acceptance or promotion;
  explicitly evaluated Mario checkpoints rank by earliest `leader/checkpoint/step`, then highest
  `eval/full/episode/return/shaped/mean`. Breakout is training-only and ranks current-contract seeded
  recipe cohorts using `train/episode/return/shaped/from/target/rolling_up_to_100/mean`, which
  excludes archive-curriculum origins and non-episode control boundaries; tied cohorts prefer fewer
  policy transitions.
- Aggregate training `across_observed_starts/cumulative/rate/*` is cumulative. Aggregate
  `across_starts/window_100/rate/*` is the latest 100 attempts. Global
  window-100 min/mean appear only after every configured start has 100 attempts. Always pair early
  cumulative aggregates with `across_starts/coverage/rate`.
- A bounded training-only search may use per-start success counts and the history peak and first
  threshold crossing of `train/outcome/success/across_starts/window_100/rate/min` to screen and rank
  recipes.
  That evidence is not checkpoint evaluation and cannot establish checkpoint promotion, goal
  acceptance, or release evidence.
- Failure-reason metrics treat each reason as a per-episode presence flag, not an occurrence count
  or necessarily the event that ended the episode. For each reason, the window rate is the number
  of unsuccessful episodes containing that reason divided by all completed episodes in the latest
  up-to-100-episode window. Multiple reasons may belong to one episode, so reason counts and rates
  need not sum to the terminal count or one; successful episodes contribute zero to every
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
  `train/outcome/success/across_starts/window_100/rate/min` reaches one. Each has one configured
  start, so this requires 100 consecutive horizon-reaching training episodes; it is training
  success evidence, not checkpoint acceptance or promotion.
- Positive actor-critic policy entropy, dominant-action rate, and the action histogram diagnose
  discrete policy collapse. For a categorical `Discrete(n)` policy, entropy has infimum zero and
  maximum `ln(n)` nats at the uniform distribution; a `MultiDiscrete(nvec)` maximum is
  `sum(ln(nvec))`, while a constrained legal-tuple categorical policy has maximum
  `ln(legal_tuple_count)` regardless of the enclosing `nvec`; a `MultiBinary(d)` maximum is
  `d * ln(2)`. Continuous `Box` policies report differential entropy, which has no finite
  action-space-only minimum or maximum and must not use those discrete bounds. Value prediction
  and advantage histograms are sampled every 64 rollouts.
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
- `train/episode/return/shaped/from/target/window_100/mean` is emitted only after 100 genuine
  target-origin training episodes and then rolls over the latest 100. It is a mature online
  behavior-policy proxy whose episodes may span learner updates, not an estimate of one frozen
  checkpoint's evaluation performance. A zero-patience threshold matches on the first qualifying
  overlapping window; `action: stop` stops there, while `action: observe` continues training and
  reports the condition state. A threshold condition with `progress_baseline` additionally emits
  `train/early_stop/{condition}/target/progress` as the current metric's clamped fraction from that
  baseline to its threshold; because it consumes the watched metric, a mature rolling-window target
  has no progress value before its full window exists. Only goal-owned checkpoint evaluation may
  establish acceptance.
- Training episode reduction aggregates return, length, outcome, success, the explicitly supported
  target-origin cell-novelty statistic, and goal-declared numeric episode progress fields.
  `VizdoomDeathmatch-v1` declares `kills`, so
  `train/progress/kills/from/target/rolling_up_to_100/mean` reports mean native monster frags over up
  to the latest 100 genuine target-origin training episodes. Native shaped return is not an exact
  substitute because different monster kills can contribute different score values.
  `eval/full/progress/kills/{mean|max}` remains frozen-checkpoint evaluation evidence.
- Snapshot-curriculum `sampling/probability/max` and `sampling/effective_cell/count` summarize the
  current cell-probability distribution. They do not report realized per-cell selection frequency
  or identify which resident cells were selected.
- Derived throughput phase timing satisfies `loop wall time = env_step_seconds +
  rollout_overhead_seconds + between_rollouts_seconds`. Compare those three phase durations on
  matching workloads to identify a training-loop bottleneck. `rollout_overhead_seconds` includes
  policy inference plus wrapper, buffer, reset, task, and callback work outside the native provider.
  `between_rollouts_seconds` includes optimizer updates, callbacks, and logging, so it is deliberately
  not named optimization time.
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
  `train/outcome/success/across_starts/window_100/rate/min` reaches one. With its single configured
  start, this requires 100 consecutive perfect-score training episodes; it is training success
  evidence, not checkpoint acceptance or promotion.
- `VizdoomHealthGathering-Plus-v1` is a surface-variant identity over the regular
  `VizdoomHealthGathering-v1` task. Both classify the 2,100-native-tic horizon as success, stop at a
  mature window-100 success rate of one, use the same neutral return-plateau fallback stop, and
  require an evaluation success rate of at least 0.95 for acceptance.
- A ViZDoom success-rate target is success-based early stopping only when its `target_reached`
  condition has `action: stop`. Every ViZDoom goal with a binary success event now stops when
  `train/outcome/success/across_starts/window_100/rate/min` reaches one. This requires 100
  consecutive successful training episodes for each configured start. For
  `VizdoomDeathmatch-v1`, reaching the 4,200-native-tic horizon is a successful outcome while the
  episode boundary remains truncated so training bootstraps from the final observation. Its single
  configured start therefore stops after 100 consecutive horizon-reaching training episodes. This
  training success evidence does not replace its goal-owned mean-kills checkpoint acceptance rule.
- Episode-return means are neither a best-episode metric nor the score of a currently visible lane:
  they reduce the latest 100 completed episodes across all applicable vector lanes. W&B chart
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

`eval/full/by_start` has one row per start and observed failure reason with these columns:

`checkpoint_step`, `start_id`, `episode_count`, `success_count`, `success_rate`,
`shaped_return_mean`, `shaped_return_std`, `shaped_return_median`, `failure_reason`,
`failure_reason_count`, `failure_reason_rate`.
The reason value is empty, with zero count and rate, when a start has no recorded failure reason.

Episode-level evidence stays in R2. Confidence intervals and start-by-reason scalar products
are intentionally computed offline rather than added to W&B history.

An acceptance contract may reject fail-fast only when the first failed outcome proves its rule
cannot pass. That rejection is complete evidence of failure, but not a complete 100-episode
evaluation, so it emits no partial `eval/full/*` result. Aggregate contracts such as mean return
disable outcome-based fail-fast, run every planned episode, and emit complete `eval/full/*` metrics
for either verdict. W&B history always receives `eval/checkpoint/step`, pass, planned/completed
episodes, and acceptance duration. Accepted projections additionally include artifact, source, and
`eval/full/by_start`.
Per-start success and failure-reason summaries are derived from immutable private-R2 episode rows.
Duplicate full duration, artifact, source, and failure count stay in typed result or checkpoint
metadata rather than the W&B-shaped metric map.

`eval/acceptance/pass` is per-checkpoint history. W&B summarizes that history with `max`, so the
summary means that some checkpoint passed; it is not the run verdict. The authoritative verdict is
the create-only private-R2 `PromotionReceipt`, whose selected result is hash-bound to the complete
acceptance evidence. At terminal publication, that receipt restamps `gradlab/goal/outcome`, the diagnostic
`leader/checkpoint/*` fields, and the accepted W&B projection. Leader fields mirror only finite
selected metrics required by the configured rank plus available diagnostics; no generic objective,
serialized rank tuple, constant acceptance alias, or fabricated default is emitted. Later rejected
checkpoint projections remain in history and never modify the active projection. Raw acceptance
aggregates and episode evidence remain authoritative in private eval R2.

## Delivery, backpressure, and recovery

Every event has a stable content-derived event ID. Delivery to W&B is at least once; the durable
`orchestration/event_seq` is also W&B's internal step, so replay after an interrupted local
acknowledgement cannot append a second scientific point. Reports and summary projection retain the
event ID for explicit deduplication. Promotion, terminal state, and early-stop authority are exactly
once through conditional private-R2 receipts.

The supervisor seals immutable metric-journal segments to private R2 every five seconds or 1,000
events and batches pending frames to W&B. A retry reconstructs its local SQLite state from those
segments before producing new events. It resumes the same W&B run with `resume="must"`.

Backpressure is sampled every 15 seconds. The supervisor reports queue depth, oldest unpublished
age, ingress and publish rates, observed publication-capacity ratio, local/R2/W&B high-water marks,
remote-visible W&B lag, checkpoint backlog, pending evals, scratch utilization, accepted-result to
stop latency, and post-learner idle-GPU tail. Publication capacity is healthy only when measured
publish capacity is at least twice peak ingress.

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
| Metric or template | Meaning | Unit | Cadence | Surface |
|---|---|---|---|---|
| `train/episode/return/shaped/across_origins/rolling_up_to_100/mean` | Rolling mean shaped return over up to the latest 100 genuine completed training episodes across target and archive origins; an archive-origin return starts at restoration, and control boundaries are excluded. | return | rollout | history |
| `train/episode/return/shaped/across_origins/rolling_up_to_100/max` | Rolling maximum shaped return over the same episodes as the across-origin rolling mean; this is observed recent headroom, not a theoretical maximum. | return | rollout | history |
| `train/episode/length/across_origins/rolling_up_to_100/mean` | Rolling mean length over up to the latest 100 completed training episodes across target and archive origins. | steps | rollout | history |
| `train/exploration/cell/unique/from/target/rolling_up_to_100/mean` | Rolling mean episodic unique-cell count over up to the latest 100 completed target-origin training episodes when cell-novelty shaping is active. Includes the reset/start cell, which is recorded without receiving a bonus. | cells | rollout | history |
| `train/progress/{progress}/from/target/rolling_up_to_100/mean` | Rolling mean of a goal-declared finite numeric episode progress field over up to the latest 100 genuine target-origin training episodes, emitted beginning with the first such episode; archive-origin episodes are excluded. | progress units | rollout | history |
| `train/episode/completed/count` | Cumulative completed episode records. | episodes | rollout | history |
| `train/outcome/failure/reason/{reason}/episode/count` | Cumulative failed episodes containing a reason. | episodes | rollout | history |
| `train/outcome/failure/reason/{reason}/window_100/rate` | Unsuccessful completed episodes containing the reason divided by all completed episodes in the latest up-to-100-episode window; presence is boolean per episode, successful episodes remain in the denominator, and the reason need not be the terminal cause. | fraction | rollout | history |
| `train/outcome/success/from/{start}/episode/count` | Cumulative successful genuine target-origin episodes from a start; archive-origin episodes are excluded. | episodes | rollout | history |
| `train/outcome/success/from/{start}/attempt/count` | Cumulative genuine target-origin episode attempts from a start; archive-origin episodes are excluded. | episodes | rollout | history |
| `train/outcome/success/from/{start}/window_100/rate` | Success rate over the latest 100 genuine target-origin attempts from a start. | fraction | rollout | history |
| `train/outcome/success/across_observed_starts/cumulative/rate/min` | Minimum cumulative target-origin success rate across starts with at least one genuine attempt. | fraction | rollout | history |
| `train/outcome/success/across_observed_starts/cumulative/rate/mean` | Mean cumulative target-origin success rate across starts with at least one genuine attempt. | fraction | rollout | history |
| `train/outcome/success/across_starts/window_100/rate/min` | Minimum target-origin window-100 success rate after every configured start has 100 genuine attempts. | fraction | rollout | history |
| `train/outcome/success/across_starts/window_100/rate/mean` | Mean target-origin window-100 success rate after every configured start has 100 genuine attempts. | fraction | rollout | history |
| `train/outcome/success/across_starts/coverage/rate` | Configured starts with a genuine target-origin attempt divided by configured starts. | fraction | rollout | history |
| `train/early_stop/{condition}/value` | Current finite value consumed by a configured metric early-stop condition. | scalar | watched metric sample | history |
| `train/early_stop/{condition}/best` | Best value retained by a configured metric early-stop condition. | scalar | watched metric sample | history |
| `train/early_stop/{condition}/patience/elapsed_steps` | Policy steps elapsed in the condition's current patience interval. | steps | watched metric sample | history |
| `train/early_stop/{condition}/patience/progress` | Condition patience progress capped at one; one means the condition would trigger. | fraction | watched metric sample | history |
| `train/early_stop/{condition}/target/progress` | Current threshold-metric progress, clamped to zero through one, from the configured `progress_baseline` to the threshold in the operator's improving direction; emitted only for threshold conditions that declare a valid baseline. | fraction | watched metric sample | history |
| `train/reward/shaped/mean` | Distribution of learner-facing per-step reward after gradlab applies the task reward scale and then clipping. | scalar | rollout | history |
| `train/reward/shaped/std` | Distribution of learner-facing per-step reward after gradlab applies the task reward scale and then clipping. | scalar | rollout | history |
| `train/reward/shaped/min` | Distribution of learner-facing per-step reward after gradlab applies the task reward scale and then clipping. | scalar | rollout | history |
| `train/reward/shaped/max` | Distribution of learner-facing per-step reward after gradlab applies the task reward scale and then clipping. | scalar | rollout | history |
| `train/reward/shaped/nonzero_rate` | Fraction of learner-facing per-step rewards that are nonzero after gradlab applies the task reward scale and then clipping. | fraction | rollout | history |
| `train/reward/raw/mean` | Distribution of completed task reward immediately before gradlab-owned scaling and clipping, emitted when distinct from shaped reward. For an identity task this is the untransformed provider reward. | scalar | rollout | history |
| `train/reward/raw/std` | Distribution of completed task reward immediately before gradlab-owned scaling and clipping, emitted when distinct from shaped reward. For an identity task this is the untransformed provider reward. | scalar | rollout | history |
| `train/reward/component/{component}/mean` | Active reward-component attribution in pre-transform task-reward units. | scalar | rollout | history |
| `train/reward/component/{component}/nonzero_rate` | Fraction of active reward-component values that are nonzero. | fraction | rollout | history |
| `train/reward/component/{component}/share` | Absolute contribution share computed from components in pre-transform task-reward units. This W&B rollout aggregate is not the player's signed contribution percentage or its post-scale activity share. | fraction | rollout | history |
| `train/reward/signal/{signal}/mean` | Configured reward-source signal. | scalar | rollout | history |
| `train/reward/signal/{signal}/max` | Configured reward-source signal. | scalar | rollout | history |
| `train/reward/signal/{signal}/nonzero_rate` | Fraction of configured reward-source signal values that are nonzero. | fraction | rollout | history |
| `train/algorithm/ppo/update/approx_kl` | Approximate KL divergence for the PPO update. | scalar | rollout | history |
| `train/algorithm/ppo/update/clip_fraction` | Fraction of sampled policy ratios outside PPO's clipping interval and therefore clipped in the surrogate objective. This is an update-pressure diagnostic, not a target to maximize; interpret it with `approx_kl` and task return. | fraction | rollout | history |
| `train/algorithm/jerk/retained/count` | Distinct action sequences retained by JERK search. | sequences | rollout | history |
| `train/algorithm/jerk/best/return_mean` | Mean observed return of JERK's highest-ranked retained sequence. | return | rollout | history |
| `train/algorithm/jerk/best/sequence_length` | Action length of JERK's highest-ranked retained sequence. | steps | rollout | history |
| `train/algorithm/jerk/archive/selected_prefix_return_mean` | Cumulative mean retained-prefix return selected for JERK archive replay. | return | rollout | history |
| `train/algorithm/jerk/exploit/probability` | Probability that JERK starts an episode by sampling a retained archive sequence. | fraction | rollout | history |
| `train/algorithm/go_explore/archive/cell_count` | Semantic cells currently retained by Go-Explore. | cells | interval | history |
| `train/algorithm/go_explore/archive/entry_count` | Immutable entries in the ephemeral working archive, including replacements not yet removed by periodic compaction. | entries | interval | history |
| `train/algorithm/go_explore/archive/blob_count` | Distinct content-addressed provider-state blobs retained by the search. | blobs | interval | history |
| `train/algorithm/go_explore/archive/blob_bytes` | Uncompressed bytes in distinct retained provider-state blobs. The local TUI presents this as a low-overhead estimate of the archive's dominant memory footprint; it excludes Python metadata, allocator overhead, and provider-specific live-handle overhead. | bytes | interval | history |
| `train/algorithm/go_explore/archive/selection_count` | Cumulative archived-cell selections for restoration. | selections | interval | history |
| `train/algorithm/go_explore/archive/visit_count` | Cumulative semantic cell visits. | visits | interval | history |
| `train/algorithm/go_explore/archive/update_count` | Cumulative replacements of an existing cell by a better trajectory. | updates | interval | history |
| `train/algorithm/go_explore/archive/recent_new_cell_rate` | New semantic cells divided by visits in the bounded recent visit window. | fraction | interval | history |
| `train/algorithm/go_explore/archive/recent_visit_window` | Visit count represented by the recent new-cell-rate window. | visits | interval | history |
| `train/algorithm/go_explore/progress_guided/cell_count` | Cells on the current best-progress lineage used for guided restores before the first success. | cells | interval | history |
| `train/algorithm/go_explore/progress_guided/selection_count` | Cumulative restores selected from the best-progress lineage before the first success. | selections | interval | history |
| `train/algorithm/go_explore/success_guided/cell_count` | Cells on the current best-success lineage. | cells | interval | history |
| `train/algorithm/go_explore/success_guided/selection_count` | Cumulative selections made by success-lineage guidance. | selections | interval | history |
| `train/algorithm/go_explore/best/progress` | Greatest task progress reached by the best retained trajectory. | progress | interval | history |
| `train/algorithm/go_explore/best/return` | Shaped return of the best retained trajectory. | return | interval | history |
| `train/algorithm/go_explore/best/program_steps` | Environment steps in the best retained action program. | steps | interval | history |
| `train/algorithm/go_explore/best/program_runs` | Canonical action runs in the best retained action program. | runs | interval | history |
| `train/algorithm/go_explore/best/completed` | Whether the best retained trajectory completed the task. | boolean | interval | history |
| `train/algorithm/go_explore/best/improvement_count` | Successful-trajectory improvements after the first completion. | improvements | interval | history |
| `train/algorithm/{algorithm}/value/explained_variance` | Actor-critic value-function explained variance. | scalar | rollout | history |
| `train/algorithm/{algorithm}/update/policy_gradient_loss` | Actor-critic policy-gradient loss. | scalar | rollout | history |
| `train/algorithm/{algorithm}/update/value_loss` | Actor-critic value loss. | scalar | rollout | history |
| `train/algorithm/{algorithm}/update/learning_rate` | Current actor-critic learning rate. | scalar | rollout | history |
| `train/algorithm/{algorithm}/policy/entropy` | Positive actor-critic policy entropy. | nats | rollout | history |
| `train/algorithm/{algorithm}/policy/entropy_bound/lower` | Theoretical lower bound for finite discrete actor-critic policy entropy; always zero and omitted for continuous action spaces. | nats | rollout | history |
| `train/algorithm/{algorithm}/policy/entropy_bound/upper` | Theoretical maximum finite discrete actor-critic policy entropy at the uniform distribution, derived from the resolved policy-facing support (including only legal tuples for a constrained `MultiDiscrete` policy) and omitted for continuous action spaces. | nats | rollout | history |
| `train/algorithm/{algorithm}/policy/distribution_std` | Continuous-action distribution standard deviation. | scalar | rollout | history |
| `train/algorithm/{algorithm}/policy/dominant_action_rate` | Fraction assigned to the most frequent sampled discrete action. | fraction | rollout | history |
| `train/algorithm/{algorithm}/policy/action_hist` | Sampled discrete-action histogram. | histogram | every 64 rollouts | history |
| `train/algorithm/{algorithm}/rollout/value_prediction/mean` | Rollout value-prediction distribution diagnostic. | scalar | rollout | history |
| `train/algorithm/{algorithm}/rollout/value_prediction/std` | Rollout value-prediction distribution diagnostic. | scalar | rollout | history |
| `train/algorithm/{algorithm}/rollout/value_prediction/min` | Rollout value-prediction distribution diagnostic. | scalar | rollout | history |
| `train/algorithm/{algorithm}/rollout/value_prediction/max` | Rollout value-prediction distribution diagnostic. | scalar | rollout | history |
| `train/algorithm/{algorithm}/rollout/value_prediction/hist` | Rollout value-prediction histogram. | histogram | every 64 rollouts | history |
| `train/algorithm/{algorithm}/rollout/advantage/mean` | Rollout advantage distribution diagnostic. | scalar | rollout | history |
| `train/algorithm/{algorithm}/rollout/advantage/std` | Rollout advantage distribution diagnostic. | scalar | rollout | history |
| `train/algorithm/{algorithm}/rollout/advantage/min` | Rollout advantage distribution diagnostic. | scalar | rollout | history |
| `train/algorithm/{algorithm}/rollout/advantage/max` | Rollout advantage distribution diagnostic. | scalar | rollout | history |
| `train/algorithm/{algorithm}/rollout/advantage/hist` | Rollout advantage histogram. | histogram | every 64 rollouts | history |
| `train/algorithm/{algorithm}/hyperparameter/entropy_coefficient` | Current scheduled entropy coefficient. | scalar | rollout | history |
| `train/throughput/loop_fps` | Policy transitions divided by rollout-start-to-next-rollout-start wall time. | steps/second | rollout | history |
| `train/throughput/env_step_fps` | Policy transitions divided by native-provider step wall time accumulated during the rollout; emitted only when the provider exposes native step timing. | steps/second | rollout | history |
| `train/throughput/rollout_seconds` | Wall time spent collecting one rollout. | seconds | rollout | history |
| `train/throughput/env_step_seconds` | Native-provider step wall time accumulated while collecting one rollout. | seconds | rollout | history |
| `train/throughput/rollout_overhead_seconds` | Rollout wall time outside native-provider step calls, including policy inference and wrapper, buffer, reset, task, and callback work. | seconds | rollout | history |
| `train/throughput/between_rollouts_seconds` | Wall time after rollout collection and before the next rollout, including optimizer updates, callbacks, and logging. | seconds | rollout | history |
| `train/artifact/save/seconds` | Local model save duration. | seconds | artifact | history |
| `train/artifact/upload/seconds` | Public R2 checkpoint publication duration. | seconds | artifact | history |
| `eval/{protocol}/episode/return/shaped/mean` | Evaluation episode-return distribution accumulated from learner-facing rewards after gradlab-owned scaling and clipping. | return | evaluation | history |
| `eval/{protocol}/episode/return/shaped/std` | Evaluation episode-return distribution accumulated from learner-facing rewards after gradlab-owned scaling and clipping. | return | evaluation | history |
| `eval/{protocol}/episode/return/shaped/median` | Evaluation episode-return distribution accumulated from learner-facing rewards after gradlab-owned scaling and clipping. | return | evaluation | history |
| `eval/full/episode/return/shaped/max` | Maximum full-evaluation episode return accumulated after gradlab-owned scaling and clipping. | return | evaluation | history |
| `eval/{protocol}/episode/length/mean` | Mean evaluation episode length. | steps | evaluation | history |
| `eval/{protocol}/episode/completed/count` | Completed evaluation episodes represented. | episodes | evaluation | history |
| `eval/{protocol}/outcome/success/across_starts/rate/min` | Minimum evaluation success rate across represented starts. | fraction | evaluation | history |
| `eval/{protocol}/outcome/success/across_starts/rate/mean` | Mean evaluation success rate across represented starts. | fraction | evaluation | history |
| `eval/full/progress/{progress}/mean` | Goal-configured full-evaluation progress summary. | value | evaluation | history |
| `eval/full/progress/{progress}/max` | Goal-configured full-evaluation progress summary. | value | evaluation | history |
| `eval/acceptance/pass` | Per-checkpoint acceptance result; W&B summarizes its history with max, not as the verdict. | boolean | acceptance evaluation | history |
| `eval/acceptance/episode/planned/count` | Exact episode identities required by the acceptance manifest. | episodes | acceptance evaluation | history |
| `eval/acceptance/episode/completed/count` | Valid planned episode rows completed before acceptance or fail-fast rejection. | episodes | acceptance evaluation | history |
| `eval/acceptance/duration/seconds` | Acceptance-worker evaluation wall duration. | seconds | acceptance evaluation | history |
| `eval/full/by_start` | Structured full-evaluation evidence by start and reason. | table | evaluation | history |
| `leader/checkpoint/outcome/success/across_starts/rate/min` | Selected-checkpoint projection of minimum success rate across starts. | fraction | selection | summary |
| `leader/checkpoint/outcome/success/across_starts/rate/mean` | Selected-checkpoint projection of mean success rate across starts. | fraction | selection | summary |
| `leader/checkpoint/episode/return/shaped/mean` | Selected-checkpoint mean shaped episode return. | return | selection | summary |
| `leader/checkpoint/episode/return/shaped/max` | Selected-checkpoint maximum shaped episode return. | return | selection | summary |
| `leader/checkpoint/progress/{progress}/mean` | Selected-checkpoint mean for one named progress dimension. | value | selection | summary |
| `leader/checkpoint/progress/{progress}/max` | Selected-checkpoint maximum for one named progress dimension. | value | selection | summary |
| `leader/checkpoint/step` | Selected checkpoint policy step. | steps | selection | summary |
| `leader/checkpoint/artifact/ref` | Selected checkpoint immutable artifact reference. | metadata | selection | summary |
| `leader/checkpoint/evaluation/source` | Selected checkpoint evaluation source. | text | selection | summary |
| `leader/checkpoint/updated_at` | Selected checkpoint projection update time. | timestamp | selection | summary |
| `train/global_step` | Scientific training X-axis: policy environment transitions consumed. | steps | frame | history |
| `eval/checkpoint/step` | Scientific evaluation X-axis: step of the evaluated checkpoint. | steps | evaluation | history |
| `orchestration/event_seq` | Monotonic local outbox event sequence used as W&B delivery order. | events | frame | history |
| `orchestration/event_id` | Stable content-derived identifier used to deduplicate at-least-once delivery. | metadata | frame | history |
| `orchestration/outbox/queue_depth` | Metric outbox frames not yet acknowledged by the W&B SDK. | events | supervisor sample | history |
| `orchestration/outbox/oldest_unpublished_seconds` | Age of the oldest metric frame not yet acknowledged by the W&B SDK. | seconds | supervisor sample | history |
| `orchestration/outbox/ingress_rate` | Observed local metric-frame creation rate over the latest supervisor interval. | events/second | supervisor sample | history |
| `orchestration/outbox/publish_rate` | Observed W&B SDK acknowledgment rate over the latest supervisor interval. | events/second | supervisor sample | history |
| `orchestration/outbox/publication_capacity_ratio` | Observed W&B publication rate divided by observed metric ingress rate. | ratio | supervisor sample | history |
| `orchestration/outbox/local_high_water` | Largest metric-frame sequence durably present in local SQLite. | events | supervisor sample | history |
| `orchestration/outbox/r2_high_water` | Largest metric-frame sequence sealed in immutable private R2 journals. | events | supervisor sample | history |
| `orchestration/outbox/wandb_high_water` | Largest metric-frame sequence acknowledged by the W&B SDK. | events | supervisor sample | history |
| `orchestration/outbox/wandb_remote_high_water` | Largest orchestration event sequence observed through the W&B API. | events | remote visibility probe | history |
| `orchestration/outbox/wandb_remote_visible_lag_seconds` | Age of the newest local metric event not yet observed through the W&B API. | seconds | remote visibility probe | history |
| `orchestration/checkpoint/backlog` | Ready local checkpoints not yet verified in public model R2. | checkpoints | supervisor sample | history |
| `orchestration/eval/pending` | Persisted evaluation intents currently pending submission or awaiting a verified result; intents deferred after acceptance are excluded. | evaluations | supervisor sample | history |
| `orchestration/eval/result_to_stop_seconds` | Time from observing an accepted eval result to signaling the learner. | seconds | accepted evaluation | history |
| `orchestration/drain/idle_gpu_tail_seconds` | Time the training container retained its GPU after the learner exited. | seconds | terminal drain | history |
| `orchestration/scratch/used_fraction` | Fraction of the task scratch filesystem currently used. | fraction | supervisor sample | history |
| `train/episode/return/shaped/from/target/rolling_up_to_100/mean` | Rolling mean shaped return over up to the latest 100 genuine target-origin training episodes, emitted beginning with the first such episode. | return | rollout | history |
| `train/episode/return/shaped/from/target/window_100/mean` | Mature online behavior-policy mean shaped return over the latest 100 genuine target-origin training episodes, emitted only once the full window exists; this is a mean return in policy-facing reward units, not a success probability, so a threshold such as `0.95` requires mean return `>= 0.95`; comparable to evaluation return only in reward contract and units, not as a frozen-checkpoint estimate. | return | rollout | history |
| `train/episode/return/shaped/from/target/rolling_up_to_100/max` | Rolling maximum shaped return over the same genuine target-origin training episodes as the warm-up rolling mean; archive-origin episodes are excluded. | return | rollout | history |
| `train/curriculum/archive/cell/count` | Current archive-curriculum cell count. | cells | rollout | history |
| `train/curriculum/archive/entry/count` | Current immutable entry count retained by the curriculum view. | entries | rollout | history |
| `train/curriculum/archive/admission/candidate/count` | Non-terminal cell-crossing candidates observed during the rollout. | transitions | rollout | history |
| `train/curriculum/archive/admission/accepted/count` | Candidate entries accepted into cell reservoirs during the rollout. | entries | rollout | history |
| `train/curriculum/archive/evicted/count` | Curriculum cells evicted during the rollout. | cells | rollout | history |
| `train/curriculum/archive/capture/call/count` | Batched portable provider-state capture calls during the rollout. | calls | rollout | history |
| `train/curriculum/archive/restore/episode/count` | Archive-origin episodes started during the rollout. | episodes | rollout | history |
| `train/curriculum/archive/restore/forced_boundary/count` | Non-episode control truncations used to activate archive lanes. | boundaries | rollout | history |
| `train/curriculum/archive/feedback/trajectory/count` | Completed archive-origin trajectories committed to the priority sampler. | trajectories | rollout | history |
| `train/curriculum/archive/transition/share` | Fraction of policy transitions whose origin is the archive curriculum. | fraction | rollout | history |
| `train/curriculum/archive/sampling/probability/max` | Largest final cell probability in the archive sampler. | fraction | rollout | history |
| `train/curriculum/archive/sampling/effective_cell/count` | Inverse-Simpson effective cell count of the archive sampling distribution. | cells | rollout | history |
| `train/curriculum/archive/capture/seconds` | Portable state capture wall time accumulated during the rollout. | seconds | rollout | history |
| `train/curriculum/archive/restore/seconds` | Provider restore wall time for reset calls containing archive lanes. | seconds | rollout | history |
<!-- METRIC_REGISTRY_END -->
