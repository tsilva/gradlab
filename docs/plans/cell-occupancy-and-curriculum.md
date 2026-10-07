# Cell occupancy and archive curriculum

Published implementation specification: [GitHub issue #41](https://github.com/tsilva/gradlab/issues/41), labeled `ready-for-agent`.

Design agreed through interview, 2026-09-11. Requested outcome: inspect where training experience is collected throughout a Run in W&B, using environment-defined cells, and later reuse those cells and archived states to allocate practice to underexposed regions. The user accepted all recommendations in questions Q1 through Q21. This records the agreed design and remaining engineering proposals; it does not authorize a training launch or change an authoritative specification.

## Confirmed decisions

| Interview questions | Agreed decision |
| --- | --- |
| Q1, Q4, Q11 | Deliver passive tracking first. Validate on Breakout and Mario, using level identity and position buckets for Mario. Add device-native GraDOOM support later. |
| Q2, Q3 | Share cell definitions and machinery, with explicit coarse views of finer cells. Declare dimensions and buckets before launch. Historical merging is supported; arbitrary retrospective rebucketing is deferred. |
| Q5, Q7 | W&B is the complete initial interface, including historical selection. Default to 100,000-transition windows, configurable before launch, with cumulative totals and explicit partial final windows. |
| Q6 | Breakout uses wall number, six-brick bands, and paddle width, with zero remaining bricks separate. |
| Q8, Q9 | Count every collected pre-action state, including waiting, repetitive play, and terminal transitions. Exclude reset machinery and optimizer reuse. Include cell-entry counts; defer per-cell rewards and learning errors. |
| Q10 | Keep counting exact and target at most 2% throughput overhead. Optimize or narrow declared support if that target is missed; do not silently sample. |
| Q12, Q13 | Default to combined occupancy with normal-start and archive-start views alongside it. Show configured zero-exposure cells as unobserved and distinguish compatible representative availability from reachability. |
| Q14 | Resume cumulative counts only with matching collector state. Otherwise start a visibly separate collection segment and preserve earlier history without fabricating continuity. |
| Q15, Q16 | Retain several bounded curriculum representatives through the existing reservoir machinery while preserving Go-Explore's own representative selection. Begin with current-run states and recovery across its attempts; defer cross-run imports. |
| Q17, Q18 | Favor underexposed restorable cells using recent combined experience. Archive practice counts toward reducing its own exposure deficit. Do not enforce uniform occupancy. |
| Q19, Q20 | Begin adaptive training with fixed 20% archive lanes and 80% normal-start lanes. Preserve continuation semantics, including elapsed episode time and remaining limits. |
| Q21 | Judge curriculum benefit by improvement on the original task from normal starts at matched training budgets. Report wall-clock cost; a more balanced heatmap alone is insufficient. |

## Recommendation

Build a shared cell assignment and occupancy module underneath telemetry and archive consumers. Keep measurement usable without snapshots, reward shaping, or curriculum activation. Extend the existing archive curriculum for coverage-based selection rather than introducing a second state store or reset mechanism.

Ship in three usable increments: occupancy and W&B inspection; bounded representative capture; then explicitly configured adaptive starts. Do not silently enable the latter when tracking is enabled.

## Existing code to reuse

| Existing implementation | Reuse and required change |
| --- | --- |
| `src/gradlab/state_archive.py`: `ArchiveCellConfig`, `ArchiveCellDimension`, `ArchiveCellDetector` | Shared declarative scalar bucketing already supports semantic signals, provider sources, bucket sizes, clamps, and equality categories. Extract a neutral cell module and add batch integer coordinates. Preserve existing serialized keys. |
| `src/gradlab/batch_runtime.py`: signal resolution, reset handling, cell detection | Own cached current-state cell assignments and collection counts here, before reset/restore state replaces transition state. This covers the ordinary vector runtime independently of the learning algorithm. |
| `src/gradlab/state_archive.py`: `StateArchive`, codecs, immutable entries, blob store, views | Reuse provider/task/runtime snapshot closure, compatibility checks, content addressing, retention, and export. Cell counts are not snapshots and must survive archive eviction. |
| `ArchiveCurriculum` and `BatchRuntime._capture_curriculum_candidates` | Existing per-cell reservoirs, lane allocation, cell-crossing capture, deterministic sampling, probability caps, and reset attribution are the starting point for admission and adaptive starts. Capture currently precedes admission, so rejected candidates can still incur snapshot work. |
| `src/gradlab/training/ppo_engine.py` and `sb3_on_policy.py` | Reuse rollout lifecycle and archive-origin attribution. Existing `value_error` feedback is mean absolute raw GAE over a completed archive-origin trajectory, not an occupancy counter. |
| `src/gradlab/go_explore.py`, `training/go_explore.py`, `cell_graph.py` | Reuse cell identity, snapshot storage, route graph, and restore operations. Preserve Go-Explore's own representative ranking and search selection. |
| `src/gradlab/task_kernels.py`: cell novelty | Another existing consumer of the same detector. Preserve its episodic novelty and reward behavior during extraction. |
| `src/gradlab/wandb_publisher.py`, metric registry, managed workspace declarations | Extend the existing SQLite outbox and sole-supervisor W&B writer with bounded aggregate cell tables and panels. No learner network calls. |

The existing Breakout PPO archive recipe uses score buckets and 20% archive lanes. Go-Explore uses wall progress, remaining bricks, ball and paddle position, and velocity direction. Neither exposes the requested time-resolved per-cell occupancy distribution.

Go-Explore's current visits are not interchangeable with occupancy. Initialization counts starting cells; subsequent observations count nonterminal destination cells. Archive curriculum admission counts cell-crossing candidates. Neither counts every pre-action state exactly once.

GraDOOM has a separate device runtime with device-side signals and currently rejects state archives. A shared Python batch hook alone does not cover it. Tracking capability and snapshot capability must be registered and tested independently.

## Cell definition and identity

Author a cell space once and let tracking, archive admission, and curriculum configuration refer to it. Named spaces should resolve to the normalized existing detector document and a semantic hash. Existing embedded definitions must normalize through the same implementation during the source refactor; do not add legacy runtime fallbacks.

Keep display labels separate from identity. Identity includes selectors, signal semantics and units, dimension order, binning rules, and cell schema version. Namespace stored keys by the resolved cell-space and environment contracts. Two arrays such as `[1, 2]` are not the same cell merely because their serialized coordinates match.

For Breakout, start with:

- `walls_cleared`, so a fresh wall cannot be confused with the first wall;
- `bricks_remaining`, in six-brick bands with zero separate;
- actual paddle width, using the validated provider field and declared units.

The current goal does not declare a paddle-width task signal, although structured observation recipes expose width. Bind an appropriate tracking selector explicitly and validate its presence on both reset and step. Do not infer width from brick count or silently alter Policy inputs.

Zero separation must be explicit in the shared definition. Existing floor division with a bucket size of six merges zero with values one through five, so that existing rule alone does not implement the agreed stage view. Preserve existing Go-Explore key semantics while adding the required stage representation.

Freeze tracked dimensions and bucket boundaries before launch. Historical views may merge recorded cells and sum their counts, but cannot split a stored bucket or add a previously unrecorded dimension. Arbitrary retrospective rebucketing and raw per-transition state logging are outside the initial scope.

These are coarse stage cells. A cell can retain several full snapshots with different brick arrangements, ball directions, and paddle positions. A cell ID cannot reconstruct any of those details.

Allow Go-Explore to retain its finer search cell space. Prefer stage views that are exact projections of those dimensions where possible. If a stage dimension such as paddle width is absent from a search key, calculate both from the shared resolved signal batch and store the stage label with captured entries. Never infer a missing dimension from an old graph. Sharing the detector and named definitions does not require degrading the search partition.

V1 dashboard spaces must have a finite declared domain and at most 256 cells. Proposed domain bounds are validation rules, not automatic clamps. Keep clamping explicit because it merges out-of-range states. Fine or unbounded archives can continue using their existing representation; report a bounded stage projection. This avoids both metric-key explosion and silently dropping rare cells from the display.

## Exact collection semantics

For window W, cell c, and episode origin o:

`count[c,o,W] = number of collected transitions whose pre-action state maps to c and whose origin is o`.

`fraction[c,o,W] = count[c,o,W] / all collected transitions of origin o in W`.

When that origin has no transitions, its fraction is unavailable, not zero. The combined fraction uses all transitions in the window as its denominator. Counts across cells must sum to the declared denominator.

Count every environment transition once, including actions ending in termination or truncation. Do not count reset operations, state restores, snapshot reads, PPO optimization epochs, evaluation episodes, or playback as training occupancy. Runtime reset no-ops are not policy transitions. This is occupancy of the collected data, not optimizer sample reuse or simulated native frames.

Waiting to serve and repetitive play remain included. Any optional filtered display must retain access to the unfiltered population and denominator rather than making that expenditure disappear.

Initialize each lane's cached cell from reset information without incrementing a count. After an executed step, attribute the transition to that cached source cell, then compute the next assignment from the real destination state. After reset or restore, replace the cached assignment only for affected lanes. Copy compact coordinates before borrowed provider arrays can be mutated. Masked resolution must ignore inactive lanes rather than demand valid values from absent reset columns.

For each cell retain occupancy counts and observed segment-entry counts. An entry occurs at the first counted transition after a reset/restore or a change of cell. Carry the previous-cell marker across windows and rollouts; a reporting flush is not a new visit. Separate natural cell crossings from reset/restore entries when archive sampling is enabled. Avoid calling these “distinct episodes reaching the cell” without implementing a separate per-episode seen set.

Keep the following separate:

| Quantity | Meaning |
| --- | --- |
| Occupancy | Transitions collected while in a cell |
| Entry count | New contiguous visits to that cell |
| Retained representatives | Full restorable snapshots currently available |
| Start selections | How often the sampler chose that cell |
| Start probability | The sampler's intended distribution |

Many transitions can come from one long stall. Frequent entries can be a loop. Neither establishes competence. Optional later event counts and progress deltas must use declared task semantics and their own metric definitions.

## Time windows and persistence

Use nonoverlapping transition windows as the canonical history. The agreed default is 100,000 policy transitions, configurable before launch. Round up once to a whole vector batch and store the resolved window size, actual start/end steps, denominator, and partial status. Close at batch boundaries, independently of rollout size; never assign an entire rollout to the window in which it ended.

The user should see recent windows first and cumulative counts as a second view. Rollout summaries can be derived for learner feedback without making rollout length the dashboard's time unit. A smooth recent view can merge adjacent windows with count-weighted denominators. Do not average percentages with unequal denominators or describe block windows as an exact sliding window.

Flush completed windows through the learner's outbox on existing logging opportunities. Preserve partial windows at orderly termination. A crash must show a documented uncovered interval if its latest uncommitted counts cannot be recovered; never fill that gap with zeros.

Each aggregate carries Run, Attempt, collection segment, cell-space hash, window sequence, step interval, origin, and counts. Persist reducer state and the last committed window ID with a consistent recovery cursor. Restore counters only when that cursor matches the learner recovery point. A model-only continuation begins a new collection segment even when its global step continues. Do not double-count replayed step ranges or invent historical counts from the global step counter. This extends the existing outbox recovery contract and needs focused recovery tests.

## W&B experience

The main panel is a heatmap: training step on X, ordered game-stage cell on Y, occupancy fraction as color. Show wall and paddle-width groups in readable order. Hover reveals raw count, denominator, interval, origin, and cell definition. Preserve zero-exposure bins in the declared domain and label them unobserved. Distinguish missing data and compatible archived representative availability. An empty cell is not evidence that it is reachable or needs more training.

Add a selected-window bar chart and a small fixed set of stage-share curves for easy run comparison. Default to combined experience, with normal-start and archive-start views immediately alongside it and each population and denominator labeled. For Go-Explore, identify search-restored trajectories explicitly rather than labeling them as independent normal-start episodes. A fixed provenance mapping belongs in the collector interface.

Combined occupancy answers what the learner trained on. Normal-start occupancy answers where the current training policy spends time when starting normally. Archive-start occupancy shows the additional practice supplied by the curriculum. Once a curriculum is active, a rise in combined late-stage exposure does not prove better natural reachability.

Proposed table payload: one row per cell and origin, with coordinates, count, entry count, fraction, cumulative count within the collection segment, window bounds, and completeness. Counts are aggregated scientific measures, not raw state or per-episode diagnostic payloads. Register a bounded table metric and only a small fixed set of scalar summaries. Do not make arbitrary cell IDs into metric paths, or expand counts into repeated samples for a histogram.

W&B supports tables at multiple history steps and custom Vega charts over table/history data. The supervisor should construct SDK table objects from validated JSON outbox payloads. Prototype and verify historical window selection and the cross-time heatmap before finalizing the payload. A latest-table bar chart alone does not meet the user's request. Keep each logged window payload bounded and avoid re-uploading the entire growing history.

If the table-history query cannot support the desired heatmap efficiently, use a fixed bounded set of per-stage scalar series or a bounded overview table. Any overview downsampling must add raw counts and denominators together. Validate the actual W&B view in the Codex in-app browser during implementation, including historical selection and multi-run cell-contract filtering.

## Efficient implementation

Compile cell definitions once. Resolve only required scalar columns once per batch and share the results across consumers with matching selectors. Use vectorized floor, equality, and clipping operations to generate integer coordinates. For bounded spaces, map coordinates to dense indices and accumulate with `bincount` or an equivalent integer reduction. Keep portable string keys at archive, graph, and serialization interfaces rather than formatting JSON for every lane on every step.

The collector interface should own configuration validation, cached lane assignments, count reduction, window closure, and export/recovery state. It accepts reset/restore notifications and executed transition batches. Consumers request immutable window summaries; they do not coordinate raw counter arrays themselves.

Memory for one bounded view is proportional to cells times origins plus lanes times dimensions, independent of run duration. With 256 cells and two origins, two int64 occupancy arrays for window and cumulative counts use 8 KiB. Entry counters and lane assignments add a small bounded amount. This is an arithmetic estimate, not a throughput result.

No frame copies, snapshots, filesystem writes, or network calls belong in the counting path. Snapshot cost must be measured separately. For device-native environments, perform assignment and reduction on the simulator device and transfer aggregates at reporting boundaries; do not add a per-step `.cpu()` or `.item()` synchronization. Initially reject unsupported tracking configurations through the capability registry until the device adapter passes the same semantic tests.

Agreed tracking-only performance target: at most 2% median throughput regression against disabled tracking on the same representative workload, with repeated measurements and exact count verification. If missed, optimize or narrow declared support before considering a separately agreed sampled mode. Exact counts remain the contract. Do not choose hardware or alter training concurrency as part of this planning task.

## Representative capture

Tracking does not require archiving. Add capture as a separate enabled consumer when valid snapshots are needed. Refactor admission so the reservoir can decide which candidates need capture before the provider serializes them; batch selected lanes in one capture call and commit only successful captures. Preserve deterministic candidate ordering and sampling randomness independently of telemetry.

Reuse several bounded representatives per curriculum cell and existing compatibility checks. Reservoir sampling broadens the retained sample beyond one best-scoring state but does not guarantee geometric diversity. Preserve Go-Explore's own representative ranking. New nonterminal cell crossings are useful capture candidates. A low-frequency within-cell candidate stream may later improve diversity, but keep its budget explicit. Crossing-only reservoirs represent cell-entry states, not uniform samples of all occupied states. Track that admission policy in provenance.

Cap entries and bytes, protect states in active use, and reuse archive garbage collection so rejected or evicted entries do not grow storage indefinitely. Existing `max_entries` limits the curriculum view; verify physical blob retention separately. Losing an archive representative must not erase the cell's occupancy history.

Continue preserving provider state, observation history, task state, episode timing, and runtime attribution through existing restore semantics. Recurrent Policy state, if supported later, needs an explicit restore contract. Do not assume a simulator snapshot alone is a complete learning start. Preserve the existing continuation behavior initially, including remaining time limits; a fresh-horizon curriculum would be a separate scientific variant.

Begin with states discovered within the current Run and preserve recovery across its Attempts. Go-Explore checkpoints must still include their portable cell-and-route graph. Archive snapshot export remains opt-in under `docs/specs/go-explore.md`. A graph-only checkpoint cannot provide arbitrary restoration for a later curriculum. Cross-run archive import requires matching contracts and explicitly retained snapshots; it is deferred.

## Adaptive starts

Extend `ArchiveCurriculum` with an explicit coverage selection strategy alongside its existing value-error strategy. Use a fixed 20% archive lane share and 80% normal-start lane share for the first experiment, with the resolved lane counts recorded. Preserve cold-cell handling, deterministic sampling, representative selection, and probability caps. A cell is eligible only when it has a compatible restorable nonterminal representative. A never-discovered cell cannot be selected into existence.

The agreed objective is to favor underexposed restorable cells without enforcing uniform occupancy. Proposed first strategy: smoothed inverse recent occupancy over eligible cells. For example, use weights proportional to `(epsilon + recent_count[cell]) ** -beta`, then apply the existing cap and a small exploration floor. Record epsilon, beta, history horizon, eligible-cell set, and update cadence in the curriculum configuration. The agreed feedback population is recent combined collected experience so archive practice counts toward satisfying its own exposure deficit. Maintain normal-start counts separately for diagnosis. The exact formula and tuning values remain engineering proposals to validate before the adaptive phase.

Freeze the selection distribution for each rollout and update from committed prior collection counts. Keep measurement and sampler inputs local; W&B latency or outage must not affect training. Do not reuse the string `priority_metric: value_error` for a coverage score. Register any published probability or coverage metric separately from internal strategy identifiers.

Uniform occupancy is not automatically the desired objective. Cells differ in difficulty, useful dwell time, and reachable volume. Inverse occupancy is an experiment baseline, not an optimal curriculum. Later allow declared target exposure shares or a separately evaluated mixture with the existing learning-potential score. Do not begin with a controller that simultaneously changes resets, reward shaping, entropy, and failure penalties.

Selecting a start distribution does not directly set occupancy: episodes leave their start cells, spend different amounts of time elsewhere, and sometimes fail immediately. Log both intended starts and realized occupancy. Smooth updates and retain a nonzero normal-start lane allocation to limit oscillation and retain access to the goal's real start distribution.

After restoring a state, collect fresh actions, log probabilities, values, and transitions under the current behavior Policy. Do not place old archive trajectories directly into PPO's rollout buffer. Starts now follow a curriculum mixture, so this changes the training distribution even though the collected actions are current-policy data. Keep evaluation on the declared goal starts. Changes to start-state semantics require the corresponding Goal Variant and provenance under existing root requirements.

## Implementation sequence and evidence

1. Extract and test the shared cell encoder. Preserve Go-Explore graph keys, novelty rewards, and archive behavior. Add cell-space references, bounded reporting-domain validation, and separate tracking/restore capability checks. Prove no enabled consumer changes behavior from the extraction alone.
2. Add the passive collector to the ordinary vector runtime and its backend logging integration. Validate first on Breakout and Mario, with level identity and position buckets for Mario, and declare tested backend/provider combinations explicitly. Add GraDOOM's device adapter later with the same semantic tests and device-side performance checks before advertising support there. Deliver occupancy and entries with no capture calls.
3. Add the registered table transport, supervisor conversion, and managed W&B panels. Include interval identity, origin, zeros, partial windows, and collection segments. Verify historical data and recovery behavior, not only the latest chart. Update `METRICS.md`, metric constants, inventory, publisher tests, and workspace applicability in the same implementation change.
4. Extract archive admission from reset selection so capture can run with restoration disabled. Add pre-capture admission and bounded retention. Verify snapshot round trips and storage budgets without altering Policy actions.
5. Add the coverage sampler as an explicit curriculum strategy. Use scripted trajectories to verify eligibility, probability caps, determinism, origin attribution, and delayed feedback. Preserve the existing value-error strategy and Go-Explore selection.
6. Run a separately authorized controlled comparison: passive tracking; capture enabled with restores disabled; then coverage-based restores with the same cell contract and training budget. Require better original-task performance from normal starts to claim curriculum benefit, and report wall-clock cost alongside realized stage exposure. Redistribution alone is insufficient. Use multiple seeds and preserve goal-owned evaluation and Acceptance rules; training-only goals do not acquire Acceptance authority from this comparison.

Required tests include source-versus-destination attribution; terminal transitions; auto-reset and masked restore; lane isolation; zero-transition origins; windows crossing rollouts; final partial windows; unchanged counts across optimizer epochs; exact totals; deterministic replay of delivery; archive eviction preserving counts; incompatible space hashes; inactive-lane missing fields; and no counting-induced change to observations, rewards, actions, or RNG streams.

Existing worktree changes include gamma scheduling, training, publication, playback, and metrics. Implementation must inspect and preserve them. This planning task changes no learner or runtime code.

## Specification fit and remaining engineering work

Root `SPECS.md` already requires declared curricula, start-state contracts, diagnostic isolation, provenance, and valid capabilities. No root change is proposed. The Go-Explore scoped requirements remain intact. The interview decisions above are agreed scope and design choices for this capability; they must not be broadened into project-wide requirements. No authoritative specification is edited by this plan revision.

The product interview is complete. Remaining engineering work includes proving W&B historical rendering, expressing zero-separated brick bands in the shared encoder, validating provider fields and Mario level identity, measuring overhead, and selecting bounded sampler and retention settings before their implementation phase. The 256-cell display limit and exact sampler formula remain implementation proposals, not interview-approved numeric requirements. An early-game majority still needs interpretation through entries, dwell time, progress, and original-task performance; there is no agreed uniform occupancy target.

References informing the plan:

- [W&B custom charts](https://docs.wandb.ai/models/app/features/custom-charts) describes table history and custom visualizations; the exact dashboard interaction still needs an implementation spike.
- [W&B logging performance guidance](https://docs.wandb.ai/models/track/limits) motivates bounded tables and avoiding arbitrary per-cell metric keys.
- [First return, then explore](https://www.nature.com/articles/s41586-020-03157-9) describes remembering states and returning before further exploration.
- [Prioritized Level Replay](https://proceedings.mlr.press/v139/jiang21b.html) motivates adaptive practice based on learning potential. The proposed occupancy strategy is a GradLab design hypothesis, not a claimed reproduction of that algorithm.
