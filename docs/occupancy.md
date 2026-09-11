# Collected cell occupancy

Occupancy records the state before every executed training transition. It is passive
unless an archive recorder and restoration are separately enabled. It does not modify
policy inputs, rewards, evaluation, Acceptance, or Promotion.

## Define a reporting space

These are `train` fields in a recipe, or equivalent resolved launch overrides. This
Breakout example distinguishes actual paddle geometry, walls, and six-brick bands.
The wall clamp deliberately groups wall three and later. Remove it and declare a
larger finite domain if those walls must remain separate.

```yaml
train:
  cell_spaces:
    breakout_stage:
      cell:
        dimensions:
        - {source: walls_cleared, bucket_size: 1, clamp: [0, 3]}
        - {source: bricks_remaining, bucket_size: 6, zero_separate: true}
        - {source: paddle_width, bucket_size: 1}
      domains:
      - [0, 1, 2, 3]
      - [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
      - [8, 16]
      units: [walls, bricks, pixels]
  occupancy:
    space: breakout_stage
    window_transitions: 100000
```

Zero-separated buckets map zero to zero, one through six to one, and seven through
twelve to two. Ordinary buckets retain floor division. The provider's raw
`paddle_width` supplies pixel width; the normalized policy context is a different
quantity. Tracking requests the additional provider columns without adding policy inputs.

For Mario Level1-1, the provider's `levelHi` and `levelLo` are zero-based world and
level indices. The task's `x` signal combines `xscrollHi` and `xscrollLo` into pixels.
A suitable definition has dimensions `source: levelHi` with size one, `source:
levelLo` with size one, and `signal: x` with size 256. Declare domains `[0]`, `[0]`,
and integers zero through 31, with units `world index`, `level index`, and `pixels`.
Other starts need their own declared level domains.

Dimensions, their order, units, environment contract, domains, and bucketing rules
identify a measurement contract. Optional unique `labels` follow the Cartesian
product of the domains. Labels do not affect the contract hash. The current reporting
limit is 256 cells, an engineering limit. Out-of-domain states fail explicitly;
only authored clamps combine them. Fine Go-Explore spaces can omit reporting domains.
Named definitions also resolve in `state_archive.recorder.cell` and in semantic-only
`task.reward.cell_novelty.cell`.

## Read W&B history

Managed Breakout and Mario workspaces include recent and historical occupancy panels,
including for runs enabled through launch overrides. Open a panel full screen for
larger domains. Each table contains up to eight consecutive windows from one segment.
The recent panel shows the latest page. The historical panel starts at publication
index zero, which can be empty when another metric was logged there; choose an
index containing an occupancy table.
In the historical panel, choose **Edit panel → Query → index** to select any recorded
table using W&B's history index. This is a publication-history coordinate, not a
training-step value; other logged events can occupy intervening indices. The
synthetic validation's indices 8 and 120 show complete early and late pages.
A retry starts a new page; inspect its bounds and segment in
the hover values. Every table version is addressable, including versions beyond 100.
The default unindexed W&B history query samples versions and is deliberately unused.

The main Breakout workspace also has **Total experience by bucket**, showing each
cell's cumulative count divided by the cumulative denominator in the latest
collection segment. Runs, segments, and cell contracts stay separate. Hover a bar
to see exact counts. Runs without occupancy enabled contribute no data.

Click a window to select its distribution. Under **Edit panel → Chart fields**, set
`origin` to `combined` (the default), `normal`, `archive`, or `search`. Set `grouping`
to `full`, `first dimension`, or `first two dimensions`. Coarser views add exact
counts and retain the original denominator. They cannot recover distinctions absent
from the recorded definition.

Each origin retains its unfiltered denominator. Empty origins have unavailable
fractions. Unobserved declared cells remain visible with zero counts. Hover shows
entries, counts, denominators, cumulative totals, bounds, completeness, segment,
and contract identity. Gaps remain blank, with their uncovered interval in metadata.
A partial final window uses its actual transition count.

An entry is the first collected transition after a reset, restore, or cell crossing.
Window and rollout boundaries create no entries. Episode origin remains normal,
archive, or search even after moving away from the starting cell. Search restores
are distinct from independent normal-start experience.

The collector owns fixed-size counters and immutable summaries. No snapshots, disk
writes, or network requests occur in passive counting. Existing learner logging
boundaries append validated summaries to the SQLite outbox. Only the supervisor
constructs W&B tables. Each upload contains at most eight windows (8,192 rows at the
256-cell limit), without a growing history payload or per-cell metric paths.
Deterministic identities reject conflicting
replays and overlapping intervals within a segment.

Model-only continuation begins a new segment. Earlier history remains available;
its totals are never silently carried into fresh environment state. The runtime's
collector export/restore interface requires a matching learner recovery cursor for
continuous recovery. Current model-only learners deliberately use new segments.

## Capture representatives without restoration

Reuse the same reporting definition and enable capture independently:

```yaml
train:
  state_archive:
    persistence: durable
    restore_semantics: continuation
    recorder:
      mode: cell_transition
      cell: breakout_stage
    curriculum:
      archive_share: 0.2
      priority_metric: value_error
      restore_entries: false
      entries_per_cell: 4
      max_entries: 1024
      max_bytes: 1073741824
```

The reservoir decides which candidates survive before the provider serializes them.
Selected lanes are captured in one call. Capture RNG is independent of the policy
and telemetry. Representatives are cell-entry states; reservoir sampling does not
promise geometric diversity or uniform occupancy samples.

Retention bounds immutable entry files and unique snapshot blobs. View and closure
metadata are separately bounded by the retained inventory. A capture batch can
transiently allocate candidate storage before the budget check. Oversized or evicted
candidates are reclaimed before publishing the next archive generation. Active cells
and cells awaiting trajectory feedback protect their representatives. Coverage cells
in the frozen rollout distribution are also protected; a full protected budget
defers admission. At a rollout boundary, one inactive cell can be retired so later
crossings can enter the next inventory without changing the active distribution.
Eviction does
not change occupancy history. Go-Explore's separate best-representative and route-graph
contracts remain unchanged, and snapshot export remains opt-in.

A curriculum view in the existing portable archive preserves the reservoir and
compatible representatives across attempts of the same Run. The existing supervisor
publishes recoverable archive generations at rollout boundaries. A shared publication
lock protects the complete generation during upload; mutation and pruning take an
exclusive lock and invalidate the previous local closure first. Capture can wait
for an in-progress upload, without retaining a second unbounded snapshot inventory.
After publication, the Run writer reclaims remote recovery objects and generations
not referenced by the current generation. Recovery holds the same exclusive writer
lease, and separately owned checkpoint exports are outside this cleanup prefix.
Restores preserve
provider state, observation history, task state, episode time, and remaining limits.
No cross-run import or graph-only restoration is introduced.

## Enable coverage starts

Set `restore_entries: true` and `strategy: coverage` in that curriculum. Coverage
requires the archive and occupancy definitions to match exactly. Its initial
`archive_share` is fixed at 0.2; the existing rounding rule records the actual lane
count and preserves normal-start lanes.

The initial engineering defaults are five completed occupancy windows plus the
current partial window, smoothing of one transition, and exponent 0.5. Eligible cell
weights are `(recent_combined_count + smoothing) ** -exponent`. Configure them with
`coverage_windows`, `coverage_smoothing`, and `coverage_exponent`. These are tunable
choices, not a claim that uniform occupancy optimizes learning.

Probabilities use prior collected experience and freeze at rollout admission.
Archive-origin practice contributes to its own exposure. Cold cells get their
existing first dispatch before regular sampling. The existing probability cap has
an effective minimum of one divided by the eligible cell count. Representative
selection remains deterministic. W&B latency does not participate in selection.
`train/curriculum/distribution` shows retained representatives, cold status,
probabilities, resolved lane counts, and coverage feedback separately from occupancy.
The default `value_error` strategy retains raw-GAE feedback semantics.

After restoration, learners collect fresh actions, values, and log probabilities.
Enabled restoration and its coverage measurement contract enter the effective Goal
Variant and Run provenance. Passive tracking and capture-only configuration do not
change the Goal Variant.
Judge any benefit using original-task performance from normal starts at matched
budgets and multiple seeds, with wall-clock cost. Implementing this feature does
not launch that scientific comparison or establish a performance improvement.

## Validation scope

`gradlab.occupancy.TRACKING_COMBINATIONS` lists the host-vector backend/provider
combinations. Breakout and Mario use the shared runtime through SB3 PPO/A2C,
GradLab PPO, and Go-Explore. GraDOOM device collection is rejected; a future adapter
must reduce on-device instead of transferring lane states each step.

See [the implementation validation report](../experiments/reports/occupancy-41.md)
for workloads, performance limits, and W&B evidence.
