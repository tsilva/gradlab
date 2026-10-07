# Issue 41 implementation validation

This report covers software behavior, not scientific curriculum benefit. No
production training or acceptance campaign was launched.

## Exact collection and integration

Scripted public-runtime tests cover borrowed buffers, masked reset fields, source
cells, terminal transitions, cell entries, archive origins, immutable summaries,
partial windows, recovery cursor mismatch, outbox replay, and overlapping ranges.
Native Breakout and Mario tests compare observations, rewards, and termination under
identical seeds and actions. Archive tests cover admission before capture, bounded
storage, recovery, active representatives, deterministic coverage feedback, and the
20% allocation. Existing Go-Explore, PPO, archive, and metric tests remain part of
validation.

## Performance

Matched CPU collection with the checked-in policy's deterministic inference used
five alternating samples of 50 vector batches after warm-up. No learning updates or
snapshot operations were included. Breakout used `ppo-ball-state`, 128 lanes, and
its declared observation/model contract. Mario Level1-1 used `ppo`, 16 lanes, and
its declared contract. Tracking counted exactly 33,280 and 4,160 transitions,
respectively, including warm-up.

Median tracking regression was 1.51% for Breakout and -1.25% for Mario. The negative
value is measurement noise, not a speedup claim. The workload includes policy
inference and meets the 2% target on this CPU check. Environment-only Breakout
measurements exceeded 2%, so these results do not establish the target for a fast
GPU policy or every backend/device combination. Rerun matched measurements on the
intended training host before extending that performance claim.

Capture and masked-restore timing are measured separately from passive tracking.
Operator-specific raw timing records and generated scripts remain outside source
control. No snapshot cost is attributed to the passive collector.

## W&B inspection

The synthetic validation project contains historical windows, unobserved cells,
normal and archive origins, an interruption gap, and a partial final window. The
native Codex in-app browser verified early-window selection, raw hover values, and
the corresponding distribution. The table-history query requires an explicit empty
`extraKeys` list. A latest-table query does not supply historical inspection.

The 122-window probe exposed sampling in an unindexed W&B table-history query.
Explicit index 120 retrieved the exact late window, so the implementation uses
indexed historical pages and a separate recent-page panel. Each upload contains at
most eight windows; every historical table remains individually addressable. It
does not repeatedly upload the whole growing history. Presentation deduplicates
aggregate identities before adding counts for coarser views.

The final [paged-history fixture](https://wandb.ai/tsilva/gradlab-occupancy-validation?nw=10ywdldcehf)
rendered complete eight-window ranges 0–48 and 672–720. Selecting 0–6 after grouping
the first dimension showed raw counts 4 and 2 with denominator 6. The partial
archive-origin window 732–734 showed count 2, denominator 2, and `complete=false`.
A [gap fixture](https://wandb.ai/tsilva/gradlab-occupancy-validation?nw=eoe51pcwm5p)
checks readable interval metadata; W&B requires this array to be serialized as text
in the table. Immutable outbox aggregates retain the original numeric interval.
The final tooltip showed `[6, 12]` for the gap before the 12–14 partial window.
Curriculum tables include Run identity so comparisons cannot stack probabilities
or representative counts from different Runs.

## Regression checks

The full suite completed with 1,806 passing tests, 438 passing subtests, three skips,
and five failures. Four recipe expectation failures reproduce on the original commit
`7daff719`; the metric registry cardinality failure was fixed for the added measures.
Focused tests then passed for the new collector, recovery gaps, bounded history,
Goal Variants, archive publication versus concurrent pruning, and managed workspaces.
The final archive/provenance/supervisor regression run passed 128 tests. Ruff and
type checks of the four new modules passed. Broader mypy checking is blocked by its
parser rejecting existing Python 3.14 exception syntax in repository modules.
The final collector/workspace check passed 35 tests, including publisher Run identity
and readable gap metadata.

## Standards review

No remaining material findings. Publication locking, deferred inventory turnover,
leased remote reclamation, and collector ownership repairs passed follow-up review.

## Spec review

No remaining material findings after entry- and byte-budget turnover, provenance,
recovery-gap, and exact historical-page repairs. Browser checks complete the external
W&B rendering evidence.
