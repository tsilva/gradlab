# Checkpoint selection verification

Issue [#44](https://github.com/tsilva/gradlab/issues/44), verified 2026-09-12.
Base: `091403d7724a369035277bcacb629e97ab740c93`.
Tested implementation: `b615289417bc88207f4d66936c03f6b1a426b757`.
The following documentation commit changes no executable files.

## Scope and review

`CheckpointSelection` owns local command identity, selection generations, routes,
foreground/background decisions, navigation readiness, loading, and completion.
SourceBrowser executes catalog and history work using its resulting view; the app
keeps its existing snapshots, frame caches, and scientific rendering. Presentation
completion uses a ticket registered before asynchronous preparation and validates
its generation, snapshot object, session epoch, and sequence. Command acceptance
still does not indicate presentation readiness.

Removed the old activation/failure predicates, app loading functions, mirrored
pending Checkpoint identity, source-sliced selection tests, and selection-specific
prototype tests. Independent synchronized-presentation tests and unrelated catalog,
rendering, chart, and fullscreen tests remain. No dependencies, server protocols,
execution policy, metrics, or product specifications changed.

## Standards

Independent review found no material Standards violations or code smells.

## Spec

The first independent review found that selecting/preparing routes still bypassed
the new owner. The repair makes the owner reconcile every foreground authoritative
route and makes both callers consume that route. Re-review found no remaining
Spec findings. An epoch-increment gate from the first implementation commit was
removed because cancellation can authoritatively retain the previous session;
local generation is not server command correlation.

Final review totals: zero unresolved Standards findings and zero unresolved Spec
findings.

## Automated checks

Commands ran from the dedicated worktree with Node 26.3.1 and the frozen CPython
3.14.5 environment. `uv sync --frozen` completed without dependency changes.

| Command | Result |
| --- | --- |
| `node --test tests/web_player/checkpoint-selection.test.mjs` | 14 passed |
| `node --test tests/web_player/checkpoint-selection.test.mjs tests/web_player/source-browser.test.mjs tests/web_player/controls.test.mjs tests/web_player/synchronized-presentation.test.mjs tests/web_player/playback-transport.test.mjs` | 118 passed, 2 baseline failures |
| `node --test tests/web_player/*.test.mjs` | 314 passed, 4 baseline failures |
| `uv run --frozen pytest tests/test_play_application.py tests/test_play_web.py -q` | 81 passed |
| `uv run --frozen pytest -q` | 1,892 passed, 5 failures in unchanged configuration tests, 3 skipped, 438 subtests passed |
| `uv run --frozen ruff check tests/web_player/fixtures/selection_player.py tests/test_play_web.py` | Passed |
| `node --check` on `app.js`, `sources/browser.js`, `checkpoint-selection.js`, and fixture `selection-checks.js` | Passed |
| `git diff --check` | Passed |

No repository typecheck command is configured. JavaScript parsing and Python lint
were checked with the commands above.

The full Python run completed during implementation; the final focused Python run
rechecked every changed Python test and the host/web-player path. No production
Python files changed.

### Baseline failures

The complete JavaScript suite was also run on an extracted immutable baseline:

```sh
mkdir -p logs/issue44-baseline
git archive 091403d7724a369035277bcacb629e97ab740c93 src/gradlab/web_player tests/web_player package.json | tar -x -C logs/issue44-baseline
node --test logs/issue44-baseline/tests/web_player/*.test.mjs
```

It reported 316 passed and these same four failures:

- `run metrics use compact labels and values`
- `checkpoint metric headers preserve semantics in one short line`
- `scientific editorial dark-theme tokens are the single CSS color source`
- `typography uses stable family roles and a five-step scale`

The Python failures are in unchanged tests, recipes, and configuration code:

- `test_default_ppo_injects_native_paddle_velocity`: expects eight context values; the recipe has nine.
- `test_breakout_ball_state_recipe_is_matched_to_fusion_control`: expects no control layers; the recipe has `[256]`.
- `test_breakout_goal_hotswaps_provider_without_changing_semantics`: expects 500 million steps; the recipe specifies one billion.
- `test_breakout_recipe_loads_with_stable_retro_start_state`: expects Stable Retro; the recipe selects the native Breakout provider.
- `test_mspacman_recipe_loads_with_breakout_base_config_and_hud_mask`: expects event rewards; the recipe uses native rewards.

A focused recheck reproduced the same five failures, with 51 passed and 255 subtests
passed, using:

```sh
uv run --frozen pytest tests/test_breakout_ball_state_runtime.py::test_default_ppo_injects_native_paddle_velocity tests/test_config_validation.py -q
```

These failures were left unchanged, as required by the issue.

## Complete-player verification

Used the native Codex in-app browser and the checked-in fixture:

```sh
uv run --frozen python -m tests.web_player.fixtures.selection_player
```

The server bound loopback with port zero and generated a local session token. It
used no credentials, remote storage, trained Policy, or external manifest fetches.
Each source provided three synthetic recorded transitions, distinguishable frames,
and paused playback. Explicit preparation/frame gates controlled ordering.

The fresh-fixture selection suite passed all eight groups:

- Catalog refresh failure retains rows; recovery does not prepare a Checkpoint.
- Observer refusal preserves the route and creates no load.
- Command rejection removes loading, reports the error, and retains the selected route.
- Keyboard selection produces the loading mask, `aria-busy`, and live status; preparation failure releases them.
- Preparation Retry uses its existing separate command and activates paused playback without a tracked mask.
- Adjacent navigation is chronological and guards pending work. Active navigation becomes ready while the mask still waits for withheld frames. Releasing frames presents the selected Checkpoint's pixel value and clears busy state.
- Local browsing survives background control snapshots without sending `browse_sources` against the active runner.
- Failed replacement preserves the prior Playback Session epoch.

The extended suite passed all six groups:

- Back/Forward preserves Run and Environment discovery routes without closing the runner.
- Repeated selection of the same Checkpoint activates a fresh paused session.
- Inspection at recorded step 102 stays paused; replacement presents the new session.
- Selection initiated with RGB hidden completes under the server's activation settings.
- Programmatic activation of the covered Cancel button preserves the previous runner and applicable presentation completion.
- Imported Playback activates a recorded trajectory without another loader activation or Checkpoint attachment execution.

Additional native-browser checks passed:

- With `sources/browser.js` withheld, a direct Checkpoint link remains intact. On release, exactly one `select_source` preparation resolves the full Environment, Research Goal, Goal Variant, Run, and Checkpoint identity. The resulting player is paused at position 2/2.
- A separate `/panel/game` viewer follows the current source and initiates adjacent selection. Its mask is visible while the original viewer has no local mask; both reconcile to position 1/2. Closing that viewer leaves its panel detached, so restore the default layout before reusing the main-window frame fixture.
- A fresh source browser with no catalog rows exposes catalog Retry. Recovery loads both rows with zero Checkpoint preparations, distinct from preparation Retry.
- Back during pending preparation restores the Run route while retaining the pending mask. Disconnect then clears the mask and `aria-busy` and displays Disconnected.

The main selection compatibility sequence and the history/inspection/RGB/import
sequence also passed with the immutable baseline frontend served by the same
fixture, using:

```sh
uv run --frozen python -m tests.web_player.fixtures.selection_player --assets-root logs/issue44-baseline/src/gradlab/web_player
```

### Characterized behavior and limits

Preparation Retry and cancellation do not create new tracked loads. The existing
mask covers the replacement-cancellation control for pointer users; this change
does not change that policy. Initial observations without applicable frames already
complete without waiting for RGB. A failed replacement can leave the locally browsed
preparation screen unchanged while clearing its mask, and pending navigation may
remain until an authoritative active snapshot. These behaviors are preserved.
Navigating Back during pending preparation likewise leaves the load pending until
presentation or a terminal event; route navigation alone does not cancel it.

The host can reject a new selection while a previous activation worker finishes
cleanup. Independent fixture cases wait for that worker to settle. Frame-order and
obsolete-completion correctness use deferred promises and explicit frame gates,
not timing sleeps. The fixture initially exposed incidental initial-observation
snapshots and retained panel placement between manual checks; those fixture setup
conditions were corrected before recording the passing results above.

Fixture diagnostics intentionally omit some scientific evidence. Their unavailable
values are not new product defects or research results. Real remote catalogs,
trained Policy execution, and remote CI were not used as local validation gates.
