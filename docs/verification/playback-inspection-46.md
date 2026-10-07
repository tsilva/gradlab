# Playback inspection verification

Issue [#46](https://github.com/tsilva/gradlab/issues/46), verified 2026-09-12.
Base: `ff5dc28b7b843c10c875851b9c7dd185d363c4f8`.
Tested implementation: `ff7b0d22`. The following documentation commit changes no executable files.

## Scope and review

`createPlaybackInspection` owns cursor and live presentation, recorded reads,
prefetch, replay scheduling, bounded retention, frame eligibility, and inspection
peer messages. The application retains source admission and Checkpoint selection.
Chart history, range, reward reference, diagnostics and renderers keep their owners.
The interface and adapter payloads are documented in `docs/playback-inspection.md`.
No dependencies, wire formats, Policy execution, metrics or specifications changed.

Independent Standards review found one redundant view notification, which was
removed. Spec review found obsolete decode errors could escape after a newer seek
or a diagnostic generation change. The final guards cover cursor, episode/session,
demand, disposal and diagnostic generation in both inspection and panel adapters.
Deferred regressions failed before these repairs and pass afterward. Re-review
reported zero unresolved findings.

A pre-existing presentation scheduler bug was reproduced before repair: reset and
a replacement offer during held preparation could leave the replacement queued
until another external event. The scheduler now drains that replacement when the
old preparation settles, while retaining the original Checkpoint ticket identity.

## Automated checks

All commands ran in the dedicated worktree with the frozen dependency environment.

| Check | Result |
| --- | --- |
| Inspection, replay, prefetch, synchronized presentation, Checkpoint selection and panel runtime tests | 70 passed |
| `node --test tests/web_player/*.test.mjs` | 340 passed, 4 baseline failures |
| `uv run --frozen pytest tests/test_play_application.py tests/test_play_web.py -q` | 81 passed |
| `uv run --frozen pytest -q` | 1,911 passed, 5 baseline failures, 3 skipped, 440 subtests passed |
| Ruff on changed Python files | Passed |
| `node --check` on changed executable JavaScript | Passed |
| `git diff --check` | Passed |

The full Python run preceded only the final JavaScript error guards and their
JavaScript tests. No production Python code changed. No typecheck command is
configured in this repository.

Focused JavaScript command:

```sh
node --test tests/web_player/playback-inspection*.test.mjs tests/web_player/panel-runtime.test.mjs tests/web_player/recorded-step-prefetch.test.mjs tests/web_player/synchronized-presentation.test.mjs tests/web_player/checkpoint-selection.test.mjs
```

### Actual baseline comparison

The full JavaScript suite ran at the base commit before editing: 314 passed and
these same four failures:

- `run metrics use compact labels and values`
- `checkpoint metric headers preserve semantics in one short line`
- `scientific editorial dark-theme tokens are the single CSS color source`
- `typography uses stable family roles and a five-step scale`

All five Python failures were individually replayed using the complete base tree
extracted with `git archive`, its tests and its `src` on `PYTHONPATH`, and the same
frozen interpreter. Each failed with the same assertion:

- `test_default_ppo_injects_native_paddle_velocity`
- `test_breakout_ball_state_recipe_is_matched_to_fusion_control`
- `test_breakout_goal_hotswaps_provider_without_changing_semantics`
- `test_breakout_recipe_loads_with_stable_retro_start_state`
- `test_mspacman_recipe_loads_with_breakout_base_config_and_hud_mask`

These configuration, labeling and theme failures were left unchanged.

## Complete-player browser checks

Used the native Codex in-app browser with the real app, renderer modules, browser
history, HTTP recorded-step endpoint, WebSocket frames and BroadcastChannel.
The fixture supplies synthetic recorded transitions and deterministic preparation,
read, bitmap and peer delivery gates. It requires no remote catalog, credentials
or trained Policy execution.

```sh
uv run --frozen python -m tests.web_player.fixtures.inspection_player
uv run --frozen python -m tests.web_player.fixtures.selection_player
```

The five gated inspection groups passed:

- Loading a Policy opens paused with the exact Checkpoint frame.
- Holding read A while selecting later steps settles only the final cursor and frame.
- Releasing earlier game and Input bitmap decodes cannot repaint an older selection.
- Scrubbing preserves the independently selected discounted-reward reference.
- Replacement loading remains visible until held frame preparation finishes, then displays the new Checkpoint.

Additional manual checks passed:

- Held peer delivery leaves the second Input window at its prior cursor. Releasing
  two pending cursor messages converges on the latest step and its recorded Input.
  Returning to the live head from that window synchronizes both windows without
  resuming inference or producing command feedback.
- Primary-button drag on Step reward selects a chart window. Seeking preserves
  that range. Changing its playbar start handle synchronizes the same offsets to
  a second window while retaining the inspection cursor.

The existing selection fixture passed all eight primary groups: catalog recovery,
observer refusal, command rejection, keyboard/failure accessibility, preparation
Retry, adjacent readiness and frames, background browsing, and failed replacement.
Its extended sequence passed Back/Forward, repeated Checkpoint selection, paused
inspection/replacement, and selection initiated with RGB hidden.

### Characterized behavior and limits

The extended fixture's covered Cancel action did not release its presentation mask.
The same sequence failed at the same check with the immutable base frontend served
through `selection_player --assets-root logs/issue46-baseline/src/gradlab/web_player`.
It remains a baseline limitation. Imported trajectory activation was checked
separately and did not execute its Checkpoint attachment or increment activations.

Fresh or idle fixture servers are required when repeating preparation checks.
A pending loader must be released and allowed to drain before another case. One
rerun collided with an earlier preparation; another lost its expired fixture
server. Those setup attempts were excluded from passing results.

The fixture intentionally lacks some scientific evidence, so unavailable diagnostic
values are expected. These checks do not claim remote catalog coverage, real Policy
execution, scientific acceptance, or CI success.
