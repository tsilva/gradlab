# Playback diagnostic-read verification

Issue: [#40](https://github.com/tsilva/gradlab/issues/40). Verified on 2026-09-12.
Base: `bcdcba0de7c5ce754ece33635a9a36bd39e1ee59`.
Final tested implementation and tests: `07b6732f987ca2aed41016d43209934d76aa026e`.

`DiagnosticReads.read(DiagnosticRead(...))` now owns chart, reward, and event
reservation, admission, isolated calculation, validation, and release. Live and
imported sources supply recording pins and copied calibration annotations. The
host binds the operation to its session epoch; the shared HTTP handler checks
that epoch again after worker delivery, including completed but unpolled jobs.

The existing short diagnostic lock, trajectory-before-diagnostic replacement
ordering, pinned recorded-step inspection, shared eight-job worker admission,
1 ms inspection polling, and 10 ms history polling remain intact. The scientific
calculations, revision encoder, production browser code, and capacity limits did
not change. No W&B metrics are emitted by the diagnostic read/calculation path.
Root and scoped specifications require no changes.

## Python checks

The clean base passed 157 focused tests before implementation. The final focused
run passed **230 tests in 135.82 seconds**, including 73 lifecycle/compatibility
tests, with this command:

```bash
uv run --frozen pytest -q tests/test_play_diagnostic_reads.py tests/test_play_chart_history.py tests/test_play_reward_history.py tests/test_play_chart_transport.py tests/test_playback_responsiveness.py tests/test_playback_episode_history.py tests/test_play_trajectory.py tests/test_play_application.py tests/test_play_web.py tests/test_playback_pacing.py tests/test_playback_worker.py
```

Coverage includes both sources and all three kinds; preparation, submission,
calculation, and validation failures; admission exhaustion; running and queued
shutdown; old-file retention during replacement; activation before old-runner
drain; shared diagnostic/export/inspection retention; and completed unpolled
worker results. A real spawned calculation process test verifies queued
cancellation, independent inference, draining, and child exit. Real worker tests
verify shared inspection capacity and cancellation translation. Imported tests
fail if Policy loading or runtime construction is attempted.

The preparation-failure test first reproduced a leaked pin. The delivery test
first reproduced a stale HTTP 200 after session replacement, then passed with
HTTP 400 for chart, reward, and event reads after the final epoch check was
restored. Direct calculation cancellation still disconnects HTTP; worker
cancellation still returns HTTP 400 with
`playback worker CancelledError: unknown worker failure`.

`uv run --frozen pytest -q` completed once with **1,886 passed, 6 failed,
3 skipped, and 438 subtests passed** in 558 seconds. One failure was an invalid
query in the new completed-job capacity test; the corrected test passes in the
final focused run. The other five failures reproduced individually against an
export of the exact base revision using the same frozen environment:

- `test_breakout_ball_state_runtime.py::test_default_ppo_injects_native_paddle_velocity`
- `test_config_validation.py::ConfigValidationTests::test_breakout_ball_state_recipe_is_matched_to_fusion_control`
- `test_config_validation.py::ConfigValidationTests::test_breakout_goal_hotswaps_provider_without_changing_semantics`
- `test_config_validation.py::ConfigValidationTests::test_breakout_recipe_loads_with_stable_retro_start_state`
- `test_config_validation.py::ConfigValidationTests::test_mspacman_recipe_loads_with_breakout_base_config_and_hud_mask`

Ruff passed on every changed Python file. Python compilation checks passed for
the changed orchestration modules. The repository configures no separate type
checker. Dependencies were installed with `uv sync --frozen`; no dependency or
lockfile changes were made.

## Browser checks

`node --test tests/web_player/*.test.mjs` produced **316 passes and 4 failures**.
The identical four failures reproduced against the exact base's browser source
and tests: compact run metric labels, checkpoint metric headers, dark-theme
color tokens, and typography scale. All chart lifecycle, revision transport,
range, synchronized presentation, and recorded-step tests passed.

Native Codex in-app Browser checks used the credential-free complete player:

```bash
uv run --frozen python -m tests.web_player.fixtures.chart_player --chart-delay 0.8
uv run --frozen python -m tests.web_player.fixtures.chart_player --imported --chart-delay 0.8
```

| Check | Observed result |
| --- | --- |
| Live and imported delayed first read | Loading replaced the plot; initial HTTP 503 recovered automatically into recorded charts. |
| Live zoom and synchronized workspace | Selecting steps 8–107 updated both windows; moving the end handle to 106 updated the shared selection. |
| Paused inspection | Live seeking retained zoom. Imported inspection moved from step 140 to 20 while the reward reference remained at 140. |
| Demand cancellation and restoration | Disabling the sole chart in the secondary live/imported workspace removed demand; enabling it while paused started a refresh. |
| Episode replacement | Live reset moved to episode 2, step 0; both windows cleared old charts and removed the selected range. |
| Continued live trajectory | Playback advanced to the synthetic episode boundary and charts followed the new recording. |
| Imported zoom | Steps 8–107 appeared in both imported windows without advancing the recorded cursor. |
| Missing scientific evidence | Missing value-contract comparison remained unavailable; imported diagnostics identified unrecorded data. |

The complete interaction checks ran at `c602e9f8`. Fresh live and imported
servers using the final production code at `f898d7c7` repeated recovery,
paused imported inspection, and live episode replacement. `07b6732f` only
corrects a test's expected cancellation message. The in-app driver did not
expose popup windows in its tab list, so secondary views were opened through
the player's existing `/workspace/<name>` route and populated through Panels.
No production JavaScript was changed.

## Review

The standards review found no material violations. Its nonblocking request to
make the source context-manager return contract explicit was applied.

The specification review found the completed-job epoch regression and missing
real-process shutdown coverage. Both were addressed and independently reviewed
again with no remaining material findings. The PR remains open for review;
this verification does not imply a merge, release, or scientific Acceptance.
