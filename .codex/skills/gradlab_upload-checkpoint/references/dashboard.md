# Game dashboards

Use this procedure after a verified game checkpoint or Release upload, or for
requested standalone dashboard creation, refresh, and evaluation backfills.
The local renderer is shared across games. The model publisher owns model
artifacts and Releases, and dashboard edits happen after model verification.

## Identify the dashboard and its rows

1. Resolve the public owner, canonical environment ID, and verified collection
   ID using the [model publication procedure](operations.md#publish-a-model-on-hugging-face).
   The canonical static Space is
   `<owner>/<environment ID>_Dashboard`. Verify the exact case and ownership.
   Reuse the Space and its published data. For an existing dashboard without
   `dashboard.json`, reconstruct the rows from its verified public artifacts
   and evidence before replacing its page.
2. Record the environment's authoritative stages in game order. Derive stages
   and names from its goals or game inventory. A game can have a single stage,
   and an environment with no defined solved criterion can omit solved status.
   Preserve other stages and published versions. Use one row per stage only
   when the chosen checkpoint or Release for that stage is unambiguous.
   Apply the game's documented selection rule or the owner's choice before
   replacing any existing selected policy.
3. Bind the row to the exact published commit and checkpoint digest. Verify the
   public card, root replay, and provider. Derive the algorithm from the executed
   recipe, including enabled add-ons such as SIL. Link the source recipe only
   after matching the immutable GitHub file to recorded provenance. The exact
   Hub recipe remains available with its model if no public source match exists.
4. Populate metrics from verified evidence for that exact checkpoint. Choose
   the game's primary metric and direction, and distinguish training summaries
   from independent evaluation. Leave unavailable results blank. A checkpoint
   can appear before evaluation and be filled in later on the same row.
   Use solved status only for the method's documented criterion. For a clear
   rate panel, report aggregate clears and sample standard deviation across
   seed clear rates. If reporting Clear Steps, average the successful episodes
   within each contributing seed, then report the mean and sample standard
   deviation across those seed averages. State any difference in contributing
   seeds. Keep other games' metrics and units as defined by their goals.

## Render the reviewed data

Prepare a public JSON document with `version` 1 and these fields.

| Field | Content |
| --- | --- |
| `environment`, `provider`, `provider_url`, `collection_url` | Canonical environment and verified public links |
| `metric_name`, `metric_column`, `metric_direction`, `methods` | Primary metric, short column label, preferred `ascending` or `descending` order, and evaluation definitions |
| `clear_steps` | Optional boolean that enables the Clear Steps column |
| `stages` | Ordered rows, each with a stable `id` and a display `label` |

Rows can add `algorithm`, `solved` (true, false, or omitted), `training_steps`,
`evaluation_seeds`, `metric`, and `clear_steps`. A published row supplies
`checkpoint_sha256` and `model_url` ending in `/tree/<full-commit>`. Use
`replay_url` for that revision's root `replay.mp4` and `recipe_url` for a
verified commit-pinned source recipe. Each metric object has numeric `value`,
visible `display`, and optional `detail`. The table sorts by `value`.
Use `current_recipe_url` for a maintained launchable recipe at a verified
GitHub commit. The table shows one Recipe link per published stage, preferring
the current recipe when supplied. Keep the exact checkpoint recipe in the
model bundle and its source URL in the manifest.

From the repository root, run the offline renderer on the reviewed data.

```sh
python3 .codex/skills/gradlab_upload-checkpoint/scripts/render_dashboard.py public-data.json stage-dir
```

The renderer writes only `index.html` and the normalized public
`dashboard.json`. For a new Space, add a `README.md` with the title
`<environment ID> Dashboard`, `sdk: static`, and `app_file: index.html`.
When maintaining a dashboard with an existing CSV export, carry its current
columns and update affected rows from the same verified receipts. Preserve its
failure breakdown and avoid changing the schema during an ordinary refresh.
Keep all draft files and receipt checks private until their intended publication.
The generated page uses stage order on load and for the single replay player,
with sortable metrics and missing values at the end. It omits a playlist when
there are no replays and keeps table replay links out of the layout.

## Publish and verify

- Review the exact Space and collection operations under the applicable
  outbound policy. The selected Checkpoint exception covers its model repo
  and collection append. Space creation, commits, and reordering follow their
  own approved route and authority. Verify transport supports Spaces and the
  required collection operations before using it.
- Commit the dashboard files together to the exact public Static Space.
  Check the expected parent and preserve unrelated files. Before retrying an
  uncertain commit, compare the published revision and bytes with the draft.
  Add the Space once and position it first while preserving the remaining
  collection order. Verify the final order via a fresh public read.
- Verify the live app and selected stage against the immutable model revision,
  the actual rendered links, metrics, default stage order, and applicable
  replay playback. Check the public dashboard file hashes and CSV when staged.
  Return the model, dashboard, and collection links. If the dashboard update
  is blocked, preserve the completed model receipt and resume from it.
