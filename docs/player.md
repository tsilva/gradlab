# GradLab Player

Open a locally trained policy by recipe or start the Player without a selection
to browse available checkpoints:

```bash
gradlab play --recipe gradlab__bandit/ppo
gradlab play
```

`gradlab play --latest` selects the highest-step published checkpoint from the
newest run with a published checkpoint. It opens paused. Select **Play** to start
inference.

## Desktop and browser

The player serves compiled assets and prints its loopback URL. The installed
package needs no Node.js or frontend build. On Linux, install GTK 3 and WebKitGTK
4.1 for the desktop viewer. Pass `--no-open` to skip the desktop runtime download
and open the printed URL yourself.

The desktop viewer caches its pinned, checksum-verified Neutralinojs runtime in
`~/.cache/gradlab/neutralino` or under `$XDG_CACHE_HOME`. Player and Stats use
separate window and app identities. **Stats** opens or focuses the synchronized
companion window. Closing the last viewer stops the session. Press Ctrl+C in the
terminal to stop the server and close its viewer windows.

## Inspect or save an episode

Reward panels show the return reference step and discount. The initial reference
is step one. Choose **Set return reference to cursor** to move it and keep that
reference while scrubbing. **Provider reward** is the environment output.
**Native reward contribution** is the weighted task component.

Use **Record** and **Download episode** to save a self-contained trajectory.
Use **Import episode** or `gradlab play --recording episode.trj` to inspect its
frames and policy inputs without running the environment. See the
[trajectory guide](player-trajectories.md) for controls, limits, and file format.

For frontend changes, see [source development](development.md#player-development).
