"""Deterministic labels for the pinned native Breakout RGB dataset.

Source: tsilva/gradlab-breakout-trajectories, revision
9a22e4c0b6b9796a1358f36a6854f2569cbee0af, annotations/breakout-bricks-v1/brick_labels.py.

The visual detector does not consume simulator labels. Occurrence-level checks
compare its result with native counts afterward; no count correction is applied.
"""

from enum import IntFlag

import numpy as np

VERSION = "breakout-bricks-v1"
COLORS = np.array(
    [[200, 72, 72], [192, 104, 56], [176, 120, 48], [160, 160, 40], [72, 160, 72], [64, 72, 200]],
    dtype=np.uint8,
)


class Issue(IntFlag):
    UNKNOWN_CELL = 1
    COUNT_MISMATCH = 2
    UNEXPECTED_COLOR = 4
    PARTIAL_BRICK = 8
    TEMPORAL_CHANGE = 16
    WALL_RESET_UNVALIDATED = 32
    NATIVE_COUNT_UNAVAILABLE = 64
    INITIAL_LAYOUT = 128


def extract_wall_batch(walls):
    """Classify N RGB wall crops [57:93, 8:152] as six rows of 18 cells.

    Returns int8 grids, uint8 matching-pixel counts, and uint16 visual issue masks.
    Grid values: 0 absent, 1 present, -1 unknown. Counts have denominator 48.
    """
    walls = np.asarray(walls)
    if walls.dtype != np.uint8 or walls.ndim != 4 or walls.shape[1:] != (36, 144, 3):
        raise ValueError("Expected uint8 wall crops with shape [N,36,144,3]")
    cells = walls.reshape(-1, 6, 6, 18, 8, 3)
    matching = np.all(cells == COLORS[None, :, None, None, None, :], axis=-1)
    counts = matching.sum((2, 4)).astype(np.uint8)
    grids = np.full(counts.shape, -1, dtype=np.int8)
    grids[counts <= 2] = 0
    grids[counts >= 36] = 1
    xs, ys = np.arange(8), np.arange(6)
    active_x = matching.any(2)
    active_y = matching.any(4).transpose(0, 1, 3, 2)
    widths = np.where(active_x, xs, -1).max(-1) - np.where(active_x, xs, 8).min(-1) + 1
    heights = np.where(active_y, ys, -1).max(-1) - np.where(active_y, ys, 6).min(-1) + 1
    grids[(grids == -1) & (widths <= 2) & (heights <= 4)] = 0
    flags = np.zeros(len(walls), dtype=np.uint16)
    flags[(grids == -1).any((1, 2))] |= Issue.UNKNOWN_CELL.value
    flags[((grids == 1) & (counts != 48)).any((1, 2))] |= Issue.PARTIAL_BRICK.value
    unexpected = ~matching & np.any(cells != 0, axis=-1)
    flags[unexpected.any((1, 2, 3, 4))] |= Issue.UNEXPECTED_COLOR.value
    return grids, counts, flags


def initial_layout_flags(grids, native_counts, walls_cleared, scores):
    """Mark the initial animation prefix, including the episode's initial frame.

    Inputs contain the initial image plus successors; native values exist only for
    successors. A reset-to-full-wall episode is identified by the first native state.
    The prefix ends at the first complete visible wall or evidence gameplay started.
    It never restarts on life loss or later walls. Animation may remove bricks before
    showing a full wall, so monotonic visible growth is deliberately not assumed.
    """
    if len(grids) != len(native_counts) + 1 or not len(native_counts):
        raise ValueError("Need initial grid plus one grid per native successor state")
    flags = np.zeros(len(grids), dtype=bool)
    if native_counts[0] != 108 or walls_cleared[0] != 0 or scores[0] != 0:
        return flags
    for i, grid in enumerate(grids):
        if np.all(grid == 1):
            break
        if i and (native_counts[i - 1] < 108 or walls_cleared[i - 1] > 0 or scores[i - 1] > 0):
            break
        flags[i] = True
    return flags


def episode_issues(grids, visual_flags, native_counts, walls_cleared, scores):
    """Return occurrence issues and startup flags without changing visual grids."""
    initial = initial_layout_flags(grids, native_counts, walls_cleared, scores)
    flags = np.array(visual_flags, dtype=np.uint16, copy=True)
    counts = (grids == 1).sum((1, 2))
    flags[0] |= Issue.NATIVE_COUNT_UNAVAILABLE.value
    flags[1:][counts[1:] != native_counts] |= Issue.COUNT_MISMATCH.value
    flags[initial] |= Issue.INITIAL_LAYOUT.value
    # Native counters can certify reset boundaries after the first successor.
    resets = np.zeros(len(grids), dtype=bool)
    resets[2:] = (np.diff(walls_cleared) > 0) | (np.diff(native_counts) > 0)
    flags[resets & ~initial] |= Issue.WALL_RESET_UNVALIDATED.value
    gains = ((grids[:-1] == 0) & (grids[1:] == 1)).any((1, 2))
    unexplained = gains & ~initial[:-1] & ~initial[1:] & ~resets[1:]
    flags[:-1][unexplained] |= Issue.TEMPORAL_CHANGE.value
    flags[1:][unexplained] |= Issue.TEMPORAL_CHANGE.value
    return flags, initial
