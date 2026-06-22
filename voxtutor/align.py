"""Segmenting frames to canonical positions: fixed vs forced alignment.

A pronunciation scorer must decide *which frames belong to which phoneme* before it can
score them. Two strategies:

* :func:`fixed_segmentation` -- split the frames into equal contiguous chunks, one per
  canonical position. Correct only when every phoneme has the same duration; under
  speaking-rate variation it drifts and mis-assigns frames.
* :func:`dtw_align` -- **forced alignment**: a dynamic-time-warping pass that lets each
  canonical phoneme consume a variable number of consecutive frames, minimizing total
  frame-to-template distance. It recovers the true segmentation under rate variation.

Both return an ``assign`` array mapping each frame index to a canonical position; the
scorers then aggregate per position.
"""

from __future__ import annotations

import numpy as np


def frame_template_cost(frames: np.ndarray, canon_templates: np.ndarray) -> np.ndarray:
    """Squared-Euclidean cost of every frame against every canonical position's template.

    ``frames`` is (n_frames, d), ``canon_templates`` is (n_positions, d) -- the template of
    the canonical phoneme at each position. Returns (n_frames, n_positions).
    """
    diff = frames[:, None, :] - canon_templates[None, :, :]
    return np.einsum("tpd,tpd->tp", diff, diff)


def fixed_segmentation(n_frames: int, n_positions: int) -> np.ndarray:
    """Uniform segmentation: contiguous equal chunks, one per canonical position."""
    bounds = np.linspace(0, n_frames, n_positions + 1).astype(int)
    assign = np.empty(n_frames, dtype=int)
    for p in range(n_positions):
        assign[bounds[p]:bounds[p + 1]] = p
    return assign


def dtw_align(cost: np.ndarray) -> np.ndarray:
    """Forced alignment by DTW over a (n_frames, n_positions) cost matrix.

    Monotonic alignment from (frame 0, position 0) to (last frame, last position); from a
    cell a frame may stay on the same position (a horizontal step -- a phoneme spanning
    several frames) or advance one position (a diagonal step). Every position is therefore
    assigned at least one frame. Returns the per-frame position assignment.
    """
    n_frames, n_positions = cost.shape
    inf = np.inf
    acc = np.full((n_frames, n_positions), inf)
    acc[0, 0] = cost[0, 0]
    # First column: all early frames stuck on position 0 until the path advances.
    for t in range(1, n_frames):
        acc[t, 0] = acc[t - 1, 0] + cost[t, 0]
    # Fill row by row, vectorized over positions. A position p at frame t comes from
    # either p (stay) or p-1 (advance) at frame t-1; positions p > t are unreachable.
    for t in range(1, n_frames):
        prev = acc[t - 1]
        advance = np.empty(n_positions)
        advance[0] = inf
        advance[1:] = prev[:-1]
        best = np.minimum(prev, advance)
        row = cost[t] + best
        if t < n_positions:  # positions beyond frame index are not yet reachable
            row[t + 1:] = inf
        acc[t] = row

    # Backtrack the monotonic path.
    assign = np.empty(n_frames, dtype=int)
    p = n_positions - 1
    for t in range(n_frames - 1, 0, -1):
        assign[t] = p
        if p > 0 and acc[t - 1, p - 1] <= acc[t - 1, p]:
            p -= 1
    assign[0] = p
    return assign
