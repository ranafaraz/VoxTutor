"""Scoring a pronunciation scorer against the ground-truth mispronunciations.

The headline metric is the AUROC of the per-position mispronunciation scores at separating
the injected mispronunciations from the correctly pronounced positions. It is
threshold-free and lives in ``[0, 1]`` -- ``1.0`` means every mispronounced position
outranks every correct one, ``0.5`` is chance. Computed from the Mann-Whitney U statistic
with average ranks for ties (no scipy/sklearn on the default path).
"""

from __future__ import annotations

import numpy as np


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks (1-based) with ties assigned the average of the positions they span."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_vals = values[order]
    i = 0
    n = len(values)
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1.0  # average 1-based rank over the tie block
        ranks[order[i:j]] = avg
        i = j
    return ranks


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve of ``scores`` against boolean ``labels``.

    Returns ``0.5`` when one class is empty (undefined ranking -> chance).
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = _average_ranks(scores)
    sum_pos = float(ranks[labels].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
