"""Optional SciPy cross-checks for the numpy core.

Two hand-rolled pieces carry the benchmark: the DTW frame-template cost matrix
(:func:`voxtutor.align.frame_template_cost`) and the Mann-Whitney AUROC
(:func:`voxtutor.metrics.auroc`). This module recomputes both with SciPy
(``scipy.spatial.distance.cdist`` and ``scipy.stats.mannwhitneyu``) so a test can confirm
they agree. Imported lazily so SciPy is never required for the offline benchmark; install
with ``pip install "voxtutor[scipy]"``.
"""

from __future__ import annotations

import numpy as np


def cost_matrix_scipy(frames: np.ndarray, canon_templates: np.ndarray) -> np.ndarray:
    """Squared-Euclidean frame-template cost via ``scipy.spatial.distance.cdist``."""
    from scipy.spatial.distance import cdist

    return cdist(frames, canon_templates, metric="sqeuclidean")


def auroc_scipy(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via the Mann-Whitney U statistic from ``scipy.stats.mannwhitneyu``."""
    from scipy.stats import mannwhitneyu

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    u = mannwhitneyu(pos, neg, alternative="greater").statistic
    return float(u) / (pos.size * neg.size)
