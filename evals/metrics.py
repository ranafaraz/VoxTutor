"""Aggregate mispronunciation-detection AUROC across seeds.

One number carries a (scorer, regime) cell: the mean AUROC of recovering the injected
mispronunciations over many seeds. A scorer with the right ingredient for a regime holds
near 1.0; one missing it drops toward chance on the regime whose distortion it cannot
handle, and recovers on the ``clean`` control.
"""

from __future__ import annotations

import numpy as np

from voxtutor.runner import run
from voxtutor.types import Config


def aurocs(method: str, regime: str, labels: str, samples: int, seeds: int) -> np.ndarray:
    return np.array(
        [run(Config(method, regime, labels, samples, s, "numpy")).auroc for s in range(seeds)]
    )


def mean_auroc(method: str, regime: str, samples: int, seeds: int,
               labels: str = "real") -> float:
    return float(aurocs(method, regime, labels, samples, seeds).mean())


def std_auroc(method: str, regime: str, samples: int, seeds: int,
              labels: str = "real") -> float:
    return float(aurocs(method, regime, labels, samples, seeds).std())
