"""The core scientific claim, on a small seed budget for speed.

Full-strength assertions (with margins) live in the eval gate; these keep the 2x2
dissociation honest in the unit-test suite too.
"""

import numpy as np

from voxtutor.runner import run
from voxtutor.types import Config

SEEDS = 6
N = 32


def _mean(method, regime):
    return float(np.mean([run(Config(method, regime, "real", N, s, "numpy")).auroc
                          for s in range(SEEDS)]))


def test_all_scorers_recover_on_clean():
    for m in ("naive", "aligned", "normalized", "gop"):
        assert _mean(m, "clean") > 0.95


def test_warp_breaks_fixed_segmentation_only():
    # forced alignment survives speaking-rate variation; fixed segmentation does not
    assert _mean("naive", "warped") < 0.80
    assert _mean("normalized", "warped") < 0.80
    assert _mean("aligned", "warped") > 0.95
    assert _mean("gop", "warped") > 0.95


def test_noise_breaks_raw_distance_only():
    # GOP normalization survives channel noise; raw distance does not
    assert _mean("naive", "noisy") < 0.80
    assert _mean("aligned", "noisy") < 0.80
    assert _mean("normalized", "noisy") > 0.95
    assert _mean("gop", "noisy") > 0.95


def test_gop_is_robust_everywhere():
    for regime in ("clean", "warped", "noisy"):
        assert _mean("gop", regime) > 0.95


def test_random_is_chance():
    for regime in ("clean", "warped", "noisy"):
        assert _mean("random", regime) < 0.65
