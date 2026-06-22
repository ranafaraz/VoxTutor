"""Optional cross-check: numpy core matches SciPy.

Skipped unless the ``[scipy]`` extra is installed; never runs on the default offline path,
so CI stays green with no extra downloads.
"""

import numpy as np
import pytest

from voxtutor.align import frame_template_cost
from voxtutor.config import Settings
from voxtutor.metrics import auroc
from voxtutor.phonemes import build_phoneme_set
from voxtutor.synth import synth_utterance

pytest.importorskip("scipy")

from voxtutor.scipy_check import auroc_scipy, cost_matrix_scipy  # noqa: E402


@pytest.mark.parametrize("regime", ["clean", "warped", "noisy"])
def test_cost_matrix_matches_scipy(regime):
    rng = Settings(seed=0).rng(1)
    ps = build_phoneme_set(rng)
    u = synth_utterance(ps, rng, regime)
    canon_t = ps.templates[u.canon]
    assert np.allclose(frame_template_cost(u.frames, canon_t),
                       cost_matrix_scipy(u.frames, canon_t), atol=1e-9)


def test_auroc_matches_scipy():
    rng = np.random.default_rng(0)
    scores = rng.standard_normal(300)
    labels = rng.random(300) < 0.3
    assert np.isclose(auroc(scores, labels), auroc_scipy(scores, labels), atol=1e-9)
