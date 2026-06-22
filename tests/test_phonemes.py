import numpy as np

from voxtutor.config import Settings
from voxtutor.phonemes import D_FEATURES, N_PHONEMES, build_phoneme_set


def test_shape_and_count():
    ps = build_phoneme_set(Settings(seed=0).rng(1))
    assert ps.templates.shape == (N_PHONEMES, D_FEATURES)
    assert ps.n_phonemes == N_PHONEMES and ps.d == D_FEATURES


def test_templates_are_unit_norm():
    ps = build_phoneme_set(Settings(seed=0).rng(1))
    norms = np.linalg.norm(ps.templates, axis=1)
    assert np.allclose(norms, 1.0)


def test_templates_are_well_separated():
    ps = build_phoneme_set(Settings(seed=0).rng(1))
    # nearest distinct templates are clearly apart (no accidental duplicate phonemes)
    for i in range(ps.n_phonemes):
        for j in range(ps.n_phonemes):
            if i != j:
                assert np.linalg.norm(ps.templates[i] - ps.templates[j]) > 0.3


def test_deterministic():
    a = build_phoneme_set(Settings(seed=4).rng(1)).templates
    b = build_phoneme_set(Settings(seed=4).rng(1)).templates
    assert np.array_equal(a, b)
