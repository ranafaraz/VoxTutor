import numpy as np
import pytest

from voxtutor.align import fixed_segmentation
from voxtutor.config import Settings
from voxtutor.phonemes import build_phoneme_set
from voxtutor.scorers import (
    ALIGNMENT,
    NORMALIZED,
    SCORERS,
    _position_scores,
    make_scorer,
    score_utterance,
)
from voxtutor.synth import N_POSITIONS, synth_utterance


def test_registry_membership():
    assert SCORERS == ("random", "naive", "aligned", "normalized", "gop")
    assert ALIGNMENT == frozenset({"aligned", "gop"})
    assert NORMALIZED == frozenset({"normalized", "gop"})


def test_two_by_two_ingredients():
    assert not make_scorer("naive").alignment and not make_scorer("naive").normalized
    assert make_scorer("aligned").alignment and not make_scorer("aligned").normalized
    assert not make_scorer("normalized").alignment and make_scorer("normalized").normalized
    assert make_scorer("gop").alignment and make_scorer("gop").normalized


def test_unknown_scorer_raises():
    with pytest.raises(ValueError):
        make_scorer("nope")


@pytest.mark.parametrize("name", SCORERS)
def test_scores_have_one_per_position(name):
    rng = Settings(seed=0).rng(1)
    ps = build_phoneme_set(rng)
    u = synth_utterance(ps, rng, "clean")
    s = score_utterance(name, u, ps, Settings(seed=0).rng(2))
    assert s.shape == (N_POSITIONS,)


def test_normalized_equals_raw_minus_within_phone_variance():
    # bias-variance identity: mean ||f - t||^2 = ||mean(f) - t||^2 + within-phone variance
    rng = Settings(seed=5).rng(1)
    ps = build_phoneme_set(rng)
    u = synth_utterance(ps, rng, "noisy")
    canon_t = ps.templates[u.canon]
    assign = fixed_segmentation(u.n_frames, u.n_positions)
    raw = _position_scores(u, canon_t, assign, normalized=False)
    norm = _position_scores(u, canon_t, assign, normalized=True)
    within = np.zeros(u.n_positions)
    for p in range(u.n_positions):
        block = u.frames[assign == p]
        within[p] = ((block - block.mean(0)) ** 2).sum(1).mean()
    assert np.allclose(raw, norm + within)


def test_mispronounced_positions_score_higher_on_clean():
    rng = Settings(seed=0).rng(1)
    ps = build_phoneme_set(rng)
    u = synth_utterance(ps, rng, "clean")
    s = score_utterance("gop", u, ps, Settings(seed=0).rng(2))
    assert s[u.is_mispronounced].min() > s[~u.is_mispronounced].max()


def test_data_is_method_independent():
    # the utterance (audio + ground truth) does not depend on the scorer
    rng1 = Settings(seed=1).rng(9)
    ps1 = build_phoneme_set(rng1)
    u1 = synth_utterance(ps1, rng1, "warped")
    rng2 = Settings(seed=1).rng(9)
    ps2 = build_phoneme_set(rng2)
    u2 = synth_utterance(ps2, rng2, "warped")
    assert np.array_equal(u1.frames, u2.frames)
    assert np.array_equal(u1.is_mispronounced, u2.is_mispronounced)
