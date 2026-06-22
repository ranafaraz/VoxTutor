import numpy as np
import pytest

from voxtutor.config import Settings
from voxtutor.phonemes import build_phoneme_set
from voxtutor.synth import (
    FRAMES_PER_PHONE,
    N_MISPRON,
    N_POSITIONS,
    REGIMES,
    synth_utterance,
)


def _utt(regime, seed=0):
    rng = Settings(seed=seed).rng(1)
    ps = build_phoneme_set(rng)
    return ps, synth_utterance(ps, rng, regime)


@pytest.mark.parametrize("regime", REGIMES)
def test_ground_truth_shape_and_count(regime):
    _, u = _utt(regime)
    assert u.canon.shape == (N_POSITIONS,)
    assert u.is_mispronounced.shape == (N_POSITIONS,)
    assert int(u.is_mispronounced.sum()) == N_MISPRON


@pytest.mark.parametrize("regime", REGIMES)
def test_no_adjacent_identical_phonemes(regime):
    _, u = _utt(regime)
    assert np.all(u.canon[1:] != u.canon[:-1])


def test_clean_has_fixed_frame_counts():
    _, u = _utt("clean")
    assert u.n_frames == N_POSITIONS * FRAMES_PER_PHONE
    # each position owns exactly FRAMES_PER_PHONE frames
    _, counts = np.unique(u.frame_owner, return_counts=True)
    assert np.all(counts == FRAMES_PER_PHONE)


def test_warped_has_variable_frame_counts():
    _, u = _utt("warped")
    _, counts = np.unique(u.frame_owner, return_counts=True)
    assert counts.min() != counts.max()  # genuine speaking-rate variation


def test_noisy_keeps_fixed_frame_counts_but_adds_variance():
    ps, u = _utt("noisy")
    assert u.n_frames == N_POSITIONS * FRAMES_PER_PHONE  # rate is not the distortion here
    # at least one position has clearly inflated within-phone spread
    spreads = []
    for p in range(N_POSITIONS):
        block = u.frames[u.frame_owner == p]
        spreads.append(((block - block.mean(0)) ** 2).sum(1).mean())
    assert max(spreads) > 3 * min(spreads)


def test_ground_truth_identical_across_regimes_for_same_seed():
    # distortions must not change *which* positions are mispronounced
    labels = [_utt(r, seed=2)[1].is_mispronounced for r in REGIMES]
    canons = [_utt(r, seed=2)[1].canon for r in REGIMES]
    for x in labels[1:]:
        assert np.array_equal(labels[0], x)
    for x in canons[1:]:
        assert np.array_equal(canons[0], x)


def test_unknown_regime_raises():
    rng = Settings(seed=0).rng(1)
    ps = build_phoneme_set(rng)
    with pytest.raises(ValueError):
        synth_utterance(ps, rng, "bogus")
