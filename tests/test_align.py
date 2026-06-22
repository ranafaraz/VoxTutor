import numpy as np

from voxtutor.align import dtw_align, fixed_segmentation, frame_template_cost
from voxtutor.config import Settings
from voxtutor.phonemes import build_phoneme_set
from voxtutor.synth import synth_utterance


def test_fixed_segmentation_partitions_all_frames():
    a = fixed_segmentation(20, 4)
    assert a.shape == (20,)
    assert sorted(np.unique(a)) == [0, 1, 2, 3]
    assert np.all(a[1:] >= a[:-1])  # monotonic non-decreasing


def test_frame_template_cost_matches_bruteforce():
    rng = Settings(seed=0).rng(1)
    frames = rng.standard_normal((7, 5))
    templ = rng.standard_normal((3, 5))
    cost = frame_template_cost(frames, templ)
    for t in range(7):
        for p in range(3):
            assert np.isclose(cost[t, p], ((frames[t] - templ[p]) ** 2).sum())


def test_dtw_is_monotonic_and_covers_all_positions():
    rng = Settings(seed=1).rng(1)
    cost = rng.standard_normal((30, 6)) ** 2
    a = dtw_align(cost)
    assert a[0] == 0 and a[-1] == 5
    assert np.all(a[1:] - a[:-1] >= 0)        # never goes backwards
    assert np.all(a[1:] - a[:-1] <= 1)        # advances at most one position per frame
    assert sorted(np.unique(a)) == [0, 1, 2, 3, 4, 5]  # every position used


def test_dtw_recovers_true_segmentation_under_warp():
    rng = Settings(seed=3).rng(1)
    ps = build_phoneme_set(rng)
    u = synth_utterance(ps, rng, "warped")
    cost = frame_template_cost(u.frames, ps.templates[u.canon])
    assign = dtw_align(cost)
    # forced alignment should match the true owner for the large majority of frames
    agreement = (assign == u.frame_owner).mean()
    assert agreement > 0.9
