"""Synthesize learner utterances with a *known* set of mispronunciations.

Each utterance is a sequence of ``N_POSITIONS`` canonical phonemes. At ``N_MISPRON``
randomly chosen positions a **mispronunciation** is injected: the spoken phoneme template
is nudged by a small vector of norm ``DELTA`` (a subtle articulation error, the way real
mispronunciations are confusable rather than obviously wrong). Each position is rendered
as a run of acoustic frames drawn around its spoken template. The ground truth is the
boolean mask of which positions were nudged.

Three regimes layer a *distortion* on top, each one designed to fool a different cheap
scorer -- and a ``clean`` control that omits the distortion so the fooled scorer recovers:

* ``clean``  -- fixed frame count per phoneme, base noise only.
* ``warped`` -- **speaking-rate variation**: each phoneme gets a different number of
  frames, so a fixed (uniform) segmentation mis-assigns frames to phonemes. Forced
  alignment (DTW) re-segments and is unaffected.
* ``noisy``  -- **heteroscedastic channel noise**: a subset of positions is recorded at a
  much higher variance, inflating the raw frame-to-template distance even when the phoneme
  is correct. Likelihood-ratio (centroid) normalization cancels the variance and is
  unaffected.

The distortions never touch *which* positions are mispronounced -- only how the audio is
rendered -- so the ground truth is identical across regimes and the dissociation is clean.
"""

from __future__ import annotations

import numpy as np

from voxtutor.phonemes import PhonemeSet
from voxtutor.types import Utterance

# --- utterance shape --------------------------------------------------------------
N_POSITIONS = 12   # phonemes per utterance
N_MISPRON = 2      # mispronunciations injected per utterance (the positives)
FRAMES_PER_PHONE = 10  # canonical frame count per phoneme (clean / noisy regimes)
BASE_SIGMA = 0.14  # per-frame acoustic noise std

# --- distortion strengths (tuned so each fooled scorer clearly collapses) ---------
# A mispronunciation nudges the spoken template by this norm. Small enough to be a
# *confusable* error (so a channel-noise or misalignment artefact can rival it), large
# enough that a correctly aligned, noise-normalized scorer still separates it cleanly.
DELTA = 0.65
# Speaking-rate variation: under `warped` each phoneme's frame count is drawn uniformly
# from this inclusive range (mean ~ FRAMES_PER_PHONE), so cumulative drift throws off a
# uniform segmentation while leaving total order intact for DTW.
WARP_MIN, WARP_MAX = 3, 16
# Channel noise: under `noisy` a fraction NOISY_FRACTION of positions is recorded with
# this multiple of the base noise std -- enough to inflate raw distance past a
# mispronunciation, but the within-phone variance still cancels under normalization.
NOISE_MULT = 2.5
NOISY_FRACTION = 0.5

REGIMES = ("clean", "warped", "noisy")


def _phoneme_sequence(rng: np.random.Generator, n_phonemes: int) -> np.ndarray:
    """A canonical sequence with no two adjacent phonemes equal (real transitions)."""
    seq = rng.integers(0, n_phonemes, size=N_POSITIONS)
    for p in range(1, N_POSITIONS):
        while seq[p] == seq[p - 1]:
            seq[p] = rng.integers(0, n_phonemes)
    return seq


def synth_utterance(phset: PhonemeSet, rng: np.random.Generator, regime: str) -> Utterance:
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}; choose from {REGIMES}")

    canon = _phoneme_sequence(rng, phset.n_phonemes)

    # Inject mispronunciations: nudge the spoken template at N_MISPRON positions.
    spoken_mu = phset.templates[canon].astype(float).copy()
    is_mis = np.zeros(N_POSITIONS, dtype=bool)
    mis_pos = rng.choice(N_POSITIONS, size=N_MISPRON, replace=False)
    for p in mis_pos:
        u = rng.standard_normal(phset.d)
        u /= np.linalg.norm(u)
        spoken_mu[p] = phset.templates[canon[p]] + DELTA * u
        is_mis[p] = True

    # Frame counts (speaking rate) and per-position noise level (channel).
    if regime == "warped":
        counts = rng.integers(WARP_MIN, WARP_MAX + 1, size=N_POSITIONS)
    else:
        counts = np.full(N_POSITIONS, FRAMES_PER_PHONE)

    sigma = np.full(N_POSITIONS, BASE_SIGMA)
    if regime == "noisy":
        noisy = rng.random(N_POSITIONS) < NOISY_FRACTION
        sigma[noisy] = BASE_SIGMA * NOISE_MULT

    frames_list, owner = [], []
    for p in range(N_POSITIONS):
        n = int(counts[p])
        block = np.repeat(spoken_mu[p][None, :], n, axis=0)
        block = block + sigma[p] * rng.standard_normal(block.shape)
        frames_list.append(block)
        owner.extend([p] * n)

    return Utterance(
        frames=np.vstack(frames_list),
        canon=canon,
        is_mispronounced=is_mis,
        frame_owner=np.array(owner),
        regime=regime,
    )
