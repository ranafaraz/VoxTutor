"""Pronunciation scorers as a 2x2 factorial over two independent design choices.

A scorer turns an utterance into a per-position mispronunciation score (higher = more
likely mispronounced). It is defined by two orthogonal switches:

* **segmentation** -- ``fixed`` uniform chunks vs ``dtw`` forced alignment (see
  :mod:`voxtutor.align`). Forced alignment is what survives speaking-rate variation.
* **normalization** -- how the frames assigned to a position are turned into a score:

  - *raw* ``= mean_t || frame_t - template ||^2`` -- the average frame-to-template
    distance. By the bias-variance identity this equals ``||centroid - template||^2``
    (the genuine pronunciation error) **plus** the within-phone frame variance, so channel
    noise inflates it even when the phoneme is correct.
  - *normalized* ``= || mean_t(frame_t) - template ||^2`` -- the distance of the position's
    *centroid* to the template. This is exactly the raw score minus the within-phone
    variance, i.e. a likelihood-ratio / Goodness-of-Pronunciation style normalization that
    cancels heteroscedastic channel noise.

The four real scorers are the corners of the square; ``random`` is the chance baseline.

=============  ============  =============  =============================================
scorer         segmentation  normalization  robust to
=============  ============  =============  =============================================
``naive``      fixed         raw            neither
``aligned``    dtw           raw            speaking-rate variation only
``normalized`` fixed         normalized     channel noise only
``gop``        dtw           normalized     both (forced alignment + GOP normalization)
=============  ============  =============  =============================================
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from voxtutor.align import dtw_align, fixed_segmentation, frame_template_cost
from voxtutor.phonemes import PhonemeSet
from voxtutor.types import Utterance

CostFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Scorer:
    """One corner of the 2x2: which segmentation and which normalization it uses."""

    name: str
    alignment: bool   # True -> DTW forced alignment, False -> fixed segmentation
    normalized: bool  # True -> centroid (GOP) normalization, False -> raw mean distance


_SCORERS: dict[str, Scorer] = {
    "naive": Scorer("naive", alignment=False, normalized=False),
    "aligned": Scorer("aligned", alignment=True, normalized=False),
    "normalized": Scorer("normalized", alignment=False, normalized=True),
    "gop": Scorer("gop", alignment=True, normalized=True),
}

# The full ordered list shown in the eval table (random first as the chance floor).
SCORERS: tuple[str, ...] = ("random", *_SCORERS)

# Which scorers use each robustness ingredient (gop has both).
ALIGNMENT: frozenset[str] = frozenset(n for n, s in _SCORERS.items() if s.alignment)
NORMALIZED: frozenset[str] = frozenset(n for n, s in _SCORERS.items() if s.normalized)


def make_scorer(name: str) -> Scorer:
    try:
        return _SCORERS[name]
    except KeyError:
        raise ValueError(f"unknown scorer {name!r}; choose from {SCORERS}") from None


def _position_scores(utt: Utterance, canon_templates: np.ndarray, assign: np.ndarray,
                     normalized: bool) -> np.ndarray:
    scores = np.zeros(utt.n_positions)
    for p in range(utt.n_positions):
        idx = np.flatnonzero(assign == p)
        if idx.size == 0:
            continue
        block = utt.frames[idx]
        tmpl = canon_templates[p]
        if normalized:  # centroid distance -- cancels within-phone (channel) variance
            scores[p] = float(((block.mean(axis=0) - tmpl) ** 2).sum())
        else:           # mean frame distance -- bias^2 + within-phone variance
            scores[p] = float(((block - tmpl) ** 2).sum(axis=1).mean())
    return scores


def score_utterance(name: str, utt: Utterance, phset: PhonemeSet,
                    rng: np.random.Generator, cost_fn: CostFn = frame_template_cost
                    ) -> np.ndarray:
    """Per-position mispronunciation scores for one utterance under scorer ``name``."""
    if name == "random":
        return rng.standard_normal(utt.n_positions)

    scorer = make_scorer(name)
    canon_templates = phset.templates[utt.canon]
    if scorer.alignment:
        assign = dtw_align(cost_fn(utt.frames, canon_templates))
    else:
        assign = fixed_segmentation(utt.n_frames, utt.n_positions)
    return _position_scores(utt, canon_templates, assign, scorer.normalized)
