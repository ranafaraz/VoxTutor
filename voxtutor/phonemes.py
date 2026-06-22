"""The phoneme inventory: a fixed set of acoustic templates.

A real pronunciation scorer compares speech frames against an acoustic model of each
phoneme. VoxTutor stands in a deterministic *template* for each phoneme -- a unit vector
in a small feature space, well separated from the others -- so a frame produced for
phoneme ``k`` sits near template ``k`` (up to noise). The inventory is fixed for a run
(it is the "language"); utterances and mispronunciations vary on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_PHONEMES = 10  # size of the (toy) phoneme inventory
D_FEATURES = 12  # acoustic feature dimensionality
TEMPLATE_SCALE = 1.0  # each phoneme template is a unit vector scaled by this


@dataclass(frozen=True)
class PhonemeSet:
    """A fixed inventory of phoneme templates, shape (n_phonemes, d)."""

    templates: np.ndarray

    @property
    def n_phonemes(self) -> int:
        return int(self.templates.shape[0])

    @property
    def d(self) -> int:
        return int(self.templates.shape[1])

    def template(self, phoneme_id: int) -> np.ndarray:
        return self.templates[phoneme_id]


def build_phoneme_set(rng: np.random.Generator) -> PhonemeSet:
    """Random, well-separated unit-vector templates -- the phoneme inventory for a run."""
    t = rng.standard_normal((N_PHONEMES, D_FEATURES))
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return PhonemeSet(templates=t * TEMPLATE_SCALE)
