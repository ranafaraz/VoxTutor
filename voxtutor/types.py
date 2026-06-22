"""Core dataclasses shared across VoxTutor.

An :class:`Utterance` is a synthesized learner recording: a sequence of acoustic feature
frames, the canonical phoneme the speaker *should* have produced at each position, and a
boolean ground-truth mask of which positions were actually mispronounced. A pronunciation
scorer assigns each position a mispronunciation score; the runner pools those scores over
many utterances and reports the AUROC of recovering the injected mispronunciations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Utterance:
    """One synthesized learner utterance with known ground-truth mispronunciations.

    ``frames`` is the acoustic feature matrix (``n_frames`` x ``d``), ``canon`` the
    canonical phoneme id the speaker targets at each of ``n_positions`` positions, and
    ``is_mispronounced`` a boolean mask that is ``True`` exactly where a mispronunciation
    was injected. ``frame_owner`` records the true position each frame belongs to (used
    only by tests / diagnostics, never by a scorer).
    """

    frames: np.ndarray
    canon: np.ndarray
    is_mispronounced: np.ndarray
    frame_owner: np.ndarray
    regime: str

    @property
    def n_positions(self) -> int:
        return int(self.canon.shape[0])

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])


@dataclass
class ScoreResult:
    """The outcome of running one scorer on one regime for one seed.

    ``scores`` and ``labels`` are pooled across every position of every utterance, so the
    headline ``auroc`` measures how well the scorer separates mispronounced positions from
    correct ones over the whole evaluation set.
    """

    method: str
    regime: str
    seed: int
    scores: np.ndarray
    labels: np.ndarray
    auroc: float = 0.0


@dataclass(frozen=True)
class Config:
    """A fully resolved experiment specification (what to run, deterministically)."""

    method: str
    regime: str
    labels: str  # "real" | "scrambled"
    samples: int  # number of utterances pooled per (method, regime, seed)
    seed: int
    backend: str


@dataclass
class Aggregate:
    """Mean AUROC of one (method, regime) cell across seeds, with its spread."""

    method: str
    regime: str
    mean: float
    std: float
    per_seed: list[float] = field(default_factory=list)
