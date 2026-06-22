"""VoxTutor: an offline pronunciation-assessment benchmark.

A spoken-language tutor flags *mispronounced* phonemes in a learner utterance. VoxTutor
synthesizes utterances from a known phoneme inventory with mispronunciations injected at
known positions, so the ground truth is exact, then scores pronunciation scorers by the
AUROC of recovering those mispronunciations. The result is a clean 2x2 dissociation:

* **forced alignment** (DTW) buys robustness to *speaking-rate variation* (time-warp);
* **likelihood-ratio normalization** (the Goodness-of-Pronunciation idea) buys robustness
  to *channel / speaker noise* (heteroscedastic variance);

and you need both -- each ablation collapses on the regime whose distortion it cannot
handle, proven by a ``clean`` control that toggles the distortion off.
"""

from __future__ import annotations

# Pin BLAS to a single thread *before* numpy is imported. The benchmark is many tiny
# matrix ops (per-frame template distances and small DTW cost matrices over a few dozen
# utterances, repeated across seeds); with multi-threaded BLAS the per-call thread-pool
# overhead dominates and contention is non-deterministic, which would make CI flaky and
# slow. One thread is both faster here and fully reproducible. setdefault so an explicit
# env wins.
import os as _os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

from voxtutor.config import Settings
from voxtutor.metrics import auroc
from voxtutor.phonemes import PhonemeSet, build_phoneme_set
from voxtutor.runner import run, run_settings
from voxtutor.scorers import ALIGNMENT, NORMALIZED, SCORERS, make_scorer
from voxtutor.synth import REGIMES, synth_utterance
from voxtutor.types import Config, ScoreResult, Utterance

__version__ = "0.1.0"

__all__ = [
    "Settings",
    "PhonemeSet",
    "build_phoneme_set",
    "Utterance",
    "Config",
    "ScoreResult",
    "REGIMES",
    "synth_utterance",
    "SCORERS",
    "ALIGNMENT",
    "NORMALIZED",
    "make_scorer",
    "run",
    "run_settings",
    "auroc",
]
