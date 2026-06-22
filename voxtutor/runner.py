"""Run one scorer on one regime and assemble a :class:`ScoreResult`.

The phoneme inventory and the utterances (including which positions are mispronounced)
depend only on the regime and seed -- never on the scorer -- so every scorer is graded on
identical audio for a fair comparison. Scores and ground-truth labels are pooled across
every position of every utterance; the headline number is the AUROC of recovering the
injected mispronunciations.
"""

from __future__ import annotations

import numpy as np

from voxtutor.align import frame_template_cost
from voxtutor.config import Settings
from voxtutor.metrics import auroc
from voxtutor.phonemes import build_phoneme_set
from voxtutor.scorers import score_utterance
from voxtutor.synth import synth_utterance
from voxtutor.types import Config, ScoreResult


def _offset(name: str) -> int:
    return sum(ord(ch) for ch in name)


def run(config: Config) -> ScoreResult:
    settings = Settings(
        method=config.method,
        regime=config.regime,
        labels=config.labels,
        samples=config.samples,
        seed=config.seed,
        backend=config.backend,
    )

    # Inventory + utterances are method-independent (same audio for every scorer).
    data_rng = settings.rng(0xDA7A, _offset(config.regime))
    phset = build_phoneme_set(data_rng)
    utterances = [synth_utterance(phset, data_rng, config.regime) for _ in range(config.samples)]

    cost_fn = frame_template_cost
    if config.backend == "scipy":
        from voxtutor.scipy_check import cost_matrix_scipy

        cost_fn = cost_matrix_scipy

    method_rng = settings.rng(0x5EED, _offset(config.method))
    all_scores, all_labels = [], []
    for utt in utterances:
        all_scores.append(score_utterance(config.method, utt, phset, method_rng, cost_fn))
        all_labels.append(utt.is_mispronounced)

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    if config.labels == "scrambled":
        label_rng = settings.rng(0x5C4A, _offset(config.regime))
        labels = labels[label_rng.permutation(len(labels))]

    return ScoreResult(
        method=config.method,
        regime=config.regime,
        seed=config.seed,
        scores=scores,
        labels=labels,
        auroc=auroc(scores, labels),
    )


def run_settings(settings: Settings) -> ScoreResult:
    return run(settings.to_config())
