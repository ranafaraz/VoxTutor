import numpy as np

from voxtutor.metrics import auroc


def test_perfect_separation():
    scores = np.array([0.1, 0.2, 0.9, 1.0])
    labels = np.array([False, False, True, True])
    assert auroc(scores, labels) == 1.0


def test_inverted_separation():
    scores = np.array([0.9, 1.0, 0.1, 0.2])
    labels = np.array([False, False, True, True])
    assert auroc(scores, labels) == 0.0


def test_chance_for_ties():
    scores = np.ones(6)
    labels = np.array([True, False, True, False, True, False])
    assert np.isclose(auroc(scores, labels), 0.5)


def test_empty_class_returns_chance():
    assert auroc(np.array([1.0, 2.0]), np.array([False, False])) == 0.5


def test_matches_rank_definition_random():
    rng = np.random.default_rng(0)
    scores = rng.standard_normal(200)
    labels = rng.random(200) < 0.3
    # brute-force probability a positive outranks a negative (ties count half)
    pos, neg = scores[labels], scores[~labels]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    assert np.isclose(auroc(scores, labels), wins / (len(pos) * len(neg)))
