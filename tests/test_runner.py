import numpy as np

from voxtutor.config import Settings
from voxtutor.runner import run, run_settings
from voxtutor.types import Config


def test_run_is_deterministic():
    a = run(Config("gop", "warped", "real", 16, 0, "numpy"))
    b = run(Config("gop", "warped", "real", 16, 0, "numpy"))
    assert a.auroc == b.auroc
    assert np.array_equal(a.scores, b.scores)


def test_run_pools_all_positions():
    r = run(Config("naive", "clean", "real", 10, 0, "numpy"))
    assert len(r.scores) == len(r.labels)
    assert len(r.scores) == 10 * 12  # samples * N_POSITIONS
    assert int(r.labels.sum()) == 10 * 2  # samples * N_MISPRON


def test_scrambled_labels_drop_to_chance():
    real = run(Config("gop", "clean", "real", 40, 0, "numpy")).auroc
    scrambled = run(Config("gop", "clean", "scrambled", 40, 0, "numpy")).auroc
    assert real > 0.95
    assert 0.35 < scrambled < 0.65


def test_run_settings_matches_run():
    s = Settings(method="aligned", regime="noisy", samples=12, seed=1)
    assert run_settings(s).auroc == run(s.to_config()).auroc


def test_auroc_in_unit_interval():
    for method in ("random", "naive", "aligned", "normalized", "gop"):
        for regime in ("clean", "warped", "noisy"):
            r = run(Config(method, regime, "real", 8, 0, "numpy"))
            assert 0.0 <= r.auroc <= 1.0
