import numpy as np

from voxtutor.config import Settings, rng_from_seed


def test_rng_is_deterministic_for_same_offsets():
    a = Settings(seed=3).rng(1, 2, 3).standard_normal(8)
    b = Settings(seed=3).rng(1, 2, 3).standard_normal(8)
    assert np.array_equal(a, b)


def test_rng_varies_with_offsets_and_seed():
    base = Settings(seed=0).rng(1).standard_normal(8)
    assert not np.array_equal(base, Settings(seed=0).rng(2).standard_normal(8))
    assert not np.array_equal(base, Settings(seed=1).rng(1).standard_normal(8))


def test_from_env_defaults(monkeypatch):
    for k in ("VOXTUTOR_METHOD", "VOXTUTOR_REGIME", "VOXTUTOR_LABELS",
              "VOXTUTOR_SAMPLES", "VOXTUTOR_SEED", "VOXTUTOR_BACKEND"):
        monkeypatch.delenv(k, raising=False)
    s = Settings.from_env()
    assert s.method == "gop" and s.regime == "clean" and s.labels == "real"
    assert s.backend == "numpy"


def test_from_env_reads_overrides(monkeypatch):
    monkeypatch.setenv("VOXTUTOR_METHOD", "Naive")
    monkeypatch.setenv("VOXTUTOR_REGIME", "WARPED")
    monkeypatch.setenv("VOXTUTOR_SEED", "7")
    s = Settings.from_env()
    assert s.method == "naive" and s.regime == "warped" and s.seed == 7


def test_to_config_roundtrip():
    cfg = Settings(method="aligned", regime="noisy", seed=2).to_config()
    assert cfg.method == "aligned" and cfg.regime == "noisy" and cfg.seed == 2


def test_rng_from_seed_matches_settings():
    a = rng_from_seed(5, 9).standard_normal(4)
    b = Settings(seed=5).rng(9).standard_normal(4)
    assert np.array_equal(a, b)
