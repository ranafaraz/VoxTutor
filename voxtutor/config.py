"""Environment-driven settings and a single deterministic RNG factory.

Every random draw in VoxTutor (the phoneme inventory, each utterance's phoneme sequence,
which positions are mispronounced, the channel noise, the random-scorer baseline) comes
from :meth:`Settings.rng`, seeded from a fixed salt plus explicit integer offsets -- never
from ``hash()`` or wall-clock time -- so a given (method, regime, seed) reproduces
bit-for-bit on every machine and Python version.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from voxtutor.types import Config

# Fixed salt so seeds are stable across machines and Python versions.
SALT = 0x76_78  # "vx" -- arbitrary constant, never change once published.

DEFAULT_SAMPLES = 40  # utterances pooled per (method, regime, seed)
DEFAULT_SEED = 0


@dataclass(frozen=True)
class Settings:
    method: str = "gop"
    regime: str = "clean"
    labels: str = "real"
    samples: int = DEFAULT_SAMPLES
    seed: int = DEFAULT_SEED
    backend: str = "numpy"

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            method=os.environ.get("VOXTUTOR_METHOD", "gop").strip().lower(),
            regime=os.environ.get("VOXTUTOR_REGIME", "clean").strip().lower(),
            labels=os.environ.get("VOXTUTOR_LABELS", "real").strip().lower(),
            samples=int(os.environ.get("VOXTUTOR_SAMPLES", DEFAULT_SAMPLES)),
            seed=int(os.environ.get("VOXTUTOR_SEED", DEFAULT_SEED)),
            backend=os.environ.get("VOXTUTOR_BACKEND", "numpy").strip().lower(),
        )

    def to_config(self) -> Config:
        return Config(
            method=self.method,
            regime=self.regime,
            labels=self.labels,
            samples=self.samples,
            seed=self.seed,
            backend=self.backend,
        )

    def rng(self, *offsets: int) -> np.random.Generator:
        """A fresh Generator seeded from SALT, the run seed, and explicit offsets."""
        state = (SALT * 0x9E3779B1) ^ (int(self.seed) & 0xFFFFFFFF)
        for off in offsets:
            state = (state * 0x100000001B3) ^ (int(off) & 0xFFFFFFFFFFFFFFFF)
            state &= 0xFFFFFFFFFFFFFFFF
        return np.random.default_rng(state & 0xFFFFFFFF)


def rng_from_seed(seed: int, *offsets: int) -> np.random.Generator:
    """Stand-alone RNG factory for code paths that don't hold a Settings."""
    return Settings(seed=seed).rng(*offsets)
