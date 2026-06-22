"""Minimal example: watch the 2x2 dissociation on three regimes.

    python examples/run_scoring.py

Prints, for each regime, the AUROC of every scorer at recovering the known injected
mispronunciations -- so you can see the fixed-segmentation scorers collapse on `warped`,
the raw scorers collapse on `noisy`, and `gop` (forced alignment + GOP normalization) stay
faithful on both.
"""

from __future__ import annotations

from voxtutor.runner import run
from voxtutor.scorers import SCORERS
from voxtutor.synth import REGIMES
from voxtutor.types import Config


def main() -> None:
    print(f"{'scorer':14s}" + "".join(f"{r:>10s}" for r in REGIMES))
    print("-" * (14 + 10 * len(REGIMES)))
    for method in SCORERS:
        cells = []
        for regime in REGIMES:
            r = run(Config(method, regime, "real", 40, 0, "numpy"))
            cells.append(f"{r.auroc:10.3f}")
        print(f"{method:14s}" + "".join(cells))
    print("\n* fixed-segmentation scorers (naive, normalized) drop on 'warped';")
    print("  raw scorers (naive, aligned) drop on 'noisy'; gop stays ~1.0 on both.")


if __name__ == "__main__":
    main()
