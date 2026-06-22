"""Quality gate: fail CI unless the 2x2 dissociation actually holds.

The gate asserts the *shape* of the result, not a single lucky number:

* every scorer recovers the mispronunciations on the ``clean`` control;
* on ``warped`` the fixed-segmentation scorers collapse and the forced-alignment scorers
  do not (alignment is the ingredient that matters for speaking-rate variation);
* on ``noisy`` the raw scorers collapse and the GOP-normalized scorers do not
  (normalization is the ingredient that matters for channel noise);
* ``gop`` (both ingredients) stays faithful on every regime and beats each single-ingredient
  ablation on the regime that ablation fails;
* the random baseline and the scrambled-label null sit at chance.

Thresholds sit well inside the measured margins (collapses land near 0.62-0.73, robust
cells near 1.0); ``np.random.default_rng`` is stable across platforms and Python versions,
so the numbers are identical on CI.
"""

from __future__ import annotations

import sys

from evals.harness import compute
from voxtutor.scorers import ALIGNMENT, NORMALIZED, SCORERS

ROBUST_MIN = 0.95     # a scorer with the right ingredient recovers
COLLAPSE_MAX = 0.80   # a scorer missing the ingredient falls clearly below recovery
DISSOC_MARGIN = 0.15  # clean control exceeds the collapsed cell by this much
GAP_MARGIN = 0.15     # gop beats the fooled ablation by this much at its failure regime
CHANCE_MAX = 0.65     # baseline / null sit near chance

REAL = [m for m in SCORERS if m != "random"]
FIXED_SEG = [m for m in REAL if m not in ALIGNMENT]   # collapse under speaking-rate warp
RAW = [m for m in REAL if m not in NORMALIZED]         # collapse under channel noise


def _check(checks: list[tuple[bool, str]], ok: bool, msg: str) -> None:
    checks.append((ok, msg))


def run_checks() -> list[tuple[bool, str]]:
    r = compute()
    mat = r["matrix"]
    null = r["null"]
    checks: list[tuple[bool, str]] = []

    def cell(method: str, regime: str) -> float:
        return mat[method][regime]["mean"]

    # ---- control: every scorer recovers on the clean regime ----
    for m in REAL:
        v = cell(m, "clean")
        _check(checks, v >= ROBUST_MIN, f"clean: {m} recovers ({v:.3f} >= {ROBUST_MIN})")

    # ---- Effect 1: forced alignment is what survives speaking-rate variation ----
    for m in FIXED_SEG:  # naive, normalized
        cl, wp = cell(m, "clean"), cell(m, "warped")
        _check(checks, wp <= COLLAPSE_MAX,
               f"warped: {m} (fixed seg) collapses ({wp:.3f} <= {COLLAPSE_MAX})")
        _check(checks, cl - wp >= DISSOC_MARGIN,
               f"{m}: clean {cl:.3f} - warped {wp:.3f} >= {DISSOC_MARGIN}")
    for m in sorted(ALIGNMENT):  # aligned, gop
        v = cell(m, "warped")
        _check(checks, v >= ROBUST_MIN,
               f"warped: {m} (forced align) unaffected ({v:.3f} >= {ROBUST_MIN})")

    # ---- Effect 2: GOP normalization is what survives channel noise ----
    for m in RAW:  # naive, aligned
        cl, ns = cell(m, "clean"), cell(m, "noisy")
        _check(checks, ns <= COLLAPSE_MAX,
               f"noisy: {m} (raw) collapses ({ns:.3f} <= {COLLAPSE_MAX})")
        _check(checks, cl - ns >= DISSOC_MARGIN,
               f"{m}: clean {cl:.3f} - noisy {ns:.3f} >= {DISSOC_MARGIN}")
    for m in sorted(NORMALIZED):  # normalized, gop
        v = cell(m, "noisy")
        _check(checks, v >= ROBUST_MIN,
               f"noisy: {m} (gop-norm) unaffected ({v:.3f} >= {ROBUST_MIN})")

    # ---- gop has both ingredients: faithful everywhere, beats each ablation ----
    for regime in ("clean", "warped", "noisy"):
        v = cell("gop", regime)
        _check(checks, v >= ROBUST_MIN, f"{regime}: gop faithful ({v:.3f} >= {ROBUST_MIN})")
    # on warped gop beats the fixed-seg ablation `normalized`; on noisy beats raw `aligned`
    gw, nw = cell("gop", "warped"), cell("normalized", "warped")
    _check(checks, gw - nw >= GAP_MARGIN,
           f"gop beats normalized on warped ({gw:.3f} - {nw:.3f} >= {GAP_MARGIN})")
    gn, an = cell("gop", "noisy"), cell("aligned", "noisy")
    _check(checks, gn - an >= GAP_MARGIN,
           f"gop beats aligned on noisy ({gn:.3f} - {an:.3f} >= {GAP_MARGIN})")

    # ---- chance floors ----
    for regime in ("clean", "warped", "noisy"):
        v = cell("random", regime)
        _check(checks, v <= CHANCE_MAX, f"{regime}: random at chance ({v:.3f} <= {CHANCE_MAX})")
    for m in SCORERS:
        v = null[m]
        _check(checks, v <= CHANCE_MAX, f"scrambled null: {m} at chance ({v:.3f} <= {CHANCE_MAX})")

    return checks


def main() -> int:
    checks = run_checks()
    passed = 0
    for ok, msg in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {msg}")
        passed += ok
    total = len(checks)
    print(f"\nGate: {passed}/{total} checks passed.")
    if passed == total:
        print("PASSED")
        return 0
    print("FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
