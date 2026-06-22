"""Run the full offline benchmark and write ``evals/RESULTS.md``.

Every scorer is graded on every regime by the AUROC of recovering the known injected
mispronunciations, averaged over seeds. The result is a 2x2 dissociation:

1. **Speaking-rate variation** -- on ``warped`` a uniform segmentation mis-assigns frames,
   so the fixed-segmentation scorers (``naive``, ``normalized``) fall toward chance while
   the forced-alignment scorers (``aligned``, ``gop``) are untouched. The ``clean`` control
   (fixed rate) shows the fixed-segmentation scorers recover.
2. **Channel noise** -- on ``noisy`` heteroscedastic variance inflates raw frame distance,
   so the raw scorers (``naive``, ``aligned``) fall toward chance while the normalized
   scorers (``normalized``, ``gop``) are untouched. The ``clean`` control (base noise)
   shows the raw scorers recover.

``gop`` (forced alignment + GOP normalization) stays near 1.0 across all three. A
scrambled-label run drops every scorer to chance, confirming the metric is honest.
"""

from __future__ import annotations

import io
from pathlib import Path

from evals.metrics import mean_auroc, std_auroc
from voxtutor.scorers import ALIGNMENT, NORMALIZED, SCORERS
from voxtutor.synth import REGIMES

SEEDS = 12
SAMPLES = 40  # utterances pooled per cell

RESULTS_PATH = Path(__file__).resolve().parent / "RESULTS.md"


def compute() -> dict:
    matrix: dict[str, dict[str, dict[str, float]]] = {}
    for method in SCORERS:
        matrix[method] = {}
        for regime in REGIMES:
            matrix[method][regime] = {
                "mean": mean_auroc(method, regime, SAMPLES, SEEDS),
                "std": std_auroc(method, regime, SAMPLES, SEEDS),
            }
    # Scrambled-label null: shuffle the ground truth -> every scorer should hit chance.
    null = {
        method: sum(
            mean_auroc(method, regime, SAMPLES, SEEDS, labels="scrambled")
            for regime in REGIMES
        ) / len(REGIMES)
        for method in SCORERS
    }
    return {"matrix": matrix, "null": null, "meta": {"seeds": SEEDS, "samples": SAMPLES}}


def _ingredients(method: str) -> str:
    if method == "random":
        return "baseline"
    return ("align" if method in ALIGNMENT else "fixed") + \
           " + " + ("gop-norm" if method in NORMALIZED else "raw")


def _render(results: dict) -> str:
    out = io.StringIO()
    w = out.write
    m = results["meta"]
    mat = results["matrix"]
    w("# VoxTutor — offline benchmark results\n\n")
    w(f"Seeds: **{m['seeds']}** · utterances/cell: **{m['samples']}** · each cell is the "
      "mean AUROC of recovering the known injected mispronunciations (1.0 = perfect, "
      "0.5 = chance). No acoustic model, no audio downloads, no API keys.\n\n")

    w("## AUROC by scorer and regime\n\n")
    w("| scorer | ingredients | clean (control) | warped | noisy |\n")
    w("|---|---|--:|--:|--:|\n")
    for method in SCORERS:
        r = mat[method]
        w(f"| {method} | {_ingredients(method)} | {r['clean']['mean']:.3f} | "
          f"{r['warped']['mean']:.3f} | {r['noisy']['mean']:.3f} |\n")
    w("\n")

    naive_clean = mat["naive"]["clean"]["mean"]
    norm_warp = mat["normalized"]["warped"]["mean"]
    aligned_warp = mat["aligned"]["warped"]["mean"]
    aligned_noisy = mat["aligned"]["noisy"]["mean"]
    norm_noisy = mat["normalized"]["noisy"]["mean"]
    gop_warp = mat["gop"]["warped"]["mean"]
    gop_noisy = mat["gop"]["noisy"]["mean"]

    w("## Effect 1 — forced alignment beats speaking-rate variation\n\n")
    w(f"On `warped`, phonemes have unequal durations, so a uniform segmentation drifts and "
      f"mis-assigns frames. Fixed-segmentation scorers fall (`normalized` "
      f"{naive_clean:.3f} clean to **{norm_warp:.3f}**), while forced alignment holds "
      f"(`aligned` **{aligned_warp:.3f}**, `gop` **{gop_warp:.3f}**) — DTW re-segments the "
      "frames against the canonical sequence.\n\n")

    w("## Effect 2 — GOP normalization beats channel noise\n\n")
    w(f"On `noisy`, a subset of phonemes is recorded at much higher variance, inflating the "
      f"raw frame-to-template distance even when the phoneme is correct. Raw scorers fall "
      f"(`aligned` **{aligned_noisy:.3f}**), while centroid / likelihood-ratio "
      f"normalization holds (`normalized` **{norm_noisy:.3f}**, `gop` **{gop_noisy:.3f}**) "
      "— it subtracts the within-phone variance and keeps only the genuine error.\n\n")

    w("Only **gop** — forced alignment *and* GOP normalization — is robust to both "
      "distortions; each single-ingredient ablation collapses on the regime it cannot "
      "handle, and `naive` (neither) collapses on both.\n\n")

    w("## Scrambled-label null (sanity)\n\n")
    w("Shuffle the ground-truth labels and every scorer collapses to chance, so the AUROC "
      "above is not an artefact of the metric.\n\n")
    w("| scorer | null AUROC |\n|---|--:|\n")
    for method in SCORERS:
        w(f"| {method} | {results['null'][method]:.3f} |\n")
    w("\n> Reproduce: `python -m evals.harness` (writes this file), "
      "`python -m evals.gate` (asserts the dissociation).\n")
    return out.getvalue()


def main() -> None:
    results = compute()
    RESULTS_PATH.write_text(_render(results), encoding="utf-8")
    print(f"wrote {RESULTS_PATH}")
    # ASCII-only console summary (Windows cp1252 safe).
    mat = results["matrix"]
    print(f"{'scorer':14s}{'clean':>9s}{'warped':>9s}{'noisy':>9s}")
    for method in SCORERS:
        r = mat[method]
        print(f"{method:14s}{r['clean']['mean']:9.3f}"
              f"{r['warped']['mean']:9.3f}{r['noisy']['mean']:9.3f}")


if __name__ == "__main__":
    main()
