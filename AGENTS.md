# VoxTutor — agent guide (AGENTS.md)

Offline pronunciation-assessment benchmark for spoken-language tutoring. Utterances are
**synthesized** from a known phoneme inventory with mispronunciations injected at **known**
positions; pronunciation scorers are graded by AUROC of recovering them. Thesis: **forced
alignment** (DTW) buys robustness to speaking-rate variation and **GOP normalization** buys
robustness to channel noise — a 2×2 ablation where each single-ingredient scorer collapses on
the regime it can't handle, proven by a `clean` control that removes the distortion.

> `CLAUDE.md` is the canonical guide this file mirrors — **edit both together**.

## Commit policy (hard rule)
Author = **Rana Faraz only**. **Never** add a `Co-Authored-By` AI/assistant trailer or any
AI/assistant branding to commit messages. Keep history tidy and incremental.

## What must stay true (don't regress)
- **Offline & deterministic.** numpy is the only runtime dep. Every draw comes from
  `Settings.rng` (salted `np.random.default_rng`), never `hash()`/clock — so CI reproduces
  `evals/RESULTS.md` bit-for-bit on Python 3.10–3.12.
- **Synthesized, exact ground truth.** Utterances + injected mispronunciations are
  method-independent (same audio for every scorer). If you touch `synth.py`, keep the ground
  truth exact and identical across regimes.
- **BLAS pinned to 1 thread** before numpy imports (`voxtutor/__init__.py`, CI `env:`,
  `tests/conftest.py`). Never remove.
- **Tune the experiment, not the scorer.** The scorers are textbook (uniform segmentation, DTW
  forced alignment, GOP centroid normalization); only distortion strengths in `synth.py`
  (`DELTA`, warp range, `NOISE_MULT`, `NOISY_FRACTION`) are tuned. Don't tweak a scorer to clear
  the gate.
- **The dissociation must hold:** `warped` collapses the fixed-segmentation scorers (naive,
  normalized) only; `noisy` collapses the raw scorers (naive, aligned) only; `gop` stays ~1.0
  everywhere; random + the scrambled-label null sit at chance. `python -m evals.gate` asserts
  all of it (29 checks).

## Layout
`voxtutor/` — `phonemes` (template inventory), `synth` (utterances + injected
mispronunciations, regimes clean/warped/noisy), `align` (fixed_segmentation · dtw_align forced
alignment), `scorers` (2×2: naive · aligned · normalized · gop + random), `metrics` (AUROC),
`runner`, `config` (Settings + SALT rng), `cli`, `scipy_check` (optional). `evals/`
(metrics · harness · gate). `tests/` (56 + 4 scipy skipped). `examples/run_scoring.py`.

## Run (offline)
```
pip install -e ".[dev]"
pytest -q                  # 56 pass (+4 skipped without the [scipy] extra)
ruff check .
python -m evals.harness    # writes evals/RESULTS.md
python -m evals.gate       # asserts the dissociation (CI gate, 29/29)
voxtutor compare --regime warped
```
Backend matrix (env): `VOXTUTOR_METHOD` random|naive|aligned|normalized|gop ·
`VOXTUTOR_REGIME` clean|warped|noisy · `VOXTUTOR_LABELS` real|scrambled ·
`VOXTUTOR_SAMPLES` (40) · `VOXTUTOR_SEED` (0) · `VOXTUTOR_BACKEND` numpy|scipy
(optional `[scipy]` cross-check of the DTW cost matrix vs cdist and AUROC vs mannwhitneyu;
importorskip in tests, never on the default path).

## Env notes
Windows 11, PowerShell + Bash. venv at `.venv/Scripts/python.exe` (`python -m venv .venv`).
Windows console is cp1252 — CLI prints ASCII; the harness writes UTF-8 `RESULTS.md`.
`gh` authed as `ranafaraz`.
