# Architecture

VoxTutor is a small, fully offline benchmark. Data flows in one direction: a synthesizer
produces learner utterances with known mispronunciations, a scorer assigns each phoneme a
mispronunciation score, and a metric grades the scores against the ground truth.

```
phonemes.py        synth.py              scorers.py + align.py        metrics.py
PhonemeSet   -->   Utterance        -->  per-position scores     -->  AUROC vs
(templates)        (frames, canon,       (segmentation x               ground-truth
                    is_mispronounced)      normalization)               mispronunciations
                          |                                                  |
                          +------------------ runner.run(Config) ------------+
                                                     |
                                      evals/ (metrics -> harness -> gate)
```

## Modules

- **`voxtutor/phonemes.py`** — the phoneme inventory: `N_PHONEMES` well-separated unit-vector
  templates in a `D_FEATURES`-dim acoustic space. Fixed for a run (it is the "language").
- **`voxtutor/synth.py`** — synthesizes one `Utterance`: a canonical phoneme sequence, with
  `N_MISPRON` positions nudged by `DELTA` (the injected mispronunciations = ground truth),
  rendered as frames. Three regimes layer a distortion: `clean` (control), `warped` (variable
  frames per phoneme — speaking-rate variation), `noisy` (heteroscedastic channel variance).
  The distortion never changes *which* positions are mispronounced.
- **`voxtutor/align.py`** — frame→position segmentation. `fixed_segmentation` (uniform chunks)
  vs `dtw_align` (forced alignment by dynamic time warping over a frame-template cost matrix).
- **`voxtutor/scorers.py`** — the 2×2 factorial. A scorer = (segmentation ∈ {fixed, DTW}) ×
  (normalization ∈ {raw mean distance, GOP centroid distance}). `naive`, `aligned`,
  `normalized`, `gop` are the corners; `random` is the chance baseline.
- **`voxtutor/metrics.py`** — threshold-free AUROC via the Mann-Whitney U statistic with
  average ranks for ties (no scipy/sklearn on the default path).
- **`voxtutor/runner.py`** — `run(Config)` builds the inventory + utterances (method-independent,
  so every scorer sees identical audio), scores each utterance, pools positions across the
  whole set, and returns the AUROC.
- **`voxtutor/config.py`** — `Settings` + the salted `np.random.default_rng` factory. Every
  random draw is reproducible from `(SALT, seed, offsets)`.
- **`voxtutor/scipy_check.py`** — optional cross-check of the cost matrix and AUROC against SciPy.
- **`evals/`** — `metrics` (seed aggregation) → `harness` (writes `RESULTS.md`) → `gate`
  (asserts the dissociation as a battery of inequalities; the CI quality gate).

## The 2×2 and why the scores behave

For a position's assigned frames, write the mean frame-to-template distance as

```
mean_t || f_t - template ||^2  =  || centroid - template ||^2  +  within-phone variance
        (raw score)                 (GOP / normalized score)        (channel noise term)
```

- **raw** keeps the variance term, so heteroscedastic channel noise (regime `noisy`) inflates
  it even when the phoneme is correct → false positives → collapse.
- **GOP / normalized** keeps only the centroid (bias) term, cancelling the channel variance.
- Both depend on the **segmentation** being right. Under `warped`, a fixed uniform split
  mis-assigns frames to the wrong phonemes, so even the centroid is wrong → collapse; DTW
  forced alignment recovers the true segmentation.

So alignment fixes `warped`, normalization fixes `noisy`, and only `gop` (both) is robust to
both. `clean` is the control where neither distortion is present and every scorer recovers.

## Determinism

`numpy` is the only runtime dependency. BLAS is pinned to one thread before numpy imports
(`voxtutor/__init__.py`, `tests/conftest.py`, CI `env:`). All randomness is salted
`np.random.default_rng`, which is stable across platforms and Python versions, so the eval
table — and the gate thresholds built around it — reproduce bit-for-bit on 3.10–3.12.
