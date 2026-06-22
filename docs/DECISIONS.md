# Design decisions

What I built and why — the choices that make VoxTutor a real, reproducible result rather than
a demo.

## Synthesize the speech, don't record it
A pronunciation-assessment benchmark needs **ground truth**: which phonemes were actually
mispronounced. With real speech that means a labelled corpus (expensive, licensed, and the
labels are themselves a judgement call). Instead I generate utterances from a known phoneme
inventory and *inject* the mispronunciations myself, so the ground truth is exact and the whole
benchmark runs offline with no downloads. The cost is realism — these are template-space
features, not waveforms — but the **mechanism** under test (segmentation + scoring) is the same
one a real GOP pipeline uses, and the failure modes are the real ones.

## A 2×2 factorial, not a leaderboard
The interesting claim is not "scorer X wins" but *why* a scorer wins: which ingredient buys
robustness to which distortion. So the scorers are the four corners of
{segmentation: fixed | DTW} × {normalization: raw | GOP}, and there are two control regimes that
each toggle one distortion. That gives a **dissociation**: `aligned` is robust to `warped` but
not `noisy`; `normalized` is the mirror image; `gop` (both) is robust to both; `naive` (neither)
fails on both. Each effect is proven by the `clean` control where the distortion is absent and
the "failing" scorer recovers — so the collapse is the missing ingredient, not the scorer.

## The normalization is the textbook GOP, derived from bias–variance
Goodness-of-Pronunciation normalizes the raw acoustic score so that intrinsic acoustic
variability doesn't look like a mispronunciation. I implement it as the distance of a position's
**centroid** to the canonical template, which — by the bias–variance identity — is exactly the
raw mean frame-distance minus the within-phone variance. That makes the channel-noise
cancellation provable (and unit-tested in `test_normalized_equals_raw_minus_within_phone_variance`),
not hand-wavy.

## Forced alignment is plain DTW
The alignment ingredient is a textbook monotonic DTW over a frame-template cost matrix, allowing
each canonical phoneme to span a variable number of frames. No HMM, no acoustic model — the
point is to show that *re-segmenting* is what survives speaking-rate variation, and DTW is the
minimal honest way to do it. Its forward pass is vectorized over positions so CI stays fast; its
correctness is cross-checked against a brute-force cost matrix and (optionally) SciPy.

## Tune the experiment, never the scorer
The scorers are fixed textbook implementations. The only knobs are the *distortion strengths* in
`synth.py` (`DELTA`, the warp range, `NOISE_MULT`, `NOISY_FRACTION`) — chosen so each failure is
clearly visible with ~0.2 AUROC of headroom to the gate thresholds. Mispronunciations are
deliberately **subtle** (`DELTA = 0.65`): a confusable error, so that a channel-noise or
misalignment artefact can plausibly rival it. If the mispronunciations were obvious full-phoneme
swaps, no distortion could fool any scorer and there would be nothing to measure.

## Offline, deterministic, gated
numpy-only, salted `np.random.default_rng`, BLAS pinned to one thread → the eval table
reproduces bit-for-bit across machines and Python 3.10–3.12. That is what lets `evals/gate.py`
assert the dissociation with tight numeric margins and fail CI if it ever regresses, rather than
eyeballing a plot. SciPy is an optional cross-check only, never on the default path.

## Why these numbers
`SAMPLES = 40` utterances × `SEEDS = 12` pools enough positions (≈ 5760 per cell) for stable
AUROC while keeping the whole benchmark — harness *and* gate — under a minute, so it fits a CI
job comfortably alongside the test suite.
