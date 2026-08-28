*AI-generated draft (Claude, Anthropic) — for review. All numbers and figures are derived from version-controlled scripts and data in this repo. Hand counts are Makayla Joseph's. Confirmed findings and hypotheses are labelled as such.*

# Image-quality decline and its effect on scale-worm counts (CAMHDA301, 2023–2024)

*Working analysis note — `scaleworm-student-lab`. Not the (phase-gated) thesis methods chapter.*
*Written 2026-08-28. Regenerate every figure/number from the scripts cited below.*

## Summary

We ran the `mushroom.pt` (v0) detector over one front-on frame per Monday recording
(367 "scene1" recordings, 2023–2024) to build a scale-worm count time series. The raw
series shows worm counts dropping from ~8–10/scene through mid-2023 to ~0–1/scene from
late 2023 onward. **This apparent decline is an artifact of declining camera image
quality (blur), not a change in the worm population.** The worms are still present; the
detector stops seeing them on the degraded footage.

## What is confirmed

1. **The worms are present the whole time.** Makayla hand-counted three months by eye:
   Jan 2023 = 14.7 worms/scene, Feb 2024 = 9.7, May 2024 = 7.5. Worms are clearly there
   in 2024 — the low model counts are not real absence.
   *(`notebooks/handcount_2023_01.csv`, `model_comparison_handcount.csv`, `handcount_2024_05.csv`.)*

2. **The detector's recall collapses on later footage.** Against those hand counts the
   model recovers 67 % of worms in Jan 2023 but only 5 % (Feb 2024) and 4 % (May 2024).
   *(`scripts/plot_recall_comparison.py`.)*

3. **The collapse tracks image sharpness, not biology.** Across all 20 months the worm
   count correlates with frame sharpness (Laplacian variance) at **r = 0.91
   (95 % CI 0.86–0.95)**, far more tightly than with global contrast (r = 0.71). Sharp
   months detect worms; blurry months do not — including a few 2023 transition months
   that are already blurry and already low-count.
   *(`scripts/compute_image_quality.py`, `plot_count_vs_blur.py`.)*

4. **It is not a camera hardware swap or format change.** Videos on both sides of the
   change are byte-for-byte the same class — 1920×1080, h264, 29.97 fps, ~15 min, ~948 MB.
   Brightness is unchanged (114.8 → 114.1). What changed is the *picture*, not the
   recording pipeline. *(ffprobe + `scripts/compute_image_quality.py`.)*

5. **The change is a loss of fine detail (blur), and it persists.** Sharpness fell from
   ~200–250 (2023) to ~130 and stayed there through all of 2024. Global RMS contrast
   dipped transiently (to ~18 in Dec 2023–Jan 2024) then recovered to ~39 — so contrast
   alone does *not* explain the persistent count failure; sharpness does.
   *(`scripts/plot_image_quality.py`.)*

The side-by-side detection figure makes the mechanism visible: on a sharp 2023 frame
(sharpness 777) the model boxes 7 worms; on a blurry 2024 frame (sharpness 118) it boxes
0, while a human counts 9 in the same frame. *(`scripts/plot_sharp_vs_blurry.py`.)*

## What is hypothesized (not yet confirmed)

- **Cause of the blur.** The signature — steady brightness, halved sharpness that never
  recovers, and a visible milky veil in the frames — is consistent with **biofouling /
  film on the camera dome or a focus drift**, developing around September 2023. This is a
  hypothesis from the image evidence, **not** a confirmed maintenance event.
- **Relation to servicing.** OOI runs annual servicing cruises; whether one coincides with
  (or caused) the change is checkable in OOI deployment/cruise records and has **not** been
  verified here. Earlier framing of a "camera replacement / servicing boundary" is
  explicitly withdrawn — the format-identity check (point 4) rules out new hardware.

## Effect on the counts — what you can and cannot say

- **Within the sharp 2023 record**, the model recovers ~⅔ of worms; counts are a usable
  *lower-bound relative index* (still an under-count, so not absolute abundance).
- **Across the September-2023 boundary the counts are not comparable.** Recall changes from
  67 % to ~5 % purely because of image quality, so any 2023-vs-2024 difference is dominated
  by the detector, not the worms. The apparent "collapse" must **not** be reported as a
  population trend.

## Recommendations

1. **Retrain v2 on the blurry post-2023 footage** (with labels drawn on that footage), so
   the detector learns worms as they appear in degraded video. Contrast enhancement alone
   (e.g. CLAHE) cannot restore detail that blur has removed.
2. **Flag the camera image-quality decline to OOI** and check deployment/cruise records for
   a cleaning/servicing event around Sep 2023 — both to explain the change and because
   persistent fouling affects all downstream video products, not just this one.
3. **Do not compute cross-boundary abundance trends** with the current model.
4. **Validate any future model** the same way — hand-counted months spanning both the sharp
   and blurry regimes, reporting recall with uncertainty.

## Reproducibility

Committed in `e449d60`. Pipeline: `run_full_timeseries.py` → per-scene counts;
`compute_image_quality.py` → per-scene brightness/contrast/sharpness; figures from
`plot_timeseries.py`, `plot_recall_comparison.py`, `plot_image_quality.py`,
`plot_count_vs_blur.py`, `plot_sharp_vs_blurry.py`. Detector `mushroom.pt` @ conf 0.25.
Bootstrap: 2000 resamples, seed 20260828, 2.5/97.5 percentiles.
