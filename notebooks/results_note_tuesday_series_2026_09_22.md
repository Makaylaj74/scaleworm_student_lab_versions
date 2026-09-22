*AI-generated draft (Claude, Anthropic) — for review. All numbers are produced by version-controlled scripts and data in this repo (SHAs cited below); scale-worm counts are Makayla Joseph's. Confirmed findings are labelled as such.*

# Results note — Tuesday scale-worm series + v2 detector saturation (for Dr. Soule)

**From:** Makayla Joseph · **Date:** 2026-09-22 · **Repo:** `scaleworm-student-lab` (branch `image-quality-survey`)
**Instrument:** OOI Cabled Array HD camera CAMHDA301 (RS03ASHS-PN03B-06), Mushroom vent, Axial Seamount

---

## 1. TL;DR

1. **The 2021–2023 manual Tuesday series is complete** — 229 front-on Scene-1 frames hand-counted on **49 Tuesdays**, **7,220 worms**, mean 31.5/frame (range 4–69). It stands on its own, like the Monday series, and needs no detector.
2. **The v2 detector saturates on these dense scenes (new, confirmed).** On the same 229 frames it recovers only **38%** of worms in aggregate, and its per-frame count explains just **20%** of the manual variance (R²=0.20). The AI count plateaus near 28 boxes while true counts reach 69. Recall falls monotonically with density (85% at <10 worms → 28% at 45+).
3. **This is a "No Borrowed Assumptions" correction.** The 2026-09-13 note reported v2 recall ≈ 66% and count agreement ≈ 92% — but that rested on **19 sparse clear frames (mean ≈ 11 worms)**. The Tuesday scenes average 31.5, so the Monday-clear recall does **not** transfer. A single scalar recall factor is invalid here.
4. **A density-corrected AI series can fill gaps but not replace counting.** Using an OLS calibration it extends coverage from 49 to **92 Tuesdays** over the same 2021–2023 window, but with deliberately wide, honest CIs — and because R²=0.20 it regresses toward the mean (~29 worms) and cannot resolve the peaks/troughs the hand counts show.

---

## 2. The manual Tuesday series (complete)

Every front-on Scene-1 frame in the weekly-Tuesday × 8-slot sampling was box-corrected by MJ (2026-09-20…22). As with the Mondays, the corrected boxes are the ground truth and `worm_count = number of boxes`, so one pass yields both the abundance series and detector-training labels.

| Metric | Value |
|---|---|
| Frames counted | 229 (`counted`) + 56 `unusable_blur` excluded |
| Worms counted | 7,220 |
| Distinct Tuesdays | 49 (2021-12-21 → 2023-08-08) |
| Mean per frame | 31.5 (range 4–69) |
| Tuesdays with an error bar (≥2 slots) | 43 (6 single-slot) |
| Split | 225 train / 60 val |

Each Tuesday = mean worm count across its up-to-8 three-hourly slots; single-slot Tuesdays carry no CI (left blank, *not* zero). **Headline (confirmed):** abundance rises from ~10/frame (early 2022) to a ~57 peak (Nov 2022), then settles to ~20–27 through 2023 — a real, resolved seasonal pattern.
Figure: `notebooks/figure_tuesday_manual_series.png`.

---

## 3. Detector saturation (new, confirmed)

`ai_count` is the raw v2 box count at conf 0.25 (`scripts/run_ai_counts.py`); all 229 Tuesday frames were scored by v2. Manual `worm_count` = label-box count, verified equal on all 229 (0 mismatches). None of the 229 were in v2 training (leakage-free).

| Manual density | n frames | AI / manual |
|---|---|---|
| 0–10 | 12 | 0.85 |
| 10–20 | 26 | 0.55 |
| 20–30 | 67 | 0.46 |
| 30–45 | 83 | 0.37 |
| 45+ | 41 | 0.28 |
| **Overall** | **229** | **0.38** |

The detector plateaus (max AI = 28 vs max manual = 69), Pearson r = 0.449. Because recall depends on density, **no constant factor is defensible**: it would badly under-correct the dense scenes. Figure: `notebooks/figure_tuesday_ai_saturation.png`.

---

## 4. Density-corrected AI series (secondary, caveated)

The 234 AI-scored frames that were never hand-counted fall on **43 entirely new Tuesdays** (zero overlap with the 49 manual ones), so a correction *adds temporal coverage within the same window* rather than extending the record.

**Method.** OLS calibration on the 229 pairs: `manual = 18.21 + 1.112 · ai` (R²=0.201, residual SD 11.9). Applied per slot, aggregated per Tuesday. Per-Tuesday uncertainty is a **paired bootstrap 95% CI** (n=10,000, seed 20260922) that resamples the 229 pairs, refits OLS, and adds a resampled residual per predicted slot (counts clipped at 0). The manual series uses the same percentile-bootstrap CI over its slots, so the two series are comparable on one axis and neither produces negative counts. Extrapolation is minimal: only 3 of 234 frames sit at AI=0, just below the calibration support (AI 1–28).

**Result.** Extended series = 92 Tuesdays (49 manual + 43 corrected). The corrected points cluster at 25–33 worms with CI widths ~17–44 (median ~19) — **conservative by construction**: with only 20% of variance explained, the estimator shrinks to the mean and cannot reproduce the hand-counted peaks (~57) or troughs (~8). It fills gaps; it does not replace counting. Figure: `notebooks/figure_tuesday_extended_series.png`.

---

## 5. Decisions for the meeting

1. **Present the corrected AI series at all, or only the manual series?** It doubles temporal coverage but flattens the signal; it may be more honest as a "why we hand-count" caveat than as a data product.
2. **Is the OLS form adequate,** or should we try a saturating/GLM model (e.g. Poisson with an offset)? Given R²=0.20, model choice barely moves the wide CIs, but worth stating.
3. **More clear/dense ground truth for v2?** The saturation is now well-constrained (n=229), but all in the clear window; blurry post-2023 frames (v3) are untested at this density.

---

## 6. Methods & reproducibility

- **Series + correction:** `scripts/build_tuesday_series.py` (+ `tests/test_build_tuesday_series.py`, 7 tests) →
  `validation/tuesday_manual_series/tuesday_manual_timeseries.csv`,
  `notebooks/tuesday_corrected_ai_timeseries.csv`,
  `notebooks/tuesday_extended_timeseries.csv`,
  `notebooks/tuesday_ai_vs_manual_pairs.csv`,
  `notebooks/tuesday_calibration.json`.
- **Figures:** `scripts/plot_tuesday_series.py` (Paper tier, 300 DPI, Okabe-Ito, per the lab timeseries rubric; PNGs are gitignored and regenerated from the CSVs).
- **AI counts:** `scripts/run_ai_counts.py`, v2 `best.pt`, conf 0.25 → `notebooks/ai_counts_tuesday_2021_2023.csv`.
- **Uncertainty:** percentile bootstrap, n=10,000, seed 20260922; counts clipped at 0.
- **Caveats:** the calibration is clear-window only (2021–2023); the corrected series is a gap-fill, not a measurement; the single 2-slot Tuesday (2022-11-01) has a wide but bounded CI.
- **Prior context:** the Monday-clear recall figures (≈66%) are in `notebooks/results_note_detector_recall_2026_09_13.md`; the Mon→Tue transfer validation is in `notebooks/results_note_tuesday_transfer_2026_09_17.md`.
