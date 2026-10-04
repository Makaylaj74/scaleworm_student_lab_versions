*AI-generated draft (Claude, Anthropic) — for review. All numbers derive from the
version-controlled scripts `plot_worm_dwdt_automated.py` and `gate_scale_curve.py` run on
2026-10-04 against committed harvest data, models, and held-out labels; regenerate with the
commands at the bottom. Ethical-check (7-point) cleared: all first-party derived data, no new
literature / PII / embargo; MJ hand-counts credited as ground truth.*

# Automated dw/dt (clear window) + Tier-B learning curve — results note (2026-10-04)

## 1. Automated dw/dt now exists (clear window)

Until today the dw/dt figure was built from the **manual** Monday series. The automated
clear-window abundance harvest (`harvest_mf_2021_2023.csv`, M-F × 8 slots, scene-sort-free,
`count_corrected` = med-of-top-3 ÷0.72) is validated (90% of manual Monday) but had never been
turned into a rate-of-change figure. It now is:
`notebooks/figure_worm_dwdt_automated.png` (`scripts/plot_worm_dwdt_automated.py`).

- **Coverage:** CLEAR window only, 2021-09-01 .. 2023-08-11 — **102 weeks → 25 months → 24
  dw/dt points**. This is the one regime where the detector is trustworthy; blurry/dense eras
  are deliberately excluded (see §2 and the v6 gate).
- **Added value over the manual figure:** ~40 M-F recordings/week vs 8 manual Mondays → a less
  noisy derivative. Weekly means are the independent unit; monthly mean/SEM are taken over
  weekly means (mirrors the manual figure's aggregation).
- **Honest level caveat:** automated monthly abundance spans **4.0–22.7/frame** vs the manual
  check **8.7–29.2/frame** — the automated series reads ~25–30% low, because ÷0.72 corrects
  moderate density but under-corrects crowded peaks (the documented density-compression limit).
  For a derivative this scales the dw/dt *amplitude* but preserves *sign and timing*, so the
  figure is a valid shape/phase record, not an absolute rate. Manual Monday is overlaid as the
  independent check.

Panels: A automated abundance (+ manual check), B automated dw/dt (gap-aware month-to-month),
C volcano reloading (caldera uplift + seismicity, full-record context).

## 2. Tier-B learning curve: density ceiling CONFIRMED (negative result)

Three nested models (clear + first N dense Tuesday-blurry labels, N ∈ {15, 30, 47}) were
trained 2026-09-30 but never scored. Gated today on the **same held-out 23 dense Tuesday
frames** as the v6 gate (conf 0.25, point-matched vs MJ box centres, bootstrap 95% CI,
leakage re-verified 0/23). `scripts/gate_scale_curve.py` →
`validation/scale_curve_gate.csv`, `notebooks/figure_scale_learning_curve.png`.

| Model | N dense labels | Point recall [95% CI] | Count-ratio | Precision | F1 |
|---|---|---|---|---|---|
| v5_all (baseline) | 0 | 24.4% [21.7, 30.7] | 29.5% | 82.5% | 37.6% |
| v6_scale_n15 | 15 | 21.9% [18.6, 26.2] | 27.1% | 80.9% | 34.5% |
| v6_scale_n30 | 30 | 27.6% [25.3, 32.5] | 35.8% | 77.3% | 40.7% |
| v6_scale_n47 | 47 | 20.9% [18.5, 25.4] | 94.5%* | 22.1% | 34.2% |

\* n47 precision 94.5% (FP=7) — it got *more conservative*, not better.

**Verdict: flat.** Slope n15→n47 = **−1.0%**; the curve is noise (n30 up, n47 back down), every
CI overlaps the v5_all baseline. Adding dense-blurry labels from 15→47 did **not** lift recall.
This reproduces and extends the v6_blurry finding (47 real dense labels barely moved F1): the
dense/blurry wall is a **detector / image-quality limit, not a label-quantity problem.**

## 3. What this means for the goal (M-F × 8/day × 2015–2024 automated dw/dt)

- **Clear window (2021-09 .. 2023-08): DONE.** Automated abundance *and* dw/dt both exist and
  are validated. This is a real, defensible deliverable today.
- **Do NOT label the remaining 59 Tuesday frames.** Tier B says it won't help; that effort is
  retired.
- **Blurry / dense / OOD eras (post-Aug-2023, 2015–2016, 2017–2020) remain NOT automatable**
  with this detector family. The path forward there is **not another retrain** but a different
  approach: image restoration/deblurring, a density/%-cover target instead of per-worm counts,
  or accepting those eras as manual. This is an advisor-level scoping decision.
- **2015–2016 (eruption-recovery motivation)** stays blocked on Dax defining the dense-mound
  count target — Tier B confirms we can't brute-force it with labels first.

## Reproduce

```bash
TV=/home/jovyan/joseph-scaleworm-thesis/.venv/bin/python3
$TV scripts/plot_worm_dwdt_automated.py      # -> notebooks/figure_worm_dwdt_automated.png
$TV scripts/gate_scale_curve.py              # -> validation/scale_curve_gate.csv + figure
```
Inputs: `notebooks/harvest_mf_2021_2023.csv`, `validation/monday_manual_series/
monday_manual_timeseries.csv`, geophysics weekly CSVs; models `99_runs/scaleworm_v6_scale_n{15,30,47}`,
`99_runs/scaleworm_v5_all`; held-out set `datasets/scaleworm_v6_blurry/tuesday_blurry_test.txt`.
Figures + model weights are gitignored (regenerable); the gate CSV is committed.
