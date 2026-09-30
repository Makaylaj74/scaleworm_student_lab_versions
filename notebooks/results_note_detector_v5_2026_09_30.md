*AI-generated draft (Claude, Anthropic) — for review. All numbers and figures are derived from version-controlled scripts and this session's verified outputs; regenerate via the scripts named below.*

# Detector retrain v5 — results note (2026-09-30)

> **⚠️ CORRECTION (2026-09-30, later same day): the blurry results below are INVALID.**
> The Monday blurry manual counts used both to *train* v5's blurry side (161 frames) and to
> *gate* it (`model_comparison_handcount`) were systematic under-counts (~5 worms/frame). v5
> learned to output ~5/frame on blurry footage regardless of true density, and the gate reused
> the same bad counts, so it "passed" circularly. Against properly hand-counted **Tuesday**
> blurry frames (mean 21/frame, dense), v5 recovers only **~32%**. **The "blurry 94%" claim and
> all blurry deployment guidance below are retracted.** The **clear-window results stand** — they
> were validated against independent *click* counts, not the suspect boxes. Fix in progress: a v6
> blurry retrain on the correct Tuesday box-labels. See project memory `scaleworm-detector-retrain-v5`.

Retrain of the scaleworm counting detector on the full accumulated manual annotation,
with a leakage-safe held-out gate, per-regime operating points, and an abundance
time series. Branch `image-quality-survey`; commits `d2493f1` (detector + gate) and
`cc9b301` (time series + 2015–2016 gate scaffold).

## Data

Training annotation = the **Monday manual series**
(`validation/monday_manual_series/`): **722 MJ box-corrected Scene-1 frames / 9,212
boxes**, 2021–2024 (~8× the earlier v2/v3 sets of 47 / 94 frames).

**Leakage control (day level).** Every prior acceptance-gate frame (39 clear + 31
blurry) also appears in the 722. Because the Monday series has up to ~8 slots per day,
we excluded any frame sharing a **calendar day** with a gate frame (`build_retrain_dataset.py`),
leaving **508 leakage-safe training frames** (347 clear + 161 blurry). The carved gate
frames are the held-out test, scored against their own independent counts (clear = clicks;
blurry = held-out MJ boxes). Train/val/test day-disjointness is asserted in code.

Two models fine-tuned from `mushroom.pt` on the Hub H100 (imgsz 1280, `train_v2.py`):
`v5_all` (all eras, 421/87) and `v5_clear` (clear only, 289/58).

## Gate results (held-out, 0/70 leakage; `gate_retrain.py`)

Point-matched detection (honest metric):

| Model | Clear rec/prec/F1 | Blurry rec/prec/F1 |
|---|---|---|
| v0 mushroom (baseline) | 48 / 81 / 60 | 9 / 94 / 16 |
| **v5_all** | 79 / 63 / 70 | **70 / 73 / 71** |
| **v5_clear** | 72 / 71 / 71 | 38 / 85 / 53 |

**Operating points (from the conf sweep, `sweep_conf_retrain.py`).** Blurred footage
yields systematically lower-confidence boxes, so a per-regime threshold keeps counts
calibrated: **v5_all @ conf 0.40 for clear, 0.25 for blurry** → count-ratio 100 % / 95 %,
F1 ~70 % both regimes. `v5_clear` is redundant with `v5_all` @ 0.40 on clear and weak on
blurry, so it is retained only as a cross-check.

**Headline:** blurry post-2023 footage — previously unusable (~9–33 %) — now recovers
**94 % of manual counts (95 % out-of-sample)**. This disproves the earlier "blurry label
ceiling" hypothesis; the fix was more blurry training labels (39 → 161), not the recipe.

## Abundance time series (`run_ai_counts.py`, `plot_v5_series.py`)

`notebooks/figure_worms_over_time_v5.png` overlays the manual ground truth and the v5_all
AI series (per-regime conf), monthly means, 2021–2024. Out-of-sample check on **213
manually-counted frames never in training: r = 0.89**, so the tracking is not memorization.

## Known limitation — clear-peak crowding under-count (documented, not corrected)

v5_all **compresses high-density clear frames**. Out-of-sample clear recovery is
density-dependent:

| Manual density (worms/frame) | AI count-ratio |
|---|---|
| 0–8 | 93 % |
| 8–15 | 87 % |
| 15–22 | 78 % |
| >22 (fall-2022 peak) | 64 % |

This is a **crowding/detection ceiling, not a threshold or resolution issue**: higher-
resolution inference does not fix it (dense bin ≈ 62–65 % at imgsz 1280/1536/1920,
`experiment_imgsz_density.py`), and dense frames were present in training. A global power
correction (`build_density_correction.py`, `manual = 1.882·ai^0.825`, out-of-sample
validated) repairs the **aggregate** (clear 82 % → 99 %, MAE 3.48 → 3.14) but cannot fix
all density bins simultaneously (it over-inflates low density), because a 1-D correction
cannot undo a bias that depends on unobserved true density.

**Decision (2026-09-30):** report v5 clear as a **conservative, shape-preserving abundance
index** (temporal pattern preserved, r = 0.89) with this caveat stated; do **not** apply a
correction. Rationale: the 2021–2024 clear record already has manual counts as ground
truth, so v5's clear value is future automation, not this window; the blurry years (the
real gain) need no correction. The correction parameters are archived in
`validation/v5_density_correction.json` should a fully-automated clear series ever be needed.

## Deployment recommendation

- **Blurry post-Aug-2023:** use `v5_all` @ conf 0.25 (94 % recovery) — the primary new capability.
- **Clear pre-Aug-2023 (2021–2024):** manual counts remain ground truth; `v5_all` @ 0.40 is a
  conservative index for automation, under-counting dense peaks by a known, documented amount.
- **New/uncounted footage:** `v5_all` with the per-regime threshold; hand-count a small gate
  before trusting any new era (as done here).
- **2015–2016 early-recovery era:** NOT covered — out-of-distribution, count target undefined.
  Gate scaffold staged at `validation/early_recovery_handcount_2015_2016/`
  (`build_early_recovery_gate.py`); blocked on the advisor count-target decision.

## Reproduce

`build_retrain_dataset.py` → `train_v2.py` (×2) → `gate_retrain.py` / `sweep_conf_retrain.py`
/ `spotcheck_retrain.py` → `run_ai_counts.py` → `plot_v5_series.py`. Tests:
`tests/test_retrain_gate.py` (full suite 204 pass). Weights + frames + figures are
gitignored (regenerable).
