*AI-generated draft (Claude, Anthropic) — for review. All parameters, statistics, and figures are derived from version-controlled scripts and data; every number below was recomputed from the committed gate CSVs (not transcribed from notes).*

# Scale-worm abundance: the density-estimation head (CSRNet)

## 1. Why this method exists — the density ceiling

The scale-worm series was originally counted with an object **detector** (YOLO lineage: `mushroom.pt` → v2–v8), which draws a bounding box around each worm and counts the boxes. This works in sparse frames but **saturates in crowded ones**: when worms pile onto the vent mound and occlude one another, the detector cannot separate adjacent individuals and systematically under-counts. This "density ceiling" was confirmed independently three times (tile-training, inference-time SAHI slicing, and a two-class region detector all failed to break it), and it is worst exactly where the science needs it most — the dense post-eruption 2015–2016 community that is the thesis motivation.

A **density-estimation head** is the standard computer-vision answer to crowded counting (crowd counting, cell counting), and is the first method in this project to beat the ceiling.

## 2. What it is and how it works

Instead of detecting and separating individual worms, a density head regresses a **density map**: a per-pixel surface whose value is "expected worms per unit area." The **integral of the map over the frame equals the predicted count**. Because it never has to resolve where one worm ends and the next begins, it is not bound by instance separation — the exact failure mode that caps the detector in dense frames.

> Intuition: to count a stadium crowd, the detector circles every head (and fails where heads overlap); the density head estimates people-per-square-metre across the stands and sums it. The second approach survives crowding.

Architecture: **CSRNet** — a torchvision ImageNet-pretrained VGG-16 front end (`features[:23]`) followed by six dilated-convolution layers and a 1×1 output, implemented in `notebooks/density_head.py`. Input frames are resized to 1024×576 (16:9 preserved); the output density map is 128×72 (stride 8). During training, ground-truth density maps are built on-the-fly from the existing single-class worm annotations, with the Gaussian spread (σ) **derived from this dataset's own annotation sizes** rather than borrowed from a crowd-counting paper (per the lab "No Borrowed Assumptions" rule); the map integral was verified to equal the exact annotation count (integration error 0.0).

## 3. How it is trained and calibrated against the hand-counts

The model is trained and judged entirely against **manual annotations** — the human count is the ground truth, and the model is measured against it, never the reverse.

- **Training data:** the leakage-safe `scaleworm_retrain_all` split (421 train / 87 validation frames, 2021–2024 Monday series), single-class worm boxes.
- **Calibration ("ratio-to-GT"):** on held-out frames the model's summed count is compared to the human count. The **ratio of predicted total to true total** is the recovery fraction. A ratio of 0.96 means the model recovers ~96 of every 100 hand-counted worms; counts are divided by the ratio to re-centre the series on the human scale. This makes each block a **GT-referenced relative index**, not a raw model output.
- **Gating:** "pass/fail" checks on held-out frames (correlation, error, ratio, with bootstrap CIs) decide whether a model is trusted. Gating *measures* reliability; it does not create it — only more/better annotation does that.

### Two-model split (adopted 2026-10-18)

One model does not fit both eras, so two calibrated models are kept:

| Model | File | Era served | Recovery ratio |
|---|---|---|---|
| **Clear** (in-distribution) | `csrnet_run/best.pt` | 2021–2024 clear frames | **0.96** → counts ÷0.96 |
| **2015–16 fine-tuned** | `csrnet_ft2015_run/best_ft2015.pt` | 2015–2016 (dense, out-of-distribution) | **0.61** → counts ÷0.61 |

The clear model trained on 2021–24 alone generalised poorly to the dense 2015–16 scene (a different visual regime: tubeworm mound vs. mature tower). Fine-tuning it on **MJ's 35 hand-counted 2015–16 anchor frames** (density maps built from her click points) roughly doubled its out-of-distribution recovery (0.34 → 0.61) — evidence the wall was a **data-coverage gap, not a fundamental limit**. The eras do not abut (the 2017–2020 record is a dead-end gap), so the seam between the two models is low-risk; both are put on a GT scale (÷0.96, ÷0.61) for any cross-era comparison.

## 4. Performance (recomputed from committed gate CSVs, bootstrap n=10 000, seed 0, percentile CIs)

**Clear / in-distribution** (`csrnet_run/gate_density_results.csv`, n=87 held-out 2021–24 frames):

| Counter | Pearson r | MAE (worms/frame) | Count ratio to GT |
|---|---|---|---|
| **Density head** | **0.943** (0.915–0.965) | **2.12** (1.67–2.63) | **0.96** (0.92–1.00) |
| Box detector v5 ÷0.72 | 0.862 | 4.85 | 1.23 (over-counts sparse) |

The density head halves the per-frame error and tracks the human count almost 1:1, with no regime-specific fudge factor.

**Dense / out-of-distribution** (`csrnet_ft2015_run/ft2015_cv_results.csv`, n=35 2015–16 anchor frames, 5-fold CV, GT mean 25.2 worms/frame):

| Counter | Pearson r | MAE | Count ratio to GT |
|---|---|---|---|
| **Density head, fine-tuned** | **0.603** (0.26–0.80) | **9.91** (8.34–11.44) | **0.61** (0.57–0.66) |
| Density head, pre-fine-tune | 0.331 | 16.83 | 0.34 |
| Box detector | 0.385 | 19.40 | 0.23 |

In the dense OOD regime the fine-tuned density head still under-counts (0.61) and is noisier (r≈0.60), but it roughly triples the detector's recovery (0.23 → 0.61) and turns a non-significant correlation into a real one.

## 5. Pros

- **Beats the density ceiling.** The only method tested that recovers crowded counts the detector cannot (clear in-dist 0.96 vs. detector 0.72-corrected over/under-count; dense OOD 0.61 vs. detector 0.23).
- **Near-1:1 and low-error where calibrated** (clear: r=0.94, MAE 2.1 worms/frame), with no sparse-vs-dense correction switch.
- **Fine-tunable into new regimes.** The 2015–16 result shows the OOD gap is a data problem — a modest set of era-specific hand-counts adapts it, opening the thesis-motivation eras.
- **Reuses existing annotations.** Density maps are built from the single-class boxes/points already produced; no new label *type* is required for the in-distribution model.
- **Fully reproducible.** Fixed seeds; every statistic regenerates from committed scripts and CSVs.

## 6. Cons and limitations

- **It is a relative index, not an absolute count.** The ÷ratio re-centres the *average* on the human scale, but the bias is **density-dependent** (sparse frames over-count, dense under-count), so a single scalar cannot make any single frame absolute.
- **The dense regime is still under-counted and noisy** (OOD ratio 0.61, r≈0.60). Fine-tuning softened the ceiling; it did not eliminate it.
- **Ground truth itself is ~2× uncertain.** Inter-rater reliability work shows human counters disagree by up to ~2× on dense frames, which caps the achievable accuracy of *any* count-based method and widens the OOD confidence intervals (ratio 0.61 CI 0.57–0.66, but r CI 0.26–0.80 on n=35).
- **Two models, two scales.** Cross-era *magnitude* comparisons (e.g. "2015 recovered to X% of 2023") cross a model seam and are the fragile claim; within-era *trends and timing* are robust.
- **Only as good as the frame it is given.** On the human-selected Scene-1 frames of 2021 it reads 0.83–0.86× hand counts (r = 0.81). On archive frames picked by the detector's max-over-grid it reads 0.38–0.51× in Jan–Aug 2021, because many recordings lack Scene-1 and the brief Scene-1 window is missed. Frame selection, not the counter, is the binding limit for the archive index (see `method_writeup_detector_2026_10_14.md` §5, §6.3).
- **No individual identities.** The method yields a count/abundance surface, not per-worm tracks — unsuitable for growth/turnover of individuals.
- **Small OOD calibration set** (n=35 for 2015–16); the dense ratio rests on relatively few anchor frames.

## 7. Recommended use

- **2021–2024 clear frames:** clear model (`best.pt`, ÷0.96) — trusted, near-absolute, low error.
- **2015–2016 dense frames:** fine-tuned model (`best_ft2015.pt`, ÷0.61) — a usable *relative* index for the post-eruption recovery era, reported with its uncertainty, not as an absolute headcount.
- **Making 2015–16 firmer** requires *more manual annotation of the type chosen by the advisor* (count vs. percent-cover), not more gating. Percent-cover is more repeatable between raters in dense mounds and may yield a genuinely absolute, reproducible metric there — at the cost of being a different quantity than the per-worm counts used for 2021–24.

## 8. Provenance

- **Model / training / gating:** `notebooks/density_head.py`, `gate_density.py`, `bootstrap_density_gate.py`, `finetune_density_2015_cv.py`, `train_deploy_density_2015.py`.
- **Series builders:** `build_worm_series_2015.py`, `build_worm_series_2016.py`, `build_worm_series_2021_2023_density.py`, `build_hybrid_series_2015_2024.py`.
- **Gate data (numbers in §4):** `csrnet_run/gate_density_results.csv` (n=87), `csrnet_ft2015_run/ft2015_cv_results.csv` (n=35).
- **Weights** (`best.pt`, `best_ft2015.pt`) are git-ignored per repo policy and regenerable from the committed scripts (fixed seeds).
- **Source data:** OOI Regional Cabled Array, ASHES vent field camera `RS03ASHS-PN03B-06-CAMHDA301` (CAMHD), public archive.
- Branch `image-quality-survey`.
