*AI-generated draft (Claude, Anthropic) — for review. All counts, model parameters, and results are derived from version-controlled scripts, datasets, and gate CSVs in the `scaleworm-student-lab` repository and were re-verified against disk on 2026-10-06. Geophysical interpretation is deliberately excluded per request.*

# Scale-worm automated counting pipeline — progress report

**Project:** automated scale-worm abundance from the OOI ASHES `CAMHDA301` seafloor camera (Mushroom hydrothermal vent), 2015–2024.
**Date:** 2026-10-06 · **Repo:** `scaleworm-student-lab` (branch `image-quality-survey`)

---

## 1. Why an automated detector is the critical path

The end goal is a scale-worm abundance time series covering **Monday–Friday, 8 recordings/day (the eight 3-hourly slots), 2015–2024** — on the order of **13,000–20,000 Scene-1 frames**. That is roughly 25× what can realistically be hand-counted (≈1,000 frames), so counting every frame by hand is impossible. The chosen route is therefore:

1. Hand-**label** a few hundred representative frames per image regime (clear, blurry, dense-mound, early-era).
2. **Fine-tune** an object detector on those labels.
3. Let the detector **count** the remaining tens of thousands of frames.

The detector is thus the bottleneck, not a side tool. Everything below is in service of making it reliable across the camera's very different image regimes over the decade.

---

## 2. Manual annotations (the training + validation foundation)

Two kinds of manual work exist, by **two different methods**, kept strictly separate:

### 2a. Box annotations (used for training)
A human draws a bounding box on every worm in a selected Scene-1 frame. Box count = worm count.

| Annotation set | Frames | Worm boxes | Mean/frame | Role |
|---|---|---|---|---|
| **Monday manual series** | 722 | 9,212 | 12.8 | Primary training + clear-era series |
| **Tuesday manual series** | 229 | 7,220 | 31.5 | Dense / blurry-era training |
| **Jan–Jul 2021 batch** | 22 | 855 | 38.9 | 2021 coverage top-up (v7) |
| **TOTAL (box-labeled)** | **973** | **17,287** | — | — |

Monday-series breakdown by year: 2021 = 67, 2022 = 288, 2023 = 205, 2024 = 162 frames.

### 2b. Click counts (used only as independent validation — never trained on)
A human clicks the centre of each worm (a point, not a box). These are a *different counting method* from box-drawing, which makes them a fair, independent test set. They are held out of training at the **calendar-day** level.

| Click-count gate | Frames | Purpose |
|---|---|---|
| Clear-window handcount | 39 | Clear-era acceptance test |
| Feb+May-2024 model comparison | 31 | Blurry-era point-match test |
| 2015–16 early-recovery | 27 | Early-era (dense mound) feasibility |
| Blurry 2024 gate | 14 | 2024 blurry recall |
| Early-2021 handcount | 4 | Jan–Aug 2021 undercount test |

### 2c. Annotation reliability (inter-rater, in progress)
A second annotator (Rater B) independently box-counted a 20-frame packet (Jan-2022 low vs Jan-2023 peak). Against Makayla's counts she tracks the frame-to-frame pattern well (Pearson r = 0.74) but marks systematically fewer worms (≈9.5 vs 16.2/frame; ICC 0.38). This interim comparison is **confounded** — Makayla's reference labels were detector-seeded then corrected, while Rater B counted from a blank frame — so Makayla is now re-counting the same frames blind for a clean, like-for-like reliability estimate. A third rater (Rater C) is outstanding.

---

## 3. What was fed to the detector (training data per model)

All models are fine-tuned from the inherited base **`mushroom.pt`** (a `YOLO26x` architecture, ≈55.6 M parameters). Training uses the **box** annotations above; the click-count gates are always held out.

| Model | Base | Training frames (train / val) | Training data |
|---|---|---|---|
| mushroom.pt (v0) | — | inherited | pre-existing weights |
| v2 | mushroom | ≈47 | clear frames (original small set) |
| v3 | mushroom | ≈94 | clear + some blurry (generalist) |
| v4 | mushroom | — | *discarded (worse than v3)* |
| **v5_all** | mushroom | 421 / 87 (≈5,396 boxes) | Monday series, all eras, leakage-safe |
| v5_clear | mushroom | 289 / 58 | Monday series, clear era only |
| v6_blurry | mushroom | 329 / 65 | clear + dense Tuesday (blurry fix) |
| v6_scale n15/n30/n47 | mushroom | 289 + {15,30,47} | learning-curve test (dense) |
| v6_early2015 | v5_all | 14 / 5 | 2015–16 pseudo-boxes *(discarded)* |
| v7_2021 (warm) | v5_all | 99 / 17 | 2021-only small set |
| v7_additive | mushroom | 443 / 87 | v5 set + 22 Jan–Jul 2021 |
| v7_gentle | v5_all | 443 / 87 | v5 set + 22 Jan–Jul 2021, gentle LR |

**Leakage control:** any frame sharing a calendar day with a gate frame is removed from training (the camera records ~8 slots/day, so one shared day would leak near-duplicates). Every reported result below was re-audited to confirm zero train/test overlap.

---

## 4. Detector model versions and results (no geophysical interpretation)

Metrics are **point-matched** recall / precision / F1 (does the detector mark the same worms a human did), which is stricter and more honest than a raw count ratio. "conf" is the detection-confidence threshold.

### Headline lineage
- **v0 → v2/v3:** first fine-tunes lifted clear-era recall from the inherited baseline but stayed weak on blurry footage.
- **v5_all (current production):** trained on the full 722-frame Monday series (leakage-carved to 421). Clear-era point-recall **≈70 %** at conf 0.40 with a near-unbiased count ratio (≈100 %). This is the reliable, deployed model.
- **v6 (blurry / density experiments):** attempts to fix dense and blurry footage.
- **v7 (2021-undercount experiments, this week):** three recipes to fix an early-2021 shortfall.

### Clear era (the trustworthy regime)
- v5_all: point-recall ≈70 % (conf 0.40), count ratio ≈100 %.
- **Density compression caveat:** accuracy declines as frames get crowded — ≈93 % count recovery at 0–8 worms/frame, falling to ≈64 % above 22 worms/frame. The detector cannot separate heavily overlapping worms, so it **compresses the tallest peaks**. This is a detection ceiling, not a threshold artifact (higher-resolution inference and tiled "SAHI" inference were both tested and did not fix it).

### Blurry era (post-Aug-2023 camera soft-focus)
- An earlier "94 % recovery" claim was found to be **invalid / circular**: the Monday blurry *training* labels had themselves been under-counted, so the model learned to under-detect and the blurry gate (built from the same bad labels) falsely passed. Corrected against properly counted dense Tuesday frames, dense-blurry point-recall is only **≈32 %**.
- Adding real dense labels (v6_blurry; learning curve n=15/30/47) did **not** lift it — recall stayed flat ≈22–28 %.

### Early-recovery / dense-mound eras (2015–16, and dense 2021)
- 2015–16 mound scenes: ≈28 % recall (out-of-distribution dense view).
- Early-2021: the detector under-counts ≈2× (it saw no Jan–Aug 2021 training frames). The three v7 retrains this week:
  - *warm-start* fixed 2021 (point-recall 25 → 39 %) but **crashed** the clear and blurry windows;
  - *additive* kept clear but **regressed blurry** and did not fix 2021;
  - *gentle warm-start* (adopted recipe) **held both clear and blurry** at v5 levels and gave a small 2021 gain (24.5 → 30.9 %) — but still could **not** cross ~31 % on the dense 2021 test frames.

### The unifying finding: a density ceiling
Across blurry, 2015–16 mound, and dense 2021, **dense scenes resist every fix** — more labels, pseudo-labels, higher resolution, tiled inference, and all three v7 recipes. Where the scene is clear and low-to-moderate density, the detector is reliable (~70 %+); where it is dense or blurry it under-counts (~25–35 %). The practical consequence: the automated series is trustworthy as a **conservative, shape-preserving relative index** in the clear window, and must be treated as provisional in dense/blurry eras.

---

## 5. The scene selector

Each recording is a slow pan-and-zoom; only a brief window shows the **front-on "Mushroom tower" Scene-1 view** that can be counted. A frame must be selected before it is counted.

- **Manual scene-sorting (settled):** a human picks the Scene-1 frame. Used for the Monday series. This is the reliable method but does not scale to ~20k recordings.
- **AI scene classifier (abandoned for primary use):** a learned Scene-1/not classifier was tested twice and failed its agreement gate (κ = −0.42, then κ = 0.231), so it is not used to auto-sort the archive. (A later held-out Wednesday validation of a related classifier transferred better, κ = 0.709, but scene sorting remains manual/settled.)
- **Detector-based frame-finder (the scalable workaround):** instead of a separate classifier, scan the pan (≈120–600 s), run the detector on sampled frames, and take the richest frame as the implicit Scene-1 pick ("max-over-window"). This is **scene-sort-free** and needs no human. It works in the **clear** window (lands within 120 s of the manual Scene-1 ≈90 % of the time) but **fails on blurry footage** (wrong frame ≈25–61 % of the time), because there the detector itself is unreliable — so frame-finding and counting fail together off the clear window.

---

## 6. What we are doing now: the Tuesday–Friday extension

**Objective.** Move from a Monday-only manual series to all weekdays (Tue–Fri), then ultimately to M–F × 8 slots/day × 2015–2024.

**Method — a scene-sort-free harvest pipeline:**
1. `build_harvest_manifest.py` enumerates the archive for a date range and weekdays, filtering to the **8 canonical 3-hourly slots**.
2. `harvest_series.py` extracts candidate frames (parallel `ffmpeg`), runs the detector across the pan window, takes the **median of the top-3 richest frames** as the count, applies a **÷0.72 recall correction**, and is parallel + resumable.

**Progress:**
- **Tier A** (clear Tue–Fri, 2021-09 … 2023-08, 3,169 recordings) reproduced the independent Monday manual abundance to **≈90 %** on never-touched days — validating the automated approach where the detector is reliable.
- **Full M–F 2021–2023** harvested (6,158 recordings) → `mf_index_weekly_2021_2023.csv` (156 weeks, regime-tagged).

**Current status of the weekday index:** it is a conservative, recall-corrected **relative abundance index** (tracks change, not absolute counts).
- **Trustworthy:** clear window, 2022–2023.
- **Provisional (under-counts):** 2021 (≈2× low — the detector coverage/density issue the v7 work targeted) and blurry post-Aug-2023 / 2024.

**How we finish:**
- Keep the clear-window M–F index as the primary deliverable.
- For 2021 absolute levels: because the shortfall is both a density ceiling *and* an independent frame-finder failure in that era, those years likely need **human Scene-1 picks + counts**, not a further detector tweak.
- Extend the harvest to remaining years only where per-era detector reliability allows; dense/blurry/early eras remain provisional or require human-in-the-loop counting.

---

## 7. One-paragraph summary

A `YOLO26x` detector was fine-tuned on **973 hand-box-labeled frames (17,287 worms)**, validated against **independent human click-counts** held out at the day level. The production model (**v5_all**, trained on 421 clear-safe frames) counts clear, low-to-moderate-density footage at ~70 % point-recall and near-unbiased totals; it systematically under-counts dense and blurry scenes (~25–35 %), a **density ceiling** that resisted every remedy tried (more labels, pseudo-labels, higher resolution, tiled inference, three v7 retrain recipes). Scene selection is manual where it matters and automated (a detector-driven frame-finder) where the detector is reliable; a learned scene classifier was tested and shelved. The active work extends the series from Mondays to all weekdays via a scene-sort-free harvest, which reproduces the manual abundance to ≈90 % in the clear 2021–2023 window and yields a weekly M–F index — trustworthy for 2022–2023, provisional for 2021 and the blurry post-2023 era.

---

## 8. Update (2026-10-06): tiling, the two bottlenecks, and an improved index

**Dense-miss diagnosis.** A within-frame analysis of which worms the detector misses on dense frames found near-zero box overlap between worms — they are small and closely-spaced but *separable*, not occluded. That is the small-object regime tiling targets, so a tiled model was trained.

**Tile-training (v8).** The detector was re-trained on 640 px image tiles (3,162 tiles from the leakage-safe set) and run with SAHI tiled inference. Result: on dense frames it lifted recall only +1–3 points — **the density ceiling held** (now confirmed three independent ways: inference-SAHI, tile-training, and a labelling learning-curve). But v8+SAHI proved the **best clear-window counter** (76 % recall, 73.5 % precision, near-unbiased 103 % count on the clear gate), better than the production v5.

**The two bottlenecks.** Re-counting the harvest frames with the better counter *did not* move the automated index — revealing that the automated series is limited by **frame-finding** (picking the true front-on Scene-1 moment), not counting. On auto-picked frames both v5 and v8 land at ~58 % of manual because the picked frame isn't the true Scene-1; the counter only shines on good frames.

**Improved index.** A validation on frames with known true Scene-1 times showed the fix is the **aggregation**, not a new finder: counting the **max over a 60 s temporal grid** with v8+SAHI reaches **100 % of manual abundance in aggregate** (vs the old v5 index's 81 %), and resolves the 2021 undercount. The full clear window (3,989 recordings) was re-harvested with this recipe. Validated against manual on the same 530 Monday recordings: **100 % overall** (2021 131 %, 2022 93 %, 2023 107 %). The residual is **density-dependent** and irreducible by aggregation — sparse frames are over-counted by max-selection (132 %) while dense peaks are still under-counted at the detection ceiling (79 %), cancelling to 100 %. It is therefore a strong **relative/aggregate** index, not a per-frame absolute count. See `figure_v8_index_vs_manual_2021_2023.png` and `v8_index_weekly_2021_2023.csv`.

**Net:** counting is solved on good frames; the density ceiling is a real (and partly ground-truth-limited) floor for dense scenes; and frame-finding + aggregation — not the counter — is the lever that moved the automated clear-window index from ~58 % to ~100 % of manual.

**Dense-peak limitation (logged).** The automated index under-reads the population peaks (dense frames recover ~79 % of manual; the densest ~50 %), and this is irreducible: detector improvements plateau at +1–3 points, a held-out-validated count calibration does **not** lift the peaks (the automated count is many-to-one — true manual counts of 1–43 map to the same automated value), and human counts themselves disagree ~2× at high density. Timing and direction-of-change are preserved, so trend/timing correlations with geophysical parameters are robust; **peak-magnitude** comparisons should be anchored to the non-saturating manual Monday series, not the automated index. Full write-up: `results_note_density_ceiling_2026_10_06.md`.
