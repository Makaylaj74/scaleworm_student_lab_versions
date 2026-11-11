*AI-generated draft (Claude, Anthropic) — for review. A scaffold for the thesis Method section; all model versions, counts, and results are derived from version-controlled scripts, datasets, and gate CSVs in `scaleworm-student-lab` and the 2026-10-06 pipeline report. Re-verify every number against disk before it enters the thesis. Edit freely — the disclosure may be removed once substantially rewritten by the author.*

# Method — automated scale-worm counting from the ASHES seafloor camera

**Scope:** scale-worm abundance at the Mushroom hydrothermal vent (OOI RS03ASHS, `CAMHDA301` camera), 2015–2024. This document is a Method-section outline: each section states *what was done* and *why*, with the decision points that shaped the pipeline.

---

## 1. Objective and overall design

The target series is **Monday–Friday × 8 recordings/day × 2015–2024** — on the order of **13,000–20,000 countable frames**, roughly 25× the ~1,000 frames that can be hand-counted. Counting every frame by hand is therefore impossible, and the pipeline is built around an automated object detector:

1. Hand-**label** a few hundred representative frames per image regime.
2. **Fine-tune** an object detector on those labels.
3. Let the detector **count** the remaining tens of thousands of frames.

The detector is the critical path; the rest of the method exists to make it reliable across a camera whose image quality changes substantially over the decade (clear 2021–2022, soft-focus/blurry after Aug 2023, dense-mound early-recovery 2015–2016).

> *Outline note:* frame this section as "why automation is necessary" → magnitude of the task vs. hand-counting capacity.

---

## 2. Data

### 2.1 Source footage and frame selection
Each recording is a slow pan-and-zoom; only a brief window shows the front-on **"Mushroom tower" Scene-1 view** that can be counted. A Scene-1 frame must be selected before counting. Selection methods (detailed in §5) are: manual scene-sorting (reliable, used for the training/validation series) and a detector-driven frame-finder (scalable, used for the archive harvest).

### 2.2 Manual annotations — two methods kept strictly separate
| Kind | Method | Role | Volume |
|---|---|---|---|
| **Box labels** | human draws a box on each worm | **training** | 722 Monday + 229 Tuesday + 22 Jan–Jul 2021 ≈ **973 frames / 17,287 worms** |
| **Click counts** | human clicks each worm center | **independent validation only — never trained on** | 5 gate sets (clear-window 39, 2024 comparison 31, 2015–16 early 27, blurry-2024 14, early-2021 4) |

Click counts are a *different counting method* from box-drawing, which makes them a fair, independent test of the detector rather than a re-scoring of its own training signal.

### 2.3 Leakage control
Any frame sharing a **calendar day** with a validation frame is removed from training (the camera records ~8 slots/day, so a shared day would leak near-duplicates). Every reported result was re-audited for zero train/test day overlap.

### 2.4 Annotation reliability (inter-rater, in progress)
A second rater (Rater B) independently box-counted a 20-frame packet: tracks the pattern (Pearson r = 0.74) but counts systematically fewer worms (≈9.5 vs 16.2/frame; ICC 0.38). That interim comparison is **confounded** (reference labels were detector-seeded then corrected, vs. counted-from-blank), so a clean blind 3-way round (Makayla / Rater B / Rater C, all from-scratch on the same fresh frames) is underway. Relevance to the method: human–human disagreement is ~2× at high density, which sets a ground-truth ceiling the detector cannot beat on dense frames (see §6).

### 2.5 Written counting rule and blind recounts (Nov 2026)
Re-running fresh pre-labels against finished counts showed the box-correction criterion had drifted between sessions. In the most extreme case, on the same Sep–Dec 2021 Mondays, a pass that boxed empty dark openings read **2.35×** a pass that did not. A **written counting rule** (`notebooks/worm_counting_rule.md`, confirmed 2026-11-11) now governs every manual count. It boxes each separable worm body, and a dark opening counts only if part of a worm body is visible inside it. All 2021 counts were redone **blind** (fresh pre-labels, annotation columns blanked) under this rule:

| Set | Frames | Result (bootstrap n = 10 000, seed 0, percentile CIs) |
|---|---|---|
| Sep–Dec 2021 recount vs Sep 2026 originals | 48 | originals were low by 1.24× [1.16, 1.33] (one earlier session) |
| 2022–2024 stratified recount vs originals | 24 | 0.95× [0.77, 1.19]: originals **validated**, kept |
| Jan–Aug 2021 recount (rule) vs earlier broad-criterion pass | 43 | rule / broad = 0.70× [0.65, 0.76]: broad pass **replaced** |

Under the rule, Jan–Aug 2021 averages **19.8 worms/frame [17.0, 22.9]** and Sep–Dec 2021 averages **13.2 [12.0, 14.4]**. These intervals come from a cluster bootstrap over Mondays. There is therefore **no early-2021 low**: the low early-2021 values in the automated series were a pipeline artifact (§5, §6.3). *Caveat:* the two 2021 blocks were counted in separate sittings, so between-session drift cannot be fully excluded; a shuffled blind Jul-vs-Oct/Nov check would settle it.

---

## 3. Detector

### 3.1 Architecture
All models are fine-tuned from the inherited base **`mushroom.pt`**, a **YOLO26-x** architecture (~59 M parameters), via the Ultralytics framework. *(Note for the thesis: earlier draft text naming "YOLOv11m" is incorrect — every checkpoint, mushroom.pt included, embeds `yolo26x.yaml`. Correct to YOLO26-x.)*

### 3.2 Inference modes (and where SAHI fits)
- **Plain full-frame inference** (default): `model(image)`.
- **SAHI (Slicing Aided Hyper Inference):** an *external, model-agnostic inference-time technique* (the `sahi` library) that slices the frame into overlapping tiles, detects per tile, and merges — so small/dense worms occupy more pixels per tile. **SAHI is not part of the detector architecture;** it is an inference wrapper evaluated separately as a density-recovery step (§6). The original model used plain inference; SAHI was added later.

### 3.3 Version lineage (why each model exists)
| Model | Base | Train (frames) | Purpose / outcome |
|---|---|---|---|
| mushroom.pt (v0) | — | inherited | pre-existing YOLO26-x weights |
| v2 | mushroom | ≈47 | clear specialist (first fine-tune) |
| v3 | mushroom | ≈94 | generalist (clear + some blurry) |
| v4 | mushroom | — | *discarded — worse than v3* |
| **v5_all** | mushroom | 421 / 87 | **production**: full Monday series, all eras, leakage-safe |
| v5_clear | mushroom | 289 / 58 | clear-era only |
| v6_blurry, v6_scale n15/30/47 | mushroom | 329 + | blurry-fix + labeling learning curve (dense) |
| v6_early2015 | v5_all | 14 / 5 | 2015–16 pseudo-boxes *(discarded)* |
| v7_2021 (warm / additive / gentle) | v5_all / mushroom | 99–443 | three recipes for the early-2021 undercount |
| **v8_tiled** | mushroom | 3,162 tiles | **tile-trained**; best clear counter with SAHI |
| scale sweep n/s/m/l/x | COCO-pretrained | 421 / 87 | architecture-scale comparison (2026-10-14, §7) |

---

## 4. Validation metric

Detection quality is scored by **point-match** recall / precision (greedy center-in-box): does the detector mark the *same worms* a human did. This is stricter and more honest than a raw count ratio, under which misses and false positives cancel and can make a poor model look accurate (a mistake that produced an early invalid "94%" blurry result — see §6).

---

## 5. Scene selection

- **Manual scene-sort (settled):** a human picks the Scene-1 frame; used for the Monday training/validation series. Reliable but does not scale to ~20k recordings.
- **Learned scene classifier (abandoned):** failed its agreement gate twice (κ = −0.42, then 0.231); not used to auto-sort the archive.
- **Detector-driven frame-finder (scalable):** scan the pan (~120–600 s), run the detector on sampled frames, take the richest frame as the implicit Scene-1 pick. Works in the clear window (~90% within 120 s of manual) but fails on blurry footage (~25–61% wrong frame), because there the detector itself is unreliable — frame-finding and counting fail together off the clear window.
- **Era-fixed Scene-1 time (density head; Nov 2026, in validation):** the camera reaches Scene-1 at a **fixed point in the pan within each camera-schedule era**. Per the Monday sort, that point is 420 s before 2021-09-01, 540 s until 2022-04-25 and 210 s after; Jan–May 2023 alternates between 210 s and 420 s, so the higher of the two readings is taken. The density head (§ `methods_density_head.md`) reads a single frame at that time, and recordings reading below **8.5 worms** are treated as off-scene. That cutoff was chosen on Monday sort labels only. Validation:
  - *Monday hand counts under the rule* (91 frames, 2021):
    - The new reading gives r = 0.80 [0.69, 0.87]; the detector-picked frame gives 0.40 [0.17, 0.57]. Reading the human-selected frame directly gives 0.81, so the new method matches that ceiling.
    - The ratio to hand counts is **0.82× in Jan–Aug and 0.86× in Sep–Dec**. The old frames gave 0.72× and 1.29×, so the new method removes the period-dependent bias.
    - These era times were derived from the same Mondays, so this check is not independent.
  - *Independent Tuesday check* (rules fixed from Mondays):
    - The era times match the Tuesday scene sort for 94% of 519 Scene-1 slots.
    - The off-scene cutoff transfers with sensitivity 0.90 and specificity 0.83 (686 slots).
    - Against 229 Tuesday box counts, which are dense (mean 31.5/frame), r improves from 0.56 to **0.64** (paired Δr +0.08 [0.01, 0.15]).
    - Day-level r improves from 0.42 to 0.58 (49 days; Δr +0.16 [−0.07, 0.40]).
    - The count ratio there is 0.67×, which is the density ceiling on dense 2022 Tuesdays rather than frame choice.
  - The Tuesday sort begins Sep 2021, so Jan–Aug 2021 is validated on Mondays only.
  - **Full index built (Nov 2026; `scripts/build_era_time_index_2021_2023.py`, 5 354 clear Mon–Fri recordings → 135 weeks).**
    - *Weekly agreement with hand counts* (bootstrap over weeks; `scripts/plot_era_index_vs_hand_2021_2023.py`):
      - All weeks: 1.02× [0.96, 1.08], r = 0.71 [0.57, 0.81], n = 108. The old index gave 0.89× and r = 0.42.
      - Sep 2021–Aug 2023: 1.08× [1.03, 1.14], r = 0.78 [0.66, 0.87], n = 87. The old index gave r = 0.61.
      - Jan–Aug 2021: 0.81× [0.72, 0.92], r = 0.52 [0.08, 0.78], n = 21. The old index gave 0.52×.
      - Late-2022 peak: 0.93×; the old index gave 0.76×.
    - *Residual biases:*
      - Early 2021 still reads about 20% low.
      - Sparse quarters read high (2022 Q1 1.22×), because the off-scene cutoff also drops genuinely sparse Scene-1 frames.
      - Jan–May 2023, where the higher of two candidate times is kept, reads 1.27× against 1.03× for the rest of Sep 2021–Aug 2023.
    - *Status:* the index tracks timing and relative change well, but it is not an absolute count.
  - *Provenance:* `scripts/era_finder_*.py` (density reads) and `scripts/era_finder_score.py` (every number above). Outputs are in `validation/era_time_finder/`.

---

## 6. Key findings that shaped the method

**6.1 The circular "blurry 94%" correction.** An early blurry result looked excellent until it was found the blurry *training labels* were themselves under-counted; the model learned to under-detect and the same-source gate falsely passed. Re-scored against properly counted dense Tuesday frames, real dense-blurry recall is ~32%. *Lesson baked into the method: validate against an independent counting method, never the training signal.*

**6.2 The density ceiling (the unifying result).** Dense scenes resist every remedy tried — more labels, pseudo-labels, higher-resolution inference, inference-SAHI, tile-training (v8), and all three v7 recipes. Clear/low-density → reliable (~70%+ point-recall, ~100% count); dense/blurry → ~25–35%. A within-frame diagnosis showed the missed worms are small and closely-spaced but *separable* (not occluded) — the small-object regime tiling targets — yet tiling lifted dense recall only +1–3 points. The ceiling is partly ground-truth-limited (humans disagree ~2× at high density). **Consequence:** the automated series is a conservative, shape-preserving **relative index**; peak *magnitudes* are anchored to the non-saturating manual Monday series, not the automated count.

**6.3 Frame-finding, not counting, is the archive bottleneck.** A better counter (v8+SAHI) did not move the automated archive index: on auto-picked frames both v5 and v8 land at ~58% of manual because the *picked frame* isn't the true Scene-1. The fix was **aggregation** — counting the **max over a 60 s temporal grid** — which lifted the clear-window index from ~58% → ~100% of manual abundance and resolved the 2021 undercount. The full clear window (3,989 recordings) was re-harvested with this recipe.
*Update (Nov 2026):* the "~100%" and "resolved" claims were judged against inadequate 2021 truth. The comparison rested on 4 dense clicked frames and on broad-criterion counts that were later found inflated. Against the rule-based 2021 recount (§2.5), the density-head index built on these harvest frames still reads Jan–Aug 2021 at **0.38–0.51×** of hand counts by quarter. From Sep 2021 to Aug 2023 it matches hand counts: 1.01× [0.94, 1.08], weekly r = 0.61 [0.41, 0.77], n = 87 weeks, though the dense late-2022 peak reads 0.76×. Diagnosis: the counter is fine on correct frames (0.83–0.86× in both periods). The failure lies in selecting recordings and frames: in Jan–Aug 2021, 37–66% of recordings per month contain no Scene-1, and the max-over-grid pick misses the brief Scene-1 window. The era-fixed Scene-1 time (§5) is the proposed fix.

**6.4 Provisional eras.** Trustworthy automated index: Sep 2021–Aug 2023 (validated against hand counts, §6.3). Manual counts under the rule cover all of 2021–Aug 2023. Provisional automated index: Jan–Aug 2021 (~0.4–0.5× low from recording/frame selection, pending the era-fixed rebuild) and blurry post-Aug-2023 / 2024. 2017–2020 cannot extend the series. A blind check of Rater B's sort found almost no Scene-1 in 2017–2018. For 2019–2020, a view check (12 blinded accepts from the Sep 2026 sort against 2021 references; `scripts/build_view_check_2019_2020.py`) judged 0/12 to be the same view, so the 2021–2023 Scene-1 view is not available before 2021. Dense early-recovery 2015–2016 is out-of-distribution and awaits an advisor decision on the count metric (worm count vs. percent cover).

---

## 7. Detector-scale comparison (2026-10-14, in progress)

Because the production model is the largest scale (YOLO26-**x**, ~59 M params), a sweep tested whether a **smaller/faster** scale holds detection quality (relevant to the ~13k-record harvest cost): yolo26 **n / s / m / l / x**, each COCO-pretrained, fine-tuned on the same 421/87 clear-safe set with identical hyperparameters (1280 px, 100 epochs, seed 0).

- **Validation mAP is ~flat across scale** (mAP50-95 0.411–0.440; 10 M `s` ≈ 59 M `x`).
- **Detection quality is flat across scale under the point-match gate, confirmed by seed replication.** A single-seed run initially suggested `s` beat `x` by ~6 recall points, but replicating each scale at 3 seeds showed this was seed variance: clear39 `s` 71.0 ± 3.6% vs `x` 70.0 ± 2.5% (gap +1.0%, combined sd 6.0%); dense_clear `s` 60.2 ± 3.1% vs `x` 58.9 ± 0.6% (gap +1.3%, combined sd 3.7%). Both gaps are **within noise** — the 10 M `s` *matches* (does not beat) the 59 M `x`. (`x` is notably more seed-stable on dense, ±0.6%.)
- **Speed on the H100 is scale-insensitive** (~15–16 img/s all scales); the detector is not the harvest throughput bottleneck.
- **Conclusion:** scale is a wash on current hardware — no quality gain from the larger model, no throughput gain from the smaller. `s` is a defensible efficiency choice (1/6 the parameters, equal quality) if the pipeline ever becomes compute-bound; otherwise the production `x` is equally valid. *Methodological note: the single-seed result was overturned by replication — reinforcing that scale comparisons require multiple seeds.*

---

## 8. One-paragraph abstract of the method

A YOLO26-x object detector was fine-tuned on ~973 hand-box-labeled Scene-1 frames (17,287 worms) and validated against independent human click-counts held out at the calendar-day level. The production model counts clear, low-to-moderate-density footage at ~70% point-recall with near-unbiased totals, but systematically under-counts dense and blurry scenes (~25–35%) — a density ceiling that resisted every remedy (more labels, pseudo-labels, higher-resolution and tiled/SAHI inference, and multiple retrain recipes) and is partly limited by ~2× human disagreement at high density. Scene selection is manual where it anchors the series and automated (a detector-driven frame-finder with max-over-grid aggregation) for the archive, reproducing manual abundance (1.01×) from Sep 2021 to Aug 2023. All manual counts follow a written counting rule, and the 2021 counts were redone blind under it. That recount showed early 2021 was not low (19.8 vs 13.2 worms/frame in late 2021). It also showed the archive frame-finder still under-reads Jan–Aug 2021, a recording/frame-selection failure rather than a counting one. Reading the density head at each camera-schedule era's fixed Scene-1 time removes the period-dependent bias on Mondays and modestly improves agreement on independent Tuesdays. The resulting weekday index is a conservative relative abundance measure: trustworthy from Sep 2021 to Aug 2023, provisional in early 2021 and in dense/blurry eras, with peak magnitudes anchored to the manual series.
