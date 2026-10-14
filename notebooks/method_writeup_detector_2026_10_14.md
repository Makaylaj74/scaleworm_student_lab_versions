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

---

## 6. Key findings that shaped the method

**6.1 The circular "blurry 94%" correction.** An early blurry result looked excellent until it was found the blurry *training labels* were themselves under-counted; the model learned to under-detect and the same-source gate falsely passed. Re-scored against properly counted dense Tuesday frames, real dense-blurry recall is ~32%. *Lesson baked into the method: validate against an independent counting method, never the training signal.*

**6.2 The density ceiling (the unifying result).** Dense scenes resist every remedy tried — more labels, pseudo-labels, higher-resolution inference, inference-SAHI, tile-training (v8), and all three v7 recipes. Clear/low-density → reliable (~70%+ point-recall, ~100% count); dense/blurry → ~25–35%. A within-frame diagnosis showed the missed worms are small and closely-spaced but *separable* (not occluded) — the small-object regime tiling targets — yet tiling lifted dense recall only +1–3 points. The ceiling is partly ground-truth-limited (humans disagree ~2× at high density). **Consequence:** the automated series is a conservative, shape-preserving **relative index**; peak *magnitudes* are anchored to the non-saturating manual Monday series, not the automated count.

**6.3 Frame-finding, not counting, is the archive bottleneck.** A better counter (v8+SAHI) did not move the automated archive index: on auto-picked frames both v5 and v8 land at ~58% of manual because the *picked frame* isn't the true Scene-1. The fix was **aggregation** — counting the **max over a 60 s temporal grid** — which lifted the clear-window index from ~58% → ~100% of manual abundance and resolved the 2021 undercount. The full clear window (3,989 recordings) was re-harvested with this recipe.

**6.4 Provisional eras.** Trustworthy: clear window, 2022–2023. Provisional (under-counts): 2021 (~2× low — density + an independent frame-finder failure in that era) and blurry post-Aug-2023 / 2024. Dense early-recovery 2015–2016 is out-of-distribution and awaits an advisor decision on the count metric (worm count vs. percent cover).

---

## 7. Detector-scale comparison (2026-10-14, in progress)

Because the production model is the largest scale (YOLO26-**x**, ~59 M params), a sweep tested whether a **smaller/faster** scale holds detection quality (relevant to the ~13k-record harvest cost): yolo26 **n / s / m / l / x**, each COCO-pretrained, fine-tuned on the same 421/87 clear-safe set with identical hyperparameters (1280 px, 100 epochs, seed 0).

- **Validation mAP is ~flat across scale** (mAP50-95 0.411–0.440; 10 M `s` ≈ 59 M `x`).
- **Under the point-match gate, the smaller `s` *beats* `x`** by ~6 recall points on both clear (75.5 vs 69.5%) and dense (64.6 vs 58.5%) — consistent with the 59 M model *overfitting* the 421-image training set — and `s`-plain nearly matches the full v8+SAHI pipeline without tiling.
- **Speed on the H100 is scale-insensitive** (~15–16 img/s all scales); the detector is not the harvest throughput bottleneck.
- **Status:** a seed replication (s and x at 3 seeds each) is running to confirm the `s` > `x` gap exceeds seed variance before any production-scale change is recommended. *[update verdict here when the replication completes]*

---

## 8. One-paragraph abstract of the method

A YOLO26-x object detector was fine-tuned on ~973 hand-box-labeled Scene-1 frames (17,287 worms) and validated against independent human click-counts held out at the calendar-day level. The production model counts clear, low-to-moderate-density footage at ~70% point-recall with near-unbiased totals, but systematically under-counts dense and blurry scenes (~25–35%) — a density ceiling that resisted every remedy (more labels, pseudo-labels, higher-resolution and tiled/SAHI inference, and multiple retrain recipes) and is partly limited by ~2× human disagreement at high density. Scene selection is manual where it anchors the series and automated (a detector-driven frame-finder with max-over-grid aggregation) for the archive, reproducing manual abundance to ~100% in the clear 2021–2023 window. The resulting weekday index is a conservative relative abundance measure — trustworthy in the clear window, provisional in dense/blurry eras — with peak magnitudes anchored to the manual series.
