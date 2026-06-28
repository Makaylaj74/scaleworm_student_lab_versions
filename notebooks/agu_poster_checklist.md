*AI-generated draft (Claude, Anthropic) — for review. All parameters and figures are derived from version-controlled scripts and data.*

# AGU Poster Preparation Checklist — Scaleworm Population Time Series

**Presentation type:** Poster  
**Dataset:** OOI CAMHD, Mushroom vent (RS03ASHS-PN03B-06-CAMHDA301), October 2024 → multi-year  
**Last updated:** 2026-06-28

---

## 0. Known sources of error and model limitations

A full accounting of where this pipeline can be wrong and why. These must be understood before any results are presented publicly. Each item is marked with its **likely impact** (Low / Medium / High) on reported population counts.

### 0.1 Scene and frame selection errors

**Variable zoom timing** *(Medium)*
The CAMHD observation profile nominally places the Mushroom vent close-up at ~305 s, but this varies by video. The scan window used for population counting (185–325 s) was calibrated on one reference video (CAMHDA301-20241001T091500.mp4, t=190 s). Videos where the zoom falls outside this window will report 0 detections, which are then dropped from the time series. These look identical to "camera not zoomed" frames and cannot be distinguished without manual review.

**Wrong camera angle in frame pool** *(Medium)*
Frames extracted during the detection phase (Section 3) cover 305–320 s at 1 fps. For videos where the camera is not yet zoomed into the vent at this timestamp, the detector runs on a wide-angle scene and produces either 0 detections or spurious detections on non-vent structures. Both outcomes contaminate the training data and the time series.

**Two distinct camera styles in training data** *(Medium)*
Training data contains at least two camera angles: a front-on close-up (T061500-style) and a wider horizontal view (T091500-style). The proportion of each in the training set is unknown. The model may be systematically better at detecting worms in the dominant style, producing angle-dependent count bias across the time series.

**1 fps frame rate** *(Low–Medium)*
Frames are extracted at 1 fps from the 15-second Scene 1 window (15 frames/video). Worms that move significantly between seconds, or that are only visible in sub-second intervals, may be missed entirely within a given video.

### 0.2 Detection errors

**No temporal tracker** *(Medium)*
YOLO runs independently on each frame with no temporal memory. The same worm can be detected in one frame and missed in the next due to minor motion, occlusion, or lighting changes. Counts are therefore frame-level snapshots, not tracked individuals. The reported "max in window" count mitigates but does not eliminate this variance.

**Low confidence threshold during training (0.1)** *(Medium)*
Detection is run at conf=0.1 to maximize recall for human verification. At this threshold a large fraction of detections are false positives (bacterial mat, tube structures, lighting artifacts). These are filtered by human labeling, but any labeling errors pass directly into training data and propagate to the retrained model.

**Max-in-window is not a census** *(Medium)*
The population count for each video is the maximum detection count across all scanned frames — not a count of unique individuals. If a dense cluster of worms is detected in one frame and a different subset is detected in another, neither frame individually captures the full count. The max is a lower bound on visible worms, not a true abundance estimate.

**Bounding box size filter (MAX_BOX_SIZE = 300 px)** *(Low)*
Any box wider or taller than 300 pixels is rejected as a non-worm artifact. Real worms are 20–100 px. This filter is appropriate for single worms but could reject an aggregate box encompassing a very dense cluster — an unlikely edge case but worth monitoring if large clusters appear in the data.

**Human labeling error** *(Low–Medium)*
The verification step relies on a single reviewer (no inter-annotator agreement check). Ambiguous crops (partial worm, blurry frame, edge of image) are particularly susceptible. Skipped detections are excluded from training, not reviewed again — any systematic skipping bias affects the final training set.

### 0.3 Training and model errors

**No train/validation split** *(Medium)*
The exported YOLO dataset uses the same frames for both `train` and `val` in `dataset.yaml`. This means mAP50 reported during training is measured on training data — it is not a true holdout validation score. The model's actual generalization performance is unknown until it is tested on a genuinely unseen set of videos.

**Single month, single location training** *(High — for multi-month deployment)*
All annotation rounds so far use October 2024 data from one vent (Mushroom, ASHES, Axial Seamount). The model has not seen other months, seasons, or environmental conditions. Lighting, turbidity, worm density, and camera behavior can all vary. Applying this model to other months without any cross-month validation is an untested generalization.

**CPU-only inference on the Hub** *(Low — functional, not quality)*
The NVIDIA driver version (12020 / CUDA 12.2) is too old for the installed PyTorch build. All inference runs on CPU. This does not affect output quality but significantly increases runtime and therefore how frequently re-runs can be attempted.

### 0.4 Counting pipeline errors

**Zero-detection dropout** *(Medium)*
Videos with max detection count = 0 are excluded from the time series plot on the assumption the camera was not zoomed. Some of these may be genuine model failures (wrong angle, turbidity, unusual worm density). The true absence rate is unknown without reviewing a random sample of excluded videos.

**Scan window not validated across all months** *(High — for multi-month deployment)*
The 185–325 s window was chosen for October 2024. CAMHD observation profiles may differ in other years or after instrument maintenance. If the zoom falls at a different timestamp in another month, the entire time series for that period will be zeros from window misses — indistinguishable from true population absence.

---

## 1. Model readiness — proof of concept (October 2024 first)

**Do not proceed to multi-month deployment until all of these are met.**

- [ ] mAP50 ≥ 0.90 on October 2024 — from YOLO training logs after the final annotation round
- [ ] Create a genuine holdout set: reserve ~10% of October 2024 frames before the next annotation round and evaluate mAP50 on those only (addresses §0.3 — current val = train)
- [ ] False positive rate < 5% at **production** confidence threshold (0.3–0.4), not the 0.1 training threshold
- [ ] Spot-check T061500 vs T091500 camera style videos — confirm counts are not systematically different between the two angles (addresses §0.1)
- [ ] Zero-detection videos: manually review a random sample of ~20 to confirm they are truly "camera not zoomed" and not model failures (addresses §0.4)
- [ ] Round-over-round comparison chart (Section 0 of `22_verify_detections.ipynb`) shows convergence — new round produces meaningfully fewer corrections than the last

**Only after the above:** expand to other months. For each new month:
- [ ] Run model on a 20–30 video sample from the new month; spot-check detections
- [ ] Confirm scan window (185–325 s) still captures the vent zoom (spot-check 5 videos manually)
- [ ] If performance looks degraded, do one targeted annotation round on that month before full deployment
- [ ] Add brightness/contrast augmentation to the retrain cell before multi-month runs (makes the model more robust to lighting/turbidity variation without extra annotation — defer until proof of concept is confirmed)

## 2. Time series data readiness

- [ ] Model run on all target months (goal: "a couple of years" of consistent data)
- [ ] Zero-detection videos audited — random sample checked to confirm camera not zoomed vs. model failure
- [ ] Counts per day reviewed for plausibility — no anomalous spikes or suspicious flat periods without an explanation
- [ ] Scan window (185–325 s) confirmed adequate for each new month (spot-check a few videos)
- [ ] All results files committed to git and backed up

## 3. Uncertainty and defensibility

- [ ] Model precision and recall reported at production confidence threshold alongside population counts
- [ ] Counts described as **maximum detections in a scan window** — not an exact census; this distinction must appear in methods or poster text
- [ ] Known limitations explicitly disclosed (in methods panel or supplemental):
  - Mixed camera styles in training data (T061500 vs T091500)
  - Variable zoom timing — scan window is an empirical estimate
  - Zero-detection dropout — some excluded videos may be model failures, not true absence
- [ ] If error bars are shown, state what they represent (model FP rate, inter-frame variance, etc.)

## 4. Figure requirements

- [ ] **Time series figure:** static matplotlib PNG, 300+ DPI — NOT the Plotly version from `24_count_timeseries.ipynb`
  - x-axis: date/time with proper UTC label
  - y-axis: "Scaleworm count (max in 185–325 s window)" or similar honest label
  - Colorblind-safe palette (Okabe-Ito for any multi-series; viridis/cmocean if continuous)
  - Caption includes: model version, confidence threshold, scan window, brief limitation note
- [ ] **Example detections panel:** 2–3 example crops showing true positives at various confidence levels; 1–2 false positives; clearly labeled
- [ ] All figures pass the AI-generated text disclosure check (caption text by human or labeled AI-draft)

## 5. Methods panel

- [ ] Workflow described: CAMHD video → Scene 1 frame extraction → YOLO detection → human-in-the-loop verification → retrain → population count
- [ ] Model architecture named (YOLOv11m, ultralytics)
- [ ] Training data described: X verified detections from October 2024, Y annotation rounds
- [ ] Production confidence threshold stated
- [ ] Scene selection method described (scan window 185–325 s, max-in-window count)

## 6. Attribution and acknowledgments

- [ ] OOI data citation (instrument: RS03ASHS-PN03B-06-CAMHDA301)
- [ ] Ultralytics / YOLO citation
- [ ] Any collaborators or prior work on the Mushroom vent / scaleworm counts acknowledged
- [ ] AI tool disclosure if any AI-generated prose or figures appear on the poster

## 7. Poster logistics

- [ ] Poster template and dimensions confirmed with conference/advisor
- [ ] Print deadline confirmed
- [ ] Final figure files exported at poster resolution before print deadline
- [ ] Backup copy of poster PDF committed to git or stored in Dropbox

---

## Open questions to resolve before AGU

1. **Multi-month generalization**: What months will be in the final time series? Has the model been validated on at least one non-October month?
2. **Production confidence threshold**: What threshold will be used for the final time series counts? Needs to be decided and justified (not 0.1).
3. **Scene window adequacy**: Does the 185–325 s window cover the vent zoom in all target months, or does it need to be widened for some months/years?
4. **Baseline comparison**: Is there any prior scaleworm count data (e.g., VIAME annotations from 2023) to cross-validate against?
