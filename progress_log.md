# Scaleworm Detection — Progress Log

Makayla Joseph · Advisor: Dax Soule · Hub: `/home/jovyan/scaleworm-student-lab`

Auto-pushed to `origin/main` (https://github.com/Makaylaj74/scaleworm-makayla) after each update.

---

## Model Lineage

| Version | File | Base | Training data | mAP50 |
|---|---|---|---|---|
| v0 — Mushroom Model | `mushroom.pt` | unknown | prior lab work | — |
| v1 | `runs/detect/verification_session/runs/scaleworm_retrained_v1/weights/best.pt` | mushroom.pt | round 1 corrected (1,156 imgs / 4,823 boxes) | **0.790** |
| v2 | `verification_session_r2/runs/scaleworm_retrained_v2/weights/best.pt` | v1 | round 2 (TBD) | — |

---

## Round 1 — Complete ✓

**Period:** 2024-10-01 → 2024-10-31  
**Completed:** ~2026-06-22  
**Notebook:** `notebooks/22_verify_detections.ipynb`  
**Model used:** `mushroom.pt` (v0)  
**Working dir:** `verification_session/`

### Detection run
- Videos processed: 247 (standard 3-hr cadence, Scene 1 @ 305–320 s, 1 FPS)
- Total candidate detections: 3,874

### Verification results
| Label | Count | % |
|---|---|---|
| ✓ Scale worm | 3,610 | 93.2% |
| ✗ Not a worm | 88 | 2.3% |
| ⟳ Skipped | 176 | 4.5% |
| False-positive rate | — | 2.4% |

### Hub annotation & retraining
- Exported YOLO dataset: 1,153 unique frames, `verification_session/export/`
- Corrected on Ultralytics Hub → downloaded as `round_2.ndjson`
- Corrected dataset: 1,156 images, 4,823 bounding boxes
- Retrained on corrected annotations: 20 epochs, `imgsz=1280`, 4 workers
- Training converged: mAP50 epoch 1 = 0.659 → epoch 20 = **0.790**; mAP50-95 = 0.679
- Best weights saved: `runs/detect/verification_session/runs/scaleworm_retrained_v1/weights/best.pt`

---

## Notebooks Built (2026-06-28)

### `23_pick_timestamp.ipynb`
Interactive tool for picking a reference video timestamp for population counting.
- Cell 1: pre-extracts frames every 5s from a reference video into `timestamp_cache/`
- Cell 2: interactive slider to scrub through the video
- Cell 3 (new): **scene verification** — shows 4 random training frames alongside the chosen timestamp frame for visual confirmation the scenes match

**Finding:** Training data contains frames from multiple camera angles (different time slots show different views of the vent). The T061500-style videos show a closer front-on view; T091500-style shows a wider horizontal profile. Both are in the training set and the model handles both.

**Reference timestamp chosen:** t = 190 s in `CAMHDA301-20241001T091500.mp4`

---

### `24_count_timeseries.ipynb`
Population count time series. **Originally** a scan-window approach (185–325 s, every 5 s,
max count per video). **Rewired 2026-08-26** to single-frame-per-scene — see the
"Scene-1 sort → single-frame counting" section below. The scan-window description is
retained here for history only.

---

## Round 2 — In progress 🔄

**Started:** 2026-06-28  
**Notebook:** `notebooks/22_verify_detections.ipynb`  
**Model used:** v1 (`scaleworm_retrained_v1/weights/best.pt`)  
**Working dir:** `verification_session_r2/`

### Status
- [ ] Frame extraction (Cell 3)
- [ ] Detection run (Cell 4)
- [ ] Crop generation (Cell 5)
- [ ] Verification (Cell 6)
- [ ] Export YOLO dataset (Cell 8)
- [ ] Hub annotation corrections
- [ ] Retrain → v2

### Notes
- Same Oct 2024 date range as round 1; retrained model expected to produce fewer false positives
- Run name for retrain: `scaleworm_retrained_v2`

---

## Scene-1 sort → single-frame counting (2026-08-26)

*AI-generated draft (Claude, Anthropic) — for review. All numbers are produced by version-controlled scripts/notebooks and the committed `sort_log.csv`.*

### Scene-1 sort complete
The full 2023–2024 Scene-1 sort finished: **835/835 recordings**, **367 scene1 / 468
not_scene1** (`scene_sorting/full_2023_2024/sort_log.csv`, committed `d30260a`). Log is
clean — 835 unique stems, no duplicates, every scene1 row carries `scene1_time_s`
(range 180–810 s), no not_scene1 row does.

**Monthly scene1 coverage surfaced a finding:** the front-on Mushroom view collapses in
autumn 2024 — **Sep/Oct/Nov 2024 have zero scene1 recordings** in the weekly-Monday sample
(Dec 2024 = 4). Dense months include Feb 2024 (31), May 2023 (35), Jan 2023 (31), Jul 2024 (28).
This also flags a tension with Round-1 detector verification, which used a *fixed* 305–320 s
window on Oct-2024 footage: the sort suggests that window may have been scoring non-front-on
scenes. Worth a closer look before trusting Round-1 counts.

### `24_count_timeseries.ipynb` rewired to single-frame-per-scene
Now reads `sort_log.csv`, keeps `decision==scene1`, extracts **one full-resolution frame per
recording at its `scene1_time_s`**, and counts `scale_worm` boxes — one recording, one frame,
one count. This uses Makayla's own annotation to place the frame instead of a borrowed window,
removing the notebook's former limitations #2 (dropping zeros on assumption) and #3 (fixed
window) and the upward bias of max-over-window. Zero-detection scene1 frames are now **flagged
as model-miss candidates, not dropped**. Shared, unit-tested logic lives in
`scripts/count_frames.py` (`tests/test_count_frames.py`).

### `26_compare_v0_v1.ipynb` — empirical model choice
Both models run on the **identical** Feb-2024 scene1 frames (n=31, **leakage-free for v1** —
v1 was trained only on Oct-2024 frames). Compares model-vs-model and model-vs-hand-count
(MAE + bootstrap 95% CI, bias, RMSE, exact-match). Hand-count template:
`notebooks/model_comparison_handcount.csv` (fill `human_count` by eye on the saved frames).

### Pilot results (Feb 2024, conf 0.25, single frame per scene1)

| Model | total worms | mean/scene | max | zero-detection |
|---|---|---|---|---|
| v0 `mushroom.pt` | 16 | 0.52 | 3 | **21 / 31** |
| v1 `best.pt` (retrained) | 39 | 1.26 | 4 | **9 / 31** |

- v0 and v1 agree on 13/31 frames, differ on 18; **in 17 of 18 disagreements v1 ≥ v0**.
- **v1 finds worms in 12 of v0's 21 zero-detection frames** → v0's high zero rate is largely
  *recall failure*, not real absence. Since Feb 2024 is unseen by v1, this is genuine
  sensitivity, not memorization.
- **Caveat:** more detections ≠ more correct — v1's extra boxes could be false positives.
  The hand-count validation (nb 26, cell 6) is what decides which sensitivity is right.
  Not yet done; counts remain a plumbing pilot, **not IRR-gated**, not thesis numbers.

### Environment / model notes
- `ultralytics` (8.4.62) is available **only** in `/home/jovyan/joseph-scaleworm-thesis/.venv`
  (Jupyter kernel **`joseph-scaleworm-thesis`**); the student-lab `.venv` and `scaleworm` conda
  env lack it. Both counting notebooks are set to that kernel. Cleaner fix: add ultralytics to
  the student-lab venv.
- GPU driver too old (CUDA 12020) → runs on CPU; 31 frames ≈ 0.8 min, fine.
- `mushroom.pt` classes = `{0: 'scale_worm'}` (single class), same as v1's training set.

### Next
Hand-count the 31 Feb-2024 frames → run nb 26 vs-human metrics → choose v0/v1 (and re-run when
v2 lands); IRR κ gate on the Scene-1 judgment; decide full counting scope (2023–2024).

---

*Log maintained by Claude Code — updated after each significant step.*
