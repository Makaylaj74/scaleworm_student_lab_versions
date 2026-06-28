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

*Log maintained by Claude Code — updated after each significant step.*
