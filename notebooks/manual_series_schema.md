*AI-generated draft (Claude, Anthropic) — for review. All parameters and figures are derived from version-controlled scripts and data.*

# Manual Scaleworm Series — Unified Schema (box-correction = counts = training labels)

**Purpose.** One annotation pass over the weekly-Monday Scene-1 series produces *both* the
manual abundance timeseries **and** the detector training labels, from the same frames, with
no chance of the count and the label drifting apart.

## Core principle

A corrected box set **is** the ground truth for a frame. The worm count is *derived*:
`worm_count = number of boxes`. Never hand-type a count independently of the boxes — the boxes
are the single source of truth, the count is a view of them.

## Two coupled artifacts

Boxes are variable-length per frame, so they don't fit one CSV cell. Store them the way
YOLO/ultralytics already expects, and keep a one-row-per-frame manifest that indexes them:

```
validation/monday_manual_series/
  frame_manifest.csv          # one row per Scene-1 frame (the timeseries + provenance + split)
  labels/                     # YOLO-format boxes, one .txt per frame  ← training labels
    CAMHDA301-20230102T001500.txt
  images/                     # extracted Scene-1 frame per row (or symlink to source)
```

- **Label file** (`labels/<frame_id>.txt`), YOLO format, one line per worm:
  `class x_center y_center width height` (normalized, class 0 = worm).
  `worm_count = line count`. This is exactly what training consumes — no conversion step.
- **Manifest** (`frame_manifest.csv`) — see columns below.

## `frame_manifest.csv` columns

| column | meaning |
|---|---|
| `frame_id` | stem, e.g. `CAMHDA301-20230102T001500` |
| `datetime_utc` | recording timestamp (ISO 8601, UTC) |
| `camera_unit` | deployment, e.g. `CAMHDA301-2023` |
| `quarter` | e.g. `2023-Q1` (for rollups) |
| `scene1_time_s` | timestamp *inside* the video where Scene-1 lands (from `sort_log`) |
| `video_path` | source `.mp4` on disk |
| `sharpness` | image-quality metric carried from the survey (for QC / covariate) |
| `frame_status` | `counted` \| `no_scene1` \| `unusable_blur` \| `pending` |
| `worm_count` | **derived** = boxes in `label_path`; blank/`NA` when not `counted` |
| `label_path` | pointer to `labels/<frame_id>.txt`; blank if no boxes |
| `annotation_method` | `box_corrected` \| `from_scratch` (identifies the anchoring-bias check subset) |
| `prelabel_model` | model/version that produced the pre-labels, e.g. `mushroom.pt@<sha>` |
| `split` | `train` \| `val` \| `holdout` — **assigned before counting** |
| `counter` | annotator initials, e.g. `MJ` |
| `date_counted` | date the frame was annotated |
| `notes` | free text (e.g. "biofouling on lens", "partial Scene-1") |

Extends the existing `validation/clear_window_handcount/handcount_sheet.csv` — same spirit,
plus boxes, split, and method provenance.

## Rules that keep it defensible

1. **`worm_count` is never independent of the boxes.** It is read from `label_path`. This is
   what removes the count/label inconsistency you flagged.
2. **No-observation ≠ zero.** `no_scene1` / `unusable_blur` rows get `worm_count = NA`, **not
   0** — a frame with no usable view is missing data, not an observed absence. Only `counted`
   frames enter the timeseries.
3. **Split is assigned before annotation** (e.g. hold out one full quarter, or every 5th
   Monday) so the model-accuracy claim can't be gamed after the fact. The full set is still the
   manual timeseries regardless of split; the split only governs what you may claim about the
   *model*.
4. **Correct in both directions.** Add worms the model missed, don't only delete false
   positives — otherwise the count inherits the detector's (known-low) recall and biases the
   series downward, worst on blurry frames. On low-recall frames, annotate as if from scratch.
5. **Anchoring-bias check (once).** On a small subset set `annotation_method = from_scratch`
   (annotate blind, no pre-labels shown) and compare to `box_corrected` counts. Agreement ⇒
   box-correction is unbiased and the rest is trustworthy; disagreement ⇒ you've *measured* the
   correction bias for the methods section.

## Building the timeseries from this

Group `counted` frames by Monday date; each Monday has up to 8 three-hourly slots:

- point estimate = **mean** `worm_count` across that Monday's counted slots
- uncertainty = spread across slots (std, or SEM = std/√n) → error bar per Monday
- exclude `no_scene1` / `unusable_blur` (missing, not zero)

The within-Monday replication is free — it already exists in the 8-slot cadence — so every
point on the series carries an error bar with no extra sampling.

## What retraining is for (given this)

If all 367 Monday Scene-1 frames are hand-corrected, the **2023–2024 Monday series needs no
model** — it's counted by hand. The retrained detector earns its keep by **extending beyond the
Mondays**: non-Monday days (densifying 2023–2024) and other years (2021–2022, 2017–2018). The
`holdout` Mondays are what license trusting it out there — and cross-domain transfer (e.g.
2017–2018) must be re-verified against a few hand-corrected frames from that era, not assumed.
