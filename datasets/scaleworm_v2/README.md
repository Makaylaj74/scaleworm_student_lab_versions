*AI-generated draft (Claude, Anthropic) — for review. All frame selection, pre-labels, and splits are produced by version-controlled scripts; the corrected boxes will be Makayla Joseph's ground truth. Confirmed vs planned steps are labelled.*

# scaleworm v2 — labeling & retrain plan (domain adaptation of mushroom.pt)

**Goal:** widen the detector's usable domain so the abundance record can extend beyond
the ~1-year -2022 clear window. The recall gate showed mushroom.pt works only in-domain
(-2022 sharp: ~68%) and fails out-of-domain — on sharp-but-different -2021 (~37%) and on
blurry post-2023 (~4-5%). v2 fine-tunes mushroom.pt on corrected labels from the new
domains to recover recall there.

## Why fine-tuning (not just hand counts)
Hand counts grade a model; they cannot train one. Training needs **bounding boxes**. We
cut the labeling cost with **model-assisted pre-labels**: `scripts/make_prelabels.py`
runs mushroom.pt on each frame and writes its boxes as YOLO labels. Because mushroom.pt
has **no false positives** (pure recall miss), a human corrects a frame by *adding the
missed worms* — rarely deleting — which is far faster than boxing from scratch.

## Leakage rules (do not break)
- The **39-frame clear-window validation set** (`validation/clear_window_handcount/`) is
  the INDEPENDENT test. Never place those recordings in `images/`—`train_frames_2022.csv`
  already excludes them.
- `train/` and `val/` here are for *training* only; the val split is a held-out slice of
  the corrected training frames, used for training metrics — NOT the gate.
- Acceptance = **`scripts/run_recall_gate.py`** re-run with `MODEL` pointed at the new
  `best.pt`, reporting recall per camera unit on the 39 held-out frames. That is the
  number that decides whether a unit becomes usable (threshold locks in Phase 04).

## Status
- **-2022 batch — READY (done 2026-09-08):** 40 frames evenly spread across 2023-01..07,
  pre-labeled (323 boxes, ~8.1/frame) in `prelabels/2022/`. Awaiting human correction.
- **-2021 batch — PENDING scene-picks:** no sort exists for -2021, so training frames must
  be scene-picked from contact sheets first (like the validation set, but DIFFERENT
  recordings than the 11 validation ones). Target ~40 frames across 2021-09..2022-07.

## Workflow (per batch)
1. **(-2021 only) pick Scene-1 times** for a fresh set of -2021 contact sheets — reuse the
   `notebooks/30_pick_scene1_2021.ipynb` pattern on a training manifest, then extract.
2. **Pre-label:** `python scripts/make_prelabels.py <manifest.csv> prelabels/<unit>`
3. **Correct:** open each frame in a box-labeling tool, ADD the worms mushroom.pt missed
   (and delete any rare wrong box). Recommended: **Label Studio** (`pip install
   label-studio`, image-bbox template, import the YOLO pre-labels, export YOLO). CVAT or
   Roboflow work too. (If you'd prefer an in-repo click-box notebook in the nb-29 idiom,
   ask — it can be built.)
4. **Split & place:** move corrected image+label pairs into `images/train`+`labels/train`
   (~85%) and `images/val`+`labels/val` (~15%). Keep whole months together where possible.
5. **Train:** `python scripts/train_v2.py` (see compute note below).
6. **Validate:** point `run_recall_gate.py` at the new `best.pt`, re-run; compare per-unit
   recall to mushroom.pt. Log the run in `joseph-scaleworm-thesis/99_experiments/`.

## Compute constraint (important)
This Hub's GPU driver (CUDA 12020) is too old for the installed torch → training runs on
**CPU**, which is impractically slow at full resolution. Options: (a) a short reduced-imgsz
smoke test here to prove the pipeline; (b) run the real training on a **GPU machine**
(Colab / lab workstation) with this same dataset dir; (c) newer-driver GPU on the Hub.

## Phased strategy (matches "recover as much record as possible")
- **Phase 1 — sharp non-2022 units (high odds):** label -2021 (+ keep the -2022 batch to
  prevent forgetting *and* teach the currently-missed worms). Expect recall on -2021 to
  climb toward -2022 levels → buys back the continuous clear record ~Sept 2021 → July 2023
  (and older clear units the multiyear survey flagged).
- **Phase 2 — blurry post-2023 (harder, has a ceiling):** label fouled-footage frames.
  Worms are still human-countable (7-9/scene) so retraining will help, but biofouling
  destroyed fine detail — expect a residual recall gap and month-varying uncertainty;
  document any months that stay detector-unreliable rather than forcing a number.

## Files
- `train_frames_2022.csv` — the 40-frame -2022 training manifest (leakage-excluded).
- `prelabels/2022/{images,labels}` — pre-labeled -2022 batch (to correct).
- `prelabels/2021/` — empty; fills after -2021 picks.
- `data.yaml` — YOLO config (1 class: scale_worm).
- `images|labels/{train,val}` — corrected ground truth goes here.
- Scripts: `scripts/make_prelabels.py`, `scripts/train_v2.py`; acceptance via
  `scripts/run_recall_gate.py`.
