*AI-generated draft (Claude, Anthropic) — for review. The frame selection and extraction are version-controlled (scripts/build_early_recovery_gate.py); any worm counts entered here are the counter's own data.*

# Early-recovery hand-count gate (2015-2016)

Held-out hand-count set for the **post-Axial-eruption early-recovery era** (the 2015
eruption is the scientific motivation for extending the scaleworm record backward). Its
purpose is to measure the **real** recall/precision of the detector on this era and/or to
seed a fine-tune (v6) — exactly the role `clear_window_handcount` played for 2021-2024.

## ⚠️ Do not count yet — the count TARGET is undefined

The 2015-2016 view is a **dense tubeworm/community mound**, visually different from the
2021-2024 mushroom-tower scene the detector was trained on. The 2021-2024 count rubric
does **not** transfer (No Borrowed Assumptions). Before anyone clicks:

1. Decide **what a countable unit is** in this scene (individual scale worms? occupied
   tubeworm patches? % cover?) **with the advisor**, using the eruption/colonization
   context (was Mushroom physically disturbed in April 2015? what is the succession
   target?).
2. Record that definition here, then count.

## Provenance

- Frames: Scene-1 recordings confirmed by the manual scene sorts
  (`scene_sorting/full_2015/`, `scene_sorting/subsample_2016/`), extracted at each
  recording's `scene1_time_s` from the CamHD archive.
- Monthly spread: 2015 = 2 per month (Jul-Dec); 2016 = all confirmed subsample Scene-1.

## Workflow (once the count target is defined)

1. Open `notebooks/29_clickcount_clear_window.ipynb`, set
   `BASE = .../validation/early_recovery_handcount_2015_2016`, kernel
   `joseph-scaleworm-thesis`.
2. Click each worm (per the agreed target); counts + click coords save to
   `handcount_sheet.csv` and `clicks/`.
3. Score the detector: `MODEL=99_runs/scaleworm_v5_all/weights/best.pt LABEL=v5_early
   ... gate_retrain.py`-style scoring against these counts.
