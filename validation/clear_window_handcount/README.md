*AI-generated draft (Claude, Anthropic) — for review. All selection logic, frames, and numbers are produced by version-controlled scripts and data in this repo; worm counts will be Makayla Joseph's. Confirmed findings are labelled as such.*

# Clear-window hand-count validation set — scale worms at Mushroom (CAMHDA301)

**For:** Makayla Joseph & Dr. Dax Soule
**Built:** 2026-09-02 · scripts committed in `scaleworm-student-lab`
**Status:** ground-truth counts pending (this is the sheet to fill)

---

## 1. What this is (one paragraph)

A stratified set of front-on Mushroom ("Scene 1") frames drawn from the **clearest
two camera-years** of the CAMHDA301 archive. Its job is to be **ground truth**: you
count scale worms by eye on each frame, and we later run the existing detector on the
**same** frames to measure how well it recovers your counts (recall) in the clear
image regime. It is also a **held-out test set** for any future re-trained model.

## 2. Why we built it (background)

- A multi-year image-quality survey (2015–2026) showed the Mushroom HD camera is
  **swapped every August**, and clarity depends on the specific **camera unit**, not
  the season. The two clearest units are **CAMHDA301-2021** (~Sept 2021 → Aug 2022)
  and **CAMHDA301-2022** (~Aug 2022 → July 2023) — a continuous ~2-year clear window.
  *(See `notebooks/figure_sharpness_2015_2026.html` and `..._weekly_2021_2023.html`.)*
- Before deciding whether to **re-train** the detector, we need to know whether the
  **existing** detector is already good enough on clear footage. This set answers that:
  it is the cheap "gate" measurement. Retraining on clear footage would *not* fix the
  blurry post-2023 years, so it is only worth doing if this test shows the current
  model underperforms on clear footage.

## 3. How frames were selected (provenance)

Deterministic, no random sampling (`scripts/build_clear_window_validation.py`):

| Camera unit | Period | Source of Scene-1 | Cadence |
|---|---|---|---|
| **CAMHDA301-2022** | 2023-01 … 2023-07 | existing sort log (`scene_sorting/full_2023_2024/sort_log.csv`) | 4 recordings/month, evenly spaced |
| **CAMHDA301-2021** | 2021-09 … 2022-07 | **contact sheet** (no sort exists yet) | 2 recordings/month, evenly spaced |

Recordings are spread across days-of-month and hours-UTC. Frames are extracted at
**full resolution** (no downscaling) for accurate counting.

## 4. Two-part workflow (important — the two units differ)

**Part A — CAMHDA301-2022 rows (2023): ready to count now.**
The Scene-1 time was already known, so the full-res frame is in `frames/<frame_id>.png`.
Just open it and count.

**Part B — CAMHDA301-2021 rows (2021–22): one extra step first.**
No sort exists for these, so instead of a frame you get a **contact sheet** in
`contact_sheets/<frame_id>.png` (a grid of the whole recording). For each:
1. Find the tile showing the **front-on Mushroom view** (same view as the 2023 frames).
2. Write that tile's time (the `t=___s` label) into the **`scene1_time_s`** column.
3. When those are filled, run once:
   `python scripts/extract_pending_frames.py`
   — this extracts the full-res frame into `frames/`. Then count as in Part A.

## 5. How to count (do this consistently)

- Count every distinct **scale worm** you can identify in the frame — same criteria
  the lab has used for the Jan-2023 / Feb-2024 / May-2024 hand counts, so numbers are
  comparable. If you and another rater will both count (recommended for inter-rater
  reliability), each uses a separate copy and we compare before merging.
- Record the number in **`worm_count`**, your initials in **`counter`**, and the date
  in **`date_counted`**. Use **`notes`** for anything unusual (heavy marine snow,
  partial view, ambiguous animals).
- If a frame is not actually a usable front-on view, put `notes = "not front-on"` and
  leave `worm_count` blank rather than guessing.

## 6. `handcount_sheet.csv` columns

| column | meaning |
|---|---|
| `frame_id` | recording stem, e.g. `CAMHDA301-20230102T001500` |
| `datetime_utc` | recording time (UTC) |
| `camera_unit` | `CAMHDA301-2021` or `-2022` |
| `quarter` | e.g. `2023-Q1` (for balance checking) |
| `sharpness_survey` | automated sharpness score where available (context only) |
| `scene1_time_s` | seconds into the clip of the front-on view (pre-filled for 2023) |
| `frame_ready` | `Y` = full-res frame in `frames/`; `pending-contact-sheet` = do Part B |
| **`worm_count`** | **← you fill this** (integer) |
| **`counter`** | **← your initials** |
| **`date_counted`** | **← date you counted** |
| `notes` | free text |
| `video_path` | source video on the Hub |

## 7. Rules & caveats

- **Leakage:** these frames are a **TEST set**. Do **not** include them in any future
  training data — that would inflate the model's apparent accuracy.
- **Scene-1 precision** is ~30 s (contact-sheet tile spacing), so the extracted frame
  may sit slightly off the ideal view.
- **IRR:** counts feeding thesis claims should be inter-rater-reliability gated
  (scene-1 IRR is still pending). Two independent counters here is ideal.
- **Sharpness scores** are a *relative* index (they mix camera scenes); they explain
  which recordings are clear, they are not worm counts.
- **Phase note:** this is validation/experiment tooling. The thesis is in Phase 02
  (lit review); detector thresholds are locked in Phase 04 before analysis.

## 8. What happens next

1. You count `worm_count` for all rows (Part A now; Part B after the scene-pick step).
2. We run the existing detector (`mushroom.pt` / `best.pt`) on the same `frames/` and
   compute **recall vs. your counts, with a bootstrap 95% CI**, split by camera unit.
3. That number decides whether to re-train — and if so, this set is the held-out test.

## 9. Reproducibility & attribution

- Build: `scripts/build_clear_window_validation.py`; re-extract: `scripts/extract_pending_frames.py`.
- Frame extraction / Scene-1 mapping reuse `scripts/count_frames.py`; contact sheets
  reuse `scripts/scene_sampler.py`.
- Data: OOI Regional Cabled Array CAMHD video (Ocean Observatories Initiative, NSF),
  camera `RS03ASHS-PN03B-06-CAMHDA301`, Hub mirror `/home/jovyan/ooi/san_data/...`.
- **Image files** (`frames/`, `contact_sheets/`) are **Hub-local and git-ignored**
  (repo ignores `*.png`). Only `handcount_sheet.csv` + `README.md` are versioned.
  On another machine / a fresh clone, regenerate the images by running
  `python scripts/build_clear_window_validation.py` on the Hub. Your counts live in
  the versioned CSV, so they are never lost even though the images are not committed.
