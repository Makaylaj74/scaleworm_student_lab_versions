*AI-generated draft (Claude, Anthropic) — for review. All parameters and figures are derived from version-controlled scripts and data.*

# Assignment — Scene-Sorting the HD Mushroom-Vent Camera, 2019–2020

**Assigned to:** _(to be assigned)_
**Parallel work:** Lyr Gada (scene-sorting 2017–2018) · Maureen Scully (OOI operational timeline 2017–2018)
**Project:** Scaleworm thesis (GEOL 795), Queens College CUNY
**Instrument:** CAMHDA301 — OOI Regional Cabled Array HD camera, Mushroom vent, ASHES vent field, Axial Seamount (`RS03ASHS`)

## Goal

Run the existing Scene-1 scene sorter over the CAMHDA301 record for **2019–2020** and report, month by month, how many **clear Scene-1** frames exist. This closes the gap between Lyr's 2017–2018 work and the already-sorted 2021–2024 record, so the team has a continuous 2017–2024 backbone. Your output tells us which 2019–2020 months have imagery good enough to count scaleworms in.

## Background you need

- CAMHDA301 records a short HD **pan sequence** on a fixed schedule; each sequence steps the camera through named framing positions ("scenes"). **Scene-1** is the specific framing we count scaleworms in: a **zoomed-in, angled top-down view showing tubeworm clumps** — *not* a zoomed-out/wide view and *not* a chimney/plume shot. The sorter's job is to separate Scene-1 frames from everything else.
- The sorter + sort-log workflow already exist in the repo and were run on **2021–2022**. You are extending that same tooling to 2019–2020. Same cadence: every Monday × the eight 3-hourly slots.
- Practically: build the contact sheets with `scripts/scene_sampler.py` (point `--start`/`--end` at 2019–2020, `--out scene_sorting/2019_2020`), then sort in `notebooks/25_sort_scenes.ipynb` (set `SESSION_DIR` to your session) using `s1(t)` / `no()` / `sk()` / `undo()`. It is resumable and writes `sort_log.csv`.

## ⚠️ Important caveat — do not assume 2019–2020 behaves like 2021–2022

Our image-quality survey found the **clearest** imagery was **2021–2022**, and a recall test showed the detector is **domain-dependent, not just blur-dependent** — it did well on 2022 imagery but noticeably worse on *sharp but visually different* 2021 imagery. **2019–2020 is an earlier era** (different camera units — swapped each August — different biofouling/servicing) and may look different again. The camera unit was also **swapped in August of each year**, so 2019–2020 spans more than one physical camera; clarity can change abruptly at those swaps.

**So part of your job is to check whether the sorter still deserves our trust here.** Report not just the sorter's numbers but whether they hold up to your eyes.

Why this matters downstream: 2019–2020 will be **counted by the AI detector**, not by hand (that's how we cover the historical years in time for the deadline). But the detector was trained on 2021–2022 footage, so before we trust its 2019–2020 counts we run a small **per-unit "gate"** — Makayla hand-counts ~20–30 frames per camera-year and we measure the detector's recall. **Your clear-Scene-1 frames are the raw material for that gate**, so the cleaner and more honest your sort, the better the gate.

## Tasks

1. **Build + sort 2019–2020.** Generate the contact sheets and run the Scene-1 sorter following the existing workflow, producing `scene_sorting/2019_2020/sort_log.csv`.
2. **Spot-check by eye.** Randomly sample ~20–30 frames the sorter/you labeled Scene-1 and ~20–30 rejected, and view them (`imshow`). Does the call match your eyes? Note any systematic issue (a whole framing style mis-called, or obviously blurry frames passed).
3. **Tabulate clear-Scene-1 yield by month.** Report the count per month; flag **zero/near-zero** months so we can tell a camera outage from a sorter/footage failure.
4. **Write a short transfer note** (½–1 page): *Does the Scene-1 criterion apply cleanly to 2019–2020?* Yes / partly / no, with evidence from your spot-check. If a camera swap changes the look mid-period, say where. A clear "this era is different" finding is valuable, not a failure.

## Deliverables

- `scene_sorting/2019_2020/sort_log.csv` (the sort).
- Per-month clear-Scene-1 counts (the table below).
- The transfer note + your spot-check sample (so the finding is reproducible).

## Handoff points

- **You need:** the 2019–2020 **operational timeline** — which months the camera was actually deployed/recording — so a zero-yield month can be labeled "camera off" vs "sorter/footage failure." If no one is assigned the 2019–2020 timeline, pull it yourself from the OOI cruise/deployment records (ask Makayla for the path used for 2017–2018).
- **The team needs from you:** the clear-Scene-1 counts, so Makayla can run the per-unit gates and the AI counting for 2019–2020.

## Deliverable table (fill as you go)

| Month (2019–2020) | Camera status | Clear Scene-1 count | Include in series? | Notes (swap? blur? outage?) |
|---|---|---|---|---|
| 2019-01 | | | | |
| 2019-02 | | | | |
| … | | | | |
| 2020-12 | | | | |

## Working conventions

- New files/dirs in `snake_case`; run `ruff format` / `ruff check` and `uv run pytest` before committing any code you touch.
- Use `imshow` (matplotlib) for viewing frames.
- Sort at the **same Monday × 8-slot cadence** as the rest of the record so the series stays comparable. If time is short, tell Makayla before dropping to a coarser cadence.
- If you write explanatory prose for the team, label it per the lab's AI-disclosure rule if AI-assisted.
