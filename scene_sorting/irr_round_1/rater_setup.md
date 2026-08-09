*AI-generated draft (Claude, Anthropic) — for review. All parameters and paths are derived from version-controlled scripts and data.*

# Scene 1 sorting — setup note for raters

You've been asked to independently sort a set of **80 CAMHD contact sheets** into
**Scene 1** vs **not Scene 1**, and — for the Scene 1 ones — record roughly *when* in the
clip the Scene 1 view appears. All three of us label the **same 80 sheets separately**; we
then compare to measure how reproducible the judgment is (inter-rater reliability). This is a
validation exercise, so **labeling independently and to the same rules is the whole point** —
please don't compare notes until everyone is done.

Budget ~30–60 min. You can stop and resume anytime; your progress is saved as you go.

---

## Before you start — what you need

1. **The repo** `scaleworm_student_lab_versions` checked out (you have this if you're reading it).
2. **Your rater folder** — one of `scene_sorting/irr_round_1/raters/rater_a`, `rater_b`, or
   `rater_c`. Makayla will tell you which ID is yours. The 80 image files are **not in git**
   (too large), so Makayla will hand you the folder separately (shared drive / zip / Hub copy).
   Drop it at `scene_sorting/irr_round_1/raters/<your_id>/` so the paths below line up.
3. **The `scaleworm` Jupyter kernel** (has matplotlib) — select it in the notebook, top right.

## Step 0 — set up your folder (you can paste this to Claude)

The `raters/<your_id>/` folder and the 80 images are **not in git**, so they won't appear from a
`git pull` — you have to place them yourself. If you'd rather not do it by hand, open Claude Code
**in the repo root** and paste the prompt below, after filling in the two `<...>` blanks
(`<YOUR_ID>` = the ID Makayla gave you; `<PATH_TO_YOUR_ZIP>` = wherever the zip Makayla sent you
landed, e.g. `~/Downloads/rater_b.zip`):

```
I'm a rater in a scale-worm "Scene 1" sorting exercise, working in this repo. My rater ID is
<YOUR_ID>. Please set up my folder without looking at any image contents or touching anyone
else's rater folder:

1. Make sure scripts/scene_sorter.py and notebooks/25_sort_scenes_clean_for_raters.ipynb exist.
   If either is missing, stop and tell me (I may need to `git pull` or be handed a file).
2. Create scene_sorting/irr_round_1/raters/ if it doesn't exist, then unzip <PATH_TO_YOUR_ZIP>
   so the images end up at exactly:
       scene_sorting/irr_round_1/raters/<YOUR_ID>/contact_sheets/*.png
   The zip's top-level folder is already named <YOUR_ID>, so extract it INTO
   scene_sorting/irr_round_1/raters/ — do NOT create a doubled path like
   raters/<YOUR_ID>/<YOUR_ID>/contact_sheets.
3. Verify the count: report how many .png files are in
   scene_sorting/irr_round_1/raters/<YOUR_ID>/contact_sheets/. It must be exactly 80.
4. Do not create, move, or edit any sort_log.csv, scene1/, or not_scene1/ folder — the notebook
   makes those on first run. Just report what you did and the final PNG count.
```

If it reports anything other than **80** contact sheets, stop and check with Makayla before
labeling — a wrong count means the wrong or partial folder.

## Step 1 — read the rubric (required)

Open **`scene_sorting/scene1_rubric.md`** and read it fully. It defines exactly what counts as
Scene 1 and, importantly, **which appearance to time**: Scene 1 can show up more than once in a
clip, so we all record the **first** tile that clearly shows the front-on Mushroom view. If you
disagree with any rule, raise it with the group *now* — changing it after we start invalidates
the comparison.

## Step 2 — point the notebook at YOUR folder

Open **`notebooks/25_sort_scenes_clean_for_raters.ipynb`** (this is the rater copy — it has no
pre-filled decisions). In the **config cell** (the first code cell), change the one `RATER_ID`
line to your ID:

```python
RATER_ID = "rater_a"  # <-- CHANGE THIS to your assigned ID
```

Run that cell — it should print `remaining: 80`. If it prints a different number or an error,
stop and check with Makayla; do **not** proceed pointing at the wrong folder.

## Step 3 — nothing to clear (rater copy is already clean)

This rater notebook ships with **no decisions in it** — just the config cell, the engine cell,
and one empty driver cell. So there is nothing to delete: don't paste in anyone else's calls, and
work only in the single driver cell (the one with the `# DRIVER CELL` comment).

## Step 4 — label the 80 sheets

Run the **config cell** and the **engine cell** once, top to bottom. The first contact sheet
appears as an image. Each sheet is a grid of frames from one recording, one frame every 30 s,
with a `t=...s` time label under each tile.

In your single **driver cell**, type one of these and press **Shift+Enter**:

| Call      | Meaning                                                                     |
|-----------|-----------------------------------------------------------------------------|
| `s1(t)`   | Scene 1 — `t` = the `t=...s` label under the **first** tile showing it (e.g. `s1(210)`) |
| `no()`    | Not Scene 1 anywhere in this recording                                      |
| `sk()`    | Can't tell (corrupt/ambiguous) — leave for group discussion. Use rarely.    |
| `undo()`  | Undo your last decision (fixes a mis-click on the sheet you just filed)     |
| `show()`  | Redraw the current sheet without deciding                                   |

Each decision files the sheet, logs it, and draws the next one. Just edit the same driver cell
and Shift+Enter again for the next sheet. The header line shows your running counts and how many
remain. `sk()` is not "probably not" — that's `no()`; skip only genuine can't-tell cases.

## Rules that keep this valid

- **Label alone.** Don't discuss individual sheets, and don't look at anyone else's folder or
  results, until all three of us have finished.
- **One pass.** Don't go back to "fix" earlier calls to match a pattern you notice later —
  `undo()` is only for an immediate mis-click.
- **First appearance only** for the time (see the rubric).

## When you're done

Tell Makayla. Your decisions live in `scene_sorting/irr_round_1/raters/<your_id>/sort_log.csv`
— she'll collect the three logs and run the agreement stats. Then we meet to talk through any
sheets we disagreed on. Thank you!
