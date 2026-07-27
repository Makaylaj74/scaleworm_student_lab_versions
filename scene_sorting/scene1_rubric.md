*AI-generated draft (Claude, Anthropic) — for review. All parameters and thresholds are derived from version-controlled scripts and data; every rater must confirm this rubric before labeling.*

# Scene 1 Labeling Rubric (CAMHD Mushroom vent)

**Version:** draft 1 · 2026-07-27
**Applies to:** the inter-rater reliability (IRR) subset under `scene_sorting/irr_round_1/`
**Raters:** three, labeling independently and blind to one another (`rater_a`, `rater_b`, `rater_c`).

This rubric exists so that a disagreement between raters measures the *difficulty of the
judgment*, not differences in how each of us defined the task. Read it fully and confirm you
agree with the pinned rules **before** opening the first contact sheet. If a rule is wrong or
ambiguous, raise it now — changing it after labeling starts invalidates the comparison.

---

## 1. What we are labeling

Each **contact sheet** is one ~14.8-minute CAMHD recording rendered as a grid of frames, one
tile every 30 s, each tile labeled with its clip-elapsed time (`t=...s`). The PTZ camera runs a
pre-programmed tour, so a single recording contains several different scenes. Your job is to
decide, per recording, **whether the Scene 1 view appears anywhere in it**, and if so, **at what
clip time**.

You are **not** counting worms and **not** judging image quality beyond what the criteria below
require.

## 2. The three decisions

The notebook exposes exactly three actions:

| Action        | Meaning                                                                 |
|---------------|-------------------------------------------------------------------------|
| `s1(t)`       | Scene 1 **is** present; `t` is the clip time (seconds) where it appears |
| `no()`        | Scene 1 is **not** present anywhere in this recording                   |
| `sk()`        | Cannot decide (corrupt tiles, ambiguous, needs discussion)              |

Use `sk()` sparingly and only for genuine can't-tell cases — a skip drops that recording from the
agreement statistics. Do **not** use `sk()` to mean "probably not Scene 1"; that is `no()`.

## 3. Definition of Scene 1 — the positive criteria

Scene 1 is the **front-on view of the Mushroom sulfide edifice**. Call it Scene 1 only when
**all** of the following hold in at least one tile:

1. **Subject:** the Mushroom tower is the dominant structure, viewed roughly **face-on** (the
   flange/cap reads as a horizontal cap over the column, not edge-on or from behind).
2. **Framing/zoom:** the camera is **zoomed in** on the edifice so it fills a large fraction of
   the frame — not the wide survey/pan view where the tower is small and distant.
3. **Composition:** the tower is roughly **centered and upright**, with the vent orifice / active
   area visible. This is the view the 2023 ground-truth annotations and the training clips were
   drawn from.

If you are unsure whether a zoomed view is "front-on enough," treat the 2023 training frames as
the reference standard for what counts (see `~/scaleworm_starter/baseline/annotations_scene1/`).

## 4. What is NOT Scene 1 — negative criteria

Mark `no()` when the recording shows only:

- **Wide / survey views** where the tower is small, distant, or one of several structures.
- **Oblique or rear angles** of the edifice (cap not reading as a horizontal cap over the column).
- **Other vents or seafloor** that are not the Mushroom tower.
- **Transit / pan / zoom-in-progress** frames only, with no tile settling on the front-on view.
- **Blue-water, blank, or lens-flare** frames with no usable structure.

A recording that merely *passes over* the tower during a pan, without a tile resting on the
front-on zoomed view, is `no()` — the detection step needs a settled frame, not a glimpse.

## 5. PINNED RULE — which appearance to time  ⚠️ confirm before starting

Because the PTZ tour drifts, Scene 1 can appear **more than once** in a recording (the warm-up
batch showed clusters near ~150–210 s and again near ~420 s). Two raters who both correctly see
Scene 1 could still record different times. To make the time comparable across raters we pin one
rule:

> **Record the time of the FIRST tile that satisfies all Section 3 criteria.**
> Read `t` directly from that tile's `t=...s` label. Times are therefore quantized to the 30 s
> tile spacing — this is a *locator* (the exact frame is refined later), not a precise timestamp.

Rationale for "first" over "clearest": "first" is a mechanical, low-judgment rule, so it
maximizes reproducibility across raters, which is the whole point of the IRR exercise. "Clearest"
introduces a second subjective judgment on top of the presence judgment and would inflate
disagreement for reasons unrelated to whether Scene 1 is present. If the team prefers "clearest"
or "longest-dwell," change this rule **here, before labeling** — do not mix conventions.

## 6. Independence and blinding — the rules that make this valid

1. **Label alone.** Do not discuss individual recordings with the other raters until all three
   logs are complete and the agreement stats have been computed.
2. **Do not look** at another rater's `scene1/` / `not_scene1/` folders or `sort_log.csv`.
3. **Same sheets, separate sessions.** Each rater points the notebook `SESSION_DIR` at their own
   folder (`scene_sorting/irr_round_1/raters/rater_a`, `.../rater_b`, `.../rater_c`). The contact
   sheets are identical copies; your decisions are logged only in your own folder.
4. **One pass.** Do not go back and "fix" earlier decisions to match a pattern you notice later —
   that couples your labels to your own memory and defeats the measure. `undo()` is only for an
   immediate mis-click on the sheet you just filed.
5. Work in whatever sittings you like; the log is resumable.

## 7. Edge cases and conventions

- **Multiple front-on appearances:** apply Section 5 — time the **first** one only.
- **Partial / cut-off tower** (edifice half out of frame) but clearly front-on and zoomed:
  count as Scene 1; time the first such tile.
- **Corrupt or missing tiles** in the middle of an otherwise clear recording: judge from the
  tiles you can see; only `sk()` if the corruption prevents a confident presence decision.
- **Borderline zoom** (medium shot, tower prominent but not filling the frame): default to `no()`
  and note the stem for group review — consistent borderline handling matters more than any
  single call.

## 8. After all three raters finish

Run `scripts/irr_agreement.py` on the three session directories. It reports:

- **Scene 1 vs not:** Fleiss' κ (all three raters) and pairwise Cohen's κ, with bootstrap 95% CIs.
- **Time agreement:** on recordings all raters called Scene 1, the fraction agreeing within
  ±1 tile (±30 s) and the mean absolute pairwise time difference, with bootstrap 95% CIs.

Then meet to adjudicate every disagreement — those cases tell us which criterion to tighten and
which recordings are genuinely hard.
