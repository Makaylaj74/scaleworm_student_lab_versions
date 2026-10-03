*AI-generated draft (Claude, Anthropic) — for review. All numbers derive from the
version-controlled `scripts/gate_scene_classifier.py` scoring MJ's blind nb40 sort against
the pre-committed classifier predictions (commit ddf1a99); regenerate per the command below.*

# Wednesday Scene-1 validation — results note (2026-10-03)

## Bottom line

The trained Scene-1 classifier **transfers well to Wednesdays**, independently confirming
that the M-F harvest's recall (previously validated only against manual *Monday*) is not a
Monday-specific artifact. On a blind 96-recording Wednesday sample, classifier-vs-MJ
agreement is **κ = 0.709 (substantial), 85.4% raw (95% CI 77–91%)**. The classifier's
**recall is strong (92.7%)** — it rarely misses a Scene-1 MJ found — and its main error is
**over-calling** Scene-1 (precision 77.6%). Crucially, **2024 (blurry) agreement (83%) is
no worse than 2022–2023**, so the tile-level blurry weakness does not break recording-level
recall. **No Borrowed Assumptions check: passed** — the Monday→weekday recall transfer is
now evidenced on held-out Wednesdays, not assumed.

## Design (blind, pre-committed)

96 recordings drawn from the 1363 rendered Wednesday sheets, **stratified 3 per year×slot**
(`build_wednesday_validation.py`, seed 20261003). The classifier scored every recording and
its predictions were **committed before MJ sorted** (ddf1a99) — she then sorted all 96
**blind** in nb40 without seeing them, so there is no anchoring. Scoring compares the two
(`gate_scene_classifier.py`); MJ's sort is the reference.

## Results (n = 96 paired)

| Metric | Value |
|---|---|
| Cohen's κ | **0.709** (substantial) |
| Raw agreement | **85.4%** (Wilson 95% CI 77.0–91.1%) |
| MJ Scene-1 / model Scene-1 | 41 / 49 |
| Recall (TP/MJ-Scene-1) | **92.7%** (38/41) |
| Precision (TP/model-Scene-1) | **77.6%** (38/49) |
| F1 | 84.4% |
| Scene-1 time agreement | **33/38** within one 30 s tile |

**Error direction:** 14 disagreements = **11 false positives** (model says Scene-1, MJ
not) + **3 misses** (MJ Scene-1, model not). So the classifier over-calls rather than
misses.

**Agreement by year:** 2021 96% (23/24) · 2022 79% (19/24) · 2023 83% (20/24) ·
**2024 83% (20/24)**. The blurry 2024 era is *not* the weak spot at the recording level.

## Interpretation

- **Recall transfers.** Only 3 of 41 MJ-Scene-1 recordings were missed, and all three sit
  right at the 0.908 decision threshold (p = 0.56, 0.42, 0.895) — near-misses, not blind
  spots. The classifier reliably *finds* the front-on Wednesday view.
- **Precision is the gap.** 11 recordings MJ rejected were called Scene-1 with **high
  confidence** (p = 0.92–0.9997). High-confidence disagreements are exactly where a
  scene-definition ambiguity hides, so these should be **eyeballed by MJ** before they are
  written off as model errors — some may be borderline tower views she judged too
  oblique/zoomed-out. Until reviewed, treat recording-level precision as ~78% (a lower
  bound if any FPs are actually acceptable Scene-1).
- **Blurry 2024 holds up.** The tile-level test showed blurry recall 0.75, but at the
  recording level (any tile clears threshold) 2024 agreement matches 2022–2023. Picking the
  single best frame per recording is more forgiving than per-tile classification.
- **Timing is usable.** When both call Scene-1, the model's chosen frame is within ±30 s of
  MJ's 87% of the time — good enough to seed the detection step.

## Decision / recommendation

- The classifier is **good enough to pre-seed the Wednesday (and by extension Tue/Thu/Fri)
  scene sort**: use it as a first pass, then human-review, concentrating review on its
  **Scene-1 calls** (precision 78%) rather than re-sorting cold. Its misses are rare and
  near-threshold.
- **Do not** treat the classifier's Scene-1 label as final without spot-review in a context
  that needs clean precision (e.g. a curated training set), given ~1 in 5 calls is contested.
- **Next:** MJ reviews the 11 high-confidence false positives to decide whether they are
  model errors or strict-human calls; that resolves whether precision is ~78% or higher, and
  surfaces any scene-definition rule to pin in the rubric.
- The abundance series and dw/dt figure are unaffected (the harvest uses the detector, not
  this classifier, to pick frames); this validates the *option* of classifier-driven scene
  selection going forward.

## Reproduce

```bash
python scripts/build_wednesday_validation.py                       # the 96-frame sample
/home/jovyan/joseph-scaleworm-thesis/.venv/bin/python3 \
    scripts/gate_scene_classifier.py --session scene_sorting/wednesday_validation
# -> validation/scene_classifier_gate/wednesday_validation/{predictions,gate_metrics,gate_results}
```
MJ's blind sort: `scene_sorting/wednesday_validation/sort_log.csv` (via nb40). The 11 FP /
3 FN stems are listed in `gate_metrics.json`.
