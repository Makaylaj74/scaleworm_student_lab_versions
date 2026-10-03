*AI-generated draft (Claude, Anthropic) — for review. All numbers derive from the
version-controlled `scripts/gate_v6_blurry.py` run against committed models/labels on
2026-10-03; regenerate with the command at the bottom.*

# v6_blurry gate — results note (2026-10-03)

## Bottom line (negative result)

Retraining the detector's blurry side on the **correct dense Tuesday box-labels**
(v6_blurry) did **not** fix blurry counting. On the honest, held-out dense-Tuesday
blurry gate, v6_blurry recovers **~26% of manual counts (23.0% point recall)** —
statistically **indistinguishable from, and by a paired test slightly *worse* than,
v5_all** (~30% / 24.5%), while v6 also regresses a little on clear footage. The blurry
post-Aug-2023 ceiling is therefore a **detector/image-quality limit, not a label
problem**: v6 had properly-counted dense labels and still could not exceed ~¼ recall.
**Keep v5_all deployed; do not deploy v6_blurry.** More blurry labels are unlikely to
help (see Tier B note).

## Why this gate (what was wrong before)

v5's retracted "blurry 94%" was circular: it was trained *and* gated on the
under-counted Monday blurry frames (`model_comparison_handcount.csv`, ~5 worms/frame).
v6_blurry instead trains the blurry side on MJ's box-corrected **dense Tuesday** frames
(mean ~25/frame) and holds out a **day-disjoint** 30% as the gate
(`datasets/scaleworm_v6_blurry/tuesday_blurry_test.txt`, 23 frames). The gate scores
v0 (baseline), v5_all, and v6_blurry on the **same** held-out frames:

- **CLEAR** — `validation/clear_window_handcount`, 39 frames, independent **click** GT,
  conf 0.40 (clear operating point). Checks for regression.
- **BLURRY** — the 23 held-out dense Tuesday frames, MJ **box centres** as GT, conf 0.25
  (blurry operating point). The headline.

Metric: greedy point-in-box recall / precision / F1 (the honest detector metric) plus
count-ratio recall, with per-frame percentile-bootstrap 95% CIs (n=10 000, seed
20260930). **Leakage re-verified: 0/62 gate frames share a day with v6's 394 training
stems (78 training days).**

## Results

Point-matched detection on the held-out gate:

| Model | Clear count-ratio | Clear recall / prec / F1 | Blurry count-ratio | Blurry recall / prec / F1 |
|---|---|---|---|---|
| v0 mushroom (baseline) | 50.6% | 43.4 / 85.9 / 57.7 | 8.3% | 7.9 / 95.8 / 14.7 |
| **v5_all** | 99.6% | 70.2 / 70.5 / 70.3 | **29.9%** | **24.5 / 82.1 / 37.8** |
| **v6_blurry** | 89.8% | 67.3 / 74.9 / 70.9 | **26.1%** | **23.0 / 88.1 / 36.4** |

Blurry per-frame 95% CIs (point recall): v5_all [21.8%, 30.8%], v6_blurry [19.9%, 28.2%]
— heavily overlapping. Both are far below the clear regime and far below usable.

**Paired test (same 23 blurry frames, v6 − v5 per-frame point recall):** mean
**−2.3%**, 95% CI **[−4.7%, −0.1%]**; v6 higher on only 4/23 frames. The interval just
excludes zero, so v6 is marginally but significantly **worse**, not better. Either way
there is **no improvement** to be had here.

This reproduces the v5 note's independent finding (v5 ≈ 32% on dense Tuesday blurry) and
confirms it on a cleanly held-out set.

## Interpretation

- **Not a label problem.** v6_blurry was trained on exactly the dense, correctly-counted
  blurry labels whose absence was blamed for v5's failure. It still plateaus at ~¼
  recall. The information needed to resolve individual worms in dense, blurred frames is
  not recoverable by the detector at these image qualities.
- **Precision stays high, recall collapses.** v6 blurry precision is 88% (few false
  positives) but recall 23% — the model finds *some* worms confidently and misses the
  rest. Consistent with blur erasing the fine structure that distinguishes crowded worms.
- **Small clear regression.** Adding dense Tuesday blurry frames pulled clear count-ratio
  from 99.6% → 89.8% (F1 unchanged at ~71%), i.e. v6 slightly under-counts clear peaks —
  another reason not to prefer it over v5_all.
- **Compounds the scene-finder failure.** The automated harvest's frame-finder *is* the
  detector; a detector that recovers ¼ of blurry worms cannot reliably find the front-on
  blurry frame either (the 2024 61%-bad-frame finding). Blur breaks both together.

## Decision / recommendation

- **Deployed model stays v5_all** (per-regime conf: 0.40 clear / 0.25 blurry). It is as
  good as v6 on blurry and better on clear.
- **Blurry post-Aug-2023 remains NOT usable for automated counting** (~25%, both models).
  Treat blurry years as requiring manual counts, or a fundamentally different approach
  (image restoration / a higher-quality frame source), not another retrain.
- **For the Tier B learning-curve test:** this result predicts *more* blurry labels will
  not lift recall (quality, not quantity, is the limiter) — worth confirming with the
  n15/n30/n47 curve before investing in more blurry hand-counting.
- The clear-window abundance series and the dw/dt figure are **unaffected** — they rest
  on the clear regime and the manual Monday series.

## Reproduce

```bash
/home/jovyan/joseph-scaleworm-thesis/.venv/bin/python3 scripts/gate_v6_blurry.py
# -> validation/v6_blurry_gate.csv  + the console table above
```
Models: `99_runs/scaleworm_v6_blurry/weights/best.pt`,
`99_runs/scaleworm_v5_all/weights/best.pt`, `mushroom.pt`. Dataset built by
`scripts/build_v6_blurry_dataset.py`. Weights/images gitignored (regenerable).
