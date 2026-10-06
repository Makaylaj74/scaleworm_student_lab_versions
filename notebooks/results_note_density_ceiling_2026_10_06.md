*AI-generated draft (Claude, Anthropic) — for review. All numbers derive from version-controlled scripts, datasets, and gate CSVs in the `scaleworm-student-lab` repository (branch `image-quality-survey`) and were re-verified against disk on 2026-10-06.*

# Results note: the dense-density counting ceiling, and what it means for population–geophysics trends

**Scope.** Documents a measurement limitation of the automated scale-worm counter at high worm density, the evidence that it is irreducible by counting or correction, and its specific consequences for comparing population to geophysical parameters.

---

## 1. The limitation, stated plainly

At high worm density — the population peaks, e.g. late 2022 / early 2023 — the automated detector **saturates and under-counts**. On the dense frames (manual > 15 worms) the detector recovers only ~79% of the hand-counted worms, and on the very densest frames (manual 30–47) it reads roughly half. Manual box counts on human-picked Scene-1 frames show peaks up to ~47 worms/frame; the automated index caps near ~20–26. **The automated index is therefore a conservative lower bound at the peaks**, while it is accurate-to-over-counting at low and moderate density.

## 2. Why it cannot be fixed by counting or correction (the evidence)

The ceiling was attacked from every tractable angle; all converge:

1. **Detector improvements plateau.** Tiled (SAHI) inference, a tile-*trained* detector (v8), higher-resolution inference, and a label-scaling learning curve each lifted dense recall by only +1–3 points. The detector cannot resolve individuals once they crowd and overlap.
2. **Count-based correction is information-theoretically impossible.** A monotonic calibration (automated → manual, isotonic regression, held-out-validated) did **not** lift the peaks (held-out dense ratio 82% → 80%). Root cause: the automated count is strongly **many-to-one** — for recordings the detector counted at 18–22, the *true* manual count ranges from **1 to 43**. A genuine dense frame, a moderate frame, and a false-positive-heavy frame all collapse to the same automated number, so no function of the count alone can separate them. (This is the same reason an earlier global power-law correction failed.)
3. **The ground truth is itself uncertain at high density.** Independent human counts disagree by roughly 2× on dense frames (inter-rater reliability: ICC 0.38 on a low-vs-peak packet). At the densest peaks, "the exact number of worms" is not a well-defined quantity even for trained human counters.

**Conclusion:** individual-worm counting saturates at high density, for machines *and* humans. This is a property of the measurement, not a tuning failure.

## 3. What the automated index *does* preserve

Despite the peak saturation:
- **Timing is preserved** — the weeks/months in which the population rises and falls are captured correctly.
- **Direction of change (the sign of dw/dt) is preserved** — the index tracks increases and decreases faithfully.
- **Aggregate level is unbiased** away from the extreme peaks (overall ratio ~100% of manual, because low-density over-counting from max-frame selection offsets dense under-counting).

So the index is a sound **relative / temporal** record; it is the peak *magnitudes* specifically that are under-read.

## 4. Consequence for population vs geophysical parameters (the operational point)

Because the population peaks are exactly the events one would correlate with geophysical forcing, the saturation matters — but it matters **only for magnitude**, and there is a clean way to handle it:

- **Trend / timing / lead–lag correlations** (does population change track or lead a geophysical signal, and when?) are **robust** in both the manual and automated series — timing and the sign of change survive the ceiling. These analyses can use the automated Monday–Friday index directly.
- **Magnitude correlations at the peaks** (how *large* a population response accompanies a given geophysical magnitude) must **not** use the automated index at the peaks, which is a lower bound there. For peak magnitude, use the **manual Monday box counts**, which do *not* saturate (humans counted the dense frames directly, up to ~47), bounded only by the ~2× human uncertainty. The manual Monday series is weekly and spans 2021–2024 — adequate for weekly/monthly geophysical comparison.
- **Practical recommendation for the geophysics comparison:** anchor peak magnitudes to the manual Monday series; use the automated Mon–Fri index for temporal coverage, trend, and timing. State the peak-magnitude caveat wherever absolute population magnitude enters a correlation.

## 5. The hard case: dense peaks in eras with no manual counts (e.g. 2015–2016)

The manual-Monday anchor in §4 only covers 2021–2024. The eras that most motivate the study — notably the **2015–2016 post-eruption colonization**, a population boom — have **no manual series**, so the anchor provides nothing there, and the automated detector both **saturates** (dense) and is **out-of-distribution** (2015 is a dense tubeworm-mound scene unlike the 2021–24 tower; the detector recovers only ~28% there, so it is unreliable even for timing). For these eras the automated pipeline cannot stand alone.

This does **not** require hand-counting every frame (~20k) — that would defeat the automation. The workable path per unmanualled dense era is:

1. **A sparse manual anchor** — count a *representative subset* (tens of frames per month, as was done for the 2017 salvage), enough to establish the trend and pin magnitude, not the whole archive.
2. **An era-appropriate dense metric** — for a dense colonization mat, *percent-cover / density* is likely the correct measurable: it does not saturate the way individual counting does, and humans agree on cover far better than on exact dense counts. Whether an early-era population is measured as a **count** or a **cover/density** is an advisor-level "count target" decision; **2015–2016 is currently blocked on exactly this decision**, which this limitation shows to be load-bearing for the core science, not a detail.
3. **Optionally, a density-estimation / segmentation model** trained on those sparse era labels to scale the anchor across the era. Crowd-counting-style density-map regression reads the image rather than detecting individuals and is the standard tool for the "too many to separate" regime; it remains bounded by the ~2× human ground-truth uncertainty and would itself need a few hundred era-specific labels (not 20k).

**Summary of the measurement strategy by regime:** 2021–2024 clear window — automated index (trend + timing) with peak magnitude anchored to manual Monday; dense unmanualled eras (2015–2016, and any future dense era) — sparse manual anchor + an advisor-chosen dense metric (count or cover), optionally scaled by a density model. The automated clear-window index is valid for its window; it is not a substitute for era-specific anchoring where the detector saturates or is out-of-distribution.

---

**Provenance.** Automated counts: `harvest_clear_v8sahi_2021_2023.csv` (v8_tiled detector + SAHI, max over a 60 s grid). Manual truth: `validation/monday_manual_series`. Calibration test: `scripts/build_count_calibration.py`, `validation/count_calibration.json`. Dense-miss mechanism: `scripts/diag_dense_miss_mechanism.py`. Inter-rater reliability: `scripts/irr_counts_agreement.py` on `validation/irr_round_2`.
