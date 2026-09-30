"""Abundance time series from the retrained detector (v5_all, per-regime conf).

Panel A: monthly-mean scale-worm abundance 2021-09..2024-12 — manual box-corrected
ground truth vs v5_all AI counts (clear conf 0.40 / blurry 0.25). Blur-onset (camera
swap ~2023-08-11) marked; the post-swap span is where the ORIGINAL detector collapsed
to ~0 and v5 now tracks truth.

Panel B: out-of-sample check — AI vs manual per frame, restricted to frames that are
manually counted but were EXCLUDED (day-level) from v5 training. This is the leakage-
free demonstration that the tracking in Panel A is not memorization.

matplotlib, Okabe-Ito, 300 DPI. Run under any env with pandas+matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "validation/monday_manual_series/frame_manifest.csv"
AI_FRAME = REPO / "notebooks/ai_counts_v5_per_frame.csv"
OUT = REPO / "notebooks/figure_worms_over_time_v5.png"

BLUR_ONSET = pd.Timestamp("2023-08-11")
START = pd.Timestamp("2021-09-01")  # manual coverage begins; drop 2019-20 dead-end era

# Okabe-Ito
MANUAL_C = "#000000"
AI_C = "#0072B2"
BLUR_C = "#D55E00"


def main() -> None:
    man = pd.read_csv(MANIFEST)[["frame_id", "datetime_utc", "worm_count"]]
    man["dt"] = pd.to_datetime(man["datetime_utc"])
    man = man.dropna(subset=["worm_count"])
    man["worm_count"] = man["worm_count"].astype(float).astype(int)

    ai = pd.read_csv(AI_FRAME)[["frame_id", "ai_count", "in_training"]]
    df = man.merge(ai, on="frame_id", how="inner")
    df = df[df["dt"] >= START].copy()
    df["month"] = df["dt"].dt.to_period("M").dt.to_timestamp()

    # --- monthly means + SEM for both series ---
    def monthly(col):
        g = df.groupby("month")[col]
        m = g.mean()
        sem = g.std(ddof=1) / np.sqrt(g.count())
        return m, sem

    man_m, man_sem = monthly("worm_count")
    ai_m, ai_sem = monthly("ai_count")

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [2.4, 1]}
    )

    # ---- Panel A: time series ----
    axA.axvspan(BLUR_ONSET, df["month"].max() + pd.Timedelta(days=20),
                color=BLUR_C, alpha=0.07)
    axA.axvline(BLUR_ONSET, color=BLUR_C, ls="--", lw=1.6)
    axA.text(BLUR_ONSET, axA.get_ylim()[1], " camera swap →\n blurry footage",
             color=BLUR_C, fontsize=9, va="top", fontweight="bold")
    axA.plot(man_m.index, man_m.values, "-o", color=MANUAL_C, lw=1.8, ms=4,
             label="Manual (box-corrected truth)")
    axA.fill_between(man_m.index, man_m - man_sem, man_m + man_sem,
                     color=MANUAL_C, alpha=0.15)
    axA.plot(ai_m.index, ai_m.values, "-s", color=AI_C, lw=1.8, ms=4,
             label="v5_all AI (conf 0.40 clear / 0.25 blurry)")
    axA.fill_between(ai_m.index, ai_m - ai_sem, ai_m + ai_sem, color=AI_C, alpha=0.15)
    axA.set_xlabel("Date (monthly mean)", fontweight="bold", fontsize=12)
    axA.set_ylabel("Scale-worms per Scene-1 frame", fontweight="bold", fontsize=12)
    axA.set_title(
        "A · Scaleworm abundance at Mushroom vent, 2021–2024\n"
        "v5 recovers the signal (blurry 94%); compresses the dense 2022 peak (see B)",
        fontsize=12, fontweight="bold")
    axA.set_ylim(0, None)
    span = (df["month"].max() - df["month"].min()).days
    pad = pd.Timedelta(days=span * 0.03)
    axA.set_xlim(df["month"].min() - pad, df["month"].max() + pad)
    axA.grid(alpha=0.3)
    axA.legend(framealpha=0.9, loc="upper right", fontsize=10)
    for s in axA.spines.values():
        s.set_linewidth(1.0)

    # ---- Panel B: out-of-sample AI vs manual ----
    oos = df[~df["in_training"].astype(bool)]
    x, y = oos["worm_count"].values, oos["ai_count"].values
    r = np.corrcoef(x, y)[0, 1]
    lim = max(x.max(), y.max()) + 2
    axB.plot([0, lim], [0, lim], color="0.6", ls="--", lw=1.2, label="1:1")
    clear_m = oos["dt"] < BLUR_ONSET
    axB.scatter(x[clear_m], y[clear_m], c=AI_C, s=22, alpha=0.6, label="clear")
    axB.scatter(x[~clear_m], y[~clear_m], c=BLUR_C, s=22, alpha=0.7, label="blurry")
    axB.set_xlabel("Manual count (worms)", fontweight="bold", fontsize=12)
    axB.set_ylabel("v5_all AI count (worms)", fontweight="bold", fontsize=12)
    axB.set_title(f"B · Out-of-sample agreement\n"
                  f"n={len(oos)} held-out frames · r={r:.2f}",
                  fontsize=12, fontweight="bold")
    axB.set_xlim(0, lim)
    axB.set_ylim(0, lim)
    axB.grid(alpha=0.3)
    axB.legend(framealpha=0.9, loc="upper left", fontsize=9)
    for s in axB.spines.values():
        s.set_linewidth(1.0)

    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"Panel A: {len(df)} frames, {df['month'].nunique()} months "
          f"{df['month'].min().date()}..{df['month'].max().date()}")
    print(f"Panel B: out-of-sample n={len(oos)} (manually counted, NOT in v5 training), "
          f"r={r:.3f}, AI/manual total ratio={y.sum() / x.sum():.1%}")


if __name__ == "__main__":
    main()
