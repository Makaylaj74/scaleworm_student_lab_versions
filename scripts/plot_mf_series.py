"""First automated Monday-Friday scaleworm abundance series (clear window 2021-2023).

Combines the MANUAL Monday counts (human ground truth) with the DETECTOR-harvested Tue-Fri
counts (max-over-window, med-of-top-3, recall-corrected ÷0.72 to human scale; off-scene
zero-count recordings dropped as 'no usable view', matching the manual convention).

The recall correction puts detector Tue-Fri on the same scale as manual Monday (validated:
Tue-Fri harvest = 90% of Monday manual over the overlap). Weekly means combine all available
M-F recordings; the figure distinguishes the two sources and shows M-F extends the sampling
from 8 recordings/week (Monday only) to ~40/week.

matplotlib, Okabe-Ito, 300 DPI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
MON = REPO / "validation/monday_manual_series/frame_manifest.csv"
HARVEST = REPO / "notebooks/harvest_tierA_clear_2021_2023.csv"
OUT = REPO / "notebooks/figure_mf_abundance_clear_2021_2023.png"
START, END = pd.Timestamp("2021-09-01"), pd.Timestamp("2023-08-15")

MON_C = "#000000"
TF_C = "#0072B2"
WK_C = "#D55E00"


def main() -> None:
    # Monday manual per-recording (clear window)
    m = pd.read_csv(MON)
    m = m[m.worm_count.astype(str).str.strip().replace("nan", "") != ""].copy()
    m["dt"] = pd.to_datetime(m.datetime_utc)
    m = m[(m.dt >= START) & (m.dt <= END)]
    m["count"] = m.worm_count.astype(float)
    m["src"] = "Mon (manual)"

    # Tue-Fri detector harvest, recall-corrected, drop off-scene zeros
    h = pd.read_csv(HARVEST)
    h["dt"] = pd.to_datetime(h.datetime_utc).dt.tz_localize(None)
    h = h[(h.dt >= START) & (h.dt <= END) & (h.count_raw > 0)].copy()
    h["count"] = h.count_corrected
    h["src"] = "Tue-Fri (detector)"

    allrec = pd.concat([m[["dt", "count", "src"]], h[["dt", "count", "src"]]], ignore_index=True)
    allrec["week"] = allrec.dt.dt.to_period("W-MON").dt.start_time

    # weekly mean over ALL M-F recordings + SEM
    wk = allrec.groupby("week")["count"].agg(["mean", "sem", "count"]).rename(columns={"count": "n"})
    # Monday-only weekly (for the overlay comparison)
    mon_wk = m.assign(week=m.dt.dt.to_period("W-MON").dt.start_time).groupby("week")["count"].mean()

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.scatter(m.dt, m["count"], s=10, color=MON_C, alpha=0.35, label="Mon recordings (manual)")
    ax.scatter(h.dt, h["count"], s=10, color=TF_C, alpha=0.25, label="Tue-Fri recordings (detector)")
    ax.plot(wk.index, wk["mean"], "-", color=WK_C, lw=2.4, label="weekly M-F mean (all 5 days)")
    ax.fill_between(wk.index, wk["mean"] - wk["sem"], wk["mean"] + wk["sem"], color=WK_C, alpha=0.2)
    ax.plot(mon_wk.index, mon_wk.values, "--", color=MON_C, lw=1.3, alpha=0.7,
            label="Monday-only weekly mean (for comparison)")

    ax.set_xlabel("Date (weekly)", fontweight="bold", fontsize=12)
    ax.set_ylabel("Scale-worms per Scene-1 frame\n(detector recall-corrected)", fontweight="bold", fontsize=12)
    ax.set_title("First automated Monday-Friday scaleworm abundance — clear window 2021-2023\n"
                 "Manual Monday + detector-harvested Tue-Fri (scene-sort-free, ~40 recordings/week)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, None)
    span = (END - START).days
    ax.set_xlim(START - pd.Timedelta(days=span * 0.02), END + pd.Timedelta(days=span * 0.02))
    ax.grid(alpha=0.3)
    ax.legend(framealpha=0.9, fontsize=9, loc="upper right")
    for s in ax.spines.values():
        s.set_linewidth(1.0)
    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"weeks {len(wk)} | M-F recordings {len(allrec)} (Mon {len(m)} + Tue-Fri {len(h)}) "
          f"| weekly M-F mean {wk['mean'].mean():.1f}/frame")


if __name__ == "__main__":
    main()
