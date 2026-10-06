"""Does the automated frame-finder pick a consistent pan-time day to day?

For each recording slot (hour of day), sorts recordings by date and measures the change in the
picked frame time (best_time_s) from one day to the next. If the camera panned identically each day,
Scene-1 would fall at the same pan-time and these day-to-day changes would cluster near 0.

Output: notebooks/figure_frame_time_consistency.png (300 dpi).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/home/jovyan/scaleworm-student-lab")
C_BAR, C_LINE = "#0072B2", "#D55E00"


def deltas_and_times():
    rows = []
    for f in ("harvest_clear_v8sahi_2021_2023.csv", "harvest_janaug2021_v8sahi.csv"):
        p = REPO / "notebooks" / f
        if p.exists():
            for r in csv.DictReader(p.open()):
                dt = datetime.strptime(  # noqa: DTZ007 (frame ids are UTC; tz not needed here)
                    r["frame_id"].split("-")[1][:15], "%Y%m%dT%H%M%S")
                rows.append((dt, float(r["best_time_s"])))
    by = defaultdict(list)
    for dt, b in rows:
        by[dt.hour].append((dt, b))
    d = []
    for recs in by.values():
        recs.sort()
        d += [abs(recs[i][1] - recs[i - 1][1]) for i in range(1, len(recs))]
    return np.array(d), np.array([b for _, b in rows])


def main() -> None:
    d, times = deltas_and_times()
    within60 = (d <= 60).mean()
    within120 = (d <= 120).mean()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # Panel A: day-to-day change in picked frame-time
    bins = np.arange(0, 600, 60)
    axA.hist(d, bins=bins, color=C_BAR, alpha=0.85, edgecolor="white")
    axA.axvline(60, color=C_LINE, lw=2, ls="--", label="60 s")
    axA.set_xlabel("Change in picked frame-time vs previous day (s)", fontweight="bold", fontsize=11)
    axA.set_ylabel("Number of recordings", fontweight="bold", fontsize=11)
    axA.set_title("A  Day-to-day consistency of the picked frame", fontsize=11, loc="left", fontweight="bold")
    axA.grid(alpha=0.3, axis="y")
    axA.text(0.52, 0.95, f"within 60 s: {within60:.0%}\nwithin 120 s: {within120:.0%}\n"
             f"median change: {np.median(d):.0f} s",
             transform=axA.transAxes, va="top", ha="left", fontsize=10,
             bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.7"})
    axA.legend(loc="upper right", framealpha=0.9, fontsize=9)

    # Panel B: how spread the picked times are overall
    axB.hist(times, bins=np.arange(60, 690, 60), color="#009E73", alpha=0.85, edgecolor="white")
    axB.set_xlabel("Picked frame-time within the pan (s)", fontweight="bold", fontsize=11)
    axB.set_ylabel("Number of recordings", fontweight="bold", fontsize=11)
    axB.set_title(f"B  Where in the pan Scene-1 is found (median {np.median(times):.0f} s)",
                  fontsize=11, loc="left", fontweight="bold")
    axB.grid(alpha=0.3, axis="y")

    for ax in (axA, axB):
        for sp in ax.spines.values():
            sp.set_linewidth(1.0)

    fig.suptitle("Automated frame-finder: temporal consistency of the picked Scene-1 frame",
                 fontsize=12.5, fontweight="bold")
    cap = ("AI-generated caption (Claude, Anthropic) — Consistency of the detector-based frame-finder "
           "(v8+SAHI, max over a 60 s grid) on OOI RS03ASHS CAMHDA301, all weekdays 2021-2023. (A) "
           "Day-to-day change in the picked frame-time (best_time_s) between consecutive recordings in the "
           "same daily slot; dashed line = 60 s (the grid spacing). Only ~49% of consecutive days land "
           "within 60 s and ~66% within 120 s (median change 120 s) — the pan timing is NOT consistent "
           "day to day, which is why a fixed-time extraction fails and the max-over-grid finder is needed. "
           "(B) Distribution of where in the pan Scene-1 is found (90-630 s grid), spread across the whole "
           "window (median 330 s). Source: harvest_clear_v8sahi_2021_2023.csv + harvest_janaug2021_v8sahi.csv.")
    fig.subplots_adjust(bottom=0.30, top=0.86, wspace=0.26)
    fig.text(0.5, 0.015, cap, ha="center", va="bottom", wrap=True, fontsize=7.5)
    fig.savefig(REPO / "notebooks/figure_frame_time_consistency.png", dpi=300, bbox_inches="tight")
    print(f"wrote figure_frame_time_consistency.png  within60={within60:.0%} within120={within120:.0%} "
          f"median_delta={np.median(d):.0f}s")


if __name__ == "__main__":
    main()
