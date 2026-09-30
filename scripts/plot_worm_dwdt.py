"""Scaleworm abundance, its rate of change (dw/dt), and the volcano-reloading context.

Uses the MANUAL box-corrected Monday series (validated ground truth, 2021-2024) — NOT the
detector — so this figure is independent of the detector's density limitations.

Panels (shared 2021-2024 monthly axis):
  A. Worm abundance: weekly points (faint) + monthly mean (line) + SEM band.
  B. dw/dt: month-to-month change in mean abundance (worms/frame per month), gap-aware
     (only between consecutive months that both have data).
  C. Volcano reloading: caldera uplift (BOTPT) + weekly seismicity — the co-accelerating
     pre-eruption signals the worms live on.

Aggregation = monthly means (the abundance signal is slow; weekly is gappy). This is VISUAL
context, NOT a correlation claim (worm<->geophysics stats are a separate step).

matplotlib, Okabe-Ito, 300 DPI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
WORM = REPO / "validation/monday_manual_series/monday_manual_timeseries.csv"
INFL = REPO / "notebooks/axial_inflation_weekly_2021_2024.csv"
SEIS = REPO / "notebooks/axial_seismicity_weekly_2021_2024.csv"
OUT = REPO / "notebooks/figure_worm_dwdt_vs_geophysics.png"

WORM_C = "#000000"
DW_UP = "#009E73"
DW_DN = "#D55E00"
INFL_C = "#0072B2"
SEIS_C = "#CC79A7"
START, END = pd.Timestamp("2021-09-01"), pd.Timestamp("2025-01-01")


def monthly(dates, vals):
    df = pd.DataFrame({"m": pd.to_datetime(dates).dt.to_period("M").dt.to_timestamp(), "v": vals})
    return df.groupby("m")["v"].mean()


def main() -> None:
    w = pd.read_csv(WORM)
    w["date"] = pd.to_datetime(w["date"])
    w = w[(w["date"] >= START) & (w["date"] < END)]
    wm = monthly(w["date"], w["mean_worms"])  # monthly mean abundance

    # monthly SEM from weekly spread within month
    g = pd.DataFrame({"m": w["date"].dt.to_period("M").dt.to_timestamp(), "v": w["mean_worms"]})
    sem = g.groupby("m")["v"].agg(lambda s: s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else np.nan)

    # gap-aware dw/dt: change per month only between consecutive calendar months
    months = wm.index
    dwdt, dwdt_x = [], []
    for i in range(1, len(months)):
        gap = (months[i].year - months[i - 1].year) * 12 + (months[i].month - months[i - 1].month)
        if gap == 1:  # consecutive months only
            dwdt.append(wm.iloc[i] - wm.iloc[i - 1])
            dwdt_x.append(months[i])

    infl = pd.read_csv(INFL); infl["week_start"] = pd.to_datetime(infl["week_start"], utc=True)
    seis = pd.read_csv(SEIS); seis["week_start"] = pd.to_datetime(seis["week_start"], utc=True)

    fig, (axA, axB, axC) = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    # A: abundance
    axA.plot(w["date"], w["mean_worms"], "o", color=WORM_C, ms=3, alpha=0.25, label="weekly Monday")
    axA.plot(wm.index, wm.values, "-", color=WORM_C, lw=2.2, label="monthly mean")
    axA.fill_between(wm.index, wm - sem, wm + sem, color=WORM_C, alpha=0.15)
    axA.set_ylabel("Scale-worms\nper Scene-1 frame", fontweight="bold", fontsize=11)
    axA.set_title("A · Scaleworm abundance at Mushroom vent (manual counts, 2021–2024)",
                  fontsize=12, fontweight="bold")
    axA.legend(framealpha=0.9, fontsize=9, loc="upper right")
    axA.set_ylim(0, None); axA.grid(alpha=0.3)

    # B: dw/dt
    colors = [DW_UP if v >= 0 else DW_DN for v in dwdt]
    axB.bar(dwdt_x, dwdt, width=22, color=colors, alpha=0.85)
    axB.axhline(0, color="0.4", lw=1)
    axB.set_ylabel("dw/dt\n(worms · frame⁻¹ · month⁻¹)", fontweight="bold", fontsize=11)
    axB.set_title("B · Rate of change of abundance (gap-aware month-to-month)",
                  fontsize=12, fontweight="bold")
    axB.grid(alpha=0.3)

    # C: volcano reloading
    axC.plot(infl["week_start"], infl["uplift_cm"], "-", color=INFL_C, lw=2, label="caldera uplift (cm)")
    axC.set_ylabel("Caldera uplift (cm)", fontweight="bold", fontsize=11, color=INFL_C)
    axC.tick_params(axis="y", labelcolor=INFL_C)
    axC2 = axC.twinx()
    axC2.plot(seis["week_start"], seis["eq_count_all"], "-", color=SEIS_C, lw=1.3, alpha=0.8,
              label="earthquakes / week")
    axC2.set_ylabel("Earthquakes / week", fontweight="bold", fontsize=11, color=SEIS_C)
    axC2.tick_params(axis="y", labelcolor=SEIS_C)
    axC.set_title("C · Volcano reloading: caldera inflation + seismicity co-accelerate",
                  fontsize=12, fontweight="bold")
    axC.set_xlabel("Date (monthly means panel A–B; weekly panel C)", fontweight="bold", fontsize=11)
    axC.grid(alpha=0.3)

    span = (END - START).days
    axC.set_xlim(START - pd.Timedelta(days=span * 0.02), END + pd.Timedelta(days=span * 0.02))
    for ax in (axA, axB, axC, axC2):
        for s in ax.spines.values():
            s.set_linewidth(1.0)
    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"worm months={len(wm)}  dw/dt points={len(dwdt)}  "
          f"abundance {wm.min():.1f}-{wm.max():.1f}/frame")


if __name__ == "__main__":
    main()
