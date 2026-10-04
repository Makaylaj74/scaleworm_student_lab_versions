"""Automated scaleworm abundance, dw/dt, and the volcano-reloading context (CLEAR window).

Companion to plot_worm_dwdt.py (which uses the MANUAL Monday series). This version derives
abundance from the **scene-sort-free automated harvest** (harvest_mf_2021_2023.csv,
count_corrected = med-of-top-3 ÷0.72, off-scene zeros dropped) over the CLEAR window only
(2021-09-01 .. 2023-08-11) — the one regime where the detector is validated (harvest = 90% of
manual). Added value over the manual figure: ~40 M-F recordings/week vs 8 manual Mondays, so a
less noisy derivative. Honest scope: clear window only; blurry/dense eras are NOT trustworthy
(see the v6 gate) and are deliberately excluded here.

Panels (shared axis):
  A. Automated abundance: weekly points (faint) + monthly mean (line) + SEM band, with the
     manual Monday monthly mean overlaid as the independent validation check.
  B. dw/dt: month-to-month change in mean abundance, gap-aware (consecutive months only).
  C. Volcano reloading: caldera uplift (BOTPT) + weekly seismicity.

The weekly mean is the independent unit (within-week 8-slot recordings view one scene), so
monthly mean/SEM are computed OVER weekly means — mirrors the manual figure's aggregation.

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
HARVEST = REPO / "notebooks/harvest_mf_2021_2023.csv"
MANUAL = REPO / "validation/monday_manual_series/monday_manual_timeseries.csv"
INFL = REPO / "notebooks/axial_inflation_weekly_2021_2024.csv"
SEIS = REPO / "notebooks/axial_seismicity_weekly_2021_2024.csv"
OUT = REPO / "notebooks/figure_worm_dwdt_automated.png"

DET_C = "#0072B2"  # automated detector series (blue)
MAN_C = "#000000"  # manual check (black)
DW_UP = "#009E73"
DW_DN = "#D55E00"
INFL_C = "#0072B2"
SEIS_C = "#CC79A7"
# clear window = the validated detector regime; dw/dt only defensible here
CLEAR0, CLEAR1 = pd.Timestamp("2021-09-01"), pd.Timestamp("2023-08-11")
# geophysics panel spans the full reloading context for visual framing
AXIS0, AXIS1 = pd.Timestamp("2021-09-01"), pd.Timestamp("2025-01-01")


def weekly_mean(dates: pd.Series, vals: pd.Series) -> pd.Series:
    """Mean per ISO week (W-MON start), the independent sampling unit."""
    df = pd.DataFrame(
        {"w": pd.to_datetime(dates).dt.to_period("W-MON").dt.start_time, "v": vals}
    )
    return df.groupby("w")["v"].mean()


def monthly_from_weekly(weekly: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Monthly mean and SEM computed OVER weekly means (weeks = independent unit)."""
    df = pd.DataFrame(
        {"m": weekly.index.to_period("M").to_timestamp(), "v": weekly.values}
    )
    mean = df.groupby("m")["v"].mean()
    sem = df.groupby("m")["v"].agg(
        lambda s: s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else np.nan
    )
    return mean, sem


def gap_aware_dwdt(monthly: pd.Series) -> tuple[list, list]:
    """Month-to-month change, only between consecutive calendar months."""
    months, dwdt, dwdt_x = monthly.index, [], []
    for i in range(1, len(months)):
        gap = (months[i].year - months[i - 1].year) * 12 + (
            months[i].month - months[i - 1].month
        )
        if gap == 1:
            dwdt.append(monthly.iloc[i] - monthly.iloc[i - 1])
            dwdt_x.append(months[i])
    return dwdt, dwdt_x


def main() -> None:
    # --- automated abundance, clear window only ---
    h = pd.read_csv(HARVEST)
    h["dt"] = pd.to_datetime(h["datetime_utc"]).dt.tz_localize(None)
    h = h[(h["count_raw"] > 0) & (h["dt"] >= CLEAR0) & (h["dt"] < CLEAR1)].copy()
    wk = weekly_mean(h["dt"], h["count_corrected"])
    wm, sem = monthly_from_weekly(wk)
    dwdt, dwdt_x = gap_aware_dwdt(wm)

    # --- manual Monday monthly mean (validation overlay, same window) ---
    m = pd.read_csv(MANUAL)
    m["dt"] = pd.to_datetime(m["date"])
    m = m[(m["dt"] >= CLEAR0) & (m["dt"] < CLEAR1)]
    man_wk = weekly_mean(m["dt"], m["mean_worms"])
    man_m, _ = monthly_from_weekly(man_wk)

    infl = pd.read_csv(INFL)
    infl["week_start"] = pd.to_datetime(infl["week_start"], utc=True)
    seis = pd.read_csv(SEIS)
    seis["week_start"] = pd.to_datetime(seis["week_start"], utc=True)

    fig, (axA, axB, axC) = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    axA.axvspan(AXIS0, CLEAR0, color="0.6", alpha=0.10)
    axA.axvspan(CLEAR1, AXIS1, color="0.6", alpha=0.10)

    # A: automated abundance + manual check
    axA.plot(
        wk.index,
        wk.values,
        "o",
        color=DET_C,
        ms=3,
        alpha=0.3,
        label="weekly M-F (automated)",
    )
    axA.plot(
        wm.index, wm.values, "-", color=DET_C, lw=2.4, label="monthly mean (automated)"
    )
    axA.fill_between(wm.index, wm - sem, wm + sem, color=DET_C, alpha=0.18)
    axA.plot(
        man_m.index,
        man_m.values,
        ":",
        color=MAN_C,
        lw=1.8,
        alpha=0.85,
        label="manual Monday (validation check)",
    )
    axA.set_ylabel("Scale-worms\nper Scene-1 frame", fontweight="bold", fontsize=11)
    axA.set_title(
        "A · Automated scaleworm abundance, Mushroom vent — CLEAR window "
        "(M-F detector, recall-corrected)",
        fontsize=12,
        fontweight="bold",
    )
    axA.legend(framealpha=0.9, fontsize=9, loc="upper right")
    axA.set_ylim(0, None)
    axA.grid(alpha=0.3)

    # B: dw/dt
    for ax in (axB,):
        ax.axvspan(AXIS0, CLEAR0, color="0.6", alpha=0.10)
        ax.axvspan(CLEAR1, AXIS1, color="0.6", alpha=0.10)
    colors = [DW_UP if v >= 0 else DW_DN for v in dwdt]
    axB.bar(dwdt_x, dwdt, width=22, color=colors, alpha=0.85)
    axB.axhline(0, color="0.4", lw=1)
    axB.set_ylabel("dw/dt\n(worms · frame⁻¹ · month⁻¹)", fontweight="bold", fontsize=11)
    axB.set_title(
        "B · Automated rate of change of abundance (gap-aware month-to-month, clear window)",
        fontsize=12,
        fontweight="bold",
    )
    axB.grid(alpha=0.3)

    # C: volcano reloading (full context span)
    axC.plot(
        infl["week_start"],
        infl["uplift_cm"],
        "-",
        color=INFL_C,
        lw=2,
        label="caldera uplift (cm)",
    )
    axC.set_ylabel("Caldera uplift (cm)", fontweight="bold", fontsize=11, color=INFL_C)
    axC.tick_params(axis="y", labelcolor=INFL_C)
    axC2 = axC.twinx()
    axC2.plot(
        seis["week_start"],
        seis["eq_count_all"],
        "-",
        color=SEIS_C,
        lw=1.3,
        alpha=0.8,
        label="earthquakes / week",
    )
    axC2.set_ylabel("Earthquakes / week", fontweight="bold", fontsize=11, color=SEIS_C)
    axC2.tick_params(axis="y", labelcolor=SEIS_C)
    axC.set_title(
        "C · Volcano reloading: caldera inflation + seismicity (full-record context)",
        fontsize=12,
        fontweight="bold",
    )
    axC.set_xlabel(
        "Date (shaded = outside validated clear window; dw/dt not shown there)",
        fontweight="bold",
        fontsize=11,
    )
    axC.grid(alpha=0.3)

    span = (AXIS1 - AXIS0).days
    axC.set_xlim(
        AXIS0 - pd.Timedelta(days=span * 0.02), AXIS1 + pd.Timedelta(days=span * 0.02)
    )
    for ax in (axA, axB, axC, axC2):
        for s in ax.spines.values():
            s.set_linewidth(1.0)
    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(
        f"automated clear-window: weeks={len(wk)}  months={len(wm)}  dw/dt points={len(dwdt)}  "
        f"abundance {wm.min():.1f}-{wm.max():.1f}/frame"
    )
    print(
        f"manual check: months={len(man_m)}  abundance {man_m.min():.1f}-{man_m.max():.1f}/frame"
    )


if __name__ == "__main__":
    main()
