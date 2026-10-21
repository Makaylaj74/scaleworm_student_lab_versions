"""Plot the 2021-2023 scale-worm series from the CLEAR density head (÷0.96).

Reads build_worm_series_2021_2023_density.py weekly output. Weekly mean ± SEM
over the clear recordings; markers at actual data-weeks (no interpolation across
gaps). Paper tier, matplotlib, Okabe-Ito. Mirrors plot_worm_series_2015/2016.
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

RUN = Path(
    "/home/jovyan/scaleworm-student-lab/notebooks/worm_series_2021_2023_density_run"
)
WEEKLY = RUN / "worm_series_2021_2023_density_weekly.csv"
OUT = RUN / "figure_worm_series_2021_2023_density.png"
C_IDX = "#0072B2"  # Okabe-Ito blue
PROVISIONAL = datetime(2021, 9, 1, tzinfo=UTC)  # manual-validated window starts here

DISCLOSURE = (
    "AI-generated draft (Claude, Anthropic) — for review. All values "
    "derive from version-controlled scripts and data."
)


def load():
    wk, mean, sem, n = [], [], [], []
    for r in csv.DictReader(WEEKLY.open()):
        wk.append(datetime.strptime(r["week"], "%Y-%m-%d").replace(tzinfo=UTC))
        mean.append(float(r["density_mean"]))
        sem.append(float(r["density_sem"]))
        n.append(int(r["n_recordings"]))
    return (np.array(wk), np.array(mean), np.array(sem), np.array(n))


def main() -> None:
    wk, mean, sem, n = load()

    fig, ax = plt.subplots(figsize=(12, 6.6))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.91, bottom=0.30)

    # shade the pre-Sep-2021 provisional (no manual truth, early-2021 undercount)
    prov = wk < PROVISIONAL
    if prov.any():
        ax.axvspan(
            mdates.date2num(wk.min()),
            mdates.date2num(PROVISIONAL),
            color="0.85",
            zorder=0,
            label="pre-Sep-2021 (provisional, unvalidated)",
        )

    ax.errorbar(
        wk,
        mean,
        yerr=sem,
        fmt="o-",
        color=C_IDX,
        lw=1.4,
        ms=4.5,
        capsize=2,
        elinewidth=0.9,
        zorder=3,
        label="weekly index ± 1 SEM",
    )

    ax.set_xlabel("Date", fontsize=13, fontweight="bold")
    ax.set_ylabel(
        "Scale-worm abundance index\n(worms / Scene-1 frame, GT-referenced)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_title(
        "Scale-worm abundance, 2021–2023 (clear density head)\n"
        "Axial Seamount — ASHES vent field (Mushroom), CAMHDA301",
        fontsize=14,
        fontweight="bold",
    )

    span = (wk.max() - wk.min()).days
    pad = span * 0.02
    ax.set_xlim(mdates.date2num(wk.min()) - pad, mdates.date2num(wk.max()) + pad)
    ax.set_ylim(0, (mean + sem).max() * 1.08)
    ax.xaxis.set_major_locator(mdates.MonthLocator((1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")
    ax.tick_params(labelsize=11)
    ax.grid(alpha=0.3, zorder=0)
    for sp in ax.spines.values():
        sp.set_linewidth(1.2)
    ax.legend(fontsize=10, framealpha=0.9, loc="upper left")

    caption = (
        f"{DISCLOSURE}\n"
        f"Weekly scale-worm abundance index, {wk.min():%d %b %Y}–{wk.max():%d %b %Y} "
        f"({len(wk)} data-weeks, {int(n.sum())} clear recordings, "
        f"{int(n.mean())}/wk). Counts from the CSRNet clear density head "
        "(csrnet_run/best.pt; in-distribution held-out ratio-to-GT 0.96; counts "
        "÷0.96 to a GT-referenced scale). Frame per recording = the v8+SAHI "
        "coarse-max best_time_s (validated ~100% of manual Scene-1 in aggregate); "
        "the counter is the density head, the frame choice is reused. Index is a "
        "conservative RELATIVE measure (single-rater GT ~2× uncertain per IRR), "
        "not a calibrated absolute count. Shaded pre-Sep-2021 window has no manual "
        "truth and a known early-2021 undercount — provisional. Markers at actual "
        "data-weeks, no interpolation across gaps. Source: "
        "build_worm_series_2021_2023_density.py / "
        "worm_series_2021_2023_density_weekly.csv."
    )
    fig.text(
        0.012,
        0.005,
        caption,
        fontsize=7.3,
        family="monospace",
        va="bottom",
        ha="left",
        wrap=True,
    )

    fig.savefig(OUT, dpi=300)
    print(f"figure -> {OUT}")


if __name__ == "__main__":
    main()
