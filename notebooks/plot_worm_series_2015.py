"""Plot the 2015 post-eruption scale-worm recovery series (timeseries-figure rubric).

Reads build_worm_series_2015.py outputs. Daily means (several Scene-1 frames per
data-day) with per-frame spread; irregular sampling shown as markers at actual
data-days (no interpolation across gaps). Paper tier, matplotlib, Okabe-Ito.
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

RUN = Path("/home/jovyan/scaleworm-student-lab/notebooks/worm_series_2015_run")
DAILY = RUN / "worm_series_2015_daily.csv"
PERFRAME = RUN / "worm_series_2015_per_frame.csv"
OUT = RUN / "figure_worm_series_2015.png"
C_IDX = "#0072B2"  # Okabe-Ito blue

DISCLOSURE = (
    "AI-generated draft (Claude, Anthropic) — for review. All values "
    "derive from version-controlled scripts and data."
)


def load():
    days, mean, std = [], [], []
    for r in csv.DictReader(DAILY.open()):
        days.append(datetime.strptime(r["date"], "%Y-%m-%d").replace(tzinfo=UTC))
        mean.append(float(r["worm_index_mean"]))
        std.append(float(r["worm_index_std"]))
    pf_d, pf_v = [], []
    for r in csv.DictReader(PERFRAME.open()):
        pf_d.append(datetime.strptime(r["date"], "%Y-%m-%d").replace(tzinfo=UTC))
        pf_v.append(float(r["worm_index"]))
    return (
        np.array(days),
        np.array(mean),
        np.array(std),
        np.array(pf_d),
        np.array(pf_v),
    )


def main() -> None:
    days, mean, std, pf_d, pf_v = load()

    fig, ax = plt.subplots(figsize=(11, 6.6))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.91, bottom=0.30)

    # per-frame points (shows within-day spread + sampling density)
    ax.scatter(
        pf_d,
        pf_v,
        s=16,
        color=C_IDX,
        alpha=0.25,
        zorder=2,
        label="per-frame index (n=120)",
    )
    # daily mean with spread; markers at actual data-days (gaps visible)
    ax.errorbar(
        days,
        mean,
        yerr=std,
        fmt="o-",
        color=C_IDX,
        lw=1.5,
        ms=6,
        capsize=3,
        elinewidth=1.0,
        zorder=3,
        label="daily mean ± 1 SD",
    )

    ax.set_xlabel("Date (2015)", fontsize=13, fontweight="bold")
    ax.set_ylabel(
        "Scale-worm abundance index\n(worms / Scene-1 frame, GT-referenced)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_title(
        "Scale-worm abundance, 2015 post-eruption recovery\n"
        "Axial Seamount — ASHES vent field (Mushroom), CAMHDA301",
        fontsize=14,
        fontweight="bold",
    )

    span = (days.max() - days.min()).days
    pad = span * 0.03
    ax.set_xlim(mdates.date2num(days.min()) - pad, mdates.date2num(days.max()) + pad)
    ax.set_ylim(0, max(pf_v.max(), (mean + std).max()) * 1.08)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")
    ax.tick_params(labelsize=11)
    ax.grid(alpha=0.3, zorder=0)
    for sp in ax.spines.values():
        sp.set_linewidth(1.2)
    ax.legend(fontsize=10, framealpha=0.9, loc="upper right")

    caption = (
        f"{DISCLOSURE}\n"
        "Daily-mean scale-worm abundance index, 16 Jul–28 Dec 2015 (21 Scene-1 "
        "data-days, 120 frames). Counts from a CSRNet density head fine-tuned on "
        "2015–16 hand-counts (OOD 5-fold CV ratio-to-GT 0.61; counts ÷0.61 to a "
        "GT-referenced scale); 33 training frames use held-out CV predictions, 87 "
        "use the deploy model (best_ft2015.pt) — leakage-free. Frames = manual "
        "Scene-1 picks (scene_sorting/full_2015) from OOI CAMHDA301 seafloor video. "
        "Index is a conservative RELATIVE measure (dense OOD; single-rater GT ~2× "
        "uncertain per IRR), not a calibrated absolute count. CAMHD record begins "
        "~11 wk post-eruption (Apr 2015) — no pre-eruption baseline. Irregular "
        "sampling: markers at actual data-days, no interpolation across gaps. "
        "Source: build_worm_series_2015.py / worm_series_2015_daily.csv."
    )
    fig.text(
        0.015,
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
