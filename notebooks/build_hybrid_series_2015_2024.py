"""Stitch the method-consistent scale-worm hybrid: 2015 + 2016 + 2021-2023.

All three blocks are counted by the CSRNet density head (one counter, no density/
box-detector seam), each put on a GT-referenced scale by dividing by its model's
ratio-to-GT:
  * 2015, 2016  -> fine-tuned model best_ft2015.pt  (OOD CV 0.61x; counts ÷0.61)
  * 2021-2023   -> clear model      best.pt         (in-dist 0.96x; counts ÷0.96)

The 2017-2020 interval is a DEAD-END gap (the 2017-20 CAMHD record lacks the
canonical Scene-1 community view; see memory) -- shown as an explicit axis break,
never interpolated.

Honest-comparison caveats, surfaced in the caption:
  (1) TWO MODELS / TWO SCALES across the break. Within-block trends are robust;
      cross-era ABSOLUTE magnitude ("recovered to level X") is the fragile claim
      -- the 0.61x vs 0.96x calibrations differ and GT is ~2x uncertain (IRR).
  (2) AGGREGATION ASYMMETRY. 2015/2016 points are ~1 Scene-1 frame each; 2021-23
      points are weekly MEANS over ~39 recordings -- so the early blocks are
      noisier with higher single-frame peaks, the late block is smoothed. Compare
      trends, not point-to-point magnitude.

Reads existing CSVs only (no re-counting). Emits a combined provenance CSV and a
two-panel broken-axis figure (Okabe-Ito, coloured by model).
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

NB = Path("/home/jovyan/scaleworm-student-lab/notebooks")
D2015 = NB / "worm_series_2015_run/worm_series_2015_daily.csv"
D2016 = NB / "worm_series_2016_run/worm_series_2016_daily.csv"
W2123 = NB / "worm_series_2021_2023_density_run/worm_series_2021_2023_density_weekly.csv"
OUT_CSV = NB / "hybrid_series_2015_2024.csv"
OUT_FIG = NB / "figure_hybrid_series_2015_2024.png"

C_FT = "#0072B2"  # Okabe-Ito blue   -- fine-tuned model (2015-2016, ÷0.61)
C_CLR = "#D55E00"  # Okabe-Ito vermillion -- clear model (2021-2023, ÷0.96)
PROVISIONAL = datetime(2021, 9, 1, tzinfo=UTC)

DISCLOSURE = (
    "AI-generated draft (Claude, Anthropic) — for review. All values "
    "derive from version-controlled scripts and data."
)


def _dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=UTC)


def load_daily(fn: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d, m, sd = [], [], []
    with fn.open() as fh:
        for r in csv.DictReader(fh):
            d.append(_dt(r["date"]))
            m.append(float(r["worm_index_mean"]))
            sd.append(float(r["worm_index_std"]))
    return np.array(d), np.array(m), np.array(sd)


def load_weekly(fn: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d, m, se = [], [], []
    with fn.open() as fh:
        for r in csv.DictReader(fh):
            d.append(_dt(r["week"]))
            m.append(float(r["density_mean"]))
            se.append(float(r["density_sem"]))
    return np.array(d), np.array(m), np.array(se)


def write_combined(blocks: list[dict]) -> None:
    with OUT_CSV.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["date", "era", "model", "ratio_to_gt", "worm_index",
                     "spread", "spread_kind"])
        for b in blocks:
            for dt, v, s in zip(b["t"], b["mean"], b["spread"], strict=True):
                wr.writerow([dt.date().isoformat(), b["era"], b["model"],
                             b["ratio"], f"{v:.2f}", f"{s:.2f}", b["spread_kind"]])
    print(f"combined -> {OUT_CSV}")


def main() -> None:
    t15, m15, s15 = load_daily(D2015)
    t16, m16, s16 = load_daily(D2016)
    t23, m23, s23 = load_weekly(W2123)

    blocks = [
        {"era": "2015", "model": "best_ft2015.pt", "ratio": 0.61,
         "t": t15, "mean": m15, "spread": s15, "spread_kind": "daily_SD"},
        {"era": "2016", "model": "best_ft2015.pt", "ratio": 0.61,
         "t": t16, "mean": m16, "spread": s16, "spread_kind": "daily_SD"},
        {"era": "2021-2023", "model": "best.pt", "ratio": 0.96,
         "t": t23, "mean": m23, "spread": s23, "spread_kind": "weekly_SEM"},
    ]
    write_combined(blocks)

    ymax = max(
        (m15 + s15).max(), (m16 + s16).max(), (m23 + s23).max()
    ) * 1.08

    # left panel = 2015-2016 (FT model); right = 2021-2023 (clear model)
    # width ratios ~ time span of each block
    span_l = (t16.max() - t15.min()).days
    span_r = (t23.max() - t23.min()).days
    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(13, 6.6), sharey=True,
        gridspec_kw={"width_ratios": [span_l, span_r], "wspace": 0.04},
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.80, bottom=0.30)

    # --- left: 2015-2016 FT model ---
    axl.errorbar(t15, m15, yerr=s15, fmt="o-", color=C_FT, lw=1.3, ms=5,
                 capsize=2, elinewidth=0.9, zorder=3)
    axl.errorbar(t16, m16, yerr=s16, fmt="s-", color=C_FT, lw=1.3, ms=4.5,
                 capsize=2, elinewidth=0.9, zorder=3, mfc="white")
    axl.set_xlim(mdates.date2num(t15.min()) - 15,
                 mdates.date2num(t16.max()) + 15)
    axl.xaxis.set_major_locator(mdates.MonthLocator((1, 7)))
    axl.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axl.set_title("Post-eruption recovery (FT model ÷0.61)",
                  fontsize=10, fontweight="bold", pad=4)

    # --- right: 2021-2023 clear model ---
    prov = t23 < PROVISIONAL
    if prov.any():
        axr.axvspan(mdates.date2num(t23.min()), mdates.date2num(PROVISIONAL),
                    color="0.88", zorder=0)
    axr.errorbar(t23, m23, yerr=s23, fmt="o-", color=C_CLR, lw=1.2, ms=3.5,
                 capsize=1.5, elinewidth=0.7, zorder=3)
    axr.set_xlim(mdates.date2num(t23.min()) - 20,
                 mdates.date2num(t23.max()) + 20)
    axr.xaxis.set_major_locator(mdates.MonthLocator((1, 7)))
    axr.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axr.set_title("Mature vent community (clear model ÷0.96)",
                  fontsize=10, fontweight="bold", pad=4)

    axl.set_ylim(0, ymax)
    for ax in (axl, axr):
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(30)
            lbl.set_ha("right")
        ax.tick_params(labelsize=10)
        ax.grid(alpha=0.3, zorder=0)
        for sp in ax.spines.values():
            sp.set_linewidth(1.2)

    # axis break (hide inner spines + diagonal marks)
    axl.spines["right"].set_visible(False)
    axr.spines["left"].set_visible(False)
    axr.tick_params(left=False)
    dkw = {"marker": [(-1, -0.6), (1, 0.6)], "markersize": 10,
           "linestyle": "none", "color": "k", "mec": "k", "mew": 1.2,
           "clip_on": False}
    axl.plot([1, 1], [0, 1], transform=axl.transAxes, **dkw)
    axr.plot([0, 0], [0, 1], transform=axr.transAxes, **dkw)

    axl.set_ylabel("Scale-worm abundance index\n(worms / Scene-1 frame, "
                   "GT-referenced)", fontsize=12, fontweight="bold")
    fig.text(0.54, 0.205, "Date", fontsize=12, fontweight="bold", ha="center")
    fig.suptitle(
        "Scale-worm abundance, 2015–2023 hybrid density series\n"
        "Axial Seamount — ASHES vent field (Mushroom), CAMHDA301",
        fontsize=14, fontweight="bold", y=0.985,
    )

    # gap label: vertical, centred in the axis-break gutter (computed post-layout)
    fig.canvas.draw()
    x_gap = (axl.get_position().x1 + axr.get_position().x0) / 2
    fig.text(x_gap, 0.55, "2017–2020  ·  no Scene-1 record  (dead-end gap)",
             rotation=90, ha="center", va="center", fontsize=8,
             style="italic", color="0.45")

    # legend (model/era + marker key)
    handles = [
        Line2D([0], [0], color=C_FT, marker="o", lw=1.3, label="2015 daily (FT ÷0.61)"),
        Line2D([0], [0], color=C_FT, marker="s", mfc="white", lw=1.3,
               label="2016 weekly (FT ÷0.61)"),
        Line2D([0], [0], color=C_CLR, marker="o", lw=1.2,
               label="2021–23 weekly mean (clear ÷0.96)"),
        Patch(color="0.88", label="pre-Sep-2021 provisional"),
    ]
    axr.legend(handles=handles, fontsize=8.5, framealpha=0.9, loc="upper left")

    caption = (
        f"{DISCLOSURE}\n"
        "Hybrid scale-worm abundance index, 2015–2023, all counted by the CSRNet "
        "density head (one counter — no density/box-detector seam). 2015 (21 "
        "data-days) & 2016 (45 weekly frames) use the 2015–16 fine-tuned model "
        "(best_ft2015.pt, OOD CV ratio-to-GT 0.61; ÷0.61); 2021–2023 (137 "
        "data-weeks, ~39 recordings/wk) uses the clear model (best.pt, in-dist "
        "0.96; ÷0.96). The 2017–2020 record lacks the canonical Scene-1 community "
        "view (dead-end) → explicit axis break, no interpolation. CAVEATS: (1) two "
        "models / two scales across the break — within-block TRENDS are robust, "
        "cross-era absolute magnitude is the fragile claim (0.61 vs 0.96 "
        "calibrations differ; GT ~2× uncertain per IRR). (2) Aggregation "
        "asymmetry — 2015/16 points are ~1 frame each (noisier, higher single-"
        "frame peaks); 2021–23 points are weekly means (smoothed). Compare trends, "
        "not point magnitude. Shaded pre-Sep-2021 is provisional (no manual truth, "
        "early-2021 undercount). Source: build_hybrid_series_2015_2024.py; "
        "worm_series_{2015,2016}_daily.csv, worm_series_2021_2023_density_weekly.csv."
    )
    fig.text(0.012, 0.005, caption, fontsize=7.1, family="monospace",
             va="bottom", ha="left", wrap=True)

    fig.savefig(OUT_FIG, dpi=300)
    print(f"figure -> {OUT_FIG}")


if __name__ == "__main__":
    main()
