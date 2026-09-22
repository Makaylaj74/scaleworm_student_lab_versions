"""Figures for the Tuesday scale-worm series (2021-2023).

Renders three Paper-tier figures (matplotlib, 300 DPI, Okabe-Ito) per the lab
timeseries rubric, from the CSVs written by ``build_tuesday_series.py``:

  figure_tuesday_manual_series.png   manual per-Tuesday mean +/- 95% CI over time
  figure_tuesday_ai_saturation.png   AI vs manual per-frame scatter (detector saturation)
  figure_tuesday_extended_series.png manual anchors + density-corrected AI fill-ins

The scientific point: the v2 detector saturates on dense Tuesday scenes (recovers
~38% of worms; per-frame count explains only ~20% of the manual variance), so the
hand counts are the real series and any AI-based fill-in carries wide, honest CIs.
"""

from __future__ import annotations

import csv
import json
import textwrap
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
MANUAL_CSV = REPO / "validation/tuesday_manual_series/tuesday_manual_timeseries.csv"
CORRECTED_CSV = REPO / "notebooks/tuesday_corrected_ai_timeseries.csv"
EXTENDED_CSV = REPO / "notebooks/tuesday_extended_timeseries.csv"
PAIRS_CSV = REPO / "notebooks/tuesday_ai_vs_manual_pairs.csv"
CALIB_JSON = REPO / "notebooks/tuesday_calibration.json"

OUT_MANUAL = REPO / "notebooks/figure_tuesday_manual_series.png"
OUT_SATUR = REPO / "notebooks/figure_tuesday_ai_saturation.png"
OUT_EXT = REPO / "notebooks/figure_tuesday_extended_series.png"

# Okabe-Ito
BLUE, VERMILLION, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#666666"


def _justify(fig, x0, y0, width_frac, text, fontsize):
    """Render `text` as a fully-justified monospace block; return bottom y (fig frac)."""
    fig.canvas.draw()  # need a renderer for word widths
    rend = fig.canvas.get_renderer()
    fig_w_px = fig.bbox.width
    avail_px = width_frac * fig_w_px
    probe = fig.text(0, 0, "x" * 100, fontsize=fontsize, family="monospace")
    char_px = probe.get_window_extent(rend).width / 100
    probe.remove()
    n_chars = max(10, int(avail_px / char_px))
    lines = textwrap.wrap(text, n_chars)
    line_h = fontsize * 1.6 / fig.bbox.height
    y = y0
    for i, line in enumerate(lines):
        last = i == len(lines) - 1
        words = line.split()
        if last or len(words) == 1:
            fig.text(
                x0, y, line, fontsize=fontsize, family="monospace", ha="left", va="top"
            )
        else:
            widths = []
            for w in words:
                t = fig.text(0, 0, w, fontsize=fontsize, family="monospace")
                widths.append(t.get_window_extent(rend).width)
                t.remove()
            slack = avail_px - sum(widths)
            gap = slack / (len(words) - 1)
            cx = x0 * fig_w_px
            for j, w in enumerate(words):
                fig.text(
                    cx / fig_w_px,
                    y,
                    w,
                    fontsize=fontsize,
                    family="monospace",
                    ha="left",
                    va="top",
                )
                cx += widths[j] + gap
        y -= line_h
    return y


def _load_series(path):
    rows = list(csv.DictReader(path.open(newline="")))
    dts = [datetime.fromisoformat(r["date"]) for r in rows]
    mean = np.array([float(r["mean_worms"]) for r in rows])

    def col(name):
        return np.array(
            [float(r[name]) if r[name] not in ("", "nan") else np.nan for r in rows]
        )

    return rows, dts, mean, col("ci_lo"), col("ci_hi")


def _break_gaps(dts, y, max_gap_days=21):
    """Segment-wise x/y with a NaN break inserted across calendar gaps."""
    seg_x, seg_y = [], []
    for i, d in enumerate(dts):
        if i > 0 and (d - dts[i - 1]) > timedelta(days=max_gap_days):
            seg_x.append(np.datetime64("NaT"))
            seg_y.append(np.nan)
        seg_x.append(np.datetime64(d))
        seg_y.append(y[i])
    return seg_x, seg_y


def _date_axis(ax, dts):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    span = (max(dts) - min(dts)).days
    pad = timedelta(days=max(1, int(span * 0.03)))
    ax.set_xlim(min(dts) - pad, max(dts) + pad)
    ax.grid(alpha=0.3, zorder=0)
    for s in ax.spines.values():
        s.set_linewidth(1.0)


def fig_manual():
    _, dts, mean, lo, hi = _load_series(MANUAL_CSV)
    yerr = np.vstack([mean - lo, hi - mean])
    fig, ax = plt.subplots(figsize=(11, 5.2))
    sx, sy = _break_gaps(dts, mean)
    ax.plot(sx, sy, color=BLUE, lw=1.2, alpha=0.6, zorder=2)
    ax.errorbar(
        dts,
        mean,
        yerr=yerr,
        fmt="o",
        ms=5,
        color=BLUE,
        ecolor=GREY,
        elinewidth=1.0,
        capsize=2.5,
        zorder=3,
        label="Tuesday mean +/- 95% CI (<=8 Scene-1 slots)",
    )
    ax.set_ylabel("Scale-worm count per Scene-1 frame", fontweight="bold", fontsize=12)
    ax.set_xlabel("Date", fontweight="bold", fontsize=12)
    ax.set_title(
        "Manual scale-worm abundance — Mushroom vent, Axial Seamount, 2021-2023\n"
        "(hand-corrected Scene-1 counts; weekly Tuesday sampling, mean +/- 95% CI)",
        fontsize=13,
    )
    _date_axis(ax, dts)
    ax.set_ylim(bottom=0)
    ax.legend(framealpha=0.92, loc="upper left", fontsize=10)
    cap = (
        "AI-generated caption (Claude, Anthropic) — Manual scale-worm (Polynoidae) abundance at the "
        "Mushroom vent, Axial Seamount, from OOI Cabled Array HD video (instrument CAMHDA301, "
        "RS03ASHS-PN03B-06). Each point is the mean worm count across that Tuesday's front-on "
        "Scene-1 recordings (up to eight three-hourly UTC slots); error bars are a percentile "
        "bootstrap 95% CI over the counted slots (n=10000, seed 20260922; absent where only one slot "
        "was usable, and wide where only 2-3 slots were). Counts "
        "are human box-corrected annotations (annotator MJ; worm_count = number of boxes) over 229 "
        "usable frames on 49 Tuesdays (2021-12 to 2023-08); 56 unusable-blur slots are excluded as "
        "missing data, not zeros, so the connecting line breaks across gaps > 21 days. Derived from "
        "validation/tuesday_manual_series/tuesday_manual_timeseries.csv (build_tuesday_series.py)."
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.9, bottom=0.42)
    _justify(fig, 0.08, 0.24, 0.89, cap, fontsize=8)
    fig.savefig(OUT_MANUAL, dpi=300)
    plt.close(fig)
    print(f"wrote {OUT_MANUAL.name}  ({len(dts)} Tuesdays)")


def fig_saturation():
    rows = list(csv.DictReader(PAIRS_CSV.open(newline="")))
    ai = np.array([float(r["ai_count"]) for r in rows])
    man = np.array([float(r["worm_count"]) for r in rows])
    calib = json.loads(CALIB_JSON.read_text())
    b0, b1, r2 = calib["b0"], calib["b1"], calib["r2"]
    recall = calib["aggregate_recall_ai_over_manual"]

    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    hi = max(ai.max(), man.max()) + 3
    ax.plot(
        [0, hi],
        [0, hi],
        color=GREY,
        ls="--",
        lw=1.3,
        zorder=1,
        label="1:1 (perfect detection)",
    )
    xs = np.array([ai.min(), ai.max()])
    ax.plot(
        xs,
        b0 + b1 * xs,
        color=VERMILLION,
        lw=2.0,
        zorder=3,
        label=f"OLS: manual = {b0:.1f} + {b1:.2f}*AI  (R^2={r2:.2f})",
    )
    ax.scatter(
        ai,
        man,
        s=26,
        color=BLUE,
        alpha=0.55,
        edgecolor="none",
        zorder=2,
        label=f"Scene-1 frames (n={len(rows)})",
    )
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(
        "v2 AI count (raw YOLO boxes @ conf 0.25)", fontweight="bold", fontsize=12
    )
    ax.set_ylabel("Manual count (hand-corrected boxes)", fontweight="bold", fontsize=12)
    ax.set_title(
        "Detector saturation: v2 AI count vs. manual count per frame\n"
        f"(Tuesday Scene-1 frames, 2021-2023; aggregate AI recall = {recall:.0%})",
        fontsize=13,
    )
    ax.grid(alpha=0.3, zorder=0)
    for s in ax.spines.values():
        s.set_linewidth(1.0)
    ax.legend(framealpha=0.92, loc="upper left", fontsize=9.5)
    cap = (
        "AI-generated caption (Claude, Anthropic) — Per-frame scale-worm count from the v2 detector "
        "(raw YOLO box count at confidence 0.25) against the human box-corrected count, for 229 "
        "front-on Scene-1 frames on 49 Tuesdays (CAMHDA301, Mushroom vent, Axial Seamount, "
        "2021-2023). Points fall far below the 1:1 line and the detector plateaus near 28 boxes while "
        "true counts reach 69: v2 recovers only 38% of worms in aggregate, and its count explains "
        "just 20% of the manual variance (R^2=0.20), so a single scalar recall factor is not "
        "defensible. The vermillion line is the ordinary-least-squares calibration used to fill "
        "un-counted Tuesdays. Derived from notebooks/tuesday_ai_vs_manual_pairs.csv "
        "(build_tuesday_series.py); AI counts from scripts/run_ai_counts.py."
    )
    fig.subplots_adjust(left=0.10, right=0.97, top=0.9, bottom=0.34)
    _justify(fig, 0.10, 0.22, 0.87, cap, fontsize=8)
    fig.savefig(OUT_SATUR, dpi=300)
    plt.close(fig)
    print(f"wrote {OUT_SATUR.name}  (n={len(rows)})")


def fig_extended():
    rows = list(csv.DictReader(EXTENDED_CSV.open(newline="")))
    man = [r for r in rows if r["source"] == "manual"]
    cor = [r for r in rows if r["source"] == "ai_corrected"]

    def unpack(rs):
        dts = [datetime.fromisoformat(r["date"]) for r in rs]
        mean = np.array([float(r["mean_worms"]) for r in rs])
        lo = np.array(
            [float(r["ci_lo"]) if r["ci_lo"] not in ("", "nan") else np.nan for r in rs]
        )
        hi = np.array(
            [float(r["ci_hi"]) if r["ci_hi"] not in ("", "nan") else np.nan for r in rs]
        )
        return dts, mean, lo, hi

    md, mm, mlo, mhi = unpack(man)
    cd, cm, clo, chi = unpack(cor)

    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    # corrected first (background), wide CI
    ax.errorbar(
        cd,
        cm,
        yerr=np.vstack([cm - clo, chi - cm]),
        fmt="s",
        ms=5,
        mfc="white",
        color=VERMILLION,
        ecolor=VERMILLION,
        elinewidth=1.0,
        capsize=2.0,
        alpha=0.85,
        zorder=2,
        label="Density-corrected v2 AI (+/- 95% bootstrap CI)",
    )
    # manual on top, tight CI
    ax.errorbar(
        md,
        mm,
        yerr=np.vstack([mm - mlo, mhi - mm]),
        fmt="o",
        ms=5.5,
        color=BLUE,
        ecolor=GREY,
        elinewidth=1.0,
        capsize=2.5,
        zorder=3,
        label="Manual hand count (+/- 95% bootstrap CI)",
    )
    alldts = md + cd
    ax.set_ylabel("Scale-worm count per Scene-1 frame", fontweight="bold", fontsize=12)
    ax.set_xlabel("Date", fontweight="bold", fontsize=12)
    ax.set_title(
        "Extended Tuesday scale-worm series — Mushroom vent, Axial Seamount, 2021-2023\n"
        "(hand-counted anchors + density-corrected v2 AI fill-ins; per-Tuesday mean)",
        fontsize=13,
    )
    _date_axis(ax, alldts)
    ax.set_ylim(bottom=0)
    ax.legend(framealpha=0.92, loc="upper left", fontsize=9.5)
    cap = (
        "AI-generated caption (Claude, Anthropic) — Weekly-Tuesday scale-worm (Polynoidae) abundance "
        "at the Mushroom vent, Axial Seamount (CAMHDA301, RS03ASHS-PN03B-06), 2021-2023. Blue "
        "circles are the 49 Tuesdays hand-counted by MJ (mean over up-to-8 Scene-1 slots; bootstrap "
        "95% CI). Open vermillion squares are 43 other Tuesdays the v2 detector scored but nobody "
        "hand-counted, with per-frame counts mapped to worms by the OLS calibration manual = 18.2 + "
        "1.11*AI and aggregated per Tuesday; error bars are a paired bootstrap 95% CI (n=10000, seed "
        "20260922) propagating both fit and residual uncertainty. The corrected series is "
        "deliberately conservative -- because AI explains only 20% of the variance it regresses "
        "toward the mean (~29 worms) and cannot resolve the peaks and troughs the hand counts show; "
        "it fills temporal gaps, it does not replace counting. Derived from "
        "notebooks/tuesday_extended_timeseries.csv (build_tuesday_series.py)."
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.9, bottom=0.44)
    _justify(fig, 0.08, 0.25, 0.89, cap, fontsize=8)
    fig.savefig(OUT_EXT, dpi=300)
    plt.close(fig)
    print(f"wrote {OUT_EXT.name}  (manual={len(man)} corrected={len(cor)})")


def main() -> None:
    fig_manual()
    fig_saturation()
    fig_extended()


if __name__ == "__main__":
    main()
