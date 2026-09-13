"""Manual scale-worm abundance timeseries (weekly Monday Scene-1 series, 2023-2024).

Reads ``monday_manual_timeseries.csv`` (per-Monday mean +/- SEM, from
``build_manual_timeseries.py``) and renders a Paper-tier figure (matplotlib, 300 DPI,
Okabe-Ito) per the lab timeseries rubric: bold labelled axes, mean marker with SEM
error bars, a connecting line that BREAKS across calendar gaps > 14 days (no silent
interpolation), and the ~Aug-2023 camera-blur onset marked for context.

The point of the figure: unlike the mushroom.pt model index (which collapses to ~0
after Sep-2023), the *manual* counts stay high across the blur boundary -> the worms
were present throughout; the model collapse was a detection artifact.
"""

from __future__ import annotations

import csv
import textwrap
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
CSV = REPO / "validation/monday_manual_series/monday_manual_timeseries.csv"
OUT_PNG = REPO / "notebooks/figure_manual_worms_over_time.png"

# Okabe-Ito
BLUE, VERMILLION, GREY = "#0072B2", "#D55E00", "#666666"
BLUR_ONSET = date(2023, 8, 10)

CAPTION = (
    "AI-generated caption (Claude, Anthropic) — Manual scale-worm (Polynoidae) abundance at the "
    "Mushroom vent, Axial Seamount, from OOI Cabled Array HD video (instrument CAMHDA301, "
    "RS03ASHS-PN03B-06), 2023-2024. Each point is the mean worm count across that Monday's "
    "front-on Scene-1 recordings (up to eight three-hourly slots); error bars are +/- 1 SEM over "
    "the counted slots (absent where only one slot was usable). Counts are derived from human "
    "box-corrected annotations (annotator MJ; worm_count = number of boxes), seeded by the v2 "
    "(clear) / v3 (blurry) detectors and corrected in both directions; 367 Scene-1 frames over 67 "
    "Mondays. Non-front-on and unusable-blur slots are excluded as missing data, not zeros, so the "
    "connecting line breaks across gaps > 14 days. The dashed line marks the ~Aug-2023 onset of "
    "persistent camera blur; abundance does not collapse across it, confirming the post-Sep-2023 "
    "drop in the automated mushroom.pt index was a detection artifact, not a real decline. "
    "Derived from validation/monday_manual_series/frame_manifest.csv."
)


def _justify(fig, x0, y0, width_frac, text, fontsize):
    """Render `text` as a fully-justified monospace block; return bottom y (fig frac)."""
    fig.canvas.draw()  # need a renderer for word widths
    rend = fig.canvas.get_renderer()
    fig_w_px = fig.bbox.width
    avail_px = width_frac * fig_w_px
    # approx chars/line from monospace advance width
    probe = fig.text(0, 0, "x" * 100, fontsize=fontsize, family="monospace")
    char_px = probe.get_window_extent(rend).width / 100
    probe.remove()
    n_chars = max(10, int(avail_px / char_px))
    lines = textwrap.wrap(text, n_chars)
    line_h = fontsize * 1.6 / fig.bbox.height  # fig-fraction line height
    y = y0
    for i, line in enumerate(lines):
        last = i == len(lines) - 1
        words = line.split()
        if last or len(words) == 1:
            fig.text(
                x0, y, line, fontsize=fontsize, family="monospace", ha="left", va="top"
            )
        else:
            # measure words, distribute slack as equal inter-word gaps
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


def main() -> None:
    rows = list(csv.DictReader(CSV.open(newline="")))
    dts = [datetime.fromisoformat(r["date"]) for r in rows]
    mean = np.array([float(r["mean_worms"]) for r in rows])
    sem = np.array([float(r["sem_worms"]) if r["sem_worms"] else np.nan for r in rows])

    fig, ax = plt.subplots(figsize=(11, 5.2))

    # connecting line that breaks across gaps > 14 days (no interpolation over gaps):
    # plot segment-wise, inserting a NaT/NaN break wherever consecutive Mondays are >14d apart.
    seg_x, seg_y = [], []
    for i, d in enumerate(dts):
        if i > 0 and (d - dts[i - 1]) > timedelta(days=14):
            seg_x.append(np.datetime64("NaT"))
            seg_y.append(np.nan)
        seg_x.append(np.datetime64(d))
        seg_y.append(mean[i])
    ax.plot(seg_x, seg_y, color=BLUE, lw=1.2, alpha=0.6, zorder=2)

    ax.errorbar(
        dts,
        mean,
        yerr=sem,
        fmt="o",
        ms=5,
        color=BLUE,
        ecolor=GREY,
        elinewidth=1.0,
        capsize=2.5,
        zorder=3,
        label="Monday mean +/- SEM (<=8 Scene-1 slots)",
    )

    # blur-onset context marker
    ax.axvline(BLUR_ONSET, color=VERMILLION, ls="--", lw=1.4, zorder=1)
    ax.text(
        BLUR_ONSET + timedelta(days=8),
        ax.get_ylim()[1] * 0.96,
        "camera blur onset (~Aug 2023)",
        color=VERMILLION,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )

    ax.set_ylabel("Scale-worm count per Scene-1 frame", fontweight="bold", fontsize=12)
    ax.set_xlabel("Date", fontweight="bold", fontsize=12)
    ax.set_title(
        "Manual scale-worm abundance — Mushroom vent, Axial Seamount, 2023-2024\n"
        "(hand-corrected Scene-1 counts; weekly Monday sampling, mean +/- SEM)",
        fontsize=13,
    )
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(alpha=0.3, zorder=0)
    ax.set_ylim(bottom=0)

    span = (max(dts) - min(dts)).days
    pad = timedelta(days=int(span * 0.03))
    ax.set_xlim(min(dts) - pad, max(dts) + pad)
    for s in ax.spines.values():
        s.set_linewidth(1.0)
    ax.legend(framealpha=0.92, loc="upper right", fontsize=10)

    # reserve space below the axes for the justified caption
    fig.subplots_adjust(left=0.08, right=0.97, top=0.9, bottom=0.42)
    _justify(fig, 0.08, 0.24, 0.89, CAPTION, fontsize=8)

    fig.savefig(OUT_PNG, dpi=300)
    print(f"wrote {OUT_PNG}")
    print(
        f"points: {len(rows)}  mean/frame overall: {mean.mean():.1f}  "
        f"range {mean.min():.1f}-{mean.max():.1f}"
    )


if __name__ == "__main__":
    main()
