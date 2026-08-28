"""Worms-over-time figure from the single-frame mushroom.pt counts.

Reads the per-scene JSON cache written by ``run_full_timeseries.py`` and renders a
publication-grade time series (matplotlib, 300 DPI) following the lab timeseries
rubric: raw per-scene counts + a monthly-mean line that BREAKS at data gaps (no
interpolation across months with no scene1), bold labelled axes, Okabe-Ito colors.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "notebooks/timeseries_results_singleframe"
OUT_PNG = REPO / "notebooks/figure_worms_over_time_2023_2024.png"

# Okabe-Ito
BLUE, ORANGE, GREY = "#0072B2", "#E69F00", "#666666"

recs = [json.loads(f.read_text()) for f in sorted(RESULTS_DIR.glob("*.json"))]
counted = [r for r in recs if r.get("count") is not None]
errored = [r for r in recs if r.get("count") is None]
for r in counted:
    r["_dt"] = datetime.fromisoformat(r["dt"])
counted.sort(key=lambda r: r["_dt"])

dts = [r["_dt"] for r in counted]
counts = np.array([r["count"] for r in counted], dtype=float)
n_zero = int((counts == 0).sum())
print(f"counted scenes : {len(counted)}")
print(f"  zero-detection: {n_zero}/{len(counted)}")
print(f"  errors        : {len(errored)} ({[r.get('error') for r in errored][:5]})")
print(
    f"  total worms   : {int(counts.sum())}  mean/scene {counts.mean():.2f}  max {int(counts.max())}"
)

# --- monthly means with explicit gaps: every calendar month in range, NaN if empty ---
months = sorted({(d.year, d.month) for d in dts})
start, end = min(months), max(months)
grid = []
y, m = start
while (y, m) <= end:
    grid.append((y, m))
    m += 1
    if m > 12:
        y, m = y + 1, 1
mid = []
mean = []
for yy, mm in grid:
    vals = [c for c, d in zip(counts, dts) if d.year == yy and d.month == mm]
    mid.append(datetime(yy, mm, 15))
    mean.append(np.mean(vals) if vals else np.nan)
mean = np.array(mean)

GREEN = "#009E73"  # Okabe-Ito, for the hand-count truth marker

fig, ax = plt.subplots(figsize=(11, 4.5))

# View-collapse months (camera pan caught no front-on scene): shade them so the
# blanks are explained, not mysterious. Sep 2023 and Sep-Nov 2024.
collapse_spans = [
    (datetime(2023, 9, 1), datetime(2023, 10, 1)),
    (datetime(2024, 9, 1), datetime(2024, 12, 1)),
]
for i, (a, b) in enumerate(collapse_spans):
    ax.axvspan(
        a,
        b,
        color=GREY,
        alpha=0.18,
        zorder=0,
        label="no front-on scene (view collapse)" if i == 0 else None,
    )

ax.scatter(
    dts,
    counts,
    s=18,
    color=BLUE,
    alpha=0.45,
    zorder=2,
    label="mushroom.pt count, one per scene",
)
# masked array so the line BREAKS over months with no scene1 data (real gaps)
ax.plot(
    mid,
    np.ma.masked_invalid(mean),
    "-o",
    color=ORANGE,
    lw=2,
    ms=5,
    zorder=3,
    label="monthly mean of model counts",
)

# Hand-count ground truth in BOTH regimes. Jan 2023: model 9.8 vs truth 14.7 (67% recall).
# Feb 2024: model 0.5 vs truth 9.7 (5% recall). Two stars show the detector tracks the eye
# on 2023 footage but goes blind on post-Sep-2023 footage — the recall collapse, made visible.
truth_pts = [(datetime(2023, 1, 15), 14.68), (datetime(2024, 2, 15), 9.74)]
ax.scatter(
    [t for t, _ in truth_pts],
    [v for _, v in truth_pts],
    marker="*",
    s=340,
    color=GREEN,
    edgecolor="black",
    linewidth=0.6,
    zorder=5,
    label="hand count = ground truth (Jan '23 & Feb '24)",
)

ax.set_ylabel("Scale worm count (per scene)", fontweight="bold", fontsize=12)
ax.set_xlabel("Date", fontweight="bold", fontsize=12)
ax.set_title(
    "Scale worm relative-abundance index — Mushroom vent, Axial Seamount, 2023–2024\n"
    "(mushroom.pt / v0 single-frame counts; lower-bound index, not absolute abundance)",
    fontsize=12,
)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
ax.grid(alpha=0.3, zorder=0)
ax.set_ylim(bottom=-0.3)
# 3% x-padding
span = (max(dts) - min(dts)).days
ax.set_xlim(
    min(dts) - np.timedelta64(int(span * 0.03), "D"),
    max(dts) + np.timedelta64(int(span * 0.03), "D"),
)
for s in ax.spines.values():
    s.set_linewidth(1.0)
ax.legend(
    framealpha=0.92,
    loc="upper right",
    fontsize=10,
    title="Key",
    title_fontproperties={"weight": "bold", "size": 10},
)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
print(f"wrote {OUT_PNG}")
