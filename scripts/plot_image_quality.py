"""Stacked figure: image quality (contrast, sharpness) over the worm-count collapse.

Answers 'step vs gradual': if contrast/sharpness drop abruptly at one date the
degradation is a discrete event; if they slide down over months it's biofouling.
Top panel = image quality; bottom = mushroom.pt monthly-mean count (from cache).
Shared date axis so the two are directly comparable.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
IQ_CSV = REPO / "notebooks/image_quality_timeseries.csv"
CACHE = REPO / "notebooks/timeseries_results_singleframe"
OUT_PNG = REPO / "notebooks/figure_image_quality_vs_count.png"

BLUE, ORANGE, GREEN, GREY = "#0072B2", "#E69F00", "#009E73", "#666666"


def monthly(dates, vals):
    """Monthly means over every calendar month in range; NaN for empty months (gaps show)."""
    months = sorted({(d.year, d.month) for d in dates})
    (y0, m0), (y1, m1) = months[0], months[-1]
    grid, y, m = [], y0, m0
    while (y, m) <= (y1, m1):
        grid.append((y, m))
        m += 1
        if m > 12:
            y, m = y + 1, 1
    mid, mean = [], []
    for yy, mm in grid:
        v = [
            x
            for x, d in zip(vals, dates)
            if d.year == yy and d.month == mm and not np.isnan(x)
        ]
        mid.append(datetime(yy, mm, 15))
        mean.append(np.mean(v) if v else np.nan)
    return mid, np.array(mean)


# image quality
iq = [r for r in csv.DictReader(IQ_CSV.open()) if r["contrast"]]
iq_dt = [datetime.fromisoformat(r["dt"]) for r in iq]
contrast = np.array([float(r["contrast"]) for r in iq])
sharp = np.array([float(r["sharpness"]) for r in iq])

# counts from cache
crecs = [json.loads(f.read_text()) for f in sorted(CACHE.glob("*.json"))]
crecs = [r for r in crecs if r.get("count") is not None]
c_dt = [datetime.fromisoformat(r["dt"]) for r in crecs]
counts = np.array([r["count"] for r in crecs], dtype=float)

collapse = [
    (datetime(2023, 9, 1), datetime(2023, 10, 1)),
    (datetime(2024, 9, 1), datetime(2024, 12, 1)),
]

fig, (axA, axB) = plt.subplots(
    2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]}
)

# ── Panel A: contrast (left) + sharpness (right), color-matched dual axis ──
for a, b in collapse:
    axA.axvspan(a, b, color=GREY, alpha=0.18, zorder=0)
axA.scatter(iq_dt, contrast, s=12, color=BLUE, alpha=0.30, zorder=2)
mid, cm = monthly(iq_dt, contrast)
axA.plot(
    mid,
    np.ma.masked_invalid(cm),
    "-o",
    color=BLUE,
    lw=2,
    ms=4,
    zorder=3,
    label="RMS contrast (monthly mean)",
)
axA.set_ylabel("RMS contrast", color=BLUE, fontweight="bold", fontsize=11)
axA.tick_params(axis="y", labelcolor=BLUE)

axA2 = axA.twinx()
mid2, sm = monthly(iq_dt, sharp)
axA2.plot(
    mid2,
    np.ma.masked_invalid(sm),
    "-s",
    color=ORANGE,
    lw=2,
    ms=4,
    zorder=3,
    label="sharpness / Laplacian var (monthly mean)",
)
axA2.set_ylabel(
    "Sharpness (Laplacian var)", color=ORANGE, fontweight="bold", fontsize=11
)
axA2.tick_params(axis="y", labelcolor=ORANGE)
axA.set_title(
    "A. Image quality over time — does it step or slide?",
    fontsize=12,
    fontweight="bold",
    loc="left",
)
axA.grid(alpha=0.3)
# hand-count month markers
for d in (datetime(2023, 1, 15), datetime(2024, 2, 15)):
    axA.axvline(d, color=GREEN, ls=":", lw=1.3, zorder=1)

# ── Panel B: worm count monthly mean ──
for a, b in collapse:
    axB.axvspan(
        a,
        b,
        color=GREY,
        alpha=0.18,
        zorder=0,
        label="no front-on scene" if (a, b) == collapse[0] else None,
    )
axB.scatter(c_dt, counts, s=12, color="#4C72B0", alpha=0.30, zorder=2)
midc, cc = monthly(c_dt, counts)
axB.plot(
    midc,
    np.ma.masked_invalid(cc),
    "-o",
    color="#C44E52",
    lw=2,
    ms=4,
    zorder=3,
    label="worm count (monthly mean)",
)
axB.set_ylabel("Scale worm count\n(per scene)", fontweight="bold", fontsize=11)
axB.set_xlabel("Date", fontweight="bold", fontsize=12)
axB.set_title(
    "B. mushroom.pt worm count (the collapse)",
    fontsize=12,
    fontweight="bold",
    loc="left",
)
axB.grid(alpha=0.3)
axB.set_ylim(bottom=-0.3)
axB.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
axB.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(axB.get_xticklabels(), rotation=45, ha="right")

# combined legend note for the green line
axA.plot(
    [], [], color=GREEN, ls=":", lw=1.3, label="hand-count months (Jan '23, Feb '24)"
)
axA.legend(loc="upper right", framealpha=0.9, fontsize=9)
axB.legend(loc="upper right", framealpha=0.9, fontsize=9)

fig.suptitle(
    "Camera image quality vs scale-worm detections — CAMHDA301, 2023–2024",
    fontsize=13,
    fontweight="bold",
)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
print(
    f"contrast Jan'23~{cm[0]:.1f}  ...  post-2023 range {np.nanmin(cm[12:]):.1f}-{np.nanmax(cm[12:]):.1f}"
)
print(f"wrote {OUT_PNG}")
