"""The corrected story in one picture: worm count tracks image sharpness (blur), not
biology. Monthly mean worm count vs sharpness, colored by regime, Pearson r with a
bootstrap 95% CI. Hand-count months annotated (worms verified present there, so low
counts = detector failing on blurry footage, not absence).
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT_PNG = REPO / "notebooks/figure_count_vs_blur.png"
BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"

# monthly sharpness + count
iq = defaultdict(list)
for r in csv.DictReader((REPO / "notebooks/image_quality_timeseries.csv").open()):
    if r["sharpness"]:
        d = datetime.fromisoformat(r["dt"])
        iq[f"{d.year}-{d.month:02d}"].append(float(r["sharpness"]))
cnt = defaultdict(list)
for f in (REPO / "notebooks/timeseries_results_singleframe").glob("*.json"):
    rr = json.loads(f.read_text())
    if rr.get("count") is not None:
        d = datetime.fromisoformat(rr["dt"])
        cnt[f"{d.year}-{d.month:02d}"].append(rr["count"])

months = sorted(set(iq) & set(cnt))
sharp = np.array([np.mean(iq[m]) for m in months])
count = np.array([np.mean(cnt[m]) for m in months])
year = np.array([int(m[:4]) for m in months])

r = np.corrcoef(sharp, count)[0, 1]
rng = np.random.default_rng(20260828)  # fixed seed: reproducible bootstrap
idx = np.arange(len(months))
boot = []
for _ in range(2000):
    s = rng.choice(idx, size=len(idx), replace=True)
    if np.std(sharp[s]) > 0 and np.std(count[s]) > 0:
        boot.append(np.corrcoef(sharp[s], count[s])[0, 1])
lo, hi = np.percentile(boot, [2.5, 97.5])
print(
    f"n={len(months)} months  Pearson r={r:.2f}  bootstrap 95% CI [{lo:.2f}, {hi:.2f}]"
)

fig, ax = plt.subplots(figsize=(8, 6))
for yr, color, lab in [
    (2023, BLUE, "2023 (sharp footage)"),
    (2024, VERM, "2024 (blurry footage)"),
]:
    mask = year == yr
    ax.scatter(
        sharp[mask],
        count[mask],
        s=90,
        color=color,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
        label=lab,
    )
# mark hand-count-verified months (worms confirmed present)
for m in ("2024-02", "2024-05"):
    if m in months:
        i = months.index(m)
        ax.scatter(
            sharp[i],
            count[i],
            s=260,
            facecolor="none",
            edgecolor=GREEN,
            linewidth=2.2,
            zorder=4,
        )
ax.scatter(
    [],
    [],
    s=120,
    facecolor="none",
    edgecolor=GREEN,
    linewidth=2.2,
    label="worms hand-verified present",
)

ax.set_xlabel(
    "Image sharpness — Laplacian variance (blurry ← → sharp)",
    fontweight="bold",
    fontsize=12,
)
ax.set_ylabel(
    "Scale worm count (monthly mean per scene)", fontweight="bold", fontsize=12
)
ax.set_title(
    "Worm detections track camera sharpness, not biology\n"
    f"monthly means, 2023–2024  ·  Pearson r = {r:.2f}  (95% CI {lo:.2f}–{hi:.2f})",
    fontsize=12,
)
ax.grid(alpha=0.3)
ax.legend(framealpha=0.9, loc="upper left", fontsize=10)
for s in ax.spines.values():
    s.set_linewidth(1.0)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
print(f"wrote {OUT_PNG}")
